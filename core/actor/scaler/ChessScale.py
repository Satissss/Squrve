import re
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset
from core.utils import parse_schema_from_df, load_dataset, save_dataset
from llama_index.core.llms.llm import LLM
from core.actor.prompts.CHESSPrompt import (
    template_generate_candidate_one,
    template_generate_candidate_two,
    template_generate_candidate_three,
    template_generate_candidate_retrieval,
    template_extract_keywords,
)

@BaseScaler.register_actor
class ChessScaler(BaseScaler):
    """Scaler implementation based on CHESS-SQL's candidate generation for producing multiple SQL candidates."""

    NAME = "ChessScaler"

    CANDIDATE_TEMPLATE = '''You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Database admin instructions:
1. **SELECT Clause:** Only select columns mentioned in the user's question.
2. **Aggregation (MAX/MIN):** Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:** Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC`.
4. **Handling NULLs:** If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:** Only include tables essential to answer the question.
6. **Strictly Follow Hints:** Adhere to all provided hints.
7. **Thorough Question Analysis:** Address all conditions mentioned in the question.
8. **DISTINCT Keyword:** Use `SELECT DISTINCT` when the question requires unique values.
9. **Column Selection:** Carefully analyze column descriptions and hints to choose the correct column.
10. **JOIN Preference:** Prioritize `INNER JOIN` over nested `SELECT` statements.
11. **SQLite Functions Only:** Use only functions available in SQLite.

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

【Database Info】
{DATABASE_SCHEMA}

【Question】
Question: {QUESTION}

Evidence: {HINT}

Reasoning Examples:
{REASONING_EXAMPLES}

【Answer】
'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 5,
            temperature: float = 0.5,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            open_parallel: bool = True,
            max_workers: int = None,
            use_external: bool = True,
            use_few_shot: bool = True,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.generate_num = generate_num
        self.temperature = temperature
        self.use_external = use_external
        self.use_few_shot = use_few_shot

    def _extract_keywords(self, question: str, hint: str = "") -> List[str]:
        """Extract keywords from the question using the same template as CHESSGenerate."""
        try:
            llm_lis = self.llm if isinstance(self.llm, list) else [self.llm]
            llm_to_use = llm_lis[0] if llm_lis else None
            if llm_to_use is None:
                logger.warning("No LLM available for keyword extraction")
                return []

            prompt = template_extract_keywords().format(QUESTION=question, HINT=hint or "")
            response = llm_to_use.complete(prompt, temperature=0.2).text
            match = re.search(r'\[.*\]', response)
            if match:
                keywords_str = match.group(0)
                keywords = eval(keywords_str)
                return keywords if isinstance(keywords, list) else []
            return []
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []

    def _retrieve_context(self, question: str, schema: str, keywords: List[str]) -> str:
        """Retrieve relevant context from schema based on keywords"""
        if not keywords:
            return schema

        relevant_tables = []
        schema_lines = schema.split('\n')

        for line in schema_lines:
            line_lower = line.lower()
            if any(keyword.lower() in line_lower for keyword in keywords):
                relevant_tables.append(line)

        if relevant_tables:
            return '\n'.join(relevant_tables)
        return schema

    def _build_evidence(self, question: str, schema_text: str, keywords: List[str], dataset_evidence: str,
                        schema_links: Union[str, List[str]] = None, sub_questions: Union[str, List[str]] = None) -> str:
        """Construct lightweight evidence in parity with CHESSGenerate."""
        lines = schema_text.split('\n') if schema_text else []
        matched_lines = []
        lower_keywords = [kw.lower() for kw in (keywords or [])]
        for line in lines:
            li = line.strip()
            lwr = li.lower()
            if any(kw in lwr for kw in lower_keywords):
                matched_lines.append(li)

        top_matches = matched_lines[:12]
        summary_parts = []
        if keywords:
            summary_parts.append(f"Identified keywords: {', '.join(keywords[:10])}.")
        if top_matches:
            summary_parts.append("Relevant schema snippets:")
            summary_parts.extend(top_matches)
        if dataset_evidence:
            summary_parts.append("Additional hints:")
            summary_parts.append(str(dataset_evidence))
        if schema_links:
            if isinstance(schema_links, list):
                schema_links_str = ', '.join(schema_links)
            else:
                schema_links_str = str(schema_links)
            summary_parts.append("Schema links (Identified Critical Tables & Columns):")
            summary_parts.append(schema_links_str)
        if sub_questions:
            if isinstance(sub_questions, list):
                sub_questions_str = '\n'.join([f"- {q}" for q in sub_questions])
            else:
                sub_questions_str = str(sub_questions)
            summary_parts.append("Sub-questions (Sub-question Decomposition of the Original Question):")
            summary_parts.append(sub_questions_str)

        return "\n".join(summary_parts).strip()

    def _generate_candidates_with_templates(
            self,
            llm_: LLM,
            question: str,
            schema: str,
            evidence: str,
            total_samples: int,
            temperature: float,
            reasoning_examples: Optional[str] = None,
    ) -> List[str]:
        """Use diversified CHESS templates to generate multiple SQL candidates."""
        template_functions = [
            template_generate_candidate_one,
            template_generate_candidate_two,
            template_generate_candidate_three,
            template_generate_candidate_retrieval,
        ]

        total_samples = max(1, int(total_samples))
        samples_per_template = max(1, total_samples // len(template_functions))

        candidates: List[str] = []
        examples_text = reasoning_examples or ""
        for template_func in template_functions:
            for _ in range(samples_per_template):
                try:
                    params = {
                        "DATABASE_SCHEMA": schema,
                        "QUESTION": question,
                        "HINT": evidence,
                        "REASONING_EXAMPLES": examples_text,
                    }
                    if template_func == template_generate_candidate_retrieval:
                        params["EXAMPLES"] = examples_text
                    prompt = template_func().format(**params)
                    response = llm_.complete(prompt, temperature=temperature).text
                    sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1).strip()
                        if sql:
                            candidates.append(sql)
                    else:
                        for line in response.split('\n'):
                            if line.strip().upper().startswith('SELECT'):
                                candidates.append(line.strip())
                                break
                except Exception as e:
                    logger.warning(f"Failed to generate candidate: {e}")
                    continue

        return candidates

    def _generate_single_candidate(self, llm_: LLM, question: str, schema: str, evidence: str) -> Optional[str]:
        """Generate a single SQL candidate"""
        try:
            prompt = self.CANDIDATE_TEMPLATE.format(
                DATABASE_SCHEMA=schema,
                QUESTION=question,
                HINT=evidence,
                REASONING_EXAMPLES=""
            )

            response = llm_.complete(prompt, temperature=self.temperature).text

            sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            else:
                lines = response.split('\n')
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        sql = line.strip()
                        logger.debug(f"Generated SQL candidate from line: {sql[:100]}...")
                        return sql
            logger.warning("No SQL found in LLM response")
            return None
        except Exception as e:
            logger.warning(f"Failed to generate candidate: {e}")
            return None

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ) -> List[str]:
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '') or kwargs.get('evidence', '') or ''
        # Load and process schema using base class method
        schema = self.process_schema(schema, item)

        # Information Retrieval (align with CHESSGenerate)
        keywords = self._extract_keywords(question, evidence)
        context = self._retrieve_context(question, schema, keywords)
        built_evidence = self._build_evidence(question, context, keywords, evidence, schema_links, sub_questions)

        if self.use_external:
            if data_logger:
                data_logger.info(f"{self.NAME}: use external knowledge to explain the question.")
            external_knowledge = self.load_external_knowledge(row.get("external", None))
            if external_knowledge:
                question += "\n" + external_knowledge
                logger.debug("已加载外部知识")

        reasoning_examples = None
        if self.use_few_shot:
            if data_logger:
                data_logger.info(f"{self.NAME}: use retrieved reasoning exapmles to enhance the results.")
            reasoning_example_path = row.get("reasoning_examples", None)
            if reasoning_example_path:
                reasoning_examples = load_dataset(reasoning_example_path)
                logger.debug(f"加载推理示例: {reasoning_example_path}")

        # 在 act 方法内部初始化 llm，考虑 self.llm 是否为列表
        if isinstance(self.llm, list) and self.llm:
            llm = self.llm[0]
        else:
            llm = self.llm

        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            logger.warning("No LLM available for SQL generation")
            return []

        # 仅使用第一个 LLM 生成 SQL 候选（使用多策略模板）
        pred_sqls = self._generate_candidates_with_templates(
            llm_=llm,
            question=question,
            schema=context,
            evidence=built_evidence,
            total_samples=self.generate_num,
            temperature=self.temperature,
            reasoning_examples=reasoning_examples,
        )

        # Deduplicate
        pred_sqls = list(dict.fromkeys(pred_sqls))

        # 确保至少有一个 SQL 结果，如果没有生成任何 SQL，创建一个默认的
        if not pred_sqls:
            logger.warning(f"No SQL candidates generated for item {item}, creating default SQL")
            pred_sqls = ["SELECT * FROM table LIMIT 1"]  # 默认 SQL

        logger.info(f"ChessScaler: Final pred_sqls for item {item}: {len(pred_sqls)} candidates")
        if data_logger:
            data_logger.info(f"{self.NAME}.candidates | count={len(pred_sqls)}")

        # Save results using base class method
        self.save_output(pred_sqls, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.pred_sqls | output={pred_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return pred_sqls
