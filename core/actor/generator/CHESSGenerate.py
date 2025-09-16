from llama_index.core.llms.llm import LLM
from typing import Union, List, Callable, Dict, Optional, Any
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger
import json
import re
from dataclasses import dataclass
import ast

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)
from core.actor.prompts.CHESSPrompt import (
    template_generate_candidate_one,
    template_generate_candidate_two,
    template_generate_candidate_three,
    template_generate_candidate_retrieval,
    template_revise_one,
    template_evaluate,
    template_generate_unit_tests,
    template_select_tables,
    template_extract_keywords
)


@dataclass
class CHESSConfig:
    """Configuration for CHESS-SQL method"""
    # Information Retriever settings
    ir_engine: str = "gpt-4o-mini"
    ir_temperature: float = 0.2
    ir_top_k: int = 5

    # Candidate Generator settings
    cg_engine: str = "gpt-4o-mini"
    cg_temperature: float = 0.5
    cg_sampling_count: int = 10

    # Unit Tester settings
    ut_engine: str = "gpt-4o-mini"
    ut_temperature: float = 0.8
    ut_unit_test_count: int = 20

    # Schema Selector settings (optional)
    use_schema_selector: bool = False
    ss_engine: str = "gpt-4o-mini"
    ss_temperature: float = 0.2


class CHESSGenerator(BaseGenerator):
    """CHESS-SQL: Contextual Harnessing for Efficient SQL Synthesis

    A multi-agent framework for efficient and scalable SQL synthesis, comprising:
    1. Information Retriever (IR): Extracts relevant data
    2. Schema Selector (SS): Prunes large schemas (optional)
    3. Candidate Generator (CG): Generates high-quality candidates and refines queries
    4. Unit Tester (UT): Validates queries through LLM-based natural language unit tests
    """

    NAME = "CHESSGenerator"


    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            config: Optional[CHESSConfig] = None,
            sql_post_process_function: Optional[Callable] = None,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ):
        self.dataset: Optional[Dataset] = dataset
        self.llm: Optional[LLM] = llm
        self.is_save = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.config = config or CHESSConfig()
        self.sql_post_process_function: Optional[Callable] = sql_post_process_function

        # Initialize database path and credentials
        if db_path is not None:
            self.db_path = db_path
        elif self.dataset is not None:
            self.db_path = self.dataset.db_path
        else:
            self.db_path = None

        if credential is not None:
            self.credential = credential
        elif self.dataset is not None:
            self.credential = self.dataset.credential
        else:
            self.credential = None


    def _build_evidence(self, question: str, schema_text: str, keywords: List[str], dataset_evidence: str) -> str:
        """Construct lightweight evidence/chain-of-thought style hints.

        This mimics baseline's aggregated CoT from table/column selection in a minimal way.
        """
        lines = schema_text.split('\n') if schema_text else []
        matched_lines = []
        lower_keywords = [kw.lower() for kw in (keywords or [])]
        for line in lines:
            li = line.strip()
            lwr = li.lower()
            if any(kw in lwr for kw in lower_keywords):
                matched_lines.append(li)

        top_matches = matched_lines[:12]  # cap for brevity
        summary_parts = []
        if keywords:
            summary_parts.append(f"Identified keywords: {', '.join(keywords[:10])}.")
        if top_matches:
            summary_parts.append("Relevant schema snippets:")
            summary_parts.extend(top_matches)
        if dataset_evidence:
            summary_parts.append("Additional hints:")
            summary_parts.append(str(dataset_evidence))

        return "\n".join(summary_parts).strip()

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from the question using LLM"""
        prompt = template_extract_keywords().format(QUESTION=question, HINT="")

        try:
            response = self.llm.complete(prompt, temperature=self.config.ir_temperature).text
            # Extract list from response
            match = re.search(r'\[.*\]', response)
            if match:
                keywords_str = match.group(0)
                # Simple parsing - in production, use ast.literal_eval for safety
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

        # Simple keyword-based filtering
        relevant_tables = []
        schema_lines = schema.split('\n')

        for line in schema_lines:
            line_lower = line.lower()
            if any(keyword.lower() in line_lower for keyword in keywords):
                relevant_tables.append(line)

        if relevant_tables:
            return '\n'.join(relevant_tables)
        return schema

    def _select_schema(self, schema: str, question: str) -> str:
        if not self.config.use_schema_selector:
            return schema
        # Use template_select_tables for schema selection
        prompt = template_select_tables().format(DATABASE_SCHEMA=schema, QUESTION=question, HINT="")
        try:
            response = self.llm.complete(prompt, temperature=self.config.ss_temperature).text
            # Parse JSON response to get table names
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                table_names = result.get("table_names", [])
                # Filter schema to include only selected tables
                schema_lines = schema.split('\n')
                filtered_lines = []
                current_table = None
                for line in schema_lines:
                    if line.strip().upper().startswith('CREATE TABLE'):
                        table_name = line.split()[2].strip('`()')
                        current_table = table_name
                    if current_table in table_names:
                        filtered_lines.append(line)
                return '\n'.join(filtered_lines)
            return schema
        except Exception as e:
            logger.warning(f"Failed to select schema: {e}")
            return schema

    def _generate_candidate_sql(self, question: str, schema: str, evidence: str = "") -> List[Dict[str, Any]]:
        """Generate candidate SQL queries using multiple baseline-style templates if available."""

        # Use diversified baseline templates
        template_functions = [
            template_generate_candidate_one,
            template_generate_candidate_two,
            template_generate_candidate_three,
            template_generate_candidate_retrieval,
        ]

        total_samples = max(1, self.config.cg_sampling_count)
        samples_per_template = max(1, total_samples // len(template_functions))

        candidates: List[Dict[str, Any]] = []
        for template_func in template_functions:
            for i in range(samples_per_template):
                try:
                    # Prepare template parameters
                    template_params = {
                        "DATABASE_SCHEMA": schema,
                        "QUESTION": question,
                        "HINT": evidence
                    }
                    
                    # Add EXAMPLES placeholder for template_generate_candidate_retrieval
                    if template_func == template_generate_candidate_retrieval:
                        template_params["EXAMPLES"] = ""  # Empty examples for now, can be enhanced later

                    prompt = template_func().format(**template_params)

                    response = self.llm.complete(prompt, temperature=self.config.cg_temperature).text

                    # Extract SQL from response
                    sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1).strip()
                        candidates.append({
                            "SQL": sql,
                            "chain_of_thought_reasoning": response,
                            "confidence": 0.8
                        })
                    else:
                        # Fallback: try to extract first SELECT line
                        lines = response.split('\n')
                        for line in lines:
                            if line.strip().upper().startswith('SELECT'):
                                candidates.append({
                                    "SQL": line.strip(),
                                    "chain_of_thought_reasoning": response,
                                    "confidence": 0.6
                                })
                                break

                except Exception as e:
                    logger.warning(f"Failed to generate candidate: {e}")
                    continue

        return candidates

    def _revise_sql(self, question: str, schema: str, sql: str, feedback: str = "") -> str:
        """Revise SQL query based on feedback"""

        # Use baseline revise template
        template = template_revise_one()

        try:
            prompt = template.format(
                DATABASE_SCHEMA=schema,
                QUESTION=question,
                HINT=feedback,  # Use HINT instead of FEEDBACK
                QUERY=sql,      # Use QUERY instead of SQL
                RESULT=feedback  # Use RESULT instead of FEEDBACK
            )

            response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text

            # Extract SQL from response
            sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            else:
                # Fallback: return the response as SQL
                return response.strip()

        except Exception as e:
            logger.warning(f"Failed to revise SQL: {e}")
            return sql

    def _generate_unit_tests(self, question: str, sql: str) -> List[str]:
        """Generate unit tests for the SQL query"""

        # Use baseline unit test template
        template = template_generate_unit_tests()

        try:
            prompt = template.format(
                QUESTION=question,
                HINT="",  # Add missing HINT parameter
                SQL=sql,
                UNIT_TEST_CAP=self.config.ut_unit_test_count,
                DATABASE_SCHEMA="",  # Not used in this template
                CANDIDATE_QUERIES=sql  # Use the SQL as candidate query
            )

            response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text

            # Extract list from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                tests_str = match.group(0)
                try:
                    tests = ast.literal_eval(tests_str)
                except Exception:
                    tests = eval(tests_str)
                return tests if isinstance(tests, list) else []
            return []

        except Exception as e:
            logger.warning(f"Failed to generate unit tests: {e}")
            return []

    def _evaluate_sql(self, question: str, sql: str, unit_tests: List[str]) -> Dict[str, Any]:
        """Evaluate SQL query using unit tests"""

        if not unit_tests:
            return {"score": 0.5, "feedback": "No unit tests available"}

        # Use baseline evaluate template
        template = template_evaluate()

        try:
            unit_tests_text = "\n".join([f"{i + 1}. {test}" for i, test in enumerate(unit_tests)])

            prompt = template.format(
                DATABASE_SCHEMA="",  # Not used in this template
                QUESTION=question,
                HINT="",  # Not used in this template
                CANDIDATE_RESPONSES=sql,
                UNIT_TEST=unit_tests_text
            )

            response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text

            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    result = json.loads(json_str)
                except Exception:
                    result = eval(json_str)
                return result if isinstance(result, dict) else {"score": 0.5, "feedback": "Invalid evaluation result"}
            return {"score": 0.5, "feedback": "Could not parse evaluation result"}

        except Exception as e:
            logger.warning(f"Failed to evaluate SQL: {e}")
            return {"score": 0.5, "feedback": f"Evaluation failed: {e}"}

    def _select_best_sql(self, candidates: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]) -> str:
        """Select the best SQL query based on evaluations"""
        if not candidates:
            return ""

        if not evaluations:
            # If no evaluations, return the first candidate
            return candidates[0]["SQL"]

        # Find the candidate with the highest evaluation score
        best_score = -1
        best_sql = candidates[0]["SQL"]

        for i, evaluation in enumerate(evaluations):
            score = evaluation.get("score", 0)
            if score > best_score:
                best_score = score
                best_sql = candidates[i]["SQL"]

        return best_sql

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ):
        """Execute the CHESS-SQL pipeline for a single item"""
        logger.info(f"CHESSGenerator 开始处理样本 {item}")

        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get("db_id", "")
        evidence = row.get('evidence', '')

        logger.debug(f"处理问题: {question[:100]}... (数据库: {db_id}, 类型: {db_type})")

        # Step 1: Load and process schema
        logger.debug("开始处理数据库模式...")
        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"从实例模式路径加载模式: {instance_schema_path}")

            if schema is None:
                logger.debug("从数据集获取数据库模式")
                schema = self.dataset.get_db_schema(item)

            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema = parse_schema_from_df(schema)
        else:
            raise Exception("Failed to load a valid database schema for the sample!")

        logger.debug("数据库模式处理完成")

        # Step 2: Information Retrieval (IR)
        logger.debug("开始信息检索...")
        keywords = self._extract_keywords(question)
        context = self._retrieve_context(question, schema, keywords)
        logger.debug(f"提取关键词: {keywords[:10]}...")

        # Step 3: Candidate Generation (CG)
        logger.debug("开始候选SQL生成...")
        # Build lightweight hints to mimic baseline CoT aggregation
        built_evidence = self._build_evidence(question, context, keywords, evidence)
        candidates = self._generate_candidate_sql(question, context, built_evidence)
        logger.debug(f"生成 {len(candidates)} 个候选SQL")

        if not candidates:
            logger.warning("没有生成任何候选SQL")
            pred_sql = ""
        else:
            # Step 4: Unit Testing (UT)
            logger.debug("开始单元测试生成...")
            unit_tests = self._generate_unit_tests(question, candidates[0]["SQL"])
            logger.debug(f"生成 {len(unit_tests)} 个单元测试")

            # Step 5: Evaluation
            logger.debug("开始SQL评估...")
            evaluations = []
            for candidate in candidates:
                evaluation = self._evaluate_sql(question, candidate["SQL"], unit_tests)
                evaluations.append(evaluation)
            logger.debug("SQL评估完成")

            # Step 6: Select best SQL
            pred_sql = self._select_best_sql(candidates, evaluations)
            logger.debug(f"选择最佳SQL: {pred_sql[:100]}...")

            # Step 7: Optional SQL revision based on feedback
            if evaluations and evaluations[0].get("feedback"):
                logger.debug("开始SQL修订...")
                revised_sql = self._revise_sql(question, context, pred_sql, evaluations[0]["feedback"])
                if revised_sql and revised_sql != pred_sql:
                    pred_sql = revised_sql
                    logger.debug("SQL修订完成")

        # SQL post-process
        if self.sql_post_process_function and pred_sql:
            pred_sql = self.sql_post_process_function(pred_sql, self.dataset)

        # Save results
        if self.is_save:
            instance_id = row.get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.name}_{instance_id}.sql"

            save_dataset(pred_sql, new_data_source=save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))
            logger.debug(f"SQL 已保存到: {save_path}")

        logger.info(f"CHESSGenerator 样本 {item} 处理完成")
        return pred_sql