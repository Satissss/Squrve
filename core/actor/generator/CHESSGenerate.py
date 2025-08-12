from llama_index.core.llms.llm import LLM
from typing import Union, List, Callable, Dict, Optional, Any
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger
import json
import re
from dataclasses import dataclass

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
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

    # Embed templates as class constants for isolation
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

【Answer】
'''

    REVISE_TEMPLATE = '''You are an expert SQL developer. Revise the following SQL query based on the feedback provided.

Database Schema:
{DATABASE_SCHEMA}

Question: {QUESTION}

Original SQL: {SQL}

Feedback: {FEEDBACK}

Please provide the corrected SQL query. Output only the SQL query without any explanations.'''

    UNIT_TEST_TEMPLATE = '''Generate natural language unit tests for the following SQL query.

Question: {QUESTION}
SQL: {SQL}

Generate {UNIT_TEST_COUNT} unit tests that can be used to validate the SQL query.
Each test should be a natural language question that the SQL should answer correctly.

Return the tests as a Python list of strings.'''

    EVALUATE_TEMPLATE = '''Evaluate the following SQL query based on the unit tests.

Question: {QUESTION}
SQL: {SQL}

Unit Tests:
{UNIT_TESTS}

Evaluate if the SQL query correctly answers the unit tests.
Return a JSON object with 'score' (0-1) and 'feedback' (string).'''

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

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from the question using LLM"""
        prompt = f"""Extract key entities and concepts from the following question that are relevant for SQL query generation. 
        Focus on table names, column names, conditions, and operations.

        Question: {question}

        Return only a Python list of keywords, for example: ['customer', 'name', 'age', '>', '30']"""

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
        # Implement schema selection similar to SelectTables tool
        prompt = f"""Select relevant tables from the following schema based on the question.

Schema:
{schema}

Question: {question}

Please return only the filtered schema as a string, with each table on a new line."""
        try:
            response = self.llm.complete(prompt, temperature=self.config.ss_temperature).text
            # Parse and filter schema
            filtered_schema_lines = [line for line in response.split('\n') if line.strip()]
            return '\n'.join(filtered_schema_lines)
        except Exception as e:
            logger.warning(f"Failed to select schema: {e}")
            return schema

    def _generate_candidate_sql(self, question: str, schema: str, evidence: str = "") -> List[Dict[str, Any]]:
        """Generate candidate SQL queries using the recursive divide-and-conquer approach"""

        # Use CANDIDATE_TEMPLATE
        template = self.CANDIDATE_TEMPLATE

        candidates = []
        for i in range(self.config.cg_sampling_count):
            try:
                prompt = template.format(
                    DATABASE_SCHEMA=schema,
                    QUESTION=question,
                    HINT=evidence
                )

                response = self.llm.complete(prompt, temperature=self.config.cg_temperature).text

                # Extract SQL from response
                sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()
                    candidates.append({
                        "SQL": sql,
                        "chain_of_thought_reasoning": response,
                        "confidence": 0.8  # Default confidence
                    })
                else:
                    # Fallback: try to extract SQL from the response
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
                logger.warning(f"Failed to generate candidate {i}: {e}")
                continue

        return candidates

    def _revise_sql(self, question: str, schema: str, sql: str, feedback: str = "") -> str:
        """Revise SQL query based on feedback"""

        # Use REVISE_TEMPLATE
        template = self.REVISE_TEMPLATE

        try:
            prompt = template.format(
                DATABASE_SCHEMA=schema,
                QUESTION=question,
                SQL=sql,
                FEEDBACK=feedback
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

        # Use UNIT_TEST_TEMPLATE
        template = self.UNIT_TEST_TEMPLATE

        try:
            prompt = template.format(
                QUESTION=question,
                SQL=sql,
                UNIT_TEST_COUNT=self.config.ut_unit_test_count
            )

            response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text

            # Extract list from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                tests_str = match.group(0)
                tests = eval(tests_str)  # In production, use ast.literal_eval
                return tests if isinstance(tests, list) else []
            return []

        except Exception as e:
            logger.warning(f"Failed to generate unit tests: {e}")
            return []

    def _evaluate_sql(self, question: str, sql: str, unit_tests: List[str]) -> Dict[str, Any]:
        """Evaluate SQL query using unit tests"""

        if not unit_tests:
            return {"score": 0.5, "feedback": "No unit tests available"}

        # Use EVALUATE_TEMPLATE
        template = self.EVALUATE_TEMPLATE

        try:
            unit_tests_text = "\n".join([f"{i + 1}. {test}" for i, test in enumerate(unit_tests)])

            prompt = template.format(
                QUESTION=question,
                SQL=sql,
                UNIT_TESTS=unit_tests_text
            )

            response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text

            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = eval(match.group(0))  # In production, use json.loads
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
        candidates = self._generate_candidate_sql(question, context, evidence)
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