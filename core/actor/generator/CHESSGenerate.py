from llama_index.core.llms.llm import LLM
from typing import Union, List, Callable, Dict, Optional, Any
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger
import json
import re
import ast
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)


class DatabaseType(Enum):
    """Database type enumeration"""
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


@dataclass
class CHESSConfig:
    """Configuration for CHESS-SQL method with enhanced validation"""
    # Information Retriever settings
    ir_engine: str = "gpt-4o-mini"
    ir_temperature: float = field(default=0.2)
    ir_top_k: int = field(default=5)

    # Candidate Generator settings
    cg_engine: str = "gpt-4o-mini"
    cg_temperature: float = field(default=0.5)
    cg_sampling_count: int = field(default=10)

    # Unit Tester settings
    ut_engine: str = "gpt-4o-mini"
    ut_temperature: float = field(default=0.8)
    ut_unit_test_count: int = field(default=20)

    # Schema Selector settings
    use_schema_selector: bool = field(default=False)
    ss_engine: str = "gpt-4o-mini"
    ss_temperature: float = field(default=0.2)

    # Retry and timeout settings
    max_retries: int = field(default=3)
    timeout_seconds: float = field(default=30.0)

    def __post_init__(self):
        """Validate configuration parameters"""
        self._validate_temperature("ir_temperature", self.ir_temperature)
        self._validate_temperature("cg_temperature", self.cg_temperature)
        self._validate_temperature("ut_temperature", self.ut_temperature)
        self._validate_temperature("ss_temperature", self.ss_temperature)

        if self.ir_top_k <= 0:
            raise ValueError("ir_top_k must be positive")
        if self.cg_sampling_count <= 0:
            raise ValueError("cg_sampling_count must be positive")
        if self.ut_unit_test_count <= 0:
            raise ValueError("ut_unit_test_count must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @staticmethod
    def _validate_temperature(name: str, value: float) -> None:
        """Validate temperature parameter"""
        if not (0.0 <= value <= 2.0):
            raise ValueError(f"{name} must be between 0.0 and 2.0, got {value}")


class CHESSGenerator(BaseGenerator):
    """CHESS-SQL: Contextual Harnessing for Efficient SQL Synthesis

    A multi-agent framework for efficient and scalable SQL synthesis, comprising:
    1. Information Retriever (IR): Extracts relevant data
    2. Schema Selector (SS): Prunes large schemas (optional)
    3. Candidate Generator (CG): Generates high-quality candidates and refines queries
    4. Unit Tester (UT): Validates queries through LLM-based natural language unit tests
    """

    NAME = "CHESSGenerator"

    # Template constants with improved formatting
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
        """Initialize CHESS Generator with enhanced validation"""
        super().__init__()

        self.dataset = dataset
        self.llm = self._validate_llm(llm)
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.config = config or CHESSConfig()
        self.sql_post_process_function = sql_post_process_function

        # Initialize database configuration
        self.db_path = self._resolve_db_path(db_path)
        self.credential = self._resolve_credential(credential)

        # Ensure save directory exists
        if self.is_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _validate_llm(self, llm: Optional[LLM]) -> Optional[LLM]:
        """Validate LLM instance"""
        if llm is not None and not isinstance(llm, LLM):
            raise TypeError("llm must be an instance of LLM or None")
        return llm

    def _resolve_db_path(self, db_path: Optional[Union[str, PathLike]]) -> Optional[Path]:
        """Resolve database path with fallback to dataset"""
        if db_path is not None:
            return Path(db_path)
        elif self.dataset is not None:
            dataset_db_path = getattr(self.dataset, 'db_path', None)
            return Path(dataset_db_path) if dataset_db_path else None
        return None

    def _resolve_credential(self, credential: Optional[Dict]) -> Optional[Dict]:
        """Resolve credential with fallback to dataset"""
        if credential is not None:
            return credential
        elif self.dataset is not None:
            return getattr(self.dataset, 'credential', None)
        return None

    @contextmanager
    def _llm_retry_context(self, operation: str):
        """Context manager for LLM operations with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                yield attempt
                break
            except Exception as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Failed {operation} after {self.config.max_retries} retries: {e}")
                    raise
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {operation}: {e}. Retrying...")

    def _safe_extract_list(self, text: str, fallback: List = None) -> List:
        """Safely extract Python list from text response"""
        if fallback is None:
            fallback = []

        try:
            # First try to find list pattern
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                list_str = match.group(0)
                # Use ast.literal_eval for safe evaluation
                result = ast.literal_eval(list_str)
                return result if isinstance(result, list) else fallback
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse list from response: {e}")

        return fallback

    def _safe_extract_json(self, text: str, fallback: Dict = None) -> Dict:
        """Safely extract JSON from text response"""
        if fallback is None:
            fallback = {"score": 0.5, "feedback": "Could not parse evaluation"}

        try:
            # First try to find JSON pattern
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                result = json.loads(json_str)
                return result if isinstance(result, dict) else fallback
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")

        return fallback

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from the question using LLM with retry logic"""
        prompt = f"""Extract key entities and concepts from the following question that are relevant for SQL query generation. 
        Focus on table names, column names, conditions, and operations.

        Question: {question}

        Return only a Python list of keywords, for example: ['customer', 'name', 'age', '>', '30']"""

        with self._llm_retry_context("keyword extraction"):
            try:
                response = self.llm.complete(prompt, temperature=self.config.ir_temperature).text
                return self._safe_extract_list(response, fallback=[])
            except Exception as e:
                logger.warning(f"Failed to extract keywords: {e}")
                return []

    def _retrieve_context(self, question: str, schema: str, keywords: List[str]) -> str:
        """Retrieve relevant context from schema based on keywords with improved filtering"""
        if not keywords:
            return schema

        # Enhanced keyword-based filtering
        relevant_lines = set()  # Use set to avoid duplicates
        schema_lines = schema.split('\n')

        # Preprocess keywords for better matching
        processed_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]

        for line in schema_lines:
            line_lower = line.lower()
            # More sophisticated matching
            if any(keyword in line_lower for keyword in processed_keywords):
                relevant_lines.add(line)
                # Also add context lines (table definitions, etc.)
                line_idx = schema_lines.index(line)
                # Add surrounding context
                for i in range(max(0, line_idx - 2), min(len(schema_lines), line_idx + 3)):
                    if schema_lines[i].strip():
                        relevant_lines.add(schema_lines[i])

        if relevant_lines:
            return '\n'.join(sorted(relevant_lines, key=lambda x: schema_lines.index(x) if x in schema_lines else 0))
        return schema

    def _select_schema(self, schema: str, question: str) -> str:
        """Select relevant schema tables with improved error handling"""
        if not self.config.use_schema_selector:
            return schema

        prompt = f"""Select relevant tables from the following schema based on the question.

Schema:
{schema}

Question: {question}

Please return only the filtered schema as a string, with each table on a new line."""

        with self._llm_retry_context("schema selection"):
            try:
                response = self.llm.complete(prompt, temperature=self.config.ss_temperature).text
                filtered_lines = [line.strip() for line in response.split('\n') if line.strip()]
                return '\n'.join(filtered_lines) if filtered_lines else schema
            except Exception as e:
                logger.warning(f"Failed to select schema: {e}")
                return schema

    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """Extract SQL query from LLM response with multiple extraction strategies"""
        # Strategy 1: Look for FINAL_ANSWER tags
        sql_match = re.search(r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # Strategy 2: Look for SQL blocks
        sql_block_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            return sql_block_match.group(1).strip()

        # Strategy 3: Look for lines starting with SELECT
        lines = response.split('\n')
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                return stripped_line

        return None

    def _generate_candidate_sql(self, question: str, schema: str, evidence: str = "") -> List[Dict[str, Any]]:
        """Generate candidate SQL queries with enhanced error handling"""
        candidates = []
        successful_generations = 0

        for i in range(self.config.cg_sampling_count):
            with self._llm_retry_context(f"candidate generation {i + 1}"):
                try:
                    prompt = self.CANDIDATE_TEMPLATE.format(
                        DATABASE_SCHEMA=schema,
                        QUESTION=question,
                        HINT=evidence or "No additional evidence provided."
                    )

                    response = self.llm.complete(prompt, temperature=self.config.cg_temperature).text
                    sql = self._extract_sql_from_response(response)

                    if sql:
                        candidates.append({
                            "SQL": sql,
                            "chain_of_thought_reasoning": response,
                            "confidence": self._calculate_confidence(response, sql),
                            "generation_id": i
                        })
                        successful_generations += 1
                    else:
                        logger.warning(f"Could not extract SQL from candidate {i + 1}")

                except Exception as e:
                    logger.warning(f"Failed to generate candidate {i + 1}: {e}")
                    continue

        logger.info(f"Successfully generated {successful_generations}/{self.config.cg_sampling_count} SQL candidates")
        return candidates

    def _calculate_confidence(self, response: str, sql: str) -> float:
        """Calculate confidence score based on response quality indicators"""
        confidence = 0.5  # Base confidence

        # Increase confidence for structured responses
        if '<FINAL_ANSWER>' in response:
            confidence += 0.2

        # Increase confidence for longer, more detailed reasoning
        if len(response) > 200:
            confidence += 0.1

        # Increase confidence for properly formatted SQL
        if sql and any(keyword in sql.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN']):
            confidence += 0.2

        return min(confidence, 1.0)

    def _revise_sql(self, question: str, schema: str, sql: str, feedback: str = "") -> str:
        """Revise SQL query based on feedback with improved error handling"""
        if not feedback:
            return sql

        with self._llm_retry_context("SQL revision"):
            try:
                prompt = self.REVISE_TEMPLATE.format(
                    DATABASE_SCHEMA=schema,
                    QUESTION=question,
                    SQL=sql,
                    FEEDBACK=feedback
                )

                response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text
                revised_sql = self._extract_sql_from_response(response)

                return revised_sql if revised_sql else sql

            except Exception as e:
                logger.warning(f"Failed to revise SQL: {e}")
                return sql

    def _generate_unit_tests(self, question: str, sql: str) -> List[str]:
        """Generate unit tests for the SQL query with improved error handling"""
        with self._llm_retry_context("unit test generation"):
            try:
                prompt = self.UNIT_TEST_TEMPLATE.format(
                    QUESTION=question,
                    SQL=sql,
                    UNIT_TEST_COUNT=self.config.ut_unit_test_count
                )

                response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text
                return self._safe_extract_list(response, fallback=[])

            except Exception as e:
                logger.warning(f"Failed to generate unit tests: {e}")
                return []

    def _evaluate_sql(self, question: str, sql: str, unit_tests: List[str]) -> Dict[str, Any]:
        """Evaluate SQL query using unit tests with improved error handling"""
        if not unit_tests:
            return {"score": 0.5, "feedback": "No unit tests available for evaluation"}

        with self._llm_retry_context("SQL evaluation"):
            try:
                unit_tests_text = "\n".join([f"{i + 1}. {test}" for i, test in enumerate(unit_tests)])

                prompt = self.EVALUATE_TEMPLATE.format(
                    QUESTION=question,
                    SQL=sql,
                    UNIT_TESTS=unit_tests_text
                )

                response = self.llm.complete(prompt, temperature=self.config.ut_temperature).text
                return self._safe_extract_json(response)

            except Exception as e:
                logger.warning(f"Failed to evaluate SQL: {e}")
                return {"score": 0.5, "feedback": f"Evaluation failed: {str(e)}"}

    def _select_best_sql(self, candidates: List[Dict[str, Any]], evaluations: List[Dict[str, Any]]) -> str:
        """Select the best SQL query based on evaluations and confidence scores"""
        if not candidates:
            return ""

        if not evaluations or len(evaluations) != len(candidates):
            # Fallback to confidence-based selection
            return max(candidates, key=lambda x: x.get("confidence", 0))["SQL"]

        # Weighted scoring: evaluation score (70%) + confidence (30%)
        best_score = -1
        best_sql = candidates[0]["SQL"]

        for i, (candidate, evaluation) in enumerate(zip(candidates, evaluations)):
            eval_score = evaluation.get("score", 0)
            confidence_score = candidate.get("confidence", 0)
            weighted_score = 0.7 * eval_score + 0.3 * confidence_score

            if weighted_score > best_score:
                best_score = weighted_score
                best_sql = candidate["SQL"]

        logger.info(f"Selected SQL with weighted score: {best_score:.3f}")
        return best_sql

    def _load_and_process_schema(self, item: int, schema: Union[str, PathLike, Dict, List]) -> str:
        """Load and process database schema with comprehensive error handling"""
        logger.debug("Processing database schema...")

        # Load schema from various sources
        processed_schema = None

        if isinstance(schema, (str, PathLike)):
            processed_schema = load_dataset(schema)
        elif schema is not None:
            processed_schema = schema
        else:
            # Try to load from dataset
            row = self.dataset[item]
            instance_schema_path = row.get("instance_schemas")

            if instance_schema_path:
                processed_schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded schema from instance path: {instance_schema_path}")
            else:
                processed_schema = self.dataset.get_db_schema(item)
                logger.debug("Loaded schema from dataset")

        if processed_schema is None:
            raise ValueError("Failed to load a valid database schema for the sample!")

        # Normalize schema format
        if isinstance(processed_schema, dict):
            processed_schema = single_central_process(processed_schema)
        elif isinstance(processed_schema, list):
            processed_schema = pd.DataFrame(processed_schema)

        if isinstance(processed_schema, pd.DataFrame):
            schema_str = parse_schema_from_df(processed_schema)
        elif isinstance(processed_schema, str):
            schema_str = processed_schema
        else:
            raise ValueError(f"Unsupported schema type: {type(processed_schema)}")

        logger.debug("Database schema processing completed")
        return schema_str

    def _save_results(self, item: int, pred_sql: str) -> None:
        """Save prediction results with proper error handling"""
        if not self.is_save or not pred_sql:
            return

        try:
            row = self.dataset[item]
            instance_id = row.get("instance_id", f"item_{item}")

            # Create save path with dataset organization
            save_path = self.save_dir
            if hasattr(self.dataset, 'dataset_index') and self.dataset.dataset_index:
                save_path = save_path / str(self.dataset.dataset_index)
                save_path.mkdir(parents=True, exist_ok=True)

            final_path = save_path / f"{self.NAME}_{instance_id}.sql"

            # Save the SQL
            save_dataset(pred_sql, new_data_source=final_path)

            # Update dataset with prediction path
            if hasattr(self.dataset, 'setitem'):
                self.dataset.setitem(item, "pred_sql", str(final_path))

            logger.debug(f"SQL saved to: {final_path}")

        except Exception as e:
            logger.error(f"Failed to save SQL results: {e}")

    def act(
            self,
            item: int,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ) -> str:
        """Execute the CHESS-SQL pipeline for a single item with comprehensive error handling"""
        try:
            logger.info(f"CHESSGenerator processing item {item}")

            if self.dataset is None:
                raise ValueError("Dataset is required for processing")

            row = self.dataset[item]
            question = row.get('question', '')
            if not question:
                raise ValueError("Question is required")

            db_type = row.get('db_type', 'sqlite')
            db_id = row.get("db_id", "unknown")
            evidence = row.get('evidence', '')

            logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

            # Step 1: Load and process schema
            schema_str = self._load_and_process_schema(item, schema)

            # Step 2: Information Retrieval (IR)
            logger.debug("Starting information retrieval...")
            keywords = self._extract_keywords(question)
            context = self._retrieve_context(question, schema_str, keywords)
            logger.debug(f"Extracted keywords: {keywords[:10]}...")

            # Step 3: Schema Selection (optional)
            if self.config.use_schema_selector:
                context = self._select_schema(context, question)

            # Step 4: Candidate Generation (CG)
            logger.debug("Starting SQL candidate generation...")
            candidates = self._generate_candidate_sql(question, context, evidence)

            if not candidates:
                logger.warning("No SQL candidates generated")
                return ""

            logger.debug(f"Generated {len(candidates)} SQL candidates")

            # Step 5: Unit Testing (UT) - use best candidate for test generation
            best_candidate_sql = max(candidates, key=lambda x: x.get("confidence", 0))["SQL"]
            logger.debug("Generating unit tests...")
            unit_tests = self._generate_unit_tests(question, best_candidate_sql)
            logger.debug(f"Generated {len(unit_tests)} unit tests")

            # Step 6: Evaluation
            logger.debug("Evaluating SQL candidates...")
            evaluations = []
            for candidate in candidates:
                evaluation = self._evaluate_sql(question, candidate["SQL"], unit_tests)
                evaluations.append(evaluation)

            # Step 7: Select best SQL
            pred_sql = self._select_best_sql(candidates, evaluations)
            logger.debug(f"Selected best SQL: {pred_sql[:100]}...")

            # Step 8: Optional SQL revision
            if evaluations and evaluations[0].get("feedback") and evaluations[0].get("score", 0) < 0.7:
                logger.debug("Attempting SQL revision...")
                revised_sql = self._revise_sql(question, context, pred_sql, evaluations[0]["feedback"])
                if revised_sql and revised_sql != pred_sql:
                    pred_sql = revised_sql
                    logger.debug("SQL revision completed")

            # Step 9: Post-processing
            if self.sql_post_process_function and pred_sql:
                pred_sql = self.sql_post_process_function(pred_sql, self.dataset)

            # Step 10: Save results
            self._save_results(item, pred_sql)

            logger.info(f"CHESSGenerator completed processing item {item}")
            return pred_sql

        except Exception as e:
            logger.error(f"CHESSGenerator failed to process item {item}: {e}")
            return ""

    @property
    def name(self) -> str:
        """Get generator name"""
        return self.NAME

    def __repr__(self) -> str:
        """String representation of the generator"""
        return f"CHESSGenerator(config={self.config}, dataset={bool(self.dataset)}, llm={bool(self.llm)})"