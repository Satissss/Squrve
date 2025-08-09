import os
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
from loguru import logger
from llama_index.core.llms.llm import LLM

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.utils import parse_schema_from_df
from core.db_connect import execute_sql


@dataclass
class SQLResult:
    """Container for SQL execution results."""
    sql: str
    result: str
    is_success: bool = False
    error_message: Optional[str] = None


class ReFoRCEGenerator(BaseGenerator):
    """
    ReFoRCE SQL Generator implementing retrieval-focused column exploration
    and iterative self-refinement for robust SQL generation.
    """

    OUTPUT_NAME = "pred_sql"
    NAME = "ReFoRCEGenerator"

    # Constants
    EMPTY_RESULT = "No data found for the specified query."
    DEFAULT_DB_TYPE = "sqlite"
    DEFAULT_CSV_MAX_LEN = 500
    DEFAULT_MAX_ITER = 5
    DEFAULT_MAX_TRY = 3
    DEFAULT_EXPLORATION_QUERIES = 10
    DEFAULT_LIMIT_ROWS = 20

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, os.PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            do_column_exploration: bool = True,
            do_self_refinement: bool = True,
            max_iter: int = DEFAULT_MAX_ITER,
            max_try: int = DEFAULT_MAX_TRY,
            csv_max_len: int = DEFAULT_CSV_MAX_LEN,
            db_path: Optional[Union[str, os.PathLike]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ):
        """Initialize ReFoRCE Generator with configuration parameters."""
        super().__init__(**kwargs)

        # Core components
        self.dataset = dataset
        self.llm = llm

        # Configuration
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.use_external = use_external
        self.use_few_shot = use_few_shot
        self.do_column_exploration = do_column_exploration
        self.do_self_refinement = do_self_refinement

        # Parameters
        self.max_iter = max_iter
        self.max_try = max_try
        self.csv_max_len = csv_max_len

        # Database configuration
        self.db_path = self._resolve_db_path(db_path)
        self.credential = self._resolve_credential(credential)

        # Validation
        self._validate_initialization()

    def _resolve_db_path(self, db_path: Optional[Union[str, os.PathLike]]) -> Optional[Path]:
        """Resolve database path from parameters or dataset."""
        if db_path:
            return Path(db_path)
        if self.dataset and hasattr(self.dataset, 'db_path') and self.dataset.db_path:
            return Path(self.dataset.db_path)
        return None

    def _resolve_credential(self, credential: Optional[Dict]) -> Optional[Dict]:
        """Resolve database credentials from parameters or dataset."""
        if credential:
            return credential
        if self.dataset and hasattr(self.dataset, 'credential'):
            return self.dataset.credential
        return None

    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.max_try < 1:
            raise ValueError("max_try must be at least 1")
        if self.csv_max_len < 1:
            raise ValueError("csv_max_len must be at least 1")

    def load_external_knowledge(self, external_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Load external knowledge if available and valid."""
        if not external_path or not self.use_external:
            return None

        try:
            external_data = load_dataset(external_path)
            if external_data and len(str(external_data)) > 50:
                return f"####[External Prior Knowledge]:\n{external_data}"
        except Exception as e:
            logger.warning(f"Failed to load external knowledge from {external_path}: {e}")

        return None

    def parse_sql_from_response(self, response: str) -> List[str]:
        """
        Parse SQL queries from LLM response, extracting from ```sql code blocks.

        Args:
            response: LLM response text

        Returns:
            List of parsed SQL queries
        """
        if not response or not isinstance(response, str):
            return []

        sqls = []
        # Match SQL code blocks with optional descriptions
        pattern = r'```sql\s*(?:--Description:.*?\n)?(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            sql = match.strip()
            if sql and not sql.startswith('--'):  # Exclude comment-only blocks
                sqls.append(sql)

        return sqls

    def _create_exploration_prompt(self, db_type: str, schema_str: str) -> str:
        """Create prompt for database column exploration."""
        dialect_notes = {
            'postgresql': 'Use ILIKE for case-insensitive string matching.',
            'mysql': 'Use LIKE with appropriate collation for string matching.',
            'sqlite': 'Use LIKE (case-insensitive by default) for string matching.',
            'mssql': 'Use LIKE with COLLATE for case-insensitive matching.'
        }

        prompt_parts = [
            f"Write at most {self.DEFAULT_EXPLORATION_QUERIES} {db_type} SQL queries from simple to complex to understand values in related columns.",
            "Each query should be different and focus on exploring data patterns.",
            f"Use DISTINCT and LIMIT {self.DEFAULT_LIMIT_ROWS} rows for efficiency.",
            "Write annotations to describe each SQL in format ```sql\n--Description: <purpose>\n<sql_query>\n```.",
            dialect_notes.get(db_type.lower(), 'Use appropriate string matching for your database.'),
            "Focus on SELECT queries only. Do not query schema or data types.",
            f"\nDatabase schema:\n{schema_str}"
        ]

        return '\n'.join(prompt_parts)

    def _execute_sql_safely(self, sql: str, db_type: str, db_path: Union[str, Path],
                            credential: Optional[Dict]) -> SQLResult:
        """Execute SQL with proper error handling and result validation."""
        try:
            result = execute_sql(db_type, db_path, sql, credential)

            if isinstance(result, str):
                is_success = (result != self.EMPTY_RESULT and
                              'ERROR' not in result.upper() and
                              'EXCEPTION' not in result.upper())
                return SQLResult(
                    sql=sql,
                    result=result,
                    is_success=is_success,
                    error_message=None if is_success else result
                )
            else:
                # Handle non-string results (e.g., DataFrames)
                return SQLResult(
                    sql=sql,
                    result=str(result),
                    is_success=True
                )

        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return SQLResult(
                sql=sql,
                result="",
                is_success=False,
                error_message=str(e)
            )

    def _self_correct_sql(self, sql: str, error: str, simplify: bool = False) -> Optional[str]:
        """Attempt to correct a failed SQL query using LLM feedback."""
        if not self.llm:
            return None

        prompt_parts = [
            f"Input SQL:\n{sql}",
            f"Error information:\n{error}"
        ]

        if simplify:
            prompt_parts.append("Since the output is empty, please simplify conditions in the SQL.")

        prompt_parts.extend([
            "Please correct the SQL and provide reasoning for your changes.",
            "Output exactly one complete SQL query in ```sql``` format."
        ])

        try:
            response = self.llm.complete('\n'.join(prompt_parts)).text
            corrected_sqls = self.parse_sql_from_response(response)
            return corrected_sqls[0] if corrected_sqls else None
        except Exception as e:
            logger.warning(f"Self-correction failed: {e}")
            return None

    def _execute_exploration_sqls(self, sqls: List[str], db_type: str,
                                  db_path: Union[str, Path], credential: Optional[Dict]) -> List[Dict[str, str]]:
        """Execute exploration SQLs with error correction."""
        successful_results = []

        for i, sql in enumerate(sqls):
            sql_result = self._execute_sql_safely(sql, db_type, db_path, credential)

            if sql_result.is_success:
                successful_results.append({'sql': sql, 'res': sql_result.result})
            else:
                # Attempt correction
                corrected_sql = self._self_correct_sql(
                    sql,
                    sql_result.error_message or sql_result.result,
                    simplify=(sql_result.result == self.EMPTY_RESULT)
                )

                if corrected_sql:
                    corrected_result = self._execute_sql_safely(corrected_sql, db_type, db_path, credential)
                    if corrected_result.is_success:
                        successful_results.append({'sql': corrected_sql, 'res': corrected_result.result})
                        logger.debug(f"Successfully corrected SQL {i + 1}")

        return successful_results

    def exploration(self, question: str, schema_str: str, db_type: str,
                    db_path: Union[str, Path], credential: Optional[Dict]) -> str:
        """Perform database exploration to understand column values and patterns."""
        if not self.llm:
            logger.warning("LLM not available for exploration")
            return ""

        try:
            prompt = self._create_exploration_prompt(db_type, schema_str)
            response = self.llm.complete(prompt).text

            sqls = self.parse_sql_from_response(response)
            if not sqls:
                logger.warning("No exploration SQLs generated")
                return ""

            logger.debug(f"Generated {len(sqls)} exploration queries")

            results = self._execute_exploration_sqls(sqls, db_type, db_path, credential)

            # Format results for context
            exploration_info = []
            for result_dict in results:
                exploration_info.extend([
                    f"Query:\n{result_dict['sql']}",
                    f"Results:\n{result_dict['res']}\n"
                ])

            return '\n'.join(exploration_info)

        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return ""

    def _generate_sql_candidates(self, question: str, schema_str: str, pre_info: str,
                                 db_type: str, db_path: Union[str, Path],
                                 credential: Optional[Dict]) -> List[Tuple[str, str]]:
        """Generate and validate multiple SQL candidates through iterative refinement."""
        candidates = []
        error_history = []

        for iteration in range(self.max_iter):
            try:
                # Build prompt
                prompt_parts = [
                    f"Database schema:\n{schema_str}"
                ]

                if pre_info:
                    prompt_parts.append(f"Exploration results:\n{pre_info}")

                prompt_parts.extend([
                    f"Question: {question}",
                    f"Generate one accurate {db_type} SQL query.",
                    "Think step by step and ensure the query addresses the question completely."
                ])

                if error_history:
                    recent_errors = error_history[-3:]  # Last 3 errors
                    prompt_parts.append(f"Avoid these recent errors: {', '.join(recent_errors)}")

                response = self.llm.complete('\n'.join(prompt_parts)).text
                sqls = self.parse_sql_from_response(response)

                if not sqls:
                    logger.debug(f"No SQL parsed in iteration {iteration + 1}")
                    continue

                sql = sqls[0]
                sql_result = self._execute_sql_safely(sql, db_type, db_path, credential)

                if sql_result.is_success:
                    candidates.append((sql, sql_result.result))
                    logger.debug(f"Successful SQL in iteration {iteration + 1}")
                else:
                    error_msg = sql_result.error_message or sql_result.result
                    error_history.append(error_msg)

                    # Try correction
                    corrected_sql = self._self_correct_sql(sql, error_msg,
                                                           simplify=(sql_result.result == self.EMPTY_RESULT))
                    if corrected_sql:
                        corrected_result = self._execute_sql_safely(corrected_sql, db_type, db_path, credential)
                        if corrected_result.is_success:
                            candidates.append((corrected_sql, corrected_result.result))
                            logger.debug(f"Corrected SQL successful in iteration {iteration + 1}")

                # Early termination for repeated empty results
                if len(error_history) >= 4 and all(e == self.EMPTY_RESULT for e in error_history[-4:]):
                    logger.info("Early termination due to repeated empty results")
                    break

            except Exception as e:
                logger.warning(f"Error in iteration {iteration + 1}: {e}")
                continue

        return candidates

    def _select_best_sql(self, candidates: List[Tuple[str, str]]) -> Optional[str]:
        """Select the best SQL from candidates using result-based voting."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][0]

        # Group candidates by results
        result_groups = defaultdict(list)
        for sql, result in candidates:
            result_groups[result].append(sql)

        # Count occurrences of each result
        result_counts = Counter({result: len(sql_list) for result, sql_list in result_groups.items()})
        most_common = result_counts.most_common()

        # Handle ties by random selection from top candidates
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            top_results = [result for result, count in most_common if count == most_common[0][1]]
            selected_result = random.choice(top_results)
        else:
            selected_result = most_common[0][0]

        # Return first SQL from selected result group
        return result_groups[selected_result][0]

    def self_refine(self, question: str, schema_str: str, pre_info: str, db_type: str,
                    db_path: Union[str, Path], credential: Optional[Dict]) -> Optional[str]:
        """Generate SQL through iterative self-refinement with candidate voting."""
        if not self.llm:
            logger.warning("LLM not available for self-refinement")
            return None

        try:
            candidates = self._generate_sql_candidates(question, schema_str, pre_info,
                                                       db_type, db_path, credential)
            return self._select_best_sql(candidates)
        except Exception as e:
            logger.error(f"Self-refinement failed: {e}")
            return None

    def _generate_fallback_sql(self, question: str, schema_str: str, db_type: str) -> str:
        """Generate a basic SQL query as fallback when main generation fails."""
        if not self.llm:
            return "/* No LLM available for SQL generation */"

        try:
            prompt = f"Generate a {db_type} SQL query for: {question}\nDatabase schema:\n{schema_str}\nOutput only the SQL in ```sql``` format."
            response = self.llm.complete(prompt).text
            sqls = self.parse_sql_from_response(response)
            return sqls[0] if sqls else "/* Failed to generate SQL */"
        except Exception as e:
            logger.error(f"Fallback SQL generation failed: {e}")
            return f"/* Fallback generation error: {e} */"

    def _resolve_database_config(self, item: int) -> Tuple[str, Union[str, Path], Optional[Dict]]:
        """Resolve database configuration for a specific item."""
        row = self.dataset[item]

        db_type = row.get('db_type', self.DEFAULT_DB_TYPE)
        db_id = row.get("db_id")

        # Resolve database path
        if self.db_path and db_type == 'sqlite' and db_id:
            db_path = self.db_path / f"{db_id}.sqlite"
        else:
            db_path = row.get('db_path', self.db_path)

        credential = self.credential or row.get('credential')

        return db_type, db_path, credential

    def _process_schema(self, item: int, schema: Optional[Union[str, os.PathLike, Dict, List]]) -> str:
        """Process and normalize schema from various input formats."""
        # Try provided schema first
        if schema is not None:
            if isinstance(schema, (str, os.PathLike)):
                try:
                    schema = load_dataset(schema)
                except Exception as e:
                    logger.warning(f"Failed to load schema from path: {e}")
                    schema = None

        # Fallback to dataset schema
        if schema is None:
            row = self.dataset[item]
            instance_schema_path = row.get("instance_schemas")

            if instance_schema_path:
                try:
                    schema = load_dataset(instance_schema_path)
                    logger.debug(f"Loaded schema from instance path: {instance_schema_path}")
                except Exception as e:
                    logger.warning(f"Failed to load instance schema: {e}")

            # Final fallback to dataset method
            if schema is None:
                try:
                    schema = self.dataset.get_db_schema(item)
                except Exception as e:
                    logger.error(f"Failed to get schema from dataset: {e}")
                    raise ValueError("Failed to load database schema") from e

        # Normalize schema format
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            return parse_schema_from_df(schema)
        elif isinstance(schema, str):
            return schema
        else:
            return str(schema)

    def _save_result(self, item: int, pred_sql: str) -> None:
        """Save the generated SQL result."""
        if not self.is_save:
            return

        try:
            row = self.dataset[item]
            instance_id = row.get("instance_id", str(item))

            # Create save path
            save_path = self.save_dir
            if self.dataset.dataset_index:
                save_path = save_path / str(self.dataset.dataset_index)

            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / f"{self.NAME}_{instance_id}.sql"

            # Save SQL
            save_dataset(pred_sql, new_data_source=save_file)

            # Update dataset
            self.dataset.setitem(item, "pred_sql", str(save_file))

            logger.debug(f"SQL saved to: {save_file}")

        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def act(
            self,
            item: int,
            schema: Optional[Union[str, os.PathLike, Dict, List]] = None,
            schema_links: Optional[Union[str, List[str]]] = None,
            **kwargs
    ) -> str:
        """
        Generate SQL for a given dataset item using ReFoRCE methodology.

        Args:
            item: Dataset item index
            schema: Database schema (optional)
            schema_links: Schema relationship information (optional)
            **kwargs: Additional arguments

        Returns:
            Generated SQL query string

        Raises:
            ValueError: If required components are missing or invalid
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for ReFoRCEGenerator")
        if self.llm is None:
            raise ValueError("LLM must be provided for ReFoRCEGenerator")

        logger.info(f"Processing item {item} with ReFoRCEGenerator")

        try:
            # Get question and database configuration
            row = self.dataset[item]
            question = row.get('question', '')
            if not question:
                raise ValueError(f"No question found for item {item}")

            db_type, db_path, credential = self._resolve_database_config(item)
            logger.debug(f"Database config - Type: {db_type}, Path: {db_path}")

            # Load external knowledge
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                question = f"{question}\n{external_knowledge}"
                logger.debug("Added external knowledge to question")

            # Process schema
            schema_str = self._process_schema(item, schema)
            if schema_links:
                if isinstance(schema_links, list):
                    schema_links = ', '.join(schema_links)
                schema_str += f"\nSchema Links: {schema_links}"

            logger.debug("Schema processed successfully")

            # Column exploration phase
            pre_info = ""
            if self.do_column_exploration:
                logger.debug("Starting column exploration")
                pre_info = self.exploration(question, schema_str, db_type, db_path, credential)
                if pre_info:
                    logger.debug("Column exploration completed successfully")

            # SQL generation with self-refinement
            pred_sql = None
            if self.do_self_refinement:
                logger.debug("Starting self-refinement process")
                pred_sql = self.self_refine(question, schema_str, pre_info, db_type, db_path, credential)
                if pred_sql:
                    logger.debug("Self-refinement generated SQL successfully")

            # Fallback generation
            if pred_sql is None:
                logger.warning("Main generation failed, using fallback")
                pred_sql = self._generate_fallback_sql(question, schema_str, db_type)

            # Save result
            self._save_result(item, pred_sql)

            logger.info(f"Successfully processed item {item}")
            return pred_sql

        except Exception as e:
            logger.error(f"Failed to process item {item}: {e}")
            raise