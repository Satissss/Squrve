import concurrent.futures
from typing import Union, List, Optional, Dict, Tuple
from pathlib import Path
from loguru import logger
from llama_index.core.llms.llm import LLM

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.db_connect import get_sql_exec_result
from core.utils import sql_clean, parse_schema_from_df
import pandas as pd


class LinkAlignOptimizer(BaseOptimizer):
    """Optimizer that debugs and refines SQL queries using experience-based and feedback-based methods from LinkAlign."""

    NAME = "LinkAlignOptimizer"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            use_feedback_debug: bool = True,
            debug_turn_n: int = 2,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.use_feedback_debug = use_feedback_debug
        self.debug_turn_n = debug_turn_n

    def _sql_debug_by_experience(
            self,
            question: str,
            schema: str,
            sql_query: str,
            db_type: str = "sqlite",
            schema_links: Union[str, List] = "None",
    ) -> str:
        """Debug SQL by experience-based analysis."""
        instruction = f"""[Instruction]
As an expert SQL analyst, critically analyze and repair the given SQL query to EXACTLY match the data requirements from the user's question using the provided database schema. Follow these steps rigorously:

1. SCHEMA VALIDATION:
- Map ALL question entities to precise schema elements (tables/columns) 
- Verify table relationships through PRIMARY/FOREIGN KEY constraints
- Identify any missing schema elements critical to the question

2. QUERY CORRECTION:
2.1 SELECT CLAUSE:
- Include ALL columns explicitly referenced in the question
- Add essential calculation columns (SUM/COUNT/AVG) implied by the question
- Apply DISTINCT only when duplicate results would occur
2.2 JOIN LOGIC:
- Rebuild joins using ACTUAL foreign key relationships
- Eliminate Cartesian products and phantom joins
- Preserve required NULL values with appropriate join types
2.3 DATA FILTERS:
- Explicitly implement ALL value conditions from the question
- Convert implicit requirements to explicit WHERE/HAVING clauses
- Validate date/number formats against {db_type} conventions
2.4 GROUPING & ORDERING:
- Enforce 1-column GROUP BY unless question explicitly requires multi-column grouping
- Synchronize SELECT columns with GROUP BY clauses
- Apply DESC ordering only when specified by the question

3. SYNTAX ENFORCEMENT:
- Validate ALL functions/operators against {db_type} specifications
- Format CTEs for queries with:
* Multiple subquery references
* Window function dependencies
* Complex logical steps
- Eliminate reserved keyword conflicts using proper quoting

Return ONLY the corrected SQL in standard {db_type} format. NEVER include explanations or markdown formatting. Prioritize completeness of result columns over query brevity."""

        prompt = (
            f"{instruction}\n"
            f"{schema}\n"
            f"#### [Question]: {question}\n"
            f"#### [Schema Links]: {schema_links}\n"
            f"#### [Existing Sql]:\n{sql_query}\n"
            f"#### Output FIXED SQL QUERY:"
        )

        debugged_sql = self.llm.complete(prompt).text
        debugged_sql = sql_clean(debugged_sql)
        return debugged_sql

    def _get_feedback_debug_prompt(self, db_type: str) -> str:
        """Get feedback debug prompt template for specific database type."""
        if db_type == "sqlite":
            return """[Instruction]
You are provided with a user question, a database schema (tables and columns), and a history of SQL execution errors. Your task is to generate a corrected SQL query that resolves all identified issues while strictly adhering to the provided schema and SQLite syntax. The final query must execute successfully in SQLite without introducing new errors.

[Requirements]
1. Error Analysis and Correction:
# Carefully analyze the Execution Error History to identify the type and location of errors (e.g., syntax errors, invalid column references, unsupported functions).
# Cross-reference these errors with the provided Tables and Columns to determine the root cause (e.g., typos, missing joins, incorrect table/column names).
# Address one error at a time, validating each correction against the schema and error history.

2. Schema Compliance:
# Use only the tables, columns, and relationships explicitly defined in the Tables and Columns section. Do not assume or infer additional schema elements.
# Ensure all table and column names match the schema exactly (respecting case sensitivity, underscores, etc.).
# Avoid using aliases or database objects not explicitly mentioned in the schema.

3. SQLite-Specific Syntax:
# Use only SQLite-supported functions, operators, and constructs. Replace any non-SQLite features with equivalent SQLite-compliant alternatives.
# Avoid unsupported SQLite features, such as certain join types (e.g., FULL OUTER JOIN), proprietary functions, or advanced SQL constructs.
# Follow SQLite best practices, such as using explicit joins and avoiding ambiguous column references.

4. Error-Free Query Construction:
# Simplify the query if necessary (e.g., break nested subqueries into smaller, manageable parts) to improve readability and reduce the risk of errors.
# Validate the final query to ensure: No syntax errors (e.g., missing commas, mismatched brackets). All referenced tables and columns exist in the schema. Joins and aliases are explicitly defined and unambiguous.

5. Final Output:
# Provide only the corrected SQL query without any additional commentary or irrelevant content.
# Ensure the query is ready for execution in SQLite and adheres to all requirements above.

### Question: {question}

### Provided Database Schema: 
{schema}

### Execution Error History: 
{error_history}

### [Output Requirements] Your final output should be the corrected SQL query only.
### Output Fixed SQL:
"""
        elif db_type == "big_query":
            return """[Instruction]
For the given question, fix the existing  SQL statement errors by using the provided database schema, execution error history. 
Correct all identified errors (syntax, schema, or BigQuery-specific) without introducing new ones. 
Ensure the final SQL query adheres strictly to the provided schema, follows BigQuery's syntax rules, and executes successfully.

[Requirements]
1. Error Analysis and Correction:
# Carefully review the Execution Error History to identify the type and location of errors (e.g., syntax errors, invalid column references, unsupported functions).
# Categorize errors into: Syntax: Missing commas, parentheses, or incorrect clauses. Schema: Unqualified columns, typos, or invalid table references.

2. BigQuery Compliance: 
# Non-standard functions , legacy SQL, or improper handling of arrays/structs.
# Cross-reference errors with the provided Tables and Columns to pinpoint root causes.
# Address one error at a time, validating each correction against the schema and error history.

3. Schema Compliance:
# Use only the tables, columns, and relationships explicitly defined in the Tables and Columns section. Do not assume or infer additional schema elements.
# Fully qualify table names as project.dataset.table to avoid ambiguity.
# Match all table and column names exactly (respecting case sensitivity and special characters).
# Avoid using aliases or database objects not explicitly mentioned in the schema.

4. BigQuery-Specific Syntax:
# Use only BigQuery Standard SQL (not legacy SQL).
# Replace non-BigQuery functions with their BigQuery equivalents (e.g., GETDATE() → CURRENT_TIMESTAMP()).
# Ensure proper handling of BigQuery-specific features, such as arrays, structs, and nested fields.
# Follow BigQuery best practices, such as using WITH clauses (CTEs) for clarity and modularity.

5. Error-Free Query Construction:
# Simplify the query if necessary (e.g., break nested subqueries into CTEs) to improve readability and reduce the risk of errors.
# Validate the final query to ensure:
(a) No syntax errors (e.g., missing commas, mismatched brackets, or incorrect backtick usage).
(b) All referenced tables and columns exist in the schema.
(c) Joins use explicit ON conditions (avoid implicit comma joins).
(d) No ambiguous column references or unqualified column names.

### Question: {question}

### Provided Database Schema: 
{schema}

### Execution Error History: 
{error_history}

### [Output Requirements] Your final output should be the corrected SQL query only. Ensure the query is ready for execution in BigQuery and adheres to all requirements above.
### Output Fixed SQL:
"""
        elif db_type == "snowflake":
            return """[Instruction]
For the given question, analyze and refine the erroneous SQL statement using the provided database schema, execution error history, and Snowflake SQL standards. Correct all identified errors (syntax, schema, or Snowflake-specific) without introducing new ones. Ensure the final SQL query adheres strictly to the provided schema, follows Snowflake's syntax rules, and executes successfully.

[Requirements]
1. Error Analysis and Correction:
# Carefully review the Execution Error History to identify the type and location of errors (e.g., syntax errors, invalid column references, unsupported functions).
# Cross-reference errors with the provided Tables and Columns to pinpoint root causes (e.g., typos, missing joins, incorrect table/column names).
# Address one error at a time, validating each correction against the schema and error history.

2. Schema Compliance:
# Use only the tables, columns, and relationships explicitly defined in the Tables and Columns section. Do not assume or infer additional schema elements.
# Ensure all table and column names match the schema exactly (respecting case sensitivity, underscores, etc.).
# Follow Snowflake's identifier conventions and use proper quoting where necessary.

3. Snowflake-Specific Syntax:
# Use only Snowflake-supported functions, operators, and constructs. Avoid unsupported features or non-standard SQL.
# Apply Snowflake best practices, such as using CTEs (Common Table Expressions) for clarity and modularity.
# Ensure proper syntax for Snowflake-specific features (e.g., QUALIFY, ILIKE, ARRAY_AGG).

4. Error-Free Query Construction:
# Simplify the query if necessary (e.g., break nested subqueries into CTEs) to improve readability and reduce the risk of errors.
# Validate the final query to ensure: No syntax errors (e.g., missing commas, mismatched brackets). All referenced tables and columns exist in the schema. Joins and aliases are explicitly defined and unambiguous.

5. Final Output:
# Provide only the corrected SQL query without any additional commentary or irrelevant content.
# Ensure the query is ready for execution in Snowflake and adheres to all requirements above.

### Question: {question}

### Provided Database Schema: 
{schema}

### Execution Error History: 
{error_history}

### [Output Requirements] Your final output should be the corrected SQL query only.
### Output Fixed SQL:
"""
        else:
            return None

    def _sql_debug_by_feedback(
            self,
            question: str,
            schema: str,
            sql_query: str,
            db_id: Optional[str] = None,
            db_path: Optional[Union[str, Path]] = None,
            db_type: str = "sqlite",
            credential: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Debug SQL by feedback-based execution and error analysis."""
        chat_history = []
        debug_args = {
            "db_type": db_type,
            "sql_query": sql_query,
            "db_path": db_path,
            "db_id": db_id,
            "credential_path": credential.get(db_type) if credential else None
        }

        for turn in range(self.debug_turn_n):
            res = get_sql_exec_result(**debug_args)
            if not res:
                raise ValueError(f"Invalid 'db_type' argument: failed to execute query on `{db_type}` database.")

            exe_flag, dbms_error_info = res
            if exe_flag is not None:
                return True, debug_args["sql_query"]

            chat_history.append((debug_args["sql_query"], dbms_error_info))

            error_history_info = "\n".join(
                f"### Turn {i + 1}\n# SQL:\n{sql};\n### Error Information:\n{err}\n"
                for i, (sql, err) in enumerate(chat_history)
            )

            prompt_template = self._get_feedback_debug_prompt(db_type)
            if not prompt_template:
                raise ValueError(
                    f"Invalid 'db_type' argument: failed to load prompt template for `{db_type}` database.")

            prompt = prompt_template.format(
                question=question,
                schema=schema,
                error_history=error_history_info
            )

            # Get revised SQL from LLM
            new_sql = self.llm.complete(prompt).text
            debug_args["sql_query"] = sql_clean(new_sql)

        # Final attempt
        exe_flag, _ = get_sql_exec_result(**debug_args)
        return (True, debug_args["sql_query"]) if exe_flag is not None else (False, debug_args["sql_query"])

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            **kwargs
    ):
        """Act method implementing the BaseOptimizer interface."""
        logger.info(f"LinkAlignOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for LinkAlignOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row.get("db_id")
        db_path = Path(self.dataset.db_path) / (
                db_id + ".sqlite") if self.dataset.db_path and db_type == "sqlite" else None
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load and process schema using base class method
        schema = self.process_schema(schema, item)

        # Load schema_links if not provided
        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        # Load pred_sql using base class method
        sql_list, is_single = self.load_pred_sql(pred_sql, item)

        def process_sql(sql):
            _, final_sql = self._sql_debug_by_feedback(
                question, schema, sql, db_id, db_path, db_type, credential
            )
            return final_sql

        optimized_sqls = []
        if self.open_parallel and len(sql_list) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_sql, sql) for sql in sql_list]
                for future in concurrent.futures.as_completed(futures):
                    optimized_sqls.append(future.result())
        else:
            for sql in sql_list:
                optimized_sqls.append(process_sql(sql))

        # Save results using base class method
        output = self.save_results(optimized_sqls, is_single, item, row.get("instance_id"))

        logger.info(f"LinkAlignOptimizer completed processing item {item}")
        return output
