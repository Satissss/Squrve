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


class DINSQLOptimizer(BaseOptimizer):
    """Optimizer that debugs and refines SQL queries using DIN-SQL's prompt-based method with optional execution feedback."""

    NAME = "DINSQLOptimizer"

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
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.use_feedback_debug = use_feedback_debug
        self.debug_turn_n = debug_turn_n
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    def _sql_debug_by_experience(
            self,
            question: str,
            schema: str,
            sql_query: str,
            db_type: str = "sqlite",
            schema_links: Union[str, List] = "None",
    ) -> str:
        """Debug SQL using DIN-SQL's experience-based prompt."""
        instruction = """#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given {db_type} SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the {db_type} SQL QUERY as is.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.

"""
        instruction = instruction.format(db_type=db_type.upper())

        # 限制 schema 长度，避免 prompt 过长
        if len(schema) > 8000:  # 设置合理的长度限制
            schema = schema[:8000] + "\n... (schema truncated)"

        prompt = (
                instruction +
                schema +
                '#### Question: ' + question +
                f'\n#### {db_type.upper()} SQL QUERY\n' + sql_query +
                f'\n#### {db_type.upper()} FIXED SQL QUERY\nSELECT'
        )

        try:
            debugged_sql = self.llm.complete(prompt).text.strip().replace("\n", " ")
            return "SELECT " + debugged_sql
        except Exception as e:
            logger.error(f"Error in _sql_debug_by_experience: {e}")
            # 如果出错，返回原始 SQL
            return sql_query

    def _get_feedback_debug_prompt(self, db_type: str) -> str:
        """Get feedback debug prompt template adapted from DIN-SQL for specific database type."""
        db_type_upper = db_type.upper()
        base_instruction = f"""#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given {db_type} SQL QUERY based on the execution errors. If there are any problems, fix them using the error history.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.

### Question: {{question}}

### Provided Database Schema: 
{{schema}}

### Execution Error History: 
{{error_history}}

#### {db_type_upper} FIXED SQL QUERY
SELECT"""
        return base_instruction

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
        """Debug SQL using execution feedback and DIN-SQL adapted prompt."""
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
                raise ValueError(f"Failed to execute query on {db_type} database.")

            exe_flag, dbms_error_info = res
            if exe_flag is not None:
                # Check if result is empty, perhaps consider as error
                # But for now, if executes, return success
                return True, debug_args["sql_query"]

            chat_history.append((debug_args["sql_query"], dbms_error_info))

            error_history_info = "\n".join(
                f"### Turn {i + 1}\n# SQL:\n{sql};\n### Error Information:\n{err}\n"
                for i, (sql, err) in enumerate(chat_history)
            )

            prompt_template = self._get_feedback_debug_prompt(db_type)
            
            # 限制 schema 长度，避免 prompt 过长
            truncated_schema = schema
            if len(schema) > 6000:  # 为错误历史留出空间
                truncated_schema = schema[:6000] + "\n... (schema truncated)"
            
            prompt = prompt_template.format(
                question=question,
                schema=truncated_schema,
                error_history=error_history_info
            )

            try:
                new_sql = self.llm.complete(prompt).text.strip().replace("\n", " ")
                debug_args["sql_query"] = "SELECT " + new_sql if not new_sql.startswith("SELECT") else new_sql
                debug_args["sql_query"] = sql_clean(debug_args["sql_query"])
            except Exception as e:
                logger.error(f"Error in _sql_debug_by_feedback turn {turn + 1}: {e}")
                # 如果出错，保持当前 SQL 不变
                break

        # Final attempt
        exe_flag, _ = get_sql_exec_result(**debug_args)
        return (True, debug_args["sql_query"]) if exe_flag is not None else (False, debug_args["sql_query"])

    def optimize_single_sql(
            self,
            sql: str,
            question: str,
            schema: str,
            db_type: str,
            schema_links: Union[str, List] = "None",
            db_id: Optional[str] = None,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[dict] = None
    ) -> str:
        """Optimize a single SQL query using experience and optional feedback debugging."""
        # Step 1: Debug by experience (DIN-SQL style)
        debugged_sql = self._sql_debug_by_experience(
            question, schema, sql, db_type, schema_links
        )

        # Step 2: Debug by feedback if enabled
        if self.use_feedback_debug:
            success, final_sql = self._sql_debug_by_feedback(
                question, schema, debugged_sql, db_id, db_path, db_type, credential
            )
            debugged_sql = final_sql

        return debugged_sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            **kwargs
    ):
        """Act method implementing the BaseOptimizer interface."""
        logger.info(f"DINSQLOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for DINSQLOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row.get("db_id")
        db_path = Path(self.dataset.db_path) / (
                    db_id + ".sqlite") if self.dataset.db_path and db_type == "sqlite" else None
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load schema if not provided
        if schema is None:
            instance_schema_path = row.get("instance_schemas", None)
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        schema = parse_schema_from_df(schema)

        # Load schema_links if not provided
        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        # Handle pred_sql input
        if pred_sql is None:
            # 尝试从数据集中获取 pred_sql
            pred_sql = row.get(self.OUTPUT_NAME)
            if pred_sql is None:
                raise ValueError("pred_sql is required for optimization")

        # Normalize pred_sql to list
        is_single = not isinstance(pred_sql, list)
        sql_list = [pred_sql] if is_single else pred_sql

        # Load SQL from paths if necessary
        sql_list = [load_dataset(sql) if isinstance(sql, (str, Path)) and Path(sql).exists() else sql for sql in
                    sql_list]

        def process_sql(sql):
            return self.optimize_single_sql(
                sql, question, schema, db_type, schema_links, db_id, db_path, credential
            )

        optimized_sqls = []
        if self.open_parallel and len(sql_list) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_sql, sql) for sql in sql_list]
                for future in concurrent.futures.as_completed(futures):
                    optimized_sqls.append(future.result())
        else:
            for sql in sql_list:
                optimized_sqls.append(process_sql(sql))

        # Return single or list based on input
        output = optimized_sqls[0] if is_single else optimized_sqls

        if self.is_save:
            instance_id = row.get("instance_id")
            save_path_base = Path(self.save_dir) / str(
                self.dataset.dataset_index) if self.dataset.dataset_index else Path(self.save_dir)
            save_path_base.mkdir(parents=True, exist_ok=True)
            if is_single:
                save_path = save_path_base / f"{self.NAME}_{instance_id}.sql"
                save_dataset(output, new_data_source=save_path)
                self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))
            else:
                paths = []
                for i, opt_sql in enumerate(optimized_sqls):
                    save_path = save_path_base / f"{self.NAME}_{instance_id}_{i}.sql"
                    save_dataset(opt_sql, new_data_source=save_path)
                    paths.append(str(save_path))
                self.dataset.setitem(item, self.OUTPUT_NAME, paths)

        logger.info(f"DINSQLOptimizer completed processing item {item}")
        return output
