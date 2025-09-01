import concurrent.futures
import re
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.db_connect import get_sql_exec_result
from core.utils import sql_clean, parse_schema_from_df
from llama_index.core.llms.llm import LLM

def parse_sql_from_string(input_string: str) -> str:
    sql_pattern = r'```sql(.*?)```'
    all_sqls = []
    for match in re.finditer(sql_pattern, input_string, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    if all_sqls:
        return all_sqls[-1]
    else:
        return "error: No SQL found in the input string"

class MACSQLOptimizer(BaseOptimizer):
    """Optimizer that debugs and refines SQL queries using MAC-SQL's refinement method with execution feedback."""

    NAME = "MACSQLOptimizer"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            debug_turn_n: int = 1,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.debug_turn_n = debug_turn_n
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    def _build_desc_str(self, schema_df: pd.DataFrame) -> str:
        desc_str = ""
        tables = schema_df['table_name'].unique()
        for table in tables:
            cols = schema_df[schema_df['table_name'] == table]
            desc_str += f"# Table: {table}\n[\n"
            for _, row in cols.iterrows():
                col = row['column_name']
                desc = row.get('description', col)
                # No actual query for values in this replication
                values = "No value examples found."
                desc_str += f"  ({col}, {desc}. Value examples: [{values}].),\n"
            desc_str += "]\n"
        return desc_str

    def _build_fk_str(self, schema_df: pd.DataFrame) -> str:
        fk_str = ""
        # Check if foreign key columns exist before accessing them
        if 'referenced_table' in schema_df.columns and 'referenced_column' in schema_df.columns:
            fks = schema_df[schema_df['referenced_table'].notna() & schema_df['referenced_column'].notna()]
            for _, row in fks.iterrows():
                fk_str += f"{row['table_name']}.`{row['column_name']}` = {row['referenced_table']}.`{row['referenced_column']}`\n"
        return fk_str

    def _refine_sql(
            self,
            question: str,
            desc_str: str,
            fk_str: str,
            evidence: str,
            original_sql: str,
            error: str
    ) -> str:
        refiner_template = '''Given the original SQL query that failed execution, the error message, the database schema, evidence, and question, generate a corrected SQL query.

【Database schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}
【Evidence】
{evidence}
【Original SQL】
{original_sql}
【Error】
{error}

Provide the corrected SQL after thinking step by step:
'''
        prompt = refiner_template.format(
            desc_str=desc_str,
            fk_str=fk_str,
            query=question,
            evidence=evidence,
            original_sql=original_sql,
            error=error
        )
        response = self.llm.complete(prompt)
        reply = response.text.strip()
        return parse_sql_from_string(reply)

    def optimize_single_sql(
            self,
            sql: str,
            question: str,
            schema: str,
            db_type: str,
            db_id: Optional[str] = None,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[Dict] = None,
            evidence: str = ""
    ) -> str:
        # For MACSQLOptimizer, we'll use the schema string directly in the prompt
        # instead of trying to parse it back to DataFrame
        desc_str = schema  # Use the schema string directly
        fk_str = ""  # Empty foreign key string since we don't have FK info in the schema string

        current_sql = sql_clean(sql)
        for turn in range(self.debug_turn_n):
            exec_args = {
                "db_type": db_type,
                "sql_query": current_sql,
                "db_path": db_path,
                "db_id": db_id
            }
            
            # Add credential_path for any database type if credential is provided
            if credential and credential.get(db_type):
                exec_args["credential_path"] = credential.get(db_type)

            exec_result = get_sql_exec_result(**exec_args)

            if isinstance(exec_result, tuple):
                if len(exec_result) == 3:
                    res, err, _ = exec_result
                else:
                    res, err = exec_result
            else:
                res = exec_result
                err = None

            if err is None and res is not None and not (isinstance(res, pd.DataFrame) and res.empty):
                return current_sql  # Success, no need to refine further

            error = err or "Empty result set"
            refined_sql = self._refine_sql(question, desc_str, fk_str, evidence, current_sql, error)
            if refined_sql.startswith("error:"):
                logger.warning(f"Failed to parse refined SQL: {refined_sql}")
                break
            current_sql = sql_clean(refined_sql)

        return current_sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List, pd.DataFrame] = None,
            schema_links: Union[str, List[str]] = None,  # Unused but kept for interface
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            **kwargs
    ):
        logger.info(f"MACSQLOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for MACSQLOptimizer")

        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '')
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get('db_id')
        db_path = Path(self.dataset.db_path) / f"{db_id}.sqlite" if self.dataset.db_path and db_type == "sqlite" else None
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
        sql_list = [load_dataset(sql) if isinstance(sql, (str, Path)) and Path(sql).exists() else sql for sql in sql_list]

        def process_sql(sql):
            return self.optimize_single_sql(
                sql, question, schema, db_type, db_id, db_path, credential, evidence
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
            save_path_base = Path(self.save_dir) / str(self.dataset.dataset_index) if self.dataset.dataset_index else Path(self.save_dir)
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

        logger.info(f"MACSQLOptimizer completed processing item {item}")
        return output 