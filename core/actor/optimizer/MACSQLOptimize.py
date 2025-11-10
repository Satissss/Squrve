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
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.debug_turn_n = debug_turn_n

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
        refiner_template = '''【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
【Query】
-- {query}
【Evidence】
{evidence}
【Database info】
{desc_str}
【Foreign keys】
{fk_str}
【old SQL】
```sql
{original_sql}
```
【SQLite error】 
{error}

Now please fixup old SQL and generate new SQL again.
【correct SQL】
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
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
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

        # Load and process schema using base class method
        schema = self.process_schema(schema, item)

        # Load schema_links if not provided
        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        # Load pred_sql using base class method
        sql_list, is_single = self.load_pred_sql(pred_sql, item)
        if data_logger:
            data_logger.info(f"{self.NAME}.input_sql_count | count={len(sql_list)}")

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

        # Save results using base class method
        output = self.save_results(optimized_sqls, is_single, item, row.get("instance_id"))

        logger.info(f"MACSQLOptimizer completed processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.optimized_sql | output={optimized_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        
        return output 