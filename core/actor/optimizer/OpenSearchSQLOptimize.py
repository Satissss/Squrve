from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Optional, Tuple, Dict
from pathlib import Path
import re
import sqlite3
import pandas as pd
from loguru import logger

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset
from core.db_connect import get_sql_exec_result
from core.utils import sql_clean

class OpenSearchSQLOptimizer(BaseOptimizer):
    NAME = "OpenSearchSQLOptimizer"

    CORRECT_PROMPT = """You are an expert in SQL. Here are some examples of fix SQL
{fewshot}

/* Database schema is as follows: */
{db_info}
{key_col_des}

/* Now Please fix the following error SQL */
#question: {q}
#Error SQL: {result_info}
{advice}

Please answer according to the format below and do not output any other content.:
```
#reason: Analysis of How to fix the error
#SQL: right SQL
```"""

    SOFT_PROMPT = """Your task is to perform a simple evaluation of the SQL.

The database system is SQLite. The SQL you need to evaluation is:
#question: {question}
#SQL: {SQL}

Answer in the following format: 
{{
"Judgment": true/false,
"SQL":If SQL is wrong, please correct SQL directly. else answer ""
}}"""

    VOTE_PROMPT = """现在有问题如下:
#question: {question}
对应这个问题有如下几个SQL,请你从中选择最接近问题要求的SQL:
{sql}

请在上面的几个SQL中选择最符合题目要求的SQL, 不要回复其他内容:
#SQL:"""

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm: Optional = None,
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/optimized_sql",
        use_feedback_debug: bool = True,
        debug_turn_n: int = 3,
        open_parallel: bool = True,
        max_workers: Optional[int] = None,
        correct_dic: dict = None,
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
        self.correct_dic = correct_dic or {"default": ""}

    def sql_raw_parse(self, sql, return_question=False):
        sql = sql.split('/*')[0].strip().replace('```sql', '').replace('```', '')
        sql = re.sub("```.*?", '', sql)
        rwq = None
        if return_question:
            rwq, sql = sql.split('#SQL:')
        else:
            sql = sql.split('#SQL:')[-1]
        if sql.startswith("\"") or sql.startswith("\'"):
            sql = sql[1:-1]
        sql = re.sub('\\s+', ' ', sql).strip()
        return sql, rwq

    def correct_sql(self, db_sqlite_path, sql, query, db_info, hint, key_col_des, new_prompt, db_col={}, foreign_set={}, L_values=[]):
        conn = sqlite3.connect(db_sqlite_path, timeout=180)
        count = 0
        raw = sql
        none_case = False
        while count <= self.debug_turn_n:
            try:
                df = pd.read_sql_query(sql, conn)
                if len(df) == 0:
                    raise ValueError("Error':Result: None")
                else:
                    break
            except Exception as e:
                if count >= self.debug_turn_n:
                    wsql = sql
                    sql = self.llm.complete(new_prompt, temperature=0.2).text
                    none_case = True
                    break
                count += 1
                tag = str(e)
                e_s = str(e).split("':")[-1]
                result_info = f"{sql}\nError: {e_s}"
            if sql.find("SELECT") == -1:
                sql = self.llm.complete(new_prompt, temperature=0.3).text
            else:
                fewshot = self.correct_dic.get("default", "")
                advice = ""
                for x in self.correct_dic:
                    if tag.find(x) != -1:
                        fewshot = self.correct_dic[x]
                        if e_s == "Result: None":
                            # Add ambiguity checks as in original
                            advice = ""  # Placeholder, implement if needed
                        elif x == "no such column":
                            advice += "Please check if this column exists in other tables"
                        break
                cor_prompt = self.CORRECT_PROMPT.format(fewshot=fewshot, db_info=db_info, key_col_des=key_col_des, q=query, result_info=result_info, advice=advice)
                sql_response = self.llm.complete(cor_prompt, temperature=0.2 + count / 5).text
                sql, _ = self.sql_raw_parse(sql_response, False)
            raw = sql
        conn.close()
        return sql, none_case

    def optimize_single_sql(
        self,
        sql: str,
        question: str,
        schema: str,
        db_type: str,
        schema_links: Union[str, List] = "None",
        db_id: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        credential: Optional[dict] = None,
        new_prompt: str = "",
        db_info="",
        key_col_des="",
        hint="",
        db_col={},
        foreign_set={},
        L_values=[],
        sub_questions=None,
        **kwargs
    ) -> str:
        if db_type != "sqlite":
            logger.warning("Currently optimized for SQLite only.")
            return sql

        db_path = Path(db_path) if db_path else None
        sql = sql_clean(sql)

        if self.use_feedback_debug:
            optimized_sql, _ = self.correct_sql(
                db_path, sql, question, db_info, hint, key_col_des, new_prompt, db_col, foreign_set, L_values
            )
        else:
            optimized_sql = sql

        return optimized_sql

    def act(
        self,
        item,
        schema: Union[str, Path, Dict, List] = None,
        schema_links: Union[str, List[str]] = None,
        pred_sql: Union[str, Path, List[str], List[Path]] = None,
        **kwargs
    ):
        logger.info(f"OpenSearchSQLOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for OpenSearchSQLOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row.get("db_id")
        db_path = Path(self.dataset.db_path) / (db_id + ".sqlite") if self.dataset.db_path and db_type == "sqlite" else None
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load schema if not provided (placeholder, adapt as needed)
        if schema is None:
            schema = ""  # Implement schema loading if required

        # Load schema_links if not provided
        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        # Handle pred_sql input
        if pred_sql is None:
            raise ValueError("pred_sql is required for optimization")

        is_single = not isinstance(pred_sql, list)
        sql_list = [pred_sql] if is_single else pred_sql

        # Load SQL from paths if necessary
        sql_list = [load_dataset(sql) if isinstance(sql, (str, Path)) and Path(sql).exists() else sql for sql in sql_list]

        def process_sql(sql):
            return self.optimize_single_sql(
                sql, question, schema, db_type, schema_links, db_id, db_path, credential, **kwargs
            )

        optimized_sqls = []
        if self.open_parallel and len(sql_list) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_sql, sql) for sql in sql_list]
                for future in as_completed(futures):
                    optimized_sqls.append(future.result())
        else:
            for sql in sql_list:
                optimized_sqls.append(process_sql(sql))

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

        logger.info(f"OpenSearchSQLOptimizer completed processing item {item}")
        return output 