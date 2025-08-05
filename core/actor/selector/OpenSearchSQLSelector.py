from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict, Any
from pathlib import Path

import time

from core.actor.selector.BaseSelector import BaseSelector
from core.data_manage import Dataset
from core.utils import load_dataset, save_dataset
from core.db_connect import execute_sql


class OpenSearchSQLSelector(BaseSelector):
    """Selector component based on OpenSearch-SQL for choosing/optimizing SQL candidates."""

    NAME = "OpenSearchSQLSelector"

    VOTE_PROMPT = """现在有问题如下:
#question: {question}
对应这个问题有如下几个SQL,请你从中选择最接近问题要求的SQL:
{sql}

请在上面的几个SQL中选择最符合题目要求的SQL, 不要回复其他内容:
#SQL:"""

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Any = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            max_workers: int = 5,
            enable_execution_voting: bool = True,
            enable_corrections: bool = True,
            enable_llm_voting: bool = True,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.enable_execution_voting = enable_execution_voting
        self.enable_corrections = enable_corrections
        self.enable_llm_voting = enable_llm_voting

    def _execute_sql_safe(self, sql: str, db_type: str, db_path: str, credential: Any = None) -> Dict[str, Any]:
        """Safely execute SQL and return result with timing."""
        start_time = time.time()
        try:
            if db_type == "sqlite":
                result = execute_sql(db_type, db_path, sql, None)
            elif db_type in ["snowflake", "big_query"]:
                credential_path = credential if isinstance(credential, str) else None
                if not credential_path and self.dataset and hasattr(self.dataset, 'credential'):
                    credential_path = self.dataset.credential.get(db_type)
                if not credential_path:
                    return {"success": False, "result": None, "error": "No credential", "time_cost": 0, "sql": sql}
                result = execute_sql(db_type, db_path, sql, credential_path)
            else:
                return {"success": False, "result": None, "error": "Unsupported db_type", "time_cost": 0, "sql": sql}

            time_cost = time.time() - start_time
            return {"success": True, "result": result, "error": None, "time_cost": time_cost, "sql": sql}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e), "time_cost": time.time() - start_time,
                    "sql": sql}

    def _compare_execution_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare execution results for voting."""
        if not results:
            return {"best_sql": None}

        result_groups = {}
        for res in results:
            if res["success"]:
                res_str = str(res["result"])
                if res_str not in result_groups:
                    result_groups[res_str] = []
                result_groups[res_str].append({"sql": res["sql"], "time_cost": res["time_cost"]})

        if result_groups:
            best_group = max(result_groups.values(), key=len)
            best_sql = min(best_group, key=lambda x: x["time_cost"])["sql"]
            return {"best_sql": best_sql}
        return {"best_sql": results[0]["sql"]}

    def _vote_chose(self, sqls: List[str], question: str) -> str:
        """Use LLM to vote on best SQL."""
        if not self.llm:
            return sqls[0] if sqls else ""

        all_sql = '\n\n'.join(sqls)
        prompt = self.VOTE_PROMPT.format(question=question, sql=all_sql)
        response = self.llm.complete(prompt).text
        return response.split("#SQL:")[-1].strip()

    def _correct_sql(self, sql: str, question: str, db_type: str, db_path: str, credential: Any) -> str:
        """Correct SQL by attempting execution and fixing errors."""
        exec_result = self._execute_sql_safe(sql, db_type, db_path, credential)
        if exec_result["success"]:
            return sql
        # Placeholder for more advanced correction logic from OpenSearch-SQL
        # For now, return original
        return sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            **kwargs
    ):
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get('db_id', '')
        db_path = row.get('db_path', db_id)
        credential = row.get('credential', None)

        # Load pred_sql
        is_single = isinstance(pred_sql, (str, Path)) or (isinstance(pred_sql, list) and len(pred_sql) == 1)
        if pred_sql is None:
            pred_sql = row.get('pred_sql', [])
        if isinstance(pred_sql, (str, Path)):
            pred_sql = [load_dataset(Path(pred_sql)) if Path(pred_sql).exists() else pred_sql]
        elif isinstance(pred_sql, list):
            loaded = []
            for p in pred_sql:
                p_path = Path(p) if isinstance(p, str) else p
                loaded.append(load_dataset(p_path) if p_path.exists() else str(p))
            pred_sql = loaded

        if not pred_sql:
            return "" if is_single else []

        # Concurrent execution
        execution_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._execute_sql_safe, sql, db_type, db_path, credential) for sql in pred_sql]
            for future in as_completed(futures):
                execution_results.append(future.result())

        # Voting
        if self.enable_execution_voting and len(pred_sql) > 1:
            voting_result = self._compare_execution_results(execution_results)
            best_sql = voting_result["best_sql"]
        else:
            best_sql = pred_sql[0]

        # Corrections
        if self.enable_corrections:
            best_sql = self._correct_sql(best_sql, question, db_type, db_path, credential)

        # LLM Voting if multiple
        if len(pred_sql) > 1 and self.enable_llm_voting:
            best_sql = self._vote_chose(pred_sql, question)

        # Save
        if self.is_save:
            instance_id = row.get('instance_id', item)
            save_path = Path(self.save_dir)
            if self.dataset.dataset_index:
                save_path /= str(self.dataset.dataset_index)
            save_path /= f"{self.NAME}_{instance_id}.sql"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_dataset(best_sql, save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))

        return best_sql if is_single else [best_sql]
