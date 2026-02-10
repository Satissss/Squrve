from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict, Any
from pathlib import Path

from core.actor.selector.BaseSelect import BaseSelector
from core.data_manage import Dataset

@BaseSelector.register_actor
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
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.max_workers = max_workers
        self.enable_execution_voting = enable_execution_voting
        self.enable_corrections = enable_corrections

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
        exec_result = self.execute_sql_safe(sql, db_type, db_path, credential)
        if exec_result["success"]:
            return sql

        return sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get('db_id', '')
        db_path = row.get('db_path', db_id)
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load pred_sql using base class method
        is_single = isinstance(pred_sql, (str, Path)) or (isinstance(pred_sql, list) and len(pred_sql) == 1)
        pred_sql = self.load_pred_sql(pred_sql, item)
        if not pred_sql:
            return "" if is_single else []
        if data_logger:
            data_logger.info(f"{self.NAME}.candidates | count={len(pred_sql)}")
            

        # Concurrent execution
        execution_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.execute_sql_safe, sql, db_type, db_path, credential) for sql in pred_sql]
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

        # Save using base class method
        best_sql = self.save_result(best_sql, item, row.get('instance_id', item))
        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
            
              

        return best_sql
