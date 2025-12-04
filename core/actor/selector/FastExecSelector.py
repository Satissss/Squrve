from typing import Union, List, Dict, Any, Tuple
from os import PathLike
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.selector.BaseSelector import BaseSelector
from core.data_manage import Dataset
from core.db_connect import get_sql_exec_result_with_time
from core.utils import compare_pandas_table


class FastExecSelector(BaseSelector):
    """
    Selector that keeps only successfully executed SQL candidates and picks the
    one with the shortest execution time.
    """

    NAME = "FastExecSelector"

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Any = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, PathLike, List[str], List[PathLike]] = None,
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row = self.dataset[item] if self.dataset else {}
        db_type = row.get("db_type", "sqlite")
        db_id = row.get("db_id", "")
        credential = getattr(self.dataset, "credential", None)
        row_db_path = row.get("db_path")
        dataset_db_root = getattr(self.dataset, "db_path", None)
        db_path = None
        if row_db_path:
            db_path = Path(row_db_path)
        elif dataset_db_root and db_type == "sqlite" and db_id:
            db_path = Path(dataset_db_root) / f"{db_id}.sqlite"

        pred_sql_list = self.load_pred_sql(pred_sql, item)
        if not pred_sql_list:
            return ""

        successful_runs = []
        for sql in pred_sql_list:
            exec_args = self._build_exec_args(db_type, sql, db_id=db_id, db_path=db_path, credential=credential)
            if data_logger:
                data_logger.info(f"{self.NAME}.exec_params | sql={sql} | params={exec_args}")
            try:
                elapsed, exec_result = get_sql_exec_result_with_time(db_type, **exec_args)
            except Exception as exc:
                if data_logger:
                    data_logger.info(f"{self.NAME}.exec_failed | sql={sql} | error={exc}")
                continue

            res, err = self._normalize_exec_result(exec_result)
            if err:
                if data_logger:
                    data_logger.info(f"{self.NAME}.exec_error | sql={sql} | error={err}")
                continue

            successful_runs.append({
                "sql": sql,
                "time_cost": elapsed,
                "result": res
            })

        if not successful_runs:
            logger.warning(f"{self.NAME} | no successful executions for item {item}")
            return ""

        # Group by execution result, select the fastest SQL from the most frequent result group
        result_groups = self._group_by_result(successful_runs)
        max_count = max(len(group) for group in result_groups)
        
        # If all groups have the same count, fallback to the fastest SQL overall
        if len(result_groups) == len(successful_runs):
            best_run = min(successful_runs, key=lambda r: r["time_cost"])
        else:
            # Find the most frequent result group(s)
            most_frequent_groups = [group for group in result_groups if len(group) == max_count]
            # Select the fastest SQL from the first most frequent group
            best_run = min(most_frequent_groups[0], key=lambda r: r["time_cost"])
        
        best_sql = best_run["sql"]
        if data_logger:
            data_logger.info(f"{self.NAME}.best_candidate | details={best_run} | group_size={max_count}")

        best_sql = self.save_result(best_sql, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return best_sql

    @staticmethod
    def _build_exec_args(
            db_type: str,
            sql: str,
            db_id: str = "",
            db_path: Union[str, Path, None] = None,
            credential: Any = None
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "sql_query": sql,
            "db_path": db_path,
            "db_id": db_id
        }
        credential_path = None
        if isinstance(credential, dict):
            credential_path = credential.get(db_type)
        elif credential:
            credential_path = credential

        if credential_path:
            args["credential_path"] = credential_path
        return args

    @staticmethod
    def _normalize_exec_result(exec_result: Any) -> Tuple[Any, Any]:
        if isinstance(exec_result, tuple):
            if len(exec_result) == 3:
                res, err, _ = exec_result
                return res, err
            if len(exec_result) == 2:
                res, err = exec_result
                return res, err
            if len(exec_result) >= 1:
                return exec_result[0], None
            return None, "Empty result tuple"

        return exec_result, None

    @staticmethod
    def _group_by_result(successful_runs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group SQL runs by their execution results using DataFrame comparison."""
        groups = []
        for run in successful_runs:
            result = run["result"]
            placed = False
            
            for group in groups:
                group_result = group[0]["result"]
                # Compare DataFrames using compare_pandas_table utility
                if isinstance(result, pd.DataFrame) and isinstance(group_result, pd.DataFrame):
                    if compare_pandas_table(result, group_result, ignore_order=True) == 1:
                        group.append(run)
                        placed = True
                        break
                # Fallback to direct comparison for non-DataFrame results
                elif result == group_result or (pd.isna(result) and pd.isna(group_result)):
                    group.append(run)
                    placed = True
                    break
            
            if not placed:
                groups.append([run])
        
        return groups

