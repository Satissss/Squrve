from typing import Union, List, Dict, Any, Tuple
from os import PathLike
from pathlib import Path
from loguru import logger

from core.actor.selector.BaseSelector import BaseSelector
from core.data_manage import Dataset
from core.db_connect import get_sql_exec_result_with_time


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

        successful_runs.sort(key=lambda r: r["time_cost"])
        best_sql = successful_runs[0]["sql"]
        if data_logger:
            data_logger.info(f"{self.NAME}.best_candidate | details={successful_runs[0]}")

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

