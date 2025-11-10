from os import PathLike
from typing import Union, Dict, List, Any
from pathlib import Path
from loguru import logger
import time
import os
from core.actor.base import Actor
from core.data_manage import Dataset
from core.utils import load_dataset, save_dataset
from core.db_connect import execute_sql
from abc import abstractmethod


class BaseSelector(Actor):
    OUTPUT_NAME: str = "pred_sql"

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Any = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir

    def load_pred_sql(self, pred_sql: Union[str, Path, List[str], List[Path]], item: int = None) -> List[str]:
        """Load and normalize pred_sql from various input formats."""
        if pred_sql is None:
            pred_sql = self.dataset[item].get('pred_sql', [])

        if isinstance(pred_sql, (str, Path)):
            pred_sql_path = Path(pred_sql)
            pred_sql = [load_dataset(pred_sql_path) if pred_sql_path.exists() else str(pred_sql)]

        elif isinstance(pred_sql, list):
            pred_sql_list = []
            for p in pred_sql:
                if isinstance(p, str) and os.path.exists(p):
                    pred_sql_list.append(load_dataset(Path(p)))
                else:
                    pred_sql_list.append(str(p))
            pred_sql = pred_sql_list

        else:
            raise ValueError("Invalid pred_sql type")

        if not pred_sql:
            logger.warning("No pred_sql provided")
            return []

        return pred_sql

    def execute_sql_safe(self, sql: str, db_type: str, db_path: str, credential: Any = None) -> Dict[str, Any]:
        """Safely execute SQL and return result with error handling."""
        try:
            # Extract credential_path from credential parameter
            credential_path = None

            # If credential_path not provided in credential parameter, try to get from dataset
            if hasattr(self.dataset, 'credential'):
                dataset_credential = self.dataset.credential
                if isinstance(dataset_credential, dict) and db_type in dataset_credential:
                    credential_path = dataset_credential[db_type]
                elif isinstance(dataset_credential, str):
                    credential_path = dataset_credential

            # Execute SQL with credential_path if available
            start_time = time.time()
            result = execute_sql(db_type, db_path, sql, credential_path)
            return {
                "success": True,
                "result": result,
                "error": None,
                "sql": sql,
                "time_cost": time.time() - start_time,
            }
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "sql": sql,
                "time_cost": 100000  # Marked as a very large number.
            }

    def save_result(self, result: str, item: int, instance_id: str = None) -> str:
        """Save the result to file and update dataset."""
        if not self.is_save:
            return result

        if instance_id is None:
            instance_id = self.dataset[item].get("instance_id", item)

        save_path = Path(self.save_dir)
        if self.dataset.dataset_index:
            save_path = save_path / str(self.dataset.dataset_index)
        save_path = save_path / f"{self.NAME}_{instance_id}.sql"

        save_dataset(result, new_data_source=save_path)
        self.dataset.setitem(item, "pred_sql", str(save_path))
        logger.debug(f"Result saved to: {save_path}")

        return result

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, PathLike, List[str], List[PathLike]] = None,
            data_logger=None,
            **kwargs
    ):
        pass
