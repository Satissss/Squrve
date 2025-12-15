from os import PathLike
from typing import Union, Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.base import Actor, MergeStrategy
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.utils import parse_schema_from_df
from abc import abstractmethod


class BaseOptimizer(Actor):
    OUTPUT_NAME = "pred_sql"
    STRATEGY = MergeStrategy.OVERWRITE.value

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[Any] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    def process_schema(self, schema: Union[str, Path, Dict, List, pd.DataFrame], item: int) -> str:
        """Load and process schema from various input formats."""
        if schema is None:
            row = self.dataset[item]
            instance_schema_path = row.get("instance_schemas", None)
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema = parse_schema_from_df(schema)

        return schema

    def load_pred_sql(self, pred_sql: Union[str, Path, List[str], List[Path]], item: int) -> tuple[List[str], bool]:
        """Load and normalize pred_sql from various input formats."""
        if pred_sql is None:
            row = self.dataset[item]
            pred_sql = row.get(self.OUTPUT_NAME)
            if pred_sql is None:
                raise ValueError("pred_sql is required for optimization")

        is_single = not isinstance(pred_sql, list)
        sql_list = [pred_sql] if is_single else pred_sql

        # Load SQL from paths if necessary
        try:
            sql_list = [load_dataset(p) if isinstance(p, (str, Path)) and Path(p).exists() else p for p in sql_list]
        except Exception as e:
            logger.info(f"Error when loading pred_sql: {e}. Treat sql_list storing the generated sqls.")

        return sql_list, is_single

    def save_output(self, optimized_sqls: List[str], item: int, instance_id: str = None) -> Union[str, List[str]]:
        """Save optimized results to files and update dataset."""
        is_single = len(optimized_sqls) == 1

        if not self.is_save:
            return optimized_sqls[0] if is_single else optimized_sqls

        if instance_id is None:
            row = self.dataset[item]
            instance_id = row.get("instance_id", item)

        save_path_base = Path(self.save_dir)
        if self.dataset.dataset_index is not None:
            save_path_base = save_path_base / str(self.dataset.dataset_index)
        save_path_base.mkdir(parents=True, exist_ok=True)

        if is_single:
            save_path = save_path_base / f"{self.NAME}_{instance_id}.sql"
            logger.info(f"Optimized SQL save path: {save_path}")
            save_dataset(optimized_sqls[0], new_data_source=save_path)
            self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))
            return optimized_sqls[0]
        else:
            paths = []
            for i, opt_sql in enumerate(optimized_sqls):
                save_path = save_path_base / f"{self.NAME}_{instance_id}_{i}.sql"
                save_dataset(opt_sql, new_data_source=save_path)
                paths.append(str(save_path))
            self.dataset.setitem(item, self.OUTPUT_NAME, paths)
            return optimized_sqls

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
