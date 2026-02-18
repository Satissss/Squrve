from os import PathLike
from typing import Union, Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.base import Actor, MergeStrategy, ActorPool
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.utils import parse_schema_from_df
from abc import abstractmethod

@ActorPool.register_actor
class BaseScaler(Actor):
    """ Use different prompt strategies to generate multiple SQL candidates matching the query. """
    OUTPUT_NAME: str = "pred_sql"
    STRATEGY = MergeStrategy.EXTEND.value

    _registered_actor_lis: List[Actor] = []

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[Any] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
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

        if isinstance(schema, (str, Path)) and Path(schema).exists():
            schema = load_dataset(schema)

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema = parse_schema_from_df(schema)
        else:
            raise Exception("Failed to load a valid database schema for the sample!")

        return schema

    def save_output(self, output: List[str], item: int, instance_id: str = None) -> List[str]:
        """Save generated SQL candidates to files and update dataset."""
        if not self.is_save:
            # 即使不保存文件，也要设置 pred_sql 字段
            if len(output) == 1:
                self.dataset.setitem(item, self.OUTPUT_NAME, output[0])
            else:
                self.dataset.setitem(item, self.OUTPUT_NAME, output)
            return output

        if instance_id is None:
            row = self.dataset[item]
            instance_id = row.get("instance_id", item)

        save_path = Path(self.save_dir)
        if self.dataset.dataset_index is not None:
            save_path = save_path / str(self.dataset.dataset_index)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save each SQL candidate in separate files
        sql_paths = []
        for i, sql in enumerate(output):
            sql_save_path = save_path / f"{self.NAME}_{instance_id}_{i}.sql"
            save_dataset(sql, new_data_source=sql_save_path)
            sql_paths.append(str(sql_save_path))

        # Set dataset field - single path if one SQL, list of paths if multiple
        if len(sql_paths) == 1:
            self.dataset.setitem(item, self.OUTPUT_NAME, sql_paths[0])
        else:
            self.dataset.setitem(item, self.OUTPUT_NAME, sql_paths)

        return output

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        pass

    @classmethod
    def syntax_check(cls, actor_str: str) -> bool:
        if not isinstance(actor_str, str):
            return False

        return actor_str.lower().endswith("scaler")

    @classmethod
    def register_actor(cls, actor_cls: Actor):
        if not issubclass(actor_cls, Actor):
            raise TypeError(f"{actor_cls.__name__} is not a subclass of Actor")

        if actor_cls not in cls._registered_actor_lis:
            cls._registered_actor_lis.append(actor_cls)
        return actor_cls

    @classmethod
    def get_all_actors(cls):
        return cls._registered_actor_lis