from os import PathLike
from typing import Union, Dict, List, Any
from pathlib import Path

from core.actor.base import Actor, MergeStrategy, ActorPool
from core.data_manage import save_dataset, Dataset
from abc import abstractmethod
from loguru import logger

@ActorPool.register_actor
class BaseGenerator(Actor):
    OUTPUT_NAME = "pred_sql"
    STRATEGY = MergeStrategy.APPEND.value

    _registered_actor_lis: List[Actor] = []
    is_save: bool
    save_dir: str

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str], Dict] = None,
            data_logger=None,
            **kwargs
    ):
        pass

    def save_output(self, sql: str, item, instance_id: str = None) -> str:
        """
        Save generated SQL to file and update dataset.
        
        Args:
            sql: The SQL query to save
            item: The dataset item index
            instance_id: The instance identifier (defaults to item if not provided)
            
        Returns:
            The input SQL (unchanged)
        """
        if not self.is_save:
            return sql

        instance_id = instance_id or str(item)
        save_path = Path(self.save_dir)

        # Add dataset index subfolder if available
        if self.dataset and hasattr(self.dataset, 'dataset_index') and self.dataset.dataset_index:
            save_path = save_path / str(self.dataset.dataset_index)

        # Construct final file path
        save_path = save_path / f"{self.name}_{instance_id}.sql"

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save SQL to file
        save_dataset(sql, new_data_source=save_path)

        # Update dataset with saved path
        if self.dataset:
            self.dataset.setitem(item, "pred_sql", str(save_path))

        logger.debug(f"SQL saved to: {save_path}")

        return sql

    @classmethod
    def syntax_check(cls, actor_str: str) -> bool:
        if not isinstance(actor_str, str):
            return False

        return actor_str.lower().endswith("generator")

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