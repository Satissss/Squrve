from os import PathLike
from typing import Union, Dict, List, Any
from pathlib import Path

from core.actor.base import Actor
from core.data_manage import save_dataset, Dataset
from abc import abstractmethod
from loguru import logger


class BaseGenerator(Actor):
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

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
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
