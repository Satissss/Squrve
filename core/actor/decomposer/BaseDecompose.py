from os import PathLike
from pathlib import Path
import pandas as pd
from typing import Union, Dict, List
from loguru import logger

from core.actor.base import Actor
from abc import abstractmethod

from core.data_manage import single_central_process, Dataset, save_dataset
from core.utils import load_dataset, parse_schema_from_df


class BaseDecomposer(Actor):
    """ Decompose complex queries into a series of sub-questions. """

    OUTPUT_NAME: str = "sub_questions"

    def __init__(
            self,
            dataset: Dataset = None,
            llm=None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/sub_questions",
            **kwargs
    ):
        """Initialize base decomposer with common parameters."""
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.kwargs = kwargs

    def process_schema(self, item, schema: Union[str, PathLike, Dict, List] = None) -> str:
        """Process and normalize database schema from various input formats."""
        logger.debug("Processing database schema...")

        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = self.dataset[item].get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded schema from: {instance_schema_path}")
            else:
                logger.debug("Fetching schema from dataset")
                schema = self.dataset.get_db_schema(item)

            if schema is None:
                raise ValueError("Failed to load a valid database schema for the sample!")

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema_str = parse_schema_from_df(schema)
        else:
            raise ValueError("Invalid schema format")

        logger.debug("Database schema processed")
        return schema_str

    def get_llm(self):
        """Get the first available LLM from the list or single LLM."""
        if isinstance(self.llm, list) and self.llm:
            return self.llm[0]
        return self.llm

    def save_output(self, output, item, instance_id: str = None, db_id: str = None):
        """Save output to file and update dataset."""
        if not self.is_save:
            return

        if instance_id is None:
            instance_id = self.dataset[item].get('instance_id', item)

        save_path = Path(self.save_dir)
        save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path

        # Handle different naming conventions for different decomposers
        filename = f"{self.NAME}_{instance_id}.json"

        save_path = save_path / filename
        save_dataset(output, new_data_source=save_path)
        self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))
        logger.debug(f"Output saved to: {str(save_path)}")

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            **kwargs
    ):
        pass
