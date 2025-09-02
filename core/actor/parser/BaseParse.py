from core.actor.base import Actor
from abc import abstractmethod
from os import PathLike
from typing import Union, Dict, List
from pathlib import Path
import pandas as pd
from loguru import logger

from core.data_manage import single_central_process, Dataset, save_dataset
from core.utils import load_dataset, parse_schema_from_df


class BaseParser(Actor):
    OUTPUT_NAME: str = "schema_links"

    def __init__(
            self,
            dataset: Dataset = None,
            llm=None,
            output_format: str = "list",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            use_external: bool = False,
            **kwargs
    ):
        """Initialize base parser with common parameters."""
        self.dataset = dataset
        self.llm = llm
        self.output_format = output_format
        self.is_save = is_save
        self.save_dir = save_dir
        self.use_external = use_external
        self.kwargs = kwargs

    def process_schema(self, item, schema: Union[str, PathLike, Dict, List] = None) -> Union[str, pd.DataFrame]:
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
            logger.debug("Database schema processed")
            return schema
        else:
            raise ValueError("Invalid schema format")

    def get_llm(self):
        """Get the first available LLM from the list or single LLM."""
        if isinstance(self.llm, list) and self.llm:
            return self.llm[0]
        return self.llm

    def save_output(self, output, item, instance_id: str = None, file_ext: str = ".json"):
        """Save output to file and update dataset."""
        if not self.is_save:
            return
            
        if instance_id is None:
            instance_id = self.dataset[item].get('instance_id', item)
        
        save_path = Path(self.save_dir)
        save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
        
        filename = f"{self.NAME}_{instance_id}{file_ext}"
        save_path = save_path / filename
        save_dataset(output, new_data_source=save_path)
        self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))
        logger.debug(f"Output saved to: {str(save_path)}")

    def format_output(self, output, output_format: str = None):
        """Format output based on output_format parameter."""
        if output_format is None:
            output_format = self.output_format
            
        if output_format == "str":
            return str(output)
        elif output_format == "list":
            return output if isinstance(output, list) else [output]
        else:
            return output

    @abstractmethod
    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, **kwargs):
        pass
