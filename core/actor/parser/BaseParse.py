from core.actor.base import Actor, MergeStrategy
from abc import abstractmethod
from os import PathLike
from typing import Union, Dict, List
from pathlib import Path
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.data_manage import single_central_process, Dataset, save_dataset
from core.utils import load_dataset, parse_schema_from_df
from core.actor.parser.parse_utils import slice_schema_df


class BaseParser(Actor):
    # The NAME variable is defined in the implementing subclasses,
    # and by convention it is recommended to use a name ending with *Parser.
    # NAME: str = "*Parser"

    OUTPUT_NAME: str = "schema_links"
    STRATEGY = MergeStrategy.OVERWRITE.value

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
        logger.debug(f"Output: {output}, saved to: {str(save_path)}")

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

    def _log_schema_entry(self, data_logger, link: str, tag: str) -> None:
        """Normalize and log a single schema link entry."""
        if data_logger is None or link is None:
            return

        normalized = " ".join(str(link).split())
        data_logger.info(f"{self.NAME}.schema_link.{tag} | {normalized}")

    def log_schema_links(self, data_logger, links, stage: str = "final") -> None:
        """Log schema linking outputs regardless of their structure."""
        if data_logger is None or links is None:
            return

        if isinstance(links, dict):
            for table, columns in links.items():
                if isinstance(columns, list):
                    for idx, column in enumerate(columns):
                        self._log_schema_entry(data_logger, f"{table}.{column}", f"{stage}.{table}.{idx}")
                else:
                    self._log_schema_entry(data_logger, f"{table}:{columns}", f"{stage}.{table}")
        elif isinstance(links, list):
            for idx, link in enumerate(links):
                self._log_schema_entry(data_logger, link, f"{stage}.{idx}")
        else:
            self._log_schema_entry(data_logger, links, stage)

    @abstractmethod
    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, update_dataset=True,
            **kwargs):
        pass


def parallel_slice_parse(func):
    slice_size = 100  # todo adjust to a external params. needed to pass
    max_workers = 5  # 最大线程数

    def wrapper(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, **kwargs):
        try:
            if data_logger:
                data_logger.info("Using parallel slice parser!")

            if not hasattr(self, "merge_results"):
                data_logger.info("The Parser has not implemented merging function! Using the original parser instead!")
                raise AttributeError("merge_results method not found")

            # get the schema and the DB size
            schema_df = self.process_schema(item, schema)
            sub_schema_lis = slice_schema_df(schema_df, slice_size=slice_size)
            if data_logger:
                data_logger.info(f"Schema Dataframe slice number is {len(sub_schema_lis)}")
            # 使用多线程处理
            results = [None] * len(sub_schema_lis)  # 预分配结果列表以保持顺序

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，保存 future 和对应的索引
                future_to_index = {
                    executor.submit(func, self, item, sub_schema, data_logger, update_dataset=False, **kwargs): idx
                    for idx, sub_schema in enumerate(sub_schema_lis)
                }

                # 按完成顺序处理结果，但保存到正确的位置
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        output = future.result()
                        results[idx] = output
                        if data_logger:
                            data_logger.info(f"Completed processing slice {idx + 1}/{len(sub_schema_lis)}")
                    except Exception as e:
                        if data_logger:
                            data_logger.error(f"Error processing slice {idx}: {e}")
                        raise

            # 合并结果
            results = self.merge_results(results)
            file_ext = ".txt" if self.output_format == "str" else ".json"
            self.save_output(results, item, file_ext=file_ext)

            if data_logger:
                data_logger.info("All slices processed and merged successfully!")
                data_logger.info(f"Final parsing results: {results}!")
            return results

        except Exception as e:
            if data_logger:
                data_logger.info(f"Parallel slice parser failed: {e}. Used default schema instead!")
            res = func(self, item, schema, data_logger=data_logger, **kwargs)
            return res

    return wrapper
