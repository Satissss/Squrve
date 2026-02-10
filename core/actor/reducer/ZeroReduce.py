from os import PathLike
from pathlib import Path
from core.actor.reducer.BaseReduce import BaseReducer
from core.data_manage import Dataset, single_central_process
from core.utils import save_dataset
from typing import Union, Dict, List
import pandas as pd

@BaseReducer.register_actor
class ZeroReducer(BaseReducer):
    """ Return all schemas of the target database for the data sample directly """

    NAME = "ZeroReducer"

    def __init__(
            self,
            dataset: Dataset = None,
            output_format: str = "dataframe",  # output in `dataframe` or `json`
            save_dir: Union[str, PathLike] = None,
            **kwargs
    ):
        self.dataset: Dataset = dataset
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

    def act(self, item, schema: Union[Dict, List] = None, data_logger=None, **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        sub_schema = self.dataset.get_db_schema(item) if not schema else schema
        if not sub_schema:
            return None

        if self.output_format == "dataframe":
            if isinstance(sub_schema, dict):
                sub_schema = single_central_process(sub_schema)
            assert isinstance(sub_schema, list)
            sub_schema = pd.DataFrame(sub_schema)

        if self.save_dir:
            instance_id = self.dataset[item].get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            if self.output_format == "dataframe":
                save_path = save_path / f"{self.name}_{instance_id}.csv"
            else:
                save_path = save_path / f"{self.name}_{instance_id}.json"
            save_dataset(sub_schema, new_data_source=save_path)
            self.dataset.setitem(item, "instance_schemas", str(save_path))
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return sub_schema
