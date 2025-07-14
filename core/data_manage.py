"""
Data Management Module for Squrve Framework

This module provides comprehensive data management capabilities for Text-to-SQL processing,
including dataset handling, schema management, and vector store operations.

Classes:
    Dataset: Encapsulates dataset and schema for single task runs
    DataLoader: Manages multiple datasets and schemas for Text-to-SQL processes

Key Features:
    - Multi-database support with flexible configuration
    - Vector store integration for schema indexing
    - Few-shot learning capabilities
    - External knowledge integration
    - Comprehensive data validation and preprocessing
"""

from core.base import Router

from llama_index.core.llms.llm import LLM
from core.LinkAlign.RagPipeline import RagPipeLines
from core.LinkAlign.SchemaLinkingTool import SchemaLinkingTool
from core.utils import load_dataset, save_dataset

from os import PathLike
from pathlib import Path
from typing import Union, Dict, List, Callable, Optional, Any
import warnings
import random


class Dataset:
    """
    Dataset encapsulates the core dataset and corresponding database schema for a single task startup_run.

    Supports flexible dataset construction from various sources (JSON file, list of dicts, etc.)
    and schema definitions, with options for random sampling and filtering.

    Key features:
    - Supports single and multi-database configurations
    - Allows schema finalization and indexing
    - Enables vector store configuration and embedding model selection
    - Compatible with both pre-defined and custom data sources
    """

    def __init__(
            self,
            data_source: Union[str, PathLike, List[Dict]],
            schema_source: Union[str, PathLike],
            is_schema_final: bool = False,
            dataset_index: Optional[Union[str, int]] = None,
            schema_index: Optional[Union[str, int]] = None,
            random_size: Optional[float] = None,
            filter_by: Optional[str] = None,
            multi_database: bool = False,
            vector_store: str = "vector_store",
            embed_model_name: str = "BAAI/bge-large-en-v1.5",
            db_credential: Optional[Dict] = None,
            db_path: Optional[Union[str, PathLike]] = None,
            **kwargs
    ):
        """
        Initialize a Dataset instance.

        Args:
            data_source: The input data source (file path or list of dicts)
            schema_source: Path or identifier for the database schema
            is_schema_final: Whether the schema is finalized and ready for use
            dataset_index: Optional identifier for the dataset
            schema_index: Optional identifier for the schema
            random_size: Proportion of data to sample randomly
            filter_by: Filtering condition for dataset records
            multi_database: Whether the dataset involves multiple databases
            vector_store: Path or name of the vector store
            embed_model_name: Name of the embedding model for vectorization
            db_credential: Optional database connection credentials
            db_path: Optional path to the database
            **kwargs: Additional keyword arguments
        """
        self._data_source = data_source
        self._dataset: List[Dict] = self.__init_data_source__(data_source, random_size, filter_by)
        self._schema_source = schema_source

        self.is_schema_final = is_schema_final
        self.dataset_index = dataset_index
        self.schema_index = schema_index
        self.multi_database = multi_database
        self.vector_store = vector_store
        self.embed_model_name = embed_model_name
        self.db_credential: dict = {} if not db_credential else db_credential
        self.db_path: Optional[Union[str, PathLike]] = db_path

        self.suffix = self.__init_suffix__(random_size, filter_by)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        return self._dataset[item]

    def setitem(self, item, key, value):
        self._dataset[item][key] = value

    def __setitem__(self, key, value):
        self._dataset[key] = value

    def if_schema_source_file(self):
        schema_source = Path(self._schema_source)
        return schema_source.is_file()

    @property
    def schema_source(self):
        source = Path(self._schema_source)
        if self.is_schema_final or source.suffix:
            return str(source)
        source = source / "multi_db" if self.multi_database else source / "single_db"
        return str(source)

    @property
    def credential(self):
        return self.db_credential

    @property
    def database_path(self):
        return self.db_path

    @property
    def data_source(self):
        return self._data_source

    @property
    def is_multi_database(self):
        return self.multi_database

    @classmethod
    def __init_data_source__(
            cls,
            data_source: Union[str, PathLike, List[Dict]],
            random_size: Optional[Union[float, int]] = None,
            filter_by: Optional[str] = None
    ) -> List[Dict]:
        if isinstance(data_source, str):
            data_source = Path(data_source)
        if isinstance(data_source, Path):
            loaded_data = load_dataset(data_source)
            if isinstance(loaded_data, list):
                data_source = loaded_data
            else:
                return []

        if not data_source or not isinstance(data_source, list):
            return []

        if random_size:
            if isinstance(random_size, float):
                sample_size = int(len(data_source) * random_size)
            else:
                sample_size = random_size
            data_source = random.sample(data_source, sample_size)

        if filter_by:
            data_source = filter_dataset(dataset_=data_source, filter_by_=filter_by)

        return data_source

    @classmethod
    def __init_suffix__(cls, random_size: Optional[Union[float, int]] = None, filter_by: Optional[str] = None):
        suffix = ""
        if random_size is not None:
            suffix += "rnd_" + str(random_size)
        if filter_by is not None:
            suffix += filter_by
        suffix = "_".join(suffix.split("."))

        return suffix

    def get_vector_index(self, item: Optional[int] = None, db_id: Optional[str] = None):
        if item is not None:
            row = self.__getitem__(item)
            db_id = row.get("db_id")

        if not db_id:
            warnings.warn("Failed to retrieve vector index: 'db_id' must not be None.", category=UserWarning)
            return None

        source_index = str(self.schema_index) if self.schema_index else ""
        vec_store_dir = get_vector_store_dir(
            source=self.schema_source,
            source_index=source_index,
            db_id=db_id,
            vector_store=self.vector_store,
            embed_model_name=self.embed_model_name,
            multi_database=self.multi_database,
        )

        vector_index = RagPipeLines.build_index_from_source(
            data_source=str(self._schema_source),
            persist_dir=str(vec_store_dir),
            is_vector_store_exist=True,
            index_method="VectorStoreIndex",
            embed_model_name=self.embed_model_name
        )

        return vector_index

    def get_db_schema(self, item: Optional[int] = None, db_id: Optional[str] = None):
        if item is not None:
            row = self.__getitem__(item)
            db_id = row["db_id"]
        assert db_id

        schema_source = Path(self.schema_source)
        if schema_source.is_file():
            all_schema = load_dataset(schema_source)
            if isinstance(all_schema, list) and all(
                    isinstance(row, list) and all(isinstance(_, dict) for _ in row)
                    for row in all_schema
            ):
                # Determine whether it is a schema file in parallel format and return a List[Dict] object.
                for db_schema_row in all_schema:
                    if db_schema_row[0]["db_id"] == db_id:
                        return db_schema_row
            elif isinstance(all_schema, list) and all(isinstance(row, dict) for row in all_schema):
                # Determine whether it is a schema file in central format and return a Dict object.
                for db_schema_row in all_schema:
                    if db_schema_row["db_id"] == db_id:
                        return db_schema_row
            return None
        elif schema_source.is_dir():
            if not self.multi_database:
                schema_source = schema_source / db_id
            db_schema_row = []
            for file in [f.name for f in schema_source.iterdir() if f.is_file()]:
                schema_item = load_dataset(schema_source / file)
                db_schema_row.append(schema_item)

            return db_schema_row

        return None

    def save_data(self, dataset_save_path: Optional[Union[str, PathLike]] = None):
        """Save dataset to the data source, optionally modifying the filename suffix."""
        path = None
        if dataset_save_path:
            dataset_save_path = Path(dataset_save_path)
            if dataset_save_path.suffix:
                path = dataset_save_path
        if not path:
            if isinstance(self._data_source, (str, PathLike)):
                path = Path(self._data_source)
            else:
                warnings.warn("Cannot save dataset: data_source is not a file path", category=UserWarning)
                return

        if self.suffix:
            filename = f"{path.stem}_{self.suffix}_{path.suffix}"
            path = path.with_name(filename)

        save_dataset(self._dataset, new_data_source=path)

    def get_instance_ids(self):
        instance_ids = [row["instance_id"] for row in self._dataset]
        return instance_ids

    def __eq__(self, other):
        """ A simple method to compare two dataset is equal. """
        if other is None:
            return False
        if not isinstance(other, Dataset):
            return NotImplemented
        # Compare the dataset
        if len(self) != len(other):
            return False

        if self.dataset_index != other.dataset_index:
            return False
        if self.schema_index != other.schema_index:
            return False

        instance_id_self = self.get_instance_ids()
        instance_id_other = other.get_instance_ids()
        if any(id_ not in instance_id_other for id_ in instance_id_self):
            return False

        if Path(self._schema_source).resolve() != Path(other._schema_source).resolve():
            return False

        return True

    def __hash__(self):
        instance_ids = frozenset(self.get_instance_ids())  # unordered and immutable
        schema_path = Path(self._schema_source).resolve()
        return hash((
            self.dataset_index,
            self.schema_index,
            instance_ids,
            str(schema_path)
        ))

    @property
    def dataset_dict(self):
        """ Transform the dataset parameter into a dictionary format. """

        main_ = {
            "dataset": self._dataset,
            "schema": str(self._schema_source),
            "dataset_index": self.dataset_index,
            "schema_index": self.schema_index
        }
        sub_ = {
            "is_schema_final": self.is_schema_final,
            "multi_database": self.multi_database,
            "vector_store": self.vector_store,
            "embed_model_name": self.embed_model_name,
            "db_credential": self.db_credential,
            "db_path": self.db_path,
            "suffix": self.suffix
        }
        dataset_dict = {"main": main_, "sub": sub_}

        return dataset_dict

    @classmethod
    def resolve_dataset_from_dict(cls, dataset_dict: Dict):
        main = dataset_dict.get("main")
        if not isinstance(main, dict):
            warnings.warn(
                "Error creating Dataset: the dataset dictionary is missing the main parameter or has a format error.",
                category=UserWarning)
            return None

        if not {"dataset", "schema"}.issubset(main):
            return None

        main_params = {
            "data_source": main.get("dataset"),
            "schema_source": main.get("schema"),
            "dataset_index": main.get("dataset_index"),
            "schema_index": main.get("schema_index")
        }
        # 合并 sub（如果存在）到 main
        sub = dataset_dict.get("sub")
        if isinstance(sub, dict):
            main_params = {**main_params, **sub}

        # Filter out None values for required parameters and provide defaults
        if main_params.get("data_source") is None or main_params.get("schema_source") is None:
            warnings.warn("Required parameters data_source and schema_source cannot be None", category=UserWarning)
            return None

        # Provide default values for optional parameters
        main_params.setdefault("is_schema_final", False)
        main_params.setdefault("multi_database", False)
        main_params.setdefault("vector_store", "vector_store")
        main_params.setdefault("embed_model_name", "BAAI/bge-large-en-v1.5")

        return Dataset(**main_params)


def update_dataset(
        self: Union[Dataset, Dict],
        other: Union[Dataset, Dict],
        merge_dataset: bool = False
) -> Union[Dataset, Dict, None]:
    if not other:
        warnings.warn("Dataset update error!", category=UserWarning)
    if not self:
        return other if isinstance(other, Dataset) else Dataset.resolve_dataset_from_dict(other)

    self_dict = self.dataset_dict if isinstance(self, Dataset) else self
    other_dict = other.dataset_dict if isinstance(other, Dataset) else other

    if "main" not in self_dict.keys() and "main" in other_dict.keys():
        self_dict["main"] = other_dict["main"]

    if merge_dataset:
        dataset_self = self_dict.get("main", {}).get("dataset", [])
        dataset_other = self_dict.get("other", {}).get("dataset", [])
        if len(dataset_self) == len(dataset_other):
            dataset_self.sort(key=lambda row: row["instance_id"])
            dataset_other.sort(key=lambda row: row["instance_id"])
            for r1, r2 in zip(dataset_self, dataset_other):
                if r1["instance_id"] == r2["instance_id"]:
                    r1.update(r2)

    if "sub" in other_dict.keys():
        self_dict.setdefault("sub", dict()).update(other_dict["sub"])

    result = Dataset.resolve_dataset_from_dict(self_dict)

    return result


class DataLoader:
    """
    DataLoader: Comprehensive Data Management for Text-to-SQL Processing

    The DataLoader class is responsible for managing all datasets and database schemas
    required for a single Text-to-SQL process. It serves as the central hub for data
    preparation, validation, and organization in the Squrve framework.

    Key Responsibilities:
    - Manages complete datasets and database schemas for Text-to-SQL tasks
    - Ensures literal path validation and security, especially when using Squrve's
      integrated baseline datasets
    - Supports partitioning central format database schemas into field-level schema
      lists for enhanced indexing and retrieval capabilities
    - Facilitates few-shot chain-of-thought learning and external knowledge extraction
    - Provides robust data source validation and preprocessing

    Features:
    - Multi-database support with flexible configuration options
    - Vector store integration for efficient schema indexing and retrieval
    - Few-shot learning capabilities with customizable examples
    - External knowledge integration through configurable functions
    - Comprehensive path validation and security checks
    - Automatic data preparation and preprocessing workflows

    Architecture:
    The DataLoader integrates with the Router component for parameter initialization
    and configuration management, ensuring consistent behavior across the Squrve
    system. It supports both single and multi-database scenarios with flexible
    schema management capabilities.
    """

    def __init__(
            self,
            router: Optional[Router] = None,
            llm: Optional[LLM] = None,
            dataset: Optional[List] = None,
            schema: Optional[List] = None,
            embed_model_source: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            data_source: Optional[Union[str, List[str], Dict]] = None,
            data_source_dir: Optional[str] = None,
            overwrite_exist_file: Optional[bool] = None,
            need_few_shot: Optional[bool] = None,
            few_shot_num: Optional[int] = None,
            few_shot_save_dir: Optional[str] = None,
            few_shot_range: Optional[Union[int, str, List[str], List[int]]] = None,
            need_external: Optional[bool] = None,
            external_function: Optional[Callable] = None,
            external_range: Optional[List[str]] = None,
            external_save_dir: Optional[str] = None,
            db_path: Optional[Union[str, List[str], Dict]] = None,
            skip_schema_init: Optional[bool] = None,
            schema_source: Optional[Union[str, List[str], Dict]] = None,
            multi_database: Optional[Union[bool, List[bool], Dict]] = None,
            vector_store: Optional[Union[str, List[str], Dict]] = None,
            schema_source_dir: Optional[str] = None,
            need_build_index: Optional[bool] = None,
            index_range: Optional[Union[bool, List[str]]] = None,
            is_prepare_data: Optional[bool] = None
    ):
        self.router = router if router else Router()

        self.llm = llm if llm else self.init_llm()

        self.embed_model_source = embed_model_source if embed_model_source else self.router.embed_model_source
        self.embed_model_name = embed_model_name if embed_model_name else self.router.embed_model_name

        self.data_source = data_source if data_source else self.router.data_source
        self.data_source_dir = data_source_dir if data_source_dir else self.router.data_source_dir
        self.default_data_file_name = self.router.default_data_file_name
        self.overwrite_exist_file = overwrite_exist_file if overwrite_exist_file is not None else self.router.overwrite_exist_file

        self.need_few_shot = need_few_shot if need_few_shot is not None else self.router.need_few_shot
        self.few_shot_num = few_shot_num if few_shot_num else self.router.few_shot_num
        self.sys_few_shot_dir = self.router.sys_few_shot_dir
        self.few_shot_save_dir = few_shot_save_dir if few_shot_save_dir else self.router.few_shot_save_dir
        self.few_shot_range = few_shot_range if few_shot_range else self.router.few_shot_range

        self.need_external = need_external if need_external is not None else self.router.need_external
        self.external_function = external_function if external_function else self.init_default_external_function(
            self.router.default_get_external_function)
        self.external_range = external_range if external_range else self.router.external_range
        self.external_save_dir = external_save_dir if external_save_dir else self.router.external_save_dir

        self.skip_schema_init = skip_schema_init if skip_schema_init is not None else self.router.skip_schema_init
        self.schema_source = schema_source if schema_source else self.router.schema_source
        """
        schema_save_source is the actual saved path, obtained through the get_schema_source_by_index method.
        - Example:
        {
            "schema_1": {
                "source_dir": "../files/schema_source/schema_1", 
                "multi_database": False,
                "vector_store": "vector_store"
            }
        }
        """
        self.schema_save_source: dict = {}
        self.multi_database = multi_database if multi_database else self.router.multi_database
        self.vector_store = vector_store if vector_store else self.router.vector_store

        self.schema_source_dir = schema_source_dir if schema_source_dir else self.router.schema_source_dir
        self.default_schema_dir_name = self.router.default_schema_dir_name
        self.need_build_index = need_build_index if need_build_index is not None else self.router.need_build_index
        self.index_range = index_range if index_range is not None else self.router.index_range

        self.obtained_db_path: dict = {}

        self.__init_data_source__(dataset, self.router.db_path if db_path is None else db_path)
        self.__init_schema_source__(schema)
        if is_prepare_data or self.router.is_prepare_data:
            self.prepare_dataset()

    def get_data_source_index(self, output_format: Optional[str] = None):
        """ Return the indexes of all data sources. """
        if self.data_source is None:
            return None

        if isinstance(self.data_source, str):
            path_ = Path(self.data_source)
            index_ = path_.stem
            return [index_] if output_format == "list" else index_

        if isinstance(self.data_source, list):
            return list(range(len(self.data_source)))

        if isinstance(self.data_source, dict):
            return list(self.data_source.keys())

        return None

    def get_data_source_by_index(self, index_: Optional[Union[int, str, List[str], List[int]]],
                                 output_format: Optional[str] = None):
        if index_ is None:
            return self.data_source

        source_ = None
        if isinstance(index_, int):
            if isinstance(self.data_source, list):
                source_ = self.data_source[index_] if 0 <= index_ < len(self.data_source) else None
            elif isinstance(self.data_source, dict):
                source_ = self.data_source.get(str(index_), None)
        elif isinstance(index_, str):
            if isinstance(self.data_source, str):
                source_ = self.data_source if Path(self.data_source).stem == index_ else None
            if isinstance(self.data_source, dict):
                source_ = self.data_source.get(index_, None)
        elif isinstance(index_, list):
            # 传入列表索引列表时，以字典形式返回所有 datasource
            unique_index = list(dict.fromkeys(index_))
            sub_data_source = {}
            if isinstance(self.data_source, dict):
                for ind in unique_index:
                    sub_data_source[str(ind)] = self.data_source.get(str(ind), None)
            elif isinstance(unique_index[0], int) and isinstance(self.data_source, list):
                for ind in unique_index:
                    if isinstance(ind, int) and 0 <= ind < len(self.data_source):
                        sub_data_source[str(ind)] = self.data_source[ind]
                    else:
                        sub_data_source[str(ind)] = None

            source_ = sub_data_source if sub_data_source else None

        if output_format == "dict":
            return source_ if not source_ or isinstance(source_, dict) else {str(index_): source_}

        return source_

    def get_schema_source_index(self):
        return list(self.schema_save_source.keys()) if self.schema_save_source else None

    def get_schema_source_by_index(self, index_: Optional[Union[int, str, List[str], List[int]]],
                                   key: Optional[str] = None):
        """
        Get schema source by index.
        
        Args:
            index_: Index or list of indices to retrieve. If None, returns all schema sources.
            key: Optional key to extract from the schema source metadata.
            
        Returns:
            Dict containing the requested schema sources, or None if not found.
        """
        if not index_:
            # Return all schema sources, optionally filtered by key
            if key is None:
                return self.schema_save_source
            else:
                # Filter all schema sources by the specified key
                filtered_sources = {}
                for idx, meta_source in self.schema_save_source.items():
                    if meta_source and key in meta_source:
                        filtered_sources[idx] = meta_source[key]
                return filtered_sources if filtered_sources else None

        sub_data_source = {}
        if isinstance(index_, (int, str)):
            meta_source = self.schema_save_source.get(str(index_), None)
            if meta_source:
                if key is None:
                    sub_data_source[str(index_)] = meta_source
                elif key in meta_source:
                    sub_data_source[str(index_)] = meta_source[key]
        else:
            # Handle list of indices
            unique_index = list(dict.fromkeys(index_))
            for ind in unique_index:
                meta_source = self.schema_save_source.get(str(ind), None)
                if meta_source:
                    if key is None:
                        sub_data_source[str(ind)] = meta_source
                    elif key in meta_source:
                        sub_data_source[str(ind)] = meta_source[key]

        return sub_data_source if sub_data_source else None

    def init_llm(self, use: Optional[str] = None, **kwargs):
        llm_ = None
        try:
            init_args = {
                "api_key": self.router.api_key,
                "model_name": self.router.model_name,
                "context_window": self.router.context_window,
                "max_token": self.router.max_token,
                "top_p": self.router.top_p,
                "temperature": self.router.temperature,
                "time_out": self.router.time_out
            }
            for key, val in kwargs.items():
                init_args[key] = val

            if use is None:
                use = self.router.use
            if use == "deepseek":
                from core.llm.DeepseekModel import DeepseekModel
                llm_ = DeepseekModel(**init_args)
            elif use == "qwen":
                from core.llm.QwenModel import QwenModel
                llm_ = QwenModel(**init_args)

        except Exception as e:
            warnings.warn(f"Failed to create LLM: {e}.", category=UserWarning)

        return llm_

    @staticmethod
    def load_llm_by_args(
            use: str,
            api_key: str,
            model_name: str,
            context_window: int = 120000,
            max_token: int = 4000,
            top_p: float = 0.8,
            temperature: float = 0.7,
            time_out: float = 300.0,
            **kwargs
    ):
        llm_ = None
        try:
            init_args = {
                "api_key": api_key,
                "model_name": model_name,
                "context_window": context_window,
                "max_token": max_token,
                "top_p": top_p,
                "temperature": temperature,
                "time_out": time_out
            }
            for key, val in kwargs.items():
                init_args[key] = val
            if use is None:
                raise Exception("The `use` for LLM selection is empty!")
            if use == "deepseek":
                from core.llm.DeepseekModel import DeepseekModel
                llm_ = DeepseekModel(**init_args)
            elif use == "qwen":
                from core.llm.QwenModel import QwenModel
                llm_ = QwenModel(**init_args)

        except Exception as e:
            warnings.warn(f"Failed to create LLM: {e}.", category=UserWarning)

        return llm_

    def __init_data_source__(self, dataset: Optional[List] = None,
                             db_path: Optional[Union[str, List[str], Dict]] = None):

        def init_single_item(data_source_: str, index_: Optional[Union[int, str]] = None):
            if data_source_.count(":") != 2:
                init_single_db_path(index_)
                return
            if not isinstance(self.data_source, str):
                warnings.warn("data_source is not a string, cannot split", category=UserWarning)
                return
            file_name_ = "_".join(self.data_source.split(":"))
            if self.data_source_dir is None:
                warnings.warn("data_source_dir is None", category=UserWarning)
                return
            save_data_source = Path(self.data_source_dir) / (file_name_ + ".json")
            if self.overwrite_exist_file or not save_data_source.exists():
                self.init_benchmark_dataset(data_source_, index_, save_data_source=save_data_source)
            if index_ is None:
                index_ = file_name_
            if isinstance(self.data_source, str):
                self.data_source = str(save_data_source)
            else:
                if isinstance(self.data_source, dict):
                    self.data_source[index_] = str(save_data_source)
            init_single_db_path(index_)

        def check_db_path(db_path_: str):
            if db_path_ is None:
                return None
            if db_path_.count(":") == 1:
                # use benchmark db path
                id_, sub_id_ = db_path_.split(":")
                db_path_result = self.router.get_benchmark_db_path(id_, sub_id_)
                if db_path_result is not None:
                    db_path_ = db_path_result
            if db_path_ and Path(db_path_).exists():
                return db_path_
            return None

        def init_single_db_path(index_: Optional[Union[int, str]] = None):
            if db_path is None:
                return
            if isinstance(db_path, str):
                db_path_ = check_db_path(db_path)
                if db_path_ and index_ is not None:
                    self.set_db_path(index_, db_path_)
            elif isinstance(db_path, list):
                if isinstance(self.data_source, list) and len(db_path) == len(self.data_source):
                    if isinstance(index_, int) and 0 <= index_ < len(db_path):
                        db_path_ = check_db_path(db_path[index_])
                        if db_path_:
                            self.set_db_path(index_, db_path_)
            elif isinstance(db_path, dict):
                if index_ is not None:
                    db_path_value = db_path.get(str(index_), None)
                    if db_path_value is not None:
                        db_path_ = check_db_path(db_path_value)
                        if db_path_:
                            self.set_db_path(index_, db_path_)

        if self.data_source is not None:
            if isinstance(self.data_source, str):
                ind = self.get_data_source_index()
                if isinstance(ind, (int, str)):
                    init_single_item(self.data_source, ind)

            elif isinstance(self.data_source, list):
                for ind, source_ in enumerate(self.data_source):
                    init_single_item(source_, ind)

            elif isinstance(self.data_source, dict):
                for key_, source_ in self.data_source.items():
                    init_single_item(source_, key_)

        if dataset:
            if self.data_source_dir is None or self.default_data_file_name is None:
                warnings.warn("data_source_dir or default_data_file_name is None", category=UserWarning)
                return
            new_data_source = Path(self.data_source_dir) / self.default_data_file_name
            if self.overwrite_exist_file or not new_data_source.exists():
                save_dataset(dataset=dataset, new_data_source=new_data_source)
            self.update_data_source(save_path_=new_data_source)
            db_path = check_db_path(db_path) if isinstance(db_path, str) else None
            if db_path:
                self.set_db_path(new_data_source.stem, db_path)

    def update_data_source(
            self,
            save_path_: Union[str, PathLike],
            data_source_index: Optional[Union[int, str]] = None
    ):
        save_path_ = str(save_path_)
        if self.data_source is None:
            self.data_source = save_path_
        elif isinstance(self.data_source, str):
            origin_data_source = self.get_data_source_index()
            dataset_index = Path(save_path_).stem if data_source_index is None else data_source_index
            if isinstance(origin_data_source, (str, int)):
                self.data_source = {
                    str(origin_data_source): self.data_source,
                    str(dataset_index): save_path_
                }
        elif isinstance(self.data_source, List):
            # data_source_index is not effective if self.data_source is list object.
            self.data_source.append(save_path_)
        elif isinstance(self.data_source, dict):
            dataset_index = Path(self.default_data_file_name).stem if data_source_index is None else data_source_index
            self.data_source[str(dataset_index)] = save_path_
            # return dataset_index

    @staticmethod
    def init_default_external_function(default_external_function: str) -> Callable:
        from core.LinkAlign.tools.external import summary_external_knowledge
        if default_external_function == "LinkAlign":
            return summary_external_knowledge

        return summary_external_knowledge

    def set_db_path(self, index: Union[int, str], db_path: Union[str, PathLike]):
        if db_path is None:
            warnings.warn("Database path cannot be None.", category=UserWarning)
            return
        if index in self.obtained_db_path.keys():
            warnings.warn(f"Datasource {index} database path already exists.", category=UserWarning)
        self.obtained_db_path[index] = str(db_path)

    def get_db_path(self, index: Union[int, str]):
        return self.obtained_db_path.get(index, None)

    def init_benchmark_dataset(
            self,
            identifier: str,
            data_source_index: Optional[Union[int, str]],
            is_save_dataset: bool = True,
            save_data_source: Optional[Union[str, PathLike]] = None,
            update_db_path: bool = True
    ):
        """ Locate dataset by identifier, initialize and return the dataset list. """

        try:
            id_, sub_id_, filter_by_ = identifier.split(":")
        except ValueError:
            warnings.warn("Identifier format error: expected 'id:sub_id:filter'.", category=UserWarning)
            return None

        meta_data = next((x for x in self.router.benchmark if x.get('id') == id_), None)
        if meta_data is None:
            warnings.warn("Invalid datasource benchmark dataset id; dataset does not exist.", category=UserWarning)
            return None

        origin_data_source = Path(meta_data.get('root_path', ''))
        external_path = None

        if sub_id_:
            if not meta_data.get('has_sub', False):
                warnings.warn("Sub-dataset not supported for this benchmark dataset.", category=UserWarning)
                return None

            sub_meta_data = next((x for x in meta_data.get('sub_data', []) if x.get('sub_id') == sub_id_), None)
            if sub_meta_data is None:
                warnings.warn("Invalid datasource benchmark dataset sub_id; sub-dataset does not exist.",
                              category=UserWarning)
                return None

            origin_data_source = origin_data_source / sub_id_ / "dataset.json"

            if sub_meta_data.get("use_local_external", False):
                external_path = Path(meta_data['root_path']) / sub_id_ / "external"
        else:
            origin_data_source = origin_data_source / "dataset.json"

        if not external_path:
            external_path = Path(meta_data['root_path']) / "external" if meta_data.get("external", False) else None

        dataset_ = load_dataset(origin_data_source)
        if dataset_ is None:
            warnings.warn("Failed to load dataset from the datasource benchmark dataset parameters.",
                          category=UserWarning)
            return None

        # Add external paths to dataset rows if external_path exists
        if external_path and isinstance(dataset_, list):
            for row in dataset_:
                if isinstance(row, dict):
                    file_name = row.get("external_path", "")
                    row["external_path"] = str(external_path / file_name) if file_name else ""

        # Filter dataset
        if isinstance(dataset_, list):
            dataset_ = filter_dataset(dataset_=dataset_, filter_by_=filter_by_)

        # Update database path if requested
        if update_db_path and data_source_index is not None:
            benchmark_db_path = self.router.get_benchmark_db_path(id_, sub_id_)
            if benchmark_db_path is not None:
                self.set_db_path(data_source_index, benchmark_db_path)

        # Save dataset if requested
        if is_save_dataset:
            if save_data_source is None:
                file_name_ = "_".join(identifier.split(":")) + ".json"
                save_data_source = Path(self.data_source_dir) / file_name_
            save_dataset(dataset=dataset_, new_data_source=save_data_source)

        return dataset_

    def __init_schema_source__(self, schema: Optional[List] = None):
        if schema is None:
            schema = []

        def init_single_item(source_: str, index_: Optional[Union[str, int]] = None):
            if ":" in Path(source_).stem:
                index_ = source_.replace(":", "_") if not index_ else index_
                multi_db_ = self.query_multi_database(index_, self.multi_database)
                vector_store_ = self.query_vector_store(index_, self.vector_store)
                if self.schema_source_dir is None:
                    warnings.warn("schema_source_dir is None, cannot proceed", category=UserWarning)
                    return
                save_schema_source = Path(self.schema_source_dir) / source_.replace(":", "_")
                if self.skip_schema_init:
                    save_schema_source = save_schema_source / "schema.json"
                self.init_benchmark_schema(source_, multi_db_, save_schema_source=save_schema_source,
                                           skip_schema_init=self.skip_schema_init)
                self.update_schema_save_source({index_: str(save_schema_source)}, multi_db_, vector_store_)
            else:
                index_ = Path(source_).stem if not index_ else index_
                multi_db_ = self.query_multi_database(index_, self.multi_database)
                vector_store_ = self.query_vector_store(index_, self.vector_store)
                save_schema_source = source_ if self.skip_schema_init else Path(self.schema_source_dir) / str(index_)
                if not self.skip_schema_init:
                    self.central_schema_process(source_, save_schema_source=save_schema_source, multi_db=multi_db_)
                self.update_schema_save_source({index_: str(save_schema_source)}, multi_db_, vector_store_)

        if isinstance(self.schema_source, str):
            source = self.schema_source
            init_single_item(source)
        elif isinstance(self.schema_source, list):
            for ind, source in enumerate(self.schema_source):
                init_single_item(source, ind)
        elif isinstance(self.schema_source, dict):
            for key_, source in self.schema_source.items():
                init_single_item(source, key_)

        if schema:
            multi_db = self.multi_database if isinstance(self.multi_database, bool) else False
            vec_store = self.vector_store if isinstance(self.vector_store, str) else "vector_store"
            if self.schema_source_dir is None or self.default_schema_dir_name is None:
                warnings.warn("schema_source_dir or default_schema_dir_name is None", category=UserWarning)
                return
            save_path = Path(self.schema_source_dir) / self.default_schema_dir_name
            save_path = save_path / "schema.json" if self.skip_schema_init else save_path
            if self.skip_schema_init:
                save_dataset(schema, new_data_source=save_path)
            else:
                self.central_schema_process(schema, save_schema_source=save_path, multi_db=multi_db)
            schema_index = self.default_schema_dir_name
            self.update_schema_save_source({schema_index: str(save_path)}, multi_db, vec_store)

    def query_multi_database(
            self,
            ind: Union[int, str],
            multi_database: Optional[Union[bool, List[bool], Dict]] = None
    ):
        if multi_database is None:
            multi_database = self.multi_database
        try:
            if isinstance(multi_database, bool):
                return multi_database
            elif isinstance(multi_database, list):
                if isinstance(ind, int) and 0 <= ind < len(multi_database):
                    return multi_database[ind]
                return False
            elif isinstance(multi_database, Dict):
                return multi_database.get(ind, False)
        except Exception as e:
            print(f"Error occurred while checking multi-database configuration: {e}")

        return False

    def query_vector_store(
            self,
            ind: Union[int, str],
            vector_store: Optional[Union[str, List[str], Dict]] = None
    ):
        if vector_store is None:
            vector_store = self.vector_store
        try:
            if isinstance(vector_store, str):
                return vector_store
            elif isinstance(vector_store, list):
                if isinstance(ind, int) and 0 <= ind < len(vector_store):
                    return vector_store[ind]
                return "vector_store"
            elif isinstance(vector_store, dict):
                return vector_store.get(ind, "vector_store")
        except Exception as e:
            print(f"Error occurred while querying vector storage path: {e}")

        return "vector_store"

    def update_schema_save_source(
            self,
            schema_source: Union[str, PathLike, List[str], Dict],
            multi_database: Union[bool, List[bool], Dict] = False,
            vector_store: Union[str, List[str], Dict] = "vector_store"
    ):
        if isinstance(schema_source, str):
            schema_source = Path(schema_source)

        if isinstance(schema_source, Path):
            schema_index = schema_source.stem
            multi_db = self.query_multi_database(schema_index, multi_database)
            vec_store = self.query_vector_store(schema_index, vector_store)

            self.schema_save_source[schema_index] = {"source_dir": str(schema_source),
                                                     "multi_database": multi_db,
                                                     "vector_store": vec_store}
        elif isinstance(schema_source, List):
            for ind, schema_ in schema_source:
                multi_db = self.query_multi_database(ind, multi_database)
                vec_store = self.query_vector_store(ind, vector_store)
                self.schema_save_source[ind] = {"source_dir": schema_,
                                                "multi_database": multi_db,
                                                "vector_store": vec_store}

        elif isinstance(schema_source, Dict):
            for key_, schema_ in schema_source.items():
                multi_db = self.query_multi_database(key_, multi_database)
                vec_store = self.query_vector_store(key_, vector_store)
                self.schema_save_source[key_] = {"source_dir": schema_,
                                                 "multi_database": multi_db,
                                                 "vector_store": vec_store}

    def central_schema_process(
            self,
            schema: Union[str, PathLike, List[Dict]],
            is_save_schema: bool = True,
            save_schema_source: Optional[Union[str, PathLike]] = None,
            multi_db: bool = False
    ):
        """
        Convert schema in central format to parallel format.

        Args:
            schema: Path to schema file or list of dicts in central format.
            is_save_schema: Whether to save the converted schema.
            save_schema_source: Path to save the schema, required if is_save_schema is True.
            multi_db: Whether the schema is for multiple databases.

        Returns:
            Converted schema in parallel format, or None if invalid input.
        """
        # Load schema from file if it's a path
        if isinstance(schema, (str, PathLike)):
            schema = Path(schema)
        if isinstance(schema, Path):
            if not schema.exists() or schema.is_dir():
                warnings.warn("Provided schema path does not exist or is a directory.", category=UserWarning)
                return None
            loaded_schema = load_dataset(schema)
            if isinstance(loaded_schema, list):
                schema = loaded_schema
            else:
                warnings.warn("Loaded schema is not a list", category=UserWarning)
                return None

        if not isinstance(schema, list) or not all(isinstance(item, dict) for item in schema):
            warnings.warn("Schema must be a list of dictionaries after loading.", category=UserWarning)
            return None

        # Process each central-format schema item into parallel format
        processed_schema = [single_central_process(row) for row in schema]

        # Save schema if required
        if is_save_schema:
            if save_schema_source is None:
                if self.schema_source_dir is None or self.default_schema_dir_name is None:
                    warnings.warn("schema_source_dir or default_schema_dir_name is None", category=UserWarning)
                    return processed_schema
                save_schema_source = Path(self.schema_source_dir) / self.default_schema_dir_name
            self.save_schema(schema=processed_schema, multi_db=multi_db, schema_save_source=save_schema_source)

        return processed_schema

    @classmethod
    def save_schema(
            cls,
            schema: Optional[List[List[Dict]]] = None,
            multi_db: bool = False,
            schema_save_source: Optional[Union[str, PathLike]] = None
    ):
        """
        Save the schema in parallel format to the specified directory.

        Args:
            schema: A list of lists of dicts, representing the parallel schema format.
            multi_db: Whether the schema contains multiple databases.
            schema_save_source: The directory path where the schema should be saved.
        """
        if not isinstance(schema, list) or not all(
                isinstance(table, list) and all(isinstance(column, dict) for column in table)
                for table in schema
        ):
            warnings.warn("Schema is not in a valid parallel format. Skipping save.", category=UserWarning)
            return None

        if schema_save_source is None:
            raise ValueError("schema_save_source must be provided.")

        schema_save_source = Path(schema_save_source)
        if schema_save_source.suffix:
            warnings.warn("The save path must be a directory, not a file.", category=UserWarning)
            return None

        subdir = "multi_db" if multi_db else "single_db"
        save_root = schema_save_source / subdir

        for db_schema in schema:
            for row in db_schema:
                file_name = transform_name(row['table_name'], row['column_name']) + ".json"
                save_path = (
                    save_root / file_name if multi_db
                    else save_root / row['db_id'] / file_name
                )
                save_dataset(row, new_data_source=save_path)

        return None

    def init_benchmark_schema(
            self,
            identifier: str,
            multi_db: bool = False,
            is_save_schema: bool = True,
            save_schema_source: Optional[Union[str, PathLike]] = None,
            skip_schema_init: Optional[bool] = None
    ):
        """
        Initialize benchmark dataset schema based on the given identifier.

        Args:
            identifier: Dataset identifier in the format 'id:sub_id'.
            multi_db: Whether the schema contains multiple databases.
            is_save_schema: Whether to save the schema after loading.
            save_schema_source: Custom path to save the schema.
            skip_schema_init: Whether to skip schema re-initialization.

        Returns:
            The loaded schema, or None if not found or invalid identifier.
        """
        if skip_schema_init is None:
            skip_schema_init = self.skip_schema_init

        try:
            id_, sub_id = identifier.split(":")
        except ValueError:
            warnings.warn("Invalid identifier format. Expected 'id:sub_id'.", category=UserWarning)
            return None

        meta_candidates = [x for x in self.router.benchmark if x['id'] == id_]
        if not meta_candidates:
            warnings.warn("Invalid benchmark dataset ID.", category=UserWarning)
            return None

        meta_data = meta_candidates[0]
        origin_schema_source = Path(meta_data['root_path'])

        if sub_id:
            if not meta_data.get('has_sub'):
                warnings.warn("Sub-dataset is not expected but sub_id provided.", category=UserWarning)
                return None
            sub_meta = [x for x in meta_data['sub_data'] if x['sub_id'] == sub_id]
            if not sub_meta:
                warnings.warn("Invalid sub-dataset ID.", category=UserWarning)
                return None
            origin_schema_source = origin_schema_source / sub_id / "schema.json"
        else:
            origin_schema_source = origin_schema_source / "schema.json"

        # Load the dataset schema
        schema = load_dataset(origin_schema_source)

        if is_save_schema:
            if self.schema_source_dir is None:
                warnings.warn("schema_source_dir is None", category=UserWarning)
                return schema
            default_path = Path(self.schema_source_dir) / identifier.replace(":", "_")
            if skip_schema_init:
                save_path = save_schema_source or (default_path / "schema.json")
                if isinstance(schema, (list, dict)):
                    save_dataset(schema, save_path)
            else:
                save_path = save_schema_source or default_path
                if isinstance(schema, (list, dict)):
                    self.central_schema_process(schema, save_schema_source=save_path, multi_db=multi_db)

        return schema

    def add_few_shot(
            self,
            source_index: Union[int, str, List[str], List[int]] = None,
            few_shot_num: int = None,
            few_shot_save_dir: Union[str, PathLike] = None,
            is_update_dataset: bool = True,
            embed_model_name: str = None
    ):
        """
        Add chain-of-thought examples for specified indices in the dataset.

        :param source_index: Data source index list. If None, apply to all data sources.
        :param few_shot_num: Number of examples to retrieve.
        :param few_shot_save_dir: Local directory to save the examples.
        :param is_update_dataset: Whether to update the dataset with the example file path.
        :param embed_model_name: Name of the embedding model.
        """
        # Set default values
        source_index = source_index or self.get_data_source_index()
        few_shot_num = few_shot_num or self.few_shot_num
        embed_model_name = embed_model_name or self.embed_model_name
        few_shot_save_dir = Path(few_shot_save_dir or self.few_shot_save_dir)

        index_dict = {}
        data_sources = self.get_data_source_by_index(source_index, output_format="dict")

        if data_sources is None:
            return
        if not isinstance(data_sources, dict):
            raise TypeError("Expected 'data_sources' to be a dictionary.")

        for idx, source_path in data_sources.items():
            source_path = Path(source_path)
            if not source_path.exists() or source_path.suffix != ".json":
                raise FileNotFoundError(f"Data source '{source_path}' not found or is not a .json file.")

            dataset = load_dataset(source_path)
            if not isinstance(dataset, list):
                raise TypeError(f"Dataset loaded from '{source_path}' is not a list.")

            save_dir = few_shot_save_dir / str(idx)
            for row in dataset:
                db_type = row.get("db_type")
                if not db_type:
                    continue  # skip if db_type is missing

                data_source_dir = Path(self.sys_few_shot_dir) / db_type
                persist_dir = data_source_dir / "vector_store" / Path(embed_model_name).name
                store_exists = persist_dir.exists() and any(persist_dir.iterdir())

                vector_index = index_dict.get(db_type)
                if not vector_index:
                    vector_index = RagPipeLines.build_index_from_source(
                        data_source=str(data_source_dir),
                        persist_dir=str(persist_dir),
                        is_vector_store_exist=store_exists,
                        index_method="VectorStoreIndex",
                        embed_model_name=embed_model_name
                    )
                    index_dict[db_type] = vector_index

                # Retrieve few-shot reasoning examples
                retriever = RagPipeLines.get_retriever(index=vector_index, similarity_top_k=few_shot_num)
                retrieved_nodes = SchemaLinkingTool.retrieve(
                    retriever_lis=[retriever],
                    query_lis=[row["question"]]
                )

                example_paths = [
                    data_source_dir / node.node.metadata["file_name"]
                    for node in retrieved_nodes
                ]
                example_paths = [p for p in example_paths if p.exists()]

                # Combine examples into context text
                context_parts = [
                    f"-- Example {i + 1}\n{load_dataset(p)}"
                    for i, p in enumerate(example_paths)
                ]
                context = "\n\n".join(context_parts).strip()

                # Save reasoning example
                example_save_path = save_dir / f"{row['instance_id']}.txt"
                save_dataset(context, new_data_source=example_save_path)

                if is_update_dataset:
                    row["reasoning_examples"] = str(example_save_path)

            if is_update_dataset:
                save_dataset(dataset, new_data_source=source_path)

    def add_external(
            self,
            llm_: LLM = None,
            source_index: Union[int, str, List[str], List[int]] = None,
            external_save_dir: Union[str, PathLike] = None,
            external_function: Callable = None,
            is_update_dataset: bool = True,
    ):
        llm_ = llm_ if llm_ else self.llm
        source_index = self.get_data_source_index() if not source_index else source_index
        external_save_dir = self.external_save_dir if not external_save_dir else external_save_dir
        external_save_dir = Path(external_save_dir)
        external_function = self.external_function if not external_function else external_function

        if not llm_ or not external_function:
            return
        temp_data_source = self.get_data_source_by_index(source_index, output_format="dict")
        if temp_data_source is None:
            return

        for index_, source in temp_data_source.items():
            source = Path(source)
            save_dir = external_save_dir / str(index_)
            if not source.exists() or source.suffix != ".json":
                return
            dataset = load_dataset(source)
            assert isinstance(dataset, dict)
            for row in dataset:
                external_path = row.get("external_path", None)
                if not external_path:
                    continue
                external = load_dataset(external_path)
                if not external:
                    continue
                external_save_path_ = save_dir / (row['instance_id'] + '.txt')
                input_args = {
                    "question": row["question"],
                    "llm": llm_,
                    "external": external,
                    "save_path": external_save_path_
                }
                external_function(**input_args)
                if is_update_dataset:
                    row["external"] = str(external_save_path_)

            if is_update_dataset:
                save_dataset(dataset, new_data_source=source)

    def build_index(
            self,
            source_index: Union[int, str, List[str], List[int]] = None,
            embed_model_name: str = None
    ):
        """Build vector indices from specified schema sources."""

        embed_model_name = embed_model_name or self.embed_model_name
        schema_sources = self.get_schema_source_by_index(source_index)

        if schema_sources is None:
            return

        for idx, meta in schema_sources.items():
            source_dir = Path(meta.get("source_dir", ""))
            multi_db = meta.get("multi_database", False)
            vec_store_base = Path(meta.get("vector_store", ""))

            if not source_dir.exists():
                raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
            if source_dir.is_file():
                continue  # Skip if source_dir is a file

            db_type_folder = "multi_db" if multi_db else "single_db"
            root_path = source_dir / db_type_folder

            if not root_path.exists():
                raise FileNotFoundError(f"Expected folder '{root_path}' does not exist.")

            # Build for multi-database mode
            if multi_db:
                vector_store = get_vector_store_dir(
                    root_path, idx,
                    vector_store=vec_store_base,
                    embed_model_name=embed_model_name,
                    multi_database=multi_db
                )

                if vector_store.exists() and any(vector_store.iterdir()):
                    continue

                RagPipeLines.build_index_from_source(
                    data_source=str(root_path),
                    persist_dir=str(vector_store),
                    is_vector_store_exist=False,
                    index_method="VectorStoreIndex",
                    embed_model_name=embed_model_name
                )

            # Build for single-database mode
            else:
                for db_dir in root_path.iterdir():
                    if not db_dir.is_dir():
                        continue

                    vector_store = get_vector_store_dir(
                        root_path, idx, db_dir.name,
                        vector_store=vec_store_base,
                        embed_model_name=embed_model_name,
                        multi_database=multi_db
                    )

                    if vector_store.exists() and any(vector_store.iterdir()):
                        continue

                    RagPipeLines.build_index_from_source(
                        data_source=str(db_dir),
                        persist_dir=str(vector_store),
                        is_vector_store_exist=False,
                        index_method="VectorStoreIndex",
                        embed_model_name=embed_model_name
                    )

    def prepare_dataset(
            self,
            need_few_shot: bool = None,
            need_external: bool = None,
            need_index: bool = None
    ):
        need_few_shot = need_few_shot if need_few_shot else self.need_few_shot
        need_external = need_external if need_external else self.need_external
        need_build_index = need_index if need_index else self.need_build_index

        if need_few_shot:
            self.add_few_shot(source_index=self.few_shot_range)
        if need_external:
            self.add_external(source_index=self.external_range)
        if need_build_index:
            self.build_index(source_index=self.index_range)

    def generate_dataset(
            self,
            data_source_index: Union[int, str],
            schema_source_index: Union[int, str],
            random_size: float = None,
            filter_by: str = None,
            is_schema_final: bool = None,
            **kwargs
    ):
        data_source = self.get_data_source_by_index(data_source_index)
        schema_source = self.get_schema_source_by_index(schema_source_index).get(schema_source_index)
        if not data_source or not schema_source:
            return None
        if is_schema_final is None:
            is_schema_final = False

        init_args = {
            "data_source": data_source,
            "schema_source": schema_source.get("source_dir"),
            "dataset_index": data_source_index,
            "schema_index": schema_source_index,
            # the parameter below can be set by the `meta` in `task meta`.
            "random_size": random_size,
            "filter_by": filter_by,
            "multi_database": schema_source.get("multi_database"),
            "vector_store": schema_source.get("vector_store"),
            "embed_model_name": self.embed_model_name,
            "db_credential": self.router.credential,
            "db_path": self.get_db_path(data_source_index),
            "is_schema_final": is_schema_final,
        }
        for key, val in kwargs.items():
            # Decode directly may throw error due to repetition
            init_args[key] = val

        dataset = Dataset(**init_args)

        return dataset


def filter_dataset(
        dataset_: List[Dict],
        filter_by_: Union[str, List[str]] = "has_label",
        spliter_: str = '.'
) -> List[Dict]:
    """
    Filter samples from dataset based on given filter criteria.

    Args:
        dataset_ (List[Dict]): List of data samples.
        filter_by_ (Union[str, List[str]]): Filter conditions, can be a string or a list of conditions.
        spliter_ (str): Separator for string-based filter input.

    Returns:
        List[Dict]: Filtered dataset.
    """

    if isinstance(filter_by_, str):
        filter_by_ = filter_by_.split(spliter_)

    # Parse filter conditions into a dictionary
    filter_dict = {}
    for condition in filter_by_:
        parts = condition.split("-")
        if len(parts) == 1:
            filter_dict[parts[0]] = {"value": True}
        elif len(parts) == 2:
            filter_dict[parts[0]] = {"value": parts[1]}
        elif len(parts) == 3:
            filter_dict[parts[0]] = {"operator": parts[1], "value": parts[2]}
        else:
            raise ValueError(f"Invalid filter condition format: {condition}")

    # Operator mapping
    op_map = {
        "l": lambda x, y: x < y,
        "e": lambda x, y: x == y,
        "m": lambda x, y: x > y,
        "le": lambda x, y: x <= y,
        "me": lambda x, y: x >= y
    }

    def apply_numeric_filter(data, key, value_key, length=False):
        op = filter_dict[key].get("operator", "l")
        val_ = int(filter_dict[key].get("value", 100))
        comparator = op_map.get(op)
        if not comparator:
            raise ValueError(f"Unsupported operator: {op}")
        if length:
            return [item for item in data if comparator(len(item.get(value_key, "")), val_)]
        return [item for item in data if comparator(item.get(value_key, 0), val_)]

    # Apply filters
    if "db_size" in filter_dict:
        dataset_ = apply_numeric_filter(dataset_, "db_size", "db_size")

    if "difficulty" in filter_dict:
        val = filter_dict["difficulty"].get("value", "easy")
        dataset_ = [row for row in dataset_ if row.get("difficulty", "easy") == val]

    if "db_type" in filter_dict:
        val = filter_dict["db_type"].get("value", "sqlite")
        dataset_ = [row for row in dataset_ if row.get("db_type", "sqlite") == val]

    if "ques_length" in filter_dict:
        dataset_ = apply_numeric_filter(dataset_, "ques_length", "question", length=True)

    if "query_length" in filter_dict:
        dataset_ = apply_numeric_filter(dataset_, "query_length", "query", length=True)

    if "has_label" in filter_dict:
        val = filter_dict["has_label"].get("value", "query")
        val = "query" if val is True else val
        dataset_ = [row for row in dataset_ if val in row]

    return dataset_


def get_vector_store_dir(
        source: Union[str, PathLike] = None,  # <schema_source_dir> / <schema_source_index> / [`multi_db`]
        source_index: str = None,
        db_id: str = None,  # multi_database 为 False，该参数不能为空
        vector_store: Union[str, PathLike] = None,
        embed_model_name: str = None,
        multi_database: bool = False,
        is_raw_source: bool = False
):
    source = Path(source)
    vector_store = Path(vector_store)

    if not embed_model_name:
        return None
    embed_model_name = Path(embed_model_name).name

    if is_raw_source and source:
        source = source / "multi_db" if multi_database else source / "single_db"

    if multi_database:
        if vector_store.is_absolute():
            vector_store = vector_store / source_index / "multi_db" / embed_model_name
        else:
            vector_store = source / vector_store.name / embed_model_name
    else:
        if vector_store.is_absolute():
            vector_store = vector_store / source_index / "single_db" / db_id / embed_model_name
        else:
            vector_store = source / db_id / vector_store.name / embed_model_name

    return vector_store


def transform_name(table_name, col_name):
    prefix = f"{table_name}_{col_name}"
    prefix = prefix if len(prefix) < 100 else prefix[:100]

    syn_lis = ["(", ")", "%", "/"]
    for syn in syn_lis:
        prefix = prefix.replace(syn, "_")

    return prefix


def single_central_process(row: Dict):
    column_info_lis = []
    db_id = row["db_id"]
    db_type = row["db_type"]
    tables = row["table_names_original"]
    columns = [col for col in row["column_names_original"] if col[0] != -1]
    types = row["column_types"]

    descriptions = row.get("column_descriptions", [])  # [Opt.]
    pro_infos = row.get("table_to_projDataset", {})  # [Opt.]

    for ind, (table_ind, col_name) in enumerate(columns):
        column_info_lis.append({
            "db_id": db_id,
            "db_type": db_type,
            "table_name": tables[table_ind],
            "column_name": col_name,
            "column_types": types[ind],
            "column_descriptions": descriptions[ind][1] if descriptions and ind < len(descriptions) else "",
            "table_to_projDataset": pro_infos[tables[table_ind]] if pro_infos else ""
        })

    return column_info_lis


if __name__ == "__main__":
    print(1)
