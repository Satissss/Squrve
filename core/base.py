import json
import warnings
from pathlib import Path
from typing import Union, Dict, List


class Router:
    """
    Router is the central configuration manager for a Text-to-SQL pipeline.

    It supports initialization via a JSON config file, parameter dictionary, or command-line arguments.
    The Router handles all parameters required for a complete Text-to-SQL run, including model settings,
    embedding configuration, dataset sources, database schemas, execution logic, and saving options.

    This class is essential for orchestrating the system and is required by other key modules,
    such as the DataLoader (for loading datasets and database schemas) and Engine (for managing the core execution flow).
    """

    _sys_config_path = "../config/sys_config.json"
    _demo_config_path = "../config/demo_config.json"

    """ CONFIG """
    _config: dict

    """ API KEY """
    _api_key: dict

    """ LLM """
    _use: str
    _model_name: str
    _context_window: int
    _max_token: int
    _top_p: float
    _temperature: float
    _time_out: float

    """ TEXT EMBED"""
    _embed_model_source: str
    _embed_model_name: str

    """ DATASET """
    _data_source: Union[None, str, List[str], Dict]
    _data_source_dir: str
    _overwrite_exist_file: bool
    _default_data_file_name: str

    _need_few_shot: bool
    _few_shot_num: int
    _sys_few_shot_dir: str
    _few_shot_save_dir: str
    _few_show_range: Union[int, str, List[str], List[int]]

    _need_external: bool
    _default_get_external_function: str
    _external_range: Union[int, str, List[str], List[int]]
    _external_save_dir: str

    _db_path: Union[str, List[str], Dict]

    """ DATABASE """
    _skip_schema_init: bool
    _schema_source: Union[None, str, List[str], Dict]
    _multi_database: Union[bool, List[bool], Dict]
    _vector_store: Union[None, str, List[str], Dict]
    _schema_source_dir: str
    _default_schema_dir_name: str

    _need_build_index: bool
    _index_method: str
    _index_range: Union[bool, List[str]]

    """ ROUTER """
    _use_demo: bool

    """ DATALOADER """
    _is_prepare_data: bool

    """ REDUCER """
    _reduce_type: Union[None, str]
    _reduce_output_format: str
    _is_save_reduce: bool
    _reduce_save_dir: str

    """ PARSER """
    _parse_type: Union[None, str]
    _is_save_parse: bool
    _parse_save_dir: Union[None, str]
    _parse_output_format: str

    """ GENERATOR"""
    _generate_type: Union[None, str]
    _is_save_generate: bool
    _generate_save_dir: bool

    """ TASK """
    _task_meta: Union[Dict, List[Dict]]
    _cpx_task_meta: Union[Dict, List[Dict]]
    _default_log_save_dir: str
    _is_save_dataset: bool
    _open_parallel: bool
    _max_workers: int

    """ ENGINE """
    _exec_process: Union[None, List[str]]

    """ BENCHMARK """
    _benchmark: List

    """ CREDENTIAL """
    _credential: dict

    def __init__(
            self,
            config_path: str = None,
            api_key: dict = None,
            use: str = "qwen",
            model_name: str = "qwen-turbo",
            context_window: int = 120000,
            max_token: int = 8000,
            top_p: float = 0.9,
            temperature: float = 0.75,
            time_out: float = 300.0,
            embed_model_source: str = "HuggingFace",
            embed_model_name: str = "BAAI/bge-large-en-v1.5",
            use_demo: bool = False,
            is_prepare_data: bool = True,
            data_source: Union[str, List[str], Dict] = None,
            data_source_dir: str = None,
            overwrite_exist_file: bool = True,
            need_few_shot: bool = False,
            few_shot_num: int = 3,
            few_shot_save_dir: str = None,
            few_show_range: Union[int, str, List[str], List[int]] = None,
            need_external: bool = False,
            external_range: Union[int, str, List[str], List[int]] = None,
            external_save_dir: str = None,
            db_path: Union[str, List[str], Dict] = None,
            skip_schema_init: bool = False,
            schema_source: Union[None, str, List[str], Dict] = None,
            multi_database: Union[bool, List[bool], Dict] = False,
            vector_store: Union[None, str, List[str], Dict] = "vector_store",
            schema_source_dir: str = None,
            need_build_index: bool = False,
            index_range: Union[bool, List[str]] = False,
            reduce_type: str = None,
            reduce_output_format: str = "dataframe",
            is_save_reduce: bool = True,
            reduce_save_dir: str = None,
            parse_type: Union[None, str] = None,
            is_save_parse: bool = True,
            parse_save_dir: Union[None, str] = None,
            parse_output_format: str = None,
            generate_type: Union[None, str] = None,
            is_save_generate: bool = True,
            generate_save_dir: bool = True,
            task_meta: Union[Dict, List[Dict]] = None,
            cpx_task_meta: Union[Dict, List[Dict]] = None,
            is_save_dataset: bool = True,
            open_parallel: bool = True,
            max_workers: int = 5,
            exec_process: Union[None, List[str]] = None,
            credential: dict = None
    ):
        """ API KEY """
        self._api_key = {} if api_key is None else api_key

        """ LLM """
        self._use = use
        self._model_name = model_name
        self._context_window = context_window
        self._max_token = max_token
        self._top_p = top_p
        self._temperature = temperature
        self._time_out = time_out

        """ TEXT EMBED """
        self._embed_model_source = embed_model_source
        self._embed_model_name = embed_model_name

        """ ROUTER """
        self._use_demo = use_demo

        """ DATALOADER """
        self._is_prepare_data = is_prepare_data

        """ DATASET """
        self._data_source = data_source
        self._data_source_dir = data_source_dir
        self._overwrite_exist_file = overwrite_exist_file

        self._need_few_shot = need_few_shot
        self._few_shot_num = few_shot_num
        self._few_shot_save_dir = few_shot_save_dir
        self._few_show_range = [] if not few_show_range else few_show_range

        self._need_external = need_external
        self._external_range = [] if not external_range else external_range
        self._external_save_dir = external_save_dir

        self._db_path = db_path

        """ DATABASE """
        self._skip_schema_init = skip_schema_init
        self._schema_source = schema_source
        self._multi_database = multi_database
        self._vector_store = vector_store
        self._schema_source_dir = schema_source_dir

        self._need_build_index = need_build_index
        self._index_range = index_range

        """ REDUCER """
        self._reduce_type = reduce_type
        self._is_save_reduce = is_save_reduce
        self._reduce_save_dir = reduce_save_dir
        self._reduce_output_format = reduce_output_format

        """ PARSER """
        self._parse_type = parse_type
        self._is_save_parse = is_save_parse
        self._parse_save_dir = parse_save_dir
        self._parse_output_format = parse_output_format

        """ GENERATOR """
        self._generate_type = generate_type
        self._is_save_generate = is_save_generate
        self._generate_save_dir = generate_save_dir

        """ TASK """
        self._task_meta = {} if not task_meta else task_meta
        self._cpx_task_meta = {} if not cpx_task_meta else cpx_task_meta
        self._is_save_dataset = is_save_dataset
        self._open_parallel: bool = open_parallel
        self._max_workers: int = max_workers

        """ ENGINE """
        self._exec_process = exec_process

        """ CREDENTIAL """
        self._credential = {} if credential is None else credential

        self._init_system_config()
        if config_path:
            config_path_ = Path(config_path)
            if config_path_.exists():
                self.init_config(config_path)
        else:
            if self._use_demo:
                self.init_config(self._demo_config_path)

    @staticmethod
    def _load_config_file(file_path: str):
        """ Loading configuration files. Currently only JSON format is supported. """
        filepath = Path(file_path)
        if filepath.suffix != ".json":
            raise Exception("The suffix of Config file must be .json.")
        with open(filepath, "r", encoding="utf-8") as file:
            config_ = json.load(file)
        return config_

    def init_config(self, config: Union[Dict, str]):
        config = config if type(config) == dict else self._load_config_file(config)
        self._config = config

        setup_keys = list(config.keys())
        if "api_key" in setup_keys:
            self.init_api_key_config(config.get("api_key"))
        if "llm" in setup_keys:
            self.init_llm_config(config.get("llm"))
        if "text_embed" in setup_keys:
            self.init_text_embed_config(config.get("text_embed"))
        if "router" in setup_keys:
            self.init_router_config(config.get("router"))
        if "dataloader" in setup_keys:
            self.init_dataloader_config(config.get("dataloader"))
        if "dataset" in setup_keys:
            self.init_dataset_config(config.get("dataset"))
        if "database" in setup_keys:
            self.init_database_config(config.get("database"))
        if "reducer" in setup_keys:
            self.init_reducer_config(config.get("reducer"))
        if "parser" in setup_keys:
            self.init_parser_config(config.get("parser"))
        if "generator" in setup_keys:
            self.init_generator_config(config.get("generator"))
        if "task" in setup_keys:
            self.init_task_config(config.get("task"))
        if "engine" in setup_keys:
            self.init_engine_config(config.get("engine"))
        if "credential" in setup_keys:
            self.init_credential_config(config.get("credential"))

    def _init_system_config(self):
        system_config_ = self._load_config_file(self._sys_config_path)
        self.init_config(system_config_)

        """ Dataset """
        self._default_data_file_name = system_config_.get("dataset").get("default_data_file_name")
        self._default_get_external_function = system_config_.get("dataset").get(
            "default_get_external_function")
        self._sys_few_shot_dir = system_config_.get("dataset").get("sys_few_shot_dir")

        """ Database """
        self._index_method = system_config_.get("database").get("index_method")
        self._default_schema_dir_name = system_config_.get("database").get("default_schema_dir_name")

        """ Benchmark """
        self._benchmark = system_config_.get("benchmark")

        """ Task """
        self._default_log_save_dir = system_config_.get("task").get("default_log_save_dir")

    def init_api_key_config(self, api_key_: Dict):
        self._api_key = api_key_

    def init_llm_config(self, llm_: Dict):
        self._use = llm_.get("use", self._use)
        self._model_name = llm_.get("model_name", self._model_name)
        self._context_window = llm_.get("context_window", self._context_window)
        self._max_token = llm_.get("max_token", self._max_token)
        self._top_p = llm_.get("top_p", self._top_p)
        self._temperature = llm_.get("temperature", self._temperature)
        self._time_out = llm_.get("time_out", self._time_out)

    def init_text_embed_config(self, text_embed_: Dict):
        self._embed_model_source = text_embed_.get("embed_model_source", self._embed_model_source)
        self._embed_model_name = text_embed_.get("embed_model_name", self._embed_model_name)

    def init_router_config(self, router_: Dict):
        self._use_demo = router_.get("use_demo", self._use_demo)

    def init_dataloader_config(self, dataloader_: Dict):
        self._is_prepare_data = dataloader_.get("is_prepare_data", self._is_prepare_data)

    def init_dataset_config(self, dataset_: Dict):
        self._data_source = dataset_.get("data_source", self._data_source)
        self._data_source_dir = dataset_.get("data_source_dir", self._data_source_dir)
        self._overwrite_exist_file = dataset_.get("overwrite_exist_file", self._overwrite_exist_file)

        self._need_few_shot = dataset_.get("need_few_shot", self._need_few_shot)
        self._few_shot_num = dataset_.get("few_shot_num", self._few_shot_num)
        self._few_shot_save_dir = dataset_.get("few_shot_save_dir", self._few_shot_save_dir)
        self._few_show_range = dataset_.get("few_show_range", self._few_show_range)

        self._need_external = dataset_.get("need_external", self._need_external)
        self._external_range = dataset_.get("external_range", self._external_range)
        self._external_save_dir = dataset_.get("external_save_dir", self._external_save_dir)

        self._db_path = dataset_.get("db_path", self._db_path)

    def init_database_config(self, database_: Dict):
        self._skip_schema_init = database_.get("skip_schema_init", self._skip_schema_init)
        self._schema_source = database_.get("schema_source", self._schema_source)
        self._multi_database = database_.get("multi_database", self._multi_database)
        self._vector_store = database_.get("vector_store", self._vector_store)
        self._schema_source_dir = database_.get("schema_source_dir", self._schema_source_dir)

        self._need_build_index = database_.get("need_build_index", self._need_build_index)
        self._index_range = database_.get("schema_source", self._schema_source)

    def init_reducer_config(self, reducer_: Dict):
        self._reduce_type = reducer_.get("reduce_type", self._reduce_type)
        self._is_save_reduce = reducer_.get("is_save_reduce", self._is_save_reduce)
        self._reduce_save_dir = reducer_.get("reduce_save_dir", self._reduce_save_dir)
        self._reduce_output_format = reducer_.get("reduce_output_format", self._reduce_output_format)

    def init_parser_config(self, parser_: Dict):
        self._parse_type = parser_.get("parse_type", self._parse_type)
        self._is_save_parse = parser_.get("is_save_parse", self._is_save_parse)
        self._parse_save_dir = parser_.get("parse_save_dir", self._parse_save_dir)
        self._parse_output_format = parser_.get("parse_output_format", self._parse_output_format)

    def init_generator_config(self, generator: Dict):
        self._generate_type = generator.get("generate_type", self._generate_type)
        self._is_save_generate = generator.get("is_save_generate", self._is_save_generate)
        self._generate_save_dir = generator.get("generate_save_dir", self._generate_save_dir)

    def init_task_config(self, task_: Dict):
        self._task_meta = task_.get("task_meta", self._task_meta)
        self._cpx_task_meta = task_.get("cpx_task_meta", self._cpx_task_meta)
        self._is_save_dataset = task_.get("is_save_dataset", self._is_save_dataset)
        self._open_parallel = task_.get("open_parallel", self._open_parallel)
        self._max_workers = task_.get("max_workers", self._max_workers)

    def init_engine_config(self, engine_: Dict):
        self._exec_process = engine_.get("exec_process", self._exec_process)

    def init_credential_config(self, credential_: Dict):
        self._credential = credential_

    def get_config_param(self, category: str, item: str = None):
        category = self._config.get(category, None)
        if not category or not item:
            return category
        item = category.get(item, None)
        return item

    def get_credential(self, db_type: str):
        if db_type not in self._credential.keys():
            raise Exception(f"The credential of `{db_type}` does not exist!")
        credential_ = self._credential.get(db_type)
        return credential_

    def get_api_key(self, use: str):
        if use not in self._api_key.keys():
            raise Exception("The API KEY does not exist!")
        key_ = self._api_key.get(use)
        return key_

    def get_benchmark_db_path(self, id_: str, sub_id_: str):
        meta_data = next((x for x in self._benchmark if x.get('id') == id_), None)
        if meta_data is None:
            warnings.warn("Invalid benchmark dataset 'id'.", category=UserWarning)
            return None

        root_path = Path(meta_data.get("root_path", ""))

        if sub_id_:
            if not meta_data.get('has_sub', False):
                warnings.warn("Sub-dataset not supported for this benchmark dataset.", category=UserWarning)
                return None

            sub_meta_data = next((x for x in meta_data.get('sub_data', []) if x.get('sub_id') == sub_id_), None)
            if sub_meta_data is None:
                warnings.warn("Invalid 'sub_id' for benchmark dataset.", category=UserWarning)
                return None

            if sub_meta_data.get("use_local_database", False):
                return str(root_path / sub_id_ / "database")

        return str(root_path / "database")

    @property
    def api_key(self):
        if self._use not in self._api_key.keys():
            raise Exception("The API KEY does not exist!")
        key_ = self._api_key.get(self._use)
        return key_

    """ LLM """

    @property
    def use(self):
        return self._use

    @property
    def model_name(self):
        return self._model_name

    @property
    def context_window(self):
        return self._context_window

    @property
    def max_token(self):
        return self._max_token

    @property
    def top_p(self):
        return self._top_p

    @property
    def temperature(self):
        return self._temperature

    @property
    def time_out(self):
        return self._time_out

    """ Text Embed """

    @property
    def embed_model_source(self):
        return self._embed_model_source

    @property
    def embed_model_name(self):
        return self._embed_model_name

    @property
    def use_demo(self):
        return self._use_demo

    """ Dataset """

    @property
    def data_source(self):
        return self._data_source

    @property
    def data_source_dir(self):
        return self._data_source_dir

    @property
    def default_data_file_name(self):
        return self._default_data_file_name

    @property
    def overwrite_exist_file(self):
        return self._overwrite_exist_file

    @property
    def need_few_shot(self):
        return self._need_few_shot

    @property
    def few_shot_num(self):
        return self._few_shot_num

    @property
    def sys_few_shot_dir(self):
        return self._sys_few_shot_dir

    @property
    def few_shot_save_dir(self):
        return self._few_shot_save_dir

    @property
    def few_shot_range(self):
        return self._few_show_range

    @property
    def need_external(self):
        return self._need_external

    @property
    def default_get_external_function(self):
        return self._default_get_external_function

    @property
    def external_range(self):
        return self._external_range

    @property
    def external_save_dir(self):
        return self._external_save_dir

    @property
    def db_path(self):
        return self._db_path

    """ Database """

    @property
    def skip_schema_init(self):
        return self._skip_schema_init

    @property
    def schema_source(self):
        return self._schema_source

    @property
    def multi_database(self):
        return self._multi_database

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def schema_source_dir(self):
        return self._schema_source_dir

    @property
    def default_schema_dir_name(self):
        return self._default_schema_dir_name

    @property
    def need_build_index(self):
        return self._need_build_index

    @property
    def index_method(self):
        return self._index_method

    @property
    def index_range(self):
        return self._index_range

    """ Dataloader """

    @property
    def is_prepare_data(self):
        return self._is_prepare_data

    """ Reducer """

    @property
    def reduce_type(self):
        return self._reduce_type

    @property
    def is_save_reduce(self):
        return self._is_save_reduce

    @property
    def reduce_save_dir(self):
        return self._reduce_save_dir

    @property
    def reduce_output_format(self):
        return self._reduce_output_format

    """ Parser """

    @property
    def parse_type(self):
        return self._parse_type

    @property
    def is_save_parse(self):
        return self._is_save_parse

    @property
    def parse_save_dir(self):
        return self._parse_save_dir

    @property
    def parse_output_format(self):
        return self._parse_output_format

    """ Generator """

    @property
    def generate_type(self):
        return self._generate_type

    @property
    def is_save_generate(self):
        return self._is_save_generate

    @property
    def generate_save_dir(self):
        return self._generate_save_dir

    """ Task """

    @property
    def task_meta(self):
        return self._task_meta

    @property
    def cpx_task_meta(self):
        return self._cpx_task_meta

    @property
    def default_log_save_dir(self):
        return self._default_log_save_dir

    @property
    def is_save_dataset(self):
        return self._is_save_dataset

    @property
    def open_parallel(self):
        return self._open_parallel

    @property
    def max_workers(self):
        return self._max_workers

    """ Engine """

    @property
    def exec_process(self):
        return self._exec_process

    """ Benchmark """

    @property
    def benchmark(self):
        return self._benchmark

    @property
    def credential(self):
        return self._credential
