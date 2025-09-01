from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict
import pandas as pd
from os import PathLike
from pathlib import Path

from core.data_manage import Dataset, single_central_process
from core.actor.parser.BaseParse import BaseParser
from core.LinkAlign.SchemaLinkingTool import SchemaLinkingTool
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset,
    parse_schema_link_from_str
)


class LinkAlignParser(BaseParser):
    """
    Extract the required schema information for a single sample using Schema Linking provided by LinkAlign
    """

    NAME = "LinkAlignParser"

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "str",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            use_external: bool = False,
            generate_num: int = 1,
            use_llm_scaling: bool = False,
            automatic: bool = True,
            parse_mode: str = "agent",  # `agent` or `pipeline`
            parse_turn_n: int = 1,
            parse_link_num: int = 3,
            open_parallel: bool = False,
            max_workers: int = None,
            **kwargs
    ):
        self.dataset: Dataset = dataset
        self.llm: Union[LLM, List[LLM]] = llm
        self.output_format: str = output_format
        self.is_save: bool = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.use_external: bool = use_external
        self.generate_num: int = generate_num
        self.use_llm_scaling: bool = use_llm_scaling
        self.automatic: bool = automatic

        self.parse_mode: str = parse_mode
        self.parse_turn_n: int = parse_turn_n
        self.parse_link_num: int = parse_link_num

        self.open_parallel: bool = open_parallel
        self.max_workers: int = max_workers

    @classmethod
    def load_turn_n(cls, db_size: int):
        if db_size < 250:
            return 1, 2
        elif db_size < 750:
            return 2, 3
        else:
            return 2, 5

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, **kwargs):
        row = self.dataset[item]
        question = row["question"]
        db_size = row["db_size"]

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                question += "\n" + external_knowledge

        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas", None)
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        # 转换 schema 为自然语言形式
        schema_context = parse_schema_from_df(schema)

        turn_n, link_num = (self.load_turn_n(db_size) if self.automatic
                            else (self.parse_turn_n, self.parse_link_num))

        if isinstance(self.llm, list) and self.llm:
            llm = self.llm[0]
        else:
            llm = self.llm

        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        # 仅使用第一个 LLM 生成 schema links
        schema_links = []
        select_args = {
            "mode": self.parse_mode,
            "query": question,
            "context": schema_context,
            "turn_n": turn_n,
            "linker_num": link_num,
            "llm": llm,
        }
        for _ in range(self.generate_num):
            result = SchemaLinkingTool.generate_selector(**select_args)
            schema_links.extend(parse_schema_link_from_str(result))

        schema_links = list(dict.fromkeys(schema_links))

        if self.is_save:
            instance_id = row.get("instance_id", item)
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            if self.output_format == "str":
                save_path = save_path / f"{self.NAME}_{instance_id}.txt"
            else:
                save_path = save_path / f"{self.NAME}_{instance_id}.json"
            save_dataset(schema_links, new_data_source=save_path)
            self.dataset.setitem(item, "schema_links", str(save_path))

        return str(schema_links) if self.output_format == "str" else schema_links
