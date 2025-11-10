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
            automatic: bool = True,
            parse_mode: str = "agent",  # `agent` or `pipeline`
            parse_turn_n: int = 1,
            parse_link_num: int = 3,
            **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)
        self.generate_num = generate_num
        self.automatic = automatic
        self.parse_mode = parse_mode
        self.parse_turn_n = parse_turn_n
        self.parse_link_num = parse_link_num

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

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row["question"]
        db_size = row["db_size"]

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                question += "\n" + external_knowledge

        # Use base class method to process schema
        schema_df = self.process_schema(item, schema)
        schema_context = parse_schema_from_df(schema_df)

        turn_n, link_num = (self.load_turn_n(db_size) if self.automatic
                            else (self.parse_turn_n, self.parse_link_num))

        # Use base class method to get LLM
        llm = self.get_llm()
        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        # Generate schema links
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
        self.log_schema_links(data_logger, schema_links, stage="final")
        output = self.format_output(schema_links)

        # Use base class method to save output
        file_ext = ".txt" if self.output_format == "str" else ".json"
        self.save_output(output, item, file_ext=file_ext)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return output
