import copy
import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.parser.LinkAlignParse import LinkAlignParser
from core.actor.parser.BaseParse import BaseParser


class ParseTask(MetaTask):
    """ Task For Text-to-SQL """

    NAME = "ParseTask"
    registered_parse_type = ["LinkAlignParser", "LinkAlign"]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            parse_type: str = "LinkAlignParser",
            output_format: str = "str",  # output in `list` or `str`
            save_dir: Union[str, PathLike] = "../files/schema_links",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.parse_type: str = parse_type
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseParser]:

        if actor_type is None:
            actor_type = self.parse_type

        output_format = self.output_format
        if "output_format" in kwargs:
            output_format = kwargs.get("output_format")

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        parse_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "output_format": output_format,
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            parse_args[key] = val

        if hasattr(self, "actor"):
            # actor = self.actor
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseParser):
                for key, val in parse_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("LinkAlignParser", "LinkAlign"):
            actor = LinkAlignParser(**parse_args)
            return actor

        warnings.warn(f"The parse_type `{actor_type}` is not available.", category=UserWarning)
        return None
