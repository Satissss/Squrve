import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.reducer.LinkAlignReduce import LinkAlignReducer
from core.actor.reducer.BaseReduce import BaseReducer


class ReduceTask(MetaTask):
    """ Task For Text-to-SQL """

    NAME = "ReduceTask"
    registered_reduce_type = ["LinkAlignReducer", "LinkAlign"]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            reduce_type: str = "LinkAlignReducer",
            output_format: str = "str",  # output in `list` or `str`
            save_dir: Union[str, PathLike] = "../files/instance_schemas",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.reduce_type: str = reduce_type
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseReducer]:

        if actor_type is None:
            actor_type = self.reduce_type

        output_format = self.output_format
        if "output_format" in kwargs:
            output_format = kwargs.get("output_format")

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        reduce_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "output_format": output_format,
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            reduce_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseReducer):
                for key, val in reduce_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("LinkAlignReducer", "LinkAlign"):
            actor = LinkAlignReducer(**reduce_args)
            return actor

        warnings.warn(f"The reduce_type `{actor_type}` is not available.", category=UserWarning)
        return None
