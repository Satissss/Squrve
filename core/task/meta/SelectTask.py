import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.actor.selector.FastExecSelector import FastExecSelector
from core.task.meta.MetaTask import MetaTask
from core.actor.selector.BaseSelector import BaseSelector
from core.actor.selector.CHESSSelect import CHESSSelector
from core.actor.selector.OpenSearchSQLSelect import OpenSearchSQLSelector


class SelectTask(MetaTask):
    """ Task For Selecting Optimal SQL """

    NAME = "SelectTask"
    registered_select_type = [
        "CHESSSelector", "CHESS",
        "OpenSearchSQLSelector", "OpenSearchSQL",
        "FastExecSelector", "FastExec"
    ]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            select_type: str = "CHESSSelector",
            output_format: str = "str",  # output in `list` or `str`
            save_dir: Union[str, PathLike] = "../files/selected_sql",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.select_type: str = select_type
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseSelector]:

        if actor_type is None:
            actor_type = self.select_type

        output_format = self.output_format
        if "output_format" in kwargs:
            output_format = kwargs.get("output_format")

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        select_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "output_format": output_format,
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            select_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseSelector):
                for key, val in select_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("CHESSSelector", "CHESS"):
            actor = CHESSSelector(**select_args)
            return actor

        elif actor_type in ("OpenSearchSQLSelector", "OpenSearchSQL"):
            actor = OpenSearchSQLSelector(**select_args)
            return actor

        elif actor_type in ("FastExecSelector", "FastExec"):
            actor = FastExecSelector(**select_args)
            return actor

        warnings.warn(f"The select_type `{actor_type}` is not available.", category=UserWarning)
        return None
