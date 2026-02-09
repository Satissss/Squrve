import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.decomposer.BaseDecompose import BaseDecomposer
from core.actor.decomposer.MACSQLDecompose import MACSQLDecomposer
from core.actor.decomposer.DINSQLDecompose import DINSQLDecomposer
from core.actor.decomposer.RecursiveDecompose import RecursiveDecomposer


class DecomposeTask(MetaTask):
    """ Task For Query Decomposition """

    NAME = "DecomposeTask"
    registered_decompose_type = [
        "MACSQLDecomposer", "MACSQL",
        "DINSQLDecomposer", "DINSQL",
    ]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            decompose_type: str = "MACSQLDecomposer",
            output_format: str = "list",  # output in `list` or `str`
            save_dir: Union[str, PathLike] = "../files/sub_questions",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.decompose_type: str = decompose_type
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseDecomposer]:

        if actor_type is None:
            actor_type = self.decompose_type

        output_format = self.output_format
        if "output_format" in kwargs:
            output_format = kwargs.get("output_format")

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        decompose_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "output_format": output_format,
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            decompose_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseDecomposer):
                for key, val in decompose_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("MACSQLDecomposer", "MACSQL"):
            actor = MACSQLDecomposer(**decompose_args)
            return actor

        elif actor_type in ("DINSQLDecomposer", "DINSQL"):
            actor = DINSQLDecomposer(**decompose_args)
            return actor

        elif actor_type in ("RecursiveDecomposer", "Recursive"):
            actor = RecursiveDecomposer(**decompose_args)
            return actor

        warnings.warn(f"The decompose_type `{actor_type}` is not available.", category=UserWarning)
        return None
