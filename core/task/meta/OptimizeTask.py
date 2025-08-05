import copy
import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.optimizer.LinkAlignOptimize import LinkAlignOptimizer
from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.actor.optimizer.DINSQLOptimize import DINSQLOptimizer
from core.actor.optimizer.CHESSOptimize import CHESSOptimizer
from core.actor.optimizer.MACSQLOptimize import MACSQLOptimizer
from core.actor.optimizer.RSLSQLOptimize import RSLSQLOptimizer
from core.actor.optimizer.OpenSearchSQLOptimize import OpenSearchSQLOptimizer


class OptimizeTask(MetaTask):
    """ Task For SQL Optimization """

    NAME = "OptimizeTask"
    registered_optimize_type = [
        "LinkAlignOptimizer", "LinkAlign",
        "DINSQLOptimizer", "DIN_SQL",
        "CHESSOptimizer", "CHESS",
        "MACSQLOptimizer", "MACSQL",
        "RSLSQLOptimizer", "RSLSQL",
        "OpenSearchSQLOptimizer", "OpenSearchSQL"
    ]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            optimize_type: str = "LinkAlignOptimizer",
            save_dir: Union[str, PathLike] = "../files/optimized_sql",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.optimize_type: str = optimize_type
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseOptimizer]:

        if actor_type is None:
            actor_type = self.optimize_type

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        optimize_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            optimize_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseOptimizer):
                for key, val in optimize_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("LinkAlignOptimizer", "LinkAlign"):
            actor = LinkAlignOptimizer(**optimize_args)
            return actor

        elif actor_type in ("DINSQLOptimizer", "DIN_SQL"):
            actor = DINSQLOptimizer(**optimize_args)
            return actor

        elif actor_type in ("CHESSOptimizer", "CHESS"):
            actor = CHESSOptimizer(**optimize_args)
            return actor

        elif actor_type in ("MACSQLOptimizer", "MACSQL"):
            actor = MACSQLOptimizer(**optimize_args)
            return actor

        elif actor_type in ("RSLSQLOptimizer", "RSLSQL"):
            actor = RSLSQLOptimizer(**optimize_args)
            return actor

        elif actor_type in ("OpenSearchSQLOptimizer", "OpenSearchSQL"):
            actor = OpenSearchSQLOptimizer(**optimize_args)
            return actor

        warnings.warn(f"The optimize_type `{actor_type}` is not available.", category=UserWarning)
        return None 