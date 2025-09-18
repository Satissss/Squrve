import copy
import warnings
from os import PathLike
from typing import Union, List, Optional

from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.scaler.ChessScale import ChessScaler
from core.actor.scaler.OpenSearchSQLScale import OpenSearchSQLScaler
from core.actor.scaler.DINSQLScale import DINSQLScaler
from core.actor.scaler.MACSQLScale import MACSQLScaler
from core.actor.scaler.RSLSQLScale import RSLSQLScaler
from core.actor.scaler.BaseScale import BaseScaler


class ScaleTask(MetaTask):
    """ Task For SQL Scaling """

    NAME = "ScaleTask"
    registered_scale_type = [
        "ChessScaler", "CHESS",
        "OpenSearchSQLScaler", "OpenSearchSQL",
        "DINSQLScaler", "DINSQL",
        "MACSQLScaler", "MACSQL", 
        "RSLSQLScaler", "RSLSQL"
    ]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            scale_type: str = "ChessScaler",
            output_format: str = "list",  # output in `list` or `str`
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.scale_type: str = scale_type
        self.output_format: str = output_format
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseScaler]:

        if actor_type is None:
            actor_type = self.scale_type

        output_format = self.output_format
        if "output_format" in kwargs:
            output_format = kwargs.get("output_format")

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        scale_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "output_format": output_format,
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            scale_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseScaler):
                for key, val in scale_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("ChessScaler", "CHESS"):
            actor = ChessScaler(**scale_args)
            return actor

        elif actor_type in ("OpenSearchSQLScaler", "OpenSearchSQL"):
            actor = OpenSearchSQLScaler(**scale_args)
            return actor

        elif actor_type in ("DINSQLScaler", "DINSQL"):
            actor = DINSQLScaler(**scale_args)
            return actor

        elif actor_type in ("MACSQLScaler", "MACSQL"):
            actor = MACSQLScaler(**scale_args)
            return actor

        elif actor_type in ("RSLSQLScaler", "RSLSQL"):
            actor = RSLSQLScaler(**scale_args)
            return actor

        warnings.warn(f"The scale_type `{actor_type}` is not available.", category=UserWarning)
        return None 