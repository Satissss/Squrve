import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.generator.LinkAlignGenerate import LinkAlignGenerator
from core.actor.generator.DINSQLGenerate import DIN_SQLGenerator
from core.actor.generator.DAILSQLGenerate import DAILSQLGenerate
from core.actor.generator.BaseGenerate import BaseGenerator


class GenerateTask(MetaTask):
    """ Task For Text-to-SQL """

    NAME = "GenerateTask"
    registered_generate_type = ["LinkAlignGenerator", "LinkAlign", "DIN_SQLGenerator", "DIN_SQL", "DAILSQLGenerator", "DAILSQL"]

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            generate_type: str = "LinkAlignGenerator",
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.generate_type: str = generate_type
        self.save_dir: Union[str, PathLike] = save_dir

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseGenerator]:
        if actor_type is None:
            actor_type = self.generate_type

        is_save = self.is_save
        if "is_save" in kwargs:
            is_save = kwargs.get("is_save")

        save_dir = self.save_dir
        if "save_dir" in kwargs:
            save_dir = kwargs.get("save_dir")

        generate_args = {
            "dataset": self.dataset,
            "llm": self.llm,
            # The arguments below can be replaced by the one provided in `actor_args`.
            "is_save": is_save,
            "save_dir": save_dir,
        }
        for key, val in kwargs.items():
            generate_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseGenerator):
                for key, val in generate_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("LinkAlignGenerator", "LinkAlign"):
            actor = LinkAlignGenerator(**generate_args)
            return actor
        
        elif actor_type in ("DIN_SQLGenerator", "DIN_SQL"):
            actor = DIN_SQLGenerator(**generate_args)
            return actor
        
        elif actor_type in ("DAILSQLGenerator", "DAILSQL"):
            actor = DAILSQLGenerate(**generate_args)
            return actor

        elif actor_type in ("CHESSGenerator", "CHESS"):
            from core.actor.generator.CHESSGenerate import CHESSGenerator
            actor = CHESSGenerator(**generate_args)
            return actor

        elif actor_type in ("MACSQLGenerator", "MACSQL"):
            from core.actor.generator.MACSQLGenerate import MACSQLGenerator
            actor = MACSQLGenerator(**generate_args)
            return actor

        elif actor_type in ("RSLSQLGenerator", "RSLSQL"):
            from core.actor.generator.RSLSQLGenerate import RSLSQLGenerator
            actor = RSLSQLGenerator(**generate_args)
            return actor

        elif actor_type in ("ReFoRCEGenerator", "ReFoRCE"):
            from core.actor.generator.ReFoRCEGenerate import ReFoRCEGenerator
            actor = ReFoRCEGenerator(**generate_args)
            return actor

        elif actor_type in ("OpenSearchSQLGenerator", "OpenSearchSQL"):
            from core.actor.generator.OpenSearchSQLGenerate import OpenSearchSQLGenerator
            actor = OpenSearchSQLGenerator(**generate_args)
            return actor

        warnings.warn(f"The generate_type `{actor_type}` is not available.", category=UserWarning)
        return None