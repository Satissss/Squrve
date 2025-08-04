from os import PathLike
from typing import Union, Dict, List

from core.actor.base import Actor
from abc import abstractmethod


class BaseOptimizer(Actor):
    OUTPUT_NAME: str = "pred_sql"

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, PathLike, List[str], List[PathLike]] = None,
            **kwargs
    ):
        pass
