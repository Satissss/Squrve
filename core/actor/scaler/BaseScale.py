from os import PathLike
from typing import Union, Dict, List

from core.actor.base import Actor
from abc import abstractmethod


class BaseScaler(Actor):
    """ Use different prompt strategies to generate multiple SQL candidates matching the query. """
    OUTPUT_NAME: str = "pred_sql"

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            **kwargs
    ):
        pass
