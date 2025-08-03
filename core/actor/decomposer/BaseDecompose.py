from os import PathLike
from typing import Union, Dict, List

from core.actor.base import Actor
from abc import abstractmethod


class BaseDecomposer(Actor):
    """ Decompose complex queries into a series of sub-questions. """

    OUTPUT_NAME: str = "sub_questions"

    @abstractmethod
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            **kwargs
    ):
        pass
