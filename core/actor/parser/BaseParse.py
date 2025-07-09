from core.actor.base import Actor
from abc import abstractmethod
from os import PathLike
from typing import Union, Dict, List


class BaseParser(Actor):
    OUTPUT_NAME: str = "schema_links"

    @abstractmethod
    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, **kwargs):
        pass
