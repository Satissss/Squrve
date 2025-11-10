from abc import abstractmethod
from typing import Union, Dict, List
from core.actor.base import Actor


class BaseReducer(Actor):
    OUTPUT_NAME: str = "schema"

    @abstractmethod
    def act(self, item, schema: Union[Dict, List] = None, data_logger=None, **kwargs):
        pass
