from abc import abstractmethod
from typing import Union, Dict, List
from core.actor.base import Actor, MergeStrategy, ActorPool

@ActorPool.register_actor
class BaseReducer(Actor):
    OUTPUT_NAME = "schema"
    STRATEGY = MergeStrategy.OVERWRITE.value

    @abstractmethod
    def act(self, item, schema: Union[Dict, List] = None, data_logger=None, **kwargs):
        pass
