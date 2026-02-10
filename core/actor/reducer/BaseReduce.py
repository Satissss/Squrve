from abc import abstractmethod
from typing import Union, Dict, List
from core.actor.base import Actor, MergeStrategy, ActorPool

@ActorPool.register_actor
class BaseReducer(Actor):
    OUTPUT_NAME = "schema"
    STRATEGY = MergeStrategy.OVERWRITE.value

    _registered_actor_lis: List[Actor] = []

    @abstractmethod
    def act(self, item, schema: Union[Dict, List] = None, data_logger=None, **kwargs):
        pass

    @classmethod
    def syntax_check(cls, actor_str: str) -> bool:
        if not isinstance(actor_str, str):
            return False

        return actor_str.lower().endswith("reducer")

    @classmethod
    def register_actor(cls, actor_cls: Actor):
        if not issubclass(actor_cls, Actor):
            raise TypeError(f"{actor_cls.__name__} is not a subclass of Actor")

        if actor_cls not in cls._registered_actor_lis:
            cls._registered_actor_lis.append(actor_cls)
        return actor_cls

    @classmethod
    def get_all_actors(cls):
        return cls._registered_actor_lis