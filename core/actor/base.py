import copy
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from core.data_manage import Dataset
from enum import Enum
from loguru import logger


class Actor(ABC):
    NAME: str
    OUTPUT_NAME: str
    STRATEGY: str
    dataset: Dataset

    @abstractmethod
    def act(self, item, data_logger=None, **kwargs):
        pass

    @property
    def name(self):
        if hasattr(self, "NAME"):
            return self.NAME
        return None

    @property
    def output_name(self):
        if hasattr(self, "OUTPUT_NAME"):
            return self.OUTPUT_NAME
        return None

    @property
    def strategy(self):
        if hasattr(self, "STRATEGY"):
            return self.STRATEGY
        return None

    def copy_instance(self):
        new_obj = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k == 'llm':
                setattr(new_obj, k, v)  # 直接引用 llm
            else:
                try:
                    setattr(new_obj, k, copy.deepcopy(v))
                except Exception:
                    setattr(new_obj, k, v)  # deepcopy 失败则直接引用
        return new_obj


class ComplexActor(Actor):
    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            actors: Optional[List[Actor]] = None,
            **kwargs
    ):
        self.dataset: Optional[Dataset] = dataset
        self.actors: List[Actor] = [] if actors is None else actors

        self.__init_check__()

    def __init_check__(self):
        if not self.actors:
            return
        actors = [actor for actor in self.actors if actor and actor.dataset is not None]
        self.actors = actors
        if not actors:
            return

        datasets = {actor.dataset for actor in actors}
        if len(datasets) > 1:
            raise ValueError(f"Inconsistent datasets found: {datasets}")

        if not self.dataset:
            self.dataset = list(datasets)[0]

    def add(self, actors: Union[Actor, List[Actor]]):
        if isinstance(actors, Actor):
            actors = [actors]

        actors = [actor for actor in actors if actor and actor.dataset is not None]
        if not actors:
            return
        datasets = {actor.dataset for actor in actors}
        if len(datasets) > 1:
            raise ValueError(f"Inconsistent datasets found: {datasets}")

        if not self.dataset:
            self.dataset = list(datasets)[0]
        elif self.dataset != list(datasets)[0]:
            raise ValueError(f"Inconsistent datasets found: {datasets}")

        self.actors.extend(actors)

    @abstractmethod
    def act(self, item, data_logger=None, **kwargs):
        pass

    @property
    def is_empty(self):
        return len(self.actors) == 0


class MergeStrategy(Enum):
    OVERWRITE = "overwrite"
    APPEND = "append"
    EXTEND = "extend"


class MergeFunction:
    @classmethod
    def get_method(cls, strategy):
        if strategy == MergeStrategy.OVERWRITE.value:
            return MergeFunction.overwrite
        elif strategy == MergeStrategy.APPEND.value:
            return MergeFunction.append
        elif strategy == MergeStrategy.EXTEND.value:
            return MergeFunction.extend
        else:
            raise ValueError(f"Unknown merge strategy {strategy}")

    @staticmethod
    def overwrite(results, key, val):
        if not isinstance(results, dict):
            raise TypeError("results must be a dict!")
        results[key] = val

    @staticmethod
    def append(results, key, val):
        if not isinstance(results, dict):
            raise TypeError("results must be a dict!")

        if key in results:
            if isinstance(results[key], list):
                results[key].append(val)
            else:
                results[key] = [results[key], val]
        else:
            results[key] = val
        return results

    @staticmethod
    def extend(results, key, val):
        if not isinstance(results, dict):
            raise TypeError("results must be a dict!")
        if not isinstance(val, list):
            logger.info("val must be a list!")
            return MergeFunction.append(results, key, val)

        if key in results:
            if isinstance(results[key], list):
                results[key].extend(val)
            else:
                results[key] = [str(results[key]), *val]
        else:
            results[key] = val
        return results


class ActorPool:
    _registered_actor_lis: List = []

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
