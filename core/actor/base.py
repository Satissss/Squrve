import copy
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from core.data_manage import Dataset


class Actor(ABC):
    NAME: str
    OUTPUT_NAME: str

    dataset: Dataset

    @abstractmethod
    def act(self, item, **kwargs):
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
    def act(self, item, **kwargs):
        pass

    @property
    def is_empty(self):
        return len(self.actors) == 0
