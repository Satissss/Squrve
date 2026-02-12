from typing import Union, List, Type
from os import PathLike
from core.actor.base import Actor, MergeStrategy
from abc import abstractmethod

from core.data_manage import Dataset


class BaseAgent(Actor):
    OUTPUT_NAME = "pred_sql"
    STRATEGY = MergeStrategy.APPEND.value

    def __init__(
            self,
            dataset: Dataset = None,
            llm=None,
            **kwargs
    ):
        """Initialize base decomposer with common parameters."""
        self.dataset = dataset
        self.llm = llm
        self.kwargs = kwargs

    @abstractmethod
    def act(self, item, data_logger=None, **kwargs):
        pass

    @classmethod
    def __actor_check__(cls, check_actor: Actor) -> bool:
        """ check the initialized actor is valid"""

        # check the output_name
        if check_actor.OUTPUT_NAME != cls.OUTPUT_NAME:
            return False

        return True