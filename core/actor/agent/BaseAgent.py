from typing import Union, List, Type
from os import PathLike
from core.actor.base import Actor
from abc import abstractmethod

from core.data_manage import Dataset


class BaseAgent(Actor):
    def __init__(
            self,
            dataset: Dataset = None,
            llm=None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "...",
            **kwargs
    ):
        """Initialize base decomposer with common parameters."""
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.kwargs = kwargs

    @abstractmethod
    def act(self, item, data_logger=None, **kwargs):
        pass
