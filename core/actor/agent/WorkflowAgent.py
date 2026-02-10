from pathlib import Path
from typing import Optional, Union, List

from llama_index.core.llms import LLM

from core.actor.agent import BaseAgent
from core.actor.base import MergeStrategy
from core.data_manage import Dataset


class WorkflowAgent(BaseAgent):
    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "...",
            actor_lis: List[str, List[str]] = None,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.actor_lis = actor_lis

    def act(self, item, data_logger=None, **kwargs):
        """
        根据 actor_lis 参数，解析为 Complex Actor 对象；仅需要执行 Complex Actor 的 act 方法即可
        """
        pass
