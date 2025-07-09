from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.actor.base import Actor
from core.task.meta.MetaTask import MetaTask


class ComplexTask(MetaTask):
    # todo 后续对 Complex 补充更多日志记录等方法
    NAME = "ComplexTask"

    def __init__(
            self,
            llm: Union[LLM, List[LLM]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm: Union[LLM, List[LLM]] = llm

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[Actor]:
        # ComplexTask relies entirely on the actor object being provided externally.
        if hasattr(self, "actor"):
            return self.actor

        return None
