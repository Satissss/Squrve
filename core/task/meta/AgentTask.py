import warnings
from os import PathLike
from typing import Union, List, Optional
from llama_index.core.llms.llm import LLM

from core.task.meta.MetaTask import MetaTask
from core.actor.agent.BaseAgent import BaseAgent

class AgentTask(MetaTask):

    NAME = "AgentTask"

    def __init__(
            self,
            llm: Union[LLM, List[LLM]],
            agent_type: str = "WorkflowAgent",
            **kwargs
    ):
        self.llm: Union[LLM, List[LLM]] = llm
        self.agent_type: str = agent_type

        super().__init__(**kwargs)

    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[BaseAgent]:
        if actor_type is None:
            actor_type = self.agent_type

        agent_args = {
            "dataset": self.dataset,
            "llm": self.llm,
        }
        for key, val in kwargs.items():
            agent_args[key] = val

        if hasattr(self, "actor"):
            actor = self.actor.copy_instance()
            if actor and isinstance(actor, BaseAgent):
                for key, val in agent_args.items():
                    setattr(actor, key, val)
                return actor

        if actor_type in ("WorkflowAgent", "Workflow"):
            from core.actor.agent.WorkflowAgent import WorkflowAgent
            actor = WorkflowAgent(**agent_args)
            return actor
        elif actor_type in ("MultiWorkflowAgent", "MultiWorkflow"):
            from core.actor.agent.WorkflowAgent import MultiWorkflowAgent
            actor = MultiWorkflowAgent(**agent_args)
            return actor

        warnings.warn(f"The decompose_type `{actor_type}` is not available.", category=UserWarning)
        return None
