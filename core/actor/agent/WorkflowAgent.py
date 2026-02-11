"""Workflow orchestration agent for composing and executing actor pipelines.

This module provides WorkflowAgent, which builds a pipeline of actors from a
declarative configuration and runs them on dataset items.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from anyio import sleep_forever
from llama_index.core.llms import LLM

from core.actor.agent.BaseAgent import BaseAgent
from core.actor.base import ActorPool
from core.data_manage import Dataset
from core.actor.nest.tree import TreeActor
from core.actor.nest.pipeline import PipelineActor
from loguru import logger


class WorkflowAgent(BaseAgent):
    """Orchestrates a pipeline of actors based on a declarative configuration.

    WorkflowAgent composes registered actors (parsers, generators, etc.) into
    a PipelineActor, optionally grouping some actors into TreeActors for
    parallel execution. The workflow is defined by `actor_lis`, where each
    element is either a single actor name (serial step) or a list of actor
    names (parallel step).

    Structure of `actor_lis`:
        - str: Single actor, executed as one step in the pipeline.
        - List[str]: Multiple actors, executed in parallel via TreeActor,
          then merged into a single output for the next step.

    Example:
        >>> agent = WorkflowAgent(
        ...     dataset=dataset,
        ...     llm=llm,
        ...     actor_lis=[
        ...         "LinkAlignParser",           # Step 1: parse
        ...         ["DINSQLGenerator", "CHESSGenerator"],  # Step 2: parallel generation
        ...         "RSLSQLOptimizer",          # Step 3: optimize
        ...     ],
        ...     actor_args={
        ...         "CHESSGenerator": {"use_schema_selector": True},
        ...     },
        ... )
        >>> result = agent.act(item_index)

    Attributes:
        actor_lis: List of pipeline steps; each step is str or List[str].
        actor_args: Optional per-actor constructor kwargs, keyed by actor NAME.
          `dataset` and `llm` are always overridden from this agent.
    """

    NAME: str = "WorkflowAgent"
    OUTPUT_NAME: str = ""  # Dynamically determine

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            actor_lis: Optional[List[Union[str, List[str]]]] = None,
            actor_args: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """Initialize WorkflowAgent.

        Args:
            dataset: Dataset to process.
            llm: Language model for actors that need it.
            is_save: Whether to save outputs.
            save_dir: Directory for saved outputs.
            actor_lis: Pipeline config; each item is str (single actor) or
                List[str] (parallel actors).
            actor_args: Per-actor kwargs, keyed by actor NAME. Will be
                merged with dataset and llm from this agent.
            **kwargs: Passed to BaseAgent.
        """
        super().__init__(dataset, llm, **kwargs)
        self.actor_lis = actor_lis
        self.actor_args = actor_args or {}

    def __init_actors__(self) -> PipelineActor:
        """Build PipelineActor from actor_lis configuration.

        Returns:
            Configured PipelineActor ready to run.

        Raises:
            ValueError: If actor_lis is empty or actor_args item is not dict.
            TypeError: If actor_lis item is not str or list.
        """
        actor_lis = self.actor_lis or []
        if not isinstance(actor_lis, list) or len(actor_lis) == 0:
            raise ValueError("The actor list must be a list of actors")

        pipe_actor = PipelineActor(dataset=self.dataset)
        actors = []
        for item in actor_lis:
            if isinstance(item, str):
                raw_args = self.actor_args.get(item, {})
                if not isinstance(raw_args, dict):
                    raise ValueError(f"actor_args for '{item}' must be a dict, got {type(raw_args).__name__}")
                args = dict(raw_args)
                args.update({"dataset": self.dataset, "llm": self.llm})
                actor = ActorPool.get_actor_by_name(item)(**args)
                actors.append(actor)
            elif isinstance(item, list):
                tree_actor = TreeActor(dataset=self.dataset)
                inner_actors = []
                for row in item:
                    raw_args = self.actor_args.get(row, {})
                    if not isinstance(raw_args, dict):
                        raise ValueError(f"actor_args for '{row}' must be a dict, got {type(raw_args).__name__}")
                    args = dict(raw_args)
                    args.update({"dataset": self.dataset, "llm": self.llm})
                    actor = ActorPool.get_actor_by_name(row)(**args)
                    inner_actors.append(actor)
                tree_actor.actors = inner_actors
                actors.append(tree_actor)
            else:
                raise TypeError(f"actor_lis item must be str or list, got {type(item).__name__}: {item}")

        pipe_actor.actors = actors

        return pipe_actor

    def act(self, item, **kwargs):
        """Execute the workflow on a single item.

        Runs the pipeline built from actor_lis. Each step receives the
        output of the previous step (or initial kwargs for the first step).

        Args:
            item: Dataset index to process.
            **kwargs: Passed to the first actor and through the pipeline.

        Returns:
            Result of the last actor in the pipeline.

        Raises:
            Exception: Re-raised on failure (after logging).
        """
        try:
            pipe_actor = self.__init_actors__()
            res = pipe_actor.act(item, **kwargs)
            self.OUTPUT_NAME = pipe_actor.output_name

            return res
        except Exception as e:
            logger.exception("WorkflowAgent failed to initialize or execute actors: %s", e)
            self.OUTPUT_NAME = "None"

            return None

