"""Workflow orchestration agents for composing and executing actor pipelines.

This module provides:
- WorkflowAgent: Builds a pipeline of actors from a declarative configuration,
  where each step references a registered actor by name.
- MultiWorkflowAgent: Extends WorkflowAgent by supporting named workflows and
  named actors; pipeline steps reference keys in pre-configured workflows/actors
  dicts.
"""

from typing import Optional, Union, List, Dict, Any

from llama_index.core.llms import LLM

from core.actor.agent.BaseAgent import BaseAgent
from core.actor.base import ActorPool, Actor
from core.actor.selector import FastExecSelector
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
            check = self.__actor_check__(pipe_actor)
            if not check:
                return ""
            return res
        except Exception as e:
            logger.exception("WorkflowAgent failed to initialize or execute actors: %s", e)
            return ""


class MultiWorkflowAgent(BaseAgent):
    """Orchestrates a pipeline mixing named workflows and named actors.

    MultiWorkflowAgent extends WorkflowAgent by allowing pipeline steps to
    reference either pre-defined workflows (sub-pipelines) or named actors.
    Workflows and actors are configured in `workflows` and `actors` dicts,
    while `actor_lis` defines the execution order by referencing these keys.

    Structure of `actor_lis`:
        - str: Single step; key in `workflows` (treated as sub-workflow) or
          in `actors` (treated as atomic actor).
        - List[str]: Multiple steps, executed in parallel via TreeActor.

    Config format:
        - workflows: {"W1": {"actor_lis": [...], "actor_args": {...}}}
        - actors: {"A1": {"actor_name": "RegisteredActorName", "actor_args": {...}}}

    Example:
        >>> agent = MultiWorkflowAgent(
        ...     dataset=dataset,
        ...     llm=llm,
        ...     workflows={"W1": {"actor_lis": ["Parser", "Generator"], "actor_args": {}}},
        ...     actors={"A1": {"actor_name": "LinkAlignParser", "actor_args": {}}},
        ...     actor_lis=["A1", ["W1"], "Optimizer"],
        ... )

    Attributes:
        workflows: Dict of workflow key -> {actor_lis, actor_args}.
        actors: Dict of actor key -> {actor_name, actor_args}.
        actor_lis: Pipeline config; each item is str (workflow/actor key) or
            List[str] (parallel steps).
    """

    NAME: str = "MultiWorkflowAgent"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            workflows: Optional[Dict[str, Any]] = None,
            actors: Optional[Dict[str, Any]] = None,
            actor_lis: Optional[List[Union[str, List[str]]]] = None,
            **kwargs
    ):
        """Initialize MultiWorkflowAgent.

        Args:
            dataset: Dataset to process.
            llm: Language model for actors that need it.
            workflows: Dict mapping workflow key -> {actor_lis, actor_args}.
            actors: Dict mapping actor key -> {actor_name, actor_args}.
            actor_lis: Pipeline config; each item is str (key) or List[str].
            **kwargs: Passed to BaseAgent.
        """
        super().__init__(dataset, llm, **kwargs)
        self.workflows = workflows or {}
        self.actors = actors or {}
        self.actor_lis = actor_lis or []

    def __init_actors__(self) -> PipelineActor:
        """Build PipelineActor from actor_lis, resolving workflow/actor keys.

        Returns:
            Configured PipelineActor ready to run.

        Raises:
            ValueError: If actor_lis is empty, or config is invalid.
            TypeError: If actor_lis item is not str or list.
        """
        actor_lis = self.actor_lis or []
        if not isinstance(actor_lis, list) or not actor_lis:
            raise ValueError("The actor list must be a non-empty list")

        def _init_workflow_(w_actor_lis: List, w_actor_args: Dict) -> WorkflowAgent:
            """Create a WorkflowAgent from workflow config."""
            return WorkflowAgent(
                dataset=self.dataset,
                llm=self.llm,
                actor_lis=w_actor_lis,
                actor_args=w_actor_args,
            )

        def _init_actor_(actor_name: str, actor_args: Dict) -> Any:
            """Create an actor instance by registered name."""
            actor_cls = ActorPool.get_actor_by_name(actor_name)
            return actor_cls(dataset=self.dataset, llm=self.llm, **actor_args)

        def _parse_actor_item(item: str):
            """Resolve a workflow or actor key into an actor instance."""
            is_workflow = item in self.workflows
            if not is_workflow and item not in self.actors:
                raise ValueError(f"Key '{item}' not found in workflows or actors: "
                                 f"workflows={list(self.workflows)}, actors={list(self.actors)}")

            if is_workflow:
                workflow = self.workflows.get(item, {})
                if not isinstance(workflow, dict):
                    raise ValueError(f"Workflow '{item}' must be a dict, got {type(workflow).__name__}")
                if "actor_lis" not in workflow:
                    raise ValueError(f"Workflow '{item}' must contain 'actor_lis', got keys: {list(workflow)}")
                return _init_workflow_(workflow["actor_lis"], workflow.get("actor_args", {}))
            else:
                actor = self.actors.get(item, {})
                if not isinstance(actor, dict):
                    raise ValueError(f"Actor '{item}' must be a dict, got {type(actor).__name__}")
                if "actor_name" not in actor:
                    raise ValueError(f"Actor '{item}' must contain 'actor_name', got keys: {list(actor)}")
                return _init_actor_(actor["actor_name"], actor.get("actor_args", {}))

        pipe_actor = PipelineActor(dataset=self.dataset)
        actors = []

        for item in actor_lis:
            if isinstance(item, str):
                actors.append(_parse_actor_item(item))
            elif isinstance(item, list):
                tree_actor = TreeActor(dataset=self.dataset)
                tree_actor.actors = [_parse_actor_item(row) for row in item]
                actors.append(tree_actor)
            else:
                raise TypeError(f"actor_lis item must be str or list, got {type(item).__name__}: {item}")

        pipe_actor.actors = actors
        return pipe_actor

    def act(self, item, **kwargs):
        """Execute the pipeline on a single item.

        Builds the pipeline from actor_lis (resolving workflow/actor keys),
        runs it, and updates OUTPUT_NAME from the last step.

        Args:
            item: Dataset index to process.
            **kwargs: Passed to the first actor and through the pipeline.

        Returns:
            Result of the last actor in the pipeline, or None on failure.
        """
        try:
            pipe_actor = self.__init_actors__()
            res = pipe_actor.act(item, **kwargs)
            check = self.__actor_check__(pipe_actor)
            if not check:
                return ""
            return res
        except Exception as e:
            logger.exception("MultiWorkflowAgent failed to initialize or execute actors: %s", e)
            return ""