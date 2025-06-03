import warnings

from core.actor.base import ComplexActor
from core.data_manage import update_dataset


class PipelineActor(ComplexActor):
    """
    PipelineActor is a subclass of ComplexActor that chains multiple individual Actors
    into a single pipeline, executing them sequentially in the order they are provided.

    It enables modular composition of actors like Reducer, Parser, and Generator into
    a unified Actor. This allows complex transformations or operations to be structured
    as a pipeline of discrete steps, each handled by a different Actor.

    The pipeline takes a single input item and passes the output of each Actor
    as part of the input to the next, enabling multi-stage data processing.
    """

    NAME: str = "PipelineActor"
    OUTPUT_NAME: str = ""  # Dynamically determine

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, item, **kwargs):
        results = kwargs
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        output_name = ""
        res = None

        for actor in self.actors:
            actor.dataset = update_dataset(dataset, actor.dataset)

            try:
                res = actor.act(item, **results)
                output_name = actor.output_name

                if output_name == "TreeOutput" and isinstance(res, dict):
                    results.update(res)
                else:
                    results[output_name] = res

                dataset = actor.dataset

            except Exception as e:
                print(f"Error occurred while executing actor '{actor.name}': {e}")

        self.OUTPUT_NAME = output_name
        self.dataset = dataset

        return res
