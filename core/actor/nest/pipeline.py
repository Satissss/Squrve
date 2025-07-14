import warnings
import traceback
from loguru import logger

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

        logger.info(f"PipelineActor 开始执行，包含 {len(self.actors)} 个 actors")
        output_name = ""
        res = None

        for i, actor in enumerate(self.actors):
            logger.info(f"执行第 {i+1}/{len(self.actors)} 个 actor: {actor.name}")
            actor.dataset = update_dataset(dataset, actor.dataset)

            try:
                res = actor.act(item, **results)
                output_name = actor.output_name
                logger.info(f"Actor {actor.name} 执行完成，输出名称: {output_name}")

                if output_name == "TreeOutput" and isinstance(res, dict):
                    results.update(res)
                else:
                    results[output_name] = res

                dataset = actor.dataset

            except Exception as e:
                error_msg = f"Error occurred while executing actor '{actor.name}': {e}"
                logger.error(error_msg)
                
                # Check for meta tensor error and provide specific guidance
                if "meta tensor" in str(e).lower() and "to_empty" in str(e).lower():
                    error_msg += "\n\nThis is a PyTorch meta tensor error. The issue has been fixed in the latest code. "
                    error_msg += "If you're still seeing this error, please ensure you're using the updated version "
                    error_msg += "of the embedding model initialization code."
                
                print(error_msg)
                print(f"Full traceback: {traceback.format_exc()}")

        self.OUTPUT_NAME = output_name
        self.dataset = dataset
        logger.info(f"PipelineActor 执行完成，最终输出名称: {output_name}")

        return res
