import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.actor.base import ComplexActor
from core.data_manage import update_dataset


class TreeActor(ComplexActor):
    """
    TreeActor is a subclass of ComplexActor that orchestrates multiple child actors
    to process a shared input and merges their outputs into a unified result.

    Purpose:
        TreeActor enables composition of several individual actors (e.g., Reducer, Parser)
        into a single cohesive actor unit. This is useful when multiple processing steps
        should be performed independently on the same input, and their results need to be combined.

    Functionality:
        - Accepts one input item and dispatches it to all child actors (`self.actors`).
        - Executes child actors either in parallel (multi-threaded) or sequentially,
          depending on the `open_actor_parallel` flag.
        - Merges results from all child actors into a single output dictionary.
        - Integrates dataset updates from each actor back into the TreeActor's dataset.
    """

    NAME = "TreeActor"
    OUTPUT_NAME: str = "TreeOutput"  # Dynamically determine

    def __init__(
            self,
            open_actor_parallel: bool = True,
            max_workers: int = 3,
            **kwargs):
        super().__init__(**kwargs)

        self.open_actor_parallel: bool = open_actor_parallel
        self.max_workers: int = max_workers

    def act(self, item, **kwargs):
        if self.open_actor_parallel:
            res = self.process_parallel(item, **kwargs)
        else:
            res = self.process_series(item, **kwargs)

        return res

    def process_series(self, item, **kwargs):
        results = {}

        dataset = self.dataset
        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        for actor in self.actors:
            actor.dataset = update_dataset(dataset, actor.dataset)
            try:
                res = actor.act(item, **kwargs)
                output_name = actor.output_name
                if output_name == "TreeOutput" and isinstance(res, dict):
                    results.update(res)
                else:
                    results[output_name] = res
            except Exception as e:
                print(f"Error occurred while executing actor '{actor.name}': {e}")

        for actor in self.actors:
            dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)
        self.dataset = dataset

        return results

    def process_parallel(self, item, **kwargs):
        results = {}
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        # Submit actor tasks to thread pool
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for actor in self.actors:
                actor.dataset = update_dataset(dataset, actor.dataset)
                futures[executor.submit(actor.act, item, **kwargs)] = actor

            for future in as_completed(futures):
                actor = futures[future]
                try:
                    res = future.result()
                    output_name = actor.output_name

                    if output_name == "TreeOutput" and isinstance(res, dict):
                        results.update(res)
                    else:
                        results[output_name] = res

                except Exception as e:
                    print(f"Error occurred while executing actor '{actor.name}': {e}")

        # Merge datasets in the main thread
        for actor in self.actors:
            # todo 如果所有 actor 完成相同的功能，直接 update 可能覆盖之前的结果，因此需添加筛选逻辑，再更新。
            dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)

        self.dataset = dataset
        return results
