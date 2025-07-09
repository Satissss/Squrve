from typing import Optional, Dict
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.task.base import BaseTask, wrap_run, TaskCompletion
from core.actor.base import Actor


class MetaTask(BaseTask):

    def __init__(
            self,
            open_parallel: bool = True,
            max_workers: int = 3,
            actor: Actor = None,
            actor_args: Dict = None,
            is_save: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.open_parallel: bool = open_parallel
        self.max_workers: int = max_workers
        self.is_save: bool = self.is_save_dataset if is_save is None else is_save

        self.actor: Actor = self.load_actor(**actor_args if actor_args else {}) if actor is None else actor

    @wrap_run
    def run(self):
        actor = self.actor
        if actor is None:
            return

        def safe_act(index):
            try:
                result = actor.act(index)
                return index, result
            except Exception as e:
                error_info = f"Error occurred while executing act() on sample {index}: {e}."
                print(error_info)
                # Log error info in dataset and task log
                row = self.dataset[index]
                row["error_info"] = error_info
                self._task_log.add_error_data(row)
                self._task_log.error(error_info)
                return None

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(safe_act, i): i for i in range(len(self.dataset))}
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    idx, val = res
                    results[idx] = val

        self.dataset = actor.dataset

        if self.is_save:
            # For ComplexTask, actor.dataset and task.dataset may differ
            self.save(self.dataset_save_path)

        return TaskCompletion(results)

    @abstractmethod
    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[Actor]:
        pass
