import warnings

from core.task.multi.MultiTask import MultiTask
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _run_task(task):
    if not hasattr(task, "actor") or not task.actor:
        raise Exception("The actor is not available")
    if hasattr(task.actor, "llm") and task.actor.llm:
        task.actor.llm.reinit_client()

    return task.run()


class ParallelTask(MultiTask):
    """ Task For Text-to-SQL """

    NAME = "ParallelTask"

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def run(self):
        if not self.tasks:
            warnings.warn(f"The `tasks` list is empty. Run is stopped. ", category=UserWarning)
            return
        max_processes = os.cpu_count() or 1
        num_processes = min(len(self.tasks), max_processes)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(_run_task, task) for task in self.tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    warnings.warn(f"Task failed with error: {e}", category=UserWarning)
