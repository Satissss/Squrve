from abc import abstractmethod
from typing import List, Union

from core.task.base import BaseTask
from core.utils import throw_hash_id, timestamp_hash_key


class MultiTask(BaseTask):
    _ticket_number: int = 0

    def __init__(
            self,
            tasks: List[BaseTask] = None,
            **kwargs
    ):
        # Initialize the task_id
        task_id = self.__init_task_id__(kwargs.get("task_id", None))
        super().__init__(task_id, None, **kwargs)

        self.tasks = [] if tasks is None else tasks

    def set(self, tasks: Union[List[BaseTask]]):
        self.tasks = tasks

    def add(self, tasks: Union[BaseTask, List[BaseTask]]):
        if isinstance(tasks, BaseTask):
            self.tasks.append(tasks)
        else:
            self.tasks.extend(tasks)

    @abstractmethod
    def run(self):
        pass

    def eval(self, force: bool = False):
        all_res = {}
        for task_ in self.tasks:
            res = task_.eval(force)
            if not res:
                continue
            if res:
                all_res.update(res)

        if all_res:
            self._eval_results.update(all_res)
            self._task_log["Evaluation Results"] = all_res
            return all_res

        return None

    @property
    def eval_results(self):
        eval_results = {}
        for task_ in self.tasks:
            if task_.is_end:
                eval_results.update(task_.eval_results)

        return eval_results

    def is_empty(self):
        return len(self.tasks) == 0

    def save(self, **kwargs):
        for task_ in self.tasks:
            task_.save()

    def __init_task_id__(self, task_id: str = None):
        if task_id is not None:
            return task_id
        task_id = f"mtk_{throw_hash_id(self._ticket_number)}_{timestamp_hash_key()}"
        self._base_number += 1

        return task_id
