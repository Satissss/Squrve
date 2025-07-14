from typing import List, Dict, Union, Callable
from abc import ABC, abstractmethod
from os import PathLike
import warnings
import functools
import time

from core.data_manage import Dataset
from core.evaluate import Evaluator
from core.log import Logger
from core.utils import throw_hash_id, timestamp_hash_key


class BaseTask(ABC):
    """ Base classes for all Task Object """

    NAME: str
    _base_number: int = 0

    _task_id: str
    task_name: str
    task_info: str
    dataset: Dataset

    _task_log: Logger
    _is_end: bool
    call_back: Callable

    """
    Note: eval_results is used to store the evaluation results of the specific tasks.
    Here is an example of eval_results:
    {
        <eval_type>: {
            "value": <A number like recall or execution accuracy>,
            "info": [Opt. Store some relevant information during the evaluation process.]
        }
    }
    """
    eval_type: List
    _eval_results: Dict

    def __init__(
            self,
            task_id: str = None,
            dataset: Dataset = None,
            task_name: str = "",
            task_info: str = "",
            eval_type: List = None,
            call_back: Callable = None,
            log_save_path: Union[str, PathLike] = None,
            is_save_dataset: bool = True,
            dataset_save_path: Union[str, PathLike] = None,
            **kwargs
    ):
        self._task_id: str = self.__init_task_id__(task_id)
        self.task_name: str = task_name
        self.task_info: str = task_info
        self.dataset: Dataset = dataset
        self.eval_type: List = [] if eval_type is None else eval_type
        self._eval_results = {}
        self._is_end = False
        self.is_save_dataset = is_save_dataset
        self.dataset_save_path: Union[str, PathLike] = dataset_save_path
        self._task_log: Logger = self.__init_logger__(log_save_path)

        self.call_back: Callable = run_call_back if call_back is None else call_back

        self.NAME = "BaseTask"

    @property
    def name(self):
        return self.NAME

    @abstractmethod
    def run(self):
        pass

    @property
    def task_id(self):
        return self._task_id

    @property
    def is_end(self):
        return self._is_end

    @property
    def eval_results(self):
        return self._eval_results

    @property
    def log(self):
        return self._task_log

    def end(self):
        self._is_end = True

    def __init_task_id__(self, task_id: str = None):
        if task_id is not None:
            return task_id
        task_id = f"tk_{throw_hash_id(self._base_number)}_{timestamp_hash_key()}"
        self._base_number += 1

        return task_id

    def __init_logger__(self, log_save_path: Union[str, PathLike]):
        logger = Logger(log_save_path)
        # add task_
        logger["Basic Task Information"] = str(self)

        return logger

    def __str__(self):
        """ Output the key information of the task. """

        output = f"The Information of {self.NAME} ({self._task_id}).\n"
        output += f"[task_id]: {self._task_id}\n"
        if self.task_name:
            output += f"[task_name]: {self.task_name}\n"
        if self.task_info:
            output += f"[task_info]: {self.task_info}\n"
        if self.dataset:
            output += f"[dataset]:\n\t[dataset]: {self.dataset.dataset_index}\t[schema]: {self.dataset.schema_index}"
        if self.eval_type:
            output += f"[eval_type]: {self.eval_type}\n"
        if self._eval_results:
            output += f"[task_id]: {self._eval_results}\n"
        if self._is_end:
            output += f"[is_end]: {self._is_end}\n"
        if self.is_save_dataset:
            output += f"[is_save_dataset]: {self.is_save_dataset}"

        return output

    def save(self, dataset_save_path: Union[str, PathLike] = None, **kwargs):
        # save the dataset
        if self.is_save_dataset and self.dataset is not None:
            if not dataset_save_path:
                dataset_save_path = self.dataset_save_path
            self.dataset.save_data(dataset_save_path)

        # save the logger
        if self._eval_results:
            self._task_log["eval_results"] = self._eval_results
        self._task_log.save()

    def eval(self, force: bool = False):
        if self._is_end or force:
            evaluator = Evaluator(dataset=self.dataset, eval_type=self.eval_type)
            res = evaluator.eval_all()
            print(f"Evaluation is over !")
            for key, val in res.items():
                print(f"eval_type: {key},\tnumber: {val.get('valid_num')}\tresult: {val.get('avg')}")

            if res:
                self._eval_results.update(res)
                self._task_log["Evaluation Results"] = res
                return res

        return None


class TaskCompletion:

    def __init__(
            self,
            value,
    ):
        """ todo 可以考虑返回一个日志对象之类的，用于存放 Task 运行时的错误信息、报错信息等 """
        self._value = value

    @property
    def value(self):
        return self._value


def wrap_run(func):
    """ A Decorator for the `startup_run()` function of class Task"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # startup_run the function
        obj = args[0]
        print(f"{obj.name}: {obj.task_id} begin to startup_run!")
        res = func(*args, **kwargs)
        print(f"{obj.name}: {obj.task_id} end to startup_run!")
        end_time = time.time()
        if isinstance(obj, BaseTask):
            obj.end()
            # startup_run the call_back function to collect information
            obj.call_back(obj, res=res, run_time=end_time - start_time)

        return res

    return wrapper


def run_call_back(
        task: BaseTask = None,
        run_log: Logger = None,
        res: TaskCompletion = None,
        run_time: float = None,
        **kwargs
):
    """ A simple implement of the call-back function after call the `startup_run` function in Task object """

    run_log = task.log if run_log is None else run_log
    if run_log is None:
        warnings.warn(f"The run_log is not available.", category=UserWarning)
        return

    if res is not None:
        run_log["run_results"] = res.value

    run_log["run_time"] = run_time
    run_log.set_by_dict(**kwargs)
