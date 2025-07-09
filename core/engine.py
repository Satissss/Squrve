import warnings
from os import PathLike
from pathlib import Path
from typing import Union, List, Dict, Optional
import time

from core.base import Router
from core.data_manage import DataLoader, Dataset
from core.task.meta.ComplexTask import ComplexTask
from core.task.meta.MetaTask import MetaTask
from core.utils import throw_hash_id, timestamp_hash_key
from core.task.base import BaseTask
from core.task.multi.ParallelTask import ParallelTask
from core.task.multi.SequenceTask import SequenceTask
from core.actor.base import Actor
from core.actor.nest.pipeline import PipelineActor
from core.actor.nest.tree import TreeActor


class Engine:
    """ 代表 Text-to-SQL 流程的一次运行，负责创建并执行所有的 Task ，收集并评估结果"""

    registered_db_type = [
        "GenerateTask",
        "generate",
        "ParseTask",
        "parse",
        "ReduceTask",
        "reduce"
    ]

    def __init__(
            self,
            router: Router = None,
            dataloader: DataLoader = None,
            tasks: Union[BaseTask, List[BaseTask], Dict] = None,
            task_meta: Union[Dict, List[Dict]] = None,
            cpx_task_meta: Union[Dict, List[Dict]] = None,
            exec_process: Union[List, Dict] = None,
            actors: Dict = None
    ):
        self.router: Router = Router() if router is None else router
        self.dataloader: DataLoader = DataLoader(self.router) if dataloader is None else dataloader
        self.actors: Dict = {} if actors is None else actors

        self.tasks: Dict = self.init_tasks(tasks, self.router.task_meta if task_meta is None else task_meta)
        self.init_complex_task(cpx_task_meta=self.router.cpx_task_meta if cpx_task_meta is None else cpx_task_meta)
        self.exec_process: Union[List, Dict] = router.exec_process if exec_process is None else exec_process

    def check_task_id(self, ind, task_id: str, all_tasks: Dict = None):
        all_tasks_ = {}
        if all_tasks is None:
            if hasattr(self, "tasks"):
                all_tasks_ = self.tasks
        else:
            all_tasks_ = all_tasks
        if task_id is None:
            # throw out a unique hash key ID.
            task_id = f"tk_{throw_hash_id(ind)}_{timestamp_hash_key()}"
        if task_id in all_tasks_.keys():
            warnings.warn(f"task id 设置重复，跳过第 {ind} 个 Task 初始化.", category=UserWarning)
            return False, task_id

        return True, task_id

    def check_task_type(self, ind, task_type: str):
        if task_type not in self.registered_db_type:
            warnings.warn(f"task_type 错误或未定义，跳过第 {ind} 个 Task 初始化.", category=UserWarning)
            return False, task_type
        return True, task_type

    def check_data_source(self, ind, data_source_index: str, dataloader: DataLoader = None):
        if dataloader is None:
            dataloader = self.dataloader

        if data_source_index is None:
            data_source_index = dataloader.get_data_source_index(output_format="list")
            if data_source_index is None or len(data_source_index) != 1:
                warnings.warn(f"data_source 参数未定义，跳过第 {ind} 个 Task 初始化.", category=UserWarning)
                return False, data_source_index
        if data_source_index.count(":") == 2:
            file_name_ = "_".join(data_source_index.split(":"))
            save_data_source = Path(dataloader.data_source_dir) / (file_name_ + ".json")
            if dataloader.overwrite_exist_file or not save_data_source.exists():
                dataloader.init_benchmark_dataset(data_source_index, file_name_,
                                                  save_data_source=save_data_source)
            dataloader.update_data_source(save_data_source, file_name_)
            data_source_index = file_name_
        else:
            if Path(data_source_index).is_file():
                index_ = Path(data_source_index).stem
                dataloader.update_data_source(data_source_index, index_)
                data_source_index = index_
            else:
                all_index = dataloader.get_data_source_index(output_format="list")
                if not all_index or data_source_index not in all_index:
                    warnings.warn(f"data_source 参数定义无效，跳过第 {ind} 个 Task 初始化.",
                                  category=UserWarning)
                    return False, data_source_index

        return True, data_source_index

    def check_schema_source(self, ind, schema_source_index: str, dataloader: DataLoader = None):
        if dataloader is None:
            dataloader = self.dataloader

        if schema_source_index is None:
            schema_source_index = dataloader.get_schema_source_index()
            if schema_source_index is None or len(schema_source_index) != 1:
                warnings.warn(f"schema_source 参数未定义，跳过第 {ind} 个 Task 初始化.", category=UserWarning)
                return False, schema_source_index

        if ":" in schema_source_index:
            file_name_ = "_".join(schema_source_index.split(":"))
            multi_db_ = dataloader.query_multi_database(file_name_)
            vector_store_ = dataloader.query_vector_store(file_name_)
            save_schema_source = Path(dataloader.schema_source_dir) / file_name_
            if dataloader.skip_schema_init:
                save_schema_source = save_schema_source / "schema.json"
            dataloader.init_benchmark_schema(schema_source_index, multi_db_,
                                             save_schema_source=save_schema_source)
            dataloader.update_schema_save_source({file_name_: str(save_schema_source)}, multi_db_,
                                                 vector_store_)
            schema_source_index = file_name_
        else:
            if Path(schema_source_index).is_file():
                index_ = Path(schema_source_index).stem
                multi_db_ = dataloader.query_multi_database(index_)
                vector_store_ = dataloader.query_vector_store(index_)
                if dataloader.skip_schema_init:
                    save_schema_source = schema_source_index
                else:
                    save_schema_source = Path(dataloader.schema_source_dir) / index_
                if not dataloader.skip_schema_init:
                    dataloader.central_schema_process(schema_source_index,
                                                      save_schema_source=save_schema_source,
                                                      multi_db=multi_db_)
                dataloader.update_schema_save_source({index_: str(save_schema_source)}, multi_db_,
                                                     vector_store_)
                schema_source_index = index_
            elif Path(schema_source_index).is_dir():
                # Only support raw schema source path
                index_ = Path(schema_source_index).stem
                multi_db_ = dataloader.query_multi_database(index_)
                vector_store_ = dataloader.query_vector_store(index_)
                dataloader.update_schema_save_source({index_: str(schema_source_index)}, multi_db_,
                                                     vector_store_)
                schema_source_index = index_
            else:
                all_index = self.dataloader.get_schema_source_index()
                if not all_index or schema_source_index not in all_index:
                    warnings.warn(f"schema_source 参数定义无效，跳过第 {ind} 个 Task 初始化.",
                                  category=UserWarning)
                    return False, schema_source_index

        return True, schema_source_index

    def init_tasks(
            self,
            tasks: Union[BaseTask, List[BaseTask], Dict] = None,
            task_meta: Union[Dict, List[Dict]] = None,
            dataloader: DataLoader = None,
    ):
        all_tasks = {}
        if hasattr(self, "tasks"):
            all_tasks.update(self.tasks)

        if dataloader is None:
            dataloader = self.dataloader

        if task_meta:
            if isinstance(task_meta, dict):
                task_meta = [task_meta]

            # check the task meta information
            for ind, meta in enumerate(task_meta):
                # check task_id
                task_id = meta.get("task_id", None)
                flag, task_id = self.check_task_id(ind, task_id, all_tasks)
                if not flag:
                    continue
                # check task_type
                task_type = meta.get("task_type", None)
                flag, task_type = self.check_task_type(ind, task_type)
                if not flag:
                    continue
                # check data_source.
                data_source_index = meta.get("data_source", None)
                flag, data_source_index = self.check_data_source(ind, data_source_index, dataloader)
                if not flag:
                    continue
                # check schema_source
                schema_source_index = meta.get("schema_source", None)
                flag, schema_source_index = self.check_schema_source(ind, schema_source_index, dataloader)
                if not flag:
                    continue
                # generate dataset
                kwargs = meta.get("meta", {})
                dataset = dataloader.generate_dataset(data_source_index, schema_source_index,
                                                      **kwargs.get("dataset", {}))
                # generate task
                generate_args = {
                    "task_id": task_id,
                    "dataset": dataset,
                    "task_type": task_type,
                    "task_name": meta.get("task_name"),
                    "task_info": meta.get("task_info"),
                    "eval_type": meta.get("eval_type"),
                    "log_save_path": meta.get("log_save_path"),
                    "is_save_dataset": meta.get("is_save_dataset"),
                    "dataset_save_path": meta.get("dataset_save_path"),
                    "open_parallel": meta.get("open_parallel"),
                    "max_workers": meta.get("max_workers"),
                    **kwargs.get("task", {}),
                    "llm_args": kwargs.get("llm", {}),
                    "actor_args": kwargs.get("actor", {})
                }
                task = self.generate_task(**generate_args)
                if task is not None:
                    all_tasks[task_id] = task

        if tasks is not None:
            if isinstance(tasks, dict):
                for id_, task_ in tasks.items():
                    if not isinstance(task_, BaseTask):
                        warnings.warn(f"未传入 Task 对象，跳过 {id_}  Task 初始化.", category=UserWarning)
                        continue
                    if id_ in all_tasks.keys():
                        warnings.warn(f"task_id 参数定义无效，跳过 {id_}  Task 初始化.", category=UserWarning)
                        continue
                    all_tasks[id_] = task_
            else:
                # Input a list of Task Object
                if not isinstance(tasks, list):
                    tasks = [tasks]
                for ind, task_ in enumerate(tasks):
                    # Generate task_id. Use 711 just because it's my girlfriend's birthday.
                    id_ = f"tk_{throw_hash_id(711 + ind)}_{timestamp_hash_key()}"
                    if not isinstance(task_, BaseTask):
                        warnings.warn(f"未传入 Task 对象，跳过 {id_}  Task 初始化.", category=UserWarning)
                        continue
                    if id_ in all_tasks.keys():
                        warnings.warn(f"task_id 参数定义无效，跳过 {id_}  Task 初始化.", category=UserWarning)
                        continue
                    all_tasks[id_] = task_

        return all_tasks

    def generate_task(
            self,
            task_id: str,
            dataset: Dataset,
            task_type: str,
            task_name: str = "",
            task_info: str = "",
            eval_type: List = None,
            log_save_path: Union[str, PathLike] = None,
            is_save_dataset: bool = None,
            dataset_save_path: Union[str, PathLike] = None,
            open_parallel: bool = True,
            max_workers: int = 5,
            actor: Actor = None,
            actor_args: Dict = None,
            llm_args: Dict = None,
            **kwargs
    ):
        if log_save_path is None:
            log_save_dir = Path(self.router.default_log_save_dir)
            log_save_path = log_save_dir / task_id / "log.txt"

        if is_save_dataset is None:
            is_save_dataset = self.router.is_save_dataset

        if open_parallel is None:
            open_parallel = self.router.open_parallel

        if max_workers is None:
            max_workers = self.router.max_workers

        if actor is None:
            actor = self.actors.get(task_id, None)

        if not llm_args:
            llm = self.dataloader.llm
        else:
            llm = self.dataloader.init_llm(**llm_args)

        init_args = {
            "llm": llm,
            "task_id": task_id,
            "dataset": dataset,
            "task_name": task_name,
            "task_info": task_info,
            "eval_type": eval_type,
            "log_save_path": log_save_path,
            "is_save_dataset": is_save_dataset,
            "dataset_save_path": dataset_save_path,
            "open_parallel": open_parallel,
            "max_workers": max_workers,
            "actor_args": actor_args,
            "actor": actor,
            **kwargs
        }

        task = None
        if task_type in ("GenerateTask", "generate"):
            task = self.generate_generate_task(**init_args)

        elif task_type in ("ParseTask", "parse"):
            task = self.generate_parse_task(**init_args)

        elif task_type in ("ReduceTask", "reduce"):
            task = self.generate_reduce_task(**init_args)

        return task

    def generate_generate_task(self, **kwargs):
        from core.task.meta.GenerateTask import GenerateTask
        # initialize parameters by router.
        if kwargs.get("save_dir", None) is None:
            kwargs["save_dir"] = self.router.generate_save_dir

        task = GenerateTask(**kwargs)
        return task

    def generate_parse_task(self, **kwargs):
        from core.task.meta.ParseTask import ParseTask
        # initialize parameters by router.
        if kwargs.get("save_dir", None) is None:
            kwargs["save_dir"] = self.router.parse_save_dir

        if kwargs.get("output_format", None) is None:
            kwargs["output_format"] = self.router.parse_output_format

        task = ParseTask(**kwargs)
        return task

    def generate_reduce_task(self, **kwargs):
        from core.task.meta.ReduceTask import ReduceTask
        # initialize parameters by router.
        if kwargs.get("save_dir", None) is None:
            kwargs["save_dir"] = self.router.reduce_save_dir

        if kwargs.get("output_format", None) is None:
            kwargs["output_format"] = self.router.reduce_output_format

        task = ReduceTask(**kwargs)
        return task

    def init_complex_task(
            self,
            tasks: Dict = None,  # 已经处理好的 tasks, todo 或许可以增加列表等方式传入
            cpx_task_meta: Union[Dict, List[Dict]] = None,
            is_update_tasks: bool = True
    ):
        meta_tasks = tasks if tasks else {}
        if self.tasks:
            meta_tasks.update(self.tasks)

        if cpx_task_meta:
            if isinstance(cpx_task_meta, dict):
                cpx_task_meta = [cpx_task_meta]

            # check the task meta information
            for ind, meta in enumerate(cpx_task_meta):
                # check task_id
                task_id = meta.get("task_id", None)
                flag, task_id = self.check_task_id(ind, task_id, meta_tasks)
                if not flag:
                    continue
                kwargs = meta.get("meta", {})
                generate_args = {
                    "task_id": task_id,
                    "task_lis": meta["task_lis"],
                    "meta_tasks": meta_tasks,
                    "task_name": meta.get("task_name"),
                    "task_info": meta.get("task_info"),
                    "eval_type": meta.get("eval_type"),
                    "log_save_path": meta.get("log_save_path"),
                    "is_save_dataset": meta.get("is_save_dataset"),
                    "dataset_save_path": meta.get("dataset_save_path"),
                    "open_parallel": meta.get("open_parallel"),
                    "max_workers": meta.get("max_workers"),
                    "open_actor_parallel": meta.get("open_actor_parallel"),
                    **kwargs.get("task", {}),
                    "actor_args": kwargs.get("actor")
                }
                task = self.generate_complex_task(**generate_args)
                if task is not None:
                    meta_tasks[task_id] = task

        if is_update_tasks:
            self.tasks = meta_tasks

        return meta_tasks

    def generate_complex_task(
            self,
            task_id: str,
            task_lis: List,
            meta_tasks: Dict = None,
            task_name: str = "",
            task_info: str = "",
            eval_type: List = None,
            log_save_path: Union[str, PathLike] = None,
            is_save_dataset: bool = None,
            dataset_save_path: Union[str, PathLike] = None,
            open_parallel: bool = True,
            max_workers: int = 5,
            open_actor_parallel: bool = True,
            actor_args: Dict = None,
            **kwargs
    ):
        if meta_tasks is None:
            meta_tasks = self.tasks

        if log_save_path is None:
            log_save_dir = Path(self.router.default_log_save_dir)
            log_save_path = log_save_dir / task_id / "log.txt"

        if is_save_dataset is None:
            is_save_dataset = self.router.is_save_dataset

        if open_parallel is None:
            open_parallel = self.router.open_parallel

        if max_workers is None:
            max_workers = self.router.max_workers

        if actor_args is None:
            actor_args = {}

        actor = self.load_complex_actor(task_lis, meta_tasks, open_actor_parallel, max_workers, **actor_args)
        if not actor:
            return None

        init_args = {
            "task_id": task_id,
            "dataset": actor.dataset,
            "actor": actor,
            "task_name": task_name,
            "task_info": task_info,
            "eval_type": eval_type,
            "log_save_path": log_save_path,
            "is_save_dataset": is_save_dataset,
            "dataset_save_path": dataset_save_path,
            "open_parallel": open_parallel,
            "max_workers": max_workers,
            **kwargs
        }
        task = ComplexTask(**init_args)

        return task

    def load_complex_actor(
            self,
            task_lis: List[Union[str, List[str]]],
            meta_tasks: Dict[str, 'MetaTask'],
            open_actor_parallel: bool = True,
            max_workers: int = 3,
            **kwargs
    ) -> Optional[ComplexTask]:
        if meta_tasks is None:
            meta_tasks = self.tasks
        actor = self.load_complex_actor_simple(task_lis, meta_tasks, open_actor_parallel, max_workers, **kwargs)

        return actor

    @classmethod
    def load_complex_actor_simple(
            cls,
            task_lis: List[Union[str, List[str]]],
            meta_tasks: Dict[str, 'MetaTask'],
            open_actor_parallel: bool = True,
            max_workers: int = 3,
            **kwargs
    ):
        """
        Parse the task_lis by simple format without recursive.
        Example: ["task1", ["task2", "task3"], "task_4"]
        """
        if not task_lis or not meta_tasks:
            return None

        def get_valid_task(task_name: str):
            if task_name not in meta_tasks:
                warnings.warn(f"The task `{task_name}` is not available.", category=UserWarning)
                return None
            task_ = meta_tasks[task_name]
            if not isinstance(task_, MetaTask):
                raise TypeError(f"Task `{task_name}` is not a valid MetaTask instance.")
            return task_

        pipeline_actor = PipelineActor()

        for item in task_lis:
            try:
                if isinstance(item, str):
                    task = get_valid_task(item)
                    if task:
                        pipeline_actor.add(task.load_actor(**kwargs.get(item, {})))

                elif isinstance(item, list):
                    tree_actor = TreeActor(open_actor_parallel=open_actor_parallel, max_workers=max_workers)
                    for sub_name in item:
                        if not isinstance(sub_name, str):
                            warnings.warn(f"Sub-task `{sub_name}` in list `{item}` is not a string.",
                                          category=UserWarning)
                            continue
                        sub_task = get_valid_task(sub_name)
                        if sub_task:
                            tree_actor.add(sub_task.load_actor(**kwargs.get(sub_name, {})))
                    if not tree_actor.is_empty:
                        pipeline_actor.add(tree_actor)

                else:
                    warnings.warn(f"Unsupported task format: `{item}`", category=UserWarning)

            except Exception as e:
                print(f"Error processing task `{item}`: {e}")

        return pipeline_actor if not pipeline_actor.is_empty else None

    def parse_exec_process(self, exec_process: Union[List, Dict]):
        """ Parse exec_process. """
        if isinstance(exec_process, list):
            exec_task = self.parse_exec_process_from_list(exec_process)
            return exec_task

        elif isinstance(exec_process, dict):
            exec_task = self.parse_exec_process_from_dict(exec_process)
            return exec_task

        warnings.warn(f"exec_process is not available or incorrect.", category=UserWarning)
        exec_task = self.parse_exec_process_simple()

        return exec_task

    def parse_exec_process_from_dict(self, exec_process: Dict):
        """
        Parse exec_process from a Dict object. Return a MultiTask Object in the end.
        Here is a simple example:
        {
            "type": "sequence",
            "tasks": [
                "task1",
                {
                    "type": "parallel",
                    "tasks": ["task2", "task3"]
                },
                "task4"
            ]
        }
        That said, if there exists a dict object in the `value list` of the `tasks key`,
        then the method will recursively call itself to parse the new Dict Object.
        """
        if not exec_process:
            return None

        multi_type = exec_process.get("type", "parallel")
        multi_tasks = exec_process.get("tasks", [])
        # check the multi_tasks list first
        if not multi_tasks or not isinstance(multi_tasks, list):
            warnings.warn("The `tasks` list is not available.", category=UserWarning)
            return None
        if multi_type in ("parallel", "para"):
            execute_task = ParallelTask()
        elif multi_type == ("sequence", "seq"):
            execute_task = SequenceTask()
        else:
            warnings.warn(f"The `{multi_type}` is incorrect.", category=UserWarning)
            return None

        for task_ in multi_tasks:
            if isinstance(task_, str):
                tk = self.get_task_by_id(task_)
            elif isinstance(task_, dict):
                tk = self.parse_exec_process_from_dict(task_)
            else:
                warnings.warn(f"The `{task_}` is out of format.", category=UserWarning)
                continue

            if tk is None:
                continue
            else:
                execute_task.add(tk)

        if execute_task.is_empty():
            return None
        return execute_task

    def parse_exec_process_from_list(self, exec_process: List):
        """
        Parse exec_process from a List object. Return a MultiTask Object in the end.
        Here is a simple example:
        ["task_1","task_2",["task_3","task_4","~p"],"~s"]
        That said, if there exists a List object in the `value` of the `tasks` list,
        then the method will recursively call itself to parse the new List Object.
        """
        if not exec_process:
            return None

        if not exec_process:
            warnings.warn("The exec_process is empty.", category=UserWarning)
            return None

        if "~p" in exec_process or "*p" in exec_process:
            execute_task = ParallelTask()
        else:
            execute_task = SequenceTask()

        for task_ in exec_process:
            if isinstance(task_, str):
                tk = self.get_task_by_id(task_)
            elif isinstance(task_, list):
                tk = self.parse_exec_process_from_list(task_)
            else:
                warnings.warn(f"The `{task_}` is out of format.", category=UserWarning)
                continue
            if tk is None:
                continue
            else:
                execute_task.add(tk)

        if execute_task.is_empty():
            return None
        return execute_task

    def parse_exec_process_simple(self):
        if self.tasks:
            return None
        tasks = list(self.tasks.values())

        execute_task = SequenceTask()
        for task_ in tasks:
            if isinstance(task_, BaseTask):
                execute_task.add(task_)

        if execute_task.is_empty():
            return None
        return execute_task

    @property
    def task_ids(self):
        return list(self.tasks.keys())

    def get_task_by_id(self, task_id: str):
        if task_id not in self.task_ids:
            warnings.warn(f"The `{task_id}` is incorrect.", category=UserWarning)
            return None

        return self.tasks.get(task_id)

    def execute(self):
        execute_task = self.parse_exec_process(self.exec_process)
        if execute_task is None:
            return
        # execute the task
        start_time = time.time()
        execute_task.run()
        end_time = time.time()
        # save the `task`
        # execute_task.save()

        print(
            f"Engine 运行完毕. {execute_task.name} ({execute_task.task_id}) 运行时间为 {end_time - start_time:.6f} s.")

    def skip_execute(self):
        for id_, task_ in self.tasks.items():
            task_.end()

    def evaluate(self, force: bool = False):
        all_res = {}
        for id_, task_ in self.tasks.items():
            if not isinstance(task_, BaseTask):
                continue
            res = task_.eval(force)
            if not res:
                continue
            if res:
                all_res.update(res)
        print(f"Engine 评估完毕!!!!")

        return all_res


