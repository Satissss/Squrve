from typing import Optional, Dict
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

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
            logger.warning("Actor 未初始化，跳过任务执行")
            return

        logger.info(f"开始执行 MetaTask: {self.name} ({self.task_id}), 数据集大小: {len(self.dataset)}")

        def safe_act(index):
            ins_id = actor.dataset[index]['instance_id']
            data_logger = self._task_log.generate_data_logger(ins_id)
            try:
                data_logger.info(f"开始处理样本 {ins_id}")
                result = actor.act(index, data_logger=data_logger)
                data_logger.info(f"样本 {ins_id} 处理完成")
                return index, result
            except Exception as e:
                error_info = f"Error occurred while executing act() on sample {ins_id}: {e}."
                data_logger.info(error_info)
                # Log error info in dataset and task log
                row = self.dataset[index]
                row["error_info"] = error_info
                self._task_log.add_error_data(row)
                self._task_log.error(error_info)
                return None
            finally:
                data_logger.save()

        results = {}
        logger.info(f"使用线程池执行任务，最大工作线程数: {self.max_workers}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(safe_act, i): i for i in range(len(self.dataset))}
            completed_count = 0
            for future in as_completed(futures):
                res = future.result()
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(self.dataset):
                    logger.info(f"任务进度: {completed_count}/{len(self.dataset)} ({completed_count/len(self.dataset)*100:.1f}%)")
                if res is not None:
                    idx, val = res
                    results[idx] = val

        logger.info(f"MetaTask 执行完成: {self.name} ({self.task_id}), 成功处理: {len(results)}/{len(self.dataset)} 个样本")
        self.dataset = actor.dataset

        if self.is_save:
            # For ComplexTask, actor.dataset and task.dataset may differ
            logger.info(f"保存任务结果到: {self.dataset_save_path}")
            self.save(self.dataset_save_path)

        return TaskCompletion(results)

    @abstractmethod
    def load_actor(self, actor_type: str = None, **kwargs) -> Optional[Actor]:
        pass
