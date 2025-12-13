import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Union

from loguru import logger

from core.actor.base import ComplexActor, Actor, MergeStrategy, MergeFunction
from core.actor.generator import BaseGenerator
from core.data_manage import update_dataset


def group_partition(actors: Union[Actor, List[Actor]]):
    from core.actor.parser.BaseParse import BaseParser
    from core.actor.scaler.BaseScale import BaseScaler
    from core.actor.optimizer.BaseOptimize import BaseOptimizer
    new_actors = {}
    rtn_actors = []
    for actor in actors:
        if isinstance(actor, BaseParser):
            if "parse_group" not in new_actors:
                new_actors["parse_group"] = ParseActorGroup()
            new_actors["parse_group"].add(actor)
        elif isinstance(actor, BaseScaler):
            if "scale_group" not in new_actors:
                new_actors["scale_group"] = ScaleActorGroup()
            new_actors["scale_group"].add(actor)
        elif isinstance(actor, BaseOptimizer):
            if "optimize_group" not in new_actors:
                new_actors["optimize_group"] = OptimizeActorGroup()
            new_actors["optimize_group"].add(actor)
        else:
            rtn_actors.append(actor)
    for _, group in new_actors.items():
        rtn_actors.append(group)

    return rtn_actors


def actor_group_partition(func):
    def wrapper(self, item, data_logger=None, **kwargs):
        # group the actor
        self.actors = group_partition(self.actors)
        if data_logger:
            data_logger.info(f"The number of actor group is: {len(self.actors)}")

        res = func(self, item, data_logger=data_logger, **kwargs)
        return res

    return wrapper


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
    OUTPUT_NAME = "TreeOutput"  # Dynamically determine

    def __init__(
            self,
            open_actor_parallel: bool = True,
            max_workers: int = 3,
            **kwargs):
        super().__init__(**kwargs)

        self.open_actor_parallel: bool = open_actor_parallel
        self.max_workers: int = max_workers

    @actor_group_partition
    def act(self, item, **kwargs):
        logger.info(f"TreeActor 开始执行，并行模式: {self.open_actor_parallel}, 包含 {len(self.actors)} 个 actors")
        if self.open_actor_parallel:
            res = self.process_parallel(item, **kwargs)
        else:
            res = self.process_series(item, **kwargs)
        logger.info(f"TreeActor 执行完成")
        return res

    def process_series(self, item, **kwargs):
        results = kwargs
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        logger.info(f"TreeActor 串行执行模式，开始处理 {len(self.actors)} 个 actors")
        for i, actor in enumerate(self.actors):
            logger.info(f"串行执行第 {i + 1}/{len(self.actors)} 个 actor: {actor.name}")
            actor.dataset = update_dataset(dataset, actor.dataset)
            output_name = actor.output_name
            try:
                res = actor.act(item, **kwargs)
                logger.info(f"Actor {actor.name} 串行执行完成，输出名称: {output_name}")
                if output_name == "TreeOutput" and isinstance(res, dict):
                    results.update(res)
                else:
                    merge_func = MergeFunction.get_method(actor.strategy)
                    merge_func(results, output_name, res)

            except Exception as e:
                error_msg = f"Error occurred while executing actor '{actor.name}': {e}"
                logger.error(error_msg)

        for actor in self.actors:
            dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)
        self.dataset = dataset

        return results

    def process_parallel(self, item, **kwargs):
        results = kwargs
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        logger.info(f"TreeActor 并行执行模式，最大工作线程数: {self.max_workers}")
        # Submit actor tasks to thread pool
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, actor in enumerate(self.actors):
                logger.info(f"提交第 {i + 1}/{len(self.actors)} 个 actor 到线程池: {actor.name}")
                actor.dataset = update_dataset(dataset, actor.dataset)
                futures[executor.submit(actor.act, item, **kwargs)] = actor

            completed_count = 0
            for future in as_completed(futures):
                actor = futures[future]
                completed_count += 1
                logger.info(f"并行执行进度: {completed_count}/{len(self.actors)} - Actor {actor.name} 完成")
                try:
                    res = future.result()
                    output_name = actor.output_name
                    logger.info(f"Actor {actor.name} 并行执行完成，输出名称: {output_name}")

                    if output_name == "TreeOutput" and isinstance(res, dict):
                        results.update(res)
                    else:
                        merge_func = MergeFunction.get_method(actor.strategy)
                        merge_func(results, output_name, res)

                except Exception as e:
                    error_msg = f"Error occurred while executing actor '{actor.name}': {e}"
                    logger.error(error_msg)

        # Merge datasets in the main thread
        logger.info("开始合并数据集...")
        for actor in self.actors:
            # todo 如果所有 actor 完成相同的功能，直接 update 可能覆盖之前的结果，因此需添加筛选逻辑，再更新。
            dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)

        self.dataset = dataset
        logger.info("数据集合并完成")
        return results


class ActorGroup(TreeActor):
    def group_preprocess(self, **kwargs):
        pass

    def group_postprocess(self, **kwargs):
        pass

    def merge_results(self, item, results: List):
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for row in results:
            for actor, res in row.items():
                if isinstance(res, list):
                    merge_result.extend(res)
                else:
                    merge_result.append(res)

        logger.info(merge_result)
        return merge_result

    def act(self, item, **kwargs):
        # todo
        # we will add process_series method in the future, now we only support the parallel method.
        results = {}
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None
        # Begin Group Pre-Process
        self.group_preprocess(**kwargs)

        logger.info(f"TreeActor 并行执行模式，最大工作线程数: {self.max_workers}")
        # Submit actor tasks to thread pool
        futures = {}
        rtn = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, actor in enumerate(self.actors):
                logger.info(f"提交第 {i + 1}/{len(self.actors)} 个 actor 到线程池: {actor.name}")
                actor.dataset = update_dataset(dataset, actor.dataset)
                futures[executor.submit(actor.act, item, **kwargs)] = actor

            completed_count = 0
            for future in as_completed(futures):
                actor = futures[future]
                completed_count += 1
                logger.info(f"并行执行进度: {completed_count}/{len(self.actors)} - Actor {actor.name} 完成")
                try:
                    res = future.result()
                    output_name = actor.output_name
                    logger.info(f"Actor {actor.name} 并行执行完成，输出名称: {output_name}")

                    rtn.append({actor.name: res})
                except Exception as e:
                    error_msg = f"Error occurred while executing actor '{actor.name}': {e}"
                    logger.error(error_msg)

        # Merge datasets in the main thread
        logger.info("开始合并数据集...")
        rtn = self.merge_results(item, rtn)
        actor.save_output(rtn, item)
        dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)

        results[output_name] = rtn
        self.dataset = dataset
        logger.info("数据集合并完成")

        # Begin Group Post-Process
        self.group_postprocess(**kwargs)

        if self.output_name == "TreeOutput":
            return results

        return rtn


class GenerateActorGroup(ActorGroup):
    OUTPUT_NAME = "pred_sql"
    STRATEGY = MergeStrategy.EXTEND.value


class ParseActorGroup(ActorGroup):
    def merge_results(self, item, results: List):
        # Merge results generated from distinct parser methods.
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for row in results:
            for parser, res in row.items():
                if parser == "RSLSQLBiDirParser":
                    merge_result.extend(res.get("columns", []))
                elif isinstance(res, list):
                    merge_result.extend(res)
                else:
                    merge_result.append(res)
        # remove the duplicate schemas
        merge_result = list(set(merge_result))

        return merge_result


class ScaleActorGroup(ActorGroup):
    OUTPUT_NAME = "pred_sql"
    STRATEGY = MergeStrategy.EXTEND.value


class OptimizeActorGroup(ActorGroup):
    def __init__(self, use_feedback_filter=None, **kwargs):
        super().__init__(**kwargs)
        self.use_feedback_filter: bool = False if use_feedback_filter is None else use_feedback_filter

    def get_execute_results(self, item, sql) -> bool:
        # This method provide a binary result for sql executable
        if not item or not sql:
            return False

        from core.db_connect import get_sql_exec_result

        row = self.dataset[item]
        db_type = row.get('db_type')
        db_id = row.get("db_id")
        db_path = Path(self.dataset.db_path) / (
                db_id + ".sqlite") if self.dataset.db_path and db_type == "sqlite" else None
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        debug_args = {
            "db_type": db_type,
            "sql_query": sql,
            "db_path": db_path,
            "db_id": db_id,
            "credential_path": credential.get(db_type) if credential else None
        }
        res = get_sql_exec_result(**debug_args)
        if not res:
            return False
        exe_flag, dbms_error_info = res
        if exe_flag is None:
            return False

        return True

    def merge_results(self, item, results: List):
        # Merge results generated from distinct optimizer methods.
        if not results:
            logger.info("Input results empty!")
        logger.info(results)
        merge_result = []
        for row in results:
            for actor, res in row.items():
                if isinstance(res, list):
                    merge_result.extend(res)
                else:
                    merge_result.append(res)

        if self.use_feedback_filter:
            # It seems that filtering here is not very meaningful. Maybe need to optimize.
            # So we set self.use_feedback_filter as False.
            filter_lis = []
            for row in merge_result:
                if self.get_execute_results(item, row):
                    filter_lis.append(row)
            merge_result = filter_lis

        return merge_result
