import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from loguru import logger

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
        logger.info(f"TreeActor 开始执行，并行模式: {self.open_actor_parallel}, 包含 {len(self.actors)} 个 actors")
        if self.open_actor_parallel:
            res = self.process_parallel(item, **kwargs)
        else:
            res = self.process_series(item, **kwargs)
        logger.info(f"TreeActor 执行完成")
        return res

    def process_series(self, item, **kwargs):
        results = {}

        dataset = self.dataset
        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        logger.info(f"TreeActor 串行执行模式，开始处理 {len(self.actors)} 个 actors")
        for i, actor in enumerate(self.actors):
            logger.info(f"串行执行第 {i + 1}/{len(self.actors)} 个 actor: {actor.name}")
            actor.dataset = update_dataset(dataset, actor.dataset)
            try:
                res = actor.act(item, **kwargs)
                output_name = actor.output_name
                logger.info(f"Actor {actor.name} 串行执行完成，输出名称: {output_name}")
                if output_name == "TreeOutput" and isinstance(res, dict):
                    results.update(res)
                else:
                    if output_name in results:
                        if isinstance(results[output_name], list):
                            results[output_name] = [*results[output_name], str(res)]
                        else:
                            results[output_name] = [str(results[output_name]), str(res)]
                    else:
                        results[output_name] = res
            except Exception as e:
                error_msg = f"Error occurred while executing actor '{actor.name}': {e}"
                logger.error(error_msg)

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
                        if output_name in results:
                            if isinstance(results[output_name], list):
                                results[output_name] = [*results[output_name], str(res)]
                            else:
                                results[output_name] = [str(results[output_name]), str(res)]
                        else:
                            results[output_name] = res

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


def actor_group_by(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return res

    return wrapper


class ActorGroup(TreeActor):
    def merge_results(self, results: List):
        pass

    def act(self, item, **kwargs):
        # todo
        # we will add process_series method in the future, now we only support the parallel method.
        results = {}
        dataset = self.dataset

        if not dataset or not self.actors:
            warnings.warn("Both 'dataset' and 'actors' must be provided.", category=UserWarning)
            return None

        logger.info(f"TreeActor 并行执行模式，最大工作线程数: {self.max_workers}")
        # Submit actor tasks to thread pool
        futures = {}
        rtn = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, actor in enumerate(self.actors):
                logger.info(f"提交第 {i + 1}/{len(self.actors)} 个 actor 到线程池: {actor.name}")
                actor.dataset = update_dataset(dataset, actor.dataset)
                futures[executor.submit(actor.act, item, update_dataset=False, **kwargs)] = actor

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
        rtn = self.merge_results(rtn)
        actor.save_output(rtn, item)
        dataset = update_dataset(dataset, actor.dataset, merge_dataset=True)

        results[output_name] = rtn
        self.dataset = dataset
        logger.info("数据集合并完成")

        return results


class ParseActorGroup(ActorGroup):
    def merge_results(self, results: List):
        # Merge results generated from distinct parser methods.
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for item in results:
            for parser, res in item.items():
                if parser == "RSLSQLBiDirParser":
                    merge_result.extend(res.get("columns", []))
                elif isinstance(res, list):
                    merge_result.extend(item)
                else:
                    merge_result.append(item)

        return merge_result


class ScaleActorGroup(ActorGroup):
    def merge_results(self, results: List):
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for row in results:
            if not isinstance(row, List):
                raise TypeError(f"Each row must be a list, but got {type(row)}: {row}")

            merge_result.extend(row)

        return merge_result


class OptimizeActorGroup(ActorGroup):
    def merge_results(self, results: List):
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for row in results:
            if not isinstance(row, List):
                raise TypeError(f"Each row must be a list, but got {type(row)}: {row}")

            merge_result.extend(row)

        return merge_result
