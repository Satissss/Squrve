import copy
from typing import Union
from core.base import Router
from core.engine import Engine
from core.evaluate import Evaluator
from core.utils import load_dataset
from pathlib import Path
from loguru import logger

# For default config, we use `generate` for task id
GENERATE_TASK_ID = "generate"
USE_PARALLEL = True


def load_router(config_path: str, identifier: str = None) -> dict:
    original_config = load_dataset(config_path)

    dataset_save_dir = original_config.pop("dataset_save_dir")
    sql_save_dir = original_config.pop("sql_save_dir")
    n = original_config.pop("generate_num")

    # make sure the dir exists
    Path(dataset_save_dir).mkdir(parents=True, exist_ok=True)
    Path(sql_save_dir).mkdir(parents=True, exist_ok=True)

    Router._sys_config_path = "../config/sys_config.json"
    router = Router()
    router.init_config(original_config)
    tasks = router.task_meta
    meta_task = [task for task in tasks if task['task_id'] == GENERATE_TASK_ID]
    if len(meta_task) == 0:
        return None
    meta_task = meta_task[0]
    task_lis, save_lis = init_task_meta(meta_task, n, dataset_save_dir, sql_save_dir, identifier)
    router._task_meta = task_lis
    router._exec_process = [task['task_id'] for task in router.task_meta]
    if USE_PARALLEL:
        router._exec_process.append("~p")

    return router, save_lis


def init_task_meta(meta_task, n, dataset_save_dir, sql_save_dir, identifier):
    task_id = meta_task['task_id']
    task_lis = []
    save_lis = []
    for ind in range(n):
        new_task = copy.deepcopy(meta_task)
        new_task['task_id'] = task_id + str((ind + 1))
        if identifier is not None:
            save_path = dataset_save_dir + f"{identifier}/task_{ind + 1}.json"
            new_task['dataset_save_path'] = save_path
            new_task["meta"]["actor"]["save_dir"] = sql_save_dir + f"{identifier}/task_{ind + 1}"
            save_lis.append(save_path)
        task_lis.append(new_task)

    return task_lis, save_lis


def _calculate_final_score(
        dataset,
        data_lists,
        eval_type: str = "execute_accuracy"
) -> float:
    """
    Calculate final score across all iterations.

    Args:
        dataset: Base dataset object
        data_lists: List of datasets from different iterations
        eval_type: Type of evaluation to perform

    Returns:
        Final score as a float
    """
    valid_count = 0
    pass_count = 0

    for row in zip(*data_lists):
        data_row = list(row)
        sub_dataset = copy.deepcopy(dataset)
        sub_dataset._dataset = data_row

        evaluator = Evaluator(dataset=sub_dataset, eval_type=eval_type)
        results = evaluator.eval_all()

        for key, value in results.items():
            if value.get("valid_num", 0) > 0:
                valid_count += 1
            if value.get("avg", 0) != 0:
                pass_count += 1

    return pass_count / valid_count if valid_count > 0 else 0.0


def _load_dataset_from_engine(config_path: str):
    """
    Load dataset from engine's generate task.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dataset object or None if not found
    """
    router = Router(config_path=config_path)
    engine = Engine(router)

    for key, task in engine.tasks.items():
        if key == GENERATE_TASK_ID:
            return task.dataset

    return None


def evaluate(save_lis, config_path: str):
    data_lists = [load_dataset(path) for path in save_lis]
    dataset = _load_dataset_from_engine(config_path)
    final_score = _calculate_final_score(dataset, data_lists)

    logger.info(f"Final score: {final_score:.4f}")
    return final_score


def main(dataset_name, method):
    identifier = f"{dataset_name}-{method}"
    config_path = f"{identifier}.json"

    router, save_lis = load_router(config_path, identifier)

    engine = Engine(router)

    # 执行任务
    print("执行自定义任务中...")
    engine.execute()

    # 评估结果
    print("评估结果中...")
    evaluate(save_lis, config_path)


if __name__ == "__main__":
    # main("spider", "dinsql")
    evaluate([
        "../files/datasets/spider-macsql/task_1.json",
        "../files/datasets/spider-macsql/task_2.json",
        "../files/datasets/spider-macsql/task_3.json"
    ], "spider-macsql.json")