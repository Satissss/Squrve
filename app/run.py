import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from typing import Any, Dict, List, Tuple
import copy
from flask import Flask, request, jsonify
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.base import Router
from core.engine import Engine

# 尝试导入func_timeout，如果没有则提供警告
try:
    from func_timeout import func_timeout, FunctionTimedOut

    HAS_FUNC_TIMEOUT = True
except ImportError:
    HAS_FUNC_TIMEOUT = False
    logger.warning(
        "func_timeout not installed. Timeout control will be disabled. Install with: pip install func-timeout")

from app.evaluation_helper import evaluate_sql_execution, calculate_score_from_evaluation

CONFIG_DIR = "app_config.json"

# 超时常量定义
TASK_MAX_WAIT_TIME = 300  # 5分钟 = 300秒，任务执行最大等待时间
SQL_MAX_WAIT_TIME = 60  # 60秒，SQL评估最大等待时间

# Global router & engine
# We assume the data_source and schema_source is pre_defined in config file, and support by our system benchmarks
# This facilitate the dynamic data loading in the running process
router = Router(config_path=CONFIG_DIR)

# The initialization of the engine
engine = Engine(router)
dataloader = engine.dataloader
# We assume the data_source and schema_source is fixed, because all samples are from the same datasets.
data_source = dataloader.get_data_source_index()
schema_source = dataloader.get_schema_source_index()[0]

dataset = dataloader.generate_dataset(data_source, schema_source)
dataset.db_path = "../benchmarks/spider/database"

app = Flask(__name__)


def parse_task_from_id(task_id: str):
    if not task_id:
        return None
    if task_id.endswith("Generator"):
        return "GenerateTask"
    elif task_id.endswith("Decomposer"):
        return "DecomposeTask"
    elif task_id.endswith("Optimizer"):
        return "OptimizeTask"
    elif task_id.endswith("Parser"):
        return "ParseTask"
    elif task_id.endswith("Reducer"):
        return "ReduceTask"
    elif task_id.endswith("Scaler"):
        return "ScaleTask"
    elif task_id.endswith("Selector"):
        return "SelectTask"
    else:
        return None


def task_type_mapping(task_type: str):
    if task_type == "GenerateTask":
        return "generate_type"
    elif task_type == "DecomposeTask":
        return "decompose_type"
    elif task_type == "OptimizeTask":
        return "optimize_type"
    elif task_type == "ParseTask":
        return "parse_type"
    elif task_type == "ReduceTask":
        return "reduce_type"
    elif task_type == "ScaleTask":
        return "scale_type"
    elif task_type == "SelectTask":
        return "select_type"
    else:
        return None


def format_task_id(task_id, instance_id):
    return f"{task_id}_{instance_id}"


def init_tasks(task_id: str, instance_id: str, **kwargs):
    # 首先判断任务是否存在，并统一返回 (task, formatted_id)
    formatted_id = format_task_id(task_id, instance_id)
    if task_id in engine.task_ids or formatted_id in engine.task_ids:
        existing = engine.tasks.get(formatted_id) or engine.tasks.get(task_id)
        return existing, formatted_id

    # Task Type
    task_type = parse_task_from_id(task_id)
    flag, task_type = engine.check_task_type(0, task_type)
    if not flag:
        return None, formatted_id

    if dataloader is None:
        return None, formatted_id

    cpy_dataset = copy.deepcopy(dataset)
    sub_data = [row for row in cpy_dataset._dataset if row["instance_id"] == instance_id]
    if not sub_data:
        return None, formatted_id
    cpy_dataset._dataset = sub_data
    # generate task
    generate_args = {
        "task_id": formatted_id,
        "dataset": cpy_dataset,
        "task_type": task_type,
        "open_parallel": True,
        "max_workers": 1,
        "llm_args": kwargs.get("llm", {}),
        "actor_args": kwargs.get("actor", {})
    }
    task_label = task_type_mapping(task_type)
    if task_label:
        generate_args.update({task_label: task_id})

    task = engine.generate_task(**generate_args)

    return task, formatted_id


def parse_task_lis_from_origin(task_lis: List):
    rtn_lis = []
    for item in task_lis:
        if isinstance(item, str):
            rtn_lis.append(item)
        elif isinstance(item, (list, tuple)):
            rtn_lis.extend(list(item))
    return rtn_lis


def init_complex_tasks(task_list: List[str], instance_id: str):
    from datetime import datetime

    # 初始化所有 Meta Tasks
    new_task_lis = parse_task_lis_from_origin(task_list)
    new_task_lis = [init_tasks(id_, instance_id) for id_ in new_task_lis]
    all_tasks = {id_: task for task, id_ in new_task_lis}

    for ind, row in enumerate(task_list):
        if isinstance(row, str):
            task_list[ind] = format_task_id(row, instance_id)
        elif isinstance(row, List):
            for iind, item in enumerate(row):
                row[iind] = format_task_id(item, instance_id)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cpx_task_id = f"{data_source}_{instance_id}_{timestamp}"
    cpx_task_args = {
        "task_id": cpx_task_id,
        "task_lis": task_list,
        "meta_tasks": all_tasks,
        "eval_type": ["execute_accuracy"],
        "is_save_dataset": False,
    }
    cpx_task = engine.generate_complex_task(**cpx_task_args)

    all_tasks.update({
        cpx_task_id: cpx_task
    })
    engine.tasks.update(all_tasks)

    return cpx_task, cpx_task_id


@app.post("/api/run")
def run_complex_actor():
    # It works for single requests, but may have issues in concurrent scenarios.
    payload = request.get_json(force=True, silent=True) or {}
    instance_id = payload.get("instance_id") or request.args.get("instance_id")
    task_lis = payload.get("task_lis")
    logger.info(f"Payload parameter parsing: instance_id {instance_id}, task_lis {task_lis}")

    if not instance_id or not isinstance(task_lis, list):
        return jsonify({"error": "instance_id and task_lis are required"}), 400

    logger.info(f"Begin to initialize complex tasks")
    cpx_task, cpx_task_id = init_complex_tasks(task_lis, instance_id)
    logger.info(f"Finish to initialize complex tasks")

    engine.exec_process = [cpx_task_id]

    started = time.perf_counter()
    try:
        engine.execute()
    finally:
        duration = time.perf_counter() - started
        engine.tasks.pop(cpx_task_id, None)

    eval_res = cpx_task.eval(force=True) or {}
    execute_accuracy = eval_res.get("execute_accuracy", {}).get("avg")

    return jsonify({
        "duration_seconds": duration,
        "execute_accuracy": execute_accuracy
    })


@app.post("/api/run_batch")
def run_batch():
    """
    批量执行任务并返回评分
    
    评分规则:
    1. 任务执行阶段（TASK_MAX_WAIT_TIME=5分钟）:
       - 超时: score = -0.5，跳过后续所有评估
       - 在时间内完成: score += 0.5
    
    2. SQL评估阶段（SQL_MAX_WAIT_TIME=60秒）:
       - 超时或无法评估: score -= 1，跳过后续所有评估
       - SQL可执行（无语法错误）: score += 1，继续下一阶段
       - SQL不可执行（有语法错误）: score -= 1，跳过后续所有评估
    
    3. SQL正确性阶段（仅当SQL可执行时才评估）:
       - 查询结果正确: score += 1.5
       - 查询结果错误: score -= 1.5
    
    Example of payload:
    {
        "val_0": [[DINSQLGenerator],[DINSQLGenerator],[DINSQLGenerator],[MACSQLGenerator]],
        "val_1": [[DINSQLGenerator],[MACSQLGenerator]],
    }
    return:
    {
        "val_0": [3.0, 3.0, 3.0, 0],
        "val_1": [-0.5, 0.5]
    }
    """
    payload = request.get_json(force=True, silent=True) or {}
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "instance_id and task_lis are required"}), 400

    if not HAS_FUNC_TIMEOUT:
        logger.warning("func_timeout not available, timeout control disabled")

    def normalize_task_signature(task_lis: List[Any]) -> Tuple[Any, ...]:
        """Convert task_lis into an immutable, comparable structure."""

        def _normalize(item: Any):
            if isinstance(item, str):
                return ("str", item)
            if isinstance(item, (list, tuple)):
                return ("list", tuple(_normalize(sub_item) for sub_item in item))
            try:
                return ("val", json.dumps(item, ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError):
                return ("val", repr(item))

        return tuple(_normalize(entry) for entry in task_lis)

    # 构建执行计划
    instance_plan: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {}
    signature_to_task: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
    exec_plan: List[str] = []

    def cleanup_tasks(restore_callbacks: bool = False):
        """清理创建的任务"""
        for info in signature_to_task.values():
            if restore_callbacks and "original_callback" in info:
                if original_cb := info["original_callback"]:
                    info["task"].call_back = original_cb
            if cpx_id := info.get("cpx_task_id"):
                engine.tasks.pop(cpx_id, None)

    # 解析payload并初始化任务
    for instance_id, task_group in payload.items():
        if not isinstance(task_group, list):
            cleanup_tasks()
            return jsonify({"error": f"task_lis for `{instance_id}` must be a list"}), 400

        instance_plan[instance_id] = []
        for index, task_lis in enumerate(task_group):
            if not isinstance(task_lis, list) or not task_lis:
                cleanup_tasks()
                return jsonify(
                    {"error": f"task_lis at position {index} for `{instance_id}` must be a non-empty list"}
                ), 400

            signature = (instance_id, normalize_task_signature(task_lis))
            instance_plan[instance_id].append(signature)

            if signature in signature_to_task:
                continue

            cpx_task, cpx_task_id = init_complex_tasks(copy.deepcopy(task_lis), instance_id)
            if cpx_task is None:
                engine.tasks.pop(cpx_task_id, None)
                cleanup_tasks()
                return jsonify({"error": f"failed to initialize task_lis `{index}` for `{instance_id}`"}), 400

            signature_to_task[signature] = {
                "task": cpx_task,
                "cpx_task_id": cpx_task_id,
            }
            exec_plan.append(cpx_task_id)

    if not exec_plan:
        return jsonify({instance_id: [] for instance_id in instance_plan})

    logger.info(f"run_batch received {len(payload)} instances, executing {len(exec_plan)} unique task groups.")

    # 记录每个任务的执行状态
    task_execution_status: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}

    def execute_single_task(cpx_task):
        """执行单个任务"""
        cpx_task.run()
        cpx_task.end()

    def evaluate_sql(cpx_task):
        """评估SQL并返回结果"""
        if not cpx_task.dataset or len(cpx_task.dataset) == 0:
            return None, "No dataset available for evaluation"

        row = cpx_task.dataset[0]
        eval_result = evaluate_sql_execution(
            pred_sql=row.get("pred_sql", ""),
            gold_sql=row.get("query", ""),
            db_id=row.get("db_id", ""),
            db_type=row.get("db_type", "sqlite"),
            db_path=dataset.db_path,
            db_credential=dataset.credential
        )
        return eval_result, None

    # 并行执行任务并评分（每个任务独立超时控制）
    result_cache: Dict[Tuple[str, Tuple[Any, ...]], float] = {}
    result_cache_lock = threading.Lock()

    def process_single_signature(signature, info):
        """处理单个signature的任务执行和评估"""
        cpx_task = info["task"]
        cpx_task_id = info["cpx_task_id"]
        score = 0.0

        # 1. 执行任务（带超时控制）
        task_timeout = False
        try:
            if HAS_FUNC_TIMEOUT:
                try:
                    func_timeout(TASK_MAX_WAIT_TIME, execute_single_task, args=(cpx_task,))
                except FunctionTimedOut:
                    task_timeout = True
                    logger.warning(f"Task {cpx_task_id} exceeded {TASK_MAX_WAIT_TIME}s timeout")
            else:
                execute_single_task(cpx_task)
        except Exception as exc:
            logger.exception(f"Task {cpx_task_id} execution failed: {exc}")
            task_timeout = True  # 执行失败也当作超时处理

        # 2. 任务超时，分数为-0.5，跳过评估
        if task_timeout:
            with result_cache_lock:
                result_cache[signature] = -0.5
            return

        # 3. 任务完成，基础分 +0.5
        score = 0.5

        # 4. SQL评估（带超时控制）
        eval_timeout = False
        eval_result = None
        eval_error = None

        try:
            if HAS_FUNC_TIMEOUT:
                try:
                    eval_result, eval_error = func_timeout(SQL_MAX_WAIT_TIME, evaluate_sql, args=(cpx_task,))
                except FunctionTimedOut:
                    eval_timeout = True
                    logger.warning(f"SQL evaluation for task {cpx_task_id} exceeded {SQL_MAX_WAIT_TIME}s timeout")
            else:
                eval_result, eval_error = evaluate_sql(cpx_task)
        except Exception as exc:
            eval_error = str(exc)
            logger.exception(f"SQL evaluation error for task {cpx_task_id}: {exc}")

        # 5. 评估超时或失败，score -= 1
        if eval_timeout or eval_error or eval_result is None:
            with result_cache_lock:
                result_cache[signature] = score - 1
            return

        # 6. SQL不可执行，score -= 1
        if not eval_result.can_execute:
            with result_cache_lock:
                result_cache[signature] = score - 1
            return

        # 7. SQL可执行，score += 1
        score += 1

        # 8. 正确性评估
        score += 1.5 if eval_result.is_correct else -1.5
        with result_cache_lock:
            result_cache[signature] = score

    # 使用线程池并行执行所有任务
    max_workers = min(len(signature_to_task), os.cpu_count() or 4)
    logger.info(f"Starting parallel execution with {max_workers} workers for {len(signature_to_task)} tasks")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_signature, signature, info): signature
            for signature, info in signature_to_task.items()
        }
        
        # 等待所有任务完成
        for future in as_completed(futures):
            signature = futures[future]
            try:
                future.result()  # 获取结果（如果有异常会在这里抛出）
            except Exception as exc:
                logger.exception(f"Unexpected error processing signature {signature}: {exc}")
                # 如果发生意外错误，确保该signature有个默认分数
                with result_cache_lock:
                    if signature not in result_cache:
                        result_cache[signature] = -0.5

    # 清理任务
    try:
        cleanup_tasks()
    except Exception as exc:
        logger.warning(f"Error during cleanup: {exc}")

    # 构建最终响应
    return jsonify({
        instance_id: [result_cache.get(sig, -0.5) for sig in signatures]
        for instance_id, signatures in instance_plan.items()
    })


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
