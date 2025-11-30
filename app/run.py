import os
import sys
import json
import time
from loguru import logger
from typing import Any, Dict, List, Tuple
import copy
from flask import Flask, request, jsonify

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.base import Router
from core.engine import Engine

CONFIG_DIR = "app_config.json"

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
    Example of payload:
    {
        "val_0": [[DINSQLGenerator],[DINSQLGenerator],[DINSQLGenerator],[MACSQLGenerator]],
        "val_1": [[DINSQLGenerator],[MACSQLGenerator],[MACSQLGenerator], [RSLSQLGenerator]],
    }
    return:
    {
        "val_0": [{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx}],
        "val_1": [{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx}],
    }]
    }
    """
    payload = request.get_json(force=True, silent=True) or {}
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "instance_id and task_lis are required"}), 400

    def normalize_task_signature(task_lis: List[Any]) -> Tuple[Any, ...]:
        """Convert task_lis into an immutable, comparable structure."""

        def _normalize(item: Any):
            if isinstance(item, str):
                return ("str", item)
            if isinstance(item, (list, tuple)):
                return ("list", tuple(_normalize(sub_item) for sub_item in item))
            try:
                serialized = json.dumps(item, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                serialized = repr(item)
            return ("val", serialized)

        return tuple(_normalize(entry) for entry in task_lis)

    instance_plan: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {}
    signature_to_task: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
    exec_plan: List[str] = []

    def cleanup_created_tasks(restore_callbacks: bool = False):
        for info in signature_to_task.values():
            if restore_callbacks and "original_callback" in info:
                original_cb = info["original_callback"]
                if original_cb is not None:
                    info["task"].call_back = original_cb
            cpx_id = info.get("cpx_task_id")
            if cpx_id:
                engine.tasks.pop(cpx_id, None)

    for instance_id, task_group in payload.items():
        if not isinstance(task_group, list):
            cleanup_created_tasks()
            return jsonify({"error": f"task_lis for `{instance_id}` must be a list"}), 400

        instance_plan[instance_id] = []
        for index, task_lis in enumerate(task_group):
            if not isinstance(task_lis, list) or not task_lis:
                cleanup_created_tasks()
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
                cleanup_created_tasks()
                return jsonify({"error": f"failed to initialize task_lis `{index}` for `{instance_id}`"}), 400

            signature_to_task[signature] = {
                "task": cpx_task,
                "cpx_task_id": cpx_task_id,
            }
            exec_plan.append(cpx_task_id)

    if not exec_plan:
        return jsonify({instance_id: [] for instance_id in instance_plan})

    duration_recorder: Dict[str, float] = {}

    def make_callback(task_id: str, original_cb):
        def _callback(task, *cb_args, **cb_kwargs):
            run_time = cb_kwargs.get("run_time")
            if run_time is not None:
                duration_recorder[task_id] = run_time
            if original_cb:
                return original_cb(task, *cb_args, **cb_kwargs)
            return None

        return _callback

    for info in signature_to_task.values():
        task_obj = info["task"]
        original_cb = getattr(task_obj, "call_back", None)
        info["original_callback"] = original_cb
        task_obj.call_back = make_callback(info["cpx_task_id"], original_cb)

    engine.exec_process = exec_plan
    execution_error = None
    try:
        logger.info(f"run_batch received {len(payload)} instances, executing {len(exec_plan)} unique task groups.")
        engine.execute()
    except Exception as exc:
        execution_error = exc
        logger.exception("Batch execution failed: {}", exc)
    finally:
        cleanup_created_tasks(restore_callbacks=True)

    if execution_error:
        return jsonify({"error": "batch execution failed", "detail": str(execution_error)}), 500

    result_cache: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
    for signature, info in signature_to_task.items():
        cpx_task = info["task"]
        eval_res = cpx_task.eval(force=True) or {}
        execute_accuracy = None
        exec_metric = eval_res.get("execute_accuracy")
        if isinstance(exec_metric, dict):
            execute_accuracy = exec_metric.get("avg")

        duration_val = duration_recorder.get(info["cpx_task_id"])
        try:
            duration_seconds = float(duration_val) if duration_val is not None else 0.0
        except (TypeError, ValueError):
            duration_seconds = 0.0

        result_cache[signature] = {
            "duration_seconds": duration_seconds,
            "execute_accuracy": execute_accuracy
        }

    final_response: Dict[str, List[Dict[str, Any]]] = {}
    for instance_id, signatures in instance_plan.items():
        results = []
        for signature in signatures:
            result = result_cache.get(signature)
            if result is None:
                results.append({"duration_seconds": None, "execute_accuracy": None})
            else:
                results.append(dict(result))
        final_response[instance_id] = results

    return jsonify(final_response)


def run_batch_debug(payload):
    """
    Example of payload:
    {
        "val_0": [[DINSQLGenerator],[DINSQLGenerator],[DINSQLGenerator],[MACSQLGenerator]],
        "val_1": [[DINSQLGenerator],[MACSQLGenerator],[MACSQLGenerator], [RSLSQLGenerator]],
    }
    return:
    {
        "val_0": [{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx}],
        "val_1": [{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx},{"duration_seconds": xx ,"execute_accuracy": xx}],
    }]
    }
    """
    payload = payload or {}
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "instance_id and task_lis are required"}), 400

    def normalize_task_signature(task_lis: List[Any]) -> Tuple[Any, ...]:
        """Convert task_lis into an immutable, comparable structure."""

        def _normalize(item: Any):
            if isinstance(item, str):
                return ("str", item)
            if isinstance(item, (list, tuple)):
                return ("list", tuple(_normalize(sub_item) for sub_item in item))
            try:
                serialized = json.dumps(item, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                serialized = repr(item)
            return ("val", serialized)

        return tuple(_normalize(entry) for entry in task_lis)

    instance_plan: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {}
    signature_to_task: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
    exec_plan: List[str] = []

    def cleanup_created_tasks(restore_callbacks: bool = False):
        for info in signature_to_task.values():
            if restore_callbacks and "original_callback" in info:
                original_cb = info["original_callback"]
                if original_cb is not None:
                    info["task"].call_back = original_cb
            cpx_id = info.get("cpx_task_id")
            if cpx_id:
                engine.tasks.pop(cpx_id, None)

    for instance_id, task_group in payload.items():
        if not isinstance(task_group, list):
            cleanup_created_tasks()
            return jsonify({"error": f"task_lis for `{instance_id}` must be a list"}), 400

        instance_plan[instance_id] = []
        for index, task_lis in enumerate(task_group):
            if not isinstance(task_lis, list) or not task_lis:
                cleanup_created_tasks()
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
                cleanup_created_tasks()
                return jsonify({"error": f"failed to initialize task_lis `{index}` for `{instance_id}`"}), 400

            signature_to_task[signature] = {
                "task": cpx_task,
                "cpx_task_id": cpx_task_id,
            }
            exec_plan.append(cpx_task_id)

    if not exec_plan:
        return jsonify({instance_id: [] for instance_id in instance_plan})

    duration_recorder: Dict[str, float] = {}

    def make_callback(task_id: str, original_cb):
        def _callback(task, *cb_args, **cb_kwargs):
            run_time = cb_kwargs.get("run_time")
            if run_time is not None:
                duration_recorder[task_id] = run_time
            if original_cb:
                return original_cb(task, *cb_args, **cb_kwargs)
            return None

        return _callback

    for info in signature_to_task.values():
        task_obj = info["task"]
        original_cb = getattr(task_obj, "call_back", None)
        info["original_callback"] = original_cb
        task_obj.call_back = make_callback(info["cpx_task_id"], original_cb)

    exec_plan.append("~p")  # add `~p` label to enable parallel.
    engine.exec_process = exec_plan
    execution_error = None
    try:
        logger.info(f"run_batch received {len(payload)} instances, executing {len(exec_plan)} unique task groups.")
        engine.execute()
    except Exception as exc:
        execution_error = exc
        logger.exception("Batch execution failed: {}", exc)
    finally:
        cleanup_created_tasks(restore_callbacks=True)

    if execution_error:
        return jsonify({"error": "batch execution failed", "detail": str(execution_error)}), 500

    result_cache: Dict[Tuple[str, Tuple[Any, ...]], Dict[str, Any]] = {}
    for signature, info in signature_to_task.items():
        cpx_task = info["task"]
        eval_res = cpx_task.eval(force=True) or {}
        execute_accuracy = None
        exec_metric = eval_res.get("execute_accuracy")
        if isinstance(exec_metric, dict):
            execute_accuracy = exec_metric.get("avg")

        duration_val = duration_recorder.get(info["cpx_task_id"])
        try:
            duration_seconds = float(duration_val) if duration_val is not None else 0.0
        except (TypeError, ValueError):
            duration_seconds = 0.0

        result_cache[signature] = {
            "duration_seconds": duration_seconds,
            "execute_accuracy": execute_accuracy
        }

    final_response: Dict[str, List[Dict[str, Any]]] = {}
    for instance_id, signatures in instance_plan.items():
        results = []
        for signature in signatures:
            result = result_cache.get(signature)
            if result is None:
                results.append({"duration_seconds": None, "execute_accuracy": None})
            else:
                results.append(dict(result))
        final_response[instance_id] = results

    print(final_response)
    return jsonify(final_response)


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
