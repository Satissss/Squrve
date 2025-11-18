import os
import sys
import json
import time
from loguru import logger
from typing import List
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


def parse_task_lis_from_origin(task_lis: List[str]):
    rtn_lis = []
    for item in task_lis:
        if isinstance(item, str):
            rtn_lis.append(item)
        elif isinstance(item, (list, tuple)):
            rtn_lis.extend(item)
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
    payload = request.get_json(force=True, silent=True) or {}
    instance_id = payload.get("instance_id") or request.args.get("instance_id")
    task_lis = payload.get("task_lis")

    if not instance_id or not isinstance(task_lis, list):
        return jsonify({"error": "instance_id and task_lis are required"}), 400

    cpx_task, cpx_task_id = init_complex_tasks(task_lis, instance_id)
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


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
