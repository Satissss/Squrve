import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.engine import Engine
from reproduce.eval_utils import load_router, evaluate


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
    main("spider", "dinsql")
