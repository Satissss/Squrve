from core.base import Router
from core.engine import Engine

if __name__ == "__main__":
    router = Router(config_path="spider_dev_config.json")

    engine = Engine(router)

    # 执行任务
    print("📋 执行自定义任务中...")
    engine.execute()

    # 评估结果
    print("📊 评估结果中...")
    engine.evaluate()

    print("✅ 自定义任务完成!")
