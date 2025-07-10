#!/usr/bin/env python3
"""
Spider Dev 数据集运行示例

这个脚本演示如何使用 Squrve 框架运行 Spider dev 数据集的 Text-to-SQL 任务。

使用方法:
1. 确保已安装所有依赖
2. 配置 API 密钥
3. 运行脚本: python run_spider_dev.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.base import Router
from core.engine import Engine
from core.log import Logger


def setup_directories():
    """创建必要的目录结构"""
    directories = [
        "files/data_source",
        "files/schema_source", 
        "files/instance_schemas",
        "files/schema_links",
        "files/pred_sql",
        "files/reasoning_examples/user",
        "files/external",
        "files/logs",
        "files/datasets",
        "files/pipeline_output",
        "vector_store"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")


def check_api_keys(config_path):
    """检查 API 密钥配置"""
    import json
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_keys = config.get('api_key', {})
    missing_keys = []
    
    for provider, key in api_keys.items():
        if key == f"your_{provider}_api_key_here":
            missing_keys.append(provider)
    
    if missing_keys:
        print("⚠️  警告: 以下 API 密钥需要配置:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n请在 config/spider_dev_config.json 中配置相应的 API 密钥")
        return False
    
    return True


def run_simple_generation():
    """运行简单的 SQL 生成任务"""
    print("\n🚀 开始运行 Spider Dev SQL 生成任务...")
    
    # 创建配置管理器
    router = Router(config_path="config/spider_dev_config.json")
    
    # 创建执行引擎
    engine = Engine(router)
    
    # 执行任务
    print("📋 执行任务中...")
    engine.execute()
    
    # 评估结果
    print("📊 评估结果中...")
    engine.evaluate()
    
    print("✅ SQL 生成任务完成!")


def run_complete_pipeline():
    """运行完整的 Text-to-SQL 流水线"""
    print("\n🚀 开始运行 Spider Dev 完整流水线...")
    
    # 创建配置管理器
    router = Router(config_path="config/spider_dev_config.json")
    
    # 修改执行流程为完整流水线
    router._exec_process = ["spider_dev_pipeline"]
    
    # 创建执行引擎
    engine = Engine(router)
    
    # 执行任务
    print("📋 执行完整流水线中...")
    engine.execute()
    
    # 评估结果
    print("📊 评估结果中...")
    engine.evaluate()
    
    print("✅ 完整流水线任务完成!")


def run_with_custom_settings():
    """使用自定义设置运行"""
    print("\n🚀 开始运行自定义设置任务...")
    
    # 创建配置管理器
    router = Router(
        use="qwen",
        model_name="qwen-turbo",
        data_source="spider:dev",
        schema_source="spider:dev",
        task_meta=[{
            "task_id": "custom_spider_task",
            "task_name": "Custom Spider Task",
            "task_info": "Custom task with specific settings",
            "task_type": "generate",
            "data_source": "spider:dev",
            "schema_source": "spider:dev",
            "eval_type": ["exact_match"],
            "meta": {
                "dataset": {
                    "random_size": 0.1,  # 只使用10%的数据
                    "filter_by": "has_label"
                },
                "llm": {
                    "temperature": 0.5,  # 降低温度参数
                    "max_token": 4000
                }
            }
        }],
        exec_process=["custom_spider_task"]
    )
    
    # 创建执行引擎
    engine = Engine(router)
    
    # 执行任务
    print("📋 执行自定义任务中...")
    engine.execute()
    
    # 评估结果
    print("📊 评估结果中...")
    engine.evaluate()
    
    print("✅ 自定义任务完成!")


def main():
    """主函数"""
    print("=" * 60)
    print("🐛 Squrve Spider Dev 数据集运行示例")
    print("=" * 60)
    
    # 检查配置文件
    config_path = "config/spider_dev_config.json"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请确保 config/spider_dev_config.json 文件存在")
        return
    
    # 检查 API 密钥
    if not check_api_keys(config_path):
        print("\n请配置 API 密钥后重新运行")
        return
    
    # 创建目录结构
    print("\n📁 创建目录结构...")
    setup_directories()
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 简单 SQL 生成任务")
    print("2. 完整 Text-to-SQL 流水线")
    print("3. 自定义设置任务")
    print("4. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                run_simple_generation()
                break
            elif choice == "2":
                run_complete_pipeline()
                break
            elif choice == "3":
                run_with_custom_settings()
                break
            elif choice == "4":
                print("👋 退出程序")
                break
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            break
    
    print("\n" + "=" * 60)
    print("🎉 程序执行完成!")
    print("📂 结果文件保存在 files/ 目录下")
    print("📊 日志文件保存在 files/logs/ 目录下")
    print("=" * 60)


if __name__ == "__main__":
    main() 