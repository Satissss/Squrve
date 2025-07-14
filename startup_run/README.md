# Spider Dev 数据集使用指南

本指南介绍如何使用 Squrve 框架运行 Spider dev 数据集的 Text-to-SQL 任务。

## 📋 目录结构

```
Squrve/
├── config/
│   ├── spider_dev_config.json    # Spider Dev 专用配置文件
│   ├── demo_config.json          # 演示配置文件
│   └── sys_config.json           # 系统配置文件
├── run/
│   ├── run_spider_dev.py         # Spider Dev 运行脚本
├── benchmarks/
│   └── spider/
│       ├── dev/
│       │   ├── dataset.json      # Spider Dev 数据集
│       │   └── schema.json       # 数据库模式文件
│       └── database/             # SQLite 数据库文件
├── squrve_api.md                 # 详细 API 文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

编辑 `config/spider_dev_config.json` 文件，配置 API 密钥：

```json
{
  "api_key": {
    "qwen": "your_actual_qwen_api_key",
    "deepseek": "your_actual_deepseek_api_key",
    "zhipu": "your_actual_zhipu_api_key"
  }
}
```

### 3. 运行 Spider Dev 任务

#### 方式一：使用交互式脚本

```bash
python startup_run/run_spider_dev.py
```

脚本会提供以下选项：

- **简单 SQL 生成任务**: 直接生成 SQL 查询
- **完整 Text-to-SQL 流水线**: 执行完整的 Reduce -> Parse -> Generate 流程
- **自定义设置任务**: 使用自定义参数运行

#### 方式二：编程方式

```python
from core.base import Router
from core.engine import Engine

# 使用 Spider Dev 配置
router = Router(config_path="spider_dev_config.json")
engine = Engine(router)

# 执行任务
engine.execute()

# 评估结果
engine.evaluate()
```

## 📁 运行成功

### 控制台输出

样例测试运行完成后，输出评估结果和任务相关统计信息，如下所示：
![img.png](../assets/img.png)

运行完成后，结果文件将保存在以下目录：

### 文件夹输出

- `files/pred_sql/`: 生成的 SQL 查询
  ![img.png](../assets/pred_sql.png)

- `files/schema_links/`: 模式链接结果
- ![img.png](../assets/schema_linking.png)
- `files/datasets/`: 处理后的数据集
  ![img.png](../assets/final_dataset.png)

## 🎯 任务类型

### 1. 简单生成任务 (`generate`)

直接生成 SQL 查询，跳过模式降维和模式链接步骤。

### 2. 完整流水线任务

执行完整的 Text-to-SQL 流程：

1. **模式降维** (`reduce`): 根据问题筛选相关数据库模式
2. **模式链接** (`parse`): 解析问题中提到的表和字段
3. **查询生成** (`generate`): 生成最终的 SQL 查询

### 3. 复杂任务

支持任务嵌套和并行执行：

- **串行执行**: 任务按顺序执行
- **并行执行**: 多个任务同时执行
- **嵌套任务**: 任务内部包含子任务

## 📊 配置文件详解

### Startup 测试配置 (`config/spider_dev_config.json`)

该配置文件提供了 Spider dev 数据集测试样例，包含以下主要部分：

#### LLM 配置

```json
{
  "llm": {
    "use": "qwen",
    "model_name": "qwen-turbo",
    "context_window": 120000,
    "max_token": 8000,
    "temperature": 0.75
  }
}
```

* use: 使用 Base LLM 类型

#### 数据集配置

```json
{
  "dataset": {
    "data_source": "spider:dev",
    "db_path": "benchmarks/spider/database"
  }
}
```

* data_source: 实验数据集，"spider:dev" benchmark已注册，支持自动解析。
* db_path: 连接数据库路径，系统配置仅提供支持的数据库配置

#### 任务配置

```json
{
  "task": {
    "task_meta": [
      {
        "task_id": "spider_dev_generate",
        "task_type": "generate",
        "data_source": "spider:dev",
        "schema_source": "spider:dev"
      }
    ]
  }
}
```

