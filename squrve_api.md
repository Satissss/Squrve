# Squrve API 详细文档

## 项目概述

Squrve 是一个支持快速开发"端到端" Text-to-SQL 模型或对现有模型进行快速评估的工具。Text-to-SQL 任务流封装了多种组件，支持模式降维（schema reduce）、模式链接(schema linking)、查询生成（query generation）的快速实现。仅依靠配置文件即可快速启动 Squrve，并可轻松实现复杂任务嵌套、多任务并发执行。

## 核心架构

### 1. Router - 配置管理器

Router 是 Squrve 的核心配置管理器，负责管理 Text-to-SQL 的全部处理流程的参数配置。

#### 初始化方式

```python
from core.base import Router

# 方式1: 通过配置文件初始化
router = Router(config_path="config/demo_config.json")

# 方式2: 通过参数初始化
router = Router(
    use="qwen",
    model_name="qwen-turbo",
    data_source="spider:dev",
    schema_source="spider:dev",
    task_meta=[{
        "task_id": "demo_task",
        "task_type": "generate",
        "data_source": "spider:dev",
        "schema_source": "spider:dev"
    }]
)
```

#### 主要配置参数

**LLM 配置**
- `use`: LLM 提供商 ("qwen", "deepseek", "zhipu")
- `model_name`: 模型名称
- `context_window`: 上下文窗口大小
- `max_token`: 最大输出token数
- `temperature`: 温度参数
- `top_p`: top-p 采样参数
- `time_out`: 超时时间

**文本嵌入配置**
- `embed_model_source`: 嵌入模型来源 ("HuggingFace")
- `embed_model_name`: 嵌入模型名称

**数据集配置**
- `data_source`: 数据源路径或标识符
- `data_source_dir`: 数据源存储目录
- `need_few_shot`: 是否需要添加 few-shot 示例
- `few_shot_num`: few-shot 示例数量
- `need_external`: 是否需要添加外部知识
- `db_path`: 数据库路径

**数据库配置**
- `schema_source`: 数据库模式源
- `multi_database`: 是否多数据库模式
- `vector_store`: 向量存储路径
- `need_build_index`: 是否需要构建索引

**任务配置**
- `task_meta`: 任务元数据列表
- `cpx_task_meta`: 复杂任务元数据
- `exec_process`: 执行流程定义

### 2. DataLoader - 数据管理器

DataLoader 负责管理多个数据集和数据库模式，提供数据预处理、few-shot 学习、外部知识集成等功能。

#### 初始化

```python
from core.data_manage import DataLoader
from core.base import Router

router = Router(config_path="config/demo_config.json")
dataloader = DataLoader(router)
```

#### 主要方法

**数据集生成**
```python
# 生成数据集
dataset = dataloader.generate_dataset(
    data_source_index="spider_dev",
    schema_source_index="spider_dev",
    random_size=0.1,  # 随机采样10%
    filter_by="has_label"  # 过滤条件
)
```

**添加 Few-shot 示例**
```python
# 添加 few-shot 示例
dataloader.add_few_shot(
    source_index="spider_dev",
    few_shot_num=3,
    few_shot_save_dir="files/reasoning_examples/user"
)
```

**添加外部知识**
```python
# 添加外部知识
dataloader.add_external(
    source_index="spider_dev",
    external_save_dir="files/external"
)
```

**构建向量索引**
```python
# 构建向量索引
dataloader.build_index(
    source_index="spider_dev",
    embed_model_name="BAAI/bge-large-en-v1.5"
)
```

### 3. Dataset - 数据集类

Dataset 封装了单个任务运行所需的数据集和数据库模式。

#### 初始化

```python
from core.data_manage import Dataset

dataset = Dataset(
    data_source="benchmarks/spider/dev/dataset.json",
    schema_source="benchmarks/spider/dev/schema.json",
    multi_database=False,
    vector_store="vector_store",
    embed_model_name="BAAI/bge-large-en-v1.5"
)
```

#### 主要属性

- `dataset_dict`: 数据集字典
- `schema_source`: 模式源路径
- `is_multi_database`: 是否多数据库
- `credential`: 数据库凭证
- `database_path`: 数据库路径

#### 主要方法

```python
# 获取向量索引
vector_index = dataset.get_vector_index(item=0)

# 获取数据库模式
schema = dataset.get_db_schema(item=0)

# 保存数据
dataset.save_data("output/dataset.json")
```

### 4. Engine - 执行引擎

Engine 是 Text-to-SQL 流程的执行引擎，负责创建并执行所有任务，收集并评估结果。

#### 初始化

```python
from core.engine import Engine
from core.base import Router

router = Router(config_path="config/demo_config.json")
engine = Engine(router)
```

#### 主要方法

**执行任务**
```python
# 执行所有任务
engine.execute()

# 跳过执行，仅评估
engine.skip_execute()

# 评估结果
engine.evaluate(force=True)
```

**任务管理**
```python
# 获取任务ID列表
task_ids = engine.task_ids

# 根据ID获取任务
task = engine.get_task_by_id("demo_task")
```

### 5. Actor 组件

Actor 是 Squrve 的核心处理组件，包括 Reducer、Parser 和 Generator。

#### 5.1 BaseReducer - 模式降维基类

```python
from core.actor.reducer.BaseReduce import BaseReducer

class CustomReducer(BaseReducer):
    OUTPUT_NAME = "instance_schemas"
    
    def act(self, item, schema=None, **kwargs):
        # 实现模式降维逻辑
        return reduced_schema
```

#### 5.2 BaseParser - 模式解析基类

```python
from core.actor.parser.BaseParse import BaseParser

class CustomParser(BaseParser):
    OUTPUT_NAME = "schema_links"
    
    def act(self, item, schema=None, **kwargs):
        # 实现模式解析逻辑
        return schema_links
```

#### 5.3 BaseGenerator - 查询生成基类

```python
from core.actor.generator.BaseGenerate import BaseGenerator

class CustomGenerator(BaseGenerator):
    OUTPUT_NAME = "pred_sql"
    
    def act(self, item, schema=None, schema_links=None, **kwargs):
        # 实现SQL生成逻辑
        return generated_sql
```

### 6. Task 任务系统

#### 6.1 MetaTask - 元任务基类

```python
from core.task.meta.MetaTask import MetaTask

class CustomTask(MetaTask):
    NAME = "CustomTask"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_actor(self, actor_type=None, **kwargs):
        # 加载对应的 Actor
        pass
    
    def run(self):
        # 执行任务逻辑
        pass
```

#### 6.2 GenerateTask - 生成任务

```python
from core.task.meta.GenerateTask import GenerateTask
from llama_index.core.llms.llm import LLM

# 创建LLM实例
llm = LLM()  # 根据具体LLM类型创建

# 创建生成任务
generate_task = GenerateTask(
    llm=llm,
    generate_type="LinkAlign",
    save_dir="files/pred_sql"
)
```

#### 6.3 ComplexTask - 复杂任务

```python
from core.task.meta.ComplexTask import ComplexTask

# 创建复杂任务
complex_task = ComplexTask(
    task_lis=["task1", "task2"],  # 任务列表
    meta_tasks={"task1": task1, "task2": task2},  # 元任务字典
    open_actor_parallel=True,  # 开启Actor并行
    max_workers=3
)
```

### 7. LLM 集成

Squrve 支持多种 LLM 提供商：

#### 7.1 QwenModel

```python
from core.llm.QwenModel import QwenModel

qwen_llm = QwenModel(
    api_key="your_api_key",
    model_name="qwen-turbo",
    context_window=120000,
    max_token=8000,
    temperature=0.75
)
```

#### 7.2 DeepseekModel

```python
from core.llm.DeepseekModel import DeepseekModel

deepseek_llm = DeepseekModel(
    api_key="your_api_key",
    model_name="deepseek-chat",
    context_window=120000,
    max_token=8000,
    temperature=0.75
)
```

#### 7.3 ZhipuModel

```python
from core.llm.ZhipuModel import ZhipuModel

zhipu_llm = ZhipuModel(
    api_key="your_api_key",
    model_name="glm-4",
    context_window=120000,
    max_token=8000,
    temperature=0.75
)
```

### 8. 配置文件格式

#### 8.1 完整配置文件示例

```json
{
  "api_key": {
    "qwen": "your_qwen_api_key_here",
    "deepseek": "your_deepseek_api_key_here",
    "zhipu": "your_zhipu_api_key_here"
  },
  "llm": {
    "use": "qwen",
    "model_name": "qwen-turbo",
    "context_window": 120000,
    "max_token": 8000,
    "top_p": 0.9,
    "temperature": 0.75,
    "time_out": 300.0
  },
  "text_embed": {
    "embed_model_source": "HuggingFace",
    "embed_model_name": "BAAI/bge-large-en-v1.5"
  },
  "dataset": {
    "data_source": "spider:dev",
    "data_source_dir": "files/data_source",
    "overwrite_exist_file": true,
    "need_few_shot": false,
    "few_shot_num": 3,
    "few_shot_save_dir": "files/reasoning_examples/user",
    "need_external": false,
    "external_save_dir": "files/external",
    "db_path": null
  },
  "database": {
    "skip_schema_init": false,
    "schema_source": "spider:dev",
    "multi_database": false,
    "vector_store": "vector_store",
    "schema_source_dir": "files/schema_source",
    "need_build_index": false
  },
  "reducer": {
    "reduce_type": "LinkAlign",
    "is_save_reduce": true,
    "reduce_save_dir": "files/instance_schemas",
    "reduce_output_format": "dataframe"
  },
  "parser": {
    "parse_type": "LinkAlign",
    "is_save_parse": true,
    "parse_save_dir": "files/schema_links",
    "parse_output_format": "dataframe"
  },
  "generator": {
    "generate_type": "LinkAlign",
    "is_save_generate": true,
    "generate_save_dir": "files/pred_sql"
  },
  "task": {
    "task_meta": [
      {
        "task_id": "spider_dev_task",
        "task_type": "generate",
        "data_source": "spider:dev",
        "schema_source": "spider:dev",
        "meta": {
          "dataset": {
            "random_size": null,
            "filter_by": null
          }
        }
      }
    ],
    "cpx_task_meta": [],
    "is_save_dataset": true,
    "open_parallel": true,
    "max_workers": 5
  },
  "engine": {
    "exec_process": ["spider_dev_task"]
  },
  "credential": {
    "big_query": "path/to/big_query_credential.json",
    "snowflake": "path/to/snowflake_credential.json"
  }
}
```

#### 8.2 任务元数据格式

**简单任务**
```json
{
  "task_id": "task_1",
  "task_name": "Spider Dev Generation",
  "task_info": "Generate SQL for Spider dev dataset",
  "task_type": "generate",
  "data_source": "spider:dev",
  "schema_source": "spider:dev",
  "eval_type": ["exact_match", "execution"],
  "meta": {
    "dataset": {
      "random_size": 0.1,
      "filter_by": "has_label"
    },
    "llm": {
      "use": "qwen",
      "model_name": "qwen-turbo"
    }
  }
}
```

**复杂任务**
```json
{
  "task_id": "complex_task",
  "task_name": "Complex Pipeline",
  "task_info": "Multi-step Text-to-SQL pipeline",
  "task_lis": ["reduce_task", "parse_task", "generate_task"],
  "eval_type": ["exact_match"],
  "open_actor_parallel": true,
  "max_workers": 3,
  "meta": {
    "actor": {
      "reduce_task": {
        "reduce_type": "LinkAlign"
      },
      "parse_task": {
        "parse_type": "LinkAlign"
      },
      "generate_task": {
        "generate_type": "LinkAlign"
      }
    }
  }
}
```

### 9. 数据格式规范

#### 9.1 数据集格式

```json
{
  "instance_id": "unique_identifier",
  "db_id": "database_name",
  "question": "自然语言问题",
  "db_type": "sqlite",
  "db_size": "medium",
  "query": "SELECT * FROM table",
  "gold_schemas": ["table1", "table2"],
  "schema_links": ["table1.column1", "table2.column2"],
  "external_path": "path/to/external/knowledge.txt",
  "external": "extracted knowledge content",
  "reasoning_examples": "path/to/reasoning/examples.txt",
  "instance_schemas": "path/to/instance/schemas.csv",
  "pred_sql": "predicted SQL query"
}
```

#### 9.2 数据库模式格式

**Central 格式**
```json
{
  "db_id": "database_name",
  "db_size": 100,
  "db_type": "sqlite",
  "table_names_original": ["table1", "table2"],
  "column_names_original": [["table1", "column1"], ["table2", "column2"]],
  "column_types": ["text", "integer"],
  "column_descriptions": ["description1", "description2"],
  "sample_rows": [["value1", "value2"]]
}
```

**Parallel 格式**
```json
[
  {
    "db_id": "database_name",
    "table_name": "table1",
    "column_name": "column1",
    "column_types": "text",
    "column_descriptions": "description",
    "sample_rows": ["value1", "value2"]
  }
]
```

### 10. 使用示例

#### 10.1 基本使用流程

```python
from core.base import Router
from core.engine import Engine

# 1. 创建配置管理器
router = Router(config_path="config/spider_dev_config.json")

# 2. 创建执行引擎
engine = Engine(router)

# 3. 执行任务
engine.execute()

# 4. 评估结果
engine.evaluate()
```

#### 10.2 自定义任务

```python
from core.task.meta.MetaTask import MetaTask
from core.actor.generator.LinkAlignGenerate import LinkAlignGenerator

class CustomGenerateTask(MetaTask):
    NAME = "CustomGenerateTask"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_actor(self, actor_type=None, **kwargs):
        return LinkAlignGenerator(
            dataset=self.dataset,
            llm=self.llm,
            **kwargs
        )
    
    def run(self):
        # 自定义执行逻辑
        pass
```

#### 10.3 多任务并行执行

```python
from core.task.multi.ParallelTask import ParallelTask

# 创建并行任务
parallel_task = ParallelTask(
    tasks=[task1, task2, task3],
    open_parallel=True,
    max_workers=3
)

# 执行并行任务
parallel_task.run()
```

### 11. 评估系统

Squrve 支持多种评估指标：

- `exact_match`: 精确匹配
- `execution`: 执行正确性
- `reduce_recall`: 模式降维召回率
- `parse_accuracy`: 模式解析准确率

### 12. 错误处理

Squrve 提供了完善的错误处理机制：

```python
import warnings

# 检查任务ID重复
if task_id in existing_task_ids:
    warnings.warn(f"Task ID {task_id} already exists")

# 检查任务类型
if task_type not in registered_task_types:
    warnings.warn(f"Task type {task_type} not registered")

# 检查数据源
if not data_source_exists:
    warnings.warn(f"Data source {data_source} not found")
```

### 13. 日志系统

Squrve 使用内置的日志系统记录执行过程：

```python
from core.log import Logger

logger = Logger(save_path="files/logs/execution.log")
logger.info("Task started")
logger.error("Error occurred")
logger.debug("Debug information")
```

这个详细的 API 文档涵盖了 Squrve 项目的所有重要组件、类、方法的使用参数和具体示例，为用户提供了完整的使用指南。 