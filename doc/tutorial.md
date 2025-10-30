# Usage Tutorial

本文是一份可直接随项目发布的完整教程，旨在帮助你用“配置驱动”的方式快速启动 Squrve、复现基线、并在本地或远程数据库上完成端到端 Text-to-SQL 实验。

## 1. 环境准备

- Python 3.8+
- 安装依赖与基准数据：

```bash
git clone https://github.com/Satissss/Squrve.git
cd Squrve
pip install -r requirements.txt

# 解压内置基准数据（首次使用必须执行）
unzip benchmarks.zip -d .
```

如需使用 LinkAlign 相关能力，请参考其官方说明进行额外配置（见项目 `README.md` 中的链接说明）。

## 2. 快速上手（5 分钟）

1) 在 `startup_run/startup_config.json` 中填写 LLM 的 `api_key`。

2) 直接运行示例脚本：

```bash
python startup_run/run.py
```

默认示例使用 Spider-dev 的一个小切片（`db_size-10`）与 CHESS 生成器；你也可以把 `startup_run/run.py` 中的配置文件路径指向你自己的 JSON 配置文件。

运行完成后，关键输出位置：
- `files/pred_sql/`：模型生成的 SQL
- `files/schema_links/`：Schema Linking 结果（若开启）
- `files/datasets/`：任务运行时保存的中间数据
- `files/logs/`：运行日志

## 3. 配置文件总览

Squrve 通过一个 JSON 配置文件完成所有设置。最小可运行示例如下（改自 `startup_run/startup_config.json`）：

```json
{
  "api_key": {
    "deepseek": "your_api_key_here",
    "qwen": "your_api_key_here",
    "zhipu": "your_api_key_here"
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
    "embed_model_name": "BAAI/bge-large-en-v1.5"
  },
  "dataset": {
    "data_source": "spider:dev:db_size-10",
    "data_source_dir": "../files/data_source",
    "need_few_shot": false,
    "need_external": false
  },
  "database": {
    "skip_schema_init": false,
    "schema_source": "spider:dev",
    "multi_database": false,
    "vector_store": "../vector_store",
    "schema_source_dir": "../files/schema_source",
    "need_build_index": false
  },
  "task": {
    "task_meta": [
      {
        "task_id": "generate",
        "task_type": "GenerateTask",
        "data_source": "spider:dev:db_size-10",
        "schema_source": "spider:dev",
        "dataset_save_path": "../files/datasets/spider_dev_generate.json",
        "is_save_dataset": true,
        "eval_type": ["execute_accuracy"],
        "meta": {
          "task": {
            "generate_type": "CHESSGenerator"
          }
        },
        "open_parallel": true,
        "max_workers": 10
      }
    ]
  },
  "engine": {
    "exec_process": [
      "generate"
    ]
  }
}
```

把上面的内容保存为任意路径的 JSON 文件，然后在代码里用：

```python
from core.base import Router
from core.engine import Engine

router = Router(config_path="<你的配置文件路径>.json")
engine = Engine(router)
engine.execute()
engine.evaluate()
```

## 4. 关键配置说明

### 4.1 LLM 配置（`api_key` 与 `llm`）

- `api_key`：按提供商填入密钥，可只填你要用的提供商。
- `llm.use`：`qwen` | `deepseek` | `zhipu`。
- 其他参数（`model_name`、`temperature` 等）决定具体调用模型与生成行为。

示例：

```json
"api_key": {
  "qwen": "your_api_key_here"
},
"llm": {
  "use": "qwen",
  "model_name": "qwen-turbo",
  "temperature": 0.75
}
```

### 4.2 数据集配置（`dataset`）

Squrve 对“基线数据集”采用统一编码：`<数据集标识符>:<子集标识符>:<筛选条件>`。

- 常见编码：
  - `spider:dev:`、`spider:test:`
  - `bird:dev:`
  - `spider2:snow:`、`spider2:lite:`
- 筛选条件示例：`db_size-10`、`has_label`、`db_type-sqlite` 等。

当 `data_source` 使用上述编码时，系统会自动从已解压的 `benchmarks/` 中定位数据并生成可用的本地文件。

自定义数据集：需满足如下行格式（存放为 JSON 列表）：

```json
{
  "instance_id": "unique_id",
  "db_id": "database_name",
  "question": "When do university students start their semester?",
  "db_type": "sqlite",
  "db_size": 10,
  "query": "SELECT * FROM table",
  "gold_schemas": ["table1", "table2"],
  "external_path": "path/to/external.txt"
}
```

如果你的自定义数据需要访问本地 sqlite 数据库目录，需提供 `db_path`（键名是数据源索引名，通常等于数据文件名去掉后缀）：

```json
"dataset": {
  "data_source": "files/data_source/text_to_sql.json",
  "db_path": {
    "text_to_sql": "files/databases"
  }
}
```

### 4.3 数据库 Schema 配置（`database`）

- `schema_source` 支持两种形式：
  - 基线编码：如 `spider:dev`（系统自动处理为内部并行格式）
  - 直接文件或目录路径：如 `/path/to/schema.json` 或 `/path/to/schema_dir/`
- `skip_schema_init=false` 时，central 格式会被转换为 parallel 格式，便于字段粒度管理与索引构建。
- `vector_store` 和 `need_build_index` 配合使用，可在 schema 目录上构建向量索引，用于更强的 schema linking 与检索。

示例：

```json
"database": {
  "schema_source": "spider:dev",
  "need_build_index": false,
  "vector_store": "files/vector_store",
  "schema_source_dir": "files/schema_source"
}
```

### 4.4 任务系统（`task`）

最常用的任务是 SQL 生成（`GenerateTask`）。你可以在 `task_meta` 中注册一个或多个任务：

```json
"task": {
  "task_meta": [
    {
      "task_id": "spider_dev_task",
      "task_type": "generate",
      "data_source": "spider:dev:",
      "schema_source": "spider:dev",
      "eval_type": ["execute_accuracy"],
      "open_parallel": true,
      "max_workers": 10,
      "meta": {
        "task": { "generate_type": "LinkAlign" },
        "llm": { "use": "qwen", "model_name": "qwen-turbo" }
      }
    }
  ]
}
```

`generate_type` 可选值包括：`LinkAlign`/`LinkAlignGenerator`、`DINSQL`、`DAILSQL`、`CHESS`/`CHESSGenerator`、`MACSQL`、`RSLSQL`、`ReFoRCE`、`OpenSearchSQL` 等。

### 4.5 执行引擎（`engine.exec_process`）

`exec_process` 定义多个任务的编排方式，支持“列表”与“字典”两种形式：

- 列表形式（含并发/串行标记）：

```json
[
  "task_1",
  ["task_2", "task_3", "~p"],
  "~s"
]
```

- 字典形式（递归定义并行/串行）：

```json
{
  "type": "sequence",
  "tasks": [
    "task_1",
    { "type": "parallel", "tasks": ["task_2", "task_3"] },
    "task_4"
  ]
}
```

## 5. 远程数据库与 SQL 执行（可选）

若你的任务需要连接远程数据库（如 BigQuery / Snowflake）执行 SQL，请在配置中添加 `credential`：

```json
"credential": {
  "big_query": "path/to/big_query_credential.json",
  "snowflake": "path/to/snowflake_credential.json"
}
```

你也可以在代码中直接执行 SQL（示例）：

```python
from core.db_connect import execute_sql

# sqlite
print(execute_sql("sqlite", db_path="/path/to/your.db", sql="SELECT 1;", credential=None))

# snowflake（db_path 传 DB 名，credential 传配置文件/字典）
print(execute_sql("snowflake", db_path="YOUR_DB", sql="SELECT 1;", credential={"snowflake": "path/to/credential.json"}))

# big_query
print(execute_sql("big_query", db_path=None, sql="SELECT 1", credential={"big_query": "path/to/credential.json"}))
```

注意：BigQuery 会通过 `GOOGLE_APPLICATION_CREDENTIALS` 读取凭证路径；Snowflake 需传入连接所需字段（在凭证 JSON 中）。

## 6. 输出与评估

执行 `engine.execute()` 后，若配置了 `eval_type`，可调用 `engine.evaluate()` 自动完成指标评估并返回结果字典。常见指标：
- `exact_match`
- `execution` / `execute_accuracy`
- `reduce_recall`
- `parse_accuracy`

默认输出目录见第 2 节“快速上手”。

## 7. 常见问题（FAQ）

- 无法找到基准数据或 schema：是否已在项目根目录解压 `benchmarks.zip`？
- API Key 无效或未配置：检查 `api_key` 与 `llm.use` 是否匹配，提供商是否可用。
- BigQuery 报凭证错误：确认凭证文件存在且 `GOOGLE_APPLICATION_CREDENTIALS` 可被设置。
- Snowflake 超时或连接失败：检查账户、region、warehouse、role 等字段是否齐全；可在配置中调大超时时间。
- Windows 路径问题：尽量使用绝对路径或 `Path` 规范路径，注意 `startup_run` 中相对路径以该目录为基准。
- 运行无输出/无任务：检查 `engine.exec_process` 是否指向了有效的 `task_id`，或简化为单任务顺序执行。

## 8. 进阶：自定义任务与多任务并发

- 自定义任务：继承 `core.task.meta.MetaTask`，实现 `load_actor` 与 `run`；或直接使用内置的 `GenerateTask` / `ParseTask` / `ReduceTask` 等。
- 复杂编排：通过 `task.cpx_task_meta` 与 `engine.exec_process` 定义串并行混合流程，开启 `open_parallel`/`open_actor_parallel` 实现多并发执行。

## 9. 参考

- 详细 API：见 `doc/API documents.md`
- 启动示例：见 `startup_run/README.md`
- 顶层说明：见根目录 `README.md`