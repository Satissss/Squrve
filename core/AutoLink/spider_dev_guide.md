# AutoLink 测试 Spider Dev 数据集指南

本文档说明如何使用 AutoLink（配合 DIN-SQL Generator）对 Spider Dev 数据集完成端到端的解析、生成与评估。

---

## 目录

1. [前置依赖](#前置依赖)
2. [Step 1：构建 FAISS Embedding 索引](#step-1构建-faiss-embedding-索引)
3. [Step 2：编写 multi_actor_config.json](#step-2编写-multi_actor_configjson)
4. [Step 3：运行流水线](#step-3运行流水线)
5. [参数说明](#参数说明)
6. [目录结构说明](#目录结构说明)

---

## 前置依赖

```bash
pip install faiss-cpu sentence-transformers tqdm
```

确保以下文件/目录存在：

```
benchmarks/spider/
├── database/          # 每个 db_id 对应一个 .sqlite 文件
└── dev/
    └── schema.json    # Spider 格式的 schema 文件
```

---

## Step 1：构建 FAISS Embedding 索引

AutoLinkParser 在运行时需要提前构建好的 FAISS 向量索引。**在项目根目录下**执行：

```bash
# 基础用法（CPU，含样本值采样）
python -m core.AutoLink.build_index \
    --schema  benchmarks/spider/dev/schema.json \
    --db_dir  benchmarks/spider/database \
    --out_dir embeddings/spider_dev \
    --dataset spider

# 使用 GPU 加速编码（推荐，速度更快）
python -m core.AutoLink.build_index \
    --schema  benchmarks/spider/dev/schema.json \
    --db_dir  benchmarks/spider/database \
    --out_dir embeddings/spider_dev \
    --dataset spider \
    --device  cuda:0

# 跳过已建好的数据库（断点续建）
python -m core.AutoLink.build_index \
    --schema  benchmarks/spider/dev/schema.json \
    --db_dir  benchmarks/spider/database \
    --out_dir embeddings/spider_dev \
    --dataset spider \
    --skip_existing
```

> **说明**：`--db_dir` 用于从 SQLite 中采样列的真实值，可以提升检索准确率。若只想快速构建索引可加 `--no_samples` 跳过此步骤。

构建完成后，`embeddings/spider_dev/` 下每个数据库会生成：
```
embeddings/spider_dev/
└── concert_singer/
    ├── index.faiss
    └── metadata.json
```

---

## Step 2：编写 multi_actor_config.json

在 `startup_run/` 目录下创建或修改 `multi_actor_config.json`：

```json
{
  "api_key": {
    "deepseek": "your_api_key_here",
    "qwen": "your_api_key_here",
    "zhipu": "your_api_key_here"
  },
  "llm": {
    "use": "deepseek",
    "model_name": "deepseek-chat",
    "context_window": 120000,
    "max_token": 4000,
    "top_p": 0.5,
    "temperature": 0.3,
    "time_out": 300.0
  },
  "text_embed": {
    "embed_model_name": "BAAI/bge-large-en-v1.5"
  },
  "dataset": {
    "data_source_dir": "../files/data_source",
    "need_few_shot": false,
    "need_external": false
  },
  "database": {
    "skip_schema_init": false,
    "schema_source_dir": "../files/schema_source",
    "multi_database": false,
    "vector_store": "../vector_store",
    "need_build_index": false
  },
  "task": {
    "task_meta": [
      {
        "task_id": "autolink_parse",
        "task_type": "ParseTask",
        "data_source": "../files/data_source/spider_dev.json",
        "schema_source": "spider:dev",
        "is_save_dataset": true,
        "open_parallel": true,
        "max_workers": 3,
        "meta": {
          "dataset": {
            "db_path": "../benchmarks/spider/database"
          },
          "task": {
            "parse_type": "AutoLinkParser",
            "embed_path": "../embeddings/spider_dev",
            "db_type": "sqlite",
            "retrieval_top_k": 3,
            "retrieval_device": "cuda:0",
            "max_turns": 5
          }
        }
      },
      {
        "task_id": "autolink_generate",
        "task_type": "GenerateTask",
        "data_source": "../files/data_source/spider_dev.json",
        "schema_source": "spider:dev",
        "is_save_dataset": true,
        "open_parallel": true,
        "max_workers": 3,
        "meta": {
          "task": {
            "generate_type": "DINSQLGenerator",
            "save_dir": "../files/pred_sql/spider_dev_dinsql",
            "use_external": true
          }
        }
      }
    ],
    "cpx_task_meta": [
      {
        "task_id": "autolink_pipeline",
        "task_lis": [
          "autolink_parse",
          "autolink_generate"
        ],
        "eval_type": [
          "execute_accuracy"
        ],
        "is_save_dataset": true,
        "dataset_save_path": "../files/datasets/spider_dev_dinsql.json",
        "meta": {
          "actor": {
            "autolink_parse": {
              "save_dir": "../files/schema_links/spider_dev"
            },
            "autolink_generate": {
              "save_dir": "../files/pred_sql/spider_dev_dinsql"
            }
          },
          "dataset": {
            "db_path": "../benchmarks/spider/database"
          }
        },
        "open_parallel": true,
        "max_workers": 3
      }
    ],
    "is_save_dataset": true
  },
  "engine": {
    "exec_process": [
      "autolink_pipeline"
    ]
  }
}
```

### 关键配置说明

| 字段 | 位置 | 说明 |
|---|---|---|
| `data_source` | `task_meta[*]` | spider dev 完整数据集 JSON 路径 |
| `schema_source` | `task_meta[*]` | 使用 `spider:dev` 让框架自动加载 Spider schema |
| `embed_path` | `task_meta[autolink_parse].meta.task` | 指向 Step 1 生成的 embedding 目录 |
| `retrieval_device` | `task_meta[autolink_parse].meta.task` | 检索设备，无 GPU 改为 `cpu` |
| `retrieval_top_k` | `task_meta[autolink_parse].meta.task` | 每次检索返回的 schema 候选数量 |
| `max_turns` | `task_meta[autolink_parse].meta.task` | AutoLink 最大迭代轮数 |
| `meta.dataset.db_path` | `cpx_task_meta[autolink_pipeline]` | **必填**，评估阶段用于执行 SQL 时定位数据库 |

### 只跑部分数据（测试用）

在 `cpx_task_meta[autolink_pipeline].meta.dataset` 加入 `random_size` 限制条数：

```json
"dataset": {
  "random_size": 5,
  "db_path": "../benchmarks/spider/database"
}
```

---

## Step 3：运行流水线

```bash
cd startup_run
python run.py
```

运行结束后，终端会输出：

```
eval_type: execute_accuracy,   number: XXX    result: X.XXXX
```

结果文件保存位置：

| 内容 | 路径 |
|---|---|
| 解析结果（schema links） | `files/schema_links/spider_dev/` |
| 生成的 SQL | `files/pred_sql/spider_dev_dinsql/` |
| 完整数据集（含 pred_sql 字段） | `files/datasets/spider_dev_dinsql.json` |

---

## 参数说明

### `build_index` 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--schema` | *(必填)* | schema.json 路径 |
| `--out_dir` | *(必填)* | 索引输出目录 |
| `--dataset` | `spider` | schema 格式，`spider` 或 `spider2` |
| `--db_dir` | `None` | SQLite 数据库目录（用于采样列值） |
| `--model` | `BAAI/bge-large-en-v1.5` | SentenceTransformer 模型名 |
| `--batch_size` | `64` | 编码批大小 |
| `--device` | `cpu` | 编码设备：`cpu` / `cuda` / `cuda:0` |
| `--skip_existing` | `False` | 跳过已建索引的数据库 |
| `--no_samples` | `False` | 不采样列值（更快但准确率略低） |

### AutoLinkParser 关键参数

| 参数 | 说明 |
|---|---|
| `embed_path` | build_index 生成的索引根目录 |
| `db_type` | 数据库类型，Spider Dev 固定为 `sqlite` |
| `retrieval_top_k` | 每轮检索返回的 schema 候选数，推荐 `3~5` |
| `retrieval_device` | 检索时使用的设备，推荐 `cuda:0` |
| `max_turns` | 最大迭代对话轮数，推荐 `3~7` |

---

## 目录结构说明

```
Squrve/
├── benchmarks/spider/
│   ├── database/          # .sqlite 文件（一个数据库一个文件）
│   └── dev/schema.json    # Spider schema
├── embeddings/
│   └── spider_dev/        # build_index 输出（Step 1 生成）
│       └── <db_id>/
│           ├── index.faiss
│           └── metadata.json
├── files/
│   ├── data_source/       # 数据集 JSON（如 spider_dev.json）
│   ├── schema_links/      # AutoLinkParser 输出
│   ├── pred_sql/          # DINSQLGenerator 输出
│   └── datasets/          # 完整结果 JSON
└── startup_run/
    ├── run.py
    └── multi_actor_config.json
```
