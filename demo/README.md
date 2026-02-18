# Squrve Text-to-SQL Demo

基于 Gradio 的 Squrve Text-to-SQL 演示项目，提供便捷的 Web UI 接口，支持数据库上传、自然语言问答与 SQL 生成执行。

## 实现功能

### 1. 数据库上传

- **.sqlite / .db 文件**：上传单个 SQLite 数据库文件，自动抽取 schema 并生成 Spider 格式的 `schema.json`
- **多个 .xlsx / .csv 文件**：将多个 Excel 或 CSV 文件合并为一个 SQLite 数据库
  - 每个文件对应一张表，表名 = 文件名（去掉扩展名）
  - 文件第一行作为列名，其余行作为数据

### 2. 数据库选择与持久化

- 上传的数据库保存在 `files/uploaded_db/{db_id}/` 目录
- 通过 `manifest.json` 记录所有已上传数据库
- 支持在同一会话或后续会话中，从已上传数据库中选择进行 Text-to-SQL

### 3. Text-to-SQL 生成

- **直接 Generator**：从下拉框选择单个 Generator（如 DINSQLGenerator、LinkAlignGenerator 等）直接生成 SQL
- **自定义 Workflow**：选择骨架（如 `parser → generator`、`decomposer → parser → generator → optimizer`），再为每个步骤选择具体 Actor

### 4. SQL 执行

- 生成 SQL 后可直接在界面中执行，并查看查询结果

## 使用方式

### 环境准备

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装 Demo 额外依赖
pip install -r demo/requirements-demo.txt
```

### 配置

1. 在 `startup_run/startup_config.json` 中配置 LLM API Key（如 qwen、deepseek 等）
2. 可选：修改 `demo/demo_config.yaml` 调整路径、端口等

### 启动 Demo

```bash
# 在项目根目录下执行
python demo/gradio_demo.py
```

默认在 `http://0.0.0.0:7860` 启动。

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | Router 配置文件路径 | `demo/startup_config.json` |
| `--server-name` | 服务监听地址 | `0.0.0.0` |
| `--server-port` | 服务端口 | `7860` |
| `--share` | 创建公共分享链接 | - |

示例：

```bash
python demo/gradio_demo.py --config demo/startup_config.json --share --server-port 8080
```

### 使用流程

1. **Upload 标签页**：上传 .sqlite 或 .xlsx/.csv 文件，点击「Process」处理
2. **Query 标签页**：
   - 在顶部选择要查询的数据库
   - 输入自然语言问题
   - 选择生成方式（直接 Generator 或自定义 Workflow）
   - 点击「Generate SQL」生成 SQL
   - 点击「Execute SQL」执行并查看结果

## 目录结构

```
demo/
├── README.md           # 本文件
├── demo_config.yaml    # Demo 配置（路径、端口等）
├── file_to_db.py       # 文件转数据库逻辑（xlsx/csv/sqlite → schema）
├── gradio_demo.py      # Gradio 主界面
├── requirements-demo.txt  # Demo 额外依赖
└── utils.py
```

## 配置说明

`demo_config.yaml` 主要配置项：

| 配置项 | 说明 |
|--------|------|
| `paths.uploaded_db_root` | 上传数据库存储根目录 |
| `paths.temp_data_dir` | 临时数据目录 |
| `router_config` | 主配置文件路径（LLM、API Key 等） |
| `server.name` / `server.port` | 服务监听地址与端口 |
