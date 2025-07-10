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
│   └── run_.py                   # 基础运行脚本
├── benchmarks/
│   └── spider/
│       ├── dev/
│       │   ├── dataset.json      # Spider Dev 数据集
│       │   └── schema.json       # 数据库模式文件
│       └── database/             # SQLite 数据库文件
├── squrve_api.md                 # 详细 API 文档
└── SPIDER_DEV_README.md          # 本文件
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

编辑 `config/spider_dev_config.json` 文件，配置您的 API 密钥：

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
python run/run_spider_dev.py
```

脚本会提供以下选项：
- **简单 SQL 生成任务**: 直接生成 SQL 查询
- **完整 Text-to-SQL 流水线**: 执行完整的 Reduce -> Parse -> Generate 流程
- **自定义设置任务**: 使用自定义参数运行

#### 方式二：使用基础脚本

```bash
python run/run_.py
```

这会使用 `config/demo_config.json` 配置文件运行。

#### 方式三：编程方式

```python
from core.base import Router
from core.engine import Engine

# 使用 Spider Dev 配置
router = Router(config_path="../config/spider_dev_config.json")
engine = Engine(router)

# 执行任务
engine.execute()

# 评估结果
engine.evaluate()
```

## 📊 配置文件详解

### Spider Dev 专用配置 (`config/spider_dev_config.json`)

该配置文件专门为 Spider dev 数据集优化，包含以下主要部分：

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

#### 数据集配置
```json
{
  "dataset": {
    "data_source": "spider:dev",
    "db_path": "benchmarks/spider/database"
  }
}
```

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

## 🔧 自定义配置

### 修改模型参数

```json
{
  "llm": {
    "use": "deepseek",
    "model_name": "deepseek-chat",
    "temperature": 0.5,
    "max_token": 4000
  }
}
```

### 启用 Few-shot 学习

```json
{
  "dataset": {
    "need_few_shot": true,
    "few_shot_num": 5
  }
}
```

### 启用外部知识

```json
{
  "dataset": {
    "need_external": true
  }
}
```

### 数据采样

在任务元数据中配置：

```json
{
  "meta": {
    "dataset": {
      "random_size": 0.1,  // 使用10%的数据
      "filter_by": "has_label"
    }
  }
}
```

## 📁 输出文件

运行完成后，结果文件将保存在以下目录：

- `files/pred_sql/`: 生成的 SQL 查询
- `files/instance_schemas/`: 模式降维结果
- `files/schema_links/`: 模式链接结果
- `files/logs/`: 执行日志
- `files/datasets/`: 处理后的数据集
- `files/reasoning_examples/user/`: Few-shot 示例
- `files/external/`: 外部知识文件

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

## 📈 评估指标

Squrve 支持多种评估指标：

- **exact_match**: 精确匹配率
- **execution**: 执行正确性
- **reduce_recall**: 模式降维召回率
- **parse_accuracy**: 模式解析准确率

## 🐛 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误: API key is invalid
   解决: 检查 config/spider_dev_config.json 中的 API 密钥配置
   ```

2. **数据集路径错误**
   ```
   错误: Data source not found
   解决: 确保 benchmarks/spider/dev/ 目录存在且包含 dataset.json 和 schema.json
   ```

3. **数据库路径错误**
   ```
   错误: Database file not found
   解决: 确保 benchmarks/spider/database/ 目录包含所需的 SQLite 数据库文件
   ```

4. **内存不足**
   ```
   错误: Out of memory
   解决: 减少 max_workers 参数或使用数据采样 (random_size)
   ```

### 调试模式

启用详细日志：

```python
from core.log import Logger

logger = Logger(save_path="files/logs/debug.log")
logger.setLevel("DEBUG")
```

## 📚 相关文档

- [Squrve API 详细文档](squrve_api.md): 完整的 API 参考
- [原始 API 文档](API.md): 项目原始文档
- [项目 README](README.md): 项目概述

## 🤝 贡献

如果您在使用过程中遇到问题或有改进建议，请：

1. 检查现有文档
2. 查看日志文件
3. 提交 Issue 或 Pull Request

## 📄 许可证

请参考项目主目录的许可证文件。 