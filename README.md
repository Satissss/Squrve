# Squrve

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-Text--to--SQL-orange.svg)

**Squrve** 是一个轻量级、模块化的端到端 Text-to-SQL 模型开发和评估框架

</div>

## 📖 概述

**Squrve** 是一个专为快速开发和评估端到端 **Text-to-SQL** 模型而设计的轻量级、模块化框架。它将模式降维（schema reduction）、模式链接（schema linking）和查询生成（query generation）集成到一个灵活的、基于配置的流水线中。

### ✨ 核心特性

- 🚀 **快速启动**: 仅需配置文件即可启动完整的 Text-to-SQL 流水线
- 🔧 **模块化设计**: 所有组件可独立实例化和调试
- ⚡ **并行执行**: 支持多任务并发执行
- 🎯 **灵活配置**: 通过 JSON 配置文件实现即插即用的模型集成
- 📊 **内置评估**: 提供多种评估指标和可视化结果
- 🔗 **多模型支持**: 支持 Qwen、DeepSeek、智谱等主流 LLM

## 🏗️ 核心架构

Squrve 采用模块化架构，主要包含以下核心组件：

- **Router**: 配置管理器，负责管理整个 Text-to-SQL 流程的参数配置
- **DataLoader**: 数据管理器，负责数据准备和加载
- **Engine**: 执行引擎，协调各个组件的执行流程
- **Actor**: 执行器，包含 Reducer、Parser、Generator 等具体执行组件
- **Task**: 任务管理器，支持复杂任务嵌套和并行执行

### 支持的 Baselines

Squrve 支持多种 Text-to-SQL baselines，通过模块化组件实现快速集成和比较：

#### Generators（查询生成器）

| Baseline 名称       | 方法介绍 |
|---------------------|----------|
| BaseGenerate       | 基础生成器，提供标准的 Text-to-SQL 查询生成功能。 |
| CHESSGenerate      | CHESS 方法实现，专注于复杂查询的层次化生成和优化。 |
| DAILSQLGenerate    | DAIL-SQL 方法，通过分治提示和链式思考实现高效 SQL 生成。 |
| DINSQLGenerate     | DIN-SQL 方法，使用分解提示处理复杂 SQL 查询生成。 |
| LinkAlignGenerate  | LinkAlign 集成生成，利用高级模式链接提升查询准确性。 |
| MACSQLGenerate     | MAC-SQL 方法，采用多代理协作机制生成高质量 SQL。 |
| OpenSearchSQLGenerate | 基于 OpenSearch 的 SQL 生成，利用搜索增强查询构建。 |
| ReFoRCEGenerate    | ReFoRCE 方法，通过强化学习框架优化 SQL 生成过程。 |
| RSLSQLGenerate     | RSL-SQL 方法，结合规则系统和学习模型生成可靠 SQL。 |

#### Parsers（模式解析器）
- **BaseParse**: 基础解析器
- **LinkAlignParse**: LinkAlign 模式链接解析

#### Reducers（模式降维器）
- **BaseReduce**: 基础降维
- **LinkAlignReduce**: LinkAlign 降维
- **ZeroReduce**: 零降维（无降维）

### 支持的 Benchmarks

Squrve 内置支持多个标准 Text-to-SQL benchmarks，便于模型评估和比较：

| Benchmark | 描述 |
|-----------|------|
| Spider   | 跨域 Text-to-SQL 基准，支持 dev 分割。 |
| BIRD     | 带外部知识的 Text-to-SQL 基准。 |
| Spider2  | Spider 的扩展版本，包含更多复杂场景。 |
| AmbiDB   | 歧义数据库查询基准，测试歧义处理能力。 |

这些 benchmarks 可通过配置文件轻松加载和评估。

## 🚀 快速开始

### 1. 环境准备

确保您的 Python 环境满足以下要求：
- Python 3.8+
- 必要的依赖包（详见 requirements.txt）

```bash
# 克隆项目
git clone <repository-url>
cd Squrve

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

编辑配置文件，添加您的 API 密钥：

```json
{
  "api_key": {
    "qwen": "your_qwen_api_key",
    "deepseek": "your_deepseek_api_key",
    "zhipu": "your_zhipu_api_key"
  }
}
```
### [Opt.] LinkAlign 配置
若使用 LinkAlign 相关组件，需要按照 https://github.com/Satissss/LinkAlign/blob/master/README.md 进行配置。

### 3. 运行示例

#### 方式一：使用启动脚本

```bash
# 运行 Spider Dev 数据集示例
python startup_run/run_spider_dev.py
```

#### 方式二：编程方式

```python
from core.base import Router
from core.engine import Engine

# 使用配置文件初始化
router = Router(config_path="startup_run/spider_dev_config.json")
engine = Engine(router)

# 执行任务
engine.execute()

# 评估结果
engine.evaluate()
```

## 📁 项目结构

```
Squrve/
├── core/                    # 核心模块
│   ├── base.py             # 基础类和配置管理
│   ├── engine.py           # 执行引擎
│   ├── data_manage.py      # 数据管理
│   ├── actor/              # 执行器组件
│   │   ├── reducer/        # 模式降维
│   │   ├── parser/         # 模式链接
│   │   └── generator/      # 查询生成
│   └── task/               # 任务管理
├── startup_run/            # 启动示例
│   ├── run_spider_dev.py   # Spider Dev 运行脚本
│   └── spider_dev_config.json  # 示例配置文件
├── config/                 # 配置文件
├── files/                  # 输出文件
│   ├── datasets/           # 处理后的数据集
│   ├── pred_sql/           # 生成的 SQL 查询
│   └── schema_links/       # 模式链接结果
└── benchmarks/             # 基准数据集
```

## 🎯 快速使用

根据场景需求，定义 Text2SQL 任务执行的配置文件，即可自动完成 SQL 生成任务并自动完成评估。具体的 startup 示例可参考 startup_run 目录。
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


## 📊 输出结果

运行完成后，您可以在以下目录查看结果：

- **`files/pred_sql/`**: 生成的 SQL 查询文件
- **`files/schema_links/`**: 模式链接结果
- **`files/datasets/`**: 处理后的数据集
- **`files/logs/`**: 执行日志

## 📚 详细文档

- **[API 文档](API%20documents.md)**: 完整的 API 参考文档，包含所有配置参数和方法的详细说明
- **[启动示例](startup_run/README.md)**: Spider Dev 数据集的使用指南和配置示例

## 🔧 配置说明

### 主要配置参数

- **LLM 配置**: 指定使用的语言模型和参数
- **数据集配置**: 数据源路径和预处理选项
- **数据库配置**: 数据库模式和向量存储设置
- **任务配置**: 任务类型和执行流程定义
- **评估配置**: 评估指标和结果保存设置

详细配置说明请参考 [API 文档](API%20documents.md)。


## 📝 TODO List

- [ ] 添加基准 baseline 方法
- [ ] 集成基准 benchmark 数据集
- [ ] 扩展数据库连接支持
- [ ] 扩展 Actor 组件库
- [ ] 扩展评估指标体系
- [ ] 扩展微服务架构
- [ ] 集成强化学习框架


## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。




