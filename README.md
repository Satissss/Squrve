# Squrve

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-Text--to--SQL-orange.svg)

**Squrve** is a lightweight, modular framework for end-to-end Text-to-SQL model development and evaluation.

</div>

## 📖 Overview

**Squrve** is a lightweight, modular framework designed for rapid development and evaluation of end-to-end **Text-to-SQL** models. It integrates schema reduction, schema linking, and query generation into a flexible, configuration-based pipeline.

### ✨ Key Features

1. Configuration-driven Text-to-SQL tasks, integrating multiple baseline methods, large model invocations, and database connections.
2. Component-level reproduction and sharing interfaces, supporting free combination and flexible switching of different method components for plug-and-play and quick startup.
3. Scalable and robust modular design, with method implementations independent of specific datasets and database types, enabling rapid extension for new methods.

## 🏗️ Core Architecture

Squrve adopts a modular architecture with the following core components:

- **Router**: Configuration manager, responsible for managing parameters for the entire Text-to-SQL process.
- **DataLoader**: Data manager, handling data preparation and loading.
- **Engine**: Execution engine, coordinating the execution flow of various components.
- **Actor**: Executor, including specific components like Reducer, Parser, and Generator.
- **Task**: Task manager, supporting complex task nesting and parallel execution.

### Supported Baselines

Squrve supports multiple Text-to-SQL baselines, enabling quick integration and comparison through modular components:

| Baseline Name      | Description                                                                 | Code Link |
|--------------------|-----------------------------------------------------------------------------|-----------|
| BaseGenerate       | Basic generator providing standard Text-to-SQL query generation.            |           |
| CHESSGenerate      | CHESS method implementation, focusing on hierarchical generation and optimization of complex queries. |           |
| DAILSQLGenerate    | DAIL-SQL method, using divide-and-conquer prompting and chain-of-thought for efficient SQL generation. |           |
| DINSQLGenerate     | DIN-SQL method, using decomposition prompting to handle complex SQL query generation. |           |
| LinkAlignGenerate  | LinkAlign integrated generation, utilizing advanced schema linking to improve query accuracy. |           |
| MACSQLGenerate     | MAC-SQL method, employing multi-agent collaboration for high-quality SQL generation. |           |
| OpenSearchSQLGenerate | OpenSearch-based SQL generation, using search enhancement for query construction. |           |
| ReFoRCEGenerate    | ReFoRCE method, optimizing SQL generation through a reinforcement learning framework. |           |
| RSLSQLGenerate     | RSL-SQL method, combining rule systems and learning models for reliable SQL generation. |           |

### Supported Benchmarks

Squrve includes built-in support for several standard Text-to-SQL benchmarks for easy model evaluation and comparison:

| Benchmark | Description                                      | Code Link |
|-----------|--------------------------------------------------|-----------|
| Spider    | Cross-domain Text-to-SQL benchmark, supporting dev split. |           |
| BIRD      | Text-to-SQL benchmark with external knowledge.   |           |
| Spider2   | Extended version of Spider with more complex scenarios. |           |
| AmbiDB    | Ambiguous database query benchmark, testing ambiguity handling. |           |

These benchmarks can be easily loaded and evaluated via configuration files.

## 🚀 Quick Start

### 1. Environment Setup

Ensure your Python environment meets the following requirements:
- Python 3.8+
- Required dependencies (see requirements.txt)

```bash
# Clone the repository
git clone https://github.com/Satissss/Squrve.git

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the configuration file to add your API keys:

```json
{
  "api_key": {
    "qwen": "your_qwen_api_key",
    "deepseek": "your_deepseek_api_key",
    "zhipu": "your_zhipu_api_key"
  }
}
```

### [Opt.] LinkAlign Configuration
If using LinkAlign-related components, configure according to https://github.com/Satissss/LinkAlign/blob/master/README.md.

### 3. Run Examples

#### Method 1: Using Startup Script

```bash
# Run Spider Dev dataset example
python startup_run/run_spider_dev.py
```

#### Method 2: Programmatic Approach

```python
from core.base import Router
from core.engine import Engine

# Initialize with configuration file
router = Router(config_path="startup_run/spider_dev_config.json")
engine = Engine(router)

# Execute task
engine.execute()

# Evaluate results
engine.evaluate()
```

## 📁 Project Structure

```
Squrve/
├── core/                    # Core modules
│   ├── base.py             # Base classes and configuration management
│   ├── engine.py           # Execution engine
│   ├── data_manage.py      # Data management
│   ├── actor/              # Executor components
│   │   ├── reducer/        # Schema reduction
│   │   ├── parser/         # Schema linking
│   │   └── generator/      # Query generation
│   └── task/               # Task management
├── startup_run/            # Startup examples
│   ├── run_spider_dev.py   # Spider Dev run script
│   └── spider_dev_config.json  # Example configuration file
├── config/                 # Configuration files
├── files/                  # Output files
│   ├── datasets/           # Processed datasets
│   ├── pred_sql/           # Generated SQL queries
│   └── schema_links/       # Schema linking results
└── benchmarks/             # Benchmark datasets
```

## 🎯 Quick Usage

Define a Text2SQL task execution configuration file based on your scenario needs to automatically complete SQL generation tasks and evaluations. For specific startup examples, refer to the startup_run directory.

```python
from core.base import Router
from core.engine import Engine

# Use Spider Dev configuration
router = Router(config_path="spider_dev_config.json")
engine = Engine(router)

# Execute task
engine.execute()

# Evaluate results
engine.evaluate()
```

## 📊 Output Results

After running, you can view the results in the following directories:

- **`files/pred_sql/`**: Generated SQL query files
- **`files/schema_links/`**: Schema linking results
- **`files/datasets/`**: Processed datasets
- **`files/logs/`**: Execution logs

## 📚 Detailed Documentation

- **[API Documentation](API%20documents.md)**: Complete API reference with detailed explanations of all configuration parameters and methods
- **[Startup Examples](startup_run/README.md)**: Usage guide and configuration examples for the Spider Dev dataset

## 🔧 Configuration Guide

### Main Configuration Parameters

- **LLM Configuration**: Specify the language model and parameters to use
- **Dataset Configuration**: Data source paths and preprocessing options
- **Database Configuration**: Database schema and vector store settings
- **Task Configuration**: Task types and execution flow definitions
- **Evaluation Configuration**: Evaluation metrics and result saving settings

For detailed configuration explanations, refer to the [API Documentation](API%20documents.md).

## 📝 TODO List

- [ ] Add benchmark baseline methods
- [ ] Integrate benchmark datasets
- [ ] Extend database connection support
- [ ] Expand Actor component library
- [ ] Expand evaluation metrics system
- [ ] Extend to microservices architecture
- [ ] Integrate reinforcement learning framework

## 📄 License

This project is licensed under the [MIT License](LICENSE).




