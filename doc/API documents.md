# Squrve API 详细文档

## 项目概述

Squrve 是一个支持快速开发"端到端" Text-to-SQL 模型或对现有模型进行快速评估的工具。Text-to-SQL 任务流封装了多种组件，支持模式降维（schema
reduce）、模式链接(schema linking)、查询生成（query generation）的快速实现。仅依靠配置文件即可快速启动
Squrve，并可轻松实现复杂任务嵌套、多任务并发执行。

## 核心架构

### 1. Router - 配置管理器

Router 是 Squrve 的核心配置管理器，负责管理 Text-to-SQL
的全部处理流程的参数配置。它可以在初始化阶段自动加载系统配置（例如：临时文件的存储位置，静态参数文件地址），并支持通过配置文件和显示传入参数的方式进行配置。如果未提供配置，将自动加载
Demo 参数配置。

#### 初始化方式

Router 只能通过 JSON 文件或显式传入参数创建。

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

##### LLM 配置

* api_key: 字典类型，包含不同 LLM 提供商的 API 密钥。支持 "deepseek"、"qwen"、"zhipu" 等。
    * deepseek: Deepseek LLM 的 API 密钥。
    * qwen: Qwen LLM 的 API 密钥。
    * zhipu: Zhipu LLM 的 API 密钥。
* llm: LLM 相关的配置参数。
    * use: 指定使用的 LLM 提供商，如 "qwen", "deepseek", "zhipu"。
    * model_name: 使用的 LLM 模型名称。
    * context_window: LLM 的上下文窗口大小。
    * max_token: LLM 生成的最大输出 token 数量。
    * temperature: 控制生成文本随机性的温度参数。
    * top_p: 控制生成文本多样性的 top-p 采样参数。
    * time_out: LLM 请求的超时时间。

##### 文本嵌入配置

* text_embed: 文本嵌入模型相关的配置参数。
    * embed_model_source: 嵌入模型来源，目前仅支持 "HuggingFace"。
    * embed_model_name: 嵌入模型名称，例如 "BAAI/bge-large-en-v1.5"。

##### 数据集配置 (dataset)

* data_source: 数据源路径或标识符。可以是字符串、字符串列表或字典，用于适配多任务场景。
    * 用户可提供自定义数据源，也可使用经典基线数据集。
    * 对于经典基线数据集，如果需要添加 few_shot 或 external，data_source 应传入 "经典数据集标识符:子数据集名称:筛选条件"
      的格式。
    * 如果不显式指定 data_source 且任务使用了经典基线数据集，系统会自动添加。
* data_source_dir: 系统默认自动保存 data_source 文件的目录，由系统配置提供。若用户未传入 data_source 而是数据本身，则默认保存在该路径下。
* default_data_file_name: 若用户未传入 data_source 而是数据本身，默认保存的文件名称。
* overwrite_exist_file: 若 data_source_dir 中出现同名文件是否覆盖原文件，默认为 True。
* need_few_shot: 是否需要添加 few-shot 示例，这些示例来自 QueryFlow 提供的思维链样本库。
* few_shot_key_name: few-shot 示例在数据样本字典中的键名，默认为 "reason_examples"。
* few_shot_range: 添加思维链示例的数据源范围，接受 bool, str, int, List[int], List[str]。
* few_shot_num: 思维链提示的数量，必须是大于 1 的整数，否则撤销添加操作。
* few_shot_save_dir: 所有数据集 few-shot 保存的根路径。实际完整路径为 few_shot_save_dir / 数据集标识符 / data_id.txt。
* sys_few_shot_dir: 系统提供 few-shot 示例的路径。
* need_external: 是否需要添加外部知识。
    * 若数据样本由用户提供，必须在数据样本键值对中指定外部知识文件地址，键名为 external_path。
    * 若用户能够自主完成 external 添加过程，该参数可不提供或设置为 False。
* default_get_external_function: 默认添加 external 的方法，由系统配置提供。
* external_range: 一个 data_source 标识符列表，保存所有需要 external 的数据集。
* external_save_dir: 默认系统保存 external 文件根目录。实际完整路径为 external_save_dir / 数据集标识符 / data_id.txt。
* external_key_name: external 键名，默认为 "external"。
* db_path: 本地数据库存储路径。

##### 数据库配置 (database)

* skip_schema_init: 是否直接跳过 schema 初始化。
    * 如果希望直接存放原始文件处理后续流程，或希望以 central 文件进行后续操作，该参数必须设置为 True。
    * Schema 初始化旨在将 schema 格式从 central 转换为 parallel。
* schema_source: 数据库模式源。可以是字符串、字符串列表或字典，用于适配多任务场景。
    * schema_save_source 的路径为 <schema_save_dir> / <schema_index>。
* multi_database: 是否多数据库模式。
    * 若为 True，处理 schema_source 文件时将所有 .json 文件放置在单一目录下。
    * 若 schema_source 为字符串，multi_database 参数仅允许提供布尔值或字典对象。
    * 若 schema_source 为列表，multi_database 参数允许提供布尔值、列表或字典对象，但列表对象必须与 schema_source 等长。
    * 若 schema_source 为字典，multi_database 参数允许提供布尔值或字典对象。
* vector_store: 向量存储路径。可以是字符串、列表或字典。
    * 若为列表，必须与 schema_source 长度相等。
    * 若为字典，仅提供需要建立索引的 schema_source 标识即可。
    * 若该参数不提供，默认为每个 "schema_source / vector_store" 目录。
* schema_source_dir: 系统默认自动保存 schema_source 处理后文件的目录，由系统配置提供。
* default_schema_dir_name: 若用户未传入 schema_source 目录而是 schema 本身，默认保存的处理后文件夹名称。
* need_build_index: 变量为 True 则开启索引创建。
* index_method: 仅支持 "VectorStoreIndex" 方法，默认由系统配置提供。
* index_range: 一个 schema_source 标识符列表，提供需要建立索引的列表。不指定该参数则默认为全部 schema_source。

##### Router 配置

* use_demo: 是否使用 demo 配置文件。若用户未传入配置文件且该参数为真时，使用 demo 配置文件。默认由系统配置提供，为 False。

##### DataLoader 配置

* is_prepare_data: 是否在 DataLoader 初始化时自动进行数据准备。数据准备包括添加 few-shot、添加 external、构建索引。

##### Reducer 配置 (reducer)

* reduce_type: 选择的 reduce 类型，可以直接通过工厂类获取对应 reducer 对象。
* reduce_output_format: 输出格式，默认仅支持 dataframe 格式输出。
* is_save_reduce: 是否保存 reduce 后的结果。
* reduce_save_dir: 保存全部 instance schema 的根路径。实际完整路径为 reduce_save_dir / 数据集标识符 / data_id.csv。

##### Parser 配置 (parser)

* parse_type: 模式提取的方法类型。若 Task 缺少该参数且需要用到 parser 时，默认使用该参数。
* is_save_parse: 是否保存 parse 后的结果。
* parse_save_dir: 保存全部 schema linking 的根路径。实际完整路径为 parse_save_dir / 数据集标识符 /
  data_id.txt。若该参数不提供，则默认保存在数据样本字典键值对中，键名为 "schema_links"。
* parse_output_format: Parser 的输出格式。

##### Generator 配置 (generator)

* generate_type: 用于决定系统提供的 Text-to-SQL 模型的类别。
* default_generator: 当使用系统提供的 generator 且未指定 generate_type 时，由系统配置提供。
* is_save_generate: 是否保存生成结果。
* generate_save_dir: 保存生成结果的路径。

##### 任务配置 (task)

* task_meta: 字典或列表。提供所有的任务定义。若提供单个字典，则仅有一个任务被执行；若需要多个任务，则需要提供所有任务元数据的列表。
    * task_id: 任务标识符，用户可定义但需确保不能重复。若不提供，则由系统自动生成。
    * task_name: 简要的任务名称，用于保存日志和打印信息。
    * task_info: 简要介绍任务的相关内容，便于保存日志。
    * task_type: 需要执行的任务类别。
    * data_id: 需要的数据标识符。若只有一个 data_source 和 schema_source，则可不提供该参数。
    * schema_id: 数据库模式的标识符。
    * eval_type: 评估任务的列表。只有数据集提供了标签才能进行评估。
* default_log_save_dir: 默认的日志保存路径，默认为 ../files/logs。
* is_save_dataset: 是否保存 dataset。对所有 Task 统一设置。
* open_parallel: 是否通过多并发的方式启动 run 方法。对所有 Task 统一设置。
* max_workers: 最大并发数量，通常小于数据集的长度。对所有 Task 统一设置。
* cpx_task_meta: 负责复杂任务定义。

##### Engine 配置 (engine)

* exec_process: 多个 Task 执行的列表。若不指定，则按照 task_meta 中的顺序串行执行每个 Task 任务。若不指定 Task
  ID，则默认使用 "task_" + task_meta 的索引。

##### 凭证配置 (credential)

* credential: 字典类型，保存远程连接数据库参数的 JSON 文件路径。
    * big_query: BigQuery 数据库的凭证文件路径。
    * snowflake: Snowflake 数据库的凭证文件路径。

### 2. DataLoader - 数据管理器

DataLoader 负责为 Text-to-SQL 流程准备数据。它同时接受查询数据和模式信息（推荐给出文件地址）。DataLoader
是统一的数据加载类，支持经典数据的测试集、验证集、用户提交的本地测试数据集以及单个数据样本的输入。它的生命周期应为一次完整的
Text-to-SQL 任务，在任务开始前创建，在预测完成后释放。DataLoader 不会实际存储任何数据，若用户提供了具体数据和
schema，系统会在配置文件默认路径下创建文件并临时存储。仅支持特定格式 Data 和 Schema 格式的输入，若格式错误则直接报错。

#### 初始化

```python
from core.data_manage import DataLoader
from core.base import Router

router = Router(config_path="config/demo_config.json")
dataloader = DataLoader(router)  # DataLoader 支持 Router 或显式传入参数创建
```

#### 主要方法

##### 数据集集成

```python
# 生成数据集
dataset = dataloader.generate_dataset(
    data_source_index="spider_dev",  # 数据源标识符
    schema_source_index="spider_dev",  # 模式源标识符
    random_size=0.1,  # 随机采样比例，例如：0.1 表示随机采样10%的数据
    filter_by="has_label"  # 过滤条件，例如：只包含有标签的数据
)
```

##### 添加 Few-shot 示例

few-shot 示例来自 QueryFlow 提供的思维链样本库。添加 few-shot 示例会将其地址添加至单个数据样本的键值对中。

```python
# 添加 few-shot 示例
dataloader.add_few_shot(
    source_index="spider_dev",  # 数据源标识符
    few_shot_num=3,  # 要添加的 few-shot 示例数量
    few_shot_save_dir="files/reasoning_examples/user"  # few-shot 示例的保存目录
)
```

##### 添加外部知识

若数据样本由用户提供，则必须在数据样本键值对中指定外部知识文件地址，键名为 external_path。若不存在 external_path
参数，则跳过该样本的添加过程。

```python
# 添加外部知识
dataloader.add_external(
    source_index="spider_dev",  # 数据源标识符
    external_save_dir="files/external"  # 外部知识的保存目录
)
```

##### 构建向量索引

根据用户是否提供 schema 文件、schema 文件的地址、是否需要建立索引（取决于Schema
Reducer）、使用哪种索引建立方式（目前可能只支持默认，即使用文本嵌入模型在文件目录建立索引，并使用具体的解析方法加载）。

```python
# 构建向量索引
dataloader.build_index(
    source_index="spider_dev",  # 模式源标识符
    embed_model_name="BAAI/bge-large-en-v1.5"  # 用于构建索引的嵌入模型名称
)
```

### 3. Dataset - 数据集类

Dataset 封装了单个任务运行所需的数据集和数据库模式。

#### 初始化

```python
from core.data_manage import Dataset

dataset = Dataset(
    data_source="benchmarks/spider/dev/dataset.json",  # 数据源文件路径
    schema_source="benchmarks/spider/dev/schema.json",  # 数据库模式文件路径
    multi_database=False,  # 是否为多数据库模式
    vector_store="vector_store",  # 向量存储路径
    embed_model_name="BAAI/bge-large-en-v1.5"  # 嵌入模型名称
)
```

#### 主要属性

* dataset_dict: 存储数据集内容的字典。
* schema_source: 模式源路径。
* is_multi_database: 布尔值，指示是否为多数据库模式。
* credential: 数据库凭证。
* database_path: 数据库文件存储路径。

#### 主要方法

```python
# 获取向量索引
vector_index = dataset.get_vector_index(item=0)  # item 为数据样本的索引或标识符

# 获取数据库模式
schema = dataset.get_db_schema(item=0)  # item 为数据样本的索引或标识符

# 保存数据
dataset.save_data("output/dataset.json")  # 保存数据集到指定路径
```

### 4. Engine - 执行引擎

Engine 定义了 Text-to-SQL 任务的一次运行，即根据输入数据进行预测。它负责创建并执行所有任务，收集并评估结果。Engine
的配置可以通过显式传入参数或传入 Router 对象。

#### 初始化

```python
from core.engine import Engine
from core.base import Router

router = Router(config_path="config/demo_config.json")
engine = Engine(router)
```

#### 主要方法

##### 执行任务

```python
# 执行所有任务
engine.execute()

# 跳过执行，仅评估 (通常在任务已经执行完毕并保存结果后调用)
engine.skip_execute()

# 评估结果
engine.evaluate(force=True)  # force=True 表示强制评估，即使任务未完成也会尝试评估
```

##### 任务管理

```python
# 获取任务ID列表
task_ids = engine.task_ids

# 根据ID获取任务
task = engine.get_task_by_id("demo_task")
```

### 5. Actor 组件

Actor 是 Squrve 的核心处理组件，包括 Reducer、Parser 和 Generator。它们是 Text-to-SQL 任务流中模式降维、模式链接和查询生成的核心实现。

#### 5.1 BaseReducer - 模式降维基类

Reducer 类用于根据样本对数据库模式进行降维，返回样本对应降维后的模式子集。它也支持根据 Router 从指定位置加载已存在的
Instance Schema。

```python
from core.actor.reducer.BaseReduce import BaseReducer


class CustomReducer(BaseReducer):
    OUTPUT_NAME = "instance_schemas"  # 定义输出结果在数据样本字典中的键名

    def act(self, item, schema=None, **kwargs):
        # 实现模式降维逻辑
        # item: 单个数据样本
        # schema: 数据库模式信息
        return reduced_schema  # 返回降维后的模式子集
```

#### 5.2 BaseParser - 模式解析基类

Parser 类用于根据问题从给定的模式中解析需要的模式信息，返回解析后的模式列表（字符串列表，<表名>.<列名>）。大多数方法都包含了
Schema Linking 组件，因此可能不需要额外定义新的 Schema Linking 组件。

```python
from core.actor.parser.BaseParse import BaseParser


class CustomParser(BaseParser):
    OUTPUT_NAME = "schema_links"  # 定义输出结果在数据样本字典中的键名

    def act(self, item, schema=None, **kwargs):
        # 实现模式解析逻辑
        # item: 单个数据样本
        # schema: 数据库模式信息
        return schema_links  # 返回解析后的模式列表
```

#### 5.3 BaseGenerator - 查询生成基类

QueryGenerator 类是所有 Text-to-SQL 方法的抽象，它会自动加载基线方法的元数据信息。能够使用的 Text-to-SQL
基线方法一方面来自学界已有的经典工作，另一方面可支持用户自定义创建。

```python
from core.actor.generator.BaseGenerate import BaseGenerator


class CustomGenerator(BaseGenerator):
    OUTPUT_NAME = "pred_sql"  # 定义输出结果在数据样本字典中的键名

    def act(self, item, schema=None, schema_links=None, **kwargs):
        # 实现SQL生成逻辑
        # item: 单个数据样本
        # schema: 数据库模式信息
        # schema_links: 模式链接信息
        return generated_sql  # 返回生成的 SQL 语句
```

### 6. Task 任务系统

Task 定义了单个 Text-to-SQL 任务。所有 Task 类必须显式实现 run() 方法，用于 Engine 调用。

#### 6.1 MetaTask - 元任务基类

BaseTask 是所有 Task 任务的基类，定义了 init 方法（接受 Router 参数）和抽象方法 run

```python
from core.task.meta.MetaTask import MetaTask


class CustomTask(MetaTask):
    NAME = "CustomTask"  # 任务的名称

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_actor(self, actor_type=None, **kwargs):
        # 加载对应的 Actor
        # actor_type: Actor 的类型
        pass

    def run(self):
        # 执行任务逻辑
        pass
```

#### 6.2 GenerateTask - 生成任务

Query Generation 任务，仅完成 Query Generation，必须验证。

```python
from core.task.meta.GenerateTask import GenerateTask
from llama_index.core.llms.llm import LLM

# 创建LLM实例
llm = LLM()  # 根据具体LLM类型创建

# 创建生成任务
generate_task = GenerateTask(
    llm=llm,  # LLM 实例
    generate_type="LinkAlign",  # 生成器类型
    save_dir="files/pred_sql"  # 生成结果的保存目录
)
```

#### 6.3 ComplexTask - 复杂任务

ComplexTask 用于定义复杂嵌套任务的必需条件。

```python
from core.task.meta.ComplexTask import ComplexTask

# 创建复杂任务
complex_task = ComplexTask(
    task_lis=["task1", "task2"],  # 元任务 task_id 列表
    meta_tasks={"task1": task1, "task2": task2},  # 元任务字典
    open_actor_parallel=True,  # TreeTask 内部在多个 Actor 任务执行时是否开启多进程并发
    max_workers=3  # 最大并发数量
)
```

### 7. LLM 集成

Squrve 支持多种 LLM 提供商，通过配置 api_key 和 llm 参数即可使用。

#### 7.1 QwenModel

```python
from core.llm.QwenModel import QwenModel

qwen_llm = QwenModel(
    api_key="your_api_key",
    model_name="qwen-turbo",  # Qwen 模型名称
    context_window=120000,  # 上下文窗口大小
    max_token=8000,  # 最大输出 token 数
    temperature=0.75  # 温度参数
)
```

#### 7.2 DeepseekModel

```python
from core.llm.DeepseekModel import DeepseekModel

deepseek_llm = DeepseekModel(
    api_key="your_api_key",
    model_name="deepseek-chat",  # Deepseek 模型名称
    context_window=120000,  # 上下文窗口大小
    max_token=8000,  # 最大输出 token 数
    temperature=0.75  # 温度参数
)
```

#### 7.3 ZhipuModel

```python
from core.llm.ZhipuModel import ZhipuModel

zhipu_llm = ZhipuModel(
    api_key="your_api_key",
    model_name="glm-4",  # Zhipu 模型名称
    context_window=120000,  # 上下文窗口大小
    max_token=8000,  # 最大输出 token 数
    temperature=0.75  # 温度参数
)
```

### 8. 配置文件格式

配置文件是 Squrve 快速启动和自定义 Text-to-SQL 工具的关键。

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
    "exec_process": [
      "spider_dev_task"
    ]
  },
  "credential": {
    "big_query": "path/to/big_query_credential.json",
    "snowflake": "path/to/snowflake_credential.json"
  }
}
```

#### 8.2 任务元数据格式

##### 简单任务 (Meta Task)

元任务定义是定义复杂嵌套任务的必需条件。

```json
{
  "task_id": "task_1", # 任务标识符，用户可定义但需确保不能重复
  "task_name": "Spider Dev Generation", # 简要的任务名称，用于保存日志和打印信息
  "task_info": "Generate SQL for Spider dev dataset", # 简要介绍任务的相关内容，便于保存日志
  "task_type": "generate", # 需要执行的任务类别
  "data_source": "spider:dev", # 需要的数据标识符
  "schema_source": "spider:dev", # 数据库模式的标识符
  "eval_type": ["exact_match", "execution"], # 评估任务的列表。传入字典或字符串使用 '.' 进行分割
  "log_save_path": "files/logs/task_1.log", # Logger 的保存路径。若不存在，则使用 Router 默认提供的日志存储路径
  "is_save_dataset": true, # 是否允许对 dataset 的修改和保存
  "dataset_save_path": "output/task_1_dataset.json", # 指定的 dataset 保存路径
  "open_parallel": false, # 是否开启多并发
  "max_workers": 1, # 最大线程数
  "meta": { # 额外参数，可包含创建 Dataset 以及 Task 所需的额外参数
    "dataset": { # 创建数据集时的额外参数
      "random_size": 0.1, # 随机采样数据大小
      "filter_by": "has_label" # 过滤条件
    },
    "llm": { # 任务自定义的 LLM 的配置参数
      "use": "qwen",
      "model_name": "qwen-turbo"
    },
    "task": {}, # Task 特定的额外参数
    "actor": {} # 被所有 actor 共享参数的容器，actor 指的是 generator, reducer 等
  }
}
```

##### 复杂任务 (Complex Task)

复杂任务定义允许更复杂的任务编排。

```json
{
  "task_id": "complex_task", # 任务标识符
  "task_name": "Complex Pipeline", # 简要的任务名称
  "task_info": "Multi-step Text-to-SQL pipeline", # 简要介绍任务的相关内容
  "task_lis": ["reduce_task", "parse_task", "generate_task"], # 元任务 task_id 列表
  "eval_type": ["exact_match"], # 评估任务的列表
  "log_save_path": "files/logs/complex_task.log", # Logger 的保存路径
  "is_save_dataset": true, # 是否允许对 dataset 的修改和保存
  "dataset_save_path": "output/complex_task_dataset.json", # 指定的 dataset 保存路径
  "open_parallel": true, # 是否开启多并发
  "max_workers": 3, # 最大线程数
  "open_actor_parallel": true, # TreeTask 内部在多个 Actor 任务执行时是否开启多进程并发
  "meta": {
    "task": {}, # Task 特定的额外参数
    "actor": { # 必须指定 task_id 才能生效
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

#### 8.3 exec_process 定义

exec_process 定义了多个 Task 的执行顺序。如果未指定，则按照 task_meta 中的顺序串行执行每个 Task 任务。

##### 字典格式定义

```json
{
  "type": "sequence",
  "tasks": [
    "task1",
    {
      "type": "parallel",
      "tasks": [
        "task2",
        "task3"
      ]
    },
    "task4"
  ]
}
```

* sequence 可以简写为 seq，parallel 可以简写为 para。

##### 列表格式定义：

```json
[
  "task_1",
  "task_2",
  [
    "task_3",
    "task_4",
    "~p"
  ],
  "~s"
]
```

* 在列表中，添加 *p 或 ~p 表示并发。
* 添加 *s 或 ~s 表示串行。

#### 8.4 基线数据集注册配置

用于注册 Squrve 支持的基线数据集。

```json
{
    "id": "spider", # 数据集标识符，例如：spider, bird, spider2, ambid_db
    "meta_info": "描述数据集的特点", # [Opt.] 描述数据集的特点
    "root_path": "files/benchmarks/spider", # 对于 DataLoader 的相对存储路径
    "db_type": "sqlite", # [Opt.] 数据库类型。若不存在子数据集，或所有子数据集在不同类别数据库上，否则需要设置该参数
    "has_sub": true, # 是否有子数据集，例如：spider-dev。若设置为 false, 则默认 dataset 保存在 root 路径下
    "external": true, # 是否提供公共的 external。若设置为 true, 则默认 external 保存在 root/external 目录下
    "database": true, # 是否提供公共的 database。若设置为 true, 则默认 database 保存在 root/database 目录下
    "sub_data":[
    	{
    		"sub_id": "dev", # 子数据集标识符，例如：dev / lite
    		"use_local_database": true, # [Opt.] database 文件是否在子数据集目录下
    		"use_local_external": false, # [Opt.] 是否使用本地的 external 目录
    		"db_type": "sqlite", # [Opt.] 子数据集的数据库类型，str 或 List
    		"has_label": true # [Opt.] 是否存在 query label 标记
		}
    ]
}
```

**基线数据集标识符**: 基线数据集标识符:子数据集标识符:筛选条件。

* 若不存在子数据集，且需要根据筛选条件挑选数据，则不能省略冒号，只需将子数据集标识符设置为空。
* 若使用筛选条件，则必须在数据样本字典中添加对应的键值对。不同筛选条件默认使用 . 作为分隔。可作为筛选条件的有：
    * db_size: 查询数据库的规模。例如：db_size-[m/l/e]-100。
    * difficulty: 样本的难度。例如：difficulty-easy。
    * db_type: 查询数据的类型。例如：db_type-sqlite。
    * ques_length: 问题的长度。例如：ques_length-[m/l/e/me/le]-200。
    * query_length: 查询的长度（需要提供 query 标记）。例如：query_length-[m/l/me/le]-200。
    * has_label: 是否存在 query 标记。默认为 query，可以为 schema_links 或自定义内容。

### 9. 数据格式规范

#### 9.1 数据集格式 (Data Row Specification)

Squrve 接受的数据行格式应遵循以下规范:

```json
{
  "instance_id": "unique_identifier", # 作为数据在当前基准数据集的唯一标识符
  "db_id": "database_name", # 查询的目标数据库
  "question": "自然语言问题", # 自然语言问题
  "db_type": "sqlite", # 目标数据库类型，例如：sqlite, big_query等
  "db_size": "medium", # 目标数据库规模
  "query": "SELECT * FROM table", # [Opt.] 标准 SQL 标记
  "gold_schemas": ["table1", "table2"], # 标准 SQL 使用的全部 db schemas
  "schema_links": ["table1.column1", "table2.column2"], # [Opt.] 标准模式链接文件路径
  "external_path": "path/to/external/knowledge.txt", # [Opt.] 存储外部知识源文档的路径
  "external": "extracted knowledge content", # [Opt.] 已提取知识的存储路径
  "reasoning_examples": "path/to/reasoning/examples.txt", # [Opt.] 采样思维链样本存储路径
  "instance_schemas": "path/to/instance/schemas.csv", # [Opt.] reduce 后的 schema 文件存储路径
  "pred_sql": "predicted SQL query" # [Opt.] 预测后的 SQL 语句
}
```

#### 9.2 数据库模式格式 (Schema Database Specification)

QueryFlow 可接受下面两种数据库 schema 格式：

##### Central 格式

绝大多数数据集 schema 的标准格式，以字典形式提供，但不利于划分和模式链接。

```json
[
  {
    "db_id": "Airlines", # 数据库名称
    "table_name": "aircrafts_data", # 表名
    "column_name": "aircraft_code",  # 列名
    "column_types": "character(3)", # 列数据类型
    "column_descriptions": "[Opt.]",  # [Opt.] 列描述
    "sample_rows": [
        "319",
        "321",
        "CR2",
        "320",
        "CN1"
    ], # [Opt.] 示例行，str 或 list[str]
    "table_to_projDataset": null  # [Opt.] 存放表归属的数据集
  }
]
```

##### Parallel 格式

LinkAlign 提供的新格式，所有字段平行，便于字段粒度管理。以列表形式提供。

```json
[
  {
    "db_id": "Airlines", # 数据库名称
    "table_name": "aircrafts_data", # 表名
    "column_name": "aircraft_code",  # 列名
    "column_types": "character(3)", # 列数据类型
    "column_descriptions": "[Opt.]",  # [Opt.] 列描述
    "sample_rows": [
        "319",
        "321",
        "CR2",
        "320",
        "CN1"
    ], # [Opt.] 示例行，str 或 list[str]
    "table_to_projDataset": null  # [Opt.] 存放表归属的数据集
  }
]
```

### 10. 使用示例

#### 10.1 基本使用流程

```python
from core.base import Router
from core.engine import Engine

# 1. 创建配置管理器
router = Router(config_path="startup_run/spider_dev_config.json")

# 2. 创建执行引擎
engine = Engine(router)

# 3. 执行任务
engine.execute()

# 4. 评估结果
engine.evaluate()
```

#### 10.2 自定义任务

用户可以通过继承 MetaTask 基类并实现 load_actor 和 run 方法来自定义任务。

```python
from core.task.meta.MetaTask import MetaTask
from core.actor.generator.LinkAlignGenerate import LinkAlignGenerator


class CustomGenerateTask(MetaTask):
    NAME = "CustomGenerateTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_actor(self, actor_type=None, **kwargs):
        # 根据 actor_type 加载对应的 Actor
        return LinkAlignGenerator(
            dataset=self.dataset,
            llm=self.llm,
            **kwargs
        )

    def run(self):
        # 自定义执行逻辑，例如调用 actor 的 act 方法
        pass
```

#### 10.3 多任务并行执行

Squrve 支持通过 ParallelTask 实现多任务并发执行。

```python
from core.task.multi.ParallelTask import ParallelTask

# 创建并行任务
parallel_task = ParallelTask(
    tasks=[task1, task2, task3], # 任务列表
    open_parallel=True, # 是否开启多并发
    max_workers=3 # 最大并发数量
)

# 执行并行任务
parallel_task.run()
```

### 11. 评估系统
Squrve 支持多种评估指标，并且所有 eval_function 都有一个唯一的标识。评估结果会以字典的方式保存在 Task 中。

* exact_match: 精确匹配。
* execution: 执行正确性。
* reduce_recall: 模式降维召回率。
* parse_accuracy: 模式解析准确率。

### 12. 日志系统
Squrve 使用内置的日志系统记录执行过程，便于进行错误检查。
```python
from core.log import Logger

logger = Logger(save_path="files/logs/execution.log")  # 日志保存路径
logger.info("Task started")  # 记录信息日志
logger.error("Error occurred")  # 记录错误日志
logger.debug("Debug information")  # 记录调试日志
```