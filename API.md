# Query

## Introduction

 Squrve 是一个支持快速开发 “端到端” Text-to-SQL 模型或对现有模型进行快速评估的工具。Text-to-SQL 任务流封装了多种组件，支持模式降维（schema reduce）、模式链接(schema linking)、查询生成（query generation）的快速实现。仅依靠配置文件即可快速启动 Squrve，并可轻松实现复杂任务嵌套、多任务并发执行。

## Base Model

### Router

负责管理 Text-to-SQL 的全部处理流程的参数配置，只能通过 Json 文件或者显示传入参数创建。

是命令行参数进行创建，或者显示创建 Engine 传入参数。配置的存在是必要的，若不存在，则自动加载 Demo 的参数配置。

* 初始化阶段，自动加载系统配置（例如：临时文件的存储位置，当 Router 不存在的静态参数文件地址）
* 支持通过配置文件和显示传入参数的方式进行参数配置。

### DataLoader

Prepare the data for the Text-to-SQL process. 同时接受查询数据以及模式信息（推荐给出文件地址）

(比如：这里可提供若干适配不同数据库规模、类型的，因此需要有一个路由器（Router）用来管理完整的处理流程（可以映射为一个 json 配置文件。配置的存在是必要的，若不存在，则自动加载 Demo 的参数配置）

[Usage]

* 创建 DataLoader 类，统一的数据加载类，无论是提出新方法的经典数据的测试集、验证集，或者是用户提交的本地测试数据集，又或者是单个数据样本的输入，都能得到支持。（我们可以假设所有输入的数据都是一个字典元素的列表）
  
  * 经典数据集相关参数，应该添加系统配置文件（例如：sys_config.json，Router 初始化时自动读取，并添加 Spider 等数据集信息）
* DataLoader 的生命周期应该是一次完整的Text-to-SQL 任务，在任务开始前创建，在预测完成后释放。
* 根据用户是否提供 schema 文件，schema 文件的地址，是否需要建立索引（取决于Schema Reducer）, 使用哪种索引建立方式（很长一段时间可能只能支持默认，也就是使用文本嵌入模型在文件目录建立索引，并使用具体的解析方法加载）

* 不会实际存储任何数据，比如：用户可能提供了具体的数据和 schema，则在系统配置文件默认路径下创建文件，并临时存储。（其实并不推荐这种方式）

* 仅支持特定格式 Data 以及 Schema 格式的输入，也就是需要进行格式判断，若错误，则直接报错。

* 支持 Router 或者显示传入参数创建

### Actor

#### Reducer

Basic components are Step One and Step Two from LinkAlign. (添加其他方法技巧，比如：跳过检索，后检索，指数衰减随机保留)

创建 Reducer 类，用于根据样本对数据库模式进行降维，返回样本对应降维后的模式子集，也支持根据 Router，从指定位置加载已经存在的  Instance Schema ，支持 Json / Dataframe 两种格式

[Usage]

* Reducer类：可以向下获取各层参数，必须实现具体的 reduce 方法
  * BaseReducer：所有 Reducer 的基类
  * ZeroReducer：将全部数据库模式作为  Instance_Schema

* 最终应该支持返回全部样本的 Instance Schema，也可以支持查询单个样本的 Instance Schema
* 封装了多种 Schema Reducer 流程，能够支持根据配置文件进行调用

#### Parser

[Optional Layer] 大多数方法都包含了 Schema Linking 组件，因而可能不需要额外定义新的 Schema Linking 组件

实现 Parser 类，用于根据问题从给定的模式中解析需要的模式信息，返回解析后的模式列表（字符串列表，<表名>.）。

[Usage]

* 若 Text-to-SQL 使用 LinkAlign 方法，则必须显示定义 Schema Linking 层
* 返回所有样本的 Schema Linking 组件

#### Generator

* DIN-SQL

* LinkAlign (Adapted version of DIN-SQL)

* Mac-SQL

  ......

[Usage]

* 创建 QueryGenerator 类，所有 Text-to-SQL 方法的抽象，自动加载基线方法的元数据信息（工具自带的方法必须提供，默认可为空，用于Router 的初始化）。
* 能够使用的 Text-to-SQL 基线方法，一方面来自学界已有的经典工作，另一方面可支持用户自定义创建（支持新的 Text-to-SQL 方法的快速开发、搭建和测试）。涉及到 Text-to-SQL  基线方法复现，可能有必要找人帮忙

* QueryGenerator 类必须能够支持三种经典数据集的测试

#### PipelineActor

#### TreeActor

### Task（全局）

定义单个 Text-to-SQL 任务。

[Usage]

* Text-to-SQL 的任务有：

  * BaseTask ( 所有Task 任务的基类)，定义 init 方法（接受 Router参数）和抽象方法 run
  
  * DataPrepareTask (支持预先完成全部的 data prepare 工作，包括添加 few_shot / external / build index)
  
  * Schema Reduce（仅完成 Schema Reduce）：可配置评估参数
  * Schema Linking (仅完成 Schema Linking)
  * Query Generation (仅完成Query Generation，必须验证)
  * Text-to-SQL (包含 Schema Reduce，Schema Linking 和 Query Generation)
  * PipelineTask（传入一个 Task 列表，串行执行所有任务）
  * ParallelTask（传入一个 Task 的列表，通过多进程执行所有任务）
  * 其他用户自定义 Task。继承 BaseTask 基类，实现 run 方法
* 所有 _Task 类必须显示实现 run() 方法，用于 Engine 调用

  * Router 对象（init 方法传入）
  * 具体参数 (init 方法传入)



### Engine (全局)

定义了 Text-to-SQL 任务的一次运行，也就是根据输入数据进行预测。

[Usage]

* 配置：显示传入参数或者传入 Router 对象，若两者存在冲突，则以 Router 对象为主，实际上，若 Router 对象不存在时才会显示通过参数创建新的 Router，后续流程完全基于 Router 管理。
* 定义配置生效的  init 方法：根据参数进行 DataLoader、SchemaReducer、SchemaLinking（Option）、QueryGenerator，以及 Task 对象（执行顺序参数由 Router 提供，例如：task1 和 task2 串行即为[task1, task2]，task1 和 task2 并行，即为 [[task1, task2]]，也就是将并行内容放在列表 / 元组 / 集合中）
* 运行 Text-to-SQL 任务：最终应该只有唯一的 task 对象，运行 task 任务，并保存运行结果
* [后续补充] Engine 运行日志，便于进行错误检查



### LLM （全局）



### Output

[Usage]



### Config 参数汇总

* api_key
  * deepseek
  * qwen
  * zhipu
  
* credential：保存远程连接数据库参数的 Json 文件路径的字典

  * big_qury
  * snowflake

* llm
  * use
  * model_name
  * context_window
  * max_token
  * temperature
  * top_p
  * time_out
  
* text_embed（仅支持Hugging Face 的文本嵌入模型）
  
  * embed_model_source: 目前仅支持 ”Hugging Face“ 模型

  * embed_model_name
  
* dataset (数据样本)

  * **data_source**：字符串 / 字符串列表 / 字典（后两者是为了适配多任务场景，定义任务需要添加具体的 dataset 标识符，也就是索引或者键）。不推荐字典的键与系统添加经典数据集真实路径使用的键一致（也就是 task id）

    * data_source 可以由用户提供，也可以是经典的基线数据集。
    * 如果用户定义的任务中使用了经典基线数据集，则不需要显示指定 data_source，但不显示指定data_source ，则无法添加 few_shot 以及 external。
    * 如果用户希望对经典基线数据集也添加 few_shot 以及 external，datasource 应传入“经典数据集标识符:子数据集名称:筛选条件"，使用 : 作为分隔，作为实际保存路径的替代，真实保存路径将在 dataloader 初始化阶段，由系统自动替换。
    * 如果用户任务使用了经典基线数据集但并未显示指定 data_source，则任务开始前的检查阶段，会自动将经典数据集的 datasource 添加值 data_source。但任务检查将会在 dataloader 初始化完成之后，因此不会执行添加 few_shot 等操作。
    * 另外，对于经典数据集来说，将不会直接对底层保存的数据文件进行操作，而是将数据文件副本保存至 default_data_source_dir 目录下，并将该路径替换

  * data_source_dir：系统默认自动保存 data_source 文件的目录，由系统配置提供。若用户并未传入data_source ，而是数据本身，则默认保存在该路径下。对于经典数据集，文件名称为 “基线数据集标识符_子数据集标识符.json”

  * default_data_file_name：若用户并未传入data_source ，而是数据本身，默认保存的文件名称。

  * overwrite_exist_file：若 data_source_dir 中出现同名文件是否覆盖原文件，默认覆盖


  * db_path：本地数据库存储路径

  

  * **need_few_shot**：是否需要添加 few_shot 示例（来自QueryFlow 提供的思维链样本库），当然添加 few_shot 需要对单个数据列表操作，检索成功后会将地址添加至单个数据样本的键值对中。另外，添加 few_shot 应该在 dataloader中完成。

  * few_shot_key_name：该参数省略，键值对命名规范，后续将统一整理。few_shot 键名为 “reason_examples"，由系统配置提供。因此，Dataloader 必须传入 Router，若不传入，则内部必须自动创建。

  * few_show_range：添加思维链示例的 data source 范围，接受 bool, str, int, List[int], List[str]

  * few_shot_num：思维链提示的数量。必须是一个大于 1 的整数，否则撤销添加操作，只有 need_few_shot 为真时生效。整数 / 列表 / 字典（默认为一个整数参数，对所有 datasource 生效）

  * few_shot_save_dir：所有数据集 few_shot 保存的根路径，但并不等于实际保存路径（实际完整路径应为 few_shot_save_dir /  数据集标识符 / data_id.txt，其中数据集标识符和 data_id 在添加过程动态提供），由系统配置提供

  * sys_few_shot_dir：系统提供 few-shot 示例的路径

    

  * **need_external**：是否需要添加外部知识。

    * 若数据样本由用户提供，则必须在数据样本键值对中指定外部知识文件地址，键名必须为 external_path
    * 若不存在 exertanl_path 参数，则跳过该样本的添加过程，external 加载完毕后会自动添加至 单个数据样本的键值对中，键名为 external_key_name。用户可以直接不提供 exertanl_path 而仅提供 external 参数。
    * 若用户能够自主完成 external 添加过程，则该参数不提供或者设置为False.
    
  * default_get_external_function：默认添加 external 的方法，由系统配置提供。该方法旨在提取单个样本对应的外部知识。若用户未传入自定义的 external 提取方法，则该参数生效。

  * external_range：一个 data_source 标识符列表，保存所有需要external 的数据集，若data_source 为单个字符串或者仅有一个元素，则该参数失效，可不进行设置。默认为全部 data_source 。

  * external_save_dir：默认系统保存  external 文件根目录，但并不等于实际保存路径（实际完整路径应为 external_save_dir/ 数据集标识符 / data_id.txt，其中数据集标识符和 data_id 在添加过程动态提供），由系统配置提供。

  * external_key_name: 该参数省略，键值对命名规范，后续将统一整理。external 键名为 “external"，由系统配置提供。因此，Dataloader 必须传入 Router，若不传入，则内部必须自动创建。实际上，external 和 external_path 可完全相同

* database（数据库模式）

  * skip_schema_init：直接跳过 schema 初始化。
    * 若用户希望直接跳过 schema 初始化，直接存放原始文件处理后续流程，将该参数必须提供
    * 由于schema init 的目的是为了将 schema 格式由 central 转变为parralle，若用户希望以 central 文件进行后续操作，则必须提供该参数，并设置为 True
    * schema init 过程若遇到目录路径将直接跳过，并直接将最后一个目录名作为标识符，添加至data save source 路径
    * schema init 仅针对central 格式的 .json 文件进行，判断两种格式的区别在于元素是 List[Dict], 还是 Dict
    * 跳过 schema init 则直接将 schema source 添加至 schema_save_source
  * 若不进行初始化，则需要确保传入参数中 schema source 和 multi_database 的一致性
  * schema_source: 字符串 / 字符串列表 / 字典（后两者是为了适配多任务场景，定义任务需要添加具体的 dataset 标识符，也就是索引或者键）。不推荐字典的键与系统添加经典数据集真实路径使用的键一致（也就是 task id）
    
    * schema_source 可以由用户提供，也可以是经典的基线数据集。
    * schema_save_source 为 <schema_save_dir> / <schema_index>
  * multi_database: 保存 multi_database 设置下的 schema_source 标识符列表。

    * 若multi_database 为真，则处理schema source文件将所有.json 文件放置在单一目录下。真实存储路径为 schema_source_dir / schema 标识符 
    * 若 schema_source 为字符串，multi_database 参数仅允许提供布尔值或者字典对象
    * 若 schema_source 为列表，multi_database 参数允许提供布尔值、列表或者字典对象，但列表对象必须与 schema_source 等长
    * 若 schema_source 为字典，multi_database 参数允许提供布尔值或者字典对象
  * vector_store：字符串 / 列表 / 字典。若为列表，则必须与 schema_source 长度相等，也就是必须为每一个 schema_source 提供 vector_store 路径。若为字典对象，则仅提供需要建立索引的 schema_source 标识即可。若该参数不提供，则默认为每个 “schema_source / vector_store” 目录。
    * 若 vector_store 为绝对路径。在 multi_db 情况下，索引保存路径为 `vector_store / <schema 标识符> / multi_db / <embed_model_name>`，在single_db情况下，索引保存路径为 `vector_store / <schema 标识符> / single_db / <db_id> / <embed_model_name>`
    * 若 vector_store 为相对路径，则默认保存在数据库元数据存储目录下 ，也就是 `<schema source> / <embed_model_name>`。相对路径的最后一个目录为有效目录
  * schema_source_dir：系统默认自动保存 schema_source 处理后文件的目录，由系统配置提供。若用户并未传入schema_source ，而是数据库模式本身，则默认保存在该路径下。对于经典数据集，文件名称为 “基线数据集标识符_子数据集标识符.json”
  * default_schema_dir_name：若用户并未传入schema_source 目录，而是schema本身，默认保存的处理后文件夹名称。（注意，保存的不是 schema 文件，而是处理后的若干.json 文件）
  * need_build_index。变量为真，则开启索引创建。只要需要创建索引，则必须将该参数设置为真。
  * index_method：仅支持 “VectorStoreIndex” 方法，默认由系统配置提供。
  * index_range：一个 schema_source 标识符列表，提供需要建立索引的列表。不指定该参数，则默认为全部 schema_source

* router

  * use_demo：是否使用 demo 配置文件，若用户未传入配置文件，且该参数为真时，使用 demo 配置文件。默认由系统配置提供，为False

* dataloader

  * is_prepare_data：是否在 dataloader 初始化时自动进行数据准备。prepare data 包括了
    * add few_shot
    * add external
    * build_index

* reducer (Instance Schema 是包含 ground truth schema 在内的 Database Schema 的子集)

  * reduce_type：选择的  reduce 类型，可以直接通过工厂类获取对应 reducer 对象
  
  * reduce_output_format：输出格式，默认仅支持 dataframe 格式输出。
  
  * is_save_reduce：是否保存
  * reduce_save_dir：保存全部 instance schema 的根路径，但并不等于实际保存路径（实际完整路径应为 reduce_save_dir/  数据集标识符 / data_id.csv，其中数据集标识符和 data_id 在添加过程动态提供）
  * 其他关键参数，后续补充。（包括使用经典数据集评估 reduce 的召回率等）
  
* parser

  * parse_type：模式提取的方法类型。实际上，该参数独立于 Task，若 Task 缺少该参数，且需要用到 parser 时，默认使用该参数。
  * is_save_parse：是否保存
  * parse_save_dir：保存全部 schema linking 的根路径，但并不等于实际保存路径（实际完整路径应为 parse_save_dir/  数据集标识符 / data_id.txt，其中数据集标识符和 data_id 在添加过程动态提供）。若该参数不提供，则默认保存在数据样本字典键值对，键名为 ”schema_links“
  * 其他关键参数，后续补充。（包括使用经典数据集评估 schema linking 的各种参数）
  
  * parse_output_format
  
* generator

  * generate_type：用于决定系统提供的 Text-to-SQL 的模型的类别。
  * default_generator：当使用系统提供的 generator 且未指定 generate_type 时，由系统配置提供。
  * is_save_generate：是否保存
  * generate_save_dir：保存 generate_save_dir 的路径
  * 其他关键参数，后续补充。（比如：使用多线程进行加速）

* task

  * task_meta：字典 / 列表。提供所有的任务定义。若提供单个字典，则仅有一个任务被执行，若需要多个任务，则需要提供所有任务元数据的列表。单个任务元数据如下所示：

    ```json
    {
        "task_id": <任务标识符>, - [Opt.]用户可定义 task_id 但需确保不能重复，否则 task 检查阶段报错。若不提供，则由系统自动生成 
    	"task_name": <任务名称>, - [Opt.]简要的任务名称，用于保存日志和打印信息
        "task_info": <任务信息>, - [Opt.]简要介绍任务的相关内容，便于保存日志
        "task_type": <任务类别>, - 需要执行的任务类别
        "data_id": <数据标识符>,  - [Opt.] 需要的数据，若仅有一个 datat_source 和 schema_source，则可不提供该参数
    	"schema_id": <数据库模式的标识符>,  -[Opt.] 
        "eval_type": [评估任务的列表], -[Opt.] 只有数据集提供了标签才能进行评估。
        - 其他关键参数，例如
    }
    ```

    

  * 其他关键参数，后续补充。（比如：任务执行完成后的回调函数，可以增加输出内容的参数）

  * default_log_save_dir：默认的日志保存路径，默认为 ../files/logs

  * is_save_dataset：是否保存 dataset。【对所有 Task 统一设置】

  * open_parallel：是否通过多并发的方式启动 run 方法。 【对所有 Task 统一设置】

  * max_workers：最大并发数量，往往小于 dataset 的长度 【对所有 Task 统一设置】

  * cpx_task_meta：负责任务定义。示例见 Complex Task 任务定义配置。

* engine

  * exec_process：多个 Task 执行的列表，若不指定，则按照 task_meta 中的顺序串行执行每个 Task任务。若不指定 Task ID, 则默认使用 “task_” + task_meta 的索引

  



### Task 任务定义配置

【Meta Task】元任务定义：一般任务定义，也是定义复杂嵌套任务的必需条件。 

```json
{
    "task_id": <任务标识符>, - [Opt.]用户可定义 task_id 但需确保不能重复，否则 task 检查阶段报错。若不提供，则由系统自动生成 
	"task_name": <任务名称>, - [Opt.]简要的任务名称，用于保存日志和打印信息
    "task_info": <任务信息>, - [Opt.]简要介绍任务的相关内容，便于保存日志
    "task_type": <任务类别>, - 需要执行的任务类别,【引入结构化的任务定义】
    "data_source": <数据标识符>,  - [Opt.] 需要的数据，若仅有一个 datat_source 和 schema_source，则可不提供该参数
	"schema_source": <数据库模式的标识符>,  -[Opt.]
	"eval_type":[], - 传入字典,或者字符串使用 . 进行分割
	"log_save_path": <Logger 的保存路径> -[Opt.] 若不存在，则使用 Router 默认提供的日志存储路径,
	"is_save_dataset": <是否允许对 dataset 的修改和保存>,
	"dataset_save_path": <指定的 dataset 保存路径>
	"open_parallel": <是否开启多并发；>,
	"max_workers": <最大线程数>,
	"meta":{
        "dataset":{...}, -[Opt.] 创建数据集时的额外参数
        "llm": {...}, - 任务自定义的 LLM 的配置参数
        "task":{...}, - meta以外是所有 Task 共享的参数，此处定义特定类需要的额外参数
		"actor":{}, -[Opt.] 一个被所有 actor 共享参数的容器，actor 指的是 generator, reducer等
    }
}
```

* Engine 负责以列表的方式扁平化存储所有的 Task 对象。

  * 首先检查 Task，比如:

    * 是否使用了相同的 task_id; 

    * task_type 是否合法；

    * data_source 和 schema_source是否合法，

      * 若合法需要检查是否使用了基准数据集，并对DataLoader 进行更新

      * 若不指定 data_source，则需要判断 router 中的 data_source 是否唯一

    * eval_type 进行初始化为列表

    * 创建 Task 对应的 Dataset

  * 接着，创建 Task 对象，并根据 exec_process 添加至 Engine 中

* Task 有个重要的参数

  * is_end：是否已经完成运行
  * 当 Task 运行完毕后，由Engine 负责开启 Task 的评估。评估时会先判断是否 is_end，若已完成则开启评估

* Engine 有两个重要的方法，分别是 start() 用于开启所有的 run ，和 eval() 对所有的任务进行评估。若 Task 为 PipelineTask 或者 PrallelTask 则只调用保存的全部 Task的评估方法

* 评估后的结果会以字典的方式保存在 Task 中

* task type。所有 task_type 和对应 Task 对象的映射应在 Engine 中进行注册。用户自定义的 Task 必须只能通过 Task 参数传入

  * data_prepare: DataPrepareTask (支持预先完成全部的 data prepare 工作，包括添加 few_shot / external / build index)
  * reduce: Schema Reduce（仅完成 Schema Reduce）：可配置评估参数
  * parse: Schema Linking (仅完成 Schema Linking)
  * generate：Query Generation (仅完成Query Generation，必须验证)
  * nl2sql：Text-to-SQL (包含 Schema Reduce，Schema Linking 和 Query Generation)
  * sequnce：SequenceTask（传入一个 Task 列表，串行执行所有任务）
  * parallel: ParallelTask（传入一个 Task 的列表，通过多进程执行所有任务）
  * 其他用户自定义 Task。继承 BaseTask 基类，实现 run 方法

* meta 可包含创建 Dataset 以及 Task 所需的额外参数：

  【Dataset】可包含的参数有 

  * random_size
  * filter_by
  * multi_database
  * vector_store
  * embed_model_name
  * db_credential
  * db_path
  * is_schema_final

### Complex Task 任务定义配置

【Complex Task】 Complex Task 定义

```json
{
    "task_id": <任务标识符>, - [Opt.]用户可定义 task_id 但需确保不能重复，否则 task 检查阶段报错。若不提供，则由系统自动生成。若使用 exec_process，则 task_id必须定义
	"task_name": <任务名称>, - [Opt.]简要的任务名称，用于保存日志和打印信息
    "task_info": <任务信息>, - [Opt.]简要介绍任务的相关内容，便于保存日志
    "task_lis": <元任务 task_id 列表>, 
	"eval_type":[], - 传入字典,或者字符串使用 . 进行分割
	"log_save_path": <Logger 的保存路径> -[Opt.] 若不存在，则使用 Router 默认提供的日志存储路径,
	"is_save_dataset": <是否允许对 dataset 的修改和保存>,
	"dataset_save_path": <指定的 dataset 保存路径>
	"open_parallel": <是否开启多并发；>,
	"max_workers": <最大线程数>,
	"open_actor_parallel": <TreeTask 内部在多个 Actor 任务执行时是否开启多进程并发>,
    "meta":{
     
            "task":{...}, - meta以外是所有 Task 共享的参数，此处定义特定类需要的额外参数,
            "actor":{
                    "task_1":{...}
                   }, -[Opt.] 必须指定 task_id 才能生效等
                 
        }

}
```

* task_lis：代表任务层级和执行顺序的列表。我们提供两种解析 task_lis 的方式，分别对应两种不同的写法。由 sys_config 文件决定初始的解析方式
  * 一种是简单解析，默认 Text-to-SQL 不会涉及到复杂的嵌套结构，最多两层。因此外层列表代表 pipeline ,内层列表代表tree
  * 另一种是保留可扩展性的复杂解析，列表中包含 "~p" 或者 "*p"  代表 pipeline，反之则代表 tree
* 【Dataset】可包含的参数有 

  * multi_database
  * vector_store
  * embed_model_name
  * db_credential
  * db_path
  * is_schema_final

### exec_process 定义

例如：task1 和 task2 串行即为[task1, task2]，task1 和 task2 并行，即为 [[task1, task2]]，也就是将并行内容放在200 / 元组 / 集合中  

* 字典格式定义：

  ```json
  {
      "type": "sequence",
      "tasks": [
          "task1",
          {
              "type": "parallel",
              "tasks": ["task2", "task3"]
          },
          "task4"
      ]
  }
  
  ```

  * sequence 可以简写为 seq , parallel 可以简写为para
  * 这种写法很容易解析

* 列表

  ```json
  ["task_1","task_2",["task_3","task_4","~p"],"~s"]
  ```

  * 在列表中，添加 *p 或者 ~p 表示并发，添加 
  * 添加*s 或者 ~s 表示串行，定义串行和并行的定义标识符在类内写死。

### benchmark 基线数据集注册配置

```json
{
    "id": <数据集标识符，例如：spider>,  - spider / bird / spider2 / ambid_db
    "meta_info": <[Opt.] 描述数据集的特点>,
    "root_path": <对于 DataLoader 的相对存储路径>, 
    "db_type": <[Opt.] 数据库类型>, - 若不存在子数据集，或者，所有子数据集在不同类别数据库上，否则需要设置该参数
    "has_sub": <是否有子数据集，比如：spider-dev>, - 若设置为 false, 则默认 dataset 保存在 root 路径下
    "external": <是否提供公共的 external>, - 若设置为 true, 则默认 external 保存在 root/external 目录下
    "database": <是否提供公共的 database>, - 若设置为 true, 则默认 database 保存在 root/database 目录下
    "sub_data":[
    	{
    		"sub_id": <子数据集标识符>, - 例如：dev / lite
    		"use_local_database": <[Opt.] database 文件是否在子数据集目录下>, - true / false
    		"use_local_external": <[Opt.] 是否使用本地的 external 目录>
    		"db_type": <[Opt.] 子数据集的数据库类型，str or List>  - 由于数据库文件存储在子数据集文件下。若数据库文件不止一个，则数据文件除了必须指定查询的数据库id, 还需要指定数据库的类型也就是 db_type 参数。
    		"has_label"：<[Opt.] 是否存在 query label 标记>
		}
    ]
}
```

* 若 has_sub 为 false，则在 root 目录下保存 dataset.json / schema.json
* 默认注册后的基线数据集中，默认已经完成 db_type 的初始化
* 若需要外部文档，则必须提供准确的`外部知识文档名称（包括后缀）`，否则默认为 "" ，存放在 external_path 目录下
* 可支持的 db_type：sqlite, big_query (BigQuery), snowflake

### benchmark 基线数据集标识符

`基线数据集标识符`：`子数据集标识符`:`筛选条件`

* 若不存在子数据集，且需要根据筛选条件市挑选数据，则不能省略冒号，只需将子数据集标识符设置为空
* 若使用筛选条件，则必须在数据样本字典中添加对应的键值对，不同筛选条件默认使用 '.' 作为分隔，或指定分隔符可作为筛选条件的有
  * db_size：查询数据库的规模。db_size-[m/l/e]-100
  
  * difficulty：样本的难度。difficulty-easy
  
  * db_type：查询数据的类型。db_type-sqlite
  
  * ques_length：问题的长度 （不需要提供额外键值对）ques_length-[m/l/e/me/le]-200
  
  * query_length：查询的长度（需要提供 query 标记）query_length-[m/l/me/le]- 200
  
  * has_label：是否存在query 标记。has_label-[query] ，默认为 query ,可以为 schema_links，或自定义内容
  
    

### baseline 基线方法注册配置



### Dataset 数据行格式规范

```json
{
    "instance_id": <作为数据在当前基准数据集的唯一标识符>,
    "db_id": <查询的目标数据库>,
	"question": <自然语言问题>,
    "db_type": <目标数据库类型，例如：sqlite, big_query等>,
    "db_size": <目标数据库规模>,
    "query": <标准 SQL 标记>, -[Opt.]
	"gold_schemas": <标准 SQL 使用的全部 db schemas>
	"schema_links": <标准模式链接文件路径>, -[Opt]
    "external_path": <存储外部知识源文档的路径>,  -[Opt.]
    "external": <已提取知识的存储路径>, -[Opt.]
	"reasoning_examples": <采样思维链样本存储路径>, -[Opt.]
	"instance_schemas": <reduce 后的 schema 文件存储路径>,  -[Opt.]
	"pred_sql": <预测后的 SQL 语句> -[Opt.]
}
```



### Schema 数据库格式规范

QueryFlow 可接受下面两种数据库 schema 格式：

* central (绝大多数数据集 schema 的标准格式，但不利于划分和模式链接)，默认格式。以字典形式提供。

  ```json
  {
  	"db_id": <数据库名称>,
      "db_size": <数据库字段数量>,
      "db_type": <需要连接的数据库类型>
      "table_names_original": [], - 存放所有表元素
  	"column_names_original": [], - 存放所有字段元素,
  	"column_types": [],  - 存放所有字段的数据类型
  	"column_descriptions": [], - [Opt.]存放所有字段的类型描述
  	"sample_rows": [], -[Opt.] 
  	"table_to_projDataset":[], - [Opt.] 存放表归属的数据集
  }
  ```

  

* parallel (LinkAlign 提供的新格式，所有字段平行，便于字段粒度管理)，以列表形式提供。

  * db_size 是指将 schema 转换为 parallel 格式后，列表的长度 

  ```json
  [
      {
          "db_id": "Airlines",
          "table_name": "aircrafts_data",
          "column_name": "aircraft_code",  
          "column_types": "character(3)",
          "column_descriptions": "[Opt.]",  
          "sample_rows": [
              "319",
              "321",
              "CR2",
              "320",
              "CN1"
          ], -[Opt.] str / list[str]
          "table_to_projDataset": null  -[Opt.]
      }
  ]
  ```

  * parrallel 使用扁平化的字典结构，以便于直接转换为 pd.Dataframe 对象



### External Function 接口规范

`Input`

* question: str,
* llm: LLM,
* external_path: Union[str, PathLike] = None,
* external: str = None,  # 已读取的外部知识文档
* need_save: bool = True,
* save_path: str = None,

`output`

* Optional (summary) 。返回提取后的知识文档



### Eval Function 设计规范

* 所有 eval_function 都有一个唯一的标识，并且只能在 Base 类中被定义，能够直接通过该标识前往 Base 类获取对应的评估方法 。例如：`reduce_recall`，reduce 表示 BaseReduce 基类，recall 表示计算召回率
* Eval Function 由 Task 类负责调用，只有所有样本在所有流程结束后开启评估。



### Eval Results 格式规范

