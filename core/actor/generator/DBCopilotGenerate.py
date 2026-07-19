from typing import Union, List, Dict, Optional, Any
from os import PathLike
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.generator.BaseGenerate import BaseGenerator
from core.actor.base import ActorPool
from core.data_manage import Dataset
from core.utils import load_dataset, save_dataset

from core.actor.generator.dbcopilot.schema_graph import from_multi_db_schema
from core.actor.generator.dbcopilot.schema_router import SchemaRouter
from core.actor.generator.dbcopilot.schema_serializer import parse_squrve_text


@BaseGenerator.register_actor
class DBCopilotGenerator(BaseGenerator):
    """
    DBCopilot: Schema Routing + SQL Generation
    
    论文: arXiv:2312.03463, EDBT 2025
    
    核心流程:
    1. Schema Routing: 使用轻量级Router识别目标数据库和表
    2. SQL Generation: 复用现有NL2SQL方法生成最终SQL
    """
    
    NAME = "DBCopilotGenerator"
    
    SKILL = """# DBCopilotGenerator

DBCopilot使用两阶段架构处理多数据库Text-to-SQL任务：

## 核心能力
- **Schema Routing**: 从自然语言自动识别目标数据库和表
- **SQL Generation**: 基于路由结果生成SQL

## 适用场景
- 多数据库环境（如企业数据仓库）
- 大规模Schema场景
- 需要数据库级路由的复杂查询

## 输入
- `schema`: 多数据库Schema集合（包含多个db_id）
- `schema_links`: 可选，预计算的Schema链接

## 输出
`pred_sql`

## 执行步骤
1. 解析多数据库Schema结构
2. Schema Routing: 识别目标数据库和候选表
3. Schema Pruning: 构建精简的子Schema
4. SQL Generation: 使用底层生成器生成SQL
5. 返回最终SQL
"""
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm: Optional[Any] = None,
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "files/pred_sql",
        # DBCopilot特有参数
        router_type: str = "llm",           # 路由类型: "llm" | "embedding" | "keyword"
        top_k_databases: int = 3,            # 候选数据库数量
        top_k_tables: int = 5,               # 每个数据库候选表数量
        sql_generator: str = "DINSQLGenerator",  # 底层SQL生成器
        use_external: bool = True,
        db_path: Optional[Union[str, PathLike]] = None,
        credential: Optional[Dict] = None,
        **kwargs
    ):
        """
        初始化DBCopilot生成器
        
        Args:
            dataset: 数据集对象
            llm: LLM模型实例
            is_save: 是否保存输出
            save_dir: 保存目录
            router_type: Schema路由策略
            top_k_databases: 选择Top-K个候选数据库
            top_k_tables: 每个数据库选择Top-K个候选表
            sql_generator: 底层SQL生成器名称
            use_external: 是否使用外部知识
            db_path: 数据库路径
            credential: 数据库凭证
        """
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.router_type = router_type
        self.top_k_databases = top_k_databases
        self.top_k_tables = top_k_tables
        self.sql_generator_name = sql_generator
        self.use_external = use_external
        
        # 初始化数据库连接参数
        self.db_path = db_path or (self.dataset.db_path if self.dataset else None)
        self.credential = credential or (self.dataset.credential if self.dataset else None)
        
        # 延迟初始化底层生成器
        self._sql_generator = None
        
    def _get_sql_generator(self):
        """延迟初始化并返回底层SQL生成器实例"""
        if self._sql_generator is None:
            GeneratorClass = ActorPool.get_actor_by_name(self.sql_generator_name)
            if not GeneratorClass:
                raise ValueError(f"SQL Generator {self.sql_generator_name} not found!")
            self._sql_generator = GeneratorClass(
                dataset=self.dataset,
                llm=self.llm,
                is_save=False,
                save_dir=self.save_dir,
                use_external=self.use_external,
                db_path=self.db_path,
                credential=self.credential
            )
        return self._sql_generator
    
    def _schema_routing(
        self,
        question: str,
        multi_db_schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        graph = from_multi_db_schema(multi_db_schema)
        router = SchemaRouter(self.router_type, self.llm)
        return router.route(question, graph, self.top_k_databases, self.top_k_tables, multi_db_schema)
    
    def _extract_table_names(self, schema_info: Any) -> List[str]:
        table_names = []
        if isinstance(schema_info, list):
            for item in schema_info:
                if isinstance(item, dict):
                    table_name = item.get("table_name") or item.get("name")
                    if table_name:
                        table_names.append(table_name)
        elif isinstance(schema_info, dict):
            if "table_names_original" in schema_info:
                table_names = schema_info["table_names_original"]
            else:
                table_names = list(schema_info.keys())
        elif isinstance(schema_info, str):
            parsed = parse_squrve_text(schema_info)
            table_names = list(parsed.keys())
        return table_names
    
    def _prune_schema(
        self,
        multi_db_schema: Dict[str, Any],
        routing_result: Dict[str, List[str]]
    ) -> list:
        pruned = []

        for db_id, selected_tables in routing_result.items():
            if db_id not in multi_db_schema:
                continue

            schema_info = multi_db_schema[db_id]

            if isinstance(schema_info, dict) and "table_names_original" in schema_info:
                pruned.extend(self._prune_spider_central(schema_info, selected_tables))
            elif isinstance(schema_info, list):
                pruned.extend(self._prune_list_format(schema_info, selected_tables))
            elif isinstance(schema_info, dict):
                pruned.extend(self._prune_dict_format(schema_info, selected_tables))
            elif isinstance(schema_info, str):
                pruned.extend(self._prune_str_format(db_id, schema_info, selected_tables))

        if not pruned:
            first_db = list(multi_db_schema.keys())[0]
            schema_info = multi_db_schema[first_db]
            if isinstance(schema_info, dict) and "table_names_original" in schema_info:
                pruned = self._prune_spider_central(schema_info, self._extract_table_names(schema_info))
            elif isinstance(schema_info, list):
                pruned = schema_info

        return pruned

    def _prune_spider_central(self, schema_info: dict, selected_tables: List[str]) -> list:
        result = []
        table_names = schema_info.get("table_names_original", [])
        column_names_original = schema_info.get("column_names_original", [])
        column_types = schema_info.get("column_types", [])
        primary_keys = schema_info.get("primary_keys", [])
        foreign_keys = schema_info.get("foreign_keys", [])
        column_descriptions = schema_info.get("column_descriptions", [])

        selected_indices = set()
        for tbl_name in selected_tables:
            if tbl_name in table_names:
                selected_indices.add(table_names.index(tbl_name))

        has_star = column_names_original and column_names_original[0][0] == -1

        for idx, (table_idx, col_name) in enumerate(column_names_original):
            if table_idx == -1:
                continue
            if table_idx not in selected_indices:
                continue

            tbl_name = table_names[table_idx]
            col_type = column_types[idx] if len(column_types) > idx else "text"
            col_desc = column_descriptions[idx] if len(column_descriptions) > idx else None

            is_pk = col_name in primary_keys

            fk_str = ""
            for fk_entry in foreign_keys:
                if isinstance(fk_entry, str) and fk_entry.startswith(f"{tbl_name}.{col_name} = "):
                    fk_str = f"[{fk_entry.split(' = ')[1]}]"
                elif isinstance(fk_entry, dict):
                    if fk_entry.get("from") == f"{tbl_name}.{col_name}":
                        fk_str = f"[{fk_entry.get('to')}]"

            result.append({
                "table_name": tbl_name,
                "column_name": col_name,
                "column_types": col_type,
                "column_descriptions": col_desc or "",
                "primary_key": is_pk,
                "foreign_key": fk_str,
            })

        return result

    def _prune_list_format(self, schema_info: list, selected_tables: List[str]) -> list:
        return [item for item in schema_info if item.get("table_name", "") in selected_tables]

    def _prune_dict_format(self, schema_info: dict, selected_tables: List[str]) -> list:
        result = []
        for table_name, table_info in schema_info.items():
            if table_name not in selected_tables:
                continue
            columns = table_info.get("columns", []) if isinstance(table_info, dict) else []
            for col in columns:
                col_name = col if isinstance(col, str) else col.get("column_name", col.get("name", ""))
                col_type = "text" if isinstance(col, str) else col.get("column_types", col.get("type", "text"))
                result.append({
                    "table_name": table_name,
                    "column_name": col_name,
                    "column_types": col_type,
                    "column_descriptions": "",
                    "primary_key": False,
                    "foreign_key": "",
                })
        return result

    def _prune_str_format(self, db_id: str, schema_str: str, selected_tables: List[str]) -> list:
        parsed = parse_squrve_text(schema_str)
        result = []
        for table_name, cols in parsed.items():
            if table_name not in selected_tables:
                continue
            for col in cols:
                result.append({
                    "table_name": table_name,
                    "column_name": col,
                    "column_types": "text",
                    "column_descriptions": "",
                    "primary_key": False,
                    "foreign_key": "",
                })
        return result
    
    def act(
        self,
        item,
        schema: Union[str, PathLike, Dict, List] = None,
        schema_links: Union[str, List[str]] = None,
        sub_questions: Union[str, List[str], Dict] = None,
        data_logger=None,
        **kwargs
    ) -> str:
        """
        DBCopilot 核心执行方法
        
        流程:
        1. 加载问题
        2. 加载多数据库Schema
        3. Schema Routing - 识别目标数据库和表
        4. Schema Pruning - 裁剪Schema
        5. SQL Generation - 调用底层生成器
        6. 保存并返回结果
        """
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"DBCopilotGenerator processing sample {item}")
        
        # Step 1: 加载问题
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        
        logger.debug(f"Processing question: {question[:100]}...")
        
        # Step 2: 加载多数据库Schema
        logger.debug("Loading multi-database schema...")
        multi_db_schema = self._load_multi_db_schema(item, schema)
        
        if not multi_db_schema:
            raise ValueError("Failed to load multi-database schema!")
        
        logger.debug(f"Loaded {len(multi_db_schema)} databases")
        
        # Step 3: Schema Routing
        logger.debug("Starting Schema Routing...")
        routing_result = self._schema_routing(question, multi_db_schema)
        
        if data_logger:
            data_logger.info(f"{self.NAME}.routing_result | result={routing_result}")
        logger.debug(f"Routing result: {routing_result}")
        
        # Step 4: Schema Pruning
        logger.debug("Pruning schema based on routing result...")
        pruned_schema = self._prune_schema(multi_db_schema, routing_result)
        
        # Step 5: 确定主数据库（用于执行）
        primary_db_id = list(routing_result.keys())[0] if routing_result else None
        
        # Step 6: SQL Generation（复用底层生成器）
        logger.debug(f"Generating SQL using {self.sql_generator_name}...")
        
        # 临时修改row的db_id为routing结果
        original_db_id = row.get("db_id")
        if primary_db_id:
            row["db_id"] = primary_db_id

        dinsql_schema = pruned_schema
        if primary_db_id and primary_db_id in multi_db_schema:
            dinsql_schema = multi_db_schema[primary_db_id]

        # 调用底层生成器
        sql_generator = self._get_sql_generator()
        pred_sql = sql_generator.act(
            item,
            schema=dinsql_schema,
            schema_links=schema_links,
            sub_questions=sub_questions,
            data_logger=data_logger,
            **kwargs
        )
        
        # 恢复原始db_id
        row["db_id"] = original_db_id
        
        # Step 7: 保存结果
        pred_sql = self.save_output(pred_sql, item, row.get("instance_id"))
        
        logger.info(f"DBCopilotGenerator sample {item} processed")
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")
            
        return pred_sql
    
    def _load_multi_db_schema(
        self,
        item,
        schema: Union[str, PathLike, Dict, List] = None
    ) -> Dict[str, Any]:
        if isinstance(schema, dict):
            first_val = next(iter(schema.values()), None)
            if first_val is not None:
                return schema
            row = self.dataset[item]
            db_id = row.get("db_id", "default")
            single_schema = self.dataset.get_db_schema(item)
            return {db_id: single_schema}

        if isinstance(schema, list):
            row = self.dataset[item]
            db_id = row.get("db_id", "default")
            return {db_id: schema}

        if isinstance(schema, (str, PathLike)):
            path = Path(schema)
            if path.exists():
                content = load_dataset(schema)
                if isinstance(content, dict):
                    return content
                if isinstance(content, list):
                    row = self.dataset[item]
                    db_id = row.get("db_id", "default")
                    return {db_id: content}
            row = self.dataset[item]
            db_id = row.get("db_id", "default")
            return {db_id: schema}

        if isinstance(schema, pd.DataFrame):
            row = self.dataset[item]
            db_id = row.get("db_id", "default")
            return {db_id: schema.to_dict(orient="records")}

        row = self.dataset[item]
        multi_schema_path = row.get("multi_db_schemas")
        if multi_schema_path and Path(multi_schema_path).exists():
            return load_dataset(multi_schema_path)

        single_schema = self.dataset.get_db_schema(item)
        if single_schema and isinstance(single_schema, list) and len(single_schema) > 0:
            all_schema_by_db = {}
            for s in single_schema:
                db_id = s.get("db_id") if isinstance(s, dict) else None
                if db_id:
                    all_schema_by_db.setdefault(db_id, []).append(s)
            if all_schema_by_db:
                return all_schema_by_db

        db_id = row.get("db_id", "default")
        return {db_id: single_schema}
