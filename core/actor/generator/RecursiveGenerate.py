from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict, Optional
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.actor.decomposer.RecursiveDecompose import RecursiveDecomposer
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)

@BaseGenerator.register_actor
class RecursiveGenerator(BaseGenerator):
    """Recursive SQL generator that uses recursive decomposition to generate complete SQL statements."""

    NAME = "RecursiveGenerator"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            decomposer: Optional[RecursiveDecomposer] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            use_feedback: bool = True,
            table_batch_size: int = 3,
            **kwargs
    ):
        self.dataset: Optional[Dataset] = dataset
        self.llm: Optional[LLM] = llm
        self.decomposer = decomposer
        self.is_save = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.use_external: bool = use_external
        self.use_feedback = use_feedback
        self.table_batch_size = table_batch_size

        # 安全地初始化 db_path 和 credential
        if db_path is not None:
            self.db_path = db_path
        elif self.dataset is not None:
            self.db_path = self.dataset.db_path
        else:
            self.db_path = None

        if credential is not None:
            self.credential = credential
        elif self.dataset is not None:
            self.credential = self.dataset.credential
        else:
            self.credential = None

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        try:
            external = load_dataset(external)
        except FileNotFoundError:
            logger.info("External file is not valid, treat it as content instead...")
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def load_schema(self, item, schema):
        """Process and normalize database schema from various input formats (same as RecursiveDecomposer)."""
        logger.debug("Processing database schema...")

        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = self.dataset[item].get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded schema from: {instance_schema_path}")
            else:
                logger.debug("Fetching schema from dataset")
                schema = self.dataset.get_db_schema(item)

            if schema is None:
                raise ValueError("Failed to load a valid database schema for the sample!")

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        return schema

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        """Generate SQL using recursive decomposition approach. Replicates RecursiveDecomposer.act flow."""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"RecursiveGenerator processing sample {item}")
        row = self.dataset[item]
        question = row.get("question", "")
        db_id = row.get("db_id", "")
        db_path = Path(self.db_path) / (db_id + ".sqlite") if self.db_path else self.db_path

        # Use same schema loading as RecursiveDecomposer.act
        schema_df = self.load_schema(item, schema)
        if not isinstance(schema_df, pd.DataFrame):
            logger.error("Failed to load a valid database schema for the sample!")
            return ""

        # External knowledge: do not append to question, pass separately (same as RecursiveDecomposer)
        external_knowledge = None
        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))

        # LLM required for decomposition (same as RecursiveDecomposer: get_llm / llm check)
        llm = self.llm
        if llm is None:
            logger.error("LLM is not initialized")
            return ""

        # Ensure decomposer exists so we can use _parse_tables_from_schema_links, _init_tables, _filter_schemas_by_tables
        if self.decomposer is None:
            self.decomposer = RecursiveDecomposer(
                dataset=self.dataset,
                llm=self.llm,
                is_save=False,
                use_feedback=self.use_feedback,
                use_external=self.use_external,
                db_path=self.db_path,
                credential=self.credential,
                table_batch_size=self.table_batch_size
            )

        # Replicate RecursiveDecomposer.act lines 648-666: resolve tables and filter schema
        tables = None
        if schema_links is None:
            schema_link_path = row.get("schema_links", None)
            if schema_link_path:
                schema_links = load_dataset(schema_link_path)
                logger.debug(f"从路径加载模式链接: {schema_link_path}")
                tables = self.decomposer._parse_tables_from_schema_links(schema_links)
                if data_logger:
                    data_logger.info(f"{self.NAME}.act schema_links from path | path={schema_link_path} | tables={tables}")
            else:
                logger.debug("使用自定义生成模式链接")
                tables = self.decomposer._init_tables(question, schema_df, llm, external_knowledge, data_logger)
                if data_logger:
                    data_logger.info(f"{self.NAME}.act schema_links from llm | tables={tables}")
        else:
            tables = self.decomposer._parse_tables_from_schema_links(schema_links)
            if data_logger:
                data_logger.info(f"{self.NAME}.act schema_links from argument | tables={tables}")

        if not isinstance(tables, list) or len(tables) == 0:
            logger.warning("No valid tables resolved, cannot run decomposition")
            return ""

        schema_df = self.decomposer._filter_schemas_by_tables(schema_df, tables)

        # Get database type (same as RecursiveDecomposer)
        db_type = row.get("db_type") or (self.dataset.db_type if hasattr(self.dataset, "db_type") and self.dataset.db_type else "sqlite")

        logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

        # Generate decomposition using filtered schema (same as RecursiveDecomposer.generate_decomposition call)
        logger.debug("Starting recursive decomposition...")
        try:
            decomposition_results = self.decomposer.generate_decomposition(
                question=question,
                schema=schema_df,
                llm=llm,
                db_id=db_id,
                db_path=db_path,
                db_type=db_type,
                external_knowledge=external_knowledge,
                data_logger=data_logger
            )
            logger.debug(f"Recursive decomposition completed with {len(decomposition_results)} results")
        except Exception as e:
            logger.error(f"Recursive decomposition failed: {e}")
            raise

        # Find the final SQL (is_final=True) and save as pred_sql
        final_sql = None
        final_container = None
        for container in decomposition_results:
            if container.get("is_final", False):
                final_sql = container.get("sql", "")
                final_container = container
                break

        if final_sql is None:
            logger.warning("No final SQL found in decomposition results, using last result")
            if decomposition_results:
                final_sql = decomposition_results[-1].get("sql", "")
                final_container = decomposition_results[-1]
            else:
                final_sql = ""
                logger.error("No SQL results generated from decomposition")

        logger.debug(f"Final SQL: {final_sql[:100]}...")

        if data_logger and final_container:
            data_logger.info(
                f"{self.NAME}.final_container | stage={final_container.get('stage', 'unknown')} | "
                f"tables={final_container.get('table', 'unknown')} | sql_length={len(final_sql)}"
            )

        final_sql = self.save_output(final_sql, item, row.get("instance_id"))

        logger.info(f"RecursiveGenerator sample {item} processing completed")
        if data_logger:
            data_logger.info(f"{self.NAME}.final_sql | sql={final_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return final_sql