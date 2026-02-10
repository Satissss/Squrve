from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict, Tuple, Optional
import pandas as pd
from os import PathLike
from pathlib import Path
import re
from loguru import logger

from core.data_manage import Dataset, single_central_process
from core.actor.decomposer.BaseDecompose import BaseDecomposer
from core.actor.prompts.RecursivePrompt import (
    SELECT_RELATED_TABLES_PROMPT,
    REMOVE_UNRELATED_TABLES_PROMPT,
    STAGE0_SINGLE_TABLE_SQL_PROMPT,
    RECURSIVE_MERGE_SQL_PROMPT
)
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset,
    parse_json_from_str
)
from core.db_connect import get_sql_exec_result
from core.actor.decomposer.decompose_utils import normalize_sub_questions

@BaseDecomposer.register_actor
class RecursiveDecomposer(BaseDecomposer):
    NAME = "RecursiveDecomposer"

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/sub_questions",
            use_feedback: bool = True,
            use_external: bool = True,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            table_batch_size: int = 3,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.use_feedback = use_feedback
        self.use_external: bool = use_external
        self.db_path = db_path or (self.dataset.db_path if self.dataset else None)
        self.credential = credential or (self.dataset.credential if self.dataset else None)
        self.table_batch_size = table_batch_size

    def load_schema(self, item, schema):
        """Process and normalize database schema from various input formats."""
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

    def _init_tables(self, question: str, schema: pd.DataFrame, llm, external_knowledge=None, data_logger=None):
        tables = schema["table_name"].unique().tolist()
        schema_str = parse_schema_from_df(schema)

        select_prompt = SELECT_RELATED_TABLES_PROMPT.format(SCHEMA=schema_str, QUESTION=question,
                                                            EXTERNAL=external_knowledge)
        remove_prompt = REMOVE_UNRELATED_TABLES_PROMPT.format(SCHEMA=schema_str, QUESTION=question,
                                                              EXTERNAL=external_knowledge)
        try:
            response = llm.complete(select_prompt).text
            select_tables = parse_json_from_str(response)['table_names']
            if data_logger:
                data_logger.info(f"{self.NAME}._init_tables select | select_tables={select_tables}")
        except Exception as e:
            logger.error(f"Failed to select tables: {e}")
            return None, None
        try:
            response = llm.complete(remove_prompt).text
            remove_tables = parse_json_from_str(response)['table_names']
            if data_logger:
                data_logger.info(f"{self.NAME}._init_tables remove | remove_tables={remove_tables}")
        except Exception as e:
            logger.error(f"Failed to remove tables: {e}")
            return None, None
        remove_tables = [x for x in remove_tables if x not in select_tables]
        final_tables = [x for x in tables if x not in remove_tables]
        if data_logger:
            data_logger.info(f"{self.NAME}._init_tables end | final_tables={final_tables}")

        return final_tables

    def _parse_tables_from_schema_links(self, schema_links: List[str] | Dict):
        if isinstance(schema_links, Dict):
            tables = schema_links.get('tables', [])
        elif isinstance(schema_links, List):
            # we default treat the first part of schema element as table 
            tables = [x.split(".")[0] for x in schema_links]
        else:
            tables = []

        return tables

    def _filter_schemas_by_tables(self, schema: pd.DataFrame, tables: List[str]):
        if not isinstance(schema, pd.DataFrame) or not isinstance(tables, list) or len(tables) == 0:
            return schema
        return schema[schema["table_name"].isin(tables)]

    def _execute_sql(
            self,
            sql_query: str,
            db_id: str = None,
            db_path: str = None,
            db_type: str = "sqlite",
            max_rows: int = 10
    ) -> str:
        """
        Execute a SQL query and return the result as a string.
        """
        # Get credential path
        credential_path = None
        if self.credential:
            credential_path = self.credential.get(db_type) if isinstance(self.credential, dict) else self.credential

        # Execute SQL with all parameters for extensibility
        result = get_sql_exec_result(
            db_type=db_type,
            sql_query=sql_query,
            db_path=db_path or self.db_path,
            db_id=db_id,
            credential_path=credential_path
        )

        if result is None:
            return f"Unsupported database type: {db_type}"

        exec_result, error_info = result

        # SQL syntax error
        if exec_result is None and error_info:
            return f"SQL Execution Error: {error_info}"

        # Empty result
        if exec_result is None or (isinstance(exec_result, pd.DataFrame) and exec_result.empty):
            return "Query returned no results"

        # DataFrame result - return first max_rows rows
        if isinstance(exec_result, pd.DataFrame):
            result_str = exec_result.head(max_rows).to_string(index=False)
            if len(exec_result) > max_rows:
                result_str += f"\n... ({len(exec_result) - max_rows} more rows)"
            return result_str

        return str(exec_result)

    def _generate_stage0(
            self,
            question: str,
            schema: pd.DataFrame,
            llm,
            db_id: str = None,
            db_path: str = None,
            db_type: str = "sqlite",
            external_knowledge: Optional[str] = None,
            data_logger=None
    ) -> List[Dict]:
        """
        Stage 0: Generate SQL for each single table.
        
        This is the initial stage where each SQL query can only query one table.
        Each SQL statement queries data from a single table to get the maximum range
        of data that could potentially be needed to answer the question.
        """
        stage0_results: List[Dict] = []

        # Group schema by table_name
        grouped = schema.groupby('table_name')
        table_groups = [(table_name, group_df) for table_name, group_df in grouped]

        if data_logger:
            data_logger.info(
                f"{self.NAME}._generate_stage0 | total_tables={len(table_groups)} | batch_size={self.table_batch_size}")

        # Divide into batches based on table_batch_size
        batches = []
        for i in range(0, len(table_groups), self.table_batch_size):
            batch = table_groups[i:i + self.table_batch_size]
            batches.append(batch)

        if data_logger:
            data_logger.info(f"{self.NAME}._generate_stage0 | total_batches={len(batches)}")

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            # Merge DataFrames within the batch
            batch_dfs = [group_df for _, group_df in batch]
            batch_schema_df = pd.concat(batch_dfs, ignore_index=True)

            # Convert to string using parse_schema_from_df
            batch_schema_str = parse_schema_from_df(batch_schema_df)

            # Get table names in this batch for logging
            batch_table_names = [table_name for table_name, _ in batch]

            if data_logger:
                data_logger.info(f"{self.NAME}._generate_stage0 batch {batch_idx} | tables={batch_table_names}")

            # Generate prompt
            prompt = STAGE0_SINGLE_TABLE_SQL_PROMPT.format(
                SCHEMA=batch_schema_str,
                QUESTION=question,
                EXTERNAL=external_knowledge if external_knowledge else "None"
            )

            try:
                # Call LLM to generate SQL for each table in the batch
                response = llm.complete(prompt).text
                batch_results = parse_json_from_str(response)

                if data_logger:
                    data_logger.info(
                        f"{self.NAME}._generate_stage0 batch {batch_idx} | generated={len(batch_results)} items")

                # Process each result and add to stage0_results
                for item in batch_results:
                    sql_container = {
                        "sql": item.get("sql", ""),
                        "sub_question": item.get("sub_question", ""),
                        "chain_of_thought": item.get("chain_of_thought", ""),
                        "table": item.get("table", ""),
                        "result": None,  # Result is set to None initially
                        "stage": 0  # Stage 0
                    }
                    stage0_results.append(sql_container)

            except Exception as e:
                logger.error(f"Failed to generate Stage 0 SQL for batch {batch_idx}: {e}")
                if data_logger:
                    data_logger.error(f"{self.NAME}._generate_stage0 batch {batch_idx} | error={str(e)}")
                continue

        # Execute SQL and get query results when use_feedback is enabled
        if self.use_feedback and stage0_results:
            if data_logger:
                data_logger.info(f"{self.NAME}._generate_stage0 | executing SQL queries for feedback")

            for idx, sql_container in enumerate(stage0_results):
                sql_query = sql_container.get("sql", "")
                if not sql_query:
                    sql_container["result"] = "No SQL query to execute"
                    continue

                # Execute SQL and get result
                result_str = self._execute_sql(
                    sql_query=sql_query,
                    db_id=db_id,
                    db_path=db_path,
                    db_type=db_type,
                    max_rows=10
                )
                sql_container["result"] = result_str

                if data_logger:
                    log_result = result_str[:100] + "..." if len(result_str) > 100 else result_str
                    data_logger.info(
                        f"{self.NAME}._generate_stage0 sql {idx} | table={sql_container.get('table')} | result={log_result}")

        if data_logger:
            data_logger.info(f"{self.NAME}._generate_stage0 completed | total_results={len(stage0_results)}")

        return stage0_results

    @staticmethod
    def _get_active_queries(sql_containers: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Identify active (unconsumed) queries from all SQL containers.
        
        A query is "consumed" if its query_id appears as a source_query_id in
        a later merge operation. Only unconsumed queries are available for 
        further merging.
        
        Args:
            sql_containers: List of all SQL containers
            
        Returns:
            List of (index, container) tuples for active queries
        """
        # Collect all consumed query IDs
        consumed_ids: set = set()
        for container in sql_containers:
            source_ids = container.get("source_query_ids", [])
            if source_ids:
                consumed_ids.update(source_ids)

        # Return containers whose query_id is NOT consumed
        active = []
        for idx, container in enumerate(sql_containers):
            query_id = f"query_{idx}"
            if query_id not in consumed_ids:
                active.append((idx, container))

        return active

    def _format_previous_sqls(self, sql_containers: List[Dict]) -> str:
        """
        Format ACTIVE (unconsumed) SQL queries for inclusion in the merge prompt.
        
        Only queries that have NOT been consumed by a later merge are shown,
        so the LLM sees exactly which queries are available for merging.
        
        Args:
            sql_containers: List of all SQL containers
            
        Returns:
            Formatted string of active SQLs
        """
        active_queries = self._get_active_queries(sql_containers)

        lines = []
        for idx, container in active_queries:
            query_id = f"query_{idx}"
            tables = container.get("table", "unknown")
            sql = container.get("sql", "")
            result = container.get("result", None)
            stage = container.get("stage", 0)
            sub_question = container.get("sub_question", "")

            # Truncate result if too long
            result_display = result if result else "Not executed"
            if result_display and len(result_display) > 500:
                result_display = result_display[:500] + "\n... (truncated)"

            lines.append(f"""### [{query_id}] (Stage {stage})
- **Tables**: {tables}
- **Sub-question**: {sub_question}
- **SQL**:
```sql
{sql}
```
- **Execution Result**: 
{result_display}
""")

        return "\n".join(lines)

    def _generate_stage_n(
            self,
            question: str,
            schema: pd.DataFrame,
            llm,
            sql_containers: List[Dict],
            current_stage: int,
            db_id: str = None,
            db_path: str = None,
            db_type: str = "sqlite",
            external_knowledge: Optional[str] = None,
            data_logger=None
    ) -> Tuple[List[Dict], bool]:
        """
        Generate SQL for Stage N (N >= 1) through recursive merging.
        
        Each stage merges pairs of SQL queries from previous stages.
        """
        schema_str = parse_schema_from_df(schema)
        previous_sqls_str = self._format_previous_sqls(sql_containers)
        active_count = len(self._get_active_queries(sql_containers))

        prompt = RECURSIVE_MERGE_SQL_PROMPT.format(
            CURRENT_STAGE=current_stage,
            QUESTION=question,
            SCHEMA=schema_str,
            EXTERNAL=external_knowledge if external_knowledge else "None",
            PREVIOUS_SQLS=previous_sqls_str,
            ACTIVE_COUNT=active_count
        )

        if data_logger:
            data_logger.info(f"{self.NAME}._generate_stage_n | stage={current_stage}")

        try:
            response = llm.complete(prompt).text
            result = parse_json_from_str(response)

            is_final = result.get("is_final", False)

            if is_final:
                # Final merge - create the final SQL container
                final_container = {
                    "sql": result.get("final_sql", ""),
                    "sub_question": f"Final SQL answering: {question}",
                    "chain_of_thought": result.get("chain_of_thought", ""),
                    "table": result.get("merged_tables", []),
                    "source_query_ids": result.get("source_query_ids", []),
                    "result": None,
                    "stage": current_stage,
                    "is_final": True
                }

                # Execute final SQL if feedback is enabled
                if self.use_feedback and final_container["sql"]:
                    result_str = self._execute_sql(
                        sql_query=final_container["sql"],
                        db_id=db_id,
                        db_path=db_path,
                        db_type=db_type,
                        max_rows=10
                    )
                    final_container["result"] = result_str

                    if data_logger:
                        log_result = result_str[:100] + "..." if len(result_str) > 100 else result_str
                        data_logger.info(f"{self.NAME}._generate_stage_n | final_sql result={log_result}")

                if data_logger:
                    data_logger.info(f"{self.NAME}._generate_stage_n | stage={current_stage} | is_final=True")

                return [final_container], True

            else:
                # Process merge operations
                merge_operations = result.get("merge_operations", [])
                new_containers: List[Dict] = []

                for merge_op in merge_operations:
                    sql_container = {
                        "sql": merge_op.get("sql", ""),
                        "sub_question": merge_op.get("sub_question", ""),
                        "chain_of_thought": merge_op.get("chain_of_thought", ""),
                        "table": merge_op.get("tables", []),
                        "source_query_ids": merge_op.get("source_query_ids", []),
                        "result": None,
                        "stage": current_stage
                    }

                    # Execute SQL if feedback is enabled
                    if self.use_feedback and sql_container["sql"]:
                        result_str = self._execute_sql(
                            sql_query=sql_container["sql"],
                            db_id=db_id,
                            db_path=db_path,
                            db_type=db_type,
                            max_rows=10
                        )
                        sql_container["result"] = result_str

                        if data_logger:
                            log_result = result_str[:100] + "..." if len(result_str) > 100 else result_str
                            data_logger.info(
                                f"{self.NAME}._generate_stage_n | stage={current_stage} | tables={sql_container['table']} | result={log_result}")

                    new_containers.append(sql_container)

                if data_logger:
                    data_logger.info(
                        f"{self.NAME}._generate_stage_n | stage={current_stage} | new_merges={len(new_containers)}")

                return new_containers, False

        except Exception as e:
            logger.error(f"Failed to generate Stage {current_stage}: {e}")
            if data_logger:
                data_logger.error(f"{self.NAME}._generate_stage_n | stage={current_stage} | error={str(e)}")
            return [], False

    def generate_decomposition(
            self,
            question: str,
            schema: pd.DataFrame,
            llm,
            db_id: str = None,
            db_path: str = None,
            db_type: str = "sqlite",
            external_knowledge=None,
            data_logger=None
    ):
        """
        Generate SQL decomposition through recursive stages.
        
        The complete SQL generation is understood as a recursive process:
        - Stage 0: Each SQL can only query one table. If there are N tables, there are N SQL statements.
        - Stage 1-n: Progressive aggregation through JOINs and other operations, 
                     gradually approaching the final query result.
        
        Args:
            question: The question to answer
            schema: DataFrame containing the database schema
            llm: Language model for generation
            db_id: Database identifier for query execution
            db_type: Type of database ("sqlite", "big_query", "snowflake")
            external_knowledge: Optional external knowledge
            data_logger: Optional logger for debugging
            
        Returns:
            List of sql_containers with all stages' results
        """
        # sql_containers: List[Dict] = [{"sql": str, "sub_question": str, "chain_of_thought": str, "table": str, "result": str, "stage": int}]
        sql_containers: List[Dict] = []

        # Stage 0: 
        # Generate sql, sub-question, COT for every single table to query the whole data required for question.
        # All generated sql contains the necessary columns to answer the question.
        stage0_results = self._generate_stage0(
            question=question,
            schema=schema,
            llm=llm,
            db_id=db_id,
            db_path=db_path,
            db_type=db_type,
            external_knowledge=external_knowledge,
            data_logger=data_logger
        )
        sql_containers.extend(stage0_results)

        if data_logger:
            data_logger.info(f"{self.NAME}.generate_decomposition | stage0_count={len(stage0_results)}")

        # Stage 1-n:
        # A recursive process that decomposes 
        # the complete SQL generation into a multi-stage, dynamically constructed directed acyclic graph, 
        # representing the full logic of SQL generation.

        # If only one table, stage 0 result is already the final result
        if len(stage0_results) <= 1:
            if stage0_results:
                stage0_results[0]["is_final"] = True
            if data_logger:
                data_logger.info(f"{self.NAME}.generate_decomposition | single table, no merge needed")
            return sql_containers

        # Recursive merging loop
        current_stage = 1
        max_stages = 2 * len(stage0_results)  # Theoretical upper bound: N-1 merges for N queries
        is_final = False

        while not is_final and current_stage <= max_stages:
            # Count active (unconsumed) queries
            active_queries = self._get_active_queries(sql_containers)
            active_count = len(active_queries)

            if data_logger:
                active_ids = [f"query_{idx}" for idx, _ in active_queries]
                data_logger.info(
                    f"{self.NAME}.generate_decomposition | stage {current_stage} | active_queries={active_count} | ids={active_ids}")

            # If only 1 active query remains, it's the final result
            if active_count <= 1:
                if active_queries:
                    active_queries[0][1]["is_final"] = True
                if data_logger:
                    data_logger.info(f"{self.NAME}.generate_decomposition | single active query, marking as final")
                is_final = True
                break

            new_containers, is_final = self._generate_stage_n(
                question=question,
                schema=schema,
                llm=llm,
                sql_containers=sql_containers,
                current_stage=current_stage,
                db_id=db_id,
                db_path=db_path,
                db_type=db_type,
                external_knowledge=external_knowledge,
                data_logger=data_logger
            )

            if not new_containers:
                logger.warning(f"Stage {current_stage} generated no results, stopping recursion")
                break

            sql_containers.extend(new_containers)

            if data_logger:
                new_active = self._get_active_queries(sql_containers)
                data_logger.info(
                    f"{self.NAME}.generate_decomposition | stage {current_stage} done | "
                    f"new_merges={len(new_containers)} | is_final={is_final} | active_remaining={len(new_active)}"
                )

            current_stage += 1

        if current_stage > max_stages:
            logger.warning(f"Reached maximum stages ({max_stages}), stopping recursion")
            if data_logger:
                data_logger.warning(f"{self.NAME}.generate_decomposition | max stages reached")

        if data_logger:
            final_count = sum(1 for c in sql_containers if c.get("is_final", False))
            data_logger.info(
                f"{self.NAME}.generate_decomposition completed | total_containers={len(sql_containers)} | final_count={final_count}")

        return sql_containers

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        """
        Decompose the whole Text-to-SQL tasks as a directed acyclic graph generation process,
        Each node in the graph is a progressive sub-question and its corresponding SQL statement.
        Every Stage in the graph is driven by the prefounded nodes, and its SQL execution Results.
        The SQL generated by sub-question can be executed optionally to gather the cell data example and inner results.
        """
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row.get("question", "")
        db_id = row.get("db_id", "")
        db_path = Path(self.db_path) / (db_id + ".sqlite") if self.db_path else self.db_path
        # Use base class method to process schema
        schema_df = self.load_schema(item, schema)
        if not isinstance(schema_df, pd.DataFrame):
            return []

        external_knowledge = None
        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))

        # Use base class method to get LLM
        llm = self.get_llm()
        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        tables = None
        if schema_links is None:
            schema_link_path = row.get("schema_links", None)
            if schema_link_path:
                schema_links = load_dataset(schema_link_path)
                logger.debug(f"从路径加载模式链接: {schema_link_path}")
                tables = self._parse_tables_from_schema_links(schema_links)
                if data_logger:
                    data_logger.info(
                        f"{self.NAME}.act schema_links from path | path={schema_link_path} | tables={tables}")
            else:
                logger.debug("使用自定义生成模式链接")
                tables = self._init_tables(question, schema_df, llm, external_knowledge, data_logger)
                if data_logger:
                    data_logger.info(f"{self.NAME}.act schema_links from llm | tables={tables}")

        if not isinstance(tables, list) or len(tables) == 0:
            return []

        schema_df = self._filter_schemas_by_tables(schema_df, tables)

        # Get database type from dataset or default to sqlite
        db_type = self.dataset.db_type if hasattr(self.dataset, 'db_type') and self.dataset.db_type else "sqlite"

        # Generate decomposition results
        sub_questions = self.generate_decomposition(
            question=question,
            schema=schema_df,
            llm=llm,
            db_id=db_id,
            db_path=db_path,
            db_type=db_type,
            external_knowledge=external_knowledge,
            data_logger=data_logger
        )
        sub_questions = normalize_sub_questions(sub_questions, output_type="C")
        # Use base class method to save output
        self.save_output(sub_questions, item, db_id=db_id)
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return sub_questions