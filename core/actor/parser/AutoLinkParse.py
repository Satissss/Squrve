"""
AutoLinkParse.py – Squrve parser that runs the AutoLink agent-based
schema-linking loop.

Design principles
-----------------
* No hardcoded paths: embed_path comes from constructor → dataset item field → kwargs.
* No OpenAI client: uses self.llm (llama-index) exclusively via ChatMessage API.
* Output format: {"tables": [...], "columns": [...]} as required by Squrve.
* Dependencies resolved via Squrve's own module tree:
    - Prompts      → core.actor.prompts.AutoLinkPrompt
    - Parser       → core.actor.parser.parse_utils.parse_model_output
    - Model/FAISS  → core.LinkAlign.embed_model.retrieval
    - DB execution → core.db_connect.execute_sql
"""

from __future__ import annotations

import json
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from core.data_manage import Dataset
from core.actor.parser.BaseParse import BaseParser
from core.db_connect import execute_sql

# Prompts
from core.actor.prompts.AutoLinkPrompt import (
    AUTOLINK_SCHEMA_LINKING,
    AUTOLINK_USER_INPUT,
    AUTOLINK_SQL_BIGQUERY,
    AUTOLINK_SQL_SNOWFLAKE,
    AUTOLINK_SQL_SQLITE,
    AUTOLINK_BIGQUERY_OPTIMIZATION,
    AUTOLINK_SNOWFLAKE_OPTIMIZATION,
    AUTOLINK_SQLITE_OPTIMIZATION,
)

# LLM output parser
from core.actor.parser.parse_utils import parse_model_output

# FAISS vector retrieval
from core.AutoLink.retrieval import get_next_k_results


# ---------------------------------------------------------------------------
# Dialect helpers
# ---------------------------------------------------------------------------

def _get_sql_type_strings(db_type: str):
    """Return (sql_type_str, sql_optimization_str) for the given db_type."""
    db_type = (db_type or "sqlite").lower()
    if db_type in ("big_query", "bigquery"):
        return AUTOLINK_SQL_BIGQUERY, AUTOLINK_BIGQUERY_OPTIMIZATION
    elif db_type == "snowflake":
        return AUTOLINK_SQL_SNOWFLAKE, AUTOLINK_SNOWFLAKE_OPTIMIZATION
    else:
        return AUTOLINK_SQL_SQLITE, AUTOLINK_SQLITE_OPTIMIZATION


def _schema_df_to_text(schema_df: pd.DataFrame) -> str:
    """Convert the Squrve schema DataFrame into a human-readable string."""
    col_map   = {c.lower(): c for c in schema_df.columns}
    table_col = col_map.get("table_name",  col_map.get("table",  None))
    col_col   = col_map.get("column_name", col_map.get("column", None))
    type_col  = col_map.get("column_type", col_map.get("type",   None))
    desc_col  = (
        col_map.get("column_description")
        or col_map.get("description")
        or col_map.get("column_comment")
        or None
    )
    val_col   = col_map.get("column_value", col_map.get("sample_value", None))

    lines = []
    prev_table = None
    for _, row in schema_df.iterrows():
        table  = str(row[table_col])  if table_col else ""
        column = str(row[col_col])    if col_col   else ""
        ctype  = str(row[type_col])   if type_col  else ""
        desc   = str(row[desc_col])   if desc_col  else ""
        val    = str(row[val_col])    if val_col   else ""

        if table != prev_table:
            lines.append(f"\nTable: {table}")
            prev_table = table

        parts = [f"  Column name: {column}", f"Column type: {ctype}"]
        if val:
            parts.append(f"Column value: [{val}]")
        if desc:
            parts.append(f"Description: {desc}")
        lines.append("; ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoLinkParser
# ---------------------------------------------------------------------------

@BaseParser.register_actor
class AutoLinkParser(BaseParser):
    """Agent-based schema linker adapted from the AutoLink project.

    Runs a multi-turn LLM loop where the model can:
    - Call @schema_retrieval – FAISS vector search over pre-built column embeddings.
    - Call @sql_execution / @sql_draft – probe the live database.
    - Call @stop() – finish and emit gathered schema elements.

    embed_path resolution (first non-None wins)
    -------------------------------------------
    1. Constructor argument ``embed_path``
    2. ``dataset[item]["embed_path"]``
    3. ``kwargs["embed_path"]`` passed to ``act()``
    """

    NAME = "AutoLinkParser"

    def __init__(
        self,
        dataset: Dataset = None,
        llm: Union[LLM, List[LLM]] = None,
        output_format: str = "list",
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "../files/schema_links",
        use_external: bool = False,
        # AutoLink-specific params
        embed_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        status_dir: Optional[str] = None,
        retrieval_device: str = "cpu",
        retrieval_top_k: int = 3,
        max_turns: int = 10,
        db_type: Optional[str] = None,
        db_path: Optional[str] = None,
        db_credential: Any = None,
        **kwargs,
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)
        self.embed_path       = embed_path
        self.cache_dir        = cache_dir
        self.status_dir       = status_dir
        self.retrieval_device = retrieval_device
        self.retrieval_top_k  = retrieval_top_k
        self.max_turns        = max_turns
        self.db_type          = db_type
        self.db_path          = db_path
        self.db_credential    = db_credential

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_embed_path(self, item, kwargs: dict) -> str:
        if self.embed_path:
            return self.embed_path
        row = self.dataset[item]
        if row.get("embed_path"):
            return row["embed_path"]
        if kwargs.get("embed_path"):
            return kwargs["embed_path"]
        raise ValueError(
            "AutoLinkParser requires an embed_path (FAISS vector store root). "
            "Supply it as a constructor argument, via dataset[item]['embed_path'], "
            "or as kwargs['embed_path'] in the act() call."
        )

    def _resolve_db_info(self, item):
        row     = self.dataset[item]
        db_type = self.db_type or row.get("db_type") or row.get("database_type") or "sqlite"
        # Try real path fields only – never use db_id/db_name as a file path
        db_path = (
            self.db_path
            or row.get("db_path")
            or row.get("sqlite_path")
            or row.get("db_file")
        )
        # For SQLite: db_path in config is a directory; construct the actual file path
        # using db_id (e.g. "...database/california_schools.sqlite")
        if db_type == "sqlite" and db_path:
            db_id = row.get("db_id", "")
            path_obj = Path(str(db_path))
            if db_id and path_obj.is_dir():
                db_path = str(path_obj / f"{db_id}.sqlite")
        cred    = self.db_credential or row.get("credential")
        return db_type, db_path, cred


    def _call_llm(self, messages: list) -> str:
        """Call self.llm via the llama-index ChatMessage API."""
        llm = self.get_llm()
        if llm is None:
            raise RuntimeError("AutoLinkParser has no LLM configured.")
        chat_messages = []
        for msg in messages:
            role_str = msg.get("role", "user").lower()
            content  = msg.get("content", "")
            role = {"system": MessageRole.SYSTEM, "assistant": MessageRole.ASSISTANT}.get(
                role_str, MessageRole.USER
            )
            chat_messages.append(ChatMessage(role=role, content=content))
        return llm.chat(chat_messages).message.content

    def _make_cache_dirs(self) -> tuple:
        base       = Path(self.save_dir)
        cache_dir  = Path(self.cache_dir)  if self.cache_dir  else base / "autolink_cache" / "cache"
        status_dir = Path(self.status_dir) if self.status_dir else base / "autolink_cache" / "status"
        cache_dir.mkdir(parents=True, exist_ok=True)
        status_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir), str(status_dir)

    # ------------------------------------------------------------------
    # act() – main entry point
    # ------------------------------------------------------------------

    def act(
        self,
        item,
        schema: Union[str, PathLike, Dict, List] = None,
        data_logger=None,
        update_dataset: bool = True,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """Run the AutoLink agent loop and return {"tables": [...], "columns": [...]}."""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row         = self.dataset[item]
        question    = row.get("question", "")
        db_name     = row.get("db_name", row.get("db_id", ""))
        instance_id = row.get("instance_id", str(item))
        external_knowledge = ""

        if self.use_external:
            # Priority 1: file-based external knowledge
            ext_path = row.get("external") or row.get("external_knowledge")
            if ext_path and Path(str(ext_path)).exists():
                try:
                    external_knowledge = Path(str(ext_path)).read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"AutoLinkParser: could not read external knowledge: {e}")
            # Priority 2: inline string (BIRD uses 'evidence' field)
            if not external_knowledge:
                raw = (
                    row.get("evidence")
                    or row.get("external")
                    or row.get("external_knowledge")
                )
                if raw and isinstance(raw, str):
                    external_knowledge = raw

        # Schema → text
        schema_df   = self.process_schema(item, schema)
        schema_text = _schema_df_to_text(schema_df)

        col_map   = {c.lower(): c for c in schema_df.columns}
        table_col = col_map.get("table_name", col_map.get("table", None))
        all_tables: List[str] = (
            list(schema_df[table_col].dropna().unique().tolist()) if table_col else []
        )

        # Dialect strings
        db_type, db_path, credential = self._resolve_db_info(item)
        sql_type_str, sql_opt_str    = _get_sql_type_strings(db_type)

        # Retrieval setup
        embed_path            = self._resolve_embed_path(item, kwargs)
        cache_dir, status_dir = self._make_cache_dirs()

        # Build initial messages
        system_prompt = AUTOLINK_SCHEMA_LINKING.format(
            SQL_TYPE=sql_type_str,
            SQL_OPTIMIZATION=sql_opt_str,
        )
        user_message = AUTOLINK_USER_INPUT.format(
            RETRIEVED_SCHEMA=json.dumps(schema_text, ensure_ascii=False),
            USER_QUESTION=question,
            EXTERNAL_KNOWLEDGE=external_knowledge,
            ALL_TABLES=json.dumps(all_tables, ensure_ascii=False),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        table_candidates:  List[str] = []
        column_candidates: List[str] = []
        is_finished = False

        # Agent loop
        for turn_i in range(self.max_turns):
            if is_finished:
                break
            if data_logger:
                data_logger.info(f"{self.NAME} | turn {turn_i}")

            try:
                model_output = self._call_llm(messages)
            except Exception as e:
                logger.error(f"AutoLinkParser: LLM call failed at turn {turn_i}: {e}")
                break

            try:
                full_lines, tool_calls = parse_model_output(model_output)
            except Exception as e:
                logger.error(f"AutoLinkParser: parse_model_output failed: {e}")
                break

            func_messages = ""

            for line, func in zip(full_lines, tool_calls):
                tool_name = func.get("tool", "")

                if tool_name == "stop":
                    is_finished = True
                    break

                elif tool_name == "schema_retrieval":
                    table       = func.get("table", "")
                    column      = func.get("column", "")
                    description = func.get("description", "")
                    if not (table or column or description):
                        continue

                    retrieve_content = (
                        f"column name: {column}\ntable name: {table}\ndescription: {description}"
                    )
                    try:
                        semantic_results, _, completion_text = get_next_k_results(
                            instance_id=instance_id,
                            question=retrieve_content,
                            db_name=db_name,
                            embed_path=embed_path,
                            top_k=self.retrieval_top_k,
                            cache_dir=cache_dir,
                            status_dir=status_dir,
                            device=self.retrieval_device,
                        )
                    except Exception as e:
                        logger.error(f"AutoLinkParser: schema retrieval failed: {e}")
                        func_messages += f"Tool: {line}\nRetrieval error: {e}\n\n"
                        continue

                    func_messages += f"Tool: {line}\nThe tool returns the following results:\n"
                    for result in semantic_results:
                        meta   = result["metadata"]
                        t_name = meta.get("table",        "")
                        c_name = meta.get("column",       "")
                        c_type = meta.get("column_type",  "")
                        c_val  = meta.get("column_value", "")
                        c_desc = meta.get("description",  "")
                        func_messages += (
                            f"Column name: {c_name}; Table: {t_name}; "
                            f"Column type: {c_type}; Column value: [{c_val}]; "
                            f"Description: {c_desc}\n"
                        )
                        column_candidates.append(c_name)
                        table_candidates.append(t_name)

                    if completion_text:
                        func_messages += f"{completion_text}\n"
                    func_messages += "\n"

                elif tool_name in ("sql_execution", "sql_draft"):
                    query = func.get("query", "").strip()
                    if not query:
                        continue
                    func_messages += f"Tool: {line}\nThe tool returns the following results:\n"
                    try:
                        exec_result = execute_sql(
                            db_type=db_type,
                            db_path=db_path,
                            sql=query,
                            credential=credential,
                        )
                        func_messages += f"{exec_result}\n\n"
                    except Exception as e:
                        func_messages += f"Execution error: {e}\n\n"
                        logger.warning(f"AutoLinkParser: SQL execution error: {e}")

            # Reminder appended after each turn (mirrors original AutoLink behaviour)
            func_messages += (
                "\nFor @sql_execution results containing column or table names, "
                "use @schema_retrieval to retrieve any missing columns in this turn.\n"
                "Pay attention to *id, *name, *text, *code columns for joins and filtering.\n"
            )

            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user",      "content": func_messages})

        # Build output
        tables  = list(dict.fromkeys(filter(None, table_candidates)))
        columns = list(dict.fromkeys(filter(None, column_candidates)))
        result  = {"tables": tables, "columns": columns}

        self.log_schema_links(data_logger, result, stage="final")

        if update_dataset:
            self.save_output(result, item, file_ext=".json")

        if data_logger:
            data_logger.info(
                f"{self.NAME}.act end | item={item} | "
                f"tables={len(tables)}, cols={len(columns)}"
            )

        return result
