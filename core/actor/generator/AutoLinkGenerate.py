"""
AutoLinkGenerate.py – SQL generation actor for the AutoLink pipeline.

Corresponds to AutoLink/run/sql_generation.py.

Key design decisions
--------------------
- Uses self.llm.complete() exclusively – no hardcoded OpenAI client.
- Generates ``num_candidates`` SQL candidates per sample (default 5).
- Builds schema prompts from the ``schema_links`` field (output of AutoLinkParser)
  or falls back to the full schema text from the dataset.
- Candidate generation is parallelised with ThreadPoolExecutor so it is safe
  under Squrve's multi-thread model and shares the per-process FAISS/LLM state.
- Saves each candidate to a separate .sql file and stores a list of paths in
  dataset[item]["pred_sql"].
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from llama_index.core.llms.llm import LLM

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import parse_schema_from_df, load_dataset, save_dataset
from core.actor.prompts.AutoLinkPrompt import (
    AUTOLINK_SQL_GENERATION,
    AUTOLINK_DIALECT_LABEL,
    AUTOLINK_SQL_BIGQUERY,
    AUTOLINK_SQL_SNOWFLAKE,
    AUTOLINK_SQL_SQLITE,
    AUTOLINK_BIGQUERY_OPTIMIZATION,
    AUTOLINK_SNOWFLAKE_OPTIMIZATION,
    AUTOLINK_SQLITE_OPTIMIZATION,
    build_filtered_schema_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dialect_strings(db_type: str):
    """Return (sql_type_str, optimization_str, dialect_label)."""
    t = (db_type or "sqlite").lower()
    if t in ("bigquery", "big_query"):
        return AUTOLINK_SQL_BIGQUERY, AUTOLINK_BIGQUERY_OPTIMIZATION, "BigQuery"
    elif t == "snowflake":
        return AUTOLINK_SQL_SNOWFLAKE, AUTOLINK_SNOWFLAKE_OPTIMIZATION, "Snowflake"
    else:
        return AUTOLINK_SQL_SQLITE, AUTOLINK_SQLITE_OPTIMIZATION, "SQLite"


def _extract_sql(text: str) -> str:
    """Extract SQL from ```sql ... ``` block; fall back to raw text."""
    # Try fenced code block first
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        # Strip reasoning tags if any (e.g. DeepSeek R1)
        sql = re.sub(r"<think>.*?</think>", "", sql, flags=re.DOTALL).strip()
        return sql
    # Fall back: strip common prefixes
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.strip()


def _build_schema_prompt(
    schema_links: Optional[Union[str, List, Dict]],
    schema_text: str,
) -> str:
    """Construct a concise schema prompt for the SQL generation call."""
    if schema_links is None:
        return schema_text

    # schema_links can be a path, a dict {tables, columns}, or a string
    if isinstance(schema_links, (str, Path)) and Path(schema_links).exists():
        try:
            schema_links = load_dataset(schema_links)
        except Exception:
            return schema_text

    if isinstance(schema_links, dict):
        tables  = schema_links.get("tables",  [])
        columns = schema_links.get("columns", [])
        parts   = []
        if tables:
            parts.append("Relevant Tables: " + ", ".join(map(str, tables)))
        if columns:
            parts.append("Relevant Columns: " + ", ".join(map(str, columns)))
        return "\n".join(parts) if parts else schema_text

    if isinstance(schema_links, list):
        return "Schema links:\n" + "\n".join(map(str, schema_links))

    return str(schema_links) if schema_links else schema_text


# ---------------------------------------------------------------------------
# AutoLinkGenerator
# ---------------------------------------------------------------------------

@BaseGenerator.register_actor
class AutoLinkGenerator(BaseGenerator):
    """Multi-candidate SQL generator for the AutoLink pipeline.

    Migrated from ``AutoLink/run/sql_generation.py`` and adapted to use
    Squrve's LLM abstraction (self.llm) and dataset model.
    """

    NAME = "AutoLinkGenerator"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm: Optional[LLM] = None,
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "../files/pred_sql",
        num_candidates: int = 5,
        db_path: Optional[Union[str, PathLike]] = None,
        credential: Optional[Dict] = None,
        use_external: bool = True,
        **kwargs,
    ):
        self.dataset       = dataset
        self.llm           = llm
        self.is_save       = is_save
        self.save_dir      = save_dir
        self.num_candidates = num_candidates
        self.use_external  = use_external

        # DB info (resolved from dataset at act-time if not provided)
        self.db_path    = db_path
        self.credential = credential

    # ------------------------------------------------------------------
    # Schema / prompt helpers
    # ------------------------------------------------------------------

    def _load_schema_df(self, item) -> Optional[pd.DataFrame]:
        """Load the full schema as a DataFrame for the given item."""
        row = self.dataset[item]
        schema = None

        instance_schema_path = row.get("instance_schemas")
        if instance_schema_path and Path(str(instance_schema_path)).exists():
            schema = load_dataset(instance_schema_path)

        if schema is None:
            schema = self.dataset.get_db_schema(item)

        if schema is None:
            return None

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)
        if isinstance(schema, pd.DataFrame):
            return schema

        return None

    def _load_schema_text(self, item) -> str:
        """Load the full schema text for the given item."""
        schema_df = self._load_schema_df(item)
        if schema_df is None:
            raise ValueError(f"AutoLinkGenerator: no schema for item {item}")
        return parse_schema_from_df(schema_df)

    def _build_prompt(self, question: str, schema_prompt: str, db_type: str) -> str:
        sql_type_str, opt_str, dialect_label = _get_dialect_strings(db_type)
        return AUTOLINK_SQL_GENERATION.format(
            SQL_TYPE=sql_type_str,
            SQL_DIALECT_OPTIMIZATION=opt_str,
            SQL_DIALECT_NAME=dialect_label,
            PROMPT=schema_prompt,
            QUESTION=question,
        )

    # ------------------------------------------------------------------
    # Candidate generation (thread-safe: one LLM call per candidate)
    # ------------------------------------------------------------------

    def _generate_one(self, prompt: str, candidate_idx: int) -> str:
        """Call LLM once and return the extracted SQL string."""
        try:
            text = self.llm.complete(prompt).text
            sql  = _extract_sql(text)
            logger.debug(f"AutoLinkGenerator: candidate {candidate_idx} generated")
            return sql
        except Exception as e:
            logger.warning(f"AutoLinkGenerator: candidate {candidate_idx} failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # save_output – override to handle list of candidates
    # ------------------------------------------------------------------

    def save_output(self, sql_list: List[str], item, instance_id: str = None) -> List[str]:  # type: ignore[override]
        if not self.is_save:
            if len(sql_list) == 1:
                self.dataset.setitem(item, "pred_sql", sql_list[0])
            else:
                self.dataset.setitem(item, "pred_sql", sql_list)
            return sql_list

        if instance_id is None:
            instance_id = str(self.dataset[item].get("instance_id", item))

        save_path = Path(self.save_dir)
        if self.dataset and hasattr(self.dataset, "dataset_index") and self.dataset.dataset_index:
            save_path = save_path / str(self.dataset.dataset_index)
        save_path.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, sql in enumerate(sql_list):
            p = save_path / f"{self.NAME}_{instance_id}_{i}.sql"
            save_dataset(sql, new_data_source=p)
            paths.append(str(p))

        if len(paths) == 1:
            self.dataset.setitem(item, "pred_sql", paths[0])
        else:
            self.dataset.setitem(item, "pred_sql", paths)

        logger.debug(f"AutoLinkGenerator: {len(paths)} candidates saved for item {item}")
        return sql_list

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(
        self,
        item,
        schema: Union[str, PathLike, Dict, List] = None,
        schema_links: Union[str, List[str]] = None,
        sub_questions=None,          # ignored – kept for BaseGenerator signature
        data_logger=None,
        **kwargs,
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row         = self.dataset[item]
        question    = row.get("question", "")
        db_type     = row.get("db_type", "sqlite")
        instance_id = str(row.get("instance_id", item))

        # External knowledge (evidence field used by BIRD)
        if self.use_external:
            external = (
                row.get("evidence")
                or row.get("external")
                or row.get("external_knowledge")
            )
            if external:
                question = f"{question}\n\n[Evidence]: {external}"

        # Schema
        schema_df   = self._load_schema_df(item)
        schema_text = parse_schema_from_df(schema_df) if schema_df is not None else ""

        # Schema links (from AutoLinkParser output)
        if schema_links is None:
            schema_link_path = row.get("schema_links")
            if schema_link_path:
                try:
                    schema_links = load_dataset(schema_link_path)
                except Exception:
                    schema_links = None

        # Use filtered schema when schema_links are available (mirrors original
        # AutoLink's final_schema_prompts/ files). Falls back to full schema
        # when schema_links is empty or None.
        if isinstance(schema_links, dict) and schema_df is not None:
            schema_prompt = build_filtered_schema_text(schema_df, schema_links, schema_text)
        else:
            schema_prompt = _build_schema_prompt(schema_links, schema_text)

        prompt = self._build_prompt(question, schema_prompt, db_type)

        if data_logger:
            data_logger.info(f"{self.NAME}: generating {self.num_candidates} candidates")

        # Generate candidates in parallel threads
        sql_list: List[str] = [""] * self.num_candidates
        if self.num_candidates == 1:
            sql_list[0] = self._generate_one(prompt, 0)
        else:
            with ThreadPoolExecutor(max_workers=self.num_candidates) as pool:
                future_map = {
                    pool.submit(self._generate_one, prompt, i): i
                    for i in range(self.num_candidates)
                }
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    try:
                        sql_list[idx] = fut.result()
                    except Exception as e:
                        logger.warning(f"AutoLinkGenerator: future {idx} error: {e}")

        # Filter empty results but keep at least one placeholder
        sql_list = [s for s in sql_list if s] or [""]

        if data_logger:
            data_logger.info(
                f"{self.NAME}: generated {len(sql_list)} non-empty candidates"
            )

        self.save_output(sql_list, item, instance_id)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return sql_list
