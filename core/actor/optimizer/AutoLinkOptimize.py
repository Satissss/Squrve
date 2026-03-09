"""
AutoLinkOptimize.py – SQL revision actor for the AutoLink pipeline.

Corresponds to AutoLink/run/sql_revise.py.

For each candidate SQL that failed execution (as recorded by AutoLinkScaler),
attempts multi-turn LLM-based revision up to ``max_revise_turns`` times.
Successful candidates are passed through unchanged.

Key design decisions
--------------------
- Multi-turn chat uses llama-index ChatMessage API (same pattern as AutoLinkParser).
- Each candidate is revised independently → parallelisable with ThreadPoolExecutor.
- exec_results written by AutoLinkScaler are consumed here; if absent, the actor
  falls back to executing each SQL itself.
- Output: revised candidate list (same length as input), stored back in pred_sql.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import LLM

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, single_central_process
from core.db_connect import execute_sql

_EMPTY_RESULT_MARKER = "No data found for the specified query"
from core.utils import parse_schema_from_df
from core.actor.prompts.AutoLinkPrompt import (
    AUTOLINK_REVISE_SYSTEM,
    AUTOLINK_REVISE_USER,
    AUTOLINK_DIALECT_LABEL,
    AUTOLINK_SQL_BIGQUERY, AUTOLINK_SQL_SNOWFLAKE, AUTOLINK_SQL_SQLITE,
    AUTOLINK_BIGQUERY_OPTIMIZATION, AUTOLINK_SNOWFLAKE_OPTIMIZATION,
    AUTOLINK_SQLITE_OPTIMIZATION,
    build_filtered_schema_text,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r"<think>.*?</think>", "", m.group(1), flags=re.DOTALL).strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_dialect_label(db_type: str) -> str:
    return AUTOLINK_DIALECT_LABEL.get((db_type or "sqlite").lower(), "SQLite")


# ---------------------------------------------------------------------------
# AutoLinkOptimizer
# ---------------------------------------------------------------------------

@BaseOptimizer.register_actor
class AutoLinkOptimizer(BaseOptimizer):
    """Multi-turn SQL revision actor for the AutoLink pipeline.

    Reads exec_results from the dataset (produced by AutoLinkScaler) and
    revises any failed candidate SQL using a multi-turn chat loop.
    """

    NAME = "AutoLinkOptimizer"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm: Optional[LLM] = None,
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/pred_sql",
        max_revise_turns: int = 3,
        open_parallel: bool = True,
        max_workers: Optional[int] = None,
        db_path: Optional[Union[str, PathLike]] = None,
        credential: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            llm=llm,
            is_save=is_save,
            save_dir=save_dir,
            open_parallel=open_parallel,
            max_workers=max_workers,
            **kwargs,
        )
        self.max_revise_turns = max_revise_turns
        self.db_path          = db_path
        self.credential       = credential

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_db_info(self, item):
        row     = self.dataset[item]
        db_type = row.get("db_type", "sqlite")
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
        cred = self.credential or row.get("credential")
        return db_type, db_path, cred

    def _load_schema_df(self, item) -> pd.DataFrame:
        """Load the full schema as a DataFrame (without converting to text)."""
        row    = self.dataset[item]
        schema = None

        sp = row.get("instance_schemas")
        if sp and Path(str(sp)).exists():
            schema = load_dataset(sp)

        if schema is None:
            schema = self.dataset.get_db_schema(item)

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)
        if isinstance(schema, pd.DataFrame):
            return schema
        return pd.DataFrame()

    def _load_schema_text(self, item) -> str:
        row    = self.dataset[item]
        schema = None

        sp = row.get("instance_schemas")
        if sp and Path(str(sp)).exists():
            schema = load_dataset(sp)

        if schema is None:
            schema = self.dataset.get_db_schema(item)

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)
        if isinstance(schema, pd.DataFrame):
            return parse_schema_from_df(schema)
        return str(schema) if schema else ""

    def _call_llm_chat(self, messages: List[Dict]) -> str:
        """Call LLM via llama-index ChatMessage API."""
        chat_msgs = []
        for m in messages:
            role_str = m.get("role", "user").lower()
            role = {
                "system":    MessageRole.SYSTEM,
                "assistant": MessageRole.ASSISTANT,
            }.get(role_str, MessageRole.USER)
            chat_msgs.append(ChatMessage(role=role, content=m.get("content", "")))
        try:
            return self.llm.chat(chat_msgs).message.content
        except Exception:
            # Fallback: concatenate messages and use complete()
            combined = "\n".join(m.get("content", "") for m in messages)
            return self.llm.complete(combined).text

    def _execute_sql_safe(self, sql: str, db_type: str, db_path: str, credential) -> Dict:
        try:
            result = execute_sql(db_type, db_path, sql, credential)
            is_empty = isinstance(result, str) and result.strip().rstrip(".") == _EMPTY_RESULT_MARKER
            return {"success": True, "is_empty": is_empty, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "is_empty": False, "result": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Revision logic for ONE candidate
    # ------------------------------------------------------------------

    def _revise_candidate(
        self,
        sql: str,
        error: str,
        question: str,
        schema_text: str,
        db_type: str,
        db_path: str,
        credential: Any,
        idx: int,
    ) -> str:
        """Try to revise a failed SQL up to max_revise_turns times."""
        dialect = _get_dialect_label(db_type)
        system_prompt = AUTOLINK_REVISE_SYSTEM.format(SQL_DIALECT_NAME=dialect)

        messages = [{"role": "system", "content": system_prompt}]
        current_sql   = sql
        current_error = error

        for turn in range(self.max_revise_turns):
            user_content = AUTOLINK_REVISE_USER.format(
                PROMPT=schema_text,
                QUESTION=question,
                SQL=current_sql,
                ERROR=current_error,
            )
            messages.append({"role": "user", "content": user_content})

            try:
                raw    = self._call_llm_chat(messages)
                new_sql = _extract_sql(raw)
            except Exception as e:
                logger.warning(
                    f"AutoLinkOptimizer: candidate {idx} turn {turn} LLM error: {e}"
                )
                break

            messages.append({"role": "assistant", "content": raw})

            # Verify revised SQL
            exec_res = self._execute_sql_safe(new_sql, db_type, db_path, credential)
            if exec_res["success"] and not exec_res.get("is_empty", False):
                logger.debug(
                    f"AutoLinkOptimizer: candidate {idx} fixed at turn {turn}"
                )
                return new_sql

            if exec_res["success"]:
                # SQL runs but still returns empty
                current_error = "The revised SQL executed successfully but still returned no results. Check column names, join conditions, and filter values."
            else:
                current_error = exec_res.get("error", "Unknown error")
            current_sql = new_sql
            logger.debug(
                f"AutoLinkOptimizer: candidate {idx} turn {turn} still failing: "
                f"{current_error[:80]}"
            )

        # Could not fix – return best attempt (last revised SQL)
        return current_sql

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(
        self,
        item,
        schema=None,
        schema_links=None,
        pred_sql=None,
        data_logger=None,
        **kwargs,
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row         = self.dataset[item]
        question    = row.get("question", "")
        instance_id = str(row.get("instance_id", item))
        db_type, db_path, credential = self._resolve_db_info(item)

        # Load pred_sql candidates
        sql_list, is_single = self.load_pred_sql(pred_sql, item)

        # Load exec_results (from AutoLinkScaler) if available
        exec_results: Optional[List[Dict]] = row.get("exec_results")

        # Build schema text once (shared across all candidates).
        # Use the Parser-filtered schema when schema_links are available.
        schema_df   = self._load_schema_df(item)
        schema_text = self._load_schema_text(item)
        if isinstance(schema_links, dict) and not schema_df.empty:
            schema_text = build_filtered_schema_text(schema_df, schema_links, schema_text)

        # Determine which candidates need revision
        revision_tasks = []  # (idx, sql, error)

        # Count candidates with actual non-empty results (used to detect suspicious empties)
        non_empty_count = sum(
            1 for er in (exec_results or [])
            if er and er.get("success") and not er.get("is_empty",
                isinstance(er.get("result", ""), str)
                and er.get("result", "").strip().rstrip(".") == _EMPTY_RESULT_MARKER
            )
        )

        for i, sql in enumerate(sql_list):
            if exec_results and i < len(exec_results):
                er = exec_results[i]
                if not er.get("success", False):
                    # Execution error → must revise
                    revision_tasks.append((i, sql, er.get("error", "Unknown error")))
                elif er.get("is_empty", isinstance(er.get("result", ""), str)
                            and er.get("result", "").strip().rstrip(".") == _EMPTY_RESULT_MARKER) \
                        and non_empty_count > 0:
                    # Empty result but peers have data → suspicious, try to revise
                    revision_tasks.append((
                        i, sql,
                        "The SQL executed without errors but returned no results. "
                        "Check column names, join conditions, and filter values. "
                        "Rewrite the SQL to correctly answer the question."
                    ))
            else:
                # No exec_results – execute to check
                er = self._execute_sql_safe(sql, db_type, db_path, credential)
                if not er["success"]:
                    revision_tasks.append((i, sql, er.get("error", "Unknown error")))

        if data_logger:
            data_logger.info(
                f"{self.NAME}: {len(revision_tasks)}/{len(sql_list)} candidates need revision"
            )

        revised_sqls = list(sql_list)  # copy

        if revision_tasks:
            workers = min(
                self.max_workers or len(revision_tasks),
                len(revision_tasks),
            ) or 1

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(
                        self._revise_candidate,
                        sql, error, question, schema_text,
                        db_type, db_path, credential, idx,
                    ): idx
                    for idx, sql, error in revision_tasks
                }
                for fut in as_completed(future_map):
                    orig_idx = future_map[fut]
                    try:
                        revised_sqls[orig_idx] = fut.result()
                    except Exception as e:
                        logger.error(
                            f"AutoLinkOptimizer: revision future {orig_idx} raised: {e}"
                        )

        result = self.save_output(revised_sqls, item, instance_id)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return result
