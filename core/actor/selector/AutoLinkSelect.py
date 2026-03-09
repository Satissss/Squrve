"""
AutoLinkSelect.py – SQL selection actor for the AutoLink pipeline.

Corresponds to AutoLink/run/sql_selection.py.

Chooses the best SQL from the N revised candidates produced by
AutoLinkOptimizer using a three-step strategy:

1. **Execution clustering**: group candidates by result-set identity.
2. **Consistency check**: if one cluster has the majority, pick its first member.
3. **LLM pairwise voting**: when no clear winner exists, do round-robin pairwise
   comparisons with the LLM and choose the candidate with the most wins.

Key design decisions
--------------------
- Uses self.llm.complete() for pairwise votes (one call per pair).
- exec_results written by AutoLinkScaler / AutoLinkOptimizer are reused; if
  absent the actor executes the candidates itself.
- Thread-safe: pairwise LLM calls are issued in parallel with ThreadPoolExecutor.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from core.actor.selector.BaseSelect import BaseSelector
from core.data_manage import Dataset, load_dataset, single_central_process
from core.db_connect import execute_sql
from core.utils import parse_schema_from_df
from core.actor.prompts.AutoLinkPrompt import (
    AUTOLINK_SELECT_SYSTEM,
    AUTOLINK_SELECT_USER,
    build_filtered_schema_text,
)

import pandas as pd

_EMPTY_RESULT_MARKERS = {
    "No data found for the specified query",
    "No data found for the specified query.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_key(result: Any) -> str:
    """Stable string key for a result set (for clustering)."""
    try:
        if isinstance(result, list):
            return str(sorted(str(r) for r in result))
        return str(result)
    except Exception:
        return str(result)


def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r"<think>.*?</think>", "", m.group(1), flags=re.DOTALL).strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# AutoLinkSelector
# ---------------------------------------------------------------------------

@BaseSelector.register_actor
class AutoLinkSelector(BaseSelector):
    """Cluster + LLM-vote SQL candidate selector for the AutoLink pipeline."""

    NAME = "AutoLinkSelector"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm=None,
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/pred_sql",
        db_path: Optional[Union[str, PathLike]] = None,
        credential: Optional[Dict] = None,
        max_vote_workers: int = 4,
        **kwargs,
    ):
        super().__init__(dataset=dataset, llm=llm, is_save=is_save, save_dir=save_dir, **kwargs)
        self.db_path          = db_path
        self.credential       = credential
        self.max_vote_workers = max_vote_workers

    # ------------------------------------------------------------------
    # Helpers
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

    def _execute_safe(self, sql: str, db_type: str, db_path: str, credential) -> Dict:
        try:
            result = execute_sql(db_type, db_path, sql, credential)
            is_empty = self._is_empty({"result": result})
            return {"success": True, "is_empty": is_empty, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "is_empty": False, "result": None, "error": str(e)}

    @staticmethod
    def _is_empty(er: dict) -> bool:
        """Backward-compatible empty-result check (handles old and new exec_results format)."""
        if er.get("is_empty") is True:
            return True
        result = er.get("result")
        if isinstance(result, str):
            return result.strip().rstrip(".") in {
                m.rstrip(".") for m in _EMPTY_RESULT_MARKERS
            }
        return False

    def _pairwise_vote(self, sql1: str, sql2: str, question: str, schema_text: str) -> int:
        """Returns 0 if SQL1 wins, 1 if SQL2 wins (or 0 on error)."""
        prompt = (
            AUTOLINK_SELECT_SYSTEM
            + "\n\n"
            + AUTOLINK_SELECT_USER.format(
                PROMPT=schema_text,
                QUESTION=question,
                SQL1=sql1,
                SQL2=sql2,
            )
        )
        try:
            text = self.llm.complete(prompt).text.strip()
            if "SQL2" in text.upper():
                return 1
            return 0
        except Exception as e:
            logger.warning(f"AutoLinkSelector: pairwise vote LLM error: {e}")
            return 0

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select_best(
        self,
        sql_candidates: List[str],
        exec_results: Optional[List[Dict]],
        question: str,
        schema_text: str,
        db_type: str,
        db_path: str,
        credential: Any,
    ) -> str:
        n = len(sql_candidates)
        if n == 0:
            return ""
        if n == 1:
            return sql_candidates[0]

        # ---- Step 1: Obtain execution results ----
        results: List[Dict] = []
        for i, sql in enumerate(sql_candidates):
            if exec_results and i < len(exec_results):
                results.append(exec_results[i])
            else:
                results.append(
                    self._execute_safe(sql, db_type, db_path, credential)
                )

        # ---- Step 2: Categorise into three tiers ----
        non_empty_indices = [
            i for i, er in enumerate(results)
            if er.get("success") and not self._is_empty(er)
        ]
        empty_indices = [
            i for i, er in enumerate(results)
            if er.get("success") and self._is_empty(er)
        ]
        # failed_indices: everything else

        # ---- Step 3: Choose candidate pool ----
        if non_empty_indices:
            # Best case: some candidates have real data – only compete among those
            candidate_pool = non_empty_indices
        elif empty_indices:
            # All successful results are empty.
            # Could be a genuinely empty answer OR all queries are semantically wrong.
            # Fall through to LLM pairwise voting to pick the "most reasonable" SQL.
            logger.warning(
                "AutoLinkSelector: all successful candidates returned empty results; "
                "using LLM voting to select best candidate"
            )
            candidate_pool = empty_indices
        else:
            # All candidates failed execution
            logger.warning("AutoLinkSelector: all candidates failed execution")
            return sql_candidates[0]

        # ---- Step 4: Cluster within pool by result identity ----
        cluster_map: Dict[str, List[int]] = {}
        for i in candidate_pool:
            key = _result_key(results[i]["result"])
            cluster_map.setdefault(key, []).append(i)

        sorted_clusters = sorted(cluster_map.values(), key=len, reverse=True)
        largest_cluster = sorted_clusters[0]
        n_pool = len(candidate_pool)

        # ---- Step 5: Majority cluster (only when there is genuine variety) ----
        # Guard: if all non-empty candidates agree on the same result, take the first.
        # But if all candidates are *empty* and cluster into one group, skip straight to
        # LLM voting (we can't distinguish by result content alone).
        if len(sorted_clusters) > 1 and len(largest_cluster) > n_pool / 2:
            winner_idx = largest_cluster[0]
            logger.debug(
                f"AutoLinkSelector: majority cluster wins "
                f"(size={len(largest_cluster)}/{n_pool})"
            )
            return sql_candidates[winner_idx]

        # Single cluster of non-empty identical results → all agree, take first
        if len(sorted_clusters) == 1 and non_empty_indices:
            winner_idx = largest_cluster[0]
            logger.debug("AutoLinkSelector: single non-empty cluster – all candidates agree")
            return sql_candidates[winner_idx]

        # ---- Step 6: LLM pairwise voting ----
        if self.llm is None:
            # No LLM available, fall back to largest cluster
            return sql_candidates[largest_cluster[0]]

        # Collect candidate indices for voting
        candidate_indices = [i for cluster in sorted_clusters for i in cluster]

        vote_scores: List[int] = [0] * n

        pairs = [
            (candidate_indices[a], candidate_indices[b])
            for a in range(len(candidate_indices))
            for b in range(a + 1, len(candidate_indices))
        ]

        with ThreadPoolExecutor(max_workers=self.max_vote_workers) as pool:
            future_map = {
                pool.submit(
                    self._pairwise_vote,
                    sql_candidates[i], sql_candidates[j],
                    question, schema_text,
                ): (i, j)
                for i, j in pairs
            }
            for fut in as_completed(future_map):
                i, j = future_map[fut]
                try:
                    winner_local = fut.result()  # 0 → i wins, 1 → j wins
                    if winner_local == 0:
                        vote_scores[i] += 1
                    else:
                        vote_scores[j] += 1
                except Exception as e:
                    logger.warning(f"AutoLinkSelector: vote future error: {e}")

        best_idx = max(candidate_indices, key=lambda k: vote_scores[k])
        logger.debug(
            f"AutoLinkSelector: LLM voting winner idx={best_idx}, "
            f"votes={vote_scores[best_idx]}"
        )
        return sql_candidates[best_idx]

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

        # Load candidates
        sql_candidates = self.load_pred_sql(pred_sql, item)

        # Load exec_results if available
        exec_results: Optional[List[Dict]] = row.get("exec_results")

        schema_df   = self._load_schema_df(item)
        schema_text = self._load_schema_text(item)

        # Use the Parser-filtered schema for voting when available.
        # This mirrors how the original AutoLink passes final_schema_prompts/
        # to sql_selection.py instead of the full raw schema.
        if isinstance(schema_links, dict) and not schema_df.empty:
            schema_text = build_filtered_schema_text(schema_df, schema_links, schema_text)

        if data_logger:
            data_logger.info(
                f"{self.NAME}: selecting from {len(sql_candidates)} candidates"
            )

        best_sql = self._select_best(
            sql_candidates=sql_candidates,
            exec_results=exec_results,
            question=question,
            schema_text=schema_text,
            db_type=db_type,
            db_path=db_path,
            credential=credential,
        )

        result = self.save_result(best_sql, item, instance_id)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return result
