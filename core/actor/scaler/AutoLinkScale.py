"""
AutoLinkScale.py – SQL execution / scaling actor for the AutoLink pipeline.

Corresponds to AutoLink/run/sql_execution.py.

Executes the N candidate SQL queries produced by AutoLinkGenerator and
records per-candidate execution outcomes (result, error) in the dataset so
that AutoLinkOptimizer and AutoLinkSelector can use them.

Key design decisions
--------------------
- No LLM required – pure database execution.
- Supports SQLite, Snowflake, BigQuery via core.db_connect.execute_sql.
- Stores execution metadata under dataset[item]["exec_results"] so that
  downstream actors can distinguish successes from failures.
- Thread-safe: each candidate is executed in an independent thread, separate
  SQLite connections are opened per call (thread-safe by design).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset, load_dataset, save_dataset
from core.db_connect import execute_sql

_EMPTY_RESULT_MARKER = "No data found for the specified query"
from core.utils import parse_schema_from_df


# ---------------------------------------------------------------------------
# AutoLinkScaler
# ---------------------------------------------------------------------------

@BaseScaler.register_actor
class AutoLinkScaler(BaseScaler):
    """Execute SQL candidates and record execution outcomes.

    Input  : dataset[item]["pred_sql"]  – str or list[str] of SQL paths/texts
    Output : dataset[item]["pred_sql"]  – unchanged (pass-through list)
             dataset[item]["exec_results"] – list[dict] with keys:
               {sql, success, result, error, time_cost}
    """

    NAME = "AutoLinkScaler"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm: Optional[Any] = None,          # unused; kept for base signature
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/pred_sql",
        sql_timeout: float = 60.0,
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
        self.sql_timeout = sql_timeout
        self.db_path     = db_path
        self.credential  = credential

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
            or getattr(self.dataset, "db_path", None)
        )
        # For SQLite: db_path in config is a directory; construct the actual file path
        # using db_id (e.g. "../benchmarks/bird/dev/database/california_schools.sqlite")
        if db_type == "sqlite" and db_path:
            db_id = row.get("db_id", "")
            path_obj = Path(str(db_path))
            if db_id and path_obj.is_dir():
                db_path = str(path_obj / f"{db_id}.sqlite")
        cred = self.credential or row.get("credential")
        return db_type, db_path, cred

    def _load_sql_candidates(self, item) -> List[str]:
        """Load SQL candidates from dataset field (str or list of str/path)."""
        row      = self.dataset[item]
        pred_sql = row.get("pred_sql")
        if pred_sql is None:
            raise ValueError(f"AutoLinkScaler: pred_sql missing for item {item}")

        if isinstance(pred_sql, str):
            pred_sql = [pred_sql]

        results = []
        for p in pred_sql:
            try:
                if Path(str(p)).exists():
                    results.append(load_dataset(p))
                else:
                    results.append(str(p))
            except Exception:
                results.append(str(p))
        return results

    def _execute_one(
        self,
        sql: str,
        db_type: str,
        db_path: str,
        credential: Any,
        idx: int,
    ) -> Dict[str, Any]:
        """Execute one SQL candidate and return a result dict."""
        start = time.time()
        try:
            result = execute_sql(db_type, db_path, sql, credential)
            elapsed = time.time() - start
            is_empty = isinstance(result, str) and result.strip().rstrip(".") == _EMPTY_RESULT_MARKER
            logger.debug(
                f"AutoLinkScaler: candidate {idx} OK (empty={is_empty}, {elapsed:.2f}s)"
            )
            return {
                "sql":       sql,
                "success":   True,
                "is_empty":  is_empty,
                "result":    result,
                "error":     None,
                "time_cost": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(
                f"AutoLinkScaler: candidate {idx} failed: {e}"
            )
            return {
                "sql":       sql,
                "success":   False,
                "is_empty":  False,
                "result":    None,
                "error":     str(e),
                "time_cost": elapsed,
            }

    # ------------------------------------------------------------------
    # act()
    # ------------------------------------------------------------------

    def act(
        self,
        item,
        schema=None,
        schema_links=None,
        sub_questions=None,
        data_logger=None,
        **kwargs,
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        sql_candidates = self._load_sql_candidates(item)
        db_type, db_path, credential = self._resolve_db_info(item)

        if data_logger:
            data_logger.info(
                f"{self.NAME}: executing {len(sql_candidates)} candidates "
                f"(db_type={db_type})"
            )

        # Execute candidates in parallel threads (each opens its own DB connection)
        exec_results: List[Dict] = [None] * len(sql_candidates)  # type: ignore[list-item]

        workers = min(
            self.max_workers or len(sql_candidates),
            len(sql_candidates),
        ) or 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    self._execute_one,
                    sql, db_type, db_path, credential, i,
                ): i
                for i, sql in enumerate(sql_candidates)
            }
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    exec_results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"AutoLinkScaler: future {idx} raised: {e}")
                    exec_results[idx] = {
                        "sql": sql_candidates[idx],
                        "success": False,
                        "result": None,
                        "error": str(e),
                        "time_cost": 100_000,
                    }

        # Persist execution metadata to dataset (used by Optimizer + Selector)
        self.dataset.setitem(item, "exec_results", exec_results)

        # Pass-through pred_sql unchanged (Optimizer will read exec_results)
        self.save_output(sql_candidates, item)

        n_ok  = sum(1 for r in exec_results if r["success"])
        n_err = len(exec_results) - n_ok
        if data_logger:
            data_logger.info(
                f"{self.NAME}.act end | item={item} | "
                f"success={n_ok}, failed={n_err}"
            )

        return sql_candidates
