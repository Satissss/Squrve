"""Offline-safe boundary for calling an injected BiomedSQL agent."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .models import BMSQLGeneration, BMSQLRequest


class BMSQLBackend(Protocol):
    """Generate SQL for one normalized BiomedSQL request."""

    def generate(self, request: BMSQLRequest) -> BMSQLGeneration:
        """Return a normalized generation result."""


def _clean_sql(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _trajectory_from_outputs(raw: tuple[Any, Any, Any, Any, Any, Any]) -> list[dict[str, Any]]:
    stages = (
        "general_sql_query",
        "general_exec_results",
        "refined_sql_query",
        "refined_exec_results",
        "answer",
        "input_tokens",
    )
    return [{"stage": stage, "output": value} for stage, value in zip(stages, raw)]


class MockBMSQLBackend:
    """Deterministic backend for offline runs and tests."""

    def __init__(
        self,
        sql_by_id: Mapping[str, str] | None = None,
        failure_ids: Mapping[str, str] | None = None,
    ) -> None:
        self.sql_by_id = dict(sql_by_id or {})
        self.failure_ids = dict(failure_ids or {})

    def generate(self, request: BMSQLRequest) -> BMSQLGeneration:
        started = time.perf_counter()
        metadata = {"backend": "mock"}
        if request.instance_id in self.failure_ids:
            return BMSQLGeneration.failure(
                self.failure_ids[request.instance_id],
                error_stage="mock",
                latency_seconds=time.perf_counter() - started,
                model_metadata=metadata,
            )

        pred_sql = self.sql_by_id.get(request.instance_id, "SELECT 1 AS mock_value")
        return BMSQLGeneration(
            pred_sql=pred_sql,
            raw_response=pred_sql,
            stage_outputs={"mock_sql_query": pred_sql},
            trajectory=[{"stage": "mock_sql_query", "output": pred_sql}],
            latency_seconds=time.perf_counter() - started,
            model_metadata=metadata,
        )


class UpstreamBMSQLBackend:
    """Thin wrapper around an injected original BMSQL ``SQLAgent`` instance."""

    def __init__(
        self,
        agent: Any = None,
        agent_factory: Callable[[BMSQLRequest], Any] | None = None,
        num_passes: int = 1,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.agent_factory = agent_factory
        self.num_passes = num_passes
        self.model_metadata = dict(model_metadata or {})

    def generate(self, request: BMSQLRequest) -> BMSQLGeneration:
        started = time.perf_counter()
        try:
            agent = self.agent_factory(request) if self.agent_factory else self.agent
            raw = agent.run_agent(question=request.question, num_passes=self.num_passes)
            if not isinstance(raw, tuple) or len(raw) != 6:
                raise ValueError("BMSQL agent returned an invalid result tuple")
            general_sql, general_rows, refined_sql, refined_rows, answer, tokens = raw
            pred_sql = _clean_sql(refined_sql) or _clean_sql(general_sql)
            if not pred_sql:
                raise ValueError("BMSQL returned no SQL")
            return BMSQLGeneration(
                pred_sql=pred_sql,
                raw_response=raw,
                stage_outputs={
                    "general_sql_query": general_sql,
                    "general_exec_results": general_rows,
                    "refined_sql_query": refined_sql,
                    "refined_exec_results": refined_rows,
                    "answer": answer,
                    "input_tokens": tokens,
                },
                trajectory=_trajectory_from_outputs(raw),
                latency_seconds=time.perf_counter() - started,
                model_metadata=self.model_metadata,
            )
        except Exception as exc:
            return BMSQLGeneration.failure(
                str(exc),
                error_stage="upstream_agent",
                latency_seconds=time.perf_counter() - started,
                model_metadata=self.model_metadata,
            )
