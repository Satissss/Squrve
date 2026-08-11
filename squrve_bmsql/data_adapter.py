"""Adapt official BiomedSQL benchmark rows for Squrve."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any


_UNSAFE_PLACEHOLDER_VALUE = re.compile(r"[^A-Za-z0-9_-]")


def _stable_id(row: Mapping[str, Any]) -> str:
    value = row.get("instance_id", row.get("uuid"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("row is missing a stable instance_id or uuid")
    return value


def _validate_placeholder_value(name: str, value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or not value or _UNSAFE_PLACEHOLDER_VALUE.search(value)
    ):
        raise ValueError(f"unsafe {name} placeholder value")


def substitute_sql_placeholders(
    sql: str, project_id: str | None = None, dataset_name: str | None = None
) -> str:
    """Safely substitute optional BigQuery project and dataset placeholders."""
    if not isinstance(sql, str):
        raise ValueError("sql must be a string")
    _validate_placeholder_value("project_id", project_id)
    _validate_placeholder_value("dataset_name", dataset_name)
    if project_id is not None:
        sql = sql.replace("{project_id}", project_id)
    if dataset_name is not None:
        sql = sql.replace("{dataset_name}", dataset_name)
    return sql


def adapt_biomedsql_row(
    row: Mapping[str, Any],
    *,
    db_id: str = "biomedsql",
    db_type: str = "big_query",
    project_id: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Map the official BiomedSQL field names to the Squrve row convention."""
    instance_id = _stable_id(row)
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    query = row.get("benchmark_query", row.get("query"))
    if not isinstance(query, str):
        raise ValueError("benchmark_query must be a string")

    metadata = {
        "template_uuid": row.get("template_uuid"),
        "answer": row.get("answer"),
        "bio_category": row.get("bio_category"),
    }
    return {
        "instance_id": instance_id,
        "question": question,
        "query": substitute_sql_placeholders(query, project_id, dataset_name),
        "db_id": db_id,
        "db_type": db_type,
        "external": row.get("bio_category"),
        "metadata": metadata,
    }


def select_pilot_rows(
    rows: Sequence[Mapping[str, Any]], *, sample_size: int = 20, seed: int = 20260730
) -> list[dict[str, Any]]:
    """Choose a stable seeded pilot sample independent of source ordering."""
    ordered = sorted((dict(row) for row in rows), key=_stable_id)
    identifiers = [_stable_id(row) for row in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate stable IDs are not allowed")
    if len(ordered) < sample_size:
        raise ValueError(f"Need at least {sample_size} rows, received {len(ordered)}")
    selected = random.Random(seed).sample(ordered, sample_size)
    return sorted(selected, key=_stable_id)
