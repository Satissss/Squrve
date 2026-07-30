"""Offline-safe evaluation and injected read-only BigQuery execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable

from .models import (
    BMSQLGeneration,
    Evaluation,
    QueryExecution,
    ResultStatus,
)


_MUTATION_KEYWORDS = frozenset(
    {
        "ALTER",
        "CALL",
        "CREATE",
        "DELETE",
        "DROP",
        "EXPORT",
        "GRANT",
        "INSERT",
        "LOAD",
        "MERGE",
        "RENAME",
        "REPLACE",
        "REVOKE",
        "TRUNCATE",
        "UPDATE",
    }
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _masked_sql(sql: str) -> str | None:
    """Mask comments and quoted values while preserving statement punctuation."""
    output: list[str] = []
    index = 0
    state = "normal"
    quote = ""

    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""

        if state == "normal":
            if char == "-" and following == "-":
                output.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "#":
                output.append(" ")
                index += 1
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                output.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char in {"'", '"', "`"}:
                quote = char
                output.append(" ")
                index += 1
                state = "quote"
                continue
            output.append(char)
            index += 1
            continue

        if state == "line_comment":
            output.append("\n" if char == "\n" else " ")
            index += 1
            if char == "\n":
                state = "normal"
            continue

        if state == "block_comment":
            if char == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                state = "normal"
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue

        if char == "\\" and following:
            output.extend((" ", " "))
            index += 2
            continue
        if char == quote:
            if following == quote:
                output.extend((" ", " "))
                index += 2
            else:
                output.append(" ")
                index += 1
                state = "normal"
            continue
        output.append("\n" if char == "\n" else " ")
        index += 1

    if state in {"block_comment", "quote"}:
        return None
    return "".join(output)


def is_read_only_sql(sql: str) -> bool:
    """Return whether *sql* is one read-only SELECT or WITH query."""
    if not isinstance(sql, str) or not sql.strip():
        return False

    masked = _masked_sql(sql)
    if masked is None:
        return False

    semicolons = [index for index, char in enumerate(masked) if char == ";"]
    if semicolons:
        if len(semicolons) != 1 or masked[semicolons[0] + 1 :].strip():
            return False
        masked = masked[: semicolons[0]]

    tokens = [token.upper() for token in _WORD_RE.findall(masked)]
    if not tokens or tokens[0] not in {"SELECT", "WITH"}:
        return False
    if tokens[0] == "WITH" and "SELECT" not in tokens:
        return False
    return not _MUTATION_KEYWORDS.intersection(tokens)


def _canonical_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        items = (
            (str(key), _canonical_value(item))
            for key, item in value.items()
        )
        return ("mapping", tuple(sorted(items, key=lambda pair: pair[0])))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_canonical_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return ("set", tuple(sorted(items, key=repr)))
    return (
        "scalar",
        f"{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


def canonical_rows(rows: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    """Canonicalize rows as an order-insensitive multiset."""
    canonical = [_canonical_value(row) for row in rows]
    return tuple(sorted(canonical, key=repr))


class Evaluator:
    def __init__(self, executor: Any | None = None):
        self.executor = executor

    def evaluate(
        self,
        generation: BMSQLGeneration,
        *,
        gold_sql: str,
        db_id: str | None = None,
    ) -> Evaluation:
        if not generation.pred_sql:
            return Evaluation(
                status=ResultStatus.GENERATION_FAILED,
                error=generation.error or "generation produced no SQL",
                metadata={"failure_stage": generation.error_stage or "generation"},
            )
        if self.executor is None:
            return Evaluation(status=ResultStatus.GENERATED_NOT_EXECUTED)

        predicted = self.executor.execute(generation.pred_sql, db_id=db_id)
        if not predicted.success:
            return Evaluation(
                status=ResultStatus.EXECUTION_FAILED,
                predicted=predicted,
                error=predicted.error,
                metadata={"failure_stage": "predicted"},
            )

        gold = self.executor.execute(gold_sql, db_id=db_id)
        if not gold.success:
            return Evaluation(
                status=ResultStatus.EXECUTION_FAILED,
                predicted=predicted,
                gold=gold,
                error=gold.error,
                metadata={"failure_stage": "gold"},
            )

        status = (
            ResultStatus.EXECUTED_RESULT_MATCH
            if canonical_rows(predicted.rows) == canonical_rows(gold.rows)
            else ResultStatus.EXECUTED_RESULT_MISMATCH
        )
        return Evaluation(status=status, predicted=predicted, gold=gold)


def _classify_error(error: Exception) -> str:
    class_names = " ".join(
        cls.__name__.lower() for cls in type(error).__mro__
    )
    message = str(error).lower()

    if (
        isinstance(error, TimeoutError)
        or "timeout" in class_names
        or "deadlineexceeded" in class_names
        or "timed out" in message
        or "deadline exceeded" in message
    ):
        return "timeout"
    if (
        isinstance(error, PermissionError)
        or "forbidden" in class_names
        or "permissiondenied" in class_names
        or "permission" in message
        or "access denied" in message
    ):
        return "permission"
    if (
        "badrequest" in class_names
        or "invalidargument" in class_names
        or "syntax" in message
        or "parse error" in message
    ):
        return "syntax"
    return "execution"


class BigQueryReadOnlyExecutor:
    """Execute validated queries through an injected BigQuery-compatible client."""

    def __init__(
        self,
        client: Any,
        *,
        maximum_bytes_billed: int | None = None,
        timeout_seconds: float | None = None,
        query_job_config_factory: Callable[[], Any] | None = None,
    ):
        self.client = client
        self.maximum_bytes_billed = maximum_bytes_billed
        self.timeout_seconds = timeout_seconds
        self.query_job_config_factory = query_job_config_factory

    def _new_query_config(self) -> Any:
        if self.query_job_config_factory is not None:
            return self.query_job_config_factory()

        from google.cloud import bigquery

        return bigquery.QueryJobConfig()

    def execute(
        self,
        sql: str,
        *,
        db_id: str | None = None,
    ) -> QueryExecution:
        if not is_read_only_sql(sql):
            return QueryExecution(
                success=False,
                error="Only one read-only SELECT or WITH query is allowed",
                error_type="unsafe_sql",
            )

        try:
            config = self._new_query_config()
            config.use_legacy_sql = False
            if self.maximum_bytes_billed is not None:
                config.maximum_bytes_billed = self.maximum_bytes_billed
            if db_id is not None:
                config.default_dataset = db_id

            job = self.client.query(sql, job_config=config)
            if self.timeout_seconds is None:
                result = job.result()
            else:
                result = job.result(timeout=self.timeout_seconds)
            rows = [dict(row) for row in result]
            metadata = {"db_id": db_id}
            for attribute in ("job_id", "total_bytes_processed"):
                value = getattr(job, attribute, None)
                if value is not None:
                    metadata[attribute] = value
            return QueryExecution(success=True, rows=rows, metadata=metadata)
        except Exception as error:
            return QueryExecution(
                success=False,
                error=str(error),
                error_type=_classify_error(error),
                metadata={"db_id": db_id},
            )
