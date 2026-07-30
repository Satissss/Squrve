"""Serializable records used by the offline BiomedSQL pilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ResultStatus(str, Enum):
    GENERATED_NOT_EXECUTED = "generated_not_executed"
    GENERATION_FAILED = "generation_failed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTED_RESULT_MISMATCH = "executed_result_mismatch"
    EXECUTED_RESULT_MATCH = "executed_result_match"


def _json_value(value: Any) -> Any:
    """Return a JSON-compatible copy without requiring a custom encoder."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass
class BMSQLRequest:
    instance_id: str
    question: str
    schema: list[dict[str, Any]] = field(default_factory=list)
    domain_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "question": self.question,
            "schema": _json_value(self.schema),
            "domain_context": self.domain_context,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BMSQLRequest":
        return cls(
            instance_id=str(value["instance_id"]),
            question=str(value["question"]),
            schema=[dict(item) for item in value.get("schema", [])],
            domain_context=value.get("domain_context"),
            metadata=_dict(value.get("metadata")),
        )


@dataclass
class BMSQLGeneration:
    pred_sql: str | None = None
    raw_response: Any = None
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    trajectory: list[Any] = field(default_factory=list)
    error: str | None = None
    error_stage: str | None = None
    latency_seconds: float | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        error_stage: str | None = None,
        latency_seconds: float | None = None,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> "BMSQLGeneration":
        return cls(
            error=error,
            error_stage=error_stage,
            latency_seconds=latency_seconds,
            model_metadata=_dict(model_metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred_sql": self.pred_sql,
            "raw_response": _json_value(self.raw_response),
            "stage_outputs": _json_value(self.stage_outputs),
            "trajectory": _json_value(self.trajectory),
            "error": self.error,
            "error_stage": self.error_stage,
            "latency_seconds": self.latency_seconds,
            "model_metadata": _json_value(self.model_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BMSQLGeneration":
        return cls(
            pred_sql=value.get("pred_sql"),
            raw_response=value.get("raw_response"),
            stage_outputs=_dict(value.get("stage_outputs")),
            trajectory=list(value.get("trajectory", [])),
            error=value.get("error"),
            error_stage=value.get("error_stage"),
            latency_seconds=value.get("latency_seconds"),
            model_metadata=_dict(value.get("model_metadata")),
        )


@dataclass
class QueryExecution:
    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rows": _json_value(self.rows),
            "error": self.error,
            "error_type": self.error_type,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryExecution":
        return cls(
            success=bool(value["success"]),
            rows=[dict(row) for row in value.get("rows", [])],
            error=value.get("error"),
            error_type=value.get("error_type"),
            metadata=_dict(value.get("metadata")),
        )


@dataclass
class Evaluation:
    status: ResultStatus
    predicted: QueryExecution | None = None
    gold: QueryExecution | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResultStatus):
            self.status = ResultStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "predicted": None if self.predicted is None else self.predicted.to_dict(),
            "gold": None if self.gold is None else self.gold.to_dict(),
            "error": self.error,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Evaluation":
        predicted = value.get("predicted")
        gold = value.get("gold")
        return cls(
            status=ResultStatus(value["status"]),
            predicted=None if predicted is None else QueryExecution.from_dict(predicted),
            gold=None if gold is None else QueryExecution.from_dict(gold),
            error=value.get("error"),
            metadata=_dict(value.get("metadata")),
        )


@dataclass
class SampleResult:
    instance_id: str
    question: str
    gold_sql: str
    generation: BMSQLGeneration
    evaluation: Evaluation
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be non-empty")

    @property
    def status(self) -> ResultStatus:
        return self.evaluation.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "generation": self.generation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleResult":
        return cls(
            instance_id=str(value["instance_id"]),
            question=str(value["question"]),
            gold_sql=str(value["gold_sql"]),
            generation=BMSQLGeneration.from_dict(value["generation"]),
            evaluation=Evaluation.from_dict(value["evaluation"]),
            metadata=_dict(value.get("metadata")),
        )
