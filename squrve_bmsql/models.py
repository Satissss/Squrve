"""Serializable records used by the offline BiomedSQL pilot."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


class ResultStatus(str, Enum):
    GENERATED_NOT_EXECUTED = "generated_not_executed"
    GENERATION_FAILED = "generation_failed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTED_RESULT_MISMATCH = "executed_result_mismatch"
    EXECUTED_RESULT_MATCH = "executed_result_match"


_NATIVE_VALUE_TAG = "__squrve_bmsql_json_v2__"
_NATIVE_KINDS = frozenset({"bytes", "date", "datetime", "decimal", "float", "time"})


def _native_envelope(kind: str, payload: str) -> dict[str, list[str]]:
    return {_NATIVE_VALUE_TAG: [kind, payload]}


def _native_parts(value: Any) -> tuple[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != {_NATIVE_VALUE_TAG}:
        return None
    payload = value[_NATIVE_VALUE_TAG]
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[0], str):
        return None
    return payload[0], payload[1]


def encode_json_value(value: Any) -> Any:
    """Encode supported values as strict JSON, rejecting unknown native objects."""
    if isinstance(value, Enum):
        return encode_json_value(value.value)
    if isinstance(value, Mapping):
        encoded = {str(key): encode_json_value(item) for key, item in value.items()}
        if _native_parts(encoded) is not None:
            return {
                _NATIVE_VALUE_TAG: [
                    "mapping",
                    [[key, item] for key, item in encoded.items()],
                ]
            }
        return encoded
    if isinstance(value, (list, tuple)):
        return [encode_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return _native_envelope("decimal", str(value))
    if isinstance(value, datetime):
        return _native_envelope("datetime", value.isoformat())
    if isinstance(value, date):
        return _native_envelope("date", value.isoformat())
    if isinstance(value, time):
        return _native_envelope("time", value.isoformat())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _native_envelope(
            "bytes", base64.b64encode(bytes(value)).decode("ascii")
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            payload = "nan"
        else:
            payload = "-infinity" if value < 0 else "+infinity"
        return _native_envelope("float", payload)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    raise TypeError(f"Unsupported JSON value type: {type_name}")


def decode_json_value(value: Any) -> Any:
    """Decode values emitted by :func:`encode_json_value`."""
    tagged = _native_parts(value)
    if tagged is not None:
        kind, payload = tagged
        if kind == "mapping":
            if not isinstance(payload, list):
                raise ValueError("Invalid encoded mapping payload")
            decoded: dict[str, Any] = {}
            for pair in payload:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise ValueError("Invalid encoded mapping entry")
                decoded[pair[0]] = decode_json_value(pair[1])
            return decoded
        if kind in _NATIVE_KINDS and not isinstance(payload, str):
            raise ValueError(f"Invalid encoded {kind} payload")
        if kind == "decimal":
            return Decimal(payload)
        if kind == "datetime":
            return datetime.fromisoformat(payload)
        if kind == "date":
            return date.fromisoformat(payload)
        if kind == "time":
            return time.fromisoformat(payload)
        if kind == "bytes":
            return base64.b64decode(payload.encode("ascii"), validate=True)
        if kind == "float":
            values = {
                "nan": float("nan"),
                "+infinity": float("inf"),
                "-infinity": float("-inf"),
            }
            if payload not in values:
                raise ValueError("Invalid encoded float payload")
            return values[payload]
    if isinstance(value, Mapping):
        return {str(key): decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    raise TypeError(f"Unsupported JSON value type: {type_name}")


def normalize_json_value(value: Any) -> Any:
    """Validate and normalize a supported native value without losing its type."""
    return decode_json_value(encode_json_value(value))


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
        return encode_json_value({
            "instance_id": self.instance_id,
            "question": self.question,
            "schema": self.schema,
            "domain_context": self.domain_context,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BMSQLRequest":
        value = decode_json_value(value)
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
        return encode_json_value({
            "pred_sql": self.pred_sql,
            "raw_response": self.raw_response,
            "stage_outputs": self.stage_outputs,
            "trajectory": self.trajectory,
            "error": self.error,
            "error_stage": self.error_stage,
            "latency_seconds": self.latency_seconds,
            "model_metadata": self.model_metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BMSQLGeneration":
        value = decode_json_value(value)
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

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("success must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return encode_json_value({
            "success": self.success,
            "rows": self.rows,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryExecution":
        value = decode_json_value(value)
        success = value["success"]
        if type(success) is not bool:
            raise TypeError("success must be a bool")
        return cls(
            success=success,
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
        return encode_json_value({
            "status": self.status.value,
            "predicted": None if self.predicted is None else self.predicted.to_dict(),
            "gold": None if self.gold is None else self.gold.to_dict(),
            "error": self.error,
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Evaluation":
        value = decode_json_value(value)
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
        return encode_json_value({
            "instance_id": self.instance_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "generation": self.generation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "metadata": self.metadata,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleResult":
        value = decode_json_value(value)
        return cls(
            instance_id=str(value["instance_id"]),
            question=str(value["question"]),
            gold_sql=str(value["gold_sql"]),
            generation=BMSQLGeneration.from_dict(value["generation"]),
            evaluation=Evaluation.from_dict(value["evaluation"]),
            metadata=_dict(value.get("metadata")),
        )
