"""Resumable, offline-safe runner for a BiomedSQL pilot batch."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from core.actor.generator.BMSQLGenerate import BMSQLGenerator
from core.data_manage import Dataset

from .evaluator import Evaluator
from .models import BMSQLGeneration, Evaluation, ResultStatus, SampleResult


_REDACTED = "[REDACTED]"
_NON_SECRET_TOKEN_KEYS = frozenset(
    {"inputtokens", "outputtokens", "tokencount", "totaltokens"}
)


def _is_sensitive_key(key: Any) -> bool:
    normalized = "".join(character for character in str(key).casefold() if character.isalnum())
    if normalized in _NON_SECRET_TOKEN_KEYS or (
        "token" in normalized and normalized.endswith(("count", "counts"))
    ):
        return False
    return (
        "apikey" in normalized
        or "password" in normalized
        or "credential" in normalized
        or "secret" in normalized
        or "privatekey" in normalized
        or "serviceaccount" in normalized
        or "token" in normalized
    )


def redact_secrets(value: Any) -> Any:
    """Return a JSON-compatible copy with recognized secret values removed."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED
                if _is_sensitive_key(key)
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _sensitive_string_values(value: Any) -> set[str]:
    """Find literal values nested beneath sensitive configuration or row keys."""
    if isinstance(value, Mapping):
        values: set[str] = set()
        for key, item in value.items():
            if _is_sensitive_key(key):
                values.update(_all_string_values(item))
            else:
                values.update(_sensitive_string_values(item))
        return values
    if isinstance(value, (list, tuple, set, frozenset)):
        return set().union(*(_sensitive_string_values(item) for item in value))
    return set()


def _all_string_values(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set().union(*(_all_string_values(item) for item in value.values()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return set().union(*(_all_string_values(item) for item in value))
    return {value} if isinstance(value, str) and value else set()


def _redact_literal_values(value: Any, sensitive_values: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_literal_values(item, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_literal_values(item, sensitive_values) for item in value]
    if isinstance(value, str):
        for sensitive_value in sorted(sensitive_values, key=len, reverse=True):
            value = value.replace(sensitive_value, _REDACTED)
        return value
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace *path* with a complete JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


class PilotRunner:
    """Generate, evaluate, and persist each pilot sample independently."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        schema: Sequence[Mapping[str, Any]],
        backend: Any,
        evaluator: Evaluator,
        output_dir: str | os.PathLike[str],
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.schema = [dict(column) for column in schema]
        self.backend = backend
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.run_config = dict(run_config or {})
        self._secret_values: set[str] = set()

    @property
    def _samples_dir(self) -> Path:
        return self.output_dir / "samples"

    def run(self, resume: bool = True) -> list[SampleResult]:
        """Run unfinished rows and rebuild the aggregate results from checkpoints."""
        self._secret_values = _sensitive_string_values(self.rows)
        self._secret_values.update(_sensitive_string_values(self.run_config))
        self._write_run_metadata()
        in_memory_results: dict[str, SampleResult] = {}
        for row in self.rows:
            instance_id = self._instance_id(row)
            if resume:
                checkpoint = self._read_checkpoint(instance_id)
                if checkpoint is not None:
                    in_memory_results[instance_id] = checkpoint
                    continue
            result = self._run_row(row)
            _atomic_write_json(
                self._checkpoint_path(instance_id), self._persistent_snapshot(result.to_dict())
            )
            in_memory_results[instance_id] = result
            self._write_results_from_checkpoints()
        self._write_results_from_checkpoints()
        return [in_memory_results[self._instance_id(row)] for row in self.rows]

    def _run_row(self, row: Mapping[str, Any]) -> SampleResult:
        instance_id = self._instance_id(row)
        question = row.get("question")
        gold_sql = row.get("query", "")
        try:
            dataset = Dataset(
                data_source=[dict(row)],
                schema_source=str(self.output_dir / "schema.json"),
                is_schema_final=True,
            )
            generator = BMSQLGenerator(
                dataset=dataset,
                backend=self.backend,
                is_save=False,
            )
            generator.act(0, schema=self.schema)
            generation = BMSQLGeneration.from_dict(dataset[0])
            evaluation = self.evaluator.evaluate(
                generation,
                gold_sql=gold_sql,
                db_id=row.get("db_id"),
            )
        except Exception as exc:
            generation = BMSQLGeneration.failure(str(exc), error_stage="runner")
            evaluation = Evaluation(
                status=ResultStatus.GENERATION_FAILED,
                error=str(exc),
                metadata={"failure_stage": "runner"},
            )
        return SampleResult(
            instance_id=instance_id,
            question=question if isinstance(question, str) else str(question),
            gold_sql=gold_sql if isinstance(gold_sql, str) else str(gold_sql),
            generation=generation,
            evaluation=evaluation,
            metadata=dict(row.get("metadata") or {}),
        )

    def _write_run_metadata(self) -> None:
        _atomic_write_json(
            self.output_dir / "run_metadata.json",
            self._persistent_snapshot(
                {
                    "run_config": self.run_config,
                    "sample_count": len(self.rows),
                }
            ),
        )

    def _write_results_from_checkpoints(self) -> list[SampleResult]:
        results = [
            checkpoint
            for row in self.rows
            if (checkpoint := self._read_checkpoint(self._instance_id(row))) is not None
        ]
        _atomic_write_json(
            self.output_dir / "results.json",
            self._persistent_snapshot([result.to_dict() for result in results]),
        )
        return results

    def _persistent_snapshot(self, value: Any) -> Any:
        return _redact_literal_values(redact_secrets(value), self._secret_values)

    def _read_checkpoint(self, instance_id: str) -> SampleResult | None:
        path = self._checkpoint_path(instance_id)
        try:
            with path.open(encoding="utf-8") as stream:
                raw = json.load(stream)
            if not isinstance(raw, Mapping) or raw.get("instance_id") != instance_id:
                return None
            result = SampleResult.from_dict(raw)
            return result if isinstance(result.status, ResultStatus) else None
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _checkpoint_path(self, instance_id: str) -> Path:
        return self._samples_dir / f"{instance_id}.json"

    @staticmethod
    def _instance_id(row: Mapping[str, Any]) -> str:
        value = row.get("instance_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("row instance_id must be non-empty")
        if Path(value).name != value:
            raise ValueError("row instance_id must not contain a path separator")
        return value
