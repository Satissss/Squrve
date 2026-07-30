"""Persisted, redacted outcome reports for offline BiomedSQL pilots."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .models import ResultStatus, SampleResult
from .runner import (
    _redact_literal_values,
    _sensitive_string_values,
    redact_secrets,
)


_REDACTED = "[REDACTED]"
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?ix)\b("
    r"api[_-]?key|access[_-]?token|auth(?:orization)?|"
    r"credential|password|private[_-]?key|secret|service[_-]?account|token"
    r")\s*([:=])\s*([\"']?)[^\s,;\]\}\"']+\3"
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_GOOGLE_OAUTH_ACCESS_TOKEN = re.compile(r"\bya29\.[A-Za-z0-9._-]{20,}\b")
_RECOGNIZABLE_API_TOKEN = re.compile(
    r"\b(?:"
    r"AIza[0-9A-Za-z_-]{35}|"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{30,255}|"
    r"github_pat_[A-Za-z0-9_]{20,}"
    r")\b"
)
_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)
_GOOGLE_SERVICE_ACCOUNT = re.compile(
    r'\{(?=[^{}]{0,8192}"type"\s*:\s*"service_account")[^{}]*\}',
    re.DOTALL,
)


def build_report(
    results: Sequence[SampleResult], *, limitations: Sequence[str] = ()
) -> dict[str, Any]:
    """Build a JSON-compatible pilot report without executing any SQL."""
    secret_values = _sensitive_values_from_results(results)
    status_counts = {status.value: 0 for status in ResultStatus}
    failure_stage_counts: Counter[str] = Counter()
    questions: list[dict[str, Any]] = []
    latencies: list[float] = []

    for result in results:
        status_counts[result.status.value] += 1
        failure_stage = _failure_stage(result)
        if failure_stage:
            failure_stage_counts[failure_stage] += 1
        latency = result.generation.latency_seconds
        if isinstance(latency, Real) and not isinstance(latency, bool) and math.isfinite(latency):
            latencies.append(float(latency))
        questions.append(_question_report(result, failure_stage))

    assert sum(status_counts.values()) == len(results)
    most_common_failure_stage = (
        sorted(failure_stage_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if failure_stage_counts
        else None
    )
    report = {
        "total_samples": len(results),
        "generated_count": sum(
            count
            for status, count in status_counts.items()
            if status != ResultStatus.GENERATION_FAILED.value
        ),
        "execution_success_count": sum(
            status_counts[status.value]
            for status in (
                ResultStatus.EXECUTED_RESULT_MISMATCH,
                ResultStatus.EXECUTED_RESULT_MATCH,
            )
        ),
        "match_count": status_counts[ResultStatus.EXECUTED_RESULT_MATCH.value],
        "status_counts": status_counts,
        "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
        "most_common_failure_stage": most_common_failure_stage,
        "latency_summary": _latency_summary(latencies),
        "limitations": list(limitations),
        "questions": questions,
    }
    return _redact_report_value(report, secret_values)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-readable, redacted account of each pilot sample."""
    clean_report = _redact_report_value(report)
    lines = ["# BMSQL Pilot Outcome Report", "", "## Summary", ""]
    lines.extend(
        (
            f"- Total samples: {clean_report.get('total_samples', 0)}",
            f"- Generated: {clean_report.get('generated_count', 0)}",
            f"- Execution successes: {clean_report.get('execution_success_count', 0)}",
            f"- Result matches: {clean_report.get('match_count', 0)}",
            "- Most common failure stage: "
            f"{_escape_markdown_prose(clean_report.get('most_common_failure_stage') or 'none')}",
            "",
            "### Status counts",
            "",
        )
    )
    for status in ResultStatus:
        lines.append(f"- {status.value}: {clean_report.get('status_counts', {}).get(status.value, 0)}")
    lines.extend(("", "### Failure stages", ""))
    failure_stages = clean_report.get("failure_stage_counts", {})
    if failure_stages:
        lines.extend(
            f"- {_escape_markdown_prose(stage)}: {count}"
            for stage, count in failure_stages.items()
        )
    else:
        lines.append("- none")
    lines.extend(("", "### Latency", ""))
    latency = clean_report.get("latency_summary", {})
    lines.append(
        "- Samples with latency: "
        f"{latency.get('count', 0)}; mean seconds: {latency.get('mean_seconds')}"
    )
    lines.extend(("", "## Limitations", ""))
    limitations = clean_report.get("limitations", [])
    if limitations:
        lines.extend(f"- {_escape_markdown_prose(limitation)}" for limitation in limitations)
    else:
        lines.append("- none recorded")

    lines.extend(("", "## Per-question evidence", ""))
    for question in clean_report.get("questions", []):
        lines.extend(
            (
                "### "
                f"{_escape_markdown_prose(question.get('instance_id', 'unknown'))}: "
                f"{_escape_markdown_prose(question.get('question', ''))}",
                "",
                f"- Status: {_escape_markdown_prose(question.get('status', ''))}",
                "- Failure stage: "
                f"{_escape_markdown_prose(question.get('failure_stage') or 'none')}",
                f"- Error: {_escape_markdown_prose(question.get('error') or 'none')}",
                "",
                "#### Gold SQL",
                "",
                _fenced_code_block(question.get("gold_sql", ""), "sql"),
                "",
                "#### Predicted SQL",
                "",
                _fenced_code_block(question.get("predicted_sql") or "", "sql"),
                "",
                "#### Metadata",
                "",
                _fenced_code_block(
                    _json_for_markdown(
                        {
                            "backend_metadata": question.get("backend_metadata", {}),
                            "model_metadata": question.get("model_metadata", {}),
                            "sample_metadata": question.get("sample_metadata", {}),
                        }
                    ),
                    "json",
                ),
                "",
            )
        )
    return "\n".join(lines)


def write_report(
    results: Sequence[SampleResult],
    output_dir: str | os.PathLike[str],
    *,
    limitations: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Write redacted JSON and Markdown evidence files to *output_dir*."""
    report = build_report(results, limitations=limitations)
    output_path = Path(output_dir)
    json_path = output_path / "report.json"
    markdown_path = output_path / "report.md"
    _atomic_write_strict_json(json_path, report)
    _atomic_write_text(markdown_path, render_markdown(report))
    return json_path, markdown_path


def _question_report(result: SampleResult, failure_stage: str | None) -> dict[str, Any]:
    evaluation = result.evaluation
    return {
        "instance_id": result.instance_id,
        "question": result.question,
        "status": result.status.value,
        "gold_sql": result.gold_sql,
        "predicted_sql": result.generation.pred_sql,
        "error": _concise_error(result),
        "failure_stage": failure_stage,
        "model_metadata": result.generation.model_metadata,
        "backend_metadata": {
            "evaluation": evaluation.metadata,
            "predicted_execution": (
                None if evaluation.predicted is None else evaluation.predicted.metadata
            ),
            "gold_execution": None if evaluation.gold is None else evaluation.gold.metadata,
        },
        "sample_metadata": result.metadata,
    }


def _failure_stage(result: SampleResult) -> str | None:
    stage = result.evaluation.metadata.get("failure_stage") or result.generation.error_stage
    if stage is None:
        return None
    text = str(stage).strip()
    return text or None


def _concise_error(result: SampleResult) -> str | None:
    evaluation = result.evaluation
    error = (
        evaluation.error
        or result.generation.error
        or (None if evaluation.predicted is None else evaluation.predicted.error)
        or (None if evaluation.gold is None else evaluation.gold.error)
    )
    if error is None:
        return None
    return str(error).strip()[:500] or None


def _latency_summary(latencies: Sequence[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {
            "count": 0,
            "min_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
        }
    return {
        "count": len(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
    }


def _sensitive_values_from_results(results: Sequence[SampleResult]) -> set[str]:
    return set().union(
        *(_sensitive_string_values(result.to_dict()) for result in results)
    )


def _redact_report_value(value: Any, sensitive_values: set[str] | None = None) -> Any:
    redacted = redact_secrets(value)
    if sensitive_values:
        redacted = _redact_literal_values(redacted, sensitive_values)
    return _normalize_non_finite_floats(_redact_error_strings(redacted))


def _redact_error_strings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_error_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_error_strings(item) for item in value]
    if isinstance(value, str):
        value = _GOOGLE_SERVICE_ACCOUNT.sub(_REDACTED, value)
        value = _PEM_PRIVATE_KEY.sub(_REDACTED, value)
        value = _RECOGNIZABLE_API_TOKEN.sub(_REDACTED, value)
        value = _GOOGLE_OAUTH_ACCESS_TOKEN.sub(_REDACTED, value)
        value = _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}", value)
        return _BEARER_VALUE.sub(f"Bearer {_REDACTED}", value)
    return value


def _normalize_non_finite_floats(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_non_finite_floats(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _escape_markdown_prose(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    for character in "`*_[]()#+-!>|":
        text = text.replace(character, f"\\{character}")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _fenced_code_block(value: Any, language: str) -> str:
    text = str(value)
    longest_backtick_run = max((len(run) for run in re.findall(r"`+", text)), default=2)
    fence = "`" * max(3, longest_backtick_run + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _json_for_markdown(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _atomic_write_strict_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write_text(path, f"{serialized}\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
