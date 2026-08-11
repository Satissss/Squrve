"""Run a prepared BMSQL pilot in explicit mock/offline mode."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FIXED_SEED = 20260730
_TOP_LEVEL_FIELDS = frozenset(
    {"version", "seed", "selected_ids", "rows", "schema", "run_config"}
)
_ROW_FIELD_ORDER = (
    "instance_id",
    "question",
    "query",
    "db_id",
    "db_type",
    "external",
    "metadata",
)
_ROW_FIELDS = frozenset(_ROW_FIELD_ORDER)
_SCHEMA_FIELD_ORDER = (
    "db_id",
    "db_type",
    "table_name",
    "column_name",
    "column_types",
    "column_descriptions",
    "sample_rows",
    "table_to_projDataset",
)
_SCHEMA_FIELDS = frozenset(_SCHEMA_FIELD_ORDER)
_RUN_CONFIG_FIELDS = frozenset({"backend", "mode", "limitations"})
_OFFLINE_LIMITATION = (
    "Mock SQL validates Squrve wiring only; it does not evaluate model or query quality."
)
_SAFE_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
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
        or "authorization" in normalized
    )


def _is_safe_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_safe_json(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str)
            and not _is_sensitive_key(key)
            and _is_safe_json(item)
            for key, item in value.items()
        )
    return False


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_path_safe_instance_id(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_INSTANCE_ID.fullmatch(value) is not None


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate and rebuild the strict, offline-only pilot manifest contract."""
    if not isinstance(manifest, Mapping) or set(manifest) != _TOP_LEVEL_FIELDS:
        raise ValueError("manifest fields are invalid")
    if manifest.get("version") != 1 or manifest.get("seed") != FIXED_SEED:
        raise ValueError("manifest version or seed is invalid")

    rows = manifest.get("rows")
    selected_ids = manifest.get("selected_ids")
    schema = manifest.get("schema")
    if (
        not isinstance(rows, list)
        or len(rows) != 20
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 20
        or not isinstance(schema, list)
        or not schema
    ):
        raise ValueError("manifest pilot collections are invalid")
    if not all(_is_path_safe_instance_id(instance_id) for instance_id in selected_ids):
        raise ValueError("manifest instance IDs are invalid")
    if len(set(selected_ids)) != 20:
        raise ValueError("manifest instance IDs must be unique")

    normalized_rows: list[dict[str, Any]] = []
    database_pair: tuple[str, str] | None = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise ValueError("manifest row fields are invalid")
        if not all(
            _is_nonempty_text(row[field])
            for field in ("question", "query", "db_id", "db_type")
        ):
            raise ValueError("manifest row text is invalid")
        if not _is_path_safe_instance_id(row["instance_id"]):
            raise ValueError("manifest row instance ID is invalid")
        if row["external"] is not None and not isinstance(row["external"], str):
            raise ValueError("manifest row external context is invalid")
        if not isinstance(row["metadata"], Mapping) or not _is_safe_json(row["metadata"]):
            raise ValueError("manifest row metadata is invalid")
        row_database_pair = (row["db_id"], row["db_type"])
        if database_pair is None:
            database_pair = row_database_pair
        elif row_database_pair != database_pair:
            raise ValueError("manifest row database settings are inconsistent")
        normalized_rows.append({field: row[field] for field in _ROW_FIELD_ORDER})

    row_ids = [row["instance_id"] for row in normalized_rows]
    if selected_ids != row_ids:
        raise ValueError("manifest selected IDs do not match rows")

    normalized_schema: list[dict[str, Any]] = []
    for column in schema:
        if not isinstance(column, Mapping) or set(column) != _SCHEMA_FIELDS:
            raise ValueError("manifest schema fields are invalid")
        if not all(
            _is_nonempty_text(column[field])
            for field in ("db_id", "db_type", "table_name", "column_name")
        ):
            raise ValueError("manifest schema text is invalid")
        if (column["db_id"], column["db_type"]) != database_pair:
            raise ValueError("manifest schema database settings are inconsistent")
        if not isinstance(column["column_types"], str):
            raise ValueError("manifest schema column type is invalid")
        if not isinstance(column["column_descriptions"], str):
            raise ValueError("manifest schema description is invalid")
        if not isinstance(column["sample_rows"], list) or not _is_safe_json(column["sample_rows"]):
            raise ValueError("manifest schema samples are invalid")
        table_dataset = column["table_to_projDataset"]
        if table_dataset is not None and not _is_nonempty_text(table_dataset):
            raise ValueError("manifest schema dataset is invalid")
        normalized_schema.append(
            {field: column[field] for field in _SCHEMA_FIELD_ORDER}
        )

    run_config = manifest.get("run_config")
    if (
        not isinstance(run_config, Mapping)
        or set(run_config) != _RUN_CONFIG_FIELDS
        or run_config.get("backend") != "mock"
        or run_config.get("mode") != "offline"
        or run_config.get("limitations") != [_OFFLINE_LIMITATION]
    ):
        raise ValueError("manifest run configuration is invalid")

    return {
        "version": 1,
        "seed": FIXED_SEED,
        "selected_ids": list(selected_ids),
        "rows": normalized_rows,
        "schema": normalized_schema,
        "run_config": {
            "backend": "mock",
            "mode": "offline",
            "limitations": [_OFFLINE_LIMITATION],
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    return validate_manifest(manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend != "mock":
        print("run_pilot: unsupported backend", file=sys.stderr)
        return 2
    try:
        manifest = _load_manifest(args.manifest)
        from squrve_bmsql.bmsql_backend import MockBMSQLBackend
        from squrve_bmsql.evaluator import Evaluator
        from squrve_bmsql.report import write_report
        from squrve_bmsql.runner import PilotRunner

        args.output_dir.mkdir(parents=True, exist_ok=True)
        schema_path = args.output_dir / "schema.json"
        schema_path.write_text(
            json.dumps(manifest["schema"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_config = {
            "backend": "mock",
            "mode": "offline",
            "seed": manifest.get("seed"),
        }
        results = PilotRunner(
            rows=manifest["rows"],
            schema=manifest["schema"],
            backend=MockBMSQLBackend(),
            evaluator=Evaluator(),
            output_dir=args.output_dir,
            run_config=run_config,
        ).run(resume=not args.no_resume)
        run_config_source = manifest.get("run_config")
        limitations = (
            run_config_source.get("limitations", [])
            if isinstance(run_config_source, Mapping)
            else []
        )
        report_json, report_markdown = write_report(
            results, args.output_dir, limitations=limitations
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        print("run_pilot: unable to run pilot", file=sys.stderr)
        return 2
    print(f"results: {len(results)}")
    print(f"results.json: {args.output_dir / 'results.json'}")
    print(f"report.json: {report_json}")
    print(f"report.md: {report_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
