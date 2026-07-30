"""Prepare a deterministic, normalized 20-sample BMSQL pilot manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from squrve_bmsql.data_adapter import adapt_biomedsql_row, select_pilot_rows
from squrve_bmsql.schema_adapter import to_squrve_parallel_schema


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pilot_20.yaml"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    value = _read_json(path)
    if isinstance(value, Mapping):
        value = value.get("rows")
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError("benchmark must be a JSON row list, {'rows': [...]}, or CSV file")
    return [dict(row) for row in value]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("pilot configuration must be a mapping")
    return dict(value)


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def build_manifest(
    benchmark_path: Path, schema_path: Path, config_path: Path
) -> dict[str, Any]:
    """Build a manifest whose settings are whitelisted from the pilot config."""
    config = _load_config(config_path)
    sample_size = config.get("sample_size", 20)
    seed = config.get("seed", 20260730)
    if sample_size != 20 or isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("pilot sample_size must be exactly 20")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("pilot seed must be an integer")

    db_id = config.get("db_id", "biomedsql")
    db_type = config.get("db_type", "big_query")
    if not isinstance(db_id, str) or not db_id or not isinstance(db_type, str) or not db_type:
        raise ValueError("pilot database settings must be non-empty strings")
    project_id = _optional_text(config.get("project_id"), "project_id")
    dataset_name = _optional_text(config.get("dataset_name"), "dataset_name")

    selected_source_rows = select_pilot_rows(
        _load_rows(benchmark_path), sample_size=sample_size, seed=seed
    )
    normalized_rows = [
        adapt_biomedsql_row(
            row,
            db_id=db_id,
            db_type=db_type,
            project_id=project_id,
            dataset_name=dataset_name,
        )
        for row in selected_source_rows
    ]
    schema = to_squrve_parallel_schema(
        _read_json(schema_path),
        db_id=db_id,
        db_type=db_type,
        project_id=project_id,
        dataset_name=dataset_name,
    )
    return {
        "version": 1,
        "seed": seed,
        "selected_ids": [row["instance_id"] for row in normalized_rows],
        "rows": normalized_rows,
        "schema": schema,
        "run_config": {
            "backend": "mock",
            "mode": "offline",
            "limitations": [
                "Mock SQL validates Squrve wiring only; it does not evaluate model or query quality."
            ],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_manifest(args.benchmark, args.schema, args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError):
        print("prepare_pilot: unable to prepare pilot manifest", file=__import__("sys").stderr)
        return 2
    print(f"samples: {len(manifest['rows'])}")
    print(f"manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
