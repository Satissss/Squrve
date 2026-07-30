"""Run a prepared BMSQL pilot in explicit mock/offline mode."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from squrve_bmsql.bmsql_backend import MockBMSQLBackend
from squrve_bmsql.evaluator import Evaluator
from squrve_bmsql.report import write_report
from squrve_bmsql.runner import PilotRunner


_ROW_FIELDS = frozenset(
    {"instance_id", "question", "query", "db_id", "db_type", "external", "metadata"}
)
_SCHEMA_FIELDS = frozenset(
    {
        "db_id",
        "db_type",
        "table_name",
        "column_name",
        "column_types",
        "column_descriptions",
        "sample_rows",
        "table_to_projDataset",
    }
)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    if manifest.get("version") != 1:
        raise ValueError("manifest version is not supported")
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest seed must be an integer")
    rows = manifest.get("rows")
    selected_ids = manifest.get("selected_ids")
    schema = manifest.get("schema")
    if (
        not isinstance(rows, list)
        or len(rows) != 20
        or not all(isinstance(row, Mapping) for row in rows)
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 20
        or not all(isinstance(instance_id, str) and instance_id for instance_id in selected_ids)
        or not isinstance(schema, list)
        or not schema
        or not all(isinstance(column, Mapping) for column in schema)
    ):
        raise ValueError("manifest must contain 20 normalized rows and a normalized schema")
    row_ids = [row.get("instance_id") for row in rows]
    if (
        any(not _ROW_FIELDS.issubset(row) for row in rows)
        or any(not isinstance(instance_id, str) or not instance_id for instance_id in row_ids)
        or len(set(row_ids)) != 20
        or selected_ids != row_ids
        or any(not isinstance(row["metadata"], Mapping) for row in rows)
        or any(not _SCHEMA_FIELDS.issubset(column) for column in schema)
    ):
        raise ValueError("manifest normalized content is inconsistent")
    run_config = manifest.get("run_config")
    if (
        not isinstance(run_config, Mapping)
        or run_config.get("backend") != "mock"
        or run_config.get("mode") != "offline"
        or not isinstance(run_config.get("limitations"), list)
        or not all(isinstance(value, str) for value in run_config["limitations"])
    ):
        raise ValueError("manifest must specify mock/offline run configuration")
    return dict(manifest)


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
