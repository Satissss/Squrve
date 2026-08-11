"""Verify that a local official BiomedSQL checkout is importable.

This check performs no model or BigQuery calls.  Install the dependencies from
the upstream checkout first (``uv sync`` there), then run this command.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from squrve_bmsql.upstream_adapter import load_official_classes, upstream_revision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault("PROJECT_ID", "smoke-project")
    os.environ.setdefault("DATASET_NAME", "smoke-dataset")
    try:
        agent_cls, handler_cls = load_official_classes(args.upstream_root)
    except Exception as exc:
        print(f"official import failed: {type(exc).__name__}: {exc}")
        return 2
    print(f"revision: {upstream_revision(args.upstream_root)}")
    print(f"agent: {agent_cls.__module__}.{agent_cls.__name__}")
    print(f"handler: {handler_cls.__module__}.{handler_cls.__name__}")
    print("external_calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
