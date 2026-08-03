"""Loader and factory for the official NIH-CARD BiomedSQL implementation.

The upstream repository is deliberately kept outside this project because its
PolyForm Noncommercial license and large dependency set are not Squrve runtime
dependencies.  This module loads that checkout explicitly and constructs the
official ``SQLAgent``/``SQLHandler`` pair without changing their prompts or
control flow.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .bmsql_backend import UpstreamBMSQLBackend


def upstream_revision(upstream_root: str | Path) -> str:
    """Return a short immutable source marker for run metadata."""
    root = Path(upstream_root)
    head = root / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            ref_path = root / ".git" / ref[5:]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:40]
        return ref[:40]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()[:16]}"


def load_official_classes(upstream_root: str | Path) -> tuple[type[Any], type[Any]]:
    """Import official ``SQLAgent`` and ``SQLHandler`` from a checkout."""
    root = Path(upstream_root).expanduser().resolve()
    if not (root / "handlers" / "sql" / "sql_agent.py").is_file():
        raise FileNotFoundError(f"not an official BiomedSQL checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    agent_module = importlib.import_module("handlers.sql.sql_agent")
    handler_module = importlib.import_module("handlers.sql.sql_handlers")
    return agent_module.SQLAgent, handler_module.SQLHandler


def build_official_backend(
    *,
    upstream_root: str | Path,
    table_info: str,
    table_info_concise: str,
    llm: Any,
    bq_handler: Any,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 3,
    num_passes: int = 1,
    project_id: str | None = None,
    dataset_name: str | None = None,
    model_metadata: Mapping[str, Any] | None = None,
) -> UpstreamBMSQLBackend:
    """Construct a Squrve backend around the untouched official BMSQL agent."""
    if project_id:
        os.environ["PROJECT_ID"] = project_id
    if dataset_name:
        os.environ["DATASET_NAME"] = dataset_name
    SQLAgent, SQLHandler = load_official_classes(upstream_root)
    handler = SQLHandler(
        table_info=table_info,
        table_info_concise=table_info_concise,
        llm=llm,
        llm_query_params={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        bq_handler=bq_handler,
    )
    agent = SQLAgent(sql_handler=handler, max_retries=max_retries)
    metadata = {
        "backend": "official_bmsql",
        "upstream_revision": upstream_revision(upstream_root),
        "upstream_entrypoint": "handlers.sql.sql_agent.SQLAgent.run_agent",
        "model": model,
        **dict(model_metadata or {}),
    }
    return UpstreamBMSQLBackend(
        agent=agent,
        num_passes=num_passes,
        model_metadata=metadata,
    )
