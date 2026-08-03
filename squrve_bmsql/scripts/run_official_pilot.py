"""Run the official BMSQL agent through Squrve (explicit paid-service guard)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--model", default=os.getenv("BMSQL_MODEL", "deepseek-chat"))
    parser.add_argument("--confirm-external", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_external:
        parser.error("add --confirm-external to authorize model and BigQuery calls")

    # Imports are delayed so ordinary Squrve tests remain offline-safe.
    import json
    from openai import OpenAI
    from google.cloud import bigquery
    from squrve_bmsql.bmsql_backend import UpstreamBMSQLBackend
    from squrve_bmsql.evaluator import BigQueryReadOnlyExecutor, Evaluator
    from squrve_bmsql.runner import PilotRunner
    from squrve_bmsql.upstream_adapter import build_official_backend

    required = ("PROJECT_ID", "DATASET_NAME", "DEEPSEEK_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"missing required environment: {', '.join(missing)}")
    # The upstream package eagerly initializes every provider at import time;
    # only DeepSeek is used here, but harmless placeholders are required for
    # the unused providers' constructors.
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "unused")
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://unused.invalid")
    os.environ.setdefault("AZURE_AI_API_KEY", "unused")
    os.environ.setdefault("AZURE_AI_ENDPOINT", "https://unused.invalid")
    os.environ.setdefault("GEMINI_API_KEY", "unused")
    os.environ.setdefault("ANTHROPIC_API_KEY", "unused")
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    if not os.getenv("SERVICE_ACCOUNT_PATH"):
        raise SystemExit("missing required environment: SERVICE_ACCOUNT_PATH")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schema = manifest["schema"]
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("BMSQL_BASE_URL", "https://api.deepseek.com"),
    )

    from handlers.llms.base_llm import BaseLLM

    # Official SQLHandler only needs this small BaseLLM surface.
    class LLM(BaseLLM):
        def query(self, *, model_name: str, max_tokens: int, temperature: float, query_text: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

    bq_client = bigquery.Client(project=os.environ["PROJECT_ID"])
    from handlers.gcp.big_query import BigQuery
    bq_handler = BigQuery(bigquery_client=bq_client)
    backend: UpstreamBMSQLBackend = build_official_backend(
        upstream_root=args.upstream_root,
        table_info=schema_text,
        table_info_concise=schema_text,
        llm=LLM(),
        bq_handler=bq_handler,
        model=args.model,
        project_id=os.environ["PROJECT_ID"],
        dataset_name=os.environ["DATASET_NAME"],
    )
    executor = BigQueryReadOnlyExecutor(bq_client)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = PilotRunner(
        rows=manifest["rows"], schema=schema, backend=backend,
        evaluator=Evaluator(executor=executor), output_dir=output,
        run_config={"backend": "official_bmsql", "mode": "external", "model": args.model},
    ).run()
    print(f"results: {len(results)}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
