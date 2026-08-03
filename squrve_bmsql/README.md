# BMSQL pilot and official upstream adapter

This package adapts BiomedSQL inputs to Squrve and provides a deterministic,
offline-safe 20-sample pilot. The default backend is an explicit mock backend:
its `SELECT 1 AS mock_value` output validates the data, generator, checkpoint,
and reporting wiring only. It does not assess a model or execute BigQuery SQL.

## Reproducible offline pilot

Use a local JSON or CSV benchmark development sample containing at least 20
BiomedSQL rows, plus a local JSON schema. The checked-in configuration fixes
the seed at `20260730` and the pilot size at 20.

```bash
python3 -m squrve_bmsql.scripts.prepare_pilot \
  --benchmark /path/to/dev_sample.csv \
  --schema /path/to/schema.json \
  --output squrve_bmsql/artifacts/pilot_20/manifest.json

python3 -m squrve_bmsql.scripts.run_pilot \
  --manifest squrve_bmsql/artifacts/pilot_20/manifest.json \
  --output-dir squrve_bmsql/artifacts/pilot_20
```

The preparation command writes a manifest with the selected IDs, seed,
normalized rows, and normalized schema. The run command defaults to
`--backend mock` and prints only result counts and artifact paths. It writes
`schema.json`, `results.json`, `run_metadata.json`, per-sample checkpoints,
`report.json`, and `report.md`; generated artifacts are intentionally ignored
by Git.

Use `--config /path/to/pilot_20.yaml` to supply another safe pilot
configuration. It may set `sample_size` (which must remain 20), `db_id`,
`db_type`, and an optional `project_id`/`dataset_name` placeholder pair (both
values must be supplied together). The seed is fixed at `20260730` and cannot
be changed. Do not put API keys, model tokens, or credential material in the
configuration.

## Official implementation hookup

The official NIH-CARD checkout is the source of the BMSQL algorithm.  Squrve
does not copy or rewrite its prompts: `squrve_bmsql.upstream_adapter` imports
`handlers.sql.sql_agent.SQLAgent` and constructs the official
`SQLHandler`/`SQLAgent` pair, then passes it through `UpstreamBMSQLBackend`.
The checkout is intentionally external because the upstream project is under a
PolyForm Noncommercial license and has a separate dependency environment.

After cloning the official repository and installing its dependencies, verify
the source entrypoint without making any paid calls:

```bash
python3 -m squrve_bmsql.scripts.check_official \
  --upstream-root /path/to/biomedsql
```

The check reports the upstream revision and the exact imported classes.  A
real run must also provide the official benchmark/database, a model provider,
and read-only BigQuery credentials; it is not silently enabled by the offline
pilot.

With an approved DeepSeek-compatible endpoint and a read-only BigQuery account,
the explicit external launcher is:

```bash
DEEPSEEK_API_KEY=... PROJECT_ID=... DATASET_NAME=... \
SERVICE_ACCOUNT_PATH=/secure/read-only.json \
python3 -m squrve_bmsql.scripts.run_official_pilot \
  --manifest /path/to/manifest.json \
  --output-dir squrve_bmsql/artifacts/official_run \
  --upstream-root /path/to/biomedsql \
  --confirm-external
```

This command is intentionally guarded because it incurs model and BigQuery
charges.  DeepSeek is an execution smoke test of the official BMSQL pipeline,
not the paper's GPT-o3-mini result; matching the reported 62.6% requires the
paper's Azure model, database snapshot, and evaluation settings.

## Later real-service execution

The pilot CLI deliberately refuses every backend other than `mock`, so tests
and ordinary local runs cannot contact a paid model or BigQuery. Before a real
launcher is enabled, obtain all of the following outside this repository:

- an approved checkout of the upstream BiomedSQL/BMSQL implementation;
- a configured model provider account and its credential mechanism;
- `PROJECT_ID` and `DATASET_NAME` for the target BigQuery data;
- read-only BigQuery credentials with the minimum dataset and job permissions.

Keep credentials outside the manifest and repository. A real launcher should
inject `UpstreamBMSQLBackend` and `BigQueryReadOnlyExecutor`, and must set the
real identifiers through its secure environment or secret manager, for example:

```bash
export PROJECT_ID=your-project-id
export DATASET_NAME=your_dataset
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/read-only-service-account.json
# Run the separately reviewed real-service launcher from the upstream checkout.
```

The evaluator permits only read-only query statements; review the service
account and model-provider controls before any real invocation.
