# BMSQL offline pilot

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
`results.json`, `run_metadata.json`, per-sample checkpoints, `report.json`,
and `report.md`; generated artifacts are intentionally ignored by Git.

Use `--config /path/to/pilot_20.yaml` to supply another safe pilot
configuration. It may set `sample_size` (which must remain 20), `seed`,
`db_id`, `db_type`, and an optional `project_id`/`dataset_name` placeholder
pair (both values must be supplied together). Do not put API keys, model
tokens, or credential material in it.

## Later real-service execution

This CLI deliberately refuses every backend other than `mock`, so tests and
ordinary local runs cannot contact a paid model or BigQuery. Before wiring a
separate real-service launcher, obtain all of the following outside this
repository:

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
