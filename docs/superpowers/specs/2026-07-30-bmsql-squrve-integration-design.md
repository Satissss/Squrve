# BMSQL–Squrve Offline Integration Design

## Goal

Integrate the original BiomedSQL BMSQL agent into Squrve as one end-to-end
Generator while preserving BMSQL's original algorithm boundary. The first
deliverable must run and test fully offline with a deterministic mock backend,
save one durable result per sample, resume interrupted runs, and report the
five experiment statuses without claiming generated SQL is correct when it was
not executed.

## Confirmed Scope

The first phase covers:

- conversion of BiomedSQL benchmark rows into Squrve rows;
- conversion of database schemas into Squrve's parallel schema format;
- deterministic selection of 20 pilot rows with seed `20260730`;
- a backend interface with mock and original-BMSQL implementations;
- a Squrve `BMSQLGenerator`;
- offline and execution-capable evaluation boundaries;
- per-sample checkpointing and resume;
- JSON and Markdown reports;
- unit tests that do not use paid models or BigQuery.

Downloading the approximately 3 GB BiomedSQL dataset, provisioning BigQuery,
calling paid models, and changing BMSQL's prompts or internal algorithm are
outside this offline phase.

## Repository Findings

Squrve's source root is the current repository. A Generator inherits
`core.actor.generator.BaseGenerate.BaseGenerator` and is registered under that
base class with `@BaseGenerator.register_actor`. Registration alone is not
enough for configuration-based loading in this Squrve revision:
`core/task/meta/GenerateTask.py` imports each concrete Generator and selects it
through explicit name branches.

Squrve rows use `instance_id`, `db_id`, `question`, optional `query`,
`db_type`, `instance_schemas`, and `pred_sql`. Its parallel schema format uses
one row per column with `db_id`, `db_type`, `table_name`, `column_name`,
`column_types`, `column_descriptions`, `sample_rows`, and
`table_to_projDataset`.

No local BiomedSQL/BMSQL source or benchmark data is present. The official
BiomedSQL repository exposes BMSQL through
`handlers.sql.sql_agent.SQLAgent.run_agent(question, num_passes)`. That call
returns general SQL/results, refined SQL/results, a natural-language answer,
and token count. Its `SQLHandler` owns schema context, LLM calls, and BigQuery
execution. The official benchmark fields include `uuid`, `template_uuid`,
`question`, `answer`, `benchmark_query`, `execution_results`, `query_type`,
`sql_category`, and `bio_category`; `benchmark_query` contains
`{project_id}` and `{dataset_name}` placeholders.

References:

- https://github.com/NIH-CARD/biomedsql
- https://huggingface.co/datasets/NIH-CARD/BiomedSQL

## Chosen Architecture

Use a separate root package, `squrve_bmsql/`, for experiment code and keep
Squrve core changes to the Generator adapter and its existing explicit loader.
This is less invasive than adding the whole experiment to `core/`, while
remaining importable from a source checkout without requiring an installation
step.

The integration package contains focused modules:

- `models.py`: requests, generation outputs, execution outputs, statuses, and
  serializable sample results;
- `data_adapter.py`: row mapping, safe placeholder substitution, and stable
  pilot selection;
- `schema_adapter.py`: central/table-oriented schemas to Squrve parallel rows;
- `bmsql_backend.py`: backend protocol, deterministic mock, and a thin wrapper
  around an injected original `SQLAgent` or agent factory;
- `evaluator.py`: offline status assignment and optional read-only execution
  result comparison;
- `runner.py`: sample isolation, atomic per-sample checkpoint writes, resume,
  version/config metadata, and secret redaction;
- `report.py`: aggregate counts, per-sample comparison, stage failures, and
  experiment limitations.

Scripts and configuration live under `squrve_bmsql/scripts/` and
`squrve_bmsql/config/`. Generated artifacts live under
`squrve_bmsql/artifacts/` and are ignored by Git.

## Backend Boundary

`BMSQLBackend.generate(request)` returns a normalized `BMSQLGeneration`.
`MockBMSQLBackend` is deterministic and labels its metadata as mock.

`UpstreamBMSQLBackend` does not copy prompts or BMSQL internals. It receives
either an already initialized upstream agent or an injected factory that
builds one from the request's schema and domain context, then calls only:

```python
agent.run_agent(question=request.question, num_passes=num_passes)
```

The normalized prediction is the refined SQL when present, otherwise the
general SQL. Both SQL variants, both upstream execution-result sets, the
answer, token count, and timing remain in `stage_outputs`/`trajectory`. An
invalid return shape is a generation failure rather than a fabricated SQL
prediction.

Because the official `SQLAgent` binds schema and BigQuery through its
`SQLHandler`, the real-agent factory cannot be finalized or integration-tested
until the upstream source, model settings, and BigQuery configuration are
available. The interface required for that factory is included now.

## Data Flow

1. `prepare_pilot.py` reads a local CSV/JSON benchmark export.
2. Rows are normalized and sorted by stable identifier before seeded sampling,
   so source-file ordering cannot change the selected set.
3. The selected 20 rows and normalized schema are written as a pilot manifest.
4. `run_pilot.py` creates a Squrve `Dataset` and `BMSQLGenerator` using an
   explicitly selected backend.
5. The runner processes unfinished instance IDs only.
6. Each generation is evaluated and atomically written to its own JSON file.
7. A combined results JSON and Markdown report are regenerated from durable
   per-sample files.

The mock mode never executes SQL and therefore produces only
`generated_not_executed` or `generation_failed`.

## Status Model

Every completed sample has exactly one status:

- `generated_not_executed`: non-empty SQL exists but no executor was supplied;
- `generation_failed`: no usable SQL was generated;
- `execution_failed`: predicted or gold SQL execution failed;
- `executed_result_mismatch`: both executed but normalized results differ;
- `executed_result_match`: both executed and normalized results match.

Execution comparison is by normalized rows, not SQL text. Mapping/dictionary
key order is ignored; row order is ignored by default and can be made strict
when a future experiment requires order-sensitive semantics.

## Failure Handling and Resume

The runner catches exceptions per sample and records a generation failure
without stopping the batch. It writes to a temporary sibling file and replaces
the target file only after JSON serialization succeeds. A sample is considered
complete only when its result file parses, has the matching `instance_id`, and
contains one valid status. Resume skips those samples and reruns missing or
corrupt files.

Configuration snapshots and version metadata are allow-listed. Keys whose
names contain secret-like terms such as `api_key`, `token`, `password`,
`credential`, or `secret` are recursively redacted before writing.

## Squrve Adapter

`core/actor/generator/BMSQLGenerate.py` defines `BMSQLGenerator` and decorates
it with `@BaseGenerator.register_actor`. Its `act()` method:

1. reads the Squrve row and resolves an explicitly passed schema or the
   dataset's schema;
2. builds a `BMSQLRequest`;
3. calls the injected backend;
4. stores raw `pred_sql`, `trajectory`, `stage_outputs`, `error`, `latency`,
   and model/backend metadata on the row;
5. returns the SQL for normal Squrve Actor composition.

`core/actor/generator/__init__.py` exports the class, and
`core/task/meta/GenerateTask.py` adds the explicit
`BMSQLGenerator`/`BMSQL` loading branch used by this Squrve revision.

## Testing Strategy

Tests use Python's standard `unittest` runner because `pytest` is not installed
in the current environment. Every production behavior is introduced after a
failing test. Coverage includes:

- benchmark field mapping and placeholder substitution;
- stable selection of exactly 20 rows;
- schema conversion and validation;
- mock and upstream-backend normalization/error handling;
- Squrve registration and row updates;
- all five statuses and result normalization;
- one failed sample not stopping later samples;
- resume after an interrupted/partial run;
- report counts summing to 20 and stage-failure aggregation;
- secret redaction.

No test imports credentials, calls an LLM, accesses BigQuery, or downloads the
benchmark.

## External Resources Still Required

Real BMSQL and execution evaluation require:

- a local checkout of the official BiomedSQL source at a pinned commit;
- a local BiomedSQL benchmark/schema export;
- a configured model provider supported by the upstream project;
- a Google Cloud project, BigQuery dataset, and read-only credentials;
- `PROJECT_ID` and `DATASET_NAME` supplied through environment/local
  configuration, never committed;
- an implementation of the injected upstream agent factory using those local
  resources.

Until those are supplied, the deliverable remains an explicitly labeled
offline integration harness rather than a BMSQL accuracy reproduction.
