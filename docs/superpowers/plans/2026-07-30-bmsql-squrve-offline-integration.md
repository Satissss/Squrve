# BMSQL–Squrve Offline Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, resumable offline pilot harness that adapts BiomedSQL data and schemas, runs BMSQL through a Squrve Generator, records all required evidence, and reports the five unambiguous result statuses.

**Architecture:** Keep experiment code in the root-level `squrve_bmsql` package and make only the minimal explicit registration changes required by this Squrve revision. Normalize the official `SQLAgent.run_agent(question, num_passes)` tuple behind an injected backend boundary, use a mock backend offline, and persist one atomic JSON result per sample before aggregating reports.

**Tech Stack:** Python 3.11 standard library, Squrve `Dataset`/`BaseGenerator`, PyYAML for CLI configuration, and `unittest` for offline tests.

## Global Constraints

- Preserve the original BMSQL algorithm, prompts, and internal logic.
- Select exactly 20 pilot samples with seed `20260730`.
- Do not call paid models or BigQuery from unit tests.
- Do not store or print API keys or Google Cloud credentials.
- Database execution must be read-only.
- Generated SQL without execution must be `generated_not_executed`.
- One sample failure must not stop the batch.
- Persist every completed sample and resume by `instance_id`.
- Compare execution results rather than SQL strings.
- Do not modify or discard unrelated existing working-tree changes.

---

### Task 1: Serializable Experiment Models and BiomedSQL Row Adapter

**Files:**
- Create: `squrve_bmsql/__init__.py`
- Create: `squrve_bmsql/models.py`
- Create: `squrve_bmsql/data_adapter.py`
- Create: `squrve_bmsql/tests/__init__.py`
- Create: `squrve_bmsql/tests/test_data_adapter.py`

**Interfaces:**
- Produces: `ResultStatus`, `BMSQLRequest`, `BMSQLGeneration`, `QueryExecution`, `Evaluation`, `SampleResult`
- Produces: `substitute_sql_placeholders(sql, project_id=None, dataset_name=None) -> str`
- Produces: `adapt_biomedsql_row(row, *, db_id="biomedsql", db_type="big_query", project_id=None, dataset_name=None) -> dict`
- Produces: `select_pilot_rows(rows, *, sample_size=20, seed=20260730) -> list[dict]`

- [ ] **Step 1: Write failing adapter and serialization tests**

```python
class DataAdapterTests(unittest.TestCase):
    def test_maps_official_fields_and_replaces_placeholders(self):
        row = {
            "uuid": "Q1.1",
            "template_uuid": "Q1",
            "question": "Which genes?",
            "benchmark_query": "SELECT * FROM `{project_id}.{dataset_name}.genes`",
            "answer": "A",
            "bio_category": "Genetics",
        }
        adapted = adapt_biomedsql_row(
            row, project_id="research-proj", dataset_name="biomed"
        )
        self.assertEqual(adapted["instance_id"], "Q1.1")
        self.assertEqual(adapted["query"], "SELECT * FROM `research-proj.biomed.genes`")
        self.assertEqual(adapted["db_type"], "big_query")
        self.assertEqual(adapted["metadata"]["template_uuid"], "Q1")

    def test_selection_is_stable_when_source_order_changes(self):
        rows = [{"instance_id": f"Q{i:02}", "question": str(i)} for i in range(40)]
        selected = select_pilot_rows(rows)
        reversed_selected = select_pilot_rows(list(reversed(rows)))
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 20)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_data_adapter -v`

Expected: import failure because `squrve_bmsql.data_adapter` and models do not exist.

- [ ] **Step 3: Implement the dataclasses and adapter**

```python
class ResultStatus(str, Enum):
    GENERATED_NOT_EXECUTED = "generated_not_executed"
    GENERATION_FAILED = "generation_failed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTED_RESULT_MISMATCH = "executed_result_mismatch"
    EXECUTED_RESULT_MATCH = "executed_result_match"

def select_pilot_rows(rows, *, sample_size=20, seed=20260730):
    ordered = sorted((dict(row) for row in rows), key=_stable_id)
    if len(ordered) < sample_size:
        raise ValueError(f"Need at least {sample_size} rows, received {len(ordered)}")
    selected = random.Random(seed).sample(ordered, sample_size)
    return sorted(selected, key=_stable_id)
```

Implement explicit `to_dict()`/`from_dict()` methods so enum values and nested
execution records round-trip through JSON without custom encoders. Reject
empty questions, duplicate/missing stable IDs, and unsafe placeholder values
containing whitespace, braces, backticks, or SQL delimiters.

- [ ] **Step 4: Run the adapter tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_data_adapter -v`

Expected: all adapter and model serialization tests pass.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/__init__.py squrve_bmsql/models.py squrve_bmsql/data_adapter.py squrve_bmsql/tests/__init__.py squrve_bmsql/tests/test_data_adapter.py
git commit -m "feat: add BiomedSQL pilot data models"
```

### Task 2: Squrve Parallel Schema Adapter

**Files:**
- Create: `squrve_bmsql/schema_adapter.py`
- Create: `squrve_bmsql/tests/test_schema_adapter.py`

**Interfaces:**
- Consumes: normalized `db_id`, `db_type`, project, and dataset settings
- Produces: `to_squrve_parallel_schema(schema, *, db_id="biomedsql", db_type="big_query", project_id=None, dataset_name=None) -> list[dict]`

- [ ] **Step 1: Write failing central, table-oriented, and parallel-format tests**

```python
def test_flattens_central_schema_and_skips_star(self):
    central = {
        "db_id": "biomedsql",
        "table_names_original": ["genes"],
        "column_names_original": [[-1, "*"], [0, "gene_id"], [0, "symbol"]],
        "column_types": ["STRING", "STRING"],
        "column_descriptions": ["identifier", "gene symbol"],
    }
    rows = to_squrve_parallel_schema(central)
    self.assertEqual([row["column_name"] for row in rows], ["gene_id", "symbol"])
    self.assertTrue(all(row["table_name"] == "genes" for row in rows))
    self.assertTrue(all(row["db_type"] == "big_query" for row in rows))
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_schema_adapter -v`

Expected: import failure because `schema_adapter.py` does not exist.

- [ ] **Step 3: Implement strict format detection and normalization**

```python
def to_squrve_parallel_schema(schema, **defaults):
    if isinstance(schema, dict) and "table_names_original" in schema:
        rows = _from_central(schema, **defaults)
    elif _is_parallel(schema):
        rows = _normalize_parallel(schema, **defaults)
    else:
        rows = _from_tables(schema, **defaults)
    if not rows:
        raise ValueError("Schema produced no Squrve columns")
    return sorted(rows, key=lambda row: (row["table_name"], row["column_name"]))
```

Every output row must contain the eight Squrve fields documented in the design.
Use `project_id.dataset_name` only as `table_to_projDataset`; never place
credentials in schema rows.

- [ ] **Step 4: Run the schema tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_schema_adapter -v`

Expected: all schema conversion tests pass.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/schema_adapter.py squrve_bmsql/tests/test_schema_adapter.py
git commit -m "feat: adapt BiomedSQL schemas for Squrve"
```

### Task 3: Mock and Original BMSQL Backend Boundary

**Files:**
- Create: `squrve_bmsql/bmsql_backend.py`
- Create: `squrve_bmsql/tests/test_bmsql_backend.py`

**Interfaces:**
- Consumes: `BMSQLRequest`
- Produces: protocol `BMSQLBackend.generate(request) -> BMSQLGeneration`
- Produces: `MockBMSQLBackend`
- Produces: `UpstreamBMSQLBackend(agent=None, agent_factory=None, num_passes=1, model_metadata=None)`

- [ ] **Step 1: Write failing mock and official-tuple normalization tests**

```python
class FakeAgent:
    def run_agent(self, question, num_passes):
        return (
            "SELECT general",
            [{"value": 1}],
            "SELECT refined",
            [{"value": 1}],
            "answer",
            42,
        )

def test_upstream_backend_calls_public_agent_entry(self):
    result = UpstreamBMSQLBackend(agent=FakeAgent(), num_passes=2).generate(REQUEST)
    self.assertEqual(result.pred_sql, "SELECT refined")
    self.assertEqual(result.stage_outputs["general_sql_query"], "SELECT general")
    self.assertEqual(result.stage_outputs["input_tokens"], 42)
    self.assertIsNone(result.error)
```

Also test that a malformed tuple, blank SQL, and upstream exception return a
generation with `error_stage="upstream_agent"` rather than fabricating SQL.

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_bmsql_backend -v`

Expected: import failure because `bmsql_backend.py` does not exist.

- [ ] **Step 3: Implement the backend protocol and thin wrapper**

```python
class UpstreamBMSQLBackend:
    def generate(self, request):
        started = time.perf_counter()
        try:
            agent = self.agent_factory(request) if self.agent_factory else self.agent
            raw = agent.run_agent(
                question=request.question,
                num_passes=self.num_passes,
            )
            general_sql, general_rows, refined_sql, refined_rows, answer, tokens = raw
            pred_sql = _clean_sql(refined_sql) or _clean_sql(general_sql)
            return BMSQLGeneration(
                pred_sql=pred_sql or None,
                raw_response=raw,
                stage_outputs={
                    "general_sql_query": general_sql,
                    "general_exec_results": general_rows,
                    "refined_sql_query": refined_sql,
                    "refined_exec_results": refined_rows,
                    "answer": answer,
                    "input_tokens": tokens,
                },
                trajectory=_trajectory_from_outputs(raw),
                error=None if pred_sql else "BMSQL returned no SQL",
                error_stage=None if pred_sql else "sql_generation",
                latency_seconds=time.perf_counter() - started,
                model_metadata=self.model_metadata,
            )
        except Exception as exc:
            return BMSQLGeneration.failure(
                str(exc), error_stage="upstream_agent",
                latency_seconds=time.perf_counter() - started,
                model_metadata=self.model_metadata,
            )
```

The mock uses an explicit SQL mapping or a deterministic `SELECT 1 AS
mock_value`, labels `backend=mock`, and supports configured failure IDs.

- [ ] **Step 4: Run the backend tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_bmsql_backend -v`

Expected: all backend tests pass without importing the upstream repository.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/bmsql_backend.py squrve_bmsql/tests/test_bmsql_backend.py
git commit -m "feat: wrap original BMSQL agent boundary"
```

### Task 4: Offline and Read-Only Execution Evaluator

**Files:**
- Create: `squrve_bmsql/evaluator.py`
- Create: `squrve_bmsql/tests/test_evaluator.py`

**Interfaces:**
- Consumes: `BMSQLGeneration`, gold SQL, and optional executor
- Produces: `Evaluator.evaluate(generation, *, gold_sql, db_id=None) -> Evaluation`
- Produces: `BigQueryReadOnlyExecutor.execute(sql, *, db_id=None) -> QueryExecution`
- Produces: `is_read_only_sql(sql) -> bool`

- [ ] **Step 1: Write one failing test for each required status**

```python
def test_offline_generation_is_not_claimed_correct(self):
    evaluation = Evaluator().evaluate(GENERATED, gold_sql="SELECT gold")
    self.assertEqual(evaluation.status, ResultStatus.GENERATED_NOT_EXECUTED)

def test_matching_results_not_sql_text_sets_match(self):
    executor = SequenceExecutor([
        QueryExecution(success=True, rows=[{"b": 2, "a": 1}]),
        QueryExecution(success=True, rows=[{"a": 1, "b": 2}]),
    ])
    evaluation = Evaluator(executor).evaluate(GENERATED, gold_sql="different SQL")
    self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MATCH)
```

Add tests for generation failure, prediction execution failure, gold execution
failure, mismatch, permission classification, timeout classification, and
rejection of DDL/DML.

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_evaluator -v`

Expected: import failure because `evaluator.py` does not exist.

- [ ] **Step 3: Implement status precedence and canonical result comparison**

```python
def evaluate(self, generation, *, gold_sql, db_id=None):
    if not generation.pred_sql:
        return Evaluation(status=ResultStatus.GENERATION_FAILED, ...)
    if self.executor is None:
        return Evaluation(status=ResultStatus.GENERATED_NOT_EXECUTED)
    predicted = self.executor.execute(generation.pred_sql, db_id=db_id)
    if not predicted.success:
        return Evaluation(status=ResultStatus.EXECUTION_FAILED, ...)
    gold = self.executor.execute(gold_sql, db_id=db_id)
    if not gold.success:
        return Evaluation(status=ResultStatus.EXECUTION_FAILED, ...)
    status = (
        ResultStatus.EXECUTED_RESULT_MATCH
        if canonical_rows(predicted.rows) == canonical_rows(gold.rows)
        else ResultStatus.EXECUTED_RESULT_MISMATCH
    )
    return Evaluation(status=status, predicted=predicted, gold=gold)
```

`BigQueryReadOnlyExecutor` must require `SELECT` or `WITH`, reject semicolon
scripts and mutation keywords, set `maximum_bytes_billed` when configured, and
classify syntax, permission, and timeout failures.

- [ ] **Step 4: Run the evaluator tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_evaluator -v`

Expected: all five statuses and read-only safety tests pass.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/evaluator.py squrve_bmsql/tests/test_evaluator.py
git commit -m "feat: evaluate BMSQL generation and execution"
```

### Task 5: Squrve `BMSQLGenerator` Registration

**Files:**
- Create: `core/actor/generator/BMSQLGenerate.py`
- Modify: `core/actor/generator/__init__.py`
- Modify: `core/task/meta/GenerateTask.py`
- Create: `squrve_bmsql/tests/test_squrve_generator.py`

**Interfaces:**
- Consumes: Squrve `Dataset`, injected `BMSQLBackend`, schema and domain context
- Produces: registered `BMSQLGenerator.act(...) -> str | None`

- [ ] **Step 1: Write failing registration and row-update tests**

```python
def test_generator_is_registered_and_updates_evidence_fields(self):
    dataset = Dataset(
        data_source=[{"instance_id": "Q1", "db_id": "bio", "question": "Q"}],
        schema_source=str(self.schema_path),
        is_schema_final=True,
    )
    generator = BMSQLGenerator(dataset=dataset, backend=MockBMSQLBackend())
    sql = generator.act(0, schema=[{"table_name": "genes"}])
    self.assertEqual(sql, "SELECT 1 AS mock_value")
    self.assertEqual(dataset[0]["pred_sql"], sql)
    self.assertIn("trajectory", dataset[0])
    self.assertIn(BMSQLGenerator, BaseGenerator.get_all_actors())
```

Also instantiate `GenerateTask` with `generate_type="BMSQLGenerator"` and
`actor_args={"backend_mode": "mock"}` to prove configuration-based discovery.

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_squrve_generator -v`

Expected: import failure because `BMSQLGenerate.py` does not exist.

- [ ] **Step 3: Implement the thin Actor and explicit Squrve loader branch**

```python
@BaseGenerator.register_actor
class BMSQLGenerator(BaseGenerator):
    NAME = "BMSQLGenerator"
    SKILL = "Run the original BiomedSQL BMSQL workflow as one Generator."

    def act(self, item, schema=None, schema_links=None, data_logger=None, **kwargs):
        row = self.dataset[item]
        request = BMSQLRequest(
            instance_id=str(row["instance_id"]),
            question=str(row["question"]),
            schema=self._resolve_schema(item, schema),
            domain_context=row.get("external") or row.get("domain_context"),
            metadata=dict(row.get("metadata") or {}),
        )
        generation = self.backend.generate(request)
        self._store_generation(item, generation)
        return generation.pred_sql
```

Catch normal backend exceptions into `BMSQLGeneration.failure`; do not catch
`KeyboardInterrupt`/`SystemExit`. Preserve raw SQL in `pred_sql`; when
`is_save=True`, write a separate `pred_sql_path` rather than replacing SQL with
a path.

- [ ] **Step 4: Run the Generator tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_squrve_generator -v`

Expected: registration, direct Actor use, and `GenerateTask` loading pass.

- [ ] **Step 5: Commit**

```bash
git add core/actor/generator/BMSQLGenerate.py core/actor/generator/__init__.py core/task/meta/GenerateTask.py squrve_bmsql/tests/test_squrve_generator.py
git commit -m "feat: register BMSQL as a Squrve generator"
```

### Task 6: Atomic Pilot Runner, Failure Isolation, Resume, and Redaction

**Files:**
- Create: `squrve_bmsql/runner.py`
- Create: `squrve_bmsql/tests/test_runner.py`

**Interfaces:**
- Consumes: adapted rows, parallel schema, backend, evaluator, output path, run config
- Produces: `PilotRunner.run(resume=True) -> list[SampleResult]`
- Produces: `redact_secrets(value) -> JSON-compatible value`

- [ ] **Step 1: Write failing isolation, interruption, resume, and secret tests**

```python
def test_failure_does_not_stop_later_samples(self):
    results = self.make_runner(
        MockBMSQLBackend(failure_ids={"Q02": "planned failure"})
    ).run()
    self.assertEqual(len(results), 3)
    self.assertEqual(results[1].status, ResultStatus.GENERATION_FAILED)
    self.assertEqual(results[2].status, ResultStatus.GENERATED_NOT_EXECUTED)

def test_resume_skips_valid_sample_files(self):
    first_backend = InterruptingBackend(interrupt_on_call=2)
    with self.assertRaises(KeyboardInterrupt):
        self.make_runner(first_backend).run()
    resumed_backend = CountingBackend()
    results = self.make_runner(resumed_backend).run(resume=True)
    self.assertEqual(len(results), 3)
    self.assertEqual(resumed_backend.calls, ["Q02", "Q03"])
```

Verify nested values under `api_key`, `token`, `password`, `credential`, and
`secret` become `"[REDACTED]"` in `run_metadata.json`.

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_runner -v`

Expected: import failure because `runner.py` does not exist.

- [ ] **Step 3: Implement per-sample atomic persistence and resume validation**

```python
def _atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
```

Build a real Squrve `Dataset` and `BMSQLGenerator`, pass the normalized schema
to `act`, save a valid result before moving to the next row, and regenerate
`results.json` from the per-sample directory. A checkpoint is valid only when
it parses, matches the expected instance ID, and contains a `ResultStatus`.

- [ ] **Step 4: Run the runner tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_runner -v`

Expected: isolation, interruption, resume, atomic-file, and redaction tests pass.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/runner.py squrve_bmsql/tests/test_runner.py
git commit -m "feat: run resumable BMSQL pilot batches"
```

### Task 7: Experiment Report and 20-Sample Accounting

**Files:**
- Create: `squrve_bmsql/report.py`
- Create: `squrve_bmsql/tests/test_report.py`

**Interfaces:**
- Consumes: `Sequence[SampleResult]` and limitations
- Produces: `build_report(results, *, limitations=()) -> dict`
- Produces: `render_markdown(report) -> str`
- Produces: `write_report(results, output_dir, *, limitations=()) -> tuple[Path, Path]`

- [ ] **Step 1: Write failing count and per-question report tests**

```python
def test_report_status_counts_sum_to_twenty(self):
    results = make_twenty_results()
    report = build_report(results, limitations=["offline mock"])
    self.assertEqual(sum(report["status_counts"].values()), 20)
    self.assertEqual(report["total_samples"], 20)
    self.assertIn("most_common_failure_stage", report)
    markdown = render_markdown(report)
    self.assertIn("Q00", markdown)
    self.assertIn("Gold SQL", markdown)
    self.assertIn("Predicted SQL", markdown)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_report -v`

Expected: import failure because `report.py` does not exist.

- [ ] **Step 3: Implement aggregate and Markdown reports**

```python
status_counts = {status.value: 0 for status in ResultStatus}
for result in results:
    status_counts[result.status.value] += 1
assert sum(status_counts.values()) == len(results)
```

Include total, generated count, execution-success count, match count, all five
status counts, failure-stage counts, most common failure stage, latency
summary, each question's gold/predicted SQL and concise error, backend/model
metadata, and limitations.

- [ ] **Step 4: Run the report tests and confirm GREEN**

Run: `python3 -m unittest squrve_bmsql.tests.test_report -v`

Expected: report totals and Markdown evidence tests pass.

- [ ] **Step 5: Commit**

```bash
git add squrve_bmsql/report.py squrve_bmsql/tests/test_report.py
git commit -m "feat: report BMSQL pilot outcomes"
```

### Task 8: Reproducible CLIs, Pilot Configuration, Documentation, and Full Verification

**Files:**
- Create: `squrve_bmsql/config/pilot_20.yaml`
- Create: `squrve_bmsql/scripts/__init__.py`
- Create: `squrve_bmsql/scripts/prepare_pilot.py`
- Create: `squrve_bmsql/scripts/run_pilot.py`
- Create: `squrve_bmsql/tests/test_cli.py`
- Create: `squrve_bmsql/artifacts/.gitkeep`
- Create: `squrve_bmsql/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `python3 -m squrve_bmsql.scripts.prepare_pilot ...`
- Produces: `python3 -m squrve_bmsql.scripts.run_pilot ...`

- [ ] **Step 1: Write failing end-to-end CLI test with local fixtures**

```python
def test_prepare_then_mock_run_creates_twenty_results_and_report(self):
    prepare = subprocess.run(
        [sys.executable, "-m", "squrve_bmsql.scripts.prepare_pilot",
         "--benchmark", str(self.benchmark), "--schema", str(self.schema),
         "--output", str(self.manifest)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    self.assertEqual(prepare.returncode, 0, prepare.stderr)
    run = subprocess.run(
        [sys.executable, "-m", "squrve_bmsql.scripts.run_pilot",
         "--manifest", str(self.manifest), "--output-dir", str(self.output)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    self.assertEqual(run.returncode, 0, run.stderr)
    results = json.loads((self.output / "results.json").read_text())
    self.assertEqual(len(results), 20)
    self.assertTrue(all(row["status"] == "generated_not_executed" for row in results))
    self.assertTrue((self.output / "report.md").is_file())
```

- [ ] **Step 2: Run the CLI test and confirm RED**

Run: `python3 -m unittest squrve_bmsql.tests.test_cli -v`

Expected: module failure because the CLI modules do not exist.

- [ ] **Step 3: Implement config loading and both CLI entry points**

The preparation CLI must accept CSV or JSON, require at least 20 rows, write
the selected IDs, seed, normalized rows, and normalized schema. The run CLI
must default to explicit mock/offline mode, refuse an unknown backend, print
only counts and artifact paths, and never print row metadata or configuration
secrets.

- [ ] **Step 4: Document exact offline and later real-service commands**

Document:

```bash
python3 -m squrve_bmsql.scripts.prepare_pilot \
  --benchmark /path/to/dev_sample.csv \
  --schema /path/to/schema.json \
  --output squrve_bmsql/artifacts/pilot_20/manifest.json

python3 -m squrve_bmsql.scripts.run_pilot \
  --manifest squrve_bmsql/artifacts/pilot_20/manifest.json \
  --output-dir squrve_bmsql/artifacts/pilot_20
```

Explain that mock SQL is wiring validation only; list the upstream checkout,
model provider, `PROJECT_ID`, `DATASET_NAME`, and read-only BigQuery credentials
needed for a real run.

- [ ] **Step 5: Run focused and complete verification**

Run:

```bash
python3 -m unittest discover -s squrve_bmsql/tests -p 'test_*.py' -v
python3 -m compileall -q squrve_bmsql core/actor/generator/BMSQLGenerate.py
git diff --check
```

Expected: all tests pass, compilation exits zero, and `git diff --check`
produces no output.

- [ ] **Step 6: Inspect scope and existing-change preservation**

Run:

```bash
git status --short
git diff --name-only HEAD
git diff -- core/actor/generator/__init__.py core/task/meta/GenerateTask.py
```

Expected: BMSQL files plus the two intentional Squrve loader files and
`.gitignore`; all pre-existing E-SQL changes remain present and unmodified by
this implementation.

- [ ] **Step 7: Commit**

```bash
git add .gitignore squrve_bmsql/config/pilot_20.yaml squrve_bmsql/scripts squrve_bmsql/tests/test_cli.py squrve_bmsql/artifacts/.gitkeep squrve_bmsql/README.md
git commit -m "docs: add reproducible BMSQL pilot workflow"
```
