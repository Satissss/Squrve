import json
import math
import unittest
from datetime import date, datetime, time
from decimal import Decimal

from squrve_bmsql.evaluator import (
    BigQueryReadOnlyExecutor,
    Evaluator,
    is_read_only_sql,
)
from squrve_bmsql.models import BMSQLGeneration, QueryExecution, ResultStatus


GENERATED = BMSQLGeneration(pred_sql="SELECT predicted")


class SequenceExecutor:
    def __init__(self, executions):
        self.executions = list(executions)
        self.calls = []

    def execute(self, sql, *, db_id=None):
        self.calls.append((sql, db_id))
        return self.executions.pop(0)


class EvaluatorTests(unittest.TestCase):
    def test_blank_or_non_string_prediction_is_generation_failure_offline(self):
        for pred_sql in ("", " \n\t ", 7):
            with self.subTest(pred_sql=pred_sql):
                evaluation = Evaluator().evaluate(
                    BMSQLGeneration(pred_sql=pred_sql),
                    gold_sql="SELECT gold",
                )

                self.assertEqual(
                    evaluation.status,
                    ResultStatus.GENERATION_FAILED,
                )

    def test_blank_or_non_string_gold_is_execution_failure_without_execution(self):
        for gold_sql in ("", " \n\t ", None, 7):
            with self.subTest(gold_sql=gold_sql):
                executor = SequenceExecutor([])

                evaluation = Evaluator(executor).evaluate(
                    GENERATED,
                    gold_sql=gold_sql,
                )

                self.assertEqual(
                    evaluation.status,
                    ResultStatus.EXECUTION_FAILED,
                )
                self.assertEqual(evaluation.metadata["failure_stage"], "gold")
                self.assertIn("gold SQL", evaluation.error)
                self.assertEqual(executor.calls, [])

    def test_generation_failure_takes_precedence_over_offline_status(self):
        evaluation = Evaluator().evaluate(
            BMSQLGeneration.failure("model unavailable", error_stage="model"),
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.GENERATION_FAILED)
        self.assertEqual(evaluation.error, "model unavailable")
        self.assertEqual(evaluation.metadata["failure_stage"], "model")

    def test_generation_failure_does_not_call_executor(self):
        executor = SequenceExecutor([])

        evaluation = Evaluator(executor).evaluate(
            BMSQLGeneration.failure("no SQL"),
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.GENERATION_FAILED)
        self.assertEqual(executor.calls, [])

    def test_offline_generation_is_not_claimed_correct(self):
        evaluation = Evaluator().evaluate(GENERATED, gold_sql="SELECT gold")

        self.assertEqual(evaluation.status, ResultStatus.GENERATED_NOT_EXECUTED)
        self.assertIsNone(evaluation.predicted)
        self.assertIsNone(evaluation.gold)

    def test_offline_valid_prediction_precedes_blank_gold_validation(self):
        evaluation = Evaluator().evaluate(GENERATED, gold_sql=" \n\t ")

        self.assertEqual(evaluation.status, ResultStatus.GENERATED_NOT_EXECUTED)
        self.assertIsNone(evaluation.error)

    def test_prediction_execution_failure_stops_before_gold(self):
        failure = QueryExecution(
            success=False,
            error="bad prediction",
            error_type="syntax",
        )
        executor = SequenceExecutor([failure])

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="SELECT gold",
            db_id="project.dataset",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTION_FAILED)
        self.assertIs(evaluation.predicted, failure)
        self.assertIsNone(evaluation.gold)
        self.assertEqual(evaluation.error, "bad prediction")
        self.assertEqual(evaluation.metadata["failure_stage"], "predicted")
        self.assertEqual(
            executor.calls,
            [("SELECT predicted", "project.dataset")],
        )

    def test_gold_execution_failure_preserves_both_executions(self):
        predicted = QueryExecution(success=True, rows=[{"answer": 1}])
        gold = QueryExecution(
            success=False,
            error="gold denied",
            error_type="permission",
        )
        executor = SequenceExecutor([predicted, gold])

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTION_FAILED)
        self.assertIs(evaluation.predicted, predicted)
        self.assertIs(evaluation.gold, gold)
        self.assertEqual(evaluation.error, "gold denied")
        self.assertEqual(evaluation.metadata["failure_stage"], "gold")

    def test_matching_results_not_sql_text_sets_match(self):
        executor = SequenceExecutor(
            [
                QueryExecution(
                    success=True,
                    rows=[
                        {"b": 2, "a": 1},
                        {"b": 4, "a": 3},
                    ],
                ),
                QueryExecution(
                    success=True,
                    rows=[
                        {"a": 3, "b": 4},
                        {"a": 1, "b": 2},
                    ],
                ),
            ]
        )

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="different SQL",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MATCH)

    def test_different_result_multiplicity_sets_mismatch(self):
        executor = SequenceExecutor(
            [
                QueryExecution(success=True, rows=[{"a": 1}, {"a": 1}]),
                QueryExecution(success=True, rows=[{"a": 1}]),
            ]
        )

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MISMATCH)

    def test_equivalent_native_numeric_values_match(self):
        executor = SequenceExecutor(
            [
                QueryExecution(
                    success=True,
                    rows=[
                        {
                            "zero": -0.0,
                            "integer": 1,
                            "decimal": Decimal("1.00"),
                            "nan": float("nan"),
                            "positive_infinity": float("inf"),
                            "negative_infinity": Decimal("-Infinity"),
                        }
                    ],
                ),
                QueryExecution(
                    success=True,
                    rows=[
                        {
                            "zero": Decimal("-0"),
                            "integer": 1.0,
                            "decimal": 1,
                            "nan": Decimal("NaN"),
                            "positive_infinity": Decimal("Infinity"),
                            "negative_infinity": float("-inf"),
                        }
                    ],
                ),
            ]
        )

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MATCH)

    def test_boolean_does_not_compare_equal_to_number(self):
        executor = SequenceExecutor(
            [
                QueryExecution(success=True, rows=[{"value": True}]),
                QueryExecution(success=True, rows=[{"value": 1}]),
            ]
        )

        evaluation = Evaluator(executor).evaluate(
            GENERATED,
            gold_sql="SELECT gold",
        )

        self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MISMATCH)


class ReadOnlySQLTests(unittest.TestCase):
    def test_accepts_select_and_with_queries(self):
        accepted = (
            "SELECT gene_id FROM genes",
            "  -- explanation\nSELECT 1",
            "WITH genes AS (SELECT 1 AS id) SELECT id FROM genes",
            "SELECT 'DELETE is data' AS text",
            "SELECT 1;",
        )

        for sql in accepted:
            with self.subTest(sql=sql):
                self.assertTrue(is_read_only_sql(sql))

    def test_rejects_blank_non_query_and_multi_statement_sql(self):
        rejected = (
            "",
            "   ",
            "EXPLAIN SELECT 1",
            "SELECT 1; SELECT 2",
            "SELECT 1; DROP TABLE genes",
        )

        for sql in rejected:
            with self.subTest(sql=sql):
                self.assertFalse(is_read_only_sql(sql))

    def test_rejects_ddl_and_dml(self):
        rejected = (
            "CREATE TABLE genes (id INT64)",
            "DROP TABLE genes",
            "ALTER TABLE genes ADD COLUMN symbol STRING",
            "TRUNCATE TABLE genes",
            "INSERT INTO genes VALUES (1)",
            "UPDATE genes SET id = 2",
            "DELETE FROM genes WHERE id = 1",
            "MERGE genes USING incoming ON genes.id = incoming.id WHEN MATCHED THEN DELETE",
            "CALL mutate_genes()",
            "EXPORT DATA OPTIONS(uri='gs://bucket/out') AS SELECT 1",
        )

        for sql in rejected:
            with self.subTest(sql=sql):
                self.assertFalse(is_read_only_sql(sql))


class FakeQueryConfig:
    def __init__(self):
        self.use_legacy_sql = None
        self.maximum_bytes_billed = None
        self.default_dataset = None


class FakeJob:
    def __init__(self, rows=None, error=None):
        self.rows = rows
        self.error = error
        self.result_calls = []
        self.job_id = "fake-job"
        self.total_bytes_processed = 123

    def result(self, *, timeout=None):
        self.result_calls.append(timeout)
        if self.error is not None:
            raise self.error
        return self.rows


class FakeClient:
    def __init__(self, job=None, error=None):
        self.job = job
        self.error = error
        self.calls = []

    def query(self, sql, *, job_config, timeout=None):
        self.calls.append((sql, job_config, timeout))
        if self.error is not None:
            raise self.error
        return self.job


class SequenceClient:
    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.calls = []

    def query(self, sql, *, job_config, timeout=None):
        self.calls.append((sql, job_config, timeout))
        return self.jobs.pop(0)


class FakeBigQueryRow:
    def __init__(self, values):
        self.values = values

    def items(self):
        return self.values.items()


class BigQueryReadOnlyExecutorTests(unittest.TestCase):
    def make_executor(self, client, **kwargs):
        return BigQueryReadOnlyExecutor(
            client,
            query_job_config_factory=FakeQueryConfig,
            **kwargs,
        )

    def test_executes_with_read_only_config_byte_cap_and_timeout(self):
        job = FakeJob([{"b": 2, "a": 1}])
        client = FakeClient(job=job)
        executor = self.make_executor(
            client,
            maximum_bytes_billed=4096,
            timeout_seconds=7,
        )

        execution = executor.execute(
            "SELECT a, b FROM genes",
            db_id="project.dataset",
        )

        self.assertTrue(execution.success)
        self.assertEqual(execution.rows, [{"b": 2, "a": 1}])
        self.assertEqual(len(client.calls), 1)
        _, config, query_timeout = client.calls[0]
        self.assertFalse(config.use_legacy_sql)
        self.assertEqual(config.maximum_bytes_billed, 4096)
        self.assertEqual(config.default_dataset, "project.dataset")
        self.assertEqual(query_timeout, 7)
        self.assertEqual(job.result_calls, [7])
        self.assertEqual(execution.metadata["db_id"], "project.dataset")
        self.assertEqual(execution.metadata["job_id"], "fake-job")
        self.assertEqual(execution.metadata["total_bytes_processed"], 123)

    def test_unsafe_sql_is_rejected_before_client_call(self):
        client = FakeClient()
        execution = self.make_executor(client).execute("DELETE FROM genes")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "unsafe_sql")
        self.assertEqual(client.calls, [])

    def test_classifies_syntax_failure(self):
        class BadRequest(Exception):
            pass

        execution = self.make_executor(
            FakeClient(error=BadRequest("Syntax error at line 1"))
        ).execute("SELECT invalid")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "syntax")
        self.assertIn("Syntax error", execution.error)

    def test_classifies_structured_invalid_query_from_result(self):
        class BadRequest(Exception):
            def __init__(self):
                super().__init__("request rejected")
                self.errors = [{"reason": "invalidQuery", "code": 400}]

        execution = self.make_executor(
            FakeClient(job=FakeJob(error=BadRequest()))
        ).execute("SELECT invalid")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "syntax")

    def test_classifies_permission_failure(self):
        class Forbidden(Exception):
            def __init__(self):
                super().__init__("request rejected")
                self.errors = [{"reason": "accessDenied", "code": 403}]

        execution = self.make_executor(
            FakeClient(error=Forbidden())
        ).execute("SELECT 1")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "permission")

    def test_classifies_timeout_failure(self):
        class DeadlineExceeded(Exception):
            def __init__(self):
                super().__init__("request rejected")
                self.errors = [{"reason": "deadlineExceeded", "code": 504}]

        execution = self.make_executor(
            FakeClient(error=DeadlineExceeded())
        ).execute("SELECT 1")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "timeout")

    def test_submission_timeout_is_classified_and_passed_to_query(self):
        client = FakeClient(error=TimeoutError("submission timed out"))

        execution = self.make_executor(
            client,
            timeout_seconds=3,
        ).execute("SELECT 1")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "timeout")
        self.assertEqual(client.calls[0][2], 3)

    def test_non_syntax_bad_request_reasons_are_execution_errors(self):
        class BadRequest(Exception):
            def __init__(self, message, errors=None):
                super().__init__(message)
                if errors is not None:
                    self.errors = errors

        class InvalidArgument(Exception):
            pass

        errors = (
            BadRequest(
                "Resources exceeded during query execution",
                [{"reason": "resourcesExceeded", "code": 400}],
            ),
            BadRequest("Maximum bytes billed limit exceeded"),
            BadRequest("Invalid query configuration"),
            InvalidArgument("Invalid job configuration"),
        )

        for error in errors:
            with self.subTest(error=error):
                execution = self.make_executor(
                    FakeClient(error=error)
                ).execute("SELECT 1")

                self.assertFalse(execution.success)
                self.assertEqual(execution.error_type, "execution_error")

    def test_bigquery_native_values_are_normalized_and_serialize_as_strict_json(self):
        row = FakeBigQueryRow(
            {
                "decimal": Decimal("1.20"),
                "date": date(2026, 7, 30),
                "datetime": datetime(2026, 7, 30, 12, 34, 56),
                "time": time(12, 34, 56),
                "bytes": b"\x00\xff",
                "nested": (
                    Decimal("2"),
                    {"when": date(2026, 7, 31)},
                ),
                "nan": float("nan"),
            }
        )

        execution = self.make_executor(
            FakeClient(job=FakeJob([row]))
        ).execute("SELECT native_values")

        self.assertTrue(execution.success)
        normalized = execution.rows[0]
        self.assertEqual(normalized["decimal"], Decimal("1.20"))
        self.assertEqual(normalized["date"], date(2026, 7, 30))
        self.assertEqual(normalized["datetime"], datetime(2026, 7, 30, 12, 34, 56))
        self.assertEqual(normalized["time"], time(12, 34, 56))
        self.assertEqual(normalized["bytes"], b"\x00\xff")
        self.assertEqual(
            normalized["nested"],
            [Decimal("2"), {"when": date(2026, 7, 31)}],
        )
        self.assertTrue(math.isnan(normalized["nan"]))
        encoded = execution.to_dict()
        self.assertEqual(
            encoded["rows"][0]["decimal"],
            {"__squrve_bmsql_json_v2__": ["decimal", "1.20"]},
        )
        json.dumps(encoded, allow_nan=False)

    def test_tagged_decimal_from_bigquery_matches_integer_execution_result(self):
        client = SequenceClient(
            [
                FakeJob(
                    [
                        FakeBigQueryRow(
                            {
                                "value": Decimal("1.0"),
                                "nested": [Decimal("2.00")],
                            }
                        )
                    ]
                ),
                FakeJob(
                    [
                        FakeBigQueryRow(
                            {
                                "value": 1,
                                "nested": [2],
                            }
                        )
                    ]
                ),
            ]
        )
        evaluator = Evaluator(self.make_executor(client))

        evaluation = evaluator.evaluate(GENERATED, gold_sql="SELECT gold")

        self.assertEqual(evaluation.status, ResultStatus.EXECUTED_RESULT_MATCH)
        json.dumps(evaluation.to_dict(), allow_nan=False)

    def test_user_mapping_matching_legacy_native_tag_is_not_decoded(self):
        evaluator = Evaluator(
            SequenceExecutor(
                [
                    QueryExecution(
                        success=True,
                        rows=[
                            {
                                "value": {
                                    "__squrve_bmsql_native_v1__": ["decimal", "1"]
                                }
                            }
                        ],
                    ),
                    QueryExecution(success=True, rows=[{"value": Decimal("1")}]),
                ]
            )
        )

        evaluation = evaluator.evaluate(GENERATED, gold_sql="SELECT gold")

        self.assertEqual(
            evaluation.status,
            ResultStatus.EXECUTED_RESULT_MISMATCH,
        )

    def test_tagged_bytes_and_date_do_not_alias_same_text_strings(self):
        client = SequenceClient(
            [
                FakeJob(
                    [
                        FakeBigQueryRow(
                            {
                                "nested": {
                                    "bytes": b"\x00\xff",
                                    "date": date(2026, 7, 30),
                                }
                            }
                        )
                    ]
                ),
                FakeJob(
                    [
                        FakeBigQueryRow(
                            {
                                "nested": {
                                    "bytes": "AP8=",
                                    "date": "2026-07-30",
                                }
                            }
                        )
                    ]
                ),
            ]
        )
        evaluator = Evaluator(self.make_executor(client))

        evaluation = evaluator.evaluate(GENERATED, gold_sql="SELECT gold")

        self.assertEqual(
            evaluation.status,
            ResultStatus.EXECUTED_RESULT_MISMATCH,
        )
        json.dumps(evaluation.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
