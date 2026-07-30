import unittest

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
    def __init__(self, rows):
        self.rows = rows
        self.result_calls = []
        self.job_id = "fake-job"
        self.total_bytes_processed = 123

    def result(self, *, timeout=None):
        self.result_calls.append(timeout)
        return self.rows


class FakeClient:
    def __init__(self, job=None, error=None):
        self.job = job
        self.error = error
        self.calls = []

    def query(self, sql, *, job_config):
        self.calls.append((sql, job_config))
        if self.error is not None:
            raise self.error
        return self.job


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
        _, config = client.calls[0]
        self.assertFalse(config.use_legacy_sql)
        self.assertEqual(config.maximum_bytes_billed, 4096)
        self.assertEqual(config.default_dataset, "project.dataset")
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

    def test_classifies_permission_failure(self):
        class Forbidden(Exception):
            pass

        execution = self.make_executor(
            FakeClient(error=Forbidden("Access Denied"))
        ).execute("SELECT 1")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "permission")

    def test_classifies_timeout_failure(self):
        class DeadlineExceeded(Exception):
            pass

        execution = self.make_executor(
            FakeClient(error=DeadlineExceeded("deadline exceeded"))
        ).execute("SELECT 1")

        self.assertFalse(execution.success)
        self.assertEqual(execution.error_type, "timeout")


if __name__ == "__main__":
    unittest.main()
