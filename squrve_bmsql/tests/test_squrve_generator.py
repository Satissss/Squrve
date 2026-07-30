import tempfile
import unittest
from pathlib import Path

from core.actor.generator.BMSQLGenerate import BMSQLGenerator
from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset
from core.task.meta.GenerateTask import GenerateTask
from squrve_bmsql.bmsql_backend import MockBMSQLBackend


class RecordingBackend:
    def __init__(self):
        self.requests = []
        self.delegate = MockBMSQLBackend()

    def generate(self, request):
        self.requests.append(request)
        return self.delegate.generate(request)


class RaisingBackend:
    def __init__(self, exception):
        self.exception = exception

    def generate(self, request):
        raise self.exception


class BMSQLGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.schema_path = Path(self.temp_dir.name) / "schema.json"
        self.schema_path.write_text("[]", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, **row_updates):
        row = {
            "instance_id": "Q1",
            "db_id": "bio",
            "question": "Which genes?",
            "external": "genetics",
            "metadata": {"split": "pilot"},
        }
        row.update(row_updates)
        return Dataset(
            data_source=[row],
            schema_source=str(self.schema_path),
            is_schema_final=True,
        )

    def test_generator_is_registered_and_updates_evidence_fields(self):
        dataset = self.make_dataset()
        generator = BMSQLGenerator(
            dataset=dataset,
            backend=MockBMSQLBackend(),
            is_save=False,
        )

        sql = generator.act(0, schema=[{"table_name": "genes"}])

        self.assertEqual(sql, "SELECT 1 AS mock_value")
        self.assertEqual(dataset[0]["pred_sql"], sql)
        self.assertEqual(
            dataset[0]["trajectory"],
            [{"stage": "mock_sql_query", "output": sql}],
        )
        self.assertEqual(dataset[0]["stage_outputs"], {"mock_sql_query": sql})
        self.assertIsNone(dataset[0]["error"])
        self.assertEqual(dataset[0]["model_metadata"], {"backend": "mock"})
        self.assertIn(BMSQLGenerator, BaseGenerator.get_all_actors())

    def test_generator_forwards_schema_domain_context_and_metadata(self):
        dataset = self.make_dataset()
        backend = RecordingBackend()
        generator = BMSQLGenerator(dataset=dataset, backend=backend, is_save=False)
        schema = [{"table_name": "genes"}]

        generator.act(0, schema=schema)

        request = backend.requests[0]
        self.assertEqual(request.instance_id, "Q1")
        self.assertEqual(request.question, "Which genes?")
        self.assertEqual(request.schema, schema)
        self.assertEqual(request.domain_context, "genetics")
        self.assertEqual(request.metadata, {"split": "pilot"})

    def test_generate_task_discovers_bmsql_generator_from_configuration(self):
        dataset = self.make_dataset()

        task = GenerateTask(
            llm=None,
            dataset=dataset,
            generate_type="BMSQLGenerator",
            actor_args={"backend_mode": "mock"},
            is_save=False,
        )

        self.assertIsInstance(task.actor, BMSQLGenerator)
        self.assertEqual(
            task.actor.act(0, schema=[{"table_name": "genes"}]),
            "SELECT 1 AS mock_value",
        )

    def test_normal_backend_exception_is_stored_as_failure_evidence(self):
        dataset = self.make_dataset()
        generator = BMSQLGenerator(
            dataset=dataset,
            backend=RaisingBackend(RuntimeError("backend unavailable")),
            is_save=False,
        )

        sql = generator.act(0, schema=[{"table_name": "genes"}])

        self.assertIsNone(sql)
        self.assertIsNone(dataset[0]["pred_sql"])
        self.assertEqual(dataset[0]["error"], "backend unavailable")
        self.assertEqual(dataset[0]["error_stage"], "backend")
        self.assertEqual(dataset[0]["trajectory"], [])

    def test_invalid_request_is_stored_as_failure_evidence(self):
        dataset = self.make_dataset(question="")
        generator = BMSQLGenerator(
            dataset=dataset,
            backend=MockBMSQLBackend(),
            is_save=False,
        )

        sql = generator.act(0, schema=[{"table_name": "genes"}])

        self.assertIsNone(sql)
        self.assertIsNone(dataset[0]["pred_sql"])
        self.assertEqual(dataset[0]["error"], "question must be non-empty")
        self.assertEqual(dataset[0]["error_stage"], "request")
        self.assertEqual(dataset[0]["trajectory"], [])

    def test_schema_resolution_failure_is_stored_as_failure_evidence(self):
        dataset = self.make_dataset()
        generator = BMSQLGenerator(
            dataset=dataset,
            backend=MockBMSQLBackend(),
            is_save=False,
        )

        sql = generator.act(0)

        self.assertIsNone(sql)
        self.assertIsNone(dataset[0]["pred_sql"])
        self.assertEqual(
            dataset[0]["error"],
            "Failed to load a valid database schema for the sample",
        )
        self.assertEqual(dataset[0]["error_stage"], "schema")
        self.assertEqual(dataset[0]["trajectory"], [])

    def test_process_control_exceptions_are_not_caught(self):
        for exception in (KeyboardInterrupt(), SystemExit()):
            with self.subTest(exception=type(exception).__name__):
                generator = BMSQLGenerator(
                    dataset=self.make_dataset(),
                    backend=RaisingBackend(exception),
                    is_save=False,
                )

                with self.assertRaises(type(exception)):
                    generator.act(0, schema=[{"table_name": "genes"}])

    def test_save_keeps_raw_sql_and_records_separate_path(self):
        dataset = self.make_dataset()
        raw_sql = "SELECT gene_id FROM genes"
        generator = BMSQLGenerator(
            dataset=dataset,
            backend=MockBMSQLBackend(sql_by_id={"Q1": raw_sql}),
            is_save=True,
            save_dir=self.temp_dir.name,
        )

        sql = generator.act(0, schema=[{"table_name": "genes"}])

        saved_path = Path(dataset[0]["pred_sql_path"])
        self.assertEqual(sql, raw_sql)
        self.assertEqual(dataset[0]["pred_sql"], raw_sql)
        self.assertEqual(saved_path.read_text(encoding="utf-8"), raw_sql)
        self.assertEqual(saved_path.name, "BMSQLGenerator_Q1.sql")


if __name__ == "__main__":
    unittest.main()
