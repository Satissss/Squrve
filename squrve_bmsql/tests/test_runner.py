import json
import tempfile
import unittest
from pathlib import Path

from squrve_bmsql.bmsql_backend import MockBMSQLBackend
from squrve_bmsql.evaluator import Evaluator
from squrve_bmsql.models import ResultStatus
from squrve_bmsql.runner import PilotRunner, redact_secrets


class CountingBackend(MockBMSQLBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, request):
        self.calls.append(request.instance_id)
        return super().generate(request)


class InterruptingBackend(CountingBackend):
    def __init__(self, interrupt_on_call):
        super().__init__()
        self.interrupt_on_call = interrupt_on_call

    def generate(self, request):
        if len(self.calls) + 1 == self.interrupt_on_call:
            raise KeyboardInterrupt()
        return super().generate(request)


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "pilot"
        self.rows = [
            {
                "instance_id": "Q01",
                "question": "Question one?",
                "query": "SELECT 1",
                "db_id": "biomedsql",
                "db_type": "big_query",
                "external": "genetics",
                "metadata": {"source": "fixture"},
            },
            {
                "instance_id": "Q02",
                "question": "Question two?",
                "query": "SELECT 2",
                "db_id": "biomedsql",
                "db_type": "big_query",
                "metadata": {"source": "fixture"},
            },
            {
                "instance_id": "Q03",
                "question": "Question three?",
                "query": "SELECT 3",
                "db_id": "biomedsql",
                "db_type": "big_query",
                "metadata": {"source": "fixture"},
            },
        ]
        self.schema = [
            {
                "db_id": "biomedsql",
                "db_type": "big_query",
                "table_name": "genes",
                "column_name": "gene_id",
                "column_types": "STRING",
                "column_descriptions": "Gene identifier",
                "sample_rows": [],
                "table_to_projDataset": None,
            }
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_runner(self, backend):
        return PilotRunner(
            rows=self.rows,
            schema=self.schema,
            backend=backend,
            evaluator=Evaluator(),
            output_dir=self.output,
            run_config={
                "api_key": "top-level-key",
                "nested": {
                    "token": "nested-token",
                    "items": [
                        {"password": "list-password"},
                        {"credential": "list-credential"},
                        {"secret": "list-secret"},
                    ],
                },
            },
        )

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

    def test_resume_regenerates_invalid_checkpoint(self):
        self.make_runner(CountingBackend()).run()
        checkpoint = self.output / "samples" / "Q02.json"
        checkpoint.write_text("{not json", encoding="utf-8")

        resumed_backend = CountingBackend()
        results = self.make_runner(resumed_backend).run(resume=True)

        self.assertEqual(len(results), 3)
        self.assertEqual(resumed_backend.calls, ["Q02"])
        self.assertEqual(results[1].instance_id, "Q02")

    def test_sample_results_are_atomic_and_results_are_regenerated(self):
        results = self.make_runner(CountingBackend()).run()

        checkpoint = self.output / "samples" / "Q01.json"
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            json.loads(checkpoint.read_text(encoding="utf-8"))["instance_id"], "Q01"
        )
        self.assertEqual(
            json.loads((self.output / "results.json").read_text(encoding="utf-8")),
            [result.to_dict() for result in results],
        )
        self.assertEqual(list((self.output / "samples").glob(".*")), [])

    def test_metadata_redacts_nested_secrets(self):
        self.make_runner(CountingBackend()).run()

        metadata = json.loads((self.output / "run_metadata.json").read_text(encoding="utf-8"))
        config = metadata["run_config"]
        self.assertEqual(config["api_key"], "[REDACTED]")
        self.assertEqual(config["nested"]["token"], "[REDACTED]")
        self.assertEqual(config["nested"]["items"][0]["password"], "[REDACTED]")
        self.assertEqual(config["nested"]["items"][1]["credential"], "[REDACTED]")
        self.assertEqual(config["nested"]["items"][2]["secret"], "[REDACTED]")

    def test_redact_secrets_returns_json_compatible_values(self):
        value = {"token": object(), "ordinary": (1, {"secret": "x"})}

        redacted = redact_secrets(value)

        self.assertEqual(redacted["token"], "[REDACTED]")
        self.assertEqual(redacted["ordinary"], [1, {"secret": "[REDACTED]"}])
        json.dumps(redacted)


if __name__ == "__main__":
    unittest.main()
