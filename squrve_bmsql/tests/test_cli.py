import csv
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class PilotCLITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.benchmark = self.root / "benchmark.json"
        self.csv_benchmark = self.root / "benchmark.csv"
        self.schema = self.root / "schema.json"
        self.manifest = self.root / "pilot" / "manifest.json"
        self.output = self.root / "run"
        self.rows = [
            {
                "instance_id": f"Q{index:02d}",
                "question": f"Question {index}?",
                "benchmark_query": f"SELECT {index}",
                "bio_category": "fixture",
                "template_uuid": f"template-{index:02d}",
            }
            for index in range(21)
        ]
        self.benchmark.write_text(json.dumps(self.rows), encoding="utf-8")
        with self.csv_benchmark.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.schema.write_text(
            json.dumps(
                {
                    "tables": [
                        {"name": "genes", "columns": {"gene_id": "STRING"}}
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _prepare(self, benchmark, *, output=None, config=None):
        command = [
            sys.executable,
            "-m",
            "squrve_bmsql.scripts.prepare_pilot",
            "--benchmark",
            str(benchmark),
            "--schema",
            str(self.schema),
            "--output",
            str(output or self.manifest),
        ]
        if config is not None:
            command.extend(("--config", str(config)))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def _run(self, *, manifest=None, output=None):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "squrve_bmsql.scripts.run_pilot",
                "--manifest",
                str(manifest or self.manifest),
                "--output-dir",
                str(output or self.output),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_prepare_then_mock_run_creates_twenty_results_and_report(self):
        prepare = self._prepare(self.benchmark)
        self.assertEqual(prepare.returncode, 0, prepare.stderr)
        self.assertNotIn("Question 0?", prepare.stdout)
        self.assertNotIn("template-00", prepare.stdout)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], 20260730)
        self.assertEqual(len(manifest["selected_ids"]), 20)
        self.assertEqual(len(manifest["rows"]), 20)
        self.assertEqual(len(manifest["schema"]), 1)

        run = self._run()
        self.assertEqual(run.returncode, 0, run.stderr)
        results = json.loads((self.output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(results), 20)
        self.assertTrue(
            all(row["evaluation"]["status"] == "generated_not_executed" for row in results)
        )
        self.assertTrue((self.output / "report.md").is_file())
        self.assertIn("results: 20", run.stdout)
        self.assertIn(str(self.output / "results.json"), run.stdout)
        self.assertIn(str(self.output / "report.md"), run.stdout)
        self.assertNotIn("Question 0?", run.stdout)
        self.assertNotIn("template-00", run.stdout)

    def test_prepare_accepts_csv_and_unknown_backend_is_refused(self):
        prepare = self._prepare(self.csv_benchmark)
        self.assertEqual(prepare.returncode, 0, prepare.stderr)

        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "squrve_bmsql.scripts.run_pilot",
                "--manifest",
                str(self.manifest),
                "--output-dir",
                str(self.output),
                "--backend",
                "unapproved-service",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("unsupported backend", run.stderr)
        self.assertFalse((self.output / "results.json").exists())

    def test_prepare_uses_fixed_seed_and_same_selection_for_json_and_csv(self):
        reversed_benchmark = self.root / "reversed.json"
        reversed_benchmark.write_text(json.dumps(list(reversed(self.rows))), encoding="utf-8")
        outputs = [self.root / f"manifest-{name}.json" for name in ("json", "reversed", "csv")]

        for benchmark, output in zip(
            (self.benchmark, reversed_benchmark, self.csv_benchmark), outputs
        ):
            prepare = self._prepare(benchmark, output=output)
            self.assertEqual(prepare.returncode, 0, prepare.stderr)

        manifests = [json.loads(output.read_text(encoding="utf-8")) for output in outputs]
        self.assertTrue(all(manifest["seed"] == 20260730 for manifest in manifests))
        self.assertEqual(manifests[0]["selected_ids"], manifests[1]["selected_ids"])
        self.assertEqual(manifests[0]["selected_ids"], manifests[2]["selected_ids"])
        self.assertEqual(manifests[0]["rows"], manifests[1]["rows"])
        self.assertEqual(manifests[0]["rows"], manifests[2]["rows"])
        self.assertEqual(len(manifests[0]["rows"]), 20)
        self.assertEqual(len(set(manifests[0]["selected_ids"])), 20)
        self.assertEqual(
            manifests[0]["schema"],
            [
                {
                    "db_id": "biomedsql",
                    "db_type": "big_query",
                    "table_name": "genes",
                    "column_name": "gene_id",
                    "column_types": "STRING",
                    "column_descriptions": "",
                    "sample_rows": [],
                    "table_to_projDataset": None,
                }
            ],
        )

    def test_prepare_rejects_alternate_seed_without_disclosing_configuration(self):
        config = self.root / "alternate-seed.yaml"
        secret = "private-config-value"
        config.write_text(
            json.dumps(
                {
                    "sample_size": 20,
                    "seed": 1,
                    "db_id": secret,
                    "db_type": "big_query",
                }
            ),
            encoding="utf-8",
        )

        prepare = self._prepare(self.benchmark, config=config)

        self.assertNotEqual(prepare.returncode, 0)
        self.assertEqual(prepare.stdout, "")
        self.assertNotIn(secret, prepare.stderr)
        self.assertNotIn("Question 0?", prepare.stderr)
        self.assertFalse(self.manifest.exists())

    def test_prepare_rejects_unknown_configuration_fields_without_disclosure(self):
        config = self.root / "credential-config.yaml"
        secret = "private-config-value"
        config.write_text(
            json.dumps(
                {
                    "sample_size": 20,
                    "seed": 20260730,
                    "db_id": "biomedsql",
                    "db_type": "big_query",
                    "api_key": secret,
                }
            ),
            encoding="utf-8",
        )

        prepare = self._prepare(self.benchmark, config=config)

        self.assertNotEqual(prepare.returncode, 0)
        self.assertEqual(prepare.stdout, "")
        self.assertEqual(
            prepare.stderr, "prepare_pilot: unable to prepare pilot manifest\n"
        )
        self.assertNotIn(secret, prepare.stderr)
        self.assertFalse(self.manifest.exists())

    def test_prepare_rejects_short_input_without_printing_fixture_metadata(self):
        short_benchmark = self.root / "short.json"
        short_benchmark.write_text(json.dumps(self.rows[:19]), encoding="utf-8")

        prepare = self._prepare(short_benchmark)

        self.assertNotEqual(prepare.returncode, 0)
        self.assertEqual(prepare.stdout, "")
        self.assertNotIn("Question 0?", prepare.stderr)
        self.assertNotIn("template-00", prepare.stderr)
        self.assertFalse(self.manifest.exists())

    def test_run_rejects_inconsistent_manifest_before_writing_artifacts(self):
        prepare = self._prepare(self.benchmark)
        self.assertEqual(prepare.returncode, 0, prepare.stderr)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))

        for name, mutate in (
            ("version", lambda value: value.__setitem__("version", 2)),
            ("seed", lambda value: value.__setitem__("seed", 1)),
            ("selected_ids", lambda value: value.__setitem__("selected_ids", value["selected_ids"][:-1])),
            ("run_config", lambda value: value.__setitem__("run_config", {"backend": "mock", "mode": "real"})),
            ("top_level_credential", lambda value: value.__setitem__("credential", "forbidden-value")),
            ("top_level_extra", lambda value: value.__setitem__("unexpected", "ordinary-value")),
            ("row_credential", lambda value: value["rows"][0]["metadata"].__setitem__("api_key", "forbidden-value")),
            ("row_extra", lambda value: value["rows"][0].__setitem__("unexpected", "ordinary-value")),
            ("row_empty_question", lambda value: value["rows"][0].__setitem__("question", "  ")),
            ("row_non_json_metadata", lambda value: value["rows"][0]["metadata"].__setitem__("score", float("nan"))),
            ("row_db_mismatch", lambda value: value["rows"][0].__setitem__("db_id", "other")),
            ("schema_credential", lambda value: value["schema"][0].__setitem__("credential", "forbidden-value")),
            ("schema_extra", lambda value: value["schema"][0].__setitem__("unexpected", "ordinary-value")),
            ("schema_empty_table", lambda value: value["schema"][0].__setitem__("table_name", "")),
            ("schema_db_mismatch", lambda value: value["schema"][0].__setitem__("db_type", "other")),
            ("unsafe_instance_id", lambda value: value["rows"][0].__setitem__("instance_id", "../unsafe")),
        ):
            with self.subTest(name=name):
                invalid_manifest = copy.deepcopy(manifest)
                mutate(invalid_manifest)
                if name == "unsafe_instance_id":
                    invalid_manifest["selected_ids"][0] = "../unsafe"
                self.manifest.write_text(json.dumps(invalid_manifest), encoding="utf-8")
                output = self.root / f"run-{name}"
                run = self._run(output=output)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
                self.assertEqual(run.stderr, "run_pilot: unable to run pilot\n")
                self.assertNotIn("forbidden-value", run.stdout + run.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
