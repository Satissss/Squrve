import csv
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

    def _prepare(self, benchmark):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "squrve_bmsql.scripts.prepare_pilot",
                "--benchmark",
                str(benchmark),
                "--schema",
                str(self.schema),
                "--output",
                str(self.manifest),
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

        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "squrve_bmsql.scripts.run_pilot",
                "--manifest",
                str(self.manifest),
                "--output-dir",
                str(self.output),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
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

        for field, value in (
            ("version", 2),
            ("selected_ids", manifest["selected_ids"][:-1]),
            ("run_config", {"backend": "mock", "mode": "real"}),
        ):
            with self.subTest(field=field):
                invalid_manifest = dict(manifest)
                invalid_manifest[field] = value
                self.manifest.write_text(json.dumps(invalid_manifest), encoding="utf-8")
                run = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "squrve_bmsql.scripts.run_pilot",
                        "--manifest",
                        str(self.manifest),
                        "--output-dir",
                        str(self.output),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse((self.output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
