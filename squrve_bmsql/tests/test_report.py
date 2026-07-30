import json
import tempfile
import unittest
from pathlib import Path

from squrve_bmsql.models import BMSQLGeneration, Evaluation, ResultStatus, SampleResult
from squrve_bmsql.report import build_report, render_markdown, write_report


def make_twenty_results():
    statuses = list(ResultStatus) * 4
    results = []
    for index, status in enumerate(statuses):
        failure_stage = None
        error = None
        if status is ResultStatus.GENERATION_FAILED:
            failure_stage = "generation"
            error = "model unavailable"
        elif status is ResultStatus.EXECUTION_FAILED:
            failure_stage = "predicted"
            error = "query denied"
        results.append(
            SampleResult(
                instance_id=f"Q{index:02d}",
                question=f"Question {index}?",
                gold_sql=f"SELECT gold_{index}",
                generation=BMSQLGeneration(
                    pred_sql=None
                    if status is ResultStatus.GENERATION_FAILED
                    else f"SELECT predicted_{index}",
                    error=error,
                    error_stage=failure_stage,
                    latency_seconds=0.1 + index / 100,
                    model_metadata={"model": "offline-test-model"},
                ),
                evaluation=Evaluation(
                    status=status,
                    error=error,
                    metadata={
                        "failure_stage": failure_stage,
                        "backend": "offline-mock",
                    }
                    if failure_stage
                    else {"backend": "offline-mock"},
                ),
                metadata={"source": "fixture"},
            )
        )
    return results


class ReportTests(unittest.TestCase):
    def test_report_status_counts_sum_to_twenty(self):
        results = make_twenty_results()

        report = build_report(results, limitations=["offline mock"])

        self.assertEqual(sum(report["status_counts"].values()), 20)
        self.assertEqual(report["total_samples"], 20)
        self.assertEqual(report["generated_count"], 16)
        self.assertEqual(report["execution_success_count"], 8)
        self.assertEqual(report["match_count"], 4)
        self.assertEqual(set(report["status_counts"]), {status.value for status in ResultStatus})
        self.assertEqual(report["failure_stage_counts"], {"generation": 4, "predicted": 4})
        self.assertEqual(report["most_common_failure_stage"], "generation")
        self.assertEqual(report["limitations"], ["offline mock"])
        self.assertEqual(len(report["questions"]), 20)

        markdown = render_markdown(report)

        self.assertIn("Q00", markdown)
        self.assertIn("Gold SQL", markdown)
        self.assertIn("Predicted SQL", markdown)
        self.assertIn("offline-test-model", markdown)

    def test_write_report_persists_redacted_json_and_markdown(self):
        results = make_twenty_results()
        results[0].generation.model_metadata["api_key"] = "sensitive-model-key"
        results[0].evaluation.metadata["access_token"] = "sensitive-backend-token"
        results[0].generation.error = "request failed: token=sensitive-error-token"

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_report(results, temp_dir)
            json_content = json_path.read_text(encoding="utf-8")
            markdown_content = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(json_path, Path(temp_dir) / "report.json")
        self.assertEqual(markdown_path, Path(temp_dir) / "report.md")
        self.assertEqual(json.loads(json_content)["questions"][0]["model_metadata"]["api_key"], "[REDACTED]")
        for secret in ("sensitive-model-key", "sensitive-backend-token", "sensitive-error-token"):
            self.assertNotIn(secret, json_content)
            self.assertNotIn(secret, markdown_content)


if __name__ == "__main__":
    unittest.main()
