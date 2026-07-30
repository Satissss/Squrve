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

    def test_write_report_removes_sensitive_literals_from_every_field(self):
        results = make_twenty_results()
        credential = "pilot-credential-value-0123456789"
        results[0].generation.model_metadata["api_key"] = credential
        results[0].question = f"Question containing {credential}"
        results[0].gold_sql = f"SELECT '{credential}'"
        results[0].generation.pred_sql = f"SELECT '{credential}'"
        results[0].generation.error = f"unlabeled free-form failure {credential}"
        results[0].metadata["ordinary_note"] = f"copied {credential}"

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_report(results, temp_dir)
            persisted_content = (
                json_path.read_text(encoding="utf-8")
                + markdown_path.read_text(encoding="utf-8")
            )

        self.assertNotIn(credential, persisted_content)

    def test_write_report_redacts_recognizable_raw_credential_patterns(self):
        results = make_twenty_results()
        google_api_key = "AIza" + "A" * 35
        pem_private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            "not-a-real-key-material\n"
            "-----END PRIVATE KEY-----"
        )
        service_account_material = (
            '{"type":"service_account","project_id":"private-project",'
            '"client_email":"service@example.invalid"}'
        )
        results[0].generation.error = " ".join(
            (google_api_key, pem_private_key, service_account_material)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_report(results, temp_dir)
            persisted_content = (
                json_path.read_text(encoding="utf-8")
                + markdown_path.read_text(encoding="utf-8")
            )

        for raw_credential in (google_api_key, pem_private_key, service_account_material):
            self.assertNotIn(raw_credential, persisted_content)

    def test_markdown_escapes_prose_and_uses_safe_fences(self):
        results = make_twenty_results()
        results[0].question = "# injected heading\n- injected list [link](https://example.invalid)"
        results[0].generation.error = "## injected error\n* [another](https://example.invalid)"
        results[0].gold_sql = "SELECT '```';\n````sql\nnot a report fence"
        results[0].generation.pred_sql = "SELECT '`````';"
        results[0].metadata["note"] = "``````json\nnot a metadata fence"

        markdown = render_markdown(build_report(results, limitations=["- injected limitation"]))

        self.assertIn(r"\# injected heading\n\- injected list \[link\]\(https://example.invalid\)", markdown)
        self.assertIn(r"\#\# injected error\n\* \[another\]\(https://example.invalid\)", markdown)
        self.assertIn(r"\- injected limitation", markdown)
        self.assertIn("`````sql\nSELECT '```';\n````sql\nnot a report fence\n`````", markdown)
        self.assertIn("```````json\n", markdown)
        self.assertIn('"note": "``````json\\nnot a metadata fence"', markdown)
        self.assertEqual(markdown.count("# BMSQL Pilot Outcome Report"), 1)

    def test_latency_summary_ignores_non_finite_values(self):
        results = make_twenty_results()
        results[0].generation.latency_seconds = float("nan")
        results[1].generation.latency_seconds = float("inf")
        results[2].generation.latency_seconds = float("-inf")
        results[3].generation.latency_seconds = True

        report = build_report(results)

        self.assertEqual(report["latency_summary"]["count"], 16)
        self.assertEqual(report["latency_summary"]["min_seconds"], 0.14)

    def test_write_report_redacts_unlabelled_google_oauth_access_tokens(self):
        results = make_twenty_results()
        oauth_token = "ya29." + "a" * 48
        results[0].generation.error = f"remote service returned {oauth_token}"

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_report(results, temp_dir)
            persisted_content = (
                json_path.read_text(encoding="utf-8")
                + markdown_path.read_text(encoding="utf-8")
            )

        self.assertNotIn(oauth_token, persisted_content)

    def test_write_report_redacts_tokens_ending_in_non_word_characters(self):
        results = make_twenty_results()
        oauth_token = "ya29." + "a" * 47 + "-"
        google_api_key = "AIza" + "A" * 34 + "_"
        results[0].generation.error = f"remote service returned {oauth_token} {google_api_key}"

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, markdown_path = write_report(results, temp_dir)
            persisted_content = (
                json_path.read_text(encoding="utf-8")
                + markdown_path.read_text(encoding="utf-8")
            )

        self.assertNotIn(oauth_token, persisted_content)
        self.assertNotIn(google_api_key, persisted_content)
        self.assertNotIn(f"[REDACTED]-", persisted_content)
        self.assertNotIn(f"[REDACTED]_", persisted_content)

    def test_write_report_normalizes_nested_non_finite_floats_for_strict_json(self):
        results = make_twenty_results()
        results[0].generation.model_metadata["nested"] = {
            "not_a_number": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
            "truth": True,
        }
        results[0].evaluation.metadata["nested"] = {"not_a_number": float("nan")}
        results[0].metadata["nested"] = [float("inf"), {"negative": float("-inf")}]

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, _ = write_report(results, temp_dir, limitations=[float("nan")])
            json_content = json_path.read_text(encoding="utf-8")
            persisted = json.loads(
                json_content,
                parse_constant=lambda value: self.fail(f"non-strict JSON constant: {value}"),
            )

        question = persisted["questions"][0]
        self.assertIsNone(question["model_metadata"]["nested"]["not_a_number"])
        self.assertIsNone(question["model_metadata"]["nested"]["positive"])
        self.assertIsNone(question["model_metadata"]["nested"]["negative"])
        self.assertIs(question["model_metadata"]["nested"]["truth"], True)
        self.assertIsNone(question["backend_metadata"]["evaluation"]["nested"]["not_a_number"])
        self.assertEqual(question["sample_metadata"]["nested"], [None, {"negative": None}])
        self.assertEqual(persisted["limitations"], [None])


if __name__ == "__main__":
    unittest.main()
