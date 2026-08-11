import unittest

from squrve_bmsql.bmsql_backend import MockBMSQLBackend, UpstreamBMSQLBackend
from squrve_bmsql.models import BMSQLRequest


REQUEST = BMSQLRequest(instance_id="Q1.1", question="Which genes?")


class FakeAgent:
    def __init__(self):
        self.calls = []

    def run_agent(self, question, num_passes):
        self.calls.append((question, num_passes))
        return (
            "SELECT general",
            [{"value": 1}],
            "SELECT refined",
            [{"value": 1}],
            "answer",
            42,
        )


class RaisingAgent:
    def run_agent(self, question, num_passes):
        raise RuntimeError("upstream unavailable")


class BMSQLBackendTests(unittest.TestCase):
    def test_mock_uses_mapping_and_labels_metadata(self):
        result = MockBMSQLBackend(sql_by_id={"Q1.1": "SELECT gene_id FROM genes"}).generate(
            REQUEST
        )

        self.assertEqual(result.pred_sql, "SELECT gene_id FROM genes")
        self.assertEqual(result.model_metadata["backend"], "mock")
        self.assertIsNone(result.error)

    def test_mock_cleans_configured_sql(self):
        result = MockBMSQLBackend(sql_by_id={"Q1.1": " SELECT gene_id FROM genes \n"}).generate(
            REQUEST
        )

        self.assertEqual(result.pred_sql, "SELECT gene_id FROM genes")
        self.assertIsNone(result.error)

    def test_mock_uses_deterministic_default_sql(self):
        result = MockBMSQLBackend().generate(REQUEST)

        self.assertEqual(result.pred_sql, "SELECT 1 AS mock_value")
        self.assertEqual(result.model_metadata["backend"], "mock")

    def test_mock_returns_configured_failure_for_matching_id(self):
        result = MockBMSQLBackend(
            failure_ids={"Q1.1": "planned failure"}
        ).generate(REQUEST)

        self.assertIsNone(result.pred_sql)
        self.assertEqual(result.error, "planned failure")
        self.assertEqual(result.error_stage, "mock")

    def test_mock_rejects_blank_or_non_string_configured_sql(self):
        for value in ("   ", 7):
            with self.subTest(value=value):
                result = MockBMSQLBackend(sql_by_id={"Q1.1": value}).generate(REQUEST)

                self.assertIsNone(result.pred_sql)
                self.assertEqual(result.error_stage, "mock")
                self.assertIn("non-empty string", result.error)

    def test_upstream_backend_calls_public_agent_entry(self):
        agent = FakeAgent()
        result = UpstreamBMSQLBackend(agent=agent, num_passes=2).generate(REQUEST)

        self.assertEqual(agent.calls, [("Which genes?", 2)])
        self.assertEqual(result.pred_sql, "SELECT refined")
        self.assertEqual(result.stage_outputs["general_sql_query"], "SELECT general")
        self.assertEqual(result.stage_outputs["input_tokens"], 42)
        self.assertIsNone(result.error)

    def test_upstream_backend_uses_general_sql_when_refined_sql_is_blank(self):
        class BlankRefinementAgent:
            def run_agent(self, question, num_passes):
                return (" SELECT general ", [], "  ", [], "answer", 0)

        result = UpstreamBMSQLBackend(agent=BlankRefinementAgent()).generate(REQUEST)

        self.assertEqual(result.pred_sql, "SELECT general")
        self.assertIsNone(result.error)

    def test_upstream_backend_rejects_malformed_tuple(self):
        class MalformedAgent:
            def run_agent(self, question, num_passes):
                return ("SELECT only",)

        result = UpstreamBMSQLBackend(agent=MalformedAgent()).generate(REQUEST)

        self.assertIsNone(result.pred_sql)
        self.assertEqual(result.error_stage, "upstream_agent")
        self.assertEqual(result.raw_response, ("SELECT only",))

    def test_upstream_backend_reports_blank_sql_as_agent_failure(self):
        class BlankSqlAgent:
            def run_agent(self, question, num_passes):
                return (" ", [], "\n", [], "answer", 0)

        result = UpstreamBMSQLBackend(agent=BlankSqlAgent()).generate(REQUEST)

        self.assertIsNone(result.pred_sql)
        self.assertEqual(result.error_stage, "upstream_agent")
        self.assertEqual(result.raw_response, (" ", [], "\n", [], "answer", 0))
        self.assertEqual(
            result.stage_outputs,
            {
                "general_sql_query": " ",
                "general_exec_results": [],
                "refined_sql_query": "\n",
                "refined_exec_results": [],
                "answer": "answer",
                "input_tokens": 0,
            },
        )
        self.assertEqual(
            result.trajectory,
            [
                {"stage": "general_sql_query", "output": " "},
                {"stage": "general_exec_results", "output": []},
                {"stage": "refined_sql_query", "output": "\n"},
                {"stage": "refined_exec_results", "output": []},
                {"stage": "answer", "output": "answer"},
                {"stage": "input_tokens", "output": 0},
            ],
        )

    def test_upstream_backend_reports_agent_exception(self):
        result = UpstreamBMSQLBackend(agent=RaisingAgent()).generate(REQUEST)

        self.assertIsNone(result.pred_sql)
        self.assertEqual(result.error_stage, "upstream_agent")
        self.assertIn("upstream unavailable", result.error)


if __name__ == "__main__":
    unittest.main()
