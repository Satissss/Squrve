import json
import unittest

from squrve_bmsql.data_adapter import (
    adapt_biomedsql_row,
    select_pilot_rows,
    substitute_sql_placeholders,
)
from squrve_bmsql.models import (
    BMSQLGeneration,
    BMSQLRequest,
    Evaluation,
    QueryExecution,
    ResultStatus,
    SampleResult,
)


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

    def test_rejects_empty_questions_duplicate_ids_and_short_sources(self):
        with self.assertRaisesRegex(ValueError, "question"):
            adapt_biomedsql_row({"uuid": "Q1", "question": "  ", "benchmark_query": "SELECT 1"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_pilot_rows(
                [{"instance_id": "Q1"}, {"instance_id": "Q1"}], sample_size=2
            )
        with self.assertRaisesRegex(ValueError, "Need at least 2 rows, received 1"):
            select_pilot_rows([{"instance_id": "Q1"}], sample_size=2)

    def test_rejects_missing_ids_and_unsafe_placeholder_values(self):
        with self.assertRaisesRegex(ValueError, "stable"):
            select_pilot_rows([{"question": "one"}], sample_size=1)
        for value in (
            "bad project",
            "bad{project",
            "bad`project",
            "bad;project",
            "bad,project",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    substitute_sql_placeholders(
                        "SELECT * FROM `{project_id}.dataset.table`", project_id=value
                    )

    def test_serialization_round_trips_enums_and_nested_execution_records(self):
        request = BMSQLRequest(
            instance_id="Q1.1",
            question="Which genes?",
            schema=[{"table_name": "genes"}],
            domain_context="Genetics",
            metadata={"source": "unit-test"},
        )
        generation = BMSQLGeneration(
            pred_sql="SELECT gene_id FROM genes",
            raw_response={"answer": "A"},
            stage_outputs={"general_sql_query": "SELECT gene_id FROM genes"},
            trajectory=[{"stage": "generate", "output": "SELECT gene_id FROM genes"}],
            latency_seconds=0.125,
            model_metadata={"backend": "mock"},
        )
        predicted = QueryExecution(success=True, rows=[{"gene_id": "G1"}])
        gold = QueryExecution(success=True, rows=[{"gene_id": "G1"}])
        evaluation = Evaluation(
            status=ResultStatus.EXECUTED_RESULT_MATCH,
            predicted=predicted,
            gold=gold,
        )
        result = SampleResult(
            instance_id="Q1.1",
            question="Which genes?",
            gold_sql="SELECT gene_id FROM genes",
            generation=generation,
            evaluation=evaluation,
            metadata={"template_uuid": "Q1"},
        )

        for value, cls in (
            (request, BMSQLRequest),
            (generation, BMSQLGeneration),
            (predicted, QueryExecution),
            (evaluation, Evaluation),
            (result, SampleResult),
        ):
            with self.subTest(cls=cls.__name__):
                encoded = json.dumps(value.to_dict())
                self.assertEqual(cls.from_dict(json.loads(encoded)), value)


if __name__ == "__main__":
    unittest.main()
