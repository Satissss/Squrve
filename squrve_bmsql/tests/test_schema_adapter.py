import unittest

from squrve_bmsql.schema_adapter import to_squrve_parallel_schema


class SchemaAdapterTests(unittest.TestCase):
    def test_flattens_central_schema_and_skips_star(self):
        central = {
            "db_id": "biomedsql",
            "table_names_original": ["genes"],
            "column_names_original": [[-1, "*"], [0, "gene_id"], [0, "symbol"]],
            "column_types": ["STRING", "STRING"],
            "column_descriptions": ["identifier", "gene symbol"],
        }

        rows = to_squrve_parallel_schema(central)

        self.assertEqual([row["column_name"] for row in rows], ["gene_id", "symbol"])
        self.assertTrue(all(row["table_name"] == "genes" for row in rows))
        self.assertTrue(all(row["db_type"] == "big_query" for row in rows))
        self.assertTrue(
            all(
                set(row)
                == {
                    "db_id",
                    "db_type",
                    "table_name",
                    "column_name",
                    "column_types",
                    "column_descriptions",
                    "sample_rows",
                    "table_to_projDataset",
                }
                for row in rows
            )
        )

    def test_flattens_table_oriented_schema_with_column_records_and_mapping(self):
        tables = [
            {
                "name": "diseases",
                "columns": {
                    "disease_id": "INT64",
                    "name": "STRING",
                },
            },
            {
                "table_name": "genes",
                "columns": [
                    {
                        "column_name": "symbol",
                        "column_types": "STRING",
                        "column_descriptions": "gene symbol",
                        "sample_rows": ["BRCA1"],
                    }
                ],
            },
        ]

        rows = to_squrve_parallel_schema(
            tables,
            db_id="bmsql",
            db_type="big_query",
            project_id="research-proj",
            dataset_name="biomed",
        )

        self.assertEqual(
            [(row["table_name"], row["column_name"]) for row in rows],
            [("diseases", "disease_id"), ("diseases", "name"), ("genes", "symbol")],
        )
        self.assertEqual(rows[-1]["column_descriptions"], "gene symbol")
        self.assertEqual(rows[-1]["sample_rows"], ["BRCA1"])
        self.assertTrue(
            all(row["table_to_projDataset"] == "research-proj.biomed" for row in rows)
        )

    def test_accepts_tables_wrapper_and_singular_column_aliases(self):
        schema = {
            "tables": [
                {
                    "name": "genes",
                    "columns": [
                        {
                            "name": "symbol",
                            "type": "STRING",
                            "description": "gene symbol",
                            "sample_rows": ["BRCA1"],
                        }
                    ],
                }
            ]
        }

        rows = to_squrve_parallel_schema(schema)

        self.assertEqual(
            rows,
            [
                {
                    "db_id": "biomedsql",
                    "db_type": "big_query",
                    "table_name": "genes",
                    "column_name": "symbol",
                    "column_types": "STRING",
                    "column_descriptions": "gene symbol",
                    "sample_rows": ["BRCA1"],
                    "table_to_projDataset": None,
                }
            ],
        )

    def test_normalizes_parallel_rows_without_retaining_extra_fields(self):
        parallel = [
            {
                "table_name": "genes",
                "column_name": "symbol",
                "column_types": "STRING",
                "column_descriptions": "gene symbol",
                "sample_rows": ["BRCA1"],
                "credential": "must not survive",
            }
        ]

        rows = to_squrve_parallel_schema(parallel, db_id="bmsql", db_type="big_query")

        self.assertEqual(rows[0]["db_id"], "bmsql")
        self.assertEqual(rows[0]["table_to_projDataset"], None)
        self.assertNotIn("credential", rows[0])

    def test_rejects_partial_project_dataset_and_invalid_schema_shapes(self):
        table = [{"table_name": "genes", "columns": {"symbol": "STRING"}}]
        for settings in (
            {"project_id": "research-proj"},
            {"dataset_name": "biomed"},
        ):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, "project_id and dataset_name"):
                    to_squrve_parallel_schema(table, **settings)
        with self.assertRaisesRegex(ValueError, "Schema produced no Squrve columns"):
            to_squrve_parallel_schema([])
        with self.assertRaisesRegex(ValueError, "table-oriented"):
            to_squrve_parallel_schema({"genes": {"symbol": "STRING"}})
        with self.assertRaisesRegex(ValueError, "column_names_original"):
            to_squrve_parallel_schema(
                {
                    "table_names_original": ["genes"],
                    "column_names_original": [[0, "gene_id"], "not-a-column"],
                }
            )


if __name__ == "__main__":
    unittest.main()
