import tempfile
import unittest
from pathlib import Path

import duckdb

from squrve_bmsql.local_duckdb import DuckDBReadOnlyExecutor


class LocalDuckDBTests(unittest.TestCase):
    def test_rewrites_bigquery_names_and_blocks_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = duckdb.connect()
            connection.execute("CREATE TABLE genes AS SELECT 'rs1' AS SNP, 0.42 AS beta")
            connection.execute(
                f"COPY genes TO '{root / 'genes.parquet'}' (FORMAT PARQUET)"
            )
            executor = DuckDBReadOnlyExecutor(root)
            result = executor.execute(
                "SELECT SNP, beta FROM `project.dataset.genes` WHERE SNP = 'rs1'"
            )
            self.assertTrue(result.success)
            self.assertEqual(result.rows[0]["SNP"], "rs1")
            self.assertEqual(executor.execute("DELETE FROM genes").error_type, "unsafe_sql")


if __name__ == "__main__":
    unittest.main()
