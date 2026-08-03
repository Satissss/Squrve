"""Local DuckDB substitute for BigQuery-backed BMSQL experiments.

This is useful when Google Cloud cannot be provisioned. It reads the official
BiomedSQL Parquet tables locally and rewrites BigQuery's three-part table names
to DuckDB views. It is intentionally labelled as a local reproduction because
SQL dialect and execution semantics can differ from BigQuery.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .evaluator import is_read_only_sql
from .models import QueryExecution


_QUALIFIED_TABLE = re.compile(
    r"`(?:[A-Za-z0-9_-]+\.){2}([A-Za-z0-9_-]+)`"
)


class DuckDBReadOnlyExecutor:
    """Execute read-only SQL against Parquet files in a local directory."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        project_id: str = "project",
        dataset_name: str = "dataset",
    ) -> None:
        import duckdb

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.connection = duckdb.connect(database=":memory:")
        self.tables: set[str] = set()
        for path in sorted(self.data_dir.glob("*.parquet")):
            table = path.stem
            parquet_path = str(path).replace("'", "''")
            self.connection.execute(
                "CREATE OR REPLACE VIEW \"" + table.replace('"', '""') +
                "\" AS SELECT * FROM read_parquet('" + parquet_path + "')"
            )
            self.tables.add(table)

    def _rewrite(self, sql: str) -> str:
        return _QUALIFIED_TABLE.sub(lambda match: '"' + match.group(1).replace('"', '""') + '"', sql)

    def execute(self, sql: str, *, db_id: str | None = None) -> QueryExecution:
        if not is_read_only_sql(sql):
            return QueryExecution(success=False, error="Only one read-only SELECT or WITH query is allowed", error_type="unsafe_sql")
        try:
            cursor = self.connection.execute(self._rewrite(sql))
            columns = [item[0] for item in cursor.description or []]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return QueryExecution(success=True, rows=rows, metadata={"backend": "duckdb", "db_id": db_id})
        except Exception as error:
            return QueryExecution(success=False, error=str(error), error_type="execution_error", metadata={"backend": "duckdb", "db_id": db_id})


class DuckDBBMSQLHandler:
    """Small ``query_db`` surface accepted by the official SQLHandler."""

    def __init__(self, executor: DuckDBReadOnlyExecutor):
        self.executor = executor

    def query_db(self, query: str) -> list[dict[str, Any]]:
        result = self.executor.execute(query)
        if not result.success:
            raise RuntimeError(result.error or "DuckDB query failed")
        return result.rows
