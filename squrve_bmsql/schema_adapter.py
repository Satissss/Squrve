"""Convert BiomedSQL schemas to Squrve's parallel column format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def to_squrve_parallel_schema(
    schema: Any,
    *,
    db_id: str = "biomedsql",
    db_type: str = "big_query",
    project_id: str | None = None,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize central, table-oriented, or parallel schemas for Squrve."""
    if (project_id is None) != (dataset_name is None):
        raise ValueError("project_id and dataset_name must be supplied together")
    table_dataset = (
        None if project_id is None else f"{project_id}.{dataset_name}"
    )
    defaults = {
        "db_id": db_id,
        "db_type": db_type,
        "table_to_projDataset": table_dataset,
    }

    if isinstance(schema, Mapping) and "table_names_original" in schema:
        rows = _from_central(schema, **defaults)
    elif _is_parallel(schema):
        rows = _normalize_parallel(schema, **defaults)
    else:
        rows = _from_tables(schema, **defaults)
    if not rows:
        raise ValueError("Schema produced no Squrve columns")
    return sorted(rows, key=lambda row: (row["table_name"], row["column_name"]))


def _is_parallel(schema: Any) -> bool:
    return (
        isinstance(schema, list)
        and bool(schema)
        and all(
            isinstance(row, Mapping)
            and "table_name" in row
            and "column_name" in row
            for row in schema
        )
    )


def _from_central(
    schema: Mapping[str, Any], **defaults: Any
) -> list[dict[str, Any]]:
    tables = schema.get("table_names_original")
    columns = schema.get("column_names_original")
    if not _is_sequence(tables) or not _is_sequence(columns):
        raise ValueError("central schema requires table_names_original and column_names_original")

    if any(not _is_sequence(column) or len(column) != 2 for column in columns):
        raise ValueError("central column_names_original entries must be [table_index, column_name]")
    actual_columns = [
        (index, column)
        for index, column in enumerate(columns)
        if column[0] != -1
    ]

    types = schema.get("column_types", [])
    descriptions = schema.get("column_descriptions", [])
    sample_rows = schema.get("sample_rows", [])
    return [
        _make_row(
            table_name=_central_table_name(tables, column[0]),
            column_name=column[1],
            column_type=_central_value(types, position, source_index, len(columns)),
            description=_description(
                _central_value(descriptions, position, source_index, len(columns))
            ),
            sample_rows=_central_value(sample_rows, position, source_index, len(columns)),
            **defaults,
        )
        for position, (source_index, column) in enumerate(actual_columns)
    ]


def _from_tables(schema: Any, **defaults: Any) -> list[dict[str, Any]]:
    tables = schema.get("tables") if isinstance(schema, Mapping) else schema
    if not isinstance(tables, list):
        raise ValueError("table-oriented schema must be a list or {'tables': [...]} mapping")

    rows: list[dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, Mapping):
            raise ValueError("table-oriented schema entries must be mappings")
        table_name = table.get("table_name", table.get("name"))
        columns = table.get("columns")
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table-oriented schema entries require table_name or name")
        if isinstance(columns, Mapping):
            for column_name, column_type in columns.items():
                rows.append(
                    _make_row(
                        table_name=table_name,
                        column_name=column_name,
                        column_type=column_type,
                        **defaults,
                    )
                )
        elif isinstance(columns, list):
            for column in columns:
                if not isinstance(column, Mapping):
                    raise ValueError("table-oriented columns must be mappings")
                rows.append(
                    _make_row(
                        table_name=table_name,
                        column_name=column.get("column_name", column.get("name")),
                        column_type=column.get("column_types", column.get("type", "")),
                        description=column.get("column_descriptions", column.get("description", "")),
                        sample_rows=column.get("sample_rows", []),
                        **defaults,
                    )
                )
        else:
            raise ValueError("table-oriented schema entries require columns")
    return rows


def _normalize_parallel(
    schema: list[Mapping[str, Any]], **defaults: Any
) -> list[dict[str, Any]]:
    return [
        _make_row(
            table_name=row["table_name"],
            column_name=row["column_name"],
            column_type=row.get("column_types", ""),
            description=row.get("column_descriptions", ""),
            sample_rows=row.get("sample_rows", []),
            **defaults,
        )
        for row in schema
    ]


def _make_row(
    *,
    table_name: Any,
    column_name: Any,
    column_type: Any,
    db_id: str,
    db_type: str,
    table_to_projDataset: str | None,
    description: Any = "",
    sample_rows: Any = None,
) -> dict[str, Any]:
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("table_name must be a non-empty string")
    if not isinstance(column_name, str) or not column_name.strip():
        raise ValueError("column_name must be a non-empty string")
    return {
        "db_id": db_id,
        "db_type": db_type,
        "table_name": table_name,
        "column_name": column_name,
        "column_types": column_type,
        "column_descriptions": description,
        "sample_rows": [] if sample_rows is None else sample_rows,
        "table_to_projDataset": table_to_projDataset,
    }


def _central_table_name(tables: Sequence[Any], table_index: Any) -> Any:
    if not isinstance(table_index, int) or not 0 <= table_index < len(tables):
        raise ValueError("central column references an invalid table index")
    return tables[table_index]


def _central_value(
    values: Any, position: int, source_index: int, source_length: int
) -> Any:
    if not _is_sequence(values):
        return None
    index = source_index if len(values) == source_length else position
    return values[index] if index < len(values) else None


def _description(value: Any) -> Any:
    if _is_sequence(value) and len(value) == 2:
        return value[1]
    return "" if value is None else value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
