"""
File to Database conversion module for Squrve Demo.

Supports:
- Multiple xlsx/csv files -> single sqlite database (each file = one table)
- Single sqlite file upload -> extract schema to Spider format
"""

import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from core.utils import load_dataset, save_dataset

UPLOAD_MANIFEST = "manifest.json"


def _get_manifest_path(base_root: Path) -> Path:
    return base_root / UPLOAD_MANIFEST


def load_upload_manifest(base_root: Union[str, Path]) -> List[Dict]:
    """Load manifest of uploaded databases."""
    path = _get_manifest_path(Path(base_root))
    if not path.exists():
        return []
    data = load_dataset(path)
    return data if isinstance(data, list) else []


def save_upload_manifest(base_root: Union[str, Path], entries: List[Dict]) -> None:
    """Save manifest of uploaded databases."""
    path = _get_manifest_path(Path(base_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(entries, new_data_source=path)


def add_to_manifest(base_root: Union[str, Path], entry: Dict) -> None:
    """Add or update an entry in the upload manifest."""
    entries = load_upload_manifest(base_root)
    db_id = entry.get("db_id")
    entries = [e for e in entries if e.get("db_id") != db_id]
    entries.append(entry)
    save_upload_manifest(base_root, entries)


def _sanitize_table_name(name: str) -> str:
    """Sanitize filename to valid SQLite table name."""
    # Remove extension if present
    name = Path(name).stem
    # Replace invalid chars with underscore
    name = re.sub(r'[^\w]', '_', name)
    return name or "table"


def xlsx_csv_to_sqlite(
    file_paths: List[Union[str, Path]],
    output_db_path: Union[str, Path],
    db_id: Optional[str] = None,
) -> str:
    """
    Convert multiple xlsx/csv files to a single SQLite database.
    Each file becomes one table; filename (without extension) = table name.
    First row of each file = column names.

    Args:
        file_paths: List of paths to xlsx or csv files
        output_db_path: Path for output sqlite file
        db_id: Database ID (default: stem of output_db_path)

    Returns:
        db_id used for the database
    """
    output_db_path = Path(output_db_path)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    db_id = db_id or output_db_path.stem

    with sqlite3.connect(output_db_path) as conn:
        for fp in file_paths:
            fp = Path(fp)
            if not fp.exists():
                logger.warning(f"File not found: {fp}")
                continue
            table_name = _sanitize_table_name(fp.name)
            if fp.suffix.lower() == ".csv":
                df = pd.read_csv(fp)
            elif fp.suffix.lower() in (".xlsx", ".xls"):
                try:
                    df = pd.read_excel(fp)
                except ImportError:
                    raise ImportError("Reading Excel files requires openpyxl. Install with: pip install openpyxl")
            else:
                logger.warning(f"Unsupported format: {fp.suffix}")
                continue
            # Sanitize column names for SQLite
            df.columns = [re.sub(r'[^\w]', '_', str(c)) for c in df.columns]
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"Imported {fp.name} -> table {table_name}")

    return db_id


def sqlite_to_schema(
    db_path: Union[str, Path],
    db_id: Optional[str] = None,
) -> Dict:
    """
    Extract schema from SQLite database to Spider central format.
    foreign_keys and primary_keys are left empty.

    Args:
        db_path: Path to sqlite file
        db_id: Database ID (default: stem of db_path)

    Returns:
        Schema dict compatible with single_central_process
    """
    db_path = Path(db_path)
    db_id = db_id or db_path.stem

    table_names = []
    column_names_original = [[-1, "*"]]
    column_names = [[-1, "*"]]
    column_types = []  # types for actual columns only (single_central_process uses types[ind] for columns excluding *)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('sqlite_sequence')"
        )
        tables = [row[0] for row in cursor.fetchall()]

    for ti, table_name in enumerate(tables):
        if table_name == "sqlite_sequence":
            continue
        table_names.append(table_name)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(f"PRAGMA table_info(`{table_name}`)")
            rows = cursor.fetchall()
        for _, col_name, col_type, _, _, _ in rows:
            column_names_original.append([ti, col_name])
            column_names.append([ti, col_name.lower()])
            ctype = str(col_type).upper()
            if "INT" in ctype or "INTEGER" in ctype:
                spider_type = "number"
            elif "REAL" in ctype or "FLOAT" in ctype or "DOUBLE" in ctype:
                spider_type = "number"
            else:
                spider_type = "text"
            column_types.append(spider_type)

    # single_central_process uses types[ind] for enumerate(columns); columns excludes *
    # So types must be [col1_type, col2_type, ...] - same length as columns
    return {
        "db_id": db_id,
        "db_type": "sqlite",
        "table_names": table_names,
        "table_names_original": table_names,
        "column_names": column_names,
        "column_names_original": column_names_original,
        "column_types": column_types,
        "primary_keys": [],
        "foreign_keys": [],
    }


def process_uploaded_files(
    files: List[Union[str, Path]],
    base_root: Union[str, Path],
    db_id: Optional[str] = None,
) -> Dict:
    """
    Process uploaded files (xlsx/csv or sqlite) into database + schema.
    Saves to base_root/{db_id}/ and updates manifest.

    - If all files are xlsx/csv: merge into one sqlite, generate schema
    - If single sqlite: copy to base_root/{db_id}, extract schema

    Args:
        files: List of file paths (from Gradio upload)
        base_root: Root directory (e.g. files/uploaded_db)
        db_id: Optional db_id (auto-generated if not provided)

    Returns:
        {
            "db_id": str,
            "db_path": str,
            "schema_path": str,
            "schema_base_dir": str,  # dir containing schema.json for Dataset
            "table_name": str or None,
            "schema_list": list,
        }
    """
    base_root = Path(base_root)
    base_root.mkdir(parents=True, exist_ok=True)

    if not files:
        raise ValueError("No files provided")

    # Normalize to Path objects; Gradio may pass path string or Path
    # Note: Path.name returns filename only, so use Path(f) for str/Path to preserve full path
    paths = []
    for f in files:
        if isinstance(f, (str, Path)):
            p = Path(f)
        else:
            p = Path(getattr(f, "name", str(f)))
            if not p.exists() and hasattr(f, "name"):
                p = Path(f.name)
        paths.append(p)

    sqlite_ext = {".sqlite", ".db"}
    xlsx_csv_ext = {".xlsx", ".xls", ".csv"}

    def is_sqlite(p: Path) -> bool:
        return p.suffix.lower() in sqlite_ext

    def is_xlsx_csv(p: Path) -> bool:
        return p.suffix.lower() in xlsx_csv_ext

    sqlite_files = [p for p in paths if is_sqlite(p)]
    xlsx_csv_files = [p for p in paths if is_xlsx_csv(p)]

    if len(sqlite_files) == 1 and len(paths) == 1:
        # Single sqlite upload
        src = sqlite_files[0]
        db_id = db_id or src.stem
        base_dir = base_root / db_id
        base_dir.mkdir(parents=True, exist_ok=True)
        db_path = base_dir / f"{db_id}.sqlite"
        shutil.copy2(src, db_path)
        schema_dict = sqlite_to_schema(db_path, db_id)
        schema_path = base_dir / "schema.json"
        save_dataset([schema_dict], new_data_source=schema_path)
        result = {
            "db_id": db_id,
            "db_path": str(db_path),
            "schema_path": str(schema_path),
            "schema_base_dir": str(base_dir),
            "table_name": None,
            "schema_list": schema_dict.get("table_names_original", []),
        }
        add_to_manifest(base_root, result)
        return result

    if xlsx_csv_files and len(xlsx_csv_files) == len(paths):
        # Multiple xlsx/csv -> one sqlite
        db_id = db_id or f"upload_{uuid.uuid4().hex[:8]}"
        base_dir = base_root / db_id
        base_dir.mkdir(parents=True, exist_ok=True)
        db_path = base_dir / f"{db_id}.sqlite"
        xlsx_csv_to_sqlite(xlsx_csv_files, db_path, db_id)
        schema_dict = sqlite_to_schema(db_path, db_id)
        schema_path = base_dir / "schema.json"
        save_dataset([schema_dict], new_data_source=schema_path)
        result = {
            "db_id": db_id,
            "db_path": str(db_path),
            "schema_path": str(schema_path),
            "schema_base_dir": str(base_dir),
            "table_name": None,
            "schema_list": schema_dict.get("table_names_original", []),
        }
        add_to_manifest(base_root, result)
        return result

    raise ValueError(
        "Invalid upload: provide either (1) one .sqlite/.db file, or "
        "(2) one or more .xlsx/.xls/.csv files"
    )
