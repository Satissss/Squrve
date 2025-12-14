"""
File to Database Converter for Squrve

This module provides functionality to convert Excel/CSV files to SQLite databases
and generate corresponding schema files in Squrve format.
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import Union, Optional, Dict, List
from loguru import logger
import uuid
import re


def sanitize_name(name: str) -> str:
    """Sanitize table/column names for SQLite compatibility"""
    # Remove or replace invalid characters
    name = re.sub(r'[^\w]', '_', name)
    # Remove leading numbers
    name = re.sub(r'^\d+', '', name)
    # Ensure it doesn't start with underscore
    if name.startswith('_'):
        name = 'col_' + name[1:]
    # Ensure non-empty
    if not name:
        name = 'column'
    return name


def infer_sqlite_type(series: pd.Series) -> str:
    """Infer SQLite type from pandas Series"""
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    elif pd.api.types.is_float_dtype(series):
        return "real"
    elif pd.api.types.is_bool_dtype(series):
        return "integer"  # SQLite uses integer for boolean
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "text"  # SQLite doesn't have native date type
    else:
        return "text"


def excel_csv_to_sqlite(
    file_path: Union[str, Path],
    db_path: Union[str, Path],
    table_name: Optional[str] = None,
    sheet_name: Optional[str] = None
) -> str:
    """
    Convert Excel or CSV file to SQLite database.
    
    Args:
        file_path: Path to Excel (.xlsx, .xls) or CSV (.csv) file
        db_path: Path where SQLite database will be created
        table_name: Name for the table (default: file stem)
        sheet_name: For Excel files, which sheet to use (default: first sheet)
    
    Returns:
        str: Name of the created table
    """
    file_path = Path(file_path)
    db_path = Path(db_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and read data
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        if table_name is None:
            table_name = sanitize_name(file_path.stem)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path, sheet_name=0)  # First sheet
        
        if table_name is None:
            if sheet_name:
                table_name = sanitize_name(sheet_name)
            else:
                # Try to get sheet name from Excel
                xl_file = pd.ExcelFile(file_path)
                table_name = sanitize_name(xl_file.sheet_names[0])
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Sanitize column names
    df.columns = [sanitize_name(str(col)) for col in df.columns]
    table_name = sanitize_name(table_name)
    
    # Ensure db_path directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create SQLite database and table
    conn = sqlite3.connect(str(db_path))
    try:
        # Write DataFrame to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        logger.info(f"Created SQLite database: {db_path} with table: {table_name}")
    finally:
        conn.close()
    
    return table_name


def generate_schema_from_db(
    db_path: Union[str, Path],
    db_id: str,
    table_name: str,
    schema_dir: Union[str, Path],
    sample_rows: int = 5
) -> List[Dict]:
    """
    Generate Squrve Parallel format schema from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        db_id: Database identifier
        table_name: Name of the table
        schema_dir: Directory to save schema files
        sample_rows: Number of sample rows to include
    
    Returns:
        List[Dict]: Schema in Parallel format
    """
    db_path = Path(db_path)
    schema_dir = Path(schema_dir)
    schema_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    try:
        # Get table schema
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        
        # Get sample data
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {sample_rows}", conn)
        
        schema_list = []
        for col_info in columns_info:
            col_id, col_name, col_type, not_null, default_val, pk = col_info
            
            # Get sample values for this column
            sample_values = df[col_name].dropna().head(sample_rows).tolist() if not df.empty else []
            # Convert to strings for JSON serialization
            sample_values = [str(v) for v in sample_values]
            
            schema_item = {
                "db_id": db_id,
                "db_type": "sqlite",
                "table_name": table_name,
                "column_name": col_name,
                "column_types": col_type.lower() if col_type else "text",
                "column_descriptions": "",
                "sample_rows": sample_values,
                "table_to_projDataset": ""
            }
            
            schema_list.append(schema_item)
            
            # Save individual schema file (Parallel format)
            schema_file = schema_dir / f"{table_name}_{col_name}.json"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema_item, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(schema_list)} schema files in {schema_dir}")
        return schema_list
        
    finally:
        conn.close()


def process_uploaded_file(
    file_path: Union[str, Path],
    db_id: Optional[str] = None,
    table_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, str]:
    """
    Process uploaded Excel/CSV file and create database + schema.
    
    Args:
        file_path: Path to uploaded file
        db_id: Database ID (default: generated from filename)
        table_name: Table name (default: from filename)
        output_dir: Output directory for database and schema (default: files/user_databases)
    
    Returns:
        Dict with db_id, db_path, schema_dir, table_name
    """
    file_path = Path(file_path)
    
    if db_id is None:
        db_id = sanitize_name(file_path.stem)
    
    if output_dir is None:
        from core.path_utils import get_project_root
        project_root = get_project_root()
        output_dir = project_root / "files" / "user_databases"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database
    db_path = output_dir / f"{db_id}.sqlite"
    actual_table_name = excel_csv_to_sqlite(file_path, db_path, table_name)
    
    # Generate schema
    # Schema directory structure for single database:
    # schemas/{db_id}/single_db/{db_id}/ (for parallel format files)
    # But Dataset.get_db_schema expects: schema_source/{db_id}/ when multi_database=False
    # So we use: schemas/{db_id}/{db_id}/ to match the expected structure
    schema_dir = output_dir / "schemas" / db_id / db_id
    schema_list = generate_schema_from_db(db_path, db_id, actual_table_name, schema_dir)
    
    # Return schema base directory (schemas/{db_id}) for use as schema_source
    schema_base_dir = output_dir / "schemas" / db_id
    
    return {
        "db_id": db_id,
        "db_path": str(db_path),
        "schema_dir": str(schema_dir),
        "schema_base_dir": str(schema_base_dir),  # This is what we use as schema_source
        "table_name": actual_table_name,
        "schema_list": schema_list
    }

