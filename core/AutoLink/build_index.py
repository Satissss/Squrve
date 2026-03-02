"""
build_index.py – One-time FAISS index builder for AutoLinkParser.

Reads Squrve benchmark schema.json files and optional SQLite databases,
then produces per-database FAISS index + metadata.json files that
AutoLinkParser uses at runtime via core/AutoLink/retrieval.py.

Supported datasets
------------------
- spider/dev   : Spider schema format, SQLite dbs, no column descriptions
- spider2/lite : Spider2 schema format, mixed BigQuery/SQLite, has descriptions
- bird/dev     : Same Spider schema format as spider/dev
- AmbiDB       : Same Spider schema format as spider/dev

Usage
-----
# Build index for spider/dev:
python -m core.AutoLink.build_index \
    --schema   benchmarks/spider/dev/schema.json \
    --db_dir   benchmarks/spider/database \
    --out_dir  embeddings/spider \
    --dataset  spider

# Build index for spider2/lite (SQLite dbs only):
python -m core.AutoLink.build_index \
    --schema   benchmarks/spider2/lite/schema.json \
    --db_dir   benchmarks/spider2/lite/database \
    --out_dir  embeddings/spider2_lite \
    --dataset  spider2

# Skip already-built databases:
python -m core.AutoLink.build_index ... --skip_existing

Output layout
-------------
<out_dir>/
└── <db_id>/
    ├── index.faiss     – FlatL2 FAISS index
    └── metadata.json   – list of {table, column, column_type,
                          column_value, description} per vector
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Lazy imports so the module can be imported without heavy deps installed
# ---------------------------------------------------------------------------

def _require_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        sys.exit("faiss-cpu is required: pip install faiss-cpu")


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers is required: pip install sentence-transformers")


# ---------------------------------------------------------------------------
# Schema parsing helpers
# ---------------------------------------------------------------------------

def _parse_spider_schema(entry: dict) -> Tuple[List[str], List[str], List[str]]:
    """Parse one entry from a Spider-format schema.json.

    Returns (table_names, column_names, column_types) all as flat lists
    aligned by index (excluding the '*' wildcard pseudo-column).
    column_types matches by global column index (after dropping '*').
    """
    table_names_orig = entry.get("table_names_original", entry.get("table_names", []))
    col_names_orig   = entry.get("column_names_original", [])  # [[table_idx, col_name], ...]
    col_types        = entry.get("column_types", [])

    tables:  List[str] = []
    columns: List[str] = []
    types:   List[str] = []

    for i, (tbl_idx, col_name) in enumerate(col_names_orig):
        if col_name == "*":       # skip wildcard
            continue
        if tbl_idx < 0:           # skip pseudo rows
            continue
        tables.append(table_names_orig[tbl_idx] if tbl_idx < len(table_names_orig) else "")
        columns.append(col_name)
        types.append(col_types[i] if i < len(col_types) else "")

    return tables, columns, types


def _parse_spider2_descriptions(entry: dict) -> Dict[Tuple[int, str], str]:
    """Extract {(table_idx, col_name): description} from a Spider2 schema entry."""
    desc_map: Dict[Tuple[int, str], str] = {}
    col_descs = entry.get("column_descriptions", [])
    col_names  = entry.get("column_names_original", [])

    for i, desc_item in enumerate(col_descs):
        if i >= len(col_names):
            break
        tbl_idx, col_name = col_names[i]
        raw_desc = desc_item[1] if isinstance(desc_item, list) and len(desc_item) > 1 else None
        if raw_desc:
            desc_map[(tbl_idx, col_name)] = str(raw_desc)

    return desc_map


# ---------------------------------------------------------------------------
# SQLite sample value extractor
# ---------------------------------------------------------------------------

def _sample_values(
    db_path: str,
    table_name: str,
    col_name: str,
    n: int = 3,
) -> str:
    """Return a comma-joined string of up to *n* distinct non-null sample values."""
    if not os.path.exists(db_path):
        return ""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        # Quote identifiers to handle spaces and reserved words
        safe_col   = f'"{col_name}"'
        safe_table = f'"{table_name}"'
        cur.execute(
            f"SELECT DISTINCT {safe_col} FROM {safe_table} "
            f"WHERE {safe_col} IS NOT NULL LIMIT {n}"
        )
        rows = [str(r[0]) for r in cur.fetchall()]
        conn.close()
        return ", ".join(rows)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def _build_metadata_spider(
    entry: dict,
    db_dir: Optional[str],
) -> List[dict]:
    """Build metadata list for one Spider-format DB entry."""
    db_id = entry["db_id"]
    tables, columns, types = _parse_spider_schema(entry)

    db_path = os.path.join(db_dir, f"{db_id}.sqlite") if db_dir else None

    records = []
    for table, col, ctype in zip(tables, columns, types):
        sample = _sample_values(db_path, table, col) if db_path else ""
        desc = (
            f"column name: {col}\n"
            f"column type: {ctype}\n"
            f"table name: {table}\n"
            f"description: \n"
            f"sample values: {sample}"
        )
        records.append({
            "table":        table,
            "column":       col,
            "column_type":  ctype,
            "column_value": sample,
            "description":  desc,
        })
    return records


def _build_metadata_spider2(
    entry: dict,
    db_dir: Optional[str],
) -> List[dict]:
    """Build metadata list for one Spider2-format DB entry.

    For SQLite databases, also queries sample values.
    For BigQuery/Snowflake, sample values are left empty.
    """
    db_id   = entry["db_id"]
    db_type = entry.get("db_type", "sqlite").lower()

    tables, columns, types = _parse_spider_schema(entry)
    col_names_orig = entry.get("column_names_original", [])
    desc_map       = _parse_spider2_descriptions(entry)

    # Build (table_idx, col_name) for each non-wildcard column
    col_keys: List[Tuple[int, str]] = []
    for tbl_idx, col_name in col_names_orig:
        if col_name == "*" or tbl_idx < 0:
            continue
        col_keys.append((tbl_idx, col_name))

    db_path = None
    if db_type == "sqlite" and db_dir:
        candidate = os.path.join(db_dir, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            db_path = candidate

    records = []
    for (tbl_idx, col_name), table, col, ctype in zip(col_keys, tables, columns, types):
        raw_desc = desc_map.get((tbl_idx, col_name), "")
        sample   = _sample_values(db_path, table, col) if db_path else ""
        desc = (
            f"column name: {col}\n"
            f"column type: {ctype}\n"
            f"table name: {table}\n"
            f"description: {raw_desc}\n"
            f"sample values: {sample}"
        )
        records.append({
            "table":        table,
            "column":       col,
            "column_type":  ctype,
            "column_value": sample,
            "description":  desc,
        })
    return records


# ---------------------------------------------------------------------------
# Index writer
# ---------------------------------------------------------------------------

def _write_index(
    records: List[dict],
    db_id: str,
    out_dir: str,
    model,
    faiss,
    batch_size: int = 64,
    skip_existing: bool = False,
) -> bool:
    """Encode *records* and write FAISS index + metadata to *out_dir/<db_id>/*."""
    if not records:
        return False

    db_out = os.path.join(out_dir, db_id)
    idx_path  = os.path.join(db_out, "index.faiss")
    meta_path = os.path.join(db_out, "metadata.json")

    if skip_existing and os.path.exists(idx_path) and os.path.exists(meta_path):
        print(f"  [skip] {db_id} – already built")
        return False

    os.makedirs(db_out, exist_ok=True)

    texts = [r["description"] for r in records]

    # Encode in batches
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.extend(vecs)

    mat = np.array(all_vecs, dtype=np.float32)
    dim = mat.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    faiss.write_index(index, idx_path)

    # Keep only serialisable fields in metadata
    meta = [
        {k: r[k] for k in ("table", "column", "column_type", "column_value", "description")}
        for r in records
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_index(
    schema_path: str,
    out_dir: str,
    dataset: str = "spider",
    db_dir: Optional[str] = None,
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
    skip_existing: bool = False,
    db_filter: Optional[List[str]] = None,
    device: str = "cpu",
    sqlite_only: bool = False,
    no_samples: bool = False,
) -> None:
    """Build FAISS indexes for all databases in *schema_path*.

    Parameters
    ----------
    schema_path : str
        Path to the Squrve schema.json file.
    out_dir : str
        Root directory to write index files into.
    dataset : str
        'spider' (Spider/BIRD/AmbiDB format) or 'spider2' (Spider2 format).
    db_dir : str, optional
        Directory containing <db_id>.sqlite files for sample value extraction.
    model_name : str
        HuggingFace SentenceTransformer model name.
    batch_size : int
        Encoding batch size.
    skip_existing : bool
        Skip databases whose index files already exist.
    db_filter : list[str], optional
        If set, only build indexes for these db_ids.
    device : str
        Device for SentenceTransformer: 'cpu', 'cuda', 'cuda:0'.
    sqlite_only : bool
        Skip schema entries whose db_type is not 'sqlite' (e.g. BigQuery/Snowflake).
    no_samples : bool
        Skip SQLite sample value extraction (faster, less accurate retrieval).
    """
    faiss_mod = _require_faiss()
    ST        = _require_sentence_transformers()

    print(f"Loading model: {model_name} (device={device})")
    model = ST(model_name, device=device)
    print("Model loaded.\n")

    print(f"Reading schema: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    print(f"  → {len(schema)} databases found.\n")

    os.makedirs(out_dir, exist_ok=True)

    built = skipped = failed = 0

    for entry in tqdm(schema, desc="Building indexes"):
        db_id = entry.get("db_id", "")
        if db_filter and db_id not in db_filter:
            continue
        if sqlite_only and entry.get("db_type", "sqlite").lower() != "sqlite":
            skipped += 1
            continue

        try:
            effective_db_dir = None if no_samples else db_dir
            if dataset == "spider2":
                records = _build_metadata_spider2(entry, effective_db_dir)
            else:
                records = _build_metadata_spider(entry, effective_db_dir)

            if not records:
                print(f"  [warn] {db_id} – no columns found, skipping")
                skipped += 1
                continue

            wrote = _write_index(
                records, db_id, out_dir,
                model, faiss_mod,
                batch_size=batch_size,
                skip_existing=skip_existing,
            )
            if wrote:
                built += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  [error] {db_id}: {e}")
            failed += 1

    print(f"\nDone. built={built}, skipped={skipped}, failed={failed}")
    print(f"Index files written to: {os.path.abspath(out_dir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(
        description="Build FAISS indexes for AutoLinkParser from Squrve benchmark schemas."
    )
    p.add_argument("--schema",    required=True,  help="Path to schema.json")
    p.add_argument("--out_dir",   required=True,  help="Output root directory")
    p.add_argument("--dataset",   default="spider",
                   choices=["spider", "spider2"],
                   help="Schema format: 'spider' (spider/bird/AmbiDB) or 'spider2'")
    p.add_argument("--db_dir",    default=None,   help="Directory with .sqlite files")
    p.add_argument("--model",     default="BAAI/bge-large-en-v1.5",
                   help="SentenceTransformer model name")
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip databases already indexed")
    p.add_argument("--db_filter", nargs="*",
                   help="Only build for these db_ids")
    p.add_argument("--device", default="cpu",
                   help="Device for SentenceTransformer encoding: cpu, cuda, cuda:0 (default: cpu)")
    p.add_argument("--sqlite_only", action="store_true",
                   help="Skip non-SQLite databases (BigQuery/Snowflake) in spider2/lite")
    p.add_argument("--no_samples", action="store_true",
                   help="Skip SQLite sample value queries (much faster, slight accuracy tradeoff)")
    args = p.parse_args()

    build_index(
        schema_path   = args.schema,
        out_dir       = args.out_dir,
        dataset       = args.dataset,
        db_dir        = args.db_dir,
        model_name    = args.model,
        batch_size    = args.batch_size,
        skip_existing = args.skip_existing,
        db_filter     = args.db_filter,
        device        = args.device,
        sqlite_only   = args.sqlite_only,
        no_samples    = args.no_samples,
    )


if __name__ == "__main__":
    _cli()
