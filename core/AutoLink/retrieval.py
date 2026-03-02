"""
retrieval.py – FAISS-based vector schema retrieval for AutoLinkParser.

Public API
----------
get_next_k_results(instance_id, question, db_name, embed_path,
                   top_k, cache_dir, status_dir, device)
    Retrieve the next top_k schema columns, skipping previously returned
    results tracked via cache files on disk.

Concurrency notes
-----------------
- FAISS indexes are cached per process in ``_faiss_cache`` so repeated
  retrieval calls within the same agent loop don't re-read from disk.
- Cache/status JSON writes are serialised per *file path* via
  ``_file_locks`` so concurrent threads (Squrve's ThreadPoolExecutor)
  can safely update different instance files in parallel.
- The model itself is managed by model_manager.py which uses a
  per-process dict, safe under both threading and multiprocessing.
"""

import os
import json
import re
import threading
import numpy as np
from collections import defaultdict

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

from .model_manager import encode as _encode


# ---------------------------------------------------------------------------
# Per-process FAISS index cache  {(embed_path, db_name): (index, metadata)}
# ---------------------------------------------------------------------------

_faiss_cache: dict = {}
_faiss_cache_lock = threading.Lock()


def _load_faiss(embed_path: str, db_name: str):
    """Return cached (index, metadata) for this process, loading on first call."""
    key = (embed_path, db_name)
    with _faiss_cache_lock:
        if key not in _faiss_cache:
            if not _HAS_FAISS:
                raise ImportError(
                    "faiss-cpu (or faiss-gpu) is required: pip install faiss-cpu"
                )
            index_path    = os.path.join(embed_path, db_name, "index.faiss")
            metadata_path = os.path.join(embed_path, db_name, "metadata.json")
            idx = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            _faiss_cache[key] = (idx, meta)
        return _faiss_cache[key]


# ---------------------------------------------------------------------------
# Per-file threading locks for cache/status JSON
# ---------------------------------------------------------------------------

_file_locks: dict = defaultdict(threading.Lock)
_file_locks_lock = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    with _file_locks_lock:
        return _file_locks[path]


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def _mask_digits(table_name: str) -> str:
    table_name = re.sub(r"\d", "*", table_name)
    return re.sub(r"\*+", "*", table_name)


def _sliding_window_table_match(metadata_table: str, target_table: str) -> bool:
    m_parts = metadata_table.lower().split(".")
    t_parts = target_table.lower().split(".")
    if len(t_parts) > len(m_parts):
        return False
    w = len(t_parts)
    return any(m_parts[i : i + w] == t_parts for i in range(len(m_parts) - w + 1))


# ---------------------------------------------------------------------------
# Cache / status persistence  (thread-safe via per-file locks)
# ---------------------------------------------------------------------------

def _load_cache(instance_id: str, cache_dir: str) -> dict:
    path = os.path.join(cache_dir, f"{instance_id}.json")
    lock = _get_file_lock(path)
    with lock:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"used_indices": []}


def _save_cache(instance_id: str, cache_dir: str, data: dict):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{instance_id}.json")
    data["used_indices"] = [int(i) for i in data.get("used_indices", [])]
    lock = _get_file_lock(path)
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _load_status(instance_id: str, status_dir: str) -> dict:
    path = os.path.join(status_dir, f"{instance_id}.json")
    lock = _get_file_lock(path)
    with lock:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"is_complete": False, "total_available": 0, "used_count": 0, "remaining_count": 0}


def _save_status(instance_id: str, status_dir: str, data: dict):
    os.makedirs(status_dir, exist_ok=True)
    path = os.path.join(status_dir, f"{instance_id}.json")
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    lock = _get_file_lock(path)
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# FAISS search core (private)
# ---------------------------------------------------------------------------

def _faiss_search_filtered(
    question: str,
    db_name: str,
    embed_path: str,
    excluded_indices: set,
    top_k: int,
    device: str,
    model_path: str = None,
):
    """Run FAISS nearest-neighbour search, skipping already-used indices."""
    index, metadata = _load_faiss(embed_path, db_name)

    q_vec = _encode(question, model_path=model_path, device=device)
    distances, indices = index.search(q_vec.reshape(1, -1).astype(np.float32), len(metadata))

    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        if 0 <= idx < len(metadata) and idx not in excluded_indices:
            results.append({
                "index":    idx,
                "distance": float(distances[0][i]),
                "metadata": metadata[idx],
            })
            if len(results) >= top_k:
                break

    return results, len(metadata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_next_k_results(
    instance_id: str,
    question: str,
    db_name: str,
    embed_path: str,
    top_k: int,
    cache_dir: str,
    status_dir: str,
    device: str = "cpu",
    model_path: str = None,
):
    """Retrieve the next top_k schema columns, skipping already-seen ones.

    Parameters
    ----------
    instance_id : str
        Unique key for cache files (one per sample).
    question : str
        Retrieval query (column name + description).
    db_name : str
        Database name; locates ``<embed_path>/<db_name>/``.
    embed_path : str
        Root directory containing FAISS index and metadata per database.
    top_k : int
        Number of new results to return per call.
    cache_dir : str
        Directory for per-instance used-index cache files.
    status_dir : str
        Directory for per-instance status files.
    device : str
        Torch device (``"cpu"`` or ``"cuda:N"``).
    model_path : str, optional
        SentenceTransformer model name.  Defaults to ``BAAI/bge-large-en-v1.5``.

    Returns
    -------
    results : list[dict]
        Each entry: {index, distance, metadata}.
    metadata_mapping : dict[int, dict]
        Maps result index to metadata dict.
    completion_text : str
        Non-empty when all columns have been exhausted.
    """
    cache  = _load_cache(instance_id, cache_dir)
    status = _load_status(instance_id, status_dir)

    used_indices = set(cache.get("used_indices", []))

    if status.get("is_complete", False):
        return [], {}, "All columns in this database are retrieved. There is no need to retrieve again."

    results, total = _faiss_search_filtered(
        question=question,
        db_name=db_name,
        embed_path=embed_path,
        excluded_indices=used_indices,
        top_k=top_k,
        device=device,
        model_path=model_path,
    )

    if status.get("total_available", 0) == 0:
        status["total_available"] = total

    all_used = list(used_indices) + [int(r["index"]) for r in results]
    cache["used_indices"] = all_used
    _save_cache(instance_id, cache_dir, cache)

    used_count      = len(all_used)
    remaining_count = total - used_count
    is_complete     = len(results) < top_k or remaining_count <= 0

    _save_status(instance_id, status_dir, {
        "is_complete":     is_complete,
        "total_available": int(total),
        "used_count":      int(used_count),
        "remaining_count": int(remaining_count),
    })

    metadata_mapping = {r["index"]: r["metadata"] for r in results}
    completion_text  = (
        "All columns in this database are retrieved. There is no need to retrieve again."
        if is_complete else ""
    )
    return results, metadata_mapping, completion_text
