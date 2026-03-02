"""
model_manager.py – process-safe SentenceTransformer manager for AutoLinkParser.

Key design
----------
- Uses a per-process dict (_process_models) instead of a classic singleton,
  so each worker process in Squrve's ProcessPoolExecutor gets its own model
  instance rather than trying to share a threading.Lock across fork boundaries.
- Within a single process, model instances are reused (one per device).
- A threading.Lock still guards the per-process dict so threads within the
  same process (Squrve's ThreadPoolExecutor) don't race on initialisation.
"""

import os
import threading
import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Per-process model cache  {device_str: SentenceTransformer}
# ---------------------------------------------------------------------------

_process_models: dict = {}        # owned by each OS process independently
_process_lock = threading.Lock()  # guards _process_models within one process


def _get_model(model_path: str, device: str) -> "SentenceTransformer":
    """Return (and lazily create) the model for (model_path, device) in this process."""
    if not _HAS_TORCH:
        raise ImportError(
            "sentence-transformers and torch are required for AutoLinkParser retrieval."
        )
    key = (model_path, device)
    with _process_lock:
        if key not in _process_models:
            import torch as _torch
            resolved_device = device
            if device.startswith("cuda") and not _torch.cuda.is_available():
                resolved_device = "cpu"
            _process_models[key] = SentenceTransformer(model_path, device=resolved_device)
    return _process_models[key]


# ---------------------------------------------------------------------------
# Public functional API  (replaces the old singleton class)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def encode(text: str, model_path: str = None, device: str = "cpu") -> "np.ndarray":
    """Encode *text* and return a numpy vector.

    Parameters
    ----------
    text : str
        Input text to embed.
    model_path : str, optional
        HuggingFace model name/path.  Defaults to ``BAAI/bge-large-en-v1.5``.
    device : str
        Torch device string (``"cpu"`` or ``"cuda:0"`` etc.).
    """
    if model_path is None:
        model_path = _DEFAULT_MODEL
    model = _get_model(model_path, device)
    with _process_lock:
        return model.encode(text, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Legacy singleton shim – keeps old import paths working
# ---------------------------------------------------------------------------

class AutoLinkModelManager:
    """Thin shim retained for backward compatibility.

    Internally delegates to the process-local functional API above,
    so it is safe under both threading and multiprocessing.
    """

    def __init__(self):
        self._model_path = _DEFAULT_MODEL
        self._device     = "cpu"

    def load_model(self, model_path: str = None, device: str = "cpu"):
        self._model_path = model_path or _DEFAULT_MODEL
        self._device     = device
        _get_model(self._model_path, self._device)   # warm up

    def encode(self, text: str) -> "np.ndarray":
        return encode(text, model_path=self._model_path, device=self._device)

    def get_device(self) -> str:
        return self._device


# module-level shim instance – existing code that does
#   from core.AutoLink.model_manager import autolink_model_manager
# continues to work unchanged.
autolink_model_manager = AutoLinkModelManager()
