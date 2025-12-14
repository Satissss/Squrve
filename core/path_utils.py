"""
Path Utilities for Squrve Framework

This module provides unified path resolution functionality that supports:
- Environment variable SQURVE_ROOT for project root
- Automatic project root detection
- Relative path resolution from project root
"""

import os
from pathlib import Path
from typing import Union, Optional
from loguru import logger

# Global cache for project root
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Get the project root directory (Squrve root).
    
    Priority:
    1. Environment variable SQURVE_ROOT (if set)
    2. Cached project root (if already determined)
    3. Auto-detect from current file location
    
    Returns:
        Path: Absolute path to project root directory
    """
    global _PROJECT_ROOT
    
    # Check environment variable first
    env_root = os.getenv('SQURVE_ROOT')
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists() and env_path.is_dir():
            _PROJECT_ROOT = env_path
            logger.debug(f"Using SQURVE_ROOT environment variable: {_PROJECT_ROOT}")
            return _PROJECT_ROOT
        else:
            logger.warning(f"SQURVE_ROOT environment variable set to invalid path: {env_root}")
    
    # Use cached value if available
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    
    # Auto-detect: assume this file is in core/path_utils.py
    current_file = Path(__file__).resolve()
    # core/path_utils.py -> project root
    detected_root = current_file.parent.parent
    
    # Verify it's the project root by checking for key files/directories
    if (detected_root / "core").exists() and (detected_root / "config").exists():
        _PROJECT_ROOT = detected_root
        logger.debug(f"Auto-detected project root: {_PROJECT_ROOT}")
        return _PROJECT_ROOT
    
    # Fallback: try to find project root by looking for core directory
    current = current_file.parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "core").exists() and (current / "config").exists():
            _PROJECT_ROOT = current
            logger.debug(f"Found project root by searching: {_PROJECT_ROOT}")
            return _PROJECT_ROOT
        current = current.parent
    
    # Last resort: use detected root anyway
    _PROJECT_ROOT = detected_root
    logger.warning(f"Could not verify project root, using: {_PROJECT_ROOT}")
    return _PROJECT_ROOT
