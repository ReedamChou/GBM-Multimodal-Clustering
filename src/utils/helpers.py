"""
Shared helpers — configuration loading, logging, path resolution, I/O.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


# ── Project root ────────────────────────────────────────────────────────────
def get_project_root() -> Path:
    """Return the project root (parent of *src/*)."""
    return Path(__file__).resolve().parents[2]


# ── Configuration ───────────────────────────────────────────────────────────
def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dict."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    return (get_project_root() / relative).resolve()


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Logging ─────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project-wide logger."""
    logger = logging.getLogger("gbm_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── I/O convenience ────────────────────────────────────────────────────────
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
