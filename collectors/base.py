"""Shared base utilities for all OpenMedallion collectors."""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

# Root paths — override via environment variables
_script_dir = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OM_REPO_ROOT", _script_dir.parent))
DATA_ROOT = Path(os.environ.get("OM_DATA_DIR", REPO_ROOT / "data"))

# Collector defaults — override via environment
DEFAULT_COLLECTOR_TIMEOUT: int = int(os.environ.get("OM_COLLECTOR_TIMEOUT", "300"))
DEFAULT_MAX_WORKERS: int = int(os.environ.get("OM_MAX_WORKERS", "8"))
HISTORY_START: str = os.environ.get("OM_HISTORY_START", "")


def repo_path(category: str, filename: str) -> Path:
    """Return absolute output path, creating parent dirs as needed."""
    p = DATA_ROOT / category / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch(url: str, params: dict | None = None, retries: int = 5, timeout: int = 30) -> requests.Response:
    """GET with exponential backoff; 429 triggers 60s wait."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                if attempt == retries:
                    resp.raise_for_status()
                wait = 60
                LOG.warning("Attempt %d/%d failed (429 Too Many Requests) — retrying in %ds", attempt, retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            LOG.warning("Attempt %d/%d failed (%s) — retrying in %ds", attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def save(df: pd.DataFrame | None, category: str, filename: str) -> None:
    """Save DataFrame to parquet. Warns and skips if empty."""
    if df is None or df.empty:
        LOG.warning("0 rows returned for %s/%s — skipping", category, filename)
        return
    out = repo_path(category, filename)
    df.to_parquet(out, index=True)
    rel = out.relative_to(REPO_ROOT)
    print(f"Saved {len(df):,} rows to data/{rel.as_posix().removeprefix('data/')}")
