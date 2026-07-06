"""
FinTS data loader.

Streams OHLCV parquet files per asset class without loading everything into RAM.
Yields (asset_id, DataFrame) tuples with standardized columns.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Generator, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# Asset classes we care about for FinTS (OHLCV-bearing directories)
ASSET_CLASS_GLOBS: dict[str, list[str]] = {
    "crypto": [
        "crypto/*.parquet",
        "crypto/**/*.parquet",
    ],
    "equities": [
        "equities/*.parquet",
        "equities/**/*.parquet",
    ],
    "forex": [
        "forex/*.parquet",
        "forex/**/*.parquet",
    ],
    "commodities": [
        "commodities/*.parquet",
        "commodities/**/*.parquet",
    ],
    "indices": [
        "indices/*.parquet",
        "indices/**/*.parquet",
    ],
    "etfs": [
        "etfs/*.parquet",
    ],
    "bonds": [
        "bonds/*.parquet",
        "bonds/**/*.parquet",
    ],
}

# Minimum rows to include a file (filters out near-empty files)
MIN_ROWS = 100

# Required OHLCV columns (close is mandatory; volume is optional)
REQUIRED_COLS = {"close"}
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AssetRecord:
    asset_id: str          # e.g. "crypto/COIN_bitcoin_1d"
    asset_class: str       # e.g. "crypto"
    source_file: str
    df: pd.DataFrame       # standardized: DatetimeIndex, columns ⊇ {close}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_glob(pattern: str) -> list[str]:
    full = os.path.join(DATA_ROOT, pattern)
    return sorted(glob.glob(full, recursive=True))


def _standardize(df: pd.DataFrame, filepath: str) -> Optional[pd.DataFrame]:
    """
    Normalize a raw parquet into a DataFrame with:
    - DatetimeIndex (UTC, tz-aware)
    - lowercase column names
    - at least a 'close' column
    Returns None if the file doesn't qualify.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Some files store date as a column instead of index
    for date_col in ("date", "datetime", "timestamp", "time"):
        if date_col in df.columns:
            df = df.set_index(date_col)
            break

    if df.index.name is None or not hasattr(df.index, "dtype"):
        return None

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            return None

    # Normalize to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "date"
    df = df.sort_index()

    # Must have close
    if "close" not in df.columns:
        # Some files use 'value' as the only price column (ECB FX rates etc.)
        if "value" in df.columns:
            df = df.rename(columns={"value": "close"})
        else:
            return None

    # Keep only OHLCV columns that exist
    keep = [c for c in OHLCV_COLS if c in df.columns]
    df = df[keep].dropna(subset=["close"])

    if len(df) < MIN_ROWS:
        return None

    return df


def _asset_id(asset_class: str, filepath: str) -> str:
    rel = os.path.relpath(filepath, DATA_ROOT)
    # strip .parquet
    return rel.replace(".parquet", "").replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_assets(
    asset_classes: Optional[list[str]] = None,
    max_per_class: Optional[int] = None,
) -> Generator[AssetRecord, None, None]:
    """
    Yield AssetRecord one at a time — no bulk RAM load.

    Parameters
    ----------
    asset_classes : list of str, optional
        Subset of ASSET_CLASS_GLOBS keys. None = all.
    max_per_class : int, optional
        Cap files per asset class (useful for quick smoke tests).
    """
    classes = asset_classes or list(ASSET_CLASS_GLOBS.keys())

    for ac in classes:
        patterns = ASSET_CLASS_GLOBS.get(ac, [])
        seen: set[str] = set()
        count = 0

        for pattern in patterns:
            for fp in _resolve_glob(pattern):
                if fp in seen:
                    continue
                seen.add(fp)

                if max_per_class is not None and count >= max_per_class:
                    break

                try:
                    raw = pd.read_parquet(fp)
                except Exception:
                    continue

                df = _standardize(raw, fp)
                if df is None:
                    continue

                count += 1
                yield AssetRecord(
                    asset_id=_asset_id(ac, fp),
                    asset_class=ac,
                    source_file=fp,
                    df=df,
                )


def load_asset_class(
    asset_class: str,
    max_files: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all assets for one class into a dict.
    Only use when the class fits in RAM (e.g. crypto ~59 files).
    """
    return {
        rec.asset_id: rec.df
        for rec in iter_assets([asset_class], max_per_class=max_files)
    }
