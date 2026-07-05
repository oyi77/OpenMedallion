"""
VIX term structure collector.
Source: CBOE free public CDN — no API key required.
Covers: VIX (spot), VIX9D, VIX3M, VIX6M — full daily history.
Output:
  data/derivatives/vix_term_structure_1d.parquet
  columns: vix, vix9d, vix3m, vix6m
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

# CBOE CDN endpoints — free, no auth
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CBOE_VIX9D = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"
CBOE_VIX3M = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
CBOE_VIX6M = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv"


def _fetch_cboe_index(url: str, col_name: str) -> pd.Series:
    """Fetch a CBOE CSV index file and return a Series named col_name (close prices)."""
    resp = fetch(url)
    # CBOE CSVs have a metadata header row; skip rows until we find 'DATE'
    text = resp.text
    lines = text.splitlines()
    # Find the header line
    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip().upper().startswith("DATE")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No DATE header found in {url}")
    csv_body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_body))
    df.columns = [c.strip().upper() for c in df.columns]
    # DATE column + CLOSE
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    close_col = next((c for c in df.columns if "CLOSE" in c), df.columns[0])
    return pd.to_numeric(df[close_col], errors="coerce").rename(col_name)


def collect_vix_term_structure() -> None:
    """Merge VIX, VIX9D, VIX3M, VIX6M into a single term-structure parquet."""
    vix = _fetch_cboe_index(CBOE_VIX, "vix")
    vix9d = _fetch_cboe_index(CBOE_VIX9D, "vix9d")
    vix3m = _fetch_cboe_index(CBOE_VIX3M, "vix3m")
    vix6m = _fetch_cboe_index(CBOE_VIX6M, "vix6m")

    df = pd.concat([vix, vix9d, vix3m, vix6m], axis=1)
    df.index.name = "date"
    df = to_datetime_index(df)
    save(df, "derivatives", "vix_term_structure_1d.parquet")


def main() -> None:
    print("Fetching: VIX term structure (VIX, VIX9D, VIX3M, VIX6M)")
    try:
        collect_vix_term_structure()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
