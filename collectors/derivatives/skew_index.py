"""
SKEW / VVIX / PUT index collector.
Source: CBOE free public CDN — no API key required.
Covers: SKEW (tail risk), VVIX (vol-of-vol), PUT (putwrite index) — full daily history.
Output:
  data/derivatives/skew_vvix_1d.parquet
  columns: skew, vvix, put_index
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

CBOE_SKEW = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
CBOE_VVIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv"
CBOE_PUT  = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv"


def _fetch_cboe_index(url: str, col_name: str) -> pd.Series:
    """Fetch a CBOE CSV index file and return a Series of close prices."""
    resp = fetch(url)
    text = resp.text
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip().upper().startswith("DATE")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No DATE header found in {url}")
    csv_body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_body))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()
    close_col = next((c for c in df.columns if "CLOSE" in c), df.columns[0])
    return pd.to_numeric(df[close_col], errors="coerce").rename(col_name)


def collect_skew_vvix() -> None:
    """Merge SKEW, VVIX, and PUT index into a single parquet."""
    skew = _fetch_cboe_index(CBOE_SKEW, "skew")
    vvix = _fetch_cboe_index(CBOE_VVIX, "vvix")
    put  = _fetch_cboe_index(CBOE_PUT,  "put_index")

    df = pd.concat([skew, vvix, put], axis=1)
    df.index.name = "date"
    df = to_datetime_index(df)
    save(df, "derivatives", "skew_vvix_1d.parquet")


def main() -> None:
    print("Fetching: SKEW, VVIX, PUT index")
    try:
        collect_skew_vvix()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
