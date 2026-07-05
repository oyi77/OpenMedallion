"""
Additional CBOE volatility indices collector.
Source: CBOE free public CDN — no API key required.
Covers: GVZ (Gold VIX), OVX (Oil VIX), VXEEM (EM VIX), RVX (Russell 2000 VIX),
        VXAZN, VXAPL, VXGS (individual stock vol indices).
Output:
  data/derivatives/cboe_vol_indices_1d.parquet
  columns: gvz, ovx, vxeem, rvx, vxazn, vxapl, vxgs
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

CBOE_INDICES: list[tuple[str, str]] = [
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv",   "gvz"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",   "ovx"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/VXEEM_History.csv", "vxeem"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/RVX_History.csv",   "rvx"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/VXAZN_History.csv", "vxazn"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/VXAPL_History.csv", "vxapl"),
    ("https://cdn.cboe.com/api/global/us_indices/daily_prices/VXGS_History.csv",  "vxgs"),
]


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


def collect_cboe_vol_indices() -> None:
    """Fetch all CBOE vol indices and merge into a single parquet."""
    series_list: list[pd.Series] = []
    for url, col_name in CBOE_INDICES:
        try:
            print(f"  Fetching {col_name.upper()}...")
            s = _fetch_cboe_index(url, col_name)
            series_list.append(s)
        except Exception as exc:
            print(f"  WARNING: {col_name.upper()} unavailable — {exc}")

    if not series_list:
        raise RuntimeError("All CBOE vol index sources failed")

    df = pd.concat(series_list, axis=1)
    df.index.name = "date"
    df = to_datetime_index(df)
    save(df, "derivatives", "cboe_vol_indices_1d.parquet")


def main() -> None:
    print("Fetching: CBOE vol indices (GVZ, OVX, VXEEM, RVX, VXAZN, VXAPL, VXGS)")
    try:
        collect_cboe_vol_indices()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
