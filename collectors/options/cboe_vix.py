"""
CBOE VIX term structure and volatility indices collector.
Source: CBOE direct CSV downloads — free, no API key.
Covers: VIX9D, VIX (30d), VIX3M, VIX6M, VVIX, SKEW, MOVE, OVX, GVZ.
Output: data/options/CBOE_<index>_1d.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# CBOE direct CSV data (free download, no key needed)
CBOE_INDICES = {
    "VIX9D":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "VIX3M":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "VIX6M":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
    "VVIX":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "SKEW":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    "VIX":    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "OVX":    "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",
    "GVZ":    "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv",
    "EUVIX":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/EUVIX_History.csv",
    "JYVIX":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/JYVIX_History.csv",
    "BPVIX":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/BPVIX_History.csv",
    "VXAPL":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXAPL_History.csv",
    "VXAZN":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXAZN_History.csv",
    "VXGS":   "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXGS_History.csv",
    "VXGOG":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXGOG_History.csv",
    "VXIBM":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXIBM_History.csv",
}

# VIX futures term structure (settlement prices)
VIX_FUTURES_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VX_History.csv"


def parse_cboe_csv(content: bytes, name: str) -> pd.DataFrame:
    """CBOE CSV typically: DATE, OPEN, HIGH, LOW, CLOSE or DATE, VALUE."""
    text = content.decode("utf-8", errors="replace")
    # Skip header lines that aren't the column row
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "DATE" in line.upper() or "date" in line.lower():
            start = i
            break
    block = "\n".join(lines[start:])
    df = pd.read_csv(io.StringIO(block))
    df.columns = [c.strip().upper() for c in df.columns]

    date_col = next((c for c in df.columns if "DATE" in c), None)
    if date_col is None:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    # Keep OHLC if available, else just CLOSE/VALUE
    ohlc_cols = [c for c in ["OPEN", "HIGH", "LOW", "CLOSE"] if c in df.columns]
    if ohlc_cols:
        df = df[ohlc_cols].rename(columns=str.lower)
    else:
        val_col = next((c for c in df.columns if c in ["CLOSE", "VALUE", "SETTLE"]), df.columns[0])
        df = df[[val_col]].rename(columns={val_col: "value"})

    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def main() -> None:
    for name, url in CBOE_INDICES.items():
        print(f"Fetching CBOE {name} ...")
        try:
            resp = fetch(url, timeout=30)
            df = parse_cboe_csv(resp.content, name)
            save(df, "options", f"CBOE_{name}_1d.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
