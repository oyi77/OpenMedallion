"""
BIS (Bank for International Settlements) forex turnover and derivatives stats.
Source: BIS statistics portal — free CSV downloads, no API key.
Covers: FX turnover by instrument and counterparty, OTC derivatives stats.
Output: data/macro/BIS_FX_Turnover_1y.parquet
         data/macro/BIS_OTC_Derivatives_1y.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# BIS data portal direct CSV endpoints
BIS_DATASETS = {
    "BIS_FX_Turnover_1y": "https://data.bis.org/topics/TRIENNIAL_STATS/BIS,WS_TRIENNIAL_FX_TC,1.0/all?format=csv",
    "BIS_OTC_Derivatives_1y": "https://data.bis.org/topics/STATS/BIS,WS_OTC_DERIV2,1.0/all?format=csv",
    "BIS_Credit_Stats_1q": "https://data.bis.org/topics/TOTAL_CREDIT/BIS,WS_TC,1.0/all?format=csv",
    "BIS_EffectiveExchangeRates_1m": "https://data.bis.org/topics/EER/BIS,WS_EER,1.0/all?format=csv",
    "BIS_PropertyPrices_1q": "https://data.bis.org/topics/PP/BIS,WS_PP_DETAILED,1.0/all?format=csv",
    "BIS_PolicyRates_1d": "https://data.bis.org/topics/CBPOL/BIS,WS_CBPOL_D,1.0/all?format=csv",
    "BIS_LiquidAssets_1q": "https://data.bis.org/topics/GLI/BIS,WS_GLI,1.0/all?format=csv",
}


def parse_bis_csv(content: bytes) -> pd.DataFrame:
    """Parse BIS SDMX-CSV format."""
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # BIS CSVs may have key:value header metadata — skip until data block
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("KEY,") or "TIME_PERIOD" in line or (i == 0 and "," in line):
            start = i
            break

    block = "\n".join(lines[start:])
    df = pd.read_csv(io.StringIO(block), low_memory=False)

    # Find time and value columns
    time_col = next((c for c in df.columns if "TIME" in c.upper() or "PERIOD" in c.upper()), None)
    val_col = next((c for c in df.columns if c.upper() in ["OBS_VALUE", "VALUE", "OBS"]), None)

    if time_col is None or val_col is None:
        # Try to detect if first col is a key, and columns are dates
        if df.columns[0] == "KEY" or "KEY" in df.columns:
            # Wide format: rows = series, cols = dates
            key_col = df.columns[0]
            date_cols = [c for c in df.columns[1:] if any(c.startswith(str(y)) for y in range(1990, 2030))]
            if date_cols:
                sub = df[[key_col] + date_cols].set_index(key_col)
                sub.columns = pd.to_datetime(sub.columns, errors="coerce")
                sub = sub.T
                sub.index = pd.DatetimeIndex(sub.index, tz="UTC")
                sub = sub.apply(pd.to_numeric, errors="coerce")
                return sub.dropna(how="all")
        return pd.DataFrame()

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[time_col, val_col])

    # If there's a series key column, pivot
    key_col = next((c for c in df.columns if c.upper() in ["KEY", "SERIES_KEY", "SERIES"]), None)
    if key_col:
        df = df.pivot_table(index=time_col, columns=key_col, values=val_col, aggfunc="first")
        df.index.name = "date"
        df = df.sort_index()
    else:
        df = df.set_index(time_col)[[val_col]].rename(columns={val_col: "value"}).sort_index()

    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def main() -> None:
    for name, url in BIS_DATASETS.items():
        print(f"Fetching BIS {name} ...")
        try:
            resp = fetch(url, timeout=120)
            df = parse_bis_csv(resp.content)
            save(df, "macro", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {name} — {exc}")


if __name__ == "__main__":
    main()
