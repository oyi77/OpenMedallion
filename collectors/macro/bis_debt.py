"""
BIS credit gap and total credit statistics collector.
Source: BIS Statistics Portal — free CSV downloads, no API key.
Covers: Credit-to-GDP gaps, total credit to private non-financial sector.
Output: data/macro/bis_credit_gap_1q.parquet
        data/macro/bis_total_credit_1q.parquet
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save

BIS_BASE = "https://stats.bis.org/api/v2/data/dataflow/BIS"

# Credit-to-GDP gap (WS_CREDIT_GAP dataflow)
CREDIT_GAP_URL = f"{BIS_BASE}/WS_CREDIT_GAP/1.0/all?format=csv"

# Total credit to private non-financial sector (WS_TC dataflow)
TOTAL_CREDIT_URL = f"{BIS_BASE}/WS_TC/1.0/all?format=csv"

# Fallback: data.bis.org topics endpoint
FALLBACK_CREDIT_GAP = "https://data.bis.org/topics/CREDIT_GAP/BIS,WS_CREDIT_GAP,1.0/all?format=csv"
FALLBACK_TOTAL_CREDIT = "https://data.bis.org/topics/TOTAL_CREDIT/BIS,WS_TC,1.0/all?format=csv"


def parse_bis_csv(content: bytes) -> pd.DataFrame:
    """
    Parse BIS SDMX-CSV.

    BIS CSVs come in two layouts:
    1. Long: KEY, FREQ, ..., TIME_PERIOD, OBS_VALUE columns
    2. Wide: KEY column + date columns as headers
    """
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Skip BIS metadata comment block (lines starting with '#' or blank)
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            start = i
            break

    block = "\n".join(lines[start:])
    if not block.strip():
        return pd.DataFrame()

    sep = ";" if lines[start].count(";") > lines[start].count(",") else ","
    df = pd.read_csv(io.StringIO(block), sep=sep, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # --- Long format: KEY + TIME_PERIOD + OBS_VALUE ---
    time_col = next(
        (c for c in df.columns if "TIME_PERIOD" in c.upper() or c.upper() == "TIME"),
        None,
    )
    val_col = next(
        (c for c in df.columns if c.upper() in ("OBS_VALUE", "VALUE", "OBS")),
        None,
    )
    key_col = next(
        (c for c in df.columns if c.upper() in ("KEY", "SERIES_KEY")),
        None,
    )

    if time_col and val_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[time_col, val_col])
        if key_col and key_col in df.columns:
            pivot = df.pivot_table(
                index=time_col, columns=key_col, values=val_col, aggfunc="first"
            ).sort_index()
            pivot.index.name = "date"
            return pivot.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return (
            df[[time_col, val_col]]
            .rename(columns={time_col: "date", val_col: "value"})
            .set_index("date")
            .sort_index()
        )

    # --- Wide format: KEY column + date columns ---
    if key_col or (df.columns[0].upper() in ("KEY", "SERIES_KEY")):
        id_col = key_col or df.columns[0]
        date_cols = [
            c for c in df.columns[1:]
            if any(c.startswith(str(y)) for y in range(1980, 2030))
        ]
        if date_cols:
            sub = df.set_index(id_col)[date_cols]
            sub.columns = pd.to_datetime(sub.columns, errors="coerce")
            sub = sub.T
            sub.index = pd.DatetimeIndex(sub.index, tz="UTC")
            return sub.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    return pd.DataFrame()


def fetch_with_fallback(primary_url: str, fallback_url: str, label: str) -> pd.DataFrame:
    """Try primary URL, fall back to secondary on failure."""
    for url in (primary_url, fallback_url):
        try:
            resp = fetch(url, timeout=120)
            df = parse_bis_csv(resp.content)
            if not df.empty:
                return df
            print(f"  {url} returned empty data")
        except Exception as exc:
            print(f"  WARNING: {label} from {url} — {exc}")
    return pd.DataFrame()


def main() -> None:
    # --- Credit-to-GDP gap ---
    print("Fetching BIS credit-to-GDP gap ...")
    df_gap = fetch_with_fallback(CREDIT_GAP_URL, FALLBACK_CREDIT_GAP, "credit_gap")
    if df_gap.empty:
        print("  WARNING: credit gap data unavailable — skipping bis_credit_gap_1q.parquet")
    else:
        print(f"  {len(df_gap):,} rows")
        save(df_gap, "macro", "bis_credit_gap_1q.parquet")

    # --- Total credit to private non-financial sector ---
    print("Fetching BIS total credit to private sector ...")
    df_tc = fetch_with_fallback(TOTAL_CREDIT_URL, FALLBACK_TOTAL_CREDIT, "total_credit")
    if df_tc.empty:
        print("  WARNING: total credit data unavailable — skipping bis_total_credit_1q.parquet")
    else:
        print(f"  {len(df_tc):,} rows")
        save(df_tc, "macro", "bis_total_credit_1q.parquet")


if __name__ == "__main__":
    main()
