"""
AQR factor data collector.
Source: AQR Capital Management — Betting Against Beta, Quality Minus Junk,
        Value and Momentum Everywhere (free research data, no API key)
Output: data/factors/AQR_BAB_1m.parquet, AQR_QMJ_1m.parquet, AQR_VME_1m.parquet
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# AQR hosts these as Excel files — we read with openpyxl/xlrd
DATASETS = {
    "AQR_BAB_1m": (
        "https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly",
        "BAB Factors",
    ),
    "AQR_QMJ_1m": (
        "https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly",
        "QMJ Factors",
    ),
}

# Direct xlsx download URLs (AQR provides these as direct links)
DIRECT_URLS = {
    "AQR_BAB_1m": "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "AQR_QMJ_1m": "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
    "AQR_VME_1m": "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Value-and-Momentum-Everywhere-Factors-Monthly.xlsx",
}


def parse_aqr_xlsx(content: bytes, name: str) -> pd.DataFrame:
    """Parse AQR Excel factor file — data typically starts after metadata rows."""
    xls = pd.ExcelFile(io.BytesIO(content))
    # Use first sheet with 'Factor' or 'Returns' in name, else first sheet
    sheet = xls.sheet_names[0]
    for s in xls.sheet_names:
        if any(k in s for k in ["Factor", "Returns", "BAB", "QMJ", "VME"]):
            sheet = s
            break

    # Read with header detection — AQR files have 2-3 metadata rows
    df_raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None)

    # Find first row where first cell looks like a date
    start_row = 0
    for i, row in df_raw.iterrows():
        val = str(row.iloc[0])
        if any(c.isdigit() for c in val) and ("/" in val or "-" in val or len(val) == 7):
            start_row = i
            break

    header_row = start_row - 1 if start_row > 0 else 0
    df = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=header_row, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df[df.index.notna()].sort_index()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def main() -> None:
    for out_name, url in DIRECT_URLS.items():
        print(f"Fetching {out_name} ...")
        try:
            resp = fetch(url, timeout=60)
            df = parse_aqr_xlsx(resp.content, out_name)
            save(df, "factors", f"{out_name}.parquet")
        except Exception as exc:
            print(f"  WARNING: {out_name} failed — {exc}")


if __name__ == "__main__":
    main()
