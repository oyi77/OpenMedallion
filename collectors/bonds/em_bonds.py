"""
Emerging market bond yields, spreads, and ETFs.

Sources:
  - FRED CSV (no API key required)
  - yfinance (EM bond ETF proxies)

FRED series:
  Daily spreads/yields (ICE BofAML indices):
    BAMLHE00EHY0EY    — EM High Yield spread
    BAMLHE00EHY0EYS   — EM High Yield total return index
    BAMLEMHBHYCRPIEY  — EM Corporate Bond Index yield
    BAMLEMCBPIOAS     — EM Corporate Bond OAS spread
  Monthly gov bond yields (OECD/IMF via FRED):
    IRLTLT01IDM156N   — Indonesia 10Y gov bond yield
    IRLTLT01BRM156N   — Brazil 10Y gov bond yield
    IRLTLT01INM156N   — India 10Y gov bond yield
    IRLTLT01MYM156N   — Malaysia 10Y gov bond yield
    IRLTLT01THM156N   — Thailand 10Y gov bond yield

yfinance ETFs:
  EMB   — iShares JP Morgan EM Bond ETF
  VWOB  — Vanguard EM Gov Bond ETF
  EMLC  — VanEck Local Currency EM Bond ETF

Output:
  data/bonds/em/<label>_<freq>.parquet
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

_SLEEP = 0.5

# FRED daily series: series_id -> (column_name, freq_suffix)
_FRED_DAILY: dict[str, tuple[str, str]] = {
    "BAMLHE00EHY0EY":   ("em_hy_spread",        "1d"),
    "BAMLHE00EHY0EYS":  ("em_hy_total_return",   "1d"),
    "BAMLEMHBHYCRPIEY": ("em_corp_yield",         "1d"),
    "BAMLEMCBPIOAS":    ("em_corp_oas_spread",    "1d"),
}

# FRED monthly series: series_id -> (column_name, label_stem)
_FRED_MONTHLY: dict[str, tuple[str, str]] = {
    "IRLTLT01IDM156N": ("yield_pct", "INDONESIA_10Y"),
    "IRLTLT01BRM156N": ("yield_pct", "BRAZIL_10Y"),
    "IRLTLT01INM156N": ("yield_pct", "INDIA_10Y"),
    "IRLTLT01MYM156N": ("yield_pct", "MALAYSIA_10Y"),
    "IRLTLT01THM156N": ("yield_pct", "THAILAND_10Y"),
}

# yfinance EM bond ETFs: ticker -> label stem
_YF_ETFS: dict[str, str] = {
    "EMB":  "EMB",
    "VWOB": "VWOB",
    "EMLC": "EMLC",
}


def _fetch_fred_csv(series_id: str, col_name: str) -> pd.DataFrame | None:
    """Fetch a FRED series via the free CSV endpoint (no API key)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = fetch(url)
        df = pd.read_csv(StringIO(resp.text))
        date_col = df.columns[0]
        val_col = df.columns[1]
        df = df.rename(columns={date_col: "date", val_col: col_name})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df[col_name] != "."].copy()
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=[col_name])
        df = to_datetime_index(df, col="date")
        return df
    except Exception as exc:
        print(f"  WARNING FRED {series_id}: {exc}")
        return None


def _fetch_yf_etf(ticker: str) -> pd.DataFrame | None:
    """Fetch EM bond ETF close price history via yfinance."""
    try:
        raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  WARNING yfinance {ticker}: empty result")
            return None
        # Flatten MultiIndex columns if present (yfinance >=0.2)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        df = close.rename("close").to_frame()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        return df.sort_index()
    except Exception as exc:
        print(f"  WARNING yfinance {ticker}: {exc}")
        return None


def collect_fred_daily() -> None:
    print("=== EM Bond Spreads/Yields — FRED daily ===")
    for series_id, (col_name, freq) in _FRED_DAILY.items():
        df = _fetch_fred_csv(series_id, col_name)
        if df is not None and not df.empty:
            save(df, "bonds/em", f"{series_id}_{freq}.parquet")
        time.sleep(_SLEEP)


def collect_fred_monthly() -> None:
    print("=== EM Gov Bond Yields — FRED monthly ===")
    for series_id, (col_name, label) in _FRED_MONTHLY.items():
        df = _fetch_fred_csv(series_id, col_name)
        if df is not None and not df.empty:
            save(df, "bonds/em", f"{label}_1m.parquet")
        time.sleep(_SLEEP)


def collect_etfs() -> None:
    print("=== EM Bond ETFs — yfinance ===")
    for ticker, label in _YF_ETFS.items():
        df = _fetch_yf_etf(ticker)
        if df is not None and not df.empty:
            save(df, "bonds/em", f"ETF_{label}_1d.parquet")
        time.sleep(_SLEEP)


def main() -> None:
    collect_fred_daily()
    collect_fred_monthly()
    collect_etfs()


if __name__ == "__main__":
    main()
