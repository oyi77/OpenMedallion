"""
US and global bond market data via FRED + yfinance.
Source: FRED (St. Louis Fed) + Yahoo Finance — no API key required
Output: data/bonds/fred/<SERIES>_1d.parquet
       data/bonds/us_treasury_yields_1d.parquet (consolidated)
Covers:
  - US Treasury yields (1Y, 2Y, 5Y, 10Y, 30Y)
  - TIPS yields / breakeven inflation
  - Yield spread (10Y-2Y, 10Y-3M)
  - Investment-grade / high-yield credit spreads
  - Global sovereign yields (DE, UK, JP, FR, IT, AU, CA)
  - Bond ETFs via yfinance
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import save, to_datetime_index

_SLEEP = 0.5


# US Treasury yields — consolidated into single parquet with date/maturity/yield_pct
_FRED_YIELDS: dict[str, str] = {
    "DGS1":  "1Y",
    "DGS2":  "2Y",
    "DGS5":  "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
}
# Non-treasury FRED series (TIPS, breakeven, global sovereign)
_FRED_SERIES: dict[str, str] = {
    # TIPS / Breakeven
    "DFII5":   "US_5Y_TIPS_Yield",
    "DFII10":  "US_10Y_TIPS_Yield",
    "T5YIE":   "US_5Y_Breakeven",
    "T10YIE":  "US_10Y_Breakeven",
    # TED spread (interbank risk — bonds context)
    "TEDRATE": "TED_Spread",
    # Global sovereign (ECB / central bank sources proxied via FRED)
    "IRLTLT01DEM156N": "DE_10Y_Yield",
    "IRLTLT01GBM156N": "UK_10Y_Yield",
    "IRLTLT01JPM156N": "JP_10Y_Yield",
    "IRLTLT01FRM156N": "FR_10Y_Yield",
    "IRLTLT01ITM156N": "IT_10Y_Yield",
    "IRLTLT01AUM156N": "AU_10Y_Yield",
    "IRLTLT01CAM156N": "CA_10Y_Yield",
    "IRLTLT01CNM156N": "CN_10Y_Yield",
    # Yield curve spreads belong here too (removed from credit_spreads which has OAS)
}

# Bond ETFs via yfinance
import yfinance as yf
_YF_BONDS: list[tuple[str, str]] = [
    ("TLT",  "TLT_20Y_Treasury"),
    ("IEF",  "IEF_7_10Y_Treasury"),
    ("SHY",  "SHY_1_3Y_Treasury"),
    ("TIP",  "TIP_TIPS"),
    ("LQD",  "LQD_IG_Corp"),
    ("HYG",  "HYG_HY_Corp"),
    ("JNK",  "JNK_HY_Corp"),
    ("EMB",  "EMB_EM_USD"),
    ("BNDX", "BNDX_Intl"),
    ("AGG",  "AGG_US_Agg"),
    ("BND",  "BND_US_Agg"),
    ("VGLT", "VGLT_LT_Treasury"),
]


def _fetch_fred_csv(series_id: str, label: str) -> pd.DataFrame | None:
    """FRED offers a free CSV export without needing an API key."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        date_col = df.columns[0]  # observation_date or DATE depending on endpoint
        df = df.rename(columns={date_col: "date", df.columns[1]: "value"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.rename(columns={"value": label.lower()})
        df = to_datetime_index(df, col="date")
        return df
    except Exception as exc:
        print(f"  WARNING FRED {series_id}: {exc}")
        return None


def _fetch_yf(ticker: str, label: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        print(f"  WARNING yf {label}: {exc}")
        return None


def _collect_treasury_yields() -> None:
    """Fetch US Treasury yields and save as consolidated long-format parquet."""
    print("  Fetching consolidated US Treasury yields...")
    frames: list[pd.DataFrame] = []
    for series_id, maturity in _FRED_YIELDS.items():
        df = _fetch_fred_csv(series_id, f"yield_{maturity}")
        if df is not None and not df.empty:
            col = df.columns[0]
            s = df[[col]].rename(columns={col: "yield_pct"})
            s["maturity"] = maturity
            frames.append(s)
        time.sleep(_SLEEP)
    if not frames:
        print("  WARNING: no Treasury yield data fetched")
        return
    merged = pd.concat(frames)
    merged.index.name = "date"
    merged = merged[["maturity", "yield_pct"]]
    save(merged, "bonds", "us_treasury_yields_1d.parquet")


def collect_bonds() -> None:
    _collect_treasury_yields()
    print(" Fetching FRED bond/yield series...")
    for series_id, label in _FRED_SERIES.items():
        df = _fetch_fred_csv(series_id, label)
        if df is not None and not df.empty:
            save(df, "bonds/fred", f"{label}_1d.parquet")
        time.sleep(_SLEEP)



def main() -> None:
    print("Fetching: Bond market data (FRED yields + yfinance ETFs)")
    collect_bonds()


if __name__ == "__main__":
    main()
