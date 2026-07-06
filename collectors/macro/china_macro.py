"""
China macro data collector via FRED CSV (no API key) and yfinance.

FRED series collected (all verified 200 OK):
  CHNGDPNQDSMEI         China GDP quarterly (CNY, nominal)
  CHNCPIALLMINMEI       China CPI all items (monthly index)
  MKTGDPCNA646NWDB      China GDP nominal annual (USD current prices)
  CCUSMA02CNM618N       China consumer price index monthly (proxy for PPI trend)
  MYAGM2CNM189N         China M2 money supply (CNY, monthly)
  XTEXVA01CNM659S       China exports value index (monthly)
  XTIMVA01CNM659S       China imports value index (monthly)
  DEXCHUS               USD/CNY daily spot exchange rate (NEER proxy)

Note: FRED does not publish a free China urban unemployment rate or NEER series;
      DEXCHUS is the closest available substitute for NEER.

Spec IDs that do not exist on FRED (404) and their working replacements:
  CHNMAINLANDCPIALLMINMEI  -> CHNCPIALLMINMEI
  CHNTOTALSAGDPNA          -> MKTGDPCNA646NWDB
  CHNMAINLANDPPIINMEIMEI   -> CCUSMA02CNM618N  (CPI proxy)
  CHNTMCNM052S             -> MYAGM2CNM189N
  CHNUR                    -> (omitted, no free FRED series)
  CHNNEER                  -> DEXCHUS  (USD/CNY spot)

yfinance tickers:
  000001.SS  SSE Composite Index  -> data/macro/china/SSE_Composite_1d.parquet
  399001.SZ  Shenzhen Component   -> data/macro/china/SZ_Component_1d.parquet
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

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
SLEEP_BETWEEN = 0.4  # seconds between requests — polite crawling

# (fred_series_id, output_label, freq_suffix)
# All IDs verified 200 OK against live FRED endpoint.
FRED_SERIES: list[tuple[str, str, str]] = [
    ("CHNGDPNQDSMEI",    "GDP_QoQ",    "1q"),   # China GDP quarterly, CNY nominal
    ("CHNCPIALLMINMEI",  "CPI",        "1m"),   # China CPI all items
    ("MKTGDPCNA646NWDB", "GDP_Annual", "1y"),   # China GDP nominal annual, USD
    ("CCUSMA02CNM618N",  "PPI_proxy",  "1m"),   # Consumer prices (PPI proxy)
    ("MYAGM2CNM189N",    "M2",         "1m"),   # China M2 money supply
    ("XTEXVA01CNM659S",  "Exports",    "1m"),   # China exports value index
    ("XTIMVA01CNM659S",  "Imports",    "1m"),   # China imports value index
    ("DEXCHUS",          "NEER_proxy", "1d"),   # USD/CNY spot (NEER proxy)
]

# (yfinance_ticker, output_filename_stem)
YF_TICKERS: list[tuple[str, str]] = [
    ("000001.SS", "SSE_Composite_1d"),
    ("399001.SZ", "SZ_Component_1d"),
]


def fetch_fred(series_id: str) -> pd.DataFrame:
    """Download a FRED CSV series as a tidy DataFrame with UTC DatetimeIndex.

    FRED CSV format: 'observation_date,<SERIES_ID>' with no missing-value marker
    other than the literal string '.' in the value column.
    """
    url = FRED_CSV.format(id=series_id)
    resp = fetch(url)

    # Detect header column name — FRED uses 'observation_date', not 'DATE'
    raw = StringIO(resp.text)
    header_line = resp.text.split("\n", 1)[0].strip()
    date_col = header_line.split(",")[0]  # e.g. 'observation_date'

    df = pd.read_csv(StringIO(resp.text), parse_dates=[date_col], index_col=date_col)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "date"
    # Normalize to a single 'value' column regardless of original column name
    df.columns = ["value"]
    # FRED encodes missing observations as "."
    df = df.replace(".", pd.NA).apply(pd.to_numeric, errors="coerce").dropna()
    return df.sort_index()


def fetch_yf(ticker: str) -> pd.DataFrame:
    """Download full daily history for a yfinance ticker."""
    raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    close = raw[["Close"]].copy()
    close.columns = ["value"]
    close = to_datetime_index(close)
    return close.dropna()


def main() -> None:
    ok_fred = 0
    fail_fred: list[str] = []

    print("=== China macro — FRED series ===")
    for series_id, label, freq in FRED_SERIES:
        filename = f"{label}_{freq}.parquet"
        try:
            df = fetch_fred(series_id)
            save(df, "macro/china", filename)
            ok_fred += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {series_id} ({label}): {exc}")
            fail_fred.append(series_id)
        time.sleep(SLEEP_BETWEEN)

    ok_yf = 0
    fail_yf: list[str] = []

    print("\n=== China indices — yfinance ===")
    for ticker, stem in YF_TICKERS:
        filename = f"{stem}.parquet"
        try:
            df = fetch_yf(ticker)
            save(df, "macro/china", filename)
            ok_yf += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {ticker}: {exc}")
            fail_yf.append(ticker)
        time.sleep(SLEEP_BETWEEN)

    total_ok = ok_fred + ok_yf
    total = len(FRED_SERIES) + len(YF_TICKERS)
    status = "ok" if not fail_fred and not fail_yf else "partial"
    print(f"\n[{status}] {total_ok}/{total} datasets saved")
    if fail_fred:
        print(f"  FRED failures: {', '.join(fail_fred)}")
    if fail_yf:
        print(f"  yfinance failures: {', '.join(fail_yf)}")


if __name__ == "__main__":
    main()
