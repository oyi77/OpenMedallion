"""
Baltic Dry Index full history collector.
Sources tried in order:
  1. BDRY ETF (Breakwave Dry Bulk Shipping ETF) via yfinance — tracks BDI directly
     - 5y daily history, ~1254 data points
  2. World Bank container port traffic (IS.SHP.GOOD.TU) — annual global TEU proxy
Output: data/alternative/baltic_dry_index_1d.parquet — date, value, source
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collectors.base import fetch, save, to_datetime_index

WORLDBANK_API = "https://api.worldbank.org/v2/en/country/WLD/indicator/IS.SHP.GOOD.TU"


def _collect_bdry_yfinance() -> pd.DataFrame:
    """
    Fetch BDRY (Breakwave Dry Bulk Shipping ETF) daily price history via yfinance.
    BDRY's NAV tracks the BDI via dry bulk freight futures.
    """
    import yfinance as yf  # optional dep — may not be installed

    ticker = yf.Ticker("BDRY")
    hist = ticker.history(period="5y")
    if hist.empty:
        raise ValueError("yfinance BDRY returned empty history")

    df = hist[["Close"]].rename(columns={"Close": "value"})
    df["source"] = "BDRY_ETF"
    # yfinance returns DatetimeTZDtype index already
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "date"
    return df.dropna(subset=["value"]).sort_index()


def _collect_bdry_yahoo_direct() -> pd.DataFrame:
    """
    Fetch BDRY via Yahoo Finance query2 endpoint (no auth, different subdomain
    avoids the query1 rate-limit in some environments).
    """
    resp = fetch(
        "https://query2.finance.yahoo.com/v8/finance/chart/BDRY",
        params={"interval": "1d", "range": "5y"},
        retries=3,
        timeout=20,
    )
    data = resp.json()
    result = data["chart"]["result"]
    if not result:
        raise ValueError("Yahoo Finance (query2) BDRY returned no result")

    r = result[0]
    timestamps = r["timestamp"]
    closes = r["indicators"]["quote"][0].get("close", [])
    if not timestamps or not closes:
        raise ValueError("BDRY: empty timestamps or closes")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True),
            "value": closes,
            "source": "BDRY_ETF",
        }
    )
    df = df.dropna(subset=["value"])
    return df.set_index("date").sort_index()


def _collect_worldbank_shipping() -> pd.DataFrame:
    """
    Fetch World Bank global container port traffic (annual).
    Indicator IS.SHP.GOOD.TU — TEU units, world aggregate.
    Used as a long-run shipping volume proxy.
    """
    resp = fetch(
        WORLDBANK_API,
        params={"format": "json", "mrv": "25", "per_page": "50"},
        retries=3,
        timeout=20,
    )
    data = resp.json()
    if len(data) < 2 or not data[1]:
        raise ValueError("World Bank returned no shipping data")

    rows = [
        {
            "date": f"{item['date']}-01-01",
            "value": item["value"],
            "source": "WB_ContainerTEU",
        }
        for item in data[1]
        if item.get("value") is not None
    ]
    if not rows:
        raise ValueError("World Bank: 0 valid rows")

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return to_datetime_index(df, col="date")


def collect_bdi() -> None:
    """Collect BDI / shipping proxies with source fallback chain."""
    frames: list[pd.DataFrame] = []

    for label, fn in [
        ("BDRY ETF via yfinance", _collect_bdry_yfinance),
        ("BDRY ETF via Yahoo query2", _collect_bdry_yahoo_direct),
        ("World Bank container TEU", _collect_worldbank_shipping),
    ]:
        print(f"  Trying source: {label}")
        try:
            df = fn()
            if not df.empty:
                print(f"    Got {len(df):,} rows")
                frames.append(df)
                # One working daily source is sufficient
                if label.startswith("BDRY"):
                    break
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    # Also always try World Bank for annual context
    try:
        wb_df = _collect_worldbank_shipping()
        if not wb_df.empty:
            print(f"    World Bank TEU: {len(wb_df):,} annual rows")
            frames.append(wb_df)
    except Exception as exc:
        print(f"    WARNING: World Bank container TEU failed — {exc}")

    if not frames:
        print("  WARNING: All BDI sources failed — skipping")
        return

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    save(combined, "alternative", "baltic_dry_index_1d.parquet")


def main() -> None:
    print("Fetching: Baltic Dry Index / shipping proxies")
    try:
        collect_bdi()
    except Exception as exc:
        print(f"  WARNING: {exc}")


if __name__ == "__main__":
    main()
