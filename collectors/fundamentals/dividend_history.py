"""
Dividend history collector for top S&P 500 stocks.
Source: Yahoo Finance via yfinance (no API key required)
Covers: Top 100 S&P 500 stocks by market cap
Output: data/fundamentals/dividend_history.parquet
Columns: date (DatetimeIndex), ticker, dividend_amount, ex_date, pay_date
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import save

_SLEEP = 0.25

# Top 100 S&P 500 stocks by market cap (as of mid-2025)
TOP_100_SP500 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK.B", "AVGO",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "COST", "HD",
    "ABBV", "MRK", "CRM", "BAC", "AMD", "NFLX", "CVX", "KO", "PEP", "TMO",
    "LIN", "WMT", "CSCO", "ACN", "ADBE", "MCD", "ABT", "DHR", "ORCL", "TXN",
    "PM", "NEE", "INTU", "IBM", "AMGN", "QCOM", "GE", "CAT", "NOW", "ISRG",
    "AMAT", "GS", "BLK", "LOW", "INTC", "PFE", "RTX", "UBER", "SPGI", "DE",
    "PLD", "LMT", "ELV", "MDLZ", "ADI", "GILD", "SYK", "CB", "MMC", "ADP",
    "CI", "REGN", "SBUX", "BDX", "FI", "VRTX", "CME", "SCHW", "MO", "ZTS",
    "CL", "SO", "DUK", "EOG", "SNPS", "ITW", "PGR", "ICE", "USB", "COP",
    "MPC", "SLB", "EQIX", "AON", "NXPI", "CDNS", "PANW", "SHW", "HUM", "MCK",
]


def fetch_dividends(ticker: str) -> pd.DataFrame | None:
    """Fetch dividend history for a single ticker."""
    try:
        tk = yf.Ticker(ticker)
        divs = tk.dividends
        if divs.empty:
            return None
        df = divs.reset_index()
        df.columns = ["ex_date", "dividend_amount"]
        df["ticker"] = ticker
        df["ex_date"] = pd.to_datetime(df["ex_date"], utc=True)
        df["date"] = df["ex_date"]
        df["pay_date"] = pd.NaT
        return df[["date", "ticker", "dividend_amount", "ex_date", "pay_date"]]
    except Exception as exc:
        print(f"  WARNING {ticker}: {exc}")
        return None


def collect_all() -> None:
    """Fetch dividend history for all tickers and save combined parquet."""
    print(f"Fetching dividend history for {len(TOP_100_SP500)} S&P 500 stocks...")
    frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(TOP_100_SP500, 1):
        print(f"  [{i:3d}/{len(TOP_100_SP500)}] {ticker}", end="")
        df = fetch_dividends(ticker)
        if df is not None and not df.empty:
            frames.append(df)
            print(f" — {len(df)} dividends")
        else:
            print(" — no data")
        time.sleep(_SLEEP)

    if not frames:
        print("WARNING: No dividend data collected.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    save(combined, "fundamentals", "dividend_history.parquet")


def main() -> None:
    collect_all()
    print("\nDone.")


if __name__ == "__main__":
    main()
