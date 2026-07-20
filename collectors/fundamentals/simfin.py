"""
SimFin fundamentals collector.
Source: SimFin API — free tier (non-commercial), requires free API key.
Covers: Income statement, balance sheet, cash flow for US + EU companies.
Set env var SIMFIN_API_KEY or pass --api-key argument.
Output: data/fundamentals/SimFin_<ticker>_income_1q.parquet etc.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import HISTORY_START, fetch, save

SIMFIN_BASE = "https://backend.simfin.com/api/v3"

# Top 50 US tickers to collect
US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "LLY", "ABBV", "AVGO", "PFE",
    "BAC", "KO", "COST", "MRK", "CVX", "TMO", "ACN", "WMT", "MCD", "CSCO",
    "ABT", "NKE", "DHR", "ADBE", "LIN", "NEE", "TXN", "CRM", "NFLX", "AMD",
    "QCOM", "PM", "RTX", "INTU", "IBM", "AMGN", "HON", "ORCL", "CAT", "GE",
]

STATEMENT_TYPES = {
    "income": "statements/income",
    "balance": "statements/balance",
    "cashflow": "statements/cashflow",
}


def simfin_headers(api_key: str) -> dict:
    return {"Authorization": f"api-key {api_key}"}


def fetch_statement(ticker: str, stmt_type: str, api_key: str, period: str = "quarterly") -> pd.DataFrame:
    url = f"{SIMFIN_BASE}/{STATEMENT_TYPES[stmt_type]}"
    params = {"ticker": ticker, "period": period, "start": HISTORY_START or "2000-01-01"}
    headers = simfin_headers(api_key)

    resp = fetch(url, params=params)
    # SimFin returns list of dicts
    data = resp.json()
    if not data or isinstance(data, dict) and "error" in data:
        return pd.DataFrame()

    # Data may be list of statements
    if isinstance(data, list):
        rows = []
        for entry in data:
            if isinstance(entry, dict):
                row = {k: v for k, v in entry.items()}
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    # Find date column
    date_col = next((c for c in df.columns if "date" in c.lower() or "period" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.set_index(date_col).sort_index()

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("SIMFIN_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("WARNING: No SimFin API key. Set SIMFIN_API_KEY env var or pass --api-key.")
        print("         Get free key at https://app.simfin.com/settings")
        print("         Skipping SimFin collection.")
        return

    for ticker in US_TICKERS:
        print(f"Fetching SimFin {ticker} ...")
        for stmt_name in STATEMENT_TYPES:
            try:
                df = fetch_statement(ticker, stmt_name, args.api_key)
                if not df.empty:
                    save(df, "fundamentals", f"SimFin_{ticker}_{stmt_name}_1q.parquet")
            except Exception as exc:
                print(f"  WARNING: {ticker}/{stmt_name} — {exc}")


if __name__ == "__main__":
    main()
