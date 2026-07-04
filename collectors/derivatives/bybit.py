"""
Bybit derivatives collector.
Source: Bybit public API v5 — free, no API key required.
Covers: OHLCV for top 50 perpetual pairs, funding rates, open interest.
Output:
  data/derivatives/Bybit_<SYMBOL>_1h.parquet
  data/derivatives/Bybit_<SYMBOL>_funding.parquet
  data/derivatives/Bybit_<SYMBOL>_open_interest.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

BYBIT_BASE = "https://api.bybit.com/v5"

# Top perpetuals to collect
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT", "ATOMUSDT",
    "NEARUSDT", "FTMUSDT", "AAVEUSDT", "SANDUSDT", "MANAUSDT",
    "OPUSDT", "ARBUSDT", "SUIUSDT", "APTUSDT", "INJUSDT",
    "TIAUSDT", "SEIUSDT", "WLDUSDT", "PYTHUSDT", "JUPUSDT",
]


def fetch_klines(symbol: str, interval: str = "60", limit: int = 200) -> pd.DataFrame:
    """Fetch OHLCV klines. interval: 1,3,5,15,30,60,120,240,D,W,M."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = fetch(f"{BYBIT_BASE}/market/kline", params=params)
    data = resp.json()
    if data.get("retCode") != 0 or not data["result"]["list"]:
        return pd.DataFrame()

    rows = data["result"]["list"]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_funding_history(symbol: str, limit: int = 200) -> pd.DataFrame:
    params = {"category": "linear", "symbol": symbol, "limit": limit}
    resp = fetch(f"{BYBIT_BASE}/market/funding/history", params=params)
    data = resp.json()
    if data.get("retCode") != 0 or not data["result"]["list"]:
        return pd.DataFrame()

    rows = data["result"]["list"]
    df = pd.DataFrame(rows)
    df["fundingRateTimestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(float), unit="ms", utc=True
    )
    df = df.set_index("fundingRateTimestamp").sort_index()
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["fundingRate"]].rename(columns={"fundingRate": "value"})


def fetch_open_interest(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    params = {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": interval,
        "limit": limit,
    }
    resp = fetch(f"{BYBIT_BASE}/market/open-interest", params=params)
    data = resp.json()
    if data.get("retCode") != 0 or not data["result"]["list"]:
        return pd.DataFrame()

    rows = data["result"]["list"]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    return df[["openInterest"]].rename(columns={"openInterest": "value"})


def main() -> None:
    for symbol in SYMBOLS:
        print(f"Fetching Bybit {symbol} ...")
        try:
            # OHLCV 1h
            df_ohlcv = fetch_klines(symbol, interval="60")
            if not df_ohlcv.empty:
                save(df_ohlcv, "derivatives", f"Bybit_{symbol}_1h.parquet")

            time.sleep(0.2)

            # Funding rate
            df_fund = fetch_funding_history(symbol)
            if not df_fund.empty:
                save(df_fund, "derivatives", f"Bybit_{symbol}_funding.parquet")

            time.sleep(0.2)

            # Open interest
            df_oi = fetch_open_interest(symbol)
            if not df_oi.empty:
                save(df_oi, "derivatives", f"Bybit_{symbol}_open_interest.parquet")

            time.sleep(0.3)

        except Exception as exc:
            print(f"  WARNING: {symbol} — {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main()
