"""
OKX OHLCV and derivatives collector.
Source: OKX public API v5 — free, no API key required.
Covers: Top 30 perp pairs OHLCV (1h/1d), funding rates, open interest.
Output: data/derivatives/OKX_<SYMBOL>_1h.parquet etc.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

OKX_BASE = "https://www.okx.com/api/v5"

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "BNB-USDT-SWAP",
    "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "ADA-USDT-SWAP", "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP", "DOT-USDT-SWAP", "MATIC-USDT-SWAP", "LTC-USDT-SWAP",
    "BCH-USDT-SWAP", "UNI-USDT-SWAP", "ATOM-USDT-SWAP", "NEAR-USDT-SWAP",
    "OP-USDT-SWAP", "ARB-USDT-SWAP", "SUI-USDT-SWAP", "APT-USDT-SWAP",
    "INJ-USDT-SWAP", "TIA-USDT-SWAP", "WLD-USDT-SWAP", "PYTH-USDT-SWAP",
    "JUP-USDT-SWAP", "FTM-USDT-SWAP", "AAVE-USDT-SWAP", "MKR-USDT-SWAP",
    "SNX-USDT-SWAP", "CRV-USDT-SWAP",
]


def clean_symbol(sym: str) -> str:
    return sym.replace("-SWAP", "").replace("-", "")


def fetch_candles(inst_id: str, bar: str = "1H", limit: int = 300) -> pd.DataFrame:
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    resp = fetch(f"{OKX_BASE}/market/candles", params=params)
    data = resp.json()
    if data.get("code") != "0" or not data.get("data"):
        return pd.DataFrame()
    rows = data["data"]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["open", "high", "low", "close", "vol"]].rename(columns={"vol": "volume"})


def fetch_funding_rate(inst_id: str, limit: int = 100) -> pd.DataFrame:
    params = {"instId": inst_id, "limit": limit}
    resp = fetch(f"{OKX_BASE}/public/funding-rate-history", params=params)
    data = resp.json()
    if data.get("code") != "0" or not data.get("data"):
        return pd.DataFrame()
    rows = data["data"]
    df = pd.DataFrame(rows)
    if "fundingTime" not in df.columns:
        return pd.DataFrame()
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(float), unit="ms", utc=True)
    df = df.set_index("fundingTime").sort_index()
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["fundingRate"]].rename(columns={"fundingRate": "value"})


def fetch_open_interest(inst_id: str, period: str = "1H", limit: int = 100) -> pd.DataFrame:
    params = {"instId": inst_id, "period": period, "limit": limit}
    resp = fetch(f"{OKX_BASE}/rubik/stat/contracts/open-interest-volume", params=params)
    data = resp.json()
    if data.get("code") != "0" or not data.get("data"):
        return pd.DataFrame()
    rows = data["data"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col].astype(float), unit="ms", utc=True)
    df = df.set_index(ts_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def main() -> None:
    for sym in SYMBOLS:
        clean = clean_symbol(sym)
        print(f"Fetching OKX {sym} ...")
        try:
            df_1h = fetch_candles(sym, bar="1H")
            if not df_1h.empty:
                save(df_1h, "derivatives", f"OKX_{clean}_1h.parquet")
            time.sleep(0.15)

            df_fund = fetch_funding_rate(sym)
            if not df_fund.empty:
                save(df_fund, "derivatives", f"OKX_{clean}_funding.parquet")
            time.sleep(0.15)

            df_oi = fetch_open_interest(sym)
            if not df_oi.empty:
                save(df_oi, "derivatives", f"OKX_{clean}_open_interest.parquet")
            time.sleep(0.2)

        except Exception as exc:
            print(f"  WARNING: {sym} — {exc}")
            time.sleep(1)


if __name__ == "__main__":
    main()
