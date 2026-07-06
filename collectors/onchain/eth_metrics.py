"""
Ethereum & multi-chain on-chain metrics collector.
Sources:
  - yfinance  — OHLCV price/volume (no key)
  - blockchain.com charts API — BTC on-chain (tx count, hash rate, fees, etc.)
  - DeFiLlama /v2/chains — TVL per chain (no key)
  - DeFiLlama /v2/historicalChainTvl — historical TVL per chain
Output: data/onchain/<symbol>_<metric>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

# ── yfinance tickers for major L1/L2 tokens ──────────────────────────────────
YF_COINS: dict[str, str] = {
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "LINK": "LINK-USD",
    "ATOM": "ATOM-USD",
    "MATIC": "MATIC-USD",
    "UNI": "UNI-USD",
    "NEAR": "NEAR-USD",
    "ARB": "ARB-USD",
    "OP": "OP-USD",
}

# ── blockchain.com chart API (BTC on-chain, fully free) ────────────────────
BLOCKCHAIN_BASE = "https://api.blockchain.info/charts/{chart}"
BLOCKCHAIN_CHARTS: dict[str, str] = {
    "BTC_TxCount_1d": "n-transactions",
    "BTC_TxFees_1d": "transaction-fees",
    "BTC_HashRate_1d": "hash-rate",
    "BTC_Difficulty_1d": "difficulty",
    "BTC_UTXOCount_1d": "utxo-count",
    "BTC_BlockSize_1d": "avg-block-size",
    "BTC_MempoolSize_1d": "mempool-size",
    "BTC_TxPerBlock_1d": "n-transactions-per-block",
    "BTC_MarketCap_1d": "market-cap",
    "BTC_TotalFees_1d": "miners-revenue",
}

# ── DeFiLlama ─────────────────────────────────────────────────────────────────
DEFILLAMA_CHAINS = "https://api.llama.fi/v2/chains"
DEFILLAMA_CHAIN_TVL = "https://api.llama.fi/v2/historicalChainTvl/{chain}"

TRACKED_CHAINS = ["Ethereum", "BSC", "Solana", "Arbitrum", "Optimism",
                  "Polygon", "Avalanche", "Base", "Tron", "Sui", "Aptos"]


# ── helpers ──────────────────────────────────────────────────────────────────

def fetch_yfinance_ohlcv(symbol: str, ticker: str, period: str = "max") -> None:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        print(f"  SKIP {ticker} — empty")
        return
    if hasattr(df.index, 'tz') and df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    else:
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    # yfinance ≥ 0.2 returns MultiIndex columns — flatten to level-0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() if isinstance(c, str) else str(c[0]).lower() for c in df.columns]
    df = df[["open", "close", "high", "low", "volume"]].dropna(how="all")
    save(df, "onchain", f"{symbol}_OHLCV_1d.parquet")
    print(f"  OK {symbol} — {len(df)} rows")


def fetch_blockchain_chart(name: str, chart: str) -> None:
    url = BLOCKCHAIN_BASE.format(chart=chart)
    resp = fetch(url, params={"timespan": "all", "format": "json", "sampled": "true"}, timeout=30)
    data = resp.json()
    values = data.get("values", [])
    if not values:
        print(f"  SKIP {name} — no data")
        return
    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.set_index("date")[["y"]].rename(columns={"y": "value"}).sort_index()
    save(df, "onchain", f"{name}.parquet")
    print(f"  OK {name} — {len(df)} rows")


def fetch_defillama_chain_tvl(chain: str) -> None:
    url = DEFILLAMA_CHAIN_TVL.format(chain=chain)
    try:
        resp = fetch(url, timeout=20)
        records = resp.json()
        if not isinstance(records, list) or not records:
            return
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
        df = df.set_index("date")[["tvl"]].rename(
            columns={"tvl": "tvl_usd"}
        ).sort_index()
        name = chain.replace(" ", "_")
        save(df, "onchain", f"{name}_TVL_1d.parquet")
        print(f"  OK {chain} TVL — {len(df)} rows")
    except Exception as exc:
        print(f"  WARN {chain} TVL — {exc}")


def main() -> None:
    # 1. yfinance OHLCV for major L1/L2
    print("=== yfinance crypto OHLCV ===")
    for symbol, ticker in YF_COINS.items():
        try:
            fetch_yfinance_ohlcv(symbol, ticker)
        except Exception as exc:
            print(f"  WARN {symbol} — {exc}")

    # 2. BTC on-chain via blockchain.com
    print("\n=== BTC on-chain (blockchain.com) ===")
    for name, chart in BLOCKCHAIN_CHARTS.items():
        try:
            fetch_blockchain_chart(name, chart)
            time.sleep(0.5)
        except Exception as exc:
            print(f"  WARN {name} — {exc}")

    # 3. DeFiLlama chain TVL history
    print("\n=== DeFiLlama chain TVL ===")
    for chain in TRACKED_CHAINS:
        fetch_defillama_chain_tvl(chain)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
