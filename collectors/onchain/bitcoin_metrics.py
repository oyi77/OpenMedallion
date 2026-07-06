"""
Bitcoin on-chain metrics collector.
Sources:
  - blockchain.com charts API — BTC on-chain series (no key)
  - CoinGecko /coins/bitcoin — market/community data (no key)
Output: data/onchain/BTC_<metric>_1d.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

BLOCKCHAIN_BASE = "https://api.blockchain.info/charts/{chart}"

# Only verified-working chart slugs
BLOCKCHAIN_CHARTS: dict[str, str] = {
    "BTC_ActiveAddresses_1d": "n-unique-addresses",
    "BTC_TotalBTC_1d": "total-bitcoins",
    "BTC_ExchangeVolume_1d": "trade-volume",
    "BTC_EstimatedTxVol_1d": "estimated-transaction-volume",
    "BTC_CostPerTx_1d": "cost-per-transaction",
    "BTC_OutputValue_1d": "output-volume",
    "BTC_MedianConfTime_1d": "median-confirmation-time",
    "BTC_Mempool_Bytes_1d": "mempool-size",
    "BTC_MinersRevenue_1d": "miners-revenue",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_blockchain_chart(name: str, chart: str) -> None:
    url = BLOCKCHAIN_BASE.format(chart=chart)
    try:
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
    except Exception as exc:
        print(f"  WARN {name} — {exc}")


def fetch_coingecko_market_chart() -> None:
    """BTC price, market cap, volume history from CoinGecko."""
    url = f"{COINGECKO_BASE}/coins/bitcoin/market_chart"
    try:
        resp = fetch(url, params={"vs_currency": "usd", "days": "max", "interval": "daily"}, timeout=30)
        data = resp.json()

        def _to_df(series: list, col: str) -> pd.DataFrame:
            df = pd.DataFrame(series, columns=["ts", col])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("date")[[col]].sort_index()

        prices = _to_df(data.get("prices", []), "price_usd")
        caps = _to_df(data.get("market_caps", []), "market_cap_usd")
        vols = _to_df(data.get("total_volumes", []), "volume_usd")

        combined = prices.join(caps, how="outer").join(vols, how="outer")
        save(combined, "onchain", "BTC_CoinGecko_1d.parquet")
    except Exception as exc:
        print(f"  WARN BTC CoinGecko chart — {exc}")


def main() -> None:
    print("=== BTC on-chain (blockchain.com) ===")
    for name, chart in BLOCKCHAIN_CHARTS.items():
        fetch_blockchain_chart(name, chart)
        time.sleep(0.4)

    print("\n=== BTC CoinGecko market chart ===")
    fetch_coingecko_market_chart()


if __name__ == "__main__":
    main()
