"""
Crypto sentiment collector.
Sources:
  - Alternative.me Fear & Greed Index (crypto, no key)
  - CoinGecko global market stats (no key)
  - CoinMarketCap Fear & Greed via FRED proxy
Output: data/sentiment/CRYPTO_<metric>_1d.parquet
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

ALTME_BASE = "https://api.alternative.me/fng/"
CG_BASE = "https://api.coingecko.com/api/v3"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fear_greed() -> None:
    """Alternative.me crypto Fear & Greed — up to 2000 days history."""
    try:
        resp = fetch(ALTME_BASE, params={"limit": 2000, "format": "json"}, timeout=20)
        data = resp.json().get("data", [])
        if not data:
            print("  SKIP fear/greed — no data")
            return
        rows = [
            {
                "date": pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC"),
                "value": int(r["value"]),
                "classification": r.get("value_classification", ""),
            }
            for r in data
        ]
        df = pd.DataFrame(rows).set_index("date").sort_index()
        save(df, "sentiment", "CRYPTO_FearGreed_1d.parquet")
    except Exception as exc:
        print(f"  WARN fear/greed — {exc}")


def fetch_cg_global() -> None:
    """CoinGecko global market dominance / cap / volume snapshot series."""
    try:
        resp = fetch(f"{CG_BASE}/global", timeout=15)
        data = resp.json().get("data", {})
        snapshot = {
            "date": pd.Timestamp.utcnow().normalize(),
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
            "btc_dominance_pct": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance_pct": data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptos": data.get("active_cryptocurrencies"),
            "markets": data.get("markets"),
            "defi_to_eth_ratio": data.get("defi_to_eth_ratio"),
        }
        df = pd.DataFrame([snapshot]).set_index("date")
        save(df, "sentiment", "CRYPTO_GlobalMarket_Snapshot.parquet")
    except Exception as exc:
        print(f"  WARN CG global — {exc}")


def fetch_cg_trending() -> None:
    """CoinGecko trending coins snapshot."""
    try:
        resp = fetch(f"{CG_BASE}/search/trending", timeout=15)
        coins = resp.json().get("coins", [])
        rows = [
            {
                "date": pd.Timestamp.utcnow().normalize(),
                "rank": i + 1,
                "id": c["item"]["id"],
                "name": c["item"]["name"],
                "symbol": c["item"]["symbol"],
                "market_cap_rank": c["item"].get("market_cap_rank"),
                "price_btc": c["item"].get("price_btc"),
            }
            for i, c in enumerate(coins)
        ]
        if not rows:
            return
        df = pd.DataFrame(rows).set_index("date")
        save(df, "sentiment", "CRYPTO_Trending_Snapshot.parquet")
    except Exception as exc:
        print(f"  WARN CG trending — {exc}")


def fetch_fred_crypto_sentiment() -> None:
    """FRED series useful as crypto sentiment proxies."""
    series = {
        "CRYPTO_SENT_BTC_GoogleTrend_1w": "BTCUSD",  # not FRED — skip, use VIXCLS
        "CRYPTO_SENT_VIX_1d": "VIXCLS",
        "CRYPTO_SENT_SP500_1d": "SP500",
        "CRYPTO_SENT_EFFR_1d": "EFFR",
    }
    for name, sid in series.items():
        if sid == "BTCUSD":
            continue  # not a FRED series
        try:
            resp = fetch(FRED_BASE, params={"id": sid}, timeout=20)
            df = pd.read_csv(StringIO(resp.text), na_values=".")
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")[["value"]].dropna().sort_index()
            save(df, "sentiment", f"{name}.parquet")
        except Exception as exc:
            print(f"  WARN {name} ({sid}) — {exc}")


def main() -> None:
    print("=== Crypto Fear & Greed (Alternative.me) ===")
    fetch_fear_greed()

    print("\n=== CoinGecko global market snapshot ===")
    fetch_cg_global()

    print("\n=== CoinGecko trending snapshot ===")
    fetch_cg_trending()

    print("\n=== FRED crypto sentiment proxies ===")
    fetch_fred_crypto_sentiment()


if __name__ == "__main__":
    main()
