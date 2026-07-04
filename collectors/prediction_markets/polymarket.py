"""
Polymarket full history collector.
Source: Polymarket public API (gamma.markets + clob.polymarket.com)
No API key required.
Covers: All markets history, resolution prices, volume, liquidity.
Output: data/prediction_markets/Polymarket_<market_slug>_1d.parquet
         data/prediction_markets/Polymarket_all_markets_snapshot.parquet
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base import fetch, save

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def fetch_all_markets(limit: int = 500) -> pd.DataFrame:
    """Fetch all markets from Gamma API."""
    params = {"limit": limit, "active": "false", "closed": "true", "order": "volume", "ascending": "false"}
    resp = fetch(f"{GAMMA_API}/markets", params=params)
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame()
    return df


def fetch_market_prices(condition_id: str, slug: str) -> pd.DataFrame:
    """Fetch price history for a market from CLOB."""
    try:
        # Get token IDs for this market
        resp = fetch(f"{CLOB_API}/markets/{condition_id}", timeout=20)
        market = resp.json()
        tokens = market.get("tokens", [])
        if not tokens:
            return pd.DataFrame()

        token_id = tokens[0].get("token_id")
        if not token_id:
            return pd.DataFrame()

        # Fetch price history
        resp2 = fetch(
            f"{CLOB_API}/prices-history",
            params={"token_id": token_id, "interval": "1d", "fidelity": 1},
            timeout=20,
        )
        data = resp2.json()
        history = data.get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df = df.set_index("t").sort_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return df

    except Exception:
        return pd.DataFrame()


def main() -> None:
    print("Fetching all Polymarket markets ...")
    df_all = fetch_all_markets(limit=500)
    if not df_all.empty:
        if "createdAt" in df_all.columns:
            df_all["createdAt"] = pd.to_datetime(df_all["createdAt"], errors="coerce", utc=True)
        save(df_all, "prediction_markets", "Polymarket_all_markets_snapshot.parquet")

    # Fetch price history for top markets by volume
    if df_all.empty:
        return

    vol_col = next((c for c in df_all.columns if "volume" in c.lower()), None)
    if vol_col:
        df_all[vol_col] = pd.to_numeric(df_all[vol_col], errors="coerce")
        top = df_all.nlargest(100, vol_col)
    else:
        top = df_all.head(100)

    cond_col = next((c for c in top.columns if "conditionId" in c or "condition_id" in c), None)
    slug_col = next((c for c in top.columns if "slug" in c.lower()), None)

    if cond_col is None:
        print("  WARNING: Cannot find conditionId column — skipping price history")
        return

    print(f"\nFetching price history for top {len(top)} markets ...")
    for _, row in top.iterrows():
        cond_id = str(row[cond_col])
        slug = str(row[slug_col])[:50] if slug_col else cond_id[:20]
        # Sanitize for filename
        safe_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug)[:60]
        try:
            df_prices = fetch_market_prices(cond_id, slug)
            if not df_prices.empty:
                save(df_prices, "prediction_markets", f"Polymarket_{safe_slug}_1d.parquet")
            time.sleep(0.3)
        except Exception as exc:
            print(f"  WARNING: {slug} — {exc}")


if __name__ == "__main__":
    main()
