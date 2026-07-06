"""
Bitcoin on-chain metrics from blockchain.com free API.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from collectors.base import save

_CHARTS = {
    "hash-rate": "hashrate_th/s",
    "difficulty": "difficulty",
    "market-price": "price_usd",
    "n-transactions": "tx_count_daily",
    "mempool-size": "mempool_bytes",
    "cost-per-transaction": "tx_fee_avg_usd",
    "estimated-transaction-volume": "tx_volume_btc",
    "average-block-size": "block_size_bytes",
    "n-unique-addresses": "unique_addresses",
    "total-bitcoins": "circulating_supply",
    "transaction-fees": "total_fees_btc",
    "median-confirmation-time": "conf_time_median_s",
}


def collect():
    """Fetch 12 Bitcoin charts from blockchain.com API."""
    import urllib.request, json, time
    rows = []
    for chart_id, label in _CHARTS.items():
        url = f"https://api.blockchain.info/charts/{chart_id}?timespan=5years&format=json"
        print(f"  {label}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            vals = data.get("values", [])
            df = pd.DataFrame(vals)
            df["date"] = pd.to_datetime(df["x"], unit="s")
            df[label] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["date", label]).set_index("date")[[label]]
            rows.append(df)
        except Exception as e:
            print(f"    WARN: {label}: {e}")
        time.sleep(0.5)

    combined = pd.concat(rows, axis=1).sort_index()
    combined.index.name = "date"
    out = save(combined, "crypto", "bitcoin_onchain_1d.parquet")
    print(f"Saved {out}: {len(combined)} rows, {len(combined.columns)} metrics")


if __name__ == "__main__":
    collect()
