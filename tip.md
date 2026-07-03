# OpenMedallion Tips & Tricks

## Loading Data

```python
import pandas as pd

# Single file
df = pd.read_parquet("data/crypto/BTCUSD_1d.parquet")

# Specific columns (faster, less memory)
df = pd.read_parquet("data/crypto/BTCUSD_1d.parquet", columns=["date", "Close", "Volume"])

# Merge multiple symbols
btc = pd.read_parquet("data/crypto/BTCUSD_1d.parquet")
eth = pd.read_parquet("data/crypto/ETHUSD_1d.parquet")
combined = pd.merge(btc, eth, on="date", suffixes=("_BTC", "_ETH"))
```

## Parallel Loading

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

files = list(Path("data/crypto").glob("*_1d.parquet"))
def load(path):
    return pd.read_parquet(path, columns=["date", "Close"])

with ThreadPoolExecutor(max_workers=8) as pool:
    dfs = list(pool.map(load, files))
```

## Portfolio Backtest

```python
spy = pd.read_parquet("data/indices/SPX500_1d.parquet").set_index("date")["Close"]
agg = pd.read_parquet("data/bonds/CoreTotalBond_ETF_1d.parquet").set_index("date")["Close"]
gold = pd.read_parquet("data/commodities/XAUUSD_GOLD_1d.parquet").set_index("date")["Close"]

portfolio = 0.6 * spy + 0.3 * agg + 0.1 * gold
returns = portfolio.pct_change().dropna()
print(f"Sharpe: {returns.mean()/returns.std() * (252**0.5):.2f}")
```

## DeFi Analysis

```python
from pathlib import Path
protocols = {}
for f in Path("data/defi").glob("tvl_*.parquet"):
    name = f.stem.replace("tvl_", "").replace("_1d", "")
    df = pd.read_parquet(f)
    protocols[name] = df.set_index("date")["totalLiquidityUSD"]

tvl = pd.DataFrame(protocols)
print(tvl.idxmax(axis=1).value_counts().head(10))
```

## On-Chain Metrics

```python
hashrate = pd.read_parquet("data/onchain/BTC_HashRate_1d.parquet")
addresses = pd.read_parquet("data/onchain/BTC_ActiveAddresses_1d.parquet")
onchain = pd.merge(hashrate, addresses, on="date")
print(onchain.corr())
```

## Timeframe Reference

| Suffix | Timeframe | History |
|--------|-----------|---------|
| `_1m` | 1-minute | 1-2 years |
| `_5m` | 5-minute | 1-2 years |
| `_15m` | 15-minute | 1-2 years |
| `_1h` | 1-hour | 2-5 years |
| `_4h` | 4-hour | 2-5 years |
| `_1d` | 1-day | 5-100 years |
| `_funding` | 8h funding rate | 1-2 years |

## Memory-Efficient Processing

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("data/forex/EURUSD_1h.parquet")
print(f"{pf.metadata.num_rows:,} rows")

for i in range(pf.metadata.num_row_groups):
    table = pf.read_row_group(i, columns=["date", "Close"])
    # process chunk...
```
