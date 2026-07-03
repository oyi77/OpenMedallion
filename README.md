---
language:
- en
license: mit
tags:
- finance
- trading
- crypto
- forex
- stocks
- commodities
- indices
- defi
- bonds
- etfs
pretty_name: OpenMedallion Financial Dataset
---

# OpenMedallion Financial Dataset

[![HF Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/oyi77/OpenMedallion)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/oyi77/OpenMedallion)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**2,609 Parquet files · 18 categories · 1,500+ assets · 87+ countries · Up to 100 years of history**

> 💡 **Full dataset** lives on [HuggingFace Datasets](https://huggingface.co/datasets/oyi77/OpenMedallion).  
> This repo contains the sync script and usage documentation.

## Quick Start

```bash
# Clone this repo
git clone https://github.com/oyi77/OpenMedallion.git
cd OpenMedallion

# Install dependencies
pip install huggingface_hub pandas pyarrow

# Download full dataset from HuggingFace (~1.3 GB)
python sync_from_hf.py

# Load and explore
python -c "
import pandas as pd
df = pd.read_parquet('data/crypto/BTCUSD_1d.parquet')
print(f'{len(df):,} rows')
print(df.head())
"
```

## Coverage

| Category | Files | Assets | Description |
|----------|------:|-------:|-------------|
| **Crypto OHLCV** | 573 | 250+ | Binance top 50 + CoinGecko Top 100 |
| **Crypto Derivatives** | 136 | 30 | Funding rates, open interest, LSR |
| **Equities (US)** | 538 | 428+ | S&P 500, DJIA 30, NASDAQ |
| **Global Stocks** | 330 | 325+ | 87+ countries |
| **ETFs** | 384 | 361+ | Sector, theme, international |
| **Forex** | 184 | 57+ | Majors + exotics |
| **Commodities** | 126 | 53+ | Gold, Silver, Oil, Agriculture |
| **Indices** | 109 | 24+ | Global + volatility (VIX) |
| **Bonds** | 60 | 25+ | Treasury, TIPS, corporate |
| **Macro** | 60 | 46+ | Currency hedged, sector ETFs |
| **DeFi** | 13 | 200+ | Protocol TVL, chain TVL, lending |
| **On-Chain** | 16 | BTC | Hash rate, addresses, mempool |

## Data Sources (All Free, No API Keys)

| Source | Coverage |
|--------|----------|
| **Binance** | Crypto pairs, funding rates, OI |
| **Yahoo Finance** | Stocks, ETFs, forex, commodities, indices |
| **CoinGecko** | Top 100 crypto, global metrics |
| **DefiLlama** | DeFi protocol TVL, stablecoins, lending |
| **Blockchain.com** | BTC on-chain metrics |
| **FinanceDatabase** | 365K ticker universe |

## Usage Tips

See [`tip.md`](tip.md) for:
- Loading data efficiently (parallel, column selection)
- Portfolio backtest examples
- DeFi and on-chain analysis
- Timeframe reference guide

## Cross-Platform

| Platform | URL | Content |
|----------|-----|---------|
| 🤗 HuggingFace | [oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion) | Full dataset (2,609 files, ~1.3 GB) |
| 🐙 GitHub | [oyi77/OpenMedallion](https://github.com/oyi77/OpenMedallion) | Sync script + documentation |

## License

Data sourced from public APIs. Each source has its own terms of use.
