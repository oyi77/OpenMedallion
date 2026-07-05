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
- backtesting
- historical-data
- indonesia
- defi
- onchain
- bonds
- etfs
- macro
- volatility
- prediction-markets
- environmental
- shipping
pretty_name: OpenMedallion Financial Dataset
---

# OpenMedallion Financial Dataset

[![HF Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/oyi77/OpenMedallion)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/oyi77/OpenMedallion)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Daily Collect](https://github.com/oyi77/OpenMedallion/actions/workflows/collect.yml/badge.svg)](https://github.com/oyi77/OpenMedallion/actions/workflows/collect.yml)

**6,000+ Parquet files · 28 categories · 1,500+ assets · 87+ countries · Up to 100 years of history**

> **Full dataset** lives on [HuggingFace Datasets](https://huggingface.co/datasets/oyi77/OpenMedallion).
> This repo contains collector scripts, the sync utility, and documentation.

---

## Quick Start

```bash
# Clone this repo
git clone https://github.com/oyi77/OpenMedallion.git
cd OpenMedallion

# Install dependencies
pip install huggingface_hub pandas pyarrow

# Download full dataset from HuggingFace (~1.3 GB)
python sync_from_hf.py

# Or download specific categories only
python sync_from_hf.py --categories crypto,forex,macro
```

```python
import pandas as pd

# Crypto OHLCV (Yahoo source)
df = pd.read_parquet("data/crypto/yahoo/BTCUSD_1d.parquet")

# US equities (Yahoo source)
df = pd.read_parquet("data/equities/yahoo/AAPL_1d.parquet")

# Indonesian stocks (Stooq source)
df = pd.read_parquet("data/equities/stooq/BBCA_1d.parquet")

# FRED macro series
df = pd.read_parquet("data/macro/FRED_CPI_All_Items_1d.parquet")

# Bitcoin on-chain
df = pd.read_parquet("data/onchain/BTC_HashRate_1d.parquet")

print(df.tail())
```

---

## Folder Structure

Each category uses source-prefixed filenames so the data origin is always clear.

```
data/
├── alternative/          Wiki pageviews, alt sentiment (Wiki_*)
├── bonds/
│   ├── fred/             Treasury yields, TIPS from FRED (FRED_*)
│   └── stooq/            Bond ETF OHLCV from Stooq
├── central_bank/         Fed, ECB, BOJ rate decisions
├── commodities/
│   ├── _flat/            Futures & ETFs (COM_*, Agriculture_ETF_*)
│   └── stooq/            Commodity OHLCV from Stooq
├── countries/
│   └── stooq/            330 global stocks by country ticker
├── credit/               Credit spreads, CDS indices (FRED_*)
├── crypto/
│   ├── _flat/            Source-prefixed: Binance_*, CC_*, CG_*, Bybit_*
│   ├── binary/           Binary/compiled format files
│   ├── defi/             On-chain DeFi metrics
│   ├── derivatives/      Funding rates, OI
│   └── yahoo/            283 crypto OHLCV pairs from Yahoo Finance
├── defi/                 377 DeFi protocol TVL (DEFI_*_tvl_1d)
├── derivatives/          Bybit + OKX funding, OI, OHLCV
├── environmental/        CO2, temperature anomaly, sea ice (NOAA_*, NASA_*, OWID_*)
├── equities/
│   ├── yahoo/            531 US equities OHLCV (Yahoo Finance)
│   └── stooq/            283 global equities OHLCV (Stooq)
├── etfs/                 552 ETF OHLCV (sector, theme, international)
├── factors/              Fama-French, AQR, CFTC COT
├── forex/
│   ├── _flat/            ECB official rates (ECB_EUR*)
│   └── stooq/            71 FX pairs OHLCV from Stooq
├── fundamentals/         SEC EDGAR facts, SimFin, Form 4 insider, FINRA short interest
├── geopolitical/         WPR geopolitical risk index
├── indices/
│   └── stooq/            64 global index OHLCV from Stooq
├── labor/                FRED labor market series
├── macro/                FRED, World Bank, IMF WEO, OECD, BIS, consumer confidence
├── onchain/              488 BTC + ETH + SOL on-chain metrics
├── options/              CBOE VIX, Deribit vol surface, put/call ratio, SPX EOD
├── prediction_markets/   Polymarket, Kalshi, Manifold Markets
├── real_estate/          FRED housing series
├── sentiment/            Fear & Greed, Reddit WSB, Twitter/ML sentiment
├── shipping/             Baltic Exchange proxies, FRED freight series, container rates
├── supply_chain/         PMI, port congestion, logistics indices
├── trade/                UN Comtrade bilateral trade flows
├── training/             AI/LLM fine-tuning finance datasets
└── weather/              OpenMeteo daily weather for 20 global cities
```

---

## Coverage by Category

| Category | Files | Sources | Notes |
|----------|------:|---------|-------|
| **macro** | 1,673 | FRED, World Bank, IMF, OECD, BIS | Up to 100y history |
| **equities** | 815 | Yahoo Finance, Stooq | US + 87 countries |
| **crypto** | 690 | Binance, CryptoCompare, CoinGecko, Yahoo | OHLCV + derivatives |
| **etfs** | 552 | Yahoo Finance | Sector, theme, international |
| **onchain** | 488 | Blockchain.com, Glassnode | BTC, ETH, SOL |
| **defi** | 377 | DefiLlama | TVL, fees, protocol metrics |
| **countries** | 330 | Stooq | 87+ countries, 325+ stocks |
| **derivatives** | 250 | Bybit, OKX | Funding, OI, OHLCV |
| **commodities** | 174 | FRED, Yahoo, Stooq | Futures + ETFs |
| **forex** | 102 | ECB, Stooq | Majors + exotics |
| **indices** | 68 | Stooq | 64 global indices |
| **bonds** | 73 | FRED, Stooq | Treasuries, TIPS, ETFs |
| **fundamentals** | 60+ | SEC EDGAR, SimFin, FINRA | Facts, Form 4, short interest |
| **alternative** | 51 | Wikipedia | Pageviews for financial entities |
| **options** | 20+ | CBOE, Deribit | VIX, vol surface, put/call ratio |
| **shipping** | 20+ | FRED, World Bank | Freight indices, container rates |
| **sentiment** | 9 | Reddit, Twitter/ML | WSB mentions, NLP sentiment |
| **prediction_markets** | 5+ | Polymarket, Kalshi, Manifold | Market probabilities |
| **environmental** | 8+ | NOAA, NASA, OWID, NSIDC | CO2, temperature, sea ice |
| **geopolitical** | 10 | WPR | Risk index |
| **weather** | 18 | OpenMeteo | 20 global cities |
| **supply_chain** | 19 | FRED, ISM | PMI, logistics |
| **trade** | 10+ | UN Comtrade | Bilateral trade flows |
| **real_estate** | 6 | FRED | Housing starts, prices |
| **labor** | 25 | FRED | Employment, wages |
| **credit** | 25 | FRED | Spreads, CDS |
| **central_bank** | 30 | Fed, ECB, BOJ | Rate decisions |
| **factors** | 11 | Fama-French, AQR, CFTC | Risk factors, COT |

---

## Data Sources

| Source | Free | Coverage |
|--------|------|----------|
| **FRED** (St. Louis Fed) | ✅ | Macro, bonds, labor, credit, housing |
| **Yahoo Finance** | ✅ | Equities, ETFs, crypto, forex, commodities |
| **Stooq** | ✅ | Global equities, forex, indices, bonds |
| **Binance** | ✅ | Crypto OHLCV + derivatives |
| **CoinGecko** | ✅ | Top 100 crypto, global metrics |
| **CryptoCompare** | ✅* | Crypto OHLCV, social (key optional) |
| **DefiLlama** | ✅ | DeFi TVL, fees, protocol metrics |
| **Blockchain.com** | ✅ | BTC on-chain metrics |
| **SEC EDGAR** | ✅ | Form 10-K/Q facts, Form 4 insider |
| **FINRA** | ✅ | Short volume data |
| **ECB / Frankfurter** | ✅ | EUR exchange rates (1999–) |
| **World Bank** | ✅ | 1,400+ macro indicators, commodities |
| **IMF WEO** | ✅ | World Economic Outlook annual |
| **OECD** | ✅ | Multi-country macro indicators |
| **BIS** | ✅ | Credit, property prices, FX stats |
| **NOAA / NASA / NSIDC** | ✅ | CO2, temperature anomaly, sea ice |
| **Our World in Data** | ✅ | CO2 emissions per country |
| **Polymarket** | ✅ | Prediction market probabilities |
| **Kalshi** | ✅ | US-regulated prediction markets |
| **Manifold Markets** | ✅ | Community prediction markets |
| **OpenMeteo** | ✅ | Weather for 20 global cities |
| **UN Comtrade** | ✅* | Bilateral trade flows (key optional) |
| **SimFin** | ✅* | Financial statements (key optional) |

*API key optional — collectors degrade gracefully without one.

---

## Running Collectors

```bash
# Install all deps
pip install pandas pyarrow requests huggingface_hub openpyxl xlrd

# Run all collectors
python run_all_collectors.py

# Run specific group
python run_all_collectors.py --group crypto
python run_all_collectors.py --group environmental
python run_all_collectors.py --group prediction_markets

# Dry run (list collectors without executing)
python run_all_collectors.py --dry-run

# Skip heavy collectors (SEC bulk, OWID full dataset)
python run_all_collectors.py --skip-heavy

# Verbose output
python run_all_collectors.py --verbose
```

### Environment Variables (all optional)

```bash
export CRYPTOCOMPARE_API_KEY=your_key   # higher rate limits
export SIMFIN_API_KEY=your_key          # financial statements
export REDDIT_CLIENT_ID=your_id         # WSB sentiment
export REDDIT_CLIENT_SECRET=your_secret
export KALSHI_API_KEY=your_key          # Kalshi markets
export HF_TOKEN=your_hf_token           # push to HuggingFace
```

---

## Cross-Platform

| Platform | URL | Content |
|----------|-----|---------|
| 🤗 HuggingFace | [oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion) | Full dataset (6,000+ files, ~1.3 GB) |
| 🐙 GitHub | [oyi77/OpenMedallion](https://github.com/oyi77/OpenMedallion) | Collector scripts + sync utility |

---

## Contributing

1. Fork the repo
2. Add a collector in `collectors/<category>/<source>.py`
3. Use `save(df, category, f"SOURCE_name_freq.parquet")` from `base.py`
4. Add your script to `run_all_collectors.py`
5. Open a PR

**Naming convention:** `SOURCE_AssetOrSeries_Freq.parquet`
- `FRED_CPI_All_Items_1d.parquet`
- `Binance_BTCUSDT_1h.parquet`
- `Stooq_BBCA_1d.parquet`

---

## License

Collector code: [MIT](LICENSE)

Data: sourced from public APIs and open datasets. Each source retains its own terms:
- FRED: [Terms of Use](https://fred.stlouisfed.org/legal/)
- Yahoo Finance: [Terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
- SEC EDGAR: Public domain
- World Bank: [CC BY 4.0](https://datacatalog.worldbank.org/public-licenses)
- NOAA/NASA: Public domain (US Government)

---

## Support

If this dataset helps your research or trading, consider [buying a coffee](https://www.tip.md/oyi77). ☕

Every contribution helps keep this dataset free and actively maintained.
