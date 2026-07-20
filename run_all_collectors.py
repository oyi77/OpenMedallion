#!/usr/bin/env python3
"""Run all data collectors in parallel batches.

Configuration via environment variables:
    OM_COLLECTOR_TIMEOUT  — default per-collector timeout in seconds (default 300)
    OM_MAX_WORKERS         — parallel worker count (default 8)

Usage:
    python run_all_collectors.py              # run all collectors
    python run_all_collectors.py collectors/macro/fred_macro.py  # run one
    python run_all_collectors.py --since -24h  # only re-run recently-changed scripts
    python run_all_collectors.py --since 2026-07-01
"""

import argparse
import os
import subprocess
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurable defaults
_DEFAULT_TIMEOUT = int(os.environ.get("OM_COLLECTOR_TIMEOUT", "300"))
_DEFAULT_WORKERS = int(os.environ.get("OM_MAX_WORKERS", "8"))

COLLECTORS = [
    # Agriculture
    "collectors/agriculture/usda.py",
    "collectors/agriculture/usda_commodities.py",
    "collectors/agriculture/indonesia_agri.py",
    "collectors/agriculture/worldbank_commodities.py",
    "collectors/agriculture/fred_agri_ppi.py",
    "collectors/agriculture/fred_agri_series.py",

    # Alternative data
    "collectors/alternative/ais_vessel_counts.py",
    "collectors/alternative/baltic_dry_index.py",
    "collectors/alternative/github_stars.py",
    "collectors/alternative/wikipedia_views.py",

    # Bonds
    "collectors/bonds/em_bonds.py",
    "collectors/bonds/fred_bonds.py",

    # Central Bank
    "collectors/central_bank/central_bank.py",
    "collectors/central_bank/central_bank_rates.py",

    # Commodities
    "collectors/commodities/rare_earths.py",
    "collectors/commodities/soft_commodities.py",
    "collectors/commodities/yfinance_futures.py",

    "collectors/credit/credit_markets.py",
    "collectors/credit/credit_spreads.py",

    # Crypto
    "collectors/crypto/coingecko_top200_historical.py",
    "collectors/crypto/cryptocompare.py",
    "collectors/crypto/exchange_volumes.py",
    "collectors/crypto/funding_rates.py",
    "collectors/crypto/liquidations.py",
    "collectors/crypto/open_interest.py",
    "collectors/crypto/dex_volume.py",
    "collectors/crypto/stablecoin_supply.py",
    "collectors/crypto/stablecoin_metrics.py",
    "collectors/crypto/bitcoin_onchain.py",
    "collectors/crypto/coingecko_coins.py",

    # DeFi
    "collectors/defi/protocol_fees.py",

    # Derivatives
    "collectors/derivatives/bybit.py",
    "collectors/derivatives/cboe_indices.py",
    "collectors/derivatives/okx.py",
    "collectors/derivatives/skew_index.py",
    "collectors/derivatives/vix_term_structure.py",
    "collectors/derivatives/equity_index_futures.py",
    "collectors/derivatives/treasury_futures.py",

    # Energy
    "collectors/energy/crude_oil_inventories.py",
    "collectors/energy/eia.py",
    "collectors/energy/eia_inventories.py",
    "collectors/energy/eia_ng_oil.py",
    "collectors/energy/energy_extended.py",

    "collectors/environmental/environmental.py",
    "collectors/environmental/usgs_earthquakes.py",
    "collectors/environmental/noaa_climate_extended.py",
    "collectors/environmental/noaa_state_temps.py",

    # Equities
    "collectors/equities/sea_equities.py",
    "collectors/equities/yfinance_equities.py",
    "collectors/equities/asia_equities.py",
    "collectors/equities/indonesia_idx.py",

    "collectors/equities/sp_index.py",

    "collectors/etfs/yfinance_etfs.py",

    # Factors
    "collectors/factors/aqr_factors.py",
    "collectors/factors/cftc_cot.py",
    "collectors/factors/fama_french.py",

    # Forex
    "collectors/forex/ecb_frankfurter.py",
    "collectors/forex/fred_forex.py",

    "collectors/forex/fx_futures.py",
    # Fundamentals

    "collectors/fundamentals/dividend_history.py",
    "collectors/fundamentals/insider_short.py",
    "collectors/fundamentals/sec_edgar.py",
    "collectors/fundamentals/sec_form4_transactions.py",
    "collectors/fundamentals/simfin.py",

    # Geopolitical
    "collectors/geopolitical/acled.py",
    "collectors/geopolitical/wpr_index.py",
    "collectors/geopolitical/ucdp_conflict.py",

    # Indices
    "collectors/indices/yfinance_indices.py",

    # Labor
    "collectors/labor/labor_market.py",

    # Misc
    "collectors/misc/noaa_climate.py",

    # Macro
    "collectors/macro/asean_macro.py",
    "collectors/macro/bis.py",
    "collectors/macro/bis_debt.py",
    "collectors/macro/china_macro.py",
    "collectors/macro/consumer_confidence.py",
    "collectors/macro/ecb_rates.py",
    "collectors/macro/fred_macro.py",
    "collectors/macro/housing_macro.py",
    "collectors/macro/imf_weo.py",
    "collectors/macro/indonesia_bi_rates.py",
    "collectors/macro/indonesia_bonds.py",
    "collectors/macro/indonesia_bps_macro.py",
    "collectors/macro/oecd.py",
    "collectors/macro/oecd_leading.py",
    "collectors/macro/worldbank_global.py",
    "collectors/macro/us_births.py",
    "collectors/macro/worldbank.py",
    "collectors/macro/sp_fundamentals.py",

    # On-chain
    "collectors/onchain/bitcoin_metrics.py",
    "collectors/onchain/eth_metrics.py",
    "collectors/onchain/sol_metrics.py",
    "collectors/onchain/bitcoin_network.py",

    # Options
    "collectors/options/cboe_vix.py",
    "collectors/options/deribit.py",
    "collectors/options/options_chains.py",
    "collectors/options/options_flow.py",

    # Prediction markets
    "collectors/prediction_markets/hf_bulk.py",
    "collectors/prediction_markets/kalshi.py",
    "collectors/prediction_markets/kalshi_history.py",
    "collectors/prediction_markets/manifold.py",
    "collectors/prediction_markets/manifold_history.py",
    "collectors/prediction_markets/metaculus_resolved.py",
    "collectors/prediction_markets/polymarket.py",
    "collectors/prediction_markets/sports_odds.py",
    "collectors/real_estate/real_estate.py",
    "collectors/real_estate/fred_housing.py",
    "collectors/real_estate/reit_prices.py",
    "collectors/real_estate/zillow_metro.py",

    # Sentiment
    "collectors/sentiment/aaii_putcall.py",
    "collectors/sentiment/crypto_sentiment.py",
    "collectors/sentiment/fear_greed_backfill.py",
    "collectors/sentiment/gdelt_themed.py",
    "collectors/sentiment/google_trends.py",
    "collectors/sentiment/news_sentiment.py",
    "collectors/sentiment/reddit_wsb.py",
    "collectors/sentiment/stocktwits_volume.py",
    "collectors/sentiment/wikipedia_pageviews.py",

    # Shipping
    "collectors/shipping/baltic_exchange.py",
    "collectors/shipping/freightos.py",
    "collectors/shipping/port_congestion.py",

    # Sports
    "collectors/sports/cricket_matches.py",
    "collectors/sports/mlb_gamelogs.py",
    "collectors/sports/nba_elo.py",
    "collectors/sports/nfl_scores.py",
    "collectors/sports/nhl_scores.py",
    "collectors/sports/football_matches.py",
    "collectors/sports/nba_teams.py",
    "collectors/sports/f1_results.py",

    # Supply chain
    "collectors/supply_chain/supply_chain.py",

    # Trade
    "collectors/trade/comtrade.py",
    "collectors/trade/fred_trade.py",
    "collectors/trade/worldbank_trade.py",

    # Weather
    "collectors/weather/openmeteo.py",

    # Rates
    "collectors/rates/term_rates.py",

    # Volatility
    "collectors/volatility/vol_surface.py",
]
# Dependency graph (documentation only — NOT enforced at runtime).
# Keys are collector paths, values are lists of collectors that should run
# first because this collector consumes their output.
DEPENDENCIES: dict[str, list[str]] = {
#   "collectors/macro/fred_bonds.py": ["collectors/macro/fred_macro.py"],
#   "collectors/rates/term_rates.py":  ["collectors/macro/fred_macro.py"],
}


# Per-collector timeout overrides (seconds).
# Unlisted collectors use the default 300s = 5 min.
TIMEOUT_PROFILES: dict[str, int] = {
    # Heavy API pulls
    "yfinance": 600,
    "simfin": 600,
    "sec_edgar": 600,
    "sec_form4": 600,
    # Crypto — many coins / deep history
    "coingecko_top200": 600,
    "cryptocompare": 600,
    # Prediction markets — multiple endpoints
    "polymarket": 600,
    "metaculus": 600,
    "kalshi": 600,
    "kalshi_history": 600,
    "manifold": 600,
    "manifold_history": 600,
    "hf_bulk": 600,
    # Shipping — multi-source aggregation
    "baltic": 600,
    "freightos": 600,
    # Geo-political — heavy payloads
    "acled": 600,
    "ucdp_conflict": 600,
    # Macro — many indicators
    "imf_weo": 600,
    "worldbank": 600,
    # On-chain — large blockchain data pulls
    "bitcoin_metrics": 600,
    "eth_metrics": 600,
    "sol_metrics": 600,
}


def _lookup_timeout(script: str, default: int = _DEFAULT_TIMEOUT) -> int:
    """Return the longest-matching timeout profile for *script*."""
    stem = Path(script).stem  # e.g. "yfinance_equities"
    best = default
    for key, value in TIMEOUT_PROFILES.items():
        if key in stem and value > best:
            best = value
    return best


def run_collector(script: str, timeout: int = _DEFAULT_TIMEOUT) -> tuple[str, bool, str]:
    """Run a single collector script, capture output."""
    try:
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return (script, success, output)
    except subprocess.TimeoutExpired:
        return (script, False, "TIMEOUT")
    except Exception as exc:
        return (script, False, str(exc))


def _parse_since(value: str) -> float:
    """Parse --since arg into a Unix timestamp cutoff."""
    if value.startswith("-"):
        import re
        m = re.match(r"-(\d+)([hdw])", value)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mult = {"h": 3600, "d": 86400, "w": 604800}
            return time_mod.time() - n * mult[unit]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Can't parse --since: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data collectors.")
    parser.add_argument("--since", help="Only run scripts modified since this time (ISO date, -Nh, -Nd, -Nw)")
    parser.add_argument("collector", nargs="?", help="Run a single collector script path")
    args = parser.parse_args()

    collectors = [args.collector] if args.collector else list(COLLECTORS)

    if args.since:
        cutoff = _parse_since(args.since)
        collectors = [c for c in collectors if Path(c).stat().st_mtime >= cutoff]
        if not collectors:
            print("No collectors modified since the given time.")
            return

    print(f"Running {len(collectors)} collectors in parallel (max {_DEFAULT_WORKERS} workers)...\n")
    failed: list[tuple[str, str]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=_DEFAULT_WORKERS) as pool:
        futures = {pool.submit(run_collector, s, _lookup_timeout(s)): s for s in collectors}
        for future in as_completed(futures):
            script, success, output = future.result()
            completed += 1
            status = "✓" if success else "✗"
            print(f"[{completed:3d}/{len(collectors)}] {status} {Path(script).name}")
            if not success:
                failed.append((script, output))

    print(f"\n{'='*60}")
    print(f"Completed: {len(collectors) - len(failed)}/{len(collectors)}")

    if failed:
        print(f"\nFailed collectors ({len(failed)}):")
        for script, output in failed:
            print(f"  - {script}")
            if "TIMEOUT" not in output:
                lines = [l.strip() for l in output.splitlines() if l.strip()]
                if lines:
                    print(f"    {lines[-1][:200]}")


if __name__ == "__main__":
    main()
