#!/usr/bin/env python3
"""Run all data collectors in parallel batches."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_collector(script: str, timeout: int = 300) -> tuple[str, bool, str]:
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


def main() -> None:
    print(f"Running {len(COLLECTORS)} collectors in parallel (max 8 workers)...\n")
    failed: list[tuple[str, str]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_collector, s): s for s in COLLECTORS}
        for future in as_completed(futures):
            script, success, output = future.result()
            completed += 1
            status = "✓" if success else "✗"
            print(f"[{completed:3d}/{len(COLLECTORS)}] {status} {Path(script).name}")
            if not success:
                failed.append((script, output))

    print(f"\n{'='*60}")
    print(f"Completed: {len(COLLECTORS) - len(failed)}/{len(COLLECTORS)}")

    if failed:
        print(f"\nFailed collectors ({len(failed)}):")
        for script, output in failed:
            print(f"  - {script}")
            if "TIMEOUT" not in output:
                # Show last meaningful line of error
                lines = [l.strip() for l in output.splitlines() if l.strip()]
                if lines:
                    print(f"    {lines[-1][:200]}")


if __name__ == "__main__":
    main()
