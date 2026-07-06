#!/usr/bin/env python3
"""Run all data collectors in parallel batches."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

COLLECTORS = [
    # Core markets
    "collectors/equities/yfinance_equities.py",
    "collectors/etfs/yfinance_etfs.py",
    "collectors/forex/fred_forex.py",
    "collectors/commodities/yfinance_futures.py",
    "collectors/indices/yfinance_indices.py",
    "collectors/bonds/fred_bonds.py",
    
    # Macro & fundamentals
    "collectors/macro/fred_macro.py",
    "collectors/labor/labor_market.py",
    "collectors/fundamentals/sec_edgar.py",
    "collectors/fundamentals/insider_short.py",
    "collectors/credit/credit_markets.py",
    "collectors/central_bank/central_bank.py",
    
    # Crypto & DeFi
    "collectors/crypto/cryptocompare.py",
    "collectors/crypto/exchange_volumes.py",
    "collectors/crypto/funding_rates.py",
    "collectors/crypto/liquidations.py",
    "collectors/crypto/open_interest.py",
    "collectors/crypto/stablecoin_supply.py",
    "collectors/onchain/eth_metrics.py",
    "collectors/defi/protocol_fees.py",
    
    # Derivatives & Options
    "collectors/derivatives/cboe_indices.py",
    "collectors/derivatives/vix_term_structure.py",
    "collectors/derivatives/skew_index.py",
    "collectors/options/options_volume.py",
    
    # Alternative data
    "collectors/sentiment/fear_greed_backfill.py",
    "collectors/sentiment/google_trends.py",
    "collectors/sentiment/stocktwits_volume.py",
    "collectors/sentiment/reddit_wsb.py",
    "collectors/geopolitical/wpr_index.py",
    "collectors/sports/mlb_gamelogs.py",
    "collectors/sports/nba_elo.py",
    "collectors/sports/cricket_matches.py",
    "collectors/environmental/environmental.py",
    "collectors/weather/openmeteo.py",
    
    # Supply chain & trade
    "collectors/supply_chain/supply_chain.py",
    "collectors/trade/worldbank_trade.py",
    "collectors/trade/fred_trade.py",
    "collectors/shipping/baltic_exchange.py",
    
    # Energy & Agriculture
    "collectors/energy/eia.py",
    "collectors/energy/eia_inventories.py",
    "collectors/agriculture/usda.py",
    "collectors/agriculture/usda_commodities.py",
    
    # Factors & Real Estate
    "collectors/factors/fama_french.py",
    "collectors/factors/aqr_factors.py",
    "collectors/real_estate/real_estate.py",
    
    # Prediction markets
    "collectors/betting/sports_odds.py",
]

def run_collector(script: str) -> tuple[str, bool, str]:
    """Run a single collector script."""
    try:
        result = subprocess.run(
            ["python3", script],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return (script, success, output)
    except subprocess.TimeoutExpired:
        return (script, False, "TIMEOUT")
    except Exception as e:
        return (script, False, str(e))

def main():
    print(f"Running {len(COLLECTORS)} collectors in parallel (max 8 workers)...\n")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_collector, c): c for c in COLLECTORS}
        
        completed = 0
        failed = []
        
        for future in as_completed(futures):
            script, success, output = future.result()
            completed += 1
            
            if success:
                print(f"[{completed}/{len(COLLECTORS)}] ✓ {Path(script).name}")
            else:
                print(f"[{completed}/{len(COLLECTORS)}] ✗ {Path(script).name}")
                failed.append((script, output))
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(COLLECTORS) - len(failed)}/{len(COLLECTORS)}")
    
    if failed:
        print(f"\nFailed collectors ({len(failed)}):")
        for script, output in failed:
            print(f"  - {script}")
            if "TIMEOUT" not in output:
                print(f"    {output[:200]}")

if __name__ == "__main__":
    main()
