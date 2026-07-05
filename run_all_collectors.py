#!/usr/bin/env python3
"""
Run all OpenMedallion collectors in parallel batches.

Usage:
    python run_all_collectors.py                  # run all
    python run_all_collectors.py --group macro    # run one group
    python run_all_collectors.py --dry-run        # list what would run
    python run_all_collectors.py --timeout 300    # per-script timeout (seconds)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Groups define run order and concurrency cap per group.
# Scripts that share rate-limited APIs are placed in the same group
# so they run sequentially within the group (max_workers=1 for that group)
# while groups themselves run in parallel via the outer executor.
GROUPS: dict[str, dict] = {
    "fred_macro": {
        "scripts": [
            "collectors/macro/consumer_confidence.py",
            "collectors/macro/housing_macro.py",
            "collectors/central_bank/central_bank.py",
            "collectors/credit/credit_markets.py",
            "collectors/labor/labor_market.py",
            "collectors/supply_chain/supply_chain.py",
            "collectors/real_estate/real_estate.py",
            "collectors/geopolitical/wpr_index.py",
            "collectors/shipping/baltic_exchange.py",
            "collectors/shipping/freightos.py",
        ],
        "workers": 3,  # FRED allows ~3 parallel requests safely
        "timeout": 180,
    },
    "macro_intl": {
        "scripts": [
            "collectors/macro/imf_weo.py",
            "collectors/macro/worldbank.py",
            "collectors/macro/bis.py",
            "collectors/macro/oecd.py",
        ],
        "workers": 2,
        "timeout": 180,
    },
    "crypto_derivatives": {
        "scripts": [
            "collectors/derivatives/bybit.py",
            "collectors/derivatives/okx.py",
        ],
        "workers": 2,
        "timeout": 150,
    },
    "options_volatility": {
        "scripts": [
            "collectors/options/cboe_vix.py",
            "collectors/options/deribit.py",
            "collectors/options/options_flow.py",
        ],
        "workers": 2,
        "timeout": 150,
    },
    "factors": {
        "scripts": [
            "collectors/factors/fama_french.py",
            "collectors/factors/cftc_cot.py",
            "collectors/factors/aqr_factors.py",
        ],
        "workers": 2,
        "timeout": 120,
    },
    "defi_onchain": {
        "scripts": [
            "collectors/defi/protocol_fees.py",
            "collectors/onchain/eth_metrics.py",
            "collectors/onchain/sol_metrics.py",
        ],
        "workers": 1,  # CoinGecko rate-limits hard — serialize
        "timeout": 150,
    },
    "fundamentals": {
        "scripts": [
            "collectors/fundamentals/sec_edgar.py",
            "collectors/fundamentals/simfin.py",
            "collectors/fundamentals/insider_short.py",
        ],
        "workers": 1,
        "timeout": 300,
    },
    "market_data": {
        "scripts": [
            "collectors/forex/ecb_frankfurter.py",
            "collectors/energy/eia.py",
            "collectors/agriculture/usda.py",
            "collectors/weather/openmeteo.py",
        ],
        "workers": 2,
        "timeout": 150,
    },
    "alternative": {
        "scripts": [
            "collectors/sentiment/wikipedia_pageviews.py",
            "collectors/sentiment/reddit_wsb.py",
            "collectors/prediction_markets/polymarket.py",
            "collectors/prediction_markets/kalshi.py",
            "collectors/prediction_markets/manifold.py",
            "collectors/prediction_markets/hf_bulk.py",
            "collectors/trade/comtrade.py",
        ],
        "workers": 2,
        "timeout": 600,
    },
    "environmental": {
        "scripts": [
            "collectors/environmental/environmental.py",
        ],
        "workers": 1,
        "timeout": 180,
    },
}


def run_script(script: str, timeout: int) -> dict:
    path = ROOT / script
    if not path.exists():
        return {"script": script, "status": "MISSING", "elapsed": 0, "output": ""}
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        elapsed = time.monotonic() - t0
        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
        output = (result.stdout + result.stderr).strip()
        return {"script": script, "status": status, "elapsed": elapsed, "output": output}
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        return {"script": script, "status": "TIMEOUT", "elapsed": elapsed, "output": f"Killed after {timeout}s"}
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {"script": script, "status": f"ERR({exc})", "elapsed": elapsed, "output": str(exc)}


def run_group(name: str, cfg: dict, verbose: bool) -> list[dict]:
    scripts = cfg["scripts"]
    workers = cfg.get("workers", 3)
    timeout = cfg.get("timeout", 180)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_script, s, timeout): s for s in scripts}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            icon = "✓" if r["status"] == "OK" else "✗"
            print(f"  [{name}] {icon} {r['script']} — {r['status']} ({r['elapsed']:.1f}s)")
            if verbose and r["output"]:
                for line in r["output"].splitlines()[-5:]:
                    print(f"       {line}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenMedallion collectors")
    parser.add_argument("--group", help="Run a specific group only")
    parser.add_argument("--dry-run", action="store_true", help="List scripts without running")
    parser.add_argument("--timeout", type=int, default=0, help="Override per-script timeout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show script output tails")
    parser.add_argument("--sequential", action="store_true", help="Run groups one at a time")
    args = parser.parse_args()

    groups = {k: v for k, v in GROUPS.items() if not args.group or k == args.group}
    if not groups:
        print(f"Unknown group: {args.group}. Available: {', '.join(GROUPS)}")
        sys.exit(1)

    if args.timeout:
        for cfg in groups.values():
            cfg["timeout"] = args.timeout

    total_scripts = sum(len(v["scripts"]) for v in groups.values())
    print(f"OpenMedallion collector run — {len(groups)} groups, {total_scripts} scripts")
    if args.dry_run:
        for name, cfg in groups.items():
            print(f"\n[{name}] ({cfg['workers']} workers, {cfg['timeout']}s timeout)")
            for s in cfg["scripts"]:
                exists = "✓" if (ROOT / s).exists() else "✗ MISSING"
                print(f"  {exists}  {s}")
        return

    t_start = time.monotonic()
    all_results: list[dict] = []

    if args.sequential:
        for name, cfg in groups.items():
            print(f"\n▶ Group: {name}")
            all_results.extend(run_group(name, cfg, args.verbose))
    else:
        # Run groups in parallel (bounded by group-level worker counts)
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = {pool.submit(run_group, name, cfg, args.verbose): name for name, cfg in groups.items()}
            for future in as_completed(futures):
                all_results.extend(future.result())

    elapsed_total = time.monotonic() - t_start
    ok = sum(1 for r in all_results if r["status"] == "OK")
    fails = [r for r in all_results if r["status"] != "OK"]

    print(f"\n{'='*60}")
    print(f"Done in {elapsed_total:.0f}s — {ok}/{len(all_results)} scripts succeeded")
    if fails:
        print(f"\nFailed ({len(fails)}):")
        for r in fails:
            print(f"  {r['status']:20s} {r['script']}")


if __name__ == "__main__":
    main()
