#!/usr/bin/env python3
"""
eval_backtest.py — Run full backtest evaluation for any trained FinTS model.

Reads the split CSV outputs from train_lgbm.py / train_patchtst.py and
produces:
  - Per-split Sharpe, drawdown, hit-rate table
  - Regime sensitivity check (does performance degrade over time?)
  - In-sample vs out-of-sample Sharpe comparison
  - Final eval report JSON (goes into model card)

Usage:
    python scripts/eval_backtest.py --input reports/lgbm_all_splits.csv --model lgbm-baseline
    python scripts/eval_backtest.py --input reports/patchtst_crypto_splits.csv --model patchtst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from eval.metrics import (
    build_eval_report,
    regime_degradation_check,
    summarize_splits,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV of per-split results")
    p.add_argument("--model", required=True, help="model name for report")
    p.add_argument("--asset-class", default="all", help="label for report")
    p.add_argument("--output", default=None, help="output JSON path (default: same dir as input)")
    return p.parse_args()


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(index=False, float_format="{:.4f}".format))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run a training script first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    required = {"asset_id", "split_id", "hit_rate", "sharpe", "max_drawdown"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    # --- Per-split aggregate (mean across assets per split) ---
    per_split = (
        df.groupby("split_id")[["hit_rate", "sharpe", "max_drawdown"]]
        .mean()
        .reset_index()
        .rename(columns={
            "hit_rate": "mean_hit_rate",
            "sharpe": "mean_sharpe",
            "max_drawdown": "mean_max_dd",
        })
    )
    print_table(per_split, "Mean metrics per walk-forward split (chronological)")

    # Regime degradation
    deg = regime_degradation_check(df, metric="sharpe")
    print(f"\nRegime check: {deg['note']}")

    # In-sample vs out-of-sample
    split_ids = sorted(df["split_id"].unique())
    if len(split_ids) >= 2:
        first_split = split_ids[0]
        in_sample = df[df["split_id"] == first_split]["sharpe"].mean()
        out_sample = df[df["split_id"] != first_split]["sharpe"].mean()
        print(f"In-sample  (split {first_split}) mean Sharpe: {in_sample:+.4f}")
        print(f"Out-of-sample (splits {split_ids[1:]}) mean Sharpe: {out_sample:+.4f}")
        divergence = abs(in_sample - out_sample)
        if divergence > 0.5:
            print(f"  ⚠  Large in/out divergence ({divergence:.3f}) — likely overfit")

    # --- Per-asset summary ---
    per_asset = (
        df.groupby("asset_id")[["hit_rate", "sharpe", "max_drawdown"]]
        .mean()
        .sort_values("sharpe", ascending=False)
        .reset_index()
    )
    print_table(per_asset.head(20), "Top 20 assets by mean Sharpe")

    # --- Bottom assets ---
    print_table(per_asset.tail(10), "Bottom 10 assets by mean Sharpe")

    # --- Overall stats ---
    print(f"\n{'='*60}")
    print(f"  Overall [{args.model}]")
    print(f"{'='*60}")
    for col in ("hit_rate", "sharpe", "max_drawdown"):
        vals = df[col].dropna()
        print(
            f"  {col:<20}  mean={vals.mean():+.4f}  "
            f"std={vals.std():.4f}  "
            f"p25={vals.quantile(0.25):+.4f}  "
            f"p75={vals.quantile(0.75):+.4f}"
        )

    # --- Regime sensitivity table ---
    print(f"\n{'='*60}")
    print("  Sharpe by split (regime sensitivity check)")
    print(f"{'='*60}")
    sharpe_by_split = df.groupby("split_id")["sharpe"].agg(["mean", "std", "count"])
    print(sharpe_by_split.to_string(float_format="{:.4f}".format))

    # --- Build and save full report ---
    report = build_eval_report(df, args.model, args.asset_class)
    report["regime_degradation"] = deg

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = input_path.parent / (input_path.stem + "_eval_report.json")

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull eval report → {out_path}")


if __name__ == "__main__":
    main()
