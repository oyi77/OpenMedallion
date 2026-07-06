#!/usr/bin/env python3
"""
train_lgbm.py — Train LightGBM baseline for FinTS.

Usage:
    python scripts/train_lgbm.py                          # all asset classes
    python scripts/train_lgbm.py --asset-class crypto     # one class
    python scripts/train_lgbm.py --max-per-class 10       # smoke test
    python scripts/train_lgbm.py --horizon 5 --splits 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# make repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from preprocessing.loader import iter_assets, ASSET_CLASS_GLOBS
from models.lgbm_baseline import train_asset
from eval.metrics import summarize_splits, build_eval_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asset-class", default=None, help="single asset class to train")
    p.add_argument("--max-per-class", type=int, default=None, help="cap files per class")
    p.add_argument("--horizon", type=int, default=1, help="forward prediction horizon (steps)")
    p.add_argument("--splits", type=int, default=5, help="number of walk-forward splits")
    p.add_argument("--output-dir", default="reports", help="where to write results")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_classes = (
        [args.asset_class]
        if args.asset_class
        else list(ASSET_CLASS_GLOBS.keys())
    )

    all_results = []
    class_reports = {}

    for ac in asset_classes:
        t0 = time.time()
        class_splits = []
        n_trained = 0
        n_skipped = 0

        for rec in iter_assets([ac], max_per_class=args.max_per_class):
            result = train_asset(
                rec.asset_id,
                rec.asset_class,
                rec.df,
                horizon=args.horizon,
                n_splits=args.splits,
            )
            if result is None:
                n_skipped += 1
                continue

            n_trained += 1
            for split in result.splits:
                all_results.append(split)
                class_splits.append(split)

            if args.verbose:
                print(
                    f"  {rec.asset_id[:50]:<50}  "
                    f"sharpe={result.mean_sharpe():+.3f}  "
                    f"hit={result.mean_hit_rate():.3f}"
                )

        if class_splits:
            df_class = summarize_splits(class_splits)
            report = build_eval_report(df_class, "lgbm-baseline", ac)
            report["n_trained"] = n_trained
            report["n_skipped"] = n_skipped
            report["elapsed_s"] = round(time.time() - t0, 1)
            class_reports[ac] = report

            print(
                f"[{ac:<12}]  assets={n_trained}  "
                f"mean_sharpe={report['sharpe_mean']:+.3f}  "
                f"mean_hit={report['hit_rate_mean']:.3f}  "
                f"time={report['elapsed_s']}s"
            )

            # per-split CSV for this class
            df_class.to_csv(out_dir / f"lgbm_{ac}_splits.csv", index=False)
        else:
            print(f"[{ac:<12}]  no qualifying assets (all skipped)")

    # --- aggregate report ---
    if all_results:
        df_all = summarize_splits(all_results)
        df_all.to_csv(out_dir / "lgbm_all_splits.csv", index=False)

        aggregate = {
            "model": "lgbm-baseline",
            "horizon": args.horizon,
            "n_splits": args.splits,
            "total_assets_trained": df_all["asset_id"].nunique(),
            "overall_mean_sharpe": float(df_all["sharpe"].mean()),
            "overall_mean_hit_rate": float(df_all["hit_rate"].mean()),
            "overall_mean_max_dd": float(df_all["max_drawdown"].mean()),
            "by_asset_class": class_reports,
        }

        report_path = out_dir / "lgbm_eval_report.json"
        with open(report_path, "w") as f:
            json.dump(aggregate, f, indent=2, default=str)
        print(f"\nEval report → {report_path}")
        print(f"Overall: sharpe={aggregate['overall_mean_sharpe']:+.3f}  "
              f"hit={aggregate['overall_mean_hit_rate']:.3f}")
    else:
        print("No results produced — check data paths.")


if __name__ == "__main__":
    main()
