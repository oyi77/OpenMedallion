#!/usr/bin/env python3
"""
train_patchtst.py — Train PatchTST for FinTS.

Usage:
    python scripts/train_patchtst.py --asset-class crypto
    python scripts/train_patchtst.py --asset-class crypto --max-per-class 5
    python scripts/train_patchtst.py --task regression --lookback 90 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from preprocessing.loader import iter_assets, ASSET_CLASS_GLOBS
from preprocessing.features import build_features, make_direction_target, make_target
from preprocessing.splits import walk_forward_splits, sequence_windows
from models.patchtst import PatchTST, PatchTSTConfig, fit_patchtst
from eval.metrics import (
    directional_sharpe,
    hit_rate,
    max_drawdown,
    summarize_splits,
    build_eval_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asset-class", default="crypto")
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--task", choices=["direction", "regression"], default="direction")
    p.add_argument("--lookback", type=int, default=64)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _train_one_asset(
    asset_id: str,
    asset_class: str,
    df,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    """Train PatchTST on one asset, return list of per-split result dicts."""
    feat = build_features(df, lookback_window=20)
    if len(feat) < args.lookback + 50:
        return []

    if args.task == "direction":
        target = make_direction_target(feat["log_ret"], args.horizon)
    else:
        target = make_target(feat["log_ret"], args.horizon)

    combined = feat.join(target).dropna()
    if len(combined) < args.lookback + 50:
        return []

    X_full = combined[feat.columns].values  # (N, n_features)
    y_full = combined[target.name].values
    idx = combined.index

    wf_splits = walk_forward_splits(idx, n_splits=args.splits)
    if not wf_splits:
        return []

    n_features = X_full.shape[1]
    config = PatchTSTConfig(
        n_features=n_features,
        lookback=args.lookback,
        d_model=args.d_model,
        n_layers=args.n_layers,
        output_dim=1,
    )

    results = []
    for split in wf_splits:
        # Build sequences for each partition
        X_tr, y_tr = sequence_windows(
            X_full[split.train_idx], y_full[split.train_idx],
            lookback=args.lookback, horizon=1,
        )
        X_va, y_va = sequence_windows(
            X_full[split.val_idx], y_full[split.val_idx],
            lookback=args.lookback, horizon=1,
        )
        X_te, y_te = sequence_windows(
            X_full[split.test_idx], y_full[split.test_idx],
            lookback=args.lookback, horizon=1,
        )

        if len(X_tr) < 20 or len(X_va) < 5 or len(X_te) < 5:
            continue

        model, log = fit_patchtst(
            X_tr, y_tr, X_va, y_va,
            config=config,
            task=args.task,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        # Predict on test
        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_te.astype("float32")).to(device)
            logits = model(X_te_t).squeeze(-1).cpu().numpy()

        if args.task == "direction":
            pred_dir = (logits > 0).astype(int)
            true_dir = y_te.astype(int)
            # Get true returns from the test window
            # y_te is direction labels; recompute returns from features
            true_rets = X_full[split.test_idx][args.lookback - 1 : len(X_full[split.test_idx])][:, 0]
            true_rets = true_rets[: len(pred_dir)]
        else:
            pred_cont = logits
            pred_dir = (pred_cont > 0).astype(int)
            true_dir = (y_te > 0).astype(int)
            true_rets = y_te

        hr = hit_rate(pred_dir, true_dir)
        sh = directional_sharpe(pred_dir, true_rets)
        mdd = max_drawdown(pred_dir, true_rets)

        results.append({
            "asset_id": asset_id,
            "split_id": split.split_id,
            "hit_rate": hr,
            "sharpe": sh,
            "max_drawdown": mdd,
            "direction_accuracy": hr,
            "val_rmse": float("nan"),
            "test_rmse": float("nan"),
            "best_epoch": log["best_epoch"],
        })

        if args.verbose:
            print(
                f"    split {split.split_id}  "
                f"hit={hr:.3f}  sharpe={sh:+.3f}  mdd={mdd:.3f}  "
                f"best_epoch={log['best_epoch']}"
            )

    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_split_rows = []
    n_trained = 0

    for rec in iter_assets([args.asset_class], max_per_class=args.max_per_class):
        t0 = time.time()
        split_results = _train_one_asset(
            rec.asset_id, rec.asset_class, rec.df, args, device
        )
        if not split_results:
            continue

        n_trained += 1
        all_split_rows.extend(split_results)
        mean_sharpe = float(np.mean([r["sharpe"] for r in split_results if not np.isnan(r["sharpe"])]))
        mean_hit = float(np.mean([r["hit_rate"] for r in split_results]))
        elapsed = time.time() - t0

        print(
            f"  {rec.asset_id[:50]:<50}  "
            f"sharpe={mean_sharpe:+.3f}  hit={mean_hit:.3f}  "
            f"splits={len(split_results)}  {elapsed:.1f}s"
        )

    if not all_split_rows:
        print("No results — check data.")
        return

    import pandas as pd
    df_all = pd.DataFrame(all_split_rows)
    df_all.to_csv(out_dir / f"patchtst_{args.asset_class}_splits.csv", index=False)

    report = build_eval_report(df_all, "patchtst", args.asset_class)
    report["n_assets_trained"] = n_trained
    report["task"] = args.task
    report["lookback"] = args.lookback
    report["horizon"] = args.horizon

    report_path = out_dir / f"patchtst_{args.asset_class}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nPatchTST [{args.asset_class}]  "
          f"assets={n_trained}  "
          f"mean_sharpe={report['sharpe_mean']:+.3f}  "
          f"mean_hit={report['hit_rate_mean']:.3f}")
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
