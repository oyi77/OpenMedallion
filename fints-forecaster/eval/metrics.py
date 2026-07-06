"""
Evaluation utilities for FinTS.

All metrics are computed per split to expose regime sensitivity.
Never aggregate first — always show per-split degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def hit_rate(pred_direction: np.ndarray, true_direction: np.ndarray) -> float:
    """Fraction of correct directional predictions."""
    return float((pred_direction == true_direction).mean())


def directional_sharpe(
    pred_direction: np.ndarray,
    true_returns: np.ndarray,
    annual_factor: float = 252.0,
) -> float:
    """
    Annualized Sharpe of a long/short signal derived from predictions.
    Long when pred=1, short when pred=0.
    Returns NaN if std is zero or inputs are empty.
    """
    if len(pred_direction) == 0:
        return float("nan")
    pnl = np.where(pred_direction == 1, true_returns, -true_returns)
    std = pnl.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((pnl.mean() / std) * np.sqrt(annual_factor))


def max_drawdown(
    pred_direction: np.ndarray,
    true_returns: np.ndarray,
) -> float:
    """
    Max drawdown (negative number) of the simulated strategy.
    Uses compounding: (1 + r1)(1 + r2)...
    """
    if len(pred_direction) == 0:
        return float("nan")
    pnl = np.where(pred_direction == 1, true_returns, -true_returns)
    cum = np.cumprod(1.0 + pnl)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / (running_max + 1e-10)
    return float(dd.min())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# ---------------------------------------------------------------------------
# Per-split summary
# ---------------------------------------------------------------------------


@dataclass
class SplitMetrics:
    split_id: int
    asset_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_test: int
    hit_rate: float
    sharpe: float
    max_drawdown: float
    rmse: Optional[float] = None  # only for regression output


def summarize_splits(results: list) -> pd.DataFrame:
    """
    Convert a list of SplitResult or SplitMetrics to a DataFrame.
    One row per split. Columns: asset_id, split_id, hit_rate, sharpe, max_drawdown.
    """
    rows = []
    for r in results:
        row = {
            "asset_id": getattr(r, "asset_id", ""),
            "split_id": getattr(r, "split_id", 0),
            "hit_rate": getattr(r, "hit_rate", float("nan")),
            "sharpe": getattr(r, "sharpe", float("nan")),
            "max_drawdown": getattr(r, "max_drawdown", float("nan")),
            "val_rmse": getattr(r, "val_rmse", float("nan")),
            "test_rmse": getattr(r, "test_rmse", float("nan")),
            "direction_accuracy": getattr(r, "direction_accuracy", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def regime_degradation_check(
    per_split_df: pd.DataFrame,
    metric: str = "sharpe",
    threshold: float = 0.5,
) -> dict:
    """
    Check whether metric degrades monotonically across splits.
    Returns a dict with 'degrading': bool and 'note': str.

    A degrading pattern suggests regime sensitivity / overfitting.
    """
    vals = per_split_df.groupby("split_id")[metric].mean().sort_index().values
    if len(vals) < 2:
        return {"degrading": False, "note": "insufficient splits"}

    diffs = np.diff(vals)
    n_declining = (diffs < 0).sum()
    frac_declining = n_declining / len(diffs)

    if frac_declining >= threshold:
        note = (
            f"{metric} declines in {n_declining}/{len(diffs)} consecutive "
            f"splits — likely regime sensitivity"
        )
        return {"degrading": True, "note": note}
    return {"degrading": False, "note": f"{metric} stable across splits"}


def build_eval_report(
    per_split_df: pd.DataFrame,
    model_name: str,
    asset_class: str,
) -> dict:
    """
    Aggregate per-split metrics into a single eval report dict.
    The report is what goes into the model card.
    """
    report = {
        "model": model_name,
        "asset_class": asset_class,
        "n_assets": per_split_df["asset_id"].nunique(),
        "n_splits": per_split_df["split_id"].nunique(),
    }

    for metric in ("hit_rate", "sharpe", "max_drawdown"):
        vals = per_split_df[metric].dropna()
        report[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        report[f"{metric}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
        report[f"{metric}_by_split"] = (
            per_split_df.groupby("split_id")[metric].mean().to_dict()
        )

    in_sample_sharpe = per_split_df[per_split_df["split_id"] == 0]["sharpe"].mean()
    out_sample_sharpe = per_split_df[per_split_df["split_id"] > 0]["sharpe"].mean()
    report["in_vs_out_sample_sharpe"] = {
        "in_sample": float(in_sample_sharpe) if not np.isnan(in_sample_sharpe) else None,
        "out_sample": float(out_sample_sharpe) if not np.isnan(out_sample_sharpe) else None,
    }

    deg = regime_degradation_check(per_split_df)
    report["regime_sensitivity"] = deg

    return report
