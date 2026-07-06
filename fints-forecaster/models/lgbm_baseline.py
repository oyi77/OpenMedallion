"""
LightGBM baseline for FinTS.

Trains one model per asset class using walk-forward splits.
Predicts: direction (up/down) and continuous forward log-return.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from preprocessing.features import build_features, make_direction_target, make_target
from preprocessing.splits import walk_forward_splits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LGBM_PARAMS_DIRECTION = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

LGBM_PARAMS_REGRESSION = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Per-split result
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    split_id: int
    asset_id: str
    # Directional accuracy
    direction_accuracy: float
    direction_f1: float
    # Return metrics
    val_rmse: float
    test_rmse: float
    # Signal simulation
    sharpe: float
    max_drawdown: float
    hit_rate: float
    # Raw predictions for further analysis
    test_dates: list
    test_pred_direction: np.ndarray
    test_true_returns: np.ndarray


@dataclass
class AssetResult:
    asset_id: str
    asset_class: str
    splits: list[SplitResult] = field(default_factory=list)

    def mean_sharpe(self) -> float:
        vals = [s.sharpe for s in self.splits if not np.isnan(s.sharpe)]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_hit_rate(self) -> float:
        vals = [s.hit_rate for s in self.splits]
        return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _sharpe_from_signal(
    signal: np.ndarray, returns: np.ndarray, annual_factor: float = 252.0
) -> float:
    """
    Simple long/short: go long when signal=1, short when signal=0.
    Returns annualized Sharpe. Returns NaN if std is zero.
    """
    pnl = np.where(signal == 1, returns, -returns)
    std = pnl.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((pnl.mean() / std) * np.sqrt(annual_factor))


def _max_drawdown(signal: np.ndarray, returns: np.ndarray) -> float:
    """Max drawdown of the simulated long/short strategy."""
    pnl = np.where(signal == 1, returns, -returns)
    cum = np.cumprod(1 + pnl)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_asset(
    asset_id: str,
    asset_class: str,
    df: pd.DataFrame,
    horizon: int = 1,
    n_splits: int = 5,
    lookback_window: int = 20,
) -> Optional[AssetResult]:
    """
    Train LightGBM on a single asset, returning per-split metrics.
    Returns None if the asset has insufficient data.
    """
    feat = build_features(df, lookback_window=lookback_window)
    if len(feat) < 150:
        return None

    direction = make_direction_target(feat["log_ret"], horizon)
    fwd_ret = make_target(feat["log_ret"], horizon)

    # Align all series
    combined = feat.join(direction).join(fwd_ret).dropna()
    if len(combined) < 150:
        return None

    X = combined[feat.columns].values
    y_dir = combined[direction.name].values
    y_ret = combined[fwd_ret.name].values
    idx = combined.index

    splits = walk_forward_splits(idx, n_splits=n_splits)
    if not splits:
        return None

    result = AssetResult(asset_id=asset_id, asset_class=asset_class)

    for split in splits:
        tr, va, te = split.train_idx, split.val_idx, split.test_idx

        # Direction model
        clf = lgb.LGBMClassifier(**LGBM_PARAMS_DIRECTION)
        clf.fit(
            X[tr], y_dir[tr],
            eval_set=[(X[va], y_dir[va])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        pred_dir = clf.predict(X[te])

        # Regression model (for RMSE)
        reg = lgb.LGBMRegressor(**LGBM_PARAMS_REGRESSION)
        reg.fit(
            X[tr], y_ret[tr],
            eval_set=[(X[va], y_ret[va])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        val_pred = reg.predict(X[va])
        test_pred_ret = reg.predict(X[te])

        val_rmse = float(np.sqrt(np.mean((val_pred - y_ret[va]) ** 2)))
        test_rmse = float(np.sqrt(np.mean((test_pred_ret - y_ret[te]) ** 2)))

        # Direction metrics
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_dir[te], pred_dir)
        f1 = f1_score(y_dir[te], pred_dir, zero_division=0)

        # Signal simulation on actual returns
        true_rets = y_ret[te]
        sharpe = _sharpe_from_signal(pred_dir, true_rets)
        mdd = _max_drawdown(pred_dir, true_rets)
        hit = float((pred_dir == y_dir[te]).mean())

        result.splits.append(
            SplitResult(
                split_id=split.split_id,
                asset_id=asset_id,
                direction_accuracy=acc,
                direction_f1=f1,
                val_rmse=val_rmse,
                test_rmse=test_rmse,
                sharpe=sharpe,
                max_drawdown=mdd,
                hit_rate=hit,
                test_dates=list(idx[te].astype(str)),
                test_pred_direction=pred_dir,
                test_true_returns=true_rets,
            )
        )

    return result


def train_asset_class(
    asset_records: list,  # list of (asset_id, asset_class, df)
    horizon: int = 1,
    n_splits: int = 5,
) -> list[AssetResult]:
    """Train all assets in a class, collect results."""
    results = []
    for asset_id, asset_class, df in asset_records:
        res = train_asset(asset_id, asset_class, df, horizon=horizon, n_splits=n_splits)
        if res is not None:
            results.append(res)
    return results
