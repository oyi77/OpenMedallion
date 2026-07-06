"""
Walk-forward / expanding-window time splits for FinTS.

Rule: NEVER shuffle. All splits are strictly temporal.
Each split yields (train_idx, val_idx, test_idx) as integer positional arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray   # positional indices into original DataFrame
    val_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    min_train_frac: float = 0.20,
) -> list[SplitWindow]:
    """
    Expanding-window walk-forward splits.

    The full timeline is divided so that each split has a test window of
    `test_frac` of total length, preceded by a val window of `val_frac`,
    with an expanding train window from t=0.

    Example with 5 splits over 1000 rows (test_frac=0.10, val_frac=0.10):
      split 0: train[0:200]  val[200:300]  test[300:400]
      split 1: train[0:300]  val[300:400]  test[400:500]
      ...
      split 4: train[0:600]  val[600:700]  test[700:800]

    Parameters
    ----------
    index : DatetimeIndex of the asset
    n_splits : number of walk-forward folds
    val_frac, test_frac : fraction of total length per window
    min_train_frac : minimum training fraction — splits that don't have
                     enough training data are skipped

    Returns
    -------
    List of SplitWindow (may be < n_splits if series is short)
    """
    n = len(index)
    test_len = max(1, int(n * test_frac))
    val_len = max(1, int(n * val_frac))
    step = test_len  # advance test window by one test_len each split

    splits: list[SplitWindow] = []
    for i in range(n_splits):
        test_end_pos = n - i * step
        test_start_pos = test_end_pos - test_len
        val_end_pos = test_start_pos
        val_start_pos = val_end_pos - val_len
        train_end_pos = val_start_pos

        if train_end_pos < int(n * min_train_frac):
            continue
        if test_start_pos <= 0 or val_start_pos <= 0:
            continue

        train_idx = np.arange(0, train_end_pos)
        val_idx = np.arange(val_start_pos, val_end_pos)
        test_idx = np.arange(test_start_pos, test_end_pos)

        splits.append(
            SplitWindow(
                split_id=i,
                train_start=index[0],
                train_end=index[train_end_pos - 1],
                val_start=index[val_start_pos],
                val_end=index[val_end_pos - 1],
                test_start=index[test_start_pos],
                test_end=index[test_end_pos - 1],
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        )

    # Return in chronological order (earliest test first)
    return list(reversed(splits))


def sequence_windows(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int = 60,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a feature matrix to produce (X, y) pairs for
    sequence models (PatchTST, LSTM, etc.).

    X[i] = features[i : i + lookback]       shape: (lookback, n_features)
    y[i] = targets[i + lookback + horizon - 1]

    Returns arrays ready for torch Dataset consumption.
    """
    n = len(features)
    xs, ys = [], []
    for start in range(n - lookback - horizon + 1):
        xs.append(features[start : start + lookback])
        ys.append(targets[start + lookback + horizon - 1])
    if not xs:
        return np.empty((0, lookback, features.shape[1])), np.empty(0)
    return np.stack(xs), np.array(ys)
