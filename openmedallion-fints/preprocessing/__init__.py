"""
Data preprocessing module for OpenMedallion-FinTS.
"""

from .loader import load_parquet_file, load_asset_class
from .features import (
    log_returns,
    rolling_volatility,
    rsi,
    macd,
    volume_zscore,
    lag_returns,
    build_features,
    make_target,
    make_direction_target,
)
from .splits import (
    walk_forward_split,
    expanding_window_split,
    single_train_test_split,
)

__all__ = [
    'load_parquet_file',
    'load_asset_class',
    'log_returns',
    'rolling_volatility',
    'rsi',
    'macd',
    'volume_zscore',
    'lag_returns',
    'build_features',
    'make_target',
    'make_direction_target',
    'walk_forward_split',
    'expanding_window_split',
    'single_train_test_split',
]
