"""
OpenMedallion-FinTS: Time-series forecasting for financial markets.
"""

__version__ = "0.1.0"

from .preprocessing import load_asset_class, build_features, make_target, make_direction_target
from .preprocessing import walk_forward_split, expanding_window_split, single_train_test_split
from .models import LGBMForecaster, PatchTSTForecaster
from .eval import calculate_all_metrics, print_metrics_report

__all__ = [
    'load_asset_class',
    'build_features',
    'make_target',
    'make_direction_target',
    'walk_forward_split',
    'expanding_window_split',
    'single_train_test_split',
    'LGBMForecaster',
    'PatchTSTForecaster',
    'calculate_all_metrics',
    'print_metrics_report',
]
