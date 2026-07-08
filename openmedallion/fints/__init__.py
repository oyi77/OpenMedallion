"""OpenMedallion Financial Time Series Forecasting Package

This package provides time-series forecasting models and utilities for financial data.
"""

# Models
from .models.lgbm_baseline import LGBMBaseline
from .models.patchtst import PatchTST

# Preprocessing
from .preprocessing.features import create_features
from .preprocessing.loader import load_data
from .preprocessing.splits import create_splits

# Evaluation
from .eval.metrics import calculate_metrics

__all__ = [
    'LGBMBaseline',
    'PatchTST',
    'create_features',
    'load_data',
    'create_splits',
    'calculate_metrics',
]

__version__ = '0.1.0'