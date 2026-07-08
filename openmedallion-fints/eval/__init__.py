"""
Evaluation module for OpenMedallion-FinTS.

Provides metrics for forecast accuracy and trading performance evaluation.
"""

from .metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    direction_accuracy,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    profit_factor,
    calculate_all_metrics,
    print_metrics_report
)

__all__ = [
    'mean_absolute_error',
    'root_mean_squared_error',
    'mean_absolute_percentage_error',
    'direction_accuracy',
    'hit_rate',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'profit_factor',
    'calculate_all_metrics',
    'print_metrics_report'
]
