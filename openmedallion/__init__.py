"""
OpenMedallion - Unified Financial AI Package

This package provides components for:
- Financial time-series forecasting (fints)
- Financial sentiment analysis (finsentiment)
- Hugging Face Hub integration (hub)
"""

from openmedallion import fints, finsentiment

__version__ = "0.1.0"

__all__ = [
    "fints",
    "finsentiment",
    "__version__",
]
