"""
Temporal walk-forward splitting for time-series validation.
No random shuffle — strict chronological order only.
"""
import pandas as pd
from typing import List, Tuple
from datetime import datetime


def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.
    
    Args:
        df: DataFrame with datetime index (sorted chronologically)
        train_size: Number of rows for training window
        test_size: Number of rows for test window
        step: Step size between splits (default: test_size)
        
    Returns:
        List of (train_df, test_df) tuples
        
    Example:
        If train_size=252, test_size=63 (1 year train, 1 quarter test):
        Split 1: rows 0-251 train, 252-314 test
        Split 2: rows 63-314 train, 315-377 test
        ...
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted chronologically")
    
    if step is None:
        step = test_size
    
    splits = []
    total_rows = len(df)
    
    start = 0
    while start + train_size + test_size <= total_rows:
        train_end = start + train_size
        test_end = train_end + test_size
        
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        splits.append((train_df, test_df))
        
        start += step
    
    return splits


def expanding_window_split(
    df: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding window splits (train set grows over time).
    
    Args:
        df: DataFrame with datetime index (sorted chronologically)
        initial_train_size: Initial training window size
        test_size: Test window size
        step: Step size between splits (default: test_size)
        
    Returns:
        List of (train_df, test_df) tuples
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted chronologically")
    
    if step is None:
        step = test_size
    
    splits = []
    total_rows = len(df)
    
    train_end = initial_train_size
    while train_end + test_size <= total_rows:
        test_end = train_end + test_size
        
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        splits.append((train_df, test_df))
        
        train_end += step
    
    return splits


def single_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split maintaining temporal order.
    
    Args:
        df: DataFrame with datetime index (sorted chronologically)
        train_ratio: Fraction of data for training
        
    Returns:
        (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted chronologically")
    
    split_idx = int(len(df) * train_ratio)
    
    return df.iloc[:split_idx], df.iloc[split_idx:]
