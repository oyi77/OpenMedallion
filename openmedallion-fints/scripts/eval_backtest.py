#!/usr/bin/env python3
"""
Backtest evaluation for OpenMedallion-FinTS models.

Runs walk-forward or expanding window backtesting to evaluate model
performance on realistic out-of-sample data with temporal validation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from openmedallion_fints import (
    load_asset_class,
    build_features,
    make_target,
    make_direction_target,
    walk_forward_split,
    expanding_window_split,
    LGBMForecaster,
    calculate_all_metrics,
    print_metrics_report
)

warnings.filterwarnings('ignore')


def backtest_walk_forward(
    df: pd.DataFrame,
    model_type: str,
    lookback: int,
    horizon: int,
    window_size: int,
    step_size: int,
    **model_kwargs
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Walk-forward backtesting.
    
    Returns:
        predictions, actuals, fold_metrics
    """
    all_preds = []
    all_actuals = []
    fold_metrics = []
    
    splits = walk_forward_split(df, window_size=window_size, step_size=step_size)
    
    print(f"Running {len(splits)} walk-forward folds...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Backtesting")):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Feature engineering
        train_features = build_features(train_df, lookback=lookback)
        test_features = build_features(test_df, lookback=lookback)
        
        # Targets
        train_y = make_target(train_df, horizon=horizon)
        test_y = make_target(test_df, horizon=horizon)
        
        train_dir_y = make_direction_target(train_df, horizon=horizon)
        test_dir_y = make_direction_target(test_df, horizon=horizon)
        
        # Align features and targets
        train_features, train_y, train_dir_y = (
            train_features.iloc[lookback:].reset_index(drop=True),
            train_y.iloc[lookback:].reset_index(drop=True),
            train_dir_y.iloc[lookback:].reset_index(drop=True)
        )
        
        test_features, test_y, test_dir_y = (
            test_features.iloc[lookback:].reset_index(drop=True),
            test_y.iloc[lookback:].reset_index(drop=True),
            test_dir_y.iloc[lookback:].reset_index(drop=True)
        )
        
        # Drop NaN
        train_mask = ~(train_features.isna().any(axis=1) | train_y.isna() | train_dir_y.isna())
        test_mask = ~(test_features.isna().any(axis=1) | test_y.isna() | test_dir_y.isna())
        
        X_train = train_features[train_mask]
        y_train = train_y[train_mask]
        dir_train = train_dir_y[train_mask]
        
        X_test = test_features[test_mask]
        y_test = test_y[test_mask]
        dir_test = test_dir_y[test_mask]
        
        if len(X_train) < 50 or len(X_test) < 10:
            continue
        
        # Train model
        if model_type == 'lgbm':
            model = LGBMForecaster(task='regression', **model_kwargs)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Collect predictions
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        
        # Calculate fold metrics
        fold_met = calculate_all_metrics(y_test.values, preds, dir_test.values)
        fold_metrics.append({
            'fold': fold_idx,
            'train_size': len(X_train),
            'test_size': len(X_test),
            **fold_met
        })
    
    return np.array(all_preds), np.array(all_actuals), fold_metrics


def backtest_expanding_window(
    df: pd.DataFrame,
    model_type: str,
    lookback: int,
    horizon: int,
    initial_train: int,
    step_size: int,
    **model_kwargs
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Expanding window backtesting.
    
    Returns:
        predictions, actuals, fold_metrics
    """
    all_preds = []
    all_actuals = []
    fold_metrics = []
    
    splits = expanding_window_split(df, initial_train=initial_train, step_size=step_size)
    
    print(f"Running {len(splits)} expanding window folds...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="Backtesting")):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Feature engineering
        train_features = build_features(train_df, lookback=lookback)
        test_features = build_features(test_df, lookback=lookback)
        
        # Targets
        train_y = make_target(train_df, horizon=horizon)
        test_y = make_target(test_df, horizon=horizon)
        
        train_dir_y = make_direction_target(train_df, horizon=horizon)
        test_dir_y = make_direction_target(test_df, horizon=horizon)
        
        # Align features and targets
        train_features, train_y, train_dir_y = (
            train_features.iloc[lookback:].reset_index(drop=True),
            train_y.iloc[lookback:].reset_index(drop=True),
            train_dir_y.iloc[lookback:].reset_index(drop=True)
        )
        
        test_features, test_y, test_dir_y = (
            test_features.iloc[lookback:].reset_index(drop=True),
            test_y.iloc[lookback:].reset_index(drop=True),
            test_dir_y.iloc[lookback:].reset_index(drop=True)
        )
        
        # Drop NaN
        train_mask = ~(train_features.isna().any(axis=1) | train_y.isna() | train_dir_y.isna())
        test_mask = ~(test_features.isna().any(axis=1) | test_y.isna() | test_dir_y.isna())
        
        X_train = train_features[train_mask]
        y_train = train_y[train_mask]
        dir_train = train_dir_y[train_mask]
        
        X_test = test_features[test_mask]
        y_test = test_y[test_mask]
        dir_test = test_dir_y[test_mask]
        
        if len(X_train) < 50 or len(X_test) < 10:
            continue
        
        # Train model
        if model_type == 'lgbm':
            model = LGBMForecaster(task='regression', **model_kwargs)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Collect predictions
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        
        # Calculate fold metrics
        fold_met = calculate_all_metrics(y_test.values, preds, dir_test.values)
        fold_metrics.append({
            'fold': fold_idx,
            'train_size': len(X_train),
            'test_size': len(X_test),
            **fold_met
        })
    
    return np.array(all_preds), np.array(all_actuals), fold_metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest FinTS model')
    parser.add_argument('--asset-class', type=str, required=True,
                        choices=['crypto', 'forex', 'commodities', 'equities'],
                        help='Asset class to backtest')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Data directory root')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for backtest results')
    parser.add_argument('--model-type', type=str, default='lgbm',
                        choices=['lgbm'],
                        help='Model type (default: lgbm)')
    parser.add_argument('--backtest-mode', type=str, default='walk_forward',
                        choices=['walk_forward', 'expanding'],
                        help='Backtesting mode (default: walk_forward)')
    parser.add_argument('--lookback', type=int, default=20,
                        help='Lookback period for features (default: 20)')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecast horizon (default: 1)')
    parser.add_argument('--window-size', type=int, default=500,
                        help='Window size for walk-forward (default: 500)')
    parser.add_argument('--step-size', type=int, default=100,
                        help='Step size for walk-forward (default: 100)')
    parser.add_argument('--initial-train', type=int, default=500,
                        help='Initial training size for expanding window (default: 500)')
    parser.add_argument('--min-rows', type=int, default=100,
                        help='Minimum rows per file (default: 100)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees for LightGBM (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate for LightGBM (default: 0.05)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Max depth for LightGBM (default: 5)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("OPENMEDALLION-FINTS BACKTESTING")
    print("=" * 70)
    print(f"Asset class: {args.asset_class}")
    print(f"Data directory: {data_dir}")
    print(f"Model: {args.model_type}")
    print(f"Mode: {args.backtest_mode}")
    print(f"Lookback: {args.lookback}, Horizon: {args.horizon}")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.asset_class} data...")
    df = load_asset_class(data_dir, args.asset_class, min_rows=args.min_rows)
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Model kwargs
    model_kwargs = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth
    }
    
    # Run backtest
    if args.backtest_mode == 'walk_forward':
        predictions, actuals, fold_metrics = backtest_walk_forward(
            df, args.model_type, args.lookback, args.horizon,
            args.window_size, args.step_size, **model_kwargs
        )
    else:
        predictions, actuals, fold_metrics = backtest_expanding_window(
            df, args.model_type, args.lookback, args.horizon,
            args.initial_train, args.step_size, **model_kwargs
        )
    
    # Calculate overall metrics
    # Infer direction from predictions vs actuals
    pred_dir = (predictions > 0).astype(int)
    actual_dir = (actuals > 0).astype(int)
    
    overall_metrics = calculate_all_metrics(actuals, predictions, actual_dir)
    
    print("\n" + "=" * 70)
    print("OVERALL BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total predictions: {len(predictions)}")
    print(f"Number of folds: {len(fold_metrics)}")
    print_metrics_report(overall_metrics)
    
    # Save results
    results = {
        'asset_class': args.asset_class,
        'model_type': args.model_type,
        'backtest_mode': args.backtest_mode,
        'lookback': args.lookback,
        'horizon': args.horizon,
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'num_predictions': len(predictions),
        'num_folds': len(fold_metrics)
    }
    
    results_path = output_dir / f'backtest_results_{args.asset_class}_{args.model_type}.json'
    print(f"\nSaving backtest results to {results_path}...")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    preds_df = pd.DataFrame({
        'prediction': predictions,
        'actual': actuals,
        'pred_direction': pred_dir,
        'actual_direction': actual_dir
    })
    
    preds_path = output_dir / f'backtest_predictions_{args.asset_class}_{args.model_type}.csv'
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to {preds_path}")
    
    print("\nBacktesting complete!")


if __name__ == '__main__':
    main()
