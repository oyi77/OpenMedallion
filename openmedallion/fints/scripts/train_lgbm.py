"""
Train LightGBM baseline model for time-series forecasting.
Usage: python train_lgbm.py --asset_class crypto --model_dir models/
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.loader import load_asset_class
from preprocessing.features import build_features, make_target
from preprocessing.splits import single_train_test_split
from models.lgbm_baseline import LGBMForecaster
from eval.metrics import calculate_all_metrics, print_metrics_report


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM baseline')
    parser.add_argument('--asset_class', type=str, required=True,
                        choices=['crypto', 'forex', 'commodities', 'equities'],
                        help='Asset class to train on')
    parser.add_argument('--data_dir', type=str, 
                        default='~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/006f38c73a17da4bd0953102713b6ea63356693d/data/training/ai/',
                        help='Root directory for parquet files')
    parser.add_argument('--model_dir', type=str, default='models/',
                        help='Directory to save trained models')
    parser.add_argument('--lookback', type=int, default=20,
                        help='Lookback window for features')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--min_rows', type=int, default=200,
                        help='Minimum rows required per file')
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type')
    
    args = parser.parse_args()
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training LightGBM Baseline - {args.asset_class.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    data_dir = Path(args.data_dir).expanduser()
    print(f"Loading {args.asset_class} data from {data_dir}...")
    data = load_asset_class(
        data_dir=data_dir,
        asset_class=args.asset_class,
        min_rows=args.min_rows
    )
    
    if not data:
        print(f"ERROR: No data loaded for {args.asset_class}")
        return
    
    print(f"Loaded {len(data)} files for {args.asset_class}")
    
    # Concatenate all DataFrames
    combined = pd.concat([df for _, df in data], ignore_index=False)
    combined = combined.sort_index()  # Sort by datetime index
    print(f"Combined dataset shape: {combined.shape}")
    
    # Make target BEFORE building features (features drops OHLCV columns)
    print(f"\nCreating target variable...")
    if args.task == 'regression':
        target = make_target(combined, col='close')
    else:
        from preprocessing.features import make_direction_target
        target = make_direction_target(combined, col='close')
    
    # Build features (function doesn't accept lookback parameter)
    print(f"Building features...")
    features = build_features(combined)
    
    # Combine features and target
    combined = pd.concat([features, target.rename('target')], axis=1)
    
    # Drop NaN rows from feature engineering
    combined = combined.dropna()
    print(f"After feature engineering: {combined.shape}")
    
    # Split data (function expects DataFrame with train_ratio, not X/y/test_size)
    print(f"\nSplitting data (train_ratio={1-args.test_split})...")
    train_df, test_df = single_train_test_split(combined, train_ratio=1-args.test_split)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train model
    print(f"\nTraining LightGBM {args.task} model...")
    model = LGBMForecaster(task=args.task)
    model.fit(X_train, y_train)
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    if args.task == 'regression':
        metrics = calculate_all_metrics(y_test.values, y_pred)
        print_metrics_report(metrics, title=f"LightGBM {args.asset_class.upper()} - Test Set")
    else:
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
    
    # Save model
    model_path = model_dir / f"lgbm_{args.asset_class}_{args.task}.pkl"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    if args.task == 'regression':
        metrics_path = model_dir / f"lgbm_{args.asset_class}_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
