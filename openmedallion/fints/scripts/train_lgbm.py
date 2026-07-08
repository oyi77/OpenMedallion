"""
Train LightGBM baseline model for time-series forecasting.
Usage: python train_lgbm.py --asset_class crypto --model_dir models/ --push_to_hub
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openmedallion.fints.preprocessing.loader import load_asset_class
from openmedallion.fints.preprocessing.features import build_features, make_target
from openmedallion.fints.preprocessing.splits import single_train_test_split
from openmedallion.fints.models.lgbm_baseline import LGBMForecaster
from openmedallion.fints.eval.metrics import calculate_all_metrics, print_metrics_report

# Optional imports for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from openmedallion.hub import push_to_hub, setup_token
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False


def save_checkpoint(model, metrics, checkpoint_dir, asset_class, epoch_or_step):
    """Save training checkpoint for resume capability."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_{asset_class}_{epoch_or_step}.pkl"
    model.save(str(checkpoint_path))
    
    metadata_path = checkpoint_dir / f"checkpoint_{asset_class}_{epoch_or_step}_metadata.json"
    metadata = {
        "asset_class": asset_class,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "epoch_or_step": epoch_or_step
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load training checkpoint for resume."""
    from openmedallion.fints.models.lgbm_baseline import LGBMForecaster
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = LGBMForecaster()
    model.load(str(checkpoint_path))
    
    metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Checkpoint metadata: {metadata}")
        return model, metadata
    
    return model, {}


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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--lookback', type=int, default=20,
                        help='Lookback window for features')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--min_rows', type=int, default=200,
                        help='Minimum rows required per file')
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Push trained model to HuggingFace Hub')
    parser.add_argument('--hub_username', type=str, default=None,
                        help='HuggingFace Hub username (required if --push_to_hub)')
    parser.add_argument('--hub_repo_name', type=str, default=None,
                        help='Custom Hub repo name (default: openmedallion-fints-{asset_class})')
    parser.add_argument('--wandb_project', type=str, default='openmedallion-fints',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Validate Hub push requirements
    if args.push_to_hub and not args.hub_username:
        parser.error("--hub_username is required when --push_to_hub is set")
    
    if args.push_to_hub and not HUB_AVAILABLE:
        print("WARNING: Hub utilities not available. Install with: pip install huggingface-hub")
        args.push_to_hub = False
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not available. Install with: pip install wandb")
            args.use_wandb = False
        else:
            run_name = args.wandb_run_name or f"lgbm_{args.asset_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "asset_class": args.asset_class,
                    "task": args.task,
                    "lookback": args.lookback,
                    "test_split": args.test_split,
                    "min_rows": args.min_rows,
                }
            )
            print(f"Weights & Biases initialized: {args.wandb_project}/{run_name}")
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Log data loading to wandb
    if args.use_wandb:
        wandb.log({"num_files": len(data)})
    
    # Concatenate all DataFrames
    combined = pd.concat([df for _, df in data], ignore_index=False)
    combined = combined.sort_index()  # Sort by datetime index
    print(f"Combined dataset shape: {combined.shape}")
    
    # Make target BEFORE building features (features drops OHLCV columns)
    print(f"\nCreating target variable...")
    if args.task == 'regression':
        target = make_target(combined, col='close')
    else:
        from openmedallion.fints.preprocessing.features import make_direction_target
        target = make_direction_target(combined, col='close')
    
    # Build features
    print(f"Building features...")
    features = build_features(combined)
    
    # Combine features and target
    combined = pd.concat([features, target.rename('target')], axis=1)
    
    # Drop NaN rows from feature engineering
    combined = combined.dropna()
    print(f"After feature engineering: {combined.shape}")
    
    # Log dataset stats to wandb
    if args.use_wandb:
        wandb.log({
            "dataset_rows": len(combined),
            "dataset_features": len(combined.columns) - 1,
            "target_mean": combined['target'].mean(),
            "target_std": combined['target'].std(),
        })
    
    # Split data
    print(f"\nSplitting data (train_ratio={1-args.test_split})...")
    train_df, test_df = single_train_test_split(combined, train_ratio=1-args.test_split)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Check for resume checkpoint
    model = None
    if args.resume_from:
        try:
            model, checkpoint_metadata = load_checkpoint(args.resume_from)
            print(f"Resumed from checkpoint: {args.resume_from}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint: {e}")
            print("Starting fresh training...")
    
    # Train model
    if model is None:
        print(f"\nTraining LightGBM {args.task} model...")
        model = LGBMForecaster(task=args.task)
        model.fit(X_train, y_train)
    else:
        print(f"Using loaded model from checkpoint")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    
    metrics = {}
    if args.task == 'regression':
        metrics = calculate_all_metrics(y_test.values, y_pred)
        print_metrics_report(metrics, title=f"LightGBM {args.asset_class.upper()} - Test Set")
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "test_mae": metrics.get("mae", 0),
                "test_rmse": metrics.get("rmse", 0),
                "test_mape": metrics.get("mape", 0),
                "test_r2": metrics.get("r2", 0),
            })
    else:
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"\nClassification Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        metrics = {"accuracy": accuracy, "f1_score": f1}
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "test_accuracy": accuracy,
                "test_f1_score": f1,
            })
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance_df = model.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
        
        # Log feature importance to wandb
        if args.use_wandb:
            wandb.log({
                "feature_importance": wandb.Table(dataframe=importance_df.head(10))
            })
    
    # Save checkpoint
    checkpoint_path = save_checkpoint(model, metrics, checkpoint_dir, args.asset_class, "final")
    
    # Save model
    model_path = model_dir / f"lgbm_{args.asset_class}_{args.task}.pkl"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics_path = model_dir / f"lgbm_{args.asset_class}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Push to Hub if requested
    if args.push_to_hub:
        print(f"\n{'='*60}")
        print("Pushing model to HuggingFace Hub...")
        print(f"{'='*60}\n")
        
        # Setup Hub token
        setup_token()
        
        # Determine repo name
        repo_name = args.hub_repo_name or f"openmedallion-fints-{args.asset_class}"
        
        try:
            # Push model directory to Hub
            repo_url = push_to_hub(
                path=str(model_dir),
                repo_name=repo_name,
                username=args.hub_username,
                repo_type="model",
                commit_message=f"Upload LightGBM {args.asset_class} {args.task} model",
                private=False
            )
            
            print(f"\nModel successfully pushed to Hub!")
            print(f"Repository: {repo_url}")
            
            # Log Hub URL to wandb
            if args.use_wandb:
                wandb.log({"hub_repo_url": repo_url})
        
        except Exception as e:
            print(f"ERROR: Failed to push to Hub: {e}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
