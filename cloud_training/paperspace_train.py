#!/usr/bin/env python3
"""
Paperspace Gradient training script for OpenMedallion.

Supports both FinSentiment (Qwen fine-tuning) and FinTS (LGBM time-series forecasting).
Automatically pushes trained models to HuggingFace Hub after training.

Usage:
    # FinSentiment training
    gradient jobs create \
        --container openmedallion:latest \
        --machineType A100 \
        --command "python cloud_training/paperspace_train.py --task finsentiment --hub-username <username>"
    
    # FinTS training (single asset class)
    gradient jobs create \
        --container openmedallion:latest \
        --machineType C7 \
        --command "python cloud_training/paperspace_train.py --task fints --asset-class equities --hub-username <username>"
    
    # FinTS training (all asset classes)
    gradient jobs create \
        --container openmedallion:latest \
        --machineType C7 \
        --command "python cloud_training/paperspace_train.py --task fints-all --hub-username <username>"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add openmedallion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openmedallion.hub import push_to_hub, setup_token

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_finsentiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "outputs/finsentiment",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 2e-4
) -> str:
    """
    Train FinSentiment model (Qwen fine-tuning).
    
    Args:
        model_name: Base model to fine-tune
        output_dir: Output directory for trained model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
    
    Returns:
        Path to trained model directory
    """
    logger.info("🚀 Starting FinSentiment training...")
    
    # Prepare sentiment dataset
    logger.info("Preparing sentiment dataset...")
    from openmedallion.finsentiment.prepare_sentiment_data import main as prepare_data
    prepare_data()
    
    # Fine-tune Qwen model
    logger.info(f"Fine-tuning {model_name}...")
    from openmedallion.finsentiment.fine_tune_qwen import main as train_qwen
    
    # Build training args
    train_args = [
        "--model_name", model_name,
        "--output_dir", output_dir,
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--learning_rate", str(learning_rate)
    ]
    
    # Override sys.argv for argparse in fine_tune_qwen.py
    original_argv = sys.argv.copy()
    sys.argv = ["fine_tune_qwen.py"] + train_args
    
    try:
        train_qwen()
    finally:
        sys.argv = original_argv
    
    logger.info(f"✅ FinSentiment training complete! Model saved to {output_dir}")
    return output_dir


def train_fints(
    asset_class: str = "equities",
    output_dir: str = "outputs/fints"
) -> str:
    """
    Train FinTS model (LGBM time-series forecasting).
    
    Args:
        asset_class: Asset class to train (equities, crypto, commodities, forex)
        output_dir: Output directory for trained model
    
    Returns:
        Path to trained model file
    """
    logger.info(f"🚀 Starting FinTS training for {asset_class}...")
    
    # Collect data
    logger.info(f"Collecting {asset_class} data...")
    if asset_class == "equities":
        from openmedallion.fints.data_collectors.yfinance_historical import main as collect_data
        collect_data()
    elif asset_class == "crypto":
        from openmedallion.fints.data_collectors.coingecko_top200_historical import main as collect_data
        collect_data()
    elif asset_class == "commodities":
        from openmedallion.fints.data_collectors.yfinance_crypto_historical import main as collect_data
        # Override sys.argv for argparse
        original_argv = sys.argv.copy()
        sys.argv = ["yfinance_crypto_historical.py", "--asset-type", "commodities"]
        try:
            collect_data()
        finally:
            sys.argv = original_argv
    elif asset_class == "forex":
        from openmedallion.fints.data_collectors.yfinance_historical import main as collect_data
        # Override sys.argv for argparse
        original_argv = sys.argv.copy()
        sys.argv = ["yfinance_historical.py", "--asset-type", "forex"]
        try:
            collect_data()
        finally:
            sys.argv = original_argv
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")
    
    # Train LGBM model
    logger.info(f"Training LGBM model for {asset_class}...")
    from openmedallion.fints.scripts.train_lgbm import main as train_lgbm
    
    # Override sys.argv for argparse
    original_argv = sys.argv.copy()
    sys.argv = ["train_lgbm.py", "--asset-class", asset_class]
    
    try:
        train_lgbm()
    finally:
        sys.argv = original_argv
    
    model_path = f"{output_dir}/{asset_class}_lgbm_regression.pkl"
    logger.info(f"✅ FinTS training complete! Model saved to {model_path}")
    return model_path


def train_all_fints(output_dir: str = "outputs/fints") -> list[str]:
    """
    Train FinTS models for all asset classes.
    
    Args:
        output_dir: Output directory for trained models
    
    Returns:
        List of paths to trained model files
    """
    asset_classes = ["equities", "crypto", "commodities", "forex"]
    model_paths = []
    
    for asset_class in asset_classes:
        try:
            model_path = train_fints(asset_class, output_dir)
            model_paths.append(model_path)
        except Exception as e:
            logger.error(f"Failed to train {asset_class}: {e}")
            # Continue with other asset classes
    
    return model_paths


def main():
    parser = argparse.ArgumentParser(description="Paperspace Gradient training for OpenMedallion")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["finsentiment", "fints", "fints-all"],
        help="Training task to run"
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        default="equities",
        choices=["equities", "crypto", "commodities", "forex"],
        help="Asset class for FinTS training (only for --task fints)"
    )
    parser.add_argument(
        "--hub-username",
        type=str,
        required=True,
        help="HuggingFace Hub username for model upload"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model for FinSentiment training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/<task>)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs for FinSentiment"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device for FinSentiment"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for FinSentiment"
    )
    
    args = parser.parse_args()
    
    # Setup HuggingFace token
    setup_token()
    
    # Set default output dir
    if args.output_dir is None:
        if args.task == "finsentiment":
            args.output_dir = "outputs/finsentiment"
        else:
            args.output_dir = "outputs/fints"
    
    # Run training
    try:
        if args.task == "finsentiment":
            model_path = train_finsentiment(
                model_name=args.model_name,
                output_dir=args.output_dir,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            # Push to Hub
            logger.info("📤 Pushing FinSentiment model to HuggingFace Hub...")
            repo_id = f"{args.hub_username}/openmedallion-finsentiment"
            push_to_hub(model_path, repo_id, model_type="finsentiment")
            logger.info(f"✅ Model pushed to {repo_id}")
            
        elif args.task == "fints":
            model_path = train_fints(
                asset_class=args.asset_class,
                output_dir=args.output_dir
            )
            
            # Push to Hub
            logger.info(f"📤 Pushing FinTS {args.asset_class} model to HuggingFace Hub...")
            repo_id = f"{args.hub_username}/openmedallion-fints-{args.asset_class}"
            push_to_hub(model_path, repo_id, model_type="fints")
            logger.info(f"✅ Model pushed to {repo_id}")
            
        elif args.task == "fints-all":
            model_paths = train_all_fints(output_dir=args.output_dir)
            
            # Push each model to Hub
            for model_path in model_paths:
                asset_class = Path(model_path).stem.replace("_lgbm_regression", "")
                logger.info(f"📤 Pushing FinTS {asset_class} model to HuggingFace Hub...")
                repo_id = f"{args.hub_username}/openmedallion-fints-{asset_class}"
                try:
                    push_to_hub(model_path, repo_id, model_type="fints")
                    logger.info(f"✅ Model pushed to {repo_id}")
                except Exception as e:
                    logger.error(f"Failed to push {asset_class} model: {e}")
        
        logger.info("🎉 Training and upload complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
