"""
Train PatchTST transformer model for time-series forecasting.
Usage: python train_patchtst.py --asset_class crypto --model_dir models/ --push_to_hub
"""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from openmedallion.fints.preprocessing.loader import load_asset_class
from openmedallion.fints.preprocessing.features import build_features, make_target
from openmedallion.fints.preprocessing.splits import single_train_test_split
from openmedallion.fints.models.patchtst import PatchTSTForecaster
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


class TimeSeriesDataset(Dataset):
    """Dataset for PatchTST training."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, asset_class):
    """Save training checkpoint for resume capability."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_{asset_class}_epoch_{epoch}.pt"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint for resume."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Train Loss: {checkpoint['train_loss']:.6f}, Val Loss: {checkpoint['val_loss']:.6f}")
    
    return checkpoint['epoch'], checkpoint['val_loss']


def train_epoch(model, loader, optimizer, criterion, device, use_wandb=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(loader)
    
    if use_wandb:
        wandb.log({"train_loss": avg_loss})
    
    return avg_loss


def evaluate_epoch(model, loader, criterion, device, use_wandb=False):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            
            total_loss += loss.item()
            predictions.extend(y_pred.cpu().numpy().flatten())
            targets.extend(y_batch.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader)
    
    if use_wandb:
        wandb.log({"val_loss": avg_loss})
    
    return avg_loss, np.array(predictions), np.array(targets)


def main():
    parser = argparse.ArgumentParser(description='Train PatchTST model')
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
    parser.add_argument('--lookback', type=int, default=64,
                        help='Lookback window for sequences')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--min_rows', type=int, default=200,
                        help='Minimum rows required per file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Push trained model to HuggingFace Hub')
    parser.add_argument('--hub_username', type=str, default=None,
                        help='HuggingFace Hub username (required if --push_to_hub)')
    parser.add_argument('--hub_repo_name', type=str, default=None,
                        help='Custom Hub repo name (default: openmedallion-fints-{asset_class}-patchtst)')
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
            run_name = args.wandb_run_name or f"patchtst_{args.asset_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "asset_class": args.asset_class,
                    "lookback": args.lookback,
                    "test_split": args.test_split,
                    "min_rows": args.min_rows,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "patience": args.patience,
                }
            )
            print(f"Weights & Biases initialized: {args.wandb_project}/{run_name}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if args.use_wandb:
        wandb.log({"device": str(device)})
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training PatchTST - {args.asset_class.upper()}")
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
    
    if args.use_wandb:
        wandb.log({"num_files": len(data)})
    
    # Concatenate all DataFrames
    combined = pd.concat([df for _, df in data], ignore_index=False)
    combined = combined.sort_index()
    print(f"Combined dataset shape: {combined.shape}")
    
    # Make target BEFORE building features
    print(f"\nCreating target variable...")
    target = make_target(combined, col='close')
    
    # Build features
    print(f"Building features...")
    features = build_features(combined)
    
    # Combine features and target
    combined = pd.concat([features, target.rename('target')], axis=1)
    combined = combined.dropna()
    print(f"After feature engineering: {combined.shape}")
    
    if args.use_wandb:
        wandb.log({
            "dataset_rows": len(combined),
            "dataset_features": len(combined.columns) - 1,
        })
    
    # Split data
    print(f"\nSplitting data (train_ratio={1-args.test_split})...")
    train_df, test_df = single_train_test_split(combined, train_ratio=1-args.test_split)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    n_features = X_train.shape[1]
    model = PatchTSTForecaster(
        n_features=n_features,
        lookback=args.lookback
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params:,} parameters")
    
    if args.use_wandb:
        wandb.log({"num_parameters": num_params})
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from:
        try:
            start_epoch, best_val_loss = load_checkpoint(args.resume_from, model, optimizer)
            start_epoch += 1  # Resume from next epoch
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint: {e}")
            print("Starting fresh training...")
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs (starting from epoch {start_epoch})...")
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.use_wandb)
        val_loss, val_preds, val_targets = evaluate_epoch(model, test_loader, criterion, device, args.use_wandb)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if args.use_wandb:
            wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, checkpoint_dir, args.asset_class)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = model_dir / f"patchtst_{args.asset_class}_best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    print(f"\nLoading best model and evaluating...")
    model.load_state_dict(torch.load(best_model_path))
    _, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion, device, args.use_wandb)
    
    # Calculate metrics
    metrics = calculate_all_metrics(test_targets, test_preds)
    print_metrics_report(metrics, title=f"PatchTST {args.asset_class.upper()} - Test Set")
    
    if args.use_wandb:
        wandb.log({
            "test_mae": metrics.get("mae", 0),
            "test_rmse": metrics.get("rmse", 0),
            "test_mape": metrics.get("mape", 0),
            "test_r2": metrics.get("r2", 0),
        })
    
    # Save final model
    final_model_path = model_dir / f"patchtst_{args.asset_class}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save metrics
    metrics_path = model_dir / f"patchtst_{args.asset_class}_metrics.json"
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
        repo_name = args.hub_repo_name or f"openmedallion-fints-{args.asset_class}-patchtst"
        
        try:
            # Push model directory to Hub
            repo_url = push_to_hub(
                path=str(model_dir),
                repo_name=repo_name,
                username=args.hub_username,
                repo_type="model",
                commit_message=f"Upload PatchTST {args.asset_class} model",
                private=False
            )
            
            print(f"\nModel successfully pushed to Hub!")
            print(f"Repository: {repo_url}")
            
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
