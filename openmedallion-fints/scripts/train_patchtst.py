"""
Train PatchTST transformer model for time-series forecasting.
Usage: python train_patchtst.py --asset_class crypto --model_dir models/
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.loader import load_asset_class
from preprocessing.features import build_features, make_target
from preprocessing.splits import single_train_test_split
from models.patchtst import PatchTSTForecaster
from eval.metrics import calculate_all_metrics, print_metrics_report


class TimeSeriesDataset(Dataset):
    """Dataset for PatchTST training."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate_epoch(model, loader, criterion, device):
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
    
    return total_loss / len(loader), np.array(predictions), np.array(targets)


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
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training PatchTST - {args.asset_class.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading {args.asset_class} data from {args.data_dir}...")
    df = load_asset_class(
        args.asset_class,
        data_dir=args.data_dir,
        min_rows=args.min_rows
    )
    
    if df is None or len(df) == 0:
        print(f"ERROR: No data loaded for {args.asset_class}")
        return
    
    print(f"Loaded {len(df)} rows")
    
    # Build features
    print(f"\nBuilding features with lookback={args.lookback}...")
    X = build_features(df, lookback=args.lookback)
    y = make_target(df.iloc[args.lookback:], target_col='close')
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    print(f"\nSplitting data (test={args.test_split})...")
    X_train, X_test, y_train, y_test = single_train_test_split(
        X, y, test_size=args.test_split
    )
    
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
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_targets = evaluate_epoch(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = model_dir / f"patchtst_{args.asset_class}_best.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    print(f"\nLoading best model and evaluating...")
    model.load_state_dict(torch.load(best_model_path))
    _, test_preds, test_targets = evaluate_epoch(model, test_loader, criterion, device)
    
    # Calculate metrics
    metrics = calculate_all_metrics(test_targets, test_preds)
    print_metrics_report(metrics, title=f"PatchTST {args.asset_class.upper()} - Test Set")
    
    # Save final model
    final_model_path = model_dir / f"patchtst_{args.asset_class}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save metrics
    metrics_path = model_dir / f"patchtst_{args.asset_class}_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
