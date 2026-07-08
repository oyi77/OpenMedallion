"""
PatchTST transformer for time-series forecasting.
Based on "A Time Series is Worth 64 Words" (ICLR 2023).
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Convert time series into patches and embed them.
    
    Input: [batch, lookback, n_features]
    Output: [batch, n_patches, d_model]
    """
    
    def __init__(self, patch_len: int, stride: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Linear projection from patch to d_model
        self.proj = nn.Linear(patch_len * n_features, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, n_features]
            
        Returns:
            patches: [batch, n_patches, d_model]
        """
        batch_size, seq_len, n_features = x.shape
        
        # Unfold into patches: [batch, n_patches, patch_len, n_features]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2)  # [batch, n_patches, n_features, patch_len]
        
        # Flatten patch: [batch, n_patches, patch_len * n_features]
        patches = patches.reshape(batch_size, -1, self.patch_len * n_features)
        
        # Project to d_model: [batch, n_patches, d_model]
        return self.proj(patches)


class PatchTST(nn.Module):
    """
    PatchTST: Patch-based Transformer for Time Series Forecasting.
    
    Architecture:
    1. Patch embedding
    2. Positional encoding
    3. Transformer encoder
    4. Prediction head
    """
    
    def __init__(
        self,
        lookback: int = 64,
        n_features: int = 10,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        pred_len: int = 1
    ):
        """
        Args:
            lookback: Input sequence length
            n_features: Number of input features
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Transformer embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            pred_len: Prediction horizon (default 1 for next-period)
        """
        super().__init__()
        
        self.lookback = lookback
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        self.n_patches = (lookback - patch_len) // stride + 1
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, n_features)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, n_features]
            
        Returns:
            predictions: [batch, pred_len]
        """
        # Patch embedding: [batch, n_patches, d_model]
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoder: [batch, n_patches, d_model]
        x = self.transformer(x)
        
        # Prediction head: [batch, pred_len]
        return self.head(x)


class PatchTSTForecaster:
    """
    Wrapper for PatchTST model with training and inference utilities.
    """
    
    def __init__(
        self,
        lookback: int = 64,
        n_features: int = 10,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.lookback = lookback
        self.n_features = n_features
        
        self.model = PatchTST(
            lookback=lookback,
            n_features=n_features,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        ).to(device)
        
        self.scaler_mean = None
        self.scaler_std = None
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        early_stopping_patience: int = 5
    ):
        """
        Train PatchTST model.
        
        Args:
            X_train: Training sequences [n_samples, lookback * n_features]
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            early_stopping_patience: Stop if val loss doesn't improve
        """
        # Standardize features
        self.scaler_mean = X_train.mean()
        self.scaler_std = X_train.std()
        
        X_train_scaled = (X_train - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # Reshape to [n_samples, lookback, n_features]
        X_train_tensor = torch.FloatTensor(
            X_train_scaled.values.reshape(-1, self.lookback, self.n_features)
        ).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = (X_val - self.scaler_mean) / (self.scaler_std + 1e-8)
            X_val_tensor = torch.FloatTensor(
                X_val_scaled.values.reshape(-1, self.lookback, self.n_features)
            ).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))
            train_loss = 0.0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_X = X_train_tensor[batch_idx]
                batch_y = y_train_tensor[batch_idx]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(indices) / batch_size)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict next-period returns.
        
        Args:
            X: Feature matrix [n_samples, lookback * n_features]
            
        Returns:
            Predictions array
        """
        self.model.eval()
        
        # Standardize
        X_scaled = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        
        # Reshape and convert to tensor
        X_tensor = torch.FloatTensor(
            X_scaled.values.reshape(-1, self.lookback, self.n_features)
        ).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        return predictions
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        return self
