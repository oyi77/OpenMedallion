"""
PatchTST — patch-based time-series transformer for FinTS.

Architecture per "A Time Series is Worth 64 Words" (Nie et al., 2023).
Trained from scratch on this GPU (RTX 2060 Super / 3060, 12GB VRAM).

Key design choices:
- Patch length = 16, stride = 8  → non-overlapping patches at lookback=64
- Channel-independence: each univariate (log_ret) treated separately,
  but feature channels concatenated as input dim
- Output: single scalar (directional logit) or forward log-return
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PatchTSTConfig:
    n_features: int = 9          # number of input features per timestep
    lookback: int = 64           # sequence length fed to model
    patch_len: int = 16          # tokens are patches of this length
    stride: int = 8              # patch stride (overlap = patch_len - stride)
    d_model: int = 128           # transformer hidden dim
    n_heads: int = 4             # attention heads
    n_layers: int = 3            # transformer encoder layers
    ffn_dim: int = 256           # feedforward dim
    dropout: float = 0.1
    output_dim: int = 1          # 1 for direction (binary) or return (regression)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """
    Divide a (B, T, C) sequence into patches and project each patch to d_model.
    """

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_patches = math.floor((config.lookback - config.patch_len) / config.stride) + 1
        patch_input_dim = config.patch_len * config.n_features
        self.proj = nn.Linear(patch_input_dim, config.d_model)
        self.pos_embed = nn.Embedding(self.n_patches, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, n_patches, d_model)"""
        B, T, C = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]          # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))  # (B, patch_len * C)
        patches_tensor = torch.stack(patches, dim=1)  # (B, n_patches, patch_len*C)
        embedded = self.proj(patches_tensor)           # (B, n_patches, d_model)
        pos = torch.arange(self.n_patches, device=x.device)
        embedded = embedded + self.pos_embed(pos)
        return embedded


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------


class PatchTST(nn.Module):
    """
    PatchTST model for time-series forecasting.

    Inputs:  (B, lookback, n_features) float32
    Outputs: (B, output_dim) float32
    """

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model * self.patch_embed.n_patches, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, lookback, n_features) → (B, output_dim)"""
        x = self.patch_embed(x)              # (B, n_patches, d_model)
        x = self.encoder(x)                  # (B, n_patches, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.flatten(1)                     # (B, n_patches * d_model)
        return self.head(x)                  # (B, output_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceDataset(torch.utils.data.Dataset):
    """Wraps numpy (X, y) arrays into a torch Dataset."""

    def __init__(self, X: "np.ndarray", y: "np.ndarray") -> None:
        import numpy as np
        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("float32"))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: PatchTST,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task: str = "direction",
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        if task == "direction":
            loss = criterion(logits, y_batch)
        else:
            loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: PatchTST,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str = "direction",
) -> tuple[float, "np.ndarray"]:
    import numpy as np
    model.eval()
    total_loss = 0.0
    all_preds = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        if task == "direction":
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
        else:
            preds = logits.cpu().numpy()
        all_preds.append(preds)
    preds_all = np.concatenate(all_preds)
    return total_loss / len(loader.dataset), preds_all


def fit_patchtst(
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    config: PatchTSTConfig,
    task: str = "direction",
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 7,
    device: "Optional[torch.device]" = None,
) -> tuple[PatchTST, dict]:
    """
    Train PatchTST and return (model, training_log).
    training_log keys: train_losses, val_losses, best_epoch
    """
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    model = PatchTST(config).to(device)

    if task == "direction":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, task)
        va_loss, _ = eval_epoch(model, val_loader, criterion, device, task)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch = epoch
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch if best_state else 0,
    }
