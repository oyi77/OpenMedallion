---
language:
- en
license: apache-2.0
library_name: pytorch
tags:
- finance
- time-series
- forecasting
- transformer
- patchtst
- deep-learning
- trading
datasets:
- custom
pipeline_tag: time-series-forecasting
model-index:
- name: OpenMedallion FinTS PatchTST
  results:
  - task:
      type: time-series-forecasting
      name: Financial Time-Series Forecasting
    metrics:
    - type: mae
      value: 0.XX
      name: Mean Absolute Error
    - type: rmse
      value: 0.XX
      name: Root Mean Squared Error
    - type: mape
      value: 0.XX
      name: Mean Absolute Percentage Error
---

# OpenMedallion FinTS PatchTST

## Model Description

**OpenMedallion FinTS PatchTST** is a Transformer-based deep learning model for financial time-series forecasting using the PatchTST (Patching Time Series Transformer) architecture. The model predicts next-day returns across multiple asset classes including **equities**, **forex**, **commodities**, and **cryptocurrencies**.

PatchTST is specifically designed for multivariate time-series forecasting and addresses the limitations of traditional Transformers by using patching to reduce computational complexity while maintaining long-range dependency modeling.

### Key Features

- 🧠 **Transformer Architecture**: Self-attention mechanism captures complex temporal patterns
- 📊 **Multivariate Support**: Models interactions between multiple time-series
- ⚡ **Efficient Patching**: Reduces sequence length via patching for faster training
- 🎯 **Channel Independence**: Models each variable independently for better generalization
- 🔄 **Walk-Forward Validation**: Prevents look-ahead bias with expanding window
- 💰 **Trading Ready**: Includes Sharpe ratio and max drawdown metrics
- 🚀 **GPU Accelerated**: Optimized for CUDA/MPS training and inference

## Architecture Overview

### PatchTST Components

```
Input Time Series (length L)
    ↓
Patching (stride S, patch length P)
    ↓
Linear Embedding + Positional Encoding
    ↓
Transformer Encoder (N layers)
    ├── Multi-Head Self-Attention
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connections
    ↓
Flattening + Linear Head
    ↓
Prediction (next H steps)
```

### Key Architectural Choices

- **Patching**: Reduces sequence length from L to L/S, enabling efficient processing
- **Channel Independence**: Each variable processed separately, reducing parameter coupling
- **Instance Normalization**: Normalizes each time-series independently
- **Reversible Instance Normalization**: Denormalizes predictions to original scale

## Supported Asset Classes

### 1. Equities
- **Coverage**: S&P 500 constituents
- **Data Source**: Yahoo Finance
- **Features**: OHLCV + technical indicators
- **Lookback**: 60 days
- **Prediction**: Next-day return

### 2. Forex
- **Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD
- **Data Source**: Yahoo Finance
- **Features**: OHLCV + carry signals
- **Lookback**: 60 days
- **Prediction**: Next-day return

### 3. Commodities
- **Coverage**: Gold, Silver, Oil, Natural Gas
- **Data Source**: Yahoo Finance
- **Features**: OHLCV + supply-demand
- **Lookback**: 60 days
- **Prediction**: Next-day return

### 4. Cryptocurrencies
- **Coverage**: Top 200 by market cap
- **Data Source**: CoinGecko + Yahoo Finance
- **Features**: OHLCV + on-chain metrics
- **Lookback**: 60 days
- **Prediction**: Next-day return

## Training Details

### Model Architecture

```python
PatchTST(
    seq_len=60,           # Input sequence length (days)
    pred_len=1,           # Prediction horizon (next day)
    patch_len=12,         # Patch size
    stride=12,            # Patch stride
    d_model=128,          # Model dimension
    n_heads=8,            # Attention heads
    e_layers=3,           # Encoder layers
    d_ff=256,             # Feed-forward dimension
    dropout=0.2,          # Dropout rate
    activation='gelu',    # Activation function
    num_features=10       # Number of input features
)
```

### Input Features (10 dimensions)

1. **Close Price** (normalized)
2. **Returns** (log returns)
3. **Volume** (normalized)
4. **High-Low Range** (normalized)
5. **RSI** (14-day)
6. **MACD** (12, 26, 9)
7. **Bollinger Band Position**
8. **ATR** (14-day, normalized)
9. **Volume MA Ratio**
10. **Day of Week** (cyclical encoding)

### Training Procedure

**Framework**: PyTorch 2.5.1  
**Optimizer**: AdamW  
**Loss Function**: Mean Squared Error (MSE)  
**Validation**: Walk-forward expanding window

#### Hyperparameters

```python
training_config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'lr_scheduler': 'ReduceLROnPlateau',
    'lr_patience': 5,
    'lr_factor': 0.5,
    'gradient_clip_norm': 1.0
}
```

#### Training Configuration

- **Train Period**: 2015-01-01 to 2023-12-31
- **Validation Split**: Walk-forward with 252-day (1 year) expanding window
- **Test Period**: 2024-01-01 to 2024-12-31
- **Normalization**: StandardScaler on training set per feature
- **Data Augmentation**: Time warping, magnitude warping (50% probability)
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM) / A100 (40GB VRAM)
- **Mixed Precision**: FP16 training enabled
- **Training Time**: ~4-6 hours per asset class

### Evaluation Metrics

#### Statistical Metrics

| Metric | Equities | Forex | Commodities | Crypto |
|--------|----------|-------|-------------|--------|
| **MAE** | 0.XX | 0.XX | 0.XX | 0.XX |
| **RMSE** | 0.XX | 0.XX | 0.XX | 0.XX |
| **MAPE** | 0.XX% | 0.XX% | 0.XX% | 0.XX% |
| **R²** | 0.XX | 0.XX | 0.XX | 0.XX |

#### Trading Metrics (Backtest)

| Metric | Equities | Forex | Commodities | Crypto |
|--------|----------|-------|-------------|--------|
| **Sharpe Ratio** | X.XX | X.XX | X.XX | X.XX |
| **Max Drawdown** | -X.XX% | -X.XX% | -X.XX% | -X.XX% |
| **Win Rate** | XX.X% | XX.X% | XX.X% | XX.X% |
| **Profit Factor** | X.XX | X.XX | X.XX | X.XX |

*Note: Update these metrics with your actual training results.*

#### Comparison vs LGBM Baseline

| Metric | PatchTST | LGBM | Improvement |
|--------|----------|------|-------------|
| **RMSE** | 0.XX | 0.XX | +X.X% |
| **Sharpe** | X.XX | X.XX | +X.X% |
| **Max DD** | -X.X% | -X.X% | +X.X% |

## Usage

### Installation

```bash
pip install torch>=2.0.0 pandas numpy scikit-learn yfinance
```

### Quick Start

```python
import torch
from openmedallion.hub import from_pretrained
from openmedallion.fints.models.patchtst import PatchTST

# Download model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-patchtst-equities",
    model_type="fints"
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatchTST.load_from_checkpoint(f"{model_path}/best_model.pth")
model.to(device)
model.eval()

# Prepare input (60 days × 10 features)
import numpy as np
from openmedallion.fints.preprocessing.features import FeatureEngineer

# Load your data (last 60 days)
df = pd.read_csv("your_data.csv")[-60:]

# Engineer features
fe = FeatureEngineer()
features = fe.transform(df)  # Shape: (60, 10)

# Convert to tensor
x = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0)  # (1, 60, 10)
x = x.to(device)

# Make prediction
with torch.no_grad():
    prediction = model(x)

print(f"Predicted next-day return: {prediction.item():.4f}")
```

### Batch Inference

```python
# Process multiple time-series
batch_features = []  # List of (60, 10) arrays

for df in dataframes:
    fe = FeatureEngineer()
    features = fe.transform(df[-60:])
    batch_features.append(features.values)

# Stack into batch
x_batch = torch.tensor(np.stack(batch_features), dtype=torch.float32)
x_batch = x_batch.to(device)

# Batch prediction
with torch.no_grad():
    predictions = model(x_batch)

print(f"Predicted returns: {predictions.squeeze().cpu().numpy()}")
```

### Training Your Own Model

```python
from openmedallion.fints.scripts.train_patchtst import train_patchtst
from openmedallion.fints.preprocessing.loader import load_asset_data

# Load data
data = load_asset_data(
    asset_class="equities",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2015-01-01",
    end_date="2024-12-31"
)

# Train model
model, metrics = train_patchtst(
    data=data,
    asset_class="equities",
    seq_len=60,
    pred_len=1,
    patch_len=12,
    stride=12,
    d_model=128,
    n_heads=8,
    e_layers=3,
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    device="cuda",
    output_dir="models/equities_patchtst",
    push_to_hub=True,
    hub_username="your-username",
    hub_repo_name="openmedallion-fints-patchtst-equities"
)

print(f"Test RMSE: {metrics['test_rmse']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Fine-Tuning from Checkpoint

```python
# Download pretrained model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-patchtst-equities",
    model_type="fints"
)

# Load model
model = PatchTST.load_from_checkpoint(f"{model_path}/best_model.pth")

# Fine-tune on new data
from openmedallion.fints.scripts.train_patchtst import fine_tune_patchtst

fine_tuned_model, metrics = fine_tune_patchtst(
    model=model,
    new_data=new_data,
    epochs=20,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    freeze_encoder=True   # Freeze encoder, only train head
)
```

## Model Interpretability

### Attention Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get attention weights from model
with torch.no_grad():
    outputs, attention_weights = model(x, return_attention=True)

# Plot attention map
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    attention_weights[0].cpu().numpy(),  # First head
    cmap='viridis',
    xticklabels=range(60),
    yticklabels=range(60),
    ax=ax
)
ax.set_title('Self-Attention Map (Layer 1, Head 1)')
ax.set_xlabel('Key Position (Days)')
ax.set_ylabel('Query Position (Days)')
plt.tight_layout()
plt.show()
```

### Feature Attribution

```python
from captum.attr import IntegratedGradients

# Initialize attribution method
ig = IntegratedGradients(model)

# Compute attributions
attributions = ig.attribute(x, target=0)

# Plot feature importance over time
feature_names = ['Close', 'Returns', 'Volume', 'HL Range', 'RSI', 
                 'MACD', 'BB Pos', 'ATR', 'Vol MA', 'DOW']

fig, ax = plt.subplots(figsize=(14, 6))
for i, name in enumerate(feature_names):
    ax.plot(attributions[0, :, i].cpu().numpy(), label=name, alpha=0.7)
ax.set_xlabel('Time Step (Days)')
ax.set_ylabel('Attribution Score')
ax.set_title('Feature Attribution Over Time')
ax.legend(loc='upper left', ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Limitations and Biases

### Known Limitations

- **Computational Cost**: Requires GPU for training; slower inference than LGBM
- **Data Requirements**: Needs longer sequences (60+ days) for good performance
- **Black Box**: Less interpretable than tree-based models
- **Overfitting Risk**: High capacity model requires careful regularization
- **Market Regime Changes**: May struggle with unprecedented market conditions
- **Fixed Window**: Requires exactly 60 days of input data

### Potential Biases

- **Survivorship Bias**: Training on current constituents excludes delisted companies
- **Look-Ahead Bias**: Carefully mitigated via walk-forward validation
- **Temporal Bias**: Recent patterns weighted more heavily due to sequential nature
- **Bull Market Bias**: Training period (2015-2024) predominantly bullish
- **Liquidity Bias**: May not generalize to illiquid assets

### Risk Mitigation

- Implement walk-forward validation to prevent look-ahead bias
- Use ensemble methods combining PatchTST + LGBM
- Set position size limits and stop-loss thresholds
- Monitor attention weights for anomalous patterns
- Retrain quarterly with latest data
- Combine with fundamental analysis and risk management

## Ethical Considerations

⚠️ **Important**: This model is provided for research and educational purposes. Trading decisions should never be based solely on automated predictions.

### Responsible Use Guidelines

1. **No Autonomous Trading**: Always require human oversight
2. **Risk Management**: Implement proper position sizing and stop-losses
3. **Backtesting Transparency**: Report realistic slippage and transaction costs
4. **Market Impact**: Consider order size relative to market liquidity
5. **Regulatory Compliance**: Ensure compliance with local trading regulations
6. **Model Monitoring**: Track performance degradation and retrain regularly
7. **Explainability**: Use attention visualization and feature attribution

## Citation

If you use this model in your research, please cite:

```bibtex
@software{openmedallion_fints_patchtst,
  title={OpenMedallion FinTS PatchTST: Financial Time-Series Forecasting with Transformers},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/OpenMedallion}
}
```

## References

- **PatchTST**: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730.
- **Transformer**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- **Time Series Forecasting**: Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. Philosophical Transactions of the Royal Society A, 379(2194), 20200209.

## Model Card Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/yourusername/OpenMedallion/issues).

## License

This model is released under the Apache 2.0 License. See [LICENSE](https://github.com/yourusername/OpenMedallion/blob/main/LICENSE) for details.

---

**Last Updated**: 2026-07-08  
**Model Version**: 1.0.0  
**OpenMedallion Version**: 1.0.0  
**PyTorch Version**: 2.5.1+
