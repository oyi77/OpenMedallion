---
license: apache-2.0
language:
- en
tags:
- time-series
- forecasting
- finance
- trading
- lightgbm
- patchtst
- cryptocurrency
- forex
- commodities
- equities
pipeline_tag: time-series-forecasting
datasets:
- oyi77/OpenMedallion
---

# OpenMedallion-FinTS

**Time-Series Forecasting Models for Financial Markets**

> ⚠️ **CRITICAL DISCLAIMER**: These models are for **backtesting and research purposes only**. They are NOT financial advice and should NOT be used for live trading without extensive validation. Financial markets are non-stationary, and past performance does not guarantee future results.

## Model Description

OpenMedallion-FinTS provides production-ready time-series forecasting models trained on the [OpenMedallion dataset](https://huggingface.co/datasets/oyi77/OpenMedallion). The repository includes:

- **LightGBM Baseline**: Fast gradient-boosted decision trees for multi-step forecasting
- **PatchTST Transformer**: State-of-the-art patch-based transformer architecture for long-horizon forecasting

Models are trained separately per asset class (crypto, forex, commodities, equities) with strict temporal splitting to prevent data leakage.

## Intended Use

### Primary Use Cases
- **Research**: Academic studies on financial time-series forecasting
- **Backtesting**: Historical strategy validation with proper temporal splits
- **Baseline Models**: Starting point for custom trading system development
- **Educational**: Learning time-series forecasting techniques

### Out-of-Scope Use
- ❌ Live trading without extensive validation
- ❌ Financial advice or recommendations
- ❌ Production deployment without risk management
- ❌ Assuming stationarity across market regimes

## Model Architecture

### LightGBM Baseline
```python
from openmedallion_fints.models import LGBMForecaster

model = LGBMForecaster(
    task='regression',           # or 'classification'
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    early_stopping_rounds=50
)
```

**Features**:
- Supports both regression (price prediction) and classification (direction prediction)
- Early stopping with validation set
- Feature importance extraction
- Fast CPU training (~30 min per asset class)

### PatchTST Transformer
```python
from openmedallion_fints.models import PatchTSTForecaster

model = PatchTSTForecaster(
    lookback=64,      # Input sequence length
    horizon=1,        # Forecast horizon
    patch_len=16,     # Patch size
    stride=8,         # Patch stride
    d_model=128,      # Model dimension
    n_heads=4,        # Attention heads
    n_layers=3,       # Transformer layers
    d_ff=256,         # Feedforward dimension
    dropout=0.1
)
```

**Features**:
- Patch-based self-attention mechanism
- Efficient long-sequence modeling
- GPU-accelerated training (12GB VRAM)
- ~30 min per epoch on RTX 3060

## Training Data

### Dataset
- **Source**: [oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion)
- **Files**: 1,913 parquet files across 30 categories
- **OHLCV Categories**: crypto (59 files), equities (579), forex (30), commodities (32), indices (66), ETFs (208), bonds (42)
- **Temporal Range**: Varies by asset class (see dataset documentation)

### Temporal Splits
All models use **strict temporal splits** with NO random shuffling:

- **Walk-Forward Split**: Sliding window for robust validation
- **Expanding Window**: Growing training set (realistic production scenario)
- **Single Train/Test**: 80/20 chronological split

Example:
```python
from openmedallion_fints.preprocessing import walk_forward_split

splits = walk_forward_split(
    df=data,
    n_splits=5,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)
```

## Evaluation Metrics

### Forecast Accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Scale-independent error
- **Direction Accuracy**: Percentage of correct up/down predictions

### Trading-Specific Metrics
```python
from openmedallion_fints.eval import calculate_trading_metrics

metrics = calculate_trading_metrics(
    y_true=actual_returns,
    y_pred=predicted_returns,
    benchmark_returns=buy_hold_returns
)
# Returns: sharpe_ratio, sortino_ratio, max_drawdown, 
#          calmar_ratio, profit_factor, hit_rate
```

- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **Profit Factor**: Gross profit / Gross loss
- **Hit Rate**: Percentage of profitable trades

## Usage Example

### Training LightGBM Baseline
```bash
python openmedallion-fints/scripts/train_lgbm.py \
    --asset-class equities \
    --split-method expanding \
    --n-splits 5 \
    --train-size 0.7 \
    --val-size 0.15 \
    --test-size 0.15 \
    --task regression \
    --n-estimators 500 \
    --learning-rate 0.05 \
    --max-depth 7 \
    --early-stopping-rounds 50 \
    --output-dir ./outputs/lgbm_equities
```

### Training PatchTST
```bash
python openmedallion-fints/scripts/train_patchtst.py \
    --asset-class crypto \
    --split-method walk_forward \
    --lookback 64 \
    --horizon 1 \
    --patch-len 16 \
    --stride 8 \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 3 \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 0.001 \
    --device cuda \
    --output-dir ./outputs/patchtst_crypto
```

### Inference
```python
from openmedallion_fints.models import LGBMForecaster
from openmedallion_fints.preprocessing import compute_features
import pandas as pd

# Load trained model
model = LGBMForecaster.load("./outputs/lgbm_equities/model.pkl")

# Prepare features
df = pd.read_parquet("your_ohlcv_data.parquet")
X, y = compute_features(df, lookback=20, horizon=1)

# Forecast
predictions = model.predict(X)
```

## Limitations and Risks

### Model Limitations
1. **Non-Stationarity**: Financial markets are non-stationary; models trained on historical data may not generalize to future regimes
2. **Black Swan Events**: Models cannot predict unprecedented events (COVID-19, financial crises, regulatory changes)
3. **Liquidity**: Predictions assume sufficient liquidity for order execution
4. **Slippage**: Does not account for transaction costs, slippage, or market impact
5. **Regime Changes**: Performance degrades when market regime shifts (bull→bear, low→high volatility)

### Data Quality Risks
- **Survivorship Bias**: Dataset may exclude delisted/bankrupt assets
- **Look-Ahead Bias**: Ensure no future data leaks into training
- **Outliers**: Extreme events may distort model calibration

### Deployment Risks
- **Overfitting**: Models may overfit to historical patterns that don't repeat
- **Concept Drift**: Market dynamics change over time (new regulations, market structure, HFT)
- **Correlated Failures**: Models trained on same data may fail simultaneously during market stress

## Ethical Considerations

- **Market Manipulation**: Using these models for coordinated trading could constitute market manipulation
- **Systemic Risk**: Widespread use of similar models can amplify market volatility
- **Fairness**: Algorithmic trading advantages institutional players over retail traders
- **Transparency**: Black-box models lack interpretability for regulatory compliance

## License

**Apache License 2.0**

This model is released under the Apache License 2.0. You are free to use, modify, and distribute this model for commercial or non-commercial purposes, with proper attribution.

See [LICENSE](LICENSE) for full terms.

## Citation

```bibtex
@misc{openmedallion-fints-2026,
  author = {oyi77},
  title = {OpenMedallion-FinTS: Time-Series Forecasting for Financial Markets},
  year = {2026},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub},
  howpublished = {\url{https://huggingface.co/oyi77/openmedallion-fints}}
}
```

## Contact

- **Repository**: [https://huggingface.co/oyi77/openmedallion-fints](https://huggingface.co/oyi77/openmedallion-fints)
- **Dataset**: [https://huggingface.co/datasets/oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion)
- **Issues**: Report bugs and feature requests via HuggingFace discussions

## Acknowledgments

- **PatchTST**: Based on [PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
- **LightGBM**: Powered by [Microsoft LightGBM](https://github.com/microsoft/LightGBM)
- **Dataset**: Built on the OpenMedallion dataset

---

**Last Updated**: 2026-07-08
