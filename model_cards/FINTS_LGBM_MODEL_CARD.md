---
language:
- en
license: apache-2.0
library_name: lightgbm
tags:
- finance
- time-series
- forecasting
- lgbm
- gradient-boosting
- trading
datasets:
- custom
pipeline_tag: tabular-regression
model-index:
- name: OpenMedallion FinTS LGBM
  results:
  - task:
      type: tabular-regression
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

# OpenMedallion FinTS LGBM

## Model Description

**OpenMedallion FinTS LGBM** is a LightGBM-based gradient boosting model for financial time-series forecasting. The model predicts next-day returns across multiple asset classes including **equities**, **forex**, **commodities**, and **cryptocurrencies**.

This model serves as the baseline in the [OpenMedallion](https://github.com/yourusername/OpenMedallion) project and provides a strong benchmark for more complex deep learning architectures.

### Key Features

- 🎯 **Multi-Asset Coverage**: Supports equities, forex, commodities, and crypto
- ⚡ **Fast Training**: Trains in minutes on CPU
- 📊 **Rich Feature Engineering**: 50+ technical indicators and market features
- 🔄 **Walk-Forward Validation**: Prevents look-ahead bias with expanding window
- 💰 **Trading Ready**: Includes Sharpe ratio and max drawdown metrics
- 🚀 **Production Optimized**: Low-latency inference suitable for live trading

## Supported Asset Classes

### 1. Equities
- **Coverage**: S&P 500 constituents
- **Data Source**: Yahoo Finance
- **Features**: Price/volume OHLCV, RSI, MACD, Bollinger Bands, ATR
- **Frequency**: Daily

### 2. Forex
- **Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD
- **Data Source**: Yahoo Finance
- **Features**: Bid-ask spread, carry trade signals, central bank rates
- **Frequency**: Daily

### 3. Commodities
- **Coverage**: Gold (GC=F), Silver (SI=F), Oil (CL=F), Natural Gas (NG=F)
- **Data Source**: Yahoo Finance
- **Features**: Supply-demand indicators, seasonality, inventory levels
- **Frequency**: Daily

### 4. Cryptocurrencies
- **Coverage**: Top 200 by market cap (Bitcoin, Ethereum, etc.)
- **Data Source**: CoinGecko API + Yahoo Finance
- **Features**: On-chain metrics, social sentiment, volatility measures
- **Frequency**: Daily

## Training Details

### Feature Engineering

The model uses 50+ engineered features across multiple categories:

#### Price-Based Features
- Returns: 1-day, 5-day, 20-day, 60-day
- Log returns and percentage changes
- High-low range, close-open spread
- Overnight gaps

#### Technical Indicators
- **Momentum**: RSI (14, 20, 30), Stochastic Oscillator, Williams %R
- **Trend**: SMA (10, 20, 50, 200), EMA (12, 26), MACD
- **Volatility**: Bollinger Bands (20, 2σ), ATR (14), Historical Volatility
- **Volume**: Volume MA, Volume Rate of Change, OBV

#### Market Microstructure
- Bid-ask spread (forex)
- Volume-weighted average price
- Intraday volatility
- Trading session indicators

#### Cross-Asset Features
- Market regime indicators
- Sector/asset class correlations
- VIX index (for equities)
- Dollar strength index (for forex)

### Training Procedure

**Algorithm**: LightGBM Gradient Boosting  
**Objective**: Regression (predict next-day return)  
**Validation**: Walk-forward expanding window

#### Hyperparameters

```python
lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'max_depth': -1,
    'num_iterations': 1000,
    'early_stopping_rounds': 50,
    'verbose': -1
}
```

#### Training Configuration

- **Train Period**: 2015-01-01 to 2023-12-31
- **Validation Split**: Walk-forward with 252-day (1 year) expanding window
- **Test Period**: 2024-01-01 to 2024-12-31
- **Missing Data**: Forward-fill with 5-day limit, then drop
- **Normalization**: StandardScaler on training set, applied to val/test

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

## Usage

### Installation

```bash
pip install lightgbm pandas numpy scikit-learn yfinance
```

### Quick Start

```python
import lightgbm as lgb
from openmedallion.hub import from_pretrained

# Download model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-lgbm-equities",
    model_type="fints"
)

# Load model
model = lgb.Booster(model_file=f"{model_path}/lgbm_model.txt")

# Prepare features (50+ engineered features)
import pandas as pd
from openmedallion.fints.preprocessing.features import FeatureEngineer

# Load your data
df = pd.read_csv("your_data.csv")

# Engineer features
fe = FeatureEngineer()
features = fe.transform(df)

# Make predictions
predictions = model.predict(features)
print(f"Predicted next-day return: {predictions[-1]:.4f}")
```

### Feature Engineering Pipeline

```python
from openmedallion.fints.preprocessing.features import FeatureEngineer

# Initialize feature engineer
fe = FeatureEngineer(
    price_windows=[1, 5, 20, 60],
    ma_windows=[10, 20, 50, 200],
    volatility_windows=[10, 20, 30],
    rsi_periods=[14, 20, 30]
)

# Transform raw OHLCV data
raw_data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

features = fe.transform(raw_data)
print(f"Generated {len(features.columns)} features")
```

### Training Your Own Model

```python
from openmedallion.fints.scripts.train_lgbm import train_lgbm
from openmedallion.fints.preprocessing.loader import load_asset_data

# Load data
data = load_asset_data(
    asset_class="equities",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2015-01-01",
    end_date="2024-12-31"
)

# Train model
model, metrics = train_lgbm(
    data=data,
    asset_class="equities",
    output_dir="models/equities_lgbm",
    push_to_hub=True,
    hub_username="your-username",
    hub_repo_name="openmedallion-fints-lgbm-equities"
)

print(f"Test RMSE: {metrics['test_rmse']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Backtest Evaluation

```python
from openmedallion.fints.scripts.eval_backtest import run_backtest

# Run backtest
results = run_backtest(
    model_path="models/equities_lgbm/lgbm_model.txt",
    test_data=test_df,
    initial_capital=100000,
    position_size=0.1,
    transaction_cost=0.001
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

## Model Interpretability

### Feature Importance

LightGBM provides built-in feature importance metrics:

```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

# Plot top 20 features
top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]
names, scores = zip(*top_features)

plt.figure(figsize=(10, 8))
plt.barh(names, scores)
plt.xlabel('Feature Importance (Gain)')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.show()
```

### SHAP Analysis

```python
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# Plot summary
shap.summary_plot(shap_values, features, max_display=20)
```

## Limitations and Biases

### Known Limitations

- **Linear Relationships**: LGBM captures non-linear patterns but may miss complex temporal dependencies
- **No Sequence Modeling**: Treats each time step independently; doesn't model long-term dependencies
- **Feature Engineering Required**: Performance depends heavily on manual feature engineering
- **Market Regime Changes**: May underperform during unprecedented market conditions
- **Data Quality**: Sensitive to missing data and outliers

### Potential Biases

- **Survivorship Bias**: Training on current constituents excludes delisted companies
- **Look-Ahead Bias**: Carefully mitigated via walk-forward validation
- **Overfitting Risk**: High-capacity model requires proper regularization
- **Bull Market Bias**: Training period (2015-2024) predominantly bullish

### Risk Mitigation

- Implement walk-forward validation to prevent look-ahead bias
- Use ensemble methods combining multiple models
- Set position size limits and stop-loss thresholds
- Monitor performance degradation and retrain regularly
- Combine with fundamental analysis and risk management

## Ethical Considerations

⚠️ **Important**: This model is provided for research and educational purposes. Trading decisions should never be based solely on automated predictions.

### Responsible Use Guidelines

1. **No Autonomous Trading**: Always require human oversight
2. **Risk Management**: Implement proper position sizing and stop-losses
3. **Backtesting Transparency**: Report realistic slippage and transaction costs
4. **Market Impact**: Consider order size relative to market liquidity
5. **Regulatory Compliance**: Ensure compliance with local trading regulations

## Citation

If you use this model in your research, please cite:

```bibtex
@software{openmedallion_fints_lgbm,
  title={OpenMedallion FinTS LGBM: Financial Time-Series Forecasting with LightGBM},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/OpenMedallion}
}
```

## References

- **LightGBM**: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.
- **Technical Analysis**: Murphy, J. J. (1999). Technical analysis of the financial markets: A comprehensive guide to trading methods and applications. Penguin.

## Model Card Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/yourusername/OpenMedallion/issues).

## License

This model is released under the Apache 2.0 License. See [LICENSE](https://github.com/yourusername/OpenMedallion/blob/main/LICENSE) for details.

---

**Last Updated**: 2026-07-08  
**Model Version**: 1.0.0  
**OpenMedallion Version**: 1.0.0
