---
license: mit
tags:
- time-series
- forecasting
- financial-markets
- {asset_class}
- openmedallion
library_name: {library_name}
pipeline_tag: time-series-forecasting
---

# OpenMedallion FinTS: {model_name}

## Model Description

This is a **{model_type}** model trained for {asset_class} price forecasting as part of the OpenMedallion project. The model predicts future price movements using historical market data and technical indicators.

**Asset Class**: {asset_class}  
**Model Type**: {model_type}  
**Training Framework**: {framework}  
**Prediction Horizon**: {horizon} time steps  

## Intended Uses

### Primary Use Cases
- Short-term price forecasting for {asset_class}
- Trading signal generation
- Market trend analysis
- Risk assessment

### Out-of-Scope Uses
- Long-term investment decisions (model optimized for short-term patterns)
- Real-time high-frequency trading (inference latency not optimized)
- Regulatory compliance or financial advice

## Training Data

### Data Sources
{data_sources}

### Features
**Input Features** ({num_features} total):
- Price data: Open, High, Low, Close, Volume
- Technical indicators: {technical_indicators}
- Time features: {time_features}

**Target Variable**: {target_variable}

### Data Preprocessing
{preprocessing_steps}

### Dataset Split
- **Training**: {train_start} to {train_end} ({train_samples} samples)
- **Validation**: {val_start} to {val_end} ({val_samples} samples)
- **Test**: {test_start} to {test_end} ({test_samples} samples)

**Validation Strategy**: {validation_strategy}

## Training Configuration

### Hyperparameters
```python
{hyperparameters}
```

### Model Architecture
{architecture_details}

### Training Environment
- **Hardware**: {hardware}
- **Training Time**: {training_time}
- **Framework Version**: {framework_version}

## Evaluation Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| MAE | {mae} |
| RMSE | {rmse} |
| MAPE | {mape}% |
| R² Score | {r2_score} |
| Directional Accuracy | {directional_accuracy}% |

### Backtest Performance (Walk-Forward)
```
{backtest_results}
```

### Performance by Market Condition
{performance_by_condition}

### Training Curves
{training_curves_description}

## Usage

### Installation
```bash
pip install openmedallion
```

### Quick Start
```python
from openmedallion.hub import from_pretrained
from openmedallion.fints.preprocessing import load_data, engineer_features
from openmedallion.fints.eval import evaluate_predictions
import pandas as pd

# Download model
model_path = from_pretrained(
    "{repo_id}",
    filename="{model_filename}"
)

# Load model
{load_model_code}

# Prepare data
data = load_data("{asset_class}", start_date="2024-01-01")
features = engineer_features(data, window=20)

# Make predictions
predictions = {predict_code}

# Evaluate
metrics = evaluate_predictions(predictions, features['target'])
print(f"Test RMSE: {metrics['rmse']:.4f}")
```

### Integration with Trading Pipeline
```python
from openmedallion.fints import {pipeline_imports}

# Initialize forecasting pipeline
pipeline = {pipeline_class}(
    model_path=model_path,
    lookback_window={lookback},
    forecast_horizon={horizon}
)

# Generate signals
signals = pipeline.generate_signals(
    data=live_data,
    confidence_threshold=0.7
)
```

## Limitations and Biases

### Known Limitations
1. **Market Regime Changes**: Model trained on specific market conditions; performance may degrade during unprecedented events
2. **Data Frequency**: Optimized for {data_frequency} data; not suitable for other frequencies
3. **Asset Coverage**: Trained only on {asset_coverage}
4. **Look-Ahead Bias Prevention**: Uses strict temporal splits, but be cautious with feature engineering
5. **Overfitting Risk**: {overfitting_notes}

### Potential Biases
- **Temporal Bias**: {temporal_bias_description}
- **Survivorship Bias**: {survivorship_bias_notes}
- **Volatility Regime**: Model may underperform in extreme volatility

### Risk Warnings
⚠️ **This model is for research and educational purposes only.**  
⚠️ **Past performance does not guarantee future results.**  
⚠️ **Always validate predictions and use proper risk management.**

## Environmental Impact

- **Training Compute**: {compute_hours} hours on {hardware_type}
- **Estimated CO₂ Emissions**: {co2_estimate} kg CO₂eq
- **Energy Consumption**: {energy_kwh} kWh

*Emissions estimated using the [Machine Learning Impact Calculator](https://mlco2.github.io/impact).*

## Model Card Authors

{authors}

## Citation

```bibtex
@misc{openmedallion_fints_{asset_class},
  author = {{authors_bibtex}},
  title = {{OpenMedallion FinTS: {model_name}}},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{{https://huggingface.co/{repo_id}}}},
  note = {{Asset class: {asset_class}, Model type: {model_type}}}
}
```

## Additional Resources

- **Project Repository**: https://github.com/oyi77/OpenMedallion
- **Documentation**: [TRAINING.md](https://github.com/oyi77/OpenMedallion/blob/main/TRAINING.md)
- **Related Models**: {related_models}
- **Issues & Support**: https://github.com/oyi77/OpenMedallion/issues

## Version History

- **v1.0.0** ({release_date}): Initial release
{version_history}
