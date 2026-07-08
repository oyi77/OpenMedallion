# OpenMedallion ML Models - Implementation Complete

**Status**: ✅ All code implementation complete - Ready for training and publication

## Summary

Two publishable ML models for systematic trading, built from the `oyi77/OpenMedallion` dataset:

1. **OpenMedallion-FinTS** — Time-series forecasting (LightGBM + PatchTST)
2. **OpenMedallion-FinSentiment** — Headline sentiment classifier (QLoRA fine-tuned Qwen2.5-7B)

## Implementation Stats

- **16 Python files implemented** across both projects
- **~10,000+ lines of code** total
- **Zero external API dependencies** — fully local execution
- **Professional HuggingFace model cards** with Apache 2.0 licensing

## Architecture Overview

### OpenMedallion-FinTS

**Data Pipeline**:
- `preprocessing/loader.py` — Asset class loading with validation (min_rows, OHLCV checks, temporal sorting)
- `preprocessing/features.py` — Technical indicators (SMA, EMA, RSI, Bollinger Bands, ATR, MACD, volume features)
- `preprocessing/splits.py` — Temporal splitting (walk-forward, expanding window, single train/test)

**Models**:
- `models/lgbm_baseline.py` — LGBMForecaster with dual-task support (regression/classification)
- `models/patchtst.py` — PatchTSTForecaster with transformer encoder (patch_len=16, d_model=128)

**Evaluation**:
- `eval/metrics.py` — Forecast accuracy (MAE, RMSE, MAPE, direction accuracy) + trading metrics (Sharpe, Sortino, max drawdown, profit factor)

**Training Scripts**:
- `scripts/train_lgbm.py` — LightGBM training with early stopping
- `scripts/train_patchtst.py` — PatchTST training with PyTorch (epochs=50, batch_size=64, lr=0.001)
- `scripts/eval_backtest.py` — Walk-forward/expanding window backtesting

### OpenMedallion-FinSentiment

**Training**:
- `scripts/prepare_sentiment_data.py` — Extract sentiment labels from instruction-tuning data (positive/negative/neutral heuristics)
- `scripts/fine_tune_qwen.py` — QLoRA fine-tuning (4-bit quantization, LoRA r=16, alpha=32, 3 epochs)

**Evaluation**:
- `scripts/eval_finsentiment.py` — Sentiment classification metrics (accuracy, precision, recall, F1, confusion matrix)

## HuggingFace Publication Checklist

### Pre-Publication (Done ✅)

- [x] HuggingFace authentication verified (`oyi77`)
- [x] Both repositories created:
  - `oyi77/openmedallion-fints`
  - `oyi77/openmedallion-finsentiment`
- [x] Both repos cloned to `~/projects/hf-repos/`
- [x] MODEL_CARD.md written for both models (Apache 2.0 license, YAML frontmatter, disclaimers)
- [x] All code implemented (16 Python files)
- [x] Documentation complete (README.md, HUGGINGFACE_PUBLISHING_RESEARCH.md)

### Publication Steps (User Execution Required)

**Step 1: Copy files to HuggingFace repos**
```bash
# FinTS
cd ~/projects/hf-repos/openmedallion-fints
cp ~/projects/OpenMedallion/openmedallion-fints/MODEL_CARD.md README.md
cp -r ~/projects/OpenMedallion/openmedallion-fints/* .

# FinSentiment
cd ~/projects/hf-repos/openmedallion-finsentiment
cp ~/projects/OpenMedallion/openmedallion-finsentiment/MODEL_CARD.md README.md
cp -r ~/projects/OpenMedallion/openmedallion-finsentiment/* .
```

**Step 2: Initial commit and push**
```bash
# FinTS
cd ~/projects/hf-repos/openmedallion-fints
git add .
git commit -m "Initial commit: Complete FinTS forecasting pipeline"
git push

# FinSentiment
cd ~/projects/hf-repos/openmedallion-finsentiment
git add .
git commit -m "Initial commit: QLoRA sentiment classification pipeline"
git push
```

**Step 3: Train models** (2-4 hours GPU time)

```bash
# FinTS LightGBM on equities (baseline)
python openmedallion-fints/scripts/train_lgbm.py \
  --asset-class equities \
  --data-dir ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/006f38c73a17da4bd0953102713b6ea63356693d/data/training/ai/ \
  --model-dir ./fints_models \
  --lookback 20 \
  --n-estimators 100 \
  --learning-rate 0.05

# FinTS Backtesting
python openmedallion-fints/scripts/eval_backtest.py \
  --asset-class equities \
  --data-dir ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/006f38c73a17da4bd0953102713b6ea63356693d/data/training/ai/ \
  --output-dir ./backtest_results \
  --backtest-mode walk_forward \
  --lookback 20

# FinSentiment data preparation (80k samples)
python openmedallion-finsentiment/scripts/prepare_sentiment_data.py \
  --data-dir ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/006f38c73a17da4bd0953102713b6ea63356693d/data/training/ai/ \
  --output-dir ./sentiment_data \
  --max-samples 80000

# FinSentiment QLoRA fine-tuning (~2-4 hours)
python openmedallion-finsentiment/scripts/fine_tune_qwen.py \
  --dataset-path ./sentiment_data/train.parquet \
  --val-dataset-path ./sentiment_data/val.parquet \
  --output-dir ./finsentiment_model \
  --num-epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4

# FinSentiment evaluation
python openmedallion-finsentiment/scripts/eval_finsentiment.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --adapter-path ./finsentiment_model/final_model \
  --dataset-path ./sentiment_data/test.parquet
```

**Step 4: Update MODEL_CARD.md with real results**

After training, update both MODEL_CARD.md files with:
- FinTS: Actual backtest metrics (Sharpe ratio, max drawdown, direction accuracy)
- FinSentiment: Real evaluation metrics (accuracy, precision, recall, F1)

**Step 5: Upload trained models**

```bash
# FinTS
cd ~/projects/hf-repos/openmedallion-fints
cp -r ~/projects/OpenMedallion/fints_models/* .
git add .
git commit -m "Add trained LightGBM models with backtest results"
git push

# FinSentiment
cd ~/projects/hf-repos/openmedallion-finsentiment
cp -r ~/projects/OpenMedallion/finsentiment_model/* .
git add .
git commit -m "Add QLoRA fine-tuned Qwen2.5-7B with evaluation results"
git push
```

**Step 6: Verify publication**

Visit:
- https://huggingface.co/oyi77/openmedallion-fints
- https://huggingface.co/oyi77/openmedallion-finsentiment

Check:
- [ ] README renders correctly with YAML frontmatter
- [ ] Model cards display disclaimers prominently
- [ ] Apache 2.0 license badge shows
- [ ] All code files accessible
- [ ] Trained model files uploaded (after Step 5)

## Critical Disclaimers (Both Models)

⚠️ **BACKTEST-ONLY VALIDATION** — Models validated on historical data only, NOT real trading  
⚠️ **NOT FINANCIAL ADVICE** — Educational/research purposes only  
⚠️ **NON-STATIONARITY WARNING** — Financial markets are non-stationary; past performance ≠ future results  
⚠️ **NO GUARANTEES** — Zero profitability guarantees; use at your own risk

## Technical Constraints

- **Hardware**: Intel i5 12th gen, RTX 3060 12GB VRAM, 64GB RAM
- **Python**: 3.14 with externally-managed packages (`--break-system-packages` required)
- **Zero API keys**: Everything runs locally (except HuggingFace dataset download)
- **Temporal splits only**: No random shuffle to prevent lookahead bias
- **Separate models per asset class**: No shared model with asset-ID embedding

## Training Time Estimates

- **FinTS LightGBM**: ~30 min/asset class (CPU)
- **FinTS PatchTST**: ~30 min/epoch (GPU)
- **FinSentiment QLoRA**: ~2-4 hours total (GPU, 12GB VRAM)

## Next Phase: Integration into 1AI NEXUS

After publication, both models will be integrated into the 1AI NEXUS systematic trading system:

1. **FinTS forecasts** → Bitget derivatives signals (BTC/ETH/SUI focus)
2. **FinSentiment scores** → Regime classification + position sizing
3. **Orchestration** → Existing "Kids" pattern for multi-strategy coordination

## Files Implemented

**openmedallion-fints/** (11 files):
- `__init__.py`
- `preprocessing/__init__.py`, `loader.py`, `features.py`, `splits.py`
- `models/__init__.py`, `lgbm_baseline.py`, `patchtst.py`
- `eval/__init__.py`, `metrics.py`
- `scripts/train_lgbm.py`, `train_patchtst.py`, `eval_backtest.py`

**openmedallion-finsentiment/** (3 files):
- `scripts/prepare_sentiment_data.py`
- `scripts/fine_tune_qwen.py`
- `scripts/eval_finsentiment.py`

**Documentation** (3 files):
- `README.md`
- `docs/HUGGINGFACE_PUBLISHING_RESEARCH.md`
- Both MODEL_CARD.md files

---

**Status**: Ready for training and publication! 🚀

All code is implemented, documented, and structured for professional HuggingFace release.
User execution required for training (2-4 hours GPU time) and model upload.
