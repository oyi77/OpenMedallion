# OpenMedallion — Financial ML Models

Two publishable ML model components for the 1AI NEXUS systematic trading system, built from the `oyi77/OpenMedallion` dataset.

---

## 1. OpenMedallion-FinTS

Time-series forecasting models for multi-asset directional prediction.

**Status**: ✅ Ready to train

- **Models**: LightGBM baseline + PatchTST transformer
- **Coverage**: Crypto, forex, equities, commodities, indices, ETFs, bonds
- **Location**: `openmedallion-fints/`
- **README**: [`openmedallion-fints/README.md`](openmedallion-fints/README.md)
- **Model Card**: [`openmedallion-fints/MODEL_CARD.md`](openmedallion-fints/MODEL_CARD.md)

**HuggingFace**: `<namespace>/openmedallion-fints`

**Quick Start**:
```bash
cd openmedallion-fints
pip install --break-system-packages lightgbm scikit-learn pandas pyarrow torch
python scripts/train_lgbm.py --asset-class equities --max-per-class 20 --splits 5
python scripts/eval_backtest.py --input reports/lgbm_equities_splits.csv
```

**Key Features**:
- Strictly temporal walk-forward splits (no data leakage)
- Per-split evaluation (exposes regime sensitivity)
- Streaming data loader (no RAM overload)
- Separate models per asset class (better stability)

---

## 2. OpenMedallion-FinSentiment

QLoRA fine-tuned LLM for financial text sentiment classification.

**Status**: ✅ Ready to train

- **Base Model**: Qwen2.5-7B-Instruct (4-bit QLoRA)
- **Task**: Financial headline/text sentiment (positive/negative/neutral)
- **Location**: `openmedallion-finsentiment/`
- **README**: [`openmedallion-finsentiment/README.md`](openmedallion-finsentiment/README.md)
- **Model Card**: [`openmedallion-finsentiment/MODEL_CARD.md`](openmedallion-finsentiment/MODEL_CARD.md)

**HuggingFace**: `<namespace>/openmedallion-finsentiment`

**Quick Start**:
```bash
cd openmedallion-finsentiment
pip install unsloth torch transformers trl datasets scikit-learn

# Prepare data (use your HF cache snapshot ID)
python scripts/prepare_sentiment_data.py \
    --hf-cache ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/<snapshot_id> \
    --target-size 80000

# Fine-tune (~2-4 hours on RTX 3060)
python scripts/fine_tune_qwen.py \
    --train data/sampled_train.jsonl \
    --val data/sampled_val.jsonl \
    --epochs 1

# Evaluate vs zero-shot baseline
python scripts/eval_finsentiment.py \
    --model checkpoints/finsentiment/final \
    --test data/sampled_val.jsonl
```

**Key Features**:
- 4-bit QLoRA (fits in 12GB VRAM)
- Unsloth accelerated training
- Zero-shot baseline comparison (proves fine-tuning value)
- Stratified sampling across source categories

---

## Data Source

Both models use the `oyi77/OpenMedallion` HuggingFace dataset:
- **OpenMedallion-FinTS**: OHLCV time-series (1,913 parquet files, 28 categories)
- **OpenMedallion-FinSentiment**: Instruction-tuning text data (~600k rows, finance Q&A)

**Local Cache**: `~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/`

---

## Hardware Requirements

| Component | CPU | GPU | RAM | Training Time |
|-----------|-----|-----|-----|---------------|
| FinTS (LightGBM) | i5 12th gen | Not required | 8GB | ~15-30 min per asset class |
| FinTS (PatchTST) | Any | RTX 3060 12GB | 16GB | ~30 min/epoch |
| FinSentiment | Any | RTX 3060 12GB | 32GB | ~2-4 hours |

---

## Disclaimers

⚠️ **Not Financial Advice**

Both models are research components validated on historical backtest data only:
- Past performance does not predict future results
- Non-stationarity means metrics **will** degrade in live deployment
- Use as input signals in a broader system with hard risk controls
- Transaction costs, slippage, and funding rates are **not** modeled
- No regulatory approval for investment advice

---

## License

MIT. Base models and data sources retain their original licenses:
- Qwen2.5: Apache 2.0
- PatchTST: Original paper implementation (educational use)
- OpenMedallion dataset: Check HuggingFace repo for terms

---

## Next Steps

1. **Train OpenMedallion-FinTS LightGBM baseline** on equities (fastest asset class)
2. **Evaluate backtest metrics** — check for regime degradation across splits
3. **Train OpenMedallion-FinSentiment** on 80k sampled instructions
4. **Compare zero-shot vs fine-tuned** — ensure fine-tuning added value
5. **Publish to HuggingFace** (separate repos):
   - `<namespace>/openmedallion-fints`
   - `<namespace>/openmedallion-finsentiment`

---

## Project Structure

```
OpenMedallion/
├── openmedallion-fints/
│   ├── preprocessing/     # loader.py, features.py, splits.py
│   ├── models/           # lgbm_baseline.py, patchtst.py
│   ├── eval/             # metrics.py
│   ├── scripts/          # train_*.py, eval_backtest.py
│   ├── README.md
│   └── MODEL_CARD.md
├── openmedallion-finsentiment/
│   ├── scripts/          # prepare_*.py, fine_tune_qwen.py, eval_*.py
│   ├── README.md
│   └── MODEL_CARD.md
├── data/                 # OHLCV parquet files (not in repo)
└── README.md             # This file
```

**All code is production-ready** — imports work, smoke tests pass, documentation complete.
