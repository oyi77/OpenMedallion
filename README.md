# OpenMedallion — Financial ML Models

Two publishable ML model components for the 1AI NEXUS systematic trading system, built from the `oyi77/OpenMedallion` dataset.

---

## Quick Start

```bash
# Install the unified package
pip install -e .

# Import components
from openmedallion.fints import LGBMForecaster, PatchTSTForecaster
from openmedallion.finsentiment import FinSentimentClassifier
from openmedallion.hub import push_to_hub, from_pretrained, setup_token
```

---

## 1. OpenMedallion-FinTS

Time-series forecasting models for multi-asset directional prediction.

**Status**: ✅ Production Ready

- **Models**: LightGBM baseline + PatchTST transformer
- **Coverage**: Crypto, forex, equities, commodities, indices, ETFs, bonds
- **README**: [`model_cards/FINTS_LGBM_MODEL_CARD.md`](model_cards/FINTS_LGBM_MODEL_CARD.md)
- **Model Cards**: [`FINTS_LGBM_MODEL_CARD.md`](model_cards/FINTS_LGBM_MODEL_CARD.md), [`FINTS_PATCHTST_MODEL_CARD.md`](model_cards/FINTS_PATCHTST_MODEL_CARD.md)

**HuggingFace**: [`oyi77/openmedallion-fints`](https://huggingface.co/oyi77/openmedallion-fints)

**Local Training**:
```bash
# Train LightGBM baseline
python openmedallion/fints/scripts/train_lgbm.py \
    --asset-class equities \
    --max-per-class 20 \
    --splits 5 \
    --push_to_hub \
    --hub_username oyi77 \
    --hub_repo_name openmedallion-fints

# Train PatchTST transformer
python openmedallion/fints/scripts/train_patchtst.py \
    --asset-class crypto \
    --epochs 50 \
    --push_to_hub \
    --hub_username <your-username>
```

**Cloud Training**: See [TRAINING.md](TRAINING.md) for Modal, RunPod, Paperspace, Docker guides.

**Key Features**:
- Strictly temporal walk-forward splits (no data leakage)
- Per-split evaluation (exposes regime sensitivity)
- Streaming data loader (no RAM overload)
- Separate models per asset class (better stability)
- Automatic HuggingFace Hub push after training

---

## 2. OpenMedallion-FinSentiment

QLoRA fine-tuned LLM for financial text sentiment classification.

**Status**: ⚠️ **GPU Required** (QLoRA training needs ≥12GB VRAM; training infrastructure complete)

- **Base Model**: Qwen2.5-7B-Instruct (4-bit QLoRA)
- **Task**: Financial headline/text sentiment (positive/negative/neutral)
- **README**: [`model_cards/FINSENTIMENT_MODEL_CARD.md`](model_cards/FINSENTIMENT_MODEL_CARD.md)
- **Model Card**: [`FINSENTIMENT_MODEL_CARD.md`](model_cards/FINSENTIMENT_MODEL_CARD.md)

**HuggingFace**: [`oyi77/openmedallion-finsentiment`](https://huggingface.co/oyi77/openmedallion-finsentiment)

**Local Training**:
```bash
# Prepare data (use your HF cache snapshot ID)
python openmedallion/finsentiment/prepare_sentiment_data.py \
    --hf-cache ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/<snapshot_id> \
    --target-size 80000

# Fine-tune (~2-4 hours on RTX 3060)
python openmedallion/finsentiment/fine_tune_qwen.py \
    --dataset-dir data/sentiment \
    --output-dir ./outputs \
    --epochs 3 \
    --batch-size 4

# Evaluate vs zero-shot baseline
python openmedallion/finsentiment/eval_finsentiment.py \
    --model checkpoints/finsentiment/final \
    --test data/sampled_val.jsonl
```

**Cloud Training**: See [TRAINING.md](TRAINING.md) for GPU cloud platforms.

**Key Features**:
- 4-bit QLoRA (fits in 12GB VRAM)
- Unsloth accelerated training
- Zero-shot baseline comparison (proves fine-tuning value)
- Stratified sampling across source categories
- Weights & Biases integration for monitoring
- Checkpointing and resume capabilities

---

## HuggingFace Hub Integration

All training scripts support automatic Hub push:

```python
from openmedallion.hub import setup_token, push_to_hub, from_pretrained

# Setup token (one-time)
setup_token()  # Reads from HF_TOKEN env var

# Push after training
push_to_hub(
    repo_id="username/openmedallion-fints",
    local_path="./reports/lgbm_equities.joblib",
    model_type="fints"
)

# Download pre-trained models
model = from_pretrained("username/openmedallion-fints")
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete Hub workflow.

---

## Data Source

Both models use the `oyi77/OpenMedallion` HuggingFace dataset:
- **OpenMedallion-FinTS**: OHLCV time-series (2,131 parquet files, 32 categories, 10.7M rows, 392 MB)
- **OpenMedallion-FinSentiment**: Instruction-tuning text data (~600k rows, finance Q&A)

**Local Cache**: `~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/`

---

## Hardware Requirements

| Component | CPU | GPU | RAM | Training Time |
|-----------|-----|-----|-----|---------------|
| FinTS (LightGBM) | i5 12th gen | Not required | 8GB | ~15-30 min per asset class |
| FinTS (PatchTST) | Any | RTX 3060 12GB | 16GB | ~30 min/epoch |
| FinSentiment | Any | RTX 3060 12GB | 32GB | ~2-4 hours |

**Cloud Options**: A100 (80GB) for FinSentiment, T4 (16GB) for FinTS — see [TRAINING.md](TRAINING.md).

---

## Project Structure

```
OpenMedallion/
├── openmedallion/                # Unified Python package
│   ├── __init__.py
│   ├── fints/                   # Time-series forecasting
│   │   ├── __init__.py
│   │   ├── preprocessing/       # loader.py, features.py, splits.py
│   │   ├── models/              # lgbm_baseline.py, patchtst.py
│   │   ├── eval/                # metrics.py
│   │   └── scripts/             # train_*.py, eval_backtest.py
│   ├── finsentiment/            # LLM sentiment classification
│   │   ├── __init__.py
│   │   ├── fine_tune_qwen.py
│   │   ├── prepare_sentiment_data.py
│   │   └── eval_finsentiment.py
│   └── hub/                     # HuggingFace Hub utilities
│       ├── __init__.py
│       ├── uploader.py          # push_to_hub, model cards
│       ├── downloader.py        # from_pretrained, cache management
│       ├── auth.py              # setup_token, verify_token
│       └── templates/           # Model card markdown templates
├── cloud_training/              # Cloud platform scripts
│   ├── modal_train.py           # Modal.com serverless
│   ├── runpod_train.py          # RunPod serverless
│   └── paperspace_train.py      # Paperspace Gradient
├── Dockerfile                   # Training environment
├── docker-compose.yml           # Multi-service orchestration
├── setup.py                     # Package metadata
├── data/                        # OHLCV parquet files (not in repo)
├── TRAINING.md                  # Cloud training guide
├── DEPLOYMENT.md                # Hub deployment workflow
└── README.md                    # This file
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/OpenMedallion.git
cd OpenMedallion

# Install in development mode
pip install -e .

# Or install from PyPI (future)
pip install openmedallion
```

**Dependencies**:
- **FinTS**: `lightgbm`, `scikit-learn`, `pandas`, `pyarrow`, `torch`
- **FinSentiment**: `unsloth`, `torch`, `transformers`, `trl`, `datasets`, `scikit-learn`
- **Hub**: `huggingface-hub`, `requests`

---

## Documentation

- **[TRAINING.md](TRAINING.md)**: Cloud training on Modal, RunPod, Paperspace, Docker
- **[model_cards/FINTS_LGBM_MODEL_CARD.md](model_cards/FINTS_LGBM_MODEL_CARD.md)** — FinTS model documentation
- **[model_cards/FINTS_PATCHTST_MODEL_CARD.md](model_cards/FINTS_PATCHTST_MODEL_CARD.md)** — PatchTST model documentation
- **[model_cards/FINSENTIMENT_MODEL_CARD.md](model_cards/FINSENTIMENT_MODEL_CARD.md)** — FinSentiment model documentation

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

## Citation

```bibtex
@software{openmedallion2026,
  author = {1AI NEXUS},
  title = {OpenMedallion: Financial ML Models for Systematic Trading},
  year = {2026},
  url = {https://github.com/<your-username>/OpenMedallion}
}
```

---

**All code is production-ready** — imports work, smoke tests pass, documentation complete.
