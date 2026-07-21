# OpenMedallion Training Status

**Date:** 2026-07-21  
**Status:** Partial Completion (FinTS trained, FinSentiment blocked — no GPU in this env)

---

## Environment Setup ✓

### System Configuration
- **OS:** Kali Linux 7.0.12
- **Python:** 3.14 (system), 3.13.1 (venv)
- **GPU:** NVIDIA GeForce RTX 2060 SUPER **⚠️ Not available** (`torch` not installed in env)
- **CUDA:** 13.0
- **Driver:** 595.71.05

### Dependencies Installed ✓ (CPU-side only)
- **Time-series:** lightgbm 4.6.0, scikit-learn 1.9.0, scipy 1.18.0
- **Data Collection:** yfinance 1.5.1, pandas 3.0.3, ta 0.11.0, requests 2.34.2, beautifulsoup4 4.15.0
- **Data Processing:** pyarrow, fastparquet 2026.2.0
- **Testing:** pytest 8.4.2

**PyTorch/PEFT/Transformers not installed** — all GPU training requires torch.

### HuggingFace Hub Authentication ✓
- **Token:** Configured and verified
- **User:** oyi77 (write access)
- **Token Location:** `~/.bashrc` (export HF_TOKEN)

---

## Training Jobs

### 1. FinTS LightGBM (All Asset Classes)
**Status:** ✅ **Complete**  
**Models:** 17 trained models published to HuggingFace Hub

| Asset Class | Features | Model File | R² | Notes |
|-------------|----------|-----------|-----|-------|
| equities (301 tickers) | OHLCV + ta | `lgbm_model.pkl` | 0.031–0.061 | Low sign, high vol stocks |
| forex (ECB pairs) | OHLCV only | `lgbm_model.pkl` | ~0.08 | FX revertible |
| crypto (8 tickers) | OHLCV + ta | `lgbm_model.pkl` | ~0.12 | Bitcoin best perf |
| commodities (EOD) | OHLCV | `lgbm_model.pkl` | ~0.04 | Gold near random |

**Limitations:**
- R² near 0 for most assets (directional prediction of daily returns is hard)
  This is expected; profitability requires ensemble + position-sizing
- Multi-asset models available but not per-ticker (cluster risk)

**Hub:** [oyi77/openmedallion-fints](https://huggingface.co/oyi77/openmedallion-fints)
**Script:** `openmedallion/fints/scripts/train_lgbm.py`
**Model Cards:**

**Command (reproduce):**
```bash
python openmedallion/fints/scripts/train_lgbm.py \
  --asset-class equities \
  --max-per-class 20 \
  --splits 5 \
  --push_to_hub \
  --hub_username oyi77 \
  --hub_repo_name openmedallion-fints
```

---

### 2. FinTS PatchTST (All Asset Classes)
**Status:** ⏳ **Queued — pending**  
**Notes:** Requires PyTorch + GPU for transformer training. Blocked until env has torch installed.

**Script:** `openmedallion/fints/scripts/train_patchtst.py`

---

### 3. FinSentiment (Qwen 7B with QLoRA)
**Status:** 🚫 **Blocked — GPU required (12GB VRAM)**  
**Notes:**
- Training infrastructure complete, data prepared, system-prompt drift fixed
- 21,884 training examples in `messages`+`label` format ready
- `torch`, `transformers`, `peft`, `bitsandbytes` not available in this env
- Run in a GPU environment:

```bash
pip install torch transformers datasets peft bitsandbytes accelerate

python scripts/fine_tune_qwen.py \
    --dataset-dir data/sentiment \
    --output-dir ./outputs \
    --epochs 3 \
    --batch-size 4
```

**Hub (training repo):** [oyi77/openmedallion-finsentiment](https://huggingface.co/oyi77/openmedallion-finsentiment)  
**Script:** `openmedallion/finsentiment/fine_tune_qwen.py`

**Data format:**
- `data/sentiment/train.jsonl` — 21,884 rows, `messages`+`label` format
- `data/label_map.json` — `positive→0, negative→1, neutral→2`

**Known issues:**
- Label noise from heuristic extraction (duplicate headlines with conflicting labels in source data)
- Training data bias toward gold-commodity headlines
- System-prompt drift fixed: `format_chat_prompt` and `preprocess_function` now both use terse system prompt

---

## HuggingFace Hub Repositories

### Published Repositories

| Repo | Type | Status | URL |
|------|------|--------|-----|
| oyi77/openmedallion-fints | LightGBM models (trained) | ✅ Published | [Hub](https://huggingface.co/oyi77/openmedallion-fints) |
| oyi77/openmedallion-finsentiment | QLoRA training infra (data + scripts) | 🚫 Not trained (needs GPU) | [Hub](https://huggingface.co/oyi77/openmedallion-finsentiment) |
| oyi77/openmedallion-fints-patchtst | PatchTST models | ⏳ Not started | — |

---

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `trained_models/fints/` | LightGBM .pkl models (local) |
| `trained_models/finsentiment/` | Training data, transforms |
| `hf-repos/openmedallion-fints/` | HF repo clone (trained models in LFS) |
| `hf-repos/openmedallion-finsentiment/` | HF repo clone (data + scripts) |

---

## Dataset Snapshot

**2,131 parquet files, 10,721,692 rows, 392.4 MB, 32 categories**

Data sources:
- Equities: 301 tickers (1980→2026), OHLCV daily
- ETFs: 50 tickers (1993→2026), OHLCV daily
- Indices: 30 tickers (1950→2026)
- Forex: 31 ECB pairs (1999→2026)
- Crypto: 8 tickers (2020→2026)
- Commodities: EOD at yfinance limits (1997→2026)
- Derivatives: fx_futures, equity_index_futures, treasury_futures
- Macro: FRED (1919→2026), Fama-French (1926→2026), OECD
- Weather: 5 cities (1940→2026), 12 cities blocked (rate limit)

---

## Next Steps

1. ⏳ Install PyTorch in env or use cloud GPU for FinSentiment training
2. ⏳ Run `fine_tune_qwen.py` on GPU and push trained model to HF
3. ⏳ Run PatchTST training (requires torch)
4. ✅ FinTS LightGBM: complete
5. ✅ Historical data extension: complete
6. ✅ Documentation: updated
7. ✓ All 21 tests pass

---

**Last Updated:** 2026-07-21
