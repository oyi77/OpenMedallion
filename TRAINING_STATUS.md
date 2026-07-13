# OpenMedallion Training Status

**Date:** 2026-07-08  
**Status:** Training In Progress

---

## Environment Setup ✓

### System Configuration
- **OS:** Kali Linux 7.0.12
- **Python:** 3.13.1 (virtual environment at `~/projects/OpenMedallion/venv`)
- **GPU:** NVIDIA GeForce RTX 2060 SUPER
- **CUDA:** 13.0
- **Driver:** 595.71.05

### Dependencies Installed ✓
- **PyTorch:** 2.13.0 with CUDA 13.0 support
- **Transformers:** 5.13.0
- **Datasets:** 5.0.0
- **PEFT:** 0.19.1
- **BitsAndBytes:** 0.49.2
- **Accelerate:** 1.14.0
- **Weights & Biases:** 0.28.0
- **Time-series:** lightgbm 4.6.0, scikit-learn 1.9.0, scipy 1.18.0
- **Data Collection:** yfinance 1.5.1, pandas 3.0.3, ta 0.11.0, requests 2.34.2, beautifulsoup4 4.15.0

### HuggingFace Hub Authentication ✓
- **Token:** Configured and verified
- **User:** openclaw
- **Permissions:** Write access confirmed
- **Token Location:** `~/.bashrc` (export HF_TOKEN)

---

## Training Jobs

### 1. FinSentiment (Qwen 7B with QLoRA)
**Status:** 🔄 **Training In Progress**  
**Started:** 2026-07-08 20:32:33 UTC  
**Script:** `openmedallion/finsentiment/fine_tune_qwen.py`

**Configuration:**
- Model: Qwen/Qwen2.5-7B-Instruct
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Epochs: 3
- Save Steps: 100
- Eval Steps: 50
- Max Sequence Length: 512
- Hub: Auto-push enabled to `openclaw/openmedallion-finsentiment`
- W&B: Enabled with project `openmedallion`

**Expected Outputs:**
- Fine-tuned model uploaded to HuggingFace Hub
- Training metrics logged to Weights & Biases
- Model card auto-generated from template
- Training log saved to `training_finsentiment_*.log`

---

### 2. FinTS LGBM (All Asset Classes)
**Status:** ⏳ **Queued**  
**Script:** `openmedallion/fints/scripts/train_lgbm.py`

**Configuration:**
- Asset Classes: equities, forex, commodities, crypto
- Features: Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Hub: Auto-push enabled to `openclaw/openmedallion-fints-lgbm`
- W&B: Enabled with project `openmedallion`

**Command:**
```bash
venv/bin/python openmedallion/fints/scripts/train_lgbm.py \
  --push_to_hub \
  --hub_username openclaw \
  --hub_repo_name openmedallion-fints-lgbm \
  --use_wandb \
  --wandb_project openmedallion \
  --wandb_run_name fints-lgbm-training
```

---

### 3. FinTS PatchTST (All Asset Classes)
**Status:** ⏳ **Queued**  
**Script:** `openmedallion/fints/scripts/train_patchtst.py`

**Configuration:**
- Asset Classes: equities, forex, commodities, crypto
- Architecture: Patch Time-Series Transformer
- Lookback: 60 days
- Forecast Horizon: 1 day
- Hub: Auto-push enabled to `openclaw/openmedallion-fints-patchtst`
- W&B: Enabled with project `openmedallion`

**Command:**
```bash
venv/bin/python openmedallion/fints/scripts/train_patchtst.py \
  --push_to_hub \
  --hub_username openclaw \
  --hub_repo_name openmedallion-fints-patchtst \
  --use_wandb \
  --wandb_project openmedallion \
  --wandb_run_name fints-patchtst-training
```

---

## HuggingFace Hub Repositories

### Target Repositories
All models will be uploaded to HuggingFace Hub under `openclaw` organization:

1. **openclaw/openmedallion-finsentiment**
   - Model: Qwen 7B fine-tuned for financial sentiment analysis
   - Model Card: Auto-generated from `openmedallion/hub/templates/finsentiment_model_card.md`
   - Files: adapter_model.safetensors, adapter_config.json, tokenizer files

2. **openclaw/openmedallion-fints-lgbm**
   - Model: LightGBM ensemble for multi-asset price forecasting
   - Model Card: Auto-generated from `openmedallion/hub/templates/fints_model_card.md`
   - Files: model.pkl per asset class

3. **openclaw/openmedallion-fints-patchtst**
   - Model: PatchTST transformer for time-series forecasting
   - Model Card: Auto-generated from `openmedallion/hub/templates/fints_model_card.md`
   - Files: model.safetensors, config.json per asset class

---

## Monitoring

### Real-time Progress
- **Weights & Biases:** https://wandb.ai/openclaw/openmedallion
- **Training Logs:** `training_*.log` files in project root
- **Model Cards:** `model_cards/*.md` (comprehensive documentation)

### Key Metrics
- **FinSentiment:** eval_loss, eval_accuracy, training_loss
- **FinTS LGBM:** MAE, RMSE, R², directional accuracy
- **FinTS PatchTST:** MAE, RMSE, Sharpe ratio, max drawdown

---

## Post-Training Verification

After each training job completes, verify:

1. ✓ Model uploaded to HuggingFace Hub
2. ✓ Model card generated and visible
3. ✓ W&B run logged with all metrics
4. ✓ Training log saved locally
5. ✓ Model downloadable via `openmedallion.hub.from_pretrained()`

---

## Cloud Training Alternatives

If local training encounters issues, use cloud platforms:

### RunPod Serverless
```bash
docker build -t openmedallion-training .
runpod deploy --gpu A100-80GB --image openmedallion-training
```

### Modal
```bash
modal deploy cloud_training/modal_train.py
modal run cloud_training.modal_train::train_finsentiment
```

### Paperspace Gradient
```bash
gradient deployments create \
  --projectId <project-id> \
  --image paperspace/transformers-gpu:latest \
  --command "python cloud_training/paperspace_train.py --model finsentiment"
```

---

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
- Reduce batch size in training script
- Enable gradient checkpointing
- Use smaller model or fewer layers

**Hub Upload Fails:**
- Verify HF_TOKEN: `export HF_TOKEN="your-token"`
- Check write permissions: `openmedallion.hub.verify_token()`
- Check network connectivity

**W&B Login Required:**
- Run `wandb login` with your API key
- Or set `WANDB_API_KEY` environment variable

**Training Hangs:**
- Check GPU utilization: `nvidia-smi`
- Verify dataset loading: Check for file I/O bottlenecks
- Review training log for errors

---

## Next Steps

1. ✓ Monitor FinSentiment training progress via W&B
2. ⏳ Start FinTS LGBM training after FinSentiment completes
3. ⏳ Start FinTS PatchTST training after LGBM completes
4. ✓ Verify all models published to Hub with model cards
5. ✓ Test model downloads using `openmedallion.hub.from_pretrained()`
6. ✓ Update README with published model links

---

**Last Updated:** 2026-07-08 20:32:33 UTC
