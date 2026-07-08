# Cloud Training Guide

Comprehensive guide for training OpenMedallion models on Modal, RunPod, Paperspace, and Docker.

---

## Overview

All cloud training scripts support:
- ✅ Automatic HuggingFace Hub push after training
- ✅ Weights & Biases integration for monitoring
- ✅ Checkpointing and resume capabilities
- ✅ Both FinTS and FinSentiment models

**Hardware Requirements**:
- **FinSentiment**: A100 (80GB) or RTX 3090/4090 (24GB)
- **FinTS**: T4 (16GB) or better

---

## 1. Modal.com (Recommended)

**Best for**: Serverless, pay-per-second, zero cold start overhead.

### Setup

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy script to Modal
modal deploy cloud_training/modal_train.py
```

### Train FinSentiment

```bash
modal run cloud_training/modal_train.py::train_finsentiment \
    --train-path data/sampled_train.jsonl \
    --val-path data/sampled_val.jsonl \
    --epochs 1 \
    --hf-token $HF_TOKEN \
    --hub-username <your-username> \
    --hub-repo-name openmedallion-finsentiment \
    --wandb-project openmedallion \
    --wandb-key $WANDB_API_KEY
```

**GPU**: A100-80GB (4-bit QLoRA fits in 24GB, but A100 for speed)  
**Cost**: ~$1.50/hour  
**Training Time**: ~1-2 hours for 1 epoch

### Train FinTS (All Asset Classes)

```bash
# LightGBM baseline (all 5 asset classes)
modal run cloud_training/modal_train.py::train_fints \
    --model-type lgbm \
    --asset-class all \
    --hf-token $HF_TOKEN \
    --hub-username <your-username> \
    --hub-repo-name openmedallion-fints

# PatchTST transformer (single asset class)
modal run cloud_training/modal_train.py::train_fints \
    --model-type patchtst \
    --asset-class crypto \
    --epochs 50 \
    --hf-token $HF_TOKEN \
    --hub-username <your-username>
```

**GPU**: T4-16GB (sufficient for FinTS)  
**Cost**: ~$0.30/hour  
**Training Time**: 15-30 min (LGBM), 30-60 min (PatchTST)

### Key Features

- **Persistent volume**: `/models` cached across runs
- **Automatic retries**: Modal handles transient failures
- **Secrets management**: Store tokens via `modal secret create`
- **Streaming logs**: Real-time training progress

---

## 2. RunPod Serverless

**Best for**: High GPU availability, flexible pricing.

### Setup

```bash
# Install RunPod SDK
pip install runpod

# Set API key
export RUNPOD_API_KEY=your_api_key_here
```

### Deploy Handler

```bash
# Build Docker image
docker build -t openmedallion-trainer .

# Push to Docker Hub
docker tag openmedallion-trainer <your-dockerhub>/openmedallion-trainer
docker push <your-dockerhub>/openmedallion-trainer

# Create RunPod endpoint via UI:
# 1. Go to https://www.runpod.io/console/serverless
# 2. Create new endpoint
# 3. Set Docker image: <your-dockerhub>/openmedallion-trainer
# 4. Set GPU: A100 (FinSentiment) or RTX 3090 (FinTS)
# 5. Copy endpoint ID
```

### Train FinSentiment

```python
import runpod

runpod.api_key = "your_api_key_here"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

job = endpoint.run({
    "handler": "train_finsentiment",
    "train_path": "data/sampled_train.jsonl",
    "val_path": "data/sampled_val.jsonl",
    "epochs": 1,
    "hf_token": "hf_...",
    "hub_username": "your-username",
    "hub_repo_name": "openmedallion-finsentiment",
    "wandb_project": "openmedallion",
    "wandb_key": "..."
})

# Poll for completion
while True:
    status = endpoint.status(job)
    print(status)
    if status["status"] in ["COMPLETED", "FAILED"]:
        break
    time.sleep(30)
```

### Train FinTS

```python
job = endpoint.run({
    "handler": "train_fints",
    "model_type": "lgbm",
    "asset_class": "equities",
    "hf_token": "hf_...",
    "hub_username": "your-username"
})
```

### Key Features

- **Serverless functions**: Pay only when running
- **GPU autoscaling**: Spins up instances on demand
- **Network volumes**: Persistent `/workspace` storage
- **Unified handler**: Single image for both models

**GPU Options**:
- A100-80GB: $1.89/hour
- RTX 3090: $0.44/hour
- RTX 4090: $0.69/hour

---

## 3. Paperspace Gradient

**Best for**: Managed notebooks + CLI workflows.

### Setup

```bash
# Install Gradient CLI
pip install gradient

# Login
gradient apiKey <your-api-key>
```

### Train FinSentiment

```bash
gradient jobs create \
    --name "openmedallion-finsentiment" \
    --projectId <your-project-id> \
    --container openmedallion-trainer:latest \
    --machineType A100 \
    --command "python cloud_training/paperspace_train.py \
        --model finsentiment \
        --train-path /storage/data/sampled_train.jsonl \
        --val-path /storage/data/sampled_val.jsonl \
        --epochs 1 \
        --hf-token $HF_TOKEN \
        --hub-username <your-username> \
        --hub-repo-name openmedallion-finsentiment \
        --wandb-project openmedallion" \
    --workspace "https://github.com/<your-username>/OpenMedallion.git"
```

### Train FinTS (All Asset Classes)

```bash
gradient jobs create \
    --name "openmedallion-fints-all" \
    --projectId <your-project-id> \
    --container openmedallion-trainer:latest \
    --machineType P4000 \
    --command "python cloud_training/paperspace_train.py \
        --model fints-all \
        --hf-token $HF_TOKEN \
        --hub-username <your-username>" \
    --workspace "https://github.com/<your-username>/OpenMedallion.git"
```

### Key Features

- **Gradient Workflows**: Multi-step pipelines
- **Persistent storage**: `/storage` volume across jobs
- **Job logs**: `gradient jobs logs <job-id>`
- **Auto-shutdown**: Configurable timeout

**GPU Options**:
- A100-80GB: $3.09/hour
- P4000: $0.51/hour
- RTX 5000: $0.82/hour

---

## 4. Docker + Docker Compose

**Best for**: Local development, on-premise GPU servers.

### Setup

```bash
# Build image
docker build -t openmedallion-trainer .

# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Train with Docker Run

```bash
# FinSentiment
docker run --gpus all -v $(pwd)/data:/app/data \
    -e HF_TOKEN=$HF_TOKEN \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    openmedallion-trainer \
    python openmedallion/finsentiment/fine_tune_qwen.py \
        --train /app/data/sampled_train.jsonl \
        --val /app/data/sampled_val.jsonl \
        --epochs 1 \
        --push_to_hub \
        --hub_username <your-username> \
        --use_wandb \
        --wandb_project openmedallion

# FinTS (LightGBM)
docker run --gpus all -v $(pwd)/data:/app/data \
    -e HF_TOKEN=$HF_TOKEN \
    openmedallion-trainer \
    python openmedallion/fints/scripts/train_lgbm.py \
        --asset-class equities \
        --push_to_hub \
        --hub_username <your-username>
```

### Train with Docker Compose

```bash
# Edit docker-compose.yml to set environment variables

# Train FinSentiment only
docker-compose --profile finsentiment up

# Train FinTS only
docker-compose --profile fints up

# Train both sequentially
docker-compose --profile finsentiment --profile fints up

# View logs
docker-compose logs -f finsentiment-trainer
```

**docker-compose.yml profiles**:
- `finsentiment`: Qwen2.5 fine-tuning
- `fints`: LGBM + PatchTST for all asset classes
- `jupyter`: Interactive notebook environment

### Key Features

- **GPU passthrough**: `--gpus all` flag
- **Volume mounting**: Persistent data and checkpoints
- **Environment secrets**: `.env` file support
- **Multi-service**: Run multiple training jobs in parallel

---

## Environment Variables

All platforms require these environment variables:

```bash
# Required
export HF_TOKEN=hf_...                    # HuggingFace token (write permission)

# Optional
export WANDB_API_KEY=...                  # Weights & Biases API key
export WANDB_PROJECT=openmedallion        # W&B project name
```

**Setting secrets**:

- **Modal**: `modal secret create huggingface HF_TOKEN=hf_...`
- **RunPod**: Set in endpoint environment variables
- **Paperspace**: `gradient secrets set HF_TOKEN hf_...`
- **Docker**: Create `.env` file in project root

---

## Training Configuration

### FinSentiment (fine_tune_qwen.py)

```bash
--train <path>              # Training JSONL file
--val <path>                # Validation JSONL file
--model <name>              # Base model (default: unsloth/Qwen2.5-7B-Instruct)
--epochs <int>              # Training epochs (default: 1)
--batch-size <int>          # Per-device batch size (default: 2)
--grad-accum <int>          # Gradient accumulation steps (default: 4)
--lr <float>                # Learning rate (default: 2e-4)
--max-seq-len <int>         # Max sequence length (default: 512)
--lora-rank <int>           # LoRA rank (default: 16)
--lora-alpha <int>          # LoRA alpha (default: 16)
--eval-steps <int>          # Evaluation frequency (default: 50)
--save-steps <int>          # Checkpoint frequency (default: 100)
--resume-from-checkpoint    # Resume from checkpoint directory
--push_to_hub               # Auto-push to HuggingFace Hub
--hub_username <name>       # Hub username
--hub_repo_name <name>      # Hub repository name
--use_wandb                 # Enable W&B logging
--wandb_project <name>      # W&B project name
--wandb_run_name <name>     # W&B run name
```

### FinTS (train_lgbm.py, train_patchtst.py)

```bash
--asset-class <class>       # Asset class: crypto, forex, equities, commodities, all
--max-per-class <int>       # Max tickers per class (default: 20)
--splits <int>              # Walk-forward splits (default: 5)
--min-rows <int>            # Min rows per ticker (default: 1000)
--epochs <int>              # Training epochs [PatchTST only] (default: 50)
--batch-size <int>          # Batch size [PatchTST only] (default: 32)
--push_to_hub               # Auto-push to HuggingFace Hub
--hub_username <name>       # Hub username
--hub_repo_name <name>      # Hub repository name
--use_wandb                 # Enable W&B logging
--wandb_project <name>      # W&B project name
--wandb_run_name <name>     # W&B run name
```

---

## Cost Comparison

| Platform | GPU | FinSentiment | FinTS (All) | Notes |
|----------|-----|--------------|-------------|-------|
| **Modal** | A100-80GB | $1.50/hr (~$2 total) | $0.30/hr (~$0.15 total) | Serverless, pay-per-second |
| **RunPod** | A100-80GB | $1.89/hr (~$2.50 total) | $0.44/hr (~$0.22 total) | High availability |
| **Paperspace** | A100-80GB | $3.09/hr (~$4 total) | $0.51/hr (~$0.26 total) | Managed notebooks |
| **Local** | RTX 3060 12GB | Free (2-4 hours) | Free (30-60 min) | Your hardware |

**Recommendation**: Modal for production, Docker for development.

---

## Troubleshooting

### Out of Memory (OOM)

**FinSentiment**:
- Reduce `--batch-size` to 1
- Increase `--grad-accum` to 8
- Reduce `--max-seq-len` to 256
- Use A100-80GB instead of A100-40GB

**FinTS (PatchTST)**:
- Reduce `--batch-size` to 16
- Reduce `--max-per-class` to 10
- Use T4-16GB minimum

### Hub Push Fails

```bash
# Verify token
python -c "from openmedallion.hub import verify_token; verify_token()"

# Check token scope
# Must have "write" permission in HuggingFace settings
```

### W&B Not Logging

```bash
# Verify API key
wandb login

# Check connection
python -c "import wandb; wandb.init(project='test'); wandb.finish()"
```

### Data Not Found

- **Modal**: Upload to `/models` persistent volume first
- **RunPod**: Use network volumes or upload to S3
- **Paperspace**: Use `/storage` or Gradient Datasets
- **Docker**: Mount local data directory with `-v`

---

## Best Practices

1. **Start small**: Test with 1 asset class and 1 epoch
2. **Monitor costs**: Set budget alerts on all platforms
3. **Use W&B**: Track experiments across platforms
4. **Version control**: Commit training configs to Git
5. **Hub push**: Always push final models to avoid re-training
6. **Checkpointing**: Enable `--save-steps` for long runs
7. **Resume capability**: Use `--resume-from-checkpoint` if interrupted

---

## Next Steps

After training:
1. Verify Hub push: `https://huggingface.co/<username>/<repo>`
2. Download and test: See [DEPLOYMENT.md](DEPLOYMENT.md)
3. Evaluate performance: Run backtest scripts
4. Monitor drift: Track metrics over time with W&B

---

## Support

- **Modal docs**: https://modal.com/docs
- **RunPod docs**: https://docs.runpod.io
- **Paperspace docs**: https://docs.paperspace.com/gradient
- **Docker docs**: https://docs.docker.com

For OpenMedallion issues: Open GitHub issue or contact maintainers.
