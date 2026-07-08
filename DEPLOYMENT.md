# Deployment Guide

Complete guide for deploying OpenMedallion models to HuggingFace Hub and using them in production.

---

## Overview

OpenMedallion models are automatically pushed to HuggingFace Hub after training completes. This guide covers:

1. **Manual Hub Push** - Push existing trained models
2. **Downloading Models** - Load models from Hub for inference
3. **Model Cards** - Automatically generated documentation
4. **Production Deployment** - Serve models via API

---

## 1. Manual Hub Push

If you trained models locally without automatic Hub push, you can upload them manually.

### Prerequisites

```bash
# Install HuggingFace Hub
pip install huggingface-hub

# Authenticate
export HF_TOKEN=hf_...  # Get from https://huggingface.co/settings/tokens
```

### Push FinSentiment Model

```python
from openmedallion.hub import push_to_hub, setup_token

# Setup authentication
setup_token()

# Push model directory
push_to_hub(
    path="openmedallion/finsentiment/checkpoints/qwen-sentiment-final",
    repo_name="openmedallion-finsentiment",
    username="your-username",
    repo_type="model",
    private=False
)
```

**What gets uploaded**:
- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `README.md` - Auto-generated model card

### Push FinTS Models

```python
from openmedallion.hub import push_to_hub, setup_token

setup_token()

# Push LGBM models (one per asset class)
push_to_hub(
    path="openmedallion/fints/models/lgbm_crypto.pkl",
    repo_name="openmedallion-fints-crypto",
    username="your-username"
)

push_to_hub(
    path="openmedallion/fints/models/lgbm_forex.pkl",
    repo_name="openmedallion-fints-forex",
    username="your-username"
)

# Push PatchTST model
push_to_hub(
    path="openmedallion/fints/models/patchtst_equities.pth",
    repo_name="openmedallion-fints-equities",
    username="your-username"
)
```

### Push Multiple Files

```python
# Push entire directory with all asset classes
push_to_hub(
    path="openmedallion/fints/models/",
    repo_name="openmedallion-fints-all",
    username="your-username"
)
```

---

## 2. Downloading Models

Load pre-trained models from Hub for inference.

### Download FinSentiment Model

```python
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Download LoRA adapter
model_path = from_pretrained(
    repo_id="your-username/openmedallion-finsentiment",
    repo_type="model"
)

# Load base model + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inference
def predict_sentiment(text):
    prompt = f"Analyze the sentiment of this financial text:\n\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
text = "Apple reported record Q4 earnings, beating analyst expectations."
print(predict_sentiment(text))
# Output: "Positive sentiment. The company exceeded forecasts..."
```

### Download FinTS Models

```python
from openmedallion.hub import from_pretrained
import joblib
import torch

# Download LGBM model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-crypto",
    filename="lgbm_crypto.pkl"
)
lgbm_model = joblib.load(model_path)

# Predict next-day returns
import pandas as pd
features = pd.DataFrame({
    'returns_1d': [0.02],
    'returns_5d': [0.05],
    'volume_ratio': [1.2],
    # ... other features
})
prediction = lgbm_model.predict(features)
print(f"Predicted return: {prediction[0]:.4f}")

# Download PatchTST model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-equities",
    filename="patchtst_equities.pth"
)
patchtst_model = torch.load(model_path)
patchtst_model.eval()

# Predict multi-step forecast
import numpy as np
lookback_window = np.random.randn(1, 96, 5)  # [batch, time, features]
forecast = patchtst_model(torch.tensor(lookback_window, dtype=torch.float32))
print(f"7-day forecast: {forecast.detach().numpy()}")
```

### List Available Files

```python
from openmedallion.hub import list_files

# List all files in a repo
files = list_files("your-username/openmedallion-fints-all")
for file_info in files:
    print(f"{file_info['path']} - {file_info['size'] / 1024:.1f} KB")

# Output:
# lgbm_crypto.pkl - 2.3 MB
# lgbm_forex.pkl - 1.8 MB
# lgbm_equities.pkl - 3.1 MB
# patchtst_crypto.pth - 45.2 MB
# README.md - 5.2 KB
```

---

## 3. Model Cards

Every model upload includes an auto-generated README.md (model card) with:

### FinSentiment Model Card

```markdown
# OpenMedallion FinSentiment

Fine-tuned Qwen2.5-7B for financial sentiment analysis.

## Model Details
- Base Model: unsloth/Qwen2.5-7B-Instruct
- Method: QLoRA (4-bit + rank-16 adapters)
- Training Data: 10K financial texts (news, earnings, tweets)
- Metrics: 0.89 F1 score, 0.91 accuracy

## Usage
[Code example here]

## Training Details
- Epochs: 1
- Batch Size: 2 (effective 8 with grad accumulation)
- Learning Rate: 2e-4
- GPU: A100-80GB
- Training Time: 1.5 hours

## Limitations
- English-only
- Trained on 2020-2024 data
- May not generalize to non-financial domains

## Citation
[BibTeX here]
```

### FinTS Model Card

```markdown
# OpenMedallion FinTS - Crypto

Time-series forecasting for cryptocurrency returns.

## Model Details
- Type: LightGBM Gradient Boosting
- Task: Next-day return prediction
- Features: Technical indicators (20), price history (5 days)
- Metrics: MAE 0.012, RMSE 0.018, Sharpe 1.4

## Usage
[Code example here]

## Training Details
- Asset Class: Cryptocurrencies (BTC, ETH, SOL, ...)
- Tickers: 200
- Walk-Forward Splits: 5
- Features: 25 (returns, volume, volatility, RSI, MACD)

## Performance
| Metric | Value |
|--------|-------|
| MAE | 0.012 |
| RMSE | 0.018 |
| Sharpe Ratio | 1.42 |
| Max Drawdown | -12.3% |

## Limitations
- Trained on historical data (no guarantee of future performance)
- Single-day forecast horizon
- Does not account for transaction costs

## Citation
[BibTeX here]
```

### Customize Model Cards

```python
from openmedallion.hub import push_to_hub

# Custom model card with additional metadata
push_to_hub(
    path="models/lgbm_crypto.pkl",
    repo_name="openmedallion-fints-crypto",
    username="your-username",
    model_card_metadata={
        "tags": ["time-series", "forecasting", "finance", "crypto"],
        "license": "apache-2.0",
        "datasets": ["coingecko-historical"],
        "metrics": {
            "MAE": 0.012,
            "RMSE": 0.018,
            "Sharpe": 1.42
        }
    }
)
```

---

## 4. Production Deployment

### FastAPI Inference Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import joblib
import torch

app = FastAPI(title="OpenMedallion API")

# Load models on startup
@app.on_event("startup")
async def load_models():
    global sentiment_model, sentiment_tokenizer, crypto_model
    
    # Load FinSentiment
    adapter_path = from_pretrained("your-username/openmedallion-finsentiment")
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype="auto"
    )
    sentiment_model = PeftModel.from_pretrained(base_model, adapter_path)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load FinTS
    crypto_model_path = from_pretrained(
        "your-username/openmedallion-fints-crypto",
        filename="lgbm_crypto.pkl"
    )
    crypto_model = joblib.load(crypto_model_path)

class SentimentRequest(BaseModel):
    text: str

class ForecastRequest(BaseModel):
    features: dict  # {feature_name: value}

@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    prompt = f"Analyze sentiment: {request.text}"
    inputs = sentiment_tokenizer(prompt, return_tensors="pt").to(sentiment_model.device)
    outputs = sentiment_model.generate(**inputs, max_new_tokens=50)
    result = sentiment_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"sentiment": result, "text": request.text}

@app.post("/forecast/crypto")
async def forecast_crypto(request: ForecastRequest):
    import pandas as pd
    df = pd.DataFrame([request.features])
    prediction = crypto_model.predict(df)[0]
    
    return {
        "predicted_return": float(prediction),
        "confidence": "high" if abs(prediction) > 0.02 else "low"
    }

# Run with: uvicorn deployment:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models on build (optional)
ENV HF_TOKEN=hf_...
RUN python -c "from openmedallion.hub import from_pretrained; \
    from_pretrained('your-username/openmedallion-finsentiment'); \
    from_pretrained('your-username/openmedallion-fints-crypto')"

# Start server
CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t openmedallion-api .
docker run --gpus all -p 8000:8000 -e HF_TOKEN=$HF_TOKEN openmedallion-api
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openmedallion-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openmedallion
  template:
    metadata:
      labels:
        app: openmedallion
    spec:
      containers:
      - name: api
        image: openmedallion-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface
              key: token
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 24Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
---
apiVersion: v1
kind: Service
metadata:
  name: openmedallion-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: openmedallion
```

Deploy:

```bash
kubectl create secret generic huggingface --from-literal=token=hf_...
kubectl apply -f deployment.yaml
```

---

## 5. Cache Management

Models are cached locally after first download to `~/.cache/huggingface/hub/`.

### Check Cache

```python
from openmedallion.hub import get_cache_dir

cache_dir = get_cache_dir()
print(f"Cache directory: {cache_dir}")

# List cached models
import os
for model_dir in os.listdir(cache_dir):
    print(f"- {model_dir}")
```

### Clear Cache

```python
from openmedallion.hub import clear_cache

# Clear all cached models
clear_cache()

# Clear specific model
clear_cache(repo_id="your-username/openmedallion-finsentiment")
```

---

## 6. Versioning & Updates

### Update Existing Model

```python
from openmedallion.hub import push_to_hub

# Push new version with commit message
push_to_hub(
    path="models/lgbm_crypto_v2.pkl",
    repo_name="openmedallion-fints-crypto",
    username="your-username",
    commit_message="Retrained with 2025 data, improved MAE by 8%"
)
```

### Download Specific Version

```python
from openmedallion.hub import from_pretrained

# Download specific commit
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-crypto",
    revision="a1b2c3d4"  # Git commit hash
)
```

### Pin to Version in Production

```python
# production_config.py
MODELS = {
    "sentiment": {
        "repo": "your-username/openmedallion-finsentiment",
        "revision": "v1.0.0",  # Git tag
        "hash": "abc123..."    # Model hash for integrity check
    },
    "crypto": {
        "repo": "your-username/openmedallion-fints-crypto",
        "revision": "main",
        "hash": "def456..."
    }
}
```

---

## 7. Monitoring & Logging

### Track Model Performance

```python
import wandb
from openmedallion.hub import from_pretrained

wandb.init(project="openmedallion-prod", name="crypto-forecast")

# Load model
model_path = from_pretrained("your-username/openmedallion-fints-crypto")
model = joblib.load(model_path)

# Log predictions
for data_batch in dataloader:
    predictions = model.predict(data_batch)
    actuals = get_next_day_returns(data_batch)
    
    mae = mean_absolute_error(actuals, predictions)
    wandb.log({"mae": mae, "timestamp": datetime.now()})
```

### Hub Model Analytics

View model stats on HuggingFace:
- **Downloads**: Track adoption
- **Likes**: Community feedback
- **Comments**: User issues and questions

Access via API:

```python
from huggingface_hub import HfApi

api = HfApi()
model_info = api.model_info("your-username/openmedallion-finsentiment")

print(f"Downloads: {model_info.downloads}")
print(f"Likes: {model_info.likes}")
print(f"Last updated: {model_info.last_modified}")
```

---

## 8. Troubleshooting

### Model Download Fails

```python
# Check repo exists
from huggingface_hub import HfApi
api = HfApi()
try:
    api.model_info("your-username/openmedallion-finsentiment")
    print("✓ Repo exists")
except:
    print("✗ Repo not found or private")

# Check authentication
from openmedallion.hub import verify_token
verify_token()
```

### Out of Disk Space

```python
from openmedallion.hub import get_cache_dir, clear_cache
import shutil

cache_dir = get_cache_dir()
cache_size = shutil.disk_usage(cache_dir).used / (1024**3)
print(f"Cache size: {cache_size:.2f} GB")

# Clear old models
clear_cache()
```

### Model Version Mismatch

```python
# Always specify expected model format
from openmedallion.hub import from_pretrained

try:
    model_path = from_pretrained("your-username/openmedallion-fints-crypto")
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Invalid model format"
except Exception as e:
    print(f"Model load failed: {e}")
    # Fallback to local model
    model = joblib.load("models/lgbm_crypto_backup.pkl")
```

---

## Best Practices

1. **Version Control**: Tag releases with semantic versioning (v1.0.0)
2. **Model Cards**: Keep README.md updated with metrics and limitations
3. **Private Repos**: Use for proprietary models (`private=True` in push_to_hub)
4. **Cache Management**: Clear cache regularly to save disk space
5. **Health Checks**: Monitor prediction latency and accuracy drift
6. **Rollback Plan**: Keep previous model versions for quick rollback
7. **API Rate Limits**: Respect HuggingFace Hub API limits (use caching)
8. **Security**: Never commit HF_TOKEN to Git; use environment variables

---

## Next Steps

- **Fine-tune models**: See [TRAINING.md](TRAINING.md)
- **Evaluate performance**: Run backtest scripts
- **Deploy to production**: Use FastAPI/Docker examples above
- **Monitor drift**: Track metrics over time

---

## Support

- **HuggingFace Docs**: https://huggingface.co/docs/hub
- **OpenMedallion Issues**: GitHub issue tracker
- **Community**: HuggingFace forums

For deployment questions, open a GitHub issue with logs and environment details.
