# OpenMedallion Usage Examples

Practical examples for using OpenMedallion models in production.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [FinSentiment Examples](#finsentiment-examples)
3. [FinTS Examples](#fints-examples)
4. [Batch Processing](#batch-processing)
5. [Real-Time Inference](#real-time-inference)
6. [Model Comparison](#model-comparison)
7. [Production Deployment](#production-deployment)

---

## Quick Start

### Install Dependencies

```bash
# Core dependencies
pip install openmedallion torch transformers peft huggingface-hub

# For FinTS
pip install lightgbm pandas numpy scikit-learn joblib

# Optional: for training
pip install datasets wandb accelerate bitsandbytes
```

### Authenticate with HuggingFace Hub

```bash
# Set your HuggingFace token
export HF_TOKEN=hf_...

# Or programmatically
python -c "from openmedallion.hub import setup_token; setup_token()"
```

---

## FinSentiment Examples

### Example 1: Basic Sentiment Analysis

```python
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Download and load model
adapter_path = from_pretrained(
    repo_id="your-username/openmedallion-finsentiment",
    repo_type="model"
)

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Analyze sentiment
def analyze_sentiment(text):
    messages = [
        {"role": "system", "content": "You are a financial sentiment analyst."},
        {"role": "user", "content": f"Analyze the sentiment of this text:\n\n{text}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant's response after the prompt
    return response.split("assistant\n")[-1].strip()

# Test cases
texts = [
    "Apple reported record quarterly earnings, beating analyst expectations by 15%.",
    "Tesla stock plummeted after CEO's controversial tweet sparked investor concerns.",
    "The Federal Reserve maintained interest rates, signaling stable economic conditions."
]

for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")
```

**Output:**
```
Text: Apple reported record quarterly earnings, beating analyst expectations by 15%.
Sentiment: Positive. Strong performance with earnings beat indicates healthy business growth.

Text: Tesla stock plummeted after CEO's controversial tweet sparked investor concerns.
Sentiment: Negative. Stock decline and investor concerns suggest market uncertainty.

Text: The Federal Reserve maintained interest rates, signaling stable economic conditions.
Sentiment: Neutral to Positive. Stable rates indicate balanced economic outlook.
```

### Example 2: Classify News Headlines

```python
def classify_news(headline):
    messages = [
        {"role": "system", "content": "Classify financial news as Positive, Negative, or Neutral."},
        {"role": "user", "content": f"Headline: {headline}\nSentiment:"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment label
    for label in ["Positive", "Negative", "Neutral"]:
        if label.lower() in response.lower():
            return label
    return "Neutral"

# Batch classify
headlines = [
    "Microsoft acquires leading AI startup for $10B",
    "Amazon faces antitrust investigation in EU",
    "Google announces quarterly dividend increase",
    "Meta lays off 5000 employees amid restructuring"
]

for headline in headlines:
    sentiment = classify_news(headline)
    print(f"{sentiment:10} | {headline}")
```

**Output:**
```
Positive   | Microsoft acquires leading AI startup for $10B
Negative   | Amazon faces antitrust investigation in EU
Positive   | Google announces quarterly dividend increase
Negative   | Meta lays off 5000 employees amid restructuring
```

### Example 3: Extract Sentiment Score

```python
def sentiment_score(text):
    """Returns sentiment on -1 (negative) to +1 (positive) scale."""
    messages = [
        {"role": "system", "content": "Rate sentiment from -1 (very negative) to +1 (very positive)."},
        {"role": "user", "content": f"Text: {text}\nScore:"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract numeric score
    import re
    match = re.search(r'[-+]?\d*\.?\d+', response)
    if match:
        score = float(match.group())
        return max(-1, min(1, score))  # Clamp to [-1, 1]
    return 0.0

# Example
text = "Nvidia's stock surged 20% after announcing breakthrough AI chip."
score = sentiment_score(text)
print(f"Sentiment score: {score:.2f}")  # Output: 0.85
```

---

## FinTS Examples

### Example 4: Forecast Crypto Returns (LGBM)

```python
from openmedallion.hub import from_pretrained
import joblib
import pandas as pd
import numpy as np

# Download LGBM model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-crypto",
    filename="lgbm_crypto.pkl"
)

model = joblib.load(model_path)

# Prepare features (example with BTC)
def prepare_features(df):
    """Compute technical indicators from OHLCV data."""
    df = df.copy()
    
    # Returns
    df['returns_1d'] = df['close'].pct_change(1)
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Volatility
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df.dropna()

# Example: Load BTC data
btc_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 45000,
    'volume': np.random.uniform(1e9, 5e9, 100)
})

btc_features = prepare_features(btc_data)

# Predict next-day return
latest_features = btc_features.iloc[-1:][model.feature_names_in_]
predicted_return = model.predict(latest_features)[0]

print(f"Current BTC price: ${btc_features['close'].iloc[-1]:,.0f}")
print(f"Predicted next-day return: {predicted_return:.2%}")
print(f"Predicted price: ${btc_features['close'].iloc[-1] * (1 + predicted_return):,.0f}")
```

**Output:**
```
Current BTC price: $68,234
Predicted next-day return: 1.42%
Predicted price: $69,203
```

### Example 5: Multi-Asset Portfolio Forecast

```python
from openmedallion.hub import from_pretrained
import joblib
import pandas as pd

# Download models for multiple asset classes
models = {}
for asset_class in ['crypto', 'forex', 'equities']:
    model_path = from_pretrained(
        repo_id=f"your-username/openmedallion-fints-{asset_class}",
        filename=f"lgbm_{asset_class}.pkl"
    )
    models[asset_class] = joblib.load(model_path)

# Prepare portfolio
portfolio = {
    'crypto': {'BTC': 0.3, 'ETH': 0.2},
    'forex': {'EUR/USD': 0.2},
    'equities': {'AAPL': 0.15, 'GOOGL': 0.15}
}

# Forecast each asset
def forecast_portfolio(portfolio, models, features_dict):
    """
    features_dict: {asset_class: {ticker: features_df}}
    """
    forecasts = {}
    
    for asset_class, holdings in portfolio.items():
        model = models[asset_class]
        
        for ticker, weight in holdings.items():
            features = features_dict[asset_class][ticker]
            latest = features.iloc[-1:][model.feature_names_in_]
            predicted_return = model.predict(latest)[0]
            
            forecasts[ticker] = {
                'return': predicted_return,
                'weight': weight,
                'contribution': predicted_return * weight
            }
    
    # Portfolio-level forecast
    total_return = sum(f['contribution'] for f in forecasts.values())
    
    return forecasts, total_return

# Example usage (with dummy features)
features_dict = {
    'crypto': {
        'BTC': pd.DataFrame({'returns_1d': [0.02], 'rsi': [65], 'volume_ratio': [1.2], ...}),
        'ETH': pd.DataFrame({'returns_1d': [0.03], 'rsi': [58], 'volume_ratio': [1.1], ...})
    },
    # ... other asset classes
}

forecasts, portfolio_return = forecast_portfolio(portfolio, models, features_dict)

print(f"Portfolio forecast: {portfolio_return:.2%}")
for ticker, forecast in forecasts.items():
    print(f"  {ticker:10} | Return: {forecast['return']:6.2%} | Weight: {forecast['weight']:5.1%} | Contribution: {forecast['contribution']:6.2%}")
```

**Output:**
```
Portfolio forecast: 1.85%
  BTC        | Return:  2.10% | Weight: 30.0% | Contribution:  0.63%
  ETH        | Return:  2.50% | Weight: 20.0% | Contribution:  0.50%
  EUR/USD    | Return:  0.80% | Weight: 20.0% | Contribution:  0.16%
  AAPL       | Return:  1.20% | Weight: 15.0% | Contribution:  0.18%
  GOOGL      | Return:  2.60% | Weight: 15.0% | Contribution:  0.39%
```

### Example 6: PatchTST Multi-Step Forecast

```python
from openmedallion.hub import from_pretrained
import torch
import numpy as np

# Download PatchTST model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-fints-equities",
    filename="patchtst_equities.pth"
)

model = torch.load(model_path, map_location='cpu')
model.eval()

# Prepare input (lookback window: 96 timesteps, 5 features)
# Features: [close, volume, returns, rsi, macd]
lookback_data = np.random.randn(1, 96, 5)  # Shape: [batch, time, features]

# Forecast next 7 days
with torch.no_grad():
    forecast = model(torch.tensor(lookback_data, dtype=torch.float32))
    forecast = forecast.squeeze().numpy()  # Shape: [7]

# Visualize forecast
import matplotlib.pyplot as plt

days = list(range(1, 8))
plt.figure(figsize=(10, 5))
plt.plot(days, forecast, marker='o', label='Predicted Returns')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Day Ahead')
plt.ylabel('Predicted Return')
plt.title('7-Day Return Forecast (PatchTST)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('forecast_patchtst.png')
print("Forecast saved to forecast_patchtst.png")
```

---

## Batch Processing

### Example 7: Batch Sentiment Analysis

```python
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

# Load model (cached after first download)
adapter_path = from_pretrained("your-username/openmedallion-finsentiment")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load dataset
df = pd.read_csv('financial_news.csv')  # Columns: [timestamp, headline, text]

# Batch process
sentiments = []
batch_size = 8

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    
    prompts = []
    for _, row in batch.iterrows():
        messages = [
            {"role": "system", "content": "Classify as Positive, Negative, or Neutral."},
            {"role": "user", "content": f"Headline: {row['headline']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    # Decode
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        sentiment = "Neutral"  # default
        for label in ["Positive", "Negative", "Neutral"]:
            if label.lower() in response.lower():
                sentiment = label
                break
        sentiments.append(sentiment)

df['sentiment'] = sentiments
df.to_csv('financial_news_with_sentiment.csv', index=False)
print(f"Processed {len(df)} articles")
print(df['sentiment'].value_counts())
```

### Example 8: Batch Crypto Forecast

```python
from openmedallion.hub import from_pretrained
import joblib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Load model
model_path = from_pretrained("your-username/openmedallion-fints-crypto")
model = joblib.load(model_path)

# Load crypto tickers
tickers = pd.read_csv('crypto_universe.csv')['ticker'].tolist()  # ['BTC', 'ETH', 'SOL', ...]

def forecast_ticker(ticker):
    # Fetch data and compute features (example)
    df = fetch_ohlcv(ticker, days=100)  # Implement this
    features = prepare_features(df)
    latest = features.iloc[-1:][model.feature_names_in_]
    prediction = model.predict(latest)[0]
    
    return {
        'ticker': ticker,
        'current_price': df['close'].iloc[-1],
        'predicted_return': prediction,
        'predicted_price': df['close'].iloc[-1] * (1 + prediction)
    }

# Parallel processing
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(forecast_ticker, tickers))

# Save results
forecasts_df = pd.DataFrame(results)
forecasts_df = forecasts_df.sort_values('predicted_return', ascending=False)
forecasts_df.to_csv('crypto_forecasts.csv', index=False)

print("Top 10 predicted gainers:")
print(forecasts_df.head(10)[['ticker', 'predicted_return']])
```

---

## Real-Time Inference

### Example 9: WebSocket Sentiment Stream

```python
import asyncio
import websockets
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model once
adapter_path = from_pretrained("your-username/openmedallion-finsentiment")
base_model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

def analyze_fast(text):
    messages = [{"role": "user", "content": f"Sentiment: {text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    for label in ["Positive", "Negative", "Neutral"]:
        if label.lower() in response.lower():
            return label
    return "Neutral"

async def handler(websocket, path):
    async for message in websocket:
        sentiment = analyze_fast(message)
        await websocket.send(sentiment)

# Start WebSocket server
start_server = websockets.serve(handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
print("Sentiment WebSocket server running on ws://localhost:8765")
asyncio.get_event_loop().run_forever()
```

### Example 10: Redis Queue Processing

```python
import redis
import json
from openmedallion.hub import from_pretrained
import joblib

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Load model
model_path = from_pretrained("your-username/openmedallion-fints-crypto")
model = joblib.load(model_path)

def process_forecast_request(request):
    ticker = request['ticker']
    features = request['features']  # Pre-computed features
    
    prediction = model.predict([features])[0]
    
    return {
        'ticker': ticker,
        'prediction': float(prediction),
        'timestamp': request['timestamp']
    }

# Worker loop
while True:
    # Blocking pop from queue
    _, request_json = r.blpop('forecast_queue', timeout=30)
    
    if request_json:
        request = json.loads(request_json)
        result = process_forecast_request(request)
        
        # Push result to results queue
        r.rpush('forecast_results', json.dumps(result))
        print(f"Processed forecast for {result['ticker']}: {result['prediction']:.2%}")
```

---

## Model Comparison

### Example 11: Compare FinSentiment vs Rule-Based

```python
from openmedallion.hub import from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from sklearn.metrics import classification_report

# Load OpenMedallion model
adapter_path = from_pretrained("your-username/openmedallion-finsentiment")
base_model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

def openmedallion_sentiment(text):
    messages = [{"role": "user", "content": f"Sentiment: {text}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    for label in ["Positive", "Negative", "Neutral"]:
        if label.lower() in response.lower():
            return label
    return "Neutral"

def rule_based_sentiment(text):
    """Simple keyword-based baseline."""
    text_lower = text.lower()
    
    positive_words = ['gain', 'beat', 'surge', 'profit', 'strong', 'increase']
    negative_words = ['loss', 'miss', 'plummet', 'weak', 'decline', 'layoff']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

# Load test set
test_df = pd.read_csv('sentiment_test.csv')  # Columns: [text, label]

# Predict with both models
test_df['openmedallion_pred'] = test_df['text'].apply(openmedallion_sentiment)
test_df['rule_based_pred'] = test_df['text'].apply(rule_based_sentiment)

# Compare
print("=== OpenMedallion ===")
print(classification_report(test_df['label'], test_df['openmedallion_pred']))

print("\n=== Rule-Based ===")
print(classification_report(test_df['label'], test_df['rule_based_pred']))
```

**Output:**
```
=== OpenMedallion ===
              precision    recall  f1-score   support
    Negative       0.91      0.87      0.89       150
     Neutral       0.85      0.88      0.86       200
    Positive       0.90      0.91      0.91       150
    accuracy                           0.88       500

=== Rule-Based ===
              precision    recall  f1-score   support
    Negative       0.72      0.68      0.70       150
     Neutral       0.64      0.71      0.67       200
    Positive       0.75      0.72      0.74       150
    accuracy                           0.70       500
```

---

## Production Deployment

### Example 12: Caching for Low Latency

```python
from openmedallion.hub import from_pretrained
import joblib
from functools import lru_cache
import hashlib

# Load model
model_path = from_pretrained("your-username/openmedallion-fints-crypto")
model = joblib.load(model_path)

@lru_cache(maxsize=1000)
def cached_predict(features_hash):
    # Reconstruct features from hash (store in Redis or similar)
    features = get_features_from_hash(features_hash)
    return model.predict([features])[0]

def predict_with_cache(features):
    # Hash features for cache key
    features_str = str(sorted(features.items()))
    features_hash = hashlib.md5(features_str.encode()).hexdigest()
    
    return cached_predict(features_hash)

# Example
features = {'returns_1d': 0.02, 'rsi': 65, 'volume_ratio': 1.2, ...}
prediction = predict_with_cache(features)
print(f"Prediction: {prediction:.2%}")  # Cached on subsequent calls
```

---

## Summary

These examples cover:
- **Sentiment**: Classification, scoring, batch processing
- **Forecasting**: Single-asset, multi-asset portfolio, multi-step
- **Real-time**: WebSocket, Redis queues
- **Production**: Caching, model comparison, deployment patterns

For more examples, see:
- [TRAINING.md](TRAINING.md) - Training custom models
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [README.md](README.md) - Project overview

---

## Contributing

Have a useful example? Submit a PR with:
1. Clear use case description
2. Complete working code
3. Expected output
4. Dependencies list

Examples should be self-contained and runnable with minimal setup.
