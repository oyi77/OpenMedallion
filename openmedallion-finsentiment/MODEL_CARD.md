---
license: apache-2.0
language:
- en
tags:
- sentiment-analysis
- finance
- nlp
- qwen2.5
- qlora
- peft
- 4bit
- instruction-tuning
pipeline_tag: text-classification
base_model: Qwen/Qwen2.5-7B-Instruct
datasets:
- oyi77/OpenMedallion
---

# OpenMedallion-FinSentiment

**Financial Sentiment Classification via QLoRA Fine-Tuning**

> ⚠️ **CRITICAL DISCLAIMER**: This model is for **research and analysis purposes only**. Sentiment predictions do NOT constitute financial advice. Market sentiment is one of many factors affecting asset prices, and should not be used in isolation for trading decisions.

## Model Description

OpenMedallion-FinSentiment is a financial sentiment classifier built by fine-tuning **Qwen2.5-7B-Instruct** using **QLoRA** (4-bit quantization + LoRA adapters). The model classifies financial headlines and news text into three sentiment categories:

- **Positive**: Bullish signals, growth indicators, positive earnings
- **Negative**: Bearish signals, downside risks, negative events
- **Neutral**: Factual reporting, mixed signals, ambiguous information

## Intended Use

### Primary Use Cases
- **Market Sentiment Analysis**: Aggregate sentiment from news feeds for market overview
- **Research**: Academic studies on sentiment's relationship with asset prices
- **Risk Monitoring**: Detect negative sentiment spikes for risk management
- **Feature Engineering**: Use sentiment scores as features in trading models

### Out-of-Scope Use
- ❌ Sole basis for trading decisions
- ❌ Real-time trading signals without validation
- ❌ Assuming sentiment → price causality without testing
- ❌ Ignoring sentiment lag and market efficiency

## Model Architecture

### Base Model
- **Foundation**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Parameters**: 7 billion (4-bit quantized)
- **Context Length**: 32,768 tokens
- **Fine-Tuning Method**: QLoRA (Quantized Low-Rank Adaptation)

### QLoRA Configuration
```python
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA Adapters
lora_config = LoraConfig(
    r=64,                     # Rank
    lora_alpha=16,            # Scaling factor
    target_modules=[          # All attention + MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Details
- **Epochs**: 3
- **Batch Size**: 4 (per device) × 4 (gradient accumulation) = 16 effective
- **Learning Rate**: 2e-4 with cosine schedule
- **Optimizer**: AdamW (8-bit via bitsandbytes)
- **Mixed Precision**: bfloat16
- **Hardware**: NVIDIA RTX 3060 (12GB VRAM)
- **Training Time**: ~2-4 hours

## Training Data

### Data Source
- **Dataset**: [oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion)
- **Files**: `finance_instruct_full_1-6.parquet`, `finance_alpaca_full_1.parquet`, `existing_p5.parquet`
- **Total Samples**: ~80,000 instruction-tuning examples

### Sentiment Extraction
Sentiment labels are extracted via heuristic keyword matching from instruction-tuning data:

**Positive Keywords**: profit, gain, growth, bullish, rally, surge, beat, outperform, upgrade, buy  
**Negative Keywords**: loss, decline, bearish, plunge, crash, miss, underperform, downgrade, sell, risk  
**Neutral**: Everything else or mixed signals

### Data Splits
```python
# Stratified splits to maintain class balance
train_split = 70%   # ~56,000 samples
val_split = 15%     # ~12,000 samples
test_split = 15%    # ~12,000 samples
```

## Evaluation Metrics

### Classification Performance
The model is evaluated using standard classification metrics on the held-out test set:

- **Accuracy**: Overall prediction correctness
- **Precision/Recall/F1**: Per-class performance (positive, negative, neutral)
- **Confusion Matrix**: Error pattern analysis

### Expected Performance Range
(To be updated after training)

```
Expected Results (based on similar models):
- Accuracy: 75-85%
- Macro F1: 0.70-0.80
- Class Imbalance: Neutral typically dominant
```

## Usage Example

### Data Preparation
```bash
python openmedallion-finsentiment/scripts/prepare_sentiment_data.py \
    --data-dir ~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/snapshots/*/data/training/ai/ \
    --output-dir ./data/sentiment \
    --train-split 0.70 \
    --val-split 0.15 \
    --test-split 0.15 \
    --max-samples 80000 \
    --seed 42
```

### Fine-Tuning
```bash
python openmedallion-finsentiment/scripts/fine_tune_qwen.py \
    --train-file ./data/sentiment/train.jsonl \
    --val-file ./data/sentiment/val.jsonl \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir ./outputs/finsentiment \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --lora-r 64 \
    --lora-alpha 16 \
    --max-length 512
```

### Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model + LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./outputs/finsentiment")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Format prompt with Qwen chat template
def predict_sentiment(text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a financial sentiment analyst. Classify the following text as positive, negative, or neutral."},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip().lower()

# Example usage
headline = "Apple reports record Q4 earnings, beats analyst expectations by 15%"
sentiment = predict_sentiment(headline)
print(f"Sentiment: {sentiment}")  # Expected: "positive"
```

## Limitations and Risks

### Model Limitations
1. **Keyword Heuristics**: Training labels extracted via keyword matching, not human annotation
2. **Context Dependency**: May miss sarcasm, irony, or context-dependent sentiment
3. **Domain Shift**: Trained on general financial text, may not generalize to specific sectors
4. **Temporal Lag**: News sentiment may lag or lead price movements unpredictably
5. **Market Efficiency**: In efficient markets, public sentiment is already priced in

### Data Quality Risks
- **Label Noise**: Heuristic labels may contain errors (neutral misclassified as positive/negative)
- **Class Imbalance**: Neutral category often dominant in financial news
- **Sampling Bias**: Dataset may overrepresent certain sectors or time periods

### Deployment Risks
- **False Positives**: Model may incorrectly flag neutral events as positive/negative
- **Adversarial Examples**: Carefully crafted headlines could fool the model
- **Hallucination**: LLM may generate confident but incorrect sentiment labels
- **Latency**: Real-time inference requires GPU acceleration (~100-200ms per headline)

## Ethical Considerations

- **Market Manipulation**: Automated sentiment analysis could be used to manipulate public opinion
- **Misinformation**: Model outputs are NOT fact-checked and may amplify false narratives
- **Bias Amplification**: Training data may contain sector/geographic biases
- **Algorithmic Trading**: High-frequency sentiment trading can increase market volatility
- **Privacy**: Ensure compliance with data privacy laws when processing financial news

## Technical Requirements

### Hardware
- **Training**: NVIDIA GPU with ≥12GB VRAM (RTX 3060 or better)
- **Inference**: GPU recommended for real-time use; CPU inference possible but slow

### Software
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- PEFT 0.7+
- bitsandbytes 0.41+

### Memory Footprint
- **Model (4-bit)**: ~3.5GB VRAM
- **LoRA Adapters**: ~250MB
- **Training Peak**: ~11GB VRAM (batch_size=4, gradient_accumulation=4)

## License

**Apache License 2.0**

This model is released under the Apache License 2.0. You are free to use, modify, and distribute this model for commercial or non-commercial purposes, with proper attribution.

The base model **Qwen2.5-7B-Instruct** is also licensed under Apache 2.0.

See [LICENSE](LICENSE) for full terms.

## Citation

```bibtex
@misc{openmedallion-finsentiment-2026,
  author = {oyi77},
  title = {OpenMedallion-FinSentiment: Financial Sentiment Classification via QLoRA},
  year = {2026},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub},
  howpublished = {\url{https://huggingface.co/oyi77/openmedallion-finsentiment}}
}

@article{qwen2.5,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2412.15115},
  year={2024}
}

@article{dettmers2024qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={NeurIPS},
  year={2023}
}
```

## Contact

- **Repository**: [https://huggingface.co/oyi77/openmedallion-finsentiment](https://huggingface.co/oyi77/openmedallion-finsentiment)
- **Dataset**: [https://huggingface.co/datasets/oyi77/OpenMedallion](https://huggingface.co/datasets/oyi77/OpenMedallion)
- **Issues**: Report bugs and feature requests via HuggingFace discussions

## Acknowledgments

- **Qwen Team**: For the excellent Qwen2.5-7B-Instruct base model
- **HuggingFace**: For Transformers, PEFT, and bitsandbytes libraries
- **QLoRA Authors**: For efficient 4-bit fine-tuning methodology

---

**Last Updated**: 2026-07-08
