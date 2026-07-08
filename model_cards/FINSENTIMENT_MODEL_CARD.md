---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- finance
- sentiment-analysis
- qwen
- qlora
- financial-news
datasets:
- financial_phrasebank
pipeline_tag: text-classification
model-index:
- name: OpenMedallion FinSentiment
  results:
  - task:
      type: text-classification
      name: Financial Sentiment Analysis
    dataset:
      name: Financial PhraseBank
      type: financial_phrasebank
    metrics:
    - type: accuracy
      value: 0.XX
      name: Accuracy
    - type: f1
      value: 0.XX
      name: F1 Score
---

# OpenMedallion FinSentiment

## Model Description

**OpenMedallion FinSentiment** is a fine-tuned [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model specialized for financial sentiment analysis. The model classifies financial news headlines and statements into three categories: **positive**, **neutral**, and **negative**.

This model is part of the [OpenMedallion](https://github.com/yourusername/OpenMedallion) project, an open-source financial ML toolkit for sentiment analysis and time-series forecasting.

### Key Features

- 🎯 **Specialized for Finance**: Fine-tuned on Financial PhraseBank dataset
- ⚡ **Efficient Training**: QLoRA (4-bit quantization + LoRA adapters) enables training on consumer GPUs
- 🔄 **Chat Template Format**: Uses Qwen's chat template for consistent inference
- 📊 **High Accuracy**: Achieves competitive performance on financial sentiment classification
- 🚀 **Production Ready**: Optimized for inference with BF16 precision

## Intended Use

### Primary Use Cases

- **Financial News Analysis**: Classify sentiment of market news and headlines
- **Trading Signal Generation**: Extract sentiment signals for quantitative trading strategies
- **Risk Assessment**: Analyze sentiment trends for risk management
- **Market Research**: Aggregate sentiment across multiple news sources

### Out-of-Scope Uses

- General-purpose sentiment analysis (not optimized for non-financial text)
- Multi-language sentiment analysis (trained on English only)
- Real-time trading decisions without human oversight
- Legal or compliance decisions

## Training Details

### Training Data

- **Dataset**: [Financial PhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)
- **Size**: ~4,840 sentences from financial news
- **Labels**: Positive, Neutral, Negative (3 classes)
- **Agreement**: Sentences with ≥66% annotator agreement
- **Split**: 80% train / 10% validation / 10% test

### Training Procedure

**Base Model**: Qwen2.5-1.5B-Instruct  
**Method**: QLoRA (4-bit quantization + LoRA fine-tuning)  
**Precision**: BF16

#### Hyperparameters

```python
# LoRA Configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Arguments
learning_rate = 2e-4
batch_size = 8
gradient_accumulation_steps = 4
max_steps = 500
warmup_steps = 100
max_seq_length = 512

# Optimization
optimizer = "paged_adamw_8bit"
lr_scheduler_type = "cosine"
weight_decay = 0.01
```

#### Training Environment

- **GPU**: NVIDIA RTX 2060 Super (8GB VRAM) / A100 (cloud)
- **Framework**: PyTorch + Transformers + PEFT + BitsAndBytes
- **Mixed Precision**: BF16
- **Gradient Checkpointing**: Enabled

### Evaluation Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 0.XX | 0.XX | 0.XX |
| **F1 Score** | 0.XX | 0.XX | 0.XX |
| **Precision** | 0.XX | 0.XX | 0.XX |
| **Recall** | 0.XX | 0.XX | 0.XX |

*Note: Update these metrics with your actual training results.*

## Usage

### Installation

```bash
pip install transformers torch peft bitsandbytes accelerate
```

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "your-username/openmedallion-finsentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Prepare input
text = "Apple reported record quarterly revenue, beating analyst expectations."
messages = [
    {"role": "system", "content": "You are a financial sentiment analyzer. Classify the sentiment as positive, neutral, or negative."},
    {"role": "user", "content": f"Classify the sentiment: {text}"}
]

# Generate prediction
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=10,
    do_sample=False,
    temperature=0.0
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using OpenMedallion Package

```python
from openmedallion.hub import from_pretrained

# Download model
model_path = from_pretrained(
    repo_id="your-username/openmedallion-finsentiment",
    model_type="finsentiment"
)

# Load for inference
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

### Batch Inference

```python
texts = [
    "Company X announced layoffs affecting 10% of workforce",
    "Stock market reaches all-time high amid economic recovery",
    "Federal Reserve maintains interest rates unchanged"
]

results = []
for text in texts:
    messages = [
        {"role": "system", "content": "You are a financial sentiment analyzer."},
        {"role": "user", "content": f"Classify: {text}"}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=10)
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(sentiment)
```

## Limitations and Biases

### Known Limitations

- **English Only**: Trained exclusively on English financial text
- **Context Window**: Limited to 512 tokens; longer texts require truncation
- **Domain Specificity**: Optimized for financial news; may not generalize to other financial text types
- **Temporal Bias**: Training data reflects historical market conditions
- **Label Imbalance**: Dataset may have class imbalance favoring neutral sentiment

### Potential Biases

- **Source Bias**: Financial PhraseBank draws from specific news sources
- **Annotator Bias**: Human annotation reflects subjective interpretation
- **Market Regime Bias**: Training data represents specific market cycles
- **Company Size Bias**: May perform differently on large-cap vs small-cap company news

### Risk Mitigation

- Always validate model outputs with domain experts
- Use ensemble methods combining multiple signals
- Implement confidence thresholds for production deployment
- Monitor model performance across different market conditions
- Regular retraining with recent data

## Ethical Considerations

⚠️ **Important**: This model is provided for research and educational purposes. Financial decisions should never be based solely on automated sentiment analysis.

### Responsible Use Guidelines

1. **No Autonomous Trading**: Always require human oversight for financial decisions
2. **Transparency**: Disclose the use of AI-generated sentiment in reports
3. **Fairness**: Monitor for disparate impact across different companies/sectors
4. **Accountability**: Maintain audit trails for model predictions
5. **Privacy**: Ensure compliance with data privacy regulations when processing news

## Citation

If you use this model in your research, please cite:

```bibtex
@software{openmedallion_finsentiment,
  title={OpenMedallion FinSentiment: Financial Sentiment Analysis with Qwen},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/OpenMedallion}
}
```

## References

- **Base Model**: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Dataset**: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology.
- **QLoRA**: Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.

## Model Card Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/yourusername/OpenMedallion/issues).

## License

This model is released under the Apache 2.0 License. See [LICENSE](https://github.com/yourusername/OpenMedallion/blob/main/LICENSE) for details.

---

**Last Updated**: 2026-07-08  
**Model Version**: 1.0.0  
**OpenMedallion Version**: 1.0.0
