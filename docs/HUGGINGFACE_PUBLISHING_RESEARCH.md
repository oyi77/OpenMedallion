# HuggingFace Model Publishing Research

**Research Date:** 2026-07-07  
**Purpose:** Document best practices for publishing OpenMedallion models to HuggingFace Hub

---

## Repositories Analyzed

1. **Unsloth Phi-4** (`unsloth/Phi-4`)
2. **Google BERT** (`google-bert/bert-base-uncased`)
3. **Facebook OPT** (`facebook/opt-350m`)
4. **Microsoft Phi-2** (`microsoft/phi-2`)
5. **Sentence Transformers** (`sentence-transformers/all-MiniLM-L6-v2`)
6. **HuggingFace Transformers** (GitHub)
7. **HuggingFace Diffusers** (GitHub)
8. **scikit-learn** (GitHub)
9. **PyTorch** (GitHub)

---

## Key Patterns Identified

### 1. YAML Frontmatter (Critical)

**All HuggingFace model cards start with YAML frontmatter** defining metadata:

```yaml
---
language: en
license: apache-2.0
tags:
- text-generation
- finance
- time-series
pipeline_tag: text-generation
library_name: transformers
base_model: <parent-model-if-finetuned>
datasets:
- oyi77/OpenMedallion
---
```

**Required fields:**
- `language` — language code(s)
- `license` — SPDX identifier or "other"
- `tags` — discoverability keywords
- `pipeline_tag` — HF pipeline type (text-generation, text-classification, time-series-forecasting, etc.)

**Optional but recommended:**
- `library_name` — transformers, lightgbm, pytorch, etc.
- `base_model` — if fine-tuned from another model
- `datasets` — training data sources
- `license_link` — if custom license

### 2. README.md Structure

**Standard sections across all analyzed repos:**

1. **Title + One-line Description**
   - Concise model purpose
   - Parameter count or key specs

2. **Model Summary / Description**
   - What the model does
   - Training objective
   - Key capabilities

3. **Intended Uses & Limitations**
   - Primary use cases
   - Out-of-scope applications
   - Known limitations

4. **How to Use**
   - Installation requirements
   - Code examples (Python)
   - Different usage patterns (pipeline, direct API)

5. **Model Variations** (if applicable)
   - Different sizes/configurations
   - Table comparing variants

6. **Training Details** (optional but recommended)
   - Dataset description
   - Training procedure
   - Hyperparameters

7. **Evaluation / Performance**
   - Benchmark results
   - Metrics and scores
   - Comparison to baselines

8. **Limitations and Bias**
   - Known failure modes
   - Bias warnings
   - Safety considerations

9. **Citation**
   - BibTeX entry
   - Paper references

**Never include:**
- Raw HTML (use Markdown)
- Absolute internal paths
- Unverified performance claims
- Financial advice or guarantees

### 3. Code Examples Best Practices

**From analyzed repos:**

```python
# Always show imports explicitly
from transformers import pipeline

# Use actual model ID from HuggingFace Hub
model = pipeline('text-generation', model='namespace/model-name')

# Provide realistic input examples
result = model("What are we having for dinner?")

# Show output format
# [{'generated_text': '...'}]
```

**Key patterns:**
- Show both `pipeline` API and raw `AutoModel` usage
- Include `torch.set_default_device("cuda")` for GPU examples
- Demonstrate different generation strategies (greedy vs sampling)
- Use `>>>` prompt for interactive examples (BERT style)

### 4. Model Card Sections from BERT

**"Model description"** — explains architecture and training objective:
- What type of model (transformer, LSTM, etc.)
- Pre-training objective (MLM, CLM, etc.)
- Bidirectional vs autoregressive

**"Model variations"** — table of related checkpoints:
```markdown
| Model | #params | Language |
|-------|---------|----------|
| model-base | 110M | English |
| model-large | 340M | English |
```

**"How to use"** — multiple examples:
- Pipeline API (simplest)
- Direct model loading
- Fine-tuning examples (link to scripts)

**"Limitations and bias"** — critical disclaimers:
- Data sources and their biases
- Not suitable for X tasks
- Requires fine-tuning for Y

### 5. Advanced Features (Unsloth Pattern)

**Model Collections:**
```yaml
# In frontmatter
collections:
- unsloth/phi-4-all-versions-677eecf93784e61afe762afa
```

**Training Notebooks:**
```markdown
| Unsloth supports | Free Notebooks | Performance |
|------------------|----------------|-------------|
| Phi-4 | [▶️ Start on Colab](link) | 2x faster, 50% less memory |
```

**Community Links:**
- Discord server buttons
- GitHub repository links
- "Made with ❤️" branding

**Performance Tables:**
```markdown
| Metric | Unsloth | Baseline |
|--------|---------|----------|
| Training Speed | 2x faster | 1x |
| Memory Usage | 50% less | 100% |
```

### 6. Disclaimer Patterns

**OPT Model (Meta AI):**
> The team releasing OPT wrote an official model card, which is available in Appendix D of the paper. Content from **this** model card has been written by the Hugging Face team.

**BERT Model:**
> Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by the Hugging Face team.

**Phi-2 (Microsoft):**
> * Phi-2 is intended for QA, chat, and code purposes. The model-generated text/code should be treated as a starting point rather than a definitive solution for potential use cases.
> * Direct adoption for production tasks without evaluation is out of scope of this project.

**Key takeaway:** Always include disclaimers about:
- Model card authorship
- Intended vs out-of-scope uses
- Need for evaluation before production use

### 7. Tags and Discoverability

**Common tag categories:**
- **Task:** `text-generation`, `text-classification`, `time-series-forecasting`, `feature-extraction`
- **Domain:** `finance`, `nlp`, `code`, `math`, `chat`
- **Architecture:** `bert`, `phi`, `transformer`, `lightgbm`
- **Library:** `transformers`, `sentence-transformers`, `unsloth`
- **Special:** `exbert` (BERT viz), `conversational`, `multilingual`

**Sentence Transformers example (extensive tags):**
```yaml
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
datasets:
- s2orc
- ms_marco
- natural_questions
# ... 20+ datasets listed
```

### 8. License Best Practices

**From analyzed repos:**
- **MIT:** Phi-4, Phi-2 (with license_link for custom terms)
- **Apache 2.0:** BERT, Sentence Transformers
- **Other:** OPT (with `commercial: false`)

**For OpenMedallion models:**
- Use **Apache 2.0** (permissive, research-friendly)
- Add disclaimer about non-commercial use if needed
- Include `license_link` if custom terms apply

---

## Application to OpenMedallion Models

### OpenMedallion-FinTS YAML Frontmatter

```yaml
---
language: en
license: apache-2.0
tags:
- time-series-forecasting
- finance
- trading
- lightgbm
- transformer
- patchtst
library_name: lightgbm
datasets:
- oyi77/OpenMedallion
pipeline_tag: time-series-forecasting
---
```

### OpenMedallion-FinSentiment YAML Frontmatter

```yaml
---
language: en
license: apache-2.0
tags:
- text-classification
- sentiment-analysis
- finance
- instruction-tuning
- qwen
- lora
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
datasets:
- oyi77/OpenMedallion
pipeline_tag: text-classification
---
```

### README Structure for Both Models

1. **Title + Summary**
   - "OpenMedallion-FinTS: Financial Time-Series Forecasting with LightGBM and PatchTST"
   - Parameter counts, model types

2. **Model Description**
   - Architecture (LightGBM + PatchTST)
   - Training objective (next-day return prediction)
   - Walk-forward validation approach

3. **⚠️ Critical Disclaimers** (FIRST, before usage)
   - Backtest-only validation
   - Not financial advice
   - Market non-stationarity warnings
   - No forward-looking guarantees

4. **How to Use**
   - Installation (`pip install lightgbm torch transformers`)
   - Loading models
   - Inference examples
   - Feature engineering requirements

5. **Model Variations**
   - Table: crypto/forex/commodities/equities models
   - Parameter counts, training times

6. **Training Details**
   - Dataset (oyi77/OpenMedallion)
   - Walk-forward splits
   - Hyperparameters
   - Hardware requirements

7. **Evaluation Metrics**
   - Sharpe ratio (in-sample vs out-of-sample)
   - Hit rate
   - Max drawdown
   - RMSE
   - Regime degradation checks

8. **Limitations**
   - Equity-heavy training data
   - No live trading validation
   - Model drift in regime shifts
   - Computational requirements

9. **Citation**
   - Link to oyi77/OpenMedallion dataset
   - GitHub repository

---

## Publishing Checklist

### Before Publishing

- [ ] YAML frontmatter complete
- [ ] README follows standard structure
- [ ] ⚠️ Disclaimers prominent (top 3 sections)
- [ ] Code examples tested and work
- [ ] License file included
- [ ] Model weights uploaded
- [ ] .gitattributes configured for LFS
- [ ] Tags cover all relevant categories

### Optional Enhancements

- [ ] Create model collection (if multiple variants)
- [ ] Add Google Colab training notebook
- [ ] Performance comparison table
- [ ] Community links (Discord, GitHub)
- [ ] Custom branding/logos

### Quality Gates

- [ ] No absolute file paths in examples
- [ ] No unverified performance claims
- [ ] No financial advice language
- [ ] All code examples use actual HF model IDs
- [ ] Hardware requirements clearly stated

---

## Key Differences: OpenMedallion vs Standard NLP Models

**Standard NLP models emphasize:**
- Pre-training datasets
- Benchmark scores (GLUE, SuperGLUE)
- Fine-tuning for downstream tasks
- Model sizes and variants

**OpenMedallion models must emphasize:**
- ⚠️ **Backtest-only validation** (no live trading)
- Walk-forward temporal splits (no random shuffle)
- Regime-specific performance (bull vs bear markets)
- Hardware requirements (GPU memory, training time)
- **Not financial advice** disclaimers
- Non-stationarity warnings
- Model drift risks

**Tone difference:**
- NLP models: "pretrained on X, achieves Y score"
- OpenMedallion: "trained on historical data, backtested metrics, no forward guarantees"

---

## Final Recommendations

1. **Start with YAML frontmatter** — defines how HF indexes the model
2. **Disclaimers FIRST** — before usage examples, make risks clear
3. **Show working code** — tested examples with actual model IDs
4. **Be honest about limitations** — market regime changes, data biases, computational costs
5. **Link to dataset** — oyi77/OpenMedallion for reproducibility
6. **Keep it simple** — Markdown only, no HTML tricks
7. **Test before publishing** — verify all code examples run

---

## Next Steps

1. Update `openmedallion-fints/MODEL_CARD.md` with YAML frontmatter and restructured sections
2. Update `openmedallion-finsentiment/MODEL_CARD.md` with YAML frontmatter and restructured sections
3. Train models and capture actual metrics for model cards
4. Publish to HuggingFace Hub: `<namespace>/openmedallion-fints` and `<namespace>/openmedallion-finsentiment`
5. (Optional) Create model collection linking both models
6. (Optional) Add Google Colab training notebooks

---

**Research complete. Patterns documented. Ready for model card updates.**
