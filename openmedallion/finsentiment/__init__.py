"""
OpenMedallion FinSentiment Module

Fine-tuned financial sentiment analysis using Qwen models.
"""

# Data Preparation
from .prepare_sentiment_data import (
    prepare_sentiment_dataset,
    parse_instruction_text,
    is_sentiment_question,
    extract_sentiment_label,
)

# Fine-tuning
from .fine_tune_qwen import (
    format_chat_prompt,
    preprocess_function,
    compute_metrics,
)

# Evaluation
from .eval_finsentiment import (
    predict_sentiment,
    evaluate_model,
    print_evaluation_report,
)

__all__ = [
    # Data Preparation
    'prepare_sentiment_dataset',
    'parse_instruction_text',
    'is_sentiment_question',
    'extract_sentiment_label',
    # Fine-tuning
    'format_chat_prompt',
    'preprocess_function',
    'compute_metrics',
    # Evaluation
    'predict_sentiment',
    'evaluate_model',
    'print_evaluation_report',
]
