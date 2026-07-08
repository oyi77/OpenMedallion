#!/usr/bin/env python3
"""
Prepare sentiment training data from OpenMedallion dataset.

Loads fingpt-headline data directly from cached parquet file,
parses instruction-formatted text to extract sentiment labels.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def parse_instruction_text(text: str) -> Optional[Dict[str, str]]:
    """
    Parse instruction-formatted text into components.
    
    Expected format:
        ## [Question text]
        
        ### Input
        [headline text]
        
        ### Response
        [Yes/No]
    
    Returns:
        Dictionary with 'question', 'headline', 'response' keys, or None if parsing fails.
    """
    # Match question (## followed by question text)
    question_match = re.search(r'##\s+(.+?)(?:\n|$)', text)
    
    # Match input section (### Input followed by headline)
    input_match = re.search(r'### Input\s*\n(.+?)(?:\n\n|\n###)', text, re.DOTALL)
    
    # Match response section (### Response followed by Yes/No)
    response_match = re.search(r'### Response\s*\n(.+?)(?:\n|$)', text, re.DOTALL)
    
    if not (question_match and input_match and response_match):
        return None
    
    return {
        'question': question_match.group(1).strip(),
        'headline': input_match.group(1).strip(),
        'response': response_match.group(1).strip()
    }


def is_sentiment_question(question: str) -> bool:
    """Check if question is sentiment-related (includes price movement questions)."""
    sentiment_keywords = [
        'positive', 'negative', 'neutral',
        'bullish', 'bearish',
        'optimistic', 'pessimistic',
        'sentiment',
        'good news', 'bad news',
        'price going up', 'price going down', 'price staying constant',
        'price in the past', 'price in the future',
        'asset', 'compare'
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in sentiment_keywords)


def extract_sentiment_label(question: str, response: str) -> Optional[str]:
    """
    Map Yes/No response to sentiment label based on question type.
    
    Price movement mapping:
    - "price going up?" + "Yes" -> "positive"
    - "price going down?" + "Yes" -> "negative"
    - "price staying constant?" + "Yes" -> "neutral"
    
    Direct sentiment mapping:
    - "positive sentiment?" + "Yes" -> "positive"
    - "negative news?" + "Yes" -> "negative"
    - "bearish?" + "No" -> "positive"
    """
    question_lower = question.lower()
    response_lower = response.lower()
    is_yes = response_lower == 'yes'
    
    # Price movement questions (fingpt-headline format)
    if 'price going up' in question_lower or 'price in the future' in question_lower:
        return 'positive' if is_yes else ('negative' if response_lower == 'no' else None)
    elif 'price going down' in question_lower or 'price in the past' in question_lower:
        return 'negative' if is_yes else ('positive' if response_lower == 'no' else None)
    elif 'price staying constant' in question_lower:
        return 'neutral' if is_yes else None
    
    # Direct sentiment questions
    if 'positive' in question_lower or 'bullish' in question_lower or 'good news' in question_lower:
        return 'positive' if is_yes else 'negative'
    elif 'negative' in question_lower or 'bearish' in question_lower or 'bad news' in question_lower:
        return 'negative' if is_yes else 'positive'
    elif 'neutral' in question_lower:
        return 'neutral' if is_yes else None
    
    return None

def prepare_sentiment_dataset(
    parquet_path: str,
    output_path: Path,
    max_samples: Optional[int] = None
) -> Dict[str, int]:
    """
    Load fingpt-headline data from parquet and extract sentiment examples.
    
    Args:
        parquet_path: Path to existing_p5.parquet file
        output_path: Output JSONL file path
        max_samples: Maximum samples to process (None = all)
    
    Returns:
        Dictionary with label distribution counts
    """
    # Load parquet file
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to first {len(df)} samples")
    
    # Process samples
    sentiment_examples = []
    skipped = 0
    label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    print("\nProcessing samples...")
    for idx, row in df.iterrows():
        text = row.get('text', '')
        if not text:
            skipped += 1
            continue
        
        # Parse instruction format
        parsed = parse_instruction_text(text)
        if not parsed:
            skipped += 1
            continue
        
        question = parsed['question']
        headline = parsed['headline']
        response = parsed['response']
        
        # Check if sentiment-related
        if not is_sentiment_question(question):
            skipped += 1
            continue
        
        # Extract sentiment label
        label = extract_sentiment_label(question, response)
        if not label:
            skipped += 1
            continue
        
        # Create training example
        sentiment_examples.append({
            'headline': headline,
            'sentiment': label
        })
        label_counts[label] += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples, extracted {len(sentiment_examples)} sentiment examples")
    
    print(f"\n✓ Processing complete:")
    print(f"  Total samples: {len(df)}")
    print(f"  Extracted: {len(sentiment_examples)}")
    print(f"  Skipped: {skipped}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = (count / len(sentiment_examples) * 100) if sentiment_examples else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in sentiment_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✓ Wrote {len(sentiment_examples)} examples to {output_path}")
    return label_counts


def main():
    import argparse
    
    # Default to cached parquet file location
    default_parquet = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--oyi77--OpenMedallion/"
        "snapshots/006f38c73a17da4bd0953102713b6ea63356693d/"
        "data/training/ai/existing_p5.parquet"
    )
    
    parser = argparse.ArgumentParser(
        description='Prepare sentiment training data from OpenMedallion dataset'
    )
    parser.add_argument(
        '--parquet-path',
        type=str,
        default=default_parquet,
        help='Path to existing_p5.parquet file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/train.jsonl'),
        help='Output JSONL file path (default: data/train.jsonl)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify parquet file exists
    if not Path(args.parquet_path).exists():
        print(f"Error: Parquet file not found: {args.parquet_path}")
        return
    
    # Prepare dataset
    prepare_sentiment_dataset(
        parquet_path=args.parquet_path,
        output_path=args.output,
        max_samples=args.max_samples
    )
    
    print("\n✓ Sentiment data preparation complete!")


if __name__ == '__main__':
    main()
