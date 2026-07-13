#!/usr/bin/env python3
"""
Download fingpt-sentiment-train dataset from HuggingFace and prepare for training.
"""

import json
from pathlib import Path
from datasets import load_dataset


def download_and_prepare(output_dir: Path, max_samples: int = 50000):
    """Download fingpt-sentiment-train dataset and save as JSONL."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.jsonl"
    
    print(f"Loading fingpt-sentiment-train dataset (max {max_samples} samples)...")
    dataset = load_dataset('FinGPT/fingpt-sentiment-train', split='train', streaming=True)
    
    sentiment_examples = []
    label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    print("Processing samples...")
    for idx, example in enumerate(dataset):
        if idx >= max_samples:
            break
            
        if idx % 1000 == 0:
            print(f"  Processed {idx} samples...")
        
        input_text = example['input']
        label = example['output'].strip().lower()
        
        if label not in ['positive', 'negative', 'neutral']:
            continue
            
        # Format as chat messages for Qwen
        messages = [
            {"role": "user", "content": f"Analyze the sentiment of this financial news:\n\n{input_text}"},
            {"role": "assistant", "content": label}
        ]
        
        sentiment_examples.append({
            "messages": messages,
            "label": label
        })
        label_counts[label] += 1
    
    print(f"\nCollected {len(sentiment_examples)} sentiment examples")
    print(f"Label distribution: {label_counts}")
    
    # Write to JSONL
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        for example in sentiment_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✓ Dataset saved to {output_file}")
    print(f"  Total examples: {len(sentiment_examples)}")
    if len(sentiment_examples) > 0:
        print(f"  Positive: {label_counts['positive']} ({label_counts['positive']/len(sentiment_examples)*100:.1f}%)")
        print(f"  Negative: {label_counts['negative']} ({label_counts['negative']/len(sentiment_examples)*100:.1f}%)")
        print(f"  Neutral: {label_counts['neutral']} ({label_counts['neutral']/len(sentiment_examples)*100:.1f}%)")


if __name__ == '__main__':
    output_dir = Path('data/sentiment')
    download_and_prepare(output_dir, max_samples=50000)
