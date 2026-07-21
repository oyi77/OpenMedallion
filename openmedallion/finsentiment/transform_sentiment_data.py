#!/usr/bin/env python3
"""
Transform raw sentiment data from {headline, sentiment} format to {messages, label}
format expected by fine_tune_qwen.py, and create label_map.json.

Also fixes the system-prompt drift: training preprocess_function uses
"You are a financial sentiment analysis assistant." (terse) while
format_chat_prompt uses a verbose variant. We store messages without system
prompts (they are added during tokenization) but ensure both paths align.
"""

import json
import argparse
from pathlib import Path

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def transform(input_path: Path, output_dir: Path) -> int:
    """Read headline+sentiment JSONL, write messages+label JSONL + label_map.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            headline = row["headline"]
            sentiment = row["sentiment"].lower()

            # Build messages format as expected by preprocess_function:
            # only the user message; system+assistant are added during tokenization
            row_out = {
                "messages": [{"role": "user", "content": headline}],
                "label": sentiment,
            }
            fout.write(json.dumps(row_out) + "\n")
            count += 1

    # Write label_map.json at the parent dir (as fine_tune_qwen.py expects)
    label_map_path = output_dir.parent / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"Wrote {count} transformed rows to {output_path}")
    print(f"Wrote label_map to {label_map_path}: {LABEL_MAP}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Transform sentiment data to training format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="trained_models/finsentiment/data/train.jsonl",
        help="Source JSONL with {headline, sentiment} rows",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models/finsentiment/data/sentiment",
        help="Output dir containing train.jsonl (parent gets label_map.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    count = transform(input_path, Path(args.output_dir))
    print(f"\nTransformed {count} examples. Ready for:")
    print(f"  --dataset-dir {args.output_dir}")


if __name__ == "__main__":
    main()
