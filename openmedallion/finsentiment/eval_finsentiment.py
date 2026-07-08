#!/usr/bin/env python3
"""
Evaluate OpenMedallion-FinSentiment model on test set.

Loads fine-tuned Qwen2.5 model and evaluates sentiment classification
accuracy, precision, recall, F1 on validation/test data.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm


def format_chat_prompt(text: str, tokenizer) -> str:
    """
    Format text as Qwen chat prompt for sentiment inference.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a financial sentiment analysis expert. Classify the sentiment of the given financial text as 'positive', 'negative', or 'neutral'."
        },
        {
            "role": "user",
            "content": f"Classify the sentiment of this financial text:\n\n{text}"
        }
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def predict_sentiment(model, tokenizer, text: str, device: str = 'cuda') -> str:
    """
    Predict sentiment for single text.
    
    Returns:
        Predicted sentiment label ('positive', 'negative', 'neutral')
    """
    prompt = format_chat_prompt(text, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated tokens
    generated_ids = outputs[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
    
    # Extract sentiment label
    if 'positive' in response:
        return 'positive'
    elif 'negative' in response:
        return 'negative'
    elif 'neutral' in response:
        return 'neutral'
    else:
        # Fallback: check first word
        first_word = response.split()[0] if response else ''
        if first_word in ['positive', 'negative', 'neutral']:
            return first_word
        return 'neutral'  # Default fallback


def evaluate_model(model, tokenizer, dataset, label_map_inv: Dict[int, str], 
                   batch_size: int = 8, max_samples: int = None) -> Dict:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    predictions = []
    ground_truth = []
    
    # Sample if max_samples specified
    eval_samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(eval_samples)} samples...")
    
    for i in tqdm(range(0, len(eval_samples), batch_size), desc="Evaluating"):
        batch = eval_samples[i:i+batch_size]
        
        for j in range(len(batch['text'])):
            text = batch['text'][j]
            true_label_id = batch['label_id'][j]
            true_label = label_map_inv[true_label_id]
            
            pred_label = predict_sentiment(model, tokenizer, text, device)
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
    
    # Calculate metrics
    label_names = ['positive', 'negative', 'neutral']
    
    accuracy = accuracy_score(ground_truth, predictions)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=label_names, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=label_names, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=label_names)
    
    results = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            label_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(label_names))
        },
        'confusion_matrix': cm.tolist(),
        'num_samples': len(predictions)
    }
    
    return results


def print_evaluation_report(results: Dict):
    """
    Print formatted evaluation report.
    """
    print("\n" + "=" * 70)
    print("OPENMEDALLION-FINSENTIMENT EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"Number of samples: {results['num_samples']}")
    
    print("\n" + "-" * 70)
    print("MACRO AVERAGES")
    print("-" * 70)
    print(f"Precision: {results['macro_precision']:.4f}")
    print(f"Recall:    {results['macro_recall']:.4f}")
    print(f"F1 Score:  {results['macro_f1']:.4f}")
    
    print("\n" + "-" * 70)
    print("WEIGHTED AVERAGES")
    print("-" * 70)
    print(f"Precision: {results['weighted_precision']:.4f}")
    print(f"Recall:    {results['weighted_recall']:.4f}")
    print(f"F1 Score:  {results['weighted_f1']:.4f}")
    
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 70)
    
    for label, metrics in results['per_class'].items():
        print(f"{label:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['support']:<12}")
    
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print("           ", "  ".join([f"{l:<10}" for l in ['positive', 'negative', 'neutral']]))
    
    labels = ['positive', 'negative', 'neutral']
    cm = results['confusion_matrix']
    
    for i, label in enumerate(labels):
        row_str = f"{label:<10} " + "  ".join([f"{cm[i][j]:<10}" for j in range(len(labels))])
        print(row_str)
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate FinSentiment model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to prepared sentiment dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation'],
                        help='Dataset split to evaluate (default: validation)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("OPENMEDALLION-FINSENTIMENT EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load label mapping
    label_map_path = dataset_dir.parent / 'label_map.json'
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    label_map_inv = {v: k for k, v in label_map.items()}
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_dir}...")
    dataset = load_from_disk(str(dataset_dir))
    eval_dataset = dataset[args.split]
    print(f"{args.split} size: {len(eval_dataset)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    # Evaluate
    results = evaluate_model(
        model, tokenizer, eval_dataset, label_map_inv,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Print report
    print_evaluation_report(results)
    
    # Save results
    results_path = output_dir / f'evaluation_results_{args.split}.json'
    print(f"\nSaving evaluation results to {results_path}...")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
