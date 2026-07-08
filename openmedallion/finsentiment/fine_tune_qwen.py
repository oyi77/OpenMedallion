#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B-Instruct for financial sentiment classification using QLoRA.

Uses 4-bit quantization with LoRA adapters for efficient training on 12GB VRAM.
Implements sentiment classification as a chat completion task.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np


def format_chat_prompt(text: str, label: str = None, tokenizer=None) -> str:
    """
    Format text as Qwen chat prompt for sentiment classification.
    
    Args:
        text: Financial text/headline
        label: Optional ground truth label ('positive', 'negative', 'neutral')
        tokenizer: Tokenizer instance for chat template
    
    Returns:
        Formatted chat prompt string
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
    
    if label is not None:
        messages.append({
            "role": "assistant",
            "content": label
        })
    
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(label is None))
    
    # Fallback manual format if no tokenizer
    prompt = "<|im_start|>system\nYou are a financial sentiment analysis expert. Classify the sentiment of the given financial text as 'positive', 'negative', or 'neutral'.<|im_end|>\n"
    prompt += f"<|im_start|>user\nClassify the sentiment of this financial text:\n\n{text}<|im_end|>\n"
    
    if label is not None:
        prompt += f"<|im_start|>assistant\n{label}<|im_end|>"
    else:
        prompt += "<|im_start|>assistant\n"
    
    return prompt


def preprocess_function(examples: Dict, tokenizer, label_map_inv: Dict[int, str], max_length: int = 512):
    """
    Preprocess dataset examples into tokenized chat format.
    """
    texts = []
    
    for i in range(len(examples['text'])):
        text = examples['text'][i]
        label_id = examples['label_id'][i]
        label = label_map_inv[label_id]
        
        prompt = format_chat_prompt(text, label, tokenizer)
        texts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Set labels for causal LM (same as input_ids)
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized


def compute_metrics(eval_pred):
    """
    Compute accuracy metric during evaluation.
    """
    logits, labels = eval_pred
    
    # For causal LM, we need to extract predictions
    # This is simplified - actual eval uses generate()
    predictions = np.argmax(logits, axis=-1)
    
    # Compare only non-padding tokens
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()
    
    return {'accuracy': accuracy}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen2.5-7B for sentiment classification')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to prepared sentiment dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for fine-tuned model')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='Base model name (default: Qwen/Qwen2.5-7B-Instruct)')
    
    # LoRA hyperparameters
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (default: 32)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size (default: 4)')
    parser.add_argument('--gradient-accum', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Warmup steps (default: 100)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Max sequence length (default: 512)')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Evaluation steps (default: 500)')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint steps (default: 500)')

    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (default: None)")

    # Resume from checkpoint
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="openmedallion-finsentiment",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (optional)")
    
    # HuggingFace Hub integration
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hub after training")
    parser.add_argument("--hub_username", type=str, default=None,
                        help="Hub username (auto-detected if logged in)")
    parser.add_argument("--hub_repo_name", type=str, default=None,
                        help="Hub repo name (default: openmedallion-finsentiment-{timestamp})")
    
    args = parser.parse_args()
    
    # Paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    

# Initialize Weights & Biases if requested
if args.use_wandb:
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_seq_length": args.max_seq_length,
            }
        )
    except ImportError:
        print("Warning: wandb not available, continuing without W&B logging")
        args.use_wandb = False
    print("=" * 60)
    print("OPENMEDALLION-FINSENTIMENT TRAINING")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print("=" * 60)
    
    # Load label mapping
    label_map_path = dataset_dir.parent / 'label_map.json'
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    label_map_inv = {v: k for k, v in label_map.items()}
    print(f"\nLabel mapping: {label_map}")
    
    # Load dataset from JSONL
    print(f"\nLoading dataset from {dataset_dir}/train.jsonl...")
    jsonl_path = dataset_dir / 'train.jsonl'
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Expected train.jsonl at {jsonl_path}")
    
    # Load JSONL into list of dicts
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples from JSONL")
    
    # Create HuggingFace Dataset
    from datasets import Dataset
    full_dataset = Dataset.from_list(examples)
    
    # Split into train/validation (90/10)
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    dataset = {
        'train': split['train'],
        'validation': split['test']
    }
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    
    # Configure 4-bit quantization
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Load base model with quantization
    print(f"\nLoading model with 4-bit quantization: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    print("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print(f"\nConfiguring LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM'
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess datasets
    print("\nPreprocessing datasets...")
    
    def preprocess_wrapper(examples):
        return preprocess_function(examples, tokenizer, label_map_inv, args.max_length)
    
    tokenized_train = dataset['train'].map(
        preprocess_wrapper,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing train set"
    )
    
    tokenized_val = dataset['validation'].map(
        preprocess_wrapper,
        batched=True,
        remove_columns=dataset['validation'].column_names,
        desc="Tokenizing validation set"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=False,
        bf16=True,
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=False
    )
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        resume_from_checkpoint=args.resume_from_checkpoint,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save final model
    print(f"\nSaving final model to {output_dir / 'final'}...")
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print("\n" + "="*80)
        print("PUSHING TO HUGGINGFACE HUB")
        print("="*80)
        
        from openmedallion.hub import push_to_hub, setup_token
        
        # Setup authentication
        hub_token = setup_token()
        if not hub_token:
            print("ERROR: HuggingFace token not found. Set HF_TOKEN environment variable.")
            print("Skipping Hub push.")
        else:
            username = args.hub_username if args.hub_username else hub_token.split('_')[0]
            repo_name = args.hub_repo_name
            
            print(f"Uploading model to {username}/{repo_name}...")
            
            try:
                repo_url = push_to_hub(
                    local_path=args.output_dir,
                    repo_name=repo_name,
                    username=username,
                    repo_type='model',
                    commit_message=f'FinSentiment Qwen model - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
                )
                print(f"✓ Successfully pushed to: {repo_url}")
            except Exception as e:
                print(f"ERROR pushing to Hub: {e}")
                print("Model saved locally but Hub push failed.")
    
    # Save training metrics
    metrics_path = output_dir / 'training_metrics.json'
    print(f"Saving training metrics to {metrics_path}...")
    
    metrics = {
        'train_loss': trainer.state.log_history[-1].get('loss', None),
        'eval_loss': trainer.state.log_history[-1].get('eval_loss', None),
        'total_steps': trainer.state.global_step,
        'best_model_checkpoint': trainer.state.best_model_checkpoint
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining pipeline complete!")


if __name__ == '__main__':
    main()
