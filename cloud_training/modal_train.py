"""
Modal cloud training script for OpenMedallion models.
Uses Modal for serverless GPU training with automatic scaling.
"""

import modal
from pathlib import Path


# Define Modal app
app = modal.App("openmedallion-training")

# Define Docker image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "datasets==2.16.0",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        "peft==0.7.1",
        "trl==0.7.9",
        "scipy",
        "scikit-learn",
        "pandas",
        "numpy",
        "lightgbm",
        "xgboost",
        "prophet",
        "yfinance",
        "requests",
        "huggingface_hub",
        "wandb",
    )
)

# Create persistent volume for models/data
volume = modal.Volume.from_name("openmedallion-volume", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,  # 4 hours
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-api-key"),
    ],
    volumes={"/data": volume},
)
def train_finsentiment(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "/data/finsentiment",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 2e-4,
    hub_repo_id: str = None,
):
    """
    Train FinSentiment model (Qwen fine-tuning) on Modal.
    
    Args:
        model_name: Base model name from HuggingFace
        output_dir: Directory to save trained model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        learning_rate: Learning rate
        hub_repo_id: HuggingFace Hub repo ID for upload (optional)
    
    Returns:
        dict: Training results and model path
    """
    import os
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, "/root")
    
    # Import training function
    from openmedallion.finsentiment.fine_tune_qwen import train_model
    
    print(f"🚀 Starting FinSentiment training on Modal")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    
    # Prepare data
    print("\n📥 Preparing training data...")
    from openmedallion.finsentiment.prepare_sentiment_data import prepare_data
    prepare_data()
    
    # Train model
    print("\n🔥 Training model...")
    trainer = train_model(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
    )
    
    # Save results
    results = {
        "model_path": output_dir,
        "training_loss": trainer.state.log_history[-1].get("loss", None),
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    
    # Upload to HuggingFace Hub if repo_id provided
    if hub_repo_id:
        print(f"\n📤 Uploading to HuggingFace Hub: {hub_repo_id}")
        from openmedallion.hub import push_to_hub
        
        push_to_hub(
            model_path=output_dir,
            repo_id=hub_repo_id,
            model_type="finsentiment",
        )
        results["hub_repo"] = hub_repo_id
    
    # Commit volume changes
    volume.commit()
    
    print("\n✅ FinSentiment training complete!")
    return results


@app.function(
    image=image,
    gpu="T4",  # FinTS doesn't need A100
    timeout=3600 * 2,  # 2 hours
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/data": volume},
)
def train_fints(
    asset_class: str = "equities",
    output_dir: str = "/data/fints",
    hub_repo_id: str = None,
):
    """
    Train FinTS model (LGBM) on Modal.
    
    Args:
        asset_class: Asset class to train (equities, crypto, commodities, forex)
        output_dir: Directory to save trained model
        hub_repo_id: HuggingFace Hub repo ID for upload (optional)
    
    Returns:
        dict: Training results and model path
    """
    import os
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, "/root")
    
    print(f"🚀 Starting FinTS training on Modal")
    print(f"Asset class: {asset_class}")
    print(f"Output: {output_dir}")
    
    # Download data based on asset class
    print("\n📥 Downloading data...")
    if asset_class in ["equities", "forex"]:
        from openmedallion.fints.data_collectors.yfinance_historical import collect_data
        collect_data(asset_type=asset_class)
    elif asset_class == "crypto":
        from openmedallion.fints.data_collectors.coingecko_top200_historical import collect_data
        collect_data()
    elif asset_class == "commodities":
        from openmedallion.fints.data_collectors.yfinance_crypto_historical import collect_data
        collect_data(asset_type="commodities")
    
    # Train model
    print("\n🔥 Training LGBM model...")
    from openmedallion.fints.scripts.train_lgbm import train_lgbm
    
    model_path = train_lgbm(asset_class=asset_class, output_dir=output_dir)
    
    results = {
        "model_path": model_path,
        "asset_class": asset_class,
    }
    
    # Upload to HuggingFace Hub if repo_id provided
    if hub_repo_id:
        print(f"\n📤 Uploading to HuggingFace Hub: {hub_repo_id}")
        from openmedallion.hub import push_to_hub
        
        push_to_hub(
            model_path=model_path,
            repo_id=hub_repo_id,
            model_type="fints",
        )
        results["hub_repo"] = hub_repo_id
    
    # Commit volume changes
    volume.commit()
    
    print(f"\n✅ FinTS {asset_class} training complete!")
    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=3600 * 8,  # 8 hours for all asset classes
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/data": volume},
)
def train_all_fints(hub_username: str = None):
    """
    Train FinTS models for all asset classes sequentially.
    
    Args:
        hub_username: HuggingFace Hub username for uploads (optional)
    
    Returns:
        dict: Results for all trained models
    """
    asset_classes = ["equities", "crypto", "commodities", "forex"]
    results = {}
    
    for asset_class in asset_classes:
        print(f"\n{'='*60}")
        print(f"Training {asset_class.upper()}")
        print('='*60)
        
        hub_repo_id = None
        if hub_username:
            hub_repo_id = f"{hub_username}/openmedallion-fints-{asset_class}"
        
        result = train_fints.remote(
            asset_class=asset_class,
            hub_repo_id=hub_repo_id,
        )
        results[asset_class] = result
    
    print("\n✅ All FinTS models trained!")
    return results


@app.local_entrypoint()
def main(
    task: str = "finsentiment",
    hub_username: str = None,
):
    """
    Local entrypoint for Modal training.
    
    Usage:
        # Train FinSentiment
        modal run cloud_training/modal_train.py --task finsentiment --hub-username YOUR_USERNAME
        
        # Train FinTS (single asset class)
        modal run cloud_training/modal_train.py --task fints-equities --hub-username YOUR_USERNAME
        
        # Train all FinTS models
        modal run cloud_training/modal_train.py --task fints-all --hub-username YOUR_USERNAME
    """
    
    if task == "finsentiment":
        hub_repo_id = None
        if hub_username:
            hub_repo_id = f"{hub_username}/openmedallion-finsentiment"
        
        result = train_finsentiment.remote(hub_repo_id=hub_repo_id)
        print("\n📊 Training Results:")
        print(result)
        
    elif task.startswith("fints-"):
        if task == "fints-all":
            results = train_all_fints.remote(hub_username=hub_username)
            print("\n📊 All Training Results:")
            for asset_class, result in results.items():
                print(f"\n{asset_class}: {result}")
        else:
            asset_class = task.replace("fints-", "")
            hub_repo_id = None
            if hub_username:
                hub_repo_id = f"{hub_username}/openmedallion-fints-{asset_class}"
            
            result = train_fints.remote(
                asset_class=asset_class,
                hub_repo_id=hub_repo_id,
            )
            print("\n📊 Training Results:")
            print(result)
    
    else:
        print(f"❌ Unknown task: {task}")
        print("\nAvailable tasks:")
        print("  - finsentiment")
        print("  - fints-equities")
        print("  - fints-crypto")
        print("  - fints-commodities")
        print("  - fints-forex")
        print("  - fints-all")


# Programmatic API for submitting jobs
class ModalTrainingClient:
    """
    Client for programmatically submitting Modal training jobs.
    """
    
    @staticmethod
    def submit_finsentiment_job(
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hub_repo_id: str = None,
        **kwargs,
    ):
        """
        Submit FinSentiment training job to Modal.
        
        Returns:
            modal.FunctionCall: Job handle
        """
        return train_finsentiment.spawn(
            model_name=model_name,
            hub_repo_id=hub_repo_id,
            **kwargs,
        )
    
    @staticmethod
    def submit_fints_job(
        asset_class: str,
        hub_repo_id: str = None,
    ):
        """
        Submit FinTS training job to Modal.
        
        Returns:
            modal.FunctionCall: Job handle
        """
        return train_fints.spawn(
            asset_class=asset_class,
            hub_repo_id=hub_repo_id,
        )
    
    @staticmethod
    def submit_all_fints_jobs(hub_username: str = None):
        """
        Submit jobs for all FinTS asset classes.
        
        Returns:
            modal.FunctionCall: Job handle
        """
        return train_all_fints.spawn(hub_username=hub_username)


if __name__ == "__main__":
    print("""
🚀 Modal Training Script for OpenMedallion

Setup:
1. Install Modal: pip install modal
2. Set up Modal token: modal token new
3. Create secrets in Modal dashboard:
   - huggingface-token (HF_TOKEN)
   - wandb-api-key (WANDB_API_KEY)

Usage:
    # Train FinSentiment
    modal run cloud_training/modal_train.py --task finsentiment --hub-username YOUR_USERNAME
    
    # Train single FinTS model
    modal run cloud_training/modal_train.py --task fints-equities --hub-username YOUR_USERNAME
    
    # Train all FinTS models
    modal run cloud_training/modal_train.py --task fints-all --hub-username YOUR_USERNAME

Programmatic API:
    from cloud_training.modal_train import ModalTrainingClient
    
    # Submit job
    job = ModalTrainingClient.submit_finsentiment_job(
        hub_repo_id="username/openmedallion-finsentiment"
    )
    
    # Get result
    result = job.get()

For more info: https://modal.com/docs
""")
