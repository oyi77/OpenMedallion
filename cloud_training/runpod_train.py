"""
RunPod GPU training script for OpenMedallion models.
Uses RunPod's Python SDK for serverless GPU deployment.
"""

import runpod
import os
import json
from pathlib import Path


def train_finsentiment_handler(job):
    """
    RunPod handler for FinSentiment training.
    
    Job input format:
    {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "push_to_hub": true,
        "hub_model_id": "username/openmedallion-finsentiment"
    }
    """
    import sys
    sys.path.insert(0, "/workspace")
    
    from openmedallion.finsentiment.fine_tune_qwen import train_model
    from openmedallion.hub import push_to_hub
    
    # Get job parameters
    job_input = job["input"]
    model_name = job_input.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    num_epochs = job_input.get("num_epochs", 3)
    batch_size = job_input.get("batch_size", 4)
    learning_rate = job_input.get("learning_rate", 2e-4)
    push_to_hub_flag = job_input.get("push_to_hub", True)
    hub_model_id = job_input.get("hub_model_id", None)
    
    output_dir = "/workspace/outputs/finsentiment"
    
    try:
        print(f"🚀 Starting FinSentiment training on RunPod")
        print(f"Model: {model_name}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Train the model
        train_model(
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        # Push to Hub if requested
        hub_url = None
        if push_to_hub_flag and hub_model_id:
            print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
            push_to_hub(
                model_path=output_dir,
                repo_id=hub_model_id,
                model_type="finsentiment",
            )
            hub_url = f"https://huggingface.co/{hub_model_id}"
            print(f"✅ Model uploaded to: {hub_url}")
        
        return {
            "status": "success",
            "output_dir": output_dir,
            "hub_url": hub_url,
        }
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def train_fints_handler(job):
    """
    RunPod handler for FinTS training.
    
    Job input format:
    {
        "asset_class": "equities",
        "model_type": "lgbm",
        "push_to_hub": true,
        "hub_model_id": "username/openmedallion-fints-equities"
    }
    """
    import sys
    sys.path.insert(0, "/workspace")
    
    from openmedallion.hub import push_to_hub
    
    # Get job parameters
    job_input = job["input"]
    asset_class = job_input.get("asset_class", "equities")
    model_type = job_input.get("model_type", "lgbm")
    push_to_hub_flag = job_input.get("push_to_hub", True)
    hub_model_id = job_input.get("hub_model_id", None)
    
    try:
        print(f"🚀 Starting FinTS {model_type.upper()} training on RunPod")
        print(f"Asset class: {asset_class}")
        
        if model_type == "lgbm":
            from openmedallion.fints.scripts.train_lgbm import train_lgbm
            output_path = train_lgbm(asset_class=asset_class)
        else:
            raise NotImplementedError(f"Model type {model_type} not yet supported")
        
        # Push to Hub if requested
        hub_url = None
        if push_to_hub_flag and hub_model_id:
            print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
            push_to_hub(
                model_path=output_path,
                repo_id=hub_model_id,
                model_type="fints",
            )
            hub_url = f"https://huggingface.co/{hub_model_id}"
            print(f"✅ Model uploaded to: {hub_url}")
        
        return {
            "status": "success",
            "model_path": output_path,
            "hub_url": hub_url,
        }
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# Register handlers with RunPod
runpod.serverless.start({
    "train_finsentiment": train_finsentiment_handler,
    "train_fints": train_fints_handler,
})


# Client script for submitting jobs
def submit_job(endpoint_id, handler_name, job_input):
    """
    Submit training job to RunPod endpoint.
    
    Args:
        endpoint_id: Your RunPod endpoint ID
        handler_name: "train_finsentiment" or "train_fints"
        job_input: Dictionary with training parameters
    
    Returns:
        Job ID and status
    
    Usage:
        # Train FinSentiment
        job_id = submit_job(
            endpoint_id="YOUR_ENDPOINT_ID",
            handler_name="train_finsentiment",
            job_input={
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "num_epochs": 3,
                "push_to_hub": True,
                "hub_model_id": "username/openmedallion-finsentiment"
            }
        )
        
        # Train FinTS
        job_id = submit_job(
            endpoint_id="YOUR_ENDPOINT_ID",
            handler_name="train_fints",
            job_input={
                "asset_class": "crypto",
                "model_type": "lgbm",
                "push_to_hub": True,
                "hub_model_id": "username/openmedallion-fints-crypto"
            }
        )
    """
    import runpod
    
    # Get API key from environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable not set")
    
    runpod.api_key = api_key
    
    # Submit job
    endpoint = runpod.Endpoint(endpoint_id)
    run_request = endpoint.run({
        "handler": handler_name,
        "input": job_input,
    })
    
    print(f"✅ Job submitted: {run_request.job_id}")
    print(f"Check status: https://www.runpod.io/console/serverless")
    
    # Wait for completion (optional)
    print("⏳ Waiting for job to complete...")
    result = run_request.output()
    
    print(f"✅ Job complete: {result}")
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("""
Usage:
    # Deploy handler
    python cloud_training/runpod_train.py
    
    # Submit FinSentiment job
    python cloud_training/runpod_train.py submit finsentiment
    
    # Submit FinTS job
    python cloud_training/runpod_train.py submit fints --asset-class crypto
""")
        sys.exit(1)
    
    if sys.argv[1] == "submit":
        task = sys.argv[2]
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
        
        if not endpoint_id:
            print("❌ RUNPOD_ENDPOINT_ID environment variable not set")
            sys.exit(1)
        
        if task == "finsentiment":
            result = submit_job(
                endpoint_id=endpoint_id,
                handler_name="train_finsentiment",
                job_input={
                    "push_to_hub": True,
                    "hub_model_id": "USERNAME/openmedallion-finsentiment",
                }
            )
        elif task == "fints":
            asset_class = "equities"
            if "--asset-class" in sys.argv:
                idx = sys.argv.index("--asset-class")
                asset_class = sys.argv[idx + 1]
            
            result = submit_job(
                endpoint_id=endpoint_id,
                handler_name="train_fints",
                job_input={
                    "asset_class": asset_class,
                    "push_to_hub": True,
                    "hub_model_id": f"USERNAME/openmedallion-fints-{asset_class}",
                }
            )
        else:
            print(f"❌ Unknown task: {task}")
            sys.exit(1)
        
        print(f"✅ Training complete: {result}")
