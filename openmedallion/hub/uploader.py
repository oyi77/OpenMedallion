"""
HuggingFace Hub Uploader

Handles uploading trained models to HuggingFace Hub with proper versioning
and model cards.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def push_to_hub(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_card_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Push model to HuggingFace Hub.
    
    Args:
        model_path: Local path to model file or directory
        repo_id: Repository ID on Hub (format: username/repo-name)
        token: HF token (if None, uses cached token)
        commit_message: Commit message for upload
        private: Whether repo should be private
        create_pr: Create pull request instead of direct push
        model_card_data: Metadata for model card generation
        
    Returns:
        URL of uploaded model
        
    Examples:
        >>> push_to_hub("models/lgbm_baseline.pkl", "oyi77/fints-lgbm")
        >>> push_to_hub("checkpoints/qwen-sentiment/", "oyi77/finsentiment-qwen", 
        ...             model_card_data={"accuracy": 0.87, "dataset": "financial_phrasebank"})
    """
    api = HfApi(token=token)
    model_path = Path(model_path)
    
    if commit_message is None:
        commit_message = f"Upload {model_path.name}"
    
    # Create repo if doesn't exist
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create/access repo {repo_id}: {e}")
    
    # Upload file or folder
    try:
        if model_path.is_file():
            url = upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
        else:
            url = upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )
        
        # Generate and upload model card if metadata provided
        if model_card_data:
            card_content = _generate_model_card(model_card_data)
            card_path = model_path.parent / "MODEL_CARD.md"
            card_path.write_text(card_content)
            upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
                commit_message="Add model card",
            )
        
        return url
    except Exception as e:
        raise RuntimeError(f"Failed to upload to {repo_id}: {e}")


def _generate_model_card(data: Dict[str, Any]) -> str:
    """Generate model card from metadata."""
    card = f"""---
license: mit
tags:
- financial-forecasting
- time-series
- sentiment-analysis
---

# {data.get('model_name', 'OpenMedallion Model')}

## Model Description
{data.get('description', 'Financial AI model trained with OpenMedallion framework')}

## Training Details
- **Dataset**: {data.get('dataset', 'N/A')}
- **Training Date**: {data.get('training_date', 'N/A')}
- **Framework**: {data.get('framework', 'OpenMedallion')}

## Performance
"""
    
    if 'metrics' in data:
        for metric, value in data['metrics'].items():
            card += f"- **{metric}**: {value}\n"
    
    card += f"""
## Usage
```python
from openmedallion.hub import from_pretrained

model = from_pretrained("{data.get('repo_id', 'username/model-name')}")
```

## Citation
```
@software{{openmedallion,
  title = {{OpenMedallion: Open-Source Financial AI Framework}},
  author = {{OpenMedallion Contributors}},
  year = {{2024}},
  url = {{https://github.com/oyi77/OpenMedallion}}
}}
```
"""
    return card
