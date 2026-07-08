"""
Model downloader from HuggingFace Hub.

Provides utilities to download models, datasets, and files from HuggingFace Hub
with caching and verification.
"""

import os
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError


def from_pretrained(
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Union[str, Path]:
    """
    Download a model file or entire repository from HuggingFace Hub.
    
    Args:
        repo_id: Repository ID (e.g., 'username/model-name')
        filename: Specific file to download (None = entire repo)
        cache_dir: Local cache directory (default: ~/.cache/huggingface/hub)
        force_download: Force re-download even if cached
        revision: Git revision (branch, tag, or commit hash)
        token: HuggingFace API token (uses HF_TOKEN env var if None)
    
    Returns:
        Path to downloaded file or repository directory
    
    Raises:
        ValueError: If repo_id is invalid or file not found
        HfHubHTTPError: If download fails
    
    Examples:
        >>> # Download specific model file
        >>> model_path = from_pretrained(
        ...     "oyi77/openmedallion-fints-lgbm",
        ...     filename="commodities_regression.pkl"
        ... )
        
        >>> # Download entire repository
        >>> repo_path = from_pretrained("oyi77/openmedallion-finsentiment")
    """
    if not repo_id:
        raise ValueError("repo_id cannot be empty")
    
    # Use HF_TOKEN from environment if token not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    try:
        if filename:
            # Download specific file
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision,
                token=token,
            )
            return Path(path)
        else:
            # Download entire repository
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision,
                token=token,
            )
            return Path(path)
    
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed. Set HF_TOKEN environment variable or pass token argument."
            ) from e
        elif e.response.status_code == 404:
            raise ValueError(
                f"Repository or file not found: {repo_id}" + 
                (f"/{filename}" if filename else "")
            ) from e
        else:
            raise


def list_files(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> List[str]:
    """
    List all files in a HuggingFace repository.
    
    Args:
        repo_id: Repository ID (e.g., 'username/model-name')
        revision: Git revision (branch, tag, or commit hash)
        token: HuggingFace API token (uses HF_TOKEN env var if None)
    
    Returns:
        List of file paths in the repository
    
    Raises:
        ValueError: If repo_id is invalid
        HfHubHTTPError: If listing fails
    
    Example:
        >>> files = list_files("oyi77/openmedallion-fints-lgbm")
        >>> print(files)
        ['commodities_regression.pkl', 'crypto_regression.pkl', ...]
    """
    if not repo_id:
        raise ValueError("repo_id cannot be empty")
    
    # Use HF_TOKEN from environment if token not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    try:
        files = list_repo_files(
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
        return files
    
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed. Set HF_TOKEN environment variable or pass token argument."
            ) from e
        elif e.response.status_code == 404:
            raise ValueError(f"Repository not found: {repo_id}") from e
        else:
            raise


def download_dataset(
    repo_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Download a dataset from HuggingFace Hub.
    
    This is a convenience wrapper around from_pretrained for datasets.
    
    Args:
        repo_id: Dataset repository ID (e.g., 'username/dataset-name')
        cache_dir: Local cache directory
        force_download: Force re-download even if cached
        revision: Git revision (branch, tag, or commit hash)
        token: HuggingFace API token (uses HF_TOKEN env var if None)
    
    Returns:
        Path to downloaded dataset directory
    
    Example:
        >>> dataset_path = download_dataset("oyi77/financial-sentiment")
    """
    return from_pretrained(
        repo_id=repo_id,
        filename=None,
        cache_dir=cache_dir,
        force_download=force_download,
        revision=revision,
        token=token,
    )


def get_cache_dir() -> Path:
    """
    Get the default HuggingFace cache directory.
    
    Returns:
        Path to cache directory (~/.cache/huggingface/hub)
    """
    return Path.home() / ".cache" / "huggingface" / "hub"


def clear_cache(repo_id: Optional[str] = None):
    """
    Clear the HuggingFace cache.
    
    Args:
        repo_id: Specific repository to clear (None = clear all)
    
    Warning:
        This will delete downloaded models. Use with caution.
    """
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        return
    
    if repo_id:
        # Clear specific repo cache
        # Cache structure: models--{org}--{name}
        repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
        if repo_cache.exists():
            import shutil
            shutil.rmtree(repo_cache)
    else:
        # Clear entire cache
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
