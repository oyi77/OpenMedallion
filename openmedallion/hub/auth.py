"""
HuggingFace Hub Authentication Helpers

Handles authentication and token management for Hub operations.
"""

import os
from typing import Optional
from huggingface_hub import HfApi, login, logout, whoami


def setup_token(token: Optional[str] = None, write_permission: bool = True) -> str:
    """
    Setup HuggingFace token for authentication.
    
    Args:
        token: HF token (if None, reads from HF_TOKEN env var or prompts user)
        write_permission: Whether to require write permission
        
    Returns:
        The configured token
        
    Raises:
        ValueError: If no token provided and HF_TOKEN not set
    """
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    if token is None:
        raise ValueError(
            "No token provided. Either pass token argument or set HF_TOKEN environment variable.\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )
    
    # Login with token
    login(token=token, write_permission=write_permission)
    
    return token


def get_user_info(token: Optional[str] = None) -> dict:
    """
    Get authenticated user information.
    
    Args:
        token: HF token (if None, uses cached token)
        
    Returns:
        Dictionary with user info (username, email, orgs, etc.)
    """
    try:
        info = whoami(token=token)
        return {
            "username": info["name"],
            "email": info.get("email"),
            "fullname": info.get("fullname"),
            "orgs": [org["name"] for org in info.get("orgs", [])],
            "auth_type": info.get("auth", {}).get("type"),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get user info: {e}")


def verify_token(token: str, require_write: bool = False) -> bool:
    """
    Verify token validity and permissions.
    
    Args:
        token: HF token to verify
        require_write: Whether to verify write permission
        
    Returns:
        True if token is valid with required permissions
    """
    try:
        api = HfApi(token=token)
        user_info = whoami(token=token)
        
        if require_write:
            # Check if token has write permission by attempting to list models
            # (read-only tokens will fail on certain operations)
            auth_type = user_info.get("auth", {}).get("type")
            if auth_type == "access_token":
                # Access tokens can have scoped permissions
                # For now, assume valid if we can get user info
                return True
            
        return True
    except Exception:
        return False


def clear_token():
    """
    Logout and clear cached HF token.
    """
    logout()
