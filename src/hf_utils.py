"""
Hugging Face Download Utilities

This module handles automatic downloading of models and datasets
from Hugging Face repository when they are not available locally.
This is essential for deployed Streamlit apps.
"""

import os
from pathlib import Path
from typing import List, Optional
from huggingface_hub import snapshot_download, hf_hub_download


def check_required_files() -> dict:
    """
    Check which required files/directories are missing.
    
    Returns:
        Dictionary with status of each required component
    """
    required = {
        'models': Path('models'),
        'features': Path('features'),
        'DS2GROCERIES': Path('DS2GROCERIES'),
        'DS3RECIPES': Path('DS3RECIPES')
    }
    
    status = {}
    for name, path in required.items():
        status[name] = path.exists() and any(path.iterdir()) if path.exists() else False
    
    return status


def download_from_huggingface(
    repo_id: str = "khedim/ML-MINI-PROJECT",
    token: Optional[str] = None,
    force_download: bool = False
) -> bool:
    """
    Download required files from Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID
        token: HF authentication token (optional for public repos)
        force_download: Force re-download even if files exist
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        # Check what's missing
        status = check_required_files()
        missing = [k for k, v in status.items() if not v]
        
        if not missing and not force_download:
            return True
        
        # Get token if not provided
        if token is None:
            token = get_hf_token()
        
        # Download the entire repository
        # This will only download files that don't exist locally
        print(f"Downloading required files from Hugging Face: {', '.join(missing)}...")
        
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=".",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("✓ Successfully downloaded required files!")
        return True
            
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print("💡 Tip: Make sure the repository is public or provide a valid HF_TOKEN")
        return False


def ensure_models_available(repo_id: str = "khedim/ML-MINI-PROJECT") -> bool:
    """
    Ensure all required models and data are available.
    Downloads from HF if missing.
    
    Args:
        repo_id: Hugging Face repository ID
        
    Returns:
        True if all files are available, False otherwise
    """
    status = check_required_files()
    
    # If everything exists, we're good
    if all(status.values()):
        return True
    
    # Otherwise, download from HF
    return download_from_huggingface(repo_id)


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from various sources.
    Priority: Streamlit secrets > token.txt file > None
    
    Returns:
        HF token or None
    """
    # Try Streamlit secrets first (for deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'HF_TOKEN' in st.secrets:
            return st.secrets['HF_TOKEN']
    except:
        pass
    
    # Try token.txt file (for local development)
    token_file = Path('src/token.txt')
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except:
            pass
    
    return None


if __name__ == "__main__":
    # Test the download functionality
    print("Checking required files...")
    status = check_required_files()
    
    for name, exists in status.items():
        print(f"  {name}: {'✓' if exists else '✗'}")
    
    if not all(status.values()):
        print("\nDownloading missing files from Hugging Face...")
        token = get_hf_token()
        success = download_from_huggingface(token=token)
        
        if success:
            print("✓ Download complete!")
        else:
            print("✗ Download failed!")
