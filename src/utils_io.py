"""
Utility functions for I/O operations
Handles saving/loading of various data formats
"""

import json
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Any, Union

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Save object as JSON file"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"✅ Saved JSON: {path}")

def load_json(path: Union[str, Path]) -> Any:
    """Load object from JSON file"""
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)

def save_numpy(arr: np.ndarray, path: Union[str, Path]) -> None:
    """Save numpy array to .npy file"""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)
    print(f"✅ Saved numpy array: {path}")

def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load numpy array from .npy file"""
    path = Path(path)
    return np.load(path)

def save_torch(obj: Any, path: Union[str, Path]) -> None:
    """Save PyTorch object to .pt file"""
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, path)
    print(f"✅ Saved PyTorch object: {path}")

def load_torch(path: Union[str, Path]) -> Any:
    """Load PyTorch object from .pt file"""
    path = Path(path)
    return torch.load(path)

def save_joblib(obj: Any, path: Union[str, Path]) -> None:
    """Save object using joblib"""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    print(f"✅ Saved joblib object: {path}")

def load_joblib(path: Union[str, Path]) -> Any:
    """Load object using joblib"""
    path = Path(path)
    return joblib.load(path)

def get_file_size_mb(path: Union[str, Path]) -> float:
    """Get file size in MB"""
    path = Path(path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0
