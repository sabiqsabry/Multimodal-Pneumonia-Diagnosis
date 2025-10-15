"""
Image Preprocessing Module
Handles image loading, transforms, and caching for the multimodal dataset
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from config import RSNA_ROOT, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CACHE_DIR
from utils_io import save_torch, load_torch

def resolve_image_path(image_id: str, rsna_root: str) -> str:
    """
    Resolve image path by searching in train/ then test/ directories
    
    Args:
        image_id: Image identifier (e.g., "0004cfab-14fd-4e49-80ba-63a80b6bddd6_001")
        rsna_root: Root directory of RSNA dataset
        
    Returns:
        Full path to the image file
        
    Raises:
        FileNotFoundError: If image is not found in either train or test directories
    """
    rsna_path = Path(rsna_root)
    
    # Extract the base patient ID (remove the _001 suffix)
    base_id = image_id.split('_')[0]
    
    # Search in Training/Images first
    train_path = rsna_path / "Training" / "Images" / f"{base_id}.png"
    if train_path.exists():
        return str(train_path)
    
    # Search in Test directory
    test_path = rsna_path / "Test" / f"{base_id}.png"
    if test_path.exists():
        return str(test_path)
    
    # If not found, raise error
    raise FileNotFoundError(f"Image not found: {image_id} (searched in {train_path} and {test_path})")

def ensure_rgb(image):
    """Ensure image is in RGB mode"""
    return image.convert('RGB') if image.mode != 'RGB' else image

def get_transforms(split: str, img_size: int = 224, mean: Tuple[float, float, float] = IMAGENET_MEAN, 
                  std: Tuple[float, float, float] = IMAGENET_STD) -> transforms.Compose:
    """
    Get image transforms for different splits
    
    Args:
        split: Data split ('train', 'val', 'test')
        img_size: Target image size
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Composed transforms
    """
    if split == 'train':
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Lambda(ensure_rgb),  # Ensure RGB
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Lambda(ensure_rgb),  # Ensure RGB
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def load_and_transform_image(image_path: str, transform: transforms.Compose, 
                           cache: bool = False, cache_dir: Optional[str] = None) -> torch.Tensor:
    """
    Load and transform an image with optional caching
    
    Args:
        image_path: Path to the image file
        transform: Transform to apply
        cache: Whether to use caching
        cache_dir: Directory for cached tensors
        
    Returns:
        Transformed image tensor [3, H, W]
    """
    if cache and cache_dir:
        # Create cache path
        cache_path = Path(cache_dir) / f"{Path(image_path).stem}.pt"
        
        # Try to load from cache
        if cache_path.exists():
            try:
                return load_torch(cache_path)
            except Exception as e:
                print(f"Warning: Failed to load cached image {cache_path}: {e}")
    
    # Load and transform image
    try:
        image = Image.open(image_path)
        tensor = transform(image)
        
        # Save to cache if enabled
        if cache and cache_dir:
            cache_path = Path(cache_dir) / f"{Path(image_path).stem}.pt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_torch(tensor, cache_path)
        
        return tensor
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a zero tensor as fallback
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)

def denormalize_image(tensor: torch.Tensor, mean: Tuple[float, float, float] = IMAGENET_MEAN, 
                     std: Tuple[float, float, float] = IMAGENET_STD) -> torch.Tensor:
    """
    Denormalize a normalized image tensor for visualization
    
    Args:
        tensor: Normalized image tensor [3, H, W]
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Denormalized tensor [3, H, W] with values in [0, 1]
    """
    # Convert to numpy for easier manipulation
    tensor_np = tensor.clone()
    
    # Denormalize
    for i in range(3):
        tensor_np[i] = tensor_np[i] * std[i] + mean[i]
    
    # Clamp to [0, 1] range
    tensor_np = torch.clamp(tensor_np, 0, 1)
    
    return tensor_np

def get_image_stats(image_path: str) -> dict:
    """
    Get basic statistics about an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image statistics
    """
    try:
        image = Image.open(image_path)
        return {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'exists': True
        }
    except Exception as e:
        return {
            'size': None,
            'mode': None,
            'format': None,
            'exists': False,
            'error': str(e)
        }

def validate_image_paths(image_ids: List[str], rsna_root: str) -> dict:
    """
    Validate that all image paths exist and are accessible
    
    Args:
        image_ids: List of image identifiers
        rsna_root: Root directory of RSNA dataset
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total': len(image_ids),
        'found': 0,
        'missing': 0,
        'errors': []
    }
    
    for image_id in image_ids:
        try:
            path = resolve_image_path(image_id, rsna_root)
            stats = get_image_stats(path)
            if stats['exists']:
                results['found'] += 1
            else:
                results['missing'] += 1
                results['errors'].append(f"Image exists but cannot be opened: {image_id}")
        except FileNotFoundError:
            results['missing'] += 1
            results['errors'].append(f"Image not found: {image_id}")
        except Exception as e:
            results['missing'] += 1
            results['errors'].append(f"Error with {image_id}: {str(e)}")
    
    return results

def main():
    """Main function for testing"""
    # Test image path resolution
    test_image_id = "0004cfab-14fd-4e49-80ba-63a80b6bddd6_001"
    try:
        path = resolve_image_path(test_image_id, str(RSNA_ROOT))
        print(f"✅ Found image: {path}")
        
        # Test transforms
        transform = get_transforms('train')
        image = load_and_transform_image(path, transform)
        print(f"✅ Image loaded and transformed: {image.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
