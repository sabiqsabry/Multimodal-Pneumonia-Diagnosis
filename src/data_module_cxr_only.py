"""
CXR-Only Dataset Module for Pneumonia Classification
Dataset that loads only chest X-ray images (no EMR features)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Dict, Any

from preprocess_images import get_transforms, resolve_image_path
from utils_io import load_numpy
from config import RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

class CXROnlyDataset(Dataset):
    """
    CXR-only dataset for pneumonia classification
    
    Loads only chest X-ray images and labels, ignoring EMR features
    """
    
    def __init__(self, image_ids, labels, rsna_root, transform=None, cache=False):
        """
        Initialize CXR-only dataset
        
        Args:
            image_ids: Array of image IDs
            labels: Array of labels (0=normal, 1=pneumonia)
            rsna_root: Path to RSNA dataset root
            transform: Image transforms to apply
            cache: Whether to cache preprocessed images
        """
        self.image_ids = image_ids
        self.labels = labels
        self.rsna_root = Path(rsna_root)
        self.transform = transform
        self.cache = cache
        
        # Cache for preprocessed images
        self.cache_dir = None
        if cache:
            self.cache_dir = Path("data/processed/cache/cxr_only")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Sample index
        
        Returns:
            dict: {
                'image': torch.Tensor of shape (3, H, W),
                'label': torch.Tensor of shape (1,),
                'image_id': str
            }
        """
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        
        # Load image
        image_path = resolve_image_path(image_id, str(self.rsna_root))
        
        if self.cache:
            # Check if cached version exists
            cache_path = self.cache_dir / f"{image_id}.pt"
            if cache_path.exists():
                image = torch.load(cache_path)
            else:
                image = self._load_and_transform_image(image_path)
                torch.save(image, cache_path)
        else:
            image = self._load_and_transform_image(image_path)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32).unsqueeze(0),
            'image_id': image_id
        }
    
    def _load_and_transform_image(self, image_path):
        """Load and transform image"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default transforms
                transform = transforms.Compose([
                    transforms.Resize(IMG_SIZE),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                image = transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)

def make_cxr_only_dataloaders(split_csv_dir: str, rsna_root: str, batch_size: int = BATCH_SIZE, 
                             num_workers: int = NUM_WORKERS, cache: bool = False) -> Dict[str, DataLoader]:
    """
    Create CXR-only data loaders for train/val/test splits
    
    Args:
        split_csv_dir: Directory containing split CSV files
        rsna_root: Path to RSNA dataset root
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        cache: Whether to cache preprocessed images
    
    Returns:
        dict: Dictionary containing train/val/test DataLoaders
    """
    print("🔄 Creating CXR-Only DataLoaders...")
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        print(f"   Loading {split} data...")
        
        # Load image IDs and labels
        image_ids_path = Path(split_csv_dir) / f"image_ids_{split}.npy"
        labels_path = Path(split_csv_dir) / f"y_{split}.npy"
        
        if not image_ids_path.exists() or not labels_path.exists():
            print(f"   ❌ Missing files for {split} split")
            continue
        
        # Load data
        image_ids = np.load(image_ids_path, allow_pickle=True)
        labels = load_numpy(labels_path)
        
        # Create dataset
        transform = get_transforms(split, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)
        dataset = CXROnlyDataset(
            image_ids=image_ids,
            labels=labels,
            rsna_root=rsna_root,
            transform=transform,
            cache=cache
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
        
        # Print dataset info
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"   ✅ {split}: {len(dataset)} samples, classes: {dict(zip(unique_labels, counts))}")
    
    return dataloaders

def compute_class_weights_cxr_only(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for CXR-only model
    
    Args:
        labels: Array of labels (0=normal, 1=pneumonia)
    
    Returns:
        torch.Tensor: Class weights
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(labels)
    class_weights = total_samples / (len(unique_classes) * counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(unique_classes)
    
    return torch.tensor(class_weights, dtype=torch.float32)

if __name__ == "__main__":
    # Test dataset creation
    print("Testing CXR-Only Dataset...")
    
    # Load sample data
    proc_dir = Path("data/processed")
    image_ids = np.load(proc_dir / "image_ids_train.npy", allow_pickle=True)[:10]
    labels = load_numpy(proc_dir / "y_train.npy")[:10]
    
    # Create dataset
    dataset = CXROnlyDataset(
        image_ids=image_ids,
        labels=labels,
        rsna_root=str(RSNA_ROOT),
        transform=get_transforms('train', IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Image ID: {sample['image_id']}")
    
    # Test data loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    
    print("✅ CXR-Only Dataset test completed!")



