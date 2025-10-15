"""
PyTorch Dataset and DataLoader Module
Implements multimodal dataset for pneumonia diagnosis
"""

import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config import RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
from preprocess_images import resolve_image_path, get_transforms, load_and_transform_image
from utils_io import load_numpy, load_torch

class PneumoniaMultimodalDataset(data.Dataset):
    """
    PyTorch Dataset for multimodal pneumonia diagnosis
    Combines chest X-ray images with EMR data
    """
    
    def __init__(self, image_ids: np.ndarray, emr_features: np.ndarray, labels: np.ndarray,
                 rsna_root: str, split: str = 'train', cache: bool = False):
        """
        Initialize the multimodal dataset
        
        Args:
            image_ids: Array of image identifiers
            emr_features: Array of EMR features [N, F]
            labels: Array of labels [N]
            rsna_root: Root directory of RSNA dataset
            split: Data split ('train', 'val', 'test')
            cache: Whether to use image caching
        """
        self.image_ids = image_ids
        self.emr_features = emr_features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.rsna_root = rsna_root
        self.split = split
        self.cache = cache
        
        # Get transforms for this split
        self.transform = get_transforms(split)
        
        # Cache directory
        self.cache_dir = Path("data/processed/cache") if cache else None
        
        # Validate data consistency
        assert len(image_ids) == len(emr_features) == len(labels), \
            f"Data length mismatch: {len(image_ids)} images, {len(emr_features)} EMR, {len(labels)} labels"
        
        print(f"✅ Initialized {split} dataset: {len(self)} samples")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: Image tensor [3, H, W]
                - emr: EMR features tensor [F]
                - label: Label tensor (scalar)
                - image_id: Image identifier (string)
        """
        # Get image
        image_id = self.image_ids[idx]
        try:
            image_path = resolve_image_path(image_id, self.rsna_root)
            image_tensor = load_and_transform_image(
                image_path, self.transform, self.cache, self.cache_dir
            )
        except Exception as e:
            print(f"Warning: Failed to load image {image_id}: {e}")
            # Return zero tensor as fallback
            image_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        
        # Get EMR features
        emr_tensor = torch.from_numpy(self.emr_features[idx]).float()
        
        # Get label
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image_tensor,
            'emr': emr_tensor,
            'label': label_tensor,
            'image_id': image_id
        }
    
    def get_class_counts(self) -> Dict[int, int]:
        """Get class distribution in the dataset"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(labels)
    num_classes = len(unique)
    
    weights = np.zeros(num_classes)
    for i, count in enumerate(counts):
        weights[i] = total_samples / (num_classes * count)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return torch.from_numpy(weights).float()

def make_dataloaders(split_csv_dir: str, emr_arrays_dir: str, rsna_root: str, 
                    batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS,
                    cache: bool = False) -> Dict[str, data.DataLoader]:
    """
    Create DataLoaders for all splits
    
    Args:
        split_csv_dir: Directory containing split CSV files
        emr_arrays_dir: Directory containing EMR arrays
        rsna_root: Root directory of RSNA dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        cache: Whether to use image caching
        
    Returns:
        Dictionary of DataLoaders for each split
    """
    dataloaders = {}
    splits = ['train', 'val', 'test']
    
    print("🔄 Creating DataLoaders...")
    
    for split in splits:
        print(f"   Loading {split} data...")
        
        # Load arrays
        emr_path = Path(emr_arrays_dir) / f"emr_{split}.npy"
        y_path = Path(emr_arrays_dir) / f"y_{split}.npy"
        ids_path = Path(emr_arrays_dir) / f"image_ids_{split}.npy"
        
        emr_features = load_numpy(emr_path)
        labels = load_numpy(y_path)
        image_ids = np.load(ids_path, allow_pickle=True)
        
        # Create dataset
        dataset = PneumoniaMultimodalDataset(
            image_ids=image_ids,
            emr_features=emr_features,
            labels=labels,
            rsna_root=rsna_root,
            split=split,
            cache=cache
        )
        
        # Create DataLoader
        shuffle = (split == 'train')  # Only shuffle training data
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')  # Drop last batch only for training
        )
        
        dataloaders[split] = dataloader
        
        # Print dataset info
        class_counts = dataset.get_class_counts()
        print(f"   ✅ {split}: {len(dataset)} samples, classes: {class_counts}")
    
    return dataloaders

def get_sample_batch(dataloader: data.DataLoader, num_samples: int = 4) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch from a DataLoader
    
    Args:
        dataloader: DataLoader to sample from
        num_samples: Number of samples to get
        
    Returns:
        Dictionary containing sample batch
    """
    batch = next(iter(dataloader))
    
    # Take only the requested number of samples
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key][:num_samples]
        elif isinstance(batch[key], list):
            batch[key] = batch[key][:num_samples]
    
    return batch

def print_batch_info(batch: Dict[str, torch.Tensor], split: str = "sample") -> None:
    """
    Print information about a batch
    
    Args:
        batch: Batch dictionary
        split: Split name for display
    """
    print(f"\n📊 {split.upper()} Batch Information:")
    print(f"   Image shape: {batch['image'].shape}")
    print(f"   EMR shape: {batch['emr'].shape}")
    print(f"   Labels shape: {batch['label'].shape}")
    print(f"   Image IDs: {batch['image_id'][:3]}...")  # Show first 3 IDs
    
    # Class distribution
    unique, counts = torch.unique(batch['label'], return_counts=True)
    print(f"   Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

def validate_dataloader(dataloader: data.DataLoader, max_batches: int = 3) -> bool:
    """
    Validate that a DataLoader works correctly
    
    Args:
        dataloader: DataLoader to validate
        max_batches: Maximum number of batches to test
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        print(f"🔍 Validating DataLoader...")
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Check batch structure
            required_keys = ['image', 'emr', 'label', 'image_id']
            for key in required_keys:
                if key not in batch:
                    print(f"❌ Missing key: {key}")
                    return False
            
            # Check tensor shapes
            if batch['image'].dim() != 4:  # [B, C, H, W]
                print(f"❌ Invalid image shape: {batch['image'].shape}")
                return False
            
            if batch['emr'].dim() != 2:  # [B, F]
                print(f"❌ Invalid EMR shape: {batch['emr'].shape}")
                return False
            
            if batch['label'].dim() != 1:  # [B]
                print(f"❌ Invalid label shape: {batch['label'].shape}")
                return False
            
            print(f"   ✅ Batch {i+1}: {batch['image'].shape[0]} samples")
        
        print("✅ DataLoader validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader validation failed: {e}")
        return False

def main():
    """Main function for testing"""
    # This would be called from the sanity check script
    print("Data module loaded successfully!")

if __name__ == "__main__":
    main()
