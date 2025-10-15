"""
Dataset and DataLoader for CXR + Real EMR Fusion Model
Handles loading and alignment of chest X-ray images with real EMR features
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Optional
import random

from preprocess_images import get_transforms, ensure_rgb
from config import RSNA_ROOT, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, ARRAY_FILES, PROC_DIR

class CXRRealEMRFusionDataset(Dataset):
    """
    Dataset for CXR + Real EMR fusion model
    
    Handles alignment between RSNA images and real EMR data
    """
    
    def __init__(self, image_ids, labels, emr_features, rsna_root, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.emr_features = emr_features
        self.rsna_root = Path(rsna_root)
        self.transform = transform
        
        # Ensure all arrays have the same length
        assert len(image_ids) == len(labels) == len(emr_features), \
            f"Mismatch in data lengths: images={len(image_ids)}, labels={len(labels)}, emr={len(emr_features)}"
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        emr_feature = self.emr_features[idx]
        
        # Resolve image path
        image_path = self._resolve_image_path(image_id)
        if not image_path:
            raise FileNotFoundError(f"Image not found for ID: {image_id}")
        
        # Load and transform image
        image = Image.open(image_path)
        image = ensure_rgb(image)  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "emr_features": torch.tensor(emr_feature, dtype=torch.float32),
            "label": torch.tensor([float(label)], dtype=torch.float32),
            "image_id": image_id
        }
    
    def _resolve_image_path(self, image_id: str) -> Optional[str]:
        """Resolve image path by searching in Training/Images and Test/Images folders"""
        # Try different extensions
        extensions = ['.png', '.jpg', '.jpeg']
        
        # Clean image ID (remove _001 suffix if present)
        clean_image_id = image_id.replace('_001', '')
        
        # Search in Training/Images folder first
        for ext in extensions:
            image_path = self.rsna_root / "Training" / "Images" / f"{clean_image_id}{ext}"
            if image_path.exists():
                return str(image_path)
        
        # Search in Test/Images folder
        for ext in extensions:
            image_path = self.rsna_root / "Test" / "Images" / f"{clean_image_id}{ext}"
            if image_path.exists():
                return str(image_path)
        
        return None

def align_datasets(rsna_image_ids, rsna_labels, real_emr_features, real_emr_labels, 
                  real_emr_image_ids=None, max_samples=None):
    """
    Align RSNA dataset with real EMR dataset
    
    Args:
        rsna_image_ids: List of RSNA image IDs
        rsna_labels: List of RSNA labels
        real_emr_features: Array of real EMR features
        real_emr_labels: Array of real EMR labels
        real_emr_image_ids: Optional list of real EMR image IDs for matching
        max_samples: Maximum number of samples to use (for testing)
    
    Returns:
        aligned_image_ids, aligned_labels, aligned_emr_features
    """
    print(f"🔗 Aligning datasets...")
    print(f"   RSNA samples: {len(rsna_image_ids)}")
    print(f"   Real EMR samples: {len(real_emr_features)}")
    
    if real_emr_image_ids is not None:
        # Try to match by image ID
        print("   Attempting to match by image ID...")
        rsna_set = set(rsna_image_ids)
        real_emr_set = set(real_emr_image_ids)
        common_ids = rsna_set.intersection(real_emr_set)
        
        if len(common_ids) > 0:
            print(f"   Found {len(common_ids)} matching image IDs")
            
            # Create mapping
            rsna_id_to_idx = {img_id: idx for idx, img_id in enumerate(rsna_image_ids)}
            real_emr_id_to_idx = {img_id: idx for idx, img_id in enumerate(real_emr_image_ids)}
            
            # Align data
            aligned_image_ids = []
            aligned_labels = []
            aligned_emr_features = []
            
            for img_id in common_ids:
                rsna_idx = rsna_id_to_idx[img_id]
                emr_idx = real_emr_id_to_idx[img_id]
                
                aligned_image_ids.append(img_id)
                aligned_labels.append(rsna_labels[rsna_idx])  # Use RSNA labels
                aligned_emr_features.append(real_emr_features[emr_idx])
            
            print(f"   Aligned {len(aligned_image_ids)} samples by image ID")
        else:
            print("   No matching image IDs found, using index-based alignment")
            # Fall back to index-based alignment
            aligned_image_ids, aligned_labels, aligned_emr_features = _index_based_alignment(
                rsna_image_ids, rsna_labels, real_emr_features, real_emr_labels, max_samples
            )
    else:
        # Use index-based alignment
        print("   Using index-based alignment...")
        aligned_image_ids, aligned_labels, aligned_emr_features = _index_based_alignment(
            rsna_image_ids, rsna_labels, real_emr_features, real_emr_labels, max_samples
        )
    
    # Limit samples if specified
    if max_samples is not None and len(aligned_image_ids) > max_samples:
        print(f"   Limiting to {max_samples} samples for testing")
        indices = random.sample(range(len(aligned_image_ids)), max_samples)
        aligned_image_ids = [aligned_image_ids[i] for i in indices]
        aligned_labels = [aligned_labels[i] for i in indices]
        aligned_emr_features = aligned_emr_features[indices]
    
    print(f"   Final aligned samples: {len(aligned_image_ids)}")
    return aligned_image_ids, aligned_labels, aligned_emr_features

def _index_based_alignment(rsna_image_ids, rsna_labels, real_emr_features, real_emr_labels, max_samples):
    """Index-based alignment when image ID matching fails"""
    min_length = min(len(rsna_image_ids), len(real_emr_features))
    
    aligned_image_ids = rsna_image_ids[:min_length]
    aligned_labels = rsna_labels[:min_length]
    aligned_emr_features = real_emr_features[:min_length]
    
    return aligned_image_ids, aligned_labels, aligned_emr_features

def make_cxr_real_emr_dataloaders(rsna_arrays_dir: str, real_emr_arrays_dir: str, 
                                 rsna_root: str, batch_size: int, num_workers: int,
                                 max_samples: Optional[int] = None) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for CXR + Real EMR fusion model
    
    Args:
        rsna_arrays_dir: Directory containing RSNA processed arrays
        real_emr_arrays_dir: Directory containing real EMR processed arrays
        rsna_root: Root directory of RSNA dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
        max_samples: Maximum number of samples to use (for testing)
    
    Returns:
        Dictionary of DataLoaders for train, val, test splits
    """
    print("🔄 Creating CXR + Real EMR DataLoaders...")
    
    # Load RSNA data
    print("   Loading RSNA data...")
    rsna_image_ids_train = np.load(Path(rsna_arrays_dir) / "image_ids_train.npy", allow_pickle=True)
    rsna_image_ids_val = np.load(Path(rsna_arrays_dir) / "image_ids_val.npy", allow_pickle=True)
    rsna_image_ids_test = np.load(Path(rsna_arrays_dir) / "image_ids_test.npy", allow_pickle=True)
    
    rsna_y_train = np.load(Path(rsna_arrays_dir) / "y_train.npy")
    rsna_y_val = np.load(Path(rsna_arrays_dir) / "y_val.npy")
    rsna_y_test = np.load(Path(rsna_arrays_dir) / "y_test.npy")
    
    print(f"   ✅ RSNA train: {len(rsna_image_ids_train)} samples")
    print(f"   ✅ RSNA val: {len(rsna_image_ids_val)} samples")
    print(f"   ✅ RSNA test: {len(rsna_image_ids_test)} samples")
    
    # Load Real EMR data
    print("   Loading Real EMR data...")
    real_emr_X_train = np.load(Path(real_emr_arrays_dir) / "X_real_emr_train.npy")
    real_emr_X_val = np.load(Path(real_emr_arrays_dir) / "X_real_emr_val.npy")
    real_emr_X_test = np.load(Path(real_emr_arrays_dir) / "X_real_emr_test.npy")
    
    real_emr_y_train = np.load(Path(real_emr_arrays_dir) / "y_real_emr_train.npy")
    real_emr_y_val = np.load(Path(real_emr_arrays_dir) / "y_real_emr_val.npy")
    real_emr_y_test = np.load(Path(real_emr_arrays_dir) / "y_real_emr_test.npy")
    
    print(f"   ✅ Real EMR train: {len(real_emr_X_train)} samples")
    print(f"   ✅ Real EMR val: {len(real_emr_X_val)} samples")
    print(f"   ✅ Real EMR test: {len(real_emr_X_test)} samples")
    
    # Align datasets for each split
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n   Processing {split} split...")
        
        # Get data for this split
        rsna_image_ids = locals()[f'rsna_image_ids_{split}']
        rsna_labels = locals()[f'rsna_y_{split}']
        real_emr_features = locals()[f'real_emr_X_{split}']
        real_emr_labels = locals()[f'real_emr_y_{split}']
        
        # Align datasets
        aligned_image_ids, aligned_labels, aligned_emr_features = align_datasets(
            rsna_image_ids, rsna_labels, real_emr_features, real_emr_labels, 
            max_samples=max_samples
        )
        
        # Create dataset
        transform = get_transforms(split, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)
        dataset = CXRRealEMRFusionDataset(
            image_ids=aligned_image_ids,
            labels=aligned_labels,
            emr_features=aligned_emr_features,
            rsna_root=rsna_root,
            transform=transform
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        
        dataloaders[split] = dataloader
        
        # Print class distribution
        unique_labels, counts = np.unique(aligned_labels, return_counts=True)
        class_dist = dict(zip(unique_labels, counts))
        print(f"   ✅ {split}: {len(dataset)} samples, classes: {class_dist}")
    
    print(f"\n✅ Created DataLoaders for all splits")
    return dataloaders

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced dataset"""
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(labels)
    class_weights = total_samples / (len(unique_classes) * counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(unique_classes)
    
    return torch.tensor(class_weights, dtype=torch.float32)

if __name__ == "__main__":
    # Test dataset creation
    print("Testing CXR + Real EMR Fusion Dataset...")
    
    # Create dummy data
    dummy_image_ids = [f"test_{i}" for i in range(10)]
    dummy_labels = [0, 1] * 5
    dummy_emr_features = np.random.randn(10, 13)  # 13 real EMR features
    
    # Create dataset
    dataset = CXRRealEMRFusionDataset(
        image_ids=dummy_image_ids,
        labels=dummy_labels,
        emr_features=dummy_emr_features,
        rsna_root=str(RSNA_ROOT),
        transform=None
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    for batch in dataloader:
        print(f"Batch shapes:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  EMR features: {batch['emr_features'].shape}")
        print(f"  Labels: {batch['label'].shape}")
        print(f"  Image IDs: {len(batch['image_id'])}")
        break
    
    print("✅ Dataset test completed!")
