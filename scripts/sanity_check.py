#!/usr/bin/env python3
"""
Sanity Check Script
Loads processed data, builds DataLoaders, and creates sample visualizations
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import PROC_DIR, RSNA_ROOT, FIG_DIR, IMAGENET_MEAN, IMAGENET_STD, LABEL_MAP
from data_module import make_dataloaders, get_sample_batch, print_batch_info, validate_dataloader
from preprocess_images import denormalize_image
from utils_io import load_json

def create_sample_visualization(dataloaders, output_path):
    """Create a sample visualization of the dataset"""
    print("🎨 Creating sample visualization...")
    
    # Get a sample batch from training data
    train_batch = get_sample_batch(dataloaders['train'], num_samples=4)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    # Load label map
    label_map_path = PROC_DIR / "label_map.json"
    if label_map_path.exists():
        label_map = load_json(label_map_path)
        # Reverse the mapping for display
        id_to_label = {v: k for k, v in label_map.items()}
    else:
        id_to_label = {0: "Normal", 1: "Pneumonia"}
    
    for i in range(4):
        # Get image and denormalize
        image = train_batch['image'][i]
        denorm_image = denormalize_image(image, IMAGENET_MEAN, IMAGENET_STD)
        
        # Convert to numpy and transpose for matplotlib
        image_np = denorm_image.permute(1, 2, 0).numpy()
        
        # Get label
        label_id = train_batch['label'][i].item()
        label_name = id_to_label.get(label_id, f"Class {label_id}")
        
        # Get EMR features
        emr_features = train_batch['emr'][i].numpy()
        
        # Plot image
        axes[i].imshow(image_np)
        axes[i].set_title(f"{label_name}\nEMR: {emr_features[:3].round(2)}...", fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sample visualization saved to: {output_path}")

def run_sanity_check():
    """Run comprehensive sanity check on the dataset"""
    print("🔍 Running Sanity Check...")
    print("="*50)
    
    # Check if processed data exists
    required_files = [
        "emr_train.npy", "emr_val.npy", "emr_test.npy",
        "y_train.npy", "y_val.npy", "y_test.npy",
        "image_ids_train.npy", "image_ids_val.npy", "image_ids_test.npy",
        "label_map.json", "emr_scaler.joblib", "class_weights.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not (PROC_DIR / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please run: python scripts/prepare_data.py --all")
        return False
    
    print("✅ All required files found")
    
    # Create DataLoaders
    print("\n🔄 Creating DataLoaders...")
    try:
        dataloaders = make_dataloaders(
            split_csv_dir=str(PROC_DIR),
            emr_arrays_dir=str(PROC_DIR),
            rsna_root=str(RSNA_ROOT),
            batch_size=8,  # Smaller batch for testing
            num_workers=2,  # Fewer workers for testing
            cache=False  # No caching for testing
        )
        print("✅ DataLoaders created successfully")
    except Exception as e:
        print(f"❌ Failed to create DataLoaders: {e}")
        return False
    
    # Validate each DataLoader
    print("\n🔍 Validating DataLoaders...")
    for split, dataloader in dataloaders.items():
        if not validate_dataloader(dataloader, max_batches=2):
            print(f"❌ {split} DataLoader validation failed")
            return False
        print(f"✅ {split} DataLoader validated")
    
    # Test batch loading
    print("\n📊 Testing batch loading...")
    for split, dataloader in dataloaders.items():
        try:
            batch = get_sample_batch(dataloader, num_samples=4)
            print_batch_info(batch, split)
        except Exception as e:
            print(f"❌ Failed to load batch from {split}: {e}")
            return False
    
    # Create sample visualization
    print("\n🎨 Creating sample visualization...")
    try:
        output_path = FIG_DIR / "sample_batch.png"
        create_sample_visualization(dataloaders, output_path)
        print(f"✅ Sample visualization saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to create visualization: {e}")
        return False
    
    # Print final statistics
    print("\n📈 Dataset Statistics:")
    total_samples = 0
    for split, dataloader in dataloaders.items():
        dataset_size = len(dataloader.dataset)
        total_samples += dataset_size
        print(f"   {split}: {dataset_size} samples")
    
    print(f"   Total: {total_samples} samples")
    
    # Check class balance
    train_dataset = dataloaders['train'].dataset
    class_counts = train_dataset.get_class_counts()
    print(f"\n⚖️ Training Class Distribution:")
    for class_id, count in class_counts.items():
        label_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(class_id)]
        percentage = count / len(train_dataset) * 100
        print(f"   {label_name}: {count} ({percentage:.1f}%)")
    
    return True

def main():
    """Main function"""
    print("🚀 FYP Pneumonia Diagnosis - Sanity Check")
    print("="*60)
    
    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run sanity check
    success = run_sanity_check()
    
    if success:
        print("\n" + "="*60)
        print("✅ Sanity check completed successfully!")
        print("   All DataLoaders are working correctly.")
        print("   Sample visualization created.")
        print("   Ready for training!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Sanity check failed!")
        print("   Please check the errors above and fix them.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
