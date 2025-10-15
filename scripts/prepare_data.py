#!/usr/bin/env python3
"""
Data Preparation Orchestrator Script
Handles EMR preprocessing, splits, and class weight computation
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import EMR_CSV, PROC_DIR, SEED, SPLITS, LABEL_MAP_FILE, CLASS_WEIGHTS_FILE, LABEL_MAP
from preprocess_emr import prepare_emr
from data_module import compute_class_weights
from utils_io import save_torch, load_numpy
import numpy as np

def prepare_emr_data():
    """Prepare EMR data: clean, encode, scale, and create splits"""
    print("🚀 Starting EMR data preparation...")
    print("="*60)
    
    # Run EMR preprocessing
    result = prepare_emr(
        emr_csv_path=str(EMR_CSV),
        output_dir=str(PROC_DIR),
        seed=SEED,
        splits=SPLITS
    )
    
    print("\n" + "="*60)
    print("✅ EMR data preparation completed!")
    print(f"   Total samples: {len(result['y'])}")
    print(f"   Features: {len(result['feature_columns'])}")
    print(f"   Splits: {list(result['splits'].keys())}")
    
    return result

def compute_class_weights_step():
    """Compute and save class weights for imbalanced dataset"""
    print("⚖️ Computing class weights...")
    print("="*40)
    
    # Load training labels
    y_train_path = PROC_DIR / "y_train.npy"
    if not y_train_path.exists():
        print(f"❌ Training labels not found: {y_train_path}")
        print("   Please run --emr first!")
        return False
    
    y_train = load_numpy(y_train_path)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Save class weights
    weights_path = PROC_DIR / CLASS_WEIGHTS_FILE
    save_torch(class_weights, weights_path)
    
    # Print class distribution and weights
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📊 Class Distribution:")
    for i, (label_val, count) in enumerate(zip(unique, counts)):
        label_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(label_val)]
        weight = class_weights[i].item()
        print(f"   {label_name}: {count} samples, weight: {weight:.3f}")
    
    print(f"\n✅ Class weights saved to: {weights_path}")
    return True

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Data Preparation Script for FYP Pneumonia Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py --emr
  python scripts/prepare_data.py --compute-class-weights
  python scripts/prepare_data.py --emr --compute-class-weights
        """
    )
    
    parser.add_argument(
        '--emr',
        action='store_true',
        help='Prepare EMR data: clean, encode, scale, and create splits'
    )
    
    parser.add_argument(
        '--compute-class-weights',
        action='store_true',
        help='Compute and save class weights for imbalanced dataset'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preparation steps'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.emr, args.compute_class_weights, args.all]):
        parser.print_help()
        return
    
    # Ensure directories exist
    from config import ensure_directories
    ensure_directories()
    
    success = True
    
    # Run EMR preparation if requested
    if args.emr or args.all:
        try:
            prepare_emr_data()
        except Exception as e:
            print(f"❌ EMR preparation failed: {e}")
            success = False
    
    # Run class weight computation if requested
    if args.compute_class_weights or args.all:
        try:
            if not compute_class_weights_step():
                success = False
        except Exception as e:
            print(f"❌ Class weight computation failed: {e}")
            success = False
    
    # Final status
    if success:
        print("\n" + "="*60)
        print("🎉 All data preparation steps completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some steps failed. Please check the errors above.")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
