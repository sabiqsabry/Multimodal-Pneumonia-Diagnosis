"""
EMR Data Preprocessing Module
Handles EMR data cleaning, encoding, scaling, and stratified splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import (
    EMR_CSV, PROC_DIR, SPLITS, SEED, CATEGORICAL_FEATURES, 
    CONTINUOUS_FEATURES, TARGET_COLUMN, LABEL_MAP, 
    LABEL_MAP_FILE, EMR_SCALER_FILE, SPLIT_FILES, ARRAY_FILES
)
from utils_io import save_json, save_joblib, save_numpy

def prepare_emr(emr_csv_path: str, output_dir: str, seed: int = 42, splits: Tuple[float, float, float] = (0.70, 0.15, 0.15)) -> Dict:
    """
    Prepare EMR data: load, clean, encode, scale, and create stratified splits
    
    Args:
        emr_csv_path: Path to EMR CSV file
        output_dir: Directory to save processed data
        seed: Random seed for reproducibility
        splits: Train/val/test split ratios
        
    Returns:
        Dictionary containing processed data and metadata
    """
    print("🔄 Starting EMR preprocessing...")
    
    # Load EMR data
    print(f"📁 Loading EMR data from: {emr_csv_path}")
    df = pd.read_csv(emr_csv_path)
    print(f"   Loaded {len(df)} records")
    
    # Clean and prepare data
    print("🧹 Cleaning data...")
    
    # Convert labels to lowercase and map to integers
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.lower()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_MAP)
    
    # Handle missing values
    df = df.dropna()
    print(f"   After cleaning: {len(df)} records")
    
    # Prepare features
    print("🔧 Preparing features...")
    
    # One-hot encode categorical features
    df_encoded = df.copy()
    for feature in CATEGORICAL_FEATURES:
        if feature == 'sex':
            # Create binary columns for sex
            df_encoded['sex_M'] = (df_encoded[feature] == 'M').astype(int)
            df_encoded['sex_F'] = (df_encoded[feature] == 'F').astype(int)
    
    # Select feature columns
    feature_columns = CONTINUOUS_FEATURES + ['sex_M', 'sex_F']
    X_emr = df_encoded[feature_columns].values.astype(np.float32)
    y = df_encoded[TARGET_COLUMN].values.astype(np.int64)
    image_ids = df_encoded['image_id'].values
    
    print(f"   Features: {len(feature_columns)} ({feature_columns})")
    print(f"   EMR shape: {X_emr.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Scale continuous features
    print("📏 Scaling continuous features...")
    scaler = StandardScaler()
    
    # Scale only continuous features (first 5 columns)
    continuous_mask = np.array([col in CONTINUOUS_FEATURES for col in feature_columns])
    X_emr[:, continuous_mask] = scaler.fit_transform(X_emr[:, continuous_mask])
    
    # Print feature statistics
    print("\n📊 Feature Statistics (after scaling):")
    for i, col in enumerate(feature_columns):
        mean_val = X_emr[:, i].mean()
        std_val = X_emr[:, i].std()
        print(f"   {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Create stratified splits
    print(f"\n🎯 Creating stratified splits: {splits}")
    train_ratio, val_ratio, test_ratio = splits
    
    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(X_emr, y))
    
    # Second split: val vs test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed)
    val_idx, test_idx = next(sss2.split(X_emr[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    
    # Create splits
    splits_data = {
        'train': (train_idx, 'train'),
        'val': (val_idx, 'val'),
        'test': (test_idx, 'test')
    }
    
    # Print split statistics
    print("\n📈 Split Statistics:")
    for split_name, (indices, _) in splits_data.items():
        split_y = y[indices]
        unique, counts = np.unique(split_y, return_counts=True)
        print(f"   {split_name.upper()}: {len(indices)} samples")
        for label_val, count in zip(unique, counts):
            label_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(label_val)]
            print(f"     - {label_name}: {count} ({count/len(indices)*100:.1f}%)")
    
    # Save processed data
    print(f"\n💾 Saving processed data to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save label map
    save_json(LABEL_MAP, output_path / LABEL_MAP_FILE)
    
    # Save scaler
    save_joblib(scaler, output_path / EMR_SCALER_FILE)
    
    # Save splits as CSV files
    for split_name, (indices, _) in splits_data.items():
        split_df = pd.DataFrame({
            'image_id': image_ids[indices],
            'label': y[indices],
            'split': split_name
        })
        split_df.to_csv(output_path / SPLIT_FILES[split_name], index=False)
        print(f"   ✅ Saved {SPLIT_FILES[split_name]}")
    
    # Save arrays
    for split_name, (indices, _) in splits_data.items():
        # EMR features
        emr_path = output_path / ARRAY_FILES['emr'].format(split=split_name)
        save_numpy(X_emr[indices], emr_path)
        
        # Labels
        y_path = output_path / ARRAY_FILES['y'].format(split=split_name)
        save_numpy(y[indices], y_path)
        
        # Image IDs (save as text file to avoid pickle issues)
        ids_path = output_path / ARRAY_FILES['image_ids'].format(split=split_name)
        np.save(ids_path, image_ids[indices], allow_pickle=True)
    
    print("✅ EMR preprocessing completed!")
    
    return {
        'X_emr': X_emr,
        'y': y,
        'image_ids': image_ids,
        'feature_columns': feature_columns,
        'splits': splits_data,
        'scaler': scaler
    }

def main():
    """Main function for testing"""
    result = prepare_emr(str(EMR_CSV), str(PROC_DIR), SEED, SPLITS)
    print(f"\n🎉 EMR preprocessing completed successfully!")
    print(f"   Total samples: {len(result['y'])}")
    print(f"   Features: {len(result['feature_columns'])}")

if __name__ == "__main__":
    main()
