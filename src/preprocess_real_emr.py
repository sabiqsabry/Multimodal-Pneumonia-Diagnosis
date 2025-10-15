"""
Real EMR Data Preprocessing for Pneumonia Classification
Processes the essienmary/pneumonia-dataset for EMR-only experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import json
from typing import Tuple, Dict, Any

from config import SEED
from utils_io import ensure_dir, save_json, save_numpy

# Real EMR specific paths
REAL_EMR_RAW_DIR = Path("data/raw/real_emr")
REAL_EMR_PROC_DIR = Path("data/processed/real_emr")
REAL_EMR_CSV = REAL_EMR_RAW_DIR / "pneumonia_dataset.csv"

# Feature mappings for real EMR dataset
REAL_EMR_FEATURES = {
    'continuous': ['age', 'oxygen_saturation', 'wbc_count', 'temperature'],
    'categorical': ['gender', 'cough', 'fever', 'shortness_of_breath', 'chest_pain', 
                   'fatigue', 'confusion', 'crackles', 'sputum_color'],
    'target': 'diagnosis'
}

# Label mapping
REAL_EMR_LABEL_MAP = {"No": 0, "Yes": 1}

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to lowercase with underscores"""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    return df_clean

def identify_available_features(df: pd.DataFrame) -> Dict[str, list]:
    """Identify which features are available in the dataset"""
    available_features = {
        'continuous': [],
        'categorical': [],
        'target': None
    }
    
    # Check continuous features
    for feature in REAL_EMR_FEATURES['continuous']:
        if feature.lower() in df.columns:
            available_features['continuous'].append(feature.lower())
    
    # Check categorical features
    for feature in REAL_EMR_FEATURES['categorical']:
        if feature.lower() in df.columns:
            available_features['categorical'].append(feature.lower())
    
    # Check target
    target = REAL_EMR_FEATURES['target']
    if target in df.columns:
        available_features['target'] = target
    
    return available_features

def handle_missing_values(df: pd.DataFrame, available_features: Dict[str, list]) -> pd.DataFrame:
    """Handle missing values using appropriate imputation strategies"""
    df_clean = df.copy()
    
    # Convert continuous features to numeric, replacing non-numeric values with NaN
    for feature in available_features['continuous']:
        if feature in df_clean.columns:
            df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
    
    # Impute continuous features with mean
    continuous_imputer = SimpleImputer(strategy='mean')
    if available_features['continuous']:
        df_clean[available_features['continuous']] = continuous_imputer.fit_transform(
            df_clean[available_features['continuous']]
        )
    
    # Impute categorical features with mode
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    if available_features['categorical']:
        df_clean[available_features['categorical']] = categorical_imputer.fit_transform(
            df_clean[available_features['categorical']]
        )
    
    return df_clean

def encode_categorical_features(df: pd.DataFrame, available_features: Dict[str, list]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder"""
    df_encoded = df.copy()
    encoders = {}
    
    for feature in available_features['categorical']:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
            encoders[feature] = le
    
    return df_encoded, encoders

def scale_continuous_features(df: pd.DataFrame, available_features: Dict[str, list], 
                            scaler_path: Path) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale continuous features using StandardScaler"""
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    if available_features['continuous']:
        df_scaled[available_features['continuous']] = scaler.fit_transform(
            df_scaled[available_features['continuous']]
        )
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        print(f"   ✅ Scaler saved to: {scaler_path}")
    
    return df_scaled, scaler

def create_feature_arrays(df: pd.DataFrame, available_features: Dict[str, list]) -> Tuple[np.ndarray, np.ndarray, list]:
    """Create feature arrays and labels"""
    # Select features in order: continuous first, then categorical
    feature_columns = available_features['continuous'] + available_features['categorical']
    
    # Create feature matrix
    X = df[feature_columns].values.astype(np.float32)
    
    # Create labels - map string labels to integers
    target_col = available_features['target']
    y = df[target_col].map(REAL_EMR_LABEL_MAP).values.astype(np.int32)
    
    return X, y, feature_columns

def create_stratified_splits(X: np.ndarray, y: np.ndarray, feature_columns: list, 
                           test_size: float = 0.15, val_size: float = 0.15, 
                           random_state: int = SEED) -> Dict[str, Any]:
    """Create stratified train/validation/test splits"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_columns': feature_columns
    }

def save_splits(splits: Dict[str, Any], output_dir: Path):
    """Save split data and metadata"""
    # Save numpy arrays
    save_numpy(splits['X_train'], output_dir / "X_real_emr_train.npy")
    save_numpy(splits['X_val'], output_dir / "X_real_emr_val.npy")
    save_numpy(splits['X_test'], output_dir / "X_real_emr_test.npy")
    save_numpy(splits['y_train'], output_dir / "y_real_emr_train.npy")
    save_numpy(splits['y_val'], output_dir / "y_real_emr_val.npy")
    save_numpy(splits['y_test'], output_dir / "y_real_emr_test.npy")
    
    # Save feature columns
    feature_info = {
        'feature_columns': splits['feature_columns'],
        'n_features': len(splits['feature_columns']),
        'feature_types': {
            'continuous': [col for col in splits['feature_columns'] if col in ['age', 'oxygen_saturation', 'wbc_count', 'temperature']],
            'categorical': [col for col in splits['feature_columns'] if col not in ['age', 'oxygen_saturation', 'wbc_count', 'temperature']]
        }
    }
    save_json(feature_info, output_dir / "feature_info.json")
    
    # Save split CSVs for reference
    for split_name in ['train', 'val', 'test']:
        split_df = pd.DataFrame({
            'split': split_name,
            'label': splits[f'y_{split_name}']
        })
        split_df.to_csv(output_dir / f"real_emr_splits_{split_name}.csv", index=False)

def print_summary(splits: Dict[str, Any], original_df: pd.DataFrame):
    """Print preprocessing summary"""
    print("\n" + "="*60)
    print("📊 REAL EMR PREPROCESSING SUMMARY")
    print("="*60)
    
    # Dataset info
    print(f"📁 Original dataset size: {len(original_df)} samples")
    print(f"🔢 Processed features: {splits['n_features']}")
    print(f"📋 Feature columns: {splits['feature_columns']}")
    
    # Split sizes
    print(f"\n📊 Split sizes:")
    for split_name in ['train', 'val', 'test']:
        n_samples = len(splits[f'X_{split_name}'])
        n_pneumonia = np.sum(splits[f'y_{split_name}'])
        n_normal = n_samples - n_pneumonia
        print(f"   {split_name.capitalize()}: {n_samples} samples ({n_pneumonia} pneumonia, {n_normal} normal)")
    
    # Class balance
    total_pneumonia = np.sum(splits['y_train']) + np.sum(splits['y_val']) + np.sum(splits['y_test'])
    total_samples = len(splits['y_train']) + len(splits['y_val']) + len(splits['y_test'])
    pneumonia_rate = total_pneumonia / total_samples * 100
    
    print(f"\n⚖️ Class balance:")
    print(f"   Pneumonia: {total_pneumonia}/{total_samples} ({pneumonia_rate:.1f}%)")
    print(f"   Normal: {total_samples - total_pneumonia}/{total_samples} ({100-pneumonia_rate:.1f}%)")
    
    # Feature statistics
    print(f"\n📈 Feature statistics (train set):")
    feature_means = np.mean(splits['X_train'], axis=0)
    feature_stds = np.std(splits['X_train'], axis=0)
    
    for i, feature in enumerate(splits['feature_columns']):
        print(f"   {feature}: mean={feature_means[i]:.3f}, std={feature_stds[i]:.3f}")
    
    print("="*60)
    print("✅ Real EMR preprocessing completed successfully!")
    print("="*60)

def preprocess_real_emr_data(csv_path: Path = REAL_EMR_CSV, output_dir: Path = REAL_EMR_PROC_DIR):
    """
    Main preprocessing function for real EMR data
    
    Args:
        csv_path: Path to real EMR CSV file
        output_dir: Output directory for processed data
    """
    print("🚀 Starting Real EMR Data Preprocessing...")
    print("="*60)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Load data
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Clean column names
    print("\n🧹 Cleaning column names...")
    df_clean = clean_column_names(df)
    print(f"   Columns: {df_clean.columns.tolist()}")
    
    # Identify available features
    print("\n🔍 Identifying available features...")
    available_features = identify_available_features(df_clean)
    print(f"   Continuous: {available_features['continuous']}")
    print(f"   Categorical: {available_features['categorical']}")
    print(f"   Target: {available_features['target']}")
    
    if not available_features['target']:
        raise ValueError("No target column found in dataset")
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    df_imputed = handle_missing_values(df_clean, available_features)
    print(f"   Missing values after imputation: {df_imputed.isnull().sum().sum()}")
    
    # Encode categorical features
    print("\n🏷️ Encoding categorical features...")
    df_encoded, encoders = encode_categorical_features(df_imputed, available_features)
    print(f"   Encoded {len(encoders)} categorical features")
    
    # Scale continuous features
    print("\n📏 Scaling continuous features...")
    scaler_path = output_dir / "real_emr_scaler.joblib"
    df_scaled, scaler = scale_continuous_features(df_encoded, available_features, scaler_path)
    
    # Create feature arrays
    print("\n🔢 Creating feature arrays...")
    X, y, feature_columns = create_feature_arrays(df_scaled, available_features)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Create stratified splits
    print("\n✂️ Creating stratified splits...")
    splits = create_stratified_splits(X, y, feature_columns)
    splits['n_features'] = len(feature_columns)
    
    # Save splits
    print("\n💾 Saving processed data...")
    save_splits(splits, output_dir)
    
    # Save label map
    label_map_path = output_dir / "real_emr_label_map.json"
    save_json(REAL_EMR_LABEL_MAP, label_map_path)
    print(f"   ✅ Label map saved to: {label_map_path}")
    
    # Print summary
    print_summary(splits, df)
    
    return splits

if __name__ == "__main__":
    # Run preprocessing
    splits = preprocess_real_emr_data()
    print(f"\n📁 All processed data saved to: {REAL_EMR_PROC_DIR}")
