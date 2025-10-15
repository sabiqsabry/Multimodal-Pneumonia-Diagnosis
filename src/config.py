"""
Configuration file for FYP Pneumonia Diagnosis Project
Centralizes all paths, constants, and hyperparameters
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RSNA_ROOT = PROJECT_ROOT / "data" / "raw" / "rsna"
EMR_CSV = PROJECT_ROOT / "data" / "synthetic_emr" / "emr_data.csv"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROC_DIR / "cache"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Image preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Data splits
SPLITS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 4

# EMR feature columns
CATEGORICAL_FEATURES = ["sex"]
CONTINUOUS_FEATURES = ["age", "fever_temp", "cough_score", "WBC_count", "SpO2"]
TARGET_COLUMN = "label"

# Label mapping
LABEL_MAP = {"pneumonia": 1, "normal": 0}

# File names
LABEL_MAP_FILE = "label_map.json"
EMR_SCALER_FILE = "emr_scaler.joblib"
CLASS_WEIGHTS_FILE = "class_weights.pt"

# Split file names
SPLIT_FILES = {
    "train": "splits_train.csv",
    "val": "splits_val.csv", 
    "test": "splits_test.csv"
}

# Array file names
ARRAY_FILES = {
    "emr": "emr_{split}.npy",
    "y": "y_{split}.npy",
    "image_ids": "image_ids_{split}.npy"
}

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [PROC_DIR, CACHE_DIR, FIG_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ All directories created/verified")
    print(f"   - Processed data: {PROC_DIR}")
    print(f"   - Cache: {CACHE_DIR}")
    print(f"   - Figures: {FIG_DIR}")

if __name__ == "__main__":
    ensure_directories()
