#!/usr/bin/env python3
"""
Synthetic EMR Dataset Generator for FYP Pneumonia Diagnosis Project
Generates synthetic Electronic Medical Records aligned with RSNA dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import random

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)

def load_rsna_metadata():
    """Load RSNA training metadata to get image IDs and labels"""
    train_metadata_path = "data/raw/rsna/stage2_train_metadata.csv"
    test_metadata_path = "data/raw/rsna/stage2_test_metadata.csv"
    
    # Load training data
    train_df = pd.read_csv(train_metadata_path)
    print(f"Loaded training metadata: {len(train_df)} records")
    
    # Load test data
    test_df = pd.read_csv(test_metadata_path)
    print(f"Loaded test metadata: {len(test_df)} records")
    
    return train_df, test_df

def extract_image_labels(train_df):
    """Extract image IDs and their corresponding labels from RSNA metadata"""
    # Group by patientId to get unique patients and their labels
    patient_labels = train_df.groupby('patientId').agg({
        'Target': 'max',  # 1 if any image has pneumonia, 0 otherwise
        'class': 'first'  # Get the class label
    }).reset_index()
    
    # Map Target values to our labels
    # Target = 1 means Lung Opacity (Pneumonia), Target = 0 means Normal/No Lung Opacity
    patient_labels['label'] = patient_labels['Target'].map({0: 'Normal', 1: 'Pneumonia'})
    
    print(f"Unique patients: {len(patient_labels)}")
    print(f"Pneumonia cases: {sum(patient_labels['label'] == 'Pneumonia')}")
    print(f"Normal cases: {sum(patient_labels['label'] == 'Normal')}")
    
    return patient_labels

def generate_synthetic_emr_data(patient_labels):
    """Generate synthetic EMR data for each patient"""
    n_patients = len(patient_labels)
    
    # Initialize lists for EMR data
    emr_data = []
    
    for idx, row in patient_labels.iterrows():
        patient_id = row['patientId']
        label = row['label']
        
        # Generate synthetic EMR features based on label
        if label == 'Pneumonia':
            # Pneumonia patients tend to have higher fever, cough scores, WBC, lower SpO2
            age = np.random.randint(1, 91)
            sex = np.random.choice(['M', 'F'])
            fever_temp = np.random.normal(38.5, 1.0)  # Higher fever for pneumonia
            fever_temp = np.clip(fever_temp, 36.5, 41.0)
            cough_score = np.random.randint(3, 6)  # Higher cough scores
            wbc_count = np.random.normal(12000, 3000)  # Higher WBC
            wbc_count = np.clip(wbc_count, 4000, 20000)
            spo2 = np.random.normal(92, 5)  # Lower SpO2
            spo2 = np.clip(spo2, 70, 100)
        else:
            # Normal patients have more normal vital signs
            age = np.random.randint(1, 91)
            sex = np.random.choice(['M', 'F'])
            fever_temp = np.random.normal(37.0, 0.5)  # Normal temperature
            fever_temp = np.clip(fever_temp, 36.5, 41.0)
            cough_score = np.random.randint(0, 3)  # Lower cough scores
            wbc_count = np.random.normal(7000, 2000)  # Normal WBC
            wbc_count = np.clip(wbc_count, 4000, 20000)
            spo2 = np.random.normal(98, 2)  # Higher SpO2
            spo2 = np.clip(spo2, 70, 100)
        
        # Create image_id (using patient_id as base)
        image_id = f"{patient_id}_001"  # Assuming one image per patient for simplicity
        
        emr_data.append({
            'image_id': image_id,
            'age': int(age),
            'sex': sex,
            'fever_temp': round(fever_temp, 1),
            'cough_score': int(cough_score),
            'WBC_count': int(wbc_count),
            'SpO2': round(spo2, 1),
            'label': label
        })
    
    return pd.DataFrame(emr_data)

def print_dataset_statistics(emr_df):
    """Print comprehensive dataset statistics"""
    print("\n" + "="*60)
    print("SYNTHETIC EMR DATASET STATISTICS")
    print("="*60)
    
    # Label distribution
    label_counts = emr_df['label'].value_counts()
    print(f"\nLabel Distribution:")
    print(f"  Pneumonia: {label_counts.get('Pneumonia', 0)} cases")
    print(f"  Normal: {label_counts.get('Normal', 0)} cases")
    print(f"  Total: {len(emr_df)} cases")
    
    # Age statistics
    print(f"\nAge Statistics:")
    print(f"  Mean: {emr_df['age'].mean():.1f} years")
    print(f"  Std: {emr_df['age'].std():.1f} years")
    print(f"  Range: {emr_df['age'].min()}-{emr_df['age'].max()} years")
    
    # WBC count statistics
    print(f"\nWBC Count Statistics:")
    print(f"  Mean: {emr_df['WBC_count'].mean():.0f}")
    print(f"  Std: {emr_df['WBC_count'].std():.0f}")
    print(f"  Range: {emr_df['WBC_count'].min()}-{emr_df['WBC_count'].max()}")
    
    # SpO2 statistics
    print(f"\nSpO2 Statistics:")
    print(f"  Mean: {emr_df['SpO2'].mean():.1f}%")
    print(f"  Std: {emr_df['SpO2'].std():.1f}%")
    print(f"  Range: {emr_df['SpO2'].min()}-{emr_df['SpO2'].max()}%")
    
    # Fever temperature statistics
    print(f"\nFever Temperature Statistics:")
    print(f"  Mean: {emr_df['fever_temp'].mean():.1f}°C")
    print(f"  Std: {emr_df['fever_temp'].std():.1f}°C")
    print(f"  Range: {emr_df['fever_temp'].min()}-{emr_df['fever_temp'].max()}°C")
    
    # Cough score statistics
    print(f"\nCough Score Statistics:")
    print(f"  Mean: {emr_df['cough_score'].mean():.1f}")
    print(f"  Std: {emr_df['cough_score'].std():.1f}")
    print(f"  Range: {emr_df['cough_score'].min()}-{emr_df['cough_score'].max()}")
    
    # Sex distribution
    sex_counts = emr_df['sex'].value_counts()
    print(f"\nSex Distribution:")
    print(f"  Male: {sex_counts.get('M', 0)} cases")
    print(f"  Female: {sex_counts.get('F', 0)} cases")
    
    # First 5 rows
    print(f"\nFirst 5 rows of EMR dataset:")
    print(emr_df.head().to_string(index=False))

def main():
    """Main function to generate synthetic EMR dataset"""
    print("Starting Synthetic EMR Dataset Generation...")
    print("="*50)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/synthetic_emr")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RSNA metadata
    train_df, test_df = load_rsna_metadata()
    
    # Extract image labels from training data
    patient_labels = extract_image_labels(train_df)
    
    # Generate synthetic EMR data
    print(f"\nGenerating synthetic EMR data for {len(patient_labels)} patients...")
    emr_df = generate_synthetic_emr_data(patient_labels)
    
    # Save to CSV
    output_path = output_dir / "emr_data.csv"
    emr_df.to_csv(output_path, index=False)
    print(f"\nSynthetic EMR dataset saved to: {output_path}")
    
    # Print statistics
    print_dataset_statistics(emr_df)
    
    print(f"\n" + "="*60)
    print("✅ SYNTHETIC EMR DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Dataset saved to: {output_path}")
    print(f"📊 Total records: {len(emr_df)}")
    print(f"🏥 Pneumonia cases: {sum(emr_df['label'] == 'Pneumonia')}")
    print(f"💚 Normal cases: {sum(emr_df['label'] == 'Normal')}")

if __name__ == "__main__":
    main()
