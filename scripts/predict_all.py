#!/usr/bin/env python3
"""
Main Inference Demo Script for Multi-Modal Pneumonia Diagnosis
Loads all 5 trained models and provides predictions from each
"""

import argparse
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import joblib

# Import all model architectures
from models import create_model
from models_cxr_only import create_cxr_only_model
from models_real_emr import create_real_emr_model
from models_cxr_real_emr import create_cxr_real_emr_fusion_model

# Import preprocessing utilities
from preprocess_images import get_transforms, ensure_rgb
from preprocess_emr import prepare_emr
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CHECKPOINTS_DIR, LOGS_DIR
from utils_io import ensure_dir, load_json, load_torch

class ModelPredictor:
    """Main predictor class that loads and runs all 5 models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.scalers = {}
        self.label_maps = {}
        self.transform = get_transforms('test', IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)
        
        print("🚀 Loading all trained models...")
        self._load_all_models()
        print("✅ All models loaded successfully!")
    
    def _load_all_models(self):
        """Load all 5 trained models"""
        
        # 1. CXR-only model
        try:
            print("   📁 Loading CXR-only model...")
            cxr_checkpoint = torch.load(CHECKPOINTS_DIR / "cxr_only_best.pt", map_location=self.device, weights_only=False)
            self.models['cxr_only'] = create_cxr_only_model()
            self.models['cxr_only'].load_state_dict(cxr_checkpoint['model_state_dict'])
            self.models['cxr_only'].to(self.device)
            self.models['cxr_only'].eval()
            print("   ✅ CXR-only model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load CXR-only model: {e}")
            self.models['cxr_only'] = None
        
        # 2. Real EMR-only model
        try:
            print("   📁 Loading Real EMR-only model...")
            real_emr_checkpoint = torch.load(CHECKPOINTS_DIR / "real_emr_best.pt", map_location=self.device, weights_only=False)
            self.models['real_emr'] = create_real_emr_model(input_dim=real_emr_checkpoint.get('input_dim', 13))
            self.models['real_emr'].load_state_dict(real_emr_checkpoint['model_state_dict'])
            self.models['real_emr'].to(self.device)
            self.models['real_emr'].eval()
            
            # Load real EMR scaler
            self.scalers['real_emr'] = joblib.load("data/processed/real_emr/real_emr_scaler.joblib")
            self.label_maps['real_emr'] = load_json("data/processed/real_emr/real_emr_label_map.json")
            print("   ✅ Real EMR-only model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load Real EMR-only model: {e}")
            self.models['real_emr'] = None
        
        # 3. CXR + Synthetic EMR Fusion model
        try:
            print("   📁 Loading CXR + Synthetic EMR Fusion model...")
            synthetic_checkpoint = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location=self.device, weights_only=False)
            self.models['cxr_synthetic_emr'] = create_model()
            self.models['cxr_synthetic_emr'].load_state_dict(synthetic_checkpoint['model_state_dict'])
            self.models['cxr_synthetic_emr'].to(self.device)
            self.models['cxr_synthetic_emr'].eval()
            
            # Load synthetic EMR scaler
            self.scalers['synthetic_emr'] = joblib.load("data/processed/emr_scaler.joblib")
            self.label_maps['synthetic_emr'] = load_json("data/processed/label_map.json")
            print("   ✅ CXR + Synthetic EMR Fusion model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load CXR + Synthetic EMR Fusion model: {e}")
            self.models['cxr_synthetic_emr'] = None
        
        # 4. CXR + Real EMR Fusion (Unweighted) model
        try:
            print("   📁 Loading CXR + Real EMR Fusion (Unweighted) model...")
            unweighted_checkpoint = torch.load(CHECKPOINTS_DIR / "cxr_real_emr_fusion_best.pt", map_location=self.device, weights_only=False)
            self.models['cxr_real_emr_unweighted'] = create_cxr_real_emr_fusion_model(
                emr_input_dim=unweighted_checkpoint.get('emr_input_dim', 13)
            )
            self.models['cxr_real_emr_unweighted'].load_state_dict(unweighted_checkpoint['model_state_dict'])
            self.models['cxr_real_emr_unweighted'].to(self.device)
            self.models['cxr_real_emr_unweighted'].eval()
            print("   ✅ CXR + Real EMR Fusion (Unweighted) model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load CXR + Real EMR Fusion (Unweighted) model: {e}")
            self.models['cxr_real_emr_unweighted'] = None
        
        # 5. CXR + Real EMR Fusion (Weighted) model
        try:
            print("   📁 Loading CXR + Real EMR Fusion (Weighted) model...")
            weighted_checkpoint = torch.load(CHECKPOINTS_DIR / "cxr_real_emr_fusion_weighted.pt", map_location=self.device, weights_only=False)
            self.models['cxr_real_emr_weighted'] = create_cxr_real_emr_fusion_model(
                emr_input_dim=weighted_checkpoint.get('emr_input_dim', 13)
            )
            self.models['cxr_real_emr_weighted'].load_state_dict(weighted_checkpoint['model_state_dict'])
            self.models['cxr_real_emr_weighted'].to(self.device)
            self.models['cxr_real_emr_weighted'].eval()
            print("   ✅ CXR + Real EMR Fusion (Weighted) model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load CXR + Real EMR Fusion (Weighted) model: {e}")
            self.models['cxr_real_emr_weighted'] = None
    
    def preprocess_image(self, image_path):
        """Preprocess chest X-ray image"""
        try:
            image = Image.open(image_path)
            image = ensure_rgb(image)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def preprocess_synthetic_emr(self, age, sex, fever_temp, cough_score, wbc_count, spo2):
        """Preprocess EMR data for synthetic EMR models"""
        # Create EMR data in the same format as training
        emr_data = {
            'age': age,
            'sex': sex.upper(),
            'fever_temp': fever_temp,
            'cough_score': cough_score,
            'wbc_count': wbc_count,
            'spO2': spo2
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([emr_data])
        
        # Apply preprocessing (same as training)
        # One-hot encode sex
        df['sex_F'] = (df['sex'] == 'F').astype(int)
        df['sex_M'] = (df['sex'] == 'M').astype(int)
        
        # Select features in the same order as training
        feature_columns = ['age', 'fever_temp', 'cough_score', 'wbc_count', 'spO2', 'sex_M', 'sex_F']
        emr_features = df[feature_columns].values.astype(np.float32)
        
        # Apply scaler only to continuous features (first 5 columns)
        if 'synthetic_emr' in self.scalers:
            continuous_features = emr_features[:, :5]  # First 5 features
            continuous_scaled = self.scalers['synthetic_emr'].transform(continuous_features)
            emr_features[:, :5] = continuous_scaled
        
        return torch.tensor(emr_features, dtype=torch.float32).to(self.device)
    
    def preprocess_real_emr(self, age, sex, fever_temp, cough_score, wbc_count, spo2):
        """Preprocess EMR data for real EMR models"""
        # Create EMR data in the same format as real EMR training
        emr_data = {
            'age': age,
            'gender': sex.upper(),
            'fever': 'High' if fever_temp > 38.5 else 'Low' if fever_temp > 37.5 else 'None',
            'cough': 'Dry' if cough_score <= 2 else 'Bloody' if cough_score >= 4 else 'Wet',
            'shortness_of_breath': 'Severe' if spo2 < 90 else 'Mild' if spo2 < 95 else 'None',
            'chest_pain': 'Mild' if cough_score >= 3 else 'None',
            'fatigue': 'Moderate' if fever_temp > 38 else 'Mild' if fever_temp > 37 else 'None',
            'confusion': 'No',
            'oxygen_saturation': spo2,
            'crackles': 'Yes' if cough_score >= 3 else 'No',
            'wbc_count': wbc_count,
            'sputum_color': 'Clear' if cough_score <= 1 else 'Yellow' if cough_score <= 3 else 'Bloody',
            'temperature': fever_temp
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([emr_data])
        
        # Clean column names (same as preprocessing)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle missing values and convert to numeric
        continuous_features = ['age', 'oxygen_saturation', 'wbc_count', 'temperature']
        for feature in continuous_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Encode categorical features (simplified mapping)
        categorical_mappings = {
            'gender': {'M': 1, 'F': 0},
            'fever': {'High': 2, 'Low': 1, 'None': 0},
            'cough': {'Bloody': 2, 'Wet': 1, 'Dry': 0},
            'shortness_of_breath': {'Severe': 2, 'Mild': 1, 'None': 0},
            'chest_pain': {'Mild': 1, 'None': 0},
            'fatigue': {'Moderate': 2, 'Mild': 1, 'None': 0},
            'confusion': {'No': 0, 'Yes': 1},
            'crackles': {'Yes': 1, 'No': 0},
            'sputum_color': {'Bloody': 3, 'Yellow': 2, 'Clear': 1, 'None': 0}
        }
        
        for feature, mapping in categorical_mappings.items():
            if feature in df.columns:
                df[feature] = df[feature].map(mapping).fillna(0)
        
        # Select features in the same order as real EMR training
        feature_columns = ['age', 'oxygen_saturation', 'wbc_count', 'temperature', 'gender', 
                          'cough', 'fever', 'shortness_of_breath', 'chest_pain', 'fatigue', 
                          'confusion', 'crackles', 'sputum_color']
        
        emr_features = df[feature_columns].values.astype(np.float32)
        
        # Apply scaler only to continuous features (first 4 columns)
        if 'real_emr' in self.scalers:
            continuous_features = emr_features[:, :4]  # First 4 features are continuous
            continuous_scaled = self.scalers['real_emr'].transform(continuous_features)
            emr_features[:, :4] = continuous_scaled
        
        return torch.tensor(emr_features, dtype=torch.float32).to(self.device)
    
    def predict_all_models(self, image_path, age, sex, fever_temp, cough_score, wbc_count, spo2):
        """Run prediction on all 5 models"""
        print("\n🔍 Running predictions on all models...")
        
        # Preprocess inputs
        try:
            image_tensor = self.preprocess_image(image_path)
            synthetic_emr_tensor = self.preprocess_synthetic_emr(age, sex, fever_temp, cough_score, wbc_count, spo2)
            real_emr_tensor = self.preprocess_real_emr(age, sex, fever_temp, cough_score, wbc_count, spo2)
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None
        
        predictions = {}
        
        # 1. CXR-only prediction
        if self.models['cxr_only'] is not None:
            try:
                with torch.no_grad():
                    logits, probabilities = self.models['cxr_only'](image_tensor)
                    prob = probabilities.item()
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    predictions['cxr_only'] = (label, prob)
            except Exception as e:
                print(f"   ❌ CXR-only prediction failed: {e}")
                predictions['cxr_only'] = ("Error", 0.0)
        else:
            predictions['cxr_only'] = ("Not Available", 0.0)
        
        # 2. Real EMR-only prediction
        if self.models['real_emr'] is not None:
            try:
                with torch.no_grad():
                    logits, probabilities = self.models['real_emr'](real_emr_tensor)
                    prob = probabilities.item()
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    predictions['real_emr'] = (label, prob)
            except Exception as e:
                print(f"   ❌ Real EMR-only prediction failed: {e}")
                predictions['real_emr'] = ("Error", 0.0)
        else:
            predictions['real_emr'] = ("Not Available", 0.0)
        
        # 3. CXR + Synthetic EMR Fusion prediction
        if self.models['cxr_synthetic_emr'] is not None:
            try:
                with torch.no_grad():
                    logits, probabilities = self.models['cxr_synthetic_emr'](image_tensor, synthetic_emr_tensor)
                    prob = probabilities.item()
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    predictions['cxr_synthetic_emr'] = (label, prob)
            except Exception as e:
                print(f"   ❌ CXR + Synthetic EMR prediction failed: {e}")
                predictions['cxr_synthetic_emr'] = ("Error", 0.0)
        else:
            predictions['cxr_synthetic_emr'] = ("Not Available", 0.0)
        
        # 4. CXR + Real EMR Fusion (Unweighted) prediction
        if self.models['cxr_real_emr_unweighted'] is not None:
            try:
                with torch.no_grad():
                    logits, probabilities = self.models['cxr_real_emr_unweighted'](image_tensor, real_emr_tensor)
                    prob = probabilities.item()
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    predictions['cxr_real_emr_unweighted'] = (label, prob)
            except Exception as e:
                print(f"   ❌ CXR + Real EMR Fusion (Unweighted) prediction failed: {e}")
                predictions['cxr_real_emr_unweighted'] = ("Error", 0.0)
        else:
            predictions['cxr_real_emr_unweighted'] = ("Not Available", 0.0)
        
        # 5. CXR + Real EMR Fusion (Weighted) prediction
        if self.models['cxr_real_emr_weighted'] is not None:
            try:
                with torch.no_grad():
                    logits, probabilities = self.models['cxr_real_emr_weighted'](image_tensor, real_emr_tensor)
                    prob = probabilities.item()
                    label = "Pneumonia" if prob > 0.5 else "Normal"
                    predictions['cxr_real_emr_weighted'] = (label, prob)
            except Exception as e:
                print(f"   ❌ CXR + Real EMR Fusion (Weighted) prediction failed: {e}")
                predictions['cxr_real_emr_weighted'] = ("Error", 0.0)
        else:
            predictions['cxr_real_emr_weighted'] = ("Not Available", 0.0)
        
        return predictions
    
    def display_results(self, predictions, age, sex, fever_temp, cough_score, wbc_count, spo2):
        """Display prediction results in a formatted block"""
        print("\n" + "="*50)
        print("Model Predictions")
        print("-" * 30)
        print(f"CXR-only: {predictions['cxr_only'][0]} ({predictions['cxr_only'][1]:.2f})")
        print(f"EMR-only (Real): {predictions['real_emr'][0]} ({predictions['real_emr'][1]:.2f})")
        print(f"CXR+Synthetic EMR: {predictions['cxr_synthetic_emr'][0]} ({predictions['cxr_synthetic_emr'][1]:.2f})")
        print(f"CXR+Real EMR Fusion (Unweighted): {predictions['cxr_real_emr_unweighted'][0]} ({predictions['cxr_real_emr_unweighted'][1]:.2f})")
        print(f"CXR+Real EMR Fusion (Weighted): {predictions['cxr_real_emr_weighted'][0]} ({predictions['cxr_real_emr_weighted'][1]:.2f})")
        print("-" * 30)
        print("Note: Weighted fusion model shows better sensitivity/specificity balance.")
        print("="*50)
    
    def save_to_log(self, predictions, age, sex, fever_temp, cough_score, wbc_count, spo2, image_path):
        """Save prediction results to main report log"""
        try:
            log_path = LOGS_DIR / "main_report.txt"
            ensure_dir(log_path.parent)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = f"""
========================================
Multi-Modal Pneumonia Diagnosis Report
Date: {timestamp}
Image: {Path(image_path).name}

Patient EMR:
  Age: {age}
  Sex: {sex}
  Fever Temperature: {fever_temp} °C
  Cough Score: {cough_score}
  WBC Count: {wbc_count} /µL
  SpO2: {spo2} %

Model Predictions:
  CXR-only: {predictions['cxr_only'][0]} ({predictions['cxr_only'][1]:.3f})
  EMR-only (Real): {predictions['real_emr'][0]} ({predictions['real_emr'][1]:.3f})
  CXR+Synthetic EMR: {predictions['cxr_synthetic_emr'][0]} ({predictions['cxr_synthetic_emr'][1]:.3f})
  CXR+Real EMR Fusion (Unweighted): {predictions['cxr_real_emr_unweighted'][0]} ({predictions['cxr_real_emr_unweighted'][1]:.3f})
  CXR+Real EMR Fusion (Weighted): {predictions['cxr_real_emr_weighted'][0]} ({predictions['cxr_real_emr_weighted'][1]:.3f})
========================================
"""
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            return True
        except Exception as e:
            print(f"⚠️ Warning: Failed to save log: {e}")
            return False

def get_user_input(prompt, input_type=str, default=None, choices=None, min_val=None, max_val=None):
    """Get user input with validation and re-prompting"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if input_type == int:
                value = int(user_input)
                if min_val is not None and value < min_val:
                    print(f"   ❌ Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"   ❌ Value must be <= {max_val}")
                    continue
                return value
            elif input_type == float:
                value = float(user_input)
                if min_val is not None and value < min_val:
                    print(f"   ❌ Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"   ❌ Value must be <= {max_val}")
                    continue
                return value
            elif choices is not None:
                if user_input.upper() not in [c.upper() for c in choices]:
                    print(f"   ❌ Please choose from: {', '.join(choices)}")
                    continue
                return user_input.upper()
            else:
                return user_input
        except ValueError:
            print(f"   ❌ Invalid input. Please enter a valid {input_type.__name__}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def main():
    """Main interactive prediction function"""
    print("🏥 Multi-Modal Pneumonia Diagnosis - Main Inference Demo")
    print("="*60)
    print("This demo will run predictions on all 5 trained models:")
    print("  • CXR-only")
    print("  • Real EMR-only") 
    print("  • CXR + Synthetic EMR Fusion")
    print("  • CXR + Real EMR Fusion (Unweighted)")
    print("  • CXR + Real EMR Fusion (Weighted)")
    print("="*60)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🔧 Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🔧 Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("🔧 Using CPU")
    
    # Initialize predictor
    predictor = ModelPredictor(device)
    
    # Get user inputs
    print("\n📋 Please provide patient information:")
    print("-" * 40)
    
    image_path = get_user_input("Chest X-ray image path (PNG/JPG)")
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    age = get_user_input("Age (years)", input_type=int, min_val=1, max_val=120)
    sex = get_user_input("Sex", choices=['M', 'F'])
    fever_temp = get_user_input("Fever temperature (°C)", input_type=float, min_val=35.0, max_val=45.0)
    cough_score = get_user_input("Cough score (0-5)", input_type=int, min_val=0, max_val=5)
    wbc_count = get_user_input("WBC count (/µL)", input_type=int, min_val=1000, max_val=50000)
    spo2 = get_user_input("SpO2 (%)", input_type=int, min_val=70, max_val=100)
    
    # Run predictions
    predictions = predictor.predict_all_models(
        image_path, age, sex, fever_temp, cough_score, wbc_count, spo2
    )
    
    if predictions is None:
        print("❌ Prediction failed. Please check your inputs and try again.")
        return
    
    # Display results
    predictor.display_results(predictions, age, sex, fever_temp, cough_score, wbc_count, spo2)
    
    # Ask about saving to log
    save_log = get_user_input("Save results to outputs/logs/main_report.txt?", choices=['Y', 'N'], default='N')
    if save_log == 'Y':
        success = predictor.save_to_log(predictions, age, sex, fever_temp, cough_score, wbc_count, spo2, image_path)
        if success:
            print("✅ Results saved to outputs/logs/main_report.txt")
        else:
            print("⚠️ Failed to save results to log file")
    else:
        print("📝 Results not saved to log file")
    
    print("\n🎉 Demo completed successfully!")
    print("Thank you for using the Multi-Modal Pneumonia Diagnosis system!")

if __name__ == "__main__":
    main()
