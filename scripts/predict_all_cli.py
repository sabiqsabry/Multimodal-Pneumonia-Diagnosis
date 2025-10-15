#!/usr/bin/env python3
"""
CLI version of predict_all.py for getting predictions from all models
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import joblib

# Import all model architectures
from models import create_model
from models_cxr_only import create_cxr_only_model
from models_real_emr import create_real_emr_model
from models_cxr_real_emr import create_cxr_real_emr_fusion_model

# Import preprocessing utilities
from preprocess_images import get_transforms, ensure_rgb
from preprocess_emr import prepare_emr
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CHECKPOINTS_DIR
from utils_io import load_json, load_torch

class ModelPredictorCLI:
    """CLI version of predictor that loads and runs all 5 models"""
    
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
        model_configs = [
            {
                'name': 'CXR-only',
                'path': CHECKPOINTS_DIR / 'cxr_only_best.pt',
                'creator': create_cxr_only_model,
                'scaler': 'emr_scaler.joblib',
                'label_map': 'label_map.json'
            },
            {
                'name': 'Real EMR-only',
                'path': CHECKPOINTS_DIR / 'real_emr_best.pt',
                'creator': create_real_emr_model,
                'scaler': 'real_emr/real_emr_scaler.joblib',
                'label_map': 'real_emr/real_emr_label_map.json'
            },
            {
                'name': 'CXR + Synthetic EMR',
                'path': CHECKPOINTS_DIR / 'best_model.pt',
                'creator': create_model,
                'scaler': 'emr_scaler.joblib',
                'label_map': 'label_map.json'
            },
            {
                'name': 'CXR + Real EMR Fusion (Unweighted)',
                'path': CHECKPOINTS_DIR / 'cxr_real_emr_fusion_best.pt',
                'creator': create_cxr_real_emr_fusion_model,
                'scaler': 'real_emr/real_emr_scaler.joblib',
                'label_map': 'real_emr/real_emr_label_map.json'
            },
            {
                'name': 'CXR + Real EMR Fusion (Weighted)',
                'path': CHECKPOINTS_DIR / 'cxr_real_emr_fusion_weighted.pt',
                'creator': create_cxr_real_emr_fusion_model,
                'scaler': 'real_emr/real_emr_scaler.joblib',
                'label_map': 'real_emr/real_emr_label_map.json'
            }
        ]
        
        for config in model_configs:
            print(f"   📁 Loading {config['name']} model...")
            try:
                # Load model
                checkpoint = load_torch(config['path'])
                model = config['creator']()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[config['name']] = model
                
                # Load scaler
                scaler_path = Path('data/processed') / config['scaler']
                self.scalers[config['name']] = joblib.load(scaler_path)
                
                # Load label map
                label_map_path = Path('data/processed') / config['label_map']
                self.label_maps[config['name']] = load_json(label_map_path)
                
                print(f"   ✅ {config['name']} model loaded")
            except Exception as e:
                print(f"   ❌ Failed to load {config['name']}: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess input image"""
        image = Image.open(image_path)
        image = ensure_rgb(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def preprocess_emr(self, age, sex, fever_temp, cough_score, wbc_count, spo2, model_name):
        """Preprocess EMR data for specific model"""
        scaler = self.scalers[model_name]
        
        # Prepare EMR features
        emr_features = prepare_emr(age, sex, fever_temp, cough_score, wbc_count, spo2)
        emr_scaled = scaler.transform([emr_features])
        emr_tensor = torch.tensor(emr_scaled, dtype=torch.float32).to(self.device)
        
        return emr_tensor
    
    def predict_all_models(self, image_path, age, sex, fever_temp, cough_score, wbc_count, spo2):
        """Run prediction on all models"""
        print(f"\n🔮 Running predictions...")
        print(f"📊 Input EMR: Age={age}, Sex={sex}, Fever={fever_temp}°C, Cough={cough_score}, WBC={wbc_count}, SpO2={spo2}%")
        
        # Preprocess image once
        image_tensor = self.preprocess_image(image_path)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"   🔍 Running {model_name}...")
                
                if 'EMR-only' in model_name:
                    # EMR-only model
                    emr_tensor = self.preprocess_emr(age, sex, fever_temp, cough_score, wbc_count, spo2, model_name)
                    with torch.no_grad():
                        outputs = model(emr_tensor)
                        if hasattr(model, 'classifier'):
                            outputs = model.classifier(emr_tensor)
                        prob = torch.sigmoid(outputs).item()
                else:
                    # CXR or fusion models
                    emr_tensor = self.preprocess_emr(age, sex, fever_temp, cough_score, wbc_count, spo2, model_name)
                    with torch.no_grad():
                        if 'CXR-only' in model_name:
                            prob = torch.sigmoid(model(image_tensor)).item()
                        else:
                            prob = torch.sigmoid(model(image_tensor, emr_tensor)).item()
                
                # Map probability to label
                label_map = self.label_maps[model_name]
                prediction = 'Pneumonia' if prob > 0.5 else 'Normal'
                
                predictions[model_name] = {
                    'probability': prob,
                    'prediction': prediction,
                    'confidence': max(prob, 1-prob)
                }
                
                print(f"      → {prediction} ({prob:.3f})")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                predictions[model_name] = None
        
        return predictions
    
    def display_results(self, predictions):
        """Display all prediction results"""
        print("\n" + "="*60)
        print("📊 ALL MODEL PREDICTIONS")
        print("="*60)
        
        for model_name, result in predictions.items():
            if result is None:
                print(f"{model_name:35} | ❌ FAILED")
                continue
                
            pred = result['prediction']
            prob = result['probability']
            conf = result['confidence']
            
            print(f"{model_name:35} | {pred:10} | {prob:.3f} | {conf:.3f}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Get predictions from all trained models')
    parser.add_argument('--image_path', required=True, help='Path to chest X-ray image')
    parser.add_argument('--age', type=int, required=True, help='Patient age')
    parser.add_argument('--sex', choices=['M', 'F'], required=True, help='Patient sex')
    parser.add_argument('--fever_temp', type=float, required=True, help='Fever temperature (°C)')
    parser.add_argument('--cough_score', type=int, required=True, help='Cough score (0-5)')
    parser.add_argument('--WBC_count', type=int, required=True, help='WBC count (/µL)')
    parser.add_argument('--SpO2', type=int, required=True, help='SpO2 (%)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"🔧 Using device: {device}")
    
    # Initialize predictor
    predictor = ModelPredictorCLI(device)
    
    # Run predictions
    predictions = predictor.predict_all_models(
        args.image_path, args.age, args.sex, args.fever_temp, 
        args.cough_score, args.WBC_count, args.SpO2
    )
    
    if predictions is None:
        print("❌ Prediction failed.")
        return
    
    # Display results
    predictor.display_results(predictions)
    
    print("\n✅ All predictions completed!")

if __name__ == "__main__":
    main()
