#!/usr/bin/env python3
"""
Explainability Script for Multimodal Pneumonia Diagnosis Model
Generates Grad-CAM and SHAP explanations for individual samples
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from data_module import make_dataloaders, PneumoniaMultimodalDataset
from explainability_simple import (
    generate_gradcam_simple as generate_gradcam, 
    explain_emr_with_shap_simple as explain_emr_with_shap
)
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, FIG_DIR, LABEL_MAP
from utils_io import load_torch, load_numpy

def find_sample_by_image_id(dataloader, target_image_id):
    """
    Find a specific sample by image ID in the dataloader
    
    Args:
        dataloader: DataLoader to search in
        target_image_id: Target image ID to find
        
    Returns:
        Tuple of (image, emr, label, image_id) or None if not found
    """
    for batch in dataloader:
        for i in range(len(batch['image_id'])):
            if batch['image_id'][i] == target_image_id:
                return (
                    batch['image'][i:i+1],  # Keep batch dimension
                    batch['emr'][i:i+1],    # Keep batch dimension
                    batch['label'][i].item(),
                    batch['image_id'][i]
                )
    return None

def get_random_sample(dataloader):
    """
    Get a random sample from the dataloader
    
    Args:
        dataloader: DataLoader to sample from
        
    Returns:
        Tuple of (image, emr, label, image_id)
    """
    batch = next(iter(dataloader))
    idx = 0  # Take first sample
    return (
        batch['image'][idx:idx+1],  # Keep batch dimension
        batch['emr'][idx:idx+1],    # Keep batch dimension
        batch['label'][idx].item(),
        batch['image_id'][idx]
    )

def load_model_and_data(model_path, split='test', device='auto'):
    """
    Load model and data for explanation
    
    Args:
        model_path: Path to trained model
        split: Data split to use
        device: Device to use
        
    Returns:
        Tuple of (model, dataloader, device)
    """
    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📁 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    print(f"📊 Loading {split} data...")
    dataloaders = make_dataloaders(
        split_csv_dir=str(PROC_DIR),
        emr_arrays_dir=str(PROC_DIR),
        rsna_root=str(RSNA_ROOT),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        cache=False
    )
    
    dataloader = dataloaders[split]
    print(f"   {split.capitalize()} batches: {len(dataloader)}")
    
    return model, dataloader, device

def run_explanation(model, image, emr, true_label, image_id, device, output_dir):
    """
    Run complete explanation pipeline
    
    Args:
        model: Trained model
        image: Image tensor [1, C, H, W]
        emr: EMR tensor [1, F]
        true_label: True label
        image_id: Image ID
        device: Device
        output_dir: Output directory
        
    Returns:
        Dictionary with explanation results
    """
    print(f"\n🔍 Running explanation for image: {image_id}")
    
    # Move tensors to device
    image = image.to(device)
    emr = emr.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits, probabilities = model(image, emr)
        prediction = (probabilities > 0.5).int().item()
        probability = probabilities.item()
    
    # Print prediction
    label_map = {0: "Normal", 1: "Pneumonia"}
    print(f"📊 Prediction: {label_map[prediction]} ({probability:.3f} probability)")
    print(f"🎯 True Label: {label_map[true_label]}")
    print(f"✅ Correct: {'Yes' if prediction == true_label else 'No'}")
    
    # Create output paths
    gradcam_path = output_dir / f"gradcam_{image_id}.png"
    shap_path = output_dir / f"shap_{image_id}.png"
    
    # Generate Grad-CAM
    print(f"\n🎨 Generating Grad-CAM...")
    try:
        # Generate Grad-CAM
        cam = generate_gradcam(
            model=model,
            image_tensor=image,
            class_idx=prediction,
            save_path=gradcam_path
        )
        if cam is not None:
            print(f"   ✅ Grad-CAM generated successfully")
        else:
            print(f"   ❌ Grad-CAM generation failed")
            gradcam_path = None
    except Exception as e:
        print(f"   ❌ Grad-CAM generation failed: {e}")
        gradcam_path = None
    
    # Generate SHAP explanation
    print(f"\n📊 Generating SHAP explanation...")
    try:
        # EMR feature names
        feature_names = ['age', 'fever_temp', 'cough_score', 'WBC_count', 'SpO2', 'sex_M', 'sex_F']
        
        # Generate SHAP explanation
        shap_results = explain_emr_with_shap(
            model=model,
            emr_tensor=emr,
            feature_names=feature_names,
            save_path=shap_path,
            max_features=5
        )
        
        if shap_results is not None:
            print(f"   ✅ SHAP explanation generated successfully")
            print(f"   Top contributing features:")
            for i, idx in enumerate(shap_results['top_features'][:3]):
                print(f"     {i+1}. {feature_names[idx]}: {shap_results['shap_values'][idx]:.3f}")
        else:
            print(f"   ⚠️  SHAP explanation not available")
            shap_path = None
    
    except Exception as e:
        print(f"   ❌ SHAP explanation failed: {e}")
        shap_path = None
    
    return {
        'prediction': prediction,
        'probability': probability,
        'true_label': true_label,
        'gradcam_path': gradcam_path,
        'shap_path': shap_path,
        'image_id': image_id
    }

def main():
    """Main explanation function"""
    parser = argparse.ArgumentParser(description="Explain Multimodal Pneumonia Diagnosis Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_id', type=str, default=None, help='Specific image ID to explain')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                       help='Data split to use')
    parser.add_argument('--output_dir', type=str, default='outputs/figures/explainability/', 
                       help='Output directory for explanations')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Multimodal Model Explainability Analysis")
    print("="*60)
    
    # Load model and data
    model, dataloader, device = load_model_and_data(args.model_path, args.split, args.device)
    
    # Get sample
    if args.image_id:
        print(f"🔍 Looking for image ID: {args.image_id}")
        sample = find_sample_by_image_id(dataloader, args.image_id)
        if sample is None:
            print(f"❌ Image ID '{args.image_id}' not found in {args.split} split")
            print("   Available image IDs (first 10):")
            for i, batch in enumerate(dataloader):
                if i >= 1:  # Only show first batch
                    break
                for j, img_id in enumerate(batch['image_id'][:10]):
                    print(f"     {img_id}")
            return
    else:
        print(f"🎲 Using random sample from {args.split} split")
        sample = get_random_sample(dataloader)
    
    image, emr, true_label, image_id = sample
    print(f"   Selected image: {image_id}")
    
    # Run explanation
    results = run_explanation(model, image, emr, true_label, image_id, device, output_dir)
    
    # Print summary
    if results['gradcam_path'] and results['shap_path']:
        summary = create_explainability_summary(
            results['gradcam_path'],
            results['shap_path'],
            results['prediction'],
            results['true_label'],
            results['probability']
        )
        print(summary)
    else:
        print(f"\n⚠️  Some explanations could not be generated")
        if results['gradcam_path']:
            print(f"   ✅ Grad-CAM: {results['gradcam_path']}")
        if results['shap_path']:
            print(f"   ✅ SHAP: {results['shap_path']}")
    
    print(f"\n🎉 Explanation analysis completed!")

if __name__ == "__main__":
    main()
