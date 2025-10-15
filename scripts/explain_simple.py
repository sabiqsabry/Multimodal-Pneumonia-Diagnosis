#!/usr/bin/env python3
"""
Simple Explainability Script for Multimodal Pneumonia Diagnosis Model
Generates basic explanations for individual samples
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from data_module import make_dataloaders
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, FIG_DIR, LABEL_MAP, IMAGENET_MEAN, IMAGENET_STD
from preprocess_images import denormalize_image

def create_simple_gradcam(model, image_tensor, save_path):
    """Create a simple Grad-CAM visualization"""
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        dummy_emr = torch.zeros(1, 7, device=image_tensor.device)
        logits, probabilities = model(image_tensor, dummy_emr)
        prediction = (probabilities > 0.5).int().item()
        probability = probabilities.item()
    
    # Create a simple heatmap (mock Grad-CAM)
    # In a real implementation, you would compute actual gradients
    heatmap = np.random.rand(224, 224)  # Mock heatmap
    
    # Denormalize image
    image = denormalize_image(image_tensor[0], IMAGENET_MEAN, IMAGENET_STD)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original CXR', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
    axes[1].set_title('Grad-CAM Heatmap (Mock)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
    axes[2].set_title('Grad-CAM Overlay (Mock)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Grad-CAM (mock) saved to: {save_path}")
    return prediction, probability

def create_simple_shap(emr_tensor, feature_names, save_path, max_features=5):
    """Create a simple SHAP-like visualization"""
    # Mock SHAP values (in real implementation, you would compute actual SHAP values)
    feature_importance = np.abs(emr_tensor.cpu().numpy()[0])
    feature_indices = np.argsort(feature_importance)[::-1][:max_features]
    
    # Get top features
    top_features = [feature_names[i] for i in feature_indices]
    top_values = feature_importance[feature_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    colors = ['green' if x > 0 else 'red' for x in top_values]
    bars = ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance (Mock SHAP)', fontsize=12)
    ax.set_title('EMR Feature Contributions (Mock SHAP)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        ax.text(value + 0.01, i, f'{value:.3f}', 
                va='center', ha='left', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='High Importance'),
        Patch(facecolor='red', alpha=0.7, label='Low Importance')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ SHAP (mock) saved to: {save_path}")

def main():
    """Main explanation function"""
    parser = argparse.ArgumentParser(description="Simple Explainability for Multimodal Model")
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
    
    print("🚀 Simple Multimodal Model Explainability Analysis")
    print("="*60)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📁 Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    print(f"📊 Loading {args.split} data...")
    dataloaders = make_dataloaders(
        split_csv_dir=str(PROC_DIR),
        emr_arrays_dir=str(PROC_DIR),
        rsna_root=str(RSNA_ROOT),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        cache=False
    )
    
    dataloader = dataloaders[args.split]
    print(f"   {args.split.capitalize()} batches: {len(dataloader)}")
    
    # Get a sample
    if args.image_id:
        print(f"🔍 Looking for image ID: {args.image_id}")
        sample = None
        for batch in dataloader:
            for i in range(len(batch['image_id'])):
                if batch['image_id'][i] == args.image_id:
                    sample = (
                        batch['image'][i:i+1].to(device),
                        batch['emr'][i:i+1].to(device),
                        batch['label'][i].item(),
                        batch['image_id'][i]
                    )
                    break
            if sample is not None:
                break
        
        if sample is None:
            print(f"❌ Image ID '{args.image_id}' not found in {args.split} split")
            return
    else:
        print(f"🎲 Using random sample from {args.split} split")
        batch = next(iter(dataloader))
        sample = (
            batch['image'][0:1].to(device),
            batch['emr'][0:1].to(device),
            batch['label'][0].item(),
            batch['image_id'][0]
        )
    
    image, emr, true_label, image_id = sample
    print(f"   Selected image: {image_id}")
    
    # Get prediction
    with torch.no_grad():
        logits, probabilities = model(image, emr)
        prediction = (probabilities > 0.5).int().item()
        probability = probabilities.item()
    
    # Print prediction
    label_map = {0: "Normal", 1: "Pneumonia"}
    print(f"\n📊 Prediction: {label_map[prediction]} ({probability:.3f} probability)")
    print(f"🎯 True Label: {label_map[true_label]}")
    print(f"✅ Correct: {'Yes' if prediction == true_label else 'No'}")
    
    # Create output paths
    gradcam_path = output_dir / f"gradcam_{image_id}.png"
    shap_path = output_dir / f"shap_{image_id}.png"
    
    # Generate Grad-CAM (mock)
    print(f"\n🎨 Generating Grad-CAM (mock)...")
    try:
        create_simple_gradcam(model, image, gradcam_path)
    except Exception as e:
        print(f"   ❌ Grad-CAM generation failed: {e}")
        gradcam_path = None
    
    # Generate SHAP (mock)
    print(f"\n📊 Generating SHAP explanation (mock)...")
    try:
        feature_names = ['age', 'fever_temp', 'cough_score', 'WBC_count', 'SpO2', 'sex_M', 'sex_F']
        create_simple_shap(emr, feature_names, shap_path)
    except Exception as e:
        print(f"   ❌ SHAP generation failed: {e}")
        shap_path = None
    
    # Print summary
    print(f"\n🔍 EXPLAINABILITY ANALYSIS SUMMARY")
    print("="*50)
    print(f"📊 Prediction: {label_map[prediction]} ({probability:.3f} probability)")
    print(f"🎯 True Label: {label_map[true_label]}")
    print(f"✅ Correct: {'Yes' if prediction == true_label else 'No'}")
    print(f"\n📁 Generated Files:")
    if gradcam_path:
        print(f"   • Grad-CAM: {gradcam_path}")
    if shap_path:
        print(f"   • SHAP: {shap_path}")
    print(f"\n💡 Note: This is a simplified mock implementation.")
    print(f"   For production use, implement proper Grad-CAM and SHAP algorithms.")
    print("="*50)
    
    print(f"\n🎉 Explanation analysis completed!")

if __name__ == "__main__":
    main()
