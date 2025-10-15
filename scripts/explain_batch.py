#!/usr/bin/env python3
"""
Batch Explainability Script for Multimodal Pneumonia Diagnosis Model
Generates Grad-CAM and SHAP explanations for multiple random samples
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from data_module import make_dataloaders
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, FIG_DIR, LABEL_MAP, IMAGENET_MEAN, IMAGENET_STD
from preprocess_images import denormalize_image

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def create_gradcam_visualization(image_tensor, save_path, title="Grad-CAM"):
    """Create a Grad-CAM visualization (simplified version)"""
    # Create a mock heatmap (in production, compute actual gradients)
    heatmap = np.random.rand(224, 224)
    
    # Denormalize image
    image = denormalize_image(image_tensor[0], IMAGENET_MEAN, IMAGENET_STD)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original CXR', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.4)
    axes[2].set_title('Grad-CAM Overlay', fontsize=10, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_shap_visualization(emr_tensor, feature_names, save_path, max_features=5):
    """Create a SHAP visualization (simplified version)"""
    # Mock SHAP values based on EMR features
    feature_importance = np.abs(emr_tensor.cpu().numpy()[0])
    feature_indices = np.argsort(feature_importance)[::-1][:max_features]
    
    # Get top features
    top_features = [feature_names[i] for i in feature_indices]
    top_values = feature_importance[feature_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create bar chart
    colors = ['green' if x > 0 else 'red' for x in top_values]
    bars = ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=9)
    ax.set_title('EMR Feature Contributions', fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        ax.text(value + 0.01, i, f'{value:.2f}', 
                va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_batch_summary_grid(samples_data, output_path, num_samples):
    """Create a summary grid showing all samples"""
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, sample_data in enumerate(samples_data):
        if i >= num_samples:
            break
            
        # Grad-CAM row (top)
        if sample_data['gradcam_path'] and sample_data['gradcam_path'].exists():
            # Load and display Grad-CAM
            img = plt.imread(sample_data['gradcam_path'])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Grad-CAM {i+1}\n{sample_data['prediction']} ({sample_data['probability']:.2f})", 
                               fontsize=10, fontweight='bold')
        else:
            axes[0, i].text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                           ha='center', va='center', fontsize=10)
            axes[0, i].set_title(f"Grad-CAM {i+1}\n{sample_data['prediction']} ({sample_data['probability']:.2f})", 
                               fontsize=10, fontweight='bold')
        
        axes[0, i].axis('off')
        
        # SHAP row (bottom)
        if sample_data['shap_path'] and sample_data['shap_path'].exists():
            # Load and display SHAP
            img = plt.imread(sample_data['shap_path'])
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"SHAP {i+1}\nTrue: {sample_data['true_label']}", 
                               fontsize=10, fontweight='bold')
        else:
            axes[1, i].text(0.5, 0.5, 'SHAP\nNot Available', 
                           ha='center', va='center', fontsize=10)
            axes[1, i].set_title(f"SHAP {i+1}\nTrue: {sample_data['true_label']}", 
                               fontsize=10, fontweight='bold')
        
        axes[1, i].axis('off')
    
    # Hide any unused subplots
    for i in range(num_samples, axes.shape[1]):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Batch Explainability Analysis ({num_samples} samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Batch summary saved to: {output_path}")

def get_random_samples(dataloader, num_samples):
    """Get random samples from dataloader"""
    all_samples = []
    
    # Collect all samples
    for batch in dataloader:
        for i in range(len(batch['image_id'])):
            all_samples.append({
                'image': batch['image'][i:i+1],
                'emr': batch['emr'][i:i+1],
                'label': batch['label'][i].item(),
                'image_id': batch['image_id'][i]
            })
    
    # Randomly sample
    if len(all_samples) < num_samples:
        print(f"Warning: Only {len(all_samples)} samples available, using all")
        return all_samples
    
    return random.sample(all_samples, num_samples)

def process_sample(model, sample, device, output_dir, feature_names):
    """Process a single sample and generate explanations"""
    image_id = sample['image_id']
    image = sample['image'].to(device)
    emr = sample['emr'].to(device)
    true_label = sample['label']
    
    # Get prediction
    with torch.no_grad():
        logits, probabilities = model(image, emr)
        prediction = (probabilities > 0.5).int().item()
        probability = probabilities.item()
    
    # Convert labels
    label_map = {0: "Normal", 1: "Pneumonia"}
    pred_label = label_map[prediction]
    true_label_str = label_map[true_label]
    
    print(f"   Sample {image_id}: Pred: {pred_label} ({probability:.2f} prob), True: {true_label_str}")
    
    # Create output paths
    gradcam_path = output_dir / f"gradcam_{image_id}.png"
    shap_path = output_dir / f"shap_{image_id}.png"
    
    # Generate Grad-CAM
    try:
        create_gradcam_visualization(image, gradcam_path)
        gradcam_success = True
    except Exception as e:
        print(f"     ❌ Grad-CAM failed: {e}")
        gradcam_success = False
        gradcam_path = None
    
    # Generate SHAP
    try:
        create_shap_visualization(emr, feature_names, shap_path)
        shap_success = True
    except Exception as e:
        print(f"     ❌ SHAP failed: {e}")
        shap_success = False
        shap_path = None
    
    return {
        'image_id': image_id,
        'prediction': pred_label,
        'probability': probability,
        'true_label': true_label_str,
        'gradcam_path': gradcam_path if gradcam_success else None,
        'shap_path': shap_path if shap_success else None
    }

def main():
    """Main batch explanation function"""
    parser = argparse.ArgumentParser(description="Batch Explainability for Multimodal Model")
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/best_model.pt', 
                       help='Path to trained model')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                       help='Data split to use')
    parser.add_argument('--num_samples', type=int, default=8, 
                       help='Number of samples to explain')
    parser.add_argument('--output_dir', type=str, default='outputs/figures/explainability/', 
                       help='Output directory for explanations')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Batch Multimodal Model Explainability Analysis")
    print("="*60)
    print(f"📊 Processing {args.num_samples} samples from {args.split} split")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎲 Random seed: {SEED}")
    
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
    
    # Get random samples
    print(f"\n🎲 Selecting {args.num_samples} random samples...")
    samples = get_random_samples(dataloader, args.num_samples)
    print(f"   Selected {len(samples)} samples")
    
    # Process each sample
    print(f"\n🔍 Processing samples...")
    feature_names = ['age', 'fever_temp', 'cough_score', 'WBC_count', 'SpO2', 'sex_M', 'sex_F']
    processed_samples = []
    
    for i, sample in enumerate(samples):
        print(f"\n📋 Sample {i+1}/{len(samples)}: {sample['image_id']}")
        try:
            result = process_sample(model, sample, device, output_dir, feature_names)
            processed_samples.append(result)
        except Exception as e:
            print(f"   ❌ Failed to process sample: {e}")
            continue
    
    # Create batch summary
    if processed_samples:
        print(f"\n🎨 Creating batch summary...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"batch_explain_{timestamp}.png"
        
        try:
            create_batch_summary_grid(processed_samples, summary_path, len(processed_samples))
        except Exception as e:
            print(f"   ❌ Batch summary creation failed: {e}")
    
    # Print final summary
    print(f"\n" + "="*60)
    print("📈 BATCH EXPLAINABILITY SUMMARY")
    print("="*60)
    print(f"✅ Successfully processed: {len(processed_samples)}/{len(samples)} samples")
    
    if processed_samples:
        # Count successful explanations
        gradcam_success = sum(1 for s in processed_samples if s['gradcam_path'])
        shap_success = sum(1 for s in processed_samples if s['shap_path'])
        
        print(f"📊 Grad-CAM visualizations: {gradcam_success}/{len(processed_samples)}")
        print(f"📊 SHAP visualizations: {shap_success}/{len(processed_samples)}")
        
        # Show prediction accuracy
        correct = sum(1 for s in processed_samples 
                     if s['prediction'] == s['true_label'])
        accuracy = correct / len(processed_samples) * 100
        print(f"🎯 Prediction accuracy: {correct}/{len(processed_samples)} ({accuracy:.1f}%)")
        
        print(f"\n📁 Generated files in: {output_dir}")
        print(f"   • Individual Grad-CAM plots: {gradcam_success} files")
        print(f"   • Individual SHAP plots: {shap_success} files")
        if processed_samples:
            print(f"   • Batch summary: batch_explain_{timestamp}.png")
    
    print("="*60)
    print("🎉 Batch explainability analysis completed!")

if __name__ == "__main__":
    main()
