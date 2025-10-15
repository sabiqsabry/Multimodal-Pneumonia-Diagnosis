#!/usr/bin/env python3
"""
Evaluation Script for CXR + Real EMR Fusion Weighted Model
Evaluates trained weighted fusion model on test set with threshold tuning
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from tqdm import tqdm

from models_cxr_real_emr import create_cxr_real_emr_fusion_model
from data_module_cxr_real_emr import make_cxr_real_emr_dataloaders
from config import CHECKPOINTS_DIR, FIG_DIR, LOGS_DIR, NUM_WORKERS
from utils_io import ensure_dir, load_torch, save_json

class ModelEvaluator:
    """Comprehensive model evaluation with visualizations"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, predictions, targets, save_path, threshold=0.5):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title(f'CXR + Real EMR Fusion Weighted - Confusion Matrix (Threshold={threshold})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, targets, probabilities, save_path):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('CXR + Real EMR Fusion Weighted - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, targets, probabilities, save_path):
        """Plot and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('CXR + Real EMR Fusion Weighted - Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_bar_chart(self, metrics, save_path, threshold=0.5):
        """Plot and save bar chart of key metrics"""
        metric_names = ['Accuracy', 'AUC', 'F1 Score', 'Precision', 'Recall', 
                       'Sensitivity', 'Specificity']
        metric_values = [
            metrics['accuracy'], metrics['auc'], metrics['f1_score'],
            metrics['precision'], metrics['recall'], metrics['sensitivity'], 
            metrics['specificity']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', 
                                                         '#d62728', '#9467bd', '#8c564b', '#e377c2'])
        plt.title(f'CXR + Real EMR Fusion Weighted - Performance Metrics (Threshold={threshold})', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_combined_results_summary(self, predictions, targets, probabilities, metrics, save_path, threshold=0.5):
        """Create combined results summary figure for thesis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'CXR + Real EMR Fusion Weighted - Results Summary (Threshold={threshold})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix (top-left)
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'])
        axes[0,0].set_title('Confusion Matrix', fontweight='bold')
        axes[0,0].set_xlabel('Predicted Label')
        axes[0,0].set_ylabel('True Label')
        
        # 2. ROC Curve (top-right)
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                      label='Random Classifier')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Metrics Bar Chart (bottom-left)
        metric_names = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
        metric_values = [
            metrics['accuracy'], metrics['auc'], metrics['f1_score'],
            metrics['precision'], metrics['recall']
        ]
        bars = axes[1,0].bar(metric_names, metric_values, 
                            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[1,0].set_title('Key Performance Metrics', fontweight='bold')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary Statistics (bottom-right)
        axes[1,1].axis('off')
        summary_text = f"""
        Model: CXR + Real EMR Fusion Weighted
        Test Samples: {len(targets)}
        Threshold: {threshold}
        
        Performance Metrics:
        • Accuracy: {metrics['accuracy']:.3f}
        • AUC: {metrics['auc']:.3f}
        • F1 Score: {metrics['f1_score']:.3f}
        • Precision: {metrics['precision']:.3f}
        • Recall: {metrics['recall']:.3f}
        • Sensitivity: {metrics['sensitivity']:.3f}
        • Specificity: {metrics['specificity']:.3f}
        
        Confusion Matrix:
        • True Positives: {cm[1,1]}
        • True Negatives: {cm[0,0]}
        • False Positives: {cm[0,1]}
        • False Negatives: {cm[1,0]}
        """
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_classification_report(self, predictions, targets, save_path):
        """Generate and save classification report"""
        report = classification_report(targets, predictions, 
                                     target_names=['Normal', 'Pneumonia'],
                                     output_dict=True)
        
        # Save as JSON
        save_json(report, save_path)
        
        # Also save as text
        text_report = classification_report(targets, predictions, 
                                          target_names=['Normal', 'Pneumonia'])
        with open(save_path.with_suffix('.txt'), 'w') as f:
            f.write(text_report)

def evaluate_model_at_threshold(model_path, test_loader, device, output_dir, threshold=0.5):
    """Evaluate model on test set at specific threshold"""
    print(f"🚀 Starting CXR + Real EMR Fusion Weighted Model Evaluation (Threshold={threshold})...")
    print("="*60)
    
    # Load model
    print(f"📁 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Create model
    model = create_cxr_real_emr_fusion_model(emr_input_dim=checkpoint['emr_input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded from epoch: {checkpoint['epoch']}")
    print(f"   Model info: {checkpoint['model_info']}")
    if 'pos_weight' in checkpoint:
        print(f"   Pos weight used in training: {checkpoint['pos_weight']:.3f}")
    
    # Evaluate on test set
    print("\n🔍 Evaluating on test set...")
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(device)
            emr_features = batch['emr_features'].to(device)
            targets = batch['label'].to(device)
            
            # Forward pass
            logits, probabilities = model(images, emr_features)
            
            # Convert to numpy
            targets_np = targets.squeeze().cpu().numpy()
            probabilities_np = probabilities.squeeze().cpu().numpy()
            
            all_targets.extend(targets_np)
            all_probabilities.extend(probabilities_np)
    
    # Convert to numpy arrays
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    # Apply threshold
    predictions = (probabilities > threshold).astype(int)
    
    print(f"   Evaluated {len(targets)} test samples")
    print(f"   Applied threshold: {threshold}")
    
    # Compute metrics
    print("\n📊 Computing evaluation metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, probabilities)
    f1 = f1_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"Threshold:      {threshold}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"AUC:            {auc:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"Sensitivity:    {sensitivity:.4f}")
    print(f"Specificity:    {specificity:.4f}")
    print(f"PPV:            {ppv:.4f}")
    print(f"NPV:            {npv:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    
    # Create visualizations
    print("\n🎨 Generating visualizations...")
    evaluator = ModelEvaluator(output_dir)
    
    # Individual plots
    evaluator.plot_confusion_matrix(predictions, targets, 
                                   output_dir / f"cxr_real_emr_weighted_confusion_matrix_th{threshold}.png", 
                                   threshold=threshold)
    print(f"   ✅ Confusion matrix saved to: {output_dir / f'cxr_real_emr_weighted_confusion_matrix_th{threshold}.png'}")
    
    evaluator.plot_roc_curve(targets, probabilities, output_dir / "cxr_real_emr_weighted_roc_curve.png")
    print(f"   ✅ ROC curve saved to: {output_dir / 'cxr_real_emr_weighted_roc_curve.png'}")
    
    evaluator.plot_precision_recall_curve(targets, probabilities, output_dir / "cxr_real_emr_weighted_pr_curve.png")
    print(f"   ✅ Precision-Recall curve saved to: {output_dir / 'cxr_real_emr_weighted_pr_curve.png'}")
    
    evaluator.plot_metrics_bar_chart(metrics, 
                                   output_dir / f"cxr_real_emr_weighted_metrics_bar_th{threshold}.png", 
                                   threshold=threshold)
    print(f"   ✅ Metrics bar chart saved to: {output_dir / f'cxr_real_emr_weighted_metrics_bar_th{threshold}.png'}")
    
    # Combined summary
    evaluator.create_combined_results_summary(predictions, targets, probabilities, metrics, 
                                            output_dir / f"cxr_real_emr_weighted_results_summary_th{threshold}.png",
                                            threshold=threshold)
    print(f"   ✅ Combined results summary saved to: {output_dir / f'cxr_real_emr_weighted_results_summary_th{threshold}.png'}")
    
    # Classification report
    evaluator.generate_classification_report(predictions, targets, 
                                           output_dir / f"cxr_real_emr_weighted_classification_report_th{threshold}.json")
    print(f"   ✅ Classification report saved to: {output_dir / f'cxr_real_emr_weighted_classification_report_th{threshold}.json'}")
    
    # Save metrics
    save_json(metrics, output_dir / f"cxr_real_emr_weighted_test_metrics_th{threshold}.json")
    print(f"✅ Saved JSON: {output_dir / f'cxr_real_emr_weighted_test_metrics_th{threshold}.json'}")
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | "
          f"Sensitivity: {sensitivity:.2f} | Specificity: {specificity:.2f}")
    print("="*60)
    print("✅ CXR + Real EMR fusion weighted evaluation completed successfully!")
    print(f"   Results saved to: {output_dir}")
    print("="*60)
    
    return metrics

def evaluate_multiple_thresholds(model_path, test_loader, device, output_dir, thresholds=[0.3, 0.4, 0.5, 0.6]):
    """Evaluate model at multiple thresholds"""
    print("🔍 Evaluating at multiple thresholds...")
    print("="*60)
    
    all_metrics = {}
    
    for threshold in thresholds:
        print(f"\n📊 Evaluating at threshold: {threshold}")
        print("-" * 40)
        
        metrics = evaluate_model_at_threshold(model_path, test_loader, device, output_dir, threshold)
        all_metrics[threshold] = metrics
        
        # Print key metrics
        print(f"   Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f} | "
              f"Sensitivity: {metrics['sensitivity']:.3f} | Specificity: {metrics['specificity']:.3f}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("📊 THRESHOLD COMPARISON TABLE")
    print("="*80)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 80)
    
    for threshold in thresholds:
        m = all_metrics[threshold]
        print(f"{threshold:<10.1f} {m['accuracy']:<10.3f} {m['f1_score']:<8.3f} {m['precision']:<10.3f} "
              f"{m['recall']:<8.3f} {m['sensitivity']:<12.3f} {m['specificity']:<12.3f}")
    
    print("="*80)
    
    # Save comparison metrics
    save_json(all_metrics, output_dir / "cxr_real_emr_weighted_threshold_comparison.json")
    print(f"✅ Threshold comparison saved to: {output_dir / 'cxr_real_emr_weighted_threshold_comparison.json'}")
    
    return all_metrics

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate CXR + Real EMR Fusion Weighted Model")
    parser.add_argument('--model_path', type=str, 
                       default=str(CHECKPOINTS_DIR / "cxr_real_emr_fusion_weighted.pt"),
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--threshold', type=float, default=0.4, 
                       help='Classification threshold')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum samples for testing')
    parser.add_argument('--multi_threshold', action='store_true',
                       help='Evaluate at multiple thresholds (0.3, 0.4, 0.5, 0.6)')
    
    args = parser.parse_args()
    
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
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the weighted model first:")
        print("   python scripts/train_cxr_real_emr.py --epochs 10 --batch_size 32 --lr 1e-4")
        sys.exit(1)
    
    # Check if required data exists
    required_dirs = [
        Path("data/processed"),
        Path("data/processed/real_emr"),
        Path("data/raw/rsna")
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        print("❌ Missing required directories:")
        for d in missing_dirs:
            print(f"   - {d}")
        print("\nPlease run preprocessing first:")
        print("   python scripts/prepare_data.py --emr")
        print("   python src/preprocess_real_emr.py")
        sys.exit(1)
    
    # Load test data
    print("\n📊 Loading test data...")
    dataloaders = make_cxr_real_emr_dataloaders(
        rsna_arrays_dir=str(Path("data/processed")),
        real_emr_arrays_dir=str(Path("data/processed/real_emr")),
        rsna_root=str(Path("data/raw/rsna")),
        batch_size=32,
        num_workers=NUM_WORKERS,
        max_samples=args.max_samples
    )
    
    test_loader = dataloaders['test']
    print(f"   Test batches: {len(test_loader)}")
    
    # Run evaluation
    if args.multi_threshold:
        metrics = evaluate_multiple_thresholds(model_path, test_loader, device, FIG_DIR)
    else:
        metrics = evaluate_model_at_threshold(model_path, test_loader, device, FIG_DIR, args.threshold)
    
    return metrics

if __name__ == "__main__":
    main()
