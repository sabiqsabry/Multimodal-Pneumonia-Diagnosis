#!/usr/bin/env python3
"""
Evaluation Script for Multimodal Pneumonia Diagnosis Model
Generates comprehensive evaluation metrics and visualizations
"""

import argparse
import sys
import json
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Warning: seaborn not available, using matplotlib defaults")
    sns = None
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from tqdm import tqdm

from models import create_model
from data_module import make_dataloaders
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, FIG_DIR, LABEL_MAP
from utils_io import load_torch, ensure_dir, save_json

class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_dataloader(self, dataloader, split_name="test"):
        """Evaluate model on a dataloader"""
        print(f"🔍 Evaluating on {split_name} set...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name}")
            
            for batch in progress_bar:
                # Move data to device
                images = batch['image'].to(self.device)
                emr_features = batch['emr'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Forward pass
                logits, probabilities = self.model(images, emr_features)
                
                # Store results
                all_predictions.extend((probabilities > 0.5).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def compute_metrics(self, predictions, targets, probabilities):
        """Compute comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        auc = roc_auc_score(targets, probabilities)
        f1 = f1_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Additional metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def plot_confusion_matrix(self, predictions, targets, save_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Pneumonia'],
                       yticklabels=['Normal', 'Pneumonia'])
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            plt.xticks([0, 1], ['Normal', 'Pneumonia'])
            plt.yticks([0, 1], ['Normal', 'Pneumonia'])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Confusion matrix saved to: {save_path}")
    
    def plot_roc_curve(self, targets, probabilities, save_path):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ ROC curve saved to: {save_path}")
    
    def plot_precision_recall_curve(self, targets, probabilities, save_path):
        """Plot and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        f1 = f1_score(targets, (probabilities > 0.5).astype(int))
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (F1 = {f1:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Precision-Recall curve saved to: {save_path}")
    
    def plot_metrics_bar_chart(self, metrics, save_path):
        """Plot and save bar chart of key metrics"""
        # Select key metrics for visualization
        metric_names = ['Accuracy', 'F1 Score', 'AUC', 'Sensitivity', 'Specificity']
        metric_values = [
            metrics['accuracy'],
            metrics['f1_score'],
            metrics['auc'],
            metrics['sensitivity'],
            metrics['specificity']
        ]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize plot
        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Metrics bar chart saved to: {save_path}")
    
    def create_combined_results_summary(self, predictions, targets, probabilities, metrics, save_path):
        """Create combined results summary figure for thesis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confusion Matrix (top-left)
        cm = confusion_matrix(targets, predictions)
        im1 = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        axes[0, 0].set_xticks([0, 1])
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_xticklabels(['Normal', 'Pneumonia'])
        axes[0, 0].set_yticklabels(['Normal', 'Pneumonia'])
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                               fontsize=14, fontweight='bold', color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        # 2. ROC Curve (top-right)
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Metrics Bar Chart (bottom-left)
        metric_names = ['Accuracy', 'F1', 'AUC', 'Sensitivity', 'Specificity']
        metric_values = [
            metrics['accuracy'],
            metrics['f1_score'],
            metrics['auc'],
            metrics['sensitivity'],
            metrics['specificity']
        ]
        
        bars = axes[1, 0].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Summary Statistics (bottom-right)
        axes[1, 1].axis('off')
        
        # Create summary text
        summary_text = f"""
Model Performance Summary

Test Set Results:
• Total Samples: {len(targets):,}
• Pneumonia Cases: {int(targets.sum()):,}
• Normal Cases: {int(len(targets) - targets.sum()):,}

Key Metrics:
• Accuracy: {metrics['accuracy']:.3f}
• AUC: {metrics['auc']:.3f}
• F1 Score: {metrics['f1_score']:.3f}
• Sensitivity: {metrics['sensitivity']:.3f}
• Specificity: {metrics['specificity']:.3f}

Confusion Matrix:
• True Positives: {metrics['true_positives']:,}
• True Negatives: {metrics['true_negatives']:,}
• False Positives: {metrics['false_positives']:,}
• False Negatives: {metrics['false_negatives']:,}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Overall title
        fig.suptitle('Multimodal Pneumonia Diagnosis Model - Final Evaluation Results', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Combined results summary saved to: {save_path}")
    
    def generate_classification_report(self, predictions, targets, save_path):
        """Generate and save detailed classification report"""
        # Get class names
        class_names = list(LABEL_MAP.keys())
        
        # Generate report
        report = classification_report(
            targets, predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as text
        text_report = classification_report(targets, predictions, target_names=class_names)
        text_path = save_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write(text_report)
        
        print(f"   ✅ Classification report saved to: {save_path}")
        print(f"   ✅ Text report saved to: {text_path}")
        
        return report

def evaluate_model(model_path, test_loader, device, output_dir):
    """Main evaluation function"""
    print("🚀 Starting Model Evaluation...")
    print("="*60)
    
    # Load model
    print(f"📁 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"   Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on test set
    predictions, targets, probabilities = evaluator.evaluate_dataloader(test_loader, "test")
    
    # Compute metrics
    print("\n📊 Computing evaluation metrics...")
    metrics = evaluator.compute_metrics(predictions, targets, probabilities)
    
    # Print metrics
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"AUC:          {metrics['auc']:.4f}")
    print(f"F1 Score:     {metrics['f1_score']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"PPV:          {metrics['ppv']:.4f}")
    print(f"NPV:          {metrics['npv']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    # Create visualizations
    print("\n🎨 Generating visualizations...")
    
    # Confusion Matrix
    cm_path = output_dir / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(predictions, targets, cm_path)
    
    # ROC Curve
    roc_path = output_dir / "roc_curve.png"
    evaluator.plot_roc_curve(targets, probabilities, roc_path)
    
    # Precision-Recall Curve
    pr_path = output_dir / "pr_curve.png"
    evaluator.plot_precision_recall_curve(targets, probabilities, pr_path)
    
    # Metrics Bar Chart
    metrics_bar_path = output_dir / "metrics_bar.png"
    evaluator.plot_metrics_bar_chart(metrics, metrics_bar_path)
    
    # Combined Results Summary
    summary_path = output_dir / "results_summary.png"
    evaluator.create_combined_results_summary(predictions, targets, probabilities, metrics, summary_path)
    
    # Classification Report
    report_path = output_dir / "classification_report.json"
    classification_report = evaluator.generate_classification_report(predictions, targets, report_path)
    
    # Save metrics to JSON
    metrics_path = output_dir / "test_metrics.json"
    save_json(metrics, metrics_path)
    print(f"   ✅ Metrics saved to: {metrics_path}")
    
    # Append to train log CSV
    print("\n📝 Updating training log...")
    try:
        train_log_path = Path("outputs/logs/train_log.csv")
        if train_log_path.exists():
            import pandas as pd
            # Read existing log
            df = pd.read_csv(train_log_path)
            
            # Create final summary row
            final_row = {
                'epoch': 'final_test',
                'split': 'test',
                'loss': 0.0,  # Not computed for final evaluation
                'accuracy': metrics['accuracy'],
                'auc': metrics['auc'],
                'f1': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'lr': 0.0  # Not applicable for final evaluation
            }
            
            # Append final row
            df = pd.concat([df, pd.DataFrame([final_row])], ignore_index=True)
            df.to_csv(train_log_path, index=False)
            print(f"   ✅ Final metrics appended to: {train_log_path}")
        else:
            print(f"   ⚠️ Training log not found at: {train_log_path}")
    except Exception as e:
        print(f"   ⚠️ Failed to update training log: {e}")
    
    # Print clean summary
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Accuracy: {metrics['accuracy']:.2f} | AUC: {metrics['auc']:.2f} | F1: {metrics['f1_score']:.2f} | Sensitivity: {metrics['sensitivity']:.2f} | Specificity: {metrics['specificity']:.2f}")
    print("="*60)
    print("✅ Evaluation completed successfully!")
    print(f"   Results saved to: {output_dir}")
    print("="*60)
    
    return metrics

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Pneumonia Diagnosis Model")
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/best_model.pt', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='outputs/figures/', help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Load test data
    print("📊 Loading test data...")
    dataloaders = make_dataloaders(
        split_csv_dir=str(PROC_DIR),
        emr_arrays_dir=str(PROC_DIR),
        rsna_root=str(RSNA_ROOT),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        cache=False
    )
    
    test_loader = dataloaders['test']
    print(f"   Test batches: {len(test_loader)}")
    
    # Evaluate model
    metrics = evaluate_model(args.model_path, test_loader, device, output_dir)
    
    return metrics

if __name__ == "__main__":
    main()
