#!/usr/bin/env python3
"""
Training Script for CXR-Only Pneumonia Classification Model
Trains a ResNet18-based CNN classifier using only chest X-ray images
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
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from models_cxr_only import create_cxr_only_model, get_model_info
from data_module_cxr_only import make_cxr_only_dataloaders, compute_class_weights_cxr_only
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, CHECKPOINTS_DIR, LOGS_DIR
from utils_io import ensure_dir, save_torch, load_torch

class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0.0
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.count = 0
    
    def update(self, loss, predictions, targets, probabilities):
        self.loss += loss
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probabilities.cpu().numpy())
        self.count += 1
    
    def compute_metrics(self):
        """Compute metrics from accumulated data"""
        if self.count == 0:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Convert predictions to binary
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(targets, binary_predictions)
        auc = roc_auc_score(targets, probabilities)
        f1 = f1_score(targets, binary_predictions)
        precision = precision_score(targets, binary_predictions)
        recall = recall_score(targets, binary_predictions)
        
        # Confusion matrix metrics
        tn = np.sum((targets == 0) & (binary_predictions == 0))
        fp = np.sum((targets == 0) & (binary_predictions == 1))
        fn = np.sum((targets == 1) & (binary_predictions == 0))
        tp = np.sum((targets == 1) & (binary_predictions == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'loss': self.loss / self.count,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

def train_epoch(model, dataloader, criterion, optimizer, device, metrics_tracker):
    """Train for one epoch"""
    model.train()
    metrics_tracker.reset()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        images = batch['image'].to(device)
        targets = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, probabilities = model(images)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics_tracker.update(
            loss.item(),
            (probabilities > 0.5).float().detach(),
            targets.squeeze().detach(),
            probabilities.squeeze().detach()
        )
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return metrics_tracker.compute_metrics()

def validate_epoch(model, dataloader, criterion, device, metrics_tracker):
    """Validate for one epoch"""
    model.eval()
    metrics_tracker.reset()
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            # Forward pass
            logits, probabilities = model(images)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Update metrics
            metrics_tracker.update(
                loss.item(),
                (probabilities > 0.5).float().detach(),
                targets.squeeze().detach(),
                probabilities.squeeze().detach()
            )
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return metrics_tracker.compute_metrics()

def train_cxr_only_model(epochs=10, batch_size=32, lr=1e-4, device='auto', 
                        patience=5, min_delta=0.001, cache=False):
    """
    Train CXR-only pneumonia classification model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        cache: Whether to cache preprocessed images
    """
    print("🚀 CXR-Only Pneumonia Classification Training")
    print("="*60)
    
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
    
    # Create output directories
    ensure_dir(CHECKPOINTS_DIR)
    ensure_dir(LOGS_DIR)
    
    # Load data
    print("\n📊 Loading data...")
    dataloaders = make_cxr_only_dataloaders(
        split_csv_dir=str(PROC_DIR),
        rsna_root=str(RSNA_ROOT),
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        cache=cache
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Compute class weights
    print("\n⚖️ Computing class weights...")
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['label'].numpy().flatten())
    train_labels = np.array(train_labels)
    
    class_weights = compute_class_weights_cxr_only(train_labels)
    print(f"   Class weights: {class_weights.numpy()}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_cxr_only_model()
    model = model.to(device)
    
    # Print model info
    model_info = get_model_info(model)
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Backbone parameters: {model_info['backbone_parameters']:,}")
    print(f"   Classifier parameters: {model_info['classifier_parameters']:,}")
    
    # Create loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print(f"\n🎯 Starting training for {epochs} epochs...")
    print("="*60)
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Initialize metrics tracker
    train_metrics_tracker = MetricsTracker()
    val_metrics_tracker = MetricsTracker()
    
    # Training log
    training_log = []
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, train_metrics_tracker)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, val_metrics_tracker)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Log metrics
        log_entry = {
            'epoch': epoch + 1,
            'split': 'train',
            'loss': train_metrics['loss'],
            'accuracy': train_metrics['accuracy'],
            'auc': train_metrics['auc'],
            'f1': train_metrics['f1'],
            'precision': train_metrics['precision'],
            'recall': train_metrics['recall'],
            'sensitivity': train_metrics['sensitivity'],
            'specificity': train_metrics['specificity'],
            'lr': current_lr
        }
        training_log.append(log_entry)
        
        log_entry_val = {
            'epoch': epoch + 1,
            'split': 'val',
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'auc': val_metrics['auc'],
            'f1': val_metrics['f1'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'sensitivity': val_metrics['sensitivity'],
            'specificity': val_metrics['specificity'],
            'lr': current_lr
        }
        training_log.append(log_entry_val)
        
        # Check for improvement
        if val_metrics['auc'] > best_val_auc + min_delta:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_loss': val_metrics['loss'],
                'class_weights': class_weights,
                'model_info': model_info
            }
            
            best_model_path = CHECKPOINTS_DIR / "cxr_only_best.pt"
            save_torch(checkpoint, best_model_path)
            print(f"   💾 New best model saved! (AUC: {val_metrics['auc']:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Save training log
    print(f"\n📝 Saving training log...")
    log_df = pd.DataFrame(training_log)
    log_path = LOGS_DIR / "cxr_only_train_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"   ✅ Training log saved to: {log_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("📈 TRAINING SUMMARY")
    print("="*60)
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best model saved to: {CHECKPOINTS_DIR / 'cxr_only_best.pt'}")
    print("="*60)
    print("✅ CXR-only training completed successfully!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train CXR-Only Pneumonia Classification Model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--cache', action='store_true', help='Cache preprocessed images')
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = [
        PROC_DIR / "image_ids_train.npy",
        PROC_DIR / "y_train.npy",
        PROC_DIR / "image_ids_val.npy",
        PROC_DIR / "y_val.npy"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run data preprocessing first:")
        print("   python scripts/prepare_data.py --emr")
        sys.exit(1)
    
    # Start training
    train_cxr_only_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        cache=args.cache
    )

if __name__ == "__main__":
    main()



