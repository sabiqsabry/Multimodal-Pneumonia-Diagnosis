#!/usr/bin/env python3
"""
Training Script for Multimodal Pneumonia Diagnosis Model
Implements training loop with metrics tracking, early stopping, and model saving
"""

import argparse
import sys
import os
import time
from pathlib import Path
import json
import csv
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

from models import create_model
from data_module import make_dataloaders
from config import PROC_DIR, RSNA_ROOT, BATCH_SIZE, NUM_WORKERS, CLASS_WEIGHTS_FILE
from utils_io import save_torch, load_torch, ensure_dir

class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, loss, predictions, targets, probabilities):
        """Update metrics with batch results"""
        self.losses.append(loss)
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute_metrics(self):
        """Compute all metrics from accumulated data"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Convert probabilities to binary predictions
        binary_preds = (probabilities > 0.5).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(targets, binary_preds)
        auc = roc_auc_score(targets, probabilities)
        f1 = f1_score(targets, binary_preds)
        
        # Compute sensitivity and specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, binary_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'loss': np.mean(self.losses),
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

def train_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Train model for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['image'].to(device)
        emr_features = batch['emr'].to(device)
        targets = batch['label'].to(device).float().unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, probabilities = model(images, emr_features)
        
        # Compute loss
        if class_weights is not None:
            # Apply class weights
            weights = torch.where(targets == 1, class_weights[1], class_weights[0])
            loss = criterion(logits, targets)
            loss = (loss * weights).mean()
        else:
            loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics.update(
            loss.item(),
            (probabilities > 0.5).float().detach(),
            targets.squeeze().detach(),
            probabilities.squeeze().detach()
        )
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return metrics.compute_metrics()

def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch"""
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(device)
            emr_features = batch['emr'].to(device)
            targets = batch['label'].to(device).float().unsqueeze(1)
            
            # Forward pass
            logits, probabilities = model(images, emr_features)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Update metrics
            metrics.update(
                loss.item(),
                (probabilities > 0.5).float().detach(),
                targets.squeeze().detach(),
                probabilities.squeeze().detach()
            )
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return metrics.compute_metrics()

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs, save_dir, class_weights=None, patience=5):
    """Main training loop"""
    
    # Create save directory
    ensure_dir(save_dir)
    
    # Initialize tracking variables
    best_auc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Training log
    log_file = Path(save_dir).parent / "logs" / "train_log.csv"
    ensure_dir(log_file.parent)
    
    # Initialize CSV log
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_f1', 
                        'val_loss', 'val_acc', 'val_auc', 'val_f1', 'val_sensitivity', 'val_specificity'])
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"📁 Checkpoints will be saved to: {save_dir}")
    print(f"📊 Training log: {log_file}")
    print("="*80)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, class_weights)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Compute epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.3f} | "
              f"Val AUC: {val_metrics['auc']:.3f} | "
              f"Val F1: {val_metrics['f1']:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['loss'],
                train_metrics['accuracy'],
                train_metrics['auc'],
                train_metrics['f1'],
                val_metrics['loss'],
                val_metrics['accuracy'],
                val_metrics['auc'],
                val_metrics['f1'],
                val_metrics['sensitivity'],
                val_metrics['specificity']
            ])
        
        # Check for improvement
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = Path(save_dir) / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics, best_model_path)
            print(f"   💾 New best model saved! AUC: {best_auc:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n⏹️  Early stopping triggered! No improvement for {patience} epochs.")
            print(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break
        
        print()
    
    print("="*80)
    print(f"✅ Training completed!")
    print(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"   Best model saved to: {Path(save_dir) / 'best_model.pt'}")
    
    return best_auc, best_epoch

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Multimodal Pneumonia Diagnosis Model")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
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
    
    # Load class weights
    class_weights_path = PROC_DIR / CLASS_WEIGHTS_FILE
    if class_weights_path.exists():
        class_weights = load_torch(class_weights_path)
        print(f"⚖️  Loaded class weights: {class_weights}")
    else:
        class_weights = None
        print("⚠️  No class weights found, training without class balancing")
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model()
    model = model.to(device)
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"   Total parameters: {param_counts['total']:,}")
    print(f"   Trainable parameters: {param_counts['trainable']:,}")
    
    # Load data
    print("📊 Loading data...")
    dataloaders = make_dataloaders(
        split_csv_dir=str(PROC_DIR),
        emr_arrays_dir=str(PROC_DIR),
        rsna_root=str(RSNA_ROOT),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        cache=False
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train model
    best_auc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        class_weights=class_weights,
        patience=args.patience
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Best epoch: {best_epoch}")

if __name__ == "__main__":
    main()
