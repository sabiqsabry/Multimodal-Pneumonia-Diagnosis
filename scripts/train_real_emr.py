#!/usr/bin/env python3
"""
Training Script for Real EMR-Only Pneumonia Classification Model
Trains an MLP classifier using only real EMR features
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

from models_real_emr import create_real_emr_model, get_model_info
from config import SEED, CHECKPOINTS_DIR, LOGS_DIR
from utils_io import ensure_dir, save_torch, load_numpy, load_json

class RealEMRDataset(torch.utils.data.Dataset):
    """Real EMR dataset for training"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'label': self.y[idx]
        }

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
        features = batch['features'].to(device)
        targets = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, probabilities = model(features)
        
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
            features = batch['features'].to(device)
            targets = batch['label'].to(device)
            
            # Forward pass
            logits, probabilities = model(features)
            
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

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    unique_classes, counts = np.unique(y_train, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(y_train)
    class_weights = total_samples / (len(unique_classes) * counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(unique_classes)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def train_real_emr_model(epochs=10, batch_size=32, lr=1e-4, device='auto', 
                        patience=5, min_delta=0.001, hidden_dims=[128, 64, 32], 
                        dropout_rate=0.3):
    """
    Train Real EMR-only pneumonia classification model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
    """
    print("🚀 Real EMR-Only Pneumonia Classification Training")
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
    
    # Load preprocessed data
    print("\n📊 Loading preprocessed data...")
    real_emr_proc_dir = Path("data/processed/real_emr")
    
    # Load feature info
    feature_info = load_json(real_emr_proc_dir / "feature_info.json")
    input_dim = feature_info['n_features']
    feature_columns = feature_info['feature_columns']
    
    print(f"   Input dimension: {input_dim}")
    print(f"   Feature columns: {feature_columns}")
    
    # Load data arrays
    X_train = load_numpy(real_emr_proc_dir / "X_real_emr_train.npy")
    X_val = load_numpy(real_emr_proc_dir / "X_real_emr_val.npy")
    y_train = load_numpy(real_emr_proc_dir / "y_real_emr_train.npy")
    y_val = load_numpy(real_emr_proc_dir / "y_real_emr_val.npy")
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples: {len(X_val)}")
    
    # Create datasets and dataloaders
    train_dataset = RealEMRDataset(X_train, y_train)
    val_dataset = RealEMRDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Compute class weights
    print("\n⚖️ Computing class weights...")
    class_weights = compute_class_weights(y_train)
    print(f"   Class weights: {class_weights.numpy()}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_real_emr_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    # Print model info
    model_info = get_model_info(model)
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Architecture: {model_info['architecture']}")
    
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
                'model_info': model_info,
                'feature_columns': feature_columns,
                'input_dim': input_dim
            }
            
            best_model_path = CHECKPOINTS_DIR / "real_emr_best.pt"
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
    log_path = LOGS_DIR / "real_emr_train_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"   ✅ Training log saved to: {log_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("📈 TRAINING SUMMARY")
    print("="*60)
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best model saved to: {CHECKPOINTS_DIR / 'real_emr_best.pt'}")
    print("="*60)
    print("✅ Real EMR training completed successfully!")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Real EMR-Only Pneumonia Classification Model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64, 32], 
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Check if required files exist
    real_emr_proc_dir = Path("data/processed/real_emr")
    required_files = [
        real_emr_proc_dir / "X_real_emr_train.npy",
        real_emr_proc_dir / "y_real_emr_train.npy",
        real_emr_proc_dir / "X_real_emr_val.npy",
        real_emr_proc_dir / "y_real_emr_val.npy",
        real_emr_proc_dir / "feature_info.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run real EMR preprocessing first:")
        print("   python src/preprocess_real_emr.py")
        sys.exit(1)
    
    # Start training
    train_real_emr_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    )

if __name__ == "__main__":
    main()



