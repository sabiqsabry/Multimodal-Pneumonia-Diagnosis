"""
Simplified Explainability Module for Multimodal Pneumonia Diagnosis
Implements basic Grad-CAM and SHAP explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from preprocess_images import denormalize_image
from config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

def generate_gradcam_simple(model, image_tensor, class_idx=None, save_path=None):
    """
    Simplified Grad-CAM implementation
    
    Args:
        model: Trained multimodal model
        image_tensor: Input image tensor [1, C, H, W]
        class_idx: Target class index (if None, uses predicted class)
        save_path: Path to save the visualization
        
    Returns:
        CAM as numpy array
    """
    model.eval()
    image_tensor.requires_grad_()
    
    # Get the ResNet18 backbone
    resnet = model.cxr_encoder.backbone
    
    # Find the last convolutional layer
    last_conv = None
    for name, module in resnet.named_modules():
        if isinstance(module, nn.Conv2d) and 'layer4' in name:
            last_conv = module
            break
    
    if last_conv is None:
        print("Warning: Could not find last conv layer, using any conv layer")
        for name, module in resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                break
    
    if last_conv is None:
        print("Error: No convolutional layer found")
        return None
    
    # Register hooks
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_backward_hook(backward_hook)
    
    try:
        # Forward pass
        with torch.enable_grad():
            # Create dummy EMR tensor
            device = image_tensor.device
            dummy_emr = torch.zeros(1, 7, device=device)
            logits, probabilities = model(image_tensor, dummy_emr)
            
            if class_idx is None:
                class_idx = torch.argmax(probabilities).item()
            
            # Backward pass
            model.zero_grad()
            logits[0, class_idx].backward()
        
        # Generate CAM
        if len(activations) > 0 and len(gradients) > 0:
            activation = activations[0][0]  # Remove batch dimension
            gradient = gradients[0][0]      # Remove batch dimension
            
            # Global average pooling of gradients
            weights = torch.mean(gradient, dim=(1, 2))
            
            # Weighted combination of activation maps
            cam = torch.zeros(activation.shape[1:], dtype=torch.float32, device=activation.device)
            for i, w in enumerate(weights):
                cam += w * activation[i, :, :]
            
            # Apply ReLU and normalize
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cam_np = cam.detach().cpu().numpy()
            
            # Create visualization
            if save_path:
                _create_gradcam_visualization_simple(image_tensor, cam_np, save_path)
            
            return cam_np
        else:
            print("Error: No activations or gradients captured")
            return None
    
    finally:
        # Cleanup hooks
        forward_handle.remove()
        backward_handle.remove()

def _create_gradcam_visualization_simple(image_tensor, cam, save_path):
    """Create and save Grad-CAM visualization"""
    # Denormalize image
    image = denormalize_image(image_tensor[0], IMAGENET_MEAN, IMAGENET_STD)
    image_np = image.permute(1, 2, 0).numpy()
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original CXR', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(cam_resized, cmap='jet', alpha=0.8)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
    axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Grad-CAM saved to: {save_path}")

def explain_emr_with_shap_simple(model, emr_tensor, feature_names, save_path=None, max_features=5):
    """
    Simplified SHAP explanation for EMR features
    
    Args:
        model: Trained multimodal model
        emr_tensor: EMR features tensor [1, F]
        feature_names: List of feature names
        save_path: Path to save the visualization
        max_features: Maximum number of features to display
        
    Returns:
        SHAP values and feature importance
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not available. Install with: pip install shap")
        return None
    
    model.eval()
    
    # Create a simple wrapper function
    def model_predict(emr_data):
        """Wrapper function for SHAP"""
        with torch.no_grad():
            # Convert to tensor if needed
            if not isinstance(emr_data, torch.Tensor):
                emr_data = torch.tensor(emr_data, dtype=torch.float32)
            
            # Move to same device as model
            device = next(model.parameters()).device
            emr_data = emr_data.to(device)
            
            # Create dummy image tensor on same device
            batch_size = emr_data.shape[0]
            dummy_image = torch.zeros(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
            
            # Get predictions
            logits, probabilities = model(dummy_image, emr_data)
            return probabilities.cpu().numpy()
    
    try:
        # Convert to CPU for SHAP
        emr_cpu = emr_tensor.cpu().numpy().reshape(1, -1)
        
        # Use a simple baseline (zeros)
        baseline = np.zeros((1, emr_cpu.shape[1]))
        
        # Use KernelExplainer
        explainer = shap.KernelExplainer(model_predict, baseline)
        shap_values = explainer.shap_values(emr_cpu)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get feature importance
        feature_importance = np.abs(shap_values[0])
        feature_indices = np.argsort(feature_importance)[::-1][:max_features]
        
        # Create visualization
        if save_path:
            _create_shap_visualization_simple(
                feature_names, feature_importance, feature_indices, 
                shap_values[0], save_path, max_features
            )
        
        return {
            'shap_values': shap_values[0],
            'feature_importance': feature_importance,
            'top_features': feature_indices
        }
    
    except Exception as e:
        print(f"Warning: SHAP explanation failed: {e}")
        return None

def _create_shap_visualization_simple(feature_names, feature_importance, top_indices, 
                                     shap_values, save_path, max_features):
    """Create and save SHAP bar chart visualization"""
    # Get top features
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    top_shap = shap_values[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    colors = ['red' if x < 0 else 'green' for x in top_shap]
    bars = ax.barh(range(len(top_features)), top_shap, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('SHAP Value (Contribution to Prediction)', fontsize=12)
    ax.set_title('EMR Feature Contributions (SHAP)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_shap)):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                va='center', ha='left' if value >= 0 else 'right', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive Contribution'),
        Patch(facecolor='red', alpha=0.7, label='Negative Contribution')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ SHAP visualization saved to: {save_path}")

def main():
    """Test the explainability functions"""
    print("🧪 Testing Simplified Explainability Module...")
    print(f"   SHAP available: {SHAP_AVAILABLE}")
    print("   ✅ Simplified explainability module loaded successfully!")

if __name__ == "__main__":
    main()
