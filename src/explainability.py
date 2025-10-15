"""
Explainability Module for Multimodal Pneumonia Diagnosis
Implements Grad-CAM for CXR images and SHAP for EMR features
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

class GradCAM:
    """
    Grad-CAM implementation for visualizing important regions in CNN predictions
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
            target_layer: Target convolutional layer for gradient extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            CAM as numpy array
        """
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad_()
        
        # Get prediction
        with torch.enable_grad():
            # Create dummy EMR tensor on same device
            device = input_tensor.device
            dummy_emr = torch.zeros(1, 7, device=device)
            logits, probabilities = self.model(input_tensor, dummy_emr)
            if class_idx is None:
                class_idx = torch.argmax(probabilities).item()
            
            # Backward pass
            self.model.zero_grad()
            logits[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Ensure we have the right dimensions
        if len(gradients.shape) == 4:  # [B, C, H, W]
            gradients = gradients[0]
        if len(activations.shape) == 4:  # [B, C, H, W]
            activations = activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def cleanup(self):
        """Remove registered hooks"""
        for handle in self.hook_handles:
            handle.remove()

def generate_gradcam(model, image_tensor, target_layer, class_idx=None, save_path=None):
    """
    Generate Grad-CAM visualization for CXR image
    
    Args:
        model: Trained multimodal model
        image_tensor: Input image tensor [1, C, H, W]
        target_layer: Target convolutional layer
        class_idx: Target class index (if None, uses predicted class)
        save_path: Path to save the visualization
        
    Returns:
        CAM as numpy array
    """
    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)
    
    try:
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor, class_idx)
        
        # Create visualization
        if save_path:
            _create_gradcam_visualization(image_tensor, cam, save_path)
        
        return cam
    
    finally:
        # Cleanup
        gradcam.cleanup()

def _create_gradcam_visualization(image_tensor, cam, save_path):
    """
    Create and save Grad-CAM visualization
    
    Args:
        image_tensor: Original image tensor [1, C, H, W]
        cam: Class activation map [H, W]
        save_path: Path to save the visualization
    """
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

def explain_emr_with_shap(model, emr_tensor, feature_names, save_path=None, max_features=5):
    """
    Explain EMR features using SHAP
    
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
    
    # Create a wrapper function for SHAP
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
    
    # Create SHAP explainer
    try:
        # Convert to CPU for SHAP
        emr_cpu = emr_tensor.cpu().numpy().reshape(1, -1)
        # Use KernelExplainer for simplicity
        explainer = shap.KernelExplainer(model_predict, emr_cpu)
        shap_values = explainer.shap_values(emr_cpu)
        
        # Get feature importance
        feature_importance = np.abs(shap_values[0])
        feature_indices = np.argsort(feature_importance)[::-1][:max_features]
        
        # Create visualization
        if save_path:
            _create_shap_visualization(
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

def _create_shap_visualization(feature_names, feature_importance, top_indices, 
                              shap_values, save_path, max_features):
    """
    Create and save SHAP bar chart visualization
    
    Args:
        feature_names: List of feature names
        feature_importance: Feature importance values
        top_indices: Indices of top features
        shap_values: SHAP values
        save_path: Path to save the visualization
    """
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

def get_resnet_last_conv_layer(model):
    """
    Get the last convolutional layer of ResNet18
    
    Args:
        model: Multimodal model
        
    Returns:
        Last convolutional layer
    """
    # Get the ResNet18 backbone
    resnet = model.cxr_encoder.backbone
    
    # Find the last convolutional layer
    last_conv = None
    for name, module in resnet.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("Could not find convolutional layer in ResNet18")
    
    return last_conv

def create_explainability_summary(gradcam_path, shap_path, prediction, true_label, probability):
    """
    Create a summary of explainability results
    
    Args:
        gradcam_path: Path to Grad-CAM image
        shap_path: Path to SHAP visualization
        prediction: Predicted label
        true_label: True label
        probability: Prediction probability
        
    Returns:
        Summary string
    """
    label_map = {0: "Normal", 1: "Pneumonia"}
    
    summary = f"""
🔍 EXPLAINABILITY ANALYSIS SUMMARY
{'='*50}
📊 Prediction: {label_map.get(prediction, 'Unknown')} ({probability:.3f} probability)
🎯 True Label: {label_map.get(true_label, 'Unknown')}
✅ Correct: {'Yes' if prediction == true_label else 'No'}

📁 Generated Files:
   • Grad-CAM: {gradcam_path}
   • SHAP: {shap_path}

💡 Interpretation:
   • Grad-CAM shows which regions in the CXR influenced the prediction
   • SHAP shows which EMR features contributed most to the decision
   • Red areas in Grad-CAM indicate high attention regions
   • Green bars in SHAP indicate positive contributions, red bars indicate negative
{'='*50}
"""
    return summary

def main():
    """Test the explainability functions"""
    print("🧪 Testing Explainability Module...")
    
    # Test imports
    print(f"   SHAP available: {SHAP_AVAILABLE}")
    
    # Test Grad-CAM (would need actual model and data)
    print("   ✅ Explainability module loaded successfully!")

if __name__ == "__main__":
    main()
