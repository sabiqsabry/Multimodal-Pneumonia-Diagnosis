"""
CXR-Only Model for Pneumonia Classification
Simple CNN classifier using pretrained ResNet18
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CXROnlyModel(nn.Module):
    """
    CXR-only pneumonia classification model using pretrained ResNet18
    
    Architecture:
    - Pretrained ResNet18 (ImageNet weights)
    - Replace final FC layer with Linear(512, 1)
    - Sigmoid activation for binary classification
    """
    
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(CXROnlyModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)  # ResNet18 features = 512
        )
        
        # Initialize classifier weights
        self._initialize_classifier_weights()
    
    def _initialize_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            logits: Raw logits of shape (batch_size, 1)
            probabilities: Sigmoid probabilities of shape (batch_size, 1)
        """
        # Extract features using pretrained backbone
        features = self.backbone(x)  # (batch_size, 512, 1, 1)
        features = torch.flatten(features, 1)  # (batch_size, 512)
        
        # Classify
        logits = self.classifier(features)  # (batch_size, 1)
        probabilities = torch.sigmoid(logits)  # (batch_size, 1)
        
        return logits, probabilities
    
    def get_feature_dim(self):
        """Get the dimension of features before classification"""
        return 512

def create_cxr_only_model(num_classes=1, dropout_rate=0.3):
    """
    Create CXR-only model
    
    Args:
        num_classes: Number of output classes (default: 1 for binary)
        dropout_rate: Dropout rate in classifier (default: 0.3)
    
    Returns:
        CXROnlyModel instance
    """
    return CXROnlyModel(num_classes=num_classes, dropout_rate=dropout_rate)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model):
    """Get model information"""
    total_params = count_parameters(model)
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'backbone_parameters': backbone_params,
        'classifier_parameters': classifier_params,
        'feature_dim': model.get_feature_dim()
    }

if __name__ == "__main__":
    # Test model creation
    model = create_cxr_only_model()
    print("CXR-Only Model Created Successfully!")
    print(f"Model Info: {get_model_info(model)}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, probabilities = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities range: [{probabilities.min().item():.3f}, {probabilities.max().item():.3f}]")



