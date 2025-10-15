"""
Multimodal Model Architecture for Pneumonia Diagnosis
Combines CNN (ResNet18) for chest X-rays with MLP for EMR data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Dict, Any
import math

class CXR_Encoder(nn.Module):
    """
    Chest X-ray Encoder using pretrained ResNet18
    Replaces final FC layer with linear projection to img_feat_dim
    """
    
    def __init__(self, img_feat_dim: int = 256, pretrained: bool = True):
        """
        Initialize CXR Encoder
        
        Args:
            img_feat_dim: Output feature dimension
            pretrained: Whether to use pretrained weights
        """
        super(CXR_Encoder, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get the number of input features to the final FC layer
        num_features = self.backbone.fc.in_features
        
        # Replace final FC layer with projection layer
        self.backbone.fc = nn.Linear(num_features, img_feat_dim)
        
        self.img_feat_dim = img_feat_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CXR encoder
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Image features [B, img_feat_dim]
        """
        return self.backbone(x)

class EMR_Encoder(nn.Module):
    """
    EMR Encoder using MLP with 2 hidden layers
    Architecture: input -> 128 -> 64 -> emr_feat_dim
    """
    
    def __init__(self, input_dim: int = 7, emr_feat_dim: int = 64, 
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.3):
        """
        Initialize EMR Encoder
        
        Args:
            input_dim: Input EMR feature dimension
            emr_feat_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(EMR_Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.emr_feat_dim = emr_feat_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(prev_dim, emr_feat_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EMR encoder
        
        Args:
            x: Input EMR features [B, input_dim]
            
        Returns:
            EMR features [B, emr_feat_dim]
        """
        return self.mlp(x)

class FusionModel(nn.Module):
    """
    Multimodal Fusion Model
    Combines image and EMR features for pneumonia classification
    """
    
    def __init__(self, img_feat_dim: int = 256, emr_feat_dim: int = 64, 
                 fusion_dim: int = 128, num_classes: int = 1, dropout: float = 0.3):
        """
        Initialize Fusion Model
        
        Args:
            img_feat_dim: Image feature dimension
            emr_feat_dim: EMR feature dimension
            fusion_dim: Fusion layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(FusionModel, self).__init__()
        
        self.img_feat_dim = img_feat_dim
        self.emr_feat_dim = emr_feat_dim
        self.fusion_dim = fusion_dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + emr_feat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, image_features: torch.Tensor, emr_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion model
        
        Args:
            image_features: Image features [B, img_feat_dim]
            emr_features: EMR features [B, emr_feat_dim]
            
        Returns:
            Tuple of (logits, probabilities)
        """
        # Concatenate features
        fused_features = torch.cat([image_features, emr_features], dim=1)
        
        # Fusion layer
        fused = self.fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused)
        
        # Apply sigmoid for binary classification
        probabilities = torch.sigmoid(logits)
        
        return logits, probabilities

class PneumoniaMultimodalModel(nn.Module):
    """
    Complete Multimodal Model for Pneumonia Diagnosis
    Combines CXR_Encoder, EMR_Encoder, and FusionModel
    """
    
    def __init__(self, img_feat_dim: int = 256, emr_feat_dim: int = 64, 
                 fusion_dim: int = 128, emr_input_dim: int = 7, 
                 num_classes: int = 1, dropout: float = 0.3, pretrained: bool = True):
        """
        Initialize complete multimodal model
        
        Args:
            img_feat_dim: Image feature dimension
            emr_feat_dim: EMR feature dimension
            fusion_dim: Fusion layer dimension
            emr_input_dim: EMR input dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            pretrained: Whether to use pretrained ResNet18
        """
        super(PneumoniaMultimodalModel, self).__init__()
        
        # Individual encoders
        self.cxr_encoder = CXR_Encoder(img_feat_dim=img_feat_dim, pretrained=pretrained)
        self.emr_encoder = EMR_Encoder(input_dim=emr_input_dim, emr_feat_dim=emr_feat_dim, dropout=dropout)
        
        # Fusion model
        self.fusion_model = FusionModel(
            img_feat_dim=img_feat_dim,
            emr_feat_dim=emr_feat_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Store dimensions for reference
        self.img_feat_dim = img_feat_dim
        self.emr_feat_dim = emr_feat_dim
        self.fusion_dim = fusion_dim
        
    def forward(self, image: torch.Tensor, emr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            image: Input images [B, 3, H, W]
            emr: Input EMR features [B, emr_input_dim]
            
        Returns:
            Tuple of (logits, probabilities)
        """
        # Encode image and EMR separately
        img_features = self.cxr_encoder(image)
        emr_features = self.emr_encoder(emr)
        
        # Fuse features and classify
        logits, probabilities = self.fusion_model(img_features, emr_features)
        
        return logits, probabilities
    
    def get_feature_embeddings(self, image: torch.Tensor, emr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature embeddings for analysis
        
        Args:
            image: Input images [B, 3, H, W]
            emr: Input EMR features [B, emr_input_dim]
            
        Returns:
            Dictionary containing all feature embeddings
        """
        with torch.no_grad():
            img_features = self.cxr_encoder(image)
            emr_features = self.emr_encoder(emr)
            fused_features = torch.cat([img_features, emr_features], dim=1)
            
            return {
                'image_features': img_features,
                'emr_features': emr_features,
                'fused_features': fused_features
            }
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters
        
        Returns:
            Dictionary with parameter counts for each component
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cxr_params = sum(p.numel() for p in self.cxr_encoder.parameters())
        emr_params = sum(p.numel() for p in self.emr_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_model.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'cxr_encoder': cxr_params,
            'emr_encoder': emr_params,
            'fusion_model': fusion_params
        }

def create_model(img_feat_dim: int = 256, emr_feat_dim: int = 64, 
                fusion_dim: int = 128, emr_input_dim: int = 7, 
                dropout: float = 0.3, pretrained: bool = True) -> PneumoniaMultimodalModel:
    """
    Factory function to create multimodal model
    
    Args:
        img_feat_dim: Image feature dimension
        emr_feat_dim: EMR feature dimension
        fusion_dim: Fusion layer dimension
        emr_input_dim: EMR input dimension
        dropout: Dropout rate
        pretrained: Whether to use pretrained ResNet18
        
    Returns:
        Initialized multimodal model
    """
    model = PneumoniaMultimodalModel(
        img_feat_dim=img_feat_dim,
        emr_feat_dim=emr_feat_dim,
        fusion_dim=fusion_dim,
        emr_input_dim=emr_input_dim,
        dropout=dropout,
        pretrained=pretrained
    )
    
    return model

def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights (for non-pretrained parts)
    
    Args:
        model: Model to initialize
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def main():
    """Test the model architecture"""
    print("🧪 Testing Multimodal Model Architecture...")
    
    # Create model
    model = create_model()
    
    # Print model info
    param_counts = model.count_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {param_counts['total']:,}")
    print(f"   Trainable: {param_counts['trainable']:,}")
    print(f"   CXR Encoder: {param_counts['cxr_encoder']:,}")
    print(f"   EMR Encoder: {param_counts['emr_encoder']:,}")
    print(f"   Fusion Model: {param_counts['fusion_model']:,}")
    
    # Test forward pass
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    emr = torch.randn(batch_size, 7)
    
    model.eval()
    with torch.no_grad():
        logits, probabilities = model(image, emr)
        
    print(f"\n✅ Forward Pass Test:")
    print(f"   Input image shape: {image.shape}")
    print(f"   Input EMR shape: {emr.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Output probabilities shape: {probabilities.shape}")
    print(f"   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

if __name__ == "__main__":
    main()
