"""
CXR + Real EMR Fusion Model for Pneumonia Classification
Combines chest X-ray images with real EMR features for multimodal classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Tuple

class CXR_Encoder(nn.Module):
    """
    Chest X-ray encoder using pretrained ResNet18
    
    Architecture:
    - Pretrained ResNet18 (ImageNet weights)
    - Remove final FC layer
    - Add projection layer to 256-dim features
    """
    
    def __init__(self, projection_dim: int = 256):
        super(CXR_Encoder, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final FC layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Add adaptive pooling and projection
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(resnet.fc.in_features, projection_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CXR encoder
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            features: CXR features of shape (batch_size, projection_dim)
        """
        # Extract features
        x = self.features(x)  # (batch_size, 512, H', W')
        
        # Global average pooling
        x = self.adaptive_pool(x)  # (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)    # (batch_size, 512)
        
        # Project to desired dimension
        x = self.projection(x)     # (batch_size, projection_dim)
        x = self.dropout(x)
        
        return x

class RealEMR_Encoder(nn.Module):
    """
    Real EMR encoder using MLP
    
    Architecture:
    - Input: number of real EMR features
    - Hidden layers: 128 → 64 → 32
    - Output: 64-dim features
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], 
                 output_dim: int = 64, dropout_rate: float = 0.3):
        super(RealEMR_Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output projection
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Real EMR encoder
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            features: EMR features of shape (batch_size, output_dim)
        """
        return self.mlp(x)

class FusionHead(nn.Module):
    """
    Fusion head for combining CXR and Real EMR features
    
    Architecture:
    - Concatenate [CXR_features, EMR_features]
    - Linear(320) → ReLU → Dropout → Linear(1) → Sigmoid
    """
    
    def __init__(self, cxr_dim: int = 256, emr_dim: int = 64, 
                 hidden_dim: int = 128, dropout_rate: float = 0.3):
        super(FusionHead, self).__init__()
        
        self.fusion_dim = cxr_dim + emr_dim
        
        self.fusion_net = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, cxr_features: torch.Tensor, emr_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for fusion head
        
        Args:
            cxr_features: CXR features of shape (batch_size, cxr_dim)
            emr_features: EMR features of shape (batch_size, emr_dim)
        
        Returns:
            logits: Raw logits of shape (batch_size, 1)
            probabilities: Sigmoid probabilities of shape (batch_size, 1)
        """
        # Concatenate features
        fused_features = torch.cat([cxr_features, emr_features], dim=1)
        
        # Forward through fusion network
        logits = self.fusion_net(fused_features)
        
        # Apply sigmoid for probabilities
        probabilities = torch.sigmoid(logits)
        
        return logits, probabilities

class CXRRealEMRFusionModel(nn.Module):
    """
    Complete CXR + Real EMR fusion model
    
    Architecture:
    - CXR_Encoder: ResNet18 → 256-dim features
    - RealEMR_Encoder: MLP → 64-dim features  
    - FusionHead: [256 + 64] → 128 → 1
    """
    
    def __init__(self, emr_input_dim: int, cxr_projection_dim: int = 256, 
                 emr_projection_dim: int = 64, fusion_hidden_dim: int = 128,
                 dropout_rate: float = 0.3):
        super(CXRRealEMRFusionModel, self).__init__()
        
        self.cxr_encoder = CXR_Encoder(projection_dim=cxr_projection_dim)
        self.emr_encoder = RealEMR_Encoder(
            input_dim=emr_input_dim,
            output_dim=emr_projection_dim,
            dropout_rate=dropout_rate
        )
        self.fusion_head = FusionHead(
            cxr_dim=cxr_projection_dim,
            emr_dim=emr_projection_dim,
            hidden_dim=fusion_hidden_dim,
            dropout_rate=dropout_rate
        )
    
    def forward(self, cxr_images: torch.Tensor, emr_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the complete fusion model
        
        Args:
            cxr_images: CXR images of shape (batch_size, 3, H, W)
            emr_features: EMR features of shape (batch_size, emr_input_dim)
        
        Returns:
            logits: Raw logits of shape (batch_size, 1)
            probabilities: Sigmoid probabilities of shape (batch_size, 1)
        """
        # Encode CXR images
        cxr_features = self.cxr_encoder(cxr_images)
        
        # Encode EMR features
        emr_features_encoded = self.emr_encoder(emr_features)
        
        # Fuse features and predict
        logits, probabilities = self.fusion_head(cxr_features, emr_features_encoded)
        
        return logits, probabilities
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by component
        cxr_params = sum(p.numel() for p in self.cxr_encoder.parameters() if p.requires_grad)
        emr_params = sum(p.numel() for p in self.emr_encoder.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion_head.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'cxr_encoder_parameters': cxr_params,
            'emr_encoder_parameters': emr_params,
            'fusion_head_parameters': fusion_params,
            'architecture': f"CXR(ResNet18→256) + EMR(MLP→64) → Fusion(320→128→1)"
        }

def create_cxr_real_emr_fusion_model(emr_input_dim: int, cxr_projection_dim: int = 256,
                                    emr_projection_dim: int = 64, fusion_hidden_dim: int = 128,
                                    dropout_rate: float = 0.3) -> CXRRealEMRFusionModel:
    """
    Create CXR + Real EMR fusion model
    
    Args:
        emr_input_dim: Number of input EMR features
        cxr_projection_dim: CXR feature dimension (default: 256)
        emr_projection_dim: EMR feature dimension (default: 64)
        fusion_hidden_dim: Fusion hidden dimension (default: 128)
        dropout_rate: Dropout rate (default: 0.3)
    
    Returns:
        CXRRealEMRFusionModel instance
    """
    return CXRRealEMRFusionModel(
        emr_input_dim=emr_input_dim,
        cxr_projection_dim=cxr_projection_dim,
        emr_projection_dim=emr_projection_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout_rate=dropout_rate
    )

if __name__ == "__main__":
    # Test model creation
    emr_input_dim = 13  # Example: 13 real EMR features
    model = create_cxr_real_emr_fusion_model(emr_input_dim)
    
    print("CXR + Real EMR Fusion Model Created Successfully!")
    print(f"Model Info: {model.get_model_info()}")
    
    # Test forward pass
    batch_size = 4
    dummy_cxr = torch.randn(batch_size, 3, 224, 224)  # CXR images
    dummy_emr = torch.randn(batch_size, emr_input_dim)  # EMR features
    
    logits, probabilities = model(dummy_cxr, dummy_emr)
    
    print(f"\nForward pass test:")
    print(f"CXR input shape: {dummy_cxr.shape}")
    print(f"EMR input shape: {dummy_emr.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities range: [{probabilities.min().item():.3f}, {probabilities.max().item():.3f}]")
    
    print("\n✅ CXR + Real EMR fusion model tests completed!")



