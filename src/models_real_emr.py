"""
Real EMR Model for Pneumonia Classification
MLP-based classifier using only EMR features from real dataset
"""

import torch
import torch.nn as nn
from typing import Tuple

class RealEMR_MLP(nn.Module):
    """
    Real EMR-only pneumonia classification model using MLP
    
    Architecture:
    - Input: number of processed EMR features
    - Hidden layers: 128 → 64 → 32
    - Activation: ReLU
    - Regularization: BatchNorm + Dropout(0.3)
    - Output: 1 neuron with Sigmoid for binary classification
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], 
                 dropout_rate: float = 0.3, num_classes: int = 1):
        super(RealEMR_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Create sequential model
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            logits: Raw logits of shape (batch_size, 1)
            probabilities: Sigmoid probabilities of shape (batch_size, 1)
        """
        # Forward through MLP
        logits = self.mlp(x)  # (batch_size, 1)
        
        # Apply sigmoid for probabilities
        probabilities = torch.sigmoid(logits)  # (batch_size, 1)
        
        return logits, probabilities
    
    def get_feature_dim(self) -> int:
        """Get the input feature dimension"""
        return self.input_dim
    
    def get_hidden_dims(self) -> list:
        """Get the hidden layer dimensions"""
        return self.hidden_dims

def create_real_emr_model(input_dim: int, hidden_dims: list = [128, 64, 32], 
                         dropout_rate: float = 0.3, num_classes: int = 1) -> RealEMR_MLP:
    """
    Create Real EMR MLP model
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout_rate: Dropout rate (default: 0.3)
        num_classes: Number of output classes (default: 1 for binary)
    
    Returns:
        RealEMR_MLP instance
    """
    return RealEMR_MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )

def count_parameters(model: RealEMR_MLP) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(model: RealEMR_MLP) -> dict:
    """Get model information"""
    total_params = count_parameters(model)
    
    # Count parameters by layer type
    linear_params = sum(p.numel() for name, p in model.named_parameters() 
                       if 'weight' in name and 'linear' in name.lower())
    bn_params = sum(p.numel() for name, p in model.named_parameters() 
                   if 'batch_norm' in name.lower())
    
    return {
        'total_parameters': total_params,
        'linear_parameters': linear_params,
        'batch_norm_parameters': bn_params,
        'input_dim': model.get_feature_dim(),
        'hidden_dims': model.get_hidden_dims(),
        'architecture': f"Input({model.get_feature_dim()}) -> {' -> '.join(map(str, model.get_hidden_dims()))} -> Output(1)"
    }

def create_model_from_config(config: dict) -> RealEMR_MLP:
    """
    Create model from configuration dictionary
    
    Args:
        config: Dictionary containing model configuration
    
    Returns:
        RealEMR_MLP instance
    """
    return create_real_emr_model(
        input_dim=config.get('input_dim', 10),
        hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        dropout_rate=config.get('dropout_rate', 0.3),
        num_classes=config.get('num_classes', 1)
    )

if __name__ == "__main__":
    # Test model creation
    input_dim = 10  # Example: 10 EMR features
    model = create_real_emr_model(input_dim)
    
    print("Real EMR Model Created Successfully!")
    print(f"Model Info: {get_model_info(model)}")
    
    # Test forward pass
    dummy_input = torch.randn(4, input_dim)  # Batch of 4 samples
    logits, probabilities = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Probabilities range: [{probabilities.min().item():.3f}, {probabilities.max().item():.3f}]")
    
    # Test with different configurations
    print("\nTesting different configurations:")
    
    # Smaller model
    small_model = create_real_emr_model(input_dim, hidden_dims=[64, 32], dropout_rate=0.2)
    print(f"Small model: {get_model_info(small_model)}")
    
    # Larger model
    large_model = create_real_emr_model(input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.4)
    print(f"Large model: {get_model_info(large_model)}")
    
    print("✅ Real EMR model tests completed!")



