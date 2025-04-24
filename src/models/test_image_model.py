"""
Test script for image model implementation.
"""

import torch
from src.models.image_model import get_efficientnet_b0, unfreeze_last_blocks, get_model_config

def test_efficientnet_model():
    """Test EfficientNet model creation and forward pass"""
    # Get model configuration
    config = get_model_config()
    print(f"Model configuration: {config}")
    
    # Create model
    model = get_efficientnet_b0(num_classes=config["num_classes"], freeze_backbone=True)
    print(f"Model created: {model.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, config["num_channels"], *config["input_size"])
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, config["num_classes"]), "Output shape mismatch"
    
    # Test unfreezing
    model = unfreeze_last_blocks(model, num_blocks=2)
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_params_after:,} ({trainable_params_after/total_params:.2%})")
    assert trainable_params_after > trainable_params, "Unfreezing failed"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_efficientnet_model() 