"""
Test script for multimodal model implementation.
"""

import torch
import sys
import os

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.multimodal_model import (
    create_multimodal_model,
    get_multimodal_config,
    MultimodalDocumentClassifier
)

def test_multimodal_model():
    """Test the multimodal model creation and forward pass"""
    print("Testing MultimodalDocumentClassifier")
    
    # Get model configuration
    config = get_multimodal_config()
    print(f"Model configuration: {config}")
    
    # Create model
    model = create_multimodal_model(
        num_classes=config["num_classes"],
        tfidf_dim=config["tfidf_dim"],
        projection_dim=config["projection_dim"],
        dropout_rate=config["dropout_rate"],
        freeze_image_backbone=True
    )
    print(f"Model created: {model.__class__.__name__}")
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Test forward pass
    batch_size = 4
    image_input = torch.randn(batch_size, config["image_channels"], *config["image_input_size"])
    text_input = torch.randn(batch_size, config["tfidf_dim"])  # TF-IDF features
    
    print(f"Image input shape: {image_input.shape}")
    print(f"Text input shape: {text_input.shape}")
    
    with torch.no_grad():
        output = model(image_input, text_input)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, config["num_classes"]), "Output shape mismatch"
    
    # Test unfreezing
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters before unfreezing: {frozen_params:,}")
    
    model.unfreeze_last_blocks(num_blocks=2)
    
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_after = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"Trainable parameters after unfreezing: {trainable_params_after:,} ({trainable_params_after/total_params:.2%})")
    print(f"Frozen parameters after unfreezing: {frozen_params_after:,}")
    
    assert trainable_params_after > trainable_params, "Unfreezing failed"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_multimodal_model() 