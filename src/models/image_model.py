"""
Image model architecture for document classification.

This module provides implementations of image models for document 
classification, focusing on transfer learning approaches with PyTorch.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any

def get_efficientnet_b0(num_classes: int = 5, freeze_backbone: bool = True) -> nn.Module:
    """
    Create a transfer learning model based on EfficientNet-B0 pre-trained on ImageNet.
    
    Parameters:
    -----------
    num_classes : int, default=5
        Number of output classes
    freeze_backbone : bool, default=True
        Whether to freeze the backbone weights
        
    Returns:
    --------
    nn.Module
        PyTorch model with EfficientNet-B0 backbone and custom classification head
    """
    # Load pre-trained EfficientNet-B0 model
    efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze the backbone if specified
    if freeze_backbone:
        for param in efficientnet.features.parameters():
            param.requires_grad = False
    
    # Replace the classifier
    # EfficientNet B0 already has an AdaptiveAvgPool2d, so we don't need to include it
    # The output of the pooling is [B, 1280, 1, 1]
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )
    
    return efficientnet

def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2) -> nn.Module:
    """
    Unfreeze the last n blocks of the EfficientNet backbone for fine-tuning.
    
    Parameters:
    -----------
    model : nn.Module
        The EfficientNet model
    num_blocks : int, default=2
        Number of MBConv blocks to unfreeze from the end
        
    Returns:
    --------
    nn.Module
        Modified model with unfrozen blocks
    """
    # EfficientNet-B0 has 8 blocks (0-7)
    # We want to unfreeze the last num_blocks
    
    # First, ensure all backbone is frozen
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Unfreeze the last n blocks
    # EfficientNet blocks are sequential, so we can count from the end
    total_blocks = 8  # EfficientNet-B0 has 8 blocks
    start_block = max(0, total_blocks - num_blocks)
    
    # Unfreeze specific blocks
    for i in range(start_block, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True
    
    return model

def get_model_config() -> Dict[str, Any]:
    """
    Return model configuration parameters.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with model configuration
    """
    return {
        "input_size": (224, 224),  # (height, width)
        "num_channels": 3,  # RGB
        "num_classes": 5,
        "normalization_mean": [0.485, 0.456, 0.406],  # ImageNet mean
        "normalization_std": [0.229, 0.224, 0.225],  # ImageNet std
        "model_type": "efficientnet_b0"
    } 