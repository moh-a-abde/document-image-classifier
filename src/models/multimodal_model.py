"""
Multimodal model architecture for document classification.

This module provides implementation of a multimodal model that combines
image features from EfficientNet-B0 and text features from TF-IDF 
vectorization for document classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import torchvision.models as models

class MultimodalDocumentClassifier(nn.Module):
    """
    Multimodal document classifier combining image and text features.
    
    This model uses intermediate fusion, with EfficientNet-B0 for image 
    feature extraction and a linear projection layer for TF-IDF text features.
    Features are concatenated and passed through an MLP head for classification.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        tfidf_dim: int = 10000,
        projection_dim: int = 128,
        dropout_rate: float = 0.4,
        freeze_image_backbone: bool = True
    ):
        """
        Initialize the multimodal model.
        
        Parameters:
        -----------
        num_classes : int, default=5
            Number of output classes
        tfidf_dim : int, default=10000
            Dimension of TF-IDF input features
        projection_dim : int, default=128
            Dimension for projecting image and text features before fusion
        dropout_rate : float, default=0.4
            Dropout rate for regularization
        freeze_image_backbone : bool, default=True
            Whether to freeze the image backbone weights
        """
        super(MultimodalDocumentClassifier, self).__init__()
        
        # Image feature extraction branch (EfficientNet-B0)
        self.image_feature_extractor = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        
        # Remove the final classifier
        self.image_feature_extractor.classifier = nn.Identity()
        
        # Get the feature dimension of EfficientNet-B0
        self.image_feature_dim = 1280  # EfficientNet-B0 feature dimension
        
        # Freeze the image backbone if specified
        if freeze_image_backbone:
            for param in self.image_feature_extractor.features.parameters():
                param.requires_grad = False
        
        # Project image features to a lower dimension
        self.image_projector = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.image_feature_dim, projection_dim),
            nn.ReLU()
        )
        
        # Text feature projection (from TF-IDF to common space)
        self.text_projector = nn.Sequential(
            nn.Linear(tfidf_dim, projection_dim),
            nn.ReLU()
        )
        
        # Combined feature dimension after concatenation
        combined_dim = projection_dim * 2  # Projected image + text features
        
        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters:
        -----------
        image : torch.Tensor
            Batch of input images, shape [batch_size, channels, height, width]
        text : torch.Tensor
            Batch of TF-IDF text features, shape [batch_size, tfidf_dim]
            
        Returns:
        --------
        torch.Tensor
            Class logits, shape [batch_size, num_classes]
        """
        # Extract image features
        image_features = self.image_feature_extractor(image)
        
        # Project image features
        image_embedding = self.image_projector(image_features)
        
        # Project text features
        text_embedding = self.text_projector(text)
        
        # Concatenate features
        combined_features = torch.cat([image_embedding, text_embedding], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits
    
    def unfreeze_last_blocks(self, num_blocks: int = 2):
        """
        Unfreeze the last n blocks of the EfficientNet backbone for fine-tuning.
        
        Parameters:
        -----------
        num_blocks : int, default=2
            Number of MBConv blocks to unfreeze from the end
        """
        # First, ensure all backbone is frozen
        for param in self.image_feature_extractor.features.parameters():
            param.requires_grad = False
        
        # Unfreeze the last n blocks
        # EfficientNet blocks are sequential, so we can count from the end
        total_blocks = 8  # EfficientNet-B0 has 8 blocks
        start_block = max(0, total_blocks - num_blocks)
        
        # Unfreeze specific blocks
        for i in range(start_block, total_blocks):
            for param in self.image_feature_extractor.features[i].parameters():
                param.requires_grad = True
                
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters in the model.
        
        Returns:
        --------
        Tuple[int, int]
            (total_parameters, trainable_parameters)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, trainable_params

def create_multimodal_model(
    num_classes: int = 5,
    tfidf_dim: int = 10000,
    projection_dim: int = 128,
    dropout_rate: float = 0.4,
    freeze_image_backbone: bool = True
) -> MultimodalDocumentClassifier:
    """
    Factory function to create a multimodal document classifier.
    
    Parameters:
    -----------
    num_classes : int, default=5
        Number of output classes
    tfidf_dim : int, default=10000
        Dimension of TF-IDF input features
    projection_dim : int, default=128
        Dimension for projecting image and text features before fusion
    dropout_rate : float, default=0.4
        Dropout rate for regularization
    freeze_image_backbone : bool, default=True
        Whether to freeze the image backbone weights
        
    Returns:
    --------
    MultimodalDocumentClassifier
        Initialized multimodal model
    """
    model = MultimodalDocumentClassifier(
        num_classes=num_classes,
        tfidf_dim=tfidf_dim,
        projection_dim=projection_dim,
        dropout_rate=dropout_rate,
        freeze_image_backbone=freeze_image_backbone
    )
    
    return model

def get_multimodal_config() -> Dict[str, Any]:
    """
    Return multimodal model configuration parameters.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with model configuration
    """
    return {
        "image_input_size": (224, 224),  # (height, width)
        "image_channels": 3,  # RGB
        "tfidf_dim": 10000,
        "projection_dim": 128,
        "num_classes": 5,
        "dropout_rate": 0.4,
        "normalization_mean": [0.485, 0.456, 0.406],  # ImageNet mean
        "normalization_std": [0.229, 0.224, 0.225],  # ImageNet std
        "model_type": "multimodal_efficientnet_tfidf"
    } 