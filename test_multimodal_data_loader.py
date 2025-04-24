"""
Test script for the multimodal data loader.
"""

import os
import sys
import pandas as pd
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.data_loader import load_data_paths, create_data_splits
from src.data_loader_multimodal import create_multimodal_data_loaders, MultimodalDocumentDataset

def visualize_batch(batch_inputs: Dict[str, torch.Tensor], batch_targets: torch.Tensor, class_names: List[str], num_samples: int = 4) -> None:
    """
    Visualize a batch of multimodal inputs and their corresponding targets.
    
    Parameters:
    -----------
    batch_inputs : Dict[str, torch.Tensor]
        Dictionary containing 'image' and 'text' tensors
    batch_targets : torch.Tensor
        Target labels
    class_names : List[str]
        List of class names
    num_samples : int, default=4
        Number of samples to visualize
    """
    # Ensure we don't try to visualize more samples than we have
    num_samples = min(num_samples, batch_inputs['image'].shape[0])
    
    # Create a figure with subplots for each sample
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Get image, text features, and label for this sample
        image = batch_inputs['image'][i].cpu().numpy()
        text_features = batch_inputs['text'][i].cpu().numpy()
        label_idx = batch_targets[i].item()
        label_name = class_names[label_idx]
        
        # Move channels from first dimension to last dimension for plotting
        image = np.transpose(image, (1, 2, 0))
        
        # Plot image
        axes[i].imshow(image)
        axes[i].set_title(f"Class: {label_name} (idx: {label_idx})")
        
        # Print TF-IDF info
        nonzero_features = np.count_nonzero(text_features)
        logger.info(f"Sample {i+1}: Class {label_name}, Text features shape: {text_features.shape}, Non-zero features: {nonzero_features}")
    
    plt.tight_layout()
    plt.savefig('multimodal_batch_visualization.png')
    logger.info(f"Batch visualization saved to multimodal_batch_visualization.png")
    plt.close(fig)

def test_multimodal_data_loader() -> None:
    """Test the multimodal data loader by loading a batch of data."""
    # Paths to data directories
    image_dir = os.path.join('data', 'images')
    ocr_dir = os.path.join('data', 'ocr')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    # Check if vectorizer exists
    if not os.path.exists(vectorizer_path):
        logger.error(f"TF-IDF vectorizer not found at {vectorizer_path}. Run the text feature extraction process first.")
        return
    
    # Load data paths
    logger.info("Loading data paths...")
    df = load_data_paths(image_dir, ocr_dir)
    
    # Create data splits
    logger.info("Creating data splits...")
    train_df, val_df, test_df = create_data_splits(df)
    
    # Create multimodal data loaders
    logger.info("Creating multimodal data loaders...")
    data_loaders = create_multimodal_data_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vectorizer_path=vectorizer_path,
        batch_size=8,
        num_workers=2  # Reduce worker count for testing
    )
    
    # Verify all loaders are created
    assert 'train' in data_loaders, "Training loader not created"
    assert 'val' in data_loaders, "Validation loader not created"
    assert 'test' in data_loaders, "Test loader not created"
    
    # Get class names
    class_names = sorted(train_df['label'].unique())
    logger.info(f"Class names: {class_names}")
    
    # Get a batch from the training loader
    logger.info("Getting a batch from the training loader...")
    train_batch = next(iter(data_loaders['train']))
    
    # Unpack the batch
    batch_inputs, batch_targets = train_batch
    
    # Print batch info
    logger.info(f"Batch inputs:")
    logger.info(f"  Image shape: {batch_inputs['image'].shape}")
    logger.info(f"  Text shape: {batch_inputs['text'].shape}")
    logger.info(f"Batch targets shape: {batch_targets.shape}")
    
    # Check that the batch has the expected shape
    batch_size = batch_inputs['image'].shape[0]
    assert batch_inputs['image'].shape[0] == batch_size, "Unexpected batch size for images"
    assert batch_inputs['text'].shape[0] == batch_size, "Unexpected batch size for text features"
    assert batch_targets.shape[0] == batch_size, "Unexpected batch size for targets"
    
    # Check that image tensors have the expected shape
    expected_image_shape = (3, 224, 224)  # (C, H, W)
    assert batch_inputs['image'].shape[1:] == expected_image_shape, f"Unexpected image shape: {batch_inputs['image'].shape[1:]} != {expected_image_shape}"
    
    # Check that targets are integer indices
    assert batch_targets.dtype == torch.long, f"Unexpected targets dtype: {batch_targets.dtype} != torch.long"
    
    # Visualize a batch
    logger.info("Visualizing a batch...")
    visualize_batch(batch_inputs, batch_targets, class_names)
    
    logger.info("Multimodal data loader test completed successfully!")

if __name__ == "__main__":
    test_multimodal_data_loader() 