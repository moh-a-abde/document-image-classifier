"""
Test script to check that the training pipeline for the image model is correctly implemented.

This script creates a small synthetic dataset and tries to run one epoch of training
to ensure that all the components of the training pipeline work together correctly.
"""

import os
import torch
import pandas as pd
import numpy as np
import logging
import argparse
import sys
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.models.image_model import get_efficientnet_b0, get_model_config
from src.training.train_image import (
    DocumentImageDataset, 
    create_data_loaders, 
    setup_training, 
    train_one_epoch, 
    validate,
    save_checkpoint
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_dataset(num_samples=10, num_classes=5):
    """
    Create a small synthetic dataset for testing the training pipeline.
    
    Parameters:
    -----------
    num_samples : int, default=10
        Number of samples per class
    num_classes : int, default=5
        Number of classes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic data
    """
    # Check if data directories exist
    image_dir = 'data/images'
    if not os.path.exists(image_dir):
        logger.error(f"Image directory '{image_dir}' not found. This test requires real data.")
        sys.exit(1)
    
    # Get class directories
    class_dirs = [d for d in os.listdir(image_dir) 
                  if os.path.isdir(os.path.join(image_dir, d)) and not d.startswith('.')]
    
    if len(class_dirs) == 0:
        logger.error(f"No class directories found in '{image_dir}'. This test requires real data.")
        sys.exit(1)
    
    # Limit to num_classes
    class_dirs = class_dirs[:num_classes]
    
    # Initialize lists for DataFrame
    image_paths = []
    labels = []
    
    # For each class, get a few sample images
    for class_name in class_dirs:
        class_dir = os.path.join(image_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith('.TIF') and not f.startswith('.')]
        
        # Limit to num_samples
        image_files = image_files[:num_samples]
        
        # Add to lists
        for img_file in image_files:
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_name)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    logger.info(f"Created synthetic dataset with {len(df)} samples across {len(df['label'].unique())} classes")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def test_dataset_and_dataloader():
    """Test the DocumentImageDataset and DataLoader."""
    logger.info("Testing DocumentImageDataset and DataLoader...")
    
    # Create synthetic dataset
    df = create_synthetic_dataset()
    
    # Get model config
    model_config = get_model_config()
    
    # Create dataset
    dataset = DocumentImageDataset(
        df, 
        target_size=model_config['input_size'],
        num_channels=model_config['num_channels']
    )
    
    # Test __len__
    assert len(dataset) == len(df), f"Dataset length mismatch: {len(dataset)} != {len(df)}"
    
    # Test __getitem__
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), f"Expected tensor, got {type(img)}"
    assert img.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {img.shape}"
    assert isinstance(label, int), f"Expected int, got {type(label)}"
    
    # Test DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    assert len(batch) == 2, f"Expected 2 items in batch, got {len(batch)}"
    assert batch[0].shape == (2, 3, 224, 224), f"Expected shape (2, 3, 224, 224), got {batch[0].shape}"
    assert batch[1].shape == (2,), f"Expected shape (2,), got {batch[1].shape}"
    
    logger.info("DocumentImageDataset and DataLoader tests passed!")
    
    return df

def test_model_and_optimizer():
    """Test model creation and optimizer setup."""
    logger.info("Testing model and optimizer setup...")
    
    # Create model and optimizer
    model, criterion, optimizer, scheduler = setup_training(num_classes=5, learning_rate=1e-3)
    
    # Check model
    assert isinstance(model, torch.nn.Module), f"Expected nn.Module, got {type(model)}"
    
    # Check criterion
    assert isinstance(criterion, torch.nn.CrossEntropyLoss), f"Expected CrossEntropyLoss, got {type(criterion)}"
    
    # Check optimizer
    assert isinstance(optimizer, torch.optim.Adam), f"Expected Adam, got {type(optimizer)}"
    
    # Check scheduler
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau), \
        f"Expected ReduceLROnPlateau, got {type(scheduler)}"
    
    logger.info("Model and optimizer setup tests passed!")
    
    return model, criterion, optimizer, scheduler

def test_training_loop(df):
    """Test the training loop with a small synthetic dataset."""
    logger.info("Testing training loop...")
    
    # Split the dataset
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_df, 
        val_df, 
        batch_size=4,
        num_workers=0  # Use 0 for debugging
    )
    
    # Set device (use CPU for testing)
    device = torch.device("cpu")
    
    # Create model and optimizer
    model, criterion, optimizer, scheduler = setup_training(num_classes=5, learning_rate=1e-3)
    model = model.to(device)
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, 
        data_loaders['train'], 
        criterion, 
        optimizer, 
        device
    )
    
    # Validate
    val_loss, val_acc = validate(
        model, 
        data_loaders['val'], 
        criterion, 
        device
    )
    
    logger.info(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Check that loss and accuracy are reasonable
    assert 0 <= train_loss <= 100, f"Train loss out of range: {train_loss}"
    assert 0 <= train_acc <= 100, f"Train accuracy out of range: {train_acc}"
    assert 0 <= val_loss <= 100, f"Validation loss out of range: {val_loss}"
    assert 0 <= val_acc <= 100, f"Validation accuracy out of range: {val_acc}"
    
    # Test saving checkpoint
    save_checkpoint(
        model, 
        optimizer, 
        1, 
        val_acc,
        filename='models/test_checkpoint.pt'
    )
    
    # Check that checkpoint file exists
    assert os.path.exists('models/test_checkpoint.pt'), "Checkpoint file was not created"
    
    # Clean up
    os.remove('models/test_checkpoint.pt')
    
    logger.info("Training loop tests passed!")

def main():
    """Run all tests."""
    logger.info("Starting training pipeline tests...")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/image_baseline', exist_ok=True)
    
    # Test dataset and dataloader
    df = test_dataset_and_dataloader()
    
    # Test model and optimizer
    test_model_and_optimizer()
    
    # Test training loop
    test_training_loop(df)
    
    logger.info("All tests passed!")

if __name__ == '__main__':
    main() 