#!/usr/bin/env python

"""
Script to test the data loaders for the image model.
This helps verify that the data loaders are working correctly before running the full training.
"""

import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data_paths, create_data_splits
from src.training.train_image import create_data_loaders
from src.models.image_model import get_model_config

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image with mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def display_batch(images, labels, class_names):
    """Display a batch of images with their labels."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, (img, label) in enumerate(zip(images[:8], labels[:8])):
        img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize if needed
        if img.min() < 0 or img.max() > 1:
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {class_names[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_loader_test.png')
    print(f"Saved sample batch to data_loader_test.png")

def main():
    print("Testing data loaders for the image model...")
    
    # Load data splits
    splits_dir = 'results/splits'
    train_path = os.path.join(splits_dir, 'train.csv')
    val_path = os.path.join(splits_dir, 'val.csv')
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Loading existing data splits...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    else:
        print("Creating new data splits...")
        # Load data
        df = load_data_paths('data/images', 'data/ocr')
        
        # Create splits
        train_df, val_df, _ = create_data_splits(
            df, 
            train_size=0.7, 
            val_size=0.15, 
            test_size=0.15, 
            random_state=42
        )
        
        # Save splits
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Get class names
    class_names = sorted(train_df['label'].unique())
    print(f"Classes: {class_names}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_df, 
        val_df, 
        batch_size=8,
        num_workers=2,
        image_config=get_model_config()
    )
    
    # Get a batch from the training loader
    for images, labels in data_loaders['train']:
        print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")
        print(f"Data types - Images: {images.dtype}, Labels: {labels.dtype}")
        print(f"First few labels: {labels[:5].tolist()}")
        
        # Display the images
        display_batch(images, labels, class_names)
        break
    
    print("Data loaders test completed successfully!")

if __name__ == "__main__":
    main() 