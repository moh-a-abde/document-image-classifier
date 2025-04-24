"""
Training pipeline for the image classification model.

This module implements the training, validation, and evaluation routines for 
the image-only baseline model for document classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import cv2
import sys
from typing import Dict, List, Tuple, Union, Optional, Any

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.preprocessing import preprocess_image
from src.models.image_model import get_efficientnet_b0, get_model_config, unfreeze_last_blocks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentImageDataset(Dataset):
    """
    PyTorch Dataset for document images.
    
    This dataset loads and preprocesses images on-the-fly from the provided
    DataFrame containing image paths and labels.
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 transform: Optional[Any] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 num_channels: int = 3,
                 class_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns 'image_path' and 'label'
        transform : callable, optional
            Optional transform to be applied to the images
        target_size : Tuple[int, int], default=(224, 224)
            Target size for the images (height, width)
        num_channels : int, default=3
            Number of channels (3 for RGB, 1 for grayscale)
        class_to_idx : Dict[str, int], optional
            Mapping from class names to indices. If None, it will be created.
        """
        self.df = df
        self.transform = transform
        self.target_size = target_size
        self.num_channels = num_channels
        
        # Create class to index mapping if not provided
        if class_to_idx is None:
            classes = sorted(df['label'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        # For PyTorch models, we need to transpose from HWC to CHW
        self.need_transpose = True
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Class mapping: {self.class_to_idx}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample to get
            
        Returns:
        --------
        Tuple[torch.Tensor, int]
            Tuple containing the preprocessed image and its label
        """
        # Get image path and label
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = self.class_to_idx[row['label']]
        
        # Load and preprocess image
        img = preprocess_image(img_path, self.target_size, self.num_channels)
        
        # Apply additional transforms if provided
        if self.transform:
            img = self.transform(img)
        
        # Convert to tensor and transpose if needed
        img_tensor = torch.FloatTensor(img)
        if self.need_transpose and img_tensor.dim() == 3:
            # Transpose from HWC to CHW
            img_tensor = img_tensor.permute(2, 0, 1)
        
        return img_tensor, label


def create_data_loaders(train_df: pd.DataFrame, 
                        val_df: pd.DataFrame, 
                        test_df: Optional[pd.DataFrame] = None,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        image_config: Optional[Dict[str, Any]] = None) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and optionally test sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame with training data
    val_df : pd.DataFrame
        DataFrame with validation data
    test_df : pd.DataFrame, optional
        DataFrame with test data, if available
    batch_size : int, default=32
        Batch size for the data loaders
    num_workers : int, default=4
        Number of worker processes for data loading
    image_config : Dict[str, Any], optional
        Configuration parameters for the image preprocessing
        
    Returns:
    --------
    Dict[str, DataLoader]
        Dictionary with data loaders for 'train', 'val', and optionally 'test'
    """
    if image_config is None:
        image_config = get_model_config()
    
    # Get image dimensions and channels
    target_size = image_config['input_size']
    num_channels = image_config['num_channels']
    
    # Create class to index mapping from the training set
    classes = sorted(train_df['label'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Create datasets
    train_dataset = DocumentImageDataset(
        train_df, 
        target_size=target_size, 
        num_channels=num_channels,
        class_to_idx=class_to_idx
    )
    
    val_dataset = DocumentImageDataset(
        val_df, 
        target_size=target_size, 
        num_channels=num_channels,
        class_to_idx=class_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create test loader if test data is provided
    if test_df is not None:
        test_dataset = DocumentImageDataset(
            test_df, 
            target_size=target_size, 
            num_channels=num_channels,
            class_to_idx=class_to_idx
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders['test'] = test_loader
    
    return loaders


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    This class implements early stopping by monitoring a validation metric
    and stopping training if it doesn't improve for a specified number of epochs.
    """
    
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 mode: str = 'max'):
        """
        Initialize early stopping.
        
        Parameters:
        -----------
        patience : int, default=5
            Number of epochs to wait after last improvement
        min_delta : float, default=0.0
            Minimum change to qualify as an improvement
        mode : str, default='max'
            'min' or 'max' depending on whether we want to minimize or maximize the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set comparison function based on mode
        self.improved = self._improved_min if mode == 'min' else self._improved_max
    
    def _improved_max(self, score: float) -> bool:
        """Check if score improved for 'max' mode."""
        return self.best_score is None or score > self.best_score + self.min_delta
    
    def _improved_min(self, score: float) -> bool:
        """Check if score improved for 'min' mode."""
        return self.best_score is None or score < self.best_score - self.min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped.
        
        Parameters:
        -----------
        score : float
            Current value of the monitored metric
            
        Returns:
        --------
        bool
            True if training should be stopped
        """
        if self.improved(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model: nn.Module, 
                    train_loader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Parameters:
    -----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        DataLoader for the training set
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    Tuple[float, float]
        (training loss, training accuracy)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': 100 * correct / total})
    
    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return train_loss, train_acc


def validate(model: nn.Module, 
             val_loader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """
    Validate the model.
    
    Parameters:
    -----------
    model : nn.Module
        The model to validate
    val_loader : DataLoader
        DataLoader for the validation set
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to use for validation
        
    Returns:
    --------
    Tuple[float, float]
        (validation loss, validation accuracy)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update metrics
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def setup_training(num_classes: int = 5, learning_rate: float = 1e-3) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Set up the model, loss function, optimizer, and scheduler.
    
    Parameters:
    -----------
    num_classes : int, default=5
        Number of output classes
    learning_rate : float, default=1e-3
        Initial learning rate for the optimizer
        
    Returns:
    --------
    Tuple
        (model, criterion, optimizer, scheduler)
    """
    # Create model
    model = get_efficientnet_b0(num_classes=num_classes, freeze_backbone=True)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3
    )
    
    return model, criterion, optimizer, scheduler


def save_checkpoint(model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int, 
                    val_acc: float,
                    filename: str = 'models/image_baseline_best.pt') -> None:
    """
    Save model checkpoint.
    
    Parameters:
    -----------
    model : nn.Module
        The model to save
    optimizer : torch.optim.Optimizer
        The optimizer
    epoch : int
        Current epoch
    val_acc : float
        Validation accuracy
    filename : str, default='models/image_baseline_best.pt'
        Path to save the checkpoint
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, filename)
    
    logger.info(f"Checkpoint saved to {filename}")


def load_checkpoint(model: nn.Module, 
                    optimizer: Optional[torch.optim.Optimizer] = None, 
                    filename: str = 'models/image_baseline_best.pt') -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load model checkpoint.
    
    Parameters:
    -----------
    model : nn.Module
        The model to load weights into
    optimizer : torch.optim.Optimizer, optional
        The optimizer to load state into
    filename : str, default='models/image_baseline_best.pt'
        Path to the checkpoint file
        
    Returns:
    --------
    Tuple
        (model, optimizer, epoch, val_acc)
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    
    logger.info(f"Loaded checkpoint from {filename} (epoch {epoch}, val_acc {val_acc:.2f}%)")
    
    return model, optimizer, epoch, val_acc


if __name__ == "__main__":
    # This is a placeholder for when this script is run directly
    # The actual training loop will be implemented in train_image_model.py
    logger.info("This module provides training components for the image model.")
    logger.info("For training execution, run the train_image_model.py script.") 