"""
Script to train the image baseline model for document classification.

This script loads the data, sets up the training pipeline, trains the model,
and evaluates it on the validation set.
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import load_data_paths, create_data_splits
from src.training.train_image import (
    create_data_loaders, 
    setup_training, 
    train_one_epoch, 
    validate, 
    EarlyStopping, 
    save_checkpoint,
    load_checkpoint
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(train_df, val_df, args):
    """
    Train the image baseline model.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame with training data
    val_df : pd.DataFrame
        DataFrame with validation data
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Training history
    """
    # Create data loaders
    data_loaders = create_data_loaders(
        train_df, 
        val_df, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup model, criterion, optimizer, scheduler
    model, criterion, optimizer, scheduler = setup_training(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate
    )
    model = model.to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    # Record initial learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history['learning_rate'].append(current_lr)
    
    # Train the model
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train one epoch
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
        
        # Update learning rate scheduler
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log if learning rate changed
        if current_lr != prev_lr:
            logger.info(f"Learning rate adjusted: {prev_lr:.6f} -> {current_lr:.6f}")
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Save checkpoint if validation accuracy improved
        if early_stopping.best_score is None or val_acc > early_stopping.best_score:
            save_checkpoint(
                model, 
                optimizer, 
                epoch, 
                val_acc,
                filename=args.checkpoint_path
            )
        
        # Check early stopping
        if early_stopping(val_acc):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Plot training history
    if args.plot_history:
        plot_training_history(history, args.results_dir)
    
    return history


def plot_training_history(history, results_dir):
    """
    Plot training and validation metrics.
    
    Parameters:
    -----------
    history : dict
        Dictionary with training history
    results_dir : str
        Directory to save the plots
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot accuracy and loss
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate
    if 'learning_rate' in history:
        plt.subplot(1, 3, 3)
        plt.plot(history['learning_rate'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    logger.info(f"Training history plot saved to {os.path.join(results_dir, 'training_history.png')}")


def plot_confusion_matrix(y_true, y_pred, class_names, results_dir):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels
    class_names : list
        List of class names
    results_dir : str
        Directory to save the plot
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix plot saved to {os.path.join(results_dir, 'confusion_matrix.png')}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train image baseline model')
    
    # Data paths
    parser.add_argument('--image-dir', type=str, default='data/images',
                        help='Path to image data directory')
    parser.add_argument('--ocr-dir', type=str, default='data/ocr',
                        help='Path to OCR data directory')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    
    # Output paths
    parser.add_argument('--checkpoint-path', type=str, default='models/image_baseline_best.pt',
                        help='Path to save model checkpoint')
    parser.add_argument('--results-dir', type=str, default='results/image_baseline',
                        help='Directory to save results')
    
    # Options
    parser.add_argument('--plot-history', action='store_true',
                        help='Plot training history')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Log training configuration
    logger.info("Training configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Check if data splits exist, otherwise load and create splits
    splits_dir = 'results/splits'
    train_path = os.path.join(splits_dir, 'train.csv')
    val_path = os.path.join(splits_dir, 'val.csv')
    test_path = os.path.join(splits_dir, 'test.csv')
    
    if (os.path.exists(train_path) and 
        os.path.exists(val_path) and 
        os.path.exists(test_path)):
        logger.info("Loading existing data splits...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        logger.info("Creating new data splits...")
        # Load data
        df = load_data_paths(args.image_dir, args.ocr_dir)
        
        # Create splits
        train_df, val_df, test_df = create_data_splits(
            df, 
            train_size=0.7, 
            val_size=0.15, 
            test_size=0.15, 
            random_state=args.seed
        )
        
        # Save splits
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
    
    # Log dataset sizes
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Train the model
    logger.info("Starting model training...")
    start_time = datetime.now()
    
    history = train_model(train_df, val_df, args)
    
    end_time = datetime.now()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time}")
    
    # Save training metrics
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(os.path.join(args.results_dir, 'training_metrics.csv'), index=False)
    logger.info(f"Training metrics saved to {os.path.join(args.results_dir, 'training_metrics.csv')}")
    
    logger.info(f"Best model saved to {args.checkpoint_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main() 