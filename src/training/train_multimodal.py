"""
Training script for the multimodal document classification model.

This script implements the training procedure for the multimodal model
that combines image and text features for document classification.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from project modules
from src.data_loader import load_data_paths, create_data_splits
from src.data_loader_multimodal import create_multimodal_data_loaders
from src.models.multimodal_model import create_multimodal_model, get_multimodal_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None,
    num_epochs: int = 30,
    device: str = 'cuda',
    patience: int = 5,
    model_save_path: str = 'models/multimodal_best.pt',
    unfreeze_epoch: Optional[int] = 5,
    results_dir: str = 'results/multimodal'
) -> Dict[str, List]:
    """
    Train the multimodal model.
    
    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    model : nn.Module
        PyTorch model to train
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    scheduler : Any, optional
        Learning rate scheduler
    num_epochs : int, default=30
        Maximum number of epochs to train for
    device : str, default='cuda'
        Device to use for training ('cuda' or 'cpu')
    patience : int, default=5
        Number of epochs to wait for improvement before early stopping
    model_save_path : str, default='models/multimodal_best.pt'
        Path to save the best model weights
    unfreeze_epoch : int, optional
        Epoch at which to unfreeze the last blocks of the image backbone
    results_dir : str, default='results/multimodal'
        Directory to save results
        
    Returns:
    --------
    Dict[str, List]
        Dictionary containing training metrics history
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize variables for training
    start_time = time.time()
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Unfreeze layers if this is the specified epoch
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            logger.info("Unfreezing last blocks of image backbone...")
            model.unfreeze_last_blocks(num_blocks=2)
            # Verify model parameter status
            total_params, trainable_params = model.count_parameters()
            logger.info(f"Trainable parameters after unfreezing: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            images = inputs['image'].to(device)
            texts = inputs['text'].to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images, texts)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                          f"Acc: {100.0 * train_correct / train_total:.2f}%")
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_targets = []
        all_val_predictions = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Move data to device
                images = inputs['image'].to(device)
                texts = inputs['text'].to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images, texts)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Store targets and predictions for F1 score
                all_val_targets.extend(targets.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_f1 = f1_score(all_val_targets, all_val_predictions, average='macro')
        
        # Step learning rate scheduler if provided
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_f1)
            history['learning_rate'].append(current_lr)
        else:
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print epoch results
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve_count = 0
            
            # Save model
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved! (Val F1: {val_f1:.4f})")
            
            # Save validation metrics at best epoch
            best_metrics = {
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch + 1,
                'confusion_matrix': confusion_matrix(all_val_targets, all_val_predictions)
            }
            
            # Create validation classification report
            class_names = [str(i) for i in range(len(np.unique(all_val_targets)))]
            class_report = classification_report(all_val_targets, all_val_predictions, 
                                                target_names=class_names, output_dict=True)
            
            # Save confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(all_val_targets, all_val_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix (Validation)')
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save normalized confusion matrix
            plt.figure(figsize=(10, 8))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Normalized Confusion Matrix (Validation)')
            plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            no_improve_count += 1
        
        # Check early stopping
        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    # Plot training history
    plot_training_history(history, results_dir)
    
    # Save training metrics
    pd.DataFrame(history).to_csv(os.path.join(results_dir, 'training_metrics.csv'), index=False)
    
    return history

def plot_training_history(history: Dict[str, List], save_dir: str) -> None:
    """
    Plot training history metrics.
    
    Parameters:
    -----------
    history : Dict[str, List]
        Dictionary containing training metrics history
    save_dir : str
        Directory to save the plots
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy and F1 score
    axes[1].plot(history['train_acc'], label='Training Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].plot(history['val_f1'], label='Validation F1 (Macro)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percentage / Score')
    axes[1].set_title('Training and Validation Metrics')
    axes[1].legend()
    axes[1].grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot learning rate
    plt.figure(figsize=(10, 4))
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(
    test_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None,
    results_dir: str = 'results/multimodal'
) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test set.
    
    Parameters:
    -----------
    test_loader : DataLoader
        DataLoader for test data
    model : nn.Module
        Trained PyTorch model to evaluate
    criterion : nn.Module
        Loss function
    device : str, default='cuda'
        Device to use for evaluation ('cuda' or 'cpu')
    class_names : List[str], optional
        Names of the classes
    results_dir : str, default='results/multimodal'
        Directory to save evaluation results
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing evaluation metrics
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Initialize variables
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_targets = []
    all_test_predictions = []
    all_test_probabilities = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Move data to device
            images = inputs['image'].to(device)
            texts = inputs['text'].to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images, texts)
            loss = criterion(outputs, targets)
            
            # Calculate probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Update statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # Store targets and predictions for metrics
            all_test_targets.extend(targets.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())
            all_test_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * test_correct / test_total
    test_f1 = f1_score(all_test_targets, all_test_predictions, average='macro')
    
    # Print results
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test F1 Score (Macro): {test_f1:.4f}")
    
    # Use default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(all_test_targets)))]
    
    # Generate and print classification report
    class_report = classification_report(all_test_targets, all_test_predictions, 
                                        target_names=class_names, output_dict=True)
    logger.info("Classification Report:")
    logger.info(classification_report(all_test_targets, all_test_predictions, 
                                    target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(all_test_targets, all_test_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test)')
    plt.savefig(os.path.join(results_dir, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (Test)')
    plt.savefig(os.path.join(results_dir, 'test_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot class-wise performance
    class_metrics = pd.DataFrame(class_report).transpose()
    class_metrics = class_metrics.drop('accuracy')  # Remove accuracy row
    
    plt.figure(figsize=(12, 8))
    class_metrics[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar')  # Exclude 'macro avg' and 'weighted avg'
    plt.title('Class-wise Performance (Test)')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'test_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(os.path.join(results_dir, 'multimodal_eval.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Test F1 Score (Macro): {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_test_targets, all_test_predictions, 
                                    target_names=class_names))
    
    # Create results dictionary
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'targets': all_test_targets,
        'predictions': all_test_predictions,
        'probabilities': all_test_probabilities
    }
    
    return results

def main():
    """Main function to run the training and evaluation process."""
    parser = argparse.ArgumentParser(description='Train and evaluate the multimodal document classification model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty)')
    parser.add_argument('--unfreeze_epoch', type=int, default=5, help='Epoch at which to unfreeze image backbone')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--results_dir', type=str, default='results/multimodal', help='Directory to save results')
    parser.add_argument('--model_save_path', type=str, default='models/multimodal_best.pt', help='Path to save best model')
    parser.add_argument('--vectorizer_path', type=str, default='models/tfidf_vectorizer.pkl', help='Path to TF-IDF vectorizer')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Paths to data directories
    image_dir = os.path.join('data', 'images')
    ocr_dir = os.path.join('data', 'ocr')
    
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
        vectorizer_path=args.vectorizer_path,
        batch_size=args.batch_size,
        num_workers=4  # Adjust based on your system
    )
    
    # Get model configuration
    config = get_multimodal_config()
    
    # Create model
    logger.info("Creating multimodal model...")
    model = create_multimodal_model(
        num_classes=config["num_classes"],
        tfidf_dim=config["tfidf_dim"],
        projection_dim=config["projection_dim"],
        dropout_rate=config["dropout_rate"],
        freeze_image_backbone=True  # Will unfreeze during training
    )
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Train model
    logger.info("Starting model training...")
    history = train_model(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        patience=args.patience,
        model_save_path=args.model_save_path,
        unfreeze_epoch=args.unfreeze_epoch,
        results_dir=args.results_dir
    )
    
    # Load best model for evaluation
    logger.info(f"Loading best model from {args.model_save_path}")
    model.load_state_dict(torch.load(args.model_save_path))
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    class_names = sorted(train_df['label'].unique())
    test_results = evaluate_model(
        test_loader=data_loaders['test'],
        model=model,
        criterion=criterion,
        device=args.device,
        class_names=class_names,
        results_dir=args.results_dir
    )
    
    logger.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 