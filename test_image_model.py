#!/usr/bin/env python

"""
Script to evaluate the trained image model (Task 2.4).
This script loads the best model saved during training and evaluates it on the test set.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data_paths, create_data_splits
from src.training.train_image import (
    create_data_loaders, 
    load_checkpoint,
    validate
)
from src.models.image_model import get_efficientnet_b0

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict(model, data_loader, device):
    """Get predictions from the model on the given data loader."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, results_dir):
    """Plot and save confusion matrix."""
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
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    logger.info(f"Confusion matrix saved to {os.path.join(results_dir, 'confusion_matrix.png')}")
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'))
    logger.info(f"Normalized confusion matrix saved to {os.path.join(results_dir, 'confusion_matrix_normalized.png')}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate image baseline model')
    
    # Model checkpoint
    parser.add_argument('--checkpoint-path', type=str, default='models/image_baseline_best.pt',
                        help='Path to model checkpoint')
    
    # Data paths
    parser.add_argument('--image-dir', type=str, default='data/images',
                        help='Path to image data directory')
    parser.add_argument('--ocr-dir', type=str, default='data/ocr',
                        help='Path to OCR data directory')
    
    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (cuda or cpu)')
    
    # Output paths
    parser.add_argument('--results-dir', type=str, default='results/image_baseline',
                        help='Directory to save results')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function to run evaluation."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available() and args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        
        # Ensure results directory exists
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Log system info
        logger.info(f"PyTorch version: {torch.__version__}")
        try:
            import numpy
            logger.info(f"NumPy version: {numpy.__version__}")
        except ImportError:
            logger.warning("NumPy not available")
        
        # Set device
        device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else "cpu")
        logger.info(f"Using device: {device}")
        
        # Check if model checkpoint exists
        if not os.path.exists(args.checkpoint_path):
            logger.error(f"Model checkpoint not found at {args.checkpoint_path}")
            # Create empty evaluation results
            with open(os.path.join(args.results_dir, 'image_baseline_eval.txt'), 'w') as f:
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: Model checkpoint not found at {args.checkpoint_path}\n")
            sys.exit(1)
        
        # Load test data
        splits_dir = 'results/splits'
        test_path = os.path.join(splits_dir, 'test.csv')
        
        if os.path.exists(test_path):
            logger.info("Loading existing test split...")
            try:
                test_df = pd.read_csv(test_path)
            except Exception as e:
                logger.error(f"Error loading test split: {e}")
                # Create empty evaluation results
                with open(os.path.join(args.results_dir, 'image_baseline_eval.txt'), 'w') as f:
                    f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Error: Failed to load test split: {e}\n")
                sys.exit(1)
        else:
            logger.error("Test split not found. Please run training first.")
            # Create empty evaluation results
            with open(os.path.join(args.results_dir, 'image_baseline_eval.txt'), 'w') as f:
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Error: Test split not found. Please run training first.\n")
            sys.exit(1)
        
        logger.info(f"Test set size: {len(test_df)}")
        
        # Create data loader for test set
        class_names = sorted(test_df['label'].unique())
        logger.info(f"Classes: {class_names}")
        
        # Create dummy DataFrames for train and val (required by create_data_loaders)
        # We'll only use the test loader
        dummy_df = pd.DataFrame({'image_path': [], 'text_path': [], 'label': []})
        
        try:
            # Create data loaders
            data_loaders = create_data_loaders(
                dummy_df,  # train_df (not used)
                dummy_df,  # val_df (not used)
                test_df=test_df,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # Create model
            model = get_efficientnet_b0(num_classes=len(class_names))
            
            # Load checkpoint
            logger.info(f"Loading model checkpoint from {args.checkpoint_path}")
            model, _, _, best_val_acc = load_checkpoint(model, filename=args.checkpoint_path)
            logger.info(f"Loaded model with validation accuracy: {best_val_acc:.2f}%")
            
            # Move model to device
            model = model.to(device)
            
            # Evaluate model
            logger.info("Evaluating model on test set...")
            predictions, true_labels = predict(model, data_loaders['test'], device)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions, target_names=class_names)
            
            # Log results
            logger.info(f"Test Accuracy: {accuracy*100:.2f}%")
            logger.info(f"Classification Report:\n{report}")
            
            # Save results to file
            results_file = os.path.join(args.results_dir, 'image_baseline_eval.txt')
            with open(results_file, 'w') as f:
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {args.checkpoint_path}\n")
                f.write(f"Test Set Size: {len(test_df)}\n")
                f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
                f.write("Classification Report:\n")
                f.write(report)
            
            logger.info(f"Evaluation results saved to {results_file}")
            
            # Plot confusion matrix
            try:
                plot_confusion_matrix(true_labels, predictions, class_names, args.results_dir)
            except Exception as e:
                logger.error(f"Error creating confusion matrix plot: {e}")
            
            logger.info("Evaluation complete!")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Create error evaluation file
            with open(os.path.join(args.results_dir, 'image_baseline_eval.txt'), 'w') as f:
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error during evaluation: {e}\n")
                f.write("\nTraceback:\n")
                import traceback
                f.write(traceback.format_exc())
            
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 