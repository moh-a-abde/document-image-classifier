"""
Script to evaluate the trained image model on the test set.

This script loads the trained model, evaluates it on the test set,
and generates evaluation metrics such as accuracy, precision, recall,
F1-score, and confusion matrix.
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import sys
from datetime import datetime
import json

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import load_data_paths, create_data_splits
from src.models.image_model import get_efficientnet_b0, get_model_config
from src.training.train_image import create_data_loaders, load_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(test_df, args):
    """
    Evaluate the trained model on the test set.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        DataFrame with test data
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Get model configuration
    model_config = get_model_config()
    
    # Create data loaders
    data_loaders = create_data_loaders(
        train_df=test_df,  # Using test_df for both to ensure class mapping consistency
        val_df=test_df,    # (we're only using the test loader)
        test_df=test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_config=model_config
    )
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = get_efficientnet_b0(num_classes=args.num_classes)
    model = model.to(device)
    
    # Load best checkpoint
    model, _, _, best_val_acc = load_checkpoint(model, filename=args.checkpoint_path)
    logger.info(f"Loaded checkpoint with validation accuracy: {best_val_acc:.2f}%")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists for predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Get class names from the DataLoader's dataset
    # If no test_df classes are present, use [0, 1, 2, 3, 4] as a fallback
    # Since we're expecting 5 classes from 0 to 4
    try:
        class_names = list(data_loaders['test'].dataset.class_to_idx.keys())
        class_idx = [data_loaders['test'].dataset.class_to_idx[cls] for cls in class_names]
    except (KeyError, AttributeError):
        class_names = ["0", "2", "4", "6", "9"]  # Default class names based on directory names
        class_idx = list(range(len(class_names)))
    
    # Ensure class names and indices are sorted correctly
    sorted_idx = np.argsort(class_idx)
    class_names = [class_names[i] for i in sorted_idx]
    
    logger.info(f"Evaluating on {len(test_df)} test samples with classes: {class_names}")
    
    # Evaluate on test set
    with torch.no_grad():
        for inputs, labels in data_loaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Create results dictionary
    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # Save results to file
    save_results(results, args.results_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, args.results_dir)
    
    return results


def save_results(results, results_dir):
    """
    Save evaluation results to a file.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    results_dir : str
        Directory to save results
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results to JSON file
    results_file = os.path.join(results_dir, 'evaluation_results.json')
    
    # Format the results for JSON serialization
    # Convert numpy arrays to lists
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results[k] = v.tolist()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save text report
    report_file = os.path.join(results_dir, 'image_baseline_eval.txt')
    with open(report_file, 'w') as f:
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        
        # Format classification report for text file
        report = results['classification_report']
        headers = ["precision", "recall", "f1-score", "support"]
        
        # Write headers
        f.write(f"{'':20} {headers[0]:>10} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10}\n\n")
        
        # Write per-class metrics
        for cls in results['class_names']:
            if cls in report:
                metrics = report[cls]
                f.write(f"{cls:20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                        f"{metrics['f1-score']:>10.4f} {metrics['support']:>10}\n")
        
        # Write average metrics
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                metrics = report[avg_type]
                f.write(f"\n{avg_type:20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                        f"{metrics['f1-score']:>10.4f} {metrics['support']:>10}\n")
    
    logger.info(f"Results saved to {results_file} and {report_file}")


def plot_confusion_matrix(cm, class_names, results_dir):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        List of class names
    results_dir : str
        Directory to save the plot
    """
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    cm_file = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_file)
    logger.info(f"Confusion matrix saved to {cm_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate image baseline model')
    
    # Data paths
    parser.add_argument('--image-dir', type=str, default='data/images',
                        help='Path to image data directory')
    parser.add_argument('--ocr-dir', type=str, default='data/ocr',
                        help='Path to OCR data directory')
    
    # Model parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation (cuda or cpu)')
    
    # Input/output paths
    parser.add_argument('--checkpoint-path', type=str, default='models/image_baseline_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--results-dir', type=str, default='results/image_baseline',
                        help='Directory to save evaluation results')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function to evaluate the image baseline model."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(args.results_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting image baseline model evaluation")
    logger.info(f"Arguments: {args}")
    
    # Load data
    logger.info("Loading data...")
    df = load_data_paths(args.image_dir, args.ocr_dir)
    
    # Create data splits
    logger.info("Creating data splits...")
    _, _, test_df = create_data_splits(df, random_state=args.seed)
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(test_df, args)
    
    logger.info("Evaluation complete!")
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")


if __name__ == '__main__':
    main() 