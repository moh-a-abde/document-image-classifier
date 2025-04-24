#!/usr/bin/env python
"""
Evaluation script for text-only baseline model.

This script loads a trained text classifier and evaluates it on the test set,
generating detailed metrics, confusion matrix, and class-wise performance analysis.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import project modules
from features.text_features import extract_text_features, load_vectorizer
from models.text_model import TextClassifier
from data_loader import load_data_paths, create_data_splits

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix array
    class_names : list
        List of class names
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Confusion matrix saved to {output_path}")

def plot_class_performance(metrics, class_names, output_path):
    """
    Plot per-class performance metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing classification report
    class_names : list
        List of class names
    output_path : str
        Path to save the plot
    """
    report = metrics['classification_report']
    
    # Extract per-class metrics
    class_metrics = {cls: report[cls] for cls in class_names}
    
    # Create DataFrame for plotting
    df = pd.DataFrame(class_metrics).T
    df = df.reset_index().rename(columns={'index': 'class'})
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    bar_width = 0.25
    x = np.arange(len(class_names))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i*bar_width, df[metric], width=bar_width, label=metric)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x + bar_width, class_names)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    logger.info(f"Class performance plot saved to {output_path}")

def main():
    """
    Main function to evaluate the text classifier.
    """
    parser = argparse.ArgumentParser(description='Evaluate text classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--image_subdir', type=str, default='images', help='Image subdirectory name')
    parser.add_argument('--ocr_subdir', type=str, default='ocr', help='OCR text subdirectory name')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for results')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where models are saved')
    parser.add_argument('--model_path', type=str, default='models/text_baseline_best.pkl', help='Path to the trained model')
    parser.add_argument('--vectorizer_path', type=str, default='models/tfidf_vectorizer.pkl', help='Path to the TF-IDF vectorizer')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.join(args.results_dir, 'text_baseline'), exist_ok=True)
    
    # Step 1: Load data paths
    logger.info("Loading data paths...")
    image_dir = os.path.join(args.data_dir, args.image_subdir)
    ocr_dir = os.path.join(args.data_dir, args.ocr_subdir)
    df = load_data_paths(image_dir, ocr_dir)
    
    # Step 2: Create data splits
    logger.info("Creating train/val/test splits...")
    train_df, val_df, test_df = create_data_splits(df)
    
    # Step 3: Get label mapping
    label_map = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]
    
    # Step 4: Load vectorizer and model
    logger.info(f"Loading TF-IDF vectorizer from {args.vectorizer_path}")
    vectorizer = load_vectorizer(args.vectorizer_path)
    if vectorizer is None:
        logger.error("Failed to load vectorizer. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loading text classifier from {args.model_path}")
    classifier = TextClassifier.load(args.model_path)
    if classifier is None:
        logger.error("Failed to load classifier. Exiting.")
        sys.exit(1)
    
    # Step 5: Prepare test data
    logger.info("Preparing test data...")
    from src.features.text_features import _load_and_preprocess_texts
    
    # Preprocess test texts
    test_texts = _load_and_preprocess_texts(test_df['text_path'])
    
    # Transform texts using loaded vectorizer
    X_test = vectorizer.transform(test_texts)
    
    # Convert labels to integers
    y_test = test_df['label'].map(label_map).values
    
    # Step 6: Evaluate on test set
    logger.info("Evaluating model on test set...")
    metrics = classifier.evaluate(X_test, y_test, class_names)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.results_dir, 'text_baseline', 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    
    # Plot class performance
    class_perf_path = os.path.join(args.results_dir, 'text_baseline', 'class_performance.png')
    plot_class_performance(metrics, class_names, class_perf_path)
    
    # Save detailed evaluation results
    eval_path = os.path.join(args.results_dir, 'text_baseline_eval.txt')
    
    with open(eval_path, 'w') as f:
        f.write("TEXT-ONLY BASELINE MODEL EVALUATION\n")
        f.write("===================================\n\n")
        f.write(f"Model: LogisticRegression (one-vs-rest)\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
        f.write("\n\nFeature Dimensionality: ")
        f.write(f"{X_test.shape[1]} features (TF-IDF)")
    
    logger.info(f"Evaluation results saved to {eval_path}")
    logger.info("Text classifier evaluation completed!")

if __name__ == "__main__":
    main() 