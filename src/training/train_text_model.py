#!/usr/bin/env python
"""
Training script for text-only baseline model.

This script trains a logistic regression classifier on TF-IDF features
extracted from the OCR text data, performs hyperparameter tuning using
GridSearchCV, and evaluates the model on a test set.
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
from sklearn.metrics import confusion_matrix, classification_report

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import project modules
from features.text_features import extract_text_features, load_vectorizer
from models.text_model import TextClassifier, tune_text_classifier
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

def save_cv_results(grid_search, output_path):
    """
    Save cross-validation results to CSV.
    
    Parameters:
    -----------
    grid_search : GridSearchCV
        Fitted grid search object
    output_path : str
        Path to save the CSV file
    """
    # Convert CV results to DataFrame
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Select key columns
    cols_to_keep = [
        'mean_test_score', 'std_test_score', 
        'mean_train_score', 'std_train_score', 
        'param_C', 'param_penalty', 'param_class_weight',
        'rank_test_score'
    ]
    cols_to_keep = [col for col in cols_to_keep if col in cv_results.columns]
    
    cv_results = cv_results[cols_to_keep].sort_values('rank_test_score')
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv_results.to_csv(output_path, index=False)
    logger.info(f"Cross-validation results saved to {output_path}")

def main():
    """
    Main function to train and evaluate the text classifier.
    """
    parser = argparse.ArgumentParser(description='Train text classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--image_subdir', type=str, default='images', help='Image subdirectory name')
    parser.add_argument('--ocr_subdir', type=str, default='ocr', help='OCR text subdirectory name')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for results')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of features for TF-IDF')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'text_baseline'), exist_ok=True)
    
    # Step 1: Load data paths
    logger.info("Loading data paths...")
    image_dir = os.path.join(args.data_dir, args.image_subdir)
    ocr_dir = os.path.join(args.data_dir, args.ocr_subdir)
    df = load_data_paths(image_dir, ocr_dir)
    
    # Step 2: Create data splits
    logger.info("Creating train/val/test splits...")
    train_df, val_df, test_df = create_data_splits(df)
    
    # Step 3: Extract text features
    logger.info("Extracting text features...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = extract_text_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_col='text_path',
        label_col='label',
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        model_dir=args.model_dir
    )
    
    # Invert label mapping for readability
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]
    
    # Step 4: Hyperparameter tuning using GridSearchCV
    logger.info("Performing hyperparameter tuning...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2', 'l1'],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize and fit GridSearchCV
    best_params, best_model, grid_search = tune_text_classifier(
        X_train, 
        y_train, 
        param_grid=param_grid, 
        cv=args.cv, 
        scoring='f1_macro'
    )
    
    # Save grid search results
    grid_search_results_path = os.path.join(args.results_dir, 'text_lr_gridsearch.csv')
    save_cv_results(grid_search, grid_search_results_path)
    
    # Step 5: Create and save best model
    logger.info("Training final model with best parameters...")
    classifier = TextClassifier(**best_params)
    classifier.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(args.model_dir, 'text_baseline_best.pkl')
    classifier.save(model_path)
    
    # Step 6: Evaluate on test set
    logger.info("Evaluating model on test set...")
    metrics = classifier.evaluate(X_test, y_test, class_names)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.results_dir, 'text_baseline', 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    
    # Save evaluation results
    eval_path = os.path.join(args.results_dir, 'text_baseline_eval.txt')
    
    with open(eval_path, 'w') as f:
        f.write("TEXT-ONLY BASELINE MODEL EVALUATION\n")
        f.write("===================================\n\n")
        f.write(f"Model: LogisticRegression (one-vs-rest)\n")
        f.write(f"Best Parameters: {best_params}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, classifier.predict(X_test), target_names=class_names))
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
        f.write("\n\nFeature Dimensionality: ")
        f.write(f"{X_train.shape[1]} features (TF-IDF)")
    
    logger.info(f"Evaluation results saved to {eval_path}")
    logger.info("Text classifier training and evaluation completed!")

if __name__ == "__main__":
    main() 