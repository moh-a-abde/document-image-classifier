#!/usr/bin/env python
"""
Test script for text feature extraction pipeline.

This script loads data, splits it into train/val/test sets,
extracts text features using TF-IDF, and saves the vectorizer.
"""

import os
import sys
import logging
import argparse
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import project modules
from src.data_loader import load_data_paths, create_data_splits
from src.features.text_features import extract_text_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_features(X_train, y_train, output_dir='results/text_baseline'):
    """
    Visualize TF-IDF features by plotting feature importance per class.
    
    Parameters:
    -----------
    X_train : scipy.sparse.csr_matrix
        TF-IDF matrix for training data
    y_train : numpy.ndarray
        Training data labels
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert sparse matrix to dense for easier manipulation
    if sparse.issparse(X_train):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train
    
    # Count classes
    n_classes = len(np.unique(y_train))
    
    # Calculate average feature values per class
    plt.figure(figsize=(12, 8))
    
    # Display only top features for visibility
    top_n_features = 20
    
    for class_idx in range(n_classes):
        # Get samples for this class
        class_mask = y_train == class_idx
        
        # Calculate average feature values for this class
        if np.any(class_mask):
            avg_features = X_dense[class_mask].mean(axis=0)
            
            # Get indices of top features
            top_indices = np.argsort(avg_features)[-top_n_features:]
            
            # Plot feature importance
            plt.subplot(n_classes, 1, class_idx + 1)
            plt.barh(range(top_n_features), avg_features[top_indices])
            plt.title(f'Class {class_idx}: Top {top_n_features} TF-IDF Features')
            plt.ylabel('Feature Index')
            plt.xlabel('Average TF-IDF Value')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'tfidf_feature_importance.png')
    plt.savefig(output_path)
    logger.info("Feature importance visualization saved to %s", output_path)

def main():
    """
    Main function to test text feature extraction pipeline.
    """
    parser = argparse.ArgumentParser(description='Test text feature extraction pipeline')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--image_subdir', type=str, default='images', help='Image subdirectory name')
    parser.add_argument('--ocr_subdir', type=str, default='ocr', help='OCR text subdirectory name')
    parser.add_argument('--output_dir', type=str, default='results/text_baseline', help='Output directory for results')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of features for TF-IDF')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    X_train, _, _, y_train, _, _, label_map = extract_text_features(
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
    
    # Step 4: Save data split information
    output_info_path = os.path.join(args.output_dir, 'text_features_info.txt')
    splits_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'feature_dim': X_train.shape[1],
        'label_map': label_map
    }
    
    # Save as text file
    with open(output_info_path, 'w', encoding='utf-8') as f:
        for key, value in splits_info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Data split information saved to %s", output_info_path)
    
    # Step 5: Visualize features
    logger.info("Visualizing TF-IDF features...")
    try:
        visualize_features(X_train, y_train, args.output_dir)
    except Exception as e:
        logger.error("Error visualizing features: %s", str(e))
    
    logger.info("Text feature extraction pipeline test completed successfully!")

if __name__ == "__main__":
    main() 