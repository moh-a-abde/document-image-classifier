"""
Text feature extraction utilities for document classification.

This module provides functions to extract TF-IDF features from text data
for the document classification project.
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
from typing import Tuple, List, Dict, Union
from sklearn.feature_extraction.text import TfidfVectorizer

# Import from project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_text
from data_loader import load_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    text_col: str = 'text_path',
    label_col: str = 'label',
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.9,
    model_dir: str = 'models'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Extract TF-IDF features from text data in the train, validation, and test sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame containing training data paths and labels
    val_df : pd.DataFrame
        DataFrame containing validation data paths and labels
    test_df : pd.DataFrame
        DataFrame containing test data paths and labels
    text_col : str
        Column name containing text file paths
    label_col : str
        Column name containing labels
    max_features : int
        Maximum number of features (vocabulary size) for TF-IDF
    ngram_range : tuple
        Range of n-grams to be extracted (min_n, max_n)
    min_df : int or float
        Minimum document frequency for a term to be included
    max_df : float
        Maximum document frequency for a term to be included
    model_dir : str
        Directory to save the vectorizer
        
    Returns:
    --------
    Tuple containing:
        X_train: TF-IDF features for training set
        X_val: TF-IDF features for validation set
        X_test: TF-IDF features for test set
        y_train: Integer labels for training set
        y_val: Integer labels for validation set
        y_test: Integer labels for test set
        label_map: Dictionary mapping class names to indices
    """
    logger.info("Starting text feature extraction process")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Step 1: Load and preprocess text data
    logger.info("Loading and preprocessing text data")
    train_texts = _load_and_preprocess_texts(train_df[text_col])
    val_texts = _load_and_preprocess_texts(val_df[text_col])
    test_texts = _load_and_preprocess_texts(test_df[text_col])
    
    # Step 2: Create label mapping
    label_map = _create_label_mapping(train_df[label_col])
    logger.info(f"Label mapping: {label_map}")
    
    # Convert labels to integers
    y_train = train_df[label_col].map(label_map).values
    y_val = val_df[label_col].map(label_map).values
    y_test = test_df[label_col].map(label_map).values
    
    # Step 3: Initialize and fit TF-IDF vectorizer on training data only
    logger.info(f"Fitting TF-IDF vectorizer with max_features={max_features}, ngram_range={ngram_range}")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit vectorizer on training data
    X_train = vectorizer.fit_transform(train_texts)
    logger.info(f"TF-IDF matrix shape for training set: {X_train.shape}")
    
    # Transform validation and test data
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    logger.info(f"TF-IDF matrix shape for validation set: {X_val.shape}")
    logger.info(f"TF-IDF matrix shape for test set: {X_test.shape}")
    
    # Step 4: Save the vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")
    
    # Log feature extraction statistics
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_map

def _load_and_preprocess_texts(file_paths: pd.Series) -> List[str]:
    """
    Load text files and preprocess the content.
    
    Parameters:
    -----------
    file_paths : pd.Series
        Series containing paths to text files
        
    Returns:
    --------
    List[str]
        List of preprocessed text strings (tokens joined back into strings)
    """
    processed_texts = []
    
    for path in file_paths:
        # Load text
        raw_text = load_text(path)
        
        # Skip if text loading failed
        if raw_text is None:
            logger.warning(f"Failed to load text from {path}, using empty string instead")
            processed_texts.append('')
            continue
        
        # Preprocess text
        tokens = preprocess_text(
            raw_text,
            remove_stopwords=True,
            perform_stemming=False,
            perform_lemmatization=True
        )
        
        # Join tokens back into string for TF-IDF
        processed_text = ' '.join(tokens)
        processed_texts.append(processed_text)
    
    return processed_texts

def _create_label_mapping(labels: pd.Series) -> Dict[str, int]:
    """
    Create a mapping from label names to integer indices.
    
    Parameters:
    -----------
    labels : pd.Series
        Series containing label names
        
    Returns:
    --------
    Dict[str, int]
        Dictionary mapping label names to integer indices
    """
    unique_labels = sorted(labels.unique())
    return {label: idx for idx, label in enumerate(unique_labels)}

def load_vectorizer(model_path: str = 'models/tfidf_vectorizer.pkl') -> Union[TfidfVectorizer, None]:
    """
    Load a saved TF-IDF vectorizer.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved vectorizer
        
    Returns:
    --------
    TfidfVectorizer or None
        Loaded vectorizer or None if loading fails
    """
    try:
        vectorizer = joblib.load(model_path)
        logger.info(f"TF-IDF vectorizer loaded from {model_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Error loading vectorizer from {model_path}: {e}")
        return None 