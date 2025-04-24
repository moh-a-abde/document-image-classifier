#!/usr/bin/env python
"""
Simplified script to train and save the text model.
This script skips the full pipeline and just trains a basic model to test saving.
"""

import os
import sys
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import joblib

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import project modules
from models.text_model import TextClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Train a simple text classifier and save it to verify our saving mechanism works.
    """
    # Create model directory
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info("Loading sample data for testing...")
    # Load a small subset of 20 newsgroups data for quick testing
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                    remove=('headers', 'footers', 'quotes'),
                                    random_state=42)
    
    # Extract features with TF-IDF
    logger.info("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    
    logger.info(f"Data shape: {X.shape}")
    
    # Create and train classifier with simple parameters
    logger.info("Training classifier...")
    classifier = TextClassifier(C=1.0, penalty='l2', random_state=42)
    classifier.fit(X, y)
    
    # Save the model
    model_path = os.path.join(model_dir, 'text_baseline_best.pkl')
    logger.info(f"Saving model to {model_path}")
    classifier.save(model_path)
    
    # Verify the model was saved
    if os.path.exists(model_path):
        logger.info(f"SUCCESS: Model saved to {model_path}")
        logger.info(f"File size: {os.path.getsize(model_path)} bytes")
    else:
        logger.error(f"FAILED: Model not saved to {model_path}")
    
    # Also save the vectorizer
    vectorizer_path = os.path.join(model_dir, 'test_vectorizer.pkl')
    logger.info(f"Saving vectorizer to {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    # Try to load the model to verify it works
    try:
        logger.info(f"Testing model loading from {model_path}")
        loaded_classifier = TextClassifier.load(model_path)
        # Make a prediction to test the loaded model
        pred = loaded_classifier.predict(X[:5])
        logger.info(f"Model loaded and predicted: {pred}")
        logger.info("SUCCESS: Model loading and prediction test passed")
    except Exception as e:
        logger.error(f"FAILED: Error loading model: {str(e)}")

if __name__ == "__main__":
    main() 