"""
Text classification model for document classification.

This module provides a text classification model using scikit-learn's
LogisticRegression classifier with TF-IDF features.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Union, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextClassifier:
    """
    Text classifier using Logistic Regression with TF-IDF features.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = 'l2',
        class_weight: Union[str, Dict, None] = None,
        random_state: int = 42,
        solver: str = 'saga',
        max_iter: int = 1000,
        n_jobs: int = -1
    ):
        """
        Initialize the text classifier.
        
        Parameters:
        -----------
        C : float
            Inverse of regularization strength
        penalty : str
            Type of regularization ('l1', 'l2')
        class_weight : str, dict or None
            Class weights for imbalanced datasets
        random_state : int
            Random seed for reproducibility
        solver : str
            Algorithm to use in optimization
        max_iter : int
            Maximum number of iterations
        n_jobs : int
            Number of CPU cores to use (-1 for all)
        """
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            n_jobs=n_jobs,
            multi_class='ovr'  # one-vs-rest strategy
        )
        self.params = {
            'C': C,
            'penalty': penalty,
            'class_weight': class_weight,
            'random_state': random_state,
            'solver': solver,
            'max_iter': max_iter,
            'n_jobs': n_jobs
        }
        
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            TF-IDF features
        y : array-like
            Target labels
            
        Returns:
        --------
        self : TextClassifier
            Fitted classifier
        """
        logger.info("Fitting LogisticRegression classifier")
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            TF-IDF features
            
        Returns:
        --------
        array
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            TF-IDF features
            
        Returns:
        --------
        array
            Predicted class probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        logger.info(f"Saving model to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        TextClassifier
            Loaded classifier
        """
        logger.info(f"Loading model from {filepath}")
        model = joblib.load(filepath)
        classifier = cls()
        classifier.model = model
        return classifier
    
    def evaluate(self, X, y_true, class_names=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : array-like or sparse matrix
            TF-IDF features
        y_true : array-like
            True class labels
        class_names : list or None
            Class names for the classification report
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Generate classification report and confusion matrix
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'classification_report': report,
            'confusion_matrix': cm
        }


def tune_text_classifier(
    X_train, 
    y_train, 
    param_grid=None, 
    cv=5, 
    scoring='f1_macro', 
    n_jobs=-1,
    verbose=1
) -> Tuple[Dict[str, Any], LogisticRegression]:
    """
    Tune hyperparameters for the text classifier using GridSearchCV.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        TF-IDF features for training
    y_train : array-like
        Target labels for training
    param_grid : dict or None
        Grid of parameters to search
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for model selection
    n_jobs : int
        Number of CPU cores to use (-1 for all)
    verbose : int
        Verbosity level
        
    Returns:
    --------
    tuple
        Dictionary of best parameters and the best estimator
    """
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2', 'l1'],
            'class_weight': [None, 'balanced']
        }
    
    logger.info("Starting hyperparameter tuning with GridSearchCV")
    logger.info(f"Parameter grid: {param_grid}")
    
    # Initialize base classifier
    base_clf = LogisticRegression(
        solver='saga',
        max_iter=1000,
        random_state=42,
        multi_class='ovr'
    )
    
    # Setup grid search
    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_estimator_ 