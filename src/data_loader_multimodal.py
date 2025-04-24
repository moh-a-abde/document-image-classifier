"""
Multimodal data loading utilities for document classification.

This module provides classes and functions to load and preprocess both image
and text data for the multimodal document classification model.
"""

import os
import torch
import numpy as np
import pandas as pd
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalDocumentDataset(Dataset):
    """
    Dataset for multimodal document classification including both image and text features.
    
    This dataset loads images and extracts TF-IDF features from preprocessed text files
    for use with the multimodal model.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        vectorizer_path: str,
        target_size: Tuple[int, int] = (224, 224),
        num_channels: int = 3,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Any] = None,
        precomputed_text_features: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize the multimodal dataset.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            DataFrame with image paths, text paths, and labels
        vectorizer_path : str
            Path to the trained TF-IDF vectorizer
        target_size : tuple, default=(224, 224)
            Target image size (height, width)
        num_channels : int, default=3
            Number of image channels (3 for RGB, 1 for grayscale)
        class_to_idx : dict or None
            Mapping from class names to indices
        transform : object or None
            Optional transforms to apply to images
        precomputed_text_features : dict or None
            Optional dictionary mapping text paths to precomputed TF-IDF features
        """
        self.data_df = data_df
        self.target_size = target_size
        self.num_channels = num_channels
        self.transform = transform
        self.precomputed_text_features = precomputed_text_features
        
        # Load the TF-IDF vectorizer
        self.vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Loaded TF-IDF vectorizer from {vectorizer_path}")
        
        # Create class to index mapping if not provided
        if class_to_idx is None:
            unique_classes = sorted(data_df['label'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
            
        logger.info(f"Dataset initialized with {len(data_df)} samples and {len(self.class_to_idx)} classes")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the sample
            
        Returns:
        --------
        tuple
            Dictionary containing 'image' and 'text' tensors, and the class label tensor
        """
        # Get the row from the dataframe
        row = self.data_df.iloc[idx]
        
        # Get label
        label = row['label']
        label_idx = self.class_to_idx[label]
        
        # Load and preprocess image
        image_path = row['image_path']
        image = self._load_and_preprocess_image(image_path)
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert image to tensor
        image_tensor = torch.FloatTensor(image)
        
        # Load and vectorize text
        text_path = row['text_path']
        
        # Use precomputed features if available
        if self.precomputed_text_features is not None and text_path in self.precomputed_text_features:
            text_features = self.precomputed_text_features[text_path]
        else:
            text_features = self._load_and_vectorize_text(text_path)
        
        # Convert text features to tensor
        text_tensor = torch.FloatTensor(text_features)
        
        # Return a dictionary of inputs and the label
        return {'image': image_tensor, 'text': text_tensor}, torch.tensor(label_idx, dtype=torch.long)
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        np.ndarray
            Preprocessed image as a numpy array
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
                # Return a blank image if loading fails
                return np.zeros((self.target_size[0], self.target_size[1], self.num_channels), dtype=np.float32)
            
            # Resize
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            
            # Convert to RGB if needed
            if self.num_channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Transpose from (H, W, C) to (C, H, W) for PyTorch
            img = np.transpose(img, (2, 0, 1))
            
            return img
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a blank image on error
            return np.zeros((self.num_channels, self.target_size[0], self.target_size[1]), dtype=np.float32)
    
    def _load_and_vectorize_text(self, text_path: str) -> np.ndarray:
        """
        Load and vectorize text using the TF-IDF vectorizer.
        
        Parameters:
        -----------
        text_path : str
            Path to the text file
            
        Returns:
        --------
        np.ndarray
            TF-IDF feature vector
        """
        try:
            # Load text
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Handle empty text
            if not text.strip():
                logger.warning(f"Empty text in {text_path}")
                return np.zeros(len(self.vectorizer.get_feature_names_out()), dtype=np.float32)
                
            # Vectorize
            text_vector = self.vectorizer.transform([text]).toarray()[0]
            return text_vector.astype(np.float32)
            
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            try:
                with open(text_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                text_vector = self.vectorizer.transform([text]).toarray()[0]
                return text_vector.astype(np.float32)
            except Exception as e:
                logger.error(f"Error with latin-1 encoding for {text_path}: {e}")
        except Exception as e:
            logger.error(f"Error vectorizing text {text_path}: {e}")
        
        # Return zeros on error
        return np.zeros(len(self.vectorizer.get_feature_names_out()), dtype=np.float32)


def create_multimodal_data_loaders(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: Optional[pd.DataFrame] = None,
    vectorizer_path: str = 'models/tfidf_vectorizer.pkl',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    num_channels: int = 3
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for multimodal training, validation, and optionally test sets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame with training data
    val_df : pd.DataFrame
        DataFrame with validation data
    test_df : pd.DataFrame, optional
        DataFrame with test data, if available
    vectorizer_path : str
        Path to the trained TF-IDF vectorizer
    batch_size : int, default=32
        Batch size for the data loaders
    num_workers : int, default=4
        Number of worker processes for data loading
    image_size : tuple, default=(224, 224)
        Target image size (height, width)
    num_channels : int, default=3
        Number of image channels
        
    Returns:
    --------
    Dict[str, DataLoader]
        Dictionary with data loaders for 'train', 'val', and optionally 'test'
    """
    # Verify the vectorizer exists
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
    
    # Create class to index mapping from the training set
    classes = sorted(train_df['label'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Create datasets
    train_dataset = MultimodalDocumentDataset(
        train_df, 
        vectorizer_path=vectorizer_path,
        target_size=image_size,
        num_channels=num_channels,
        class_to_idx=class_to_idx
    )
    
    val_dataset = MultimodalDocumentDataset(
        val_df, 
        vectorizer_path=vectorizer_path,
        target_size=image_size,
        num_channels=num_channels,
        class_to_idx=class_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create test loader if test data is provided
    if test_df is not None:
        test_dataset = MultimodalDocumentDataset(
            test_df, 
            vectorizer_path=vectorizer_path,
            target_size=image_size,
            num_channels=num_channels,
            class_to_idx=class_to_idx
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders['test'] = test_loader
    
    return loaders 