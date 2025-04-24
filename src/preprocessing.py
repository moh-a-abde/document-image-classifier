"""
Preprocessing utilities for image and text data.
"""

import numpy as np
from typing import List, Tuple, Union
import cv2
import os
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path_or_array: Union[str, np.ndarray], 
                     target_size: Tuple[int, int] = (224, 224),
                     num_channels: int = 3) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Parameters:
    -----------
    image_path_or_array : str or np.ndarray
        Path to the image file or a numpy array containing the image
    target_size : tuple of int
        Target size (height, width) for the output image
    num_channels : int
        Number of channels in the output image (1 for grayscale, 3 for RGB)
        
    Returns:
    --------
    np.ndarray
        Preprocessed image as a numpy array
    """
    # Load image if path is provided
    if isinstance(image_path_or_array, str):
        try:
            img = cv2.imread(image_path_or_array)
            if img is None:
                logger.error(f"Failed to load image: {image_path_or_array}")
                return None
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_array}: {e}")
            return None
    else:
        img = image_path_or_array.copy()
    
    # Check if image is loaded successfully
    if img is None or img.size == 0:
        logger.error(f"Empty or invalid image")
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3 and img.shape[2] == 3 and num_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert from grayscale to RGB if needed
    if len(img.shape) == 2 and num_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Resize the image
    if img.shape[:2] != target_size:
        # Choose appropriate interpolation method
        # INTER_AREA is better for shrinking, INTER_CUBIC for enlarging
        if img.shape[0] > target_size[0] or img.shape[1] > target_size[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=interpolation)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Ensure correct shape
    if num_channels == 1 and len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    return img


def preprocess_text(text: str, 
                    remove_stopwords: bool = True, 
                    perform_stemming: bool = False,
                    perform_lemmatization: bool = True) -> List[str]:
    """
    Preprocess text data for model input.
    
    Parameters:
    -----------
    text : str
        Raw OCR text to process
    remove_stopwords : bool, default=True
        Whether to remove common English stopwords
    perform_stemming : bool, default=False
        Whether to apply stemming (reducing words to their stem)
    perform_lemmatization : bool, default=True
        Whether to apply lemmatization (reducing words to their base form)
        
    Returns:
    --------
    List[str]
        List of cleaned tokens
    """
    # Check if text is None or empty
    if text is None or not text.strip():
        logger.warning("Empty or None text received")
        return []
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation, numbers, and special characters
        # Keep only letters and whitespace
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple tokenization by splitting on whitespace
        # This avoids the need for nltk.word_tokenize which was causing issues
        tokens = text.split()
        
        # Remove stopwords if requested
        if remove_stopwords:
            try:
                # Try to get stopwords
                try:
                    nltk.download('stopwords', quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK stopwords: {e}")
                
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            except Exception as e:
                logger.warning(f"Stopword removal failed: {e}")
                logger.warning("Continuing without stopword removal")
        
        # Perform stemming if requested (not applied if lemmatization is used)
        if perform_stemming and not perform_lemmatization:
            try:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(token) for token in tokens]
            except Exception as e:
                logger.warning(f"Stemming failed: {e}")
                logger.warning("Continuing without stemming")
        
        # Perform lemmatization if requested
        if perform_lemmatization:
            try:
                # Try using spaCy for lemmatization (faster for multiple tokens)
                nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                doc = nlp(" ".join(tokens))
                tokens = [token.lemma_ for token in doc]
            except Exception as e:
                logger.warning(f"SpaCy lemmatization failed: {e}. Falling back to NLTK")
                try:
                    # Fallback to NLTK lemmatizer
                    lemmatizer = WordNetLemmatizer()
                    try:
                        nltk.download('wordnet', quiet=True)
                    except Exception:
                        logger.warning("Failed to download wordnet, lemmatization may be limited")
                    tokens = [lemmatizer.lemmatize(token) for token in tokens]
                except Exception as e2:
                    logger.error(f"NLTK lemmatization also failed: {e2}")
                    logger.warning("Continuing without lemmatization")
        
        # Filter out single character tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return [] 