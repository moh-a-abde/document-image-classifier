"""
Preprocessing utilities for image and text data.
"""

import numpy as np
from typing import List, Tuple, Union
import cv2
import os
import logging

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


def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text data for model input.
    
    Parameters:
    -----------
    text : str
        Raw OCR text to process
        
    Returns:
    --------
    List[str]
        List of cleaned tokens
    """
    # This will be implemented in Task 1.5
    pass 