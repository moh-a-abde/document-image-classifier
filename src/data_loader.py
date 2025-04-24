"""
Data loading utilities for the document classification project.
"""

import os
import glob
from typing import Dict, List, Tuple, Union
import pandas as pd
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_paths(image_dir: str, ocr_dir: str) -> pd.DataFrame:
    """
    Load paths to image and OCR text files along with their class labels.
    
    Parameters:
    -----------
    image_dir : str
        Path to the directory containing image subdirectories for each class
    ocr_dir : str
        Path to the directory containing OCR text subdirectories for each class
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['image_path', 'text_path', 'label']
    """
    # Check if directories exist
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(ocr_dir):
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")
    
    # Initialize lists to store data
    image_paths = []
    text_paths = []
    labels = []
    
    # Get all class directories (subdirectories in the image_dir)
    class_dirs = [d for d in os.listdir(image_dir) 
                  if os.path.isdir(os.path.join(image_dir, d)) and not d.startswith('.')]
    
    # Keep track of files that don't have matching pairs
    mismatches = []
    
    # Iterate through each class directory
    for class_name in class_dirs:
        logger.info(f"Processing class: {class_name}")
        
        # Get image paths for this class
        img_class_dir = os.path.join(image_dir, class_name)
        ocr_class_dir = os.path.join(ocr_dir, class_name)
        
        # Check if the corresponding OCR directory exists
        if not os.path.exists(ocr_class_dir):
            logger.warning(f"OCR directory not found for class {class_name}")
            continue
        
        # Get all image files in this class directory
        img_files = glob.glob(os.path.join(img_class_dir, "*.TIF"))
        
        # Process each image file
        for img_path in img_files:
            # Extract the base filename
            base_filename = os.path.basename(img_path)
            
            # Construct the expected text file path
            text_path = os.path.join(ocr_class_dir, f"{base_filename}.txt")
            
            # Check if the text file exists
            if os.path.exists(text_path):
                image_paths.append(img_path)
                text_paths.append(text_path)
                labels.append(class_name)
            else:
                mismatches.append((img_path, "No matching text file"))
    
    # Check for text files that don't have matching image files
    for class_name in class_dirs:
        ocr_class_dir = os.path.join(ocr_dir, class_name)
        if not os.path.exists(ocr_class_dir):
            continue
            
        txt_files = glob.glob(os.path.join(ocr_class_dir, "*.TIF.txt"))
        
        for txt_path in txt_files:
            # Extract the base filename (without .txt)
            base_filename = os.path.basename(txt_path)
            base_filename = re.sub(r'\.txt$', '', base_filename)  # Remove .txt extension
            
            # Construct the expected image file path
            img_path = os.path.join(image_dir, class_name, base_filename)
            
            # Check if the image file exists and if we haven't already added this pair
            if not os.path.exists(img_path) and txt_path not in text_paths:
                mismatches.append((txt_path, "No matching image file"))
    
    # Report mismatches
    if mismatches:
        logger.warning(f"Found {len(mismatches)} files without matching pairs")
        for file_path, reason in mismatches[:5]:  # Show first 5 examples only
            logger.warning(f"  {reason}: {file_path}")
        if len(mismatches) > 5:
            logger.warning(f"  ... and {len(mismatches) - 5} more")
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'text_path': text_paths,
        'label': labels
    })
    
    logger.info(f"Loaded {len(df)} matched image-text pairs across {len(df['label'].unique())} classes")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def load_image(image_path: str) -> Union[None, 'numpy.ndarray']:
    """
    Load an image file as a numpy array.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    numpy.ndarray or None
        The loaded image as a numpy array, or None if loading fails
    """
    try:
        import cv2
        # Read image with OpenCV
        img = cv2.imread(image_path)
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def load_text(text_path: str) -> Union[None, str]:
    """
    Load a text file as a string.
    
    Parameters:
    -----------
    text_path : str
        Path to the text file
        
    Returns:
    --------
    str or None
        The content of the text file, or None if loading fails
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        try:
            with open(text_path, 'r', encoding='latin-1') as f:
                text = f.read()
            logger.warning(f"File {text_path} loaded with latin-1 encoding instead of utf-8")
            return text
        except Exception as e:
            logger.error(f"Error loading text {text_path} with latin-1 encoding: {e}")
            return None
    except Exception as e:
        logger.error(f"Error loading text {text_path}: {e}")
        return None

def create_data_splits(df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, 
                      test_size: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets in a stratified manner.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['image_path', 'text_path', 'label']
    train_size : float, default=0.7
        Proportion of the dataset to include in the training split
    val_size : float, default=0.15
        Proportion of the dataset to include in the validation split
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Ensure proportions sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"
    
    # First split: train and temp (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_size + test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: divide temp into val and test
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    # Log split information
    logger.info(f"Data split complete:")
    logger.info(f"  Training set: {len(train_df)} samples ({train_size*100:.1f}%)")
    logger.info(f"  Validation set: {len(val_df)} samples ({val_size*100:.1f}%)")
    logger.info(f"  Test set: {len(test_df)} samples ({test_size*100:.1f}%)")
    
    # Verify stratification
    logger.info("Class distribution in splits:")
    logger.info(f"  Original: {df['label'].value_counts(normalize=True).to_dict()}")
    logger.info(f"  Train: {train_df['label'].value_counts(normalize=True).to_dict()}")
    logger.info(f"  Validation: {val_df['label'].value_counts(normalize=True).to_dict()}")
    logger.info(f"  Test: {test_df['label'].value_counts(normalize=True).to_dict()}")
    
    return train_df, val_df, test_df 