"""
Environment setup utilities.

This module provides functions to set up the environment for training and evaluation,
especially in Continuous Integration (CI) environments where CUDA may not be available.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def setup_ci_environment():
    """
    Set up environment variables for CI environments.
    
    This function sets environment variables to ensure PyTorch doesn't
    try to use CUDA in environments where it's not available, such as
    GitHub Actions runners.
    
    Returns:
        bool: True if we're in a CI environment and settings were applied
    """
    is_ci = "CI" in os.environ or "GITHUB_ACTIONS" in os.environ
    
    # Set common environment variables to avoid CUDA issues
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    
    # Disable CUDA in CI environments
    if is_ci:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("CI environment detected. Disabling CUDA.")
    
    return is_ci

def get_device_setting(device_arg='auto'):
    """
    Determine the appropriate device setting based on availability.
    
    Args:
        device_arg (str): Requested device ('auto', 'cuda', 'mps', or 'cpu')
        
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu')
    """
    # Import torch here to avoid circular imports
    try:
        import torch
        
        # If device_arg is 'auto', detect the best available device
        if device_arg == 'auto':
            if torch.cuda.is_available():
                logger.info("CUDA is available. Using GPU.")
                return 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS is available. Using Apple Silicon GPU.")
                return 'mps'
            else:
                logger.info("No GPU detected. Using CPU.")
                return 'cpu'
        
        # If device_arg is 'cuda', check if it's available
        elif device_arg == 'cuda':
            if torch.cuda.is_available():
                logger.info("Using CUDA as specified.")
                return 'cuda'
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
        
        # If device_arg is 'mps', check if it's available
        elif device_arg == 'mps':
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Using MPS as specified.")
                return 'mps'
            else:
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return 'cpu'
        
        # If device_arg is 'cpu', use CPU
        elif device_arg == 'cpu':
            logger.info("Using CPU as specified.")
            return 'cpu'
        
        # If device_arg is something else, warn and fall back to CPU
        else:
            logger.warning(f"Unknown device '{device_arg}'. Falling back to CPU.")
            return 'cpu'
    
    except ImportError:
        logger.warning("PyTorch not available. Defaulting to CPU.")
        return 'cpu'

def setup_training_environment(device_arg='auto'):
    """
    Set up the environment for training and evaluation.
    
    This is a convenience function that combines setup_ci_environment()
    and get_device_setting() to set up the environment for training
    and evaluation.
    
    Args:
        device_arg (str): Requested device ('auto', 'cuda', 'mps', or 'cpu')
        
    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu')
    """
    # Set up environment variables
    setup_ci_environment()
    
    # Get device setting
    return get_device_setting(device_arg) 