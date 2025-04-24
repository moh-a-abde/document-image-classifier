#!/usr/bin/env python

"""
Script to run the training for Task 2.3: Model Training (Image-Only).
This is a wrapper script that calls the training code in src/training/train_image_model.py.
"""

import sys
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up environment before importing torch
try:
    from src.utils.environment import setup_training_environment
    device = setup_training_environment()
    logger.info(f"Using device: {device}")
except ImportError:
    # Fallback if utility module not available
    logger.warning("Could not import environment utilities. Using basic environment setup.")
    
    # Setting environment variables to avoid CUDA errors if CUDA is not available
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    
    # Disable CUDA in CI environments to avoid errors
    if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("CI environment detected. Disabling CUDA.")
    
    # Now safe to import torch
    import torch
    
    # Determine best available device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available. Using GPU for training.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS is available. Using Apple Silicon GPU for training.")
    else:
        device = "cpu"
        logger.info("No GPU detected. Using CPU for training (this will be slower).")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Ensure results directory exists
    os.makedirs("results/image_baseline", exist_ok=True)
    
    logger.info("Starting image model training...")
    
    # Build the command to run the training script
    cmd = [
        "python", "src/training/train_image_model.py",
        "--batch-size", "32",
        "--epochs", "50",
        "--learning-rate", "1e-3",
        "--patience", "5",
        "--device", device,
        "--plot-history"
    ]
    
    # Run the command
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        logger.info(f"Training completed with exit code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code: {e.returncode}")
        sys.exit(1)
    
    logger.info("Training completed successfully!") 