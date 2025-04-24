#!/usr/bin/env python

"""
Check the environment for PyTorch and its dependencies.

This script checks if PyTorch is installed and available,
and determines the best device to use for training.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables to ensure PyTorch doesn't crash if CUDA is not available
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '0'

# Disable CUDA in CI environments to avoid errors
is_ci = "CI" in os.environ or "GITHUB_ACTIONS" in os.environ
if is_ci:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("CI environment detected. Disabling CUDA.")

try:
    # Try importing torch without requiring CUDA
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability safely
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU for computations.")
    except RuntimeError as cuda_error:
        print(f"Error checking CUDA: {cuda_error}")
        print("CUDA check failed. Using CPU for computations.")
        cuda_available = False
        
    # Check for MPS (Apple Metal) availability
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    # Determine available device
    if cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
except ImportError:
    print("PyTorch is not installed.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while checking PyTorch: {e}")
    print("Continuing with CPU only.")
    device = torch.device("cpu")
    print(f"Using device: {device}")

# Check if the project modules can be imported
try:
    # Try importing a module to test the setup
    from src.utils import environment
    print("Successfully imported utility modules.")
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")

print("Environment check completed successfully.") 