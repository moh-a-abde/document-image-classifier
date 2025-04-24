#!/usr/bin/env python

"""
Script to run the training for Task 2.3: Model Training (Image-Only).
This is a wrapper script that calls the training code in src/training/train_image_model.py.
"""

import sys
import subprocess
import os
import torch

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Ensure results directory exists
    os.makedirs("results/image_baseline", exist_ok=True)
    
    print("Starting image model training...")
    
    # Determine best available device
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU for training.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("MPS is available. Using Apple Silicon GPU for training.")
    else:
        device = "cpu"
        print("No GPU detected. Using CPU for training (this will be slower).")
    
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
        result = subprocess.run(cmd, check=True)
        print(f"Training completed with exit code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        sys.exit(1)
    
    print("Training completed successfully!") 