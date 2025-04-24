#!/usr/bin/env python

"""
Script to run the training for Task 2.3: Model Training (Image-Only).
This is a wrapper script that calls the training code in src/training/train_image_model.py.
"""

import sys
import subprocess
import os

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Ensure results directory exists
    os.makedirs("results/image_baseline", exist_ok=True)
    
    print("Starting image model training...")
    
    # Build the command to run the training script
    cmd = [
        "python", "src/training/train_image_model.py",
        "--batch-size", "32",
        "--epochs", "50",
        "--learning-rate", "1e-3",
        "--patience", "5",
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