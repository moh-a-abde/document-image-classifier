#!/bin/bash

# Script to run only the Task 2.4: Baseline Evaluation (Image-Only)
# This skips the data loader testing and model training steps
# Run this after you've already trained the model with run_baseline_image_model.sh

set -e  # Exit on error

# Ensure results directory exists
mkdir -p results/image_baseline

echo "======================================="
echo "Evaluating the image model (Task 2.4)"
echo "======================================="
python test_image_model.py
echo ""

echo "======================================="
echo "Evaluation completed successfully!"
echo "======================================="
echo ""
echo "Results saved to:"
echo "- Evaluation results: results/image_baseline/image_baseline_eval.txt"
echo "- Confusion matrix: results/image_baseline/confusion_matrix.png"
echo "- Normalized confusion matrix: results/image_baseline/confusion_matrix_normalized.png" 