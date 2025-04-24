#!/bin/bash

# Script to run the baseline image model pipeline (Tasks 2.3 and 2.4)
# This includes data loader testing, model training, and model evaluation

set -e  # Exit on error

# Create necessary directories
mkdir -p models
mkdir -p results/image_baseline

echo "======================================="
echo "Step 1: Testing data loaders"
echo "======================================="
python test_data_loaders.py
echo ""

echo "======================================="
echo "Step 2: Training the model (Task 2.3)"
echo "======================================="
python src/training/train_image_model.py --batch-size 32 --epochs 50 --learning-rate 1e-3 --patience 5 --plot-history
echo ""

echo "======================================="
echo "Step 3: Evaluating the model (Task 2.4)"
echo "======================================="
python test_image_model.py
echo ""

echo "======================================="
echo "Pipeline completed successfully!"
echo "======================================="
echo ""
echo "Results and model saved to:"
echo "- Model: models/image_baseline_best.pt"
echo "- Training metrics: results/image_baseline/training_metrics.csv"
echo "- Evaluation results: results/image_baseline/image_baseline_eval.txt"
echo "- Confusion matrix: results/image_baseline/confusion_matrix.png" 