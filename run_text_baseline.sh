#!/bin/bash
# Shell script to run the text feature extraction, model training, and evaluation

echo "Starting text model baseline training and evaluation..."

# Step 1: Extract text features and create feature visualizations
echo "Extracting text features and creating visualizations..."
python test_text_features.py --output_dir results/text_baseline

# Step 2: Train the text classifier with hyperparameter tuning
echo "Training text classifier with hyperparameter tuning..."
python src/training/train_text_model.py --data_dir data --image_subdir images --ocr_subdir ocr --results_dir results --model_dir models

# Step 3: Evaluate the trained model on the test set
echo "Evaluating text classifier on test set..."
python src/training/evaluate_text_model.py --data_dir data --image_subdir images --ocr_subdir ocr --results_dir results --model_dir models

echo "Text baseline model training and evaluation completed!"
echo "Results can be found in the results/ directory."
echo "  - Cross-validation results: results/text_lr_gridsearch.csv"
echo "  - Evaluation metrics: results/text_baseline_eval.txt"
echo "  - Confusion matrix: results/text_baseline/confusion_matrix.png"
echo "  - Class performance: results/text_baseline/class_performance.png" 