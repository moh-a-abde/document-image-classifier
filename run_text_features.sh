#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p results/text_baseline

# Run the text feature extraction test
python test_text_features.py --data_dir data --image_subdir images --ocr_subdir ocr --output_dir results/text_baseline --model_dir models

# Display results location
echo "Feature extraction complete. Results saved to results/text_baseline/"
echo "TF-IDF vectorizer saved to models/tfidf_vectorizer.pkl" 