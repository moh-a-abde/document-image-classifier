#!/bin/bash

# Shell script to run the multimodal model training and evaluation pipeline

# Set up environment
echo "Setting up environment..."
source venv/bin/activate

# Create results directory
mkdir -p results/multimodal
mkdir -p models

# First, check if TF-IDF vectorizer exists
if [ ! -f "models/tfidf_vectorizer.pkl" ]; then
    echo "TF-IDF vectorizer not found. Running text feature extraction first..."
    bash run_text_features.sh
fi

# Test multimodal data loader
echo "Testing multimodal data loader..."
python test_multimodal_data_loader.py

# Test multimodal model architecture
echo "Testing multimodal model architecture..."
python src/models/test_multimodal_model.py

# Run the multimodal model training and evaluation
echo "Training and evaluating multimodal model..."
python src/training/train_multimodal.py \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --unfreeze_epoch 5 \
    --patience 7 \
    --device cuda \
    --results_dir results/multimodal \
    --model_save_path models/multimodal_best.pt \
    --vectorizer_path models/tfidf_vectorizer.pkl

echo "Multimodal model training and evaluation completed!"