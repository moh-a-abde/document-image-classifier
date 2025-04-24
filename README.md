# Document Image Classification

This project develops a machine learning model to classify scanned document images into five predefined categories (0, 2, 4, 6, 9), leveraging both image data and OCR text.

## Project Structure

```
ML/
├── data/                      # Data directory
│   ├── images/                # TIF image files by class (0, 2, 4, 6, 9)
│   └── ocr/                   # OCR text files by class (0, 2, 4, 6, 9)
├── models/                    # Saved model checkpoints
│   └── image_baseline_best.pt # Best image model checkpoint
├── notebooks/                 # Jupyter notebooks
│   └── eda.ipynb              # Exploratory Data Analysis
├── results/                   # Results and evaluation outputs
│   ├── image_baseline/        # Image model results
│   │   ├── training_metrics.csv        # Training metrics
│   │   ├── training_history.png        # Training plots
│   │   ├── image_baseline_eval.txt     # Evaluation report
│   │   ├── confusion_matrix.png        # Confusion matrix
│   │   └── confusion_matrix_normalized.png # Normalized confusion matrix
│   ├── preprocessing/         # Preprocessing results
│   │   ├── text_preprocessing_results.csv
│   │   └── text_preprocessing_summary.md
│   └── splits/                # Dataset splits
│       ├── train.csv          # Training set (70%)
│       ├── val.csv            # Validation set (15%)
│       ├── test.csv           # Test set (15%)
│       ├── class_distribution.png
│       └── split_summary.md
├── src/                       # Source code
│   ├── models/                # Model definitions
│   │   └── image_model.py     # Image model architecture (EfficientNet-B0)
│   ├── training/              # Training scripts
│   │   ├── train_image.py     # Image model training components
│   │   └── train_image_model.py # Image model training script
│   ├── data_loader.py         # Data loading functions
│   ├── preprocessing.py       # Preprocessing pipelines
│   ├── dataset_split.py       # Dataset splitting implementation
│   └── test_dataset_split.py  # Test script for dataset splitting
├── test_data_loaders.py       # Test data loaders for image model
├── test_image_model.py        # Evaluate trained image model
├── train_model.py             # Run image model training
├── run_baseline_image_model.sh # Full image model pipeline
├── run_evaluation.sh          # Evaluate image model only
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download required NLTK data: 
   ```python
   import nltk
   nltk.download('stopwords')
   ```
6. Download required spaCy model (if not installed via requirements.txt):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Data Exploration

To explore the dataset, you can run the Jupyter notebook:
```bash
jupyter lab notebooks/eda.ipynb
```

### Text Preprocessing

To test the text preprocessing implementation:
```bash
python src/test_text_preprocessing.py
```

### Dataset Splitting

To perform the dataset splitting:
```bash
python src/test_dataset_split.py
```

### Image Model Training and Evaluation

To test the image data loaders:
```bash
./test_data_loaders.py
```

To train the image baseline model:
```bash
./train_model.py
```

To evaluate the trained image model:
```bash
./run_evaluation.sh
```

To run the complete image model pipeline (data loader testing, training, and evaluation):
```bash
./run_baseline_image_model.sh
```

## Development Sprints

The project is organized into the following sprints:

1. **Project Setup, Data Exploration, and Preprocessing Pipeline** (Completed)
   - Environment setup and dependency management
   - Exploratory data analysis on image and text data
   - Data loading implementation
   - Image preprocessing pipeline
   - Text preprocessing pipeline
   - Dataset splitting with stratification
   - Initial requirements.txt

2. **Baseline Model Development (Image-Only)** (Completed)
   - Feature extraction from images using EfficientNet-B0
   - Transfer learning approach with frozen backbone
   - Custom classifier head with dropout for regularization
   - Training image-only classifier with learning rate scheduling
   - Model evaluation on test set (81.33% accuracy)
   - Confusion matrix analysis
   - Updated requirements.txt with PyTorch dependencies

3. **Baseline Model Development (Text-Only)** (Planned)
   - Feature extraction from OCR text
   - Training text-only classifier
   - Model evaluation and tuning

4. **Multimodal Model Development and Hyperparameter Tuning** (Planned)
   - Combined model architecture
   - Multimodal training and evaluation
   - Hyperparameter optimization

5. **Final Evaluation, Documentation, and Packaging** (Planned)
   - Comprehensive evaluation
   - Documentation and reporting
   - Model packaging and deployment guidance 