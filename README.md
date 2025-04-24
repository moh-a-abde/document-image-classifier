# Document Image Classification

This project develops a machine learning model to classify scanned document images into five predefined categories (0, 2, 4, 6, 9), leveraging both image data and OCR text.

## Dataset

- **Overview**: 2500 scanned documents split into 5 classes (500 per class), creating a balanced dataset
- **Classes**: 0, 2, 4, 6, 9 (representing different document types)
- **Data Types**:
  - **Images**: Black and white TIF format, located in `data/images/{class_name}/*.TIF`
  - **OCR Text**: Corresponding text data in `data/ocr/{class_name}/*.TIF.txt`
- **Exploratory Data Analysis**: See `notebooks/eda.ipynb` for a comprehensive analysis of image properties and text characteristics

## Project Structure

```
ML/
├── data/                      # Data directory
│   ├── images/                # TIF image files by class (0, 2, 4, 6, 9)
│   └── ocr/                   # OCR text files by class (0, 2, 4, 6, 9)
├── models/                    # Saved model checkpoints
│   ├── image_baseline_best.pt # Best image model checkpoint
│   ├── text_baseline_best.pkl # Best text model checkpoint
│   └── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
├── notebooks/                 # Jupyter notebooks
│   └── eda.ipynb              # Exploratory Data Analysis
├── results/                   # Results and evaluation outputs
│   ├── image_baseline/        # Image model results
│   │   ├── training_metrics.csv        # Training metrics
│   │   ├── training_history.png        # Training plots
│   │   ├── image_baseline_eval.txt     # Evaluation report
│   │   ├── confusion_matrix.png        # Confusion matrix
│   │   └── confusion_matrix_normalized.png # Normalized confusion matrix
│   ├── text_baseline/         # Text model results
│   │   ├── tfidf_feature_importance.png  # Feature importance visualization
│   │   ├── confusion_matrix.png          # Confusion matrix
│   │   ├── class_performance.png         # Per-class metrics
│   │   └── text_features_info.txt        # Feature information
│   ├── text_baseline_eval.txt  # Text model evaluation report
│   ├── text_lr_gridsearch.csv  # Text model hyperparameter tuning results
│   ├── preprocessing/         # Preprocessing results
│   │   ├── text_preprocessing_results.csv
│   │   ├── text_preprocessing_summary.md
│   │   ├── top_tokens.png                # Most frequent tokens visualization
│   │   ├── token_count_distribution.png  # Distribution of token counts
│   │   ├── avg_tokens_by_class.png       # Average tokens per class
│   │   └── sample_*.png                  # Sample document visualizations
│   └── splits/                # Dataset splits
│       ├── train.csv          # Training set (70%)
│       ├── val.csv            # Validation set (15%)
│       ├── test.csv           # Test set (15%)
│       ├── class_distribution.png # Distribution of classes across splits
│       └── split_summary.md    # Summary of data splitting process
├── src/                       # Source code
│   ├── features/              # Feature extraction
│   │   └── text_features.py   # Text TF-IDF feature extraction
│   ├── models/                # Model definitions
│   │   ├── image_model.py     # Image model architecture (EfficientNet-B0)
│   │   └── text_model.py      # Text model (LogisticRegression)
│   ├── training/              # Training scripts
│   │   ├── train_image.py     # Image model training components
│   │   ├── train_image_model.py # Image model training script
│   │   ├── train_text_model.py  # Text model training script
│   │   └── evaluate_text_model.py # Text model evaluation script
│   ├── data_loader.py         # Data loading functions
│   ├── preprocessing.py       # Preprocessing pipelines
│   ├── dataset_split.py       # Dataset splitting implementation
│   └── test_dataset_split.py  # Test script for dataset splitting
├── test_data_loaders.py       # Test data loaders for image model
├── test_image_model.py        # Evaluate trained image model
├── test_text_features.py      # Extract and analyze text features
├── train_model.py             # Run image model training
├── run_baseline_image_model.sh # Full image model pipeline
├── run_text_baseline.sh       # Full text model pipeline
├── run_evaluation.sh          # Evaluate image model only
├── run_text_features.sh       # Generate text features
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
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
6. Download required spaCy model:
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

To test the text preprocessing:
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
python test_data_loaders.py
```

To train the image baseline model:
```bash
python train_model.py
```
Or use the complete pipeline script:
```bash
bash run_baseline_image_model.sh
```

To evaluate the trained image model:
```bash
bash run_evaluation.sh
```

### Text Model Training and Evaluation

To extract text features and visualize their importance:
```bash
python test_text_features.py
```
Or use the script for feature extraction:
```bash
bash run_text_features.sh
```

To train and evaluate the text baseline model:
```bash
bash run_text_baseline.sh
```

## Model Architecture and Performance

### Image Baseline Model
- **Architecture**: EfficientNet-B0 (pre-trained) with custom classifier head
- **Approach**: Transfer learning with frozen backbone
- **Regularization**: Dropout (0.2) to prevent overfitting
- **Training**: Learning rate scheduling, early stopping
- **Performance**:
  - **Accuracy**: 81.33%
  - **Macro F1**: 0.81
  - **Per-class results**:
    - Class 0: Precision=0.85, Recall=0.88, F1=0.86
    - Class 2: Precision=0.88, Recall=0.91, F1=0.89
    - Class 4: Precision=0.78, Recall=0.87, F1=0.82
    - Class 6: Precision=0.79, Recall=0.76, F1=0.78
    - Class 9: Precision=0.75, Recall=0.65, F1=0.70
  - **Confusion Matrix**: See `results/image_baseline/confusion_matrix.png`
  - **Training History**: See `results/image_baseline/training_history.png`

### Text Baseline Model
- **Features**: TF-IDF vectorization (10,000 features)
- **Model**: LogisticRegression with hyperparameter tuning
- **Strategy**: One-vs-rest with L1/L2 regularization
- **Performance**: 
  - Results available in `results/text_baseline_eval.txt`
  - Feature importance visualization: `results/text_baseline/tfidf_feature_importance.png`

## Preprocessing Pipeline

### Image Preprocessing
- Resize to 224x224 pixels
- Convert to RGB (3 channels)
- Apply normalization using ImageNet mean and std
- Image augmentation during training (random rotation, flip, etc.)

### Text Preprocessing
- Tokenization and lowercasing
- Stopword removal
- Lemmatization using spaCy
- Visualizations:
  - Token distribution: `results/preprocessing/token_count_distribution.png`
  - Top tokens: `results/preprocessing/top_tokens.png`
  - Tokens by class: `results/preprocessing/avg_tokens_by_class.png`

## Data Splitting
- **Strategy**: Stratified split to maintain class distribution
- **Split Ratio**: 70% training, 15% validation, 15% test
- **Random Seed**: Fixed seed (42) to ensure reproducibility
- **Visualization**: Class distribution across splits (`results/splits/class_distribution.png`)

## Reproducibility
This project uses fixed random seeds in various components to ensure reproducible results:
- Data splitting: Seed 42 in `src/dataset_split.py`
- Model training: Seeds in training scripts
- Data loaders: Fixed seeds for consistent batching

## Visualizations

The project includes various visualizations to help understand the data, models, and results:

### Data Exploration
- Class distribution across splits: `results/splits/class_distribution.png`
- Sample documents: `results/preprocessing/sample_*.png`

### Text Analysis
- Token count distribution: `results/preprocessing/token_count_distribution.png`
- Top tokens by frequency: `results/preprocessing/top_tokens.png` 
- Average tokens by class: `results/preprocessing/avg_tokens_by_class.png`
- TF-IDF feature importance: `results/text_baseline/tfidf_feature_importance.png`

### Model Performance
- Training metrics history: `results/image_baseline/training_history.png`
- Confusion matrix: `results/image_baseline/confusion_matrix.png`
- Normalized confusion matrix: `results/image_baseline/confusion_matrix_normalized.png`
- Text model class performance: `results/text_baseline/class_performance.png`

To view all visualizations, explore the `results/` directory and its subdirectories.

## License
This project is provided for educational purposes only.

## Future Work
- Implement multimodal model combining image and text features
- Explore advanced architectures and fine-tuning strategies
- Deploy model with a simple API for inference 