# Document Image Classification

This project develops a machine learning model to classify scanned document images into five predefined categories (0, 2, 4, 6, 9), leveraging both image data and OCR text.

## Project Structure

```
ML/
├── data/                      # Data directory
│   ├── images/                # TIF image files by class (0, 2, 4, 6, 9)
│   └── ocr/                   # OCR text files by class (0, 2, 4, 6, 9)
├── notebooks/                 # Jupyter notebooks
│   └── eda.ipynb              # Exploratory Data Analysis
├── results/                   # Results and evaluation outputs
│   ├── preprocessing/         # Preprocessing results
│   │   ├── text_preprocessing_results.csv
│   │   └── text_preprocessing_summary.md
│   └── splits/                # Dataset splits
│       ├── train_split.csv    # Training set (70%)
│       ├── val_split.csv      # Validation set (15%)
│       ├── test_split.csv     # Test set (15%)
│       ├── class_distribution.png
│       └── split_summary.md
├── src/                       # Source code
│   ├── models/                # Model definitions
│   ├── training/              # Training scripts
│   ├── data_loader.py         # Data loading functions
│   ├── preprocessing.py       # Preprocessing pipelines
│   ├── dataset_split.py       # Dataset splitting implementation
│   └── test_dataset_split.py  # Test script for dataset splitting
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

2. **Baseline Model Development (Image-Only)** (Planned)
   - Feature extraction from images
   - Training image-only classifier
   - Model evaluation and tuning

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