# Document Image Classification

This project develops a machine learning model to classify scanned document images into five predefined categories (0, 2, 4, 6, 9), leveraging both image data and OCR text.

## Dataset

- **Overview**: 2500 scanned documents split into 5 classes (500 per class), creating a balanced dataset
- **Classes**: 0, 2, 4, 6, 9 (representing different document types)
- **Data Types**:
  - **Images**: Black and white TIF format, located in `data/images/{class_name}/*.TIF`
  - **OCR Text**: Corresponding text data in `data/ocr/{class_name}/*.TIF.txt`
- **Exploratory Data Analysis**: See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for a comprehensive analysis of image properties and text characteristics

## Project Structure

```text
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
│   │   ├── training_metrics.csv
│   │   ├── training_history.png
│   │   ├── image_baseline_eval.txt
│   │   ├── confusion_matrix.png
│   │   └── confusion_matrix_normalized.png
│   ├── text_baseline/         # Text model results
│   │   ├── tfidf_feature_importance.png
│   │   ├── confusion_matrix.png
│   │   ├── class_performance.png
│   │   └── text_features_info.txt
│   ├── text_baseline_eval.txt
│   ├── text_lr_gridsearch.csv
│   ├── preprocessing/
│   │   ├── text_preprocessing_results.csv
│   │   ├── text_preprocessing_summary.md
│   │   ├── top_tokens.png
│   │   ├── token_count_distribution.png
│   │   ├── avg_tokens_by_class.png
│   │   └── sample_*.png
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── class_distribution.png
│       └── split_summary.md
├── src/
│   ├── features/
│   │   └── text_features.py
│   ├── models/
│   │   ├── image_model.py
│   │   └── text_model.py
│   ├── training/
│   │   ├── train_image.py
│   │   ├── train_image_model.py
│   │   ├── train_text_model.py
│   │   └── evaluate_text_model.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset_split.py
│   └── test_dataset_split.py
├── test_data_loaders.py
├── test_image_model.py
├── test_text_features.py
├── train_model.py
├── run_baseline_image_model.sh
├── run_text_baseline.sh
├── run_evaluation.sh
├── run_text_features.sh
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - **Windows**: `venv\Scripts\activate`
   - **Mac/Linux**: `source venv/bin/activate`
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

```bash
jupyter lab notebooks/eda.ipynb
```

### Text Preprocessing

```bash
python src/test_text_preprocessing.py
```

### Dataset Splitting

```bash
python src/test_dataset_split.py
```

### Image Model Training and Evaluation

```bash
python test_data_loaders.py      # test loaders
python train_model.py            # train model
bash run_baseline_image_model.sh # full pipeline
bash run_evaluation.sh           # evaluate model
```

### Text Model Training and Evaluation

```bash
python test_text_features.py  # feature extraction & viz
bash run_text_features.sh     # TF‑IDF pipeline
bash run_text_baseline.sh     # train & evaluate text model
```

## Model Architecture and Performance

### Image Baseline Model
- **Architecture**: EfficientNet‑B0 (pre‑trained) with custom classifier head
- **Approach**: Transfer learning with frozen backbone
- **Regularization**: Dropout (0.2) to prevent overfitting
- **Training**: Learning‑rate scheduling & early stopping

#### Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **81.33 %** |
| **Macro F1** | **0.81** |

**Per‑class results**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 0 | 0.85 | 0.88 | 0.86 |
| 2 | 0.88 | 0.91 | 0.89 |
| 4 | 0.78 | 0.87 | 0.82 |
| 6 | 0.79 | 0.76 | 0.78 |
| 9 | 0.75 | 0.65 | 0.70 |

**Confusion Matrix**

![Confusion Matrix](results/image_baseline/confusion_matrix.png)

**Normalized Confusion Matrix**

![Normalized Confusion Matrix](results/image_baseline/confusion_matrix_normalized.png)

**Training History**

![Training History](results/image_baseline/training_history.png)

### Text Baseline Model
- **Features**: TF‑IDF vectorization (10 000 features)
- **Model**: Logistic Regression (one‑vs‑rest) with grid search hyper‑tuning

**Feature Importance (top coefficients)**

![TF‑IDF Feature Importance](results/text_baseline/tfidf_feature_importance.png)

**Class‑wise Performance**

![Class Performance](results/text_baseline/class_performance.png)

Evaluation report available at `results/text_baseline_eval.txt`.

## Preprocessing Pipeline

### Image Preprocessing
- Resize to 224×224 pixels
- Convert to RGB (3 channels)
- Normalize with ImageNet mean & std
- On‑the‑fly augmentation during training (random rotation, flip, …)

### Text Preprocessing
- Tokenisation & lower‑casing
- Stop‑word removal
- Lemmatisation with spaCy

**Token Count Distribution**

![Token Count Distribution](results/preprocessing/token_count_distribution.png)

**Top Tokens by Frequency**

![Top Tokens](results/preprocessing/top_tokens.png)

**Average Tokens per Class**

![Avg Tokens by Class](results/preprocessing/avg_tokens_by_class.png)

## Data Splitting
- **Strategy**: Stratified (70 % train • 15 % val • 15 % test)
- **Random Seed**: 42 for reproducibility

![Class Distribution Across Splits](results/splits/class_distribution.png)

## Sample Documents

A few example pages after preprocessing:

![Sample 1](results/preprocessing/sample_1.png)
![Sample 2](results/preprocessing/sample_2.png)
![Sample 3](results/preprocessing/sample_3.png)

## Reproducibility
This project fixes seeds in data splitting, data loaders and training scripts to enable repeatable results.

## Future Work
- Build a multimodal fusion model combining image & text features
- Explore advanced architectures and fine‑tuning

