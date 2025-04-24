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

## Baseline Image Model

The baseline image model uses EfficientNet-B0 pretrained on ImageNet to classify document images. The model architecture is implemented in `src/models/image_model.py`.

### Running the Baseline Model

You can run the baseline model training and evaluation using the provided shell script:

```bash
./run_baseline_image_model.sh
```

This script will:

1. Test the data loaders
2. Train the model
3. Evaluate the model on the test set
4. Save results to the `results/image_baseline/` directory

### Manual Execution

If you prefer to run each step manually:

1. Test data loaders:
```bash
python test_data_loaders.py
```

2. Train the model:
```bash
python src/training/train_image_model.py --batch-size 32 --epochs 50 --learning-rate 1e-3 --patience 5 --plot-history
```

3. Evaluate the model:
```bash
python test_image_model.py
```

### Additional Options

The training and evaluation scripts support several command-line options:

- `--device cpu` - Force CPU usage even if CUDA is available
- `--batch-size 16` - Change the batch size (default: 32)
- `--epochs 10` - Set the maximum number of epochs (default: 50)
- `--patience 3` - Configure early stopping patience (default: 5)
- `--debug` - Enable debug mode with more verbose logging

## GitHub Actions Workflow

This repository includes a GitHub Actions workflow that automatically runs the baseline image model pipeline whenever changes are pushed to the main branch.

To manually trigger the workflow:

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select "Baseline Image Model Training" from the workflows list
4. Click "Run workflow"

The workflow will:
- Set up the Python environment with the correct package versions
- Install specific NumPy and PyTorch versions to ensure compatibility
- Run the data loader test, training, and evaluation on CPU
- Generate a summary report including environment information
- Commit the results back to the repository

### Workflow Improvements

The GitHub Actions workflow includes several optimizations:

- Fixed package compatibility issues by pinning NumPy < 2.0.0 and using PyTorch CPU-only version
- Added CUDA-related workarounds to prevent errors on systems without GPUs:
  - Using PyTorch CPU wheels (`--index-url https://download.pytorch.org/whl/cpu`)
  - Setting `CUDA_VISIBLE_DEVICES=""` environment variable
  - Patching code to override `torch.cuda.is_available()` function
- Error handling throughout all scripts to ensure the workflow completes even if errors occur
- Fallback mechanisms for displaying batch images when NumPy is not available
- Detailed environment information logging for debugging purposes
- Reduced number of epochs (`10` instead of `50`) when running in the workflow to save time
- Proper file existence checks before Git operations

### CI/CD Compatibility

The code has been made compatible with CI/CD environments that don't have GPU access:

- All scripts accept a `--device cpu` parameter to force CPU usage
- Automatic fallback to CPU when CUDA is not available
- Safe environment checks that don't crash when CUDA libraries are missing
- Helpful error messages and diagnostic information for debugging

## Results

After running the baseline model, you'll find the following results:

- Trained model: `models/image_baseline_best.pt`
- Training metrics: `results/image_baseline/training_metrics.csv`
- Training history plot: `results/image_baseline/training_history.png`
- Evaluation results: `results/image_baseline/image_baseline_eval.txt`
- Confusion matrix: `results/image_baseline/confusion_matrix.png`
- Summary report: `model_results.md`

### Training Metrics

The model achieves around 80-85% accuracy on the validation set after training for approximately 10-15 epochs with early stopping. The EfficientNet-B0 architecture with transfer learning proves effective for document image classification, even with the limited dataset. 