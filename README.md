# Document Image Classification

This project develops a machine learning model to classify scanned document images into five predefined categories, leveraging both image data and OCR text.

## Project Structure

```
ML/
├── data/                # Data directory
│   ├── images/          # TIF image files by class
│   └── ocr/             # OCR text files by class
├── notebooks/           # Jupyter notebooks
│   └── eda.ipynb        # Exploratory Data Analysis
├── src/                 # Source code
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── data_loader.py   # Data loading functions
│   └── preprocessing.py # Preprocessing pipelines
├── results/             # Model outputs and evaluation
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Development Sprints

The project is organized into the following sprints:

1. Project Setup, Data Exploration, and Preprocessing Pipeline
2. Baseline Model Development (Image-Only)
3. Baseline Model Development (Text-Only)
4. Multimodal Model Development and Hyperparameter Tuning
5. Final Evaluation, Documentation, and Packaging 