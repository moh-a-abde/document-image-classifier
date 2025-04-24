# Project Process Description: Document Image Classification

This document outlines the process taken to develop and evaluate machine learning models for classifying scanned document images into five categories using both image and text data.

## How did you solve the problem?

The problem was approached systematically, following a structured plan akin to agile sprints:

1.  **Project Setup & Environment:** Established a project structure, set up a virtual environment, and installed necessary dependencies (`requirements.txt`).
2.  **Exploratory Data Analysis (EDA):** Analyzed the dataset (`notebooks/eda.ipynb`) to understand image properties (dimensions, color profiles) and text characteristics (token counts, common words, distributions across classes). This helped inform preprocessing strategies.
3.  **Data Splitting:** Implemented a stratified split (70% train, 15% validation, 15% test) to ensure class balance across sets and saved the splits for consistent use (`src/dataset_split.py`).
4.  **Preprocessing Pipelines:**
    *   **Image:** Developed a pipeline using `torchvision` transforms including resizing (to 224x224), conversion to RGB, normalization using ImageNet statistics, and data augmentation (random rotations, flips) during training (`src/preprocessing.py`, `src/data_loader.py`).
    *   **Text:** Created a text preprocessing pipeline involving tokenization, lowercasing, stop-word removal (NLTK), and lemmatization (spaCy). Generated TF-IDF features using `sklearn` (`src/preprocessing.py`, `src/features/text_features.py`).
5.  **Baseline Model Development:**
    *   **Image Baseline:** Implemented a transfer learning approach using a pre-trained EfficientNet-B0 model. The backbone features were extracted, and a custom classifier head (MLP with Dropout) was added and trained. Techniques like learning rate scheduling and early stopping were used (`src/models/image_model.py`, `src/training/train_image_model.py`).
    *   **Text Baseline:** Used the preprocessed text and TF-IDF features to train a Logistic Regression model. Hyperparameter tuning (Grid Search CV) was performed to find the optimal regularization strength (`src/models/text_model.py`, `src/training/train_text_model.py`).
6.  **Multimodal Model Development:**
    *   Designed an intermediate fusion architecture. Features were extracted from the best-performing image model backbone (EfficientNet-B0) and the TF-IDF text representation.
    *   These feature vectors were concatenated and fed into a final MLP classification head.
    *   Training involved careful learning rate selection and progressive unfreezing strategies (`src/models/multimodal_model.py`, `src/training/train_multimodal.py`).
7.  **Evaluation:** Consistently evaluated all models on the held-out test set using standard classification metrics: Accuracy, Macro F1-score, Precision, Recall, and Confusion Matrices (`run_evaluation.sh`, `run_text_baseline.sh`, `run_multimodal_model.sh`).
8.  **Documentation:** Maintained a `README.md` file documenting the project structure, setup, usage, model architectures, and results.

## Why did you choose to solve the problem the way you did?

The chosen approach was based on several factors:

*   **Structured Methodology:** Starting with EDA and baselines provides a solid foundation and allows for quantifying the benefits of more complex models. Following sprint-like stages helps manage complexity.
*   **Leveraging Pre-trained Models:** Using EfficientNet-B0 for images significantly reduces training time and leverages knowledge learned from large datasets (ImageNet), which is crucial given the dataset size. EfficientNet-B0 offers a good balance between performance and computational cost.
*   **Standard Baselines:** TF-IDF with Logistic Regression is a strong and standard baseline for text classification. It's computationally efficient and provides interpretable results (feature importance).
*   **Hypothesis-Driven Multimodality:** The core hypothesis was that combining visual layout information (from images) and semantic content (from text) would yield superior performance. This is common in document understanding tasks.
*   **Intermediate Fusion:** Concatenating features before a final classification layer is a robust and widely used fusion technique that allows the model to learn interactions between modalities.
*   **Standard Libraries:** Relying on established libraries like PyTorch, Scikit-learn, Pandas, NLTK, and spaCy ensures code quality, maintainability, and access to optimized implementations.
*   **Reproducibility:** Fixing random seeds and using version-controlled dependencies (`requirements.txt`) ensures that the results can be reproduced.

## What challenges did you face?

*   **OCR Quality:** The presence of empty OCR files (noted in logs) and the inherent potential for OCR errors presented a challenge. While the TF-IDF approach is somewhat robust to minor noise, significant errors could impact performance. The model had to learn despite potential inconsistencies in the text data.
*   **Hyperparameter Tuning:** Optimizing hyperparameters, especially for the multimodal model (MLP architecture, learning rates, dropout rates, training schedule), required careful experimentation and validation. Finding the best C value for Logistic Regression also involved grid search.
*   **Computational Cost:** Training the EfficientNet-based models, even with transfer learning, requires significant computational resources (GPU recommended) and time. Iteration cycles for tuning were therefore longer.
*   **Feature Engineering:** Deciding on the optimal parameters for TF-IDF (e.g., `max_features`, `ngram_range`) involved some experimentation based on EDA insights.
*   **Balancing Modalities:** Ensuring that neither modality completely dominated the other during multimodal fusion required careful design of the fusion layer and training strategy.

## If you had more time, what would you do differently?

*   **Advanced Text Representations:** Explore transformer-based embeddings (e.g., BERT, Sentence-BERT, or domain-specific variants) instead of TF-IDF. Fine-tuning a transformer on the document text could capture deeper semantic meaning.
*   **More Sophisticated Fusion:** Experiment with attention mechanisms (self-attention, cross-modal attention) to allow the model to dynamically weigh the importance of different features from each modality.
*   **End-to-End Models:** Investigate models designed for joint image and text processing from the start, such as LayoutLM or ViLT, which might capture finer-grained interactions between text position and content.
*   **Deeper Error Analysis:** Conduct a more thorough analysis of misclassified samples for the multimodal model to identify specific weaknesses or class confusions that could be targeted for improvement.
*   **Advanced Image Augmentation:** Explore more sophisticated image augmentation techniques specific to document images.
*   **Explore Different Architectures:** Try alternative CNN backbones for the image branch or different classifiers for the text branch and fusion head.
*   **Handle OCR Errors Explicitly:** Implement strategies to detect or potentially correct OCR errors, or design models more robust to this specific type of noise.

---

This structured approach, starting from data understanding and building complexity incrementally through baselines to a final multimodal model, allowed for effective problem-solving and yielded a model with significantly improved performance compared to using either modality alone. 