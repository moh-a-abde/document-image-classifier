# Dataset Splitting - Implementation Summary

## Overview

This document summarizes the implementation of Task 1.6 (Dataset Splitting) from Sprint 1. The task involved creating a stratified train/validation/test split of the document classification dataset to ensure balanced class representation in all splits.

## Implementation Details

The implementation can be found in `src/dataset_split.py` which uses the `create_data_splits()` function from `src/data_loader.py`. The splitting process includes the following steps:

1. **Data Loading**:
   - Used the `load_data_paths()` function to gather paths to all image and OCR text files
   - Created a DataFrame with columns `image_path`, `text_path`, and `label`

2. **Stratified Split**:
   - Applied `sklearn.model_selection.train_test_split` in a two-step process:
     - First split: Separated training set (70%) from a temporary set (30%)
     - Second split: Divided the temporary set into validation (15%) and test (15%) 
   - Ensured class proportions were maintained using the `stratify` parameter

3. **Verification**:
   - Compared class distributions across splits to verify proper stratification
   - Calculated actual proportions to ensure they match the expected 70%/15%/15% ratio
   - Visualized class distribution to provide clear evidence of successful stratification

4. **Data Persistence**:
   - Saved each split as a separate CSV file for future use
   - Created a detailed report summarizing the splitting process and results
   - Generated visualization showing class distribution across splits

## Configuration Options

The `create_data_splits()` function accepts several parameters to control the splitting behavior:

- `df`: DataFrame containing image paths, text paths, and labels
- `train_size`: Proportion of data to include in the training set (default: 0.7)
- `val_size`: Proportion of data to include in the validation set (default: 0.15)
- `test_size`: Proportion of data to include in the test set (default: 0.15)
- `random_state`: Seed for random number generator to ensure reproducibility (default: 42)

## Testing and Evaluation

The dataset splitting was tested to ensure it meets all requirements:

1. **File Creation Test**: Verified that all expected output files were created
2. **Proportion Test**: Confirmed that the actual split proportions match the intended 70%/15%/15% ratio
3. **Stratification Test**: Validated that class distributions are consistent across all splits

All tests were successful, demonstrating that the implementation meets the requirements specified in Task 1.6.

## Dataset Statistics

The dataset was successfully split with the following statistics:

| Split | Sample Count | Percentage |
|-------|--------------|------------|
| Training | 1750 | 70.0% |
| Validation | 375 | 15.0% |
| Test | 375 | 15.0% |
| **Total** | 2500 | 100.0% |

Class distributions were maintained across all splits within a 2% tolerance, ensuring that each class is properly represented in the training, validation, and test sets.

## Output Files

The implementation produces the following output files:

- `results/splits/train_split.csv`: Contains paths and labels for the training set
- `results/splits/val_split.csv`: Contains paths and labels for the validation set
- `results/splits/test_split.csv`: Contains paths and labels for the test set
- `results/splits/class_distribution.png`: Visualization of the class distribution across splits
- `results/splits/split_summary.md`: Detailed report of the splitting process and results

## Conclusion

The dataset splitting implementation satisfies all requirements specified in Task 1.6 of Sprint 1. It provides a solid foundation for model development in subsequent sprints by ensuring that all classes are properly represented in each split, enabling fair evaluation of model performance. 