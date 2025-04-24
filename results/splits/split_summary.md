# Dataset Splitting Summary

## Overview

This document summarizes the implementation of Task 1.6 (Dataset Splitting) from Sprint 1. The dataset was split into training, validation, and test sets using stratified sampling to ensure balanced class representation in each split.

## Split Proportions

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 1750 | 70.0% |
| Validation | 375 | 15.0% |
| Test | 375 | 15.0% |
| **Total** | 2500 | 100.0% |

## Class Distribution

| Class | Original | Training | Validation | Test |
|-------|----------|----------|------------|------|
| 0 | 500 | 350 | 75 | 75 |
| 2 | 500 | 350 | 75 | 75 |
| 4 | 500 | 350 | 75 | 75 |
| 6 | 500 | 350 | 75 | 75 |
| 9 | 500 | 350 | 75 | 75 |

## Class Distribution (Percentage)

| Class | Original | Training | Validation | Test |
|-------|----------|----------|------------|------|
| 0 | 20.0% | 20.0% | 20.0% | 20.0% |
| 2 | 20.0% | 20.0% | 20.0% | 20.0% |
| 4 | 20.0% | 20.0% | 20.0% | 20.0% |
| 6 | 20.0% | 20.0% | 20.0% | 20.0% |
| 9 | 20.0% | 20.0% | 20.0% | 20.0% |

## Implementation Details

The dataset splitting was implemented using the following steps:

1. **Data Loading**: Used `load_data_paths` function to load all image and text file paths with labels.
2. **Stratified Split**: Applied `sklearn.model_selection.train_test_split` twice:
   - First split: Separated training set (70%) from a temporary set (30%)
   - Second split: Divided the temporary set into validation (15%) and test (15%)
3. **Verification**: Confirmed that class proportions remained consistent across all splits.
4. **Persistence**: Saved the split information to CSV files for future use.

## Verification Results

The stratification was successful, with all splits maintaining class proportions within a 2% tolerance of the original dataset.

## Location of Split Files

The split information is saved in the following files:
- `results/splits/train_split.csv`
- `results/splits/val_split.csv`
- `results/splits/test_split.csv`

This information can be used in subsequent sprints to ensure consistent dataset usage across different model implementations.
