"""
Test script for dataset splitting implementation.

This script runs the dataset splitting functionality from dataset_split.py
and verifies that the result matches the expected criteria from Task 1.6.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_split import main as run_dataset_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_split_files_exist():
    """Test that all expected split files were created."""
    split_files = [
        "results/splits/train_split.csv",
        "results/splits/val_split.csv",
        "results/splits/test_split.csv",
        "results/splits/class_distribution.png",
        "results/splits/split_summary.md"
    ]
    
    for file_path in split_files:
        if not os.path.exists(file_path):
            logger.error(f"Missing expected file: {file_path}")
            return False
    
    logger.info("All expected split files were created successfully.")
    return True

def test_split_proportions():
    """Test that the split proportions match the specified ratios."""
    # Load split data
    train_df = pd.read_csv("results/splits/train_split.csv")
    val_df = pd.read_csv("results/splits/val_split.csv")
    test_df = pd.read_csv("results/splits/test_split.csv")
    
    # Calculate total count
    total_count = len(train_df) + len(val_df) + len(test_df)
    
    # Calculate actual proportions
    train_prop = len(train_df) / total_count
    val_prop = len(val_df) / total_count
    test_prop = len(test_df) / total_count
    
    # Check against expected proportions (with 1% tolerance)
    tolerance = 0.01
    train_check = abs(train_prop - 0.7) < tolerance
    val_check = abs(val_prop - 0.15) < tolerance
    test_check = abs(test_prop - 0.15) < tolerance
    
    if train_check and val_check and test_check:
        logger.info("Split proportions match the expected values (70/15/15).")
        logger.info(f"  Train: {train_prop:.3f} ({len(train_df)} samples)")
        logger.info(f"  Validation: {val_prop:.3f} ({len(val_df)} samples)")
        logger.info(f"  Test: {test_prop:.3f} ({len(test_df)} samples)")
        return True
    else:
        logger.error("Split proportions do not match expected values.")
        logger.error(f"  Expected: 0.7/0.15/0.15, Got: {train_prop:.3f}/{val_prop:.3f}/{test_prop:.3f}")
        return False

def test_stratification():
    """Test that class distribution is maintained across splits."""
    # Load split data
    train_df = pd.read_csv("results/splits/train_split.csv")
    val_df = pd.read_csv("results/splits/val_split.csv")
    test_df = pd.read_csv("results/splits/test_split.csv")
    
    # Combine to get original distribution
    all_df = pd.concat([train_df, val_df, test_df])
    
    # Calculate class distributions
    all_dist = all_df['label'].value_counts(normalize=True).sort_index()
    train_dist = train_df['label'].value_counts(normalize=True).sort_index()
    val_dist = val_df['label'].value_counts(normalize=True).sort_index()
    test_dist = test_df['label'].value_counts(normalize=True).sort_index()
    
    # Compare distributions (within 5% tolerance)
    tolerance = 0.05
    train_check = all((abs(train_dist - all_dist) < tolerance).values)
    val_check = all((abs(val_dist - all_dist) < tolerance).values)
    test_check = all((abs(test_dist - all_dist) < tolerance).values)
    
    if train_check and val_check and test_check:
        logger.info("Stratification verified: Class distributions are consistent across splits.")
        return True
    else:
        logger.error("Stratification issue: Class distributions differ between splits.")
        logger.error(f"  Original distribution: {all_dist.to_dict()}")
        logger.error(f"  Train distribution: {train_dist.to_dict()}")
        logger.error(f"  Validation distribution: {val_dist.to_dict()}")
        logger.error(f"  Test distribution: {test_dist.to_dict()}")
        return False

def main():
    """Run all tests for the dataset splitting implementation."""
    logger.info("Running dataset splitting...")
    run_dataset_split()
    
    logger.info("\nRunning tests...")
    
    # Run tests
    files_exist = test_split_files_exist()
    proportions_correct = test_split_proportions()
    stratification_correct = test_stratification()
    
    # Report overall results
    logger.info("\nTest results summary:")
    logger.info(f"  Files created correctly: {'PASS' if files_exist else 'FAIL'}")
    logger.info(f"  Split proportions correct: {'PASS' if proportions_correct else 'FAIL'}")
    logger.info(f"  Stratification maintained: {'PASS' if stratification_correct else 'FAIL'}")
    
    if files_exist and proportions_correct and stratification_correct:
        logger.info("\nAll tests PASSED! Task 1.6 successfully implemented.")
    else:
        logger.warning("\nSome tests FAILED. Review the logs above for details.")

if __name__ == "__main__":
    main() 