"""
Test script for the data_loader.py module.
"""

import os
import sys
import pandas as pd

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data_paths, create_data_splits

def test_data_loading():
    """Test loading data paths and splitting the dataset."""
    # Paths to data directories
    image_dir = os.path.join('data', 'images')
    ocr_dir = os.path.join('data', 'ocr')
    
    # Load data paths
    print("Loading data paths...")
    df = load_data_paths(image_dir, ocr_dir)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {len(df['label'].unique())}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Verify a few random samples by checking file existence
    print("\nVerifying random samples:")
    for i in range(min(5, len(df))):
        sample = df.iloc[i]
        img_exists = os.path.exists(sample['image_path'])
        txt_exists = os.path.exists(sample['text_path'])
        print(f"Sample {i+1}:")
        print(f"  Label: {sample['label']}")
        print(f"  Image: {sample['image_path']} (Exists: {img_exists})")
        print(f"  Text: {sample['text_path']} (Exists: {txt_exists})")
    
    # Test data splitting
    print("\nTesting data splitting...")
    train_df, val_df, test_df = create_data_splits(df)
    
    # Verify split sizes
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify class distribution in splits
    print("\nClass distribution in splits:")
    original_dist = df['label'].value_counts(normalize=True)
    train_dist = train_df['label'].value_counts(normalize=True)
    val_dist = val_df['label'].value_counts(normalize=True)
    test_dist = test_df['label'].value_counts(normalize=True)
    
    comparison = pd.DataFrame({
        'Original': original_dist,
        'Train': train_dist,
        'Validation': val_dist,
        'Test': test_dist
    })
    print(comparison)
    
    return df, train_df, val_df, test_df

if __name__ == "__main__":
    df, train_df, val_df, test_df = test_data_loading()
    print("Data loading and splitting completed successfully!") 