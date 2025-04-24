"""
Test script for the preprocessing.py module.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data_paths
from src.preprocessing import preprocess_image

def test_image_preprocessing():
    """Test the image preprocessing pipeline."""
    # Paths to data directories
    image_dir = os.path.join('data', 'images')
    ocr_dir = os.path.join('data', 'ocr')
    
    # Load data paths
    print("Loading data paths...")
    df = load_data_paths(image_dir, ocr_dir)
    
    # Select random samples from each class for testing
    samples_per_class = 2
    test_samples = []
    
    for class_name in df['label'].unique():
        class_samples = df[df['label'] == class_name].sample(samples_per_class, random_state=42)
        test_samples.append(class_samples)
    
    test_df = pd.concat(test_samples).reset_index(drop=True)
    print(f"Selected {len(test_df)} samples for testing")
    
    # Test preprocessing with different parameters
    test_configs = [
        {"target_size": (224, 224), "num_channels": 3, "name": "RGB 224x224"},
        {"target_size": (224, 224), "num_channels": 1, "name": "Grayscale 224x224"},
        {"target_size": (128, 128), "num_channels": 3, "name": "RGB 128x128"}
    ]
    
    for i, sample in test_df.iterrows():
        image_path = sample['image_path']
        label = sample['label']
        
        # Load original image for reference
        original_img = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        
        print(f"\nProcessing sample {i+1}/{len(test_df)}")
        print(f"  Class: {label}")
        print(f"  Image path: {image_path}")
        print(f"  Original shape: {original_img.shape}")
        
        # Process with each configuration
        processed_images = []
        
        for config in test_configs:
            processed = preprocess_image(
                image_path,
                target_size=config["target_size"],
                num_channels=config["num_channels"]
            )
            
            # Check if processing was successful
            if processed is None:
                print(f"  Error: Processing failed for {config['name']}")
                continue
                
            # Convert to display format if needed
            if config["num_channels"] == 1:
                display_img = processed.squeeze()  # Remove channel dimension for display
            else:
                # Convert from float32 [0,1] back to uint8 [0,255] for display
                display_img = (processed * 255).astype(np.uint8)
                # Convert from BGR to RGB for display if 3 channels
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            processed_images.append({
                "name": config["name"],
                "image": display_img,
                "shape": processed.shape,
                "min": processed.min(),
                "max": processed.max(),
                "dtype": processed.dtype
            })
            
            print(f"  {config['name']}:")
            print(f"    Shape: {processed.shape}")
            print(f"    Min: {processed.min():.4f}, Max: {processed.max():.4f}")
            print(f"    Data type: {processed.dtype}")
        
        # Visualize results
        if processed_images:
            fig, axes = plt.subplots(1, len(processed_images) + 1, figsize=(5 * (len(processed_images) + 1), 5))
            
            # Display original image
            axes[0].imshow(original_img_rgb)
            axes[0].set_title(f"Original\n{original_img.shape}")
            axes[0].axis('off')
            
            # Display processed images
            for j, proc in enumerate(processed_images):
                if proc["image"].ndim == 2:
                    # Grayscale image
                    axes[j+1].imshow(proc["image"], cmap='gray')
                else:
                    # RGB image
                    axes[j+1].imshow(proc["image"])
                    
                axes[j+1].set_title(f"{proc['name']}\n{proc['shape']}")
                axes[j+1].axis('off')
            
            plt.tight_layout()
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join('results', 'preprocessing')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the figure
            fig_path = os.path.join(results_dir, f"sample_{i+1}_{label}.png")
            plt.savefig(fig_path)
            print(f"  Visualization saved to: {fig_path}")
            plt.close(fig)

if __name__ == "__main__":
    import pandas as pd
    test_image_preprocessing()
    print("\nImage preprocessing test completed!") 