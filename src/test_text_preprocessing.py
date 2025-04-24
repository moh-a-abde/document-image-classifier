"""
Test script for text preprocessing utilities.
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk

# Add src directory to path if running from project root
if os.path.exists(os.path.join(os.getcwd(), 'src')):
    sys.path.append(os.getcwd())
    
from src.preprocessing import preprocess_text
from src.data_loader import load_data_paths, load_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

def test_preprocess_text():
    """Test the text preprocessing function."""
    logger.info("Testing text preprocessing functionality...")
    
    # Set paths to data directories
    image_dir = os.path.join('data', 'images')
    ocr_dir = os.path.join('data', 'ocr')
    
    # Load paths to data
    logger.info("Loading data paths...")
    df = load_data_paths(image_dir, ocr_dir)
    
    # Sample a few documents from each class for testing
    samples_per_class = 3
    results_dir = os.path.join('results', 'preprocessing')
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare dataframe for results
    results = []
    
    # Process samples from each class
    for class_name in df['label'].unique():
        logger.info(f"Processing samples from class: {class_name}")
        class_samples = df[df['label'] == class_name].sample(n=samples_per_class, random_state=42)
        
        for i, (_, row) in enumerate(class_samples.iterrows()):
            text_path = row['text_path']
            text = load_text(text_path)
            
            if text is None:
                logger.warning(f"Failed to load text: {text_path}")
                continue
                
            # Process with default settings
            tokens_default = preprocess_text(text)
            
            # Process with stemming instead of lemmatization
            tokens_stemming = preprocess_text(text, perform_stemming=True, perform_lemmatization=False)
            
            # Process without stopword removal
            tokens_with_stopwords = preprocess_text(text, remove_stopwords=False)
            
            # Process without any additional processing (just cleaning and tokenization)
            tokens_minimal = preprocess_text(text, remove_stopwords=False, perform_stemming=False, perform_lemmatization=False)
            
            # Add to results
            results.append({
                'class': class_name,
                'file': os.path.basename(text_path),
                'original_text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for display
                'original_length': len(text),
                'original_word_count': len(text.split()),
                'default_tokens_count': len(tokens_default),
                'stemming_tokens_count': len(tokens_stemming),
                'with_stopwords_count': len(tokens_with_stopwords),
                'minimal_tokens_count': len(tokens_minimal),
                'default_tokens': tokens_default[:10],  # First 10 tokens for display
                'stemming_tokens': tokens_stemming[:10],
                'with_stopwords_tokens': tokens_with_stopwords[:10],
                'minimal_tokens': tokens_minimal[:10]
            })
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, 'text_preprocessing_results.csv'), index=False)
    
    # Display summary statistics
    logger.info(f"Average tokens per document (default preprocessing): {results_df['default_tokens_count'].mean():.2f}")
    logger.info(f"Average tokens per document (with stemming): {results_df['stemming_tokens_count'].mean():.2f}")
    logger.info(f"Average tokens per document (with stopwords): {results_df['with_stopwords_count'].mean():.2f}")
    logger.info(f"Average tokens per document (minimal processing): {results_df['minimal_tokens_count'].mean():.2f}")
    
    # Create a bar plot of average token counts by class
    plt.figure(figsize=(12, 6))
    class_avg = results_df.groupby('class').agg({
        'default_tokens_count': 'mean',
        'stemming_tokens_count': 'mean',
        'with_stopwords_count': 'mean',
        'minimal_tokens_count': 'mean'
    }).reset_index()
    
    class_avg_melted = pd.melt(
        class_avg, 
        id_vars=['class'], 
        value_vars=['default_tokens_count', 'stemming_tokens_count', 'with_stopwords_count', 'minimal_tokens_count'],
        var_name='Preprocessing Method', 
        value_name='Average Token Count'
    )
    
    class_avg_melted['Preprocessing Method'] = class_avg_melted['Preprocessing Method'].map({
        'default_tokens_count': 'Default (Lemmatization)',
        'stemming_tokens_count': 'Stemming',
        'with_stopwords_count': 'With Stopwords',
        'minimal_tokens_count': 'Minimal Processing'
    })
    
    sns.barplot(data=class_avg_melted, x='class', y='Average Token Count', hue='Preprocessing Method')
    plt.title('Average Token Count by Class and Preprocessing Method')
    plt.xlabel('Class')
    plt.ylabel('Average Token Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'avg_tokens_by_class.png'))
    
    # Create boxplots of token counts
    plt.figure(figsize=(12, 6))
    token_counts = results_df[['default_tokens_count', 'stemming_tokens_count', 'with_stopwords_count', 'minimal_tokens_count']]
    token_counts.columns = ['Default (Lemmatization)', 'Stemming', 'With Stopwords', 'Minimal Processing']
    sns.boxplot(data=token_counts)
    plt.title('Distribution of Token Counts by Preprocessing Method')
    plt.ylabel('Token Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'token_count_distribution.png'))
    
    # Get most common tokens across all samples (default preprocessing)
    all_tokens = []
    for tokens in results_df['default_tokens']:
        all_tokens.extend(tokens)
    
    token_freq = Counter(all_tokens)
    
    # Plot top 20 most common tokens
    plt.figure(figsize=(12, 6))
    token_df = pd.DataFrame.from_dict(token_freq, orient='index', columns=['count']).reset_index()
    token_df = token_df.sort_values('count', ascending=False).head(20)
    sns.barplot(data=token_df, x='index', y='count')
    plt.title('Top 20 Most Common Tokens (Default Preprocessing)')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'top_tokens.png'))
    
    logger.info(f"Text preprocessing results saved to {results_dir}")
    logger.info("Text preprocessing test completed successfully.")
    
if __name__ == "__main__":
    test_preprocess_text() 