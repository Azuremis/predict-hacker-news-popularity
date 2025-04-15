import os
import pandas as pd
import re
import logging
import nltk
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text for Word2Vec training."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, HTML tags, and special characters
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Tokenize text using NLTK's word_tokenize."""
    try:
        # Download NLTK data if not already downloaded
        nltk.download('punkt', quiet=True)
        
        # Clean the text before tokenizing
        text = clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens (optional)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return []

def preprocess_wikipedia_corpus(input_file, output_file=None):
    """Preprocess Wikipedia corpus for Word2Vec training.
    
    Args:
        input_file: Path to Wikipedia dump or text file
        output_file: Optional path to save processed text
        
    Returns:
        List of tokenized sentences
    """
    logger.info(f"Preprocessing Wikipedia corpus from {input_file}")
    
    # This is a placeholder - you'll need to implement specific logic
    # for your Wikipedia corpus format (XML, text, etc.)
    sentences = []
    
    # Example simplified implementation:
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Tokenize the line
            tokens = tokenize_text(line)
            if tokens:
                sentences.append(tokens)
                
            # Log progress periodically
            if i % 10000 == 0:
                logger.info(f"Processed {i} lines from Wikipedia corpus")
    
    # Optionally save processed sentences
    if output_file:
        # Save as text file with one tokenized sentence per line
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(' '.join(sentence) + '\n')
    
    logger.info(f"Preprocessed {len(sentences)} sentences from Wikipedia corpus")
    return sentences

def process_hacker_news_titles(items_path):
    """Process Hacker News titles for Word2Vec fine-tuning.
    
    Args:
        items_path: Path to the items dataset (parquet format)
        
    Returns:
        List of tokenized titles
    """
    logger.info(f"Processing Hacker News titles from {items_path}")
    
    # Load the items dataset
    df_items = pd.read_parquet(items_path)
    
    # Extract and tokenize titles
    titles = []
    for title in df_items['title'].dropna():
        tokens = tokenize_text(title)
        if tokens:
            titles.append(tokens)
    
    logger.info(f"Processed {len(titles)} Hacker News titles")
    return titles

if __name__ == "__main__":
    # Example usage
    print("Example tokenization:", tokenize_text("This is a sample text for Word2Vec tokenization.")) 