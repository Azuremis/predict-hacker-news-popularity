import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import logging
import time
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import re

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

def train_word2vec_wikipedia(sentences, model_path, vector_size=100, window=5, min_count=5, workers=4, sg=1):
    """Train Word2Vec model on Wikipedia corpus.
    
    Args:
        sentences: List of tokenized sentences
        model_path: Path to save the trained model
        vector_size: Dimensionality of the embeddings
        window: Maximum distance between current and predicted word
        min_count: Minimum word count to be included in the vocabulary
        workers: Number of CPU cores to use
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        
    Returns:
        Trained Word2Vec model
    """
    logger.info(f"Training Word2Vec model on Wikipedia corpus with {len(sentences)} sentences")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, sg={sg}")
    
    start_time = time.time()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Word2Vec training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Vocabulary size: {len(model.wv.key_to_index)}")
    
    return model

def finetune_word2vec_hn_titles(base_model_path, titles, output_model_path, epochs=5):
    """Fine-tune pre-trained Word2Vec model on Hacker News titles.
    
    Args:
        base_model_path: Path to pre-trained Word2Vec model
        titles: List of tokenized Hacker News titles
        output_model_path: Path to save the fine-tuned model
        epochs: Number of training epochs
        
    Returns:
        Fine-tuned Word2Vec model
    """
    logger.info(f"Fine-tuning Word2Vec model on {len(titles)} Hacker News titles")
    
    # Load pre-trained model
    model = Word2Vec.load(base_model_path)
    logger.info(f"Loaded base model from {base_model_path}")
    logger.info(f"Initial vocabulary size: {len(model.wv.key_to_index)}")
    
    # Update vocabulary with HN titles
    model.build_vocab(titles, update=True)
    logger.info(f"Updated vocabulary size: {len(model.wv.key_to_index)}")
    
    # Train on HN titles
    model.train(
        titles,
        total_examples=len(titles),
        epochs=epochs
    )
    
    # Save the fine-tuned model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save(output_model_path)
    logger.info(f"Fine-tuned model saved to {output_model_path}")
    
    return model

def evaluate_word2vec_model(model, word_pairs=None):
    """Evaluate Word2Vec model by printing similar words to common terms.
    
    Args:
        model: Trained Word2Vec model
        word_pairs: Optional list of word pairs to check for similarity
    """
    logger.info("Evaluating Word2Vec model")
    
    # Check similar words for some common tech terms
    test_words = ['data', 'python', 'algorithm', 'startup', 'security']
    
    for word in test_words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            logger.info(f"Words similar to '{word}': {similar_words}")
        else:
            logger.info(f"Word '{word}' not in vocabulary")
    
    # Check word pairs if provided
    if word_pairs:
        for word1, word2 in word_pairs:
            if word1 in model.wv and word2 in model.wv:
                similarity = model.wv.similarity(word1, word2)
                logger.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            else:
                logger.info(f"Cannot compute similarity - words not in vocabulary")

def get_title_embedding(title, model):
    """Get embedding for a Hacker News title using Word2Vec model.
    
    Args:
        title: String containing the title text
        model: Trained Word2Vec model
        
    Returns:
        Numpy array with title embedding (average of word vectors)
    """
    # Tokenize the title
    tokens = tokenize_text(title)
    
    # Get embeddings for tokens in vocabulary
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    
    # Return average embedding or zero vector if no tokens in vocabulary
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def main():
    """Main function to run the Word2Vec pipeline."""
    # Paths
    wiki_input_path = '../data/raw/wikipedia_corpus.txt'  # Replace with actual path
    wiki_processed_path = '../data/processed/wikipedia_processed.txt'
    wiki_model_path = '../data/processed/word2vec_wiki.model'
    items_path = '../data/raw/items_100k.parquet'
    finetuned_model_path = '../data/processed/word2vec_hn_finetuned.model'
    
    # 1. Check if we need to train from scratch or use existing model
    if os.path.exists(wiki_model_path) and not os.path.exists(wiki_input_path):
        logger.info(f"Using existing Word2Vec model at {wiki_model_path}")
        model = Word2Vec.load(wiki_model_path)
    else:
        # 2. Preprocess Wikipedia corpus (if available)
        if os.path.exists(wiki_input_path):
            wiki_sentences = preprocess_wikipedia_corpus(wiki_input_path, wiki_processed_path)
            
            # 3. Train Word2Vec on Wikipedia
            model = train_word2vec_wikipedia(wiki_sentences, wiki_model_path)
        else:
            logger.warning("No Wikipedia corpus found and no pre-trained model. Cannot proceed with pre-training.")
            return
    
    # 4. Process Hacker News titles
    if os.path.exists(items_path):
        hn_titles = process_hacker_news_titles(items_path)
        
        # 5. Fine-tune Word2Vec on HN titles
        finetuned_model = finetune_word2vec_hn_titles(wiki_model_path, hn_titles, finetuned_model_path)
        
        # 6. Evaluate the model
        evaluate_word2vec_model(finetuned_model)
    else:
        logger.error(f"Items dataset not found at {items_path}")
        return

if __name__ == "__main__":
    main() 