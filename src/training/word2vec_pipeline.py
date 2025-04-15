import os
import logging
import time

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import components from separate modules
from text_preprocessing import preprocess_wikipedia_corpus, process_hacker_news_titles, tokenize_text
from vocabulary import Word2VecVocab
from dataset import SkipGramDataset
from word2vec_model import Word2VecModel
from training import train_word2vec_custom, finetune_word2vec_custom
from embedding import get_title_embedding, evaluate_word2vec_model

def main():
    """Main function to orchestrate the entire Word2Vec pipeline."""
    logger.info("Starting Word2Vec pipeline")
    start_time = time.time()
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Paths
    wiki_raw_path = 'data/raw/wikipedia_sample.txt'
    wiki_processed_path = 'data/processed/wiki_processed.txt'
    items_path = 'data/raw/items_100k.parquet'
    wiki_model_dir = 'data/processed/word2vec_wiki'
    hn_model_dir = 'data/processed/word2vec_hn'
    
    # Step 1: Preprocess Wikipedia corpus
    logger.info("Step 1: Preprocessing Wikipedia corpus")
    if not os.path.exists(wiki_processed_path):
        sentences = preprocess_wikipedia_corpus(wiki_raw_path, wiki_processed_path)
    else:
        # Load preprocessed sentences
        sentences = []
        with open(wiki_processed_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
        logger.info(f"Loaded {len(sentences)} preprocessed sentences")
    
    # Step 2: Train Word2Vec on Wikipedia
    logger.info("Step 2: Training Word2Vec on Wikipedia")
    wiki_model, wiki_vocab = train_word2vec_custom(
        sentences=sentences,
        output_dir=wiki_model_dir,
        vector_size=100,
        window=5,
        min_count=5,
        epochs=5
    )
    
    # Step 3: Process Hacker News titles
    logger.info("Step 3: Processing Hacker News titles")
    titles = process_hacker_news_titles(items_path)
    
    # Step 4: Fine-tune on HN titles
    logger.info("Step 4: Fine-tuning Word2Vec on Hacker News titles")
    hn_model, hn_vocab = finetune_word2vec_custom(
        base_model_dir=wiki_model_dir,
        titles=titles,
        output_dir=hn_model_dir,
        epochs=5
    )
    
    # Step 5: Evaluate model
    logger.info("Step 5: Evaluating fine-tuned model")
    evaluate_word2vec_model(hn_model, hn_vocab)
    
    # Example: Get embedding for a sample title
    sample_title = "Show HN: A new approach to word embeddings"
    embedding = get_title_embedding(sample_title, hn_model_dir)
    logger.info(f"Sample title: '{sample_title}'")
    logger.info(f"Embedding shape: {embedding.shape}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Word2Vec pipeline completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 