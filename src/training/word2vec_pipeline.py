import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import time
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import re
import random
from collections import Counter, defaultdict
import math

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

class Word2VecVocab:
    """Custom vocabulary builder for Word2Vec."""
    
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
        self.total_words = 0
        self.discard_probs = {}  # For subsampling frequent words
        self.sample_threshold = 1e-5
        self.neg_table = []
        self.neg_table_size = 100000000
        
    def build(self, sentences):
        """Build vocabulary from tokenized sentences."""
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        for sentence in sentences:
            self.word_counts.update(sentence)
            self.total_words += len(sentence)
        
        # Filter by minimum count and create mappings
        vocab = [word for word, count in self.word_counts.items() 
                if count >= self.min_count]
        
        # Create word-to-id mapping
        for i, word in enumerate(vocab):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
        
        vocab_size = len(self.word_to_id)
        logger.info(f"Vocabulary size: {vocab_size} (from {len(self.word_counts)} unique words)")
        
        # Compute subsampling probabilities for frequent words
        self._compute_subsample_probs()
        
        # Build negative sampling table
        self._build_neg_sampling_table()
        
        return vocab_size
    
    def _compute_subsample_probs(self):
        """Compute probability of keeping frequent words during training."""
        logger.info("Computing subsampling probabilities...")
        
        for word in self.word_to_id:
            word_freq = self.word_counts[word] / self.total_words
            # Formula from original Word2Vec paper
            self.discard_probs[word] = 1.0 - np.sqrt(self.sample_threshold / word_freq)
    
    def _build_neg_sampling_table(self):
        """Build table for negative sampling (unigram distribution raised to 3/4 power)."""
        logger.info("Building negative sampling table...")
        
        counts = np.array([self.word_counts[self.id_to_word[i]] for i in range(len(self.id_to_word))])
        pow_counts = np.power(counts, 0.75)
        norm_counts = pow_counts / pow_counts.sum()
        
        # Create a table where more frequent words appear more often
        self.neg_table = np.zeros(self.neg_table_size, dtype=np.int32)
        p = 0
        i = 0
        for j in range(len(self.id_to_word)):
            p += norm_counts[j]
            while i < self.neg_table_size and i / self.neg_table_size < p:
                self.neg_table[i] = j
                i += 1
    
    def get_neg_samples(self, pos_word_id, num_samples=5):
        """Get negative samples for training."""
        neg_samples = []
        while len(neg_samples) < num_samples:
            neg_id = self.neg_table[random.randint(0, self.neg_table_size - 1)]
            if neg_id != pos_word_id:
                neg_samples.append(neg_id)
        return neg_samples
    
    def subsample_sentence(self, sentence):
        """Apply subsampling to a sentence."""
        result = []
        for word in sentence:
            if word in self.word_to_id:
                prob = self.discard_probs.get(word, 0)
                if random.random() > prob:
                    result.append(word)
        return result


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram with negative sampling."""
    
    def __init__(self, sentences, vocab, window_size=5, neg_samples=5):
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.data = []
        
        logger.info("Creating Skip-gram dataset...")
        
        for sentence in sentences:
            # Apply subsampling
            subsampled = vocab.subsample_sentence(sentence)
            
            # Create word pairs
            for i, center_word in enumerate(subsampled):
                if center_word not in vocab.word_to_id:
                    continue
                    
                center_id = vocab.word_to_id[center_word]
                
                # Define context window
                window_start = max(0, i - window_size)
                window_end = min(len(subsampled), i + window_size + 1)
                
                # Get context words
                for j in range(window_start, window_end):
                    if i == j:
                        continue
                        
                    context_word = subsampled[j]
                    if context_word not in vocab.word_to_id:
                        continue
                        
                    context_id = vocab.word_to_id[context_word]
                    self.data.append((center_id, context_id))
        
        logger.info(f"Created dataset with {len(self.data)} word pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        center_id, context_id = self.data[idx]
        
        # Get negative samples
        neg_ids = self.vocab.get_neg_samples(context_id, self.neg_samples)
        
        return center_id, context_id, neg_ids


class Word2VecModel(nn.Module):
    """PyTorch implementation of Word2Vec Skip-gram with negative sampling."""
    
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        
        # Embedding matrices (input/output)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0, 0)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def forward(self, center_word, context_word, neg_words):
        """Forward pass.
        
        Args:
            center_word: Center word IDs [batch_size]
            context_word: Context word IDs [batch_size]
            neg_words: Negative sample word IDs [batch_size, neg_samples]
            
        Returns:
            Loss value
        """
        # Get embeddings
        center_embed = self.center_embeddings(center_word)  # [batch_size, embed_dim]
        context_embed = self.context_embeddings(context_word)  # [batch_size, embed_dim]
        
        # Positive score: dot product between center and context
        pos_score = torch.sum(center_embed * context_embed, dim=1)  # [batch_size]
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score)))
        
        # Negative scores
        neg_embed = self.context_embeddings(neg_words)  # [batch_size, neg_samples, embed_dim]
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)  # [batch_size, neg_samples]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -torch.mean(torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1))
        
        return pos_loss + neg_loss
    
    def get_center_embeddings(self):
        """Get the center (input) embeddings."""
        return self.center_embeddings.weight.detach().cpu().numpy()
    
    def get_context_embeddings(self):
        """Get the context (output) embeddings."""
        return self.context_embeddings.weight.detach().cpu().numpy()


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

def train_word2vec_custom(sentences, output_dir, vector_size=100, window=5, 
                      min_count=5, neg_samples=5, epochs=5, batch_size=512, 
                      lr=0.025, min_lr=0.0001):
    """Train custom Word2Vec model on corpus.
    
    Args:
        sentences: List of tokenized sentences
        output_dir: Directory to save the trained model
        vector_size: Dimensionality of the embeddings
        window: Maximum distance between current and predicted word
        min_count: Minimum word count to be included in the vocabulary
        neg_samples: Number of negative samples per positive context
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Trained Word2Vec model, vocabulary
    """
    logger.info(f"Training custom Word2Vec model on corpus with {len(sentences)} sentences")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, neg_samples={neg_samples}")
    
    start_time = time.time()
    
    # Build vocabulary
    vocab = Word2VecVocab(min_count=min_count)
    vocab_size = vocab.build(sentences)
    
    # Create dataset and dataloader
    dataset = SkipGramDataset(sentences, vocab, window_size=window, neg_samples=neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = Word2VecModel(vocab_size=vocab_size, embedding_dim=vector_size)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader), eta_min=min_lr
    )
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        for i, (center_ids, context_ids, neg_ids) in enumerate(dataloader):
            # Forward pass
            loss = model(center_ids, context_ids, neg_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress
            if i % 1000 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as NumPy arrays
    in_embeddings = model.get_center_embeddings()
    out_embeddings = model.get_context_embeddings()
    
    np.save(os.path.join(output_dir, "word2vec_in.npy"), in_embeddings)
    np.save(os.path.join(output_dir, "word2vec_out.npy"), out_embeddings)
    
    # Save vocabulary
    vocab_data = {
        'word_to_id': vocab.word_to_id,
        'id_to_word': vocab.id_to_word
    }
    
    torch.save(vocab_data, os.path.join(output_dir, "vocab.pth"))
    
    # Save full model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': vector_size
    }, os.path.join(output_dir, "word2vec_model.pth"))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Word2Vec training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Model saved to {output_dir}")
    
    return model, vocab

def finetune_word2vec_custom(base_model_dir, titles, output_dir, epochs=5, lr=0.01, min_lr=0.0001, batch_size=256):
    """Fine-tune pre-trained Word2Vec model on Hacker News titles.
    
    Args:
        base_model_dir: Directory containing pre-trained Word2Vec model
        titles: List of tokenized Hacker News titles
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        lr: Initial learning rate
        min_lr: Minimum learning rate
        batch_size: Batch size for training
        
    Returns:
        Fine-tuned Word2Vec model, vocabulary
    """
    logger.info(f"Fine-tuning Word2Vec model on {len(titles)} Hacker News titles")
    
    # Load pre-trained model
    model_path = os.path.join(base_model_dir, "word2vec_model.pth")
    vocab_path = os.path.join(base_model_dir, "vocab.pth")
    
    checkpoint = torch.load(model_path)
    vocab_data = torch.load(vocab_path)
    
    # Initialize model with pre-trained weights
    model = Word2VecModel(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reconstruct vocabulary
    vocab = Word2VecVocab(min_count=1)  # Lower threshold for fine-tuning
    vocab.word_to_id = vocab_data['word_to_id']
    vocab.id_to_word = vocab_data['id_to_word']
    
    # Update vocabulary with HN-specific words
    logger.info(f"Updating vocabulary with HN-specific words")
    all_words = set()
    for title in titles:
        all_words.update(title)
    
    new_words = all_words - set(vocab.word_to_id.keys())
    logger.info(f"Found {len(new_words)} new words in HN titles")
    
    # Add new words to vocabulary
    if new_words:
        # Expand embedding matrices
        old_vocab_size = len(vocab.word_to_id)
        embedding_dim = model.embedding_dim
        
        # Create new model with expanded vocabulary
        new_vocab_size = old_vocab_size + len(new_words)
        new_model = Word2VecModel(vocab_size=new_vocab_size, embedding_dim=embedding_dim)
        
        # Copy existing embeddings
        with torch.no_grad():
            new_model.center_embeddings.weight[:old_vocab_size] = model.center_embeddings.weight
            new_model.context_embeddings.weight[:old_vocab_size] = model.context_embeddings.weight
        
        # Update vocabulary
        for i, word in enumerate(new_words, start=old_vocab_size):
            vocab.word_to_id[word] = i
            vocab.id_to_word[i] = word
        
        model = new_model
    
    # Create dataset and dataloader
    dataset = SkipGramDataset(titles, vocab, window_size=5, neg_samples=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader), eta_min=min_lr
    )
    
    # Training loop
    logger.info(f"Starting fine-tuning for {epochs} epochs")
    for epoch in range(epochs):
        total_loss = 0
        for i, (center_ids, context_ids, neg_ids) in enumerate(dataloader):
            # Forward pass
            loss = model(center_ids, context_ids, neg_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress
            if i % 100 == 0 and i > 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
    
    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as NumPy arrays
    in_embeddings = model.get_center_embeddings()
    out_embeddings = model.get_context_embeddings()
    
    np.save(os.path.join(output_dir, "word2vec_hn_in.npy"), in_embeddings)
    np.save(os.path.join(output_dir, "word2vec_hn_out.npy"), out_embeddings)
    
    # Save vocabulary
    vocab_data = {
        'word_to_id': vocab.word_to_id,
        'id_to_word': vocab.id_to_word
    }
    
    torch.save(vocab_data, os.path.join(output_dir, "vocab_hn.pth"))
    
    # Save full model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab.word_to_id),
        'embedding_dim': model.embedding_dim
    }, os.path.join(output_dir, "word2vec_hn_model.pth"))
    
    logger.info(f"Fine-tuned model saved to {output_dir}")
    
    return model, vocab

def evaluate_word2vec_model(model, vocab, word_pairs=None):
    """Evaluate Word2Vec model by printing similar words to common terms.
    
    Args:
        model: Trained Word2Vec model
        vocab: Word2Vec vocabulary
        word_pairs: Optional list of word pairs to check for similarity
    """
    logger.info("Evaluating Word2Vec model")
    
    # Get embeddings
    embeddings = model.get_center_embeddings()
    
    # Normalize embeddings for cosine similarity
    norms = np.sqrt((embeddings ** 2).sum(axis=1))
    normalized_embeddings = embeddings / norms[:, np.newaxis]
    
    # Check similar words for some common tech terms
    test_words = ['data', 'python', 'algorithm', 'startup', 'security']
    
    for word in test_words:
        if word in vocab.word_to_id:
            word_id = vocab.word_to_id[word]
            word_vec = normalized_embeddings[word_id]
            
            # Compute similarities with all words
            similarities = np.dot(normalized_embeddings, word_vec)
            
            # Get top 5 similar words (excluding the word itself)
            most_similar = []
            for i in np.argsort(similarities)[-10:]:
                if i != word_id:
                    sim_word = vocab.id_to_word[i]
                    sim_score = similarities[i]
                    most_similar.append((sim_word, sim_score))
            
            most_similar = sorted(most_similar, key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Words similar to '{word}': {most_similar}")
        else:
            logger.info(f"Word '{word}' not in vocabulary")
    
    # Check word pairs if provided
    if word_pairs:
        for word1, word2 in word_pairs:
            if word1 in vocab.word_to_id and word2 in vocab.word_to_id:
                id1 = vocab.word_to_id[word1]
                id2 = vocab.word_to_id[word2]
                
                vec1 = normalized_embeddings[id1]
                vec2 = normalized_embeddings[id2]
                
                similarity = np.dot(vec1, vec2)
                logger.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            else:
                logger.info(f"Cannot compute similarity - words not in vocabulary")

def get_title_embedding(title, model_dir):
    """Get embedding for a Hacker News title using Word2Vec model.
    
    Args:
        title: String containing the title text
        model_dir: Directory containing the saved model
        
    Returns:
        Numpy array with title embedding (average of word vectors)
    """
    # Load model data
    vocab_path = os.path.join(model_dir, "vocab_hn.pth")
    embedding_path = os.path.join(model_dir, "word2vec_hn_in.npy")
    
    # Load vocabulary and embeddings
    vocab_data = torch.load(vocab_path)
    word_to_id = vocab_data['word_to_id']
    embeddings = np.load(embedding_path)
    
    # Tokenize the title
    tokens = tokenize_text(title)
    
    # Get embeddings for tokens in vocabulary
    token_embeddings = []
    for token in tokens:
        if token in word_to_id:
            token_embeddings.append(embeddings[word_to_id[token]])
    
    # Return average embedding or zero vector if no tokens in vocabulary
    if token_embeddings:
        return np.mean(token_embeddings, axis=0)
    else:
        return np.zeros(embeddings.shape[1])

def main():
    """Main function to train Word2Vec models."""
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Paths
    wiki_raw_path = 'data/raw/wikipedia_sample.txt'
    wiki_processed_path = 'data/processed/wiki_processed.txt'
    items_path = 'data/raw/items_100k.parquet'
    wiki_model_dir = 'data/processed/word2vec_wiki'
    hn_model_dir = 'data/processed/word2vec_hn'
    
    # Check if we need to preprocess Wikipedia
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
    
    # Train Word2Vec on Wikipedia
    wiki_model, wiki_vocab = train_word2vec_custom(
        sentences=sentences,
        output_dir=wiki_model_dir,
        vector_size=100,
        window=5,
        min_count=5,
        epochs=5
    )
    
    # Process HN titles
    titles = process_hacker_news_titles(items_path)
    
    # Fine-tune on HN titles
    hn_model, hn_vocab = finetune_word2vec_custom(
        base_model_dir=wiki_model_dir,
        titles=titles,
        output_dir=hn_model_dir,
        epochs=5
    )
    
    # Evaluate model
    evaluate_word2vec_model(hn_model, hn_vocab)
    
    logger.info("Word2Vec training pipeline completed")

if __name__ == "__main__":
    main() 