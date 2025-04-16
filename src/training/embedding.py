import os
import torch
import numpy as np
import logging
from text_preprocessing import tokenize_text

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_title_embedding(title, model_dir):
    """Get embedding for a Hacker News title using Word2Vec model.
    
    Args:
        title: String containing the title text
        model_dir: Directory containing the saved model
        
    Returns:
        Numpy array with title embedding (average of word vectors)
    """
    # Check if this is a CBOW with softmax or Skip-gram model based on file names
    dir_name = os.path.basename(model_dir)
    is_cbow_softmax = 'cbow_softmax' in dir_name
    
    # Set file paths based on model type
    if is_cbow_softmax:
        vocab_path = os.path.join(model_dir, "vocab_hn.pth")
        embedding_path = os.path.join(model_dir, "cbow_softmax_hn_in.npy")
    else:  # Skip-gram
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

def evaluate_word2vec_model(model, vocab, word_pairs=None):
    """Evaluate Word2Vec model by printing similar words to common terms.
    
    Args:
        model: Trained Word2Vec model
        vocab: Word2Vec vocabulary
        word_pairs: Optional list of word pairs to check for similarity
    """
    logger.info("Evaluating Word2Vec model")
    
    # Get embeddings - use appropriate method based on model type
    if hasattr(model, 'get_center_embeddings'):
        # Skip-gram model
        embeddings = model.get_center_embeddings()
    else:
        # CBOW softmax model
        embeddings = model.get_context_embeddings()
    
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

def find_analogy(model, vocab, word1, word2, word3, top_n=5):
    """Find analogies like 'word1 is to word2 as word3 is to ???'.
    
    Args:
        model: Trained Word2Vec model
        vocab: Word2Vec vocabulary
        word1, word2, word3: Words for the analogy
        top_n: Number of results to return
        
    Returns:
        List of (word, similarity) tuples for top matches
    """
    # Get and normalize embeddings
    if hasattr(model, 'get_center_embeddings'):
        # Skip-gram model
        embeddings = model.get_center_embeddings()
    else:
        # CBOW softmax model
        embeddings = model.get_context_embeddings()
    
    norms = np.sqrt((embeddings ** 2).sum(axis=1))
    normalized_embeddings = embeddings / norms[:, np.newaxis]
    
    # Check if all words are in vocabulary
    if not all(word in vocab.word_to_id for word in [word1, word2, word3]):
        missing = [word for word in [word1, word2, word3] if word not in vocab.word_to_id]
        logger.warning(f"Words not in vocabulary: {missing}")
        return []
    
    # Get word vectors
    vec1 = normalized_embeddings[vocab.word_to_id[word1]]
    vec2 = normalized_embeddings[vocab.word_to_id[word2]]
    vec3 = normalized_embeddings[vocab.word_to_id[word3]]
    
    # Compute target vector: vec2 - vec1 + vec3
    target_vec = vec2 - vec1 + vec3
    
    # Normalize target vector
    target_vec = target_vec / np.sqrt((target_vec ** 2).sum())
    
    # Compute similarities with all words
    similarities = np.dot(normalized_embeddings, target_vec)
    
    # Find top matches, excluding the input words
    results = []
    exclude_ids = [vocab.word_to_id[word] for word in [word1, word2, word3]]
    
    for i in np.argsort(similarities)[-30:]:  # Get extra to filter out input words
        if i not in exclude_ids:
            results.append((vocab.id_to_word[i], similarities[i]))
    
    # Sort and return top_n
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


def load_embeddings_model(model_dir):
    """Load embeddings and vocabulary from model directory.
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        model: Loaded model
        vocab: Loaded vocabulary
    """
    # Check model type based on directory name
    dir_name = os.path.basename(model_dir)
    is_cbow_softmax = 'cbow_softmax' in dir_name
    
    if is_cbow_softmax:
        from word2vec_model import CBOWSoftmaxModel
        model_path = os.path.join(model_dir, "cbow_softmax_model.pth")
    else:
        from word2vec_model import Word2VecModel
        model_path = os.path.join(model_dir, "word2vec_model.pth")
    
    vocab_path = os.path.join(model_dir, "vocab.pth")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        logger.error(f"Model files not found in {model_dir}")
        return None, None
    
    # Load model data
    checkpoint = torch.load(model_path)
    vocab_data = torch.load(vocab_path)
    
    # Initialize model with trained weights
    if is_cbow_softmax:
        model = CBOWSoftmaxModel(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
    else:
        model = Word2VecModel(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reconstruct vocabulary
    from vocabulary import Word2VecVocab
    vocab = Word2VecVocab(min_count=1)
    vocab.word_to_id = vocab_data['word_to_id']
    vocab.id_to_word = vocab_data['id_to_word']
    
    return model, vocab


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent))
    
    # Load a trained model
    model_dir = "example_skipgram_model"  # Replace with your model directory
    
    model, vocab = load_embeddings_model(model_dir)
    
    if model and vocab:
        # Get embedding for a sample title
        title = "How to effectively train word embeddings"
        embedding = get_title_embedding(title, model_dir)
        print(f"Title: '{title}'")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}...")
        
        # Evaluate model
        evaluate_word2vec_model(model, vocab)
    else:
        print(f"Model not found in {model_dir}. Run training.py first to train a model.") 