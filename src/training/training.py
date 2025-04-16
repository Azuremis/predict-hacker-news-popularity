import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    # Import here to avoid circular imports
    from vocabulary import Word2VecVocab
    from dataset import SkipGramDataset
    from word2vec_model import Word2VecModel
    
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


def train_cbow_custom(sentences, output_dir, vector_size=100, window=5, 
                      min_count=5, neg_samples=5, epochs=5, batch_size=512, 
                      lr=0.025, min_lr=0.0001):
    """Train custom CBOW model on corpus.
    
    Args:
        sentences: List of tokenized sentences
        output_dir: Directory to save the trained model
        vector_size: Dimensionality of the embeddings
        window: Maximum distance between context and predicted word
        min_count: Minimum word count to be included in the vocabulary
        neg_samples: Number of negative samples per positive context
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Trained CBOW model, vocabulary
    """
    # Import here to avoid circular imports
    from vocabulary import Word2VecVocab
    from dataset import CBOWDataset
    from word2vec_model import CBOWModel
    
    logger.info(f"Training custom CBOW model on corpus with {len(sentences)} sentences")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, neg_samples={neg_samples}")
    
    start_time = time.time()
    
    # Build vocabulary
    vocab = Word2VecVocab(min_count=min_count)
    vocab_size = vocab.build(sentences)
    
    # Create dataset and dataloader
    dataset = CBOWDataset(sentences, vocab, window_size=window, neg_samples=neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = CBOWModel(vocab_size=vocab_size, embedding_dim=vector_size)
    
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
        for i, (context_ids, target_ids, neg_ids, num_context) in enumerate(dataloader):
            # Forward pass
            loss = model(context_ids, target_ids, neg_ids, num_context)
            
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
    in_embeddings = model.get_context_embeddings()
    out_embeddings = model.get_target_embeddings()
    
    np.save(os.path.join(output_dir, "cbow_in.npy"), in_embeddings)
    np.save(os.path.join(output_dir, "cbow_out.npy"), out_embeddings)
    
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
    }, os.path.join(output_dir, "cbow_model.pth"))
    
    elapsed_time = time.time() - start_time
    logger.info(f"CBOW training completed in {elapsed_time:.2f} seconds")
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
    # Import here to avoid circular imports
    from vocabulary import Word2VecVocab
    from dataset import SkipGramDataset
    from word2vec_model import Word2VecModel
    
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


def finetune_cbow_custom(base_model_dir, titles, output_dir, epochs=5, lr=0.01, min_lr=0.0001, batch_size=256):
    """Fine-tune pre-trained CBOW model on Hacker News titles.
    
    Args:
        base_model_dir: Directory containing pre-trained CBOW model
        titles: List of tokenized Hacker News titles
        output_dir: Directory to save the fine-tuned model
        epochs: Number of training epochs
        lr: Initial learning rate
        min_lr: Minimum learning rate
        batch_size: Batch size for training
        
    Returns:
        Fine-tuned CBOW model, vocabulary
    """
    # Import here to avoid circular imports
    from vocabulary import Word2VecVocab
    from dataset import CBOWDataset
    from word2vec_model import CBOWModel
    
    logger.info(f"Fine-tuning CBOW model on {len(titles)} Hacker News titles")
    
    # Load pre-trained model
    model_path = os.path.join(base_model_dir, "cbow_model.pth")
    vocab_path = os.path.join(base_model_dir, "vocab.pth")
    
    checkpoint = torch.load(model_path)
    vocab_data = torch.load(vocab_path)
    
    # Initialize model with pre-trained weights
    model = CBOWModel(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
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
        new_model = CBOWModel(vocab_size=new_vocab_size, embedding_dim=embedding_dim)
        
        # Copy existing embeddings
        with torch.no_grad():
            new_model.context_embeddings.weight[:old_vocab_size] = model.context_embeddings.weight
            new_model.target_embeddings.weight[:old_vocab_size] = model.target_embeddings.weight
        
        # Update vocabulary
        for i, word in enumerate(new_words, start=old_vocab_size):
            vocab.word_to_id[word] = i
            vocab.id_to_word[i] = word
        
        model = new_model
    
    # Create dataset and dataloader
    dataset = CBOWDataset(titles, vocab, window_size=5, neg_samples=5)
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
        for i, (context_ids, target_ids, neg_ids, num_context) in enumerate(dataloader):
            # Forward pass
            loss = model(context_ids, target_ids, neg_ids, num_context)
            
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
    in_embeddings = model.get_context_embeddings()
    out_embeddings = model.get_target_embeddings()
    
    np.save(os.path.join(output_dir, "cbow_hn_in.npy"), in_embeddings)
    np.save(os.path.join(output_dir, "cbow_hn_out.npy"), out_embeddings)
    
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
    }, os.path.join(output_dir, "cbow_hn_model.pth"))
    
    logger.info(f"Fine-tuned CBOW model saved to {output_dir}")
    
    return model, vocab


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent))
    
    from text_preprocessing import tokenize_text
    
    # Create sample sentences
    sentences = [
        tokenize_text("This is a sample sentence for training Word2Vec."),
        tokenize_text("Another example to demonstrate how the training works."),
        tokenize_text("Word2Vec models learn word embeddings from context.")
    ]
    
    # Train Skip-gram model
    sg_output_dir = "example_skipgram_model"
    sg_model, sg_vocab = train_word2vec_custom(
        sentences=sentences,
        output_dir=sg_output_dir,
        vector_size=50,
        window=2,
        min_count=1,
        epochs=3,
        batch_size=2
    )
    
    # Train CBOW model
    cbow_output_dir = "example_cbow_model"
    cbow_model, cbow_vocab = train_cbow_custom(
        sentences=sentences,
        output_dir=cbow_output_dir,
        vector_size=50,
        window=2,
        min_count=1,
        epochs=3,
        batch_size=2
    )
    
    # Fine-tune Skip-gram model
    titles = [
        tokenize_text("How to train Word2Vec models effectively"),
        tokenize_text("Understanding embeddings in NLP")
    ]
    
    finetuned_sg_model, finetuned_sg_vocab = finetune_word2vec_custom(
        base_model_dir=sg_output_dir,
        titles=titles,
        output_dir="finetuned_skipgram_model",
        epochs=2
    )
    
    # Fine-tune CBOW model
    finetuned_cbow_model, finetuned_cbow_vocab = finetune_cbow_custom(
        base_model_dir=cbow_output_dir,
        titles=titles,
        output_dir="finetuned_cbow_model",
        epochs=2
    )
    
    print("Training and fine-tuning completed successfully!") 