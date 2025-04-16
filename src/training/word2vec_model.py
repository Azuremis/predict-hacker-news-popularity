import torch
import torch.nn as nn
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Word2VecModel(nn.Module):
    """PyTorch implementation of Word2Vec Skip-gram with negative sampling."""
    
    def __init__(self, vocab_size, embedding_dim=100):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimensionality of the embeddings
        """
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
            center_word: Center word IDs [batch_size, 1]
            context_word: Context word IDs [batch_size, 1]
            neg_words: Negative sample word IDs [batch_size, neg_samples]
            
        Returns:
            Loss value
        """
        # Reshape tensors if needed
        center_word = center_word.squeeze()
        context_word = context_word.squeeze()
        
        # Get embeddings
        center_embed = self.center_embeddings(center_word)  # [batch_size, embed_dim]
        context_embed = self.context_embeddings(context_word)  # [batch_size, embed_dim]
        
        # Positive score: dot product between center and context
        pos_score = torch.sum(center_embed * context_embed, dim=1)  # [batch_size]
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score)))
        
        # Negative scores
        neg_embed = self.context_embeddings(neg_words)  # [batch_size, neg_samples, embed_dim]
        
        # Reshape center_embed for batch matrix multiplication
        center_embed_reshaped = center_embed.unsqueeze(2)  # [batch_size, embed_dim, 1]
        
        # Batch matrix multiplication
        neg_score = torch.bmm(neg_embed, center_embed_reshaped).squeeze(2)  # [batch_size, neg_samples]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -torch.mean(torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1))
        
        # Total loss
        total_loss = pos_loss + neg_loss
        
        return total_loss
    
    def get_center_embeddings(self):
        """Get the center (input) embeddings."""
        return self.center_embeddings.weight.detach().cpu().numpy()
    
    def get_context_embeddings(self):
        """Get the context (output) embeddings."""
        return self.context_embeddings.weight.detach().cpu().numpy()
    
    def get_word_vector(self, word_id):
        """Get vector for a specific word."""
        return self.center_embeddings.weight[word_id].detach().cpu().numpy()
    
    def save_embeddings(self, output_file):
        """Save embeddings to numpy file."""
        embeddings = self.get_center_embeddings()
        np.save(output_file, embeddings)
        logger.info(f"Saved embeddings to {output_file}")


class CBOWModel(nn.Module):
    """PyTorch implementation of Word2Vec CBOW with negative sampling."""
    
    def __init__(self, vocab_size, embedding_dim=100):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimensionality of the embeddings
        """
        super().__init__()
        
        # Embedding matrices (input/output)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.target_embeddings.weight.data.uniform_(-0, 0)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def forward(self, context_words, target_word, neg_words, num_context_words):
        """Forward pass.
        
        Args:
            context_words: Context word IDs [batch_size, max_context_len]
            target_word: Target word IDs [batch_size, 1]
            neg_words: Negative sample word IDs [batch_size, neg_samples]
            num_context_words: Actual number of context words for each example [batch_size, 1]
            
        Returns:
            Loss value
        """
        batch_size = context_words.size(0)
        max_context_len = context_words.size(1)
        
        # Reshape tensors if needed
        target_word = target_word.squeeze()
        num_context_words = num_context_words.squeeze()
        
        # Create a mask for valid context words (non-padding)
        mask = torch.arange(max_context_len).expand(batch_size, max_context_len).to(context_words.device)
        mask = mask < num_context_words.unsqueeze(1)
        
        # Get context embeddings with mask to handle variable lengths
        context_embed = self.context_embeddings(context_words)  # [batch_size, max_context_len, embed_dim]
        
        # Apply mask - set padding embeddings to zero
        mask = mask.unsqueeze(2).float()  # Add embedding dimension [batch_size, max_context_len, 1]
        context_embed = context_embed * mask
        
        # Average context embeddings for each example
        context_sum = torch.sum(context_embed, dim=1)  # [batch_size, embed_dim]
        
        # Make sure to avoid division by zero
        valid_counts = torch.clamp(num_context_words, min=1).unsqueeze(1).float()
        avg_context_embed = context_sum / valid_counts  # [batch_size, embed_dim]
        
        # Get target embedding
        target_embed = self.target_embeddings(target_word)  # [batch_size, embed_dim]
        
        # Positive score: dot product between averaged context and target
        pos_score = torch.sum(avg_context_embed * target_embed, dim=1)  # [batch_size]
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_loss = -torch.mean(torch.log(torch.sigmoid(pos_score)))
        
        # Negative scores
        neg_embed = self.target_embeddings(neg_words)  # [batch_size, neg_samples, embed_dim]
        
        # Reshape context_embed for batch matrix multiplication
        avg_context_reshaped = avg_context_embed.unsqueeze(2)  # [batch_size, embed_dim, 1]
        
        # Batch matrix multiplication
        neg_score = torch.bmm(neg_embed, avg_context_reshaped).squeeze(2)  # [batch_size, neg_samples]
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_loss = -torch.mean(torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1))
        
        # Total loss
        total_loss = pos_loss + neg_loss
        
        return total_loss
    
    def get_context_embeddings(self):
        """Get the context (input) embeddings."""
        return self.context_embeddings.weight.detach().cpu().numpy()
    
    def get_target_embeddings(self):
        """Get the target (output) embeddings."""
        return self.target_embeddings.weight.detach().cpu().numpy()
    
    def get_word_vector(self, word_id):
        """Get vector for a specific word."""
        return self.context_embeddings.weight[word_id].detach().cpu().numpy()
    
    def save_embeddings(self, output_file):
        """Save embeddings to numpy file."""
        embeddings = self.get_context_embeddings()
        np.save(output_file, embeddings)
        logger.info(f"Saved embeddings to {output_file}")


if __name__ == "__main__":
    # Simple test
    vocab_size = 1000
    batch_size = 16
    neg_samples = 5
    embed_dim = 50
    max_context_len = 8
    
    # Test Skip-gram model
    sg_model = Word2VecModel(vocab_size=vocab_size, embedding_dim=embed_dim)
    
    # Create some dummy data
    center_words = torch.randint(0, vocab_size, (batch_size, 1))
    context_words = torch.randint(0, vocab_size, (batch_size, 1))
    neg_words = torch.randint(0, vocab_size, (batch_size, neg_samples))
    
    # Forward pass
    sg_loss = sg_model(center_words, context_words, neg_words)
    print(f"Skip-gram model loss: {sg_loss.item()}")
    
    # Test CBOW model
    cbow_model = CBOWModel(vocab_size=vocab_size, embedding_dim=embed_dim)
    
    # Create dummy data for CBOW
    context_words = torch.randint(0, vocab_size, (batch_size, max_context_len))
    target_words = torch.randint(0, vocab_size, (batch_size, 1))
    neg_words = torch.randint(0, vocab_size, (batch_size, neg_samples))
    num_context_words = torch.randint(1, max_context_len + 1, (batch_size, 1))
    
    # Forward pass
    cbow_loss = cbow_model(context_words, target_words, neg_words, num_context_words)
    print(f"CBOW model loss: {cbow_loss.item()}") 