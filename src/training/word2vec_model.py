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


if __name__ == "__main__":
    # Example usage
    vocab_size = 1000
    embedding_dim = 100
    batch_size = 3
    neg_samples = 5
    
    # Create model
    model = Word2VecModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # Create dummy batch
    center_words = torch.LongTensor(batch_size, 1).random_(0, vocab_size)
    context_words = torch.LongTensor(batch_size, 1).random_(0, vocab_size)
    neg_words = torch.LongTensor(batch_size, neg_samples).random_(0, vocab_size)
    
    # Forward pass
    loss = model(center_words, context_words, neg_words)
    
    print(f"Model parameters: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    print(f"Example batch loss: {loss.item():.4f}")
    print(f"Center embedding shape: {model.get_center_embeddings().shape}")
    print(f"Context embedding shape: {model.get_context_embeddings().shape}") 