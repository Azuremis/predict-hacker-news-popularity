import torch
from torch.utils.data import Dataset
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SkipGramDataset(Dataset):
    """Dataset for Skip-gram with negative sampling."""
    
    def __init__(self, sentences, vocab, window_size=5, neg_samples=5):
        """
        Args:
            sentences: List of tokenized sentences
            vocab: Word2VecVocab instance
            window_size: Maximum distance between current and predicted word
            neg_samples: Number of negative samples per positive context
        """
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.data = []
        
        logger.info("Creating Skip-gram dataset...")
        self._create_word_pairs(sentences)
        
    def _create_word_pairs(self, sentences):
        """Create word pairs for training."""
        for sentence in sentences:
            # Apply subsampling to reduce frequent words
            subsampled = self.vocab.subsample_sentence(sentence)
            
            # Create word pairs
            for i, center_word in enumerate(subsampled):
                if center_word not in self.vocab.word_to_id:
                    continue
                    
                center_id = self.vocab.word_to_id[center_word]
                
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(subsampled), i + self.window_size + 1)
                
                # Get context words
                for j in range(window_start, window_end):
                    if i == j:  # Skip the center word itself
                        continue
                        
                    context_word = subsampled[j]
                    if context_word not in self.vocab.word_to_id:
                        continue
                        
                    context_id = self.vocab.word_to_id[context_word]
                    self.data.append((center_id, context_id))
        
        logger.info(f"Created dataset with {len(self.data)} word pairs")
    
    def __len__(self):
        """Return the number of word pairs."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a training sample: (center_word, context_word, negative_samples)."""
        center_id, context_id = self.data[idx]
        
        # Get negative samples
        neg_ids = self.vocab.get_neg_samples(context_id, self.neg_samples)
        
        # Convert to torch tensors
        center_tensor = torch.LongTensor([center_id])
        context_tensor = torch.LongTensor([context_id])
        neg_tensor = torch.LongTensor(neg_ids)
        
        return center_tensor, context_tensor, neg_tensor


class CBOWSoftmaxDataset(Dataset):
    """Dataset for Continuous Bag of Words (CBOW) with full softmax."""
    
    def __init__(self, sentences, vocab, window_size=5):
        """
        Args:
            sentences: List of tokenized sentences
            vocab: Word2VecVocab instance
            window_size: Maximum distance between context and predicted word
        """
        self.vocab = vocab
        self.window_size = window_size
        self.data = []
        
        logger.info("Creating CBOW Softmax dataset...")
        self._create_context_target_pairs(sentences)
        
    def _create_context_target_pairs(self, sentences):
        """Create context-target pairs for CBOW training."""
        for sentence in sentences:
            # Apply subsampling to reduce frequent words
            subsampled = self.vocab.subsample_sentence(sentence)
            
            # For each position, target word is the center, context words are the surroundings
            for i, target_word in enumerate(subsampled):
                if target_word not in self.vocab.word_to_id:
                    continue
                    
                target_id = self.vocab.word_to_id[target_word]
                
                # Define context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(subsampled), i + self.window_size + 1)
                
                # Get context words
                context_ids = []
                for j in range(window_start, window_end):
                    if i == j:  # Skip the target word itself
                        continue
                        
                    context_word = subsampled[j]
                    if context_word not in self.vocab.word_to_id:
                        continue
                        
                    context_ids.append(self.vocab.word_to_id[context_word])
                
                if context_ids:  # Only add if we have at least one context word
                    self.data.append((context_ids, target_id))
        
        logger.info(f"Created CBOW Softmax dataset with {len(self.data)} context-target pairs")
    
    def __len__(self):
        """Return the number of context-target pairs."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a training sample: (context_words, target_word, num_context_words)."""
        context_ids, target_id = self.data[idx]
        
        # Convert to torch tensors
        # Pad the context_ids to a fixed length for batch processing
        max_context_len = 2 * self.window_size
        padded_context_ids = context_ids[:max_context_len] + [0] * (max_context_len - len(context_ids))
        
        context_tensor = torch.LongTensor(padded_context_ids)
        target_tensor = torch.LongTensor([target_id]).squeeze()  # Squeeze for compatibility with CrossEntropyLoss
        
        # Also return the actual number of context words (for averaging)
        num_context_words = len(context_ids)
        
        return context_tensor, target_tensor, torch.LongTensor([num_context_words]).squeeze()


if __name__ == "__main__":
    # Example usage (requires vocabulary.py)
    from vocabulary import Word2VecVocab
    
    # Sample data
    sample_sentences = [
        ["this", "is", "a", "sample", "sentence"],
        ["another", "example", "sentence", "with", "common", "words"],
        ["this", "sentence", "has", "some", "common", "words", "too"]
    ]
    
    # Create vocabulary
    vocab = Word2VecVocab(min_count=1)
    vocab.build(sample_sentences)
    
    # Create Skip-gram dataset
    sg_dataset = SkipGramDataset(sample_sentences, vocab, window_size=2, neg_samples=3)
    
    # Print a few samples
    print("Skip-gram samples:")
    for i in range(min(3, len(sg_dataset))):
        center, context, neg = sg_dataset[i]
        print(f"Center: {vocab.id_to_word[center.item()]}, Context: {vocab.id_to_word[context.item()]}, Negs: {[vocab.id_to_word[neg_id.item()] for neg_id in neg]}")
    
    # Create CBOW softmax dataset
    cbow_softmax_dataset = CBOWSoftmaxDataset(sample_sentences, vocab, window_size=2)
    
    # Print a few samples
    print("\nCBOW Softmax samples:")
    for i in range(min(3, len(cbow_softmax_dataset))):
        context, target, num_context = cbow_softmax_dataset[i]
        
        # Filter out padding zeros
        valid_context = [vocab.id_to_word[ctx_id.item()] for ctx_id in context if ctx_id != 0]
        
        print(f"Context: {valid_context}, Target: {vocab.id_to_word[target.item()]}, Num context words: {num_context.item()}") 