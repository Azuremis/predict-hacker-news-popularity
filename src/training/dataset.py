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
    
    # Create dataset
    dataset = SkipGramDataset(sample_sentences, vocab, window_size=2, neg_samples=3)
    
    # Show first few examples
    for i in range(min(5, len(dataset))):
        center, context, negs = dataset[i]
        print(f"Example {i+1}:")
        print(f"  Center word: {vocab.id_to_word[center.item()]}")
        print(f"  Context word: {vocab.id_to_word[context.item()]}")
        print(f"  Negative samples: {[vocab.id_to_word[neg.item()] for neg in negs]}")
        print() 