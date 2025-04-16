import numpy as np
import logging
import random
from collections import Counter

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

    def __len__(self):
        """Return vocabulary size."""
        return len(self.word_to_id)
    
    def __contains__(self, word):
        """Check if word is in vocabulary."""
        return word in self.word_to_id

if __name__ == "__main__":
    # Example usage
    sample_sentences = [
        ["this", "is", "a", "sample", "sentence"],
        ["another", "example", "sentence", "with", "common", "words"],
        ["this", "sentence", "has", "some", "common", "words", "too"]
    ]
    
    vocab = Word2VecVocab(min_count=1)
    vocab_size = vocab.build(sample_sentences)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Word IDs: {vocab.word_to_id}")
    print(f"Negative samples for 'sentence': {vocab.get_neg_samples(vocab.word_to_id['sentence'], 3)}") 