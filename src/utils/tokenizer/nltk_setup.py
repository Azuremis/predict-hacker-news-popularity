import nltk
import os

# Create a directory for NLTK data within utils
nltk_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data'))
os.makedirs(nltk_data_dir, exist_ok=True)

# Download NLTK data to the custom directory
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('words', download_dir=nltk_data_dir)

# Function to get the NLTK data directory path
def get_nltk_data_dir():
    return nltk_data_dir