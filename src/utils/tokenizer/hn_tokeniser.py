#
# Hacker News Tokenizer using the functions from tokenizer_one.py and tokenizer_two.py
#
import collections
import pickle
import json
import psycopg2
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import random
import numpy as np
import logging
from tokenizer_one import getMorphemeList, getMorphemeSet, getTokens
from nltk_setup import get_nltk_data_dir, download_nltk_resources
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hn_tokeniser.log"),
        logging.StreamHandler()
    ]
)

# Download required NLTK resources
download_nltk_resources()

# Set NLTK data path to our custom directory
nltk.data.path.insert(0, get_nltk_data_dir())

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define root directory for saving tokens
def get_tokens_dir():
    return os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)
                )
            )
        ), 'tokens'
    )

#
# Connect to Postgres and fetch Hacker News data
#
logging.info("Connecting to database and fetching Hacker News data...")
conn = psycopg2.connect("postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
cur = conn.cursor()
cur.execute("""SELECT title, score FROM "hacker_news"."items" 
               WHERE title IS NOT NULL AND score IS NOT NULL;""")
data = cur.fetchall()
titles = [row[0] for row in data]
scores = [row[1] for row in data]
conn.close()

logging.info(f"Loaded {len(titles)} titles from Hacker News")
logging.info(f"Sample titles: {titles[:3]}")
logging.info(f"Sample scores: {scores[:3]}")

#
# Process all titles using tokenizer_one functions
#
logging.info("Processing titles using tokenizer functions...")

all_morphemes = []
title_morphemes = []
# Create a progress bar for processing titles
titles_with_progress = tqdm(titles, desc="Tokenizing titles", unit="title")

logging.info("Starting tokenization with progress tracking...")

for title in titles_with_progress:
    # Get morphemes for each title using the getMorphemeList function
    # This function already handles lowercasing and removing punctuation
    morphemes = getMorphemeList(title)
    
    # Add stopwords filtering since the original had it
    stop_words = set(stopwords.words('english'))
    morphemes = [word for word in morphemes if word not in stop_words]
    
    all_morphemes.extend(morphemes)
    title_morphemes.append(morphemes)

# Get English morphemes only using getMorphemeSet
unique_morphemes = getMorphemeSet(all_morphemes)

logging.info(f"Total morphemes: {len(all_morphemes)}")
logging.info(f"Unique English morphemes: {len(unique_morphemes)}")
logging.info(f"Sample morphemes: {list(unique_morphemes)[:10]}")

# Create token mappings using getTokens
morpheme_to_id = getTokens(unique_morphemes)

# Add PAD token
if '<PAD>' not in morpheme_to_id:
    max_id = max(morpheme_to_id.values()) if morpheme_to_id else -1
    morpheme_to_id['<PAD>'] = max_id + 1

# Create id_to_morpheme mapping
id_to_morpheme = {id: word for word, id in morpheme_to_id.items()}

# Convert title morphemes to IDs
title_token_ids = []
for morphemes in title_morphemes:
    # Convert each title's morphemes to IDs, handle unknown words
    title_ids = [morpheme_to_id.get(word, morpheme_to_id['<PAD>']) for word in morphemes]
    title_token_ids.append(title_ids)

#
# Save all data in both JSON and pickle formats
#
tokens_dir = get_tokens_dir()
os.makedirs(tokens_dir, exist_ok=True)

# Define paths for saving files
file_paths = {
    'corpus': os.path.join(tokens_dir, 'corpus'),
    'title_tokens': os.path.join(tokens_dir, 'title_tokens'),
    'scores': os.path.join(tokens_dir, 'scores'),
    'morpheme_to_id': os.path.join(tokens_dir, 'morpheme_to_id'),
    'id_to_morpheme': os.path.join(tokens_dir, 'id_to_morpheme'),
    'title_token_ids': os.path.join(tokens_dir, 'title_token_ids')
}

# Save each file in both pickle and JSON formats
for name, path in file_paths.items():
    data_to_save = locals()[name]
    
    # Save as pickle
    with open(f"{path}.pkl", 'wb') as f:
        pickle.dump(data_to_save, f)
    
    # Save as JSON for data that can be serialized
    try:
        with open(f"{path}.json", 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {name} to both pickle and JSON")
    except TypeError:
        logging.info(f"Saved {name} to pickle only (couldn't serialize to JSON)")

logging.info("Tokenization complete. Files saved:")
for name in file_paths.keys():
    logging.info(f"- {name}.pkl and {name}.json")
