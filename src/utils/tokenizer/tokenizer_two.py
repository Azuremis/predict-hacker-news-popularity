import json
import os
import pickle
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from tokenizer_one import getMorphemeList, getMorphemeSet, getTokens
import logging
import nltk
from nltk_setup import get_nltk_data_dir

# Set NLTK data path to our custom directory
nltk.data.path.insert(0, get_nltk_data_dir())

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("tokenizer_two.log"),
        logging.StreamHandler()
    ]
)

# Try to load from pickle first, fall back to JSON if pickle file doesn't exist
tokens_dir = os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens'
)
pickle_path = os.path.join(tokens_dir, 'tokens.pkl')
json_path = os.path.join(tokens_dir, 'tokens.json')

try:
    with open(pickle_path, "rb") as f:
        token_dict = pickle.load(f)
    logging.info("Loaded tokens from pickle file")
except (FileNotFoundError, pickle.PickleError):
    logging.info("Pickle file not found or invalid, falling back to JSON")
    with open(json_path, "r", encoding="utf-8") as f:
        token_dict = json.load(f)

max_id = max(map(int, token_dict.values()), default=0)
next_id = max_id + 1

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD")
)
cursor = conn.cursor()
cursor.execute("SELECT title FROM hacker_news.items WHERE type = 'story' AND title IS NOT NULL")

chunk_size = 1000
total_listed = 0
total_processed = 0

while True:
    rows = cursor.fetchall()
    if not rows:
        break

    logging.info("Tokenizing titles...")

    morpheme_list = []
    last_logged = 0

    for title, in rows:
        if not title:
            continue
        morphemes = getMorphemeList(title)
        morpheme_list.extend(morphemes)
        if len(morpheme_list) % 1000 == 0 and len(morpheme_list) > last_logged:
            logging.info(f"Holding {len(morpheme_list)} words in list")
            last_logged = len(morpheme_list)
    
    morpheme_set = getMorphemeSet(morpheme_list)

    if len(morpheme_set) > 0:
        logging.info(f"Holding {len(morpheme_set)} words in set")

    number_added = 0

    for morpheme in morpheme_set[:5]:
        token_dict[0] = '<PAD>'
        if morpheme not in token_dict:
            token_dict[morpheme] = int(next_id)
            number_added += 1
            next_id += 1
        
    
    logging.info(f"Added {number_added} tokens")
    
  

# Save updated tokens to both JSON and pickle formats
tokens_upgrade_json = os.path.join(tokens_dir, 'tokens_upgrade.json')
tokens_upgrade_pkl = os.path.join(tokens_dir, 'tokens_upgrade.pkl')

# Save as JSON
with open(tokens_upgrade_json, "w", encoding="utf-8") as f:
    json.dump(token_dict, f, ensure_ascii=False, indent=2)
logging.info(f"Saved updated tokens to JSON: {tokens_upgrade_json}")

# Save as pickle
with open(tokens_upgrade_pkl, "wb") as f:
    pickle.dump(token_dict, f)
logging.info(f"Saved updated tokens to pickle: {tokens_upgrade_pkl}")

logging.info(f"Updated dictionary now has {len(token_dict)} tokens.")