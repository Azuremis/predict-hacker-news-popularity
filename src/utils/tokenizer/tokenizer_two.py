import json
import os
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

with open(os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens.json'
  ), "r", encoding="utf-8") as f:
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
    
  

with open(os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens_upgrade.json'
  ), "r+", encoding="utf-8") as f:
    f.seek(0)
    json.dump(token_dict, f, ensure_ascii=False, indent=2)
    f.truncate()

logging.info(f"Updated dictionary now has {len(token_dict)} tokens.")