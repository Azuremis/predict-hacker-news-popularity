import json
import os
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from tokenizer_one import getMorphemeList, getMorphemeSet, getTokens
import logging

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
cursor.execute("SELECT title FROM hacker_news.items_by_month_2024_10 WHERE type = 'story' AND title IS NOT NULL")

chunk_size = 1000
total_processed = 0

while True:
    rows = cursor.fetchmany(chunk_size)
    if not rows:
        break

    logging.info("Tokenizing titles...")
    for title, in rows:
        if not title:
            continue
        morphemes = getMorphemeList(title)
        morpheme_set = getMorphemeSet(morphemes)
        tokens = getTokens(morpheme_set)
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = str(next_id)
                next_id += 1
        total_processed += 1
        if total_processed % 1000 == 0:
            logging.info(f"Processed {total_processed} titles so far")
            logging.info(f"Holding {len(token_dict)} tokens so far")

with open(os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens.json'
  ), "r+", encoding="utf-8") as f:
    f.seek(0)
    json.dump(token_dict, f, ensure_ascii=False, indent=2)
    f.truncate()

logging.info(f"Updated dictionary now has {len(token_dict)} tokens.")