import json
import os
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from tokenizer_one import getMorphemeList, getMorphemeSet, getTokens

load_dotenv()  # Loads .env into environment variables

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

cursor.execute("SELECT title FROM hacker_news.items_by_month_2024_10 WHERE type = 'story'")

chunk_size = 1000
while True:
    rows = cursor.fetchmany(chunk_size)
    if not rows:
        break

    for (title,) in rows:
        if not title:
            continue
        morphemes = getMorphemeList(title)
        morpheme_set = getMorphemeSet(morphemes)
        tokens = getTokens(morpheme_set)
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = str(next_id)
                next_id += 1

with open(os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens.json'
  ), "w", encoding="utf-8") as f:
    json.dump(token_dict, f, ensure_ascii=False, indent=2)

print(f"Updated dictionary now has {len(token_dict)} tokens.")