import json
import os
import psycopg2
from dotenv import load_dotenv
from tokenizer import get_tokens

load_dotenv()  # Loads .env into environment variables

# Load existing tokens
with open("wiki.json", "r", encoding="utf-8") as f:
    token_dict = json.load(f)

# Start new IDs from the max existing ID + 1
max_id = max(map(int, token_dict.values()), default=0)
next_id = max_id + 1

# Connect to the DB
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD")
)
cursor = conn.cursor()

# Fetch story titles
cursor.execute("SELECT title FROM items WHERE type = 'story'")
rows = cursor.fetchall()

# Tokenize and add to dictionary
for (title,) in rows:
    if not title:
        continue
    tokens = get_tokens(title)
    for token in tokens:
        if token not in token_dict:
            token_dict[token] = str(next_id)
            next_id += 1

# Save updated dictionary
with open("wiki.json", "w", encoding="utf-8") as f:
    json.dump(token_dict, f, ensure_ascii=False, indent=2)

print(f"Updated dictionary now has {len(token_dict)} tokens.")