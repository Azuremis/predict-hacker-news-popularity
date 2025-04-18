from collections import Counter
from nltk.tokenize import word_tokenize
import re
import os
import json
import pickle
from nltk.corpus import words
import logging
import nltk
from nltk_setup import get_nltk_data_dir

# Set NLTK data path to our custom directory
nltk.data.path.insert(0, get_nltk_data_dir())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("tokenizer_one.log"),
        logging.StreamHandler()
    ]
)

inputs = {
  'wiki': os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'data', 'raw', 'text8'
  ),
  'sentence': "I am writing a file in Python. It's supposed to break text into tokens with unique indices that, hopefully, equate to morphemes. Now, let's repeat some of those words so that we can check that we're discarding duplicates. I am writing a file in Python. It's supposed to break text into tokens with unique indices."
}

def getMorphemeList(text):
  text = text.lower()
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  morpheme_list = word_tokenize(text)
  
  # Count occurrences of each word
  word_counts = Counter(morpheme_list)
  
  # Filter out words that only appear once
  morpheme_list = [word for word in morpheme_list if word_counts[word] > 1]
  
  return morpheme_list

def getMorphemeSet(list):
  # Get the set of English words
  english_words = set(words.words())
  
  # Create set from input list
  morpheme_set = set(list)
  
  # Filter out non-English words
  morpheme_set = {word for word in morpheme_set if word.lower() in english_words}
  
  return morpheme_set

def getTokens(morphemes):
  token_dict = {}
  for i, morpheme in enumerate(sorted(morphemes)):
    token_dict[morpheme] = i
  return token_dict

def clean(input):
  print('=== Input ===')
  if os.path.isfile(input):
    with open(input, 'r', encoding='utf-8') as f:
      input = f.read()
  # print(input)

  print('--------------------------------')

  print('=== Morphemes (All) ===:')
  morphemeList = sorted(getMorphemeList(input))
  # print(morphemeList)

  print('--------------------------------')

  print('=== Morphemes (Unique) ===:')
  morphemeSet = sorted(list(getMorphemeSet(morphemeList)))
  # print(morphemeSet)

  print('--------------------------------')

  print('=== Tokens ===')
  tokens = getTokens(morphemeSet)
  print(tokens)

  print('--------------------------------')

  # Save tokens to file - JSON format
  json_output_path = os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens.json'
  )
  with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)
  logging.info(f'Saved tokens to JSON: {json_output_path}')
  
  # Save tokens to file - Pickle format
  pickle_output_path = os.path.join(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(__file__)
        )
      )
    ), 'tokens', 'tokens.pkl'
  )
  with open(pickle_output_path, "wb") as f:
    pickle.dump(tokens, f)
  logging.info(f'Saved tokens to pickle: {pickle_output_path}')

  return tokens

if __name__ == "__main__":
  clean(inputs['wiki'])
