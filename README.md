# Hacker News Upvote Predictor

This project aims to predict the number of upvotes a Hacker News post will receive based on post metadata (title, author, domain, etc.) and user attributes (karma, account age, etc.).

## Project Structure

```
project-root/
├── data/
│   ├── raw/           # Database extracts (items, users)
│   └── processed/     # Cleaned datasets, embeddings
├── notebooks/
│   ├── EDA_HackerNews.ipynb      # Exploratory Data Analysis
│   └── Training.ipynb # Model training experiments
├── src/
│   ├── training/      # Word2Vec and model training code
│   │   ├── text_preprocessing.py  # Text cleaning and tokenization
│   │   ├── vocabulary.py          # Vocabulary building
│   │   ├── dataset.py             # Dataset preparation
│   │   ├── word2vec_model.py      # Model architecture
│   │   ├── training.py            # Training functions
│   │   ├── embedding.py           # Embedding generation
│   │   └── word2vec_pipeline.py   # Main orchestrator
│   ├── utils/         # Helper functions and data processing
│   └── api/           # FastAPI service for predictions
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository**

2. **Set up a Python virtual environment**
   ```bash
   # Create a virtual environment in the project directory
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Make sure your virtual environment is activated
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit the .env file with your database credentials if needed
   # The default values should work for team members
   ```

5. **Extract data from the database**
   ```bash
   python src/utils/data_extraction.py
   ```
   This will:
   - Connect to the Hacker News database using credentials from your .env file
   - Extract 100k items (sorted by time, newest first)
   - Extract user data for authors of those items
   - Save the data to parquet files in `data/raw/`

6. **Run the EDA notebook**
   ```bash
   # If you haven't installed Jupyter yet, install it in your virtual environment
   pip install jupyter
   
   # Start Jupyter notebook
   jupyter notebook notebooks/EDA_HackerNews.ipynb
   ```

## Development Workflow

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

To return to work, activate the virtual environment again before running any scripts or notebooks:
```bash
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
   
## Project Workflow

1. **Data Collection**: Extract and save data from the Hacker News database using the data extraction script.

2. **Exploratory Data Analysis**: Analyze post-level and user-level features to understand patterns in the data.

3. **Word2Vec Training**: 
   - Pre-train Word2Vec models on Wikipedia corpus
   - Fine-tune on Hacker News titles
   
4. **Feature Engineering**:
   - Extract title embeddings
   - Create user-level features (karma, account age)
   - Add time-based features (hour, day of week)
   - Extract domain information

5. **Model Training**:
   - Combine all features
   - Train an MLP regression model
   - Evaluate performance
   
6. **Deployment**:
   - Package model and dependencies in Docker container
   - Deploy FastAPI service
   - Test predictions with sample posts

## Word2Vec Architecture

We've implemented a custom Word2Vec from scratch, broken down into modular components for better understanding and team collaboration. Each component handles a specific part of the Word2Vec pipeline:

### Available Models

Our implementation supports two Word2Vec architectures:

1. **Skip-gram with Negative Sampling**
   - Predicts context words from a center word
   - Uses negative sampling for efficient training
   - Better for infrequent words and larger datasets
   - Generally produces higher quality embeddings

2. **CBOW (Continuous Bag of Words) with Softmax**
   - Predicts a target word from surrounding context words
   - Uses full softmax for output probability distribution
   - Computationally more expensive but potentially more accurate
   - Better for smaller datasets with frequent terms

To specify which model to use, set the `--model_type` parameter when running the pipeline:
```bash
python src/training/word2vec_pipeline.py --model_type skipgram
# or
python src/training/word2vec_pipeline.py --model_type cbow_softmax
```

### Components

1. **Text Preprocessing (`text_preprocessing.py`)**
   - Functions for text cleaning, tokenization, and corpus preparation
   - Handles both Wikipedia and Hacker News text data
   - Main functions: `clean_text()`, `tokenize_text()`, `preprocess_wikipedia_corpus()`, `process_hacker_news_titles()`

2. **Vocabulary Management (`vocabulary.py`)**
   - Builds vocabulary from tokenized text
   - Implements frequency-based subsampling of common words
   - Constructs negative sampling tables using unigram distribution
   - Main class: `Word2VecVocab`

3. **Dataset Preparation (`dataset.py`)**
   - Creates training pairs based on model architecture
   - For Skip-gram: center word + context word pairs
   - For CBOW with softmax: context words + target word pairs
   - Main classes: `SkipGramDataset` and `CBOWSoftmaxDataset`

4. **Model Architecture (`word2vec_model.py`)**
   - Implements both available architectures using PyTorch
   - Skip-gram with negative sampling: `Word2VecModel` class
   - CBOW with softmax: `CBOWSoftmaxModel` class
   - Maintains appropriate embedding matrices for each approach

5. **Training Process (`training.py`)**
   - Functions for training and fine-tuning Word2Vec models
   - Implements learning rate scheduling for better convergence
   - Handles model saving and loading
   - Main functions: `train_word2vec_custom()`, `train_cbow_softmax_custom()`, `finetune_word2vec_custom()`, `finetune_cbow_softmax_custom()`

6. **Embedding Utilities (`embedding.py`)**
   - Functions for generating embeddings from trained models
   - Provides evaluation metrics and similarity calculations
   - Includes analogy finding capabilities
   - Main functions: `get_title_embedding()`, `evaluate_word2vec_model()`, `find_analogy()`

7. **Pipeline Orchestration (`word2vec_pipeline.py`)**
   - Ties together all components into a coherent workflow
   - Handles the end-to-end training and evaluation process
   - Main function: `main()`

### Data Flow

The Word2Vec pipeline follows this flow:

1. Text data is cleaned and tokenized (`text_preprocessing.py`)
2. Vocabulary is built from tokens (`vocabulary.py`)
3. Training dataset is created based on the chosen model architecture (`dataset.py`)
4. Model is initialized with random embeddings (`word2vec_model.py`)
5. Training occurs through batched gradient descent (`training.py`)
6. Trained model generates embeddings for new text (`embedding.py`)
7. The entire process is orchestrated by the pipeline (`word2vec_pipeline.py`)

### For Team Members

- If you're interested in **text processing**, focus on `text_preprocessing.py`
- If you want to understand **vocabulary building** and **negative sampling**, explore `vocabulary.py`
- For **context window sampling** and **dataset creation**, look at `dataset.py`
- To learn about the **neural network architecture**, examine `word2vec_model.py`
- For **training dynamics** and **optimization**, study `training.py`
- If you're interested in **using the embeddings**, investigate `embedding.py`
- To see how everything fits together, review `word2vec_pipeline.py`

Each file includes examples that can be run directly to demonstrate its functionality.

## Database Connection Details

The database connection details are stored in the `.env` file. The default values are:

- Host: 178.156.142.230
- Port: 5432
- Database: hd64m1ki
- Schema: hacker_news
- Tables: items, users

## Model Features

### Post-level Features
- Title embeddings (via Word2Vec)
- Title length and word count
- Post time (hour, day of week)
- Domain

### User-level Features
- Author karma
- Account age at post time

## API Implementation

The project includes a FastAPI service that serves predictions through an HTTP endpoint:

### API Features
- Automatically detects and loads either Skip-gram or CBOW softmax model
- Prioritizes Skip-gram if available, falls back to CBOW softmax
- Returns the model type used in the prediction response
- Handles optional user features (karma, account age)
- Returns both log-transformed and exponentiated predictions
- Includes detailed feature information used for the prediction

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
     "title": "Show HN: My new AI plugin",
     "author": "pg",
     "url": "http://paulgraham.com",
     "post_time": "2023-05-10T14:30:00Z",
     "user_karma": 15000,
     "user_age_days": 3650
}'
```

### Example Response
```json
{
  "predicted_upvotes": 42,
  "log_predicted_upvotes": 3.7612,
  "title": "Show HN: My new AI plugin",
  "author": "pg",
  "features_used": {
    "title_embedding_size": 100,
    "title_length": 22,
    "title_word_count": 5,
    "account_age_days": 3650,
    "log_karma": 9.6158,
    "year": 2023,
    "month": 5,
    "day_of_week": 2,
    "hour": 14
  },
  "embedding_model": "skipgram"
}
```

## Contributors

- @AlexVOiceover @Ardrito @dimitar-seraffimov @JasonWarrenUK @Liam40