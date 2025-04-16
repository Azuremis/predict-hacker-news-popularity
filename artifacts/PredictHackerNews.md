Below is an **updated** end-to-end implementation guide, **removing reliance on Gensim** or any other pre-implemented Word2Vec library. Instead, you'll implement the **Word2Vec** algorithm **from scratch** (in either pure Python + NumPy or PyTorch). All other aspects (data fusion, MLP training, FastAPI deployment, Docker, etc.) remain the same, but now you'll build your word embeddings with your own custom code.

---

# 🚀 AI @ FAC Week 1 Project – Ultimate Implementation Reference 

> **Goal**: Build a **regression model** that predicts **Hacker News upvotes** from post metadata (title, author, domain, etc.). Implement a **Word2Vec** *from scratch*, fuse features, and deploy a **FastAPI** service in a **Docker** container, returning an estimated upvote count via an HTTP POST endpoint.

---

## 🗂 Table of Contents

1. [Project Overview](#project-overview)  
2. [Core Requirements](#core-requirements)  
3. [Architecture Summary](#architecture-summary)  
4. [Detailed Steps](#detailed-steps)  
   1. [Phase 1: Exploratory Data Analysis (EDA)](#phase-1-exploratory-data-analysis-eda)  
   2. [Phase 2: Pre-train Word2Vec (From Scratch)](#phase-2-pre-train-word2vec-from-scratch)  
   3. [Phase 3: Fine-tune Word2Vec for HN Titles](#phase-3-fine-tune-word2vec-for-hn-titles)  
   4. [Phase 4: Feature Fusion and Modeling](#phase-4-feature-fusion-and-modeling)  
   5. [Phase 5: Model Training Workflow](#phase-5-model-training-workflow)  
   6. [Phase 6: Deployment](#phase-6-deployment)  
5. [Development Environment Setup](#development-environment-setup)
6. [Collaboration & Workflow](#collaboration--workflow)  
7. [Codebase Organization](#codebase-organization)  
8. [Testing & Evaluation](#testing--evaluation)  
9. [FAQ & Tips](#faq--tips)  
10. [Final Checklist](#final-checklist)  

---

## 1. Project Overview

Your team's assignment is to **predict how many upvotes** a Hacker News post is likely to receive, based on:
- **Title**: Text input (topic/buzzwords)
- **Author**: Some authors get more traction
- **URL/Domain**: Certain domains (e.g., `paulgraham.com`) might be more popular
- **Submission Age**: Posts from 2006 vs. 2025 can have different distributions
- (Optionally) **Comments**: Watch out for potential data leakage via comment counts

Ultimately, you'll provide a **REST endpoint** that accepts post metadata in JSON and outputs a **predicted upvote count**.

### Enhanced Approach with User-Level Features

In our implementation, we're enriching the model by incorporating both:
- **Post-level features**: Title, domain, posting time, etc.
- **User-level features**: Author karma, account age at posting time

This captures not just what and when was posted, but also who posted it, giving the model stronger predictive power by accounting for author reputation and experience.

---

## 2. Core Requirements
 
1. **EDA**:  
   - Investigate data distribution, scale, missing values, and outliers.

2. **Pre-train Word2Vec**:  
   - Use Wikipedia text to capture general semantic understanding.

3. **Fine-tune Word2Vec**:  
   - Adapt these embeddings to Hacker News titles for domain relevance.

4. **Fuse Features**:  
   - Combine embeddings from text with numeric/categorical features (author, domain, age).

5. **Model**:  
   - Build a **multilayer perceptron (MLP)** for the final regression.

6. **Deploy**:  
   - Serve predictions via a **FastAPI** endpoint.
   - Containerize with **Docker** and host on a server (e.g., Computa).
   - Provide IP + port so we can send test requests.

---

## 3. Architecture Summary
A high-level flow of your system:
```
                ┌─────────────────────────┐
                │ Wikipedia (text corpus) │
                └───────────┬─────────────┘
                            │
                            ▼
                Custom Word2Vec Training
                 (from scratch, no gensim)
                            │
                            ▼
                    Word2Vec Embeddings
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
       ▼                                         ▼
  HN Title Data                          Fine-tune Word2Vec
  (tokenized)   ←------------------------→  (again, from scratch)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
                      Title Embeddings
                            │
   Author Embedding/One-hot │  Domain Embedding/One-hot
        Age (numeric)       │
                            ▼
                      ┌───────────┐
                      │   Fusion  │  (Concatenate all features)
                      └───────────┘
                            │
                            ▼
                       MLP Regression
                            │
                            ▼
                 Predicted Hacker News Upvotes
```

---

## 4. Detailed Steps

### Phase 1: Exploratory Data Analysis (EDA)

1. **Connect to the Database**  
   - Use **psycopg2** or **SQLAlchemy** to query:  
     ```sql
     SELECT * 
     FROM "hacker_news"."items"
     LIMIT 10000;
     ```
   - Start small (e.g., 10k rows) to avoid 28GB timeouts.
   
   **Implementation Details**:
   - We're using SQLAlchemy with environment variables for DB connection
   - Credentials stored in `.env` file (not committed to Git)
   - Extraction script in `src/utils/data_extraction.py`
   - Saving data as parquet files for faster loading

2. **Explore**  
   - Plot histograms of `score` distribution.  
   - Check popular domains, authors, average title length.  
   - Inspect potential correlations (domain → upvotes? author → upvotes?).
3.   **Identify Data Issues**  
   - Missing or null authors?  
   - Very old posts with suspicious zero or extremely high `score`?  
   - Overlapping data for training vs. testing?

   **Pro Tip**: 
   - Conduct time-based splits if you want to simulate "future data." 
   - Summarize EDA findings in a shared notebook or short markdown report.

   **User-Level Analysis**:
   - We've expanded the EDA to include author karma, account age, and their relationship with post scores
   - Examining whether high-karma authors tend to receive more upvotes on average
   - Investigating post performance as a function of author account age at post time

### Phase 2: Pre-train Word2Vec (From Scratch)

Our implementation uses PyTorch to build Word2Vec from scratch, with the following components:

#### Available Word2Vec Models

We support two Word2Vec architectures:

1. **Skip-gram with Negative Sampling**
   - Predicts context words from a center word
   - Uses negative sampling to efficiently approximate the full softmax
   - Handles rare words better than CBOW
   - Computationally efficient for large vocabularies

2. **CBOW (Continuous Bag of Words) with Softmax**
   - Predicts a target word from its surrounding context words
   - Uses full softmax over the vocabulary for output probabilities
   - Computationally more intensive but can be more accurate
   - May perform better on smaller datasets with frequent terms

The model type can be specified when running the Word2Vec pipeline:
```bash
python src/training/word2vec_pipeline.py --model_type skipgram
# or
python src/training/word2vec_pipeline.py --model_type cbow_softmax
```

#### Implementation Components

1. **Vocabulary Management** (`vocabulary.py`):
   - Built word-to-id and id-to-word dictionaries for efficient token lookup
   - Implemented negative sampling with frequency-based distribution
   - Added subsampling of frequent words to improve training efficiency

2. **Dataset Preparation** (`dataset.py`):
   - `SkipGramDataset`: For Skip-gram with negative sampling
     - Creates center-context word pairs from tokenized sentences
     - Generates negative samples for each positive example
   - `CBOWSoftmaxDataset`: For CBOW with softmax
     - Creates context-target word pairs from tokenized sentences
     - Handles variable-length contexts with padding

3. **Model Architecture** (`word2vec_model.py`):
   - `Word2VecModel`: Skip-gram with negative sampling
     - Separate embedding layers for center and context words
     - Optimized loss function with positive and negative sampling
   - `CBOWSoftmaxModel`: CBOW with softmax
     - Context word embeddings and output projection layer
     - Uses CrossEntropyLoss for the softmax computation

4. **Training Pipeline** (`training.py`):
   - Optimized with Adam optimizer and learning rate scheduling
   - Implemented batch processing for efficient training
   - Added checkpointing to save model state during training
   - Included comprehensive logging for monitoring training progress

5. **Text Processing** (`text_preprocessing.py`):
   - Specialized tokenization for Wikipedia corpus and HN titles
   - Cleaned and normalized text data for better embedding quality
   - Managed sentence segmentation and special character handling

**Our Implementation Checklist**:
- [x] **Vocabulary-building**: Created dictionaries for word-to-id and id-to-word mappings
- [x] **Negative sampling**: Implemented efficient training approach with unigram distribution
- [x] **Context window sampling**: Created appropriate word pairs for both model types
- [x] **Learning rate scheduling**: Added cosine annealing scheduler for better convergence

---

### Phase 3: Fine-tune Word2Vec for HN Titles

1. **Extract Titles**  
   ```sql
   SELECT title 
   FROM "hacker_news"."items" 
   WHERE title IS NOT NULL
   ```
2. **Preprocess & Tokenize** 
   - Convert to lowercase  
   - Remove punctuation  
   - Possibly remove stopwords if it helps
3. **Initialize from Pre-trained Weights**  
   - Load the pretrained embeddings from Wikipedia training
   - For Skip-gram: load center and context embeddings
   - For CBOW with softmax: load context embeddings and output weights
4. **Fine-tuning Process**
   - Using the same model architecture (Skip-gram or CBOW softmax)
   - Train for a few epochs on Hacker News titles
   - Lower learning rate to avoid catastrophic forgetting
   - Expand vocabulary as needed for HN-specific terms
5. **Handling New Vocabulary**
   - Identify new words in HN titles not present in Wikipedia
   - Expand embedding matrices to accommodate new vocabulary
   - Initialize new word vectors with random values
   - Update vocabulary mappings with new words
6. **Save Fine-tuned Model**
   - Save the model state dict for later use
   - Export embeddings as NumPy arrays for easier integration
   - Save updated vocabulary for consistent tokenization

---

### Phase 4: Feature Fusion and Modeling

1. **Title Embeddings**  
   - Each title → tokens → average (or sum) the embeddings from your `W_in`.  
   - Example:
     ```python
     def get_title_embedding(token_list, W_in, word_to_id):
         vectors = []
         for token in token_list:
             if token in word_to_id:
                 idx = word_to_id[token]
                 vectors.append(W_in[idx])
         if vectors:
             return np.mean(vectors, axis=0)
         else:
             return np.zeros(W_in.shape[1])  # embedding_dim
     ```

2. **Author Embedding**:  
   - Option 1: One-hot encode authors if you have memory for up to ~1M. Possibly too large. 
   - Option 2: Train a smaller embedding matrix for authors (like an ID → embedding). 
   - Option 3: Just keep it simpler if high cardinality is a concern. (Or only encode top X authors.)

3. **Domain**:  
   - Similar approach as author: either a small embedding or a numeric ID.
   - Consider cleaning domain names (e.g., `http://www.example.com` → `example.com`).

4. **Age**:  
   - Numeric, might need normalization (e.g., min-max scaling).

5. **User Features** (Our Enhanced Approach):
   - **Author Karma**: Log-transformed to handle skewness
   - **Account Age at Post Time**: The difference between post time and author account creation
   - These features help the model learn that established users with higher karma tend to get different upvote patterns

6. **Concatenate** all these into a single vector:
   ```python
   fused_vector = np.concatenate([
       title_embedding,   # e.g., shape (100,)
       author_embedding,  # e.g., shape (16,)
       domain_embedding,  # e.g., shape (8,)
       [age_normalized],  # shape (1,)
       [log_karma],       # shape (1,)
       [account_age]      # shape (1,)
   ])
   ```

---

### Phase 5: Model Training Workflow

1. **Prepare Training Data**:  
   - For each row in your dataset (post), create the fused feature vector + the target (`score`).
   - Split into **train / val / test** (e.g., 70/15/15 or 80/10/10). 
   - Possibly do a **temporal split**: older data = train, newer data = test.

2. **Build an MLP** (Multilayer Perceptron)  
   - A simple example in PyTorch:
     ```python
     import torch
     import torch.nn as nn

     class HNUpvoteModel(nn.Module):
         def __init__(self, input_dim, hidden_dim=128):
             super().__init__()
             self.fc1 = nn.Linear(input_dim, hidden_dim)
             self.relu = nn.ReLU()
             self.fc2 = nn.Linear(hidden_dim, 1)  # 1 output for regression

         def forward(self, x):
             x = self.fc1(x)
             x = self.relu(x)
             x = self.fc2(x)
             return x
     ```
3. **Train**  
   ```python
   import torch.optim as optim

   model = HNUpvoteModel(input_dim=fused_vector_length)
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(num_epochs):
       for batch_x, batch_y in dataloader:
           optimizer.zero_grad()
           preds = model(batch_x)
           loss = criterion(preds.squeeze(), batch_y)
           loss.backward()
           optimizer.step()
   ```
4. **Evaluate**  
   - Use **MSE**, **MAE**, or **RMSE** on the validation/test set. 
   - Plot predictions vs. actual scores. 
   - Watch out for outliers (very large upvote counts).

**Log-Transformation**:
- Our implementation applies log transformation (`log(score + 1)`) to the target variable
- This makes the distribution more normal and improves model performance
- Remember to convert predictions back to original scale: `exp(pred) - 1`

---

### Phase 6: Deployment

1. **FastAPI Setup**:
   ```python
   # src/api/main.py
   from fastapi import FastAPI
   from pydantic import BaseModel
   import torch

   app = FastAPI()

   # Define schema for incoming JSON
   class HNPost(BaseModel):
       title: str
       author: str
       url: str
       age: float

   # Load your model and word2vec in startup event or global scope
   model = torch.load("hn_upvote_model.pth")
   word2vec = # load your Word2Vec

   @app.post("/predict")
   def predict_upvotes(post: HNPost):
       # 1) Tokenize title, get embedding
       # 2) Get author/domain embedding
       # 3) Combine with age
       # 4) Pass through MLP
       # 5) Return predicted upvotes as JSON
       fused_input = create_fused_vector(post, word2vec)
       with torch.no_grad():
           prediction = model(fused_input).item()
       return {"predicted_upvotes": prediction}
   ```

2. **Dockerize**:
   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . /app

   # Expose port
   EXPOSE 8000

   CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Build & Run**:
   ```bash
   docker build -t hn-upvote-predictor .
   docker run -d -p 8000:8000 hn-upvote-predictor
   ```

4. **Test**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
        "title": "Show HN: My new AI plugin",
        "author": "pg",
        "url": "http://paulgraham.com",
        "age": 567890
   }'
   ```

**Enhanced API Implementation**:
- Our API accepts optional user-level features (karma, account age)
- If not provided, the API uses reasonable defaults
- API returns both the predicted upvotes and the log-transformed prediction

---

## 5. Development Environment Setup

### Virtual Environment

Setting up a consistent development environment helps ensure all team members can run the code:

1. **Create a Python virtual environment**:
   ```bash
   # Create a virtual environment in the project directory
   python -m venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Deactivate when done**:
   ```bash
   deactivate
   ```

### Environment Variables

For database credentials and sensitive information, we use environment variables:

1. **Create a `.env` file** (copy from `.env.example`):
   ```
   # Database credentials
   DB_USERNAME=your_username
   DB_PASSWORD=your_password
   DB_HOST=your_host
   DB_PORT=5432
   DB_NAME=your_dbname
   ```

2. **Load variables in code**:
   ```python
   from dotenv import load_dotenv
   import os
   
   # Load .env file
   load_dotenv()
   
   # Access variables
   username = os.getenv('DB_USERNAME')
   ```

3. **Add `.env` to `.gitignore** to prevent committing credentials.

---

## 6. Collaboration & Workflow

1. **Standups**: Start the day with 5-10 minute check-ins.  
2. **Pair Programming**: For tricky tasks like Word2Vec fine-tuning or Docker, pair up.  
3. **Version Control**:  
   - Create feature branches (`feature/EDA`, `feature/model`, etc.).  
   - Use Pull Requests to merge into `main`.
   - Add CI/CD workflow for testing

---

## 7. Codebase Organization

Our actual implemented directory structure:

```
project-root/
├── data/
│   ├── raw/              # Wikipedia, Database extracts (items, users)
│   └── processed/        # Cleaned datasets, embeddings, models, tokenized data
├── notebooks/
│   ├── EDA.ipynb         # Exploratory data analysis
│   └── TrainingExperiments.ipynb    # Model training experiments
├── src/
│   ├── training/
│   │   ├── word2vec_pipeline.py   # Word2Vec training and fine-tuning pipeline
│   │   ├── word2vec_model.py      # PyTorch implementation of Word2Vec
│   │   ├── vocabulary.py          # Vocabulary building and negative sampling
│   │   ├── dataset.py             # SkipGramDataset for training
│   │   ├── text_preprocessing.py  # Text cleaning and tokenization
│   │   ├── training.py            # Training loop and optimization
│   │   ├── embedding.py           # Embedding utilities
│   │   └── model.py               # PyTorch MLP model definition
│   ├── utils/
│   │   ├── data_prep.py           # Data transformation utilities
│   │   └── data_extraction.py     # Database extraction script
│   └── api/
│       ├── main.py                # FastAPI endpoint
│       └── model_loader.py        # loads model, embeddings
├── artifacts/            # Project documentation
│   └── PredictHackerNews.md       # This implementation guide
├── .env.example          # Template for environment variables
├── .gitignore            # Files to exclude from version control
├── Dockerfile            # Docker container definition
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and setup instructions
```

---

## 8. Testing & Evaluation

1. **Local Testing**:  
   - Unit tests on data transformations.  
   - Smoke tests on the FastAPI endpoint.

2. **Validation**:
   - Keep a **validation dataset** separate from training to tune hyperparameters.

3. **Performance Metrics**:
   - **RMSE** (Root Mean Squared Error): penalizes large errors.  
   - **MAE** (Mean Absolute Error): more robust to outliers.  

4. **Sanity Checks**:
   - For random or nonsense titles, does your model return a realistic upvote range?  
   - For known high-buzzword titles ("AI beats crypto again!"), do you see a higher upvote count?

---

## 9. FAQ & Tips

1. **How do we handle extremely high upvote outliers?**  
   - Consider a **log-transform** on the scores if a small fraction of items have super high upvotes. 
   - Then you can exponentiate your predictions.

2. **Should we use Comments as a Feature?**  
   - Potentially. But watch out for **future leakage** and missing timestamp alignment.  
   - If you do it, consider time-based logic to only include comments known at the submission's "current" time.

3. **Data Too Large?**  
   - Sample from the DB or use chunking. 
   - If the DB is too slow, store partial data in local parquet/csv files.

4. **Can we add more advanced models?**  
   - Yes, but the requirement is to build from scratch. No off-the-shelf large transformers. 
   - If time remains, explore advanced approaches like RNN or even a small transformer.

5. **Is Docker mandatory?**  
   - Yes—this ensures a consistent environment. And you'll run on Computa with Docker.

6. **How to handle database credentials?**
   - Use a `.env` file to store credentials
   - Load them with `python-dotenv` at runtime
   - Never commit credentials to version control

7. **How to develop as a team?**
   - Use virtual environments for consistency
   - Share the `.env.example` template without actual credentials
   - Run the same data extraction to ensure everyone has the same dataset

---

## 10. Final Checklist

- [x] **Environment Setup**: Virtual environment, dependencies, and .env file.
- [x] **EDA Completed**: SQL queries, distribution analysis, outlier detection, summary stats.  
- [ ] **Word2Vec Pre-trained**: Custom Word2Vec implementation from scratch on Wikipedia corpus.  
- [ ] **Word2Vec Fine-tuned**: Fine-tuned on Hacker News titles for domain adaptation.  
- [ ] **Feature Fusion**: Title embeddings, author/domain encoding, numeric features, user features.  
- [ ] **MLP Model Trained**: Evaluate MSE/RMSE/MAE.  
- [ ] **API Deployed**: `/predict` endpoint returning upvote predictions.  
- [ ] **Docker**: Container builds successfully, runs your FastAPI app.  
- [ ] **Test**: Curl or Postman request returns a sensible numeric prediction.  
- [ ] **Documentation**: README plus any docstrings explaining how to run end-to-end.

---

**With this blueprint, we have a comprehensive, step-by-step guide** that ensures your entire team can **track progress**, understand how to **transform data** and **build models**, and know exactly **how to collaborate** on code and deployment. You'll produce a robust upvote prediction system while practicing real-world data science and MLOps skills.

**Best of luck building your Hacker News upvote predictor!** 🌟