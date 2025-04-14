Below is a **vastly expanded, more detailed** version of the end-to-end implementation guide. It’s designed to **streamline collaboration** among your team, **accelerate** your model development, and **make deployment a breeze**. Each section includes additional context, best practices, and pro tips so that any team member—no matter their background—can jump in and move forward confidently.

---

# 🚀 AI @ FAC Week 1 Project – Ultimate Implementation Reference

> **Goal**: Build a **regression model** that predicts **Hacker News upvotes** from post metadata (title, author, domain, etc.). Implement a **Word2Vec** pipeline, fuse features, and deploy a **FastAPI** service in a **Docker** container, returning an estimated upvote count via an HTTP POST endpoint.

---

## 🗂 Table of Contents

1. [Project Overview](#project-overview)  
2. [Core Requirements](#core-requirements)  
3. [Architecture Summary](#architecture-summary)  
4. [Detailed Steps](#detailed-steps)  
   1. [Phase 1: Exploratory Data Analysis (EDA)](#phase-1-exploratory-data-analysis-eda)  
   2. [Phase 2: Pre-train Word2Vec](#phase-2-pre-train-word2vec)  
   3. [Phase 3: Fine-tune Word2Vec for HN Titles](#phase-3-fine-tune-word2vec-for-hn-titles)  
   4. [Phase 4: Feature Fusion and Modeling](#phase-4-feature-fusion-and-modeling)  
   5. [Phase 5: Model Training Workflow](#phase-5-model-training-workflow)  
   6. [Phase 6: Deployment](#phase-6-deployment)  
5. [Collaboration & Workflow](#collaboration--workflow)  
6. [Codebase Organization](#codebase-organization)  
7. [Testing & Evaluation](#testing--evaluation)  
8. [FAQ & Tips](#faq--tips)  
9. [Final Checklist](#final-checklist)  

---

## 1. Project Overview

Your team’s assignment is to **predict how many upvotes** a Hacker News post is likely to receive based on:
- **Title**: Text input, can reveal topic, buzzwords, popularity
- **Author**: Certain authors might have historically high or low traction
- **URL/Domain**: Well-known domains (e.g., `paulgraham.com`) might correlate with more upvotes
- **Submission Age**: Posts from 2006 vs. 2025 might have different distributions
- (Optionally) **Comments**: Beware of **future leakage**; if you use comment counts, factor in the timestamp

Ultimately, you’ll provide a **REST endpoint** that accepts post metadata in JSON and outputs a **predicted upvote count**.

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
                    Word2Vec Training
                            │
                            ▼
                    Word2Vec Model (.bin)
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
       ▼                                         ▼
  HN Title Data                          Fine-tune Word2Vec
  (tokenized)                                   
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

2. **Explore**  
   - Plot histograms of `score` distribution.  
   - Check popular domains, authors, average title length.  
   - Inspect potential correlations (domain → upvotes? author → upvotes?).

3. **Identify Data Issues**  
   - Missing or null authors?  
   - Very old posts with suspicious zero or extremely high `score`?  
   - Overlapping data for training vs. testing?

**Pro Tip**: 
- Conduct time-based splits if you want to simulate “future data.” 
- Summarize EDA findings in a shared notebook or short markdown report.

---

### Phase 2: Pre-train Word2Vec

#### Purpose
Capture broad language semantics from Wikipedia so your model can interpret words more effectively.

#### Steps
1. **Obtain Wikipedia Corpus**  
   - A subset or full dump. 
   - Ensure you have a pipeline for cleaning HTML markup and removing non-text.

2. **Tokenize the Articles**  
   - Use `spacy`, `NLTK`, or a custom regex-based tokenizer.
   - Convert text into sequences of tokens (lowercase, remove punctuation, etc.).

3. **Train Word2Vec**  
   - Example with `gensim`:
     ```python
     from gensim.models import Word2Vec
     
     # Suppose 'all_sentences' is an iterable of token lists
     w2v_model = Word2Vec(
         sentences=all_sentences,
         vector_size=100,   # Try 100 to start; can go bigger
         window=5,         
         min_count=5,      # Ignores rare words
         workers=4,        # Adjust to CPU cores
         sg=1              # Skip-gram model
     )
     
     w2v_model.save("wiki_word2vec.model")
     ```
   - **Validate**: Check if synonyms or related words cluster well. 
   - **Pro Tip**: Keep an eye on training time and memory usage. Smaller subsets can still yield decent embeddings.

---

### Phase 3: Fine-tune Word2Vec for HN Titles

1. **Extract Titles**  
   ```python
   SELECT title 
   FROM "hacker_news"."items" 
   WHERE title IS NOT NULL
   ```
2. **Tokenize** similarly to Wikipedia corpus for consistency.
3. **Load Pre-trained Model**  
   ```python
   from gensim.models import Word2Vec

   w2v_model = Word2Vec.load("wiki_word2vec.model")
   ```
4. **Fine-tune**  
   ```python
   w2v_model.build_vocab(hn_titles, update=True)
   w2v_model.train(
       hn_titles,
       total_examples=w2v_model.corpus_count,
       epochs=5
   )
   w2v_model.save("hn_finetuned_word2vec.model")
   ```
   - Now your embeddings better reflect Hacker News style/terminology.

---

### Phase 4: Feature Fusion and Modeling

**Main Idea**: Convert each piece of data into numeric vectors, then merge.

1. **Title Embeddings**:  
   - For a given title, you might average the embeddings of each token. 
   - Alternatively, sum them or use a more advanced approach (like an LSTM).  
   - Let’s keep it simple: **average** your token embeddings.
   
   ```python
   def get_title_embedding(title_tokens, model):
       embeddings = []
       for token in title_tokens:
           if token in model.wv:
               embeddings.append(model.wv[token])
       if embeddings:
           return np.mean(embeddings, axis=0)
       else:
           # Return a zero vector if no known tokens
           return np.zeros(model.vector_size)
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

5. **Concatenate** all these into a single vector:
   ```python
   fused_vector = np.concatenate([
       title_embedding,   # e.g., shape (100,)
       author_embedding,  # e.g., shape (16,)
       domain_embedding,  # e.g., shape (8,)
       [age_normalized]   # shape (1,)
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

---

## 5. Collaboration & Workflow

1. **Standups**: Start the day with 5-10 minute check-ins.  
2. **Pair Programming**: For tricky tasks like Word2Vec fine-tuning or Docker, pair up.  
3. **Version Control**:  
   - Create feature branches (`feature/EDA`, `feature/model`, etc.).  
   - Use Pull Requests to merge into `main`.

---

## 6. Codebase Organization

A suggested directory structure:

```
project-root/
├── data/
│   ├── raw/           # Wikipedia, HackerNews dumps
│   └── processed/     # Cleaned subsets, tokenized data
├── notebooks/
│   ├── EDA.ipynb
│   └── TrainingExperiments.ipynb
├── src/
│   ├── training/
│   │   ├── word2vec_pipeline.py   # Pretrain + fine-tune methods
│   │   └── model.py               # PyTorch MLP class
│   ├── utils/
│   │   ├── data_prep.py           # tokenization, transformations
│   │   └── sql_queries.py
│   └── api/
│       ├── main.py                # FastAPI endpoints
│       └── model_loader.py        # loads model, embeddings
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 7. Testing & Evaluation

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
   - For known high-buzzword titles (“AI beats crypto again!”), do you see a higher upvote count?

---

## 8. FAQ & Tips

1. **How do we handle extremely high upvote outliers?**  
   - Consider a **log-transform** on the scores if a small fraction of items have super high upvotes. 
   - Then you can exponentiate your predictions.

2. **Should we use Comments as a Feature?**  
   - Potentially. But watch out for **future leakage** and missing timestamp alignment.  
   - If you do it, consider time-based logic to only include comments known at the submission’s “current” time.

3. **Data Too Large?**  
   - Sample from the DB or use chunking. 
   - If the DB is too slow, store partial data in local parquet/csv files.

4. **Can we add more advanced models?**  
   - Yes, but the requirement is to build from scratch. No off-the-shelf large transformers. 
   - If time remains, explore advanced approaches like RNN or even a small transformer.

5. **Is Docker mandatory?**  
   - Yes—this ensures a consistent environment. And you’ll run on Computa with Docker.

---

## 9. Final Checklist

- [ ] **EDA Completed**: SQL queries, distribution analysis, outlier detection, summary stats.  
- [ ] **Word2Vec Pre-trained**: On Wikipedia or a large enough textual corpus.  
- [ ] **Word2Vec Fine-tuned**: On Hacker News titles for domain adaptation.  
- [ ] **Feature Fusion**: Title embeddings, author/domain encoding, numeric features.  
- [ ] **MLP Model Trained**: Evaluate MSE/RMSE/MAE.  
- [ ] **API Deployed**: `/predict` endpoint returning upvote predictions.  
- [ ] **Docker**: Container builds successfully, runs your FastAPI app.  
- [ ] **Test**: Curl or Postman request returns a sensible numeric prediction.  
- [ ] **Documentation**: README plus any docstrings explaining how to run end-to-end.

---

**With this blueprint, you have a comprehensive, step-by-step guide** that ensures your entire team can **track progress**, understand how to **transform data** and **build models**, and know exactly **how to collaborate** on code and deployment. You’ll produce a robust upvote prediction system while practicing real-world data science and MLOps skills.

**Best of luck building your Hacker News upvote predictor!** 🌟