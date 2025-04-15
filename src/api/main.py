import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Optional
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.word2vec_pipeline import get_title_embedding
from training.model import UpvotePredictor

# Initialize FastAPI app
app = FastAPI(title="Hacker News Upvote Predictor", 
              description="Predicts the number of upvotes a Hacker News post will receive",
              version="1.0.0")

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         "data/processed/model")
WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "data/processed/word2vec_hn_finetuned.model")

# Input model
class HackerNewsPost(BaseModel):
    title: str
    author: str
    url: Optional[str] = None
    post_time: Optional[datetime] = None
    user_karma: Optional[int] = None
    user_age_days: Optional[int] = None
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        return v
    
    @validator('author')
    def author_must_not_be_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Author cannot be empty')
        return v

# Output model
class PredictionResponse(BaseModel):
    predicted_upvotes: int
    log_predicted_upvotes: float
    title: str
    author: str
    features_used: dict

# Global variables for loaded models
word2vec_model = None
predictor_model = None
scaler = None
input_dim = None

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global word2vec_model, predictor_model, scaler, input_dim
    
    try:
        # Import Word2Vec here to avoid loading it in global scope
        from gensim.models import Word2Vec
        
        # Load Word2Vec model
        if os.path.exists(WORD2VEC_PATH):
            word2vec_model = Word2Vec.load(WORD2VEC_PATH)
            print(f"Loaded Word2Vec model with {len(word2vec_model.wv.key_to_index)} vocabulary items")
        else:
            print(f"WARNING: Word2Vec model not found at {WORD2VEC_PATH}. Using dummy embeddings.")
            # Create a dummy model for testing
            word2vec_model = type('obj', (object,), {
                'vector_size': 100,
                'wv': {'__getitem__': lambda self, key: np.zeros(100)}
            })
            word2vec_model.wv.__getitem__ = lambda key: np.zeros(100)
        
        # Load PyTorch model
        model_path = os.path.join(MODEL_DIR, "upvote_model.pth")
        if os.path.exists(model_path):
            # Get the input dimension from the first layer of the model
            # This should be stored somewhere, but for now we'll infer it from the model
            input_dim = 108  # 100 for embeddings + 8 for other features
            predictor_model = UpvotePredictor(input_dim=input_dim)
            predictor_model.load_state_dict(torch.load(model_path))
            predictor_model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        else:
            print(f"WARNING: PyTorch model not found at {model_path}. Using dummy model.")
            # Create a dummy model for testing
            input_dim = 108
            predictor_model = UpvotePredictor(input_dim=input_dim)
        
        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        else:
            print(f"WARNING: Scaler not found at {scaler_path}. Using dummy scaler.")
            # Create a dummy scaler for testing
            scaler = type('obj', (object,), {
                'transform': lambda self, X: X
            })
        
        print("All models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Don't raise here, as we want the API to start even if models fail to load
        # We'll check for None models in the prediction endpoint

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hacker News Upvote Predictor API",
        "docs": "/docs",
        "models_loaded": {
            "word2vec": word2vec_model is not None,
            "predictor": predictor_model is not None,
            "scaler": scaler is not None
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_upvotes(post: HackerNewsPost):
    """Predict upvotes for a Hacker News post."""
    # Check if models are loaded
    if word2vec_model is None or predictor_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")
    
    try:
        # Extract title embedding
        title_embedding = get_title_embedding(post.title, word2vec_model)
        
        # Process other features
        title_length = len(post.title)
        title_word_count = len(post.title.split())
        
        # Use current time if post_time not provided
        if post.post_time is None:
            post_time = datetime.now()
        else:
            post_time = post.post_time
            
        # Extract time features
        year = post_time.year
        month = post_time.month
        day_of_week = post_time.weekday()
        hour = post_time.hour
        
        # Handle user features
        log_karma = np.log1p(post.user_karma if post.user_karma is not None else 0)
        account_age_days = post.user_age_days if post.user_age_days is not None else 0
        
        # Combine features
        numeric_features = np.array([
            title_length, title_word_count, 
            account_age_days, log_karma,
            year, month, day_of_week, hour
        ]).reshape(1, -1)
        
        # Create feature vector
        X = np.hstack((title_embedding.reshape(1, -1), numeric_features))
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            log_pred = predictor_model(X_tensor).item()
        
        # Convert from log scale back to original scale
        pred = np.expm1(log_pred)
        
        # Round to integer
        pred_int = max(1, int(round(pred)))
        
        # Prepare feature dictionary for response
        features_used = {
            "title_embedding_size": title_embedding.shape[0],
            "title_length": title_length,
            "title_word_count": title_word_count,
            "account_age_days": account_age_days,
            "log_karma": log_karma,
            "year": year,
            "month": month,
            "day_of_week": day_of_week,
            "hour": hour
        }
        
        # Return prediction
        return PredictionResponse(
            predicted_upvotes=pred_int,
            log_predicted_upvotes=float(log_pred),
            title=post.title,
            author=post.author,
            features_used=features_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "word2vec": word2vec_model is not None,
            "predictor": predictor_model is not None,
            "scaler": scaler is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 