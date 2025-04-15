import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import time

# Import project modules
import sys
sys.path.append('..')
from training.word2vec_pipeline import get_title_embedding, tokenize_text

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class HNDataset(Dataset):
    """Dataset for Hacker News upvote prediction."""
    
    def __init__(self, features, targets):
        """
        Args:
            features: Tensor of input features
            targets: Tensor of target values (scores)
        """
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class UpvotePredictor(nn.Module):
    """MLP model for predicting Hacker News upvotes."""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        """
        Args:
            input_dim: Dimensionality of input features
            hidden_dim: Hidden layer size
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)

def extract_features(df_items, df_users, word2vec_model, log_transform=True):
    """Extract and combine features for model training.
    
    Args:
        df_items: DataFrame of Hacker News items
        df_users: DataFrame of Hacker News users
        word2vec_model: Trained Word2Vec model
        log_transform: Whether to log-transform the target variable
        
    Returns:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
    """
    logger.info("Extracting features from items and users data")
    
    # Merge items and users data
    df = df_items.merge(df_users, left_on='author', right_on='id', how='left', suffixes=('', '_user'))
    
    # Get title embeddings
    logger.info("Extracting title embeddings")
    title_embeddings = []
    for title in df['title']:
        embedding = get_title_embedding(title, word2vec_model)
        title_embeddings.append(embedding)
    
    # Convert to numpy array
    title_embeddings = np.array(title_embeddings)
    
    # Add title length features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    
    # Extract temporal features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['hour'] = df['time'].dt.hour
    
    # Calculate account age at post time
    df['account_age_days'] = (df['time'] - df['created']).dt.days
    
    # Clean up NaN values in user features
    for col in ['karma', 'account_age_days']:
        df[col] = df[col].fillna(0)
    
    # Log transform karma (add 1 to avoid log(0))
    df['log_karma'] = np.log1p(df['karma'])
    
    # Combine features
    numeric_features = [
        'title_length', 'title_word_count', 
        'account_age_days', 'log_karma',
        'year', 'month', 'day_of_week', 'hour'
    ]
    
    # Get numeric feature matrix
    X_numeric = df[numeric_features].values
    
    # Combine with title embeddings
    X = np.hstack((title_embeddings, X_numeric))
    
    # Create feature names (for interpretability)
    embedding_names = [f'title_emb_{i}' for i in range(title_embeddings.shape[1])]
    feature_names = embedding_names + numeric_features
    
    # Target variable
    if log_transform:
        y = np.log1p(df['score'].values)
    else:
        y = df['score'].values
    
    logger.info(f"Extracted {X.shape[1]} features for {X.shape[0]} samples")
    
    return X, y, feature_names

def train_model(X_train, y_train, X_val, y_val, input_dim, hidden_dim=128, 
                lr=0.001, batch_size=64, epochs=50, patience=5):
    """Train the MLP model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        input_dim: Input feature dimension
        hidden_dim: Hidden layer size
        lr: Learning rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        model: Trained PyTorch model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    logger.info(f"Training model with {input_dim} input features, {hidden_dim} hidden units")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Create datasets and dataloaders
    train_dataset = HNDataset(X_train_tensor, y_train_tensor)
    val_dataset = HNDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = UpvotePredictor(input_dim=input_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test, log_transformed=True):
    """Evaluate the trained model.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test targets
        log_transformed: Whether the target was log-transformed
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
    
    # If log-transformed, convert back to original scale
    if log_transformed:
        y_pred_original = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
    else:
        y_pred_original = y_pred
        y_test_original = y_test
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate metrics on original scale
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_original': mse_original,
        'rmse_original': rmse_original,
        'mae_original': mae_original
    }
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"Original scale:")
    logger.info(f"  MSE: {mse_original:.4f}")
    logger.info(f"  RMSE: {rmse_original:.4f}")
    logger.info(f"  MAE: {mae_original:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Log(Score)')
    plt.ylabel('Predicted Log(Score)')
    plt.title('Predicted vs Actual Log-Transformed Scores')
    plt.savefig('../data/processed/prediction_scatter.png')
    plt.close()
    
    return metrics

def save_model_artifacts(model, scaler, metrics, output_dir):
    """Save model artifacts for later use.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(output_dir, 'upvote_model.pth'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    logger.info(f"Model artifacts saved to {output_dir}")

def main():
    """Main function to run the model training pipeline."""
    # Paths
    items_path = '../data/raw/items_100k.parquet'
    users_path = '../data/raw/users_100k.parquet'
    word2vec_path = '../data/processed/word2vec_hn_finetuned.model'
    output_dir = '../data/processed/model'
    
    # Load data
    logger.info(f"Loading items data from {items_path}")
    df_items = pd.read_parquet(items_path)
    
    logger.info(f"Loading users data from {users_path}")
    df_users = pd.read_parquet(users_path)
    
    # Load Word2Vec model
    logger.info(f"Loading Word2Vec model from {word2vec_path}")
    word2vec_model = Word2Vec.load(word2vec_path)
    
    # Extract features
    X, y, feature_names = extract_features(df_items, df_users, word2vec_model, log_transform=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    input_dim = X_train_scaled.shape[1]
    model, train_losses, val_losses = train_model(
        X_train_scaled, y_train, 
        X_val_scaled, y_val, 
        input_dim=input_dim
    )
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, log_transformed=True)
    
    # Save model artifacts
    save_model_artifacts(model, scaler, metrics, output_dir)
    
    logger.info("Model training pipeline completed")

if __name__ == "__main__":
    main() 