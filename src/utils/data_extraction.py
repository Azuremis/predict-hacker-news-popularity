import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def connect_to_db():
    """Connect to the Hacker News PostgreSQL database."""
    USERNAME = os.getenv('DB_USERNAME', 'sy91dhb')
    PASSWORD = os.getenv('DB_PASSWORD', 'g5t49ao')
    HOST     = os.getenv('DB_HOST', '178.156.142.230')
    PORT     = os.getenv('DB_PORT', '5432')
    DBNAME   = os.getenv('DB_NAME', 'hd64m1ki')
    
    connection_string = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
    engine = create_engine(connection_string)
    
    return engine

def extract_items(engine, limit=100000, order_by='time DESC'):
    """Extract Hacker News items from the database.
    
    Args:
        engine: SQLAlchemy engine
        limit: Number of items to extract
        order_by: SQL ORDER BY clause
        
    Returns:
        DataFrame of items
    """
    query = f"""
    SELECT id, title, score, "time", by AS author, url, text, descendants
    FROM "hacker_news"."items"
    WHERE title IS NOT NULL AND score IS NOT NULL
    ORDER BY {order_by}
    LIMIT {limit};
    """
    
    print(f"Extracting {limit} items from database...")
    df_items = pd.read_sql(query, engine)
    
    # Convert timestamp to datetime if needed
    if 'time' in df_items.columns and not pd.api.types.is_datetime64_any_dtype(df_items['time']):
        df_items['time'] = pd.to_datetime(df_items['time'])
    
    print(f"Extracted {len(df_items)} items.")
    return df_items

def extract_users(engine, user_ids):
    """Extract user data for a list of user IDs.
    
    Args:
        engine: SQLAlchemy engine
        user_ids: List of user IDs
        
    Returns:
        DataFrame of users
    """
    # Convert list of user IDs to a comma-separated string of quoted values
    user_ids_str = "', '".join(user_ids)
    
    query = f"""
    SELECT id, created, karma, about, submitted
    FROM "hacker_news"."users"
    WHERE id IN ('{user_ids_str}');
    """
    
    print(f"Extracting data for {len(user_ids)} users...")
    df_users = pd.read_sql(query, engine)
    
    # Convert timestamp to datetime if needed
    if 'created' in df_users.columns and not pd.api.types.is_datetime64_any_dtype(df_users['created']):
        df_users['created'] = pd.to_datetime(df_users['created'])
    
    print(f"Extracted data for {len(df_users)} users.")
    return df_users

def main():
    """Main function to extract and save data."""
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Connect to database
    engine = connect_to_db()
    
    # Extract items
    df_items = extract_items(engine)
    
    # Get unique user IDs
    user_ids = df_items['author'].dropna().unique().tolist()
    print(f"Found {len(user_ids)} unique authors.")
    
    # Extract user data
    df_users = extract_users(engine, user_ids)
    
    # Save to parquet files
    items_path = 'data/raw/items_100k.parquet'
    users_path = 'data/raw/users_100k.parquet'
    
    df_items.to_parquet(items_path)
    df_users.to_parquet(users_path)
    
    print(f"Saved items to {items_path}")
    print(f"Saved users to {users_path}")
    
    # Optional: Create a merged dataset with user features
    df_merged = df_items.merge(df_users, left_on='author', right_on='id', how='left', suffixes=('', '_user'))
    merged_path = 'data/raw/items_users_merged_100k.parquet'
    df_merged.to_parquet(merged_path)
    print(f"Saved merged dataset to {merged_path}")

if __name__ == "__main__":
    main() 