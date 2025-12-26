"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import re
from typing import Optional
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import os


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw candidate data from CSV file.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file in the raw data directory
        
    Returns:
    --------
    pd.DataFrame
        Loaded candidate data
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} candidates from {filename}")
    return df


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def parse_connections(connection_str: str) -> int:
    """
    Parse connection count from string format.
    Handles '500+' format and converts to numeric.
    
    Parameters:
    -----------
    connection_str : str
        Connection count as string (e.g., '85', '500+')
        
    Returns:
    --------
    int
        Numeric connection count
    """
    if pd.isna(connection_str):
        return 0
    
    conn_str = str(connection_str).strip()
    
    if '500+' in conn_str or '500 +' in conn_str:
        return 500
    
    try:
        return int(conn_str)
    except ValueError:
        return 0


def normalize_connections(df: pd.DataFrame, column: str = 'connection') -> pd.Series:
    """
    Normalize connection counts to 0-1 scale.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing connection data
    column : str
        Name of the connection column
        
    Returns:
    --------
    pd.Series
        Normalized connection scores
    """
    connections_numeric = df[column].apply(parse_connections)
    max_conn = connections_numeric.max()
    
    if max_conn > 0:
        return connections_numeric / max_conn
    
    return connections_numeric


def preprocess_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to candidate data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw candidate DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    # Preprocess job titles
    df_processed['processed_title'] = df_processed['job_title'].apply(preprocess_text)
    
    # Parse connections
    df_processed['connection_numeric'] = df_processed['connection'].apply(parse_connections)
    
    # Normalize connections
    df_processed['connection_normalized'] = normalize_connections(df_processed)
    
    print(f"Preprocessing complete. {len(df_processed)} candidates ready.")
    
    return df_processed


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to the processed data directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed DataFrame to save
    filename : str
        Output filename
    """
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load processed data from the processed data directory.
    
    Parameters:
    -----------
    filename : str
        Name of the processed CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded processed data
    """
    filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(filepath)
    print(f"Loaded processed data from {filepath}")
    return df
