"""
Model training and persistence utilities.
"""

import pandas as pd
import joblib
import os
from typing import List, Optional
from .ranking import TalentRankingSystem
from ..config import MODELS_DIR


def train_ranking_model(
    df: pd.DataFrame,
    keywords: str,
    starred_candidates: Optional[List[int]] = None,
    use_connections: bool = True
) -> TalentRankingSystem:
    """
    Train a talent ranking model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Candidate DataFrame
    keywords : str
        Target role keywords
    starred_candidates : List[int], optional
        Pre-existing starred candidate indices
    use_connections : bool
        Whether to use connection features
        
    Returns:
    --------
    TalentRankingSystem
        Trained ranking system
    """
    print("Training ranking model...")
    print(f"Keywords: {keywords}")
    print(f"Training samples: {len(df)}")
    
    # Initialize system
    ranking_system = TalentRankingSystem(keywords, use_connections=use_connections)
    
    # Set starred candidates if provided
    if starred_candidates:
        ranking_system.starred_candidates = starred_candidates
        print(f"Using {len(starred_candidates)} starred candidates")
    
    # Perform initial ranking
    ranked_df = ranking_system.rank_candidates(df, starred_candidates)
    
    print(f"Training complete!")
    print(f"Top candidate fit score: {ranked_df.iloc[0]['fit']:.4f}")
    
    return ranking_system


def save_model(ranking_system: TalentRankingSystem, model_name: str = 'ranking_model.pkl') -> str:
    """
    Save trained ranking model to disk.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    model_name : str
        Name for the saved model file
        
    Returns:
    --------
    str
        Path to saved model
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(ranking_system, model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(model_name: str = 'ranking_model.pkl') -> TalentRankingSystem:
    """
    Load a trained ranking model from disk.
    
    Parameters:
    -----------
    model_name : str
        Name of the model file
        
    Returns:
    --------
    TalentRankingSystem
        Loaded ranking system
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    ranking_system = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    return ranking_system


def retrain_with_feedback(
    ranking_system: TalentRankingSystem,
    df: pd.DataFrame,
    new_starred_rank: int
) -> TalentRankingSystem:
    """
    Retrain model with new feedback (starred candidate).
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Existing ranking system
    df : pd.DataFrame
        Current ranked DataFrame
    new_starred_rank : int
        Rank of newly starred candidate
        
    Returns:
    --------
    TalentRankingSystem
        Updated ranking system
    """
    # Star the candidate and re-rank
    df_updated = ranking_system.star_candidate(df, new_starred_rank)
    
    print(f"Model retrained with feedback")
    print(f"Total starred candidates: {len(ranking_system.starred_candidates)}")
    
    return ranking_system
