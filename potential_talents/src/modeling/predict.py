"""
Model inference and prediction utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from .ranking import TalentRankingSystem
from ..dataset import preprocess_candidates


def predict_fitness_scores(
    ranking_system: TalentRankingSystem,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict fitness scores for candidates.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    df : pd.DataFrame
        Candidate DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with fitness scores and ranks
    """
    # Preprocess if needed
    if 'processed_title' not in df.columns:
        df = preprocess_candidates(df)
    
    # Get rankings
    ranked_df = ranking_system.rank_candidates(df, ranking_system.starred_candidates)
    
    return ranked_df


def predict_top_k(
    ranking_system: TalentRankingSystem,
    df: pd.DataFrame,
    k: int = 10
) -> pd.DataFrame:
    """
    Get top K candidates.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    df : pd.DataFrame
        Candidate DataFrame
    k : int
        Number of top candidates to return
        
    Returns:
    --------
    pd.DataFrame
        Top K candidates
    """
    ranked_df = predict_fitness_scores(ranking_system, df)
    return ranked_df.head(k)


def predict_with_threshold(
    ranking_system: TalentRankingSystem,
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Get candidates above a fitness threshold.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    df : pd.DataFrame
        Candidate DataFrame
    threshold : float
        Minimum fitness score
        
    Returns:
    --------
    pd.DataFrame
        Filtered candidates above threshold
    """
    ranked_df = predict_fitness_scores(ranking_system, df)
    return ranking_system.filter_candidates(ranked_df, threshold)


def batch_predict(
    ranking_system: TalentRankingSystem,
    df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Predict in batches for large datasets.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    df : pd.DataFrame
        Candidate DataFrame
    batch_size : int
        Size of each batch
        
    Returns:
    --------
    pd.DataFrame
        All predictions concatenated
    """
    results = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_predictions = predict_fitness_scores(ranking_system, batch)
        results.append(batch_predictions)
        print(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
    
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values('fit', ascending=False).reset_index(drop=True)
    final_df['rank'] = range(1, len(final_df) + 1)
    
    return final_df


def get_candidate_explanation(
    ranking_system: TalentRankingSystem,
    candidate_id: int,
    ranked_df: pd.DataFrame
) -> dict:
    """
    Get explanation for why a candidate was ranked at their position.
    
    Parameters:
    -----------
    ranking_system : TalentRankingSystem
        Trained ranking system
    candidate_id : int
        ID of the candidate
    ranked_df : pd.DataFrame
        Ranked DataFrame
        
    Returns:
    --------
    dict
        Explanation dictionary
    """
    candidate_row = ranked_df[ranked_df['id'] == candidate_id].iloc[0]
    
    explanation = {
        'candidate_id': candidate_id,
        'rank': int(candidate_row['rank']),
        'fit_score': float(candidate_row['fit']),
        'job_title': candidate_row['job_title'],
        'location': candidate_row['location'],
        'connections': candidate_row['connection'],
        'percentile': (1 - (candidate_row['rank'] - 1) / len(ranked_df)) * 100
    }
    
    # Add context
    if candidate_row['fit'] >= 0.7:
        explanation['assessment'] = 'Excellent match'
    elif candidate_row['fit'] >= 0.5:
        explanation['assessment'] = 'Good match'
    elif candidate_row['fit'] >= 0.3:
        explanation['assessment'] = 'Moderate match'
    else:
        explanation['assessment'] = 'Weak match'
    
    return explanation
