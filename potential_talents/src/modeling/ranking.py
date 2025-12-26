"""
Core ranking algorithm for candidate evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..features import FeatureExtractor
from ..dataset import preprocess_candidates
from ..config import (
    USE_CONNECTIONS, CONNECTION_WEIGHT, TEXT_WEIGHT,
    INITIAL_STARRED_WEIGHT, MAX_STARRED_WEIGHT, STARRED_WEIGHT_INCREMENT
)


class TalentRankingSystem:
    """
    A ranking system that scores candidates and learns from human feedback.
    """
    
    def __init__(self, keywords: str, use_connections: bool = USE_CONNECTIONS):
        """
        Initialize the ranking system.
        
        Parameters:
        -----------
        keywords : str or list
            Target role keywords to match against
        use_connections : bool
            Whether to factor in number of connections
        """
        self.keywords = keywords if isinstance(keywords, str) else ' '.join(keywords)
        self.use_connections = use_connections
        self.feature_extractor = FeatureExtractor(self.keywords)
        self.starred_candidates = []
        self.ranking_history = []
        self.candidate_vectors = None
        
    def calculate_initial_fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate initial fitness scores using TF-IDF and cosine similarity.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Candidate DataFrame (must have 'processed_title' column)
            
        Returns:
        --------
        np.ndarray
            Fitness scores for all candidates
        """
        # Extract text features
        keyword_similarities, self.candidate_vectors = self.feature_extractor.fit_transform(df)
        
        # Factor in connections if enabled
        if self.use_connections and 'connection_normalized' in df.columns:
            connection_scores = df['connection_normalized'].values
            final_scores = TEXT_WEIGHT * keyword_similarities + CONNECTION_WEIGHT * connection_scores
        else:
            final_scores = keyword_similarities
        
        return final_scores
    
    def calculate_fit_with_feedback(self, df: pd.DataFrame, starred_indices: List[int]) -> np.ndarray:
        """
        Recalculate fitness scores incorporating starred candidates as positive examples.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Candidate DataFrame
        starred_indices : List[int]
            List of starred candidate indices
            
        Returns:
        --------
        np.ndarray
            Updated fitness scores
        """
        # Get keyword similarities
        keyword_similarities, self.candidate_vectors = self.feature_extractor.fit_transform(df)
        
        if len(starred_indices) > 0:
            # Calculate similarity to starred candidates
            max_starred_sim = self.feature_extractor.compute_starred_similarities(
                self.candidate_vectors, starred_indices
            )
            
            # Adaptive weighting: increase starred weight as we get more feedback
            weight_starred = min(
                MAX_STARRED_WEIGHT,
                INITIAL_STARRED_WEIGHT + (len(starred_indices) * STARRED_WEIGHT_INCREMENT)
            )
            weight_keyword = 1 - weight_starred
            
            # Combine scores
            combined_scores = (weight_keyword * keyword_similarities + 
                             weight_starred * max_starred_sim)
        else:
            combined_scores = keyword_similarities
        
        # Factor in connections
        if self.use_connections and 'connection_normalized' in df.columns:
            connection_scores = df['connection_normalized'].values
            final_scores = 0.85 * combined_scores + 0.15 * connection_scores
        else:
            final_scores = combined_scores
        
        return final_scores
    
    def rank_candidates(self, df: pd.DataFrame, starred_indices: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Rank candidates and return sorted dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Candidate DataFrame
        starred_indices : List[int], optional
            List of starred candidate indices
            
        Returns:
        --------
        pd.DataFrame
            Ranked candidate DataFrame with 'fit' and 'rank' columns
        """
        df_copy = df.copy()
        
        # Ensure data is preprocessed
        if 'processed_title' not in df_copy.columns:
            df_copy = preprocess_candidates(df_copy)
        
        # Calculate fitness scores
        if starred_indices is None or len(starred_indices) == 0:
            fit_scores = self.calculate_initial_fit(df_copy)
        else:
            fit_scores = self.calculate_fit_with_feedback(df_copy, starred_indices)
        
        df_copy['fit'] = fit_scores
        df_copy = df_copy.sort_values('fit', ascending=False).reset_index(drop=True)
        df_copy['rank'] = range(1, len(df_copy) + 1)
        
        # Store ranking history
        self.ranking_history.append(df_copy[['id', 'rank', 'fit']].copy())
        
        return df_copy
    
    def star_candidate(self, df: pd.DataFrame, candidate_rank: int) -> pd.DataFrame:
        """
        Star a candidate at given rank position and re-rank all candidates.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Current ranked DataFrame
        candidate_rank : int
            Rank position of candidate to star
            
        Returns:
        --------
        pd.DataFrame
            Re-ranked DataFrame
        """
        # Get the candidate's original index
        candidate_idx = df.index[df['rank'] == candidate_rank].tolist()[0]
        
        if candidate_idx not in self.starred_candidates:
            self.starred_candidates.append(candidate_idx)
            candidate_title = df.loc[candidate_idx, 'job_title']
            print(f"✭ Starred candidate at rank {candidate_rank}: {candidate_title}")
        
        # Re-rank with updated feedback
        return self.rank_candidates(df, self.starred_candidates)
    
    def filter_candidates(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Filter candidates below a fitness threshold.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Ranked DataFrame
        threshold : float
            Minimum fitness score
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        filtered_df = df[df['fit'] >= threshold].copy()
        filtered_df = filtered_df.sort_values('fit', ascending=False).reset_index(drop=True)
        filtered_df['rank'] = range(1, len(filtered_df) + 1)
        
        print(f"Filtered {len(df) - len(filtered_df)} candidates (kept {len(filtered_df)})")
        
        return filtered_df
    
    def get_ranking_stats(self) -> dict:
        """
        Get statistics about current ranking state.
        
        Returns:
        --------
        dict
            Dictionary of ranking statistics
        """
        if len(self.ranking_history) == 0:
            return {}
        
        current_ranking = self.ranking_history[-1]
        
        stats = {
            'total_candidates': len(current_ranking),
            'num_iterations': len(self.ranking_history),
            'num_starred': len(self.starred_candidates),
            'top_fit_score': current_ranking['fit'].max(),
            'mean_fit_score': current_ranking['fit'].mean(),
            'median_fit_score': current_ranking['fit'].median(),
            'std_fit_score': current_ranking['fit'].std()
        }
        
        return stats
