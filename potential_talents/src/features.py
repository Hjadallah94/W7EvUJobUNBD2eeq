"""
Feature engineering for candidate ranking.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
from .config import TFIDF_PARAMS
from .dataset import preprocess_text


class FeatureExtractor:
    """
    Extract features from candidate data for ranking.
    """
    
    def __init__(self, keywords: str, tfidf_params: dict = None):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        keywords : str
            Target role keywords
        tfidf_params : dict, optional
            TF-IDF vectorizer parameters
        """
        self.keywords = keywords
        self.tfidf_params = tfidf_params or TFIDF_PARAMS
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.tfidf_matrix = None
        
    def fit_transform(self, df: pd.DataFrame, text_column: str = 'processed_title') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit TF-IDF vectorizer and compute similarity features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Candidate DataFrame
        text_column : str
            Name of the preprocessed text column
            
        Returns:
        --------
        keyword_similarities : np.ndarray
            Cosine similarities to keywords
        tfidf_matrix : np.ndarray
            TF-IDF matrix for all candidates
        """
        # Prepare texts: keywords + all candidate titles
        all_texts = [self.keywords] + df[text_column].tolist()
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Compute similarities
        keyword_vector = self.tfidf_matrix[0:1]
        candidate_vectors = self.tfidf_matrix[1:]
        
        keyword_similarities = cosine_similarity(keyword_vector, candidate_vectors).flatten()
        
        return keyword_similarities, candidate_vectors
    
    def compute_starred_similarities(self, candidate_vectors: np.ndarray, starred_indices: list) -> np.ndarray:
        """
        Compute similarities to starred candidates.
        
        Parameters:
        -----------
        candidate_vectors : np.ndarray
            TF-IDF vectors for all candidates
        starred_indices : list
            Indices of starred candidates
            
        Returns:
        --------
        np.ndarray
            Max similarity to any starred candidate for each candidate
        """
        if len(starred_indices) == 0:
            return np.zeros(candidate_vectors.shape[0])
        
        starred_vectors = candidate_vectors[starred_indices]
        starred_similarities = cosine_similarity(candidate_vectors, starred_vectors)
        max_starred_sim = starred_similarities.max(axis=1)
        
        return max_starred_sim
    
    def extract_keywords_from_top_candidates(self, df: pd.DataFrame, top_n: int = 30) -> dict:
        """
        Extract common keywords from top ranked candidates.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Ranked candidate DataFrame
        top_n : int
            Number of top candidates to analyze
            
        Returns:
        --------
        dict
            Word frequency dictionary
        """
        from collections import Counter
        import re
        
        top_titles = df.head(top_n)['job_title'].tolist()
        all_words = []
        
        for title in top_titles:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Remove common stop words
        stop_words = {'and', 'the', 'for', 'with', 'area', 'seeking', 'student', 'from', 'inc'}
        word_freq = {k: v for k, v in word_freq.items() if k not in stop_words}
        
        return word_freq
