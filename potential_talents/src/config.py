"""
Configuration settings for the talent ranking system.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Model parameters
DEFAULT_KEYWORDS = "aspiring human resources seeking human resources"
USE_CONNECTIONS = True
CONNECTION_WEIGHT = 0.2  # Weight for connection count in scoring
TEXT_WEIGHT = 0.8  # Weight for text similarity in scoring

# TF-IDF parameters
TFIDF_PARAMS = {
    'stop_words': 'english',
    'ngram_range': (1, 3),
    'min_df': 1,
    'lowercase': True
}

# Filtering thresholds
DEFAULT_PERCENTILE_CUTOFF = 75
MIN_FITNESS_SCORE = 0.0

# Active learning parameters
INITIAL_STARRED_WEIGHT = 0.3
MAX_STARRED_WEIGHT = 0.6
STARRED_WEIGHT_INCREMENT = 0.05

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
DPI = 100
