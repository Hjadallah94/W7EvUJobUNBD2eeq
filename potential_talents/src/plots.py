"""
Visualization utilities for talent ranking analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
from .config import PLOT_STYLE, FIGURE_SIZE, DPI, FIGURES_DIR
import os


# Set style
plt.style.use(PLOT_STYLE)
sns.set_palette("husl")


def plot_fitness_distribution(df: pd.DataFrame, save: bool = False, filename: str = 'fitness_distribution.png'):
    """
    Plot the distribution of fitness scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ranked DataFrame with 'fit' column
    save : bool
        Whether to save the figure
    filename : str
        Output filename if saving
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    ax.hist(df['fit'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(df['fit'].median(), color='red', linestyle='--', linewidth=2,
               label=f'Median: {df["fit"].median():.3f}')
    ax.set_xlabel('Fitness Score', fontsize=12)
    ax.set_ylabel('Number of Candidates', fontsize=12)
    ax.set_title('Distribution of Candidate Fitness Scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.tight_layout()
    return fig


def plot_top_candidates(df: pd.DataFrame, top_n: int = 20, save: bool = False, filename: str = 'top_candidates.png'):
    """
    Plot top N candidates by fitness score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ranked DataFrame
    top_n : int
        Number of top candidates to show
    save : bool
        Whether to save the figure
    filename : str
        Output filename if saving
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_df = df.head(top_n)
    ax.barh(range(len(top_df)), top_df['fit'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels([f"Rank {r}" for r in top_df['rank']], fontsize=9)
    ax.set_xlabel('Fitness Score', fontsize=12)
    ax.set_title(f'Top {top_n} Candidates by Fitness Score', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.tight_layout()
    return fig


def plot_ranking_progression(ranking_history: List[pd.DataFrame], top_n: int = 15, 
                             save: bool = False, filename: str = 'ranking_progression.png'):
    """
    Plot how candidate rankings change over iterations.
    
    Parameters:
    -----------
    ranking_history : List[pd.DataFrame]
        List of ranking DataFrames across iterations
    top_n : int
        Number of initial top candidates to track
    save : bool
        Whether to save the figure
    filename : str
        Output filename if saving
    """
    if len(ranking_history) < 2:
        print("Need at least 2 iterations to show progression")
        return None
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Get initial top candidate IDs
    initial_top_ids = ranking_history[0].head(top_n)['id'].tolist()
    
    # Track their ranks
    for candidate_id in initial_top_ids:
        ranks = []
        for iteration in ranking_history:
            rank = iteration[iteration['id'] == candidate_id]['rank'].values[0]
            ranks.append(rank)
        ax.plot(range(len(ranks)), ranks, marker='o', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('Iteration (Starring Events)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rank Position', fontsize=12, fontweight='bold')
    ax.set_title(f'Rank Changes for Top {top_n} Initial Candidates', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(ranking_history)))
    ax.set_xticklabels([f'Iter {i}' for i in range(len(ranking_history))])
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.tight_layout()
    return fig


def plot_cutoff_analysis(df: pd.DataFrame, recommended_threshold: float = None,
                         save: bool = False, filename: str = 'cutoff_analysis.png'):
    """
    Plot fitness scores with cut-off threshold visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ranked DataFrame
    recommended_threshold : float, optional
        Recommended threshold to highlight
    save : bool
        Whether to save the figure
    filename : str
        Output filename if saving
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sorted_fit = df.sort_values('fit', ascending=False)['fit'].values
    
    # Plot 1: Fitness scores vs rank
    axes[0].plot(range(len(sorted_fit)), sorted_fit, linewidth=2, color='steelblue')
    if recommended_threshold:
        axes[0].axhline(recommended_threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Recommended: {recommended_threshold:.3f}')
    axes[0].axhline(np.percentile(sorted_fit, 75), color='orange', linestyle='--', linewidth=2,
                   label=f'75th %ile: {np.percentile(sorted_fit, 75):.3f}')
    axes[0].set_xlabel('Candidate Rank', fontsize=12)
    axes[0].set_ylabel('Fitness Score', fontsize=12)
    axes[0].set_title('Fitness Scores vs Rank Position', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rate of change
    differences = np.abs(np.diff(sorted_fit))
    axes[1].plot(range(len(differences)), differences, linewidth=2, color='darkgreen')
    max_drop_idx = np.argmax(differences)
    axes[1].axvline(max_drop_idx, color='red', linestyle='--', linewidth=2,
                   label=f'Largest Drop at Rank {max_drop_idx + 1}')
    axes[1].set_xlabel('Rank Position', fontsize=12)
    axes[1].set_ylabel('Absolute Fitness Difference', fontsize=12)
    axes[1].set_title('Rate of Fitness Score Change', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.tight_layout()
    return fig


def plot_diversity_analysis(df: pd.DataFrame, top_n: int = 30,
                            save: bool = False, filename: str = 'diversity_analysis.png'):
    """
    Plot location and connection diversity in top candidates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ranked DataFrame
    top_n : int
        Number of top candidates to analyze
    save : bool
        Whether to save the figure
    filename : str
        Output filename if saving
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    top_df = df.head(top_n)
    
    # Location distribution
    location_dist = top_df['location'].value_counts().head(10)
    location_dist.plot(kind='barh', ax=axes[0], color='teal', edgecolor='black')
    axes[0].set_xlabel('Number of Candidates', fontsize=11)
    axes[0].set_title(f'Top 10 Locations in Top {top_n} Candidates', fontsize=12, fontweight='bold')
    
    # Connection distribution
    connection_dist = top_df['connection'].value_counts()
    connection_dist.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_xlabel('Connection Range', fontsize=11)
    axes[1].set_ylabel('Number of Candidates', fontsize=11)
    axes[1].set_title(f'Connection Distribution in Top {top_n}', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.tight_layout()
    return fig


def create_summary_report(df: pd.DataFrame, ranking_system, save: bool = True):
    """
    Create a comprehensive visual report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Ranked DataFrame
    ranking_system : TalentRankingSystem
        The ranking system object
    save : bool
        Whether to save figures
    """
    print("Generating summary report...")
    
    # Create all plots
    plot_fitness_distribution(df, save=save, filename='01_fitness_distribution.png')
    plot_top_candidates(df, top_n=20, save=save, filename='02_top_candidates.png')
    
    if len(ranking_system.ranking_history) > 1:
        plot_ranking_progression(ranking_system.ranking_history, save=save, 
                                filename='03_ranking_progression.png')
    
    # Determine recommended threshold
    sorted_fit = df.sort_values('fit', ascending=False)['fit'].values
    differences = np.abs(np.diff(sorted_fit))
    max_drop_idx = np.argmax(differences)
    recommended_threshold = sorted_fit[max_drop_idx]
    
    plot_cutoff_analysis(df, recommended_threshold, save=save, filename='04_cutoff_analysis.png')
    plot_diversity_analysis(df, save=save, filename='05_diversity_analysis.png')
    
    print("Report generation complete!")
    
    plt.show()
