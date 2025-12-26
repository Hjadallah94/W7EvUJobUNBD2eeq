# Potential Talents - Talent Ranking System

An intelligent ML-powered pipeline for ranking and selecting candidates based on job role requirements with active learning capabilities.

## 🎯 Project Overview

This project addresses the challenge of identifying and ranking talented candidates for specific roles in the technology sector. The system uses Natural Language Processing (NLP) and machine learning to:

- **Rank candidates** based on their fitness for a given role
- **Learn from human feedback** through an interactive starring mechanism
- **Improve rankings** progressively with each manual review
- **Filter out irrelevant candidates** to save review time
- **Reduce human bias** through data-driven decision making

## 🚀 Key Features

- **TF-IDF based similarity matching** for initial candidate ranking
- **Active learning** system that improves with recruiter feedback
- **Interactive re-ranking** when candidates are starred as ideal
- **Automated filtering** with configurable cut-off thresholds
- **Comprehensive visualizations** for ranking analysis
- **Diversity analysis** to prevent location and connection bias

## 📁 Project Structure

```
potential_talents/
├── README.md                  <- This file
├── requirements.txt           <- Python dependencies
├── setup.py                   <- Makes project pip installable
├── .gitignore                <- Files to ignore in version control
│
├── data/
│   ├── raw/                  <- Original, immutable data
│   ├── interim/              <- Intermediate transformed data
│   └── processed/            <- Final datasets for modeling
│
├── notebooks/                <- Jupyter notebooks for exploration and analysis
│   └── 01-initial-exploration.ipynb
│
├── src/                      <- Source code for the project
│   ├── __init__.py
│   ├── config.py             <- Configuration parameters
│   ├── dataset.py            <- Data loading and preprocessing
│   ├── features.py           <- Feature engineering
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── ranking.py        <- Core ranking algorithm
│   │   ├── train.py          <- Model training logic
│   │   └── predict.py        <- Inference and prediction
│   └── plots.py              <- Visualization utilities
│
├── models/                   <- Trained models and vectorizers
│
├── reports/                  <- Generated analysis reports
│   └── figures/              <- Graphics and visualizations
│
└── references/               <- Data dictionaries, manuals, project docs
    └── PROJECT_DESCRIPTION.md
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone or download this project

2. Navigate to the project directory:
```bash
cd potential_talents
```

3. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install the project as a package:
```bash
pip install -e .
```

## 📊 Usage

### Quick Start

1. Place your candidate data in `data/raw/`

2. Run the exploration notebook:
```bash
jupyter notebook notebooks/01-initial-exploration.ipynb
```

3. Use the ranking system:
```python
from src.modeling.ranking import TalentRankingSystem

# Initialize system
keywords = "aspiring human resources seeking human resources"
ranking_system = TalentRankingSystem(keywords)

# Rank candidates
ranked_df = ranking_system.rank_candidates(df)

# Star a candidate to improve rankings
ranked_df = ranking_system.star_candidate(ranked_df, candidate_rank=7)
```

### Command Line Interface (Future)
```bash
# Rank candidates
python -m src.modeling.predict --data data/raw/candidates.csv --keywords "aspiring human resources"

# Train with feedback
python -m src.modeling.train --data data/raw/candidates.csv --starred-ids 1,5,12
```

## 🧪 Methodology

### Initial Ranking
- **TF-IDF Vectorization**: Converts job titles to numerical features
- **N-gram Analysis**: Captures phrases (1-3 words)
- **Cosine Similarity**: Measures text similarity to target keywords
- **Connection Weighting**: Factors in candidate network size (20% weight)

### Active Learning & Re-ranking
- **Starred Candidates**: Manual selections become positive training examples
- **Similarity Propagation**: Finds candidates similar to starred profiles
- **Adaptive Weighting**: Increases weight on human feedback with more data
- **Hybrid Scoring**: Combines keyword matching + starred similarity

### Filtering
- **Percentile-based Thresholds**: Flexible cut-off points
- **Elbow Method**: Detects natural break points in fitness scores
- **Quality vs Quantity Trade-off**: Balance coverage and precision

## 📈 Performance

The system demonstrates:
- **Progressive improvement** with each starring iteration
- **Robust initial rankings** even without training data
- **Effective filtering** reducing candidate pool by 50-75%
- **Low bias** through text-based semantic matching

## 🤝 Contributing

For internal development:
1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit for review

## 📝 License

Proprietary - All rights reserved

## 👥 Team

**Project Lead**: Apziva Data Science Team
**Client**: Talent Sourcing & Management Company

## 📧 Contact

For questions or support, contact: [your-email@company.com]

---

**Version**: 1.0.0  
**Last Updated**: December 2025
