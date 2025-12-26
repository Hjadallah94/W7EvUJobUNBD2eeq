# Quick Start Guide - Talent Ranking System

## 🚀 Getting Started in 5 Minutes

### Step 1: Installation

Open a terminal/command prompt and navigate to the project folder:

```bash
cd potential_talents
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the project package:

```bash
pip install -e .
```

### Step 2: Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `notebooks/01-initial-exploration.ipynb` and run all cells sequentially.

### Step 3: Understanding the Workflow

The notebook will automatically:
1. ✅ Load the candidate data from `data/raw/`
2. ✅ Rank all 104 candidates based on fitness scores
3. ✅ Show you the top candidates
4. ✅ Demonstrate the starring mechanism
5. ✅ Generate visualizations and analysis

## 📊 Using the Python API

For programmatic access, you can import and use the modules directly:

```python
# Import the necessary modules
from src.dataset import load_raw_data, preprocess_candidates
from src.modeling.ranking import TalentRankingSystem

# Load and preprocess data
df = load_raw_data('potential-talents - Aspiring human resources - seeking human resources.csv')
df = preprocess_candidates(df)

# Initialize ranking system
keywords = "aspiring human resources seeking human resources"
ranking_system = TalentRankingSystem(keywords)

# Rank candidates
ranked_df = ranking_system.rank_candidates(df)

# View top 10
print(ranked_df[['rank', 'job_title', 'fit']].head(10))

# Star a candidate (e.g., rank 7)
ranked_df = ranking_system.star_candidate(ranked_df, candidate_rank=7)

# Star another candidate (e.g., rank 3)
ranked_df = ranking_system.star_candidate(ranked_df, candidate_rank=3)

# View updated rankings
print(ranked_df[['rank', 'job_title', 'fit']].head(10))
```

## 🎯 Key Features to Explore

### 1. Initial Ranking
The system automatically ranks candidates using TF-IDF similarity matching.

### 2. Interactive Starring
Star candidates you find ideal, and the system learns from your choices:
```python
ranked_df = ranking_system.star_candidate(ranked_df, candidate_rank=7)
```

### 3. Filtering
Remove low-quality candidates with a threshold:
```python
filtered_df = ranking_system.filter_candidates(ranked_df, threshold=0.5)
```

### 4. Visualizations
Generate comprehensive visual reports:
```python
from src.plots import create_summary_report
create_summary_report(ranked_df, ranking_system, save=True)
```

### 5. Save/Load Models
Persist trained models for reuse:
```python
from src.modeling.train import save_model, load_model

# Save
save_model(ranking_system, 'my_hr_model.pkl')

# Load
loaded_system = load_model('my_hr_model.pkl')
```

## 📁 Where to Find Things

| What You Need | Where to Find It |
|---------------|------------------|
| Original candidate data | `data/raw/` |
| Processed/cleaned data | `data/processed/` |
| Exploration notebook | `notebooks/01-initial-exploration.ipynb` |
| Trained models | `models/` |
| Generated charts | `reports/figures/` |
| Documentation | `references/` |
| Source code | `src/` |

## 🔄 Typical Workflow

1. **Load Data**: Import candidate CSV
2. **Initial Ranking**: Run the ranking algorithm
3. **Review Top Candidates**: Manually inspect the top 20-30 candidates
4. **Star Ideal Candidates**: Mark 3-5 candidates as ideal
5. **Re-rank**: System automatically improves rankings
6. **Filter**: Apply threshold to remove poor fits
7. **Export**: Save final candidate list
8. **Repeat**: For new roles, use different keywords

## 🆘 Common Issues

### Issue: "Module not found"
**Solution**: Make sure you ran `pip install -e .` in the project directory

### Issue: "No such file or directory"
**Solution**: Check that you're running commands from the `potential_talents` folder

### Issue: "Kernel died" in Jupyter
**Solution**: Restart the kernel (Kernel → Restart) and run cells again

## 💡 Tips for Best Results

1. **Star Diverse Candidates**: Don't just star the top 3. Star a candidate from rank 7-15 to teach the system about hidden gems.

2. **Use Multiple Keywords**: Experiment with different keyword combinations:
   - "aspiring human resources"
   - "seeking hr position"
   - "human resources generalist"

3. **Review the Metrics**: Pay attention to:
   - Fitness score (higher = better)
   - Connection count (more connections might indicate experience)
   - Location (consider geographic preferences)

4. **Iterate**: The system learns from feedback. Star 3-5 candidates and observe how rankings improve.

## 📧 Support

For questions or issues:
- Review the full README.md
- Check references/PROJECT_DESCRIPTION.md
- Contact: [your-support-email@company.com]

---

**Ready to go?** Open `notebooks/01-initial-exploration.ipynb` and start ranking! 🎉
