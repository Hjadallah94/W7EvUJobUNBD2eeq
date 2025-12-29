# Embedding Techniques for Talent Ranking

This directory contains notebooks exploring different embedding techniques for ranking potential talent candidates. Each notebook compares a specific embedding method with the baseline TF-IDF approach.

## Notebooks Overview

### 01-initial-exploration.ipynb
**Original notebook** - Implements the baseline TF-IDF ranking system with interactive feedback learning.
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Implements cosine similarity for matching
- Includes interactive starring/feedback mechanism
- Incorporates connection count as a ranking factor

### 02-word2vec-embeddings.ipynb
**Word2Vec Embeddings** - Dense vector representations capturing semantic relationships.
- Trains Word2Vec model on job titles using skip-gram architecture
- Creates document embeddings by averaging word vectors
- Better at understanding synonyms and related terms
- Compares results with TF-IDF baseline

**Key Features:**
- Semantic understanding (e.g., "HR" ≈ "Human Resources")
- Dense vector representations (100 dimensions)
- Context-based learning from word co-occurrence

**Dependencies:**
```bash
pip install gensim
```

### 03-glove-embeddings.ipynb
**GloVe Embeddings** - Pre-trained global word vectors from large corpora.
- Uses pre-trained GloVe embeddings (or simulates with Word2Vec for demo)
- Leverages knowledge from billions of words
- Better generalization through external pre-training
- Compares semantic understanding vs TF-IDF

**Key Features:**
- Pre-trained on massive datasets (Wikipedia, Common Crawl)
- Global co-occurrence statistics
- Available in multiple dimensions (50d, 100d, 200d, 300d)

**Dependencies:**
```bash
pip install gensim
# Download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/
```

### 04-fasttext-embeddings.ipynb
**FastText Embeddings** - Character n-gram based embeddings for robust word representation.
- Uses subword information (character n-grams)
- Can handle out-of-vocabulary words and typos
- Better at morphological variations (e.g., "recruit", "recruiter", "recruiting")
- Demonstrates robustness to text quality issues

**Key Features:**
- Character n-grams (3-6 characters)
- Handles misspellings and rare words
- Morphological awareness
- OOV word generation capability

**Dependencies:**
```bash
pip install gensim
```

### 05-sbert-embeddings.ipynb
**Sentence-BERT (SBERT)** - Transformer-based sentence embeddings for state-of-the-art semantic similarity.
- Uses pre-trained SBERT models (all-MiniLM-L6-v2)
- Context-aware sentence-level embeddings
- Best performance on semantic similarity tasks
- Demonstrates deep semantic understanding

**Key Features:**
- Pre-trained transformer models (BERT-based)
- Sentence-level representations (not word averaging)
- State-of-the-art semantic similarity
- Multiple model options for speed/quality tradeoffs

**Dependencies:**
```bash
pip install sentence-transformers
```

**Model Options:**
- `all-MiniLM-L6-v2`: Fast, 384 dimensions (recommended for most tasks)
- `all-mpnet-base-v2`: Best quality, 768 dimensions (slower)
- `paraphrase-MiniLM-L6-v2`: Optimized for paraphrase detection

### 06-comprehensive-comparison.ipynb
**All Methods Comparison** - Side-by-side comparison of all embedding techniques.
- Runs all five methods (TF-IDF, Word2Vec, GloVe, FastText, SBERT)
- Performance benchmarking (speed and accuracy)
- Correlation analysis between methods
- Score distribution comparisons
- Top candidates analysis across methods

**Includes:**
- Computational performance comparison
- Score distribution visualizations
- Spearman rank correlation matrix
- Common vs unique top candidates
- Statistical summary and recommendations

## Quick Start

### Install All Dependencies
```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Word embeddings (Word2Vec, GloVe, FastText)
pip install gensim

# Sentence-BERT
pip install sentence-transformers
```

### Run the Notebooks

1. **Start with the baseline:**
   ```
   01-initial-exploration.ipynb
   ```

2. **Explore individual embedding methods:**
   ```
   02-word2vec-embeddings.ipynb
   03-glove-embeddings.ipynb
   04-fasttext-embeddings.ipynb
   05-sbert-embeddings.ipynb
   ```

3. **Compare all methods:**
   ```
   06-comprehensive-comparison.ipynb
   ```

## Comparison Summary

| Method | Semantic Understanding | Speed | Handles OOV | Pre-trained | Best For |
|--------|----------------------|-------|-------------|-------------|----------|
| **TF-IDF** | ❌ No | ⚡ Fastest | ❌ No | ✅ N/A | Keyword matching |
| **Word2Vec** | ✅ Good | 🔶 Moderate | ❌ No | ❌ No | Related terms |
| **GloVe** | ✅ Good | ⚡ Fast* | ❌ No | ✅ Yes | Generalization |
| **FastText** | ✅ Good | 🔶 Moderate | ✅ Yes | ✅ Yes | Typos, variations |
| **SBERT** | ✅ Excellent | 🐌 Slowest | ✅ Yes | ✅ Yes | Deep semantics |

*Fast when using pre-trained embeddings

## Output Files

Each notebook generates comparison CSV files:
- `word2vec_vs_tfidf_comparison.csv`
- `glove_vs_tfidf_comparison.csv`
- `fasttext_vs_tfidf_comparison.csv`
- `sbert_vs_tfidf_comparison.csv`
- `all_methods_comparison.csv` (comprehensive)
- `embedding_methods_summary.csv` (statistical summary)

## Recommendations

### For Production Systems:
1. **Limited Resources**: Use **TF-IDF**
   - Fast, reliable, good for exact keyword matching
   - No training required

2. **Best Quality**: Use **SBERT**
   - State-of-the-art semantic understanding
   - Best for fuzzy matching and semantic search
   - Requires more computational resources

3. **Balanced Approach**: Use **FastText**
   - Good semantic understanding
   - Robust to typos and variations
   - Reasonable speed

### Hybrid Approach:
Consider combining methods:
1. **TF-IDF** for initial filtering (fast broad search)
2. **SBERT** for final ranking (quality semantic matching)
3. **FastText** for handling text variations

## Data

All notebooks use the same dataset:
```
potential-talents - Aspiring human resources - seeking human resources.csv
```

The dataset contains candidate information:
- `id`: Unique candidate identifier
- `job_title`: Candidate's job title
- `location`: Geographic location
- `connection`: Number of LinkedIn connections

## Key Insights

### TF-IDF (Baseline)
- ✅ Fast and efficient
- ✅ Works well for exact keyword matching
- ❌ No semantic understanding
- ❌ Misses related terms and synonyms

### Word2Vec
- ✅ Captures word relationships
- ✅ Understands synonyms
- ❌ Averaging effect loses context
- ❌ Limited by training corpus size

### GloVe
- ✅ Benefits from pre-trained knowledge
- ✅ Good generalization
- ❌ Fixed vocabulary
- ❌ Same vector regardless of context

### FastText
- ✅ Handles unknown words via subwords
- ✅ Robust to typos and variations
- ✅ Morphological awareness
- ❌ Still uses simple averaging

### SBERT
- ✅ Best semantic understanding
- ✅ Sentence-level representations
- ✅ State-of-the-art performance
- ❌ Slower computation
- ❌ Requires more resources

## License

This project is part of the Potential Talents analysis for Apziva.

## Contact

For questions or issues, please refer to the main project README.
