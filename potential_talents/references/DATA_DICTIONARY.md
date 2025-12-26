# Data Dictionary

## Candidate Dataset

**Filename**: `potential-talents - Aspiring human resources - seeking human resources.csv`

**Description**: Dataset containing candidate information for Human Resources roles

**Total Records**: 104 candidates

### Column Descriptions

| Column Name | Data Type | Description | Example Values |
|------------|-----------|-------------|----------------|
| `id` | Integer | Unique identifier for each candidate | 1, 2, 3, ... |
| `job_title` | String | Current job title or professional description of the candidate | "Aspiring Human Resources Professional", "HR Senior Specialist" |
| `location` | String | Geographic location of the candidate | "Houston, Texas", "Greater New York City Area" |
| `connection` | String/Integer | Number of professional connections. "500+" indicates 500 or more connections | "85", "500+", "44" |
| `fit` | Float | **Target variable**. Fitness score between 0-1 indicating how well the candidate matches the role requirements. Initially empty, calculated by the ranking algorithm. | 0.0 - 1.0 |

### Data Processing Notes

- **Missing Values**: The `fit` column is initially empty and populated by the ranking algorithm
- **Connection Parsing**: "500+" is treated as numeric value 500 in calculations
- **Text Preprocessing**: Job titles are lowercased and cleaned of special characters for similarity matching
- **Duplicates**: Some job titles may appear multiple times (different candidates with similar backgrounds)

### Derived Features

The following features are computed during preprocessing:

| Feature Name | Type | Description |
|--------------|------|-------------|
| `processed_title` | String | Cleaned and preprocessed job title for NLP analysis |
| `connection_numeric` | Integer | Numeric representation of connections (500+ → 500) |
| `connection_normalized` | Float | Normalized connection score (0-1 scale) |
| `rank` | Integer | Candidate's position in the ranked list (1 = best fit) |

### Target Role Keywords

**Current Keywords**: "aspiring human resources" OR "seeking human resources"

These keywords are used to calculate initial fitness scores based on semantic similarity between the candidate's job title and the target role requirements.

### Example Records

```
id,job_title,location,connection,fit
1,"2019 C.T. Bauer College of Business Graduate (Magna Cum Laude) and aspiring Human Resources professional","Houston, Texas",85,0.8542
3,"Aspiring Human Resources Professional","Raleigh-Durham, North Carolina Area",44,0.9123
10,"Seeking Human Resources HRIS and Generalist Positions","Greater Philadelphia Area","500+",0.8876
```

### Data Quality

- **Completeness**: All core fields (id, job_title, location, connection) are populated
- **Consistency**: Location formats vary (city-state, metro areas, international)
- **Privacy**: Personal identifying information has been removed
- **Immutability**: Raw data should never be modified; all changes are made to processed copies
