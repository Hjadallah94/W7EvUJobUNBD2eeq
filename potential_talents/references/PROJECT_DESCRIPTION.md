# Project Description

## Background

As a talent sourcing and management company, we are interested in finding talented individuals for sourcing these candidates to technology companies. Finding talented candidates is not easy, for several reasons:

1. **Understanding the Role**: One needs to understand what the role is very well to fill in that spot, which requires understanding the client's needs and what they are looking for in a potential candidate.

2. **Identifying Excellence**: One needs to understand what makes a candidate shine for the role we are in search for.

3. **Finding Talent**: Where to find talented individuals is another challenge.

## Problem Statement

The nature of our job requires a lot of human labor and is full of manual operations. Towards automating this process, we want to build a better approach that could save us time and help us spot potential candidates that could fit the roles we are in search for.

We want to develop a machine learning powered pipeline that could:
- Spot talented individuals
- Rank them based on their fitness for specific roles

## Current Process

We are right now semi-automatically sourcing a few candidates. Therefore, the sourcing part is not a concern at this time, but we expect to first determine best matching candidates based on how fit these candidates are for a given role.

We generally make these searches based on keywords such as:
- "full-stack software engineer"
- "engineering manager"
- "aspiring human resources"

These keywords might change, and specific keywords will be provided for each role.

## Interactive Learning Requirement

Assuming that we were able to list and rank fitting candidates, we then employ a review procedure. Each candidate needs to be reviewed and then determined how good a fit they are through manual inspection.

**Key Insight**: At the end of manual review, we might choose not the first fitting candidate in the list but maybe the 7th candidate. If that happens, we are interested in being able to **re-rank the previous list** based on this information.

This supervisory signal is supplied by **"starring"** the 7th candidate in the list. Starring one candidate actually sets this candidate as an ideal candidate for the given role. Then, we expect the list to be re-ranked each time a candidate is starred.

## Data Description

The data comes from our sourcing efforts. We removed any field that could directly reveal personal details and gave a unique identifier for each candidate.

### Attributes:
- **id**: unique identifier for candidate (numeric)
- **job_title**: job title for candidate (text)
- **location**: geographical location for candidate (text)
- **connections**: number of connections candidate has, 500+ means over 500 (text)

### Target Output:
- **fit**: how fit the candidate is for the role? (numeric, probability between 0-1)

### Current Role Keywords:
"Aspiring human resources" or "seeking human resources"

## Project Goals

1. **Predict** how fit the candidate is based on their available information (variable fit)

## Success Metrics

1. Rank candidates based on a fitness score
2. Re-rank candidates when a candidate is starred
3. Demonstrate improvement with each starring action

## Current Challenges

1. We are interested in a **robust algorithm** - tell us how your solution works and show us how your ranking gets better with each starring action.

2. **Filtering**: How can we filter out candidates which in the first place should not be in this list?

3. **Cut-off Determination**: Can we determine a cut-off point that would work for other roles without losing high potential candidates?

4. **Automation & Bias Reduction**: Do you have any ideas that we should explore so that we can even automate this procedure to prevent human bias?

## Data Source

Original data available at:
https://docs.google.com/spreadsheets/d/117X6i53dKiO7w6kuA1g1TpdTlv1173h_dPlJt5cNNMU/edit?usp=sharing
