# Large Analysis Files Management

This document provides instructions for handling large analysis files in the UAVarPrior project.

## Output Directory

Large analysis files should be stored in the `uavarprior/interpret/outputs/` directory, which is not tracked by Git.

## Contents

The outputs directory should store:
- Large pickle files (`.pkl`)
- Analysis results
- Variant matrices
- Feature importance data
- Visualization data

## Guidelines

1. Files in the outputs directory are **not tracked** by Git (as specified in the root `.gitignore` file)
2. Files larger than 100MB **cannot** be pushed to GitHub
3. Share these files with collaborators via cloud storage or other file sharing mechanisms
4. Consider using relative paths in your code to reference these files:

```python
import os
import pickle

# Get the absolute path to the outputs directory
interpret_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs')

# Example: Save analysis results
results_path = os.path.join(interpret_dir, 'variant_analysis_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

# Example: Load analysis results
with open(results_path, 'rb') as f:
    results = pickle.load(f)
```

## Current Large Files

The outputs directory currently contains:
- `unique_names_group_1.pkl` (~145MB)
- `unique_names_group_1_pred1.pkl` (~190MB)
- `variant_membership_matrix_group_1.pkl` (~842MB)
- `variant_membership_matrix_group_1_pred1.pkl` (~1.1GB)
