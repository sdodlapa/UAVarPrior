# Large Data Files Management

This document provides instructions for handling large data files in the UAVarPrior project.

## Output Directory

Large data files should be stored in the `uavarprior/data/outputs/` directory, which is not tracked by Git.

## Contents

The outputs directory should store:
- Large `.parquet.gz` files
- Large `.h5` or `.hdf5` files
- Any other data files larger than 50MB

## Guidelines

1. Files in the outputs directory are **not tracked** by Git (as specified in the root `.gitignore` file)
2. Files larger than 100MB **cannot** be pushed to GitHub
3. Share these files with collaborators via cloud storage or other file sharing mechanisms
4. Consider using relative paths in your code to reference these files:

```python
import os

# Get the absolute path to the outputs directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs')

# Example: Load a file
file_path = os.path.join(data_dir, 'combined_variant_positions.parquet.gz')
data = pd.read_parquet(file_path)
```

## Documentation

Always document the source and processing steps that generated files in the outputs directory, 
either in a README within that directory or in a separate metadata file.
