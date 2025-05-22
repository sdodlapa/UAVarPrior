# Profile Similarity Analysis Usage Guide

This document provides instructions for running the profile similarity analysis scripts.

## Overview

The profile similarity analysis tool analyzes genetic variant profiles to identify cells with similar variant patterns. It calculates:
- Common variants between pairs of profiles
- Jaccard similarity between profiles
- Most similar profile pairs
- Visualizations of the similarity matrix

## Available Scripts

1. `analyze_profile_similarity.py` - Main analysis script with visualization
2. `simple_profile_analysis.py` - Simplified analysis script without heavy dependencies
3. `standalone_test.py` - Direct matrix analysis with minimal dependencies
4. `create_test_matrix.py` - Generate synthetic test matrices
5. `check_dependencies.py` - Check for required Python modules
6. `copy_existing_matrices.py` - Copy matrices from a source location
7. `run_profile_analysis.csh` - Shell script to run the analysis
8. `run_external_analysis.csh` - Shell script to run analysis with external data

## Getting Started

### Option 1: Run with external path (for existing matrices)

```bash
./run_external_analysis.csh
```

This script will use the existing matrices at `/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana/`.

### Option 2: Run with specific paths

```bash
python analyze_profile_similarity.py --ext-path=/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana
```

or

```bash
python analyze_profile_similarity.py --matrix-file=/path/to/matrix.npz --files-file=/path/to/files.pkl
```

### Option 3: Generate test data

```bash
python create_test_matrix.py --profiles=100 --variants=500 --density=0.05
```

This will create test matrices in the `outputs` directory.

## Troubleshooting

If you encounter dependency issues:

1. Check dependencies:
   ```bash
   python check_dependencies.py
   ```

2. Install required packages:
   ```bash
   pip install numpy scipy
   ```

3. For visualization:
   ```bash
   pip install matplotlib seaborn
   ```

4. Try the standalone test:
   ```bash
   python standalone_test.py --matrix-type=similarity
   ```

## Command Line Arguments

The analysis scripts accept the following arguments:

- `--model`: Model name (default: pred1)
- `--group`: Group number (default: 1)
- `--output-dir`: Directory for output files (default: ./outputs)
- `--matrix-file`: Direct path to a similarity matrix file
- `--files-file`: Direct path to a files list file
- `--ext-path`: External path containing matrix files

For `create_test_matrix.py`:
- `--profiles`: Number of profiles to generate (default: 100)
- `--variants`: Number of variants to generate (default: 500)
- `--density`: Density of non-zero elements (default: 0.05)

For `standalone_test.py`:
- `--matrix-type`: Type of matrix to analyze (membership, similarity, or both)
