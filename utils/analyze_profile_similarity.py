#!/usr/bin/env python3
"""
Script to demonstrate the profile similarity matrix functionality.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import numpy as np
from scipy import sparse
import importlib.util
import pickle

# Load the variant_analysis.py module directly rather than through the package
# This avoids dependencies on the full UAVarPrior package
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'uavarprior/interpret/variant_analysis.py')

if not os.path.exists(module_path):
    print(f"Error: Could not find variant_analysis.py at {module_path}")
    sys.exit(1)

# Load the module directly using importlib
spec = importlib.util.spec_from_file_location("variant_analysis", module_path)
variant_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant_analysis)

# Get the required functions from the module
load_profile_similarity_matrix = variant_analysis.load_profile_similarity_matrix
visualize_profile_similarity = variant_analysis.visualize_profile_similarity

# Import matplotlib and seaborn for visualization if needed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    has_visualization = True
except ImportError:
    print("Warning: matplotlib and/or seaborn not available. Visualization will be disabled.")
    has_visualization = False

def analyze_similarity_matrix(pred_model="pred1", group=1, output_dir=None):
    """
    Analyze the profile similarity matrix for a specific prediction model and group.
    
    Args:
        pred_model: Prediction model identifier (default: "pred1")
        group: Group number (default: 1)
        output_dir: Output directory containing saved files (default: None, uses ./outputs)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    print(f"Loading profile similarity matrix for {pred_model} (Group {group})...")
    similarity_matrix, files = load_profile_similarity_matrix(
        pred_model=pred_model,
        group=group,
        output_dir=output_dir,
        verbose=True
    )
    
    # Visualize the similarity matrix if visualization libraries are available
    if has_visualization:
        print("\nVisualizing profile similarity matrix...")
        output_file = os.path.join(output_dir, f'similarity_heatmap_{pred_model}_group_{group}.png')
        visualize_profile_similarity(
            similarity_matrix=similarity_matrix,
            files=files,
            sample_size=50,  # Sample 50 profiles for visualization
            output_file=output_file
        )
    else:
        print("\nSkipping visualization due to missing libraries.")
        print(f"To enable visualization, install matplotlib and seaborn:")
        print("  conda install -c conda-forge matplotlib seaborn")
    
    # Calculate additional statistics
    print("\nCalculating additional statistics...")
    
    # Extract non-diagonal elements (similarity between different profiles)
    rows, cols = similarity_matrix.nonzero()
    non_diag_mask = rows != cols
    rows_nd, cols_nd = rows[non_diag_mask], cols[non_diag_mask]
    values_nd = similarity_matrix.data[non_diag_mask]
    
    print(f"Average similarity between different profiles: {values_nd.mean():.2f}")
    print(f"Median similarity between different profiles: {np.median(values_nd):.2f}")
    print(f"Max similarity between different profiles: {values_nd.max()}")
    
    # Find the most similar profile pairs
    top_indices = np.argsort(-values_nd)[:10]  # Top 10 most similar pairs
    print("\nTop 10 most similar profile pairs:")
    print("Profile 1, Profile 2, Common Variants")
    for idx in top_indices:
        i, j = rows_nd[idx], cols_nd[idx]
        common = similarity_matrix[i, j]
        print(f"{os.path.basename(files[i])}, {os.path.basename(files[j])}, {common}")
    
    # Calculate Jaccard similarity for a subset of profile pairs
    print("\nCalculating Jaccard similarity for top pairs...")
    for idx in top_indices[:5]:  # Top 5 most similar pairs
        i, j = rows_nd[idx], cols_nd[idx]
        common = similarity_matrix[i, j]
        variants_i = similarity_matrix[i, i]
        variants_j = similarity_matrix[j, j]
        jaccard = common / (variants_i + variants_j - common)
        print(f"Profiles {os.path.basename(files[i])} and {os.path.basename(files[j])}:")
        print(f"  Common variants: {common}")
        print(f"  Total variants in profile {os.path.basename(files[i])}: {variants_i}")
        print(f"  Total variants in profile {os.path.basename(files[j])}: {variants_j}")
        print(f"  Jaccard similarity: {jaccard:.4f}")
    
    print("\nAnalysis complete!")
    
def load_matrix_directly(matrix_path, files_path=None):
    """
    Load similarity matrix directly from file paths without going through the 
    variant_analysis module.
    
    Args:
        matrix_path: Path to the similarity matrix NPZ file
        files_path: Path to the files list pickle file (optional)
        
    Returns:
        tuple: (similarity_matrix, files)
    """
    print(f"Loading matrix directly from {matrix_path}...")
    similarity_matrix = sparse.load_npz(matrix_path)
    
    files = None
    if files_path and os.path.exists(files_path):
        print(f"Loading files from {files_path}")
        with open(files_path, 'rb') as f:
            files = pickle.load(f)
    else:
        # Create dummy file names
        n = similarity_matrix.shape[0]  # Number of profiles
        print(f"Files list not found, creating {n} dummy file names")
        files = [f"profile_{i+1}" for i in range(n)]
    
    return similarity_matrix, files

def analyze_matrix_from_paths(matrix_path, files_path=None, output_dir=None):
    """
    Analyze a similarity matrix directly from file paths.
    
    Args:
        matrix_path: Path to the similarity matrix NPZ file
        files_path: Path to the files list pickle file (optional)
        output_dir: Output directory for visualization output (default: ./outputs)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matrix and files
    similarity_matrix, files = load_matrix_directly(matrix_path, files_path)
    
    # Generate output file path for visualization
    output_file = os.path.join(output_dir, f'similarity_heatmap_custom.png')
    
    # The rest of the analysis is the same as analyze_similarity_matrix
    # Visualize the similarity matrix if visualization libraries are available
    if has_visualization:
        print("\nVisualizing profile similarity matrix...")
        visualize_profile_similarity(
            similarity_matrix=similarity_matrix,
            files=files,
            sample_size=50,  # Sample 50 profiles for visualization
            output_file=output_file
        )
    else:
        print("\nSkipping visualization due to missing libraries.")
        print(f"To enable visualization, install matplotlib and seaborn:")
        print("  conda install -c conda-forge matplotlib seaborn")
    
    # Calculate additional statistics
    print("\nCalculating additional statistics...")
    
    # Extract non-diagonal elements (similarity between different profiles)
    rows, cols = similarity_matrix.nonzero()
    non_diag_mask = rows != cols
    rows_nd, cols_nd = rows[non_diag_mask], cols[non_diag_mask]
    values_nd = similarity_matrix.data[non_diag_mask]
    
    print(f"Average similarity between different profiles: {values_nd.mean():.2f}")
    print(f"Median similarity between different profiles: {np.median(values_nd):.2f}")
    print(f"Max similarity between different profiles: {values_nd.max()}")
    
    # Find the most similar profile pairs
    top_indices = np.argsort(-values_nd)[:10]  # Top 10 most similar pairs
    print("\nTop 10 most similar profile pairs:")
    print("Profile 1, Profile 2, Common Variants")
    for idx in top_indices:
        i, j = rows_nd[idx], cols_nd[idx]
        common = similarity_matrix[i, j]
        print(f"{os.path.basename(files[i])}, {os.path.basename(files[j])}, {common}")
    
    # Calculate Jaccard similarity for a subset of profile pairs
    print("\nCalculating Jaccard similarity for top pairs...")
    for idx in top_indices[:5]:  # Top 5 most similar pairs
        i, j = rows_nd[idx], cols_nd[idx]
        common = similarity_matrix[i, j]
        variants_i = similarity_matrix[i, i]
        variants_j = similarity_matrix[j, j]
        jaccard = common / (variants_i + variants_j - common)
        print(f"Profiles {os.path.basename(files[i])} and {os.path.basename(files[j])}:")
        print(f"  Common variants: {common}")
        print(f"  Total variants in profile {os.path.basename(files[i])}: {variants_i}")
        print(f"  Total variants in profile {os.path.basename(files[j])}: {variants_j}")
        print(f"  Jaccard similarity: {jaccard:.4f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze profile similarity matrices")
    parser.add_argument("--model", type=str, default="pred1", 
                        help="Prediction model (default: pred1)")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs)")
    parser.add_argument("--matrix-file", type=str, default=None,
                        help="Direct path to a similarity matrix file (.npz)")
    parser.add_argument("--files-file", type=str, default=None,
                        help="Direct path to a files list file (.pkl)")
    parser.add_argument("--ext-path", type=str, default=None,
                        help="External path containing matrix files")
    
    args = parser.parse_args()
    
    # If matrix file path is provided, use direct file loading
    if args.matrix_file:
        analyze_matrix_from_paths(
            matrix_path=args.matrix_file,
            files_path=args.files_file,
            output_dir=args.output_dir
        )
    # If external path is provided, use it for loading
    elif args.ext_path:
        # Construct file paths from external path
        matrix_file = os.path.join(args.ext_path, f'profile_similarity_matrix_{args.model}_group_{args.group}.npz')
        files_file = os.path.join(args.ext_path, f'profile_similarity_files_{args.model}_group_{args.group}.pkl')
        
        # Check if files exist at external path
        if os.path.isfile(matrix_file):
            analyze_matrix_from_paths(
                matrix_path=matrix_file,
                files_path=files_file,
                output_dir=args.output_dir
            )
        else:
            # Try alternative matrix file name
            alt_matrix_file = os.path.join(args.ext_path, f'similarity_matrix_{args.model}_group_{args.group}.npz')
            if os.path.isfile(alt_matrix_file):
                analyze_matrix_from_paths(
                    matrix_path=alt_matrix_file,
                    files_path=files_file,
                    output_dir=args.output_dir
                )
            else:
                print(f"Error: Could not find matrix file at {args.ext_path}")
                print(f"Tried: {matrix_file}")
                print(f"Tried: {alt_matrix_file}")
                sys.exit(1)
    # Otherwise use the standard approach
    else:
        analyze_similarity_matrix(
            pred_model=args.model,
            group=args.group,
            output_dir=args.output_dir
        )
