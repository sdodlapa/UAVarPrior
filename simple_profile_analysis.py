#!/usr/bin/env python3
"""
Simple script to analyze profile similarity matrix without visualization dependencies.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import numpy as np
import pickle
from scipy import sparse

def load_similarity_matrix(matrix_file, files_file):
    """
    Load similarity matrix and file names.
    
    Args:
        matrix_file: Path to the NPZ file with the similarity matrix
        files_file: Path to the PKL file with the file names
        
    Returns:
        tuple: (similarity_matrix, files)
    """
    print(f"Loading matrix from {matrix_file}...")
    similarity_matrix = sparse.load_npz(matrix_file)
    
    print(f"Loading file names from {files_file}...")
    with open(files_file, 'rb') as f:
        files = pickle.load(f)
    
    print(f"Loaded matrix with shape {similarity_matrix.shape}")
    print(f"Number of non-zero elements: {similarity_matrix.count_nonzero()}")
    nonzero_percentage = 100 * similarity_matrix.count_nonzero() / (similarity_matrix.shape[0] * similarity_matrix.shape[1])
    print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
    
    return similarity_matrix, files

def analyze_similarity(similarity_matrix, files):
    """
    Analyze the similarity matrix and print statistics.
    
    Args:
        similarity_matrix: The sparse similarity matrix
        files: List of file names
    """
    # Calculate basic statistics on the diagonal (variants per profile)
    diag_values = similarity_matrix.diagonal()
    print("\nStatistics of diagonal elements (variants per profile):")
    print(f"  Min: {diag_values.min()}")
    print(f"  Max: {diag_values.max()}")
    print(f"  Mean: {diag_values.mean():.2f}")
    print(f"  Median: {np.median(diag_values):.2f}")
    
    # Extract non-diagonal elements (similarity between different profiles)
    rows, cols = similarity_matrix.nonzero()
    non_diag_mask = rows != cols
    rows_nd, cols_nd = rows[non_diag_mask], cols[non_diag_mask]
    values_nd = np.array([similarity_matrix[i, j] for i, j in zip(rows_nd, cols_nd)])
    
    print(f"\nAverage similarity between different profiles: {np.mean(values_nd):.2f}")
    print(f"Median similarity between different profiles: {np.median(values_nd):.2f}")
    print(f"Max similarity between different profiles: {np.max(values_nd)}")
    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze profile similarity matrices without visualization")
    parser.add_argument("--model", type=str, default="pred1", choices=["pred1", "pred150"],
                        help="Prediction model (default: pred1)")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs)")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    else:
        output_dir = args.output_dir
    
    # Define file paths
    matrix_file = os.path.join(output_dir, f'profile_similarity_matrix_{args.model}_group_{args.group}.npz')
    files_file = os.path.join(output_dir, f'profile_similarity_files_{args.model}_group_{args.group}.pkl')
    
    # Check for alternative files format if the original files aren't found
    if not os.path.isfile(matrix_file):
        alt_matrix_file = os.path.join(output_dir, f'similarity_matrix_{args.model}_group_{args.group}.npz')
        if os.path.isfile(alt_matrix_file):
            print(f"Found alternative matrix file format: {alt_matrix_file}")
            matrix_file = alt_matrix_file
    
    # Check if files exist
    if not os.path.isfile(matrix_file) or not os.path.isfile(files_file):
        print(f"Error: Required files not found:")
        if not os.path.isfile(matrix_file):
            print(f"  Missing: {matrix_file}")
        if not os.path.isfile(files_file):
            print(f"  Missing: {files_file}")
        print("\nYou need to run the variant analysis first to generate these files.")
        print("Example:")
        print("  python -m uavarprior.interpret.variant_analysis --model pred1 --group 1")
        
        # Try to look for other relevant files in the output directory
        print("\nChecking for any similarity matrix files in the output directory...")
        import glob
        matrix_files = glob.glob(os.path.join(output_dir, "*similarity*matrix*.npz"))
        if matrix_files:
            print("Found potential similarity matrix files:")
            for f in matrix_files:
                print(f"  {f}")
            print("\nTry using one of these files instead.")
        
        sys.exit(1)
    
    # Load and analyze the similarity matrix
    similarity_matrix, files = load_similarity_matrix(matrix_file, files_file)
    analyze_similarity(similarity_matrix, files)
