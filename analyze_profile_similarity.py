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
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uavarprior.interpret.variant_analysis import (
    load_profile_similarity_matrix,
    visualize_profile_similarity
)

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
    
    # Visualize the similarity matrix
    print("\nVisualizing profile similarity matrix...")
    output_file = os.path.join(output_dir, f'similarity_heatmap_{pred_model}_group_{group}.png')
    visualize_profile_similarity(
        similarity_matrix=similarity_matrix,
        files=files,
        sample_size=50,  # Sample 50 profiles for visualization
        output_file=output_file
    )
    
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
    parser.add_argument("--model", type=str, default="pred1", choices=["pred1", "pred150"],
                        help="Prediction model (default: pred1)")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs)")
    
    args = parser.parse_args()
    
    analyze_similarity_matrix(
        pred_model=args.model,
        group=args.group,
        output_dir=args.output_dir
    )
