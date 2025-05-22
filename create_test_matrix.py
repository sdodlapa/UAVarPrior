#!/usr/bin/env python3
"""
Script to create a test similarity matrix for analyzing profile similarities.
This script avoids importing the full UAVarPrior package and creates a minimal
test matrix that can be used for testing the analysis scripts.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import numpy as np
from scipy import sparse
import pickle
import argparse
import datetime

def create_test_matrix(n_profiles=100, n_variants=500, density=0.05, output_dir=None):
    """
    Create a synthetic test membership matrix and similarity matrix for testing.
    
    Args:
        n_profiles: Number of profiles (cells)
        n_variants: Number of variants
        density: Density of non-zero elements in the membership matrix
        output_dir: Directory to save the output files
    """
    print(f"Creating test matrices with {n_profiles} profiles and {n_variants} variants...")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fake file names
    files = [f"cell_{i+1}.vcf" for i in range(n_profiles)]
    
    # Generate fake variant names
    variant_names = [f"variant_{i+1}" for i in range(n_variants)]
    
    # Create a mapping from variant names to indices
    name_to_idx = {name: i for i, name in enumerate(variant_names)}
    
    # Create a random membership matrix (variants × profiles)
    # Each element represents whether a variant is present in a profile
    random_data = np.random.random((n_variants, n_profiles))
    membership_matrix = sparse.csr_matrix(random_data < density)
    
    print(f"Created membership matrix with shape {membership_matrix.shape}")
    print(f"Number of non-zero elements: {membership_matrix.count_nonzero()}")
    nonzero_percentage = 100 * membership_matrix.count_nonzero() / (membership_matrix.shape[0] * membership_matrix.shape[1])
    print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
    
    # Calculate profile similarity matrix (profiles × profiles)
    # This represents how many variants are shared between each pair of profiles
    print("\nCalculating profile similarity matrix...")
    similarity_matrix = membership_matrix.T.dot(membership_matrix)
    
    print(f"Created similarity matrix with shape {similarity_matrix.shape}")
    print(f"Number of non-zero elements: {similarity_matrix.count_nonzero()}")
    nonzero_percentage = 100 * similarity_matrix.count_nonzero() / (similarity_matrix.shape[0] * similarity_matrix.shape[1])
    print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
    
    # Save the matrices and associated data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save membership matrix
    membership_file = os.path.join(output_dir, f'membership_matrix_test_{timestamp}.npz')
    sparse.save_npz(membership_file, membership_matrix)
    print(f"\nSaved membership matrix to: {membership_file}")
    
    # Save similarity matrix in two formats for compatibility
    similarity_file1 = os.path.join(output_dir, f'profile_similarity_matrix_test_group_1.npz')
    similarity_file2 = os.path.join(output_dir, f'similarity_matrix_test_group_1.npz')
    sparse.save_npz(similarity_file1, similarity_matrix)
    sparse.save_npz(similarity_file2, similarity_matrix)
    print(f"Saved similarity matrix to: {similarity_file1}")
    print(f"Saved similarity matrix to: {similarity_file2}")
    
    # Save file names
    files_file = os.path.join(output_dir, f'profile_similarity_files_test_group_1.pkl')
    with open(files_file, 'wb') as f:
        pickle.dump(files, f)
    print(f"Saved file names to: {files_file}")
    
    # Save name-to-index mapping
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_test_group_1.pkl')
    with open(name_to_idx_file, 'wb') as f:
        pickle.dump(name_to_idx, f)
    print(f"Saved name-to-index mapping to: {name_to_idx_file}")
    
    print("\nTest data generation complete!")
    return similarity_matrix, files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test matrices for profile similarity analysis")
    parser.add_argument("--profiles", type=int, default=100,
                        help="Number of profiles/cells (default: 100)")
    parser.add_argument("--variants", type=int, default=500,
                        help="Number of variants (default: 500)")
    parser.add_argument("--density", type=float, default=0.05,
                        help="Density of non-zero elements (default: 0.05)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs)")
    
    args = parser.parse_args()
    
    try:
        create_test_matrix(
            n_profiles=args.profiles,
            n_variants=args.variants,
            density=args.density,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
