#!/usr/bin/env python3
"""
Standalone test script for membership and similarity matrix functionality.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import time
import pickle
import numpy as np
from scipy import sparse
import glob
import argparse

def check_matrix_exists(pred_model, group, output_dir, matrix_type="membership"):
    """Check if matrix files exist in the output directory."""
    if matrix_type == "membership":
        matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
        files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
        name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
        
        return all(os.path.isfile(f) for f in [matrix_file, files_file, name_to_idx_file])
    elif matrix_type == "similarity":
        # Try different possible file patterns for similarity matrix
        patterns = [
            os.path.join(output_dir, f'profile_similarity_matrix_{pred_model}_group_{group}.npz'),
            os.path.join(output_dir, f'similarity_matrix_{pred_model}_group_{group}.npz')
        ]
        return any(os.path.isfile(p) for p in patterns)

def load_matrix(pred_model, group, output_dir, matrix_type="membership"):
    """Load the matrix and associated data from files."""
    if matrix_type == "membership":
        matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
        files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
        name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
        
        matrix = sparse.load_npz(matrix_file)
        
        with open(files_file, 'rb') as f:
            files = pickle.load(f)
        
        with open(name_to_idx_file, 'rb') as f:
            name_to_idx = pickle.load(f)
        
        print(f"Loaded membership matrix with shape {matrix.shape}")
        print(f"Number of files: {len(files)}")
        print(f"Number of variants: {len(name_to_idx)}")
        
        return matrix, files, name_to_idx
    
    elif matrix_type == "similarity":
        # Try different possible file patterns for similarity matrix
        possible_matrix_files = [
            os.path.join(output_dir, f'profile_similarity_matrix_{pred_model}_group_{group}.npz'),
            os.path.join(output_dir, f'similarity_matrix_{pred_model}_group_{group}.npz')
        ]
        
        matrix_file = None
        for f in possible_matrix_files:
            if os.path.isfile(f):
                matrix_file = f
                break
        
        if matrix_file is None:
            # Try to find any similarity matrix in the directory
            all_npz = glob.glob(os.path.join(output_dir, "*similarity*matrix*.npz"))
            if all_npz:
                matrix_file = all_npz[0]
                print(f"Found similarity matrix: {matrix_file}")
            else:
                raise FileNotFoundError("No similarity matrix file found")
        
        # Try to load the matrix
        matrix = sparse.load_npz(matrix_file)
        
        # Try to find associated files list
        possible_files_files = [
            os.path.join(output_dir, f'profile_similarity_files_{pred_model}_group_{group}.pkl'),
            os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
        ]
        
        files_file = None
        for f in possible_files_files:
            if os.path.isfile(f):
                files_file = f
                break
        
        files = None
        if files_file:
            try:
                with open(files_file, 'rb') as f:
                    files = pickle.load(f)
                print(f"Loaded {len(files)} file names")
            except Exception as e:
                print(f"Error loading files list: {e}")
        
        print(f"Loaded similarity matrix with shape {matrix.shape}")
        print(f"Number of non-zero elements: {matrix.count_nonzero()}")
        
        return matrix, files

def analyze_similarity_matrix(matrix, files=None):
    """Analyze a similarity matrix."""
    print(f"\nSimilarity matrix shape: {matrix.shape} (profiles × profiles)")
    print(f"Number of non-zero elements: {matrix.count_nonzero()}")
    nonzero_percentage = 100 * matrix.count_nonzero() / (matrix.shape[0] * matrix.shape[1])
    print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
    
    # Calculate basic statistics on the diagonal (variants per profile)
    diag_values = matrix.diagonal()
    print("\nStatistics of diagonal elements (variants per profile):")
    print(f"  Min: {diag_values.min()}")
    print(f"  Max: {diag_values.max()}")
    print(f"  Mean: {diag_values.mean():.2f}")
    print(f"  Median: {np.median(diag_values):.2f}")
    
    # Extract non-diagonal elements (similarity between different profiles)
    rows, cols = matrix.nonzero()
    non_diag_mask = rows != cols
    rows_nd, cols_nd = rows[non_diag_mask], cols[non_diag_mask]
    values_nd = np.array([matrix[i, j] for i, j in zip(rows_nd, cols_nd)])
    
    print(f"\nAverage similarity between different profiles: {np.mean(values_nd):.2f}")
    print(f"Median similarity between different profiles: {np.median(values_nd):.2f}")
    print(f"Max similarity between different profiles: {np.max(values_nd)}")
    
    # Find the most similar profile pairs
    top_indices = np.argsort(-values_nd)[:10]  # Top 10 most similar pairs
    print("\nTop 10 most similar profile pairs:")
    print("Profile 1, Profile 2, Common Variants")
    for idx in top_indices:
        i, j = rows_nd[idx], cols_nd[idx]
        common = matrix[i, j]
        
        if files is not None:
            print(f"{os.path.basename(files[i])}, {os.path.basename(files[j])}, {common}")
        else:
            print(f"Profile {i}, Profile {j}, {common}")
    
    # Calculate Jaccard similarity for a subset of profile pairs
    print("\nCalculating Jaccard similarity for top pairs...")
    for idx in top_indices[:5]:  # Top 5 most similar pairs
        i, j = rows_nd[idx], cols_nd[idx]
        common = matrix[i, j]
        variants_i = matrix[i, i]
        variants_j = matrix[j, j]
        jaccard = common / (variants_i + variants_j - common)
        
        if files is not None:
            print(f"Profiles {os.path.basename(files[i])} and {os.path.basename(files[j])}:")
        else:
            print(f"Profiles {i} and {j}:")
            
        print(f"  Common variants: {common}")
        print(f"  Total variants in profile {i if files is None else os.path.basename(files[i])}: {variants_i}")
        print(f"  Total variants in profile {j if files is None else os.path.basename(files[j])}: {variants_j}")
        print(f"  Jaccard similarity: {jaccard:.4f}")

def standalone_test(matrix_type="both", pred_model="pred1", group=1, output_dir=None):
    """Test loading a matrix from disk and analyze it."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    success = False
    
    if matrix_type in ["membership", "both"]:
        print(f"Checking if membership matrix exists in {output_dir}...")
        if check_matrix_exists(pred_model, group, output_dir, "membership"):
            print("Membership matrix files found! Loading matrix...")
            start_time = time.time()
            matrix, files, name_to_idx = load_matrix(pred_model, group, output_dir, "membership")
            load_time = time.time() - start_time
            
            print(f"Matrix loaded in {load_time:.2f} seconds")
            print(f"Matrix has {matrix.count_nonzero()} non-zero elements")
            print(f"Memory usage: {matrix.data.nbytes / 1024**2:.2f} MB (data)")
            
            # Calculate some basic statistics
            # Count variants per file (column sums)
            file_variant_counts = matrix.sum(axis=0).A1
            print(f"Average variants per file: {file_variant_counts.mean():.2f}")
            print(f"Max variants in a file: {file_variant_counts.max()}")
            print(f"Min variants in a file: {file_variant_counts.min()}")
            
            # Count how many files each variant appears in (row sums)
            variant_file_counts = matrix.sum(axis=1).A1
            print(f"Cell-specific variants (appear in exactly 1 file): {sum(variant_file_counts == 1)}")
            print(f"Cell-nonspecific variants (appear in at least 80 files): {sum(variant_file_counts >= 80)}")
            
            success = True
        else:
            print("Membership matrix files not found in the output directory.")
    
    if matrix_type in ["similarity", "both"]:
        print(f"Checking if similarity matrix exists in {output_dir}...")
        if check_matrix_exists(pred_model, group, output_dir, "similarity"):
            print("Similarity matrix file found! Loading matrix...")
            try:
                start_time = time.time()
                result = load_matrix(pred_model, group, output_dir, "similarity")
                load_time = time.time() - start_time
                
                if len(result) == 2:
                    sim_matrix, files = result
                else:
                    sim_matrix = result[0]
                    files = None
                
                print(f"Similarity matrix loaded in {load_time:.2f} seconds")
                
                # Analyze the similarity matrix
                analyze_similarity_matrix(sim_matrix, files)
                
                success = True
            except Exception as e:
                print(f"Error loading similarity matrix: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Similarity matrix files not found in the output directory.")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone test for loading and analyzing matrices")
    parser.add_argument("--model", type=str, default="pred1", choices=["pred1", "pred150"],
                        help="Prediction model (default: pred1)")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./outputs)")
    parser.add_argument("--matrix-type", type=str, default="both", choices=["membership", "similarity", "both"],
                        help="Type of matrix to analyze (default: both)")
    
    args = parser.parse_args()
    
    try:
        success = standalone_test(
            matrix_type=args.matrix_type,
            pred_model=args.model,
            group=args.group,
            output_dir=args.output_dir
        )
        
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed - requested matrix files not found.")
            print("You need to run the variant_analysis.py script first to generate the matrix files.")
            
            # List available matrix files in the output directory
            if args.output_dir is None:
                output_dir = os.path.join(os.getcwd(), 'outputs')
            else:
                output_dir = args.output_dir
                
            if os.path.isdir(output_dir):
                print("\nAvailable matrix files in the output directory:")
                all_npz = glob.glob(os.path.join(output_dir, "*.npz"))
                for f in all_npz:
                    print(f"  {os.path.basename(f)}")
                
                print("\nTry running with --matrix-type=similarity")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
