#!/usr/bin/env python3
"""
Standalone test script for membership matrix functionality.
"""

import os
import sys
import time
import pickle
import numpy as np
from scipy import sparse

def check_matrix_exists(pred_model, group, output_dir):
    """Check if matrix files exist in the output directory."""
    matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
    files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
    
    return all(os.path.isfile(f) for f in [matrix_file, files_file, name_to_idx_file])

def load_matrix(pred_model, group, output_dir):
    """Load the membership matrix and associated data from files."""
    matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
    files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
    
    matrix = sparse.load_npz(matrix_file)
    
    with open(files_file, 'rb') as f:
        files = pickle.load(f)
    
    with open(name_to_idx_file, 'rb') as f:
        name_to_idx = pickle.load(f)
    
    print(f"Loaded matrix with shape {matrix.shape}")
    print(f"Number of files: {len(files)}")
    print(f"Number of variants: {len(name_to_idx)}")
    
    return matrix, files, name_to_idx

def standalone_test():
    """Test loading a matrix from disk."""
    output_dir = os.path.join(os.getcwd(), 'outputs')
    pred_model = 'pred1'
    group = 1
    
    print(f"Checking if matrix exists in {output_dir}...")
    if check_matrix_exists(pred_model, group, output_dir):
        print("Matrix files found! Loading matrix...")
        start_time = time.time()
        matrix, files, name_to_idx = load_matrix(pred_model, group, output_dir)
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
        
        return True
    else:
        print("Matrix files not found in the output directory.")
        return False

if __name__ == "__main__":
    try:
        success = standalone_test()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed - matrix files not found.")
            print("You need to run the variant_analysis.py script first to generate the matrix files.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
