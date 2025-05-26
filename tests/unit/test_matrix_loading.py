#!/usr/bin/env python3
"""
Simple script to test the membership matrix caching functionality.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import time
import importlib.util
import inspect

# Directly import the module without relying on package imports
# This helps avoid dependency issues with the entire package
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'uavarprior/interpret/variant_analysis.py')
spec = importlib.util.spec_from_file_location("variant_analysis", module_path)
variant_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variant_analysis)

# Get the run_full_analysis function from the module
run_full_analysis = variant_analysis.run_full_analysis

def test_matrix_caching():
    """
    Test the membership matrix caching functionality.
    """
    # Define output directory
    output_dir = os.path.join(os.getcwd(), 'test_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # First run: build matrix and save it
    print("\n===== FIRST RUN: BUILDING MEMBERSHIP MATRIX =====")
    start_time = time.time()
    run_full_analysis(
        group=1, 
        save_results=True, 
        output_dir=output_dir,
        model="pred1",  # Only test one model to save time
        use_cached_matrix=False  # Force rebuilding the matrix
    )
    first_run_time = time.time() - start_time
    print(f"\nFirst run time: {first_run_time:.2f} seconds")
    
    # Second run: load matrix from cache
    print("\n===== SECOND RUN: LOADING MEMBERSHIP MATRIX FROM CACHE =====")
    start_time = time.time()
    run_full_analysis(
        group=1, 
        save_results=True, 
        output_dir=output_dir,
        model="pred1",
        use_cached_matrix=True  # Use cached matrix
    )
    second_run_time = time.time() - start_time
    print(f"\nSecond run time: {second_run_time:.2f} seconds")
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"First run time (build matrix): {first_run_time:.2f} seconds")
    print(f"Second run time (load matrix): {second_run_time:.2f} seconds")
    print(f"Time savings: {first_run_time - second_run_time:.2f} seconds")
    print(f"Speed improvement: {first_run_time / second_run_time:.2f}x faster")

if __name__ == "__main__":
    try:
        # Check if we have the necessary modules
        print("Testing module imports...")
        import numpy as np
        import pandas as pd
        import scipy.sparse
        print("Basic scientific libraries imported successfully")
        
        # Check if the run_full_analysis function is available
        if not callable(run_full_analysis):
            print(f"ERROR: run_full_analysis is not callable: {type(run_full_analysis)}")
            sys.exit(1)
            
        # Run the test
        print("Starting test...")
        test_matrix_caching()
    except ImportError as e:
        print(f"\nERROR: Required module import failed: {e}")
        print("\nSolution: You may need to update your conda environment:")
        print("conda update -c conda-forge scikit-learn numpy scipy pandas")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
