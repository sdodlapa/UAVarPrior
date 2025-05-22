#!/usr/bin/env python3
"""
Test script for the variant profile counts functionality in variant_analysis.py
"""

import os
import numpy as np
from scipy import sparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import tempfile

from variant_analysis import calculate_variant_profile_counts

def test_calculate_variant_profile_counts():
    """
    Test the calculate_variant_profile_counts function with a simple test case
    """
    # Create a test matrix (3 variants x 4 files)
    # Variant 0 appears in files 0, 1, 2 (3 total)
    # Variant 1 appears in files 0, 3 (2 total)
    # Variant 2 appears in file 1 only (1 total)
    row_indices = [0, 0, 0, 1, 1, 2]
    col_indices = [0, 1, 2, 0, 3, 1]
    data_values = [1, 1, 1, 1, 1, 1]
    test_matrix = sparse.csr_matrix((data_values, (row_indices, col_indices)), shape=(3, 4))
    
    # Create test name_to_idx dict
    name_to_idx = {
        "variant_A": 0,
        "variant_B": 1,
        "variant_C": 2
    }
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the function
        df_counts = calculate_variant_profile_counts(
            matrix=test_matrix,
            name_to_idx=name_to_idx,
            pred_model="test_model",
            group=1,
            output_dir=temp_dir
        )
        
        # Check if the CSV and parquet files were created
        csv_path = os.path.join(temp_dir, "variant_profile_counts_test_model_group_1.csv")
        parquet_path = os.path.join(temp_dir, "variant_profile_counts_test_model_group_1.parquet")
        
        assert os.path.exists(csv_path), f"CSV file was not created at {csv_path}"
        assert os.path.exists(parquet_path), f"Parquet file was not created at {parquet_path}"
        
        # Check the contents of the DataFrame
        assert len(df_counts) == 3, f"Expected 3 rows in DataFrame, got {len(df_counts)}"
        
        # Map variant IDs to expected counts
        expected_counts = {"variant_A": 3, "variant_B": 2, "variant_C": 1}
        
        for _, row in df_counts.iterrows():
            variant_id = row["variant_id"]
            profile_count = row["profile_count"]
            expected_count = expected_counts[variant_id]
            
            assert profile_count == expected_count, \
                f"Expected count for {variant_id} to be {expected_count}, got {profile_count}"
        
        print("All tests passed successfully!")

if __name__ == "__main__":
    print("Testing calculate_variant_profile_counts function...")
    test_calculate_variant_profile_counts()
