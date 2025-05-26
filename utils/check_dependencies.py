#!/usr/bin/env python3
"""
Direct script to check for a similarity matrix in the outputs directory.
This script uses minimal imports to avoid dependency issues.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import glob

def check_for_required_modules():
    """Check if the required modules are available."""
    missing_modules = []
    
    try:
        import numpy
    except ImportError:
        missing_modules.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing_modules.append("scipy")
    
    if missing_modules:
        print("Missing required modules:", ", ".join(missing_modules))
        print("Please install them using:")
        print(f"  pip install {' '.join(missing_modules)}")
        return False
    
    print("All required modules are available!")
    return True

def find_matrix_files(output_dir=None):
    """Find similarity matrix files in the output directory."""
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs")
    
    # Check if outputs directory exists
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory {output_dir} does not exist")
        return []
    
    # Look for matrix files with "similarity" in the name
    matrix_files = glob.glob(os.path.join(output_dir, "*similarity*matrix*.npz"))
    
    if not matrix_files:
        # Try a more general search
        matrix_files = glob.glob(os.path.join(output_dir, "*.npz"))
    
    return matrix_files

if __name__ == "__main__":
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Check for similarity matrix files")
        parser.add_argument("--output-dir", type=str, default=None,
                          help="Output directory to search in (default: ./outputs)")
        
        args = parser.parse_args()
        
        print("Checking for required Python modules...")
        # Check if required modules are installed
        has_modules = check_for_required_modules()
        
        print("\nSearching for matrix files...")
        # Find matrix files
        matrix_files = find_matrix_files(args.output_dir)
        
        if matrix_files:
            print("\nFound the following matrix files:")
            for i, f in enumerate(matrix_files, 1):
                print(f"  {i}. {os.path.basename(f)}")
            
            if has_modules:
                print("\nYou have all required modules installed!")
                print("Run one of these commands to analyze the matrix:")
                print(f"  python standalone_test.py --matrix-file=\"{matrix_files[0]}\"")
                print("  OR")
                print("  ./run_profile_analysis.csh")
        else:
            print("\nNo matrix files found.")
            print("You need to run the variant analysis script first to generate the matrix files.")
        
        print("\nDone!")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
