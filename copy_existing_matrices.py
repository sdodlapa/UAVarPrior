#!/usr/bin/env python3
"""
Script to copy existing similarity matrix files from the specified source location
to the local outputs directory.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import sys
import shutil
import glob
import argparse

def copy_matrix_files(source_dir, dest_dir=None, model="pred1", group=1):
    """
    Copy similarity matrix files from source_dir to dest_dir.
    
    Args:
        source_dir: Source directory containing the matrix files
        dest_dir: Destination directory (default: ./outputs)
        model: Model name (default: pred1)
        group: Group number (default: 1)
    """
    if dest_dir is None:
        dest_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define source file patterns
    source_patterns = [
        f"profile_similarity_matrix_{model}_group_{group}.npz",
        f"profile_similarity_files_{model}_group_{group}.pkl",
        f"similarity_matrix_{model}_group_{group}.npz",
        f"files_list_{model}_group_{group}.pkl",
        f"*similarity*matrix*{model}*group*{group}*.npz",
        f"*similarity*files*{model}*group*{group}*.pkl"
    ]
    
    # Find all matching files
    source_files = []
    for pattern in source_patterns:
        matches = glob.glob(os.path.join(source_dir, pattern))
        source_files.extend(matches)
    
    if not source_files:
        print(f"No similarity matrix files found in {source_dir}")
        return False
    
    # Copy each file to the destination directory
    print(f"Copying files to {dest_dir}...")
    copied_files = []
    
    for source_file in source_files:
        filename = os.path.basename(source_file)
        dest_file = os.path.join(dest_dir, filename)
        
        try:
            shutil.copy2(source_file, dest_file)
            copied_files.append(dest_file)
            print(f"Copied: {filename}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    if copied_files:
        print(f"\nSuccessfully copied {len(copied_files)} files.")
        print("\nFiles are now ready for analysis. You can run:")
        print("  python analyze_profile_similarity.py")
        print("  OR")
        print("  ./run_profile_analysis.csh")
        return True
    else:
        print("Failed to copy any files.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy similarity matrix files to local outputs directory")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Source directory containing the matrix files")
    parser.add_argument("--dest-dir", type=str, default=None,
                        help="Destination directory (default: ./outputs)")
    parser.add_argument("--model", type=str, default="pred1",
                        help="Model name (default: pred1)")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    
    args = parser.parse_args()
    
    try:
        success = copy_matrix_files(
            source_dir=args.source_dir,
            dest_dir=args.dest_dir,
            model=args.model,
            group=args.group
        )
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
