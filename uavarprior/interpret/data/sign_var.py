"""
Filter variants based on a significance threshold and save the results.

This script reads parquet files containing variant effect data, filters variants 
based on an absolute threshold value, and saves the filtered results to new parquet files.
"""
import os
import pandas as pd
import time
import argparse
from typing import List, Tuple


def make_dir(dirname: str) -> bool:
    """
    Create a directory if it doesn't already exist.
    
    Args:
        dirname: Path of the directory to create
        
    Returns:
        bool: True if directory was created, False if it already existed
    """
    if os.path.exists(dirname):
        print(f'Already Exists: {dirname}')
        return False
    else:
        os.makedirs(dirname)
        print(f'Created: {dirname}')
        return True


def load_variant_labels(input_path: str) -> pd.DataFrame:
    """
    Load variant labels from a parquet file.
    
    Args:
        input_path: Path to the directory containing the row_labels.parquet.gzip file
        
    Returns:
        pd.DataFrame: DataFrame containing variant labels
    """
    start_time = time.time()
    file_path = os.path.join(input_path, 'row_labels.parquet.gzip')
    var_label = pd.read_parquet(file_path)
    var_label.reset_index(drop=True, inplace=True)
    print(f'Loaded variant labels, shape: {var_label.shape}')
    elapsed_time = time.time() - start_time
    print(f'Time taken: {elapsed_time/60:.2f} minutes')
    return var_label


def filter_variants(input_path: str, output_path: str, variant_labels: pd.DataFrame, 
                   file_list: List[str], threshold: float) -> None:
    """
    Filter variants based on significance threshold and save results.
    
    Args:
        input_path: Path to input directory
        output_path: Path to output directory
        variant_labels: DataFrame containing variant labels
        file_list: List of files to process
        threshold: Absolute threshold value for filtering
    """
    for file in file_list:
        start_time = time.time()
        file_path = os.path.join(input_path, file)
        
        try:
            data = pd.read_parquet(file_path)
            data.reset_index(drop=True, inplace=True)
            
            # Filter variants where absolute value exceeds threshold
            mask = data.iloc[:, 0].abs() > threshold
            filtered_var = variant_labels.loc[mask].copy()
            filtered_var['gve'] = data.loc[mask, data.columns[0]]
            filtered_var.reset_index(drop=True, inplace=True)
            
            # Save filtered results
            output_file = os.path.join(output_path, file)
            filtered_var.to_parquet(output_file, compression='gzip', index=False)
            
            elapsed_time = time.time() - start_time
            print(f'Processed {file} (shape: {data.shape}): {elapsed_time/60:.2f} minutes')
        except Exception as e:
            print(f'Error processing file {file}: {str(e)}')


def main():
    """Main function to run the variant filtering script."""
    parser = argparse.ArgumentParser(description='Filter variants based on significance threshold')
    parser.add_argument('--group', type=int, default=1, help='Group number')
    parser.add_argument('--threshold', type=float, default=0.10, help='Significance threshold')
    parser.add_argument('--base_path', type=str, 
                        default='/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr',
                        help='Base path for input/output directories')
    args = parser.parse_args()
    
    # Set up paths
    input_path = os.path.join(args.base_path, str(args.group))
    output_path = os.path.join(args.base_path, f'thr{args.threshold}', str(args.group))
    
    # Create output directory
    make_dir(output_path)
    
    # Get list of files to process
    all_files = os.listdir(input_path)
    files_to_process = [f for f in all_files if f != 'row_labels.parquet.gzip']
    
    # Load variant labels
    variant_labels = load_variant_labels(input_path)
    
    # Process files
    filter_variants(input_path, output_path, variant_labels, files_to_process, args.threshold)


if __name__ == "__main__":
    main()