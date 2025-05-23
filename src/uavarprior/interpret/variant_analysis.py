#!/usr/bin/env python3
"""
Variant Analysis Module

This module provides functionality for analyzing genetic variants across
different cell types and prediction models. It helps identify cell-specific
and cell-nonspecific variants and analyze their minor allele frequency (MAF)
distribution.

Created from the signVar_by_profile.ipynb notebook.

Created by: Sanjeeva Reddy Dodlapati
Date: May 22, 2025
"""

import os
import re
import pandas as pd
import pickle
import numpy as np
from scipy import sparse
import time
from typing import Dict, List, Set, Tuple, Union, Optional
from collections import Counter
import datetime


# MAF extraction and classification
def extract_maf_regex(info_str: str) -> Optional[float]:
    """
    Extract Minor Allele Frequency (MAF) from a variant info string using regex.
    
    Args:
        info_str: String containing variant information
        
    Returns:
        float: MAF value if found, None otherwise
    """
    _maf_re = re.compile(r'\bMAF=([^;]+)')
    m = _maf_re.search(info_str)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def classify_maf(maf: float) -> str:
    """
    Classify variant by Minor Allele Frequency (MAF).
    
    Args:
        maf: Minor Allele Frequency value
        
    Returns:
        str: 'rare' if MAF < 0.001, 'common' if MAF > 0.05, 'undefined' otherwise
    """
    if maf < 0.001:
        return 'rare'
    elif maf > 0.05:
        return 'common'
    else:
        return 'undefined'


def load_maf_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare MAF data from parquet file.
    
    Args:
        file_path: Path to the MAF parquet file
        
    Returns:
        pd.DataFrame: DataFrame with MAF data and categories
    """
    df_maf = pd.read_parquet(file_path)
    df_maf['category'] = df_maf['maf'].apply(classify_maf)
    df_maf.dropna(inplace=True, ignore_index=True)
    return df_maf


# Variant collection functions
def collect_unique_variants(directory: str, verbose: bool = True) -> Set[str]:
    """
    Collect unique variant names from parquet files in a directory.
    
    Args:
        directory: Path to directory containing variant parquet files
        verbose: If True, print progress information
        
    Returns:
        Set[str]: Set of unique variant names
    """
    files = os.listdir(directory)
    files.sort()
    
    # Initialize an empty set to store unique name values
    unique_names = set()
    
    # Track progress
    total_files = len(files)
    processed = 0
    
    # Process files
    for file in files:
        try:
            # Only load the 'name' column to minimize memory usage
            data = pd.read_parquet(os.path.join(directory, file), columns=['name'])
            
            # Update the set with new unique values
            unique_names.update(data['name'])
            
            # Update and display progress
            processed += 1
            if verbose and (processed % 10 == 0 or processed == total_files):
                print(f"Processed {processed}/{total_files} files. Current unique names: {len(unique_names)}")
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if verbose:
        print(f"\nTotal unique names collected: {len(unique_names)}")
        memory_usage = sum(len(name) for name in unique_names) / (1024*1024)
        print(f"Memory used by unique_names set: {memory_usage:.2f} MB")
        
    return unique_names


def save_unique_names(unique_names: Set[str], output_file: str) -> None:
    """
    Save unique variant names to a pickle file.
    
    Args:
        unique_names: Set of unique variant names
        output_file: Path to save the pickle file
        
    Returns:
        None
    """
    with open(output_file, 'wb') as f:
        pickle.dump(unique_names, f)
    print(f"Saved {len(unique_names)} unique names to: {output_file}")


def load_unique_names(input_file: str, verbose: bool = True) -> Set[str]:
    """
    Load unique variant names from a pickle file.
    
    Args:
        input_file: Path to the pickle file
        verbose: If True, print information about loaded data
        
    Returns:
        Set[str]: Set of unique variant names
    """
    if os.path.isfile(input_file):
        with open(input_file, 'rb') as f:
            unique_names = pickle.load(f)
        
        if verbose:
            print(f"Loaded {len(unique_names)} unique names from {input_file}")
            memory_usage = sum(len(name) for name in unique_names) / (1024*1024)
            print(f"Memory used by unique_names set: {memory_usage:.2f} MB")
            
            # Preview some unique names
            print("\nSample names:")
            print(list(unique_names)[:5])
            
        return unique_names
    else:
        print(f"File {input_file} not found.")
        return set()


# Binary membership matrix construction
def build_membership_matrix(variant_names: Set[str], directory: str, verbose: bool = True) -> Tuple[sparse.csr_matrix, List[str], Dict[str, int]]:
    """
    Build a binary membership matrix indicating which variants appear in which files.
    
    Args:
        variant_names: Set of unique variant names
        directory: Directory containing variant files
        verbose: If True, print progress information
    
    Returns:
        Tuple containing:
        - sparse.csr_matrix: Binary membership matrix (variants × files)
        - List[str]: List of file names
        - Dict[str, int]: Mapping of variant names to row indices
    """
    files = os.listdir(directory)
    files.sort()
    
    # Create a mapping of variant names to row indices
    start_time = time.time()
    name_to_idx = {name: idx for idx, name in enumerate(variant_names)}
    
    if verbose:
        print(f"Created name to index mapping in {time.time() - start_time:.2f} seconds")
    
    # Initialize lists to store the sparse matrix coordinates and values
    row_indices = []
    col_indices = []
    data_values = []
    
    # Process files to build the membership matrix
    start_time = time.time()
    total_files = len(files)
    processed = 0
    
    for file_idx, file in enumerate(files):
        try:
            # Only load the 'name' column
            file_data = pd.read_parquet(os.path.join(directory, file), columns=['name'])
            
            # Get unique names in this file (we only need each name once per file)
            file_names = set(file_data['name'])
            
            # For each name in this file, add a 1 to the matrix
            for name in file_names:
                if name in name_to_idx:  # This should always be true but checking to be safe
                    row_indices.append(name_to_idx[name])
                    col_indices.append(file_idx)
                    data_values.append(1)
            
            # Update progress
            processed += 1
            if verbose and (processed % 10 == 0 or processed == total_files):
                print(f"Processed {processed}/{total_files} files for matrix construction")
                print(f"Current non-zero elements: {len(data_values)}")
                
        except Exception as e:
            print(f"Error processing {file} for matrix: {str(e)}")
    
    # Create a sparse matrix in CSR format (efficient for row operations)
    num_variants = len(variant_names)
    num_files = len(files)
    
    membership_matrix = sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(num_variants, num_files)
    )
    
    if verbose:
        print(f"\nMatrix shape: {membership_matrix.shape} (variants × files)")
        print(f"Number of non-zero elements: {membership_matrix.count_nonzero()} (variant occurrences)")
        sparsity = 100 - 100 * membership_matrix.count_nonzero() / (num_variants * num_files)
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Memory usage: {membership_matrix.data.nbytes / 1024**2:.2f} MB (data)")
        print(f"Total construction time: {time.time() - start_time:.2f} seconds")
    
    return membership_matrix, files, name_to_idx


def save_membership_matrix(
    matrix: sparse.csr_matrix,
    files: List[str],
    name_to_idx: Dict[str, int],
    pred_model: str,
    group: int,
    output_dir: str = None
) -> None:
    """
    Save the membership matrix and associated data to files.
    
    Args:
        matrix: Binary membership matrix (variants × files)
        files: List of file names
        name_to_idx: Mapping of variant names to row indices
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory for saving results
        
    Returns:
        None
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sparse matrix
    matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
    sparse.save_npz(matrix_file, matrix)
    
    # Save the files list
    files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
    with open(files_file, 'wb') as f:
        pickle.dump(files, f)
    
    # Save the name to index mapping
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
    with open(name_to_idx_file, 'wb') as f:
        pickle.dump(name_to_idx, f)
    
    print(f"Saved membership matrix and associated data to {output_dir}")


def load_membership_matrix(
    pred_model: str,
    group: int,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, List[str], Dict[str, int]]:
    """
    Load the membership matrix and associated data from files.
    
    Args:
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory containing saved files
        verbose: If True, print information about loaded data
        
    Returns:
        Tuple containing:
        - sparse.csr_matrix: Binary membership matrix (variants × files)
        - List[str]: List of file names
        - Dict[str, int]: Mapping of variant names to row indices
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Load the sparse matrix
    matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
    matrix = sparse.load_npz(matrix_file)
    
    # Load the files list
    files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
    with open(files_file, 'rb') as f:
        files = pickle.load(f)
    
    # Load the name to index mapping
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
    with open(name_to_idx_file, 'rb') as f:
        name_to_idx = pickle.load(f)
    
    if verbose:
        print(f"\nLoaded membership matrix with shape {matrix.shape} (variants × files)")
        print(f"Number of non-zero elements: {matrix.count_nonzero()} (variant occurrences)")
        sparsity = 100 - 100 * matrix.count_nonzero() / (matrix.shape[0] * matrix.shape[1])
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Memory usage: {matrix.data.nbytes / 1024**2:.2f} MB (data)")
        print(f"Loaded {len(files)} files and {len(name_to_idx)} variant names")
    
    return matrix, files, name_to_idx


def check_membership_matrix_exists(
    pred_model: str,
    group: int,
    output_dir: str = None
) -> bool:
    """
    Check if a previously computed membership matrix exists in the output directory.
    
    Args:
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory to check
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Check if all required files exist
    matrix_file = os.path.join(output_dir, f'membership_matrix_{pred_model}_group_{group}.npz')
    files_file = os.path.join(output_dir, f'files_list_{pred_model}_group_{group}.pkl')
    name_to_idx_file = os.path.join(output_dir, f'name_to_idx_{pred_model}_group_{group}.pkl')
    
    return os.path.isfile(matrix_file) and os.path.isfile(files_file) and os.path.isfile(name_to_idx_file)


# Analysis functions
def analyze_membership_matrix(
    matrix: sparse.csr_matrix, 
    files: List[str], 
    name_to_idx: Dict[str, int],
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Analyze the membership matrix to find cell-specific and cell-nonspecific variants.
    
    Args:
        matrix: Binary membership matrix (variants × files)
        files: List of file names
        name_to_idx: Mapping of variant names to row indices
        verbose: If True, print analysis results
    
    Returns:
        Tuple containing:
        - List[str]: Names of cell-specific variants
        - List[str]: Names of cell-nonspecific variants
    """
    # Count variants per file (column sums)
    file_variant_counts = matrix.sum(axis=0).A1  # A1 converts to 1D array
    
    # Count how many files each variant appears in (row sums)
    variant_file_counts = matrix.sum(axis=1).A1
    
    # Show distribution of variant occurrence
    occurrence_dist = Counter(variant_file_counts)
    
    if verbose:
        # Show files with most and least variants
        print(f"Average variants per file: {file_variant_counts.mean():.2f}")
        max_idx = file_variant_counts.argmax()
        min_idx = file_variant_counts.argmin()
        print(f"Max variants in a file: {file_variant_counts.max()} (file #{max_idx}: {files[max_idx]})")
        print(f"Min variants in a file: {file_variant_counts.min()} (file #{min_idx}: {files[min_idx]})")
        
        # Print variant occurrence distribution
        print("\nVariant occurrence distribution:")
        for count, num_variants in sorted(occurrence_dist.items())[:10]:
            print(f"{num_variants} variants appear in exactly {int(count)} files")
    
    # Find cell-specific variants (appear in exactly 1 file)
    cell_specific_variants = np.where(variant_file_counts == 1)[0]
    
    # Find cell-nonspecific variants (appear in at least 80 files)
    cell_nonspecific_variants = np.where(variant_file_counts >= 80)[0]
    
    # Convert indices to variant names
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    cell_specific_variant_names = [idx_to_name[idx] for idx in cell_specific_variants]
    cell_nonspecific_variant_names = [idx_to_name[idx] for idx in cell_nonspecific_variants]
    
    if verbose:
        # Cell-specific variants stats
        specific_percentage = 100 * len(cell_specific_variants) / len(variant_file_counts)
        print(f"\nNumber of cell-specific variants (appear in exactly 1 file): {len(cell_specific_variants)}")
        print(f"This represents {specific_percentage:.2f}% of all variants")
        
        # Cell-nonspecific variants stats
        nonspecific_percentage = 100 * len(cell_nonspecific_variants) / len(variant_file_counts)
        print(f"\nNumber of cell-nonspecific variants (appear in at least 80 files): {len(cell_nonspecific_variants)}")
        print(f"This represents {nonspecific_percentage:.2f}% of all variants")
        
        # Universal variants (appear in all files)
        universal_variants = np.where(variant_file_counts == len(files))[0]
        print(f"\nNumber of variants appearing in all {len(files)} files: {len(universal_variants)}")
    
    return cell_specific_variant_names, cell_nonspecific_variant_names


def analyze_maf_distribution(df_maf: pd.DataFrame, variant_names: List[str]) -> Tuple[int, int, int]:
    """
    Analyze the MAF distribution of a set of variants.
    
    Args:
        df_maf: DataFrame with MAF data
        variant_names: List of variant names to analyze
    
    Returns:
        Tuple[int, int, int]: Count of rare, common, and undefined variants
    """
    ind = df_maf['id'].isin(variant_names)
    df_subset = df_maf[ind]
    rare_count = (df_subset['category'] == 'rare').sum()
    common_count = (df_subset['category'] == 'common').sum()
    undefined_count = (df_subset['category'] == 'undefined').sum()
    
    print(f"Total variants analyzed: {len(df_subset)}")
    print(f"Rare variants (MAF < 0.001): {rare_count}")
    print(f"Common variants (MAF > 0.05): {common_count}")
    print(f"Undefined variants: {undefined_count}")
    
    return rare_count, common_count, undefined_count


def calculate_variant_profile_counts(
    matrix: sparse.csr_matrix, 
    name_to_idx: Dict[str, int],
    pred_model: str,
    group: int,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Calculate for each variant the number of profiles (cells) in which it appears.
    
    Args:
        matrix: Binary membership matrix (variants × files)
        name_to_idx: Mapping of variant names to row indices
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory for saving results
        
    Returns:
        pd.DataFrame: DataFrame with variant names and their profile counts
    """
    # Count how many files each variant appears in (row sums)
    variant_file_counts = matrix.sum(axis=1).A1  # A1 converts to 1D array
    
    # Create a reverse mapping from indices to variant names
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    
    # Create a list of (variant_name, profile_count) tuples
    variant_counts = [(idx_to_name[idx], count) for idx, count in enumerate(variant_file_counts)]
    
    # Create a DataFrame
    df_counts = pd.DataFrame(variant_counts, columns=['variant_id', 'profile_count'])
    
    # If output directory is provided, save the DataFrame
    if output_dir:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'outputs')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f'variant_profile_counts_{pred_model}_group_{group}.csv')
        df_counts.to_csv(csv_path, index=False)
        print(f"Saved variant profile counts to: {csv_path}")
        
        # Save as parquet for more efficient storage and retrieval
        parquet_path = os.path.join(output_dir, f'variant_profile_counts_{pred_model}_group_{group}.parquet')
        df_counts.to_parquet(parquet_path, index=False)
        print(f"Saved variant profile counts to: {parquet_path}")
    
    return df_counts


def calculate_profile_similarity_matrix(
    membership_matrix: sparse.csr_matrix,
    files: List[str],
    pred_model: str,
    group: int,
    output_dir: str = None,
    verbose: bool = True
) -> sparse.csr_matrix:
    """
    Calculate a similarity matrix showing the number of common variants between each pair of profiles.
    
    Args:
        membership_matrix: Binary membership matrix (variants × profiles)
        files: List of file names
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory for saving results
        verbose: If True, print information about the resulting matrix
        
    Returns:
        sparse.csr_matrix: Similarity matrix (profiles × profiles) where each element [i,j] is 
                          the count of variants shared between profiles i and j
    """
    start_time = time.time()
    
    # Transpose the membership matrix to get a matrix of profiles × variants
    profiles_matrix = membership_matrix.transpose()
    
    # Multiply the profiles matrix by the original membership matrix to get the similarity matrix
    # This will be a profiles × profiles matrix where each element [i,j] is the number of variants
    # shared between profile i and profile j
    similarity_matrix = profiles_matrix.dot(membership_matrix)
    
    if verbose:
        print(f"\nProfile similarity matrix shape: {similarity_matrix.shape} (profiles × profiles)")
        print(f"Number of non-zero elements: {similarity_matrix.count_nonzero()}")
        nonzero_percentage = 100 * similarity_matrix.count_nonzero() / (similarity_matrix.shape[0] * similarity_matrix.shape[1])
        print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
        print(f"Computation time: {time.time() - start_time:.2f} seconds")
        
        # Calculate some statistics
        similarity_values = similarity_matrix.data
        print(f"\nStatistics of non-zero similarity values:")
        print(f"  Min: {similarity_values.min()}")
        print(f"  Max: {similarity_values.max()}")
        print(f"  Mean: {similarity_values.mean():.2f}")
        print(f"  Median: {np.median(similarity_values):.2f}")
        
        # Histogram of diagonal elements (self-similarity, i.e., number of variants in each profile)
        diag_values = similarity_matrix.diagonal()
        print(f"\nStatistics of diagonal elements (variants per profile):")
        print(f"  Min: {diag_values.min()}")
        print(f"  Max: {diag_values.max()}")
        print(f"  Mean: {diag_values.mean():.2f}")
        print(f"  Median: {np.median(diag_values):.2f}")
    
    # If output directory is provided, save the similarity matrix
    if output_dir:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'outputs')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the sparse similarity matrix
        matrix_file = os.path.join(output_dir, f'profile_similarity_matrix_{pred_model}_group_{group}.npz')
        sparse.save_npz(matrix_file, similarity_matrix)
        print(f"Saved profile similarity matrix to: {matrix_file}")
        
        # Save the file names to help interpret the matrix
        files_file = os.path.join(output_dir, f'profile_similarity_files_{pred_model}_group_{group}.pkl')
        with open(files_file, 'wb') as f:
            pickle.dump(files, f)
        print(f"Saved profile names to: {files_file}")
        
        # Optionally save the top N most similar pairs
        if verbose:
            # Extract upper triangular part of the matrix (excluding diagonal)
            rows, cols = similarity_matrix.nonzero()
            mask = rows < cols  # Upper triangular only
            rows, cols = rows[mask], cols[mask]
            values = similarity_matrix.data[mask]
            
            # Sort by similarity value in descending order
            top_indices = np.argsort(-values)[:100]  # Top 100 most similar pairs
            
            # Create a CSV file with the top pairs
            csv_path = os.path.join(output_dir, f'top_similar_profiles_{pred_model}_group_{group}.csv')
            with open(csv_path, 'w') as f:
                f.write("Profile1,Profile2,CommonVariants,Profile1_TotalVariants,Profile2_TotalVariants,JaccardSimilarity\n")
                for idx in top_indices:
                    i, j = rows[idx], cols[idx]
                    common = similarity_matrix[i, j]
                    variants_i = similarity_matrix[i, i]
                    variants_j = similarity_matrix[j, j]
                    jaccard = common / (variants_i + variants_j - common) if common > 0 else 0
                    f.write(f"{files[i]},{files[j]},{common},{variants_i},{variants_j},{jaccard:.6f}\n")
            print(f"Saved top 100 most similar profile pairs to: {csv_path}")
    
    return similarity_matrix


def load_profile_similarity_matrix(
    pred_model: str,
    group: int,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Load the profile similarity matrix and associated file names.
    
    Args:
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory containing saved files
        verbose: If True, print information about loaded data
        
    Returns:
        Tuple containing:
        - sparse.csr_matrix: Similarity matrix (profiles × profiles)
        - List[str]: List of file names
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Load the sparse matrix
    matrix_file = os.path.join(output_dir, f'profile_similarity_matrix_{pred_model}_group_{group}.npz')
    similarity_matrix = sparse.load_npz(matrix_file)
    
    # Load the files list
    files_file = os.path.join(output_dir, f'profile_similarity_files_{pred_model}_group_{group}.pkl')
    with open(files_file, 'rb') as f:
        files = pickle.load(f)
    
    if verbose:
        print(f"\nLoaded profile similarity matrix with shape {similarity_matrix.shape}")
        print(f"Number of non-zero elements: {similarity_matrix.count_nonzero()}")
        nonzero_percentage = 100 * similarity_matrix.count_nonzero() / (similarity_matrix.shape[0] * similarity_matrix.shape[1])
        print(f"Non-zero percentage: {nonzero_percentage:.2f}%")
    
    return similarity_matrix, files


def visualize_profile_similarity(
    similarity_matrix: sparse.csr_matrix, 
    files: List[str], 
    sample_size: int = 50,
    output_file: str = None
) -> None:
    """
    Visualize the profile similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: Similarity matrix (profiles × profiles)
        files: List of file names
        sample_size: Number of profiles to sample for visualization
        output_file: Path to save the visualization (default: None, just display)
        
    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Check if the matrix is too large to visualize in full
        if similarity_matrix.shape[0] > sample_size:
            print(f"Matrix is large ({similarity_matrix.shape[0]} x {similarity_matrix.shape[0]}), sampling {sample_size} profiles for visualization.")
            
            # Sample a subset of profiles
            indices = np.random.choice(similarity_matrix.shape[0], sample_size, replace=False)
            indices.sort()  # Sort to preserve some order
            
            # Extract submatrix
            submatrix = similarity_matrix[indices, :][:, indices].toarray()
            file_labels = [os.path.basename(files[i]) for i in indices]
        else:
            # Use the full matrix
            submatrix = similarity_matrix.toarray()
            file_labels = [os.path.basename(f) for f in files]
        
        # Create the heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            submatrix, 
            cmap='viridis', 
            xticklabels=file_labels if len(file_labels) <= 30 else False,
            yticklabels=file_labels if len(file_labels) <= 30 else False
        )
        plt.title(f'Profile Similarity Matrix (Common Variants)')
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file, dpi=150)
            print(f"Saved visualization to: {output_file}")
        else:
            plt.show()
            
    except ImportError:
        print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
        print("You can still analyze the similarity matrix programmatically.")
        
    return None


# File operations
def save_variant_lists(
    cell_specific_variants: List[str],
    cell_nonspecific_variants: List[str],
    pred_model: str,
    group: int,
    output_dir: str = None
) -> None:
    """
    Save cell-specific and cell-nonspecific variant lists to files.
    
    Args:
        cell_specific_variants: List of cell-specific variant names
        cell_nonspecific_variants: List of cell-nonspecific variant names
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory (default: create 'outputs' in current directory)
    
    Returns:
        None
    """
    try:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'outputs')
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")
        
        # Save cell-specific variant names
        specific_output = os.path.join(output_dir, f'cell_specific_variants_{pred_model}_group_{group}.pkl')
        with open(specific_output, 'wb') as f:
            pickle.dump(cell_specific_variants, f)
        print(f"Saved {len(cell_specific_variants)} cell-specific variants ({pred_model}) to: {specific_output}")
        
        # Save cell-nonspecific variant names
        nonspecific_output = os.path.join(output_dir, f'cell_nonspecific_variants_{pred_model}_group_{group}.pkl')
        with open(nonspecific_output, 'wb') as f:
            pickle.dump(cell_nonspecific_variants, f)
        print(f"Saved {len(cell_nonspecific_variants)} cell-nonspecific variants ({pred_model}) to: {nonspecific_output}")
        
        # Optional: Save as text files (one variant name per line) for easier inspection
        specific_txt = os.path.join(output_dir, f'cell_specific_variants_{pred_model}_group_{group}.txt')
        with open(specific_txt, 'w') as f:
            for name in cell_specific_variants:
                f.write(f"{name}\n")
        
        nonspecific_txt = os.path.join(output_dir, f'cell_nonspecific_variants_{pred_model}_group_{group}.txt')
        with open(nonspecific_txt, 'w') as f:
            for name in cell_nonspecific_variants:
                f.write(f"{name}\n")
                
    except PermissionError as e:
        print(f"ERROR: Permission denied when writing to {output_dir}")
        print(f"Details: {e}")
        print("Suggestion: Check if you have write permissions to this directory or provide an alternative directory.")
    except OSError as e:
        print(f"ERROR: Failed to save files to {output_dir}")
        print(f"Details: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error while saving files")
        print(f"Details: {e}")


def load_variant_lists(
    pred_model: str,
    group: int,
    output_dir: str = None
) -> Tuple[List[str], List[str]]:
    """
    Load cell-specific and cell-nonspecific variant lists from files.
    
    Args:
        pred_model: Prediction model identifier (e.g., 'pred1', 'pred150')
        group: Group number
        output_dir: Output directory (default: look in 'outputs' in current directory)
    
    Returns:
        Tuple containing:
        - List[str]: Cell-specific variant names
        - List[str]: Cell-nonspecific variant names
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Load cell-specific variants
    specific_file = os.path.join(output_dir, f'cell_specific_variants_{pred_model}_group_{group}.pkl')
    with open(specific_file, 'rb') as f:
        cell_specific_variants = pickle.load(f)
    print(f"Loaded {len(cell_specific_variants)} cell-specific variants for {pred_model}")
    
    # Load cell-nonspecific variants
    nonspecific_file = os.path.join(output_dir, f'cell_nonspecific_variants_{pred_model}_group_{group}.pkl')
    with open(nonspecific_file, 'rb') as f:
        cell_nonspecific_variants = pickle.load(f)
    print(f"Loaded {len(cell_nonspecific_variants)} cell-nonspecific variants for {pred_model}")
    
    return cell_specific_variants, cell_nonspecific_variants


def create_readme(output_dir: str) -> None:
    """
    Create a README file in the output directory explaining the files.
    
    Args:
        output_dir: Path to the output directory
    """
    try:
        # Read template
        readme_template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README_template.md')
        
        # Check if template exists
        if not os.path.exists(readme_template_path):
            # Create a basic template
            content = f"""# Variant Analysis Results

This directory contains the results of running the variant analysis module on genetic variant data.

## Files Description

For each prediction model (`pred1`, `pred150`) and group number, the following files are generated:

- `cell_specific_variants_<model>_group_<n>.pkl`: Pickle file containing cell-specific variant names (appearing in exactly 1 file)
- `cell_specific_variants_<model>_group_<n>.txt`: Text file with the same variant names, one per line
- `cell_nonspecific_variants_<model>_group_<n>.pkl`: Pickle file containing cell-nonspecific variant names (appearing in at least 80 files)
- `cell_nonspecific_variants_<model>_group_<n>.txt`: Text file with the same variant names, one per line
- `variant_profile_counts_<model>_group_<n>.csv`: CSV file containing counts of how many profiles (cells) each variant appears in
- `variant_profile_counts_<model>_group_<n>.parquet`: Parquet file with the same data for efficient storage and retrieval
- `profile_similarity_matrix_<model>_group_<n>.npz`: Sparse matrix showing the number of common variants between each pair of profiles
- `profile_similarity_files_<model>_group_<n>.pkl`: List of profile names corresponding to the similarity matrix indices
- `top_similar_profiles_<model>_group_<n>.csv`: CSV file containing the top 100 most similar profile pairs

## Analysis Date

These results were generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Generated By

Variant Analysis module from the UAVarPrior project.
"""
        else:
            # Read from template file
            with open(readme_template_path, 'r') as f:
                content = f.read()
            
            # Replace date placeholder
            content = content.replace('$(date)', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Write README to output directory
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(content)
        
        print(f"Created README file at: {readme_path}")
    
    except Exception as e:
        print(f"Warning: Could not create README file: {e}")
        # Non-fatal error, continue execution


# Main workflow functions
def process_prediction_model(
    group: int,
    model_name: str,
    input_path: str,
    maf_data: pd.DataFrame,
    save_results: bool = True,
    output_dir: str = None,
    use_cached_matrix: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Process a prediction model to identify cell-specific and cell-nonspecific variants.
    
    Args:
        group: Group number
        model_name: Name of the prediction model (e.g., 'pred1', 'pred150')
        input_path: Path to the directory containing variant files
        maf_data: DataFrame with MAF data
        save_results: If True, save results to files
        output_dir: Output directory for saving results
        use_cached_matrix: If True, try to load previously computed membership matrix
    
    Returns:
        Tuple containing:
        - List[str]: Cell-specific variant names
        - List[str]: Cell-nonspecific variant names
    """
    print(f"\n{'='*80}\nProcessing {model_name} (Group {group})\n{'='*80}")
    
    # Check if membership matrix already exists
    if use_cached_matrix and check_membership_matrix_exists(model_name, group, output_dir):
        print("\nFound existing membership matrix. Loading from files...")
        membership_matrix, files, name_to_idx = load_membership_matrix(
            model_name, group, output_dir
        )
    else:
        # Step 1: Collect unique variants
        print("\nStep 1: Collecting unique variants...")
        unique_names = collect_unique_variants(input_path)
        
        # Step 2: Build membership matrix
        print("\nStep 2: Building membership matrix...")
        membership_matrix, files, name_to_idx = build_membership_matrix(unique_names, input_path)
        
        # Save the membership matrix for future use
        if save_results:
            print("\nSaving membership matrix for future use...")
            save_membership_matrix(
                membership_matrix, 
                files, 
                name_to_idx,
                model_name, 
                group, 
                output_dir
            )
    
    # Step 3: Analyze membership matrix
    print("\nStep 3: Analyzing membership matrix...")
    cell_specific_variants, cell_nonspecific_variants = analyze_membership_matrix(
        membership_matrix, files, name_to_idx
    )
    
    # Step 4: Analyze MAF distribution
    print("\nStep 4: Analyzing MAF distribution for cell-specific variants...")
    analyze_maf_distribution(maf_data, cell_specific_variants)
    
    print("\nStep 4: Analyzing MAF distribution for cell-nonspecific variants...")
    analyze_maf_distribution(maf_data, cell_nonspecific_variants)
    
    # Step 5: Calculate and save variant profile counts
    print("\nStep 5: Calculating and saving variant profile counts...")
    calculate_variant_profile_counts(
        membership_matrix,
        name_to_idx,
        model_name,
        group,
        output_dir
    )
    
    # Step 6: Calculate profile similarity matrix
    print("\nStep 6: Calculating profile similarity matrix...")
    calculate_profile_similarity_matrix(
        membership_matrix,
        files,
        model_name,
        group,
        output_dir
    )
    
    # Step 7: Save results if requested
    if save_results:
        print("\nStep 7: Saving results...")
        save_variant_lists(
            cell_specific_variants,
            cell_nonspecific_variants,
            model_name,
            group,
            output_dir
        )
        create_readme(output_dir)
    
    return cell_specific_variants, cell_nonspecific_variants


def run_full_analysis(group: int = 1, save_results: bool = True, output_dir: str = None, model: str = "both", use_cached_matrix: bool = True) -> Dict:
    """
    Run the full analysis pipeline for both prediction models.
    
    Args:
        group: Group number (default: 1)
        save_results: If True, save results to files
        output_dir: Output directory for saving results
        model: Which prediction model to process ("pred1", "pred150", or "both")
        
    Returns:
        Dict: Dictionary containing all results
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create README file
    if save_results:
        create_readme(output_dir)
    
    # Load MAF data
    print("Loading MAF data...")
    maf_data_path = '/scratch/ml-csm/datasets/genomics/ref-genome/human/GRCh38/ensembl/variants/processed/1000GENOMES-release114-maf.parquet.gz'
    maf_data = load_maf_data(maf_data_path)
    print(f"MAF data loaded with {len(maf_data)} variants")
    
    # Initialize variables
    cell_specific_variants_pred150 = None
    cell_nonspecific_variants_pred150 = None
    cell_specific_variants_pred1 = None
    cell_nonspecific_variants_pred1 = None
    
    # Process pred150 model if requested
    if model in ["pred150", "both"]:
        pred150_path = f'/home/sdodl001/Desktop/DNA_Methylation_Scripts/cpg_util_scripts/data/kmeans/uncert_gve_direction/{group}/pred200_merged/'
        cell_specific_variants_pred150, cell_nonspecific_variants_pred150 = process_prediction_model(
            group, 'pred150', pred150_path, maf_data, save_results, output_dir, use_cached_matrix=use_cached_matrix
        )
    
    # Process pred1 model if requested
    if model in ["pred1", "both"]:
        thr = 0.10
        pred1_path = f'/scratch/ml-csm/projects/fgenom/gve/output/kmeans/pred1/aggr/thr{thr}/{group}/'
        cell_specific_variants_pred1, cell_nonspecific_variants_pred1 = process_prediction_model(
            group, 'pred1', pred1_path, maf_data, save_results, output_dir, use_cached_matrix=use_cached_matrix
        )
    
    # Return all results
    return {
        'pred1': {
            'cell_specific': cell_specific_variants_pred1,
            'cell_nonspecific': cell_nonspecific_variants_pred1
        },
        'pred150': {
            'cell_specific': cell_specific_variants_pred150,
            'cell_nonspecific': cell_nonspecific_variants_pred150
        },
        'maf_data': maf_data
    }


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Variant Analysis Module")
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, 
                        default="/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana/",
                        help="Output directory for saving results (default: /scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to files")
    parser.add_argument("--model", type=str, choices=["pred1", "pred150", "both"], default="both",
                       help="Which prediction model to process (default: both)")
    parser.add_argument("--rebuild-matrix", action="store_true",
                        help="Always rebuild the membership matrix even if a cached one exists")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    print("Variant Analysis Module")
    print(f"Running analysis for group {args.group}")
    
    if args.output_dir:
        print(f"Results will be saved to: {args.output_dir}")
    elif not args.no_save:
        default_output_dir = os.path.join(os.getcwd(), 'outputs')
        print(f"Results will be saved to default directory: {default_output_dir}")
    else:
        print("Results will not be saved (--no-save flag is set)")
        
    # Run the analysis with the specified parameters
    # Note: use_cached_matrix is the opposite of rebuild_matrix
    results = run_full_analysis(
        group=args.group,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        model=args.model,
        use_cached_matrix=not args.rebuild_matrix
    )
    
    print("\nAnalysis complete!")
    
    # Example of other ways to use the module:
    """
    # Load MAF data
    maf_data = load_maf_data('/path/to/maf_data.parquet.gz')
    
    # Process specific prediction model
    cell_specific, cell_nonspecific = process_prediction_model(
        group=1,
        model_name='pred1',
        input_path='/path/to/pred1/data',
        maf_data=maf_data
    )
    
    # Or load previously saved results
    cell_specific, cell_nonspecific = load_variant_lists('pred1', 1)
    """

