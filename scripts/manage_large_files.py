#!/usr/bin/env python
"""
Large File Management Utility

This script provides utilities for managing large files in the UAVarPrior project.
It helps to identify large files, move them to appropriate output directories,
and generate documentation.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path

def get_file_size_mb(file_path):
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

def find_large_files(directory, size_threshold=50, extensions=None):
    """
    Find files larger than the threshold in the given directory.
    
    Args:
        directory (str): Directory to search
        size_threshold (float): Size threshold in MB
        extensions (list): List of file extensions to include
    
    Returns:
        list: List of tuples (path, size_mb)
    """
    large_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip output directories
            if 'outputs' in root.split(os.sep):
                continue
                
            file_path = os.path.join(root, file)
            
            # Check extensions if provided
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue
                
            # Check file size
            size_mb = get_file_size_mb(file_path)
            if size_mb > size_threshold:
                large_files.append((file_path, size_mb))
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)

def move_to_outputs(file_path):
    """
    Move a file to the appropriate outputs directory.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: New path of the file
    """
    # Determine the module (data, model, interpret)
    parts = file_path.split(os.sep)
    try:
        uavarprior_idx = parts.index('uavarprior')
        if len(parts) > uavarprior_idx + 1:
            module = parts[uavarprior_idx + 1]
        else:
            module = 'data'  # Default
    except ValueError:
        module = 'data'  # Default
    
    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join('uavarprior', module, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Get the filename
    filename = os.path.basename(file_path)
    
    # New path
    new_path = os.path.join(outputs_dir, filename)
    
    # Move the file
    shutil.move(file_path, new_path)
    
    return new_path

def main():
    parser = argparse.ArgumentParser(description='Manage large files in the UAVarPrior project')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find large files')
    find_parser.add_argument('--dir', default='.', help='Directory to search')
    find_parser.add_argument('--threshold', type=float, default=50, help='Size threshold in MB')
    find_parser.add_argument('--extensions', nargs='+', help='File extensions to include')
    
    # Move command
    move_parser = subparsers.add_parser('move', help='Move large files to outputs directories')
    move_parser.add_argument('--dir', default='.', help='Directory to search')
    move_parser.add_argument('--threshold', type=float, default=50, help='Size threshold in MB')
    move_parser.add_argument('--extensions', nargs='+', help='File extensions to include')
    move_parser.add_argument('--dry-run', action='store_true', help='Just print what would be moved')
    
    args = parser.parse_args()
    
    if args.command == 'find':
        large_files = find_large_files(args.dir, args.threshold, args.extensions)
        if large_files:
            print(f"Found {len(large_files)} large files (>{args.threshold} MB):")
            for path, size in large_files:
                print(f"{size:.2f} MB: {path}")
        else:
            print(f"No files larger than {args.threshold} MB found.")
    
    elif args.command == 'move':
        large_files = find_large_files(args.dir, args.threshold, args.extensions)
        if large_files:
            print(f"Found {len(large_files)} large files (>{args.threshold} MB):")
            for path, size in large_files:
                if args.dry_run:
                    print(f"Would move {path} ({size:.2f} MB)")
                else:
                    new_path = move_to_outputs(path)
                    print(f"Moved {path} ({size:.2f} MB) to {new_path}")
        else:
            print(f"No files larger than {args.threshold} MB found.")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
