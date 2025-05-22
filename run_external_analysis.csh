#!/bin/tcsh

# Script to run the profile similarity analysis with the external data path
# Created by: Sanjeeva Reddy Dodlapati
# Date: May 22, 2025

# Use the path where the similarity matrix files are located
set EXT_PATH = "/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana"

echo "Running profile similarity analysis with external data path..."
echo "Using path: $EXT_PATH"
echo ""

# Call the main analysis script with the external path
./run_profile_analysis.csh --ext-path="$EXT_PATH" $argv[*]
