#!/bin/tcsh
# 
# This script runs the test_matrix_loading.py script with proper environment setup
# 
# Created by: Sanjeeva Reddy Dodlapati
# Date: May 22, 2025
#
# Print banner
echo ""
echo "========================================================"
echo "Running membership matrix caching test with proper setup"
echo "========================================================"
echo ""

# Check if conda is available
which conda >& /dev/null
if ($status != 0) then
    echo "ERROR: conda command not found"
    echo "Please make sure conda is installed and in your PATH"
    exit 1
endif

# Set up environment variables to prevent conflicts
setenv PYTHONNOUSERSITE 1
setenv PYTHONPATH ""

# Run the test script using conda run
echo "Running test_matrix_loading.py..."
conda run -p ~/envs/fugep python test_matrix_loading.py

# Check if the run was successful
if ($status == 0) then
    echo ""
    echo "Test completed successfully!"
else
    echo ""
    echo "Test failed with exit code $status"
    echo ""
    echo "If you're seeing GLIBCXX version errors, try updating your environment:"
    echo "conda install -c conda-forge -y libstdcxx-ng"
endif
