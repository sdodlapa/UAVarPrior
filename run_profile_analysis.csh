#!/bin/tcsh

# Script to run the profile similarity analysis with the conda environment
# Created by: Sanjeeva Reddy Dodlapati
# Date: May 22, 2025

# Print banner
echo ""
echo "========================================================="
echo "   Profile Similarity Matrix Analysis"
echo "========================================================="
echo ""

# Check if python is available
which python3 >& /dev/null
if ($status != 0) then
    which python >& /dev/null
    if ($status != 0) then
        echo "ERROR: Python not found"
        echo "Please make sure python is installed and in your PATH"
        exit 1
    else
        set PYTHON = "python"
    endif
else
    set PYTHON = "python3"
endif

# Check for conda (optional)
which conda >& /dev/null
set HAS_CONDA = $status
if ($HAS_CONDA == 0) then
    echo "Found conda, will try to use appropriate environment"
else
    echo "Conda not found, will use system Python"
endif

# Parse command line arguments
set model = "pred1"
set group = "1"
set output_dir = ""

# Loop through arguments
foreach arg ($argv)
    if ("$arg" =~ "--model=*") then
        set model = `echo $arg | sed 's/--model=//'`
    else if ("$arg" =~ "--group=*") then
        set group = `echo $arg | sed 's/--group=//'`
    else if ("$arg" =~ "--output-dir=*") then
        set output_dir = `echo $arg | sed 's/--output-dir=//'`
    endif
end

# Set options for the script
set options = ""
if ("$model" != "") then
    set options = "$options --model $model"
endif
if ("$group" != "") then
    set options = "$options --group $group"
endif
if ("$output_dir" != "") then
    set options = "$options --output-dir $output_dir"
endif

# Set up environment variables to prevent conflicts
setenv PYTHONNOUSERSITE 1
setenv PYTHONPATH ""

# Run the script with the appropriate Python interpreter
echo "Running profile similarity analysis..."
echo "Using model: $model, group: $group"
echo ""

# First, try using the standalone test with matrix_type=similarity
echo "Attempting to run standalone analysis for similarity matrix..."
if ($HAS_CONDA == 0) then
    conda run -p ~/envs/fugep $PYTHON standalone_test.py --matrix-type=similarity $options
else
    $PYTHON standalone_test.py --matrix-type=similarity $options
endif

# Check if the standalone analysis was successful
if ($status == 0) then
    echo ""
    echo "Standalone analysis completed successfully!"
else
    echo ""
    echo "Standalone analysis failed with exit code $status"
    echo ""
    echo "Trying simple analysis..."
    
    # Try running the simple profile analysis script as a fallback
    if ($HAS_CONDA == 0) then
        conda run -p ~/envs/fugep $PYTHON simple_profile_analysis.py $options
    else
        $PYTHON simple_profile_analysis.py $options
    endif
    
    if ($status == 0) then
        echo "Simple analysis completed successfully!"
    else
        echo ""
        echo "All analysis attempts failed."
        echo ""
        echo "If you're getting ImportError related to torch or other missing dependencies, try:"
        echo "1. Checking available matrices:"
        echo "   ls -l outputs/*.npz"
        echo "2. Running the standalone test directly:"
        echo "   $PYTHON standalone_test.py --matrix-type=similarity $options"
        echo "3. If using conda and want to install visualization dependencies:"
        echo "   conda activate ~/envs/fugep"
        echo "   conda install -c conda-forge matplotlib seaborn scipy"
    endif
endif
