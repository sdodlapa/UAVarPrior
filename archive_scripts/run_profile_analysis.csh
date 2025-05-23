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

# Check if numpy is available
$PYTHON -c "import numpy" >& /dev/null
if ($status != 0) then
    echo "WARNING: NumPy is not installed in your Python environment"
    echo "NumPy is required for matrix analysis"
    echo ""
    echo "Would you like to install NumPy now? (y/n)"
    set answer = $<
    
    if ("$answer" == "y" || "$answer" == "Y") then
        echo "Installing NumPy..."
        $PYTHON -m pip install numpy scipy
        
        if ($status == 0) then
            echo "NumPy installed successfully!"
        else
            echo "Failed to install NumPy. Please install it manually:"
            echo "  $PYTHON -m pip install numpy scipy"
            echo ""
        endif
    endif
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
set ext_path = ""
set matrix_file = ""
set files_file = ""

# Loop through arguments
foreach arg ($argv)
    if ("$arg" =~ "--model=*") then
        set model = `echo $arg | sed 's/--model=//'`
    else if ("$arg" =~ "--group=*") then
        set group = `echo $arg | sed 's/--group=//'`
    else if ("$arg" =~ "--output-dir=*") then
        set output_dir = `echo $arg | sed 's/--output-dir=//'`
    else if ("$arg" =~ "--ext-path=*") then
        set ext_path = `echo $arg | sed 's/--ext-path=//'`
    else if ("$arg" =~ "--matrix-file=*") then
        set matrix_file = `echo $arg | sed 's/--matrix-file=//'`
    else if ("$arg" =~ "--files-file=*") then
        set files_file = `echo $arg | sed 's/--files-file=//'`
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
if ("$ext_path" != "") then
    set options = "$options --ext-path \"$ext_path\""
endif
if ("$matrix_file" != "") then
    set options = "$options --matrix-file \"$matrix_file\""
endif
if ("$files_file" != "") then
    set options = "$options --files-file \"$files_file\""
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
    conda run -p ~/envs/UAVarPrior_TF2170 $PYTHON standalone_test.py --matrix-type=similarity $options
    set TEST_STATUS = $status
else
    $PYTHON standalone_test.py --matrix-type=similarity $options
    set TEST_STATUS = $status
endif

# Check if the standalone analysis was successful
if ($TEST_STATUS == 0) then
    echo ""
    echo "Standalone analysis completed successfully!"
else
    echo ""
    echo "Standalone analysis failed with exit code $TEST_STATUS"
    echo ""
    echo "Trying simple analysis..."
    
    # Try running the simple profile analysis script as a fallback
    if ($HAS_CONDA == 0) then
        conda run -p ~/envs/UAVarPrior_TF2170 $PYTHON simple_profile_analysis.py $options
        set SIMPLE_STATUS = $status
    else
        $PYTHON simple_profile_analysis.py $options
        set SIMPLE_STATUS = $status
    endif
    
    if ($SIMPLE_STATUS == 0) then
        echo "Simple analysis completed successfully!"
    else
        echo ""
        echo "All analysis attempts failed."
        echo "Simple analysis failed with exit code $SIMPLE_STATUS"
        echo ""
        echo "You may be missing required Python packages. Here are some options:"
        echo ""
        echo "1. Checking available matrices:"
        echo "   ls -l outputs/*.npz"
        echo ""
        echo "2. Install the required Python packages:"
        echo "   pip install numpy scipy"
        echo ""
        echo "3. If using conda:"
        echo "   conda install -c conda-forge numpy scipy matplotlib seaborn"
        echo ""
        echo "4. For direct analysis, try running one of these commands:"
        if ($HAS_CONDA == 0) then
            echo "   conda run -p ~/envs/UAVarPrior_TF2170 $PYTHON standalone_test.py --matrix-type=similarity $options"
        else
            echo "   $PYTHON standalone_test.py --matrix-type=similarity $options"
        endif
    endif
endif
