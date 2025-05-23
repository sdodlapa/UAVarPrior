#!/bin/tcsh

# Script to install UAVarPrior in editable mode
echo "Checking and loading required modules..."

# slurm/23.11 and container_env/0.1 are loaded by default

# Check if tensorflow-gpu module is already loaded
set module_loaded = `module list 2>&1 | grep -c "tensorflow-gpu/2.17.0"`
if ($module_loaded == 0) then
    echo "Loading tensorflow-gpu/2.17.0 module..."
    module load tensorflow-gpu/2.17.0
endif

echo "Using crun to install package..."
which crun

# Clean up any previous build files that might be causing issues
echo "Cleaning up previous build files..."
rm -rf build dist *.egg-info
rm -f uavarprior/data/sequences/_sequence.c
rm -f uavarprior/data/targets/_genomic_features.c

# Install with verbose output
echo "Installing package..."
crun -p ~/envs/UAVarPrior/ pip install -e . -v

echo "Installation complete. Check for any errors above."
