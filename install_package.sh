#!/bin/bash

# Script to install UAVarPrior in editable mode
echo "Checking and loading required modules..."

# slurm/23.11 and container_env/0.1 are loaded by default

# Check if tensorflow-gpu module is already loaded
if ! module list 2>&1 | grep -q "tensorflow-gpu/2.17.0"; then
    echo "Loading tensorflow-gpu/2.17.0 module..."
    module load tensorflow-gpu/2.17.0
fi

echo "Using crun to install package..."
which crun
crun -p ~/envs/UAVarPrior/ pip install -e .

echo "Installation complete. Check for any errors above."
