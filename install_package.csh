#!/bin/tcsh

# Script to install UAVarPrior in editable mode
echo "Activating environment..."
source ~/envs/UAVarPrior_TF2170/bin/activate.csh

echo "Installing package in editable mode..."
pip install -e .

echo "Installation complete. Check for any errors above."
