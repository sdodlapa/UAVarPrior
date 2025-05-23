#!/bin/bash

echo "Loading tensorflow-gpu/2.17.0 module..."
module load tensorflow-gpu/2.17.0

echo "Cleaning up previous build artifacts..."
rm -rf build dist *.egg-info uavarprior.egg-info
rm -f uavarprior/data/sequences/_sequence.c
rm -f uavarprior/data/targets/_genomic_features.c

echo "Installing Cython and NumPy first..."
crun -p ~/envs/UAVarPrior/ pip install --upgrade cython numpy

echo "Building Cython extensions separately first..."
crun -p ~/envs/UAVarPrior/ python -m pip install -e . --no-build-isolation

echo "Installation attempt complete. Check for errors above."
