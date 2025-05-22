#!/bin/tcsh

# Script to install and test UAVarPrior

echo "Loading tensorflow-gpu/2.17.0 module..."
module load tensorflow-gpu/2.17.0

echo "Cleaning up previous build artifacts..."
rm -rf build dist *.egg-info uavarprior.egg-info
rm -f uavarprior/data/sequences/_sequence.c
rm -f uavarprior/data/targets/_genomic_features.c

echo "Installing package with build output saved to install_log.txt..."
crun -p ~/envs/UAVarPrior/ pip install -e . > install_log.txt 2>&1

echo "Testing import..."
crun -p ~/envs/UAVarPrior/ python -c "try: import uavarprior; print(f'SUCCESS: UAVarPrior version {uavarprior.__version__} imported successfully!'); except Exception as e: print(f'ERROR: {e}')" > test_output.txt 2>&1

echo "==== INSTALLATION LOG SUMMARY ===="
tail -20 install_log.txt
echo ""
echo "==== IMPORT TEST OUTPUT ===="
cat test_output.txt

echo ""
echo "Installation process complete"
