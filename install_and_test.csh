#!/bin/tcsh

# Script to install and test UAVarPrior with compatible package versions

echo "Loading tensorflow-gpu/2.17.0 module..."
module load tensorflow-gpu/2.17.0

echo "Cleaning up previous build artifacts..."
rm -rf build dist *.egg-info uavarprior.egg-info
rm -f uavarprior/data/sequences/_sequence.c
rm -f uavarprior/data/targets/_genomic_features.c

echo "Installing compatible versions of key dependencies..."
crun -p ~/envs/UAVarPrior/ pip install --upgrade pip >& pip_upgrade.log
crun -p ~/envs/UAVarPrior/ pip install 'numpy<2.0.0' 'cython>=0.29.0,<3.0.0' 'pydantic<2.0.0' >& deps_install_1.log
crun -p ~/envs/UAVarPrior/ pip install 'torch<2.0.0' 'h5py<4.0.0' 'scikit-learn<1.3.0' >& deps_install_2.log
crun -p ~/envs/UAVarPrior/ pip install 'matplotlib<3.7.0' 'pandas<2.0.0' 'scipy<1.10.0' 'biopython<1.82' >& deps_install_3.log
crun -p ~/envs/UAVarPrior/ pip install 'pyyaml>=5.1,<6.0' 'seaborn<0.13.0' 'statsmodels<0.14.0' >& deps_install_4.log

echo "Installing package with build output saved to install_log.txt..."
crun -p ~/envs/UAVarPrior/ pip install -e . --no-deps >& install_log.txt

echo "Testing compatibility..."
crun -p ~/envs/UAVarPrior/ python test_compatibility.py >& compatibility_test.txt

echo "==== INSTALLATION LOG SUMMARY ===="
tail -20 install_log.txt
echo ""
echo "==== COMPATIBILITY TEST OUTPUT ===="
cat compatibility_test.txt

echo ""
echo "Installation process complete"
