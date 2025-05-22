#!/bin/bash

# Comprehensive script to install UAVarPrior with compatible dependency versions
# BASH version of the installation script

echo "===== UAVarPrior Installation Script ====="
echo "Starting installation process at $(date)"
echo ""

# ===== Module Management =====
echo "Step 1: Checking and loading required modules..."
# Check if modules are loaded
slurm_loaded=$(module list 2>&1 | grep -c "slurm/23.11")
container_env_loaded=$(module list 2>&1 | grep -c "container_env/0.1")
tensorflow_loaded=$(module list 2>&1 | grep -c "tensorflow-gpu/2.17.0")

if [ $slurm_loaded -eq 0 ]; then
    echo "Loading slurm/23.11 module..."
    module load slurm/23.11
else
    echo "slurm/23.11 module already loaded"
fi

if [ $container_env_loaded -eq 0 ]; then
    echo "Loading container_env/0.1 module..."
    module load container_env/0.1
else
    echo "container_env/0.1 module already loaded"
fi

if [ $tensorflow_loaded -eq 0 ]; then
    echo "Loading tensorflow-gpu/2.17.0 module..."
    module load tensorflow-gpu/2.17.0
else
    echo "tensorflow-gpu/2.17.0 module already loaded"
fi

echo "Modules loaded. Current modules:"
module list

# ===== Clean Previous Build Artifacts =====
echo ""
echo "Step 2: Cleaning up previous build artifacts..."
rm -rf build dist *.egg-info uavarprior.egg-info
rm -f uavarprior/data/sequences/_sequence.c
rm -f uavarprior/data/targets/_genomic_features.c
echo "Cleanup completed."

# ===== Install Dependencies =====
echo ""
echo "Step 3: Installing compatible versions of dependencies..."
echo "Upgrading pip..."
crun -p ~/envs/UAVarPrior/ pip install --upgrade pip &> pip_upgrade.log

# Core build dependencies - these must be installed first
echo "Installing core build dependencies..."
crun -p ~/envs/UAVarPrior/ pip install 'numpy>=1.19.0,<2.0.0' 'setuptools>=49.0.0,<69.0.0' 'wheel>=0.37.0,<0.42.0' &> deps_core.log

# Cython - essential for compiling the extension modules
echo "Installing Cython..."
crun -p ~/envs/UAVarPrior/ pip install 'cython>=0.29.0,<3.0.0' &> deps_cython.log

# Numerical and scientific computing packages
echo "Installing numerical/scientific packages..."
crun -p ~/envs/UAVarPrior/ pip install 'scipy>=1.7.0,<1.10.0' 'scikit-learn>=1.0.0,<1.3.0' &> deps_scientific.log

# Data handling and visualization packages
echo "Installing data handling packages..."
crun -p ~/envs/UAVarPrior/ pip install 'pandas>=1.3.0,<2.0.0' 'h5py>=3.1.0,<4.0.0' &> deps_data.log

# Visualization packages
echo "Installing visualization packages..."
crun -p ~/envs/UAVarPrior/ pip install 'matplotlib>=3.4.0,<3.7.0' 'seaborn>=0.11.0,<0.13.0' 'plotly>=5.8.0,<5.15.0' &> deps_viz.log

# Machine learning framework
echo "Installing PyTorch..."
crun -p ~/envs/UAVarPrior/ pip install 'torch>=1.10.0,<2.0.0' 'torchinfo>=1.7.0,<1.8.0' &> deps_torch.log

# Bioinformatics packages
echo "Installing bioinformatics packages..."
crun -p ~/envs/UAVarPrior/ pip install 'biopython>=1.79,<1.82' 'pyfaidx>=0.7.0,<0.8.0' 'pytabix>=0.1,<0.2' &> deps_bio.log

# Other dependencies
echo "Installing other dependencies..."
crun -p ~/envs/UAVarPrior/ pip install 'pyyaml>=5.1,<6.0' 'statsmodels>=0.13.0,<0.14.0' 'click>=8.0.0,<8.2.0' 'pydantic>=1.9.0,<2.0.0' &> deps_other.log

echo "All dependencies installed."

# ===== Install UAVarPrior Package =====
echo ""
echo "Step 4: Installing UAVarPrior package in editable mode..."
crun -p ~/envs/UAVarPrior/ pip install -e . --no-deps -v &> install_log.txt
echo "Installation completed."

# ===== Test Installation =====
echo ""
echo "Step 5: Testing UAVarPrior installation..."
echo "Running compatibility test..."
crun -p ~/envs/UAVarPrior/ python test_compatibility.py &> compatibility_test.txt

# ===== Display Results =====
echo ""
echo "===== INSTALLATION SUMMARY ====="
echo "Install log (last 15 lines):"
tail -15 install_log.txt

echo ""
echo "Compatibility test output:"
cat compatibility_test.txt

echo ""
echo "===== ADDITIONAL VALIDATION ====="
echo "Running comprehensive validation script..."
crun -p ~/envs/UAVarPrior/ python validate_install.py

echo ""
echo "Generating dependency report..."
crun -p ~/envs/UAVarPrior/ python dependency_report.py

echo ""
echo "===== INSTALLATION COMPLETE ====="
echo "Installation process finished at $(date)"
echo ""
echo "If installation was successful, you should see 'SUCCESS' in the test outputs above."
echo "Check dependency_report.txt for detailed information about installed packages."
