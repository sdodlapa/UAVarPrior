#!/usr/bin/env python
# Comprehensive validation script for UAVarPrior
# This script attempts to import and use key components of the package
# to verify that the installation is working correctly

import os
import sys
import traceback
from datetime import datetime

def print_separator():
    print("-" * 60)

def run_test(test_name, test_function):
    print_separator()
    print(f"TEST: {test_name}")
    try:
        test_function()
        print(f"✅ SUCCESS: {test_name} completed successfully")
        return True
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def test_basic_import():
    """Test basic import of the uavarprior package"""
    import uavarprior
    print(f"UAVarPrior version: {getattr(uavarprior, '__version__', 'unknown')}")
    print("Package path:", os.path.dirname(uavarprior.__file__))

def test_dependencies():
    """Test import of all key dependencies"""
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
    
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    import h5py
    print(f"h5py version: {h5py.__version__}")
    
    import Bio
    print(f"BioPython version: {Bio.__version__}")

def test_cython_modules():
    """Test import of Cython extension modules"""
    try:
        from uavarprior.data.sequences import _sequence
        print("Successfully imported _sequence Cython module")
    except ImportError as e:
        print(f"Failed to import _sequence: {e}")
        raise
    
    try:
        from uavarprior.data.targets import _genomic_features
        print("Successfully imported _genomic_features Cython module")
    except ImportError as e:
        print(f"Failed to import _genomic_features: {e}")
        raise

def test_python_modules():
    """Test import of key Python modules from the package"""
    # Import core modules (adjust these based on actual module structure)
    modules_to_test = [
        'uavarprior.data',
        'uavarprior.interpret',
        'uavarprior.cli'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"Successfully imported {module_name}")
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")
            raise

def print_system_info():
    """Print system information for debugging purposes"""
    print("SYSTEM INFORMATION:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Path:")
    for p in sys.path:
        print(f"  - {p}")

def main():
    print_separator()
    print(f"UAVarPrior VALIDATION TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    print_system_info()
    
    tests = [
        ("Basic Import Test", test_basic_import),
        ("Dependencies Test", test_dependencies),
        ("Cython Modules Test", test_cython_modules),
        ("Python Modules Test", test_python_modules)
    ]
    
    results = []
    for name, func in tests:
        results.append(run_test(name, func))
    
    print_separator()
    print("TEST SUMMARY:")
    total = len(results)
    passed = sum(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - UAVarPrior appears to be installed correctly!")
        return 0
    else:
        print(f"❌ {total - passed} TEST(S) FAILED - Check the output above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
