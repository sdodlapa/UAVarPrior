#!/usr/bin/env python
# Basic compatibility test for UAVarPrior
# This script only tests imports that exist in the package

import sys
import importlib
from importlib.util import find_spec

def test_module_exists(module_name):
    """Check if a module exists in the package"""
    try:
        spec = find_spec(module_name)
        if spec is None:
            return False
        return True
    except ModuleNotFoundError:
        return False

def print_result(name, success):
    """Print test result with formatting"""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"{status}: {name}")
    return success

def main():
    """Run all tests"""
    results = []
    
    # Test basic package import
    try:
        import uavarprior
        results.append(print_result("Basic package import", True))
        print(f"Package version: {getattr(uavarprior, '__version__', 'unknown')}")
    except ImportError as e:
        results.append(print_result(f"Basic package import: {str(e)}", False))
        # If we can't import the basic package, we can't continue
        return 1
    
    # Test modules that should exist
    modules_to_test = [
        "uavarprior.data",
        "uavarprior.model",
        "uavarprior.samplers",
        "uavarprior.utils",
        "uavarprior.interpret",
        "uavarprior.data.sequences",
        "uavarprior.data.targets"
    ]
    
    for module in modules_to_test:
        results.append(print_result(f"Module: {module}", test_module_exists(module)))
    
    # Test specific Cython extension modules
    try:
        from uavarprior.data.sequences import _sequence
        results.append(print_result("Cython module: _sequence", True))
    except ImportError as e:
        results.append(print_result(f"Cython module: _sequence ({str(e)})", False))
    
    try:
        from uavarprior.data.targets import _genomic_features
        results.append(print_result("Cython module: _genomic_features", True))
    except ImportError as e:
        results.append(print_result(f"Cython module: _genomic_features ({str(e)})", False))
    
    # Test specific imports that should work based on __init__.py
    try:
        from uavarprior.data.sequences import Sequence, Genome
        results.append(print_result("Import: data.sequences.Sequence & Genome", True))
    except ImportError as e:
        results.append(print_result(f"Import: data.sequences.Sequence & Genome ({str(e)})", False))
    
    try:
        from uavarprior.data.targets import Target, GenomicFeatures
        results.append(print_result("Import: data.targets.Target & GenomicFeatures", True))
    except ImportError as e:
        results.append(print_result(f"Import: data.targets.Target & GenomicFeatures ({str(e)})", False))
    
    # Summary
    total = len(results)
    passed = sum(results)
    print(f"\nTest summary: {passed}/{total} tests passed")
    
    # Return success if all tests passed
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
