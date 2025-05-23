#!/usr/bin/env python
"""
Test script to verify the fixes for analyzer class path extraction.
"""
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_class_path")

# Add UAVarPrior to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import UAVarPrior modules
try:
    from src.uavarprior.setup import load_path, parse_configs_and_run
except ImportError as e:
    logger.error(f"Failed to import UAVarPrior modules: {e}")
    sys.exit(1)

def run_test(config_file):
    """Run test with a specific config file."""
    logger.info(f"Testing with config file: {config_file}")
    try:
        configs = load_path(config_file, instantiate=False)
        logger.info(f"Config keys: {list(configs.keys())}")
        if "analyzer" in configs:
            logger.info(f"Analyzer config: {configs['analyzer']}")
        
        logger.info("Calling parse_configs_and_run()")
        parse_configs_and_run(configs)
        logger.info("Success!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    config_files = [
        # Test with the original file
        "test-short-analyze.yml",
        # Test with the fixed file
        "fixed-test-analyze.yml",
    ]
    
    results = {}
    for config_file in config_files:
        logger.info(f"\n\n=== Testing {config_file} ===")
        results[config_file] = run_test(config_file)
    
    logger.info("\n\n=== Results ===")
    for config_file, success in results.items():
        logger.info(f"{config_file}: {'PASSED' if success else 'FAILED'}")
