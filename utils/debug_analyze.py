#!/usr/bin/env python
"""
Debug test runner for UAVarPrior analyze operation.
"""

import os
import sys
import logging
import yaml

# Configure logging to be more verbose
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Add UAVarPrior to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.uavarprior.setup import load_path, parse_configs_and_run

# Path to test config
CONFIG_PATH = "test_analyze.yml"

def main():
    try:
        logger.info(f"Loading config from {CONFIG_PATH}")
        configs = load_path(CONFIG_PATH, instantiate=False)
        
        logger.info("Config keys:")
        for key in configs:
            logger.info(f"  {key}: {type(configs[key])}")

        logger.info("Running parse_configs_and_run")
        parse_configs_and_run(configs)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
