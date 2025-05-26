#!/usr/bin/env python3
"""
Test script to validate analyze configuration handling
"""
import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test-analyze")

# Add the project directory to sys.path if needed
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

from uavarprior.setup import load_path
from uavarprior.setup.run import parse_configs_and_run

def main():
    """Test the analyze configuration handling"""
    config_path = os.path.join(project_dir, "test-configs", "test-analyze.yml")
    logger.info(f"Loading config from: {config_path}")
    
    try:
        configs = load_path(config_path, instantiate=False)
        logger.info(f"Loaded configuration with keys: {list(configs.keys())}")
        logger.info(f"Operations: {configs.get('ops', [])}")
        
        logger.info("Calling parse_configs_and_run...")
        parse_configs_and_run(configs)
        logger.info("✅ PASSED: Configuration was processed successfully")
    except Exception as e:
        logger.error(f"❌ FAILED: Error processing configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
