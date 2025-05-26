#!/usr/bin/env python
"""
Simple test for UAVarPrior analyze functionality
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.uavarprior.setup import load_path, parse_configs_and_run
    logger.info("Successfully imported UAVarPrior modules")
except ImportError as e:
    logger.error(f"Failed to import UAVarPrior modules: {e}")
    sys.exit(1)

# Create a simple analyze config
config = {
    "ops": ["analyze"],
    "model": {
        "class": "DummyClass",
        "built": "pytorch",
        "wrapper": "DummyWrapper",
        "mult_predictions": 1,
        "classArgs": {}
    },
    "analyzer": {
        "class": "dummy.Analyzer"
    },
    "variant_effect_prediction": {},
    "output_dir": "/tmp/test-output"
}

try:
    # Mock the execute function to avoid actual execution
    import src.uavarprior.setup.run
    original_execute = src.uavarprior.setup.run.execute
    src.uavarprior.setup.run.execute = lambda configs: logger.info(f"Execute called with ops: {configs.get('ops')}")
    
    # Run the test
    logger.info("Calling parse_configs_and_run")
    parse_configs_and_run(config)
    
    # Restore the original function
    src.uavarprior.setup.run.execute = original_execute
    
    logger.info("Test passed successfully!")
except Exception as e:
    logger.error(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
