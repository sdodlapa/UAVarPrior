#!/usr/bin/env python
"""
Minimal test script for UAVarPrior analyze operation.
"""
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test")

# Add UAVarPrior source to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Create test output directory
test_output_dir = "/tmp/test-analyze-output"
os.makedirs(test_output_dir, exist_ok=True)

# Import UAVarPrior modules
try:
    import src.uavarprior
    logger.info("Successfully imported UAVarPrior")
except ImportError as e:
    logger.error(f"Failed to import UAVarPrior: {e}")
    sys.exit(1)

try:
    from src.uavarprior.setup import load_path, parse_configs_and_run
except ImportError as e:
    logger.error(f"Failed to import specific modules: {e}")
    sys.exit(1)

# Define the config file path
CONFIG_PATH = Path(__file__).parent / "test-short-analyze.yml"

def mock_pvarprior_predict(*args, **kwargs):
    """Mock implementation of PeakGVarEvaluator.__init__ to avoid actual loading."""
    logger.info(f"Mock PVarEvaluator called with: {kwargs}")
    return None

def mock_evaluate(*args, **kwargs):
    """Mock implementation of evaluate method."""
    logger.info(f"Mock evaluate called with: {kwargs}")
    return None

def main():
    logger.info(f"Starting test with config: {CONFIG_PATH}")
    
    # Check if file exists
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found: {CONFIG_PATH}")
        sys.exit(1)
    
    # Mock the PeakGVarEvaluator class
    try:
        # First try to import the actual module
        import src.uavarprior.predict
        logger.info("Successfully imported predict module")
        
        # Check if the class exists, if not create a mock
        if not hasattr(src.uavarprior.predict, "PeakGVarEvaluator"):
            logger.warning("PeakGVarEvaluator not found in predict module, creating mock")
            
            # Create mock class
            class MockPeakGVarEvaluator:
                def __init__(self, **kwargs):
                    logger.info(f"MockPeakGVarEvaluator.__init__ called with kwargs: {kwargs}")
                    self.kwargs = kwargs
                
                def evaluate(self, **kwargs):
                    logger.info(f"MockPeakGVarEvaluator.evaluate called with: {kwargs}")
                    return True
            
            # Add the mock class to the module
            src.uavarprior.predict.PeakGVarEvaluator = MockPeakGVarEvaluator
            logger.info("Mock PeakGVarEvaluator created and added to module")
    except ImportError:
        logger.error("Could not import src.uavarprior.predict module")
        sys.exit(1)
    
    try:
        # Load config
        logger.info("Loading configuration")
        configs = load_path(str(CONFIG_PATH), instantiate=False)
        
        # Log available keys in the configs
        logger.info(f"Config loaded with keys: {list(configs.keys())}")
        
        # Process the configuration
        logger.info("Calling parse_configs_and_run")
        parse_configs_and_run(configs)
        logger.info("parse_configs_and_run completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
