#!/usr/bin/env python
"""
Test YAML object instantiation for UAVarPrior 
"""
import os
import sys
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.uavarprior.setup.config import load_path, instantiate

# Path to test config
CONFIG_PATH = "test_obj_instantiate.yml"

def main():
    try:
        logger.info(f"Creating test YAML file at {CONFIG_PATH}")
        with open(CONFIG_PATH, "w") as f:
            f.write("""
# Test YAML for object instantiation
test_obj: !obj:uavarprior.utils.function_a
  arg1: 123
  arg2: hello

test_dict:
  class: uavarprior.utils.function_a
  arg1: 456
  arg2: world
""")

        # Create dummy function for testing
        logger.info("Adding dummy function to uavarprior.utils for testing")
        import uavarprior.utils
        def function_a(arg1=None, arg2=None, model=None, outputDir=None):
            logger.info(f"function_a called with: arg1={arg1}, arg2={arg2}, model={model}, outputDir={outputDir}")
            return {"arg1": arg1, "arg2": arg2, "model": model, "outputDir": outputDir}
            
        uavarprior.utils.function_a = function_a
        
        # Test loading and instantiation
        logger.info(f"Loading config from {CONFIG_PATH}")
        configs = load_path(CONFIG_PATH, instantiate=False)
        
        logger.info("Config keys:")
        for key in configs:
            logger.info(f"  {key}: {type(configs[key])}")
        
        # Test if !obj: tag creates an object with .bind method
        test_obj = configs.get("test_obj")
        logger.info(f"test_obj type: {type(test_obj)}")
        if hasattr(test_obj, "bind"):
            logger.info("test_obj has bind method!")
            test_obj.bind(model="test_model", outputDir="test_dir")
            result = instantiate(test_obj)
            logger.info(f"Instantiated result: {result}")
        else:
            logger.error("test_obj does NOT have bind method!")
        
        # Test instantiating from plain dict
        test_dict = configs.get("test_dict")
        logger.info(f"test_dict type: {type(test_dict)}")
        if isinstance(test_dict, dict):
            logger.info("test_dict is a dictionary")
            test_dict["model"] = "test_model"
            test_dict["outputDir"] = "test_dir"
            
            class_path = test_dict.pop("class")
            logger.info(f"class_path = {class_path}")
            
            module_path, class_name = class_path.rsplit('.', 1)
            logger.info(f"module_path={module_path}, class_name={class_name}")
            
            import importlib
            module = importlib.import_module(module_path)
            class_obj = getattr(module, class_name)
            
            result = class_obj(**test_dict)
            logger.info(f"Direct instantiation result: {result}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
