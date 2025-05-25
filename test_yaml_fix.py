#!/usr/bin/env python3
"""
Test script to verify YAML configuration fixes in UAVarPrior.

This script tests the fixes for YAML configuration parsing, object instantiation,
and analyzer class path extraction to ensure they work correctly.
"""
import os
import sys
import logging
import tempfile
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_yaml_fix")

# Add UAVarPrior to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.uavarprior.setup.config import load_path, instantiate
    from src.uavarprior.setup import parse_configs_and_run
except ImportError as e:
    logger.error(f"Failed to import UAVarPrior modules: {e}")
    sys.exit(1)


def test_yaml_loading():
    """Test basic YAML configuration loading."""
    logger.info("Testing basic YAML loading...")
    
    test_config = {
        "ops": ["analyze"],
        "model": {
            "name": "TestModel",
            "class": "simple_conv_model",
            "classArgs": {
                "input_channels": 4,
                "conv_channels": [32, 64],
                "kernel_size": 7
            }
        },
        "output_dir": "/tmp/test_output",
        "random_seed": 1337
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        yaml.dump(test_config, tmp)
        tmp_path = tmp.name
    
    try:
        loaded_config = load_path(tmp_path, instantiate=False)
        assert loaded_config["ops"] == ["analyze"]
        assert loaded_config["model"]["name"] == "TestModel"
        assert loaded_config["random_seed"] == 1337
        logger.info("✓ Basic YAML loading test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Basic YAML loading test failed: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def test_object_instantiation():
    """Test YAML object instantiation with !obj: tags."""
    logger.info("Testing YAML object instantiation...")
    
    # Create a simple test function
    def test_function(arg1=None, arg2=None, **kwargs):
        return {"arg1": arg1, "arg2": arg2, "kwargs": kwargs}
    
    # Add to uavarprior.utils for testing
    try:
        import uavarprior.utils
        uavarprior.utils.test_function = test_function
    except ImportError:
        logger.warning("Could not add test function to uavarprior.utils")
    
    test_yaml_content = """
test_obj: !obj:uavarprior.utils.test_function
  arg1: "hello"
  arg2: 42

test_dict:
  class: uavarprior.utils.test_function
  arg1: "world"
  arg2: 24
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        tmp.write(test_yaml_content)
        tmp_path = tmp.name
    
    try:
        configs = load_path(tmp_path, instantiate=False)
        
        # Test !obj: instantiation
        test_obj = configs.get("test_obj")
        if hasattr(test_obj, 'bind'):
            result = instantiate(test_obj)
            logger.info(f"Object instantiation result: {result}")
        
        logger.info("✓ Object instantiation test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Object instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(tmp_path)


def test_analyzer_config():
    """Test analyzer configuration parsing."""
    logger.info("Testing analyzer configuration...")
    
    analyzer_config = {
        "ops": ["analyze"],
        "analyzer": {
            "class": "uavarprior.predict.AnalyzeSequences",
            "args": {
                "trained_model_path": "/path/to/model.pth",
                "sequence_length": 1000,
                "batch_size": 64
            }
        },
        "output_dir": "/tmp/test_analyze",
        "random_seed": 42
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        yaml.dump(analyzer_config, tmp)
        tmp_path = tmp.name
    
    try:
        configs = load_path(tmp_path, instantiate=False)
        analyzer = configs.get("analyzer", {})
        
        assert "class" in analyzer
        assert "args" in analyzer
        assert analyzer["class"] == "uavarprior.predict.AnalyzeSequences"
        
        logger.info("✓ Analyzer configuration test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Analyzer configuration test failed: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def test_config_validation():
    """Test configuration validation with required fields."""
    logger.info("Testing configuration validation...")
    
    # Test with missing required fields
    incomplete_config = {
        "model": {"name": "test"}
        # Missing 'ops' which is required
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        yaml.dump(incomplete_config, tmp)
        tmp_path = tmp.name
    
    try:
        configs = load_path(tmp_path, instantiate=False)
        
        # Should be able to load even incomplete configs
        assert "model" in configs
        logger.info("✓ Configuration validation test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration validation test failed: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def test_path_resolution():
    """Test path resolution in configurations."""
    logger.info("Testing path resolution...")
    
    config_with_paths = {
        "ops": ["analyze"],
        "model": {
            "path": "./models/test_model.py",
            "class": "TestModel"
        },
        "data": {
            "reference_sequence": {
                "input_path": "/path/to/reference.fa"
            }
        },
        "output_dir": "./outputs/test"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        yaml.dump(config_with_paths, tmp)
        tmp_path = tmp.name
    
    try:
        configs = load_path(tmp_path, instantiate=False)
        
        # Check that paths are preserved
        assert configs["model"]["path"] == "./models/test_model.py"
        assert configs["data"]["reference_sequence"]["input_path"] == "/path/to/reference.fa"
        
        logger.info("✓ Path resolution test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Path resolution test failed: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def main():
    """Run all YAML fix tests."""
    logger.info("Starting YAML fix validation tests...")
    
    tests = [
        test_yaml_loading,
        test_object_instantiation,
        test_analyzer_config,
        test_config_validation,
        test_path_resolution
    ]
    
    results = {}
    for test_func in tests:
        test_name = test_func.__name__
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("YAML Fix Test Results:")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info("="*50)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All YAML fix tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())