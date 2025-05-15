import os
import pytest
from uavarprior.setup import load_path

def test_load_valid_yaml():
    """Test loading a valid YAML configuration."""
    # Create a temporary config file
    import tempfile
    import yaml
    
    config_data = {
        "model": {
            "name": "test_model",
            "layers": [64, 32, 16],
            "dropout_rate": 0.2
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        },
        "data": {
            "dataset_path": "/path/to/dataset"
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as tmp:
        yaml.dump(config_data, tmp)
    
    try:
        # Test loading the config
        loaded_config = load_path(tmp.name)
        assert loaded_config["model"]["name"] == "test_model"
        assert loaded_config["training"]["batch_size"] == 32
    finally:
        # Clean up
        os.unlink(tmp.name)