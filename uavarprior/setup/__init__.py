"""
Setup module for UAVarPrior.

This module provides utilities for loading and executing configurations.
"""
from uavarprior.setup.run import parse_configs_and_run, validate_config

# Import load_path from FuGEP's config module for backwards compatibility
try:
    from uavarprior.setup.config import load_path, instantiate
except ImportError:
    # If this fails, try importing from fugep
    try:
        from fugep.setup.config import load_path, instantiate
    except ImportError:
        # Define a simple load_path function as fallback
        import yaml
        def load_path(path, instantiate=False):
            """Load YAML configuration from path"""
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config