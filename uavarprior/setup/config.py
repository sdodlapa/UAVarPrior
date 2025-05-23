#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities for UAVarPrior.

This module provides utilities for loading and processing configurations.
"""
import os
import yaml
from typing import Dict, Any

def load_path(path: str, instantiate: bool = False) -> Dict[str, Any]:
    """
    Load YAML configuration from path
    
    Args:
        path: Path to YAML configuration file
        instantiate: Whether to instantiate objects in the configuration
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if instantiate:
        config = instantiate_config(config)
    
    return config

def instantiate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively instantiate objects in configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with instantiated objects
    """
    # This is a placeholder for the actual implementation
    # which would recursively process the config and instantiate
    # any classes specified in the configuration
    return config
