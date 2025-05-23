#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for UAVarPrior.

This module provides utility functions used throughout the package.
"""
import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Set up logging configuration
    
    Args:
        level: Logging level
        log_file: Optional path to log file
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_device_info():
    """
    Get information about available computing devices
    
    Returns:
        Dictionary with device information
    """
    import torch
    
    device_info = {
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_device_cap"] = torch.cuda.get_device_capability(0)
    
    return device_info

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    return True