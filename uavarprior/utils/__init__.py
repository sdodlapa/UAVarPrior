#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for UAVarPrior.

This module provides utility functions used throughout the package.
"""

from .logging import setup_logging
import numpy as np
import os
import sys
import logging

def load_features_list(input_path):
    """
    Load features list from file
    
    Args:
        input_path: Path to file containing features list
        
    Returns:
        List of features
    """
    features = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features