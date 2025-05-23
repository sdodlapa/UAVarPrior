#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory for UAVarPrior.

This module provides factory functions to create models based on configuration.
"""
import importlib
import logging
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim

from uavarprior.model.seq_model import SequenceModel

logger = logging.getLogger(__name__)

def create_model(config: Dict[str, Any], train: bool = True) -> SequenceModel:
    """
    Factory function to create models based on configuration
    
    Args:
        config: Model configuration dictionary
        train: Whether to set up the model for training
        
    Returns:
        An initialized model
    """
    # Import the model module
    model_class = config["class"]
    module_name = f"uavarprior.model.nn.{model_class.lower()}"
    
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        try:
            # Try without lowercase
            module_name = f"uavarprior.model.nn.{model_class}"
            module = importlib.import_module(module_name)
        except ImportError:
            # Try as a fully qualified module path
            try:
                module_path, class_name = model_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                get_model = getattr(module, f"get_{class_name.lower()}")
                criterion = getattr(module, "criterion")
                get_optimizer = getattr(module, "get_optimizer")
            except (ValueError, ImportError, AttributeError):
                raise ValueError(f"Could not import model module for {model_class}")
    
    # Create the backbone model
    model_args = config.get("classArgs", {})
    try:
        backbone = module.get_model(**model_args)
    except AttributeError:
        # Handle case where module has the model class directly
        backbone_class = getattr(module, model_class.split('.')[-1])
        backbone = backbone_class(**model_args)
    
    # Create loss function
    loss_args = config.get("criterionArgs", {})
    try:
        loss_fn = module.criterion(**loss_args)
    except AttributeError:
        # Handle case with standard loss function
        loss_fn = nn.MSELoss()  # Default loss
        logger.warning(f"No criterion found in module. Using default MSELoss.")
    
    # Create optimizer if in training mode
    optimizer = None
    if train:
        lr = config.get("lr")
        if lr is None and "training" in config:
            lr = config.get("training", {}).get("lr")
            
        if lr is not None:
            lr = float(lr)
            try:
                optimizer_class, optimizer_args = module.get_optimizer(lr)
                optimizer = optimizer_class(backbone.parameters(), **optimizer_args)
            except AttributeError:
                # Use default Adam optimizer
                optimizer = optim.Adam(backbone.parameters(), lr=lr)
                logger.warning(f"No get_optimizer found in module. Using default Adam optimizer.")
    
    # Create and return the model
    return SequenceModel(backbone, loss_fn, optimizer)
