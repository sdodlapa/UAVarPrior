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
    if not config:
        raise ValueError("Model configuration is empty")
    
    # Check for required configuration fields
    if "class" not in config:
        raise ValueError("Model configuration missing 'class' field")
    
    # Import the model module
    model_class = config["class"]
    logger.info(f"Creating model of class: {model_class}")
    
    module_name = f"uavarprior.model.nn.{model_class.lower()}"
    module = None
    
    # Try different module import strategies
    import_errors = []
    
    try:
        # Strategy 1: Use lowercase module name in standard location
        module = importlib.import_module(module_name)
        logger.info(f"Successfully imported module: {module_name}")
    except ImportError as e:
        import_errors.append(f"Failed to import {module_name}: {str(e)}")
        
        try:
            # Strategy 2: Use exact module name case in standard location
            module_name = f"uavarprior.model.nn.{model_class}"
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported module: {module_name}")
        except ImportError as e:
            import_errors.append(f"Failed to import {module_name}: {str(e)}")
            
            try:
                # Strategy 3: Try as a fully qualified module path
                if '.' in model_class:
                    module_path, class_name = model_class.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    logger.info(f"Successfully imported module from qualified path: {module_path}")
                else:
                    raise ImportError(f"Not a qualified path: {model_class}")
            except ImportError as e:
                import_errors.append(f"Failed to import {model_class} as qualified path: {str(e)}")
                
                # If all standard imports fail, try one last approach with standard PyTorch models
                try:
                    if model_class in dir(torch.nn):
                        logger.info(f"Using PyTorch built-in model: {model_class}")
                        module = torch.nn
                    else:
                        raise ImportError(f"Not a PyTorch built-in model: {model_class}")
                except ImportError as e:
                    import_errors.append(f"Failed to use as PyTorch model: {str(e)}")
                    error_msg = "\n".join(import_errors)
                    raise ValueError(f"Could not import model module for {model_class}. Errors:\n{error_msg}")
    
    # Create the backbone model
    model_args = config.get("classArgs", {})
    logger.info(f"Creating backbone with args: {model_args}")
    
    try:
        # Strategy 1: Use get_model function
        if hasattr(module, 'get_model'):
            backbone = module.get_model(**model_args)
            logger.info("Created backbone using get_model() function")
        # Strategy 2: Use the class directly
        elif hasattr(module, model_class.split('.')[-1]):
            backbone_class = getattr(module, model_class.split('.')[-1])
            backbone = backbone_class(**model_args)
            logger.info(f"Created backbone using class {model_class.split('.')[-1]} directly")
        # Strategy 3: For PyTorch built-ins
        elif model_class in dir(torch.nn):
            backbone_class = getattr(torch.nn, model_class)
            backbone = backbone_class(**model_args)
            logger.info(f"Created backbone using PyTorch built-in {model_class}")
        else:
            raise AttributeError(f"No model constructor found for {model_class}")
    except Exception as e:
        raise ValueError(f"Failed to create backbone model: {str(e)}")
    
    # Verify the backbone is a valid PyTorch module
    if not isinstance(backbone, nn.Module):
        raise TypeError(f"Created backbone is not a torch.nn.Module but {type(backbone)}")
    
    # Create loss function
    loss_args = config.get("criterionArgs", {})
    logger.info(f"Setting up loss function with args: {loss_args}")
    
    try:
        # Strategy 1: Use criterion function from the module
        if hasattr(module, 'criterion'):
            loss_fn = module.criterion(**loss_args)
            logger.info("Using criterion() function from module")
        # Strategy 2: Fall back to MSELoss
        else:
            loss_fn = nn.MSELoss()
            logger.warning(f"No criterion found in module. Using default MSELoss.")
    except Exception as e:
        logger.warning(f"Error creating loss function: {str(e)}, falling back to MSELoss")
        loss_fn = nn.MSELoss()
    
    # Create optimizer if in training mode
    optimizer = None
    if train:
        logger.info("Setting up optimizer for training")
        
        # Determine learning rate from various sources
        lr = config.get("lr")
        if lr is None and "training" in config:
            lr = config.get("training", {}).get("lr")
            
        if lr is not None:
            try:
                lr = float(lr)
                logger.info(f"Using learning rate: {lr}")
                
                try:
                    # Strategy 1: Use get_optimizer from module
                    if hasattr(module, 'get_optimizer'):
                        optimizer_class, optimizer_args = module.get_optimizer(lr)
                        optimizer = optimizer_class(backbone.parameters(), **optimizer_args)
                        logger.info(f"Created optimizer: {optimizer_class.__name__} with args {optimizer_args}")
                    # Strategy 2: Fall back to Adam optimizer
                    else:
                        optimizer = optim.Adam(backbone.parameters(), lr=lr)
                        logger.warning(f"No get_optimizer found in module. Using default Adam optimizer with lr={lr}")
                except Exception as e:
                    logger.warning(f"Error creating optimizer: {str(e)}, falling back to Adam")
                    optimizer = optim.Adam(backbone.parameters(), lr=lr)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid learning rate '{lr}': {str(e)}")
        else:
            logger.warning("No learning rate specified, skipping optimizer creation")
    
    # Create and return the model
    logger.info(f"Model creation complete: backbone={type(backbone).__name__}, optimizer={type(optimizer).__name__ if optimizer else None}")
    return SequenceModel(backbone, loss_fn, optimizer)
