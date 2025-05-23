#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run module for UAVarPrior.

This module provides functions for running operations based on configuration.
"""
import os
import sys
import logging
import importlib
from typing import Dict, Any, Tuple, List, Optional
import torch
from torch.utils.data import DataLoader

from uavarprior.setup.config import instantiate, load_path
from uavarprior.model.factory import create_model
from uavarprior.training.trainer import Trainer
from uavarprior.analysis.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)

def _import_class(class_path):
    """Helper to import a class from its string path"""
    if '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}. Must be in format 'module.Class'")
    
    module_path, class_name = class_path.rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Try alternative import path - prepend uavarprior
        if not module_path.startswith('uavarprior.'):
            try:
                alt_module_path = f"uavarprior.{module_path}"
                module = importlib.import_module(alt_module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass
        
        # Try src.uavarprior path
        if not module_path.startswith('src.uavarprior.'):
            try:
                alt_module_path = f"src.uavarprior.{module_path}"
                module = importlib.import_module(alt_module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass
        
        # If all attempts fail, raise the original error
        raise ValueError(f"Could not import {class_path}: {e}")

def setup_data_loaders(data_config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Set up data loaders from configuration
    
    Args:
        data_config: Data configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    dataset_class_name = data_config.get("dataset_class")
    if not dataset_class_name:
        raise ValueError("Missing dataset_class in data configuration")
        
    # Import dataset class
    dataset_class = _import_class(dataset_class_name)
    
    # Create datasets
    train_dataset = dataset_class(
        **data_config.get("train_args", {}),
        split="train"
    )
    
    val_dataset = None
    if "val_args" in data_config:
        val_dataset = dataset_class(
            **data_config.get("val_args", {}),
            split="val"
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get("batch_size", 32),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get("batch_size", 32),
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
            pin_memory=data_config.get("pin_memory", True)
        )
    
    return train_loader, val_loader

def execute(configs: Dict[str, Any]) -> None:
    """
    Execute operations based on configuration
    
    Args:
        configs: Full configuration dictionary
    """
    # Extract operations
    ops = configs.get('ops', [])
    if not ops:
        raise ValueError("No operations specified in configuration")
    
    logger.info(f"Executing operations: {', '.join(ops)}")
    
    # Create output directory
    output_dir = configs.get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle train operation
    if 'train' in ops:
        # Check for required configs
        if 'model' not in configs:
            raise ValueError("Missing 'model' configuration for train operation")
        if 'data' not in configs:
            raise ValueError("Missing 'data' configuration for train operation")
            
        # Create model
        model = create_model(configs['model'], train=True)
        
        # Setup data loaders
        train_loader, val_loader = setup_data_loaders(configs['data'])
        
        # Training configuration
        train_config = configs.get('training', {})
        epochs = train_config.get('epochs', 10)
        
        # Create and run trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            checkpoint_dir=output_dir
        )
        
        trainer.train()
    
    # Handle evaluate operation
    if 'evaluate' in ops:
        # Check for required configs
        if 'model' not in configs:
            raise ValueError("Missing 'model' configuration for evaluate operation")
        if 'data' not in configs:
            raise ValueError("Missing 'data' configuration for evaluate operation")
        
        # Create model
        model = create_model(configs['model'], train=False)
        
        # Load model weights if specified
        if 'model_path' in configs:
            model.load(configs['model_path'])
        
        # Setup data loader (use validation data)
        _, val_loader = setup_data_loaders(configs['data'])
        if not val_loader:
            raise ValueError("No validation data available for evaluation")
        
        # Run evaluation
        model.backbone.eval()
        metrics = {}
        
        for batch_idx, batch in enumerate(val_loader):
            step_metrics = model.eval_step(batch)
            
            # Update metrics
            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    if k not in metrics:
                        metrics[k] = 0
                    metrics[k] += v
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(val_loader)
        
        # Log metrics
        logger.info("Evaluation results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    # Handle analyze operation
    if 'analyze' in ops:
        # Check for required configs
        if 'model' not in configs:
            raise ValueError("Missing 'model' configuration for analyze operation")
        if 'analyzer' not in configs:
            raise ValueError("Missing 'analyzer' configuration for analyze operation")
        
        # Create analysis pipeline
        pipeline = AnalysisPipeline(
            model_config=configs['model'],
            analyzer_config=configs['analyzer'],
            output_dir=output_dir
        )
        
        # Run analysis
        data_config = configs.get('prediction', {})
        results = pipeline.run_analysis(data_config)
        
        # Log results summary
        logger.info("Analysis completed")
        if isinstance(results, dict):
            logger.info(f"Generated {len(results)} result items")
    
    logger.info("All operations completed successfully")

def validate_config(configs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration without running it
    
    Args:
        configs: Configuration dictionary
        
    Returns:
        Tuple of (valid: bool, messages: List[str])
    """
    messages = []
    valid = True
    
    # Check for required keys
    if 'ops' not in configs:
        valid = False
        messages.append("Missing 'ops' key in configuration")
    
    # Validate specific operation configs
    ops = configs.get('ops', [])
    
    if 'train' in ops:
        if 'model' not in configs:
            valid = False
            messages.append("Missing 'model' configuration for 'train' operation")
        if 'data' not in configs:
            valid = False
            messages.append("Missing 'data' configuration for 'train' operation")
    
    if 'evaluate' in ops:
        if 'model' not in configs:
            valid = False
            messages.append("Missing 'model' configuration for 'evaluate' operation")
        if 'data' not in configs:
            valid = False
            messages.append("Missing 'data' configuration for 'evaluate' operation")
    
    if 'analyze' in ops:
        if 'analyzer' not in configs:
            valid = False
            messages.append("Missing 'analyzer' configuration for 'analyze' operation")
        elif 'class' not in configs['analyzer']:
            valid = False
            messages.append("Missing 'class' in analyzer configuration")
        if 'prediction' not in configs:
            valid = False
            messages.append("Missing 'prediction' configuration for 'analyze' operation")
    
    # If valid so far, add some positive messages
    if valid:
        messages.append(f"Found operations: {', '.join(ops)}")
        if 'output_dir' in configs:
            messages.append(f"Output directory: {configs['output_dir']}")
    
    return valid, messages

def parse_configs_and_run(configs: Dict[str, Any]) -> None:
    """
    Parse configurations and run operations
    
    Args:
        configs: Configuration dictionary parsed from YAML
    """
    # Check if we have a legacy config structure
    if 'model' in configs and 'wrapper' in configs['model']:
        # Convert legacy wrapper configuration to new format
        logger.warning("Detected legacy wrapper configuration. Converting to new format.")
        
        # Extract old model wrapper info
        wrapper_class = configs['model']['wrapper']
        model_built = configs['model'].get('built', 'pytorch')
        mult_predictions = configs['model'].get('mult_predictions', 1)
        
        # Keep original model info but remove wrapper-specific keys
        model_config = configs['model'].copy()
        if 'wrapper' in model_config:
            del model_config['wrapper']
            
        # Create updated config without legacy wrappers
        configs['model'] = model_config
    
    # Execute operations with new or converted config
    execute(configs)