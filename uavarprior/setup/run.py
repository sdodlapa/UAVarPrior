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
    
    # Validate schema first
    try:
        from uavarprior.setup.schema import validate_config_schema
        schema_valid, schema_errors = validate_config_schema(configs)
        if not schema_valid:
            valid = False
            messages.extend(schema_errors)
    except ImportError:
        messages.append("Schema validation skipped: could not import validation module")
    
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
        elif not validate_model_config(configs['model'], messages):
            valid = False
            
        if 'data' not in configs:
            valid = False
            messages.append("Missing 'data' configuration for 'train' operation")
        elif not validate_data_config(configs['data'], messages):
            valid = False
            
        # Validate training parameters
        if 'training' in configs and not validate_training_config(configs['training'], messages):
            valid = False
    
    if 'evaluate' in ops:
        if 'model' not in configs:
            valid = False
            messages.append("Missing 'model' configuration for 'evaluate' operation")
        elif not validate_model_config(configs['model'], messages):
            valid = False
            
        if 'data' not in configs:
            valid = False
            messages.append("Missing 'data' configuration for 'evaluate' operation")
        elif not validate_data_config(configs['data'], messages):
            valid = False
    
    if 'analyze' in ops:
        if 'model' not in configs:
            valid = False
            messages.append("Missing 'model' configuration for 'analyze' operation")
        elif not validate_model_config(configs['model'], messages, require_path=True):
            valid = False
            
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

def validate_model_config(model_config: Dict[str, Any], messages: List[str], require_path: bool = False) -> bool:
    """
    Validate model configuration
    
    Args:
        model_config: Model configuration section
        messages: List to append validation messages to
        require_path: Whether to require a path to a saved model
        
    Returns:
        Whether the configuration is valid
    """
    valid = True
    
    # Check for required class
    if 'class' not in model_config:
        valid = False
        messages.append("Missing 'class' in model configuration")
    else:
        model_class = model_config['class']
        messages.append(f"Model class: {model_class}")
    
    # Check for path if required for loading
    if require_path and 'path' not in model_config:
        valid = False
        messages.append("Missing 'path' in model configuration. A saved model path is required for this operation.")
    
    # Validate class arguments if present
    if 'classArgs' in model_config and not isinstance(model_config['classArgs'], dict):
        valid = False
        messages.append("'classArgs' in model configuration must be a dictionary")
    
    # Check for criterion arguments if present
    if 'criterionArgs' in model_config and not isinstance(model_config['criterionArgs'], dict):
        valid = False
        messages.append("'criterionArgs' in model configuration must be a dictionary")
    
    return valid

def validate_data_config(data_config: Dict[str, Any], messages: List[str]) -> bool:
    """
    Validate data configuration
    
    Args:
        data_config: Data configuration section
        messages: List to append validation messages to
        
    Returns:
        Whether the configuration is valid
    """
    valid = True
    
    # Check for dataset class
    if 'dataset_class' not in data_config:
        valid = False
        messages.append("Missing 'dataset_class' in data configuration")
    
    # Check batch size is valid
    if 'batch_size' in data_config:
        try:
            batch_size = int(data_config['batch_size'])
            if batch_size <= 0:
                valid = False
                messages.append("'batch_size' must be a positive integer")
        except (ValueError, TypeError):
            valid = False
            messages.append("'batch_size' must be a valid integer")
    
    # Validate train_args if present
    if 'train_args' in data_config and not isinstance(data_config['train_args'], dict):
        valid = False
        messages.append("'train_args' in data configuration must be a dictionary")
    
    # Validate val_args if present
    if 'val_args' in data_config and not isinstance(data_config['val_args'], dict):
        valid = False
        messages.append("'val_args' in data configuration must be a dictionary")
    
    return valid

def validate_training_config(training_config: Dict[str, Any], messages: List[str]) -> bool:
    """
    Validate training configuration
    
    Args:
        training_config: Training configuration section
        messages: List to append validation messages to
        
    Returns:
        Whether the configuration is valid
    """
    valid = True
    
    # Check for learning rate
    if 'lr' in training_config:
        try:
            lr = float(training_config['lr'])
            if lr <= 0:
                valid = False
                messages.append("'lr' must be a positive number")
        except (ValueError, TypeError):
            valid = False
            messages.append("'lr' must be a valid number")
    
    # Check for epochs
    if 'epochs' in training_config:
        try:
            epochs = int(training_config['epochs'])
            if epochs <= 0:
                valid = False
                messages.append("'epochs' must be a positive integer")
        except (ValueError, TypeError):
            valid = False
            messages.append("'epochs' must be a valid integer")
    
    return valid

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

def test_config_initialization(configs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Test initialization of components from configuration
    
    Args:
        configs: Configuration dictionary
        
    Returns:
        Dictionary with initialization status for each component
    """
    results = {}
    
    # Try to initialize model if present
    if 'model' in configs:
        results['model'] = {'success': False}
        try:
            from uavarprior.model.factory import create_model
            model = create_model(configs['model'], train='train' in configs.get('ops', []))
            results['model'] = {
                'success': True,
                'details': [
                    f"Model type: {type(model).__name__}",
                    f"Backbone: {type(model.backbone).__name__}",
                    f"Device: {model.device}"
                ]
            }
        except Exception as e:
            results['model'] = {
                'success': False,
                'error': str(e)
            }
    
    # Try to initialize data loaders if present
    if 'data' in configs:
        results['data_loaders'] = {'success': False}
        try:
            train_loader, val_loader = setup_data_loaders(configs['data'])
            details = []
            
            if train_loader:
                details.append(f"Train loader: {len(train_loader)} batches, batch size: {train_loader.batch_size}")
            
            if val_loader:
                details.append(f"Validation loader: {len(val_loader)} batches, batch size: {val_loader.batch_size}")
                
            results['data_loaders'] = {
                'success': True,
                'details': details
            }
        except Exception as e:
            results['data_loaders'] = {
                'success': False,
                'error': str(e)
            }
    
    # Try to initialize analyzer if present
    if 'analyzer' in configs:
        results['analyzer'] = {'success': False}
        try:
            analyzer_config = configs['analyzer']
            analyzer_class_path = analyzer_config.get('class')
            
            if analyzer_class_path:
                module_path, class_name = analyzer_class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                analyzer_class = getattr(module, class_name)
                
                analyzer_args = analyzer_config.get('args', {})
                analyzer = analyzer_class(**analyzer_args)
                
                results['analyzer'] = {
                    'success': True,
                    'details': [
                        f"Analyzer type: {type(analyzer).__name__}",
                    ]
                }
            else:
                results['analyzer'] = {
                    'success': False,
                    'error': "Missing 'class' in analyzer configuration"
                }
        except Exception as e:
            results['analyzer'] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def test_config_initialization(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Test initialization of configuration components without running operations
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of initialization results for each component
    """
    results = {}
    
    # Test model initialization
    if 'model' in config:
        try:
            model = create_model(config['model'], train=False)
            results['model'] = {
                'success': True,
                'details': [
                    f"Successfully created model of class {config['model']['class']}",
                    f"Model parameters: {sum(p.numel() for p in model.backbone.parameters()):,}"
                ]
            }
        except Exception as e:
            results['model'] = {
                'success': False,
                'error': str(e)
            }
    
    # Test data loaders
    if 'data' in config:
        try:
            train_loader, val_loader = setup_data_loaders(config['data'])
            data_details = [
                f"Successfully created dataset of class {config['data']['dataset_class']}",
                f"Training batches: {len(train_loader)}"
            ]
            if val_loader:
                data_details.append(f"Validation batches: {len(val_loader)}")
            
            # Try loading a batch
            x, y = next(iter(train_loader))
            data_details.append(f"Input batch shape: {x.shape}")
            data_details.append(f"Target batch shape: {y.shape}")
            
            results['data'] = {
                'success': True,
                'details': data_details
            }
        except Exception as e:
            results['data'] = {
                'success': False,
                'error': str(e)
            }
    
    # Test analyzer
    if 'analyzer' in config:
        try:
            analyzer_config = config['analyzer']
            analyzer_class_path = analyzer_config["class"]
            module_path, class_name = analyzer_class_path.rsplit('.', 1)
            
            # Try to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # Try with alternative paths
                if not analyzer_class_path.startswith('uavarprior.'):
                    module_path = f"uavarprior.analysis.{analyzer_class_path}"
                    module = importlib.import_module(module_path)
            
            # Get the class
            analyzer_class = getattr(module, class_name)
            
            # Create an instance
            analyzer_args = analyzer_config.get("args", {})
            analyzer = analyzer_class(**analyzer_args)
            
            results['analyzer'] = {
                'success': True,
                'details': [
                    f"Successfully created analyzer of class {class_name}"
                ]
            }
        except Exception as e:
            results['analyzer'] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration before running
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (valid, messages)
    """
    from uavarprior.setup.schema import validate_config_schema
    
    messages = []
    
    # Validate against schema
    schema_valid, schema_messages = validate_config_schema(config)
    messages.extend(schema_messages)
    
    # Basic operational validation
    ops = config.get('ops', [])
    if not ops:
        messages.append("No operations specified in configuration")
    else:
        for op in ops:
            if op not in ['train', 'evaluate', 'analyze']:
                messages.append(f"Unknown operation: {op}")
    
    # Output directory
    if 'output_dir' in config:
        output_dir = config['output_dir']
        if output_dir:
            # Check if parent directory exists and is writable
            parent_dir = os.path.dirname(os.path.abspath(output_dir))
            if not os.path.exists(parent_dir):
                messages.append(f"Parent directory does not exist: {parent_dir}")
            elif not os.access(parent_dir, os.W_OK):
                messages.append(f"Parent directory is not writable: {parent_dir}")
    
    # Model validation
    if 'model' in config:
        model_config = config['model']
        if 'class' not in model_config:
            messages.append("Model configuration missing 'class' field")
        
        if 'path' in model_config:
            path = model_config['path']
            if not os.path.exists(path):
                messages.append(f"Model weights file does not exist: {path}")
    
    # Data validation
    if 'data' in config:
        data_config = config['data']
        if 'dataset_class' not in data_config:
            messages.append("Data configuration missing 'dataset_class' field")
        
        for split in ['train_args', 'val_args']:
            if split in data_config and 'data_path' in data_config[split]:
                path = data_config[split]['data_path']
                if not os.path.exists(path):
                    messages.append(f"Data file does not exist: {path} (for {split})")
    
    # Check if operations have required configurations
    if 'train' in ops:
        if 'model' not in config:
            messages.append("Missing 'model' configuration for train operation")
        if 'data' not in config:
            messages.append("Missing 'data' configuration for train operation")
    
    if 'evaluate' in ops:
        if 'model' not in config:
            messages.append("Missing 'model' configuration for evaluate operation")
        if 'data' not in config:
            messages.append("Missing 'data' configuration for evaluate operation")
    
    if 'analyze' in ops:
        if 'model' not in config:
            messages.append("Missing 'model' configuration for analyze operation")
        if 'analyzer' not in config:
            messages.append("Missing 'analyzer' configuration for analyze operation")
    
    # Return validation result
    return len([m for m in messages if "does not exist" in m or "missing" in m.lower()]) == 0, messages

def parse_configs_and_run(configs: Dict[str, Any]) -> None:
    """
    Parse configurations and run operations
    
    Args:
        configs: Configuration dictionary
    """
    # Validate the configuration
    valid, messages = validate_config(configs)
    
    if not valid:
        error_messages = [m for m in messages if "does not exist" in m or "missing" in m.lower()]
        raise ValueError(f"Invalid configuration: {'; '.join(error_messages)}")
    
    # Execute operations
    execute(configs)