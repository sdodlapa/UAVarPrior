#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line interface for UAVarPrior.

A modern, extensible CLI that supports multiple command groups and operations.
"""
import os
import sys
import click
import logging
from typing import Optional, Tuple
import yaml
from pathlib import Path
import importlib

from uavarprior import __version__
from uavarprior.setup import load_path, parse_configs_and_run
from uavarprior.utils import setup_logging

@click.group()
@click.version_option(__version__)
@click.option('--verbose', '-v', count=True, help='Increase verbosity (can be used multiple times)')
@click.option('--cuda-blocking', is_flag=True, help='Enable CUDA launch blocking for debugging')
def cli(verbose: int, cuda_blocking: bool):
    """UAVarPrior - Uncertainty-Aware Variational Prior framework."""
    # Setup logging based on verbosity
    log_level = max(logging.WARNING - verbose*10, logging.DEBUG)
    setup_logging(log_level)
    
    if cuda_blocking:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--override', '-o', multiple=True, help='Override config parameters: -o section.param=value')
@click.option('--lr', type=float, help='Learning rate override (for training operations)')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed error tracking')
def run(config_path: str, override: Optional[Tuple[str]]=None, lr: Optional[float]=None, debug: bool=False):
    """Run model training, evaluation, or inference using the specified configuration file."""
    configs = load_path(config_path, instantiate=False)
    
    # Handle overrides
    if override:
        for o in override:
            if '=' not in o:
                click.echo(f"Warning: Ignoring invalid override '{o}'. Format should be 'param=value'")
                continue
            key, value = o.split('=', 1)
            # Parse value to appropriate type
            try:
                # Try as number or boolean
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value and value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value.isdigit():
                    value = int(value)
            except ValueError:
                pass  # Keep as string if parsing fails
                
            # Handle nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                current = configs
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                configs[key] = value
    
    # Handle learning rate override
    if lr is not None:
        if 'training' in configs:
            configs['training']['lr'] = lr
        else:
            configs['lr'] = lr
            
    try:
        parse_configs_and_run(configs)
    except Exception as e:
        if debug:
            click.echo(f"Error: {e}")
            import traceback
            traceback.print_exc()
        else:
            click.echo(f"Error: {str(e)}")
            click.echo("Run with --debug for more information")
        exit(1)

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--dry-run', is_flag=True, help='Create model and data loaders without training')
def validate(config_path: str, dry_run: bool = False):
    """Validate configuration file without running the model.
    
    If --dry-run is specified, also attempts to initialize model and data loaders.
    """
    try:
        configs = load_path(config_path, instantiate=False)
        from uavarprior.setup.run import validate_config, test_config_initialization
        
        # Basic validation
        valid, messages = validate_config(configs)
        
        if not valid:
            click.echo("❌ Configuration is invalid")
            for msg in messages:
                click.echo(f"  - {msg}")
            exit(1)
        
        # Print validation messages
        click.echo("✅ Configuration is valid")
        for msg in messages:
            click.echo(f"  - {msg}")
        
        # If dry-run is specified, attempt to initialize components
        if dry_run:
            click.echo("\nPerforming dry-run initialization:")
            init_results = test_config_initialization(configs)
            
            # Print initialization results
            for component, status in init_results.items():
                if status['success']:
                    click.echo(f"✅ Successfully initialized {component}")
                    if 'details' in status:
                        for detail in status['details']:
                            click.echo(f"  - {detail}")
                else:
                    click.echo(f"❌ Failed to initialize {component}")
                    click.echo(f"  - Error: {status['error']}")
    except Exception as e:
        click.echo(f"Error validating configuration: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        exit(1)

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', type=click.Path(), help='Path to write expanded config with defaults')
def debug_config(config_path: str, output: Optional[str] = None):
    """Debug a configuration file by showing resolved values and importing modules.
    
    This command loads the configuration file, resolves all default values, checks
    if modules can be imported, and prints a detailed report.
    """
    try:
        # Load the configuration
        configs = load_path(config_path, instantiate=False)
        
        # Set up debug logging
        setup_logging(logging.DEBUG)
        
        click.echo(f"Configuration file: {config_path}")
        click.echo("=" * 50)
        
        # Print basic info
        click.echo(f"Operations: {configs.get('ops', [])}")
        click.echo(f"Output directory: {configs.get('output_dir', 'Not specified')}")
        click.echo("=" * 50)
        
        # Check model configuration
        if 'model' in configs:
            model_config = configs['model']
            click.echo("Model Configuration:")
            click.echo(f"  Class: {model_config.get('class', 'Not specified')}")
            
            # Try to import the model module
            try:
                model_class = model_config.get('class')
                if model_class:
                    module_name = f"uavarprior.model.nn.{model_class.lower()}"
                    click.echo(f"  Trying to import module: {module_name}")
                    try:
                        module = importlib.import_module(module_name)
                        click.echo(f"  ✅ Successfully imported {module_name}")
                        
                        # Check for required functions
                        if hasattr(module, 'get_model'):
                            click.echo(f"  ✅ Found get_model() function")
                        else:
                            click.echo(f"  ❌ Missing get_model() function")
                            
                        if hasattr(module, 'criterion'):
                            click.echo(f"  ✅ Found criterion() function")
                        else:
                            click.echo(f"  ❌ Missing criterion() function")
                            
                        if hasattr(module, 'get_optimizer'):
                            click.echo(f"  ✅ Found get_optimizer() function")
                        else:
                            click.echo(f"  ❌ Missing get_optimizer() function")
                            
                    except ImportError as e:
                        click.echo(f"  ❌ Failed to import {module_name}: {str(e)}")
                        # Try alternative import strategies
                        alt_module_name = f"uavarprior.model.nn.{model_class}"
                        click.echo(f"  Trying alternative import: {alt_module_name}")
                        try:
                            module = importlib.import_module(alt_module_name)
                            click.echo(f"  ✅ Successfully imported {alt_module_name}")
                        except ImportError as e2:
                            click.echo(f"  ❌ Failed to import {alt_module_name}: {str(e2)}")
                            
                            # Try as fully qualified path
                            if '.' in model_class:
                                module_path, class_name = model_class.rsplit('.', 1)
                                click.echo(f"  Trying as qualified path: {module_path}")
                                try:
                                    module = importlib.import_module(module_path)
                                    click.echo(f"  ✅ Successfully imported {module_path}")
                                except ImportError as e3:
                                    click.echo(f"  ❌ Failed to import {module_path}: {str(e3)}")
            except Exception as e:
                click.echo(f"  ❌ Error checking model module: {str(e)}")
                
            click.echo("  Model arguments:")
            for k, v in model_config.get('classArgs', {}).items():
                click.echo(f"    {k}: {v}")
                
            if 'path' in model_config:
                path = model_config['path']
                click.echo(f"  Model path: {path}")
                if os.path.exists(path):
                    click.echo(f"  ✅ Model file exists")
                else:
                    click.echo(f"  ❌ Model file does not exist")
                    
            click.echo("=" * 50)
        
        # Check data configuration
        if 'data' in configs:
            data_config = configs['data']
            click.echo("Data Configuration:")
            dataset_class = data_config.get('dataset_class')
            click.echo(f"  Dataset class: {dataset_class}")
            
            # Try to import the dataset class
            if dataset_class:
                try:
                    dataset_module_path, class_name = dataset_class.rsplit('.', 1)
                    click.echo(f"  Trying to import module: {dataset_module_path}")
                    try:
                        module = importlib.import_module(dataset_module_path)
                        click.echo(f"  ✅ Successfully imported {dataset_module_path}")
                        
                        if hasattr(module, class_name):
                            click.echo(f"  ✅ Found {class_name} class")
                        else:
                            click.echo(f"  ❌ Missing {class_name} class")
                    except ImportError as e:
                        click.echo(f"  ❌ Failed to import {dataset_module_path}: {str(e)}")
                except Exception as e:
                    click.echo(f"  ❌ Error parsing dataset class: {str(e)}")
                    
            # Check data paths
            for split in ['train_args', 'val_args']:
                if split in data_config:
                    split_args = data_config[split]
                    click.echo(f"  {split.replace('_args', '').title()} arguments:")
                    for k, v in split_args.items():
                        click.echo(f"    {k}: {v}")
                        
                    if 'data_path' in split_args:
                        path = split_args['data_path']
                        click.echo(f"    Data path: {path}")
                        if os.path.exists(path):
                            click.echo(f"    ✅ Data file exists")
                        else:
                            click.echo(f"    ❌ Data file does not exist")
                            
            click.echo("=" * 50)
        
        # Check analyzer configuration
        if 'analyzer' in configs:
            analyzer_config = configs['analyzer']
            click.echo("Analyzer Configuration:")
            analyzer_class = analyzer_config.get('class')
            click.echo(f"  Analyzer class: {analyzer_class}")
            
            # Try to import the analyzer class
            if analyzer_class:
                try:
                    module_path, class_name = analyzer_class.rsplit('.', 1)
                    click.echo(f"  Trying to import module: {module_path}")
                    try:
                        module = importlib.import_module(module_path)
                        click.echo(f"  ✅ Successfully imported {module_path}")
                        
                        if hasattr(module, class_name):
                            click.echo(f"  ✅ Found {class_name} class")
                        else:
                            click.echo(f"  ❌ Missing {class_name} class")
                    except ImportError as e:
                        click.echo(f"  ❌ Failed to import {module_path}: {str(e)}")
                except Exception as e:
                    click.echo(f"  ❌ Error parsing analyzer class: {str(e)}")
            
            click.echo("  Analyzer arguments:")
            for k, v in analyzer_config.get('args', {}).items():
                click.echo(f"    {k}: {v}")
                
            click.echo("=" * 50)
        
        # Write expanded config if requested
        if output:
            with open(output, 'w') as f:
                yaml.dump(configs, f, default_flow_style=False)
            click.echo(f"Expanded configuration written to {output}")
            
    except Exception as e:
        click.echo(f"Error debugging configuration: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    cli()