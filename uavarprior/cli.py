#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line interface for UAVarPrior.

A modern, extensible CLI that supports multiple command groups and operations.
"""
import os
import click
import logging
from typing import Optional, Tuple
import yaml
from pathlib import Path

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
def validate(config_path: str):
    """Validate configuration file without running the model."""
    try:
        configs = load_path(config_path, instantiate=False)
        from uavarprior.setup.run import validate_config
        valid, messages = validate_config(configs)
        
        if valid:
            click.echo("✅ Configuration is valid")
            for msg in messages:
                click.echo(f"  - {msg}")
        else:
            click.echo("❌ Configuration is invalid")
            for msg in messages:
                click.echo(f"  - {msg}")
            exit(1)
    except Exception as e:
        click.echo(f"Error validating configuration: {e}")
        exit(1)

if __name__ == "__main__":
    cli()