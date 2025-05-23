#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema validation for UAVarPrior configurations.

This module provides utilities for validating configuration dictionaries 
against JSON schemas.
"""
import os
import logging
import json
from typing import Dict, Any, Tuple, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    logger.warning("jsonschema package not installed. Schema validation will be disabled.")
    HAS_JSONSCHEMA = False

# Define schemas for configuration validation
MODEL_SCHEMA = {
    "type": "object",
    "required": ["class"],
    "properties": {
        "class": {"type": "string"},
        "path": {"type": "string"},
        "classArgs": {"type": "object"},
        "criterionArgs": {"type": "object"},
        "lr": {"type": ["number", "string"]},
    }
}

DATA_SCHEMA = {
    "type": "object",
    "required": ["dataset_class"],
    "properties": {
        "dataset_class": {"type": "string"},
        "batch_size": {"type": "integer", "minimum": 1},
        "num_workers": {"type": "integer", "minimum": 0},
        "pin_memory": {"type": "boolean"},
        "train_args": {"type": "object"},
        "val_args": {"type": "object"}
    }
}

TRAINING_SCHEMA = {
    "type": "object",
    "properties": {
        "epochs": {"type": "integer", "minimum": 1},
        "lr": {"type": ["number", "string"], "minimum": 0},
        "checkpoint_interval": {"type": "integer", "minimum": 1},
        "early_stopping": {"type": "object"}
    }
}

ANALYZER_SCHEMA = {
    "type": "object",
    "required": ["class"],
    "properties": {
        "class": {"type": "string"},
        "args": {"type": "object"}
    }
}

CONFIG_SCHEMA = {
    "type": "object",
    "required": ["ops"],
    "properties": {
        "ops": {
            "type": "array",
            "items": {"type": "string", "enum": ["train", "evaluate", "analyze"]}
        },
        "output_dir": {"type": "string"},
        "model": MODEL_SCHEMA,
        "data": DATA_SCHEMA,
        "training": TRAINING_SCHEMA,
        "analyzer": ANALYZER_SCHEMA,
        "prediction": {"type": "object"}
    }
}

def validate_against_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration dictionary against a JSON schema
    
    Args:
        config: Configuration dictionary
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (valid: bool, error_messages: List[str])
    """
    if not HAS_JSONSCHEMA:
        return True, ["Schema validation skipped: jsonschema package not installed"]
    
    try:
        jsonschema.validate(instance=config, schema=schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        return False, [f"Schema validation error: {e.message}"]

def validate_config_schema(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a complete configuration against the main schema
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (valid: bool, error_messages: List[str])
    """
    return validate_against_schema(config, CONFIG_SCHEMA)
