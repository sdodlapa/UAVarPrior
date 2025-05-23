#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis pipeline for UAVarPrior.

This module provides a high-level API for running analysis pipelines.
"""
import os
import logging
import importlib
from typing import Dict, Any, Optional, Union, List
import torch
import numpy as np

from uavarprior.model.factory import create_model

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """
    High-level API for running analysis pipelines
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        analyzer_config: Dict[str, Any],
        output_dir: Optional[str] = None
    ):
        """
        Initialize analysis pipeline
        
        Args:
            model_config: Model configuration
            analyzer_config: Analyzer configuration
            output_dir: Directory to save analysis results
        """
        self.model_config = model_config
        self.analyzer_config = analyzer_config
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create model
        self.model = create_model(model_config, train=False)
        
        # Load weights if specified
        if "path" in model_config:
            self.model.load(model_config["path"])
            logger.info(f"Loaded model weights from {model_config['path']}")
            
        # Create analyzer
        analyzer_class_path = analyzer_config["class"]
        try:
            module_path, class_name = analyzer_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            analyzer_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            # Try with alternative paths
            try:
                if not analyzer_class_path.startswith('uavarprior.'):
                    module_path = f"uavarprior.analysis.{analyzer_class_path}"
                    module = importlib.import_module(module_path)
                    analyzer_class = getattr(module, class_name)
            except (ValueError, ImportError, AttributeError):
                raise ValueError(f"Could not import analyzer class {analyzer_class_path}: {e}")
                
        analyzer_args = analyzer_config.get("args", {})
        self.analyzer = analyzer_class(**analyzer_args)
        logger.info(f"Created analyzer: {analyzer_class.__name__}")
    
    def run_analysis(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analysis pipeline
        
        Args:
            data_config: Configuration for input data
            
        Returns:
            Analysis results
        """
        # Prepare input data
        # (Implementation depends on specific data format)
        
        # Run analysis
        results = self.analyzer.analyze(self.model, data_config)
        
        # Save results if output directory is specified
        if self.output_dir:
            self._save_results(results)
            
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to output directory"""
        # Save different types of results
        for key, value in results.items():
            output_path = os.path.join(self.output_dir, f"{key}")
            
            # Handle different types of results
            if isinstance(value, np.ndarray):
                np.save(f"{output_path}.npy", value)
            elif isinstance(value, torch.Tensor):
                torch.save(value, f"{output_path}.pt")
            elif isinstance(value, dict):
                import json
                with open(f"{output_path}.json", "w") as f:
                    json.dump(value, f, indent=2)
            else:
                # For other types, try using pickle
                try:
                    import pickle
                    with open(f"{output_path}.pkl", "wb") as f:
                        pickle.dump(value, f)
                except Exception as e:
                    logger.warning(f"Could not save result '{key}': {e}")
        
        logger.info(f"Saved analysis results to {self.output_dir}")
