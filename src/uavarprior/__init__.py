"""
This is the main module for UAVarPrior.
"""

from .version import __version__
from .evaluate import ModelEvaluator
from .cli import cli as main

__all__ = ["data", "model", "samplers", "utils",
           'train', "predict", "interpret", "__version__",
           'ModelEvaluator', 'main']