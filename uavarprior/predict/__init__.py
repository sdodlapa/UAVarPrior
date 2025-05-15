"""
This module contains classes and methods for making and analyzing
predictions with models that have already been trained.
"""
from .pred import PeakEventPredictor
from .seq_ana import PeakISMSeqAnalyzer
from .seq_ana import PeakGVarEvaluator
from .seq_ana import MethylVarEvaluator

__all__ = ['PeakEventPredictor', 
           'PeakISMSeqAnalyzer',
           'PeakGVarEvaluator',
           'MethylVarEvaluator']
