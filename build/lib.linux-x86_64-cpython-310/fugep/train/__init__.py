'''
Collection of implementations for model training
'''

from .trainer import StandardSGDTrainer
# from .trainer import DeepSVSDTrainer
from .utils import LossTracker
from .losses import weightedBCELoss

__all__ = ['StandardSGDTrainer', 
           'LossTracker',
           'weightedBCELoss']