#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model interface for UAVarPrior.

This module defines the base interface that all UAVarPrior models should implement.

When creating a custom model for UAVarPrior, you should:
1. Create a class that extends ModelInterface
2. Implement all required methods
3. Register your model in the factory system

Example implementation:
```python
class MyCustomModel(ModelInterface):
    def __init__(self, backbone, loss_fn, optimizer=None):
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone.to(self.device)
        
    def forward(self, x):
        return self.backbone(x)
        
    # Implement all other required methods
```

Then, create a module that provides factory functions:
```python
# In your_model_module.py
def get_model(**args):
    # Create and return your PyTorch model
    return YourBackboneModel(**args)
    
def criterion(**args):
    # Create and return your loss function
    return YourLossFunction(**args)
    
def get_optimizer(lr):
    # Return optimizer class and arguments
    return (torch.optim.Adam, {'lr': lr, 'weight_decay': 1e-4})
```
"""
from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional, Union

class ModelInterface(ABC):
    """
    Base interface for all UAVarPrior models
    
    This abstract class defines the interface that all models in UAVarPrior
    must implement. It provides a consistent API for training, evaluation,
    prediction, and model saving/loading.
    
    Implementation note:
    - All models must have a 'backbone' attribute containing the PyTorch nn.Module
    - All models must handle device placement (CPU/GPU)
    - All models must implement serialization methods
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass
        
    @abstractmethod
    def train_step(self, batch, device=None) -> Dict[str, Any]:
        """
        Perform a single training step
        
        Args:
            batch: The input batch
            device: Optional device to use
            
        Returns:
            Dictionary of metrics from this step
        """
        pass
        
    @abstractmethod
    def eval_step(self, batch, device=None) -> Dict[str, Any]:
        """
        Perform a single evaluation step
        
        Args:
            batch: The input batch
            device: Optional device to use
            
        Returns:
            Dictionary of metrics and outputs from this step
        """
        pass
        
    @abstractmethod
    def predict(self, batch, device=None) -> torch.Tensor:
        """
        Generate predictions
        
        Args:
            batch: The input batch
            device: Optional device to use
            
        Returns:
            Model predictions
        """
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model weights
        
        Args:
            path: Path to save model weights
        """
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model weights
        
        Args:
            path: Path to load model weights from
        """
        pass
