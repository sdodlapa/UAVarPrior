#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model interface for UAVarPrior.

This module defines the base interface that all UAVarPrior models should implement.
"""
from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional, Union

class ModelInterface(ABC):
    """
    Base interface for all UAVarPrior models
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
