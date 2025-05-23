#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence model implementation for UAVarPrior.

This module provides a concrete implementation of the ModelInterface for sequence data.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Callable
from uavarprior.model.interface import ModelInterface

class SequenceModel(ModelInterface):
    """
    Sequence model implementation that handles standard DNA/RNA sequence inputs
    """
    
    def __init__(
        self, 
        backbone: nn.Module, 
        loss_fn: Callable, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a sequence model
        
        Args:
            backbone: Neural network model
            loss_fn: Loss function
            optimizer: Optional optimizer for training
            device: Device to run model on
        """
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone.to(self.device)
        
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
        
    def train_step(self, batch, device=None) -> Dict[str, Any]:
        """Perform a single training step"""
        device = device or self.device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        self.optimizer.zero_grad()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
        
    def eval_step(self, batch, device=None) -> Dict[str, Any]:
        """Perform a single evaluation step"""
        device = device or self.device
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            outputs = self(x)
            loss = self.loss_fn(outputs, y)
            
        return {'loss': loss.item(), 'outputs': outputs.cpu(), 'targets': y.cpu()}
        
    def predict(self, batch, device=None) -> torch.Tensor:
        """Generate predictions"""
        device = device or self.device
        x = batch
        if isinstance(x, tuple) and len(x) >= 1:
            # Handle case where batch is (x, y) or similar
            x = x[0]
            
        x = x.to(device)
        
        with torch.no_grad():
            outputs = self(x)
            
        return outputs.cpu()
        
    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'model_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, path)
        
    def load(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path)
        self.backbone.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
