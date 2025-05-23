#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer class for UAVarPrior.

This module provides a Trainer class that handles the training loop and validation.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class Trainer:
    """
    Model trainer that handles the training loop and validation
    """
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        callbacks: List[Callable] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            callbacks: List of callback functions to call after each epoch
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.callbacks = callbacks or []
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Run the training loop"""
        logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = {}
            if self.val_dataloader:
                val_metrics = self.validate_epoch()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Log metrics
            metrics_str = f"Epoch {epoch}/{self.epochs} - Time: {epoch_time:.2f}s - "
            metrics_str += " - ".join([f"Train {k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                metrics_str += " - " + " - ".join([f"Val {k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(metrics_str)
            
            # Save checkpoint
            if self.checkpoint_dir:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                self.model.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Run callbacks
            for callback in self.callbacks:
                callback(epoch=epoch, model=self.model, train_metrics=train_metrics, val_metrics=val_metrics)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run a single training epoch
        
        Returns:
            Dictionary of averaged metrics for the epoch
        """
        self.model.backbone.train()
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_metrics = self.model.train_step(batch, self.device)
            
            # Update epoch metrics
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0
                epoch_metrics[k] += v
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_dataloader)
            
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Run validation for one epoch
        
        Returns:
            Dictionary of averaged metrics for the epoch
        """
        self.model.backbone.eval()
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.val_dataloader):
            step_metrics = self.model.eval_step(batch, self.device)
            
            # Update epoch metrics (excluding non-scalar outputs)
            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.val_dataloader)
            
        return epoch_metrics
