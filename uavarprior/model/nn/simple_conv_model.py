#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example model implementation for UAVarPrior.

This module provides an example of a complete model implementation that
can be used with UAVarPrior's model factory system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleConvModel(nn.Module):
    """
    A simple convolutional neural network for sequence data
    
    This model takes sequence data (e.g., one-hot encoded DNA) and
    applies a series of convolutional layers followed by fully connected
    layers to predict outputs.
    """
    
    def __init__(
        self,
        input_channels=4,  # DNA has 4 channels (A, C, G, T)
        conv_channels=[16, 32, 64],
        kernel_size=3,
        pool_size=2,
        dropout=0.2,
        linear_features=[128, 64],
        output_features=1
    ):
        """
        Initialize the model
        
        Args:
            input_channels: Number of input channels (4 for DNA)
            conv_channels: List of channel sizes for each conv layer
            kernel_size: Size of convolutional kernels
            pool_size: Size of pooling windows
            dropout: Dropout probability
            linear_features: List of feature sizes for fully connected layers
            output_features: Number of output features
        """
        super().__init__()
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            in_channels = out_channels
        
        # Calculate size after convolutions and pooling
        self.pool = nn.MaxPool1d(pool_size)
        
        # Create linear layers
        self.fc_layers = nn.ModuleList()
        
        # First linear layer gets flattened conv output
        # We don't know the exact size yet, will set in forward()
        self.fc1_size = None
        
        # Create the rest of the FC layers
        for i in range(len(linear_features)):
            if i == 0:
                # Placeholder, will be set in forward()
                self.fc_layers.append(None)
            else:
                self.fc_layers.append(nn.Linear(linear_features[i-1], linear_features[i]))
        
        # Output layer
        if linear_features:
            self.output_layer = nn.Linear(linear_features[-1], output_features)
        else:
            # Direct connection from flattened conv output
            self.output_layer = None  # Will be set in forward()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, output_features)
        """
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = self.pool(x)
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Create first FC layer if it's the first call
        if self.fc_layers[0] is None:
            fc1_input_size = x.size(1)
            self.fc_layers[0] = nn.Linear(fc1_input_size, self.fc_layers[1].in_features)
            
            # If no FC layers in config, set output layer
            if self.output_layer is None:
                self.output_layer = nn.Linear(fc1_input_size, self.output_features)
        
        # Apply FC layers
        for i, fc in enumerate(self.fc_layers):
            if fc is not None:  # Skip None placeholders
                x = F.relu(fc(x))
                if i < len(self.fc_layers) - 1:  # Don't dropout after last FC layer
                    x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

# Factory functions required by UAVarPrior

def get_model(**kwargs):
    """
    Create and return a model instance
    
    This function is required by UAVarPrior's factory system.
    
    Args:
        **kwargs: Model parameters from configuration
        
    Returns:
        An initialized model
    """
    return SimpleConvModel(**kwargs)

def criterion(**kwargs):
    """
    Create and return a loss function
    
    This function is required by UAVarPrior's factory system.
    
    Args:
        **kwargs: Loss function parameters from configuration
        
    Returns:
        A loss function
    """
    # Default to BCE loss for binary classification
    if kwargs.get('task', 'binary') == 'binary':
        return nn.BCEWithLogitsLoss(**{k: v for k, v in kwargs.items() if k != 'task'})
    elif kwargs.get('task') == 'regression':
        return nn.MSELoss(**{k: v for k, v in kwargs.items() if k != 'task'})
    else:
        # Multi-class classification
        return nn.CrossEntropyLoss(**{k: v for k, v in kwargs.items() if k != 'task'})

def get_optimizer(lr):
    """
    Get optimizer class and arguments
    
    This function is required by UAVarPrior's factory system.
    
    Args:
        lr: Learning rate
        
    Returns:
        Tuple of (optimizer_class, optimizer_kwargs)
    """
    return (optim.Adam, {'lr': lr, 'weight_decay': 1e-4})
