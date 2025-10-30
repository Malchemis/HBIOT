"""Example model for testing pipeline implementation.

This module provides a simple mock model for testing and demonstrating
the pipeline structure without complex model logic.
"""

import torch
import torch.nn as nn
import logging


class MockModel(nn.Module):
    """Simple mock model for testing pipeline implementation."""

    def __init__(self, input_shape, output_shape, **kwargs):
        """Initialize the mock model.
        
        Args:
            input_shape: Input shape for one training sample, such as (C, T) -> channels, timesamples.
            output: Target output shape after forward (corresponding to labels if any), such as (L) -> labels.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.fc = nn.Linear(input_shape[0] * input_shape[1], output_shape[0])
        logging.info(f"MockModel initialized with input shape {input_shape} and output shape {output_shape}")

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (B, C, T) -> batch_size, channels, timesamples.
            
        Returns:
            Output tensor of shape (B, L) -> batch_size, labels.
        """
        logging.debug(f"Input shape: {x.shape}")
        x = x.view(x.size(0), -1)
        logging.debug(f"Reshaped input to: {x.shape}")
        x = self.fc(x)
        logging.debug(f"Output shape: {x.shape}")
        return x