"""Shallow Fully Convolutional Network (SFCN) for MEG spike detection.

This module implements a shallow CNN architecture with batch normalization
and configurable convolutional layers for processing MEG data.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SFCN(nn.Module):
    """Shallow Fully Convolutional Network for MEG classification.

    A lightweight CNN architecture with multiple convolutional layers,
    batch normalization, and pooling operations.
    """
    def __init__(
        self, 
        n_classes=1, 
        conv_channels=[32, 64, 128, 256, 64],
        conv_kernel_sizes=[5, 5, 5, 5, 1],
        pool_kernel_size=2,
        dropout=0.5,
        **kwargs
    ):
        super(SFCN, self).__init__()
        
        # Store configurable parameters
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.n_classes = n_classes
        
        # Build convolutional layers dynamically
        in_channels = 1
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            
        self.fc1 = nn.LazyLinear(n_classes)

    def forward(self, x):
        """Forward pass of the SFCN.

        Args:
            x: Input tensor of shape (batch_size, n_sensors, n_samples_per_window).

        Returns:
            Output tensor of shape (batch_size, n_classes).
        """
        x = x.unsqueeze(1)
        
        # Apply convolutional layers dynamically
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x)
            
            # Apply pooling based on layer position
            if i < len(self.conv_layers) - 1:  # Max pooling for all but last layer
                x = F.max_pool2d(x, self.pool_kernel_size)
            else:  # Average pooling for last layer
                x = F.avg_pool2d(x, self.pool_kernel_size)
                
        emb = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.dropout(emb, p=self.dropout)
        x = torch.sigmoid(self.fc1(x))  # output (batch_size, n_classes)
        return x


if __name__ == "__main__":
    # Example usage
    model = SFCN()
    input_tensor = torch.randn(32, 80, 275)
    output = model(input_tensor)
    print(output)
