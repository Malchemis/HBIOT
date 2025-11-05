"""Simple Fully Convolutional Network (SFCN) for MEG spike detection."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SFCN(nn.Module):
    """Simple Fully Convolutional Network for MEG classification.
    
    Lightweight CNN with configurable normalization, activation, and initialization.
    """
    def __init__(
        self, 
        input_shape=(1, 275, 40),
        n_classes=1, 
        conv_channels=[32, 64, 128, 256, 64],
        conv_kernel_sizes=[5, 5, 5, 5, 1],
        pool_kernel_size=2,
        dropout=0.5,
        norm_type='batch',  # 'batch', 'layer'
        activation='leaky_relu',  # 'leaky_relu', 'gelu', 'silu'
        *args,
        **kwargs
    ):
        super().__init__()
        
        self.n_channels, self.n_samples = input_shape[1], input_shape[2]
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.n_classes = n_classes
        
        # Activation function
        self.act_fn = {'leaky_relu': F.leaky_relu, 'gelu': F.gelu, 'silu': F.silu}[activation]
        
        # Build conv layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels, kernel_size in zip(conv_channels, conv_kernel_sizes):
            # Disable bias when using normalization (redundant)
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False)
            )
            self.norm_layers.append(self._make_norm_layer(out_channels, norm_type))
            in_channels = out_channels
        
        # Final classifier
        in_features = self._compute_flattened_size()
        self.fc1 = nn.Linear(in_features, n_classes)
        
        self._init_weights(activation)
        
    def _make_norm_layer(self, channels, norm_type):
        """Factory for normalization layers."""
        return {
            'batch': lambda: nn.BatchNorm2d(channels),
            'layer': lambda: nn.GroupNorm(1, channels),  # LayerNorm equivalent for 2D
        }[norm_type]()
    
    def _init_weights(self, activation):
        """Kaiming initialization for ReLU-family activations."""
        nonlinearity = 'leaky_relu' if activation == 'leaky_relu' else 'relu'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                nn.init.constant_(m.bias, 0)
    
    def _compute_flattened_size(self):
        """Compute flattened feature size via forward pass with dummy input."""
        x = torch.zeros(1, 1, self.n_channels, self.n_samples)
        
        for i in range(len(self.conv_channels) - 1):
            x = self.act_fn(self.norm_layers[i](self.conv_layers[i](x)))
            x = F.max_pool2d(x, self.pool_kernel_size)
        
        x = self.norm_layers[-1](self.conv_layers[-1](x))
        x = F.avg_pool2d(x, self.pool_kernel_size)
        
        return x.flatten(1).shape[1]

    def forward(self, x, *args, **kwargs):
        """Forward pass.
        
        Args:
            x: (batch_size, n_channels, n_samples) or (batch_size, 1, n_channels, n_samples)
        
        Returns:
            Logits of shape (batch_size,) or (batch_size, n_classes)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        # All conv layers except last
        for conv, norm in zip(self.conv_layers[:-1], self.norm_layers[:-1]):
            x = F.max_pool2d(self.act_fn(norm(conv(x))), self.pool_kernel_size)
        
        # Last conv layer (no activation, avg pooling)
        x = self.norm_layers[-1](self.conv_layers[-1](x))
        x = F.avg_pool2d(x, self.pool_kernel_size)
        
        # Classifier with dropout
        x = F.dropout(x.flatten(1), p=self.dropout, training=self.training)
        return self.fc1(x).squeeze(-1)  # Return logits, not sigmoid


if __name__ == "__main__":
    # Test different configurations
    for norm_type in ['batch', 'layer']:
        model = SFCN(norm_type=norm_type, activation='gelu').to('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(32, 275, 40).to(model.fc1.weight.device)
        out = model(x)
        print(f"{norm_type}: {out.shape}, range=[{out.min():.2f}, {out.max():.2f}]")