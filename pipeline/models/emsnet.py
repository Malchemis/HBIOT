import torch
import torch.nn as nn
import torch.nn.functional as F


class EMSNET(nn.Module):
    """
    Adapted EMS-Net for (B, 1, 275, 40) input: 275 channels, 40 timepoints.
    
    Uses standard Conv2d instead of LocallyConnected2d to reduce parameters.
    
    Architecture:
    - Module 1: Local representation learning (1D CNN per channel) 
    - Module 2: Global representation learning (2D CNN with shared weights)
    - Module 3: Weighted feature combination
    
    Key simplification from original paper:
    - Replaced unshared-weight locally connected layer with standard Conv2d
    - Trade-off: Lose position-specific learning, gain parameter efficiency
    - Still captures spatial patterns via shared convolutional filters
    
    References:
    - Zheng et al. (2020) "EMS-Net: A Deep Learning Method for Autodetecting 
      Epileptic Magnetoencephalography Spikes" IEEE TMI
    
    Args:
        input_shape: (1, n_channels, n_timepoints) 
        n_channels: Number of MEG channels (default: 275)
        n_timepoints: Number of time samples (default: 40)
    """
    
    def __init__(self, input_shape=(1, 275, 40), n_channels=275, n_timepoints=40, *args, **kwargs):
        super().__init__()
        
        if input_shape is None:
            self.n_channels = n_channels
            self.n_timepoints = n_timepoints
        else:
            self.n_channels = input_shape[1]
            self.n_timepoints = input_shape[2]
        
        # ===== MODULE 1: Local (single-channel) representation learning =====
        # Process each channel independently with 1D CNNs to extract temporal features
        # Architecture: 6 conv layers, 3 pooling, 4 dropout, 1 dense
        
        self.local_conv = nn.ModuleList([
            # Layer 1: Conv -> LeakyReLU
            # 40 -> 36 (40 - 5 + 1)
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5),  
                nn.LeakyReLU(0.1, inplace=True)
            ),
            
            # Layer 2: Conv -> LeakyReLU -> Pool -> Dropout
            # 36 -> 32 -> 16
            nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=5),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.5)
            ),
            
            # Layer 3: Conv -> LeakyReLU
            # 16 -> 14
            nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3),
                nn.LeakyReLU(0.1, inplace=True)
            ),
            
            # Layer 4: Conv -> LeakyReLU -> Pool -> Dropout
            # 14 -> 12 -> 6
            nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=3),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.5)
            ),
            
            # Layer 5: Conv -> LeakyReLU
            # 6 -> 4
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3),
                nn.LeakyReLU(0.1, inplace=True)
            ),
            
            # Layer 6: Conv -> LeakyReLU -> AdaptivePool -> Dropout
            # 4 -> 2 -> 1
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveMaxPool1d(1),
                nn.Dropout(0.5)
            )
        ])
        
        # Dense layer for local feature refinement
        self.local_dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # ===== MODULE 2: Global (multi-channel) representation learning =====
        # Capture spatial patterns across channels using 2D convolutions with SHARED weights
        # Input: (B, 1, 275, 40)
        
        # Standard Conv2d with shared weights (replacing LocallyConnected2d)
        self.global_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                kernel_size=(9, self.n_timepoints),  # (9, 40)
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(0.1, inplace=True)
        ) # Output: (B, 16, 267, 1)
        
        # Second conv layer: 1×1 to increase feature depth
        # Output: (B, 64, 267, 1)
        self.global_conv2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Global pooling to aggregate spatial information
        # Output: (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Dense layer for global feature refinement
        self.global_dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # ===== MODULE 3: Weighted feature combination =====
        # Combine local and global features with learnable weights
        # Equation: h_combi = g(W_combi * [w1*h_local + w2*h_global] + b)
        
        # Learnable scalar weights for balancing local vs global features
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        
        # Final classification layers
        # Input: 275*64 (local) + 64 (global) = 17,664 features
        self.combination = nn.Sequential(
            nn.Linear(self.n_channels * 64 + 64, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input tensor
               - (B, 1, 275, 40): batch, channel, sensors, time
               - (B, 275, 40): batch, sensors, time (will add channel dim)
        
        Returns:
            predictions: (B,) - spike probability logits for each sample
        """
        batch_size = x.shape[0]
        
        # Ensure input has channel dimension
        if x.shape == (batch_size, self.n_channels, self.n_timepoints):
            x = x.unsqueeze(1)  # (B, 1, 275, 40)
        
        # ===== MODULE 1: Local features (process each channel separately) =====
        local_features = []
        for i in range(self.n_channels):
            # Extract single channel: (B, 1, 40)
            channel_data = x[:, :, i, :]
            
            # Process through 6 convolutional layers
            h = channel_data
            for layer in self.local_conv:
                h = layer(h)
            
            # h shape: (B, 64, 1)
            h = h.squeeze(-1)  # (B, 64)
            
            # Apply dense layer
            h = self.local_dense(h)  # (B, 64)
            local_features.append(h)
        
        # Concatenate features from all channels: (B, 275*64)
        h_local = torch.cat(local_features, dim=1)
        
        # ===== MODULE 2: Global features (multi-channel patterns) =====
        # Input: (B, 1, 275, 40)
        h_global = self.global_conv1(x)     # (B, 16, 267, 1)
        h_global = self.global_conv2(h_global)  # (B, 64, 267, 1)
        h_global = self.global_pool(h_global)   # (B, 64, 1, 1)
        h_global = h_global.view(batch_size, -1)  # (B, 64)
        h_global = self.global_dense(h_global)    # (B, 64)
        
        # ===== MODULE 3: Weighted combination =====
        # Apply learnable weights to balance local vs global information
        combined = torch.cat([
            self.w1 * h_local,   # (B, 17600)
            self.w2 * h_global   # (B, 64)
        ], dim=1)  # (B, 17664)
        
        # Final classification
        output = self.combination(combined)  # (B, 1)
        
        return output.squeeze(1)  # (B,)


def test_emsnet():
    """Test the adapted EMS-Net architecture."""
    print("=" * 70)
    print("Testing adapted EMS-Net (Conv2d instead of LocallyConnected2d)")
    print("=" * 70)
    
    # Create model
    model = EMSNET(n_channels=275, n_timepoints=40).to(device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Test with batch of data
    batch_size = 32
    x = torch.randn(batch_size, 1, 275, 40).to(device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Expected:    (batch=8, channels=1, sensors=275, time=40)")
    
    # Forward pass
    output = model(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected:     ({batch_size},)")
    print(f"Output:       {output}")
    
    from torchsummary import summary
    print("\nModel Summary:")
    summary(model, input_size=(1, 275, 40))
    

if __name__ == "__main__":
    # Run tests
    test_emsnet()
