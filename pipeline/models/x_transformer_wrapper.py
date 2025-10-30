#!/usr/bin/env python3
"""
X-Transformers wrapper for BIOT architecture.

This module provides a wrapper around the x-transformers library's Encoder
to replace the current full attention transformer implementation with state-of-the-art
attention mechanisms including Flash Attention and RMSNorm, while maintaining
compatibility with BIOT's custom positional embedding strategy.
"""

import torch
import torch.nn as nn
from typing import Optional
from x_transformers import Encoder


class XTransformerEncoder(nn.Module):
    """
    Wrapper for x-transformers Encoder to replace FullAttentionTransformer.
    
    Provides a drop-in replacement for the current full attention implementation
    with advanced features while maintaining compatibility with BIOT's encoding process.
    
    Key improvements over standard attention:
    - Flash Attention for memory efficiency with long sequences
    - RMSNorm for improved training stability
    - Proper parameter mapping for drop-in replacement
    
    IMPORTANT: This wrapper is designed to work with BIOT's manual positional embeddings.
    RoPE and memory tokens are disabled to avoid conflicts with the existing encoding pipeline.
    
    Architecture Benefits for MEG data:
    - Efficient processing of 321-token sequences per window
    - Maintains exact sequence length for downstream processing
    - Compatible with existing channel and positional embedding strategy
    - Reduced memory usage with Flash Attention
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 8,
        max_seq_len: int = 1024,
        dim_head: Optional[int] = None,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_flash_attn: bool = True,
        use_rmsnorm: bool = True,
        **kwargs
    ):
        """
        Initialize the X-Transformers Encoder wrapper with BIOT-compatible settings.
        
        Args:
            dim: Embedding dimension size
            depth: Number of transformer layers  
            heads: Number of attention heads
            max_seq_len: Maximum sequence length (matches BIOT's calculated sequence length)
            dim_head: Dimension per attention head (default: dim // heads)
            attn_dropout: Attention dropout probability
            ff_dropout: Feed-forward dropout probability
            use_flash_attn: Whether to use Flash Attention for memory efficiency
            use_rotary_pos_emb: DISABLED - BIOT uses manual positional embeddings
            use_rmsnorm: Whether to use RMSNorm for improved training stability
            **kwargs: Additional arguments for compatibility with LinearAttentionTransformer
        """
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        
        # Calculate dim_head if not provided
        if dim_head is None:
            dim_head = dim // heads
            
        # Filter out LinearAttentionTransformer-specific kwargs that don't apply to x-transformers
        xtransformer_kwargs = {}
        for key, value in kwargs.items():
            # Skip LinearAttentionTransformer-specific parameters
            if key not in ['causal', 'ff_chunks', 'attn_layer_dropout', 'local_heads', 'local_window_size']:
                xtransformer_kwargs[key] = value
        
        # Initialize the x-transformers Encoder with BIOT-compatible configuration
        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            # Core improvements enabled
            attn_flash=use_flash_attn,           # Enable Flash Attention for efficiency
            use_rmsnorm=use_rmsnorm,             # Use RMSNorm for stability
            # BIOT compatibility settings
            rotary_pos_emb=False,                # DISABLED: BIOT adds manual positional embeddings
            # Additional compatibility settings
            rel_pos_bias=False,                  # No relative position bias (BIOT handles this)
            **xtransformer_kwargs
        )
        
        # Store configuration for logging/debugging
        self.config = {
            'dim': dim,
            'depth': depth, 
            'heads': heads,
            'max_seq_len': max_seq_len,
            'use_flash_attn': use_flash_attn,
            'use_rmsnorm': use_rmsnorm,
            'attn_dropout': attn_dropout,
            'ff_dropout': ff_dropout
        }
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass through the x-transformers encoder with BIOT compatibility.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
                For BIOT: (batch_size, 321, emb_size) where 321 = 1 CLS + 320 patches
            mask: Optional attention mask of shape (batch_size, seq_len)
                Compatible with BIOT's input_mask parameter
            **kwargs: Additional arguments for compatibility with LinearAttentionTransformer
                
        Returns:
            torch.Tensor: Encoded representations of shape (batch_size, seq_len, dim)
                Exact same shape as input - no memory tokens added
                Compatible with BIOT's downstream processing expectations
        """
        return self.encoder(x, mask=mask)

    def get_config(self) -> dict:
        """Return the configuration used for this encoder."""
        return self.config.copy()


def create_x_transformer_encoder(
    dim: int,
    depth: int,
    heads: int = 8,
    max_seq_len: int = 1024,
    **kwargs
) -> XTransformerEncoder:
    """
    Factory function to create an XTransformerEncoder with BIOT-compatible defaults.
    
    This factory ensures proper configuration for drop-in replacement of 
    LinearAttentionTransformer in BIOT architecture.
    
    Args:
        dim: Embedding dimension (typically 256 or 512 for BIOT)
        depth: Number of transformer layers
        heads: Number of attention heads
        max_seq_len: Maximum sequence length (calculated by BIOT based on windows/channels)
        **kwargs: Additional configuration options
        
    Returns:
        Configured XTransformerEncoder instance compatible with BIOT
    """
    # BIOT-compatible configuration optimized for MEG spike detection
    biot_compatible_config = {
        # DISABLED features that conflict with BIOT's encoding
        'use_rotary_pos_emb': False,     # DISABLED: BIOT uses manual positional embeddings
        'rel_pos_bias': False,           # DISABLED: BIOT handles positional relationships
        
        # ENABLED features that improve performance without conflicts
        'use_flash_attn': True,          # Enable Flash Attention for efficiency
        'use_rmsnorm': True,             # Use RMSNorm for training stability
        
        # Conservative dropout defaults
        'attn_dropout': 0.1,
        'ff_dropout': 0.1,
    }
    
    # Merge user config with BIOT-compatible defaults (user config takes precedence)
    config = {**biot_compatible_config, **kwargs}
    
    return XTransformerEncoder(
        dim=dim,
        depth=depth,
        heads=heads,
        max_seq_len=max_seq_len,
        **config
    )


# Compatibility alias for drop-in replacement
XTransformerFullAttention = XTransformerEncoder