"""Enhanced Full Attention Transformer using x-transformers library.

This module provides an upgraded transformer implementation with Flash Attention
and RMSNorm for improved memory efficiency and training stability.
"""

import torch.nn as nn
from pipeline.models.x_transformer_wrapper import XTransformerEncoder


class FullAttentionTransformer(nn.Module):
    """Enhanced Full Attention Transformer using x-transformers library.

    Replaces the original PyTorch MultiheadAttention implementation with
    state-of-the-art attention mechanisms including Flash Attention and RMSNorm.

    Key improvements:
    - Flash Attention for memory efficiency
    - RMSNorm for improved training stability
    """

    def __init__(
            self,
            dim,
            depth,
            max_seq_len,
            heads=8,
            ff_dropout=0.2,
            attn_dropout=0.2,
            use_flash_attn=True,
            use_rmsnorm=True,
            **kwargs  # To handle any extra params from LinearAttentionTransformer
    ):
        super().__init__()
        
        # Create x-transformers encoder
        self.transformer = XTransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_flash_attn=use_flash_attn,          # Enable Flash Attention for efficiency
            use_rmsnorm=use_rmsnorm,                # Use RMSNorm for stability
            **kwargs
        )

    def forward(self, x, mask=None):
        """
        Forward pass through the enhanced transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            mask: Optional attention mask

        Returns:
            Transformed tensor of same shape as input (excluding memory tokens)
        """
        # Forward through x-transformers encoder
        output = self.transformer(x, mask=mask)
        return output


class FullAttentionBlock(nn.Module):
    """
    Full attention block using PyTorch's MultiheadAttention.
    """

    def __init__(self, dim, heads, dim_head, causal=False, dropout=0., attn_dropout=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.causal = causal

        # PyTorch's MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True  # Input is [batch, seq, feature]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Forward pass through the attention block

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attn_mask: Optional attention mask

        Returns:
            Attention output tensor
        """
        # Apply layer normalization
        x_norm = self.norm(x)

        # Apply attention
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=self.causal
        )

        # Apply dropout
        return self.dropout(attn_output)


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection.
    """

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass through the feed-forward network"""
        return self.net(self.norm(x))
