"""Common neural network modules for MEG spike detection models.

This module provides reusable building blocks for MEG data processing including:
- Patch embedding modules (time, frequency, and feature-based)
- Positional encoding for transformer models
- Classification heads (simple and attention-based)
- STFT transformation for spectral analysis

These components are used across different model architectures (BIOT, HBIOT, etc.).
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ------ FEATURE EXTRACTION MODULES ------ #
# Frequency and time domain feature extraction modules for MEG data.
def stft(sample: torch.Tensor, n_fft, hop_length) -> torch.Tensor:
    """Compute Short-Time Fourier Transform for spectral patch embedding.

    Transforms raw temporal MEG data into frequency-domain representation for
    spectral processing mode. Uses rectangular window for simple frequency analysis.

    PROCESSING STEPS:
    1. Remove channel dimension: (B, 1, T) → (B, T)
    2. Apply STFT with rectangular window
    3. Take magnitude: Complex → Real
    4. Return frequency-time representation

    SHAPE TRANSFORMATION:
    (batch_size, 1, time_samples) → (batch_size, freq_bins, time_frames)
    Example: (800, 1, 80) → (800, 101, time_frames)

    Args:
        sample (torch.Tensor): Input temporal data.
            Shape: (batch_size, 1, time_samples)
            Example: (800, 1, 80) - single channel temporal data
        n_fft (int): Number of FFT points, determines frequency resolution.
            Example: 200 → 101 frequency bins (n_fft//2 + 1)
        hop_length (int): Number of samples between successive frames.
            Controls temporal resolution vs. frequency resolution tradeoff

    Returns:
        torch.Tensor: Magnitude spectrogram.
            Shape: (batch_size, freq_bins, time_frames)
            Example: (800, 101, time_frames)
            Contains magnitude values for each frequency bin at each time frame
    """
    spectral = torch.stft(
        input=sample.squeeze(1),  # from shape (batch_size, 1, ts) to (batch_size, ts)
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.ones(n_fft, device=sample.device),  # this is a rectangular window
        center=False,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(spectral)


class PatchFrequencyEmbedding(nn.Module):
    """Embedding module for spectral domain MEG data processing.

    Transforms STFT frequency representations into fixed-size embeddings for
    transformer processing. Used in spectral mode of BIOT encoder.

    PROCESSING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: STFT magnitude spectrogram                                       │
    │        Shape: (batch_size, freq_bins, time_frames)                      │
    │        Example: (800, 101, time_frames)                                 │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Transpose: (batch, freq, time) → (batch, time, freq)                   │
    │            Preparation for linear projection along frequency dimension  │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Linear Projection: (batch, time, freq) → (batch, time, emb_size)       │
    │                   Each time frame's frequency profile → embedding       │
    └─────────────────────────────────────────────────────────────────────────┘

    FREQUENCY EMBEDDING STRATEGY:
    • Each time frame in the spectrogram becomes a patch
    • Linear layer projects frequency profile to embedding space
    • Preserves temporal sequence structure for transformer processing
    • Enables learning of frequency-domain patterns in MEG signals

    GRADIENT FLOW:
    • Embeddings receive gradients from transformer attention
    • Linear layer learns optimal frequency feature combinations
    • Shared across all time frames within channel

    Attributes:
        projection (nn.Linear): Projects frequency vectors to embedding space.
            Input size: n_freq (number of frequency bins)
            Output size: emb_size (embedding dimension)
    """

    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        """Initialize the patch frequency embedding.

        Args:
            emb_size: Size of the embedding vector.
            n_freq: Number of frequency components.
        """
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)
        self.logger = logging.getLogger(__name__ + ".PatchFrequencyEmbedding")
        self.logger.debug(f"Initialized with emb_size={emb_size}, n_freq={n_freq}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frequency-to-embedding projection.

        SHAPE TRANSFORMATION DETAILS:
        Input:  (batch_size, freq_bins, time_frames)
        Transpose: (batch_size, time_frames, freq_bins)  
        Project: (batch_size, time_frames, emb_size)

        PROCESSING LOGIC:
        1. Transpose to put time dimension first for sequence processing
        2. Apply linear projection along frequency dimension
        3. Each time frame's frequency profile becomes an embedding vector

        Args:
            x (torch.Tensor): STFT magnitude spectrogram.
                Shape: (batch_size, freq_bins, time_frames)
                Example: (800, 101, time_frames)
                Contains magnitude values for each frequency bin at each time frame

        Returns:
            torch.Tensor: Temporal sequence of frequency embeddings.
                Shape: (batch_size, time_frames, emb_size)
                Example: (800, time_frames, 256)
                Each time frame becomes a 256-dimensional embedding vector
                Ready for transformer sequence processing
        """
        # Permute to (batch, time, freq) for linear projection along freq dimension
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PatchTimeEmbedding(nn.Module):
    """Embedding module for raw temporal MEG data with overlapping patches.

    Processes raw time-series MEG data by creating overlapping temporal patches
    and projecting them to fixed-size embeddings. Used in raw mode of BIOT encoder.

    PROCESSING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: Raw temporal data (single channel)                               │
    │        Shape: (batch_size, 1, time_samples)                             │
    │        Example: (800, 1, 80) [400ms at 200Hz]                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Remove Channel Dimension: (batch, 1, time) → (batch, time)              │
    │                          Prepare for patch extraction                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Unfold Operation: Create overlapping patches                            │
    │ • patch_size=40, overlap=0.5 → stride=20                                │
    │ • (batch, 80) → (batch, n_patches, 40)                                  │
    │ • Example: (800, 80) → (800, 5, 40)                                     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Linear Projection: (batch, n_patches, patch_size) →                     │
    │                   (batch, n_patches, emb_size)                          │
    │ Each 40-sample patch → 256-dimensional embedding                        │
    └─────────────────────────────────────────────────────────────────────────┘

    EXAMPLE PATCH CALCULATION:
    For 80 samples, patch_size=40, overlap=0.5:
    • stride = 40 x (1 - 0.5) = 20 samples
    • n_patches = (80 - 40) / 20 + 1 = 3 patches
    • Patch positions: [0:40], [20:60], [40:80]

    Attributes:
        patch_size (int): Number of samples per temporal patch.
            Example: 40 (400ms at 200Hz sampling rate)
        overlap (float): Overlap ratio between adjacent patches (0.0-1.0).
            Example: 0.5 (50% overlap for smooth temporal coverage)
        projection (nn.Linear): Projects raw patches to embedding space.
            Input size: patch_size, Output size: emb_size
    """

    def __init__(self, emb_size: int = 256, patch_size: int = 100, overlap: float = 0.0):
        """Initialize the patch time embedding.

        Args:
            emb_size: Size of the embedding vector.
            patch_size: Size of each time patch.
            overlap: Amount of overlap between adjacent patches (0.0-1.0).
        """
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.projection = nn.Linear(patch_size, emb_size)
        
        # Calculate stride based on overlap
        stride = int(self.patch_size * (1 - self.overlap))  # e.g., 40 samples * (1 - 0.75) = 10 samples
        self.stride = max(1, stride)  # Ensure stride is at least 1 sample
        
        self.logger = logging.getLogger(__name__ + ".PatchTimeEmbedding")
        self.logger.debug(f"Initialized with emb_size={emb_size}, patch_size={patch_size}, overlap={overlap}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with overlapping patch extraction and embedding.

        PATCH CALCULATION EXAMPLE:
        Input: 80 samples, patch_size=40, overlap=0.5
        • stride = 40 x (1 - 0.5) = 20 samples
        • Patches: [0:40], [20:60], [40:80]
        • Total patches: (80-40)/20 + 1 = 3 patches

        SHAPE TRANSFORMATIONS:
        (batch, 1, time) → (batch, time) → (batch, n_patches, patch_size) → 
        (batch, n_patches, emb_size)

        Args:
            x (torch.Tensor): Raw temporal MEG data for single channel.
                Shape: (batch_size, 1, time_samples)
                Example: (800, 1, 80)
                - 800 = batch_size x n_windows (e.g., 32 x 25)
                - 1 = single channel (processed one at a time)
                - 80 = temporal samples (400ms at 200Hz)

        Returns:
            torch.Tensor: Sequence of temporal patch embeddings.
                Shape: (batch_size, n_patches, emb_size)
                Example: (800, 3, 256)
                - 3 patches with 50% overlap
                - Each patch embedded to 256 dimensions
                - Ready for transformer sequence processing

        Raises:
            ValueError: If input length is less than patch_size
        """
        time_steps = x.shape[2]

        # Ensure we have enough time steps/samples for at least one patch
        if time_steps < self.patch_size:
            error_msg = f"Input length ({time_steps}) must be >= patch_size ({self.patch_size})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Use unfold with the calculated stride to create overlapping patches
        x = x.squeeze(1)  # Remove channel dim: (batch, time)
        x = x.unfold(1, self.patch_size, self.stride)  # (batch, n_patches, patch_size)

        # Project to embedding dimension
        x = self.projection(x)  # (batch, n_patches, emb_size)
        return x


class PatchFeatureEmbedding(nn.Module):
    """Embedding module for feature-based MEG data processing with overlapping patches.

    Similar to PatchTimeEmbedding, but extracts handcrafted features from each patch
    instead of using raw temporal data. This enables combining classical ML features
    with transformer architectures for MEG spike detection.

    PROCESSING PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: Raw temporal data (single channel)                               │
    │        Shape: (batch_size, 1, time_samples)                             │
    │        Example: (800, 1, 80) [400ms at 200Hz]                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Extract Overlapping Patches: Same as PatchTimeEmbedding                 │
    │ • patch_size=40, overlap=0.5 → stride=20                                │
    │ • (batch, 1, 80) → patches: (batch, n_patches, 40)                      │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Feature Extraction: Extract features from each patch                    │
    │ • Morphological features: amplitude, slope, energy, etc.               │
    │ • Spectral features: band powers, peak frequency, entropy, etc.        │
    │ • (batch, n_patches, 40) → (batch, n_patches, n_features)              │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Linear Projection: (batch, n_patches, n_features) →                     │
    │                   (batch, n_patches, emb_size)                          │
    │ Each feature vector → embedding dimension                               │
    └─────────────────────────────────────────────────────────────────────────┘

    FEATURE EXTRACTION STRATEGY:
    • Extract both morphological and spectral features from each patch
    • Creates rich feature representation per temporal patch
    • Maintains sequence structure for transformer processing
    • Combines benefits of handcrafted features with deep learning

    Attributes:
        patch_size (int): Number of samples per temporal patch.
        overlap (float): Overlap ratio between adjacent patches (0.0-1.0).
        projection (nn.Linear): Projects feature vectors to embedding space.
    """
    def __init__(
        self,
        emb_size: int = 256,
        patch_size: int = 60,
        overlap: float = 0.5,
        sfreq: float = 200.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.sfreq = sfreq

        # Feature counts
        self.n_morph = 13
        self.n_spec = 9
        self.n_feats = self.n_morph + self.n_spec #+ self.n_wav

        # Group-wise normalization
        self.norm_morph = nn.LayerNorm(self.n_morph)
        self.norm_spec = nn.LayerNorm(self.n_spec)

        # Projection MLP: n_feats -> emb_size
        self.projection = nn.Sequential(
            nn.Linear(self.n_feats, emb_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_size // 2, emb_size),
            nn.LayerNorm(emb_size),
        )

        # Logger
        self.logger = logging.getLogger(__name__ + ".PatchFeatureEmbedding")

    def extract_features_from_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Extract morphological, spectral, and wavelet features from each example in a patch.

        Args:
            patch: Tensor of shape (batch_size, patch_size)
        Returns:
            Tensor of shape (batch_size, emb_size)
        """
        from scipy import signal as scipy_signal
        from scipy import stats

        batch_size = patch.shape[0]
        data = patch.detach().cpu().numpy()

        morph_list, spec_list, wav_list = [], [], []

        for i in range(batch_size):
            x = data[i]
            
            # === Morphological (13) ===
            morph = [
                np.max(x),
                np.min(x),
                np.ptp(x),
                np.mean(x),
                np.std(x),
                np.argmax(np.abs(x)) / len(x),
            ]
            dx = np.diff(x)
            morph += [
                np.max(dx) if len(dx) > 0 else 0.0,
                np.min(dx) if len(dx) > 0 else 0.0,
                np.mean(np.abs(dx)) if len(dx) > 0 else 0.0,
                np.sum(np.diff(np.sign(x + 1e-10)) != 0) / len(x),
                np.sum(x ** 2),
                stats.kurtosis(x, nan_policy='omit'),
                stats.skew(x, nan_policy='omit'),
            ]
            # Clean
            morph = [0.0 if np.isnan(v) or np.isinf(v) else float(v) for v in morph]
            morph_list.append(morph)

            # === Spectral (9) ===
            try:
                # Use smaller nperseg for short signals
                nperseg = min(len(x) // 2, 32)  # Ensure we have at least 2 windows
                if nperseg < 4:  # Too short for Welch
                    spec = [0.0] * self.n_spec
                else:
                    freqs, psd = scipy_signal.welch(x, fs=self.sfreq, nperseg=nperseg)
                    
                    if len(psd) == 0 or np.sum(psd) == 0:
                        spec = [0.0] * self.n_spec
                    else:
                        total = np.sum(psd) + 1e-12
                        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 99)]
                        spec = []
                        
                        # Band powers
                        for lo, hi in bands:
                            mask = (freqs >= lo) & (freqs <= hi)
                            if np.any(mask):
                                spec.append(np.sum(psd[mask]) / total)
                            else:
                                spec.append(0.0)
                        
                        # Peak frequency
                        spec.append(freqs[np.argmax(psd)] if psd.size > 0 else 0.0)
                        
                        # 95% frequency
                        cumsum = np.cumsum(psd)
                        idx_95 = np.where(cumsum >= 0.95 * total)[0]
                        spec.append(freqs[idx_95[0]] if len(idx_95) > 0 else freqs[-1])
                        
                        # Spectral entropy
                        psd_norm = psd / total
                        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                        spec.append(entropy)
                        
                        # Total power
                        spec.append(total)
                        
            except Exception as e:
                self.logger.warning(f"Spectral error for patch {i}: {e}")
                spec = [0.0] * self.n_spec
                
            spec = [0.0 if np.isnan(v) or np.isinf(v) else float(v) for v in spec]
            spec_list.append(spec)


        # Convert to tensors with explicit float32 dtype
        morph_t = self.norm_morph(torch.tensor(morph_list, device=patch.device, dtype=torch.float32))
        spec_t  = self.norm_spec(torch.tensor(spec_list, device=patch.device, dtype=torch.float32))

        combined = torch.cat([morph_t, spec_t], dim=-1)
        embeddings = self.projection(combined)
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, time_samples)
        Returns:
            Tensor of shape (batch_size, n_patches, emb_size)
        """
        batch, _, T = x.shape
        stride = max(1, int(self.patch_size * (1 - self.overlap)))
        
        if T < self.patch_size:
            raise ValueError(f"Input length {T} < patch_size {self.patch_size}")

        x = x.squeeze(1)  # (batch, T)
        patches = x.unfold(1, self.patch_size, stride)  # (batch, n_patches, patch_size)
        n_patches = patches.size(1)

        # Extract embeddings per patch
        embs = []
        for i in range(n_patches):
            patch = patches[:, i, :]
            embs.append(self.extract_features_from_patch(patch))

        return torch.stack(embs, dim=1)  # (batch, n_patches, emb_size)


class PositionalEncoding(nn.Module):
    """Module for adding positional encoding to embeddings.

    This implementation uses sinusoidal position encoding to give the model
    information about the position of tokens in the sequence.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Precomputed positional encoding. (registered as to not be a model parameter)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 800):
        """Initialize the positional encoding.

        Args:
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.logger = logging.getLogger(__name__ + ".PositionalEncoding")
        self.logger.debug(f"Initialized with d_model={d_model}, dropout={dropout}, max_len={max_len}")

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.

        Args:
            x: Input embeddings tensor of shape (batch, max_len, d_model).

        Returns:
            Embeddings with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]  # type: ignore
        return self.dropout(x)


class ClassificationHead(nn.Sequential):
    """Module for classification head.

    This module takes the embeddings and produces class probabilities.

    Attributes:
        clshead (nn.Sequential): Sequential module with the classification layers.
    """

    def __init__(self, emb_size: int, n_classes: int):
        """Initialize the classification head.

        Args:
            emb_size: Size of the input embedding.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, emb_size // 2),
            nn.ELU(),
            nn.Linear(emb_size // 2, n_classes),
        )
        self.logger = logging.getLogger(__name__ + ".ClassificationHead")
        self.logger.debug(f"Initialized with emb_size={emb_size}, n_classes={n_classes}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the module.

        Args:
            x: Input embedding tensor
        Returns:
            Class logits.
        """
        # Classify each window
        # x: (batch, emb_size) or (batch * n_windows, emb_size)
        return self.clshead(x).squeeze(-1)


class AttentionClassificationHead(nn.Module):
    """Attention-based classification head for hierarchical token aggregation.

    Aggregates a sequence of tokens into a single classification output per batch
    by means of a learnable query and multi-head attention.
    """

    def __init__(
        self,
        emb_size: int,
        n_classes: int,
        n_tokens_per_window: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens_per_window = n_tokens_per_window

        # learnable classification query token
        self.classification_query = nn.Parameter(torch.randn(1, 1, emb_size))

        # attend from the query to the token sequence
        self.token_attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # final MLP to map the attended embedding to class‐scores
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, E)  -- batch of token sequences
        returns: (B, n_classes)
        """
        B, T, E = x.shape
        assert T == self.n_tokens_per_window, f"Expected {self.n_tokens_per_window} tokens, got {T}"

        # expand the query to match batch‐size
        query = self.classification_query.expand(B, 1, E)  # (B, 1, E)

        # attend: output (B, 1, E)
        attn_out, _ = self.token_attention(query, x, x)

        # squeeze out the sequence dimension → (B, E)
        pooled = attn_out.squeeze(1)

        # classifier → (B, n_classes)
        logits = self.classifier(pooled)

        return logits.squeeze(-1)

