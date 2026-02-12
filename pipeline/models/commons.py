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
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class SirenLayer(nn.Module):
    """Single SIREN layer: sin(w0 * (Wx + b)) with principled initialization.

    Ref: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020.
    """

    def __init__(self, dim_in: int, dim_out: int, w0: float = 1.0, is_first: bool = False, w0_initial: float = 30.0,
                 c: float = 6.0):
        super().__init__()
        self.w0 = w0_initial if is_first else w0
        self.linear = nn.Linear(dim_in, dim_out)

        # Principled initialization
        with torch.no_grad():
            if is_first:
                bound = 1.0 / dim_in
            else:
                bound = math.sqrt(c / dim_in) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SirenNet(nn.Module):
    """Stack of SirenLayers with a final linear output (no sine on last layer)."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, n_layers: int = 3,
                 w0_initial: float = 30.0, w0: float = 1.0, c: float = 6.0):
        super().__init__()
        layers = []
        for i in range(n_layers):
            is_first = (i == 0)
            layer_in = dim_in if is_first else dim_hidden
            layers.append(SirenLayer(layer_in, dim_hidden, w0=w0, is_first=is_first, w0_initial=w0_initial))
        self.net = nn.Sequential(*layers)

        # Final linear projection (no sine activation)
        self.final_linear = nn.Linear(dim_hidden, dim_out)
        with torch.no_grad():
            bound = math.sqrt(c / dim_hidden) / w0
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.net(x))


def _hilbert_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute the analytic signal via FFT-based Hilbert transform.

    Args:
        x: Real-valued input tensor of shape (..., T).

    Returns:
        Complex-valued analytic signal of shape (..., T).
    """
    T = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)

    # Build step function: h[0]=1, h[1..N/2-1]=2, h[N/2]=1 (if even), h[N/2+1..]=0
    h = torch.zeros(T, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if T % 2 == 0:
        h[1:T // 2] = 2.0
        h[T // 2] = 1.0
    else:
        h[1:(T + 1) // 2] = 2.0

    return torch.fft.ifft(X * h, dim=-1)


def compute_plv_gpu(x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute Phase Locking Value (PLV) connectivity matrix on GPU via bmm.

    Memory-efficient: uses bmm on (B, C, T) complex phase vectors rather than
    materializing the full (B, C, C, T) tensor.

    Args:
        x: Input tensor of shape (B, C, T) — real-valued time series.
        channel_mask: Optional mask (B, C), True=valid, False=masked.

    Returns:
        PLV adjacency matrix of shape (B, C, C), values in [0, 1].
    """
    B, C, T = x.shape

    # Analytic signal → instantaneous phase → unit complex phasor
    analytic = _hilbert_torch(x)  # (B, C, T) complex
    phase = torch.angle(analytic)  # (B, C, T) real
    exp_phase = torch.exp(1j * phase.to(torch.float32))  # (B, C, T) complex64

    # PLV = |<e^{j(phi_i - phi_j)}>_t| = |sum_t e^{j*phi_i} * e^{-j*phi_j}| / T
    plv = torch.abs(torch.bmm(exp_phase, exp_phase.conj().transpose(1, 2))) / T  # (B, C, C)

    # Zero out rows/columns for masked channels
    if channel_mask is not None:
        mask_2d = channel_mask.unsqueeze(2) & channel_mask.unsqueeze(1)  # (B, C, C)
        plv = plv * mask_2d.float()

    return plv


def compute_spectral_coords(
    adjacency: torch.Tensor,
    n_eigenvectors: int,
    channel_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute spectral coordinates from a graph adjacency matrix.

    Builds the normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2},
    performs eigendecomposition, and returns the bottom-k non-trivial eigenvectors
    as spectral coordinates for each node.

    Args:
        adjacency: Adjacency matrix (B, C, C), non-negative.
        n_eigenvectors: Number of spectral coordinates (k).
        channel_mask: Optional mask (B, C), True=valid.
        eps: Small constant for numerical stability.

    Returns:
        Spectral coordinates of shape (B, C, k).
    """
    B, C, _ = adjacency.shape
    device = adjacency.device

    # Degree matrix
    degree = adjacency.sum(dim=-1)  # (B, C)

    # D^{-1/2}, safe for zero-degree nodes
    d_inv_sqrt = torch.zeros_like(degree)
    valid = degree > eps
    d_inv_sqrt[valid] = degree[valid].rsqrt()  # 1/sqrt(d)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # D^{-1/2} A D^{-1/2} = diag(d_inv_sqrt) @ A @ diag(d_inv_sqrt)
    d_left = d_inv_sqrt.unsqueeze(2)   # (B, C, 1)
    d_right = d_inv_sqrt.unsqueeze(1)  # (B, 1, C)
    norm_adj = d_left * adjacency * d_right  # (B, C, C)
    laplacian = torch.eye(C, device=device).unsqueeze(0).expand(B, -1, -1) - norm_adj  # (B, C, C)

    # Eigendecomposition (detached — eigh gradients are numerically unstable)
    with torch.no_grad():
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)  # sorted ascending
        # eigenvalues: (B, C), eigenvectors: (B, C, C)

    # Take eigenvectors 1..k+1 (skip index 0 = trivial constant eigenvector)
    k = min(n_eigenvectors, C - 1)
    coords = eigenvectors[:, :, 1:k + 1].clone()  # (B, C, k)

    # Pad with zeros if fewer valid channels than k
    if k < n_eigenvectors:
        pad = torch.zeros(B, C, n_eigenvectors - k, device=device, dtype=coords.dtype)
        coords = torch.cat([coords, pad], dim=-1)  # (B, C, n_eigenvectors)

    # Sign ambiguity fix: max-abs-positive convention per eigenvector
    # For each eigenvector, find the element with the largest absolute value and
    # flip the sign so that element is positive.
    max_abs_idx = coords.abs().argmax(dim=1, keepdim=True)  # (B, 1, k)
    signs = torch.gather(coords, 1, max_abs_idx).sign()  # (B, 1, k)
    signs[signs == 0] = 1.0
    coords = coords * signs  # (B, C, k)

    # Zero out masked channels
    if channel_mask is not None:
        coords = coords * channel_mask.unsqueeze(-1).float()  # (B, C, k)

    return coords


class SpectralChannelEmbedding(nn.Module):
    """Derive channel embeddings from functional connectivity graph structure.

    Computes PLV-based connectivity on GPU, extracts spectral coordinates
    (graph Laplacian eigenvectors), rescales them following the generalised INR
    framework, and maps them to embedding space through a SIREN network.

    The √n rescaling follows Grattarola & Vandergheynst (NeurIPS 2022): generalised
    spectral embeddings e_i = √n · [u_{1,i}, ..., u_{k,i}] ensure scale invariance
    across graphs of different sizes, so similar nodes have comparable embeddings
    regardless of the number of channels.

    Pipeline:
        Input (B, C, T) → PLV (B, C, C) → Laplacian eigenvectors (B, C, k)
        → √n rescale → [augmentation] → SIREN → LayerNorm → (B, C, E)
    """

    def __init__(
        self,
        emb_size: int = 256,
        n_eigenvectors: int = 16,
        projection_type: str = "siren",
        siren_hidden_dim: int = 128,
        siren_n_layers: int = 3,
        siren_w0_initial: float = 30.0,
        siren_w0: float = 1.0,
        eigen_eps: float = 1e-6,
        sqrt_n_rescale: bool = True,
        output_norm: bool = True,
        augmentation_noise_std: float = 0.0,
    ):
        super().__init__()
        self.n_eigenvectors = n_eigenvectors
        self.eigen_eps = eigen_eps
        self.sqrt_n_rescale = sqrt_n_rescale
        self.augmentation_noise_std = augmentation_noise_std
        self.logger = logging.getLogger(__name__ + ".SpectralChannelEmbedding")

        if projection_type == "siren":
            self.projection = SirenNet(
                dim_in=n_eigenvectors,
                dim_hidden=siren_hidden_dim,
                dim_out=emb_size,
                n_layers=siren_n_layers,
                w0_initial=siren_w0_initial,
                w0=siren_w0,
            )
        elif projection_type == "linear":
            self.projection = nn.Linear(n_eigenvectors, emb_size)
        else:
            raise ValueError(f"projection_type must be 'siren' or 'linear', got '{projection_type}'")

        # Output normalization (stabilizes embedding scale across batches)
        self.output_norm = nn.LayerNorm(emb_size) if output_norm else nn.Identity()

        self.logger.info(
            f"Initialized SpectralChannelEmbedding: n_eigenvectors={n_eigenvectors}, "
            f"projection={projection_type}, emb_size={emb_size}, "
            f"sqrt_n_rescale={sqrt_n_rescale}, output_norm={output_norm}, "
            f"augmentation_noise_std={augmentation_noise_std}"
        )

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute spectral channel embeddings from raw signal.

        Args:
            x: Input tensor (B, C, T) — concatenated windows for robust PLV.
            channel_mask: Optional mask (B, C), True=valid.

        Returns:
            Spectral embeddings of shape (B, C, emb_size).
        """
        # Force float32 for complex FFT and eigendecomposition
        x = x.float()

        # 1. PLV connectivity
        plv = compute_plv_gpu(x, channel_mask)  # (B, C, C)

        # 2. Spectral coordinates
        coords = compute_spectral_coords(plv, self.n_eigenvectors, channel_mask, self.eigen_eps)  # (B, C, k)

        # 3. GINR √n rescaling: e_i = √n · [u_{1,i}, ..., u_{k,i}]
        # Ensures scale invariance across different channel counts
        if self.sqrt_n_rescale:
            if channel_mask is not None:
                n_valid = channel_mask.sum(dim=1, keepdim=True).unsqueeze(-1).float().clamp(min=1)  # (B, 1, 1)
            else:
                n_valid = torch.tensor(x.shape[1], device=x.device, dtype=torch.float32)
            coords = coords * torch.sqrt(n_valid)

        # 4. Augmentation: Gaussian noise on spectral coordinates during training
        if self.training and self.augmentation_noise_std > 0:
            noise = torch.randn_like(coords) * self.augmentation_noise_std
            if channel_mask is not None:
                noise = noise * channel_mask.unsqueeze(-1).float()
            coords = coords + noise

        # 5. Project to embedding space + normalize
        emb = self.output_norm(self.projection(coords))  # (B, C, emb_size)

        return emb


class FourierSpatialEmbedding(nn.Module):
    """REVE-style Fourier positional encoding from 3D sensor coordinates.

    Encodes each channel's 3D spatial position into an embedding vector using:
    1. Deterministic Fourier encoding: sin/cos at multiple frequencies per dimension
    2. Parallel learnable linear projection
    3. LayerNorm on the sum

    Pipeline:
        coordinates (C, 3) → Fourier PE (C, D_fourier) → Linear → (C, E)
                           ↘ Linear(3, E) ↗
                             → sum → LayerNorm → (C, E)

    Reference: El Ouahidi et al., "REVE: A Foundation Model for EEG", NeurIPS 2025.
    """

    def __init__(
        self,
        emb_size: int = 256,
        n_frequencies: int = 8,
        coordinate_dim: int = 3,
        learnable_linear: bool = True,
        augmentation_noise_std: float = 0.0,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.coordinate_dim = coordinate_dim
        self.augmentation_noise_std = augmentation_noise_std
        self.logger = logging.getLogger(__name__ + ".FourierSpatialEmbedding")

        # Fourier PE dimension: sin + cos for each frequency for each coordinate dim
        n_fourier_features = 2 * n_frequencies * coordinate_dim

        # Projection from Fourier features to embedding space
        self.fourier_proj = nn.Linear(n_fourier_features, emb_size)

        # Parallel learnable linear path (REVE: compensates for Fourier truncation)
        self.learnable_linear = learnable_linear
        if learnable_linear:
            self.linear_proj = nn.Sequential(
                nn.Linear(coordinate_dim, emb_size),
                nn.GELU(),
            )

        # Output normalization
        self.output_norm = nn.LayerNorm(emb_size)

        # Frequency bands: 2^0*π, 2^1*π, ..., 2^(n_freq-1)*π
        freq_bands = (2.0 ** torch.arange(n_frequencies).float()) * math.pi
        self.register_buffer('freq_bands', freq_bands)

        # Coordinate buffer — set via set_coordinates()
        self.register_buffer('coordinates', torch.empty(0))

        self.logger.info(
            f"Initialized FourierSpatialEmbedding: n_frequencies={n_frequencies}, "
            f"coord_dim={coordinate_dim}, emb_size={emb_size}, "
            f"learnable_linear={learnable_linear}, "
            f"augmentation_noise_std={augmentation_noise_std}"
        )

    def set_coordinates(
        self,
        coordinates_dict: Dict[str, List[float]],
        channel_order: List[str],
    ) -> None:
        """Set the 3D coordinates tensor from a dictionary.

        Args:
            coordinates_dict: Mapping from channel name to [x, y, z] coordinates.
            channel_order: Ordered list of channel names (defines index mapping).
        """
        C = len(channel_order)
        coords = torch.zeros(C, self.coordinate_dim)
        n_found = 0
        for i, ch_name in enumerate(channel_order):
            if ch_name in coordinates_dict:
                coord_vals = coordinates_dict[ch_name][:self.coordinate_dim]
                coords[i] = torch.tensor(coord_vals, dtype=torch.float32)
                n_found += 1
        self.coordinates = coords.to(self.freq_bands.device)
        self.logger.info(
            f"Set coordinates for {n_found}/{C} channels "
            f"(range: [{coords.min():.4f}, {coords.max():.4f}])"
        )

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute Fourier encoding of coordinates.

        Args:
            coords: (..., coordinate_dim) tensor.

        Returns:
            Fourier features of shape (..., 2 * n_frequencies * coordinate_dim).
        """
        # coords: (..., D), freq_bands: (F,)
        # outer product per dimension: (..., D, F)
        proj = coords.unsqueeze(-1) * self.freq_bands  # (..., D, F)
        # sin and cos concatenated: (..., D, 2F)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        # flatten last two dims: (..., 2*F*D)
        return fourier.flatten(start_dim=-2)

    def forward(self, channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Fourier spatial embeddings.

        Args:
            channel_mask: Optional (B, C) mask. Not used for computation (coordinates
                are fixed), but included for API consistency.

        Returns:
            Spatial embeddings of shape (C, emb_size). Batch-independent since
            coordinates are static; caller broadcasts to batch dimension.
        """
        if self.coordinates.numel() == 0:
            raise RuntimeError(
                "Coordinates not set. Call set_coordinates() before forward()."
            )

        coords = self.coordinates  # (C, D)

        # Augmentation: Gaussian noise on coordinates during training
        if self.training and self.augmentation_noise_std > 0:
            noise = torch.randn_like(coords) * self.augmentation_noise_std
            coords = coords + noise

        # Fourier path
        fourier_features = self._fourier_encode(coords)  # (C, 2*F*D)
        fourier_emb = self.fourier_proj(fourier_features)  # (C, emb_size)

        # Combine with learnable linear path
        if self.learnable_linear:
            linear_emb = self.linear_proj(coords)  # (C, emb_size)
            combined = fourier_emb + linear_emb
        else:
            combined = fourier_emb

        return self.output_norm(combined)  # (C, emb_size)


class ChannelEmbeddingComposer(nn.Module):
    """Composes multiple channel embedding strategies into a single embedding.

    Manages any combination of:
    - learned: Fixed per-channel-index nn.Parameter table
    - spectral: Data-driven PLV-based spectral embeddings (GINR)
    - fourier: Fixed 3D coordinate Fourier PE (REVE)

    All enabled strategies produce (B, C, emb_size) or (C, emb_size) tensors
    that are summed to form the final channel embedding.

    Also owns special token embeddings (missing, unknown) for channel masking.
    """

    def __init__(
        self,
        n_channels: int,
        emb_size: int,
        config: Dict[str, Any],
        reference_coordinates: Optional[Dict[str, List[float]]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.strategies: List[str] = []
        self.logger = logging.getLogger(__name__ + ".ChannelEmbeddingComposer")

        # Strategy 1: Learned table (per channel index)
        learned_cfg = config.get('learned', {})
        self.use_learned = learned_cfg.get('enabled', True)
        if self.use_learned:
            self.channel_embedding = nn.Parameter(torch.randn(n_channels, emb_size))
            self.strategies.append('learned')

        # Strategy 2: Spectral (PLV-based, data-driven)
        spectral_cfg = config.get('spectral', {})
        self.use_spectral = spectral_cfg.get('enabled', False)
        self.spectral_module: Optional[SpectralChannelEmbedding] = None
        if self.use_spectral:
            sce_params = {k: v for k, v in spectral_cfg.items() if k != 'enabled'}
            self.spectral_module = SpectralChannelEmbedding(emb_size=emb_size, **sce_params)
            self.strategies.append('spectral')

        # Strategy 3: Fourier (3D coordinate-based, REVE-style)
        fourier_cfg = config.get('fourier', {})
        self.use_fourier = fourier_cfg.get('enabled', False)
        self.fourier_module: Optional[FourierSpatialEmbedding] = None
        if self.use_fourier:
            fourier_params = {
                k: v for k, v in fourier_cfg.items()
                if k not in ('enabled', 'spatial_coordinates_path')
            }
            self.fourier_module = FourierSpatialEmbedding(emb_size=emb_size, **fourier_params)

            # Load coordinates from path if not provided directly
            spatial_path = fourier_cfg.get('spatial_coordinates_path')
            if reference_coordinates is None and spatial_path is not None:
                import pickle
                spatial_path = Path(spatial_path) if not isinstance(spatial_path, Path) else spatial_path
                with open(spatial_path, 'rb') as f:
                    reference_coordinates = pickle.load(f)
                self.logger.info(f"Loaded spatial coordinates from {spatial_path}")

            if reference_coordinates is not None and channel_names is not None:
                self.fourier_module.set_coordinates(reference_coordinates, channel_names)
            self._fourier_reference_coordinates = reference_coordinates
            self.strategies.append('fourier')

        # Special token embeddings (always present for masking)
        self.missing_channel_embedding = nn.Parameter(torch.randn(1, emb_size))
        self.unk_channel_embedding = nn.Parameter(torch.randn(1, emb_size))

        if not self.strategies:
            raise ValueError(
                "At least one channel embedding strategy must be enabled. "
                "Set learned.enabled=true, spectral.enabled=true, or fourier.enabled=true."
            )

        self.logger.info(f"ChannelEmbeddingComposer strategies: {self.strategies}")

    def set_fourier_coordinates(
        self,
        coordinates_dict: Dict[str, List[float]],
        channel_names: List[str],
    ) -> None:
        """Set Fourier spatial coordinates after initialization.

        Call this once channel names are known (e.g. from the data module).

        Args:
            coordinates_dict: Mapping from channel name to [x, y, z].
            channel_names: Ordered list of channel names.
        """
        if self.fourier_module is None:
            raise RuntimeError(
                "Cannot set Fourier coordinates: fourier strategy is not enabled."
            )
        self.fourier_module.set_coordinates(coordinates_dict, channel_names)
        self._fourier_reference_coordinates = coordinates_dict

    def compute_spectral_embeddings(
        self,
        x_for_plv: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute spectral embeddings from signal data.

        Called at the hierarchical model level where the full signal is available.

        Args:
            x_for_plv: Signal for PLV computation (B, C, T_effective).
            channel_mask: Optional mask (B, C), True=valid.

        Returns:
            Spectral embeddings (B, C, emb_size) or None if spectral is not enabled.
        """
        if self.spectral_module is None:
            return None
        return self.spectral_module(x_for_plv, channel_mask)

    def forward(
        self,
        batch_size: int,
        n_channels: int,
        device: torch.device,
        channel_mask: Optional[torch.Tensor] = None,
        spectral_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose channel embeddings from all enabled strategies.

        Args:
            batch_size: B (or B*N for virtual batch in hierarchical model).
            n_channels: Number of channels C.
            device: Target device.
            channel_mask: Optional (B, C) validity mask.
            spectral_embs: Precomputed spectral embeddings (B, C, E), or None.

        Returns:
            Combined channel embeddings of shape (B, C, emb_size).
        """
        combined = torch.zeros(
            batch_size, n_channels, self.emb_size, device=device
        )

        if self.use_learned:
            combined = combined + self.channel_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        if self.use_spectral and spectral_embs is not None:
            combined = combined + spectral_embs

        if self.use_fourier and self.fourier_module is not None:
            fourier_emb = self.fourier_module(channel_mask)  # (C, E)
            combined = combined + fourier_emb.unsqueeze(0).expand(batch_size, -1, -1)

        return combined


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

