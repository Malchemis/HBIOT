#!/usr/bin/env python3
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer

from pipeline.models.commons import AttentionClassificationHead, ClassificationHead, PatchFrequencyEmbedding, PatchTimeEmbedding, PatchFeatureEmbedding, stft
from pipeline.models.full_attention_transformer import FullAttentionTransformer


def log_tensor_statistics(tensor: torch.Tensor, name: str, logger_obj: Optional[logging.Logger] = None) -> None:
    """Log detailed statistics about a tensor for debugging NaN/inf issues.

    Args:
        tensor: Tensor to analyze
        name: Name/description of the tensor
        logger_obj: Logger to use (defaults to module logger)
    """
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__)

    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    n_total = tensor.numel()

    if n_nan > 0 or n_inf > 0:
        logger_obj.error(f"ALERT {name}: NaN={n_nan}/{n_total} ({100*n_nan/n_total:.2f}%), Inf={n_inf}/{n_total} ({100*n_inf/n_total:.2f}%)")

    if n_nan == 0 and n_inf == 0:
        logger_obj.debug(f"OK {name}: shape={tuple(tensor.shape)}, mean={tensor.float().mean():.4f}, std={tensor.float().std():.4f}, "
                        f"min={tensor.float().min():.4f}, max={tensor.float().max():.4f}, NaN=0, Inf=0")


class BIOTClassifier(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) Classifier.
    
    This model uses the BIOT encoder for feature extraction and adds a classification head.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        classifier (ClassificationHead): Classification head.
    """
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, n_classes: int = 1, 
                 mode: str = "spec", overlap: float = 0.0, log_dir: Optional[str] = None, n_selected_tokens: int = 3, 
                 use_cls_token: bool = True, use_mean_pool: bool = True, use_max_pool: bool = True, **kwargs):
        """Initialize the BIOT classifier.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_classes: Number of output classes.
            mode: Processing mode ("raw", "spec", or "features").
            overlap: Overlap between patches.
            log_dir: Optional directory for log files. If None, logs to console only.
            n_selected_tokens: Number of tokens to select and return (for multi-token classification).
            use_cls_token: Whether to include CLS token in output.
            use_mean_pool: Whether to include mean pooling token in output.
            use_max_pool: Whether to include max pooling token in output.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTClassifier")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "biot_classifier.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing BIOT classifier with emb_size={emb_size}, heads={heads}, "
                         f"depth={depth}, n_classes={n_classes}, mode={mode}")

        # Initialize the BIOT encoder: Channel and Positional embeddings, CLS token, and (linear attention) transformer
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, mode=mode, 
                                overlap=overlap, n_selected_tokens=n_selected_tokens, 
                                use_cls_token=use_cls_token, use_mean_pool=use_mean_pool, 
                                use_max_pool=use_max_pool, **kwargs)
        self.n_classes: int = n_classes
        
        # Get total number of tokens from encoder
        n_tokens_per_window = self.biot.n_tokens_per_window
        
        if n_tokens_per_window > 1:
            # Multi-token classification head for handling different representation types
            self.classifier = AttentionClassificationHead(
                emb_size=emb_size, 
                n_classes=n_classes,
                n_tokens_per_window=n_tokens_per_window,
            )
        else:
            # Single token - use simple classification head
            self.classifier = ClassificationHead(emb_size=emb_size, n_classes=n_classes)

    def forward(self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the BIOT classifier.

        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            channel_mask: Optional batch-aware mask (B, C) where True=valid, False=masked.

        Returns:
            Logits for each class.
        """
        biot_output = self.biot(x, channel_mask=channel_mask, *args, **kwargs)  # shape: (batch_size, n_tokens, emb_size) or (batch_size, emb_size)
        logits = self.classifier(biot_output)
        return logits  # shape: (batch_size, n_classes)

 
# TODO: Update to current BIOTEncoder
class UnsupervisedPretrainBiot(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=275, **kwargs):
        super(UnsupervisedPretrainBiot, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x):
        emb = self.biot(x, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x)
        return emb, pred_emb


class BIOTEncoder(nn.Module):
    """Modified BIOT Encoder with multi-representation output for window processing.

    This encoder implements the core window processing logic for the hierarchical BIOT model.
    It processes individual MEG windows through patch embeddings, channel embeddings,
    transformer attention, and multiple output representations.

    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ INPUT: (batch_size x n_windows, n_channels, n_samples_per_window)│
    │        Example: (800, 64, 80) [treating each window independently]          │
    └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 1: Per-Channel Patch Embedding                                        │
    │ • Raw mode: Split temporal data into overlapping patches                    │
    │ • Spectral mode: STFT → frequency domain patches                            │
    │ • Each channel: (1, 80) → (5, 256) [5 patches per channel]                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 2: Channel Token Addition                                             │
    │ • Add learnable channel identity embeddings to each patch                   │
    │ • Enables model to distinguish between different MEG channels               │
    │ • Shape: (BxN, 64x5, 256) = (BxN, 320, 256)                                 │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 3: CLS Token Addition & Positional Encoding                           │
    │ • Prepend learnable CLS token for global window representation             │
    │ • Add positional encodings for temporal patch order                         │
    │ • Shape: (BxN, 321, 256) [1 CLS + 320 patches]                              │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 4: Transformer Processing                                             │
    │ • Linear attention for efficient processing of long sequences               │
    │ • Models inter-channel and intra-temporal dependencies                      │
    │ • Shape: (BxN, 321, 256) → (BxN, 321, 256)                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 5: Multi-Representation Output                                        │
    │ • CLS Token: Global window summary                                         │
    │ • Mean: Average all token emb                                               │
    │ • Max: Max of all token emb                                                 │
    │ • Selected: Top-k most informative patches via attention                    │
    │ • Combined: (BxN, variable, 256) [configurable token types]                │
    └─────────────────────────────────────────────────────────────────────────────┘

    MULTI-REPRESENTATION STRATEGY:
    1. CLS Token: Learned global representation optimized through attention
    2. Pooled Representation: Statistical summary (mean and max) of all patches
    3. Selected Tokens: Attention-weighted selection of most discriminative patches
    
    This approach provides configurable views of window information:
    - Global context (CLS) - if use_cls_token=True
    - Statistical summary (pooled) - if use_mean_pool/use_max_pool=True
    - Local discriminative features (selected) - if n_selected_tokens > 0

    GRADIENT FLOW:
    • CLS token receives gradients from final classification loss
    • Selected tokens receive attention-weighted gradients based on importance
    • Pooled representation receives uniform gradients from all patches
    • Channel embeddings shared across all patches from same channel

    Attributes:
        training (bool): Training mode flag for augmentations
        patch_frequency_embedding (PatchFrequencyEmbedding): Processes spectral domain data
        patch_time_embedding (PatchTimeEmbedding): Processes raw temporal data with overlap
        cls_token (nn.Parameter): Learnable global window representation token
        transformer (Transformer): Linear/full attention transformer for sequence processing
        positional_embedding (nn.Parameter): Learnable positional encodings for patch order
        channel_tokens (nn.Embedding): Learnable channel identity embeddings
        spatial_channel_embedding (SpatialChannelEmbedding): Location-based channel embeddings
        token_selector (nn.Sequential): MLP for scoring patch importance
        cls_proj (nn.Linear): Projects CLS token to final embedding space
        pool_proj (nn.Linear): Projects pooled representation to final embedding space
        selected_proj (nn.Linear): Projects selected tokens to final embedding space
    """

    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            depth: int = 4,
            n_selected_tokens: int = 0,
            selection_temperature: float = 1.0,
            n_channels: int = 275,
            n_samples_per_window: int = 40,
            token_size: int = 200,
            overlap: float = 0.0,
            mode: str = "spec",
            sfreq: float = 200.0,
            linear_attention: bool = False,
            attn_dropout: float = 0.2,
            ff_dropout: float = 0.2,
            use_cls_token: bool = True,
            use_mean_pool: int = 0,
            use_max_pool: bool = False,
            use_min_pool: bool = False,
            channel_embedding_composer: Optional['ChannelEmbeddingComposer'] = None,
            **kwargs
    ):
        """Initialize the modified BIOT encoder.

        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_selected_tokens: Number of tokens to select and return. (instead of a single CLS token for example)
            use_cls_token: Whether to include CLS token in output.
            use_mean_pool: Whether to include moments in output. It is an integer indicating that the first k moments should be used.
            use_max_pool: Whether to include max pooling token in output.
            use_min_pool: Whether to include min pooling token in output.
            selection_temperature: Temperature for token selection.
            n_samples_per_window: Number of samples in any given raw window.
            n_channels: Number of input channels.
            token_size: Number of FFT points for Spectral mode / Number of samples for Raw mode.
            overlap: Overlap between patches for raw data mode.
            linear_attention: Whether to use linear attention or full attention.
            channel_embedding_composer: Shared ChannelEmbeddingComposer for channel embeddings.
                If None, a default learned-only composer is created (standalone use).
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".ModifiedBIOTEncoder")
        self.n_fft = token_size
        self.hop_length = int(token_size * (1 - overlap))
        self.mode = mode
        self.sfreq = sfreq

        # Validate mode parameter
        valid_modes = ["raw", "spec", "features"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        self.logger.info(f"Modified BIOT encoder initialized")
        self.logger.info(f"Processing mode: {mode}")

        # Create embedding modules for different processing modes
        if mode == "spec":
            self.patch_embedding = PatchFrequencyEmbedding(
                emb_size=emb_size, n_freq=self.n_fft // 2 + 1
            )
        elif mode == "raw":
            self.patch_embedding = PatchTimeEmbedding(
                emb_size=emb_size, patch_size=token_size, overlap=overlap
            )
        elif mode == "features":
            self.patch_embedding = PatchFeatureEmbedding(
                emb_size=emb_size, patch_size=token_size, overlap=overlap, sfreq=sfreq
            )
        else:
            # warn and use raw mode as fallback
            self.logger.warning(f"Invalid mode '{mode}', defaulting to 'raw'")
            self.patch_embedding = PatchTimeEmbedding(
                emb_size=emb_size, patch_size=token_size, overlap=overlap
            )

        # CLS token for aggregating window information
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Configurable token selection parameters
        self.use_cls_token = use_cls_token
        self.use_mean_pool = use_mean_pool
        self.use_max_pool = use_max_pool
        self.use_min_pool = use_min_pool
        self.n_selected_tokens = n_selected_tokens
        self.selection_temperature = selection_temperature
        
        # Calculate total number of tokens per window
        n_tokens_per_window = 0
        if use_cls_token:
            n_tokens_per_window += 1
            self.cls_proj = nn.Linear(emb_size, emb_size)
        if use_mean_pool:
            # k = 1: mean, k = 2: mean+variance, etc.
            for k in range(1, use_mean_pool + 1):
                setattr(self, f'mean_pool_proj_{k}', nn.Linear(emb_size, emb_size))
                n_tokens_per_window += 1
        if use_max_pool:
            n_tokens_per_window += 1
            self.max_pool_proj = nn.Linear(emb_size, emb_size)

        if use_min_pool:
            n_tokens_per_window += 1
            self.min_pool_proj = nn.Linear(emb_size, emb_size)
        
        if n_tokens_per_window == 0:
            raise ValueError("At least one token type must be enabled (CLS, mean_pool, max_pool, or selected_tokens > 0)")
                
        if n_selected_tokens > 0:
            n_tokens_per_window += n_selected_tokens
            self.token_selector = nn.Sequential(
                nn.Linear(emb_size, emb_size // 2),
                nn.ReLU(),
                nn.Linear(emb_size // 2, 1)  # Score for each token
            )
            self.selected_proj = nn.Linear(emb_size, emb_size)

        self.n_tokens_per_window = n_tokens_per_window
        
        self.logger.info(f"Token configuration: CLS={use_cls_token}, Mean={use_mean_pool}, Max={use_max_pool}, Selected={n_selected_tokens}")
        self.logger.info(f"Total tokens per window: {n_tokens_per_window}")

        # Transformer with increased max_seq_len to accommodate the extra CLS token
        n_tokens = int((n_samples_per_window - token_size) / (token_size * (1 - overlap)) + 1) * n_channels
        self.logger.info(f"Max sequence length for transformer: {n_tokens + 1}")
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=n_tokens + 1,  # +1 for CLS token
            attn_dropout=attn_dropout,
        ) if linear_attention else FullAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=n_tokens + 1,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_flash_attn=True,
            use_rmsnorm=True
        )

        # Positional embeddings for temporal order within each channel
        self.n_patches_per_channel = int((n_samples_per_window - token_size) / (token_size * (1 - overlap)) + 1)
        self.logger.info(f"Patches per channel: {self.n_patches_per_channel}")
        self.positional_embedding = nn.Parameter(torch.randn(self.n_patches_per_channel, emb_size))
       
        # Channel embeddings to distinguish MEG channels
        # Use provided composer or create a default learned-only one (standalone mode)
        if channel_embedding_composer is not None:
            self.channel_embedding_composer = channel_embedding_composer
        else:
            from pipeline.models.commons import ChannelEmbeddingComposer
            self.channel_embedding_composer = ChannelEmbeddingComposer(
                n_channels=n_channels,
                emb_size=emb_size,
                config={'learned': {'enabled': True}},
            )

    def forward(self, x: torch.Tensor, channel_mask: Optional[torch.Tensor], unk_augment: float = 0.0, unknown_mask: Optional[torch.Tensor] = None, spectral_channel_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with detailed multi-stage processing and shape tracking.

        DATA FLOW:

        Input → Per-Channel Processing → Channel Tokens → CLS Token →
        Positional Encoding → Transformer → Multi-Representation Output

        SHAPE TRANSFORMATIONS:
        (BxN, C, T) → Per-channel: (BxN, 1, T) → (BxN, P, E) →
        Concatenated: (BxN, C*P, E) → With CLS: (BxN, C*P+1, E) →
        Final: (BxN, n_tokens_per_window, E)

        PROCESSING STAGES:
        1. Channel-wise patch embedding (raw or spectral)
        2. Channel identity token addition
        3. CLS token prepending and positional encoding
        4. Transformer attention processing
        5. Multi-representation extraction

        Args:
            x (torch.Tensor): Input MEG window data.
                Shape: (batch_size x n_windows, n_channels, n_samples_per_window)
                Example: (800, 275, 80)
                - 800 = batch_size x n_windows (e.g., 32 x 25)
                - 275 = number of MEG channels
                - 80 = samples per window (400ms at 200Hz)

            loc: Mapping from channel names to positions.
            channel_mask: Channel mask (BxN, C) where 1=valid, 0=masked.
            unk_augment: Percentage of channels to randomly replace with [UNK] token during training (0.0-1.0)
            unknown_mask: Optional mask (BxN, C) where 1=unknown, 0=known. Overrides unk_augment if provided. This is for inference-time unknown channel handling.

        Returns:
            torch.Tensor: Multi-representation window embeddings.
                Shape: (batch_size x n_windows, n_tokens_per_window, emb_size)
                Example: (800, 6, 256) where 6 depends on token configuration
                Contains configurable representations per window based on enabled token types:
                - CLS token (if use_cls_token=True): global window summary
                - Mean pooling (if use_mean_pool>0): k-th moment statistics
                - Max pooling (if use_max_pool=True): max embeddings
                - Min pooling (if use_min_pool=True): min embeddings
                - Selected tokens (if n_selected_tokens > 0): discriminative local features
        """
        batch_size = x.shape[0]  # BxN (e.g., 800)
        n_channels = x.shape[1]  # Number of channels

        # Log input statistics
        log_tensor_statistics(x, f"BIOTEncoder input (batch_size={batch_size}, n_channels={n_channels})", self.logger)
        if channel_mask is not None:
            n_valid_channels = channel_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f"BIOTEncoder channel_mask: avg valid channels={n_valid_channels:.1f}/{n_channels}")

        emb_seq = []

        # Process each MEG channel independently to create temporal patches
        for i in range(n_channels):
            # Extract single channel data
            channel_data = x[:, i:i + 1, :]  # (BxN, 1, T)

            # DEBUG: Check first channel only
            if i == 0:
                log_tensor_statistics(channel_data, f"BIOTEncoder channel {i} input", self.logger)

            # Choose embedding strategy based on processing mode
            if self.mode == "spec":
                # Shape: (BxN, 1, T) → STFT → (BxN, freq, time)
                channel_data = stft(channel_data, n_fft=self.n_fft, hop_length=self.hop_length)
                if i == 0:
                    log_tensor_statistics(channel_data, f"BIOTEncoder channel {i} after STFT", self.logger)

            # Shape: (BxN, 1, T) → (BxN, P, E) where P = n_patches_per_channel
            channel_tokens = self.patch_embedding(channel_data)

            if i == 0:
                log_tensor_statistics(channel_tokens, f"BIOTEncoder channel {i} after patch_embedding", self.logger)
                log_tensor_statistics(self.positional_embedding, f"BIOTEncoder positional_embedding", self.logger)

            # Add positional embeddings for temporal order within channel
            channel_tokens = channel_tokens + self.positional_embedding[:channel_tokens.size(1)].unsqueeze(0)
            # Shape: (BxN, P, E) + (1, P, E) → (BxN, P, E)

            if i == 0:
                log_tensor_statistics(channel_tokens, f"BIOTEncoder channel {i} after adding positional", self.logger)

            # add to list of channel embeddings
            emb_seq.append(channel_tokens)

        ## Combine all channel embeddings into single sequence
        emb = torch.cat(emb_seq, dim=1)  # (BxN, C*P, E)
        log_tensor_statistics(emb, "BIOTEncoder after patch embedding concatenation", self.logger)

        ## Add channel identity embeddings (composed from enabled strategies)
        channel_embs = self.channel_embedding_composer(
            batch_size=batch_size,
            n_channels=n_channels,
            device=x.device,
            channel_mask=channel_mask,
            spectral_embs=spectral_channel_embs,
        )  # (BxN, C, E)

        if channel_mask is None:
            channel_mask = torch.ones((batch_size, n_channels), dtype=torch.bool, device=x.device)

        # Handle missing channels (where channel_mask is False)
        # channel_mask: (BxN, C) - True=valid, False=masked
        missing_mask = ~channel_mask  # (BxN, C) - True where channel is missing
        channel_embs = torch.where(
            missing_mask.unsqueeze(-1),  # (BxN, C, 1)
            self.channel_embedding_composer.missing_channel_embedding.expand(batch_size, n_channels, -1),
            channel_embs
        )

        # Apply unknown channel augmentation during training
        if self.training and unk_augment > 0.0:
            # Randomly select valid channels to mark as unknown
            valid_mask = channel_mask  # (BxN, C) - True where channel is valid
            # Create random mask for augmentation
            aug_mask = torch.rand(batch_size, n_channels, device=x.device) < unk_augment
            # Only augment valid channels (not already missing)
            unk_mask = valid_mask & aug_mask  # (BxN, C)

            channel_embs = torch.where(
                unk_mask.unsqueeze(-1),  # (BxN, C, 1)
                self.channel_embedding_composer.unk_channel_embedding.expand(batch_size, n_channels, -1),
                channel_embs
            )
        elif unknown_mask is not None:
            # Apply provided unknown mask (inference-time handling)
            channel_embs = torch.where(
                unknown_mask.unsqueeze(-1),  # (BxN, C, 1)
                self.channel_embedding_composer.unk_channel_embedding.expand(batch_size, n_channels, -1),
                channel_embs
            )

        # Expand channel embeddings to all patches and add to patch embeddings
        # channel_embs: (BxN, C, E) -> (BxN, C, P, E) -> (BxN, C*P, E)
        n_patches_per_channel = emb.size(1) // n_channels
        channel_embs_expanded = channel_embs.unsqueeze(2).expand(-1, -1, n_patches_per_channel, -1)  # (BxN, C, P, E)
        channel_embs_flat = channel_embs_expanded.reshape(batch_size, -1, channel_embs.size(-1))  # (BxN, C*P, E)

        # Add channel embeddings to patch embeddings
        emb = emb + channel_embs_flat  # (BxN, C*P, E)
        log_tensor_statistics(emb, "BIOTEncoder after adding channel embeddings", self.logger)

        ## CLS Token Addition & Positional Encoding
        # Add learnable CLS token for global window representation
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # (BxN, 1, E)
        emb_with_cls = torch.cat([cls_tokens, emb], dim=1)   # (BxN, CxP+1, E)
        
        ## Transformer Processing
        padded_mask = None
        if channel_mask is not None:
            # Create patch-level mask from channel-level mask
            # channel_mask: (BxN, C) -> (BxN, C, P) -> (BxN, C*P)
            patch_mask = channel_mask.unsqueeze(-1).expand(-1, -1, n_patches_per_channel).reshape(batch_size, -1)  # (BxN, C*P)
            # Add True for CLS token at start
            padded_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=x.device), patch_mask], dim=1)  # (BxN, C*P+1)
        output = self.transformer(emb_with_cls, mask=padded_mask)  # (BxN, CxP+1, E) → (BxN, CxP+1, E)
        log_tensor_statistics(output, "BIOTEncoder after transformer", self.logger)

        ## Multi-Representation Output Generation
        representations = []
        
        # 1. CLS TOKEN REPRESENTATION: Global window summary
        if self.use_cls_token:
            cls_output = self.cls_proj(output[:, 0, :])  # (BxN, E)
            representations.append(cls_output.unsqueeze(1))  # (BxN, 1, E)
        
        # Get sequence output (exclude CLS token)
        sequence_output = output[:, 1:, :]  # (BxN, CxP, E) - exclude CLS token
        
        # We should not consider masked patches for pooled representations to avoid contamination
        sequence_mask = padded_mask[:, 1:] if padded_mask is not None else None  # (BxN, CxP)
        
        # 2. POOLED REPRESENTATIONS: Statistical summary of all patches
        if self.use_mean_pool:
            # Compute mean once outside the loop: either by masked mean or regular mean
            if sequence_mask is not None:
                masked_seq = sequence_output * sequence_mask.unsqueeze(-1).float()  # (BxN, CxP, E)
                sum_mask = sequence_mask.sum(dim=1, keepdim=True).float().unsqueeze(-1)  # (BxN, 1, 1)
                mean = masked_seq.sum(dim=1, keepdim=True) / sum_mask.clamp(min=1e-6)  # (BxN, 1, E)
            else:
                mean = sequence_output.mean(dim=1, keepdim=True)  # (BxN, 1, E)
            
            for k in range(1,self.use_mean_pool + 1):
                if k == 1:
                    # First moment is the mean
                    kth_moment = mean.squeeze(1)  # (BxN, E)
                else:
                    # Central moments: E[(X - μ)^k]
                    # Note: for k=2 this gives variance, k=3 gives skewness (unnormalized), etc.
                    centered = sequence_output - mean  # (BxN, CxP, E)
                    if sequence_mask is not None:
                        centered = centered * sequence_mask.unsqueeze(-1).float()  # Masked
                        sum_mask = sequence_mask.sum(dim=1, keepdim=True).float()  # (BxN, 1)
                        kth_moment = centered.pow(k).sum(dim=1) / sum_mask.clamp(min=1e-6)  # (BxN, E)
                    else:
                        kth_moment = centered.pow(k).mean(dim=1)  # (BxN, E)
                
                # Project and add to representations
                moment_proj = getattr(self, f'mean_pool_proj_{k}')(kth_moment)  # (BxN, E)
                representations.append(moment_proj.unsqueeze(1))  # (BxN, 1, E)
               
        if self.use_max_pool:
            if sequence_mask is not None:
                # Masked max pooling: set masked positions to large negative value
                masked_seq = sequence_output.masked_fill(~sequence_mask.unsqueeze(-1), float('-inf'))
                max_pool = self.max_pool_proj(masked_seq.max(dim=1).values)  # (BxN, E)
            else:
                max_pool = self.max_pool_proj(sequence_output.max(dim=1).values)  # (BxN, E)
            representations.append(max_pool.unsqueeze(1))  # (BxN, 1, E)
            
        if self.use_min_pool:
            if sequence_mask is not None:
                # Masked min pooling: set masked positions to large positive value
                masked_seq = sequence_output.masked_fill(~sequence_mask.unsqueeze(-1), float('inf'))
                min_pool = self.min_pool_proj(masked_seq.min(dim=1).values)  # (BxN, E)
            else:
                min_pool = self.min_pool_proj(sequence_output.min(dim=1).values)  # (BxN, E)
            representations.append(min_pool.unsqueeze(1))  # (BxN, 1, E)
        
        # 3. SELECTED TOKEN REPRESENTATION: Most discriminative local features
        if self.n_selected_tokens > 0:
            token_scores = self.token_selector(sequence_output).squeeze(-1)  # (BxN, CxP)
            
            # Apply temperature-controlled softmax for differentiable selection
            selection_weights = torch.softmax(token_scores / self.selection_temperature, dim=-1)
            
            # Select top-k most important tokens
            _, topk_indices = torch.topk(selection_weights, self.n_selected_tokens, dim=-1)
            
            # Gather selected tokens using indices
            emb_size = sequence_output.size(-1)  # E
            selected_tokens = []
            for i in range(self.n_selected_tokens):
                indices = topk_indices[:, i].unsqueeze(-1).unsqueeze(-1)  # (BxN, 1, 1)
                indices = indices.expand(-1, -1, emb_size)  # (BxN, 1, E)
                selected = torch.gather(sequence_output, 1, indices).squeeze(1)  # (BxN, E)
                selected_tokens.append(self.selected_proj(selected).unsqueeze(1))  # (BxN, 1, E)
            
            selected_tokens = torch.cat(selected_tokens, dim=1)  # (BxN, n_selected, E)
            representations.append(selected_tokens)
        
        # Combine all enabled representations
        if len(representations) == 1:
            # Single token type - return without extra dimension
            log_tensor_statistics(representations[0], "BIOTEncoder final output (single token)", self.logger)
            return representations[0]
        else:
            # Multiple token types - concatenate along sequence dimension
            combined = torch.cat(representations, dim=1)  # (BxN, total_tokens, E)
            log_tensor_statistics(combined, f"BIOTEncoder final output (combined {len(representations)} token types)", self.logger)
            return combined


def setup_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO):
    """Set up logging configuration.
    
    Args:
        log_dir: Directory for log files. If None, logs to console only.
        log_level: Logging level.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    # File handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "biot.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)


if __name__ == "__main__":
    # Set up command-line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BIOT models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log files")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--channels", type=int, default=2, help="Number of channels in test data")
    parser.add_argument("--time_steps", type=int, default=2000, help="Number of time steps in test data")
    parser.add_argument("--n_fft", type=int, default=200, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=100, help="Hop length for STFT")
    parser.add_argument("--patch_size", type=int, default=100, help="Patch size for raw mode")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap for raw mode (0.0-1.0)")
    parser.add_argument("--raw", action="store_true", help="Use raw time-domain processing")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_dir, log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing BIOT models")
    logger.info(f"Arguments: {args}")
    
    # Create test data
    data = torch.randn(args.batch_size, args.channels, args.time_steps)
    logger.info(f"Test data shape: {data.shape}")

    # Test with spectral processing
    if not args.raw:
        logger.info("Testing spectral processing model")
        model = BIOTClassifier(
            n_fft=args.n_fft, 
            hop_length=args.hop_length, 
            depth=4, 
            heads=8, 
            raw=False,
            log_dir=args.log_dir
        )
        out = model(data)
        logger.info(f"Spectral output shape: {out.shape}")

    # Test with raw data processing (no overlap)
    logger.info("Testing raw processing model (no overlap)")
    model_raw = BIOTClassifier(
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        depth=4, 
        heads=8, 
        raw=True, 
        patch_size=args.patch_size, 
        overlap=0.0,
        log_dir=args.log_dir
    )
    out_raw = model_raw(data)
    logger.info(f"Raw output shape (no overlap): {out_raw.shape}")

    # Test with raw data processing (with overlap)
    if args.overlap > 0:
        logger.info(f"Testing raw processing model (with {args.overlap*100}% overlap)")
        model_raw_overlap = BIOTClassifier(
            n_fft=args.n_fft, 
            hop_length=args.hop_length, 
            depth=4, 
            heads=8, 
            raw=True, 
            patch_size=args.patch_size,
            overlap=args.overlap,
            log_dir=args.log_dir
        )
        out_raw_overlap = model_raw_overlap(data)
        logger.info(f"Raw output shape (with overlap): {out_raw_overlap.shape}")
    
    logger.info("All tests completed successfully")
