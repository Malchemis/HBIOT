#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer

from pipeline.models.full_attention_transformer import FullAttentionTransformer
from pipeline.models.biot import BIOTEncoder
from pipeline.models.commons import ChannelEmbeddingComposer


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


class BIOTHierarchicalClassifier(nn.Module):
    """BIOT Hierarchical Encoder for MEG spike detection.

    This encoder implements a sophisticated two-stage hierarchical attention mechanism
    specifically designed for processing long sequences of high-dimensional MEG data:
    
    ARCHITECTURE OVERVIEW:
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │ INPUT: (batch_size, n_windows, n_channels, n_samples_per_window)               │
    │       Default: (B, 25, 275, 80) - 25 windows of 400ms at 200Hz, 275 MEG channels│
    └──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 1: Intra-window Attention (Per window Processing)                    │
    │ • Process each window independently using BIOTEncoder                       │
    │ • Extract multiple representations: CLS + Pooled + Selected tokens (e.g. 6t.)│
    │ • Shape: (BxN, 275, 80) → (B, N, 6, emb_size)                                │
    └──────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 2: window Positional Encoding                                        │
    │ • Add learnable position embeddings for temporal window order              │
    │ • Shape: (B, N, 6, emb_size) → (B, N, 6, emb_size)                          │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 3: Inter-window Attention                                            │
    │ • Model long-range temporal dependencies between windows                   │
    │ • Shape: (B, Nx6, emb_size) → (B, Nx6, emb_size)                            │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ STAGE 4: Classification Head with Attention                                 │
    │ • Attention-based aggregation of tokens per window                         │
    │ • Shape: (B, Nx6, emb_size) → (B, N, n_classes)                             │
    └─────────────────────────────────────────────────────────────────────────────┘

    INFORMATION EXTRACTION HIERARCHY:
    1. Local temporal patterns via patch embeddings per channel
    2. Inter-channel interactions through window-level attention
    3. window-level feature extraction via multiple representation types
    4. Long-range temporal context through inter-window attention
    5. Classification via attention-based token aggregation

    Attributes:
        window_encoder (BIOTEncoder): Processes individual windows with intra-attention
        window_pos_embedding (nn.Parameter): Learnable positional embeddings for windows
        inter_window_transformer (Transformer): Models dependencies between windows
        classifier (AttentionClassificationHead): Attention-based classification head
        n_windows (int): Number of temporal windows in input
        n_tokens_per_window (int): Number of tokens extracted per window (CLS + pooled + selected)
    """

    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            window_encoder_depth: int = 4,
            inter_window_depth: int = 4,
            token_size: int = 40,
            overlap: float = 0.5,
            mode: str = "raw",
            linear_attention: bool = True,
            input_shape: Optional[Tuple[int, int, int]] = None,
            transformer: Optional[dict] = None,
            token_selection: Optional[dict] = None,
            classifier: Optional[dict] = None,
            n_classes: int = 1,
            max_virtual_batch_size: int = 640,
            channel_embedding: Optional[dict] = None,
            window_overlap: float = 0.5,
            **kwargs
    ):
        """Initialize the BIOT hierarchical encoder.

        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            window_encoder_depth: Number of transformer layers for intra-window attention.
            inter_window_depth: Number of transformer layers for inter-window attention.
            token_size: Size of each token window.
            overlap: Overlap percentage between windows.
            mode: Processing mode ("raw", "spec", or "features").
            linear_attention: Whether to use linear attention for transformers.
            input_shape: Shape of the input data non batched (n_windows, n_channels, n_samples_per_window)
            transformer: Parameters for transformer layers.
                Must include 'attn_dropout' and 'ff_dropout' keys.
            token_selection: Parameters for token selection.
                Can include 'n_selected_tokens', 'use_cls_token', 'use_mean_pool', 'use_max_pool' keys.
            classifier: Parameters for the classification head.
            n_classes: Number of output classes for classification.
            max_virtual_batch_size: Maximum virtual batch size (BxN) to process in a single forward pass. Defaults to 640 = 32x20 (tested empirically on h100 with 80GB RAM).
                When BxN exceeds this value, processing is done in chunks. Default: 640.
            channel_embedding: Composable channel embedding config with keys:
                learned: {enabled: bool}
                spectral: {enabled: bool, n_eigenvectors, projection_type, ...}
                fourier: {enabled: bool, n_frequencies, learnable_linear, ...}
            window_overlap: Overlap between consecutive temporal windows (must match
                dataset_config.window_overlap). Used to reconstruct non-overlapping signal
                for PLV computation.
            **kwargs: Additional keyword arguments for flexibility.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTHierarchicalEncoder")
        self.max_virtual_batch_size = max_virtual_batch_size

        assert input_shape is not None, "Input shape must be provided for the encoder."
        assert transformer is not None, "Transformer parameters must be provided."
        assert token_selection is not None, "Token selection parameters must be provided."
        assert classifier is not None, "Classifier parameters must be provided."

        print(f"Initializing BIOTHierarchicalClassifier with input shape: {input_shape}")
        n_windows, n_channels, n_samples_per_window = input_shape

        self.logger.info(f"Max virtual batch size set to {max_virtual_batch_size}. "
                        f"Expected virtual batch size for input shape: {input_shape[0]} windows = "
                        f"B x {input_shape[0]} = variable virtual batch")

        # Save channel dimensions
        self.n_channels = n_channels
        self.n_windows = n_windows
        
        # Calculate n_tokens_per_window based on token selection configuration
        use_cls_token = token_selection.get("use_cls_token", True)
        use_mean_pool = token_selection.get("use_mean_pool", 1)     # 1: mean, 2: mean+variance, etc.
        use_max_pool = token_selection.get("use_max_pool", True)
        use_min_pool = token_selection.get("use_min_pool", True)
        n_selected_tokens = token_selection.get("n_selected_tokens", 3)
        
        self.n_tokens_per_window = 0
        if use_cls_token:
            self.n_tokens_per_window += 1
        if use_mean_pool:
            self.n_tokens_per_window += use_mean_pool
        if use_max_pool:
            self.n_tokens_per_window += 1
        if use_min_pool:
            self.n_tokens_per_window += 1
        self.n_tokens_per_window += n_selected_tokens
        
        if self.n_tokens_per_window == 0:
            raise ValueError("At least one token type must be enabled in token_selection config")

        # Window overlap for PLV signal reconstruction
        self.window_overlap = window_overlap
        assert 0.0 <= self.window_overlap < 1.0, "window_overlap must be in the range [0.0, 1.0)"

        # Channel embedding composer: composable strategy (learned, spectral, fourier)
        # Must be created before BIOTEncoder which receives a reference to it.
        channel_emb_config = channel_embedding if channel_embedding is not None else {'learned': {'enabled': True}}
        self.channel_embedding_composer = ChannelEmbeddingComposer(
            n_channels=n_channels,
            emb_size=emb_size,
            config=channel_emb_config,
        )

        # Modified BIOT encoder for window-level processing with configurable tokens
        self.window_encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=window_encoder_depth,
            n_selected_tokens=n_selected_tokens,          # Number of selected tokens per window
            use_cls_token=use_cls_token,                  # Whether to use CLS token
            use_mean_pool=use_mean_pool,                  # Whether to use mean pooling token
            use_max_pool=use_max_pool,                    # Whether to use max pooling token
            use_min_pool=use_min_pool,                    # Whether to use min pooling token
            n_channels=n_channels,                        # Use full channel count
            n_samples_per_window=n_samples_per_window,    # Number of tokens depends on this, token_size, and overlap
            token_size=token_size,
            overlap=overlap,
            mode=mode,                                    # Processing mode: "raw", "spec", or "features"
            linear_attention=linear_attention,            # Use linear attention or full attention for window encoder
            channel_embedding_composer=self.channel_embedding_composer,  # Shared composer for channel embeddings
        )

        # Learnable embeddings for each window position
        # Each window gets the same positional encoding for all its tokens
        self.window_pos_embedding = nn.Parameter(
            torch.randn(n_windows, emb_size)
        )  # (N, emb_size)

        # Inter-window transformer
        self.logger.info(f"Using {'linear' if linear_attention else 'full'} attention for transformers")
        max_seq_len = n_windows * self.n_tokens_per_window  # Total sequence length for inter-window transformer
        self.inter_window_transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=inter_window_depth,
            max_seq_len=max_seq_len,
            attn_dropout=transformer.get("attn_dropout", 0.2),
        ) if linear_attention else FullAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=inter_window_depth,
            max_seq_len=max_seq_len,
            attn_dropout=transformer.get("attn_dropout", 0.2),
            ff_dropout=transformer.get("ff_dropout", 0.2),
            use_flash_attn=True,
            use_rmsnorm=True
        )

        # Classification head - choose based on number of tokens
        self.n_classes = n_classes
        if self.n_tokens_per_window > 1:
            # Multi-token classification with attention aggregation
            self.classifier = AttentionClassificationHead(
                emb_size=emb_size,
                n_classes=n_classes,
                n_tokens_per_window=self.n_tokens_per_window,
                **classifier
            )
        else:
            # Single token classification with simple MLP
            from pipeline.models.commons import ClassificationHead
            self.classifier = ClassificationHead(emb_size=emb_size, n_classes=n_classes)

    def forward(self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None,
                window_mask: Optional[torch.Tensor] = None, unk_augment: float = 0.0,
                unknown_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the BIOT hierarchical encoder with detailed shape tracking.

        DETAILED DATA FLOW:
        
        Input → window Processing → Position Encoding → Inter-window Attention → Classification
        
        SHAPE TRANSFORMATIONS:
        (B, N, Nch, Ns) → (B, N, n_tokens_per_window, E) → (B, Nxn_tokens_per_window, E) → (B, N, 1)
        
        Where:
            B = batch_size, N = n_windows, E = emb_size, Ns = n_samples_per_window, Nch = n_channels
            n_tokens_per_window = configurable (CLS + mean_pool + max_pool + selected tokens)
            1 = n_classes (binary spike detection)

        Args:
            x (torch.Tensor): Input MEG data tensor.
                Shape: (batch_size, n_windows, n_channels, n_samples_per_window)
                Default shape: (B, 25, 275, 80)
                - 25 windows of 400ms duration at 200Hz each
                - 275 MEG channels
                - 80 samples per window (at 200Hz sampling rate)
            channel_mask (Optional[torch.Tensor]): Channel mask (B, C) where True=valid, False=padded.
            window_mask (Optional[torch.Tensor]): Window mask (B, N) where True=valid, False=padded.
            unk_augment (float): Probability of augmenting unknown tokens during training.
                Default: 0.0 (no augmentation)
                Only concerns BIOTEncoder stage.
            unknown_mask (Optional[torch.Tensor]): Mask indicating which channels are unknown (B, C).
                Only concerns BIOTEncoder stage for inference time.
        Returns:
            torch.Tensor: Classification logits for each window.
                Shape: (batch_size, n_windows, n_classes)
                Default shape: (B, 25, 1)
                - Binary classification logits for each of 25 windows
                - Values are raw logits (pre-softmax/sigmoid)
        """
        batch_size, n_windows, n_channels, n_samples = x.shape  # (B, N, Nch, Ns)

        # Log input statistics
        log_tensor_statistics(x, f"HBIOT input (B={batch_size}, N={n_windows}, C={n_channels}, S={n_samples})", self.logger)
        if channel_mask is not None:
            n_valid_channels = channel_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f"HBIOT channel_mask: avg valid channels={n_valid_channels:.1f}/{n_channels}")
        if window_mask is not None:
            n_valid_windows = window_mask.sum(dim=1).float().mean().item()
            self.logger.debug(f"HBIOT window_mask: avg valid windows={n_valid_windows:.1f}/{n_windows}")

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 1: Intra-window Processing (Independent window Encoding)
        # Purpose: Extract rich representations from each window independently
        # ═══════════════════════════════════════════════════════════════════════════
        # Reshape for batch processing: treat each window as independent sample
        virtual_batch_size = batch_size * n_windows
        x_reshaped = x.view(virtual_batch_size, n_channels, n_samples)
        # Shape: (B, N, Nch, Ns) → (BxN, Nch, Ns)
        log_tensor_statistics(x_reshaped, "HBIOT after reshaping to process windows independently", self.logger)

        # same for channel mask if provided
        channel_mask_reshaped = None
        if channel_mask is not None:
            channel_mask_reshaped = channel_mask.unsqueeze(1).repeat(1, n_windows, 1)
            # Shape: (B, C) → (B, 1, C) → (B, N, C)
            channel_mask_reshaped = channel_mask_reshaped.view(virtual_batch_size, n_channels)
            # Shape: (B, N, C) → (BxN, C)
        unknown_mask_reshaped = None
        if unknown_mask is not None:
            unknown_mask_reshaped = unknown_mask.unsqueeze(1).repeat(1, n_windows, 1)
            # Shape: (B, C) → (B, 1, C) → (B, N, C)
            unknown_mask_reshaped = unknown_mask_reshaped.view(virtual_batch_size, n_channels)
            # Shape: (B, N, C) → (BxN, C)

        # Compute spectral channel embeddings from signal (if spectral strategy is enabled)
        spectral_embs_expanded = None
        if self.channel_embedding_composer.use_spectral:
            # Reconstruct non-overlapping signal for PLV to avoid duplicated samples.
            # With window_overlap > 0, consecutive windows share samples. Naive concatenation
            # (B, N, C, T) → (B, C, N*T) would repeat overlapping portions, biasing PLV.
            stride = int(n_samples * (1.0 - self.window_overlap))
            if 0 < stride < n_samples:
                # Take only the stride portion from each window, full last window
                parts = [x[:, w, :, :stride] for w in range(n_windows - 1)]
                parts.append(x[:, -1, :, :])  # (B, C, n_samples)
                x_for_plv = torch.cat(parts, dim=-1)  # (B, C, stride*(N-1) + n_samples)
            else:
                # No overlap: concatenate all windows
                x_for_plv = x.transpose(1, 2).reshape(batch_size, n_channels, n_windows * n_samples)

            spectral_embs = self.channel_embedding_composer.compute_spectral_embeddings(
                x_for_plv, channel_mask
            )  # (B, C, E)
            if spectral_embs is not None:
                # Expand for virtual batch: (B, C, E) → (B, 1, C, E) → (B, N, C, E) → (B*N, C, E)
                spectral_embs_expanded = spectral_embs.unsqueeze(1).expand(-1, n_windows, -1, -1)
                spectral_embs_expanded = spectral_embs_expanded.reshape(virtual_batch_size, n_channels, -1)
                log_tensor_statistics(spectral_embs_expanded, "HBIOT spectral channel embeddings (expanded)", self.logger)

        # Process windows through BIOT encoder with optional chunking for large virtual batches
        if virtual_batch_size <= self.max_virtual_batch_size:
            # Fast path: single forward pass for all windows
            self.logger.debug(f"HBIOT processing virtual_batch_size={virtual_batch_size} in single pass")
            window_tokens = self.window_encoder(x_reshaped, channel_mask_reshaped, unk_augment=unk_augment, unknown_mask=unknown_mask_reshaped, spectral_channel_embs=spectral_embs_expanded)
            # Shape: (BxN, Nch, Ns) → (BxN, n_tokens_per_window, emb_size)
        else:
            # Slow path: chunk processing when virtual batch exceeds maximum
            self.logger.info(f"HBIOT chunking virtual_batch_size={virtual_batch_size} into chunks of {self.max_virtual_batch_size}")
            chunks = []
            for chunk_start in range(0, virtual_batch_size, self.max_virtual_batch_size):
                chunk_end = min(chunk_start + self.max_virtual_batch_size, virtual_batch_size)
                self.logger.debug(f"HBIOT processing chunk [{chunk_start}:{chunk_end}]")

                # Extract chunk
                chunk_x = x_reshaped[chunk_start:chunk_end]
                chunk_mask = channel_mask_reshaped[chunk_start:chunk_end] if channel_mask_reshaped is not None else None
                chunk_spectral = spectral_embs_expanded[chunk_start:chunk_end] if spectral_embs_expanded is not None else None

                # Process chunk through encoder
                chunk_tokens = self.window_encoder(chunk_x, chunk_mask, unk_augment=unk_augment, unknown_mask=unknown_mask, spectral_channel_embs=chunk_spectral)
                chunks.append(chunk_tokens)

            # Concatenate all chunks
            window_tokens = torch.cat(chunks, dim=0)
            # Shape: (BxN, n_tokens_per_window, emb_size)
            self.logger.debug(f"HBIOT concatenated {len(chunks)} chunks into shape {window_tokens.shape}")

        log_tensor_statistics(window_tokens, "HBIOT after window_encoder (flat)", self.logger)

        # Reshape back to batch structure with windows
        window_tokens = window_tokens.view(batch_size, n_windows, self.n_tokens_per_window, -1)
        # Shape: (BxN, n_tokens_per_window, emb_size) → (B, N, n_tokens_per_window, emb_size)
        log_tensor_statistics(window_tokens, "HBIOT after reshaping window tokens back", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 2: window Positional Encoding
        # Purpose: Add temporal order information to windows
        # ═══════════════════════════════════════════════════════════════════════════ 
        # Create positional encoding matrix for all tokens in all windows
        window_pos_matrix = self.window_pos_embedding[:n_windows].unsqueeze(1).repeat(1, self.n_tokens_per_window, 1)
        # Shape: (N, emb_size) → (N, 1, emb_size) → (N, n_tokens_per_window, emb_size)
        
        # Add positional encodings to window tokens
        window_tokens = window_tokens + window_pos_matrix.unsqueeze(0)
        # Shape: (B, N, n_tokens_per_window, emb_size) + (1, N, n_tokens_per_window, emb_size) → (B, N, n_tokens_per_window, emb_size)
        log_tensor_statistics(window_tokens, "HBIOT after adding positional encoding", self.logger)


        # Stage 2bis : Masking tokens of padded windows if mask is provided
        token_mask = None
        if window_mask is not None:
            token_mask = window_mask.view(batch_size, n_windows, 1).repeat(1, 1, self.n_tokens_per_window)
            # Shape: (B, N) → (B, N, 1) → (B, N, n_tokens_per_window)
            # Example: (32, 25) → (32, 25, 1) → (32, 25, 6)
            window_tokens = window_tokens * token_mask.unsqueeze(-1).float()
            # Shape: (B, N, n_tokens_per_window, emb_size) * (B, N, n_tokens_per_window, 1) → (B, N, n_tokens_per_window, emb_size)
            log_tensor_statistics(window_tokens, "HBIOT after applying window mask", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 3: Inter-window Attention
        # Purpose: Model long-range temporal dependencies between windows
        # ═══════════════════════════════════════════════════════════════════════════
        # Flatten windows and tokens for sequence processing
        all_embeddings = window_tokens.view(batch_size, n_windows * self.n_tokens_per_window, -1)
        # Shape: (B, N, n_tokens_per_window, emb_size) → (B, Nxn_tokens_per_window, emb_size)

        # Flatten token mask for transformer
        token_mask_flat = None
        if token_mask is not None:
            token_mask_flat = token_mask.view(batch_size, n_windows * self.n_tokens_per_window).bool()
            # Shape: (B, N, n_tokens_per_window) → (B, Nxn_tokens_per_window)
            # Example: (32, 25, 6) → (32, 150)

        # Process through inter-window transformer for temporal context
        # The transformer library expects mask where 1=valid, 0=ignore (same as our convention)
        output_embeddings = self.inter_window_transformer(all_embeddings, mask=token_mask_flat)
        # Shape: (B, Nxn_tokens_per_window, emb_size) → (B, Nxn_tokens_per_window, emb_size)
        log_tensor_statistics(output_embeddings, "HBIOT after inter_window_transformer", self.logger)

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 4: Classification with Masked Attention
        # Purpose: Generate predictions for each window using attention-based aggregation
        # ═══════════════════════════════════════════════════════════════════════════
        logits = self.classifier(output_embeddings, mask=token_mask_flat)
        # Shape: (B, Nxn_tokens_per_window, emb_size) → (B, N, n_classes)
        log_tensor_statistics(logits, "HBIOT final output logits", self.logger)
        return logits


class AttentionClassificationHead(nn.Module):
    """Attention-based classification head for hierarchical token aggregation.

    This module implements an attention mechanism to intelligently aggregate multiple
    tokens per window into a single classification decision. Instead of simple pooling,
    it learns to focus on the most informative tokens for spike detection.

    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Input: (batch_size, n_windows x n_tokens_per_window, emb_size)          │
    │        Example: (32, 150, 256) [25 windows x 6 tokens each]             │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Reshape: (batch_size x n_windows, n_tokens_per_window, emb_size)        │
    │          Example: (800, 6, 256) [process each window independently]     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Multi-Head Attention:                                                   │
    │ • Query: Learnable classification query (1, 1, emb_size)                │
    │ • Key/Value: All tokens from window (n_tokens_per_window, emb_size)     │
    │ • Output: Attended representation (1, emb_size)                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Classification MLP:                                                     │
    │ • LayerNorm + Dropout + Linear + GELU + Dropout + Linear                │
    │ • Output: (batch_size x n_windows, n_classes)                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Reshape: (batch_size, n_windows, n_classes) or (batch_size, n_windows)  │
    │          Example: (32, 25) for binary classification [n_classes=1]      │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, emb_size: int, n_classes: int, n_tokens_per_window: int, num_heads: int = 4, dropout: float = 0.1):
        """Initialize attention-based classification head.

        Args:
            emb_size (int): Embedding dimension size.
                Example: 256 (standard BIOT embedding size)
            n_classes (int): Number of output classes.
                Example: 1 (binary spike detection: spike vs. non-spike)
            n_tokens_per_window (int): Number of tokens per window to aggregate.
                Example: 6 (1 CLS + 2 pooled + 3 selected tokens from window encoder)
            dropout (float, optional): Dropout probability for regularization.
                Default: 0.1 (10% dropout)
        """
        super().__init__()
        self.n_tokens_per_window = n_tokens_per_window
        self.n_classes = n_classes

        # Learnable classification query - optimized during training to focus on
        # discriminative features for spike detection
        self.classification_query = nn.Parameter(torch.randn(1, 1, emb_size))
        # Shape: (1, 1, emb_size) - single query vector shared across all windows
        # This query learns to "ask" for the most relevant information for classification

        # Multi-head attention for intelligent token aggregation
        self.token_attention = nn.MultiheadAttention(
            embed_dim=emb_size,          # 256-dimensional embeddings
            num_heads=num_heads,         # Configurable number of attention heads for diverse attention patterns
            dropout=dropout,             # Attention dropout for regularization
            batch_first=True,            # Batch dimension first in input tensors
        )

        # Classification MLP with progressive dimension reduction
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),                      # Normalize inputs for stable training
            nn.Dropout(dropout),                                        # Input dropout for regularization
            nn.Linear(emb_size, emb_size // 2),  # 256 → 128: Feature compression
            nn.GELU(),                                                    # Smooth activation function (better than ReLU)
            nn.Dropout(dropout),                                        # Hidden dropout for additional regularization
            nn.Linear(emb_size // 2, n_classes)  # 128 → n_classes: Final classification layer
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Forward pass with attention-based token aggregation per window.

            DETAILED PROCESSING STEPS:

            1. RESHAPE: Group tokens by window for independent processing
            2. IDENTIFY: Detect fully masked windows to skip in attention (production approach)
            3. ATTENTION: Use learnable query to attend to valid windows only
            - Skips fully masked windows entirely (no wasted computation, no NaN)
            - Properly masks out padded tokens within valid windows
            4. CLASSIFICATION: Process attended representation through MLP
            5. RESHAPE: Return per-window predictions with masked windows zeroed

            SHAPE TRANSFORMATIONS:
            - Multi-class (C > 1): (B, NxT, E) → (BxN, T, E) → (BxN_valid, 1, E) → (BxN, C) → (B, N, C)
            - Binary (C = 1):      (B, NxT, E) → (BxN, T, E) → (BxN_valid, 1, E) → (BxN, 1) → (B, N)

            Where:
                B = batch_size, N = n_windows, T = n_tokens_per_window,
                E = emb_size, C = n_classes, N_valid = number of valid windows

            Args:
                x (torch.Tensor): Token embeddings from inter-window transformer.
                    Shape: (batch_size, n_windows * n_tokens_per_window, emb_size)
                    Example: (32, 150, 256) for n_tokens_per_window=6, or (32, 25, 256) for n_tokens_per_window=1

                mask (Optional[torch.Tensor]): Token validity mask (1=valid, 0=masked/padded).
                    Shape: (batch_size, n_windows * n_tokens_per_window)
                    Example: (32, 150) where masked tokens should be ignored in attention

            Returns:
                torch.Tensor: Classification logits for each window.
                    Shape: (batch_size, n_windows, n_classes) if n_classes > 1
                           (batch_size, n_windows) if n_classes == 1
                    Example: (32, 25) for binary classification
                    Raw logits for spike classification per window
            """
            batch_size = x.size(0)
            n_windows = x.size(1) // self.n_tokens_per_window
            emb_size = x.size(2)

            # Log input
            logger = logging.getLogger(__name__)
            log_tensor_statistics(x, f"AttentionClassificationHead input (B={batch_size}, N={n_windows})", logger)
            if mask is not None:
                n_valid_tokens = mask.sum(dim=1).float().mean().item()
                logger.debug(f"AttentionClassificationHead mask: avg valid tokens={n_valid_tokens:.1f}/{x.size(1)}")

            # Handle single token case - no attention needed
            if self.n_tokens_per_window == 1:
                x = x.view(batch_size * n_windows, -1)  # (B*N, E)
                logits = self.classifier(x)  # (B*N, n_classes)

                # Zero out invalid windows
                if mask is not None:
                    window_valid_mask = mask.view(batch_size * n_windows)  # (B*N,)
                    logits = logits * window_valid_mask.unsqueeze(-1).float()

                logits = logits.view(batch_size, n_windows, -1)

                # For binary classification (n_classes=1), squeeze last dimension to get (B, N)
                if self.n_classes == 1:
                    logits = logits.squeeze(-1)

                log_tensor_statistics(logits, "AttentionClassificationHead output (single token)", logger)
                return logits

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 1: Reshape to Process Each Window Independently (Multi-token case)
            # ═══════════════════════════════════════════════════════════════════════════
            x = x.view(batch_size * n_windows, self.n_tokens_per_window, emb_size)
            # Shape: (B, NxT, E) → (BxN, T, E)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 2: Identify Valid Windows (Skip Fully Masked - Production Approach)
            # ═══════════════════════════════════════════════════════════════════════════
            valid_windows_mask = None
            key_padding_mask_valid = None
            
            if mask is not None:
                # Reshape mask to per-window: (B, N*T) → (B*N, T)
                mask_reshaped = mask.view(batch_size * n_windows, self.n_tokens_per_window)
                
                # Identify windows with at least one valid token
                valid_windows_mask = mask_reshaped.any(dim=1)  # (BxN,) True where window has >=1 valid token
                n_valid = valid_windows_mask.sum().item()
                n_total = batch_size * n_windows
                
                if n_valid < n_total:
                    logger.debug(f"AttentionClassificationHead: skipping {n_total - n_valid}/{n_total} fully masked windows")
                
                # Prepare key_padding_mask only for valid windows
                # PyTorch MultiheadAttention: True = ignore, False = attend
                # Our convention: 1 = valid, 0 = masked → invert
                if n_valid > 0:
                    key_padding_mask_valid = ~mask_reshaped[valid_windows_mask].bool()  # (N_valid, T)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 3: Attention-based Token Aggregation (Valid Windows Only)
            # ═══════════════════════════════════════════════════════════════════════════
            # Initialize output for all windows (zeros for masked windows)
            attended_output = torch.zeros(
                batch_size * n_windows, 1, emb_size,
                device=x.device, dtype=x.dtype
            )  # (BxN, 1, E)
            
            # Process only valid windows through attention (skip fully masked)
            if valid_windows_mask is None or valid_windows_mask.all():
                # Fast path: all windows valid
                query = self.classification_query.expand(batch_size * n_windows, 1, emb_size).to(dtype=x.dtype)
                attended_output, _ = self.token_attention(
                    query, x, x,
                    key_padding_mask=key_padding_mask_valid
                )
            elif valid_windows_mask.any():              
                # Slow path: some windows masked, skip them entirely
                n_valid = int(valid_windows_mask.sum().item())
                query_valid = self.classification_query.expand(n_valid, 1, emb_size).to(dtype=x.dtype)
                x_valid = x[valid_windows_mask]  # (N_valid, T, E)

                attended_output_valid, _ = self.token_attention(
                    query_valid, x_valid, x_valid,
                    key_padding_mask=key_padding_mask_valid
                )  # (N_valid, 1, E)

                # Place valid outputs back into full tensor
                attended_output[valid_windows_mask] = attended_output_valid.to(dtype=attended_output.dtype)
            # else: all windows masked, keep zeros
                
            log_tensor_statistics(attended_output, "AttentionClassificationHead after attention", logger)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 4: Classification Through MLP
            # ═══════════════════════════════════════════════════════════════════════════
            logits = self.classifier(attended_output.squeeze(1))
            # Shape: (BxN, 1, E) → (BxN, E) → (BxN, n_classes)
            
            # Zero out logits for fully masked windows (redundant but explicit)
            if valid_windows_mask is not None and not valid_windows_mask.all():
                logits[~valid_windows_mask] = 0.0
                
            log_tensor_statistics(logits, "AttentionClassificationHead after classifier MLP", logger)

            # ═══════════════════════════════════════════════════════════════════════════
            # STEP 5: Reshape to Window Structure
            # ═══════════════════════════════════════════════════════════════════════════
            logits = logits.view(batch_size, n_windows, -1)
            # Shape: (BxN, n_classes) → (B, N, n_classes)

            # For binary classification (n_classes=1), squeeze last dimension to get (B, N)
            # This ensures consistency with ground truth labels shape
            if self.n_classes == 1:
                logits = logits.squeeze(-1)
                # Shape: (B, N, 1) → (B, N)

            log_tensor_statistics(logits, "AttentionClassificationHead final output", logger)
            return logits
