#!/usr/bin/env python3
"""
PyTorch Lightning module that integrates all registries for configurable training
"""

import logging
import pickle

import torch
import torch.nn as nn
import lightning.pytorch as L
from typing import Tuple, Dict, Any, Optional

from ..models.model_registry import create_model
from ..optim.loss_registry import create_loss
from ..optim.optimizer_registry import create_optimizer
from ..optim.scheduler_registry import create_scheduler

logger = logging.getLogger(__name__)


def log_tensor_statistics(tensor: torch.Tensor, name: str, logger_obj: Optional[logging.Logger] = None) -> None:
    """Log detailed statistics about a tensor for debugging NaN/inf issues.

    Args:
        tensor: Tensor to analyze
        name: Name/description of the tensor
        logger_obj: Logger to use (defaults to module logger)
    """
    if logger_obj is None:
        logger_obj = logger

    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    n_total = tensor.numel()

    if n_nan > 0 or n_inf > 0:
        logger_obj.error(f"ALERT {name}: NaN={n_nan}/{n_total} ({100*n_nan/n_total:.2f}%), Inf={n_inf}/{n_total} ({100*n_inf/n_total:.2f}%)")

    if n_nan == 0 and n_inf == 0:
        logger_obj.debug(f"OK {name}: shape={tuple(tensor.shape)}, mean={tensor.float().mean():.4f}, std={tensor.float().std():.4f}, "
                        f"min={tensor.float().min():.4f}, max={tensor.float().max():.4f}, NaN=0, Inf=0")


class MEGSpikeDetector(L.LightningModule):
    """Lightning module for spike detection in MEG data.

    This module handles training, validation, and testing of MEG spike detection models.
    All metrics computation and reporting is handled by the MetricsEvaluationCallback.

    Attributes:
        config: Configuration dictionary containing all component settings
        model: The neural network model for spike detection
        loss_fn: The loss function for training
        threshold: Classification threshold for binary predictions (updated by callback)
    """

    def __init__(self, config: Dict[str, Any], input_shape: Tuple[int, int, int], log_dir: str, **_kwargs) -> None:
        """Initialize the Lightning module with configuration.
        
        Args:
            config: Configuration dictionary containing model, loss, optimizer settings
            input_shape: Shape of the input data (channels, time_points)
            log_dir: Directory for logging
            **kwargs: Additional keyword arguments
            
        Raises:
            ValueError: If required configuration keys are missing
            TypeError: If input_shape is not a tuple
        """
        # Input validation
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got {type(config)}")
        
        required_keys = ['model', 'loss', 'optimizer', 'data', 'evaluation']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a tuple of length 3, got {input_shape}")
        super().__init__()
        logger.info("Initializing ConfigurableLightningModule")
        self.config = config
        self.log_dir = log_dir
        self.input_shape = input_shape
        config["model"][config["model"]["name"]]["input_shape"] = list(input_shape)
        config["model"][config["model"]["name"]]["log_dir"] = log_dir
        self.save_hyperparameters(config)
        
        # Create model and processing flags
        self.contextual = config["model"][config["model"]["name"]].get("contextual", False)
        self.sequential_processing = config["model"][config["model"]["name"]].get("sequential_processing", False)
        self.model = create_model(config["model"])

        # Create loss function
        loss_config = config["loss"]
        logger.info("Creating loss function")
        self.loss_fn = create_loss(loss_config)
        
        # Store optimizer and scheduler configs for later use
        self.optimizer_config = config["optimizer"]
        self.scheduler_config = config.get("scheduler", None)
        logger.info(f"Optimizer config: {self.optimizer_config}")
        if self.scheduler_config:
            logger.info(f"Scheduler config: {self.scheduler_config}")

        # Log the graph to TensorBoard
        logger.info("ConfigurableLightningModule initialized successfully")
              
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore threshold and temperature from checkpoint if available."""
        super().on_load_checkpoint(checkpoint)
        if "hyper_parameters" in checkpoint:
            if "threshold" in checkpoint["hyper_parameters"]:
                self.threshold = checkpoint["hyper_parameters"]["threshold"]
                logger.info(f"Restored threshold from checkpoint: {self.threshold:.4f}")
    
    def dummy_forward_init(self) -> None:
        """Runs a dummy forward pass to initialize the model.

        This is required for some models to properly initialize internal parameters.
        Only runs for non-contextual models that process individual windows.

        Note:
            This initialization uses the actual input shape from configuration
            rather than hardcoded values, ensuring compatibility across different
            dataset configurations.
        """
        if not self.contextual:
            with torch.no_grad():
                # Use actual input shape for proper initialization
                dummy_input = torch.randn(1, self.input_shape[1], self.input_shape[2])
                self.model(dummy_input)

    def forward(self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None, window_mask: Optional[torch.Tensor] = None, force_sequential: bool = False, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the model with contextual and sequential processing support.

        Handles different processing modes:
        - Contextual: Pass full sequence [batch_size, n_windows, n_channels, n_timepoints] to model
        - Non-contextual + batch mode: Reshape to [BxN_window, n_channels, n_timepoints]
        - Non-contextual + sequential: Loop through windows individually

        Args:
            x: Input tensor of shape [batch_size, n_windows, n_channels, n_timepoints]
            channel_mask: Optional channel mask tensor (B, C) where True=valid, False=masked.
            window_mask: Optional window mask tensor (B, N) where True=valid, False=masked.
            force_sequential: Whether to force sequential processing mode.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, n_windows, n_classes]
        """
        if self.contextual:
            # Contextual models process the full sequence with temporal context
            return self.model(x, channel_mask, window_mask, *args, **kwargs)
        
        # Non-contextual processing for window-level models
        batch_size, n_windows, n_channels, n_timepoints = x.shape
        if self.sequential_processing or force_sequential:
            # Sequential mode: Process each window individually in a loop
            window_outputs = []
            for seg_idx in range(n_windows):
                window = x[:, seg_idx, :, :]  # [batch_size, n_channels, n_timepoints]
                window_output = self.model(window, channel_mask, *args, **kwargs)  # [batch_size, n_classes]
                window_outputs.append(window_output.unsqueeze(1))  # [batch_size, 1, n_classes]

            return torch.cat(window_outputs, dim=1)  # [batch_size, n_windows, n_classes]
        else:
            # Batch mode: Reshape to process all windows simultaneously
            x = x.view(batch_size * n_windows, n_channels, n_timepoints)  # [B×N_window, n_channels, n_timepoints]
            channel_mask = channel_mask.unsqueeze(1).repeat(1, n_windows, 1).view(batch_size * n_windows, n_channels) if channel_mask is not None else None  # [B×N_window, n_channels]
            if 'unknown_mask' in kwargs and kwargs['unknown_mask'] is not None:
                unknown_mask = kwargs['unknown_mask'].unsqueeze(1).repeat(1, n_windows, 1).view(batch_size * n_windows, n_channels)
                kwargs['unknown_mask'] = unknown_mask  # [B×N_window, n_channels]
            kwargs['channel_mask'] = channel_mask
            result = self.model(x, *args, **kwargs)  # [B×N_window, n_classes]
            return result.view(batch_size, n_windows, -1)  # [batch_size, n_windows, n_classes]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]) -> torch.Tensor:
        """Perform a single training step.

        Processes one batch of training data through the model, computes loss,
        and logs metrics.

        Args:
            batch: Training batch containing:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - y: Target labels [batch_size, n_windows]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata, used for channel coordinates (loc should be consistent across batch)

        Returns:
            torch.Tensor: Computed loss value for backpropagation
        """
        X, y, window_mask, channel_mask, metadata = batch

        # Log inputs
        log_tensor_statistics(X, f"Training input X (B={X.shape[0]}, N={X.shape[1]}, C={X.shape[2]}, S={X.shape[3]})", logger)
        log_tensor_statistics(y, "Training labels y", logger)
        if channel_mask is not None:
            n_valid = channel_mask.sum(dim=1).float().mean().item()
            logger.debug(f"Training channel_mask: avg valid={n_valid:.1f}/{X.shape[2]}")
        if window_mask is not None:
            n_valid = window_mask.sum(dim=1).float().mean().item()
            logger.debug(f"Training window_mask: avg valid={n_valid:.1f}/{X.shape[1]}")

        # Forward pass with batch-aware channel mask
        logits = self.forward(X, channel_mask=channel_mask, window_mask=window_mask, unk_augment=torch.rand(1).item()) # generate a random threshold between 0 and 1 for unk augmentation (-> at each batch, different number and density of channels will be masked)
        log_tensor_statistics(logits, "Training logits after forward", logger)

        # Calculate loss using window mask only (channel masking is handled in model)
        loss = self.loss_fn(logits, y, mask=window_mask)

        # Log training metrics with distributed training support
        batch_size = X.shape[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], prog_bar=True, sync_dist=True, batch_size=batch_size)  # type: ignore
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]) -> Dict[str, Any]:
        """Perform a single validation step.

        Processes one batch of validation data through the model, computes loss,
        and returns outputs for the MetricsEvaluationCallback to collect.

        Args:
            batch: Validation batch containing:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - y: Target labels [batch_size, n_windows]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata

        Returns:
            Dictionary of outputs for callback collection
        """
        X, y, window_mask, channel_mask, metadata = batch

        # Log inputs
        log_tensor_statistics(X, f"Validation input X (B={X.shape[0]}, N={X.shape[1]}, C={X.shape[2]}, S={X.shape[3]})", logger)
        log_tensor_statistics(y, "Validation labels y", logger)
        if channel_mask is not None:
            n_valid = channel_mask.sum(dim=1).float().mean().item()
            logger.debug(f"Validation channel_mask: avg valid={n_valid:.1f}/{X.shape[2]}")
        if window_mask is not None:
            n_valid = window_mask.sum(dim=1).float().mean().item()
            logger.debug(f"Validation window_mask: avg valid={n_valid:.1f}/{X.shape[1]}")

        # Forward pass with batch-aware channel mask
        logits = self.forward(X, channel_mask=channel_mask, window_mask=window_mask)
        log_tensor_statistics(logits, "Validation logits after forward", logger)

        val_loss = self.loss_fn(logits, y, mask=window_mask)
        batch_size = X.shape[0]
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        # Return outputs for callback collection
        return self._collect_batch_outputs(batch, logits=logits)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict],
                  _batch_idx: int) -> Dict[str, Any]:
        """Perform a single test step.

        Processes one batch of test data through the model and returns outputs
        for the MetricsEvaluationCallback to collect.

        Args:
            batch: Test batch containing:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - y: Target labels [batch_size, n_windows]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata for result export

        Returns:
            Dictionary of outputs for callback collection
        """
        X, y, window_mask, channel_mask, metadata = batch

        # Forward pass with batch-aware channel mask
        logits = self.forward(X, channel_mask=channel_mask, window_mask=window_mask)

        # Return outputs for callback collection
        return self._collect_batch_outputs(batch, logits)

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step.

        Args:
            batch: Batch data (X, window_mask, channel_mask, metadata) where:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata for result export
            batch_idx: Index of the batch

        Returns:
            Dictionary containing predictions, probabilities, and metadata
        """
        X, window_mask, channel_mask, metadata = batch
        
        # Channel mask is actually true everywhere but for padded channels
        # We actually don't know if good channels are really good at inference time, we just known that this is real data
        # So we use an unknown mask that is all True where channel_mask is given
        unknown_mask = torch.ones_like(channel_mask, dtype=torch.bool) if channel_mask is not None else None

        # Forward pass with batch-aware channel mask
        logits = self.forward(X, channel_mask=channel_mask, window_mask=window_mask, force_sequential=True, unknown_mask=unknown_mask)
        probs = torch.sigmoid(logits).cpu().detach()

        # Apply threshold for binary predictions
        predictions = (probs >= self.threshold).float()

        # Prepare outputs
        outputs = {
            'logits': logits.cpu().detach(),
            'probs': probs,
            'predictions': predictions,
            'batch_size': X.shape[0],
            'n_windows': X.shape[1] if len(X.shape) > 2 else 1,
            'batch_idx': batch_idx,
            'metadata': metadata if metadata else {},
            'channel_mask': channel_mask.cpu().detach().float().numpy() if channel_mask is not None else None,
            'window_mask': window_mask.cpu().detach().float().numpy() if window_mask is not None else None
        }
        return outputs


    def _collect_batch_outputs(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict], logits: torch.Tensor) -> Dict[str, Any]:
        """Collect outputs for a single batch.

        Args:
            batch: Input batch (X, y, window_mask, channel_mask, metadata)
            logits: Model output logits for the batch

        Returns:
            Dictionary with batch outputs including per-window losses
        """
        X, y, window_mask, _channel_mask, metadata = batch

        if hasattr(self.loss_fn, 'history_presence_loss') and hasattr(self.loss_fn, 'history_onset_loss'):
            per_window_loss = self.loss_fn.history_presence_loss.pop().cpu().detach().float().numpy() if self.loss_fn.history_presence_loss else None    # type: ignore
            onset_loss = self.loss_fn.history_onset_loss.pop().cpu().detach().float().numpy() if self.loss_fn.history_onset_loss else None        # type: ignore
        else:
            per_window_loss = None
            onset_loss = None

        has_onset = logits.size(-1) == 2

        if has_onset:
            presence_logits = logits[..., 0]
            onset_logits = logits[..., 1]
            presence_probs = torch.sigmoid(presence_logits)
            onset_probs = torch.sigmoid(onset_logits)
        else:
            presence_logits = logits
            presence_probs = torch.sigmoid(logits)
            onset_probs = None

        gt_presence = y[..., 0] if y.dim() > 2 and y.size(-1) >= 2 else y
        gt_onsets = y[..., 1] if y.dim() > 2 and y.size(-1) >= 2 else None

        return {
            "logits": logits.cpu().detach().float().numpy(),
            "probs": presence_probs.cpu().detach().float().numpy(),
            "onset_probs": onset_probs.cpu().detach().float().numpy() if onset_probs is not None else None,
            "predictions": (presence_probs >= self.threshold).float().cpu().detach().numpy(),
            "gt": gt_presence.cpu().detach().float().numpy(),
            "gt_onsets": gt_onsets.cpu().detach().float().numpy() if gt_onsets is not None else None,
            "mask": window_mask.cpu().detach().float().numpy(),
            "losses": per_window_loss,
            "onset_losses": onset_loss,
            "metadata": metadata if metadata else {},
            "batch_size": X.shape[0],
            "n_windows": X.shape[1]
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        logger.info("Configuring optimizers and schedulers")
        
        # Create optimizer
        optimizer = create_optimizer(self.optimizer_config, self.model.parameters())
        
        if self.scheduler_config is None:
            logger.info("No scheduler configured, using optimizer only")
            return optimizer
        
        scheduler_name = self.scheduler_config.get("name")
        assert self.trainer.max_epochs is not None, "Trainer max_epochs must be set to configure schedulers"
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        if scheduler_name == "OneCycleLR":
            # OneCycleLR requires additional parameters
            self.scheduler_config[scheduler_name]["steps_per_epoch"] = steps_per_epoch
            self.scheduler_config[scheduler_name]["epochs"] = self.trainer.max_epochs
        elif scheduler_name == "CosineAnnealingWarmRestarts":
            # CosineAnnealingWarmRestarts requires T_0 and T_mult
            assert "T_0" in self.scheduler_config[scheduler_name], "CosineAnnealingWarmRestarts requires 'T_0' parameter"
            self.scheduler_config[scheduler_name]["T_0"] *= steps_per_epoch
      
        scheduler = create_scheduler(self.scheduler_config, optimizer, steps_per_epoch=int(steps_per_epoch))
        
        # Handle ReduceLROnPlateau special case
        if scheduler_name == "ReduceLROnPlateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_config.get("monitor", "val_loss"),
                    "interval": "epoch"   # ReduceLROnPlateau works per epoch
                }
            }
        
        # For all other schedulers, use step-based scheduling for fine-grained control
        logger.info("Optimizer and scheduler configured successfully with step-based updates")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "lr_scheduler"
            }
        }
