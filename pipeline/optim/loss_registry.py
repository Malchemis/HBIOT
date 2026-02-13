"""Loss function registry for configurable loss selection.

This module provides custom loss functions (e.g., Focal Loss, Class-Balanced Loss)
and utilities for creating loss instances from configuration dictionaries.
"""

import logging
import traceback
from typing import Dict, Type, Any, Optional
import torch
import torch.nn as nn

from collections import deque

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


#-- Custom Loss Classes --#
class FocalLoss(nn.Module):
    """Focal loss implementation for imbalanced classification.
    
    Attributes:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, **kwargs):
        super().__init__()
        
        # Input validation
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
            
        self.alpha = alpha
        self.gamma = gamma
        
        # Use BCE with reduction='none' for proper focal loss calculation
        self.bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        # Use an Onset loss for onset prediction if needed
        self.onset_loss_fn = nn.SmoothL1Loss(reduction='none')  # might be better than MSE regarding outliers
        
        # Per Batch and Per Window loss history for monitoring
        self.history_presence_loss = deque([], maxlen=5)
        self.history_onset_loss = deque([], maxlen=5)
    
    def forward(self, predictions, targets, mask=None, **_kwargs):
        """
        Focal loss implementation. You can pass a mask to ignore certain elements like padding.
        Expects predictions: (B, N_s) for binary classification, targets: (B, N_s)

        Returns:
            Scalar focal loss value
        """
        # Log inputs
        # log_tensor_statistics(predictions, "FocalLoss predictions input", logger)
        # log_tensor_statistics(targets, "FocalLoss targets input", logger)
        if predictions.dim() != targets.dim():
            raise ValueError(f"Predictions and targets must have the same number of dimensions, got {predictions.dim()} and {targets.dim()}")
        
        B, N_s, *_ = predictions.shape

        # Infer if onset prediction is included based on target shape
        onset_and_prediction = targets.size(-1) == 2
        if onset_and_prediction:
            # Split presence and onset targets
            presence_predictions = predictions[..., 0]
            presence_targets = targets[..., 0]
            onset_predictions = torch.sigmoid(predictions[..., 1])  # Onset predictions should be in [0, 1], as targets are
            onset_targets = targets[..., 1]
        else:
            presence_predictions = predictions
            presence_targets = targets
        
        # Flatten to (B*N_s) for binary classification
        predictions_flat = presence_predictions.view(-1)
        targets_flat = presence_targets.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1)
            n_valid = mask_flat.sum().item()
            logger.debug(f"FocalLoss mask: {n_valid}/{mask_flat.numel()} valid elements")

        # log_tensor_statistics(predictions_flat, "FocalLoss predictions_flat", logger)
        # log_tensor_statistics(targets_flat, "FocalLoss targets_flat", logger)

        # Calculate BCE loss (element-wise)
        bce_loss = self.bce_fn(predictions_flat, targets_flat.float())
        log_tensor_statistics(bce_loss, "FocalLoss bce_loss", logger)

        # Calculate focal loss
        pt = torch.exp(-bce_loss)
        # log_tensor_statistics(pt, "FocalLoss pt", logger)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        # log_tensor_statistics(focal_loss, "FocalLoss focal_loss (before masking)", logger)            

        if mask is not None:
            focal_loss = focal_loss * mask_flat.float()
            self.history_presence_loss.append(focal_loss.reshape(B, N_s))
            # log_tensor_statistics(focal_loss, "FocalLoss focal_loss (after masking)", logger)
            
            if onset_and_prediction:
                # Only penalize onset loss when presence should have been predicted
                onset_loss = self.onset_loss_fn(onset_predictions.view(-1), onset_targets.view(-1)) * presence_targets.view(-1) * mask_flat.float()
                self.history_onset_loss.append(onset_loss.reshape(B, N_s))
            
            mask_sum = mask_flat.sum()
            if mask_sum > 0:
                final_loss = focal_loss.sum() / mask_sum
                
                if onset_and_prediction:
                    onset_loss = onset_loss.sum() / mask_sum
                    final_loss += onset_loss
                
                return final_loss
            else:
                logger.warning("FocalLoss: All elements masked out, returning 0.0")
                return torch.tensor(0.0, device=focal_loss.device, dtype=focal_loss.dtype)
        else:
            self.history_presence_loss.append(focal_loss.reshape(B, N_s))
            final_loss = focal_loss.mean()
            
            if onset_and_prediction:
                onset_loss = self.onset_loss_fn(onset_predictions.view(-1), onset_targets.view(-1)) * presence_targets.view(-1)
                self.history_onset_loss.append(onset_loss.reshape(B, N_s))
                onset_loss = onset_loss.mean()
                final_loss += onset_loss
            
            return final_loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss wrapper for registry compatibility."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, predictions, targets, mask=None, **_kwargs):
        if mask is not None:
            # Apply mask to cross-entropy loss
            return (self.loss_fn(predictions, targets) * mask.float()).sum() / mask.sum() if mask.sum() > 0 else self.loss_fn(predictions, targets)
        return self.loss_fn(predictions, targets)


class MSELoss(nn.Module):
    """MSE loss wrapper for registry compatibility."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.MSELoss(**kwargs)

    def forward(self, predictions, targets, mask=None, **_kwargs):
        if mask is not None:
            # Apply mask to MSE loss
            return (self.loss_fn(predictions, targets) * mask.float()).sum() / mask.sum() if mask.sum() > 0 else self.loss_fn(predictions, targets)
        return self.loss_fn(predictions, targets)
    

class BCEWithLogitsLoss(nn.Module):
    """BCE with logits loss wrapper for registry compatibility."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Default to reduction='none' for proper mask handling
        kwargs.setdefault('reduction', 'none')
        self.loss_fn = nn.BCEWithLogitsLoss(**kwargs)
        self.reduction = kwargs.get('reduction', 'none')

    def forward(self, predictions, targets, mask=None, **_kwargs):
        """
        Expects predictions: (B, N_s) for binary classification, targets: (B, N_s)
        """
        # Flatten to (B*N_s) for binary classification
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1)
        
        loss = self.loss_fn(predictions_flat, targets_flat.float())
        
        if mask is not None:
            # Apply mask
            loss = loss * mask_flat.float()
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.sum() / mask_flat.sum() if mask_flat.sum() > 0 else torch.tensor(0.0, device=loss.device)
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss


# Dictionary mapping loss names to their classes
LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MSELoss": MSELoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "FocalLoss": FocalLoss,
}


def get_loss_class(loss_name: str) -> Type[nn.Module]:
    """Get loss class from registry by name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Loss class
        
    Raises:
        ValueError: If the loss name is not found in the registry
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{loss_name}' not found in registry. Available losses: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_name]


def create_loss(loss_config: Dict[str, Any]) -> nn.Module:
    """Create a loss function instance based on configuration.
    
    Args:
        loss_config: Loss configuration dictionary containing 'name' field
            and loss-specific parameters under a key matching the loss name
        
    Returns:
        Instantiated loss function
        
    Raises:
        TypeError: If loss_config is not a dictionary
        ValueError: If required configuration fields are missing
        RuntimeError: If loss instantiation fails
    """
    # Input validation
    if not isinstance(loss_config, dict):
        raise TypeError(f"loss_config must be a dictionary, got {type(loss_config)}")
    
    if "name" not in loss_config:
        raise ValueError("loss_config must contain 'name' field")
    
    logger.info(f"Creating loss from config: {loss_config}")
    
    config = loss_config.copy()
    
    loss_name = config.pop("name")
    logger.info(f"Loss name: {loss_name}")
    
    loss_class = get_loss_class(loss_name)
    loss_params = config.get(loss_name, {})
    
    # Validate loss-specific parameters
    if loss_name == "FocalLoss":
        alpha = loss_params.get("alpha", 1.0)
        gamma = loss_params.get("gamma", 2.0)
        if alpha <= 0:
            raise ValueError(f"FocalLoss alpha must be positive, got {alpha}")
        if gamma < 0:
            raise ValueError(f"FocalLoss gamma must be non-negative, got {gamma}")
    
    logger.info(f"Loss parameters: {loss_params}")
    
    try:
        loss_fn = loss_class(**loss_params)
        logger.info(f"Loss created successfully: {loss_fn.__class__.__name__}")
        return loss_fn
    except Exception as e:
        logger.error(f"Failed to instantiate loss {loss_name}")
        logger.error(f"Loss parameters: {loss_params}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to instantiate loss {loss_name}: {e}")


def register_loss(name: str, loss_class: Type[nn.Module]) -> None:
    """Register a new loss function in the registry.
    
    Args:
        name: Name for the loss function
        loss_class: Loss class to register
        
    Raises:
        TypeError: If loss_class is not a subclass of nn.Module
        ValueError: If name is empty
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Loss name must be a non-empty string")
    
    if not issubclass(loss_class, nn.Module):
        raise TypeError(f"Loss class must be a subclass of nn.Module, got {type(loss_class)}")
    
    if name in LOSS_REGISTRY:
        logger.warning(f"Overwriting existing loss registration: {name}")
    
    logger.info(f"Registering loss: {name} -> {loss_class.__name__}")
    LOSS_REGISTRY[name] = loss_class