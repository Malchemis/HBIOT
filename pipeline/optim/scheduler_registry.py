"""Learning rate scheduler registry for configurable scheduler selection.

This module provides a centralized registry for PyTorch learning rate schedulers
with utilities for creating scheduler instances from configuration dictionaries.
"""

import logging
import traceback
from typing import Dict, Type, Any, Optional
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    PolynomialLR,
    SequentialLR
)

logger = logging.getLogger(__name__)


# Dictionary mapping scheduler names to their classes
SCHEDULER_REGISTRY: Dict[str, Type[LRScheduler]] = {
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "OneCycleLR": OneCycleLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "LinearLR": LinearLR,
    "PolynomialLR": PolynomialLR,
    "SequentialLR": SequentialLR
}


def get_scheduler_class(scheduler_name: str) -> Type:
    """Get scheduler class from registry by name.
    
    Args:
        scheduler_name: Name of the scheduler
        
    Returns:
        Scheduler class
        
    Raises:
        ValueError: If the scheduler name is not found in the registry
    """
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{scheduler_name}' not found in registry. Available schedulers: {list(SCHEDULER_REGISTRY.keys())}")
    
    return SCHEDULER_REGISTRY[scheduler_name]


def create_scheduler(scheduler_config: Dict[str, Any], optimizer, steps_per_epoch: Optional[int] = None) -> LRScheduler:
    """Create a scheduler instance based on configuration.
    
    Args:
        scheduler_config: Scheduler configuration dictionary containing 'name' field
            and scheduler-specific parameters under a key matching the scheduler name
        optimizer: Optimizer instance to schedule
        steps_per_epoch: Number of steps per epoch (optional, used for some schedulers)
        
    Returns:
        Instantiated scheduler
        
    Raises:
        TypeError: If scheduler_config is not a dictionary
        ValueError: If required configuration fields are missing or parameters are invalid
        RuntimeError: If scheduler instantiation fails
    """
    # Input validation
    if not isinstance(scheduler_config, dict):
        raise TypeError(f"scheduler_config must be a dictionary, got {type(scheduler_config)}")
    
    if "name" not in scheduler_config:
        raise ValueError("scheduler_config must contain 'name' field")
    
    if optimizer is None:
        raise ValueError("optimizer cannot be None")
    
    if steps_per_epoch is not None and (not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0):
        raise ValueError(f"steps_per_epoch must be a positive integer, got {steps_per_epoch}")
    
    logger.info(f"Creating scheduler from config: {scheduler_config}")
    
    config = scheduler_config.copy()
    
    scheduler_name = config.pop("name")
    
    logger.info(f"Scheduler name: {scheduler_name}")
    
    scheduler_class = get_scheduler_class(scheduler_name)
    scheduler_params = config.get(scheduler_name, {})
    
    logger.info(f"Scheduler parameters: {scheduler_params}")
    
    # Validate scheduler-specific parameters
    if scheduler_name == "OneCycleLR" and "max_lr" not in scheduler_params:
        raise ValueError("OneCycleLR requires 'max_lr' parameter")
    
    if scheduler_name == "CosineAnnealingWarmRestarts" and "T_0" not in scheduler_params:
        raise ValueError("CosineAnnealingWarmRestarts requires 'T_0' parameter")
    
    try:
        scheduler = scheduler_class(optimizer, **scheduler_params)
        logger.info(f"Scheduler created successfully: {scheduler.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to instantiate scheduler {scheduler_name}")
        logger.error(f"Scheduler parameters: {scheduler_params}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to instantiate scheduler {scheduler_name}: {e}")
    
    if scheduler_config.get("warmup", {}).get("enabled", False):
        warmup_config = scheduler_config.get("warmup", {})
        logger.info(f"Warmup configuration found: {warmup_config}")
        assert scheduler_name != "ReduceLROnPlateau", "Warmup (due to SequentialLR) cannot be used with ReduceLROnPlateau scheduler"
        # Create a SequentialLR with warmup
        # Create a LinearLR for warmup
        warmup_start_lr = warmup_config.get("warmup_start_lr", 1.0e-6)
        warmup_end_lr = warmup_config.get("warmup_end_lr", 1.0e-3)
        warmup_epochs = warmup_config.get("warmup_epochs", 5)
        
        # Validate warmup parameters
        if warmup_start_lr <= 0 or warmup_end_lr <= 0:
            raise ValueError(f"Warmup learning rates must be positive: start_lr={warmup_start_lr}, end_lr={warmup_end_lr}")
        if warmup_epochs <= 0:
            raise ValueError(f"Warmup epochs must be positive: {warmup_epochs}")
        
        # Calculate warmup factors relative to optimizer's initial LR
        initial_lr = optimizer.param_groups[0]['lr']
        start_factor = warmup_start_lr / initial_lr
        end_factor = warmup_end_lr / initial_lr
        
        if steps_per_epoch is not None:
            warmup_epochs *= steps_per_epoch
        
        logger.info(f"Warmup factors: start_factor={start_factor:.6f}, end_factor={end_factor:.6f}")
        
        # Update optimizer's initial LR to warmup_end_lr for the main scheduler
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = warmup_end_lr
            param_group['lr'] = warmup_end_lr
        
        # Recreate main scheduler with the updated optimizer state
        main_scheduler = scheduler_class(optimizer, **scheduler_params)
        
        # Reset optimizer LR back to original for warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        
        warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=warmup_epochs)
        # Create a SequentialLR to combine warmup and the main scheduler
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    return scheduler


def register_scheduler(name: str, scheduler_class: Type[LRScheduler]) -> None:
    """Register a new scheduler in the registry.
    
    Args:
        name: Name for the scheduler
        scheduler_class: Scheduler class to register
        
    Raises:
        TypeError: If scheduler_class is not a subclass of LRScheduler
        ValueError: If name is empty
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Scheduler name must be a non-empty string")
    
    # Note: ReduceLROnPlateau doesn't inherit from LRScheduler, so we check for the base class
    if not (issubclass(scheduler_class, LRScheduler) or scheduler_class == ReduceLROnPlateau):
        raise TypeError(f"Scheduler class must be a subclass of LRScheduler or ReduceLROnPlateau, got {type(scheduler_class)}")
    
    if name in SCHEDULER_REGISTRY:
        logger.warning(f"Overwriting existing scheduler registration: {name}")
    
    logger.info(f"Registering scheduler: {name} -> {scheduler_class.__name__}")
    SCHEDULER_REGISTRY[name] = scheduler_class