"""Optimizer registry for configurable optimizer selection and instantiation.

This module provides a centralized registry for PyTorch optimizers with utilities
for creating optimizer instances from configuration dictionaries.
"""

import logging
import traceback
from typing import Dict, Type, Any
import torch.optim as optim

logger = logging.getLogger(__name__)


# Dictionary mapping optimizer names to their classes
OPTIMIZER_REGISTRY: Dict[str, Type[optim.Optimizer]] = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad,
}


def get_optimizer_class(optimizer_name: str) -> Type[optim.Optimizer]:
    """Get optimizer class from registry by name.
    
    Args:
        optimizer_name: Name of the optimizer
        
    Returns:
        Optimizer class
        
    Raises:
        ValueError: If the optimizer name is not found in the registry
    """
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{optimizer_name}' not found in registry. Available optimizers: {list(OPTIMIZER_REGISTRY.keys())}")
    
    return OPTIMIZER_REGISTRY[optimizer_name]


def create_optimizer(optimizer_config: Dict[str, Any], model_parameters) -> optim.Optimizer:
    """Create an optimizer instance based on configuration.
    
    Args:
        optimizer_config: Optimizer configuration dictionary containing 'name' field
            and optimizer-specific parameters under a key matching the optimizer name
        model_parameters: Model parameters to optimize (must be iterable)
        
    Returns:
        Instantiated optimizer
        
    Raises:
        TypeError: If optimizer_config is not a dictionary
        ValueError: If required configuration fields are missing or parameters are invalid
        RuntimeError: If optimizer instantiation fails
    """
    # Input validation
    if not isinstance(optimizer_config, dict):
        raise TypeError(f"optimizer_config must be a dictionary, got {type(optimizer_config)}")
    
    if "name" not in optimizer_config:
        raise ValueError("optimizer_config must contain 'name' field")
    
    # Validate model parameters
    try:
        # Check if model_parameters is iterable
        iter(model_parameters)
    except TypeError:
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise TypeError("model_parameters must be iterable")
        
    logger.info(f"Creating optimizer from config: {optimizer_config}")
    
    config = optimizer_config.copy()
    
    optimizer_name = config.pop("name")
    logger.info(f"Optimizer name: {optimizer_name}")
    
    optimizer_class = get_optimizer_class(optimizer_name)
    optimizer_params = config.get(optimizer_name, {})
    
    # Validate optimizer-specific parameters
    if optimizer_name in ["Adam", "AdamW"] and "lr" in optimizer_params:
        lr = optimizer_params["lr"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
    
    if optimizer_name == "SGD" and "lr" in optimizer_params:
        lr = optimizer_params["lr"]
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
    
    logger.info(f"Optimizer parameters: {optimizer_params}")
    
    try:
        optimizer = optimizer_class(model_parameters, **optimizer_params)
        logger.info(f"Optimizer created successfully: {optimizer.__class__.__name__}")
        return optimizer
    except Exception as e:
        logger.error(f"Failed to instantiate optimizer {optimizer_name}")
        logger.error(f"Optimizer parameters: {optimizer_params}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to instantiate optimizer {optimizer_name}: {e}")


def register_optimizer(name: str, optimizer_class: Type[optim.Optimizer]) -> None:
    """Register a new optimizer in the registry.
    
    Args:
        name: Name for the optimizer
        optimizer_class: Optimizer class to register
        
    Raises:
        TypeError: If optimizer_class is not a subclass of optim.Optimizer
        ValueError: If name is empty
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Optimizer name must be a non-empty string")
    
    if not issubclass(optimizer_class, optim.Optimizer):
        raise TypeError(f"Optimizer class must be a subclass of optim.Optimizer, got {type(optimizer_class)}")
    
    if name in OPTIMIZER_REGISTRY:
        logger.warning(f"Overwriting existing optimizer registration: {name}")
    
    logger.info(f"Registering optimizer: {name} -> {optimizer_class.__name__}")
    OPTIMIZER_REGISTRY[name] = optimizer_class