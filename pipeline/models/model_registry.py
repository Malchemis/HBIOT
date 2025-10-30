"""Model registry for MEG spike detection models.

This module provides a centralized registry for all available models and utilities
for creating model instances from configuration dictionaries. Supports dynamic
model registration and validation of model parameters.
"""

import logging
import traceback
from typing import Dict, Type, Any
import torch.nn as nn
from pipeline.models.biot import BIOTClassifier
from pipeline.models.hbiot import BIOTHierarchicalClassifier
from pipeline.models.sfcn import SFCN
from pipeline.models.famed import FAMEDWrapper


logger = logging.getLogger(__name__)

# Dictionary mapping model names to their classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "BIOT": BIOTClassifier,
    "BIOTHierarchical": BIOTHierarchicalClassifier,
    "SFCN": SFCN,
    "FAMED": FAMEDWrapper
}


def get_model_class(model_name: str) -> Type[nn.Module]:
    """Get model class from registry by name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class
        
    Raises:
        ValueError: If the model name is not found in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name]


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create a model instance based on configuration.
    
    Args:
        model_config: Model configuration dictionary containing 'name' field
            and model-specific parameters under a key matching the model name

    Returns:
        Instantiated model
        
    Raises:
        TypeError: If model_config is not a dictionary
        ValueError: If required configuration fields are missing
        RuntimeError: If model instantiation fails
    """
    # Input validation
    if not isinstance(model_config, dict):
        raise TypeError(f"model_config must be a dictionary, got {type(model_config)}")
    
    if "name" not in model_config:
        raise ValueError("model_config must contain 'name' field")
    
    logger.info(f"Creating model from config: {model_config}")
    
    # Copy for safety
    config = model_config.copy()
    
    # Extract model-specific parameters from config
    model_name = config.pop("name")
    
    # Remove global flags that don't belong to the specific model
    config.pop("contextual", None)
    config.pop("sequential_processing", None)
    
    logger.info(f"Model name: {model_name}")
    
    model_class = get_model_class(model_name)
    model_params = config.get(model_name, {})
    
    # Validate required parameters for specific models
    if model_name == "BIOTHierarchical" and "input_shape" not in model_params:
        raise ValueError("BIOTHierarchical requires 'input_shape' parameter")
    if model_name == "BIOT" and "input_shape" not in model_params:
        raise ValueError("BIOT requires 'input_shape' parameter")
    if model_name == "SFCN" and "input_shape" not in model_params:
        raise ValueError("SFCN requires 'input_shape' parameter")
    
    try:
        model = model_class(**model_params)
        
        # Set contextual flag based on model type
        # BIOTHierarchical processes full sequences with temporal context
        # Other models process individual windows
        setattr(model, 'contextual', model_name == "BIOTHierarchical")
        
        logger.info(f"Model created successfully: {model.__class__.__name__} (contextual={model.contextual})")
        return model
    except Exception as e:
        logger.error(f"Failed to instantiate model {model_name}")
        logger.error(f"Model parameters: {model_params}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to instantiate model {model_name}: {e}")
        


def register_model(name: str, model_class: Type[nn.Module]) -> None:
    """Register a new model in the registry.
    
    Args:
        name: Name for the model
        model_class: Model class to register
        
    Raises:
        TypeError: If model_class is not a subclass of nn.Module
        ValueError: If name is empty or already exists
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string")
    
    if not issubclass(model_class, nn.Module):
        raise TypeError(f"Model class must be a subclass of nn.Module, got {type(model_class)}")
    
    if name in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model registration: {name}")
    
    logger.info(f"Registering model: {name} -> {model_class.__name__}")
    MODEL_REGISTRY[name] = model_class
