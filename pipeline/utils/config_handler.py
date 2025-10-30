#!/usr/bin/env python3
"""
Configuration handling utilities with comprehensive validation.
"""

import copy
import os
import traceback
from typing import Dict, Any, Union
from pathlib import Path
import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path], validate: bool = True) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional validation.
    
    Args:
        config_path: Path to the configuration file
        validate: Whether to validate the loaded configuration
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing failed for {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to load configuration: {e}")
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    if validate:
        validate_config(config)

    def expand_env_vars(obj: Any, max_recursion_depth: int = 20) -> Any:
        """Recursively expand environment variables in strings within the config."""
        if max_recursion_depth <= 0:
            raise ValueError("Maximum recursion depth reached while expanding environment variables. Default is 20.")
        
        # if this is a dict, recurse into values
        if isinstance(obj, dict):
            return {k: expand_env_vars(v, max_recursion_depth - 1) for k, v in obj.items()}
        # if this is a list, recurse into items
        elif isinstance(obj, list):
            return [expand_env_vars(i, max_recursion_depth - 1) for i in obj]
        # if this is a string, expand env vars and user (~)
        elif isinstance(obj, str):
            expanded = os.path.expandvars(os.path.expanduser(obj))
            return expanded
        else:
            return obj

    config = expand_env_vars(config)
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
        
    Raises:
        OSError: If directory creation or file writing fails
    """
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    except (OSError, IOError) as e:
        logger.error(f"Failed to save config to {save_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise OSError(f"Failed to save config to {save_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving config to {save_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to save configuration: {e}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any], max_depth: int = 20) -> Dict[str, Any]:
    """Update configuration with new values recursively.
    
    Args:
        config: Original configuration dictionary
        updates: Updates to apply
        max_depth: Maximum recursion depth to prevent infinite recursion
        
    Returns:
        Updated configuration dictionary
        
    Raises:
        ValueError: If maximum recursion depth is reached
        TypeError: If inputs are not dictionaries
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config)}")
    if not isinstance(updates, dict):
        raise TypeError(f"updates must be a dictionary, got {type(updates)}")
    
    if max_depth <= 0:
        raise ValueError("Maximum recursion depth reached")
    
    result = copy.deepcopy(config)
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = update_config(result[key], value, max_depth - 1)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary for required fields and valid values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If validation fails
        KeyError: If required keys are missing
    """
    logger.info("Validating configuration...")
    
    try:
        # Check required top-level keys
        required_keys = ['model', 'loss', 'optimizer', 'data', 'trainer', 'experiment']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")
        
        # Validate model configuration
        _validate_model_config(config['model'])
        
        # Validate loss configuration
        _validate_loss_config(config['loss'])
        
        # Validate optimizer configuration
        _validate_optimizer_config(config['optimizer'])
        
        # Validate data configuration
        _validate_data_config(config['data'])
        
        # Validate trainer configuration
        _validate_trainer_config(config['trainer'])
        
        # Validate experiment configuration
        _validate_experiment_config(config['experiment'])
        
        logger.info("Configuration validation passed")
        
    except Exception as e:
        logger.error(f"Configuration validation failed")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def _validate_model_config(model_config: Dict[str, Any]) -> None:
    """Validate model configuration."""
    if 'name' not in model_config:
        raise ValueError("Model configuration must contain 'name' field")
    
    model_name = model_config['name']
    valid_models = ['BIOT', 'BIOTHierarchical', 'SFCN', 'FAMED']
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name: {model_name}. Valid models: {valid_models}")
    
    # Check if model-specific config exists
    if model_name not in model_config:
        logger.warning(f"No specific configuration found for model {model_name}")


def _validate_loss_config(loss_config: Dict[str, Any]) -> None:
    """Validate loss configuration."""
    if 'name' not in loss_config:
        raise ValueError("Loss configuration must contain 'name' field")
    
    loss_name = loss_config['name']
    valid_losses = ['FocalLoss', 'CrossEntropyLoss', 'MSELoss', 'BCEWithLogitsLoss']
    if loss_name not in valid_losses:
        raise ValueError(f"Invalid loss name: {loss_name}. Valid losses: {valid_losses}")
    
    # Validate FocalLoss specific parameters
    if loss_name == 'FocalLoss' and loss_name in loss_config:
        focal_params = loss_config[loss_name]
        if 'alpha' in focal_params and focal_params['alpha'] <= 0:
            raise ValueError(f"FocalLoss alpha must be positive, got {focal_params['alpha']}")
        if 'gamma' in focal_params and focal_params['gamma'] < 0:
            raise ValueError(f"FocalLoss gamma must be non-negative, got {focal_params['gamma']}")


def _validate_optimizer_config(optimizer_config: Dict[str, Any]) -> None:
    """Validate optimizer configuration."""
    if 'name' not in optimizer_config:
        raise ValueError("Optimizer configuration must contain 'name' field")
    
    optimizer_name = optimizer_config['name']
    valid_optimizers = ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad']
    if optimizer_name not in valid_optimizers:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}. Valid optimizers: {valid_optimizers}")
    
    # Validate learning rate if specified
    if optimizer_name in optimizer_config:
        opt_params = optimizer_config[optimizer_name]
        if 'lr' in opt_params:
            lr = opt_params['lr']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"Learning rate must be positive, got {lr}")


def _validate_data_config(data_config: Dict[str, Any]) -> None:
    """Validate data configuration."""
    if 'name' not in data_config:
        raise ValueError("Data configuration must contain 'name' field")
    
    data_name = data_config['name']
    valid_data_modules = ['MEGDataModule', 'MEGOnTheFlyDataModule']
    if data_name not in valid_data_modules:
        raise ValueError(f"Invalid data module name: {data_name}. Valid modules: {valid_data_modules}")
    
    # Validate batch sizes if specified
    if data_name in data_config and 'dataloader_config' in data_config[data_name]:
        dataloader_config = data_config[data_name]['dataloader_config']
        for split in ['train', 'val', 'test']:
            if split in dataloader_config and 'batch_size' in dataloader_config[split]:
                batch_size = dataloader_config[split]['batch_size']
                if not isinstance(batch_size, int) or batch_size <= 0:
                    raise ValueError(f"Batch size for {split} must be a positive integer, got {batch_size}")


def _validate_trainer_config(trainer_config: Dict[str, Any]) -> None:
    """Validate trainer configuration."""
    # Validate max_epochs
    if 'max_epochs' in trainer_config:
        max_epochs = trainer_config['max_epochs']
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError(f"max_epochs must be a positive integer, got {max_epochs}")
    
    # Validate accelerator
    if 'accelerator' in trainer_config:
        accelerator = trainer_config['accelerator']
        valid_accelerators = ['cpu', 'gpu', 'cuda', 'auto']
        if accelerator not in valid_accelerators:
            raise ValueError(f"Invalid accelerator: {accelerator}. Valid accelerators: {valid_accelerators}")
    
    # Validate precision
    if 'precision' in trainer_config:
        precision = trainer_config['precision']
        valid_precisions = ['32', '16', '16-mixed', 'bf16', 'bf16-mixed']
        if str(precision) not in valid_precisions:
            raise ValueError(f"Invalid precision: {precision}. Valid precisions: {valid_precisions}")


def _validate_experiment_config(experiment_config: Dict[str, Any]) -> None:
    """Validate experiment configuration."""
    # Validate seed
    if 'seed' in experiment_config:
        seed = experiment_config['seed']
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got {seed}")
    
    # Validate experiment name
    if 'name' in experiment_config:
        name = experiment_config['name']
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Experiment name must be a non-empty string, got {name}")
