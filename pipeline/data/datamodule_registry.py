"""DataModule registry for configurably selecting and instantiating data modules.

This module provides a registry pattern for managing Lightning DataModule classes,
allowing dynamic instantiation based on configuration dictionaries.
"""

import logging
from typing import Dict, Type, Any
import lightning.pytorch as L
from .meg_datamodules import MEGOnTheFlyDataModule, PredictionDataModule

logger = logging.getLogger(__name__)

DATAMODULE_REGISTRY: Dict[str, Type[L.LightningDataModule]] = {
    "MEGOnTheFlyDataModule": MEGOnTheFlyDataModule,
    "PredictionDataModule": PredictionDataModule,
}


def get_datamodule_class(datamodule_name: str) -> Type[L.LightningDataModule]:
    """Get datamodule class from registry by name.

    Args:
        datamodule_name: Name of the datamodule to retrieve.

    Returns:
        The DataModule class corresponding to the given name.

    Raises:
        ValueError: If the datamodule name is not found in the registry.
    """
    if datamodule_name not in DATAMODULE_REGISTRY:
        raise ValueError(f"DataModule '{datamodule_name}' not found in registry. Available datamodules: {list(DATAMODULE_REGISTRY.keys())}")
    
    return DATAMODULE_REGISTRY[datamodule_name]


def create_datamodule(datamodule_config: Dict[str, Any]) -> L.LightningDataModule:
    """Create a datamodule instance based on configuration.

    Args:
        datamodule_config: Configuration dictionary containing 'name' and parameters.

    Returns:
        Instantiated datamodule ready for use.

    Raises:
        ValueError: If 'name' field is missing from configuration.
    """
    logger.info(f"Creating datamodule from config: {datamodule_config}")
    
    config = datamodule_config.copy()
    
    datamodule_name = config.pop("name", None)
    if datamodule_name is None:
        raise ValueError("DataModule configuration must contain 'name' field")
    
    logger.info(f"DataModule name: {datamodule_name}")
    
    datamodule_class = get_datamodule_class(datamodule_name)
    datamodule_params = config.get(datamodule_name, {})
    
    logger.info(f"DataModule parameters: {datamodule_params}")
    
    datamodule = datamodule_class(**datamodule_params)
    logger.info(f"DataModule created successfully: {datamodule.__class__.__name__}")
    return datamodule


def register_datamodule(name: str, datamodule_class: Type[L.LightningDataModule]) -> None:
    """Register a new datamodule in the registry.

    Args:
        name: Name identifier for the datamodule.
        datamodule_class: DataModule class to register.
    """
    DATAMODULE_REGISTRY[name] = datamodule_class
