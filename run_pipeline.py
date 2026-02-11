#!/usr/bin/env python3
"""MEG Spike Detection Training Pipeline.

A configurable PyTorch Lightning pipeline for training and testing MEG spike detection models.
Provides registry-based architecture for models, losses, optimizers, callbacks, and datamodules.
Supports distributed data parallel (DDP) training for multi-GPU environments.
"""

import os
import re
import logging
import random
import numpy as np
import torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
import argparse
import lightning as L
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from pipeline.utils.config_handler import load_config
from pipeline.data.datamodule_registry import create_datamodule
from pipeline.training.lightning_module import MEGSpikeDetector
from pipeline.training.callback_registry import create_callbacks


def setup_logging(log_level: str = "INFO", log_file: str = "pipeline.log") -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file.

    Returns:
        Configured logger instance.

    Raises:
        ValueError: If log_level is invalid.
    """
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across all frameworks.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    L.seed_everything(seed, workers=True)


def setup_ddp_strategy(find_unused_parameters: bool = True):
    """Setup DDP strategy for multi-GPU training.

    Args:
        find_unused_parameters: Whether to find unused parameters in DDP.

    Returns:
        DDPStrategy instance if multiple GPUs available, None otherwise.
    """
    logger = logging.getLogger(__name__)
    strategy = None

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logger.info(f"Multiple GPUs detected ({gpu_count}), setting up DDP")
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)
    else:
        logger.info(f"Single GPU or CPU detected ({gpu_count} GPUs), using single device training")

    return strategy


def find_best_checkpoint(
    checkpoint_dir: str,
    metric: str = "pr_auc",
    mode: str = "max"
) -> Optional[str]:
    """Find the best checkpoint file based on a specific metric.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        metric: Metric name to optimize for (e.g., 'pr_auc', 'f1', 'loss').
        mode: 'max' for metrics to maximize, 'min' for metrics to minimize.

    Returns:
        Path to the best checkpoint file, or None if no matching checkpoints found.
    """
    logger = logging.getLogger(__name__)
    ckpt_dir = Path(checkpoint_dir)

    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {ckpt_dir}")
        return None

    if not ckpt_dir.is_dir():
        logger.error(f"Checkpoint path is not a directory: {ckpt_dir}")
        return None

    pattern = rf"epoch=(\d+)-val_{re.escape(metric)}=([0-9]*\.?[0-9]+)\.ckpt"
    best_value = float('-inf') if mode == 'max' else float('inf')
    best_checkpoint = None

    for ckpt_file in ckpt_dir.glob("*.ckpt"):
        if ckpt_file.name.startswith("last"):
            continue

        match = re.search(pattern, ckpt_file.name)
        if match:
            epoch, value = match.groups()
            value = float(value)

            if (mode == 'max' and value > best_value) or (mode == 'min' and value < best_value):
                best_value = value
                best_checkpoint = ckpt_file

    if best_checkpoint:
        logger.info(f"Best checkpoint for {metric}: {best_checkpoint} (value: {best_value:.4f})")
        return str(best_checkpoint)
    else:
        logger.warning(f"No checkpoint found for metric: {metric}")
        return None


def main(config_path: str, resume: bool = False, test_only: bool = False, token_selection_dict: Optional[Dict[str, Any]] = None, n_windows: Optional[int] = None, batch_size: Optional[int] = None):
    """Main training and testing function.

    Args:
        config_path: Path to configuration file.
        resume: If True, resume training from checkpoint. If no checkpoint is specified, the latest checkpoint in the log directory will be used.
        test_only: If True, skip training and only run testing.
        token_selection_dict: Dictionary containing token selection parameters.
        n_windows: Number of windows to use (overrides config if provided).
        batch_size: Batch size to use (overrides config if provided).

    Raises:
        RuntimeError: If pipeline execution fails.
    """
    config = load_config(config_path, validate=True)

    token_selection_dict = token_selection_dict or {}
    token_selection_updates = {k: v for k, v in token_selection_dict.items() if v not in (None, False, 0)}
    config["model"]["BIOTHierarchical"]["token_selection"] = {
        **config["model"]["BIOTHierarchical"].get("token_selection", {}),
        **token_selection_updates
    }

    if token_selection_dict.get('use_cls_token', False):
        config["experiment"]["version"] += "-cls"
    if token_selection_dict.get('use_mean_pool', 0):
        config["experiment"]["version"] += f"-mean{token_selection_dict['use_mean_pool']}"
    if token_selection_dict.get('use_max_pool', False):
        config["experiment"]["version"] += "-max"
    if token_selection_dict.get('use_min_pool', False):
        config["experiment"]["version"] += "-min"
    if token_selection_dict.get('n_selected_tokens', 0) > 0:
        config["experiment"]["version"] += f"-{token_selection_dict['n_selected_tokens']}tok"

    if n_windows is not None and n_windows > 0:
        config["data"]["MEGOnTheFlyDataModule"]["dataset_config"]["n_windows"] = n_windows
        config["experiment"]["version"] += f"-{n_windows}win"

    if batch_size is not None and batch_size > 0:
        config["data"]["MEGOnTheFlyDataModule"]["dataloader_config"]["train"]["batch_size"] = batch_size
        config["data"]["MEGOnTheFlyDataModule"]["dataloader_config"]["val"]["batch_size"] = batch_size
        config["experiment"]["version"] += f"-bs{batch_size}"

    logging_config = config.get("logging", {})
    experiment_config = config.get("experiment", {})

    log_dir = Path(logging_config.get("log_dir", "logs"))
    experiment_name = experiment_config.get("name", "experiment")
    experiment_log_dir = log_dir / experiment_name
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging_config.get('log_level', 'INFO')
    pipeline_log_config = logging_config.get("pipeline_log", None)
    if pipeline_log_config:
        log_file_path = Path(pipeline_log_config)
        if not log_file_path.is_absolute():
            log_file = str(log_dir / log_file_path.name)
        else:
            log_file = str(log_file_path)
    else:
        log_file = str(log_dir / "pipeline.log")

    logger = setup_logging(log_level, log_file)
    separator_config = logging_config.get("separator", {"char": "=", "length": 50})
    separator = separator_config["char"] * separator_config["length"]

    logger.info(separator)
    logger.info("STARTING MEG SPIKE DETECTION PIPELINE")
    logger.info(separator)

    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Experiment version: {experiment_config.get('version', 'version_0')}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Experiment log directory: {experiment_log_dir}")
    logger.info(f"Pipeline log file: {log_file}")

    seed = experiment_config.get("seed", 42)
    logger.info(f"Setting random seed: {seed}")
    set_seed(seed)

    trainer_config = config.get("trainer", {})
    ddp_config = trainer_config.get("ddp", {})
    if "ddp" in trainer_config:
        trainer_config = {k: v for k, v in trainer_config.items() if k != "ddp"}
        config["trainer"] = trainer_config
    find_unused_params = ddp_config.get("find_unused_parameters", True)
    strategy = setup_ddp_strategy(find_unused_parameters=find_unused_params)

    logger.info("Creating datamodule...")
    datamodule = create_datamodule(config["data"])
    logger.info(f"DataModule created: {config['data']['name']}")

    datamodule.prepare_data()
    datamodule.setup(stage='fit' if not test_only else 'test')
    input_shape = datamodule.get_input_shape() # type: ignore

    ckpt_path = experiment_config.get("checkpoint_path", None)
    if ckpt_path:
        logger.info(f"Using checkpoint: {ckpt_path}")

    version = experiment_config.get("version", None)
    if ckpt_path:
        version = Path(ckpt_path).parent.name

    logger.info(f"Creating TensorBoard logger with save_dir={experiment_log_dir}, version={version}")
    tb_logger = TensorBoardLogger(
        save_dir=str(experiment_log_dir),
        version=version,
        name=""
    )
    logger.info(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")

    logger.info("Creating Lightning module...")
    lightning_module = MEGSpikeDetector(config, input_shape, log_dir=tb_logger.log_dir)
    logger.info(f"Model created: {config['model']['name']}")

    logger.info("Creating callbacks...")
    callbacks = create_callbacks(config["callbacks"], tb_logger.log_dir)
    logger.info(f"Created {len(callbacks)} callbacks")

    logger.info("Creating trainer...")
    trainer_config = config["trainer"]

    trainer_kwargs = {
        **trainer_config,
        "callbacks": callbacks,
        "logger": tb_logger,
        # "deterministic": True,
    }

    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
        logger.info(f"Using strategy: {strategy.__class__.__name__}")

    trainer = L.Trainer(**trainer_kwargs)

    logger.info("PIPELINE CONFIGURATION SUMMARY")
    logger.info("-"*30)
    logger.info(f"Experiment: {experiment_config.get('name', 'Unknown')}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Loss: {config['loss']['name']}")
    logger.info(f"Optimizer: {config['optimizer']['name']}")
    logger.info(f"DataModule: {config['data']['name']}")
    logger.info(f"Max Epochs: {trainer_config.get('max_epochs', 50)}")
    logger.info(f"Batch Size: {config['data'].get(config['data']['name'], {}).get('dataloader_config', {}).get('train', {}).get('batch_size', 'Unknown')}")
    logger.info(f"Accelerator: {trainer_kwargs['accelerator']}")
    logger.info(f"Devices: {trainer_kwargs['devices']}")

    logger.info("Preparing data...")
    datamodule.prepare_data()
    
    if resume:
        logger.info("Resuming training...")
        if not ckpt_path:
            logger.info("No checkpoint specified, searching for latest checkpoint in log directory...")
            last_ckpt = Path(tb_logger.log_dir) / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                logger.info(f"Resuming from latest checkpoint: {ckpt_path}")
            else:
                logger.warning("No latest checkpoint found, starting fresh training")
            logger.info(f"Best checkpoint found: {ckpt_path}")

    if not test_only:
        logger.info("Starting training...")
        trainer.fit(lightning_module, datamodule, ckpt_path=ckpt_path)
        logger.info("Training completed successfully.")
    else:
        logger.info("Skipping training (test-only mode)")

    logger.info("Starting testing...")
    if not ckpt_path:
        ckpt_path = find_best_checkpoint(tb_logger.log_dir)
        logger.info(f"Best checkpoint found: {ckpt_path}")
    trainer.test(lightning_module, datamodule, ckpt_path=ckpt_path)
    logger.info("Testing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MEG spike detection pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file (default: configs/default_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training, if no checkpoint is specified, the latest checkpoint in the log directory will be used"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and only run testing on a pre-trained model"
    )
    parser.add_argument(
        "--use_cls_token",
        action="store_true",
        help="Use CLS token for classification"
    )
    parser.add_argument(
        "--use_mean_pool",
        type=int,
        help="Use k-th moment pooling for classification"
    )
    parser.add_argument(
        "--use_max_pool",
        action="store_true",
        help="Use max pooling for classification"
    )
    parser.add_argument(
        "--use_min_pool",
        action="store_true",
        help="Use min pooling for classification"
    )
    parser.add_argument(
        "--n_selected_tokens",
        type=int,
        help="Number of attention-selected tokens"
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        help="Number of windows to use (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use (overrides config)"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not config_path.suffix.lower() in ['.yaml', '.yml']:
        raise ValueError(f"Configuration file must be a YAML file, got: {config_path.suffix}")

    if args.test_only:
        logger = logging.getLogger(__name__)
        logger.info("Running in test-only mode")

    token_selection_dict = {
        "use_cls_token": args.use_cls_token,
        "use_mean_pool": args.use_mean_pool,
        "use_max_pool": args.use_max_pool,
        "use_min_pool": args.use_min_pool,
    }

    if args.n_selected_tokens is not None:
        if args.n_selected_tokens < 0:
            raise ValueError(f"n_selected_tokens must be non-negative, got: {args.n_selected_tokens}")
        token_selection_dict["n_selected_tokens"] = args.n_selected_tokens

    main(str(config_path), resume=args.resume, test_only=args.test_only, token_selection_dict=token_selection_dict, n_windows=args.n_windows, batch_size=args.batch_size)
