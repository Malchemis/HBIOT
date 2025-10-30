#!/usr/bin/env python3
"""MEG spike detection prediction script.

This module provides functionality for predicting spike events in MEG recordings
using trained models. It can be used as a standalone script or imported as a module.

The prediction pipeline loads a trained model checkpoint, processes MEG data through
the model, and generates predictions with timing information and confidence scores.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import lightning as L

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.data.datamodule_registry import create_datamodule
from pipeline.training.lightning_module import MEGSpikeDetector
from pipeline.utils.config_handler import load_config

logger = logging.getLogger(__name__)


def predict_spikes(
    file_path: str,
    config_path: str,
    checkpoint_path: str,
    compute_gfp_peaks: bool = True,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Predict spikes in a MEG file.

    Args:
        file_path: Path to the MEG file (.fif or .ds).
        config_path: Path to the configuration file.
        checkpoint_path: Path to model checkpoint.
        compute_gfp_peaks: Whether to compute GFP peaks for onset adjustment,
            otherwise use the center of windows.
        output_csv: Path to save CSV results (optional).

    Returns:
        DataFrame with columns: onset, duration, probas.

    Example:
        >>> df = predict_spikes(
        ...     file_path="data/patient_001.fif",
        ...     config_path="configs/default_config.yaml",
        ...     checkpoint_path="checkpoints/best.ckpt"
        ... )
        >>> print(df.head())
           onset  duration    probas
        0  12.34       0.0  0.823456
        1  15.67       0.0  0.234567
    """
    logging.basicConfig(level=logging.WARNING)

    config = load_config(config_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Using checkpoint: {checkpoint_path}")
    input_shape = tuple(config["model"][config["model"]["name"]]["input_shape"])

    model = MEGSpikeDetector.load_from_checkpoint(
        checkpoint_path, config=config, input_shape=input_shape, log_dir=None
    )

    print(f"Best threshold from training: {model.threshold:.4f}")

    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )

    prediction_config = {
        "name": "PredictionDataModule",
        "PredictionDataModule": {
            "file_path": file_path,
            "dataset_config": config["data"][config["data"]["name"]]["dataset_config"],
            "dataloader_config": config["data"][config["data"]["name"]]["dataloader_config"],
        }
    }

    datamodule = create_datamodule(prediction_config)
    datamodule.setup(stage='predict')

    predictions = trainer.predict(model, datamodule=datamodule)

    results = []
    assert predictions is not None, "No predictions returned from the model."
    for batch_predictions in predictions:
        if not isinstance(batch_predictions, dict):
            continue
        batch_metadata = batch_predictions.get('metadata', [])
        probs = batch_predictions['probs']
        mask = batch_predictions.get('mask', None)

        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        elif probs.dim() == 3:
            probs = probs.squeeze(-1)

        batch_size = probs.shape[0]
        for i in range(batch_size):
            if i >= len(batch_metadata):
                continue

            metadata = batch_metadata[i]

            if 'window_times' in metadata:
                window_times = metadata['window_times']
                n_windows = metadata['n_windows']

                sample_probs = probs[i].squeeze()
                sample_mask = mask[i] if mask is not None and mask.ndim == 2 else None

                for j in range(n_windows):
                    if j >= len(window_times):
                        continue

                    if sample_mask is not None and sample_mask[j] == 0:
                        continue

                    seg_time = window_times[j]
                    prob = float(sample_probs[j].item()) if sample_probs.ndim == 1 else float(sample_probs[j, 0].item())

                    onset = seg_time['center_time'] if not compute_gfp_peaks else seg_time['peak_time']
                    results.append({
                        'onset': onset,
                        'duration': 0,
                        'probas': prob
                    })

    df = pd.DataFrame(results)

    print(f"Total predictions before filtering: {len(df)}")

    if not df.empty:
        df = df.drop_duplicates(subset='onset')

    if not df.empty:
        df = df.sort_values('onset').reset_index(drop=True)

    print(f"Total predictions after removing duplicates: {len(df)}")

    logger.info(f"Generated {len(df)} predictions for {file_path}")

    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to: {output_csv}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict spikes in MEG data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to MEG file (.fif, .ds, or 4D format)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (auto-generated if not provided)')
    parser.add_argument('--no-gfp', action='store_true',
                        help='Skip GFP peak calculation for onset adjustment')

    args = parser.parse_args()

    df = predict_spikes(
        file_path=args.file_path,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        compute_gfp_peaks=not args.no_gfp,
        output_csv=args.output
    )

    print(f"\nPrediction Summary:")
    print(f"Input file: {args.file_path}")
    print(f"Total windows: {len(df)}")
    print(f"Mean probability: {df['probas'].mean():.4f}")
    print(f"Output file: {args.output}")
