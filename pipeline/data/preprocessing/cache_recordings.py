#!/usr/bin/env python3
"""Cache preprocessed MEG recordings for efficient loading and online windowing.

This module provides functionality for caching preprocessed MEG recordings to HDF5 files
to enable fast loading and efficient online windowing during training. It supports parallel
preprocessing with configuration-based cache invalidation.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pipeline.data.preprocessing.annotation import compile_annotation_patterns, get_spike_annotations
from pipeline.data.preprocessing.file_manager import get_patient_group
from pipeline.data.preprocessing.signal_processing import load_and_process_meg_data

logger = logging.getLogger(__name__)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic hash of preprocessing configuration.

    Only includes parameters that affect preprocessing output to avoid
    unnecessary cache invalidation.

    Args:
        config: Dataset configuration dictionary.

    Returns:
        8-character hex hash string.
    """
    preprocessing_params = {
        'sampling_rate': config.get('sampling_rate'),
        'l_freq': config.get('l_freq'),
        'h_freq': config.get('h_freq'),
        'notch_freq': config.get('notch_freq'),
        'normalization': config.get('normalization'),
        'median_filter_temporal_window_ms': config.get('median_filter_temporal_window_ms'),
        'special_case_handling': config.get('special_case_handling'),
    }

    config_str = json.dumps(preprocessing_params, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]


def get_cache_path(file_path: str, cache_dir: str, config: Dict[str, Any]) -> Path:
    """Generate cache path for a preprocessed recording.

    Args:
        file_path: Original MEG file path.
        cache_dir: Base directory for cached files.
        config: Dataset configuration (for hash computation).

    Returns:
        Path to cached HDF5 file.
    """
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    config_hash = compute_config_hash(config)

    parts = Path(file_path).parts
    if len(parts) >= 2:
        patient_id = parts[-2]
        filename = Path(parts[-1]).stem
        cache_name = f"{patient_id}_{filename}_{file_hash}_{config_hash}.h5"
    else:
        cache_name = f"recording_{file_hash}_{config_hash}.h5"

    return Path(cache_dir) / cache_name


def preprocess_recording(
    file_path: str,
    config: Dict[str, Any],
    good_channels: List[str],
    compiled_patterns: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Preprocess a single MEG recording.

    Wraps existing preprocessing pipeline to return data for caching.

    Args:
        file_path: Path to MEG file.
        config: Dataset configuration.
        good_channels: Ordered list of reference channel names.
        compiled_patterns: Precompiled annotation patterns (optional).

    Returns:
        Tuple of (meg_data, spike_samples, metadata, channel_info).
    """
    logger.debug(f"Preprocessing {file_path}")

    if compiled_patterns is None:
        compiled_patterns = compile_annotation_patterns(config.get('annotation_rules', {}))

    raw, meg_data, channel_info = load_and_process_meg_data(
        file_path, config, good_channels, close_raw=False
    )

    spike_onsets = []
    group = get_patient_group(file_path)
    if len(raw.annotations) > 0:
        spike_onsets = get_spike_annotations(raw.annotations, group, compiled_patterns)

    sampling_rate = raw.info['sfreq']
    raw.close()

    spike_samples = np.array([int(onset * sampling_rate) for onset in spike_onsets], dtype=np.int64)

    metadata = {
        'file_name': file_path,
        'patient_id': str(Path(file_path).parts[-2]) if len(Path(file_path).parts) >= 2 else 'unknown',
        'original_filename': Path(file_path).name,
        'group': group,
        'sampling_rate': float(sampling_rate),
        'n_samples': int(meg_data.shape[1]),
        'n_spikes_in_recording': len(spike_samples),
        'duration_s': float(meg_data.shape[1] / sampling_rate),
        'preprocessing_config': {
            'sampling_rate': config['sampling_rate'],
            'l_freq': config.get('l_freq', 0.5),
            'h_freq': config.get('h_freq', 95.0),
            'notch_freq': config.get('notch_freq', 50.0),
            'normalization': config.get('normalization', {}),
            'median_filter_temporal_window_ms': config.get('median_filter_temporal_window_ms', 0.0),
        },
    }

    return meg_data, spike_samples, metadata, channel_info


def save_preprocessed_recording(
    output_path: Path,
    meg_data: np.ndarray,
    spike_samples: np.ndarray,
    metadata: Dict[str, Any],
    channel_info: Dict[str, Any]
) -> None:
    """Save preprocessed recording to HDF5 file.

    Args:
        output_path: Path to output HDF5 file.
        meg_data: MEG data array (n_channels, n_samples).
        spike_samples: Spike sample indices.
        metadata: Recording metadata.
        channel_info: Channel selection and mask information.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            'meg_data',
            data=meg_data,
            dtype=np.float32,
            # compression='gzip',
            # compression_opts=4,
        )

        f.create_dataset(
            'spike_samples',
            data=spike_samples,
            dtype=np.int64,
        )

        f.create_dataset(
            'metadata',
            data=json.dumps(metadata),
            dtype=h5py.string_dtype('utf-8'),
        )

        f.create_dataset(
            'channel_info',
            data=json.dumps({
                'selected_channels': channel_info.get('selected_channels', []),
                'channel_mask': channel_info.get('channel_mask', []).tolist() if hasattr(channel_info.get('channel_mask', []), 'tolist') else channel_info.get('channel_mask', []),
            }),
            dtype=h5py.string_dtype('utf-8'),
        )

    logger.debug(f"Saved preprocessed recording to {output_path}")


def load_preprocessed_recording(
    cache_path: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Load preprocessed recording from HDF5 file.

    Thread-safe for concurrent reads in DDP mode.

    Args:
        cache_path: Path to cached HDF5 file

    Returns:
        Tuple of (meg_data, spike_samples, metadata, channel_info)
    """
    with h5py.File(cache_path, 'r', swmr=True) as f:  # SWMR = Single Writer Multiple Reader
        meg_data = f['meg_data'][:]                # type: ignore
        spike_samples = f['spike_samples'][:]      # type: ignore
        metadata = json.loads(f['metadata'][()])                              # type: ignore
        channel_info = json.loads(f['channel_info'][()])                      # type: ignore

    return meg_data, spike_samples, metadata, channel_info # type: ignore


def _preprocess_and_cache_single_file(
    file_path: str,
    config: Dict[str, Any],
    good_channels: List[str],
    cache_dir: str,
    force: bool,
    compiled_patterns: Dict
) -> bool:
    """Worker function for parallel preprocessing.

    Returns:
        True if preprocessing was performed, False if cached file was used.
    """
    cache_path = get_cache_path(file_path, cache_dir, config)

    if cache_path.exists() and not force:
        logger.debug(f"Using cached file: {cache_path}")
        return False

    try:
        meg_data, spike_samples, metadata, channel_info = preprocess_recording(
            file_path, config, good_channels, compiled_patterns
        )

        save_preprocessed_recording(
            cache_path, meg_data, spike_samples, metadata, channel_info
        )

        return True

    except Exception as e:
        logger.error(f"Error preprocessing {file_path}: {e}")
        raise


def preprocess_and_cache_files(
    file_paths: List[str],
    config: Dict[str, Any],
    good_channels: List[str],
    cache_dir: str,
    force: bool = False,
    n_workers: int = 4
) -> Dict[str, int]:
    """Preprocess and cache multiple MEG files in parallel.

    Args:
        file_paths: List of MEG file paths to preprocess.
        config: Dataset configuration.
        good_channels: Ordered list of reference channel names.
        cache_dir: Directory to store cached files.
        force: If True, reprocess even if cached files exist.
        n_workers: Number of parallel workers.

    Returns:
        Dictionary with statistics (n_processed, n_cached, n_failed).
    """
    logger.info(f"Preprocessing {len(file_paths)} files with {n_workers} workers")
    logger.info(f"Cache directory: {cache_dir}")

    compiled_patterns = compile_annotation_patterns(config.get('annotation_rules', {}))

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    results: List[bool] = Parallel(n_jobs=n_workers, backend='threading')(
        delayed(_preprocess_and_cache_single_file)(
            fp, config, good_channels, cache_dir, force, compiled_patterns
        )
        for fp in tqdm(file_paths, desc="Preprocessing files", unit="file")
    ) # type: ignore
    
    n_processed = sum(results)
    n_cached = len(results) - n_processed
    n_failed = len(file_paths) - len(results)

    stats = {
        'n_total': len(file_paths),
        'n_processed': n_processed,
        'n_cached': n_cached,
        'n_failed': n_failed,
    }

    logger.info(f"Preprocessing complete: {n_processed} processed, {n_cached} cached, {n_failed} failed")

    return stats


def check_cache_exists(
    file_paths: List[str],
    config: Dict[str, Any],
    cache_dir: str
) -> Tuple[List[str], List[str]]:
    """Check which files have cached versions.

    Args:
        file_paths: List of file paths to check.
        config: Dataset configuration.
        cache_dir: Cache directory.

    Returns:
        Tuple of (cached_files, missing_files).
    """
    cached = []
    missing = []

    for fp in file_paths:
        cache_path = get_cache_path(fp, cache_dir, config)
        if cache_path.exists():
            cached.append(fp)
        else:
            missing.append(fp)

    return cached, missing
