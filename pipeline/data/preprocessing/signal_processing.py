"""Signal processing operations for MEG data."""

import logging
from typing import List, Dict, Any, Tuple, Optional

import os
import pathlib
import numpy as np
import torch
from scipy.ndimage import median_filter
import mne
from mne.io.base import BaseRaw
from scipy import stats

logger = logging.getLogger(__name__)


def log_array_statistics(array: np.ndarray, name: str, logger_obj: Optional[logging.Logger] = None) -> None:
    """Log detailed statistics about an array for debugging NaN/inf issues.

    Args:
        array: Array to analyze
        name: Name/description of the array
        logger_obj: Logger to use (defaults to module logger)
    """
    if logger_obj is None:
        logger_obj = logger

    n_nan = np.isnan(array).sum()
    n_inf = np.isinf(array).sum()
    n_total = array.size

    if n_nan > 0 or n_inf > 0:
        logger_obj.error(f"ALERT {name}: NaN={n_nan}/{n_total} ({100*n_nan/n_total:.2f}%), Inf={n_inf}/{n_total} ({100*n_inf/n_total:.2f}%)")

    if n_nan == 0 and n_inf == 0:
        logger_obj.debug(f"OK {name}: shape={array.shape}, mean={array.mean():.4f}, std={array.std():.4f}, "
                        f"min={array.min():.4f}, max={array.max():.4f}, NaN=0, Inf=0")


def load_and_process_meg_data(
    file_path: str,
    config: Dict[str, Any],
    good_channels: Optional[List[str]] = None,
    n_channels: int = 275,
    close_raw: bool = True
) -> Tuple[BaseRaw, np.ndarray, Dict[str, Any]]:
    """Load and process MEG data for prediction.
    
    Args:
        file_path: Path to the MEG data file.
        config: Configuration dictionary with preprocessing parameters.
        good_channels: List of channels that should be present. If None, use all available channels (useful for inference on new systems).
        n_channels: Number of MEG channels to use (default: 275) to enforce consistent input size
        close_raw: Whether to close the MNE Raw object after processing to free memory.
            
    Returns:
        Tuple containing:
            - raw: MNE Raw object after processing.
            - data: Processed MEG data array (n_channels, n_timepoints).
            - channel_info: loc information and channel mask.
    """
    try:
        if ".ds" in file_path:
            raw = mne.io.read_raw_ctf(file_path, preload=False).pick(picks=['meg'], exclude='bads').load_data()
        elif ".fif" in file_path:
            raw = mne.io.read_raw_fif(file_path, preload=False).pick(picks=['meg'], exclude='bads').load_data()
        elif os.path.isdir(file_path):
            subject_path = pathlib.Path(file_path)
            files = list(subject_path.glob("*"))
            raw_fname = next((f for f in files if "rfDC" in f.name and f.suffix == ""), None)
            config_fname = next((f for f in files if "config" in f.name.lower()), None)
            hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

            if not all([raw_fname, config_fname, hs_fname]):
                raise ValueError("Missing BTi raw/config/hs files.")

            raw = mne.io.read_raw_bti(
                pdf_fname=str(raw_fname),
                config_fname=str(config_fname),
                head_shape_fname=str(hs_fname),
                preload=False,
                verbose=False,
            ).pick(picks=['meg'], exclude='bads').load_data()
        else:
            raise ValueError("Unsupported file type for subject path.")

        # Special case handling for specific file patterns
        for pattern, channels in config.get('special_case_handling', {}).items():
            if pattern in file_path:
                # Drop problematic channels before selecting
                channels_to_drop = [ch for ch in channels if ch in raw.ch_names]
                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)
                    logger.info(f"Dropped {len(channels_to_drop)} special case channels for pattern '{pattern}'")
        
        if good_channels is None:
            good_channels = list(raw.ch_names)  # Use all available channels if no reference provided
            logger.info(f"No good_channels provided, using all {len(good_channels)} available channels")
        
        # Select channels based on good channels and location information
        raw, channel_info = select_channels(raw, good_channels)
        
        # Resample and filter
        if raw.info['sfreq'] != config['sampling_rate']:
            raw.resample(sfreq=config['sampling_rate'])
        raw.filter(l_freq=config.get('l_freq', 0.5), h_freq=config.get('h_freq', 95.0))
        
        if config.get('notch_freq', 50.0) > 0:
            freqs = np.arange(config['notch_freq'], config['sampling_rate']/2, config['notch_freq']).tolist()
            raw.notch_filter(freqs=freqs)
        
        # Get raw data from MNE (in order of selected_channels)
        raw_data = np.array(raw.get_data())  # Shape: (n_selected_channels, n_timepoints)
        n_timepoints = raw_data.shape[1]
        log_array_statistics(raw_data, f"Raw data after filtering (file: {os.path.basename(file_path)})")

        # Now normalize and filter
        raw_data = normalize_data(raw_data, config.get('normalization', {'method': 'robust_zscore', 'axis': None}))
        log_array_statistics(raw_data, f"Data after normalization (file: {os.path.basename(file_path)})")

        if config.get('median_filter_temporal_window_ms', 0) > 0:
            raw_data = apply_median_filter(raw_data, config['sampling_rate'], config['median_filter_temporal_window_ms'])
            log_array_statistics(raw_data, f"Data after median filter (file: {os.path.basename(file_path)})")

        if close_raw:
            raw.close()
        
        # Reorder data to match good_channels exactly
        # This ensures all samples in batch have data at same positions
        # Position i in data array ALWAYS represents good_channels[i]
        num_channels = max(n_channels, len(good_channels))
        data = np.zeros((num_channels, n_timepoints), dtype=raw_data.dtype)
        channel_mask = torch.zeros(num_channels, dtype=torch.bool)

        # Create index mapping for efficiency
        good_channels_index = {ch: i for i, ch in enumerate(good_channels)}

        # Place each channel's data at its correct position
        for ch_idx, ch_name in enumerate(channel_info['selected_channels']):
            if ch_name in good_channels_index:
                target_idx = good_channels_index[ch_name]
                data[target_idx, :] = raw_data[ch_idx, :]
                channel_mask[target_idx] = True
            else:
                logger.warning(f"Channel {ch_name} not in good_channels reference - skipping")

        n_valid = channel_mask.sum().item()
        logger.debug(f"Channel masking (file: {os.path.basename(file_path)}): {n_valid}/{len(good_channels)} valid channels")
        log_array_statistics(data, f"Final data after channel reordering (file: {os.path.basename(file_path)})")

        # Store channel mask for batch collation
        channel_info['channel_mask'] = channel_mask

        return raw, data, channel_info

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


def select_channels(raw: BaseRaw, good_channels: List[str]) -> Tuple[BaseRaw, Dict[str, Any]]:
    """Select channels ensuring consistent ordering across all samples for batch compatibility.

    Args:
        raw: MNE Raw object containing MEG data
        good_channels: ORDERED list of reference channel names (defines canonical ordering)

    Returns:
        Tuple of (processed_raw, channel_info) where channel_info contains:
            - 'loc': Dictionary mapping selected channel names to coordinates (legacy)
            - 'selected_channels': ORDERED list of channel names matching good_channels order
            - 'n_selected': Number of channels actually present in raw data
            - 'n_with_coordinates': Number of channels with coordinate info (legacy)
    """
    logger.debug(f"Raw channels available: {len(raw.ch_names)} channels")
    logger.debug(f"Good channels reference: {len(good_channels)} channels")
    
    # Get available MEG channels from the raw data
    available_channels = set(raw.ch_names)  # Use set for O(1) lookup
    logger.debug(f"Available MEG channels: {len(available_channels)}")

    # Ensure batch consistency - all samples have same channel ordering
    selected_channels = [ch for ch in good_channels if ch in available_channels]

    if len(selected_channels) == 0:
        raise ValueError(f"No channels from good_channels found in raw data! "
                        f"Raw has: {list(raw.ch_names)[:10]}..., "
                        f"Expected: {good_channels[:10]}...")

    logger.debug(f"Selected {len(selected_channels)}/{len(good_channels)} channels from reference list")

    # Pick only the selected channels in the raw object
    raw = raw.pick_channels(selected_channels)
    channel_info = {
        'ch_info': raw.info['chs'],  # Full channel info from MNE},
        'selected_channels': selected_channels,
    }
    return raw, channel_info


def find_bad_channels(raw: BaseRaw, flat_th: float = 1e-20, noisy_th: float = 20.0) -> Tuple[List[str], List[str]]:
    """
    Find flat and noisy channels in MEG data.
    
    Args:
        raw: MNE Raw object containing MEG data
        flat_th: Threshold for flat channels (variance below this value, default 1e-15 for MEG)
        noisy_th: Threshold for noisy channels (z-score above this value, default 5.0)

    Returns:
        Tuple of (flat_channel_names, noisy_channel_names)
    """
    logger.debug(f"Finding bad channels in {len(raw.ch_names)} channels")
    
    # Get data for all channels
    data = raw.get_data()
    channel_names = raw.ch_names
    
    # Find flat channels (low variance)
    channel_vars = np.var(data, axis=1)
    flat_indices = np.where(channel_vars < flat_th)[0]
    flat_channels = [channel_names[i] for i in flat_indices]
    
    # Find noisy channels using Median Absolute Deviation (MAD)
    channel_medians = np.median(data, axis=1)
    channel_mads = stats.median_abs_deviation(data, axis=1)
    global_median = np.median(channel_medians)
    global_mad = np.median(channel_mads)
    
    # Z-score based on deviation from global statistics
    zscore = np.abs((channel_medians - global_median) / (global_mad + 1e-20))
    noisy_indices = np.where(zscore > noisy_th)[0]
    noisy_channels = [channel_names[i] for i in noisy_indices]
    
    if flat_channels:
        logger.info(f"Found {len(flat_channels)} flat channels: {flat_channels}")
    if noisy_channels:
        logger.info(f"Found {len(noisy_channels)} noisy channels: {noisy_channels}")
    
    return flat_channels, noisy_channels


def normalize_data(data: np.ndarray, norm_config: Dict, eps: Optional[float] = None) -> np.ndarray:
    """Normalize data using specified method."""
    if eps is None:
        eps = norm_config.get('epsilon', 1e-20)
    
    method = norm_config.get('method', 'robust_zscore')
    axis = norm_config.get('axis', None)
    
    if method == 'percentile':
        percentile = norm_config.get('percentile', 95)
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        q = np.percentile(np.abs(data), percentile, axis=axis, keepdims=True)
        return data / (q + eps)
    
    elif method == 'robust_normalize':
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + eps)
    
    elif method == 'robust_zscore':
        median = np.median(data, axis=axis, keepdims=True)
        mad = stats.median_abs_deviation(data, axis=axis)  # type: ignore
        if axis is not None:
            mad = np.expand_dims(mad, axis=axis)
        return (data - median) / (mad + eps)  # type: ignore
    
    elif method == 'zscore':
        return (data - np.mean(data, axis=axis, keepdims=True)) / (np.std(data, axis=axis, keepdims=True) + eps)
    
    elif method == 'minmax':
        min_v = np.min(data, axis=axis, keepdims=True)
        max_v = np.max(data, axis=axis, keepdims=True)
        return (data - min_v) / (max_v - min_v + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Supported methods: percentile, zscore, minmax, robust_normalize, robust_zscore")


def apply_median_filter(data: np.ndarray, sfreq: float, temporal_window_ms: float) -> np.ndarray:
    """Apply median filter with adaptive kernel size based on sampling frequency.
    
    Args:
        data: MEG data array of shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        temporal_window_ms: Temporal smoothing window in milliseconds
        
    Returns:
        Filtered data with same shape as input
    """
    if temporal_window_ms <= 0:
        return data
    
    # Calculate kernel size based on sampling frequency and temporal window
    kernel_samples = int(temporal_window_ms * sfreq / 1000)
    # Ensure odd kernel size for symmetric filtering
    kernel_size = kernel_samples if kernel_samples % 2 == 1 else kernel_samples + 1
    
    # Apply median filter along time axis (axis=1) for each channel
    return median_filter(data, size=(1, kernel_size))


def augment_data(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Augment data with random noise."""
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    return data


def compute_gfp(meg_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Global Field Power (GFP) from MEG data.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints) or (n_timepoints, n_channels)
        axis: Axis along which channels are located (0 for first dim, 1 for second dim)
        
    Returns:
        GFP values, shape (n_timepoints,)
    """
    # Compute GFP as standard deviation across channels
    gfp = np.std(meg_data, axis=axis)
    return gfp


def find_gfp_peak_in_window(
    meg_data: np.ndarray,
    window_start: int,
    window_end: int,
    sampling_rate: float
) -> Tuple[int, float]:
    """Find the peak GFP within a window.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints)
        window_start: Start sample of the window
        window_end: End sample of the window
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (peak_sample, peak_time_in_seconds)
    """
    # Extract window
    window_data = meg_data[:, window_start:window_end]
    
    # Compute GFP
    gfp = compute_gfp(window_data, axis=0)
    
    # Find peak
    peak_idx = np.argmax(gfp)
    peak_sample = window_start + peak_idx
    peak_time = peak_sample / sampling_rate
    
    return int(peak_sample), float(peak_time)