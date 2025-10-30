"""Feature extraction utilities for MEG spike detection.

This module provides optimized feature extraction functions for MEG epochs including:
- Morphological features (amplitude, slope, zero-crossing rate, etc.)
- Spectral features (band powers, peak frequency, spectral entropy)
- Wavelet features (discrete wavelet decomposition coefficients)

Features are extracted in parallel for efficient processing of large datasets.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Tuple

import numpy as np
import pywt
from numba import jit, prange
from scipy import signal, stats
from tqdm import tqdm


@jit(nopython=True, parallel=True)
def _compute_morphological_features_batch(
    epochs_flat: np.ndarray, 
    n_samples: int, 
    n_features: int = 13
) -> np.ndarray:
    """Numba-optimized morphological feature computation.
    
    Process all epochs and channels in parallel.
    
    Args:
        epochs_flat: Flattened epochs array.
        n_samples: Number of samples per epoch.
        n_features: Number of features to extract.
        
    Returns:
        Features array.
    """
    n_total = epochs_flat.shape[0]
    features = np.empty((n_total, n_features), dtype=np.float32)
    
    for i in prange(n_total):
        channel_data = epochs_flat[i]
        
        # Basic amplitude features
        features[i, 0] = np.max(channel_data)  # max_amp
        features[i, 1] = np.min(channel_data)  # min_amp
        features[i, 2] = features[i, 0] - features[i, 1]  # peak_to_peak
        features[i, 3] = np.mean(channel_data)  # mean_amp
        features[i, 4] = np.std(channel_data)  # std_amp
        
        # Time to peak
        max_idx = np.argmax(np.abs(channel_data))
        features[i, 5] = max_idx / n_samples  # normalized time_to_peak
        
        # Slope features - compute differences efficiently
        diff = np.diff(channel_data)
        features[i, 6] = np.max(diff)  # max_slope
        features[i, 7] = np.min(diff)  # min_slope
        features[i, 8] = np.mean(np.abs(diff))  # mean_abs_slope
        
        # Zero crossing rate
        signs = np.sign(channel_data + 1e-10)
        zero_crossings = 0
        for j in range(1, len(signs)):
            if signs[j] != signs[j-1]:
                zero_crossings += 1
        features[i, 9] = zero_crossings / n_samples  # zcr
        
        # Energy
        features[i, 10] = np.sum(channel_data ** 2)
        
        # Placeholder for kurtosis and skewness (computed separately due to Numba limitations)
        features[i, 11] = 0.0  # kurtosis
        features[i, 12] = 0.0  # skewness
    
    return features


def extract_morphological_features(epochs: np.ndarray) -> np.ndarray:
    """Extract morphological features from epochs using optimized NumPy operations.
    
    Args:
        epochs: np.ndarray of shape (n_epochs, n_channels, n_samples).
    
    Returns:
        Morphological features array.
    
    Features include:
    - Peak amplitude (max, min, peak-to-peak, mean, std)
    - Normalized time to peak (of absolute max)
    - Slope features (max, min, mean absolute diff)
    - Zero-crossing rate
    - Signal energy
    - Kurtosis and skewness
    """
    n_epochs, n_channels, n_samples = epochs.shape
    epochs_reshaped = epochs.reshape(-1, n_samples)
    
    # Use Numba-optimized function for main computations
    features_flat = _compute_morphological_features_batch(
        epochs_reshaped.astype(np.float32), n_samples
    )
    
    # Compute kurtosis and skewness using vectorized scipy operations
    # These can't be easily done in Numba, so compute separately
    kurt = stats.kurtosis(epochs_reshaped, axis=1)
    skew = stats.skew(epochs_reshaped, axis=1)
    
    # Replace NaN values with 0
    kurt = np.nan_to_num(kurt, nan=0.0)
    skew = np.nan_to_num(skew, nan=0.0)

    features_flat[:, 11] = kurt
    features_flat[:, 12] = skew
    
    # Reshape to (n_epochs, n_channels * n_features)
    features = features_flat.reshape(n_epochs, n_channels * features_flat.shape[1])
    
    return features.astype(np.float32)


def process_single_epoch(
    epoch_data: np.ndarray, 
    bands: Dict[str, Tuple[float, float]], 
    sfreq: float
) -> np.ndarray:
    """Process a single epoch's spectral features.
    
    Args:
        epoch_data: Single epoch data.
        bands: Frequency bands dictionary.
        sfreq: Sampling frequency.
        
    Returns:
        Extracted features for the epoch.
    """
    n_channels, n_samples = epoch_data.shape
    epoch_features = []
    
    for channel_idx in range(n_channels):
        channel_signal = epoch_data[channel_idx]
        
        # Compute PSD
        freqs, psd = signal.welch(channel_signal, fs=sfreq, 
                                nperseg=min(256, n_samples))
        
        total_power = np.sum(psd)
        
        # Band powers
        band_powers = []
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.sum(psd[band_mask])
            relative_power = band_power / total_power if total_power > 0 else 0
            band_powers.append(relative_power)
        
        # Peak frequency
        peak_freq = freqs[np.argmax(psd)] if psd.size > 0 else 0
        
        # Spectral edge frequency
        if total_power > 0:
            cumsum_psd = np.cumsum(psd)
            edge_freq_idx = np.where(cumsum_psd >= 0.95 * total_power)[0]
            edge_freq = freqs[edge_freq_idx[0]] if len(edge_freq_idx) > 0 else freqs[-1]
        else:
            edge_freq = 0
        
        # Spectral entropy
        if np.sum(psd) > 0:
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        else:
            spectral_entropy = 0
        
        channel_features = band_powers + [peak_freq, edge_freq, spectral_entropy, total_power]
        epoch_features.extend(channel_features)
    
    return np.array(epoch_features, dtype=np.float32)


def extract_spectral_features(
    epochs: np.ndarray, 
    sfreq: float = 200.0,
    n_cpus: int = mp.cpu_count() // 2
) -> np.ndarray:
    """Extract spectral features from epochs.
    
    Args:
        epochs: np.ndarray of shape (n_epochs, n_channels, n_samples).
        sfreq: Sampling frequency.
    
    Returns:
        Spectral features array.
        
    Features include:
    - Power in different frequency bands (delta, theta, alpha, beta, gamma)
    - Spectral edge frequency
    - Peak frequency
    - Spectral entropy
    """
    n_epochs, n_channels, n_samples = epochs.shape
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 99)
    }
    
    # Use parallel processing for epochs
    n_jobs = min(n_cpus, n_epochs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        features_list = list(tqdm(
            executor.map(process_single_epoch, 
                         epochs, 
                         [bands] * n_epochs, 
                         [sfreq] * n_epochs),
            total=n_epochs,
            desc="Extracting spectral features"
        ))
    
    return np.array(features_list, dtype=np.float32)


def process_single_epoch_wavelet(
    epoch_data: np.ndarray,
    wavelet: str,
    max_level: int
) -> np.ndarray:
    """Process a single epoch's wavelet features.
    
    Args:
        epoch_data: Single epoch data of shape (n_channels, n_samples).
        wavelet: Wavelet name (e.g., 'db4', 'sym5').
        max_level: Maximum decomposition level.
        
    Returns:
        Extracted wavelet features for the epoch.
    """
    n_channels, n_samples = epoch_data.shape
    epoch_features = []
    
    for channel_idx in range(n_channels):
        channel_signal = epoch_data[channel_idx]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(channel_signal, wavelet, level=max_level)
        
        channel_features = []
        
        # Extract features from approximation coefficients
        approx = coeffs[0]
        channel_features.extend([
            np.mean(approx),
            np.std(approx),
            np.max(approx),
            np.min(approx),
            np.sqrt(np.mean(approx**2))  # RMS energy
        ])
        
        # Extract features from detail coefficients at each level
        for level in range(1, len(coeffs)):
            detail = coeffs[level]
            channel_features.extend([
                np.mean(detail),
                np.std(detail),
                np.max(detail),
                np.min(detail),
                np.sqrt(np.mean(detail**2))  # RMS energy
            ])
        
        # Add wavelet entropy
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energies)
        if total_energy > 0:
            rel_energies = np.array(energies) / total_energy
            wavelet_entropy = -np.sum(rel_energies * np.log2(rel_energies + 1e-10))
        else:
            wavelet_entropy = 0
        channel_features.append(wavelet_entropy)
        
        epoch_features.extend(channel_features)
    
    return np.array(epoch_features, dtype=np.float32)


def extract_wavelet_features(
    epochs: np.ndarray,
    wavelet: str = 'db4',
    n_cpus: int = mp.cpu_count() // 2
) -> np.ndarray:
    """Extract discrete wavelet features from epochs.
    
    Args:
        epochs: np.ndarray of shape (n_epochs, n_channels, n_samples).
        sfreq: Sampling frequency (used to determine max decomposition level).
        wavelet: Wavelet name for decomposition.
        n_cpus: Number of CPUs for parallel processing.
        
    Returns:
        Wavelet features array of shape (n_epochs, n_channels * n_features).
        
    Features include (for each decomposition level):
    - Mean, std, max, min, RMS energy of coefficients
    - Wavelet entropy across all levels
    """
    n_epochs, n_channels, n_samples = epochs.shape

    # Limit to reasonable level (typically 4-6 is sufficient)
    max_level = pywt.dwt_max_level(n_samples, wavelet)
    max_level = min(max_level, 5)
    
    # Use parallel processing for epochs
    n_jobs = min(n_cpus, n_epochs)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        features_list = list(tqdm(
            executor.map(process_single_epoch_wavelet,
                         epochs,
                         [wavelet] * n_epochs,
                         [max_level] * n_epochs),
            total=n_epochs,
            desc="Extracting wavelet features"
        ))
    
    return np.array(features_list, dtype=np.float32)


def reshape_features_to_3d(features_2d: np.ndarray, n_channels: int) -> np.ndarray:
    """Reshape features from (n_epochs, n_channels * n_features) to (n_epochs, n_channels, n_features).
    
    Args:
        features_2d: Features array of shape (n_epochs, n_channels * n_features).
        n_channels: Number of channels.
        
    Returns:
        Reshaped features of shape (n_epochs, n_channels, n_features).
    """
    n_epochs = features_2d.shape[0]
    n_features_per_channel = features_2d.shape[1] // n_channels
    
    # Reshape to separate channels and features
    features_3d = features_2d.reshape(n_epochs, n_channels, n_features_per_channel)
    
    return features_3d


def raw_epochs_to_features(
    epochs: List[np.ndarray],
    sfreq: float = 200.0,
    n_cpus: int = mp.cpu_count() // 2,
    wavelet: str = 'db4',
) -> List[np.ndarray]:
    """Convert raw epochs to feature vectors with shape (n_epochs, n_channels, n_features).
    
    Args:
        epochs: List of length n_subjects, containing arrays each of shape (n_epochs, n_channels, n_samples).
        sfreq: Sampling frequency.
        n_cpus: Number of CPUs for parallel processing.
        wavelet: Wavelet name for decomposition.
        
    Returns:
        Tuple of:
        - List of feature arrays for each subject, each of shape (n_epochs_i, n_channels, n_features)
    """
    # Assume n_channels is consistent across all subjects
    n_channels = epochs[0].shape[1]
    
    # Track epoch counts per subject for later separation
    epoch_counts = [ep.shape[0] for ep in epochs]
    
    # Stack all epochs
    all_epochs = np.vstack(epochs)  # Shape: (n_total_epochs, n_channels, n_samples)
    
    # Extract all features
    morph_feats = extract_morphological_features(all_epochs)
    spec_feats = extract_spectral_features(all_epochs, sfreq=sfreq, n_cpus=n_cpus)
    wavelet_feats = extract_wavelet_features(all_epochs, wavelet=wavelet, n_cpus=n_cpus)
    
    # Reshape each feature type to 3D
    morph_3d = reshape_features_to_3d(morph_feats, n_channels)
    spec_3d = reshape_features_to_3d(spec_feats, n_channels)
    wavelet_3d = reshape_features_to_3d(wavelet_feats, n_channels)
    
    # Concatenate along feature dimension
    all_features_3d = np.concatenate([morph_3d, spec_3d, wavelet_3d], axis=2)
    
    # Split back into subjects
    subject_features = []
    start_idx = 0
    for n_epochs in epoch_counts:
        end_idx = start_idx + n_epochs
        subject_features.append(all_features_3d[start_idx:end_idx])
        start_idx = end_idx
    
    return subject_features
