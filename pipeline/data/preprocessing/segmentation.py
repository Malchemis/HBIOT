"""Data segmentation operations for MEG data."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import random


def create_windows(
    meg_data: np.ndarray,
    sampling_rate: float,
    window_duration_s: float,
    window_overlap: float,
) -> np.ndarray:
    """Create windows from MEG data.
    
    Args:
        meg_data: MEG data array (n_channels, n_timepoints)
        sampling_rate: Sampling rate in Hz
        window_duration_s: Duration of each window in seconds
        window_overlap: Overlap between windows (0.0 to 1.0)
        
    Returns:
        Array of windows with shape (n_windows, n_channels, n_samples_per_window)
    """
    window_duration_samples = int(window_duration_s * sampling_rate)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    
    windows = []
    seg_start = 0
    
    while seg_start + window_duration_samples <= meg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(meg_data[:, seg_start:seg_end])
        seg_start += window_step
    
    return np.array(windows)


def create_chunks(meg_data: np.ndarray, spike_samples: List[int], config: Dict) -> Tuple:
    """Create chunks from MEG data."""
    window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
    window_overlap = config.get('window_overlap', 0.0)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    n_windows = config['n_windows']

    # Create windows
    windows, sp_pos, seg_labels = [], [], []
    seg_start = 0

    while seg_start + window_duration_samples <= meg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(meg_data[:, seg_start:seg_end])

        # Find spikes whose peak falls inside this window
        seg_spikes = []
        for onset in spike_samples:
            if seg_start <= onset < seg_end:
                seg_spikes.append(onset - seg_start)

        sp_pos.append(seg_spikes)
        seg_labels.append(1.0 if len(seg_spikes) > 0 else 0.0)
        
        # Move to next window with configured overlap
        seg_start += window_step
    
    # Create chunks
    chunks, chunk_labels, start_positions, chunk_sp_pos = [], [], [], []
    i = 0
    
    while i < len(windows):
        chunk_end = min(i + n_windows, len(windows))
        chunks.append(np.array(windows[i:chunk_end]))
        chunk_labels.append(np.array(seg_labels[i:chunk_end], dtype=np.float32))
        start_positions.append(i * window_step)
        chunk_sp_pos.append(sp_pos[i:chunk_end])
        
        i += n_windows
        if i >= len(windows):
            break
    
    return chunks, chunk_labels, np.array(start_positions, dtype=np.int32), chunk_sp_pos


def calculate_chunk_duration(config: Dict[str, Any]) -> int:
    """Calculate chunk duration in samples accounting for window overlap.

    Args:
        config: Dataset configuration with window parameters

    Returns:
        Chunk duration in samples
    """
    window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
    window_overlap = config.get('window_overlap', 0.0)
    n_windows = config['n_windows']

    # With overlap: step_size = window_duration * (1 - overlap)
    # Total duration = (n_windows - 1) * step_size + window_duration
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    chunk_duration_samples = (n_windows - 1) * window_step + window_duration_samples

    return chunk_duration_samples


def extract_random_chunk(
    meg_data: np.ndarray,
    spike_samples: List[int],
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract a random chunk from MEG recording with smart boundary handling.

    This function implements the user's proposed strategy:
    1. Calculate chunk duration accounting for window overlap
    2. Randomly select an onset position
    3. Extract chunk from recording (handling partial chunks at boundaries)
    4. Create windows within chunk
    5. Assign labels based on spikes in chunk

    Args:
        meg_data: MEG data array (n_channels, n_samples)
        spike_samples: List of spike sample indices in the full recording
        config: Dataset configuration
        seed: Random seed (None = random, int = deterministic)

    Returns:
        Tuple of:
            - windows: Array of windows (n_windows, n_channels, window_samples)
            - labels: Window labels (n_windows,)
            - metadata: Chunk metadata dictionary
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    recording_length = meg_data.shape[1]
    chunk_duration_samples = calculate_chunk_duration(config)

    # Smart boundary handling
    if recording_length >= chunk_duration_samples:
        # Normal case: can extract full chunk
        max_onset = recording_length - chunk_duration_samples
        onset = random.randint(0, max_onset)
        offset = onset + chunk_duration_samples
    else:
        # Edge case: recording shorter than chunk duration
        # Take entire recording (will be padded by collate_fn)
        onset = 0
        offset = recording_length

    # Extract chunk MEG data
    chunk_meg = meg_data[:, onset:offset]

    # Adjust spike positions relative to chunk
    chunk_spikes = [s - onset for s in spike_samples if onset <= s < offset]

    # Create windows within chunk
    windows = create_windows(
        chunk_meg,
        config['sampling_rate'],
        config['window_duration_s'],
        config.get('window_overlap', 0.0),
    )

    # Calculate labels for windows
    labels = calculate_window_labels_from_spikes(
        windows,
        chunk_spikes,
        config,
    )

    # Calculate window parameters
    window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
    window_overlap = config.get('window_overlap', 0.0)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

    # Calculate window indices based on chunk position in recording
    # For random extraction, start_window_idx is the first window that would start at onset
    start_window_idx = onset // window_step if window_step > 0 else 0
    end_window_idx = start_window_idx + len(windows)

    # Create metadata
    metadata = {
        # Chunk position in recording
        'chunk_onset_sample': onset,
        'chunk_offset_sample': offset,
        'chunk_duration_samples': offset - onset,

        # Window-level traceability (approximate for random extraction)
        'start_window_idx': start_window_idx,
        'end_window_idx': end_window_idx,
        'n_windows': len(windows),

        # Spike information
        'n_spikes_in_chunk': len(chunk_spikes),
        'spike_positions_in_chunk': chunk_spikes,

        # Extraction mode
        'extraction_mode': 'random',
    }

    return windows, labels, metadata


def extract_spike_centered_chunk(
    meg_data: np.ndarray,
    spike_samples: List[int],
    target_spike: int,
    config: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract a chunk guaranteed to contain a specific spike.

    The chunk onset is randomized so the target spike appears at a different
    position each time, giving the model varied temporal context.

    Args:
        meg_data: MEG data array (n_channels, n_samples).
        spike_samples: All spike sample indices in the recording.
        target_spike: The spike sample index this chunk must contain.
        config: Dataset configuration (same keys as ``extract_random_chunk``).
        seed: Random seed for reproducibility.

    Returns:
        Same as ``extract_random_chunk``: (windows, labels, metadata).
    """
    if seed is not None:
        random.seed(seed)

    recording_length = meg_data.shape[1]
    chunk_duration = calculate_chunk_duration(config)
    window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
    margin = window_duration_samples // 2

    # Valid onset range that keeps the target spike inside the chunk
    earliest_onset = max(0, target_spike - chunk_duration + margin)
    latest_onset = min(recording_length - chunk_duration, target_spike - margin)

    if latest_onset < earliest_onset:
        # Very short recording or spike near boundary — best effort
        onset = max(0, min(target_spike - chunk_duration // 2,
                           recording_length - chunk_duration))
        onset = max(0, onset)
    else:
        onset = random.randint(earliest_onset, latest_onset)

    offset = min(onset + chunk_duration, recording_length)

    # Extract chunk
    chunk_meg = meg_data[:, onset:offset]
    chunk_spikes = [s - onset for s in spike_samples if onset <= s < offset]

    # Create windows and labels
    windows = create_windows(
        chunk_meg,
        config['sampling_rate'],
        config['window_duration_s'],
        config.get('window_overlap', 0.0),
    )

    labels = calculate_window_labels_from_spikes(windows, chunk_spikes, config)

    window_overlap = config.get('window_overlap', 0.0)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    start_window_idx = onset // window_step if window_step > 0 else 0

    metadata = {
        'chunk_onset_sample': onset,
        'chunk_offset_sample': offset,
        'chunk_duration_samples': offset - onset,
        'start_window_idx': start_window_idx,
        'end_window_idx': start_window_idx + len(windows),
        'n_windows': len(windows),
        'n_spikes_in_chunk': len(chunk_spikes),
        'spike_positions_in_chunk': chunk_spikes,
        'extraction_mode': 'spike_centered',
        'target_spike_sample': target_spike,
    }

    return windows, labels, metadata


def calculate_window_labels_from_spikes(
    windows: np.ndarray,
    spike_positions: List[int],
    config: Dict[str, Any]
) -> np.ndarray:
    """Calculate window-level labels from spike positions within a chunk.

    Reuses the labeling logic from create_chunks() but as a standalone function.

    Args:
        windows: Array of windows (n_windows, n_channels, window_samples)
        spike_positions: Spike sample indices relative to chunk start
        config: Dataset configuration

    Returns:
        Array of labels (n_windows,) with values 0.0 or 1.0, or (n_windows, 2) if onset prediction
    """
    window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
    window_overlap = config.get('window_overlap', 0.0)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

    predict_event_onset = config.get('predict_event_onset', False)

    labels = []

    for i in range(len(windows)):
        seg_start = i * window_step
        seg_end = seg_start + window_duration_samples

        # A window is positive only if a spike peak falls inside it
        best_spike_pos = None

        for spike_pos in spike_positions:
            if seg_start <= spike_pos < seg_end:
                best_spike_pos = spike_pos
                break  # Use first spike peak found in window

        presence = 1.0 if best_spike_pos is not None else 0.0

        if predict_event_onset:
            onset = (best_spike_pos - seg_start) / window_duration_samples if best_spike_pos is not None else 0.0
            labels.append([presence, onset])
        else:
            labels.append(presence)

    return np.array(labels, dtype=np.float32)
