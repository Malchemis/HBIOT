"""File operations for MEG data processing."""

import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def find_ds_files(root_dir: str, patient_groups: Dict[str, str], skip_patterns: List[re.Pattern]) -> List[Dict[str, str]]:
    """Find all .ds directories in the hierarchical directory structure.

    Args:
        root_dir: Root directory to search.
        patient_groups: Dictionary to populate with patient group information.
        skip_patterns: List of regex patterns to skip certain files.

    Returns:
        List of dictionaries containing paths to .ds directories, patient IDs, and groups.
    """
    ds_files = []
    logger.debug(f"Searching for .ds files in {root_dir}")

    # Get group directories
    group_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    logger.debug(f"Found {len(group_dirs)} groups: {group_dirs}")

    # Sequential processing
    for group_dir in group_dirs:
        group_results, group_dir = process_group(root_dir, group_dir, skip_patterns)
        ds_files.extend(group_results)

        # Update patient groups
        for result in group_results:
            patient_groups[result['patient_id']] = result['group']

    logger.info(f"Found {len(ds_files)} .ds files across {len(patient_groups)} patients")
    return ds_files


def process_group(root_dir: str, group_dir: str, skip_patterns: List[re.Pattern]) -> Tuple[List[Dict[str, str]], str]:
    """Helper function to process a group directory to find .ds files.

    Args:
        root_dir: Root directory containing group directories.
        group_dir: Name of the group directory to process.
        skip_patterns: List of regex patterns to skip certain files.

    Returns:
        Tuple of (group_results, group_dir)
    """
    group_results = []
    group_path = str(os.path.join(root_dir, group_dir))

    # Process each patient directory within the group
    for patient_dir in os.listdir(group_path):
        patient_path = os.path.join(group_path, patient_dir)

        # Skip if not a directory
        if not os.path.isdir(patient_path):
            continue

        # Look for .ds files in the original structure (Group/Patient)
        found_ds = False
        for item in os.listdir(patient_path):
            if item.endswith('.ds') and os.path.isdir(os.path.join(patient_path, item)):
                ds_path = os.path.join(patient_path, item)

                # Check if file should be skipped
                if should_skip_file(ds_path, skip_patterns):
                    continue

                group_results.append({
                    'path': ds_path,
                    'patient_id': patient_dir,
                    'group': group_dir,
                    'filename': item
                })
                found_ds = True

        if not found_ds:
            logger.warning(f"No .ds files found for patient {patient_dir} in group {group_dir}")

    return group_results, group_dir


def get_patient_group(file_path: str, group_mappings: Optional[Dict[str, str]] = None) -> str:
    """Determine patient group from file path.
    
    Args:
        file_path: Path to the patient file
        group_mappings: Optional custom group mappings. If None, uses default mappings.
        
    Returns:
        Patient group string
    """
    if group_mappings is None:
        # Default group mappings - can be overridden via parameter
        group_mappings = {
            'Holdout': 'Holdout', 
            'IterativeLearningFeedback1': 'IterativeLearningFeedback', 
            'IterativeLearningFeedback2': 'IterativeLearningFeedback', 
            'MEG': 'MEG', 
            'Omega': 'Omega'
        }
    
    for key, val in group_mappings.items():
        if key in file_path:
            return val
    return 'Default'


def should_skip_file(file_path: str, skip_patterns: List[re.Pattern]) -> bool:
    """Check if a file should be skipped based on skip patterns.

    Args:
        file_path: Path to the file.
        skip_patterns: List of compiled regex patterns to match against the filename.

    Returns:
        True if the file should be skipped, False otherwise.
    """
    filename = os.path.basename(file_path)

    # Check against each skip pattern
    for pattern in skip_patterns:
        if pattern.search(filename) or pattern.search(file_path):
            logger.info(f"Skipping file {filename} - matches skip pattern {pattern.pattern}")
            return True
    return False


def save_chunks(chunks: List[np.ndarray], labels: List[np.ndarray],
                start_positions: List[int], spike_positions: List[List[List[Optional[int]]]],
                file_info: Dict[str, str], split: str, output_dir: str, window_duration_s: float,
                window_duration_samples: int, sampling_rate: int, is_test_set: bool,
                l_freq: float, h_freq: float, notch_freq: float, normalization: Dict[str, Any],
                save_as_pickle: bool) -> Dict[str, Any]:
    """Save preprocessed chunks to disk and return processing information.

    Args:
        chunks: List of preprocessed chunks.
        labels: List of labels for each window in each chunk.
        start_positions: Original starting position of each chunk in samples.
        spike_positions: Position of all spikes within each window for each chunk. If no spikes are present, the list is empty for the window.
        file_info: Dictionary with file information.
        split: Data split ('train', 'val', or 'test').
        output_dir: Output directory.
        window_duration_s: Length of each clip in seconds.
        window_duration_samples: Length of each clip in samples.
        sampling_rate: Sampling rate in Hz.
        is_test_set: Whether this file belongs to the test set.
        l_freq: Low frequency cutoff for filtering.
        h_freq: High frequency cutoff for filtering.
        notch_freq: Notch frequency for filtering.
        normalization: Normalization parameters.
        save_as_pickle: Whether to save the data as pickle files or PyTorch tensors.

    Returns:
        Dictionary with processing information including files, class counts, etc.
    """
    patient_id = file_info['patient_id']
    filename_origin = file_info['filename'].split('.')[0]
    group = file_info['group']

    # Prepare common metadata for all chunks from this file
    common_metadata = {
        'patient_id': patient_id,
        'original_filename': filename_origin,
        'group': group,
        'preprocessing_config': {
            'sampling_rate': sampling_rate,
            'l_freq': l_freq,
            'h_freq': h_freq,
            'notch_freq': notch_freq,
            'normalization': normalization,
        },
        'window_duration_s': window_duration_s,
        'window_duration_samples': window_duration_samples,
        'is_test_set': is_test_set
    }

    # Initialize processed_info
    processed_info = {
        'split': split,
        'files': [],
        'class_counts': {0: 0, 1: 0},
        'n_chunks': len(chunks),
        'n_windows': sum(len(chunk) for chunk in chunks),
        'chunk_metadata': []  # store chunk metadata
    }

    # Process each chunk
    for i in range(len(chunks)):
        # Create filename
        file_prefix = f"{patient_id}_{filename_origin}_{i:04d}".replace('__', '_')
        file_name = f"{file_prefix}.pt" if not save_as_pickle else f"{file_prefix}.pkl"
        file_path = os.path.join(output_dir, split, file_name)

        # Convert to PyTorch tensors or numpy arrays depending on the config
        if not save_as_pickle:
            chunk_tensor = np.array(chunks[i], dtype=np.float32)
            label_tensor = np.array(labels[i], dtype=np.float32)
        else:
            chunk_tensor = torch.tensor(chunks[i], dtype=torch.float32)
            label_tensor = torch.tensor(labels[i], dtype=torch.float32)

        # Determine if chunk has spikes
        has_spike = bool(torch.any(label_tensor == 1.0).item())
        spike_count = sum(len(pos) for pos in spike_positions[i])

        # Format spike positions as string
        spike_positions_str = ""
        for window_idx, positions in enumerate(spike_positions[i]):
            if positions:
                for pos in positions:
                    spike_positions_str += f"{window_idx}:{pos};"

        # Remove trailing semicolon
        if spike_positions_str.endswith(';'):
            spike_positions_str = spike_positions_str[:-1]

        # Create chunk-specific metadata
        chunk_metadata = {
            'data': chunk_tensor,
            'label': label_tensor,
            'start_position': int(start_positions[i]),
            'original_index': i,
            'spike_positions': spike_positions[i],
            'n_windows': len(chunks[i]),
        }

        # Combine and save the data
        full_metadata = {**common_metadata, **chunk_metadata}
        if save_as_pickle:
            with open(file_path, 'wb') as f:
                pickle.dump(full_metadata, f)
        else:
            torch.save(full_metadata, file_path)

        # Add file to processed_info
        processed_info['files'].append(file_name)

        # Update class counts
        for label in label_tensor:
            processed_info['class_counts'][int(label)] += 1

        # Collect CSV metadata
        processed_info['chunk_metadata'].append({
            'split': split,
            'file_name': file_name,
            'patient_id': patient_id,
            'original_filename': filename_origin,
            'group': group,
            'start_position': int(start_positions[i]),
            'original_index': i,
            'spike_positions': spike_positions_str,
            'has_spike': has_spike,
            'spike_count': spike_count,
            'sampling_rate': sampling_rate,
            'n_windows': len(chunks[i]),
        })

    return processed_info

