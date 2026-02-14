"""Generate stratified cross-validation splits for MEG spike detection datasets.

This script creates stratified K-fold cross-validation splits ensuring balanced distribution
of spike counts across folds. It separates a test set (holdout patients) from the train/val pool,
then generates stratified folds based on spike count distributions.

Supports optional quality filtering:
- Per-spike filtering based on multi-channel activity (reject low-channel-count spikes)
- Subject-level filtering based on spike rate (remove subjects with suspiciously low rates)
"""
import json
import os
import re
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import logging
import argparse

import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from joblib import Parallel, delayed

import sys
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.data.preprocessing.annotation import get_spike_annotations, compile_annotation_patterns
from pipeline.data.preprocessing.file_manager import find_ds_files
from pipeline.data.preprocessing.filtering_stats import FilteringStatistics
from pipeline.data.preprocessing.signal_processing import apply_standard_filters, filter_spikes_by_channel_activity
from pipeline.utils.config_handler import load_config


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = "generate_splits.log") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file.

    Raises:
        ValueError: If log_level is invalid.
    """
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ],
        force=True
    )

def load_meg_data_and_count_spikes(
    file_path: str,
    patient_group: str,
    annotation_rules: Dict[str, Any],
    spike_quality_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load MEG data, count valid spike annotations, and optionally filter by quality.

    Args:
        file_path: Path to the MEG file (.ds, .fif, or BTi directory).
        patient_group: Patient group for annotation rules.
        annotation_rules: Dictionary containing annotation rules for each group.
        spike_quality_config: Optional config for per-spike quality filtering.
            Keys: enabled, threshold_zscore, min_active_channels, epoch_half_duration_s.
        preprocessing_config: Optional config with ``sampling_rate``, ``l_freq``,
            ``h_freq``, ``notch_freq``. Required when spike_quality_config is enabled.

    Returns:
        Dictionary with keys:
            - spike_count: Raw spike count (before quality filtering).
            - filtered_spike_count: Spike count after quality filtering (equals spike_count if disabled).
            - duration_s: Recording duration in seconds.
            - filter_stats: Per-spike filtering statistics (None if disabled).
    """
    mne.set_log_level(verbose='ERROR')
    result = {'spike_count': 0, 'filtered_spike_count': 0, 'duration_s': 0.0, 'filter_stats': None}

    try:
        if ".ds" in file_path:
            raw = mne.io.read_raw_ctf(file_path, preload=False, verbose=False).pick(picks=['meg'], exclude='bads').load_data()
        elif ".fif" in file_path:
            raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False).pick(picks=['meg'], exclude='bads').load_data()
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

        result['duration_s'] = raw.n_times / raw.info['sfreq']

        # Extract annotations
        annotations = raw.annotations
        if annotations is None or len(annotations) == 0:
            return result

        # Get spike annotations using the annotation processor
        spike_onsets = get_spike_annotations(annotations, patient_group, annotation_rules)
        result['spike_count'] = len(spike_onsets)
        result['filtered_spike_count'] = len(spike_onsets)

        # Optional per-spike quality filtering
        sq = spike_quality_config or {}
        if sq.get('enabled', False) and len(spike_onsets) > 0:
            # Apply the same preprocessing pipeline used during caching
            apply_standard_filters(raw, preprocessing_config)

            meg_data = raw.get_data()
            kept, rejected, fstats = filter_spikes_by_channel_activity(
                meg_data,
                spike_onsets,
                raw.info['sfreq'],
                epoch_half_duration_s=sq.get('epoch_half_duration_s', 0.05),
                threshold_zscore=sq.get('threshold_zscore', 2.5),
                min_active_channels=sq.get('min_active_channels', 5),
            )
            result['filtered_spike_count'] = len(kept)
            result['filter_stats'] = fstats

        return result

    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")
        return result


def _process_patient(patient_id, files, patient_group, annotation_rules,
                     spike_quality_config, preprocessing_config):
    total_spikes = 0
    total_filtered_spikes = 0
    total_duration_s = 0.0
    for file_info in files:
        file_result = load_meg_data_and_count_spikes(
            file_info['path'], patient_group, annotation_rules,
            spike_quality_config, preprocessing_config,
        )
        total_spikes += file_result['spike_count']
        total_filtered_spikes += file_result['filtered_spike_count']
        total_duration_s += file_result['duration_s']

    total_duration_min = total_duration_s / 60.0
    spikes_per_minute = total_filtered_spikes / max(total_duration_min, 1e-6)

    stats = {
        'group': patient_group,
        'total_spikes': total_spikes,
        'total_filtered_spikes': total_filtered_spikes,
        'total_duration_s': total_duration_s,
        'spikes_per_minute': spikes_per_minute,
        'file_count': len(files),
        'files': [f['path'] for f in files],
    }
    logger.debug(
        f"Patient {patient_id} ({patient_group}): "
        f"{total_spikes} raw spikes, {total_filtered_spikes} filtered spikes, "
        f"{total_duration_min:.1f} min, {spikes_per_minute:.2f} spikes/min"
    )
    return patient_id, stats


def generate_patient_spike_statistics(
    ds_files: List[Dict],
    patient_groups: Dict[str, str],
    annotation_rules: Dict[str, Any],
    spike_quality_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    njobs: int = 1,
) -> Dict[str, Dict]:
    """Generate spike statistics for each patient."""
    logger.info("Analyzing spike statistics for all patients...")

    patient_files = defaultdict(list)
    for file_info in ds_files:
        patient_files[file_info['patient_id']].append(file_info)

    results = Parallel(n_jobs=njobs, verbose=10)(
        delayed(_process_patient)(
            patient_id, files, patient_groups[patient_id],
            annotation_rules, spike_quality_config, preprocessing_config,
        )
        for patient_id, files in patient_files.items()
    )

    patient_stats = {patient_id: stats for patient_id, stats in results}
    return patient_stats


def create_stratification_bins(spike_counts: List[int], n_bins: int = 5) -> List[int]:
    """Create stratification bins based on spike counts.
    
    Args:
        spike_counts: List of spike counts
        n_bins: Number of bins to create
        
    Returns:
        List of bin assignments for each patient
    """
    if len(spike_counts) == 0:
        return []
    
    # Create percentile-based bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(spike_counts, percentiles)
    
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    
    # Assign bins
    bins = np.digitize(spike_counts, bin_edges) - 1
    # Ensure bins are in valid range [0, n_bins-1]
    bins = np.clip(bins, 0, len(bin_edges) - 2)
    
    return bins.tolist()


def generate_stratified_splits(train_val_patients: List[str], patient_stats: Dict[str, Dict], 
                               n_splits: int, random_state: int) -> List[Tuple[List[str], List[str]]]:
    """Generate stratified K-fold splits for train/validation.
    
    Args:
        train_val_patients: List of patient IDs for train/val splitting
        patient_stats: Patient statistics dictionary
        n_splits: Number of folds to generate
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_patients, val_patients) tuples for each fold
    """
    if len(train_val_patients) < n_splits:
        raise ValueError(f"Cannot create {n_splits} folds with only {len(train_val_patients)} patients")
    
    # Extract spike counts for stratification (use filtered counts if available)
    spike_counts = [
        patient_stats[patient_id].get('total_filtered_spikes', patient_stats[patient_id]['total_spikes'])
        for patient_id in train_val_patients
    ]
    
    # Create stratification bins
    stratification_bins = create_stratification_bins(spike_counts, n_bins=min(5, len(train_val_patients)))
    
    logger.info(f"Created {len(set(stratification_bins))} stratification bins based on spike counts")
    
    # Create stratified K-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_patients, stratification_bins)):
        train_patients = [train_val_patients[i] for i in train_idx]
        val_patients = [train_val_patients[i] for i in val_idx]
        
        splits.append((train_patients, val_patients))
        
        # Log fold statistics
        train_spikes = sum(patient_stats[p]['total_spikes'] for p in train_patients)
        val_spikes = sum(patient_stats[p]['total_spikes'] for p in val_patients)
        
        logger.info(f"Fold {fold_idx + 1}: {len(train_patients)} train patients ({train_spikes} spikes), "
                   f"{len(val_patients)} val patients ({val_spikes} spikes)")
    
    return splits


def log_split_statistics(splits: List[Tuple[List[str], List[str]]], test_patients: List[str], 
                        patient_stats: Dict[str, Dict]):
    """Log detailed statistics for each split.
    
    Args:
        splits: List of (train_patients, val_patients) tuples
        test_patients: List of test patient IDs
        patient_stats: Patient statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION SPLIT STATISTICS")
    logger.info("=" * 60)
    
    # Test set statistics
    test_spikes = sum(patient_stats[p]['total_spikes'] for p in test_patients)
    test_files = sum(patient_stats[p]['file_count'] for p in test_patients)
    logger.info(f"Test set: {len(test_patients)} patients, {test_files} files, {test_spikes} spikes")
    
    # Group statistics for test set
    test_groups = defaultdict(int)
    for patient_id in test_patients:
        test_groups[patient_stats[patient_id]['group']] += 1
    logger.info(f"Test set groups: {dict(test_groups)}")
    
    logger.info("-" * 60)
    
    # Fold statistics
    for fold_idx, (train_patients, val_patients) in enumerate(splits):
        logger.info(f"FOLD {fold_idx + 1}:")
        
        # Train statistics
        train_spikes = sum(patient_stats[p]['total_spikes'] for p in train_patients)
        train_files = sum(patient_stats[p]['file_count'] for p in train_patients)
        train_groups = defaultdict(int)
        for patient_id in train_patients:
            train_groups[patient_stats[patient_id]['group']] += 1
        
        # Validation statistics
        val_spikes = sum(patient_stats[p]['total_spikes'] for p in val_patients)
        val_files = sum(patient_stats[p]['file_count'] for p in val_patients)
        val_groups = defaultdict(int)
        for patient_id in val_patients:
            val_groups[patient_stats[patient_id]['group']] += 1
        
        logger.info(f"  Train: {len(train_patients)} patients, {train_files} files, {train_spikes} spikes")
        logger.info(f"  Train groups: {dict(train_groups)}")
        logger.info(f"  Val: {len(val_patients)} patients, {val_files} files, {val_spikes} spikes")
        logger.info(f"  Val groups: {dict(val_groups)}")
        logger.info(f"  Spike ratio (train:val): {train_spikes / max(val_spikes, 1):.2f}:1")
        
        if fold_idx < len(splits) - 1:
            logger.info("-" * 40)
    
    logger.info("=" * 60)


def save_splits(splits: List[Tuple[List[str], List[str]]], test_patients: List[str], 
               patient_stats: Dict[str, Dict], output_dir: str):
    """Save the generated splits to files.
    
    Args:
        splits: List of (train_patients, val_patients) tuples
        test_patients: List of test patient IDs
        patient_stats: Patient statistics dictionary
        output_dir: Directory to save split files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test set (same for all folds)
    test_files = []
    for patient_id in test_patients:
        test_files.extend(patient_stats[patient_id]['files'])
    
    test_split_path = os.path.join(output_dir, 'test_files.json')
    with open(test_split_path, 'w') as f:
        json.dump({
            'patient_ids': test_patients,
            'file_paths': test_files,
            'statistics': {
                'n_patients': len(test_patients),
                'n_files': len(test_files),
                'total_spikes': sum(patient_stats[p]['total_spikes'] for p in test_patients)
            }
        }, f, indent=2)
    
    logger.info(f"Saved test split to {test_split_path}")
    
    # Save each fold
    for fold_idx, (train_patients, val_patients) in enumerate(splits):
        fold_data = {
            'fold': fold_idx + 1,
            'train': {
                'patient_ids': train_patients,
                'file_paths': [f for p in train_patients for f in patient_stats[p]['files']],
                'statistics': {
                    'n_patients': len(train_patients),
                    'n_files': sum(patient_stats[p]['file_count'] for p in train_patients),
                    'total_spikes': sum(patient_stats[p]['total_spikes'] for p in train_patients)
                }
            },
            'val': {
                'patient_ids': val_patients,
                'file_paths': [f for p in val_patients for f in patient_stats[p]['files']],
                'statistics': {
                    'n_patients': len(val_patients),
                    'n_files': sum(patient_stats[p]['file_count'] for p in val_patients),
                    'total_spikes': sum(patient_stats[p]['total_spikes'] for p in val_patients)
                }
            }
        }
        
        fold_path = os.path.join(output_dir, f'fold_{fold_idx + 1}.json')
        with open(fold_path, 'w') as f:
            json.dump(fold_data, f, indent=2)
        
        logger.info(f"Saved fold {fold_idx + 1} to {fold_path}")
    
    # Save patient statistics
    stats_path = os.path.join(output_dir, 'patient_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(patient_stats, f, indent=2)
    
    logger.info(f"Saved patient statistics to {stats_path}")


def split_patients_by_group(patient_groups: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Split patients into train/val and test sets based on group.
    
    Args:
        patient_groups: Dictionary mapping patient IDs to groups
        
    Returns:
        Tuple of (train_val_patients, test_patients)
    """
    train_val_patients = []
    test_patients = []
    
    for patient_id, patient_group in patient_groups.items():
        if patient_group == 'Holdout':
            test_patients.append(patient_id)
        else:
            train_val_patients.append(patient_id)
    
    return train_val_patients, test_patients


def filter_low_spike_rate_subjects(
    patient_stats: Dict[str, Dict],
    train_val_patients: List[str],
    test_patients: List[str],
    min_spikes_per_minute: float = 0.5,
    protect_groups: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Remove subjects with suspiciously low spike rates from train/val pool.

    Subjects with 0 spikes (controls) are KEPT — they are legitimate negatives.
    Subjects with 0 < spikes_per_minute < threshold are REMOVED as unreliable.
    Test/holdout subjects and protected groups are never removed.

    Args:
        patient_stats: Patient statistics (must include 'total_filtered_spikes', 'spikes_per_minute').
        train_val_patients: List of train/val patient IDs.
        test_patients: List of test patient IDs (never filtered).
        min_spikes_per_minute: Minimum spike rate to keep a spike-positive subject.
        protect_groups: List of group names that are never filtered.

    Returns:
        Tuple of (filtered_train_val, removed_patients, stats_dict).
    """
    protect_groups = set(protect_groups or [])
    test_set = set(test_patients)

    filtered = []
    removed = []
    removed_details = []

    for patient_id in train_val_patients:
        stats = patient_stats[patient_id]
        n_spikes = stats['total_filtered_spikes']
        rate = stats['spikes_per_minute']
        group = stats['group']

        # Never filter test subjects or protected groups
        if patient_id in test_set or group in protect_groups:
            filtered.append(patient_id)
            continue

        # Keep controls (0 spikes) — they are legitimate negatives
        if n_spikes == 0:
            filtered.append(patient_id)
            continue

        # Remove subjects with low but non-zero spike rate
        if rate < min_spikes_per_minute:
            removed.append(patient_id)
            removed_details.append({
                'patient_id': patient_id,
                'group': group,
                'spikes': n_spikes,
                'spikes_per_minute': round(rate, 4),
                'duration_min': round(stats['total_duration_s'] / 60, 1),
            })
            continue

        filtered.append(patient_id)

    filter_stats = {
        'subjects_before': len(train_val_patients),
        'subjects_after': len(filtered),
        'subjects_removed': len(removed),
        'min_spikes_per_minute_threshold': min_spikes_per_minute,
        'removed_details': removed_details,
    }

    return filtered, removed, filter_stats


def generate_splits(config: Dict[str, Any], njobs: int = 1) -> Tuple[List[Tuple[List[str], List[str]]], List[str], Dict[str, Dict]]:
    """Generate stratified cross-validation splits for MEG dataset.

    Supports optional quality filtering controlled by config keys:
    - ``spike_quality_filtering``: Per-spike channel activity filtering.
    - ``subject_filtering``: Subject-level spike rate filtering.

    Args:
        config: Configuration dictionary.
        njobs: Number of parallel jobs to run.
    """
    filtering_stats = FilteringStatistics()

    # Find all .ds files
    logger.info("Discovering MEG data files...")
    ds_files = []
    patient_groups = {}
    skip_patterns = [re.compile(pattern) for pattern in config['skip_files']]
    annotation_rules = compile_annotation_patterns(config['annotation_rules'])

    for root_dir in config['root_dirs']:
        logger.info(f"Searching for .ds files in {root_dir}")
        root_files = find_ds_files(root_dir, patient_groups, skip_patterns)
        ds_files.extend(root_files)

    if not ds_files:
        error_msg = f"No .ds files found in {config['root_dirs']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Found {len(ds_files)} .ds files across {len(patient_groups)} patients")

    # Split patients into train/val and test groups
    logger.info("Separating test set from train/val patients...")
    train_val_patients, test_patients = split_patients_by_group(patient_groups)

    logger.info(f"Train/Val pool: {len(train_val_patients)} patients")
    logger.info(f"Test set: {len(test_patients)} patients")

    # Generate patient spike statistics (with optional per-spike quality filtering)
    spike_quality_config = config.get('spike_quality_filtering')
    preprocessing_config = {
        'sampling_rate': config['sampling_rate'],
        'l_freq': config['l_freq'],
        'h_freq': config['h_freq'],
        'notch_freq': config['notch_freq'],
    }
    patient_stats = generate_patient_spike_statistics(
        ds_files, patient_groups, annotation_rules,
        spike_quality_config, preprocessing_config,
        njobs=njobs
    )

    # Stage 1: Raw counts
    total_raw_spikes = sum(s['total_spikes'] for s in patient_stats.values())
    total_filtered_spikes = sum(s['total_filtered_spikes'] for s in patient_stats.values())
    total_duration_min = sum(s['total_duration_s'] for s in patient_stats.values()) / 60.0
    filtering_stats.add_stage('raw_counts', {
        'total_patients': len(patient_stats),
        'total_recordings': len(ds_files),
        'total_duration_minutes': round(total_duration_min, 1),
        'total_raw_spikes': total_raw_spikes,
    })

    # Stage 2: Per-spike quality filtering stats
    sq = spike_quality_config or {}
    if sq.get('enabled', False):
        filtering_stats.add_stage('spike_quality_filter', {
            'spikes_before': total_raw_spikes,
            'spikes_after': total_filtered_spikes,
            'spikes_removed': total_raw_spikes - total_filtered_spikes,
            'rejection_rate': round(
                (total_raw_spikes - total_filtered_spikes) / max(total_raw_spikes, 1), 4
            ),
            'threshold_zscore': sq.get('threshold_zscore', 2.5),
            'min_active_channels': sq.get('min_active_channels', 5),
        })

    # Stage 3: Subject-level filtering
    subject_filter_config = config.get('subject_filtering', {})
    if subject_filter_config.get('enabled', False):
        logger.info("Applying subject-level spike rate filtering...")
        train_val_patients, removed_patients, subj_filter_stats = filter_low_spike_rate_subjects(
            patient_stats,
            train_val_patients,
            test_patients,
            min_spikes_per_minute=subject_filter_config.get('min_spikes_per_minute', 0.5),
            protect_groups=subject_filter_config.get('protect_groups'),
        )
        filtering_stats.add_stage('subject_filter', subj_filter_stats)

        if removed_patients:
            logger.info(
                f"Removed {len(removed_patients)} subjects with low spike rates: "
                f"{removed_patients}"
            )
        logger.info(f"Train/Val pool after filtering: {len(train_val_patients)} patients")

    # Generate stratified K-fold splits (use filtered spike counts for stratification)
    logger.info(f"Generating {config['n_splits']} stratified folds...")
    splits = generate_stratified_splits(
        train_val_patients, patient_stats, config['n_splits'], config['random_state']
    )

    # Stage 4: Final split stats
    filtering_stats.add_stage('final_splits', {
        'n_folds': config['n_splits'],
        'train_val_patients': len(train_val_patients),
        'test_patients': len(test_patients),
        'total_filtered_spikes': total_filtered_spikes,
    })

    # Log detailed statistics
    log_split_statistics(splits, test_patients, patient_stats)
    filtering_stats.log_summary(logger)

    # Save splits and filtering statistics
    if 'splits_output_dir' in config:
        save_splits(splits, test_patients, patient_stats, config['splits_output_dir'])
        stats_path = os.path.join(config['splits_output_dir'], 'filtering_statistics.json')
        filtering_stats.save(stats_path)

    logger.info("Split generation completed successfully!")
    return splits, test_patients, patient_stats


def main():
    """Main function to generate cross-validation splits."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate cross-validation splits")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--njobs', type=int, required=True, help='Number of parallel jobs to run')
    args = parser.parse_args()

    config = load_config(args.config, validate=False)

    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', 'generate_splits.log')
    setup_logging(log_level, log_file)

    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Logging to {log_file} with level {log_level}")

    mne.set_log_level(verbose=logging.ERROR)
    generate_splits(config, njobs=args.njobs)

if __name__ == "__main__":
    main()