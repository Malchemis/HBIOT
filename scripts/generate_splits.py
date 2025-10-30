"""Generate stratified cross-validation splits for MEG spike detection datasets.

This script creates stratified K-fold cross-validation splits ensuring balanced distribution
of spike counts across folds. It separates a test set (holdout patients) from the train/val pool,
then generates stratified folds based on spike count distributions.
"""
import json
import os
import re
import pickle
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import logging
import argparse

import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from pipeline.data.preprocessing.annotation import get_spike_annotations, compile_annotation_patterns
from pipeline.data.preprocessing.file_manager import find_ds_files
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

def load_meg_data_and_count_spikes(file_path: str, patient_group: str, annotation_rules: Dict[str, Any]) -> int:
    """Load MEG data and count valid spike annotations.
    
    Args:
        file_path: Path to the .ds file
        patient_group: Patient group for annotation rules
        annotation_rules: Dictionary containing annotation rules for each group

    Returns:
        Number of valid spike annotations in the file
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
            
        # Extract annotations
        annotations = raw.annotations
        if annotations is None or len(annotations) == 0:
            return 0
            
        # Get spike annotations using the annotation processor
        spike_onsets = get_spike_annotations(annotations, patient_group, annotation_rules)
        return len(spike_onsets)
        
    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")
        return 0


def generate_patient_spike_statistics(ds_files: List[Dict], patient_groups: Dict[str, str], annotation_rules: Dict[str, Any]) -> Dict[str, Dict]:
    """Generate spike statistics for each patient.
    
    Args:
        ds_files: List of file dictionaries with metadata
        patient_groups: Dictionary mapping patient IDs to groups
        annotation_rules: Dictionary containing annotation rules for each group

    Returns:
        Dictionary with patient statistics
    """
    logger.info("Analyzing spike statistics for all patients...")
    
    # Group files by patient
    patient_files = defaultdict(list)
    for file_info in ds_files:
        patient_id = file_info['patient_id']
        patient_files[patient_id].append(file_info)
    
    patient_stats = {}
    
    for patient_id, files in tqdm(patient_files.items(), desc="Processing patients"):
        patient_group = patient_groups[patient_id]
        total_spikes = 0
        file_count = len(files)
        
        for file_info in files:
            file_path = file_info['path']
            spike_count = load_meg_data_and_count_spikes(
                file_path, patient_group, annotation_rules
            )
            total_spikes += spike_count
        
        patient_stats[patient_id] = {
            'group': patient_group,
            'total_spikes': total_spikes,
            'file_count': file_count,
            'files': [f['path'] for f in files]
        }
        
        logger.debug(f"Patient {patient_id} ({patient_group}): {total_spikes} spikes across {file_count} files")
    
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
    
    # Extract spike counts for stratification
    spike_counts = [patient_stats[patient_id]['total_spikes'] for patient_id in train_val_patients]
    
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


def generate_splits(config: Dict[str, Any]):
    """Generate stratified cross-validation splits for MEG dataset.
    
    Args:
        config: Configuration dictionary
    """   
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
    
    # Generate patient spike statistics
    patient_stats = generate_patient_spike_statistics(
        ds_files, patient_groups, annotation_rules
    )
    
    # Generate stratified K-fold splits
    logger.info(f"Generating {config['n_splits']} stratified folds...")
    splits = generate_stratified_splits(
        train_val_patients, patient_stats, config['n_splits'], config['random_state']
    )

    # Log detailed statistics
    log_split_statistics(splits, test_patients, patient_stats)

    # Save splits to files
    if 'splits_output_dir' in config:
        save_splits(splits, test_patients, patient_stats, config['splits_output_dir'])

    logger.info("Split generation completed successfully!")
    return splits, test_patients, patient_stats


def main():
    """Main function to generate cross-validation splits."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate cross-validation splits")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config, validate=False)

    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', 'generate_splits.log')
    setup_logging(log_level, log_file)

    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Logging to {log_file} with level {log_level}")

    mne.set_log_level(verbose=logging.ERROR)
    generate_splits(config)

if __name__ == "__main__":
    main()