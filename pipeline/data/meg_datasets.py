"""MEG Spike Detection Datasets

This module provides dataset classes for MEG spike detection with unified metadata naming.

Unified Metadata Convention:
    All datasets use consistent field names for compatibility:

    Chunk-level fields:
    - chunk_onset_sample: Start sample position in the original recording
    - chunk_offset_sample: End sample position in the original recording
    - chunk_duration_samples: Duration of chunk in samples
    - chunk_idx: Chunk index within the file (0-based)

    Window-level fields (for traceability):
    - start_window_idx: Index of first window in this chunk (relative to file)
    - end_window_idx: Index after last window in this chunk (relative to file)
    - n_windows: Number of windows in this chunk
    - window_duration_s: Duration of each window in seconds
    - window_duration_samples: Duration of each window in samples

    Spike-related fields:
    - spike_positions_in_chunk: List of spike sample positions (chunk-relative)
    - n_spikes_in_chunk: Number of actual spikes in this chunk

    File identification:
    - file_name: Full path to the source MEG file
    - patient_id: Patient identifier
    - original_filename: Original filename without path
    - group: Patient group

    Processing metadata:
    - preprocessing_config: Dictionary with preprocessing parameters
    - is_test_set: Boolean indicating if this is test data
    - extraction_mode: How the chunk was extracted ('fixed', 'random', 'sequential')

    Dataset-specific fields:
    - global_chunk_idx: (OnlineWindowDataset only) Dataset-wide chunk index
    - window_times: (PredictDataset only) Detailed per-window timing information

Datasets:
    - PreloadedDataset: Preloads all chunks into memory for fast access
    - OnlineWindowDataset: Loads recordings once, extracts chunks on-the-fly
    - PredictDataset: For inference on new MEG files
"""
import json
import logging
import random
import traceback
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import mne
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

mne.set_log_level(verbose=logging.ERROR)
torch.multiprocessing.set_sharing_strategy('file_system')

from pipeline.data.preprocessing.file_manager import get_patient_group
from pipeline.data.preprocessing.segmentation import (
    create_chunks,
    extract_random_chunk,
)
from pipeline.data.preprocessing.signal_processing import load_and_process_meg_data, augment_data, find_gfp_peak_in_window
from pipeline.data.preprocessing.annotation import compile_annotation_patterns, get_spike_annotations
from pipeline.data.preprocessing.cache_recordings import (
    get_cache_path,
    preprocess_recording,
    save_preprocessed_recording,
    load_preprocessed_recording,
)


class PreloadedDataset(torch.utils.data.Dataset):
    """Dataset that preloads all chunks into memory for fast access.

    This dataset processes all MEG files during initialization, extracting and storing
    all chunks in memory. Best for datasets that fit in RAM.

    Returns:
        Tuple of (data, labels, metadata) where metadata follows the unified convention
        with fields: chunk_onset_sample, global_chunk_idx, spike_positions_in_chunk, etc.
    """

    def __init__(self, json_data: Dict[str, Any], config: Dict[str, Any], good_channels: List[str], n_workers: Optional[int] = None, is_test: bool = True):
        self.json_data = json_data
        self.config = config
        self.file_paths = json_data['file_paths']
        self.statistics = json_data.get('statistics', {})
        self.is_test = is_test
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Statistics: {self.statistics}")
        
        # Extract parameters
        self.sampling_rate = config['sampling_rate']
        self.n_windows = config['n_windows']
        self.window_duration_s = config['window_duration_s']
        self.window_duration_samples = int(self.window_duration_s * self.sampling_rate)
        self.window_overlap = config.get('window_overlap', 0.0)
        self.spike_overlap_threshold = config.get('spike_overlap_threshold', 0.9)
        self.estimated_spike_duration_s = config.get('estimated_spike_duration_s', 0.1)
        self.first_half_spike_duration = config.get('first_half_spike_duration', 0.05)
        self.second_half_spike_duration = config.get('second_half_spike_duration', 0.05)
        
        self.compiled_patterns = compile_annotation_patterns(config.get('annotation_rules', {}))
        
        # Channel information for masking
        self.good_channels = good_channels
        
        # Store samples in cache
        self.samples = []
        self.samples_weight = []  # Weight based on 'n_spikes_in_chunk'

        # Limit workers to prevent resource contention and hanging
        max_workers = cpu_count() // 4
        self.n_workers = n_workers or max_workers
        self.preloaded = False

    def preload(self):
        """Preload all files and create chunks using multiprocessing with progress bar."""
        if self.preloaded:
            self.logger.info("Data already preloaded, skipping")
            return
            
        self.logger.info(f"Preloading {len(self.file_paths)} files using {self.n_workers} workers")                
        process_func = partial(self._process_full_file, 
                               config=self.config,
                               good_channels=self.good_channels,
                               compiled_patterns=self.compiled_patterns,
                               is_test=self.is_test)
        
        try:
            results = Parallel(n_jobs=self.n_workers, backend='threading')(
                delayed(process_func)(file_path) for file_path in tqdm(self.file_paths, desc="Processing files", unit="file", position=0, leave=True)
            )
            
            if results:
                for samples in results:
                    if samples:
                        self.samples.extend(samples)
                    
        except Exception as e:
            self.logger.error(f"Joblib parallel processing failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.info("Falling back to sequential processing")
        
        if not self.is_test:
            random.shuffle(self.samples)

        for sample in self.samples:
            self.samples_weight.append(sample[2]['n_spikes_in_chunk'])

        self.preloaded = True
        self.logger.info(f"Preloading completed: {len(self.samples)} samples loaded")

    @staticmethod
    def _process_full_file(file_path: str, config: Dict, good_channels: List[str], compiled_patterns: Dict, is_test: bool) -> List[Tuple]:
        """Process a single file and return all chunks.

        Args:
            file_path: Path to MEG file.
            config: Dataset configuration.
            good_channels: List of channel names.
            compiled_patterns: Compiled annotation patterns.
            is_test: Whether this is test data.

        Returns:
            List of (data, labels, metadata) tuples.
        """
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug(f"Processing file: {file_path}")

            raw, meg_data, channel_info = load_and_process_meg_data(
                file_path, config, good_channels, close_raw=False
            )

            spike_onsets = []
            group = get_patient_group(file_path)
            if len(raw.annotations) > 0:
                spike_onsets = get_spike_annotations(raw.annotations, group, compiled_patterns)
            raw.close()
            spike_samples = [int(onset * config['sampling_rate']) for onset in spike_onsets]

            if meg_data is None:
                logger.warning(f"No MEG data loaded from {file_path}")
                return []

            chunks, labels, start_pos, spike_pos = create_chunks(
                meg_data, spike_samples, config
            )

            common_meta = {
                'file_name': file_path,
                'patient_id': file_path.split('/')[-2],
                'original_filename': file_path.split('/')[-1],
                'group': group,
                'preprocessing_config': {
                    'sampling_rate': config['sampling_rate'],
                    'l_freq': config.get('l_freq', 0.5),
                    'h_freq': config.get('h_freq', 95.0),
                    'notch_freq': config.get('notch_freq', 50.0),
                    'normalization': config.get('normalization', {}),
                },
                'window_duration_s': config['window_duration_s'],
                'window_duration_samples': int(config['window_duration_s'] * config['sampling_rate']),
                'is_test_set': is_test,
            }

            window_duration_samples = int(config['window_duration_s'] * config['sampling_rate'])
            window_overlap = config.get('window_overlap', 0.0)
            window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
            n_windows = config['n_windows']

            samples = []
            for i, (chunk, label, start, sp) in enumerate(zip(chunks, labels, start_pos, spike_pos)):
                chunk_duration_samples = len(chunk) * window_step + (window_duration_samples - window_step)
                chunk_offset_sample = int(start + chunk_duration_samples)

                start_window_idx = i * n_windows
                end_window_idx = start_window_idx + len(chunk)

                chunk_meta = {
                    'chunk_onset_sample': int(start),
                    'chunk_offset_sample': chunk_offset_sample,
                    'chunk_duration_samples': chunk_duration_samples,
                    'chunk_idx': i,
                    'start_window_idx': start_window_idx,
                    'end_window_idx': end_window_idx,
                    'n_windows': len(chunk),
                    'spike_positions_in_chunk': sp,
                    'n_spikes_in_chunk': sum(label),
                    'extraction_mode': 'fixed',
                }
                full_meta = {**common_meta, **chunk_meta, **channel_info}
                samples.append((
                    torch.tensor(chunk, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32),
                    full_meta
                ))

            return samples
            
        except Exception as e:
            logging.error(f"Error processing test file {file_path}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]:
        if not self.preloaded:
            raise RuntimeError("Dataset not preloaded. Call preload() before accessing samples.")
        
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")

        if self.is_test:
            return self.samples[idx]
        else:
            data = augment_data(self.samples[idx][0].numpy(), self.config.get('noise_level', 0.01))
            return torch.tensor(data, dtype=torch.float32), self.samples[idx][1], self.samples[idx][2]

    @classmethod
    def from_test_file(cls, file_path: str, config: Dict[str, Any], good_channels: List[str], n_workers: Optional[int] = None) -> 'PreloadedDataset':
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return cls(json_data, config, good_channels=good_channels, n_workers=n_workers, is_test=True)

    @classmethod
    def from_split_file(cls, split_file_path: str, config: Dict[str, Any], good_channels: List[str],
                        split_type: str = "train", n_workers: Optional[int] = None) -> 'PreloadedDataset':
        with open(split_file_path, 'r') as f:
            split_data = json.load(f)
        
        if split_type not in split_data:
            raise ValueError(f"Split type '{split_type}' not found in {split_file_path}")

        return cls(split_data[split_type], config, good_channels=good_channels, n_workers=n_workers, is_test=False)


class PredictDataset(torch.utils.data.Dataset):
    """Dataset for prediction using sequential chunk extraction.

    Uses the same sequential extraction pattern as OnlineWindowDataset in test mode
    for consistency with training/validation data processing.

    Returns:
        Tuple of (chunk_data, metadata) with unified metadata convention including
        chunk_onset_sample, chunk_idx, window_times, etc.
    """

    def __init__(
        self,
        file_path: str,
        dataset_config: Dict[str, Any],
        n_channels: int = 275,
    ):
        """Initialize prediction dataset with sequential chunk extraction.

        Args:
            file_path: Path to the MEG file (.fif or .ds).
            dataset_config: Configuration for data processing.
            n_channels: Number of MEG channels (default: 275) for consistent input size.
        """
        self.file_path = file_path
        self.dataset_config = dataset_config
        self.n_channels = n_channels

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PredictDataset for {file_path}")

        # Load and preprocess the recording once
        self.meg_data = None
        self.channel_info = None
        self.sampling_rate = None
        self.n_chunks = 0

        self._load_recording()

    def _load_recording(self):
        """Load and preprocess the MEG recording once."""
        try:
            raw, self.meg_data, self.channel_info = load_and_process_meg_data(
                self.file_path,
                self.dataset_config,
                good_channels=None,
                n_channels=self.n_channels,
            )

            self.sampling_rate = raw.info['sfreq']
            raw.close()

            from .preprocessing.segmentation import create_windows
            self.all_windows = create_windows(
                self.meg_data,
                self.sampling_rate,
                self.dataset_config['window_duration_s'],
                self.dataset_config.get('window_overlap', 0.0),
            )

            num_context_windows = self.dataset_config['n_windows']
            total_windows = len(self.all_windows)
            self.n_chunks = (total_windows + num_context_windows - 1) // num_context_windows

            self.logger.info(f"Loaded recording: {self.meg_data.shape[1]} samples, "
                           f"{total_windows} windows, {self.n_chunks} chunks")

        except Exception as e:
            self.logger.error(f"Error loading file {self.file_path}: {e}")
            raise

    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract a chunk sequentially for prediction.

        Args:
            idx: Chunk index (0-based).

        Returns:
            Tuple of (chunk_data, metadata) with chunk_data as tensor of shape
            (n_windows, n_channels, window_samples) and metadata dictionary.
        """
        num_context_windows = self.dataset_config['n_windows']
        window_duration_samples = int(self.dataset_config['window_duration_s'] * self.sampling_rate)
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        start_window_idx = idx * num_context_windows
        end_window_idx = min(start_window_idx + num_context_windows, len(self.all_windows))

        windows = self.all_windows[start_window_idx:end_window_idx]

        chunk_onset_sample = start_window_idx * window_step

        window_times = []
        for local_idx, global_idx in enumerate(range(start_window_idx, end_window_idx)):
            window_start = global_idx * window_step
            window_end = window_start + window_duration_samples
            window_center = window_start + window_duration_samples // 2
            
            assert self.sampling_rate is not None, "Sampling rate not set"
            assert self.meg_data is not None, "MEG data not loaded"
            peak_sample, peak_time = find_gfp_peak_in_window(
                self.meg_data, window_start, window_end, self.sampling_rate
            )

            window_times.append({
                'start_sample': int(window_start),
                'end_sample': int(window_end),
                'center_sample': int(window_center),
                'peak_sample': int(peak_sample),
                'start_time': float(window_start / self.sampling_rate),
                'end_time': float(window_end / self.sampling_rate),
                'center_time': float(window_center / self.sampling_rate),
                'peak_time': float(peak_time),
                'window_idx_in_chunk': local_idx,
                'global_window_idx': global_idx,
            })

        metadata = {
            'chunk_onset_sample': chunk_onset_sample,
            'chunk_offset_sample': chunk_onset_sample + len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_duration_samples': len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_idx': idx,
            'start_window_idx': start_window_idx,
            'end_window_idx': end_window_idx,
            'n_windows': len(windows),
            'window_times': window_times,
            'window_duration_s': self.dataset_config['window_duration_s'],
            'window_duration_samples': window_duration_samples,
            'file_name': self.file_path,
            'patient_id': self.file_path.split('/')[-2] if '/' in self.file_path else 'unknown',
            'channel_mask': self.channel_info.get('channel_mask', None) if self.channel_info else None,
            'selected_channels': self.channel_info.get('selected_channels', []) if self.channel_info else [],
            'n_selected_channels': len(self.channel_info.get('selected_channels', [])) if self.channel_info else 0,
            'sampling_rate': self.sampling_rate,
            'is_test_set': False,
            'extraction_mode': 'sequential',
        }

        return torch.tensor(windows, dtype=torch.float32), metadata


class OnlineWindowDataset(torch.utils.data.Dataset):
    """Dataset that loads recordings once and extracts chunks on-the-fly.

    Loads preprocessed MEG recordings into memory and extracts chunks dynamically during
    training for temporal diversity. Supports both random (training) and sequential (test)
    extraction modes.

    Returns:
        Tuple of (data, labels, metadata) with unified metadata convention including
        chunk_onset_sample, global_chunk_idx, spike_positions_in_chunk, etc.
    """

    def __init__(
        self,
        json_data: Dict[str, Any],
        config: Dict[str, Any],
        good_channels: List[str],
        preprocessed_dir: str,
        samples_per_recording: int = 45,
        is_test: bool = False,
        force_preprocess: bool = False,
    ):
        """Initialize online windowing dataset.

        Args:
            json_data: Split data with file_paths and statistics.
            config: Dataset configuration.
            good_channels: Ordered list of reference channel names.
            preprocessed_dir: Directory for cached preprocessed files.
            samples_per_recording: Number of chunks to sample per recording per epoch.
            is_test: If True, use deterministic sequential chunk extraction.
            force_preprocess: If True, reprocess even if cached.
        """
        self.config = config
        self.good_channels = good_channels
        self.preprocessed_dir = preprocessed_dir
        self.samples_per_recording = samples_per_recording
        self.is_test = is_test
        self.force_preprocess = force_preprocess

        self.file_paths = json_data['file_paths']
        self.statistics = json_data.get('statistics', {})

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing OnlineWindowDataset with {len(self.file_paths)} recordings")
        self.logger.info(f"Samples per recording: {samples_per_recording}")

        self.compiled_patterns = compile_annotation_patterns(config.get('annotation_rules', {}))

        self.recordings = []
        self.all_windows_per_recording = []
        self.chunks_per_recording = []
        self.cumulative_chunks = []

        self._load_recordings()
        self._build_index_map()

    def _load_recordings(self):
        """Load or preprocess all recordings into memory."""
        self.logger.info(f"Loading {len(self.file_paths)} recordings into memory")

        for file_path in tqdm(self.file_paths, desc="Loading recordings", unit="file"):
            cache_path = get_cache_path(file_path, self.preprocessed_dir, self.config)

            if not cache_path.exists() or self.force_preprocess:
                self.logger.debug(f"Preprocessing {file_path}")
                try:
                    meg_data, spike_samples, metadata, channel_info = preprocess_recording(
                        file_path, self.config, self.good_channels, self.compiled_patterns
                    )
                    save_preprocessed_recording(
                        cache_path, meg_data, spike_samples, metadata, channel_info
                    )
                except Exception as e:
                    self.logger.error(f"Error preprocessing {file_path}: {e}")
                    continue
            else:
                try:
                    meg_data, spike_samples, metadata, channel_info = load_preprocessed_recording(cache_path)
                except Exception as e:
                    self.logger.error(f"Error loading cached file {cache_path}: {e}")
                    continue

            self.recordings.append((meg_data, spike_samples, metadata, channel_info))

        self.logger.info(f"Loaded {len(self.recordings)} recordings into memory")

    def _build_index_map(self):
        """Build index mapping from global index to (recording_idx, local_chunk_idx).

        Training mode uses samples_per_recording with random sampling.
        Test/validation mode uses all possible chunks for exhaustive coverage.
        """
        if self.is_test:
            from .preprocessing.segmentation import create_windows

            self.chunks_per_recording = []
            num_context_windows = self.config['n_windows']

            for meg_data, _, _, _ in self.recordings:
                all_windows = create_windows(
                    meg_data,
                    self.config['sampling_rate'],
                    self.config['window_duration_s'],
                    self.config.get('window_overlap', 0.0),
                )
                self.all_windows_per_recording.append(all_windows)

                total_windows = len(all_windows)
                n_chunks = (total_windows + num_context_windows - 1) // num_context_windows
                self.chunks_per_recording.append(n_chunks)

            self.cumulative_chunks = np.cumsum([0] + self.chunks_per_recording)

            total_chunks = sum(self.chunks_per_recording)
            total_windows = sum(len(w) for w in self.all_windows_per_recording)
            self.logger.info(f"Test dataset size: {total_chunks} chunks ({total_windows} windows) "
                           f"from {len(self.recordings)} recordings (exhaustive sequential extraction)")
        else:
            self.chunks_per_recording = [self.samples_per_recording for _ in self.recordings]
            self.cumulative_chunks = np.cumsum([0] + self.chunks_per_recording)

            self.logger.info(f"Train dataset size: {len(self)} chunks ({len(self.recordings)} recordings x "
                           f"{self.samples_per_recording} random samples)")

    def __len__(self) -> int:
        """Return total number of chunks."""
        return int(self.cumulative_chunks[-1]) if len(self.cumulative_chunks) > 0 else 0

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Extract a chunk from a recording.

        Args:
            idx: Global sample index.

        Returns:
            Tuple of (chunk_data, labels, metadata).
        """
        rec_idx = int(np.searchsorted(self.cumulative_chunks, idx, side='right') - 1)
        local_chunk_idx = idx - self.cumulative_chunks[rec_idx]

        meg_data, spike_samples, metadata, channel_info = self.recordings[rec_idx]

        if self.config.get('refine_spike_positions', False):
            new_spike_samples = []
            n_samples_total = meg_data.shape[1]
            for spike_sample in spike_samples:
                peak_sample, peak_time = find_gfp_peak_in_window(
                    meg_data,
                    max(0, spike_sample - int(self.config['first_half_spike_duration'] * self.config['sampling_rate'])),
                    min(n_samples_total, spike_sample + int(self.config['second_half_spike_duration'] * self.config['sampling_rate'])),
                    self.config['sampling_rate']
                )
                new_spike_samples.append(peak_sample)
            spike_samples = sorted(set(new_spike_samples)) # Remove duplicates if peaks overlap

        if self.is_test:
            from .preprocessing.segmentation import calculate_window_labels_from_spikes

            all_windows = self.all_windows_per_recording[rec_idx]
            num_context_windows = self.config['n_windows']

            # Calculate window indices for this chunk
            start_window_idx = local_chunk_idx * num_context_windows
            end_window_idx = min(start_window_idx + num_context_windows, len(all_windows))

            # Get windows for this chunk
            windows = all_windows[start_window_idx:end_window_idx]

            # Build chunk metadata (need this first to adjust spike positions)
            window_duration_samples = int(self.config['window_duration_s'] * self.config['sampling_rate'])
            window_overlap = self.config.get('window_overlap', 0.0)
            window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

            chunk_onset_sample = start_window_idx * window_step
            chunk_offset_sample = chunk_onset_sample + len(windows) * window_step + (window_duration_samples - window_step)

            # Adjust spike positions to be relative to chunk onset
            chunk_spike_samples = [s - chunk_onset_sample for s in spike_samples
                                  if chunk_onset_sample <= s < chunk_offset_sample]

            # Calculate labels for these windows (with chunk-relative spike positions)
            labels = calculate_window_labels_from_spikes(
                windows,
                chunk_spike_samples,
                self.config,
            )

            chunk_meta = {
                # Chunk position in recording
                'chunk_onset_sample': chunk_onset_sample,
                'chunk_offset_sample': chunk_offset_sample,
                'chunk_duration_samples': chunk_offset_sample - chunk_onset_sample,
                'chunk_idx': local_chunk_idx,

                # Window-level traceability
                'start_window_idx': start_window_idx,
                'end_window_idx': end_window_idx,
                'n_windows': len(windows),

                # Spike information
                'n_spikes_in_chunk': len([s for s in spike_samples if chunk_onset_sample <= s < chunk_offset_sample]),
                'spike_positions_in_chunk': [int(s - chunk_onset_sample) for s in spike_samples
                                            if chunk_onset_sample <= s < chunk_offset_sample],

                # Extraction mode
                'extraction_mode': 'sequential',
            }
        else:
            # Training mode: Random extraction for temporal diversity
            # Different position each time this recording is sampled
            windows, labels, chunk_meta = extract_random_chunk(
                meg_data,
                spike_samples.tolist(),
                self.config,
                seed=None  # Fully random
            )

            # Apply augmentation (training only)
            windows = augment_data(windows, self.config.get('noise_level', 0.01))

        # Merge all metadata with unified naming convention
        full_metadata = {
            **metadata,
            **chunk_meta,
            **channel_info,
            'recording_idx': rec_idx,              # Index of recording in dataset
            'local_chunk_idx': local_chunk_idx,    # Index of chunk within this recording
            'global_chunk_idx': idx,               # Global index across entire dataset
            'is_test_set': self.is_test,
            'extraction_mode': 'sequential' if self.is_test else 'random',
        }

        return (
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            full_metadata
        )

    @classmethod
    def from_test_file(
        cls,
        file_path: str,
        config: Dict[str, Any],
        good_channels: List[str],
        preprocessed_dir: str,
        samples_per_recording: int = 10,
        force_preprocess: bool = False,
    ) -> 'OnlineWindowDataset':
        """Create dataset from test file JSON.

        Args:
            file_path: Path to test_files.json
            config: Dataset configuration
            good_channels: Ordered list of reference channel names
            preprocessed_dir: Directory for cached files
            samples_per_recording: Chunks per recording
            force_preprocess: Force reprocessing

        Returns:
            OnlineWindowDataset instance
        """
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        return cls(
            json_data,
            config,
            good_channels,
            preprocessed_dir,
            samples_per_recording=samples_per_recording,
            is_test=True,
            force_preprocess=force_preprocess,
        )

    @classmethod
    def from_split_file(
        cls,
        split_file_path: str,
        config: Dict[str, Any],
        good_channels: List[str],
        preprocessed_dir: str,
        split_type: str = "train",
        samples_per_recording: int = 10,
        force_preprocess: bool = False,
    ) -> 'OnlineWindowDataset':
        """Create dataset from fold split file JSON.

        Args:
            split_file_path: Path to fold_X.json
            config: Dataset configuration
            good_channels: Ordered list of reference channel names
            preprocessed_dir: Directory for cached files
            split_type: 'train' or 'val'
            samples_per_recording: Chunks per recording
            force_preprocess: Force reprocessing

        Returns:
            OnlineWindowDataset instance
        """
        with open(split_file_path, 'r') as f:
            split_data = json.load(f)

        if split_type not in split_data:
            raise ValueError(f"Split type '{split_type}' not found in {split_file_path}")

        is_test = (split_type == 'val')  # Val uses deterministic extraction

        return cls(
            split_data[split_type],
            config,
            good_channels,
            preprocessed_dir,
            samples_per_recording=samples_per_recording,
            is_test=is_test,
            force_preprocess=force_preprocess,
        )
        

class WeightedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    A memory-efficient BatchSampler that creates batches with a similar total weight
    by yielding them one at a time (generator-based) instead of pre-computing.
    
    This avoids storing all batch indices in memory, making it suitable for
    very large datasets.
    """
    def __init__(self, weights: Union[List[float], torch.Tensor], batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(self.weights)
        
        # Calculate the target batch weight.
        self.target_batch_weight = self.weights.mean().item() * self.batch_size
        # Handle the edge case of all-zero weights.
        if self.target_batch_weight == 0:
            # Fallback to a non-zero value to avoid division by zero.
            # Here, we can simply treat it as a standard sampler.
            self.target_batch_weight = self.batch_size
            self.weights = torch.ones_like(self.weights, dtype=torch.double)
            print("Warning: All weights are zero. Falling back to standard batching.")

        # Calculate the estimated length for progress bars and schedulers.
        if self.drop_last:
            self.estimated_len = self.num_samples // self.batch_size
        else:
            self.estimated_len = (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yields batches one by one without storing them all in memory.
        """
        # Generate a new random permutation for each epoch.
        indices = torch.randperm(self.num_samples).tolist()
        
        current_batch = []
        current_batch_weight = 0.0
        
        for idx in indices:
            current_batch.append(idx)
            current_batch_weight += self.weights[idx].item()
            
            # When the batch weight target is met or exceeded, yield the batch.
            if current_batch_weight >= self.target_batch_weight:
                yield current_batch
                # Reset for the next batch
                current_batch = []
                current_batch_weight = 0.0
        
        # Handle the last batch if drop_last is False and there are remaining samples.
        if len(current_batch) > 0 and not self.drop_last:
            yield current_batch

    def __len__(self) -> int:
        """
        Returns the *estimated* number of batches.
        This is crucial for DataLoader and progress bars.
        """
        return self.estimated_len
