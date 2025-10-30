"""
MEG Data Preprocessing Pipeline

This module provides a unified interface for all MEG data preprocessing operations.
It contains reusable functions for signal processing, annotation handling, data segmentation,
and configuration management.

Main components:
- signal_processing: Signal processing functions (filtering, normalization)
- annotation: Spike annotation extraction and processing
- segmentation: Data chunking and segmentation
- config: Configuration management utilities
- file_manager: File I/O operations

Example usage:
    from pipeline.data.preprocessing import process_file, create_chunks, generate_splits
    
    # Process a single MEG file
    file_path, meg_data, spike_samples, group = process_file(args)
    
    # Create chunks from processed data
    chunks, labels, positions, spike_pos = create_chunks(meg_data, spike_samples, config)
"""

# Import main preprocessing functions
from .signal_processing import (
    load_and_process_meg_data,
    normalize_data,
    apply_median_filter,
    augment_data
)

from .annotation import (
    compile_annotation_patterns,
    get_spike_annotations
)

from .segmentation import (
    create_chunks,
    create_windows
)

from .file_manager import (
    find_ds_files,
    get_patient_group,
    should_skip_file,
    save_chunks
)

__all__ = [
    # Signal processing
    'load_and_process_meg_data',
    'normalize_data',
    'apply_median_filter',
    'augment_data',
    
    # Annotation processing
    'compile_annotation_patterns',
    'get_spike_annotations',
    
    # Data segmentation
    'create_chunks',
    'create_windows',
    
    # File management
    'find_ds_files',
    'get_patient_group',
    'should_skip_file',
    'save_chunks',
]