"""Configuration management for MEG preprocessing pipeline."""

import argparse
import json
import logging
import os
from typing import Dict, Any, Optional

import yaml


class ConfigManager:
    """Manages configuration for MEG preprocessing pipeline.

    This class handles loading configuration from files and command-line arguments,
    validating parameters, and providing access to configuration settings.

    Attributes:
        config: Dictionary containing configuration parameters.
        logger: Logger instance for this class.
    """

    DEFAULT_CONFIG = {
        # Entry point arguments
        'save_as_pickle': False,  # toggle saving with pickle.dump vs torch.save

        # Logging configuration
        'log_level': 'INFO',
        # Not forced to specify log file, in that case will just log to the console
        'log_file': None,
        
        # Performance configuration
        'n_jobs': 1,

        # Patient split configuration
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        'random_state': 42,
        
        # Preprocessing configuration
        'n_windows': 100,
        'window_duration_s': 0.2,  # Reduced from 400ms to 200ms
        'window_overlap': 0.5,     # NEW: Overlap between consecutive windows within sequences
        'overlap': 0.0,             # Reduced chunk overlap to prevent redundancy 
        'sampling_rate': 400,       # Doubled from 200Hz to 400Hz to maintain same samples per window
        'spike_overlap_threshold': 0.8,  # Increased threshold for well-centered spikes
        'l_freq': 1.0,
        'h_freq': 70.0,
        'notch_freq': 50.0,
        # Normalization methods are 'percentile', 'zscore', 'minmax', 'none'. Any other value will be considered 'none'.
        'normalization': {
            'method': 'percentile',
            'percentile': 95,
            'per_channel': True,
        },
        
        # Dataset processing configuration
        'min_spikes_per_min': 10,
        'process_no_spike_files': {
            'enabled': False,
            'sample_ratio': 0.1
        },
        'skip_files': ['.*brana_Epi-001_20100420_05.*'],
        'special_case_handling': {  # drop the channels in the list if the pattern is found in the file path
            'Liogier_AllDataset1200Hz': ['MRO22-2805', 'MRO23-2805', 'MRO24-2805']
        },

        'annotation_rules': {
            'Holdout': {
                'include': ['jj_add', 'JJ_add', 'jj_valid', 'JJ_valid'],  # will look for any of these in the annotation
                'exclude': []
            },
            'IterativeLearningFeedback1': {
                'include': [],                                          # when empty, all annotations are included
                'exclude': ['detected_spike']                           # but those defined here
            },
            'IterativeLearningFeedback2': {
                'include': [],
                'exclude': ['detected_spike']                   # don't consider "detected_spike" as a valid annotation
            },
            'Default': {
                'include': [],
                'exclude': ['true_spike', 'detected_spike']     # don't consider "true_spike" and "detected_spike"
            }
        }
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the configuration manager.

        Args:
            logger: Optional logger instance.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.logger = logger or logging.getLogger(__name__)

    def load_yaml(self, yaml_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file is not found.
            ValueError: If there's an error parsing the YAML.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            self._merge_config(yaml_config)
            self.logger.info(f"Loaded configuration from {yaml_path}")
        except Exception as e:
            raise ValueError(f"Error loading YAML configuration: {str(e)}")

    def load_args(self, args: argparse.Namespace) -> None:
        """Load configuration from command-line arguments.

        Args:
            args: Parsed command-line arguments.
        """
        # Convert args to dict, skipping None values (not provided)
        args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

        # Handle special cases for nested parameters
        normalization = {}
        process_no_spike_files = {}

        for key, value in args_dict.copy().items():
            if key.startswith('normalization_'):
                # Convert 'normalization_method' to 'method'
                norm_key = key.replace('normalization_', '')
                normalization[norm_key] = value
                del args_dict[key]
            elif key.startswith('process_no_spike_files_'):
                # Convert 'process_no_spike_files_enabled' to 'enabled'
                process_key = key.replace('process_no_spike_files_', '')
                process_no_spike_files[process_key] = value
                del args_dict[key]

        # Update config dict
        if normalization:
            self.config['normalization'] = {**self.config['normalization'], **normalization}
        if process_no_spike_files:
            self.config['process_no_spike_files'] = {**self.config['process_no_spike_files'], **process_no_spike_files}

        # Merge remaining args
        self._merge_config(args_dict)
        self.logger.info("Loaded configuration from command-line arguments")

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing configuration.
        The new configuration takes precedence over existing parameters.

        Args:
            new_config: New configuration dictionary to merge.
        """
        for key, value in new_config.items():
            # If both the existing and new values are dictionaries, merge them
            if (key in self.config and isinstance(self.config[key], dict)
                    and isinstance(value, dict)):
                self.config[key].update(value)
            else:
                # Otherwise replace the value
                self.config[key] = value

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Check required parameters
        required_params = ['root_dirs', 'output_dir', 'good_channels_file_path', 'loc_meg_channels_file_path']
        missing_params = [param for param in required_params if param not in self.config]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Check if paths exist
        for path_param in self.config["root_dirs"]:
            if not os.path.exists(path_param):
                raise ValueError(f"Path does not exist: {path_param} (parameter: root_dirs)")
        for path_param in ['good_channels_file_path', 'loc_meg_channels_file_path']:
            if path_param in self.config and not os.path.exists(self.config[path_param]):
                raise ValueError(f"Path does not exist: {self.config[path_param]} (parameter: {path_param})")

        # Create output directory if it doesn't exist
        if 'output_dir' in self.config:
            os.makedirs(self.config['output_dir'], exist_ok=True)

        # Check if train and val ratios sum to 1
        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        if abs(train_ratio + val_ratio - 1.0) > 1e-7:
            raise ValueError(f"Train and val ratios must sum to 1. Got {train_ratio + val_ratio}")

        # Check if overlap is valid
        overlap = self.config['overlap']
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and less than 1.0")

        # Check if sampling_rate is positive
        sampling_rate = self.config['sampling_rate']
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive. Got {sampling_rate}")

        # Check if window_duration_s is positive
        window_duration_s = self.config['window_duration_s']
        if window_duration_s <= 0:
            raise ValueError(f"Clip length must be positive. Got {window_duration_s}")

        # Check if n_windows is positive
        n_windows = self.config['n_windows']
        if n_windows <= 0:
            raise ValueError(f"Chunk size must be positive. Got {n_windows}")

        # Check normalization parameters
        norm_method = self.config['normalization'].get('method', 'percentile')
        if norm_method not in ['percentile', 'zscore', 'minmax', 'none']:
            self.logger.warning(
                f"Unknown normalization method: {norm_method}. Valid options are: 'percentile', 'zscore', 'minmax', 'none'."
            )

        # Check if percentile value is valid for percentile normalization
        if norm_method == 'percentile':
            percentile = self.config['normalization'].get('percentile', 95)
            if not (0 <= percentile <= 100):
                self.logger.warning(f"Percentile value ({percentile}) should be between 0 and 100.")

        # Check if n_jobs is valid
        n_jobs = self.config['n_jobs']
        import multiprocessing
        max_jobs = multiprocessing.cpu_count()
        if n_jobs > max_jobs:
            self.logger.warning(
                f"Number of jobs ({n_jobs}) exceeds available CPU cores ({max_jobs}). "
                f"Consider reducing to improve performance."
            )

        # Warn about potentially unused parameters
        known_params = set(self.DEFAULT_CONFIG.keys()) | {'root_dirs', 'output_dir', 'good_channels_file_path',
                                                          'loc_meg_channels_file_path', 'log_file'
                                                          'validate', 'validation_output_dir', 'visualize',
                                                          'config', 'annotation_rules'}
        # Check for unknown parameters
        unknown_params = set(self.config.keys()) - known_params
        if unknown_params:
            self.logger.warning(f"Found potentially unused parameters: {', '.join(unknown_params)}")

    def save(self, output_path: str, file_format: str = 'yaml') -> None:
        """Save configuration to a file.

        Args:
            output_path: Path to save the configuration file.
            file_format: Format to save ('yaml'/'yml' or 'json').

        Raises:
            ValueError: If the format is not supported.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if file_format.lower() in ['yaml', 'yml']:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)  # False to avoid inline style
        elif file_format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)  # Indent for readability
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'yaml' or 'json'.")

        self.logger.info(f"Saved configuration to {output_path}")

    def get_config_dict(self) -> Dict[str, Any]:
        """Get a deep copy of the configuration dictionary.

        Returns:
            Copy of the configuration dictionary.
        """
        import copy
        return copy.deepcopy(self.config)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the MEG preprocessing pipeline.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description='MEG Data Preprocessing')

    # Entry point arguments
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--root_dirs', nargs='+', help='List of directories containing patient folders containing .ds files')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed data')
    parser.add_argument('--good_channels_file_path', type=str,
                        help='Path to the file containing the list of good channels')
    parser.add_argument('--loc_meg_channels_file_path', type=str,
                        help='Path to the file containing the list of locations of meg channels')
    parser.add_argument('--save_as_pickle', type=bool, help='Save processed data as pickle files instead of torch files')

    # Logging arguments
    parser.add_argument('--log_level', type=str, help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--log_file', type=str, help='Path to log file')

    # Validation and testing arguments
    parser.add_argument('--validate', type=str, help='Path to a specific .ds file to validate after processing')
    parser.add_argument('--validation_output_dir', type=str, help='Output directory for validation visualizations')
    parser.add_argument('--visualize', type=bool, help='Show generated plots')

    # Data splitting options
    parser.add_argument('--train_ratio', type=float, help='Ratio of patients for training')
    parser.add_argument('--val_ratio', type=float, help='Ratio of patients for validation')
    parser.add_argument('--random_state', type=int, help='Random seed')

    # Preprocessing options
    parser.add_argument('--n_windows', type=int, help='Number of clips to aggregate into a chunk')
    parser.add_argument('--window_duration_s', type=float, help='Length of each clip in seconds')
    parser.add_argument('--window_overlap', type=float, help='Fraction of overlap between consecutive windows within sequences (0.0 to <1.0)')
    parser.add_argument('--overlap', type=float, help='Fraction of overlap between consecutive chunks (0.0 to <1.0)')
    parser.add_argument('--sampling_rate', type=int, help='Target sampling rate in Hz')
    parser.add_argument('--spike_overlap_threshold', type=float, help='Minimum overlap percentage to consider a window as containing a spike')
    parser.add_argument('--l_freq', type=float, help='Low-pass filter frequency in Hz')
    parser.add_argument('--h_freq', type=float, help='High-pass filter frequency in Hz')
    parser.add_argument('--notch_freq', type=float, help='Notch filter frequency in Hz')

    # Normalization options
    parser.add_argument('--normalization_method', type=str,
                        choices=['percentile', 'zscore', 'minmax'], help='Method for data normalization')
    parser.add_argument('--normalization_percentile', type=int, help='Percentile value for percentile normalization')

    # Dataset processing options
    parser.add_argument('--min_spikes_per_min', type=int, help='Minimum number of spikes required to process a file')
    parser.add_argument('--process_no_spike_files_enabled', type=bool, help='Process files with no valid spikes')
    parser.add_argument('--process_no_spike_files_sample_ratio', type=float, help='Ratio of samples to take from no-spike files')
    # we don't allow skip_files to be set from the command line, use the config file instead (see default config)

    # Performance options
    parser.add_argument('--n_jobs', type=int, help='Number of parallel jobs to use')

    # Annotation rules
    # we don't allow annotation rules to be set from the command line,
    # use the config file instead (see default config)

    return parser
