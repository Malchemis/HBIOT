#!/usr/bin/env python3
"""MEG recording preprocessing and caching script.

This script preprocesses MEG recordings and saves them to HDF5 files for efficient
loading during training. It is particularly useful for preprocessing data once before
launching distributed training jobs on compute clusters.

The script supports:
- Batch preprocessing of multiple recordings from split files
- Parallel processing with configurable worker count
- Cache checking to avoid reprocessing
- Selective fold processing
- Force reprocessing option

Example:
    Preprocess all files in splits:
        $ python scripts/preprocess_recordings.py \\
            --config configs/default_config.yaml \\
            --splits-dir /path/to/splits \\
            --output-dir /path/to/preprocessed

    Force reprocessing with 8 workers:
        $ python scripts/preprocess_recordings.py \\
            --config configs/default_config.yaml \\
            --splits-dir /path/to/splits \\
            --output-dir /path/to/preprocessed \\
            --force --n-workers 8

    Preprocess specific folds only:
        $ python scripts/preprocess_recordings.py \\
            --config configs/default_config.yaml \\
            --splits-dir /path/to/splits \\
            --output-dir /path/to/preprocessed \\
            --folds 1 2 3
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.data.preprocessing.cache_recordings import preprocess_and_cache_files, check_cache_exists
from pipeline.utils.config_handler import load_config


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def collect_file_paths(splits_dir: str, folds: Optional[List[int]] = None, include_test: bool = True) -> Set[str]:
    """Collect all unique file paths from split files.

    Args:
        splits_dir: Directory containing split JSON files.
        folds: List of fold numbers to process. If None, process all folds.
        include_test: Whether to include test files.

    Returns:
        Set of unique file paths from the specified splits.
    """
    file_paths = set()

    fold_files = sorted(Path(splits_dir).glob("fold_*.json"))

    for fold_file in fold_files:
        fold_num = int(fold_file.stem.split('_')[1])

        if folds is not None and fold_num not in folds:
            continue

        with open(fold_file, 'r') as f:
            fold_data = json.load(f)

        for split_type in ['train', 'val']:
            if split_type in fold_data:
                file_paths.update(fold_data[split_type].get('file_paths', []))

    if include_test:
        test_file = Path(splits_dir) / "test_files.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            file_paths.update(test_data.get('file_paths', []))

    return file_paths


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MEG recordings and cache to HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--splits-dir',
        type=str,
        required=True,
        help='Directory containing split JSON files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for cached HDF5 files'
    )

    parser.add_argument(
        '--reference-coordinates',
        type=str,
        default=None,
        help='Path to reference coordinates file (overrides config)'
    )

    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        help='Specific fold numbers to process (default: all folds)'
    )

    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip preprocessing test files'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if cached files exist'
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check which files are cached, do not preprocess'
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    dataset_config = config['data']['MEGOnTheFlyDataModule']['dataset_config']

    ref_coords_path = args.reference_coordinates or config['data']['MEGOnTheFlyDataModule']['reference_coordinates']
    logger.info(f"Loading reference coordinates from {ref_coords_path}")

    with open(ref_coords_path, 'rb') as f:
        good_channels = pickle.load(f)

    logger.info(f"Loaded {len(good_channels)} reference channels")

    logger.info(f"Collecting file paths from {args.splits_dir}")
    file_paths = collect_file_paths(
        args.splits_dir,
        folds=args.folds,
        include_test=not args.no_test
    )

    logger.info(f"Found {len(file_paths)} unique files to preprocess")

    if len(file_paths) == 0:
        logger.error("No files found to preprocess!")
        return 1

    if args.check_only or not args.force:
        logger.info("Checking cache status...")
        cached, missing = check_cache_exists(list(file_paths), dataset_config, args.output_dir)

        logger.info(f"Cache status: {len(cached)} cached, {len(missing)} missing")

        if args.check_only:
            if missing:
                logger.info("Missing files:")
                for fp in sorted(missing)[:10]:
                    logger.info(f"  - {fp}")
                if len(missing) > 10:
                    logger.info(f"  ... and {len(missing) - 10} more")
            return 0

    logger.info("Starting preprocessing...")
    stats = preprocess_and_cache_files(
        file_paths=list(file_paths),
        config=dataset_config,
        good_channels=good_channels,
        cache_dir=args.output_dir,
        force=args.force,
        n_workers=args.n_workers
    )

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Total files:      {stats['n_total']}")
    logger.info(f"  Processed:        {stats['n_processed']}")
    logger.info(f"  Cached (skipped): {stats['n_cached']}")
    logger.info(f"  Failed:           {stats['n_failed']}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 60)

    return 0 if stats['n_failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
