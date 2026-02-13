"""Filtering statistics tracking for the MEG spike detection pipeline."""

import json
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FilteringStatistics:
    """Accumulates and reports filtering statistics across pipeline stages."""

    def __init__(self):
        self.stages: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def add_stage(self, name: str, stats: Dict[str, Any]) -> None:
        """Record statistics for a filtering stage.

        Args:
            name: Stage name (e.g. 'raw_counts', 'spike_quality_filter').
            stats: Dictionary of statistics for this stage.
        """
        self.stages[name] = stats

    def log_summary(self, log: Optional[logging.Logger] = None) -> None:
        """Log a formatted summary of all stages."""
        log = log or logger
        log.info("=" * 70)
        log.info("FILTERING STATISTICS SUMMARY")
        log.info("=" * 70)

        for stage_name, stats in self.stages.items():
            log.info(f"\n--- {stage_name} ---")
            for key, value in stats.items():
                if isinstance(value, (list, dict)) and len(str(value)) > 120:
                    log.info(f"  {key}: <{type(value).__name__} with {len(value)} entries>")
                else:
                    log.info(f"  {key}: {value}")

        log.info("=" * 70)

    def to_dict(self) -> Dict[str, Any]:
        """Return all stats as a JSON-serializable dictionary."""
        return dict(self.stages)

    def save(self, path: str) -> None:
        """Save statistics to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved filtering statistics to {path}")
