"""Callback registry for PyTorch Lightning training callbacks.

This module provides custom callbacks for MEG spike detection training including
metrics evaluation, model checkpointing, and comprehensive reporting capabilities.
"""

import os
import logging
from typing import Union, Dict, Type, Any, List, Optional

import torch
import torch.nn as nn
import lightning.pytorch as L
from pipeline.eval.metrics import MetricsAggregator
from pipeline.training.lightning_module import MEGSpikeDetector
from pipeline.training.unsupervised_pretrain_lit_mod import MEGUnsupervisedPretrainer

logger = logging.getLogger(__name__)
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
    RichProgressBar,
    RichModelSummary,
    ModelSummary,
    StochasticWeightAveraging
)

try:
    from lightning import LightningModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LightningModule = None
    LIGHTNING_AVAILABLE = False

from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def is_fsdp_model(pl_module):
    """Check if a module uses FSDP (Fully Sharded Data Parallel)."""
    return isinstance(pl_module, FSDP) or any(isinstance(m, FSDP) for m in pl_module.modules())


## --- Custom Callbacks --- ##
class MetricsEvaluationCallback(L.Callback):
    """Callback to handle all metrics computation, reporting, and CSV generation.

    This callback separates evaluation logic from the training module, providing:
    - Validation and test metrics computation
    - Patient-specific and group-level analysis
    - Threshold optimization
    - Comprehensive reporting
    - CSV export of predictions and detailed metrics

    The callback reads all evaluation settings from the Lightning module's config
    (config["evaluation"]) to avoid duplication and ensure consistency.

    The callback uses the MetricsAggregator from pipeline.eval.metrics for all
    computation, ensuring consistency across validation and test phases.

    Attributes:
        window_overlap: Overlap ratio between consecutive windows (for relaxed metrics)
        validation_outputs: Collected outputs from validation batches
        test_outputs: Collected outputs from test batches
        metrics_aggregator: MetricsAggregator instance for computation
        metrics_config: Configuration dictionary read from pl_module (populated in setup)
        logger: Python logger for console output
    """

    def __init__(
        self,
        window_overlap: float = 0.5,
    ):
        """Initialize metrics evaluation callback.

        Args:
            window_overlap: Overlap ratio between consecutive windows (for relaxed metrics).
                           This should match the window_overlap from your dataset configuration.

        Note:
            All other evaluation settings (threshold, threshold_optimization, etc.) are read
            from pl_module.config["evaluation"] in the setup() method.
        """
        super().__init__()
        self.window_overlap = window_overlap
        self.logger = logging.getLogger(__name__)

        # Storage for outputs
        self.validation_outputs: List[Dict[str, Any]] = []
        self.test_outputs: List[Dict[str, Any]] = []

        # Metrics configuration and aggregator (will be initialized in setup)
        self.metrics_config: Dict[str, Any] = {}  # Populated from pl_module.config["evaluation"] in setup()
        self.metrics_aggregator: Optional[MetricsAggregator] = None

    def setup(
        self,
        trainer: L.Trainer,
        pl_module: Union[MEGSpikeDetector, MEGUnsupervisedPretrainer],
        stage: str
    ) -> None:
        """Initialize metrics aggregator with settings from Lightning module's evaluation config."""
        # Read evaluation config from Lightning module (single source of truth)
        self.metrics_config = pl_module.config.get("evaluation", {})

        # Get threshold from Lightning module
        threshold = getattr(pl_module, 'threshold', self.metrics_config.get('default_threshold', 0.5))

        self.metrics_aggregator = MetricsAggregator(
            compute_relaxed=self.metrics_config.get('compute_relaxed', True),
            threshold=threshold,
            window_overlap=self.window_overlap
        )
        self.logger.info(
            f"MetricsEvaluationCallback initialized from config.evaluation: "
            f"threshold={threshold:.3f}, "
            f"compute_relaxed={self.metrics_config.get('compute_relaxed', True)}, "
            f"threshold_optimization={self.metrics_config.get('threshold_optimization', False)}"
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Collect validation batch outputs for epoch-end processing."""
        if outputs is not None and isinstance(outputs, dict):
            self.validation_outputs.append(outputs)

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Reset validation outputs and metrics aggregator."""
        self.validation_outputs = []
        if self.metrics_aggregator:
            self.metrics_aggregator.reset()

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Compute and log validation metrics."""
        if not self.validation_outputs:
            self.logger.warning("No validation outputs collected")
            return

        # Compute metrics with optional threshold optimization
        optimize_threshold = self.metrics_config.get('threshold_optimization', False)
        metrics = self._process_outputs(
            self.validation_outputs,
            optimize_threshold=optimize_threshold
        )

        # Update threshold in both aggregator and Lightning module
        if optimize_threshold and self.metrics_aggregator:
            threshold = self.metrics_aggregator.threshold
            pl_module.threshold = threshold  # type: ignore
            pl_module.log("best_threshold", threshold, sync_dist=True, on_epoch=True)
            pl_module.hparams["threshold"] = threshold  # type: ignore
            self.logger.info(f"Optimized threshold: {threshold:.4f}")

        # Log all metrics
        for key, value in metrics.items():
            pl_module.log(f"val_{key}", value, sync_dist=True, on_epoch=True)

        # Log probability distribution histograms to TensorBoard
        self._log_probability_distributions(trainer)

        # Clear outputs to free memory
        self.validation_outputs = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Collect test batch outputs for epoch-end processing."""
        if outputs is not None and isinstance(outputs, dict):
            self.test_outputs.append(outputs)

    def on_test_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Reset test outputs and metrics aggregator."""
        self.test_outputs = []
        if self.metrics_aggregator:
            self.metrics_aggregator.reset()
            # Use optimized threshold from validation if available
            if self.metrics_config.get('threshold_optimization', False):
                threshold = getattr(pl_module, 'threshold', self.metrics_config.get('default_threshold', 0.5))
                self.metrics_aggregator.threshold = threshold

    def on_test_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Compute test metrics, generate reports, and save results."""
        if not self.test_outputs:
            self.logger.warning("No test outputs collected")
            return

        # Compute metrics with optional threshold optimization
        test_optimize_threshold = self.metrics_config.get('test_threshold_optimization', False)
        metrics = self._process_outputs(
            self.test_outputs,
            optimize_threshold=test_optimize_threshold
        )

        if test_optimize_threshold and self.metrics_aggregator:
            threshold = self.metrics_aggregator.threshold
            self.logger.info(f"Optimal test threshold: {threshold:.4f}")

        # Log all metrics
        for key, value in metrics.items():
            pl_module.log(f"test_{key}", value, sync_dist=True, on_epoch=True)

        # Generate and log comprehensive report (only on rank 0)
        if trainer.is_global_zero and self.metrics_aggregator:
            report = self.metrics_aggregator.generate_comprehensive_report()
            self.logger.info(f"Test evaluation:\n{report}")

            # Save predictions CSV if metadata is available
            if any('metadata' in output for output in self.test_outputs):
                self._save_predictions_csv(trainer, pl_module)

            # Save detailed metrics results
            metrics_df = self.metrics_aggregator.get_dataframe()
            if metrics_df is not None and len(metrics_df) > 0:
                self._save_detailed_metrics(trainer)

        # Clear outputs to free memory
        self.test_outputs = []

    def _process_outputs(
        self,
        outputs: List[Dict[str, Any]],
        optimize_threshold: bool = False
    ) -> Dict[str, float]:
        """Process collected outputs and compute metrics.

        Args:
            outputs: List of output dictionaries from batches
            optimize_threshold: Whether to optimize threshold based on F1 score

        Returns:
            Dictionary of computed metrics
        """
        if not outputs or not self.metrics_aggregator:
            return {}

        # Reset aggregator
        self.metrics_aggregator.reset()

        for batch_output in outputs:
            self.metrics_aggregator.update(
                probs=batch_output["probs"],
                gt=batch_output["gt"],
                mask=batch_output.get("mask"),
                batch_size=batch_output["batch_size"],
                n_windows=batch_output["n_windows"],
                metadata=batch_output.get("metadata"),
                losses=batch_output.get("losses"),
                onset_probs=batch_output.get("onset_probs"),
                gt_onsets=batch_output.get("gt_onsets"),
                onset_losses=batch_output.get("onset_losses"),
            )

        # Compute metrics
        all_metrics = self.metrics_aggregator.compute(optimize_threshold=optimize_threshold)

        # Extract scalar metrics for logging (patient/group metrics logged separately)
        scalar_metrics = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float))}
        return scalar_metrics  # type: ignore

    def _log_probability_distributions(self, trainer: L.Trainer) -> None:
        """Log probability distributions for presence and onset predictions to TensorBoard.

        Args:
            trainer: Lightning trainer instance
        """
        if not trainer.logger or not hasattr(trainer.logger, 'experiment'):
            return

        import numpy as np
        import torch

        all_probs = []
        all_gt = []
        all_masks = []
        all_onset_probs = []
        all_gt_onsets = []

        for output in self.validation_outputs:
            all_probs.append(output["probs"].flatten())
            all_gt.append(output["gt"].flatten())
            if output.get("mask") is not None:
                all_masks.append(output["mask"].flatten())
            if output.get("onset_probs") is not None:
                all_onset_probs.append(output["onset_probs"].flatten())
            if output.get("gt_onsets") is not None:
                all_gt_onsets.append(output["gt_onsets"].flatten())

        probs = np.concatenate(all_probs)
        gt = np.concatenate(all_gt)

        if all_masks:
            masks = np.concatenate(all_masks)
            valid_indices = masks > 0
            probs = probs[valid_indices]
            gt = gt[valid_indices]

        class_0_probs = probs[gt == 0]
        class_1_probs = probs[gt == 1]

        try:
            if len(class_0_probs) > 0:
                trainer.logger.experiment.add_histogram(
                    'val_probability_distribution/presence_non_spike',
                    torch.from_numpy(class_0_probs),
                    global_step=trainer.current_epoch
                )

            if len(class_1_probs) > 0:
                trainer.logger.experiment.add_histogram(
                    'val_probability_distribution/presence_spike',
                    torch.from_numpy(class_1_probs),
                    global_step=trainer.current_epoch
                )

            if all_onset_probs and all_gt_onsets:
                onset_probs = np.concatenate(all_onset_probs)
                gt_onsets = np.concatenate(all_gt_onsets)

                if all_masks:
                    onset_probs = onset_probs[valid_indices]
                    gt_onsets = gt_onsets[valid_indices]

                spike_mask = gt == 1
                if spike_mask.sum() > 0:
                    spike_onset_preds = onset_probs[spike_mask]
                    spike_onset_gt = gt_onsets[spike_mask]

                    trainer.logger.experiment.add_histogram(
                        'val_onset_distribution/predicted_onsets',
                        torch.from_numpy(spike_onset_preds),
                        global_step=trainer.current_epoch
                    )

                    trainer.logger.experiment.add_histogram(
                        'val_onset_distribution/ground_truth_onsets',
                        torch.from_numpy(spike_onset_gt),
                        global_step=trainer.current_epoch
                    )

        except Exception as e:
            self.logger.debug(f"Could not log distributions: {e}")

    def _save_predictions_csv(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Save test predictions to CSV file.
        This includes metadata and formatted predictions for each sample.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
        """
        from datetime import datetime
        import pandas as pd

        if not trainer.logger or not trainer.logger.log_dir:
            self.logger.warning("Cannot save predictions: logger or log_dir not available")
            return

        output_dir = trainer.logger.log_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")

        # Get threshold from aggregator or module
        threshold = self.metrics_aggregator.threshold if self.metrics_aggregator else 0.5

        csv_data = []
        for output in self.test_outputs:
            if 'metadata' not in output:
                continue

            predictions = output['probs']
            ground_truth = output['gt']
            batch_size = output['batch_size']
            metadata_list = output['metadata']

            onset_preds = output.get('onset_probs')
            onset_gt = output.get('gt_onsets')

            if not isinstance(metadata_list, list):
                self.logger.error(f"Expected metadata to be a list, got {type(metadata_list)}")
                continue

            for i in range(batch_size):
                if i >= len(metadata_list):
                    self.logger.warning(f"Metadata list length ({len(metadata_list)}) < batch_size ({batch_size}), skipping sample {i}")
                    continue

                sample_meta = metadata_list[i]

                pred_probs = predictions[i].flatten()
                pred_classes = (pred_probs >= threshold).astype(int)
                gt_flat = ground_truth[i].flatten()

                chunk_onset = sample_meta.get('chunk_onset_sample', 0)
                global_chunk_idx = sample_meta.get('global_chunk_idx', None)
                chunk_idx = sample_meta.get('chunk_idx', sample_meta.get('chunk_index', 0))
                spikes = sample_meta.get('spike_positions_in_chunk', [])
                start_window_idx = sample_meta.get('start_window_idx', 0)
                end_window_idx = sample_meta.get('end_window_idx', 0)

                preprocessing_config = sample_meta.get('preprocessing_config', {})
                sampling_rate = preprocessing_config.get('sampling_rate', 0) if isinstance(preprocessing_config, dict) else 0

                row = {
                    'split': 'test',
                    'file_name': sample_meta.get('file_name', 'unknown'),
                    'patient_id': sample_meta.get('patient_id', 'unknown'),
                    'original_filename': sample_meta.get('original_filename', 'unknown'),
                    'group': sample_meta.get('group', 'Unknown'),
                    'global_chunk_idx': global_chunk_idx,
                    'chunk_onset_sample': chunk_onset,
                    'chunk_index': chunk_idx,
                    'chunk_idx': chunk_idx,
                    'start_window_idx': start_window_idx,
                    'end_window_idx': end_window_idx,
                    'spike_positions_in_chunk': spikes,
                    'extraction_mode': sample_meta.get('extraction_mode', 'fixed'),
                    'sampling_rate': sampling_rate,
                    'predicted_probs': ','.join(f"{float(p):.4f}" for p in pred_probs),
                    'predicted_classes': ','.join(map(str, pred_classes)),
                    'ground_truth': ','.join(map(str, gt_flat)),
                }

                if onset_preds is not None:
                    onset_pred_flat = onset_preds[i].flatten()
                    row['predicted_onsets'] = ','.join(f"{float(o):.4f}" for o in onset_pred_flat)

                if onset_gt is not None:
                    onset_gt_flat = onset_gt[i].flatten()
                    row['ground_truth_onsets'] = ','.join(f"{float(o):.4f}" for o in onset_gt_flat)

                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved predictions to {csv_path}")

    def _save_detailed_metrics(self, trainer: L.Trainer) -> None:
        """Save detailed metrics results to CSV files.

        Args:
            trainer: Lightning trainer instance
        """
        from datetime import datetime

        if not trainer.logger or not trainer.logger.log_dir or not self.metrics_aggregator:
            return

        output_dir = trainer.logger.log_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.metrics_aggregator.save_detailed_results(output_dir, timestamp)


class GradientNormLogger(L.Callback):
    """Enhanced callback with integrated adaptive gradient clipping and comprehensive logging.

    This callback provides:
    - Adaptive gradient clipping (integrated ZClip functionality)
    - Global gradient norm tracking
    - Per-module gradient norms
    - Gradient distribution histograms
    - Percentile tracking (P50, P90, P95, P99)
    - Gradient sparsity metrics
    - Comprehensive TensorBoard logging of all metrics
    - Efficient single-pass gradient norm computation

    Attributes:
        log_global_norm: Whether to log the global gradient norm
        log_module_norms: Whether to log per-module gradient norms
        log_param_norms: Whether to log per-parameter gradient norms
        log_histograms: Whether to log gradient distribution histograms
        log_percentiles: Whether to log gradient norm percentiles
        log_sparsity: Whether to log gradient sparsity metrics
        log_every_n_steps: Log gradients every N training steps
        norm_type: Type of norm to compute (default: 2.0 for L2 norm)
        modules_to_track: List of module name patterns to track

        # Adaptive clipping parameters (integrated ZClip)
        use_adaptive_clipping: Whether to enable adaptive gradient clipping
        clip_alpha: EMA smoothing factor for mean and variance
        clip_z_thresh: Z-score threshold for outlier detection
        clip_max_norm: Optional maximum gradient norm (applied after adaptive clipping)
        clip_eps: Small constant to avoid division by zero
        clip_warmup_steps: Number of steps before enabling adaptive clipping
        clip_mode: Clipping mode ("zscore" or "percentile")
        clip_option: Clipping strategy for zscore mode ("adaptive_scaling" or "mean")
        clip_factor: Multiplier for adaptive threshold
        clip_skip_update_on_spike: Whether to skip EMA update when spike detected

        # Adaptive clipping state
        clip_buffer: Buffer for warmup gradient norms
        clip_initialized: Whether warmup is complete
        clip_mean: EMA mean of gradient norms
        clip_var: EMA variance of gradient norms

        logger: Python logger for console output
    """

    def __init__(
        self,
        # Logging configuration
        log_global_norm: bool = True,
        log_module_norms: bool = True,
        log_param_norms: bool = False,  # Default False - very verbose
        log_histograms: bool = True,
        log_percentiles: bool = True,
        log_sparsity: bool = True,
        log_every_n_steps: int = 50,
        norm_type: float = 2.0,
        modules_to_track: Optional[List[str]] = None,

        # Adaptive clipping configuration (integrated ZClip)
        use_adaptive_clipping: bool = False,
        clip_alpha: float = 0.97,
        clip_z_thresh: float = 2.5,
        clip_max_norm: Optional[float] = None,
        clip_eps: float = 1e-6,
        clip_warmup_steps: int = 25,
        clip_mode: str = "zscore",
        clip_option: str = "adaptive_scaling",
        clip_factor: float = 1.0,
        clip_skip_update_on_spike: bool = False,
    ):
        """Initialize the gradient norm logger callback with integrated adaptive clipping.

        Args:
            # Logging parameters
            log_global_norm: Whether to log the global gradient norm
            log_module_norms: Whether to log gradient norms for each module
            log_param_norms: Whether to log gradient norms for individual parameters (very verbose)
            log_histograms: Whether to log gradient distribution histograms to TensorBoard
            log_percentiles: Whether to log percentiles (P50, P90, P95, P99) of gradient norms
            log_sparsity: Whether to log gradient sparsity (percentage of near-zero gradients)
            log_every_n_steps: Frequency of logging (every N training steps)
            norm_type: Type of norm to compute (2.0 for L2 norm, inf for infinity norm)
            modules_to_track: Optional list of module names to track specifically

            # Adaptive clipping parameters (integrated from ZClip)
            use_adaptive_clipping: Whether to enable adaptive gradient clipping
            clip_alpha: Smoothing factor for EMA mean and variance (default: 0.97)
            clip_z_thresh: Z-score threshold for outlier detection (default: 2.5)
            clip_max_norm: Optional maximum gradient norm (applied after adaptive clipping)
            clip_eps: Small constant to avoid division by zero (default: 1e-6)
            clip_warmup_steps: Number of steps to collect gradients before enabling clipping (default: 25)
            clip_mode: Clipping mode - "zscore" or "percentile" (default: "zscore")
                      - "percentile": Always clip to mean + (z_thresh × std)
                      - "zscore": Use z-score based adaptive clipping
            clip_option: Strategy for zscore mode - "adaptive_scaling" or "mean" (default: "adaptive_scaling")
                        - "adaptive_scaling": Adaptive threshold based on outlier magnitude
                        - "mean": Clip to EMA mean when z-score exceeds threshold
            clip_factor: Multiplier for the adaptive threshold (default: 1.0)
            clip_skip_update_on_spike: If True, skip EMA update when spike detected (default: False)
        """
        super().__init__()

        # Logging configuration
        self.log_global_norm = log_global_norm
        self.log_module_norms = log_module_norms
        self.log_param_norms = log_param_norms
        self.log_histograms = log_histograms
        self.log_percentiles = log_percentiles
        self.log_sparsity = log_sparsity
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type
        self.modules_to_track = modules_to_track
        self.logger = logging.getLogger(__name__)
        self.step_count = 0

        # Adaptive clipping configuration (integrated ZClip)
        self.use_adaptive_clipping = use_adaptive_clipping
        self.clip_alpha = clip_alpha
        self.clip_z_thresh = clip_z_thresh
        self.clip_max_norm = clip_max_norm
        self.clip_eps = clip_eps
        self.clip_warmup_steps = clip_warmup_steps
        self.clip_mode = clip_mode.lower()
        self.clip_factor = clip_factor
        self.clip_skip_update_on_spike = clip_skip_update_on_spike

        # Validate clip_mode and clip_option
        if self.clip_mode == "zscore":
            assert clip_option in ["mean", "adaptive_scaling"], (
                "For zscore mode, clip_option must be either 'mean' or 'adaptive_scaling'."
            )
            self.clip_option = clip_option.lower()
        elif self.clip_mode == "percentile":
            self.clip_option = None  # Not used in percentile mode
        else:
            raise ValueError("clip_mode must be either 'zscore' or 'percentile'.")

        # Adaptive clipping state (integrated from ZClip)
        if self.use_adaptive_clipping:
            self.clip_buffer: List[float] = []
            self.clip_initialized = False
            self.clip_mean: Optional[float] = None
            self.clip_var: Optional[float] = None
            self.logger.info(
                f"Adaptive gradient clipping enabled: mode={self.clip_mode}, "
                f"z_thresh={self.clip_z_thresh}, warmup_steps={self.clip_warmup_steps}, "
                f"clip_option={self.clip_option}, max_norm={self.clip_max_norm}"
            )

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Compute gradient norm once, apply adaptive clipping, and log all metrics.

        This hook is called after backward() but before optimizer.step().

        Workflow:
        1. Compute gradient norm ONCE (reused for clipping and logging)
        2. Apply adaptive clipping if enabled (modifies gradients in-place)
        3. Log all metrics if this is a logging step

        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            optimizer: Optimizer being used (not used but required by Lightning API)
        """
        # Increment step counter
        self.step_count += 1
        should_log = (self.step_count % self.log_every_n_steps == 0)

        # STEP 1: Compute gradient norm ONCE (will be reused for clipping and logging)
        pre_clip_norm = self._compute_global_grad_norm(pl_module)

        # STEP 2: Apply adaptive clipping if enabled (happens every step, not just logging steps)
        clip_occurred = False
        post_clip_norm = pre_clip_norm
        adaptive_threshold = None
        z_score = None
        is_outlier = False

        if self.use_adaptive_clipping:
            clip_occurred, post_clip_norm, adaptive_threshold, z_score, is_outlier = (
                self._apply_adaptive_clipping(pl_module, pre_clip_norm)
            )

        # STEP 3: Log all metrics if this is a logging step
        if should_log:
            self._log_all_gradient_metrics(
                pl_module=pl_module,
                trainer=trainer,
                pre_clip_norm=pre_clip_norm,
                post_clip_norm=post_clip_norm,
                clip_occurred=clip_occurred,
                adaptive_threshold=adaptive_threshold,
                z_score=z_score,
                is_outlier=is_outlier
            )

    # ==================== Adaptive Clipping Methods (Integrated ZClip) ====================

    def _apply_adaptive_clipping(
        self,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        grad_norm: float
    ) -> tuple[bool, float, Optional[float], Optional[float], bool]:
        """Apply adaptive gradient clipping (integrated ZClip functionality).

        This method implements the core ZClip algorithm integrated directly into the callback.
        It computes the clipping threshold adaptively based on EMA statistics and applies
        clipping in-place to the model's gradients.

        Args:
            pl_module: Lightning module instance
            grad_norm: Pre-computed gradient norm (from _compute_global_grad_norm)

        Returns:
            Tuple containing:
            - clip_occurred (bool): Whether clipping was applied
            - post_clip_norm (float): Gradient norm after clipping
            - adaptive_threshold (float|None): Computed adaptive threshold (None during warmup)
            - z_score (float|None): Z-score of gradient norm (None during warmup)
            - is_outlier (bool): Whether gradient was detected as outlier
        """
        # During warmup: collect gradient norms
        if not self.clip_initialized:
            self.clip_buffer.append(grad_norm)

            # Initialize EMA when warmup is complete
            if len(self.clip_buffer) >= self.clip_warmup_steps:
                self.clip_mean = sum(self.clip_buffer) / len(self.clip_buffer)
                self.clip_var = sum((x - self.clip_mean) ** 2 for x in self.clip_buffer) / len(self.clip_buffer)
                self.clip_initialized = True
                self.clip_buffer = []  # Clear buffer
                self.logger.info(
                    f"Adaptive clipping initialized: mean={self.clip_mean:.4f}, "
                    f"std={self.clip_var**0.5:.4f}"
                )

            # During warmup, optionally apply max_grad_norm if specified
            if self.clip_max_norm is not None:
                self._apply_max_norm_clipping(pl_module, grad_norm, self.clip_max_norm)
                post_clip_norm = self._compute_global_grad_norm(pl_module)
                clip_occurred = abs(post_clip_norm - grad_norm) > 1e-6
                return clip_occurred, post_clip_norm, None, None, False

            # No clipping during warmup (unless max_grad_norm is set)
            return False, grad_norm, None, None, False

        # After warmup: compute adaptive threshold and apply clipping
        assert self.clip_mean is not None and self.clip_var is not None

        std = self.clip_var ** 0.5
        z_score = (grad_norm - self.clip_mean) / (std + self.clip_eps)
        is_outlier = z_score > self.clip_z_thresh

        # Compute adaptive threshold based on mode
        adaptive_threshold = self._compute_adaptive_threshold(grad_norm, std, z_score)

        # Determine final clipping threshold (merge adaptive + max_grad_norm)
        if adaptive_threshold is not None:
            if self.clip_max_norm is not None:
                effective_threshold = min(adaptive_threshold, self.clip_max_norm)
            else:
                effective_threshold = adaptive_threshold
        else:
            # No adaptive clipping needed, but may still apply max_grad_norm
            effective_threshold = self.clip_max_norm if self.clip_max_norm is not None else None

        # Apply clipping if threshold is set and exceeded
        clip_occurred = False
        post_clip_norm = grad_norm

        if effective_threshold is not None and grad_norm > effective_threshold:
            self._apply_in_place_clipping(pl_module, grad_norm, effective_threshold)
            post_clip_norm = self._compute_global_grad_norm(pl_module)
            clip_occurred = True

        # Update EMA statistics (unless skipping on spike)
        if not (is_outlier and self.clip_skip_update_on_spike):
            # Update with the effective norm (clipped if clipping occurred)
            effective_norm = post_clip_norm if clip_occurred else grad_norm
            self.clip_mean = self.clip_alpha * self.clip_mean + (1 - self.clip_alpha) * effective_norm
            self.clip_var = self.clip_alpha * self.clip_var + (1 - self.clip_alpha) * (effective_norm - self.clip_mean) ** 2

        return clip_occurred, post_clip_norm, adaptive_threshold, z_score, is_outlier

    def _compute_adaptive_threshold(
        self,
        grad_norm: float,
        std: float,
        z_score: float
    ) -> Optional[float]:
        """Compute adaptive clipping threshold based on mode and current statistics.

        Args:
            grad_norm: Current gradient norm
            std: Standard deviation from EMA
            z_score: Computed z-score

        Returns:
            Adaptive threshold or None if no clipping needed
        """
        assert self.clip_mean is not None

        if self.clip_mode == "percentile":
            # Percentile mode: always use fixed threshold = mean + (z_thresh × std)
            threshold = self.clip_mean + self.clip_z_thresh * std
            if grad_norm > threshold:
                return threshold
            return None

        elif self.clip_mode == "zscore":
            # Z-score mode: only clip if z-score exceeds threshold
            if z_score > self.clip_z_thresh:
                if self.clip_option == "adaptive_scaling":
                    # Adaptive scaling: threshold scales inversely with outlier magnitude
                    eta = z_score / self.clip_z_thresh
                    threshold = self.clip_mean + (self.clip_z_thresh * std) / eta
                    threshold = threshold * self.clip_factor
                elif self.clip_option == "mean":
                    # Mean mode: clip to EMA mean
                    threshold = self.clip_mean
                else:
                    raise ValueError(f"Unknown clip_option: {self.clip_option}")
                return threshold
            return None

        else:
            raise ValueError(f"Unknown clip_mode: {self.clip_mode}")

    def _apply_in_place_clipping(
        self,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        grad_norm: float,
        max_norm: float
    ) -> None:
        """Apply gradient clipping in-place to all parameters.

        Args:
            pl_module: Lightning module instance
            grad_norm: Current gradient norm
            max_norm: Maximum allowed norm
        """
        clip_coef = (max_norm / (grad_norm + self.clip_eps)) if grad_norm > max_norm else 1.0

        if clip_coef < 1.0:
            for param in pl_module.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def _apply_max_norm_clipping(
        self,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        grad_norm: float,
        max_norm: float
    ) -> None:
        """Apply maximum gradient norm clipping (for warmup or as fallback).

        Handles both regular models and FSDP models.

        Args:
            pl_module: Lightning module instance
            grad_norm: Current gradient norm
            max_norm: Maximum allowed norm
        """
        if is_fsdp_model(pl_module):
            # Use FSDP's built-in clip_grad_norm_
            if isinstance(pl_module, FSDP):
                pl_module.clip_grad_norm_(max_norm)
            elif LIGHTNING_AVAILABLE and LightningModule is not None and isinstance(pl_module, LightningModule):
                pl_module.trainer.model.clip_grad_norm_(max_norm)
            else:
                # Find root FSDP module
                for m in pl_module.modules():
                    if isinstance(m, FSDP) and m._is_root:
                        m.clip_grad_norm_(max_norm)
                        break
        else:
            # Regular clipping for non-FSDP models
            torch.nn.utils.clip_grad_norm_(pl_module.parameters(), max_norm)

    # ==================== Comprehensive Logging Method ====================

    def _log_all_gradient_metrics(
        self,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        trainer: L.Trainer,
        pre_clip_norm: float,
        post_clip_norm: float,
        clip_occurred: bool,
        adaptive_threshold: Optional[float],
        z_score: Optional[float],
        is_outlier: bool
    ) -> None:
        """Log all gradient metrics to TensorBoard in one comprehensive method.

        This method logs:
        - Global gradient norms (pre/post clip)
        - Adaptive clipping metrics (warmup progress, EMA stats, thresholds, z-scores)
        - Module-level gradient norms
        - Parameter-level gradient norms (if enabled)
        - Gradient distributions (histograms, percentiles, sparsity)

        Args:
            pl_module: Lightning module instance
            trainer: Lightning trainer instance
            pre_clip_norm: Gradient norm before clipping
            post_clip_norm: Gradient norm after clipping
            clip_occurred: Whether clipping was applied
            adaptive_threshold: Computed adaptive threshold (None during warmup)
            z_score: Z-score of gradient norm (None during warmup)
            is_outlier: Whether gradient was detected as outlier
        """
        # === Global Gradient Norms ===
        if self.log_global_norm:
            if self.use_adaptive_clipping:
                pl_module.log("grad_norm/pre_clip", pre_clip_norm, on_step=True, on_epoch=False, logger=True, sync_dist=False)
                pl_module.log("grad_norm/post_clip", post_clip_norm, on_step=True, on_epoch=False, logger=True, sync_dist=False)
            else:
                pl_module.log("grad_norm/global", pre_clip_norm, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        # === Adaptive Clipping Metrics ===
        if self.use_adaptive_clipping:
            # Check if still in warmup
            if not self.clip_initialized:
                # Log warmup progress
                warmup_progress = len(self.clip_buffer) / self.clip_warmup_steps
                pl_module.log("adaptive_clip/warmup_progress", warmup_progress, on_step=True, on_epoch=False, logger=True, sync_dist=False)
            else:
                # Log EMA statistics
                assert self.clip_mean is not None and self.clip_var is not None
                std = self.clip_var ** 0.5

                pl_module.log("adaptive_clip/ema_mean", self.clip_mean, on_step=True, on_epoch=False, logger=True, sync_dist=False)
                pl_module.log("adaptive_clip/ema_std", std, on_step=True, on_epoch=False, logger=True, sync_dist=False)
                pl_module.log("adaptive_clip/ema_var", self.clip_var, on_step=True, on_epoch=False, logger=True, sync_dist=False)

                # Log z-score and outlier detection
                if z_score is not None:
                    pl_module.log("adaptive_clip/z_score", z_score, on_step=True, on_epoch=False, logger=True, sync_dist=False)
                    pl_module.log("adaptive_clip/is_outlier", float(is_outlier), on_step=True, on_epoch=False, logger=True, sync_dist=False)

                # Log adaptive threshold (if computed)
                if adaptive_threshold is not None:
                    pl_module.log("adaptive_clip/threshold", adaptive_threshold, on_step=True, on_epoch=False, logger=True, sync_dist=False)

                    # Log eta (scaling factor) for adaptive_scaling mode
                    if self.clip_mode == "zscore" and self.clip_option == "adaptive_scaling" and z_score is not None and z_score > self.clip_z_thresh:
                        eta = z_score / self.clip_z_thresh
                        pl_module.log("adaptive_clip/eta_scaling", eta, on_step=True, on_epoch=False, logger=True, sync_dist=False)

                # Log clipping statistics
                pl_module.log("adaptive_clip/clip_occurred", float(clip_occurred), on_step=True, on_epoch=False, logger=True, sync_dist=False)

                if clip_occurred and pre_clip_norm > 1e-6:
                    clip_ratio = post_clip_norm / pre_clip_norm
                    clip_magnitude = pre_clip_norm - post_clip_norm
                    pl_module.log("adaptive_clip/clip_ratio", clip_ratio, on_step=True, on_epoch=False, logger=True, sync_dist=False)
                    pl_module.log("adaptive_clip/clip_magnitude", clip_magnitude, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        # === Module-level Gradient Norms ===
        if self.log_module_norms:
            module_norms = self._compute_module_grad_norms(pl_module)
            for module_name, norm_value in module_norms.items():
                pl_module.log(f"grad_norm/module/{module_name}", norm_value, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        # === Parameter-level Gradient Norms (verbose) ===
        if self.log_param_norms:
            param_norms = self._compute_param_grad_norms(pl_module)
            for param_name, norm_value in param_norms.items():
                pl_module.log(f"grad_norm/param/{param_name}", norm_value, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        # === Gradient Distributions ===
        if self.log_histograms or self.log_percentiles or self.log_sparsity:
            self._log_gradient_distributions(pl_module, trainer)

    # ==================== Gradient Computation Methods ====================

    def _compute_global_grad_norm(self, pl_module: nn.Module) -> float:
        """Compute the global gradient norm across all parameters (FSDP-aware).

        For FSDP models, aggregates norms across sharded parameters using all-reduce.
        For regular models, computes norm using all local parameters.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            Global gradient norm value
        """
        parameters = [p for p in pl_module.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0

        first_param = parameters[0]
        assert first_param.grad is not None, "First parameter has no gradient."
        device = first_param.grad.device
        dtype = first_param.dtype

        # Handle FSDP models specially (aggregate across shards)
        if is_fsdp_model(pl_module):
            if self.norm_type == 2.0:
                # L2 norm for FSDP: sum squared norms and all-reduce
                local_norm_sq = torch.stack(
                    [p.grad.to(dtype).norm(2).pow(2) for p in parameters if p.grad is not None]
                ).to(device)
                local_norm_sq = torch.sum(local_norm_sq)
                dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
                total_norm = torch.sqrt(local_norm_sq)
                return total_norm.item()
            else:
                # For other norms, fall back to local computation
                # (Note: may not be accurate for FSDP with non-L2 norms)
                pass

        # Regular model or non-L2 norm
        if self.norm_type == float('inf'):
            # Infinity norm: max of absolute values
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters if p.grad is not None)
        else:
            # L2 norm (most common)
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach(), self.norm_type).to(device)
                    for p in parameters
                    if p.grad is not None
                ]),
                self.norm_type
            )

        return total_norm.item()

    def _compute_module_grad_norms(self, pl_module: nn.Module) -> Dict[str, float]:
        """Compute gradient norms for each module in the model.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            Dictionary mapping module names to their gradient norms
        """
        module_norms = {}

        # Handle both LightningModule (with .model attribute) and regular nn.Module
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module

        for module_name, module in model.named_modules(): # type: ignore
            # Skip empty module names and containers without parameters
            if not module_name:
                continue

            # Filter by modules_to_track if specified
            if self.modules_to_track is not None:
                if not any(pattern in module_name for pattern in self.modules_to_track):
                    continue

            # Collect parameters with gradients from this module
            parameters = [p for p in module.parameters(recurse=False) if p.grad is not None]
            if len(parameters) == 0:
                continue

            # Compute norm for this module
            assert parameters[0].grad is not None, "Parameter has no gradient."
            device = parameters[0].grad.device
            if self.norm_type == float('inf'):
                norm = max(p.grad.detach().abs().max().to(device) for p in parameters if p.grad is not None)
            else:
                norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach(), self.norm_type).to(device)
                        for p in parameters
                        if p.grad is not None
                    ]),
                    self.norm_type
                )

            # Clean up module name for logging (replace dots with underscores)
            clean_name = module_name.replace('.', '/')
            module_norms[clean_name] = norm.item()

        return module_norms

    def _compute_param_grad_norms(self, pl_module: nn.Module) -> Dict[str, float]:
        """Compute gradient norms for individual parameters.

        Warning: This can be very verbose for large models.

        Args:
            pl_module: PyTorch Lightning module

        Returns:
            Dictionary mapping parameter names to their gradient norms
        """
        param_norms = {}

        # Handle both LightningModule (with .model attribute) and regular nn.Module
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module

        for param_name, param in model.named_parameters(): # type: ignore
            if param.grad is None:
                continue

            # Filter by modules_to_track if specified
            if self.modules_to_track is not None:
                if not any(pattern in param_name for pattern in self.modules_to_track):
                    continue

            # Compute norm for this parameter
            norm = torch.norm(param.grad.detach(), self.norm_type)

            # Clean up parameter name for logging
            clean_name = param_name.replace('.', '/')
            param_norms[clean_name] = norm.item()

        return param_norms

    def _log_gradient_distributions(
        self,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        trainer: L.Trainer
    ) -> None:
        """Log gradient distribution statistics (histograms, percentiles, sparsity).

        Args:
            pl_module: PyTorch Lightning module
            trainer: Lightning trainer instance
        """
        # Collect all gradients into a flat tensor
        all_grads = []
        for param in pl_module.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().flatten())

        if not all_grads:
            return

        # Concatenate all gradients
        flat_grads = torch.cat(all_grads)

        # Log histogram to TensorBoard
        if self.log_histograms and trainer.logger is not None:
            try:
                # TensorBoardLogger has an experiment attribute (SummaryWriter)
                trainer.logger.experiment.add_histogram(
                    'grad_distribution/all_gradients',
                    flat_grads,
                    global_step=trainer.global_step
                )
            except Exception as e:
                self.logger.debug(f"Could not log histogram: {e}")

        # Compute and log percentiles
        if self.log_percentiles:
            grad_norms = torch.abs(flat_grads)
            percentiles = {
                'p50': torch.quantile(grad_norms, 0.50).item(),
                'p90': torch.quantile(grad_norms, 0.90).item(),
                'p95': torch.quantile(grad_norms, 0.95).item(),
                'p99': torch.quantile(grad_norms, 0.99).item(),
            }
            for percentile_name, percentile_value in percentiles.items():
                pl_module.log(
                    f"grad_percentile/{percentile_name}",
                    percentile_value,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=False
                )

        # Compute and log sparsity (percentage of near-zero gradients)
        if self.log_sparsity:
            # Define "near-zero" as gradients with absolute value < 1e-7
            sparsity_threshold = 1e-7
            near_zero = (torch.abs(flat_grads) < sparsity_threshold).float().mean().item()
            sparsity_pct = near_zero * 100

            pl_module.log(
                "grad_sparsity/pct_near_zero",
                sparsity_pct,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False
            )


class TemperatureScalingCallback(L.Callback):
    """Callback to perform temperature scaling calibration on validation set.

    Temperature scaling improves the calibration of model predictions by learning
    a single scalar parameter (temperature) that scales the logits before applying
    sigmoid. This is particularly important for clinical applications where
    confidence estimates need to be reliable.

    The temperature is optimized to minimize the negative log-likelihood (NLL)
    on the validation set at the end of each validation epoch. The optimization
    is performed using LBFGS for efficient convergence.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
        On calibration of modern neural networks. ICML 2017.

    Attributes:
        enabled: Whether temperature scaling is enabled
        max_iter: Maximum iterations for LBFGS optimizer
        tolerance: Convergence tolerance for LBFGS
        optimize_every_n_epochs: Optimize temperature every N validation epochs
        validation_outputs: Collected outputs from validation batches
        epoch_counter: Counter for validation epochs
        logger: Python logger for console output
    """

    def __init__(
        self,
        enabled: bool = True,
        max_iter: int = 50,
        tolerance: float = 1e-5,
        optimize_every_n_epochs: int = 1
    ):
        """Initialize temperature scaling callback.

        Args:
            enabled: Whether to enable temperature scaling
            max_iter: Maximum iterations for LBFGS optimizer
            tolerance: Convergence tolerance for LBFGS
            optimize_every_n_epochs: Optimize temperature every N validation epochs
        """
        super().__init__()
        self.enabled = enabled
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.optimize_every_n_epochs = optimize_every_n_epochs
        self.validation_outputs: List[Dict[str, Any]] = []
        self.epoch_counter = 0
        self.logger = logging.getLogger(__name__)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Collect validation batch outputs for temperature optimization.

        Note: We collect raw logits (before temperature scaling) for optimization.
        """
        if not self.enabled or outputs is None or not isinstance(outputs, dict):
            return

        # Store only what we need for temperature optimization
        self.validation_outputs.append({
            'logits': outputs.get('logits'),  # Raw logits (numpy array)
            'gt': outputs.get('gt'),  # Ground truth (numpy array)
            'mask': outputs.get('mask')  # Window mask (numpy array)
        })

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Reset validation outputs at the start of each validation epoch."""
        self.validation_outputs = []

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Optimize temperature on validation set at the end of validation epoch."""
        if not self.enabled:
            return

        self.epoch_counter += 1

        # Only optimize every N epochs
        if self.epoch_counter % self.optimize_every_n_epochs != 0:
            self.validation_outputs = []
            return

        if not self.validation_outputs:
            self.logger.warning("No validation outputs collected for temperature scaling")
            self.validation_outputs = []
            return

        # Collect and concatenate all logits, ground truth, and masks
        import numpy as np

        all_logits = []
        all_gt = []
        all_masks = []

        for output in self.validation_outputs:
            if output['logits'] is not None and output['gt'] is not None:
                # Flatten each batch individually to handle variable N across batches
                all_logits.append(output['logits'].flatten())
                all_gt.append(output['gt'].flatten())
                if output.get('mask') is not None:
                    all_masks.append(output['mask'].flatten())

        if not all_logits:
            self.logger.warning("No valid logits collected for temperature scaling")
            self.validation_outputs = []
            return

        # Concatenate flattened arrays (all are now 1D)
        logits = np.concatenate(all_logits)
        gt = np.concatenate(all_gt)

        # Apply mask if available
        if all_masks:
            masks = np.concatenate(all_masks)
            valid_indices = masks > 0
            logits = logits[valid_indices]
            gt = gt[valid_indices]

        # Convert to tensors
        logits_tensor = torch.from_numpy(logits).float().to(pl_module.device)
        gt_tensor = torch.from_numpy(gt).float().to(pl_module.device)

        # Optimize temperature
        assert isinstance(pl_module.temperature, nn.Parameter)
        initial_temp = pl_module.temperature.item()
        optimized_temp = self._optimize_temperature(logits_tensor, gt_tensor, pl_module.temperature)

        # Update temperature in Lightning module
        pl_module.temperature.data = torch.tensor([optimized_temp], device=pl_module.device)
        pl_module.hparams["temperature"] = optimized_temp

        # Log the optimized temperature
        pl_module.log("temperature", optimized_temp, sync_dist=True, on_epoch=True)

        self.logger.info(
            f"Temperature scaling optimized: {initial_temp:.4f} -> {optimized_temp:.4f} "
            f"(epoch {trainer.current_epoch}, {len(logits)} samples)"
        )

        # Clear outputs to free memory
        self.validation_outputs = []

    def _optimize_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperature_param: nn.Parameter
    ) -> float:
        """Optimize temperature parameter to minimize NLL on validation set.

        Args:
            logits: Validation logits [N]
            labels: Ground truth labels [N]
            temperature_param: Current temperature parameter

        Returns:
            Optimized temperature value
        """
        # Create a new temperature tensor for optimization
        temperature = nn.Parameter(temperature_param.data.clone())
        temperature.requires_grad = True

        # Use LBFGS optimizer for efficient convergence
        optimizer = torch.optim.LBFGS([temperature], max_iter=self.max_iter, tolerance_change=self.tolerance)

        def closure():
            optimizer.zero_grad()
            # Apply temperature scaling and compute NLL
            scaled_logits = logits / temperature
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss

        # Optimize
        optimizer.step(closure)

        # Ensure temperature is positive and reasonable (between 0.1 and 10)
        optimized_temp = temperature.item()
        optimized_temp = max(0.1, min(10.0, optimized_temp))

        return optimized_temp


class CalibrationDiagnosticCallback(L.Callback):
    """Callback to log calibration metrics and reliability diagrams.

    This callback helps verify that temperature scaling is improving probability calibration.
    It computes and logs:
    - Expected Calibration Error (ECE): measures deviation from perfect calibration
    - Reliability diagrams: visualize calibration quality in TensorBoard
    - Per-bin statistics: confidence vs accuracy across probability bins

    Use this to verify temperature scaling is working correctly. Lower ECE = better calibration.

    Attributes:
        n_bins: Number of bins for calibration curve
        log_every_n_epochs: Frequency of logging (every N validation epochs)
        validation_outputs: Collected validation outputs
        logger: Python logger for console output
    """

    def __init__(self, n_bins: int = 10, log_every_n_epochs: int = 1):
        """Initialize calibration diagnostic callback.

        Args:
            n_bins: Number of bins for reliability diagram (default: 10)
            log_every_n_epochs: Log calibration metrics every N epochs (default: 1)
        """
        super().__init__()
        self.n_bins = n_bins
        self.log_every_n_epochs = log_every_n_epochs
        self.logger = logging.getLogger(__name__)
        self.validation_outputs: List[Dict[str, Any]] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"],
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Collect validation outputs for calibration analysis."""
        if outputs is not None and isinstance(outputs, dict):
            self.validation_outputs.append({
                'probs': outputs['probs'],
                'gt': outputs['gt'],
                'mask': outputs.get('mask')
            })

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Reset outputs at epoch start."""
        self.validation_outputs = []

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]
    ) -> None:
        """Compute and log calibration metrics at epoch end."""
        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self.validation_outputs = []
            return

        if not self.validation_outputs:
            return

        import numpy as np

        # Collect all probabilities and labels
        all_probs = []
        all_gt = []
        all_masks = []

        for output in self.validation_outputs:
            # Flatten each batch individually to handle variable N across batches
            all_probs.append(output['probs'].flatten())
            all_gt.append(output['gt'].flatten())
            if output.get('mask') is not None:
                all_masks.append(output['mask'].flatten())

        # Concatenate flattened arrays (all are now 1D)
        probs = np.concatenate(all_probs)
        gt = np.concatenate(all_gt)

        # Apply mask if available
        if all_masks:
            masks = np.concatenate(all_masks)
            valid_indices = masks > 0
            probs = probs[valid_indices]
            gt = gt[valid_indices]

        # Compute calibration metrics
        ece = self._compute_ece(probs, gt, self.n_bins)
        bin_stats = self._compute_bin_statistics(probs, gt, self.n_bins)

        # Log ECE
        pl_module.log("calibration/ece", ece, sync_dist=True, on_epoch=True)
        self.logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

        # Log reliability diagram to TensorBoard
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            self._log_reliability_diagram(trainer, bin_stats)

        # Clear outputs
        self.validation_outputs = []

    def _compute_ece(
        self,
        probs: 'np.ndarray',  # type: ignore
        labels: 'np.ndarray',  # type: ignore
        n_bins: int
    ) -> float:
        """Compute Expected Calibration Error.

        ECE measures the difference between predicted probabilities and actual frequencies.
        Lower is better (perfect calibration = 0).

        Args:
            probs: Predicted probabilities [N]
            labels: Ground truth labels [N]
            n_bins: Number of bins

        Returns:
            Expected calibration error (0 = perfect, higher = worse)
        """
        import numpy as np

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (probs > bin_lower) & (probs <= bin_upper)

            if np.sum(in_bin) > 0:
                # Average predicted probability in bin
                avg_confidence = np.mean(probs[in_bin])

                # Actual fraction of positives in bin
                avg_accuracy = np.mean(labels[in_bin])

                # Weighted contribution to ECE
                bin_weight = np.sum(in_bin) / len(probs)
                ece += bin_weight * np.abs(avg_confidence - avg_accuracy)

        return ece

    def _compute_bin_statistics(
        self,
        probs: 'np.ndarray',  # type: ignore
        labels: 'np.ndarray',  # type: ignore
        n_bins: int
    ) -> Dict[str, 'np.ndarray']:  # type: ignore
        """Compute per-bin statistics for reliability diagram.

        Args:
            probs: Predicted probabilities [N]
            labels: Ground truth labels [N]
            n_bins: Number of bins

        Returns:
            Dictionary with bin centers, confidences, accuracies, and counts
        """
        import numpy as np

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            count = np.sum(in_bin)

            if count > 0:
                confidence = np.mean(probs[in_bin])
                accuracy = np.mean(labels[in_bin])
            else:
                confidence = bin_centers[i]
                accuracy = 0.0

            bin_confidences.append(confidence)
            bin_accuracies.append(accuracy)
            bin_counts.append(count)

        return {
            'centers': bin_centers,
            'confidences': np.array(bin_confidences),
            'accuracies': np.array(bin_accuracies),
            'counts': np.array(bin_counts)
        }

    def _log_reliability_diagram(
        self,
        trainer: L.Trainer,
        bin_stats: Dict[str, 'np.ndarray']  # type: ignore
    ) -> None:
        """Log reliability diagram to TensorBoard.

        Creates a visual plot showing predicted probability vs actual frequency.
        Points closer to the diagonal indicate better calibration.

        Args:
            trainer: Lightning trainer instance
            bin_stats: Per-bin statistics (from _compute_bin_statistics)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image

            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot diagonal (perfect calibration)
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)

            # Plot actual calibration
            ax.plot(
                bin_stats['confidences'],
                bin_stats['accuracies'],
                'o-',
                label='Model Calibration',
                linewidth=2,
                markersize=8,
                color='#2ca02c'
            )

            # Shade the gap (calibration error)
            ax.fill_between(
                bin_stats['confidences'],
                bin_stats['accuracies'],
                bin_stats['confidences'],
                alpha=0.2,
                color='red',
                label='Calibration Gap'
            )

            # Add bin counts as text annotations
            for conf, acc, count in zip(bin_stats['confidences'], bin_stats['accuracies'], bin_stats['counts']):
                if count > 0:
                    ax.annotate(
                        f'n={int(count)}',
                        (conf, acc),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        alpha=0.7
                    )

            ax.set_xlabel('Predicted Probability (Confidence)', fontsize=12)
            ax.set_ylabel('Actual Frequency (Accuracy)', fontsize=12)
            ax.set_title('Reliability Diagram - Model Calibration', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlim((0, 1))
            ax.set_ylim((0, 1))
            ax.set_aspect('equal')

            # Convert plot to image tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            # Convert to tensor and log
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

            assert trainer.logger is not None
            trainer.logger.experiment.add_image(
                'calibration/reliability_diagram',
                img_tensor,
                global_step=trainer.current_epoch
            )

            plt.close(fig)
            buf.close()

        except Exception as e:
            self.logger.debug(f"Could not log reliability diagram: {e}")


class ProjectionLayerCallback(L.Callback):
    """Callback to handle channel projection layer freezing.

    This callback implements a strategy to freeze the projection layer
    after a certain number of epochs or when validation metrics reach a plateau.

    Attributes:
        freeze_after_epochs: Number of epochs after which to freeze the projection
        freeze_on_plateau: Whether to freeze on validation metric plateau
        patience: Number of epochs to wait for improvement before freezing on plateau
        best_val_metric: Best validation metric value seen so far
        epochs_without_improvement: Counter for epochs without improvement
        frozen: Flag indicating if projection layer is frozen
        logger: Custom logger instance
    """

    def __init__(
            self,
            freeze_after_epochs: int = 10,
            freeze_on_plateau: bool = True,
            patience: int = 3,
            monitor: str = 'val_pr_auc',
            mode: str = 'max'
    ):
        """Initialize the projection layer callback.

        Args:
            freeze_after_epochs: Number of epochs after which to freeze the projection
            freeze_on_plateau: Whether to freeze on validation metric plateau
            patience: Number of epochs to wait for improvement before freezing on plateau
            monitor: Metric to monitor for plateau detection
            mode: Mode for monitoring ('min' or 'max')
        """
        super().__init__()
        self.freeze_after_epochs = freeze_after_epochs
        self.freeze_on_plateau = freeze_on_plateau
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_val_metric = 0
        self.epochs_without_improvement = 0
        self.frozen = False
        self.logger = logging.getLogger(__name__)

    def setup(self, trainer: L.Trainer, pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"], stage: str) -> None:
        """Set up the callback at the start of training.

        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
            stage: Current stage of training
        """
        self.logger.info(f"Projection layer callback setup with freeze_after_epochs={self.freeze_after_epochs}, "
                         f"freeze_on_plateau={self.freeze_on_plateau}, patience={self.patience}")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"], stage: str) -> None:
        """Check whether to freeze the projection layer at the end of each epoch.

        Args:
            trainer: Lightning trainer instance
            pl_module: Lightning module instance
        """
        current_epoch = trainer.current_epoch

        # Check if we should freeze based on epoch count
        if not self.frozen and current_epoch >= self.freeze_after_epochs:
            self._freeze_projection(pl_module)
            return

        # Check if we should freeze based on validation plateau
        if self.freeze_on_plateau and not self.frozen:
            current_val_metric = trainer.callback_metrics.get(self.monitor, 0)

            # Convert to scalar if tensor
            if isinstance(current_val_metric, torch.Tensor):
                current_val_metric = current_val_metric.item()

            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Validation metric {self.monitor} plateaued for {self.patience} epochs.")
                self._freeze_projection(pl_module)

    def _freeze_projection(self, pl_module: Union["MEGSpikeDetector", "MEGUnsupervisedPretrainer"]) -> None:
        """Freeze the projection layer in the model.

        Args:
            pl_module: Lightning module instance
        """
        if hasattr(pl_module.model, 'freeze_projection'):
            attr = pl_module.model.freeze_projection
            if callable(attr):
                attr()
            elif isinstance(attr, torch.Tensor):
                attr.requires_grad_(False)
                self.logger.info("Projection tensor requires_grad set to False.")
            else:
                self.logger.warning("Model's 'freeze_projection' attribute is not callable or tensor.")
            self.frozen = True
            self.logger.info("Projection layer frozen after training phase.")
        else:
            self.logger.warning("Model does not have freeze_projection method.")

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dictionary for checkpointing.

        Returns:
            State dictionary
        """
        return {
            "best_val_metric": self.best_val_metric,
            "epochs_without_improvement": self.epochs_without_improvement,
            "frozen": self.frozen
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dictionary from a checkpoint.

        Args:
            state_dict: State dictionary
        """
        self.best_val_metric = state_dict.get("best_val_metric", 0)
        self.epochs_without_improvement = state_dict.get("epochs_without_improvement", 0)
        self.frozen = state_dict.get("frozen", False)


# Dictionary mapping callback names to their classes
CALLBACK_REGISTRY: Dict[str, Type[Callback]] = {
    "ModelCheckpoint": ModelCheckpoint,
    "EarlyStopping": EarlyStopping,
    "LearningRateMonitor": LearningRateMonitor,
    "TQDMProgressBar": TQDMProgressBar,
    "RichProgressBar": RichProgressBar,
    "RichModelSummary": RichModelSummary,
    "ProjectionLayerCallback": ProjectionLayerCallback,
    "GradientNormLogger": GradientNormLogger,
    "MetricsEvaluationCallback": MetricsEvaluationCallback,
    "TemperatureScalingCallback": TemperatureScalingCallback,
    "CalibrationDiagnosticCallback": CalibrationDiagnosticCallback,
    "ModelSummary": ModelSummary,
    "StochasticWeightAveraging": StochasticWeightAveraging,
}


def get_callback_class(callback_name: str) -> Type[Callback]:
    """Get callback class from registry by name.
    
    Args:
        callback_name: Name of the callback
        
    Returns:
        Callback class
        
    Raises:
        ValueError: If the callback name is not found in the registry
    """
    if callback_name not in CALLBACK_REGISTRY:
        raise ValueError(f"Callback '{callback_name}' not found in registry. Available callbacks: {list(CALLBACK_REGISTRY.keys())}")
    
    return CALLBACK_REGISTRY[callback_name]


def create_callback(callback_config: Dict[str, Any]) -> Callback:
    """Create a callback instance based on configuration.
    
    Args:
        callback_config: Callback configuration dictionary
        
    Returns:
        Instantiated callback
    """
    config = callback_config.copy()
    
    callback_name: str = config.pop("name", None)
    if callback_name is None:
        raise ValueError("Callback configuration must contain 'name' field")
    
    callback_class = get_callback_class(callback_name)
    callback_params: Dict[str, Any] = config.get(callback_name, {})
    
    if issubclass(callback_class, RichProgressBar):
        # Define the RichProgressBarTheme
        from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
        theme = RichProgressBarTheme(metrics_format='.5f')
        callback_params['theme'] = theme
    return callback_class(**callback_params)


def create_callbacks(callbacks_config: List[Dict[str, Any]], log_dir: Union[str, None] = None) -> List[Callback]:
    """Create multiple callback instances based on configuration.
    
    Args:
        callbacks_config: List of callback configuration dictionaries
        log_dir: Directory for logs and checkpoints (optional)
        
    Returns:
        List of instantiated callbacks
    """
    logger.info(f"Creating {len(callbacks_config)} callbacks")
    
    callbacks = []
    for i, callback_config in enumerate(callbacks_config):
        logger.info(f"Creating callback {i+1}/{len(callbacks_config)}: {callback_config.get('name', 'Unknown')}")
        
        # Handle ModelCheckpoint specially to set dirpath to log_dir if not specified
        if callback_config.get('name') == 'ModelCheckpoint' and log_dir:
            config_copy = callback_config.copy()
            checkpoint_params = config_copy.get('ModelCheckpoint', {})
            if 'dirpath' not in checkpoint_params:
                checkpoint_params['dirpath'] = log_dir
            config_copy['ModelCheckpoint'] = checkpoint_params
            callback = create_callback(config_copy)
        else:
            callback = create_callback(callback_config)
        
        callbacks.append(callback)
    
    logger.info(f"Successfully created {len(callbacks)} callbacks")
    return callbacks


def register_callback(name: str, callback_class: Type[Callback]) -> None:
    """Register a new callback in the registry.
    
    Args:
        name: Name for the callback
        callback_class: Callback class to register
    """
    CALLBACK_REGISTRY[name] = callback_class
