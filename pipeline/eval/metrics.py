#!/usr/bin/env python3
"""
Metrics system for MEG spike detection evaluation.

This module provides comprehensive evaluation capabilities including overall metrics
and patient-stratified analysis using pandas for efficient vectorized operations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix


class MetricsAggregator:
    """Metrics aggregator for overall and patient-specific evaluation.

    This class uses pandas DataFrames for efficient vectorized operations,
    eliminating redundant computations and sequence length bias.

    Features:
    - Overall dataset metrics with relaxed temporal tolerance
    - Patient-specific performance analysis
    - Patient group comparisons
    - Cross-patient variance statistics
    - Threshold optimization
    - Three evaluation modes:
      1. Default: All windows with equal weight
      2. Strided: Non-overlapping windows only (discards overlapping information)
      3. Weighted: Composite predictions averaging overlapping windows
    - Comprehensive reporting and CSV export
    - Returns DataFrame for TensorBoard visualization
    """

    def __init__(
        self,
        compute_relaxed: bool = True,
        threshold: float = 0.5,
        window_overlap: float = 0.5,
    ):
        """Initialize  metrics aggregator.

        Args:
            compute_relaxed: Whether to compute relaxed metrics allowing temporal tolerance
            threshold: Classification threshold for binary predictions
            window_overlap: Overlap ratio between consecutive windows
        """
        self.logger = logging.getLogger(__name__)
        self.compute_relaxed = compute_relaxed
        self.threshold = threshold
        self.window_overlap = window_overlap

        # Data storage - use list for efficient appending, convert to DataFrame on compute
        self.data_buffer: List[Dict[str, Any]] = []
        self.df: Optional[pd.DataFrame] = None
        self.composite_df: Optional[pd.DataFrame] = None  # Composite predictions for weighted metrics

        # Cache for computed metrics to avoid redundant calculations
        self._patient_metrics_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._group_metrics_cache: Optional[Dict[str, Dict[str, float]]] = None

    def reset(self):
        """Reset all metrics state."""
        self.data_buffer.clear()
        self.df = None
        self.composite_df = None
        self._patient_metrics_cache = None
        self._group_metrics_cache = None

    def update(
        self,
        probs: np.ndarray,
        gt: np.ndarray,
        mask: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        n_windows: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        losses: Optional[np.ndarray] = None,  # NEW: Per-window losses
    ):
        """Update metrics with batch data.

        Args:
            probs: Predicted probabilities (batch_size, n_windows)
            gt: Ground truth labels (batch_size, n_windows)
            mask: Window mask (batch_size, n_windows) - 1=valid, 0=padded
            batch_size: Batch size for sequence data
            n_windows: Number of windows per sequence
            metadata: List of metadata dictionaries containing patient/file information
            losses: Per-window losses (batch_size, n_windows) - optional
        """
        # Ensure inputs are numpy arrays
        probs = np.asarray(probs)
        gt = np.asarray(gt)

        # Validate probability ranges
        if np.any(probs < 0) or np.any(probs > 1):
            self.logger.warning(f"Probabilities outside [0,1] range detected: min={np.min(probs):.4f}, max={np.max(probs):.4f}")
            probs = np.clip(probs, 0.0, 1.0)

        # Ensure batch_size and n_windows are provided
        if batch_size is None or n_windows is None:
            raise ValueError("batch_size and n_windows must be provided for sequence-aware metrics")

        # Reshape from flat to [B, N_windows] if needed
        if probs.ndim == 1:
            probs = probs.reshape(batch_size, n_windows)
            gt = gt.reshape(batch_size, n_windows)
            if losses is not None:
                losses = losses.reshape(batch_size, n_windows)

        # Convert soft targets to hard targets
        gt = (gt >= 0.5).astype(int)

        # Compute valid indices mask
        mask_arr = (mask > 0) if mask is not None else np.ones((batch_size, n_windows), dtype=bool)

        # Build DataFrame records - one row per valid window
        for i in range(batch_size):
            valid_indices = mask_arr[i]
            if not np.any(valid_indices):
                continue

            # Extract metadata for this sample
            meta = metadata[i] if metadata is not None else {}
            patient_id = meta.get('patient_id', 'unknown')
            group = meta.get('group', 'Unknown')
            file_name = meta.get('file_name', 'unknown')

            # Get window positions for sorting (needed for file-level relaxed metrics)
            # Use start_window_idx (new unified field) with fallback to old field names
            start_window_idx = meta.get('start_window_idx', meta.get('start_position', 0))

            # Extract valid data
            valid_probs = probs[i, valid_indices]
            valid_gt = gt[i, valid_indices]
            valid_losses = losses[i, valid_indices] if losses is not None else np.zeros(valid_indices.sum())

            # Get window indices (within chunk)
            window_indices = np.where(valid_indices)[0]

            # Create records for each valid window
            for j, (window_idx, prob, label, loss) in enumerate(zip(window_indices, valid_probs, valid_gt, valid_losses)):
                # Global window position = start_window_idx + window_idx
                global_window_idx = start_window_idx + window_idx

                record = {
                    'prob': float(prob),
                    'gt': int(label),
                    'pred': int(prob >= self.threshold),
                    'loss': float(loss) if losses is not None else np.nan,
                    'patient_id': patient_id,
                    'group': group,
                    'file_name': file_name,
                    'sample_idx': len(self.data_buffer) + j,  # Unique sample ID
                    'window_idx': int(window_idx),  # Window index within chunk
                    'global_window_idx': int(global_window_idx),  # Global position in file
                    'batch_idx': i,
                    # Add confidence metrics
                    'confidence': float(abs(prob - 0.5)),  # Distance from decision boundary
                    'correct': int((prob >= self.threshold) == label),
                }
                self.data_buffer.append(record)

    def _build_dataframe(self):
        """Build pandas DataFrame from buffer if not already built."""
        if self.df is None and self.data_buffer:
            self.df = pd.DataFrame(self.data_buffer)
            # Sort by file and global window index for proper file-level operations
            self.df = self.df.sort_values(['file_name', 'global_window_idx']).reset_index(drop=True)

            # Add composite predictions for weighted metrics
            self._add_composite_predictions()

            # Add strided selection flag (non-overlapping windows)
            self._add_strided_selection()

            self.logger.info(f"Built DataFrame with {len(self.df)} windows from {self.df['patient_id'].nunique()} patients")

    def _add_temporal_weights_old(self):
        """[DEPRECATED] Add temporal contribution weight for each window.

        This is the old approach that weighted windows by their overlap count.
        Kept for reference but not used. The new approach uses composite predictions instead.

        For overlapping windows, each time point appears in multiple windows.
        The weight for each window is proportional to its unique temporal contribution.

        With overlap ratio r (e.g., 0.5 for 50% overlap):
        - Each time point appears in approximately 1/(1-r) windows
        - Each window's weight is approximately (1-r)

        This is computed per-file to handle boundaries correctly.
        """
        if self.df is None or self.df.empty:
            return

        weights = []

        for file_name, file_df in self.df.groupby('file_name'):
            file_df = file_df.sort_values('global_window_idx')
            n_windows = len(file_df)

            # For each window, count how many neighboring windows overlap with it
            file_weights = np.ones(n_windows, dtype=float)

            if self.window_overlap > 0 and n_windows > 1:
                # Calculate expected number of overlapping windows for each position
                # With overlap r, each time point appears in 1/(1-r) windows on average
                overlap_factor = 1.0 / (1.0 - self.window_overlap)

                # Weight each window by its inverse contribution
                # Interior windows have full overlap on both sides
                # Boundary windows have less overlap
                for i in range(n_windows):
                    # Count actual overlapping neighbors
                    # A window at position i overlaps with windows within the overlap range
                    # For 50% overlap, adjacent windows overlap
                    overlap_count = 1.0  # Count self

                    # Check overlap with previous windows
                    if i > 0:
                        overlap_count += self.window_overlap

                    # Check overlap with next windows
                    if i < n_windows - 1:
                        overlap_count += self.window_overlap

                    # Weight is inversely proportional to overlap count
                    file_weights[i] = 1.0 / overlap_count

            weights.extend(file_weights.tolist())

        self.df['temporal_weight'] = weights

        # Normalize weights to sum to the number of windows (for easier interpretation)
        total_weight = self.df['temporal_weight'].sum()
        if total_weight > 0:
            self.df['temporal_weight'] *= len(self.df) / total_weight

    def _add_composite_predictions(self):
        """Add composite predictions for non-overlapping temporal regions.

        Instead of weighting individual windows, this creates composite predictions by
        averaging overlapping windows for each strided position. This approach:
        1. Eliminates redundant counting (like strided metrics)
        2. Leverages information from all overlapping windows
        3. Creates weighted composite predictions for each temporal region

        For each strided window at position i:
        composite_prob[i] = (prob[i] + overlap * prob[i-1] + overlap * prob[i+1]) / (1 + 2*overlap)

        The composite predictions are stored in new columns for use in weighted metrics.
        """
        if self.df is None or self.df.empty:
            return

        composite_probs = []
        composite_gts = []
        composite_flags = []  # Which rows represent composite predictions
        
        overlap_weight = self.window_overlap
        # overlap_weight = 0.2  # Fixed overlap weight for stability

        for file_name, file_df in self.df.groupby('file_name'):
            file_df = file_df.sort_values('global_window_idx').reset_index(drop=True)
            n_windows = len(file_df)

            # Calculate stride based on overlap
            if self.window_overlap > 0:
                stride = int(1.0 / (1.0 - self.window_overlap))
            else:
                stride = 1

            # Extract arrays for this file
            probs = file_df['prob'].to_numpy()
            gts = file_df['gt'].to_numpy()

            # For each strided position, compute composite prediction
            for i in range(0, n_windows, stride):
                # Start with current window
                total_weight = 1.0
                weighted_prob = probs[i]
                weighted_gt = gts[i]

                # Add contribution from previous window if it exists
                if i > 0:
                    total_weight += overlap_weight
                    weighted_prob += overlap_weight * probs[i - 1]
                    weighted_gt += overlap_weight * gts[i - 1]

                # Add contribution from next window if it exists
                if i + 1 < n_windows:
                    total_weight += overlap_weight
                    weighted_prob += overlap_weight * probs[i + 1]
                    weighted_gt += overlap_weight * gts[i + 1]

                # Normalize
                composite_prob = weighted_prob / total_weight
                composite_gt = weighted_gt / total_weight

                composite_probs.append(composite_prob)
                composite_gts.append(composite_gt)
                composite_flags.append(file_df.index[i])  # Store original index

        # Create composite DataFrame
        if composite_flags:
            self.composite_df = self.df.loc[composite_flags].copy()
            self.composite_df['composite_prob'] = composite_probs
            self.composite_df['composite_gt'] = composite_gts
            self.composite_df['composite_pred'] = (np.array(composite_probs) >= self.threshold).astype(int)
        else:
            self.composite_df = None

    def _add_strided_selection(self):
        """Add boolean flag indicating if window should be included in strided (non-overlapping) evaluation.

        Selects windows with stride = window_size (no overlap) for independent evaluation.
        With 50% overlap, this selects every other window (stride of 2).
        """
        if self.df is None or self.df.empty:
            return

        strided_flags = []

        for file_name, file_df in self.df.groupby('file_name'):
            file_df = file_df.sort_values('global_window_idx').reset_index(drop=True)
            n_windows = len(file_df)

            # Calculate stride based on overlap
            # overlap = 0.5 means stride = 2 (every other window)
            # overlap = 0.75 means stride = 4 (every 4th window)
            if self.window_overlap > 0:
                stride = int(1.0 / (1.0 - self.window_overlap))
            else:
                stride = 1

            # Select windows at regular stride intervals
            file_flags = np.zeros(n_windows, dtype=bool)
            file_flags[::stride] = True

            strided_flags.extend(file_flags.tolist())

        self.df['strided'] = strided_flags

    def _compute_binary_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: Optional[float] = None, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute standard binary classification metrics.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            threshold: Classification threshold (uses self.threshold if None)
            sample_weight: Optional sample weights for weighted metrics

        Returns:
            Dictionary of computed metrics
        """
        if threshold is None:
            threshold = self.threshold

        y_pred = (y_prob >= threshold).astype(int)

        metrics = {}

        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight).ravel()

        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0

        # Confusion matrix components
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)

        # AUC metrics
        if len(np.unique(y_true)) > 1:  # Need both classes for AUC
            precision, recall, _ = precision_recall_curve(y_true, y_prob, sample_weight=sample_weight)
            metrics['pr_auc'] = auc(recall, precision)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
        else:
            metrics['pr_auc'] = 0.0
            metrics['roc_auc'] = 0.0

        return metrics

    def _compute_relaxed_metrics_per_file(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute relaxed metrics with temporal tolerance on file-level contiguous data.

        Args:
            df: DataFrame with columns: prob, gt, file_name, global_window_idx

        Returns:
            Dictionary of relaxed metrics
        """
        if df.empty:
            return {}

        relaxed_tp = 0
        relaxed_fp = 0
        relaxed_fn = 0

        # Group by file to ensure contiguous data
        for _file_name, file_df in df.groupby('file_name'):
            # Sort by global window index to ensure temporal ordering
            file_df = file_df.sort_values('global_window_idx').reset_index(drop=True)

            true = file_df['gt'].to_numpy()
            pred = (file_df['prob'].to_numpy() >= self.threshold).astype(int)

            # Create a new array for relaxed predictions
            relaxed_pred = np.zeros_like(pred)

            # Apply relaxation rules sequentially for each time point
            for ind in range(len(true)):
                # Calculate window bounds using min/max to handle boundaries
                left_bound = max(0, ind - 1)
                right_bound = min(len(true) - 1, ind + 1)

                # Check true positive case (Mark as TP if prediction exists in neighborhood)
                if true[ind] == 1:
                    # Current, left, or right window is predicted as positive
                    if pred[ind] == 1 or pred[left_bound] == 1 or pred[right_bound] == 1:
                        relaxed_pred[ind] = 1
                # Check isolated false positive case (keep it as a FP)
                elif pred[ind] == 1 and true[left_bound] == 0 and true[right_bound] == 0:
                    relaxed_pred[ind] = 1

            # Accumulate metrics for this file
            relaxed_tp += np.sum((true == 1) & (relaxed_pred == 1))
            relaxed_fp += np.sum((true == 0) & (relaxed_pred == 1))
            relaxed_fn += np.sum((true == 1) & (relaxed_pred == 0))

        # Compute relaxed metrics
        metrics = {}
        metrics['relaxed_precision'] = relaxed_tp / (relaxed_tp + relaxed_fp) if (relaxed_tp + relaxed_fp) > 0 else 0.0
        metrics['relaxed_recall'] = relaxed_tp / (relaxed_tp + relaxed_fn) if (relaxed_tp + relaxed_fn) > 0 else 0.0
        metrics['relaxed_f1'] = 2 * metrics['relaxed_precision'] * metrics['relaxed_recall'] / (metrics['relaxed_precision'] + metrics['relaxed_recall']) if (metrics['relaxed_precision'] + metrics['relaxed_recall']) > 0 else 0.0
        metrics['relaxed_tp'] = int(relaxed_tp)
        metrics['relaxed_fp'] = int(relaxed_fp)
        metrics['relaxed_fn'] = int(relaxed_fn)

        return metrics

    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """Optimize threshold based on F1 score.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, best_f1_score)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        return best_threshold, best_f1

    def compute_overall_metrics(self, optimize_threshold: bool = False) -> Dict[str, float]:
        """Compute overall dataset metrics.

        Args:
            optimize_threshold: Whether to optimize threshold based on F1 score

        Returns:
            Dictionary of overall metrics including default, strided, and weighted variants
        """
        self._build_dataframe()

        if self.df is None or self.df.empty:
            self.logger.warning("No data available for overall metrics computation")
            return {}

        # Extract all probabilities and ground truth (equal weight per window)
        all_probs = self.df['prob'].to_numpy()
        all_gt = self.df['gt'].to_numpy()

        # Optimize threshold if requested
        if optimize_threshold:
            self.threshold, _ = self._optimize_threshold(all_gt, all_probs)
            # Update predictions with new threshold
            self.df['pred'] = (self.df['prob'] >= self.threshold).astype(int)

        # 1. Compute standard metrics (DEFAULT - all windows with equal weight)
        metrics = self._compute_binary_metrics(all_gt, all_probs)

        # Add dataset statistics
        metrics['n_samples'] = len(self.df)
        metrics['n_positive'] = int(all_gt.sum())
        metrics['n_negative'] = int((1 - all_gt).sum())
        metrics['positive_rate'] = float(all_gt.mean())

        # 2. Compute STRIDED metrics (non-overlapping windows only)
        if 'strided' in self.df.columns:
            strided_df = self.df[self.df['strided']]
            if not strided_df.empty:
                strided_probs = strided_df['prob'].to_numpy()
                strided_gt = strided_df['gt'].to_numpy()
                strided_metrics = self._compute_binary_metrics(strided_gt, strided_probs)

                # Add strided metrics with prefix
                for key, value in strided_metrics.items():
                    if key not in ['tp', 'fp', 'tn', 'fn']:  # Skip confusion matrix for brevity
                        metrics[f'strided_{key}'] = value

                metrics['strided_n_samples'] = len(strided_df)
                metrics['strided_n_positive'] = int(strided_gt.sum())

        # 3. Compute WEIGHTED metrics (composite predictions from overlapping windows)
        if self.composite_df is not None and not self.composite_df.empty:
            composite_probs = self.composite_df['composite_prob'].to_numpy()
            composite_gt = self.composite_df['composite_gt'].to_numpy()
            # Convert soft composite_gt to hard labels
            composite_gt_binary = (composite_gt >= 0.5).astype(int)

            weighted_metrics = self._compute_binary_metrics(composite_gt_binary, composite_probs)

            # Add weighted metrics with prefix
            for key, value in weighted_metrics.items():
                if key not in ['tp', 'fp', 'tn', 'fn']:  # Skip confusion matrix for brevity
                    metrics[f'weighted_{key}'] = value

            # Weighted sample counts
            metrics['weighted_n_samples'] = len(self.composite_df)
            metrics['weighted_n_positive'] = int(composite_gt_binary.sum())

        # Compute relaxed metrics if enabled (on default, strided, and weighted)
        if self.compute_relaxed:
            # Default relaxed
            relaxed_metrics = self._compute_relaxed_metrics_per_file(self.df)
            metrics.update(relaxed_metrics)

            # Strided relaxed
            if 'strided' in self.df.columns:
                strided_df = self.df[self.df['strided']]
                if not strided_df.empty:
                    strided_relaxed = self._compute_relaxed_metrics_per_file(strided_df)
                    for key, value in strided_relaxed.items():
                        metrics[f'strided_{key}'] = value

        return metrics

    def compute_patient_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each patient individually using vectorized operations.

        Returns:
            Dictionary mapping patient IDs to their metrics (includes default, strided, and weighted)
        """
        self._build_dataframe()

        if self.df is None or self.df.empty:
            return {}

        # Use cached results if available
        if self._patient_metrics_cache is not None:
            return self._patient_metrics_cache

        patient_metrics = {}

        for patient_id, patient_df in self.df.groupby('patient_id'):
            if patient_df.empty:
                continue

            patient_probs = patient_df['prob'].to_numpy()
            patient_gt = patient_df['gt'].to_numpy()

            # Optimize threshold for this patient
            if len(np.unique(patient_gt)) > 1:  # Need both classes for optimization
                best_threshold, best_f1 = self._optimize_threshold(patient_gt, patient_probs)
            else:
                best_threshold = self.threshold
                best_f1 = 0.0

            # 1. Compute DEFAULT metrics with global threshold
            metrics = self._compute_binary_metrics(patient_gt, patient_probs)

            # Compute metrics with patient-specific optimal threshold
            optimal_metrics = self._compute_binary_metrics(patient_gt, patient_probs, best_threshold)

            # Add threshold information
            metrics['best_threshold'] = best_threshold
            metrics['best_f1'] = best_f1
            metrics['threshold_diff'] = best_threshold - self.threshold

            # Add optimal threshold metrics with prefix
            for key, value in optimal_metrics.items():
                metrics[f'optimal_{key}'] = value

            # Add patient statistics
            metrics['n_samples'] = len(patient_df)
            metrics['n_positive'] = int(patient_gt.sum())
            metrics['positive_rate'] = float(patient_gt.mean())

            # 2. Compute STRIDED metrics
            if 'strided' in patient_df.columns:
                strided_patient_df = patient_df[patient_df['strided']]
                if not strided_patient_df.empty:
                    strided_probs = strided_patient_df['prob'].to_numpy()
                    strided_gt = strided_patient_df['gt'].to_numpy()
                    strided_metrics = self._compute_binary_metrics(strided_gt, strided_probs)

                    for key, value in strided_metrics.items():
                        if key not in ['tp', 'fp', 'tn', 'fn']:
                            metrics[f'strided_{key}'] = value

                    metrics['strided_n_samples'] = len(strided_patient_df)
                    metrics['strided_n_positive'] = int(strided_gt.sum())

            # 3. Compute WEIGHTED metrics (composite predictions)
            if self.composite_df is not None:
                patient_composite_df = self.composite_df[self.composite_df['patient_id'] == patient_id]
                if not patient_composite_df.empty:
                    composite_probs = patient_composite_df['composite_prob'].to_numpy()
                    composite_gt = patient_composite_df['composite_gt'].to_numpy()
                    composite_gt_binary = (composite_gt >= 0.5).astype(int)

                    weighted_metrics = self._compute_binary_metrics(composite_gt_binary, composite_probs)

                    for key, value in weighted_metrics.items():
                        if key not in ['tp', 'fp', 'tn', 'fn']:
                            metrics[f'weighted_{key}'] = value

                    metrics['weighted_n_samples'] = len(patient_composite_df)
                    metrics['weighted_n_positive'] = int(composite_gt_binary.sum())

            # Add loss statistics if available
            if 'loss' in patient_df.columns:
                metrics['mean_loss'] = float(patient_df['loss'].mean())
                metrics['std_loss'] = float(patient_df['loss'].std())

            # Add relaxed metrics if enabled
            if self.compute_relaxed:
                # Default relaxed
                relaxed_metrics = self._compute_relaxed_metrics_per_file(patient_df)
                metrics.update(relaxed_metrics)

                # Strided relaxed
                if 'strided' in patient_df.columns:
                    strided_patient_df = patient_df[patient_df['strided']]
                    if not strided_patient_df.empty:
                        strided_relaxed = self._compute_relaxed_metrics_per_file(strided_patient_df)
                        for key, value in strided_relaxed.items():
                            metrics[f'strided_{key}'] = value

            patient_metrics[patient_id] = metrics

        # Cache results
        self._patient_metrics_cache = patient_metrics
        return patient_metrics

    def compute_group_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics grouped by patient groups using vectorized operations.

        Returns:
            Dictionary mapping group names to group metrics (includes default, strided, and weighted)
        """
        self._build_dataframe()

        if self.df is None or self.df.empty:
            return {}

        # Use cached results if available
        if self._group_metrics_cache is not None:
            return self._group_metrics_cache

        group_metrics = {}

        for group, group_df in self.df.groupby('group'):
            if group_df.empty:
                continue

            group_probs = group_df['prob'].to_numpy()
            group_gt = group_df['gt'].to_numpy()

            # 1. Compute DEFAULT group metrics
            metrics = self._compute_binary_metrics(group_gt, group_probs)

            # Add group statistics
            metrics['n_samples'] = len(group_df)
            metrics['n_patients'] = group_df['patient_id'].nunique()
            metrics['n_positive'] = int(group_gt.sum())
            metrics['positive_rate'] = float(group_gt.mean())

            # 2. Compute STRIDED metrics
            if 'strided' in group_df.columns:
                strided_group_df = group_df[group_df['strided']]
                if not strided_group_df.empty:
                    strided_probs = strided_group_df['prob'].to_numpy()
                    strided_gt = strided_group_df['gt'].to_numpy()
                    strided_metrics = self._compute_binary_metrics(strided_gt, strided_probs)

                    for key, value in strided_metrics.items():
                        if key not in ['tp', 'fp', 'tn', 'fn']:
                            metrics[f'strided_{key}'] = value

                    metrics['strided_n_samples'] = len(strided_group_df)
                    metrics['strided_n_positive'] = int(strided_gt.sum())

            # 3. Compute WEIGHTED metrics (composite predictions)
            if self.composite_df is not None:
                group_composite_df = self.composite_df[self.composite_df['group'] == group]
                if not group_composite_df.empty:
                    composite_probs = group_composite_df['composite_prob'].to_numpy()
                    composite_gt = group_composite_df['composite_gt'].to_numpy()
                    composite_gt_binary = (composite_gt >= 0.5).astype(int)

                    weighted_metrics = self._compute_binary_metrics(composite_gt_binary, composite_probs)

                    for key, value in weighted_metrics.items():
                        if key not in ['tp', 'fp', 'tn', 'fn']:
                            metrics[f'weighted_{key}'] = value

                    metrics['weighted_n_samples'] = len(group_composite_df)
                    metrics['weighted_n_positive'] = int(composite_gt_binary.sum())

            # Add loss statistics if available
            if 'loss' in group_df.columns:
                metrics['mean_loss'] = float(group_df['loss'].mean())
                metrics['std_loss'] = float(group_df['loss'].std())

            # Add relaxed metrics
            if self.compute_relaxed:
                # Default relaxed
                relaxed_metrics = self._compute_relaxed_metrics_per_file(group_df)
                metrics.update(relaxed_metrics)

                # Strided relaxed
                if 'strided' in group_df.columns:
                    strided_group_df = group_df[group_df['strided']]
                    if not strided_group_df.empty:
                        strided_relaxed = self._compute_relaxed_metrics_per_file(strided_group_df)
                        for key, value in strided_relaxed.items():
                            metrics[f'strided_{key}'] = value

            group_metrics[group] = metrics

        # Cache results
        self._group_metrics_cache = group_metrics
        return group_metrics

    def compute_cross_patient_statistics(self) -> Dict[str, float]:
        """Compute statistics across patients using cached patient metrics.

        Returns:
            Dictionary of cross-patient statistics (includes default, strided, and weighted)
        """
        patient_metrics = self.compute_patient_metrics()

        if not patient_metrics:
            return {}

        # Extract metrics for analysis - include default, strided, and weighted variants
        stats = {}

        # Base metric names
        base_metrics = ['pr_auc', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']

        # Prefixes for different variants
        prefixes = ['', 'strided_', 'weighted_']

        for prefix in prefixes:
            for metric_name in base_metrics:
                full_metric_name = f'{prefix}{metric_name}'
                values = [metrics.get(full_metric_name, 0.0) for metrics in patient_metrics.values() if full_metric_name in metrics]

                if values:
                    stats[f'{full_metric_name}_mean'] = np.mean(values)
                    stats[f'{full_metric_name}_std'] = np.std(values)
                    stats[f'{full_metric_name}_min'] = np.min(values)
                    stats[f'{full_metric_name}_max'] = np.max(values)
                    stats[f'{full_metric_name}_range'] = stats[f'{full_metric_name}_max'] - stats[f'{full_metric_name}_min']

        return stats

    def compute_calibration_analysis(self) -> Dict[str, float]:
        """Analyze calibration issues by comparing patient-specific vs global thresholds.

        Returns:
            Dictionary of calibration analysis metrics
        """
        patient_metrics = self.compute_patient_metrics()

        if not patient_metrics:
            return {}

        # Extract threshold information
        best_thresholds = [metrics.get('best_threshold', self.threshold) for metrics in patient_metrics.values()]
        threshold_diffs = [metrics.get('threshold_diff', 0.0) for metrics in patient_metrics.values()]
        f1_improvements = [metrics.get('optimal_f1', 0.0) - metrics.get('f1', 0.0) for metrics in patient_metrics.values()]

        # Calibration statistics
        calibration_stats = {}

        # Threshold distribution analysis
        calibration_stats['threshold_mean'] = np.mean(best_thresholds)
        calibration_stats['threshold_std'] = np.std(best_thresholds)
        calibration_stats['threshold_min'] = np.min(best_thresholds)
        calibration_stats['threshold_max'] = np.max(best_thresholds)
        calibration_stats['threshold_range'] = calibration_stats['threshold_max'] - calibration_stats['threshold_min']

        # Global threshold calibration
        calibration_stats['global_threshold'] = self.threshold
        calibration_stats['mean_threshold_diff'] = np.mean(threshold_diffs)
        calibration_stats['std_threshold_diff'] = np.std(threshold_diffs)

        # Performance impact analysis
        calibration_stats['mean_f1_improvement'] = np.mean(f1_improvements)
        calibration_stats['std_f1_improvement'] = np.std(f1_improvements)
        calibration_stats['max_f1_improvement'] = np.max(f1_improvements)

        # Calibration quality indicators
        calibration_stats['patients_above_global_threshold'] = np.sum(np.array(best_thresholds) > self.threshold)
        calibration_stats['patients_below_global_threshold'] = np.sum(np.array(best_thresholds) < self.threshold)
        calibration_stats['well_calibrated_patients'] = np.sum(np.abs(threshold_diffs) < 0.1)  # Within 0.1 of global threshold
        calibration_stats['poorly_calibrated_patients'] = np.sum(np.abs(threshold_diffs) > 0.3)  # More than 0.3 away from global

        # Relative calibration metrics
        total_patients = len(patient_metrics)
        if total_patients > 0:
            calibration_stats['pct_well_calibrated'] = calibration_stats['well_calibrated_patients'] / total_patients * 100
            calibration_stats['pct_poorly_calibrated'] = calibration_stats['poorly_calibrated_patients'] / total_patients * 100
            calibration_stats['pct_above_global'] = calibration_stats['patients_above_global_threshold'] / total_patients * 100
            calibration_stats['pct_below_global'] = calibration_stats['patients_below_global_threshold'] / total_patients * 100

        return calibration_stats

    def compute(self, optimize_threshold: bool = False) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all metrics (overall, patient-specific, and cross-patient statistics).

        Single-pass computation with caching to avoid redundancy.

        Args:
            optimize_threshold: Whether to optimize threshold based on F1 score

        Returns:
            Dictionary containing all computed metrics
        """
        results = {}

        # Build DataFrame once
        self._build_dataframe()

        # Overall metrics
        overall_metrics = self.compute_overall_metrics(optimize_threshold)
        results.update(overall_metrics)

        # Patient-specific metrics (computed once and cached)
        if self.df is not None and 'patient_id' in self.df.columns:
            results['patient_metrics'] = self.compute_patient_metrics()
            results['group_metrics'] = self.compute_group_metrics()
            results['cross_patient_stats'] = self.compute_cross_patient_statistics()
            results['calibration_analysis'] = self.compute_calibration_analysis()

        return results

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the underlying DataFrame for custom analysis or TensorBoard logging.

        Returns:
            DataFrame with all window-level data and predictions
        """
        self._build_dataframe()
        return self.df

    def format_all_metrics(self, metrics: Dict[str, Union[float, Dict[str, Any]]]) -> str:
        """Format all metrics for display.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Formatted string representation
        """
        lines = []

        # Overall metrics
        lines.append("Overall Metrics:")
        lines.append("-" * 20)
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:>15}: {value:.4f}")

        # Cross-patient statistics
        if 'cross_patient_stats' in metrics and isinstance(metrics['cross_patient_stats'], dict):
            lines.append("\nCross-Patient Statistics:")
            lines.append("-" * 25)
            stats = metrics['cross_patient_stats']
            for metric_base in ['pr_auc', 'roc_auc', 'f1', 'accuracy']:
                if f'{metric_base}_mean' in stats and f'{metric_base}_std' in stats:
                    lines.append(f"{metric_base:>10}: {stats[f'{metric_base}_mean']:.4f} ± {stats[f'{metric_base}_std']:.4f}")

        return "\n".join(lines)

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report.

        Returns:
            Formatted comprehensive report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("COMPREHENSIVE MEG SPIKE DETECTION EVALUATION")
        lines.append("=" * 60)

        # Compute all metrics
        all_metrics = self.compute()

        # Overall performance
        lines.append("\nOVERALL PERFORMANCE:")
        lines.append("-" * 30)
        for metric, value in all_metrics.items():
            if isinstance(value, float):
                lines.append(f"{metric:>15}: {value:.4f}")

        # Cross-patient analysis
        if 'cross_patient_stats' in all_metrics:
            lines.append("\nCROSS-PATIENT ANALYSIS:")
            lines.append("-" * 30)
            stats = all_metrics['cross_patient_stats']
            assert isinstance(stats, dict), "Expected cross_patient_stats to be a dictionary"
            for metric_base in ['pr_auc', 'roc_auc', 'f1']:
                if f'{metric_base}_mean' in stats:
                    lines.append(f"{metric_base.upper():>10}: {stats[f'{metric_base}_mean']:.4f} ± {stats[f'{metric_base}_std']:.4f}")
                    lines.append(f"{'Range':>10}: [{stats[f'{metric_base}_min']:.4f}, {stats[f'{metric_base}_max']:.4f}]")

        # Calibration analysis
        if 'calibration_analysis' in all_metrics and isinstance(all_metrics['calibration_analysis'], dict):
            lines.append("\nCALIBRATION ANALYSIS:")
            lines.append("-" * 30)
            calib = all_metrics['calibration_analysis']
            lines.append(f"Global threshold: {calib.get('global_threshold', 0.0):.4f}")
            lines.append(f"Optimal threshold: {calib.get('threshold_mean', 0.0):.4f} ± {calib.get('threshold_std', 0.0):.4f}")
            lines.append(f"Threshold range: [{calib.get('threshold_min', 0.0):.4f}, {calib.get('threshold_max', 0.0):.4f}]")
            lines.append(f"Well calibrated: {calib.get('pct_well_calibrated', 0.0):.1f}% of patients")
            lines.append(f"Poorly calibrated: {calib.get('pct_poorly_calibrated', 0.0):.1f}% of patients")
            lines.append(f"F1 improvement: {calib.get('mean_f1_improvement', 0.0):.4f} ± {calib.get('std_f1_improvement', 0.0):.4f}")

        # Group performance
        if 'group_metrics' in all_metrics and isinstance(all_metrics['group_metrics'], dict):
            lines.append("\nGROUP PERFORMANCE:")
            lines.append("-" * 30)
            for group, group_metrics in all_metrics['group_metrics'].items():
                if isinstance(group_metrics, dict):
                    lines.append(f"\n{group}:")
                    for metric, value in group_metrics.items():
                        if isinstance(value, float):
                            lines.append(f"  {metric:>12}: {value:.4f}")

        # Dataset summary
        if self.df is not None:
            lines.append(f"\nEVALUATION SUMMARY:")
            lines.append("-" * 30)
            lines.append(f"Total windows: {len(self.df)}")
            lines.append(f"Total patients: {self.df['patient_id'].nunique()}")
            lines.append(f"Total files: {self.df['file_name'].nunique()}")

            group_counts = self.df.groupby('group')['patient_id'].nunique()
            for group, count in group_counts.items():
                lines.append(f"  {group}: {count} patients")

        return "\n".join(lines)

    def save_detailed_results(self, output_dir: str, timestamp: str):
        """Save detailed results to CSV files.

        Args:
            output_dir: Directory to save results
            timestamp: Timestamp string for file naming
        """
        import os

        # Save patient metrics
        patient_metrics = self.compute_patient_metrics()
        if patient_metrics:
            rows = []

            for patient_id, metrics in patient_metrics.items():
                # Get group from DataFrame
                patient_rows = self.df[self.df['patient_id'] == patient_id] if self.df is not None else pd.DataFrame()
                group = patient_rows['group'].iloc[0] if not patient_rows.empty else 'Unknown'

                row = {
                    'patient_id': patient_id,
                    'group': group,
                }
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                rows.append(row)

            df = pd.DataFrame(rows)
            patient_csv_path = os.path.join(output_dir, f"patient_metrics_{timestamp}.csv")
            df.to_csv(patient_csv_path, index=False)
            self.logger.info(f"Patient metrics saved to {patient_csv_path}")

        # Save calibration analysis
        calibration_analysis = self.compute_calibration_analysis()
        if calibration_analysis:
            calib_df = pd.DataFrame([calibration_analysis])
            calib_csv_path = os.path.join(output_dir, f"calibration_analysis_{timestamp}.csv")
            calib_df.to_csv(calib_csv_path, index=False)
            self.logger.info(f"Calibration analysis saved to {calib_csv_path}")

        # Save group metrics
        group_metrics = self.compute_group_metrics()
        if group_metrics:
            group_rows = []
            for group, metrics in group_metrics.items():
                row: Dict[str, Union[str, float]] = {'group': group}
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                group_rows.append(row)

            group_df = pd.DataFrame(group_rows)
            group_csv_path = os.path.join(output_dir, f"group_metrics_{timestamp}.csv")
            group_df.to_csv(group_csv_path, index=False)
            self.logger.info(f"Group metrics saved to {group_csv_path}")

        # NEW: Save window-level DataFrame for detailed analysis
        if self.df is not None:
            window_csv_path = os.path.join(output_dir, f"window_level_predictions_{timestamp}.csv")
            self.df.to_csv(window_csv_path, index=False)
            self.logger.info(f"Window-level predictions saved to {window_csv_path}")