#!/usr/bin/env python3
"""Statistical significance analysis for HBIOT pipeline outputs.

Standalone script — no imports from pipeline/.
Consumes patient_metrics_*.csv and optionally window_level_predictions_*.csv
to run non-parametric and parametric significance tests across configs/folds.

Usage:
    python scripts/statistical_analysis.py --logs-dir logs/
    python scripts/statistical_analysis.py --logs-dir logs/ --filter "BIOT*" --plots
    python scripts/statistical_analysis.py --logs-dir logs/ --list-configs
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Discovery & Loading
# ---------------------------------------------------------------------------

DEFAULT_FOLD_PATTERN = r"fold[_-]?(\d+)"
DEFAULT_METRICS = ["f1", "pr_auc", "roc_auc", "relaxed_f1"]


def discover_experiments(
    logs_dir: Path,
    fold_pattern: str = DEFAULT_FOLD_PATTERN,
    filters: list[str] | None = None,
) -> list[dict]:
    """Walk *logs_dir* and find every directory containing patient_metrics_*.csv.

    Returns a list of dicts with keys: config, fold, path, rel_path.
    """
    fold_re = re.compile(fold_pattern, re.IGNORECASE)
    experiments: list[dict] = []

    for csv_path in sorted(logs_dir.rglob("patient_metrics_*.csv")):
        directory = csv_path.parent
        rel = directory.relative_to(logs_dir)
        rel_str = str(rel)

        # Optional glob filtering
        if filters:
            if not any(fnmatch.fnmatch(rel_str, f) for f in filters):
                continue

        # Extract fold from path components
        parts = list(rel.parts)
        fold_id: str | None = None
        config_parts: list[str] = []
        for part in parts:
            m = fold_re.search(part)
            if m and fold_id is None:
                fold_id = m.group(1)
                # Strip the fold portion from this component
                stripped = fold_re.sub("", part).strip("-_ ")
                if stripped:
                    config_parts.append(stripped)
            else:
                config_parts.append(part)

        config_id = "/".join(config_parts) if config_parts else rel_str

        experiments.append(
            dict(
                config=config_id,
                fold=fold_id,
                path=directory,
                rel_path=rel_str,
            )
        )

    return experiments


def find_latest_csv(directory: Path, prefix: str) -> Path | None:
    """Return the most recent (lexicographic) CSV matching *prefix*_*.csv."""
    candidates = sorted(directory.glob(f"{prefix}_*.csv"))
    return candidates[-1] if candidates else None


def load_patient_metrics(experiments: list[dict]) -> pd.DataFrame:
    """Build a unified DataFrame: (patient_id, group, fold, config, ...metrics)."""
    frames: list[pd.DataFrame] = []
    for exp in experiments:
        csv_path = find_latest_csv(exp["path"], "patient_metrics")
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        df["config"] = exp["config"]
        df["fold"] = exp["fold"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_window_predictions(experiments: list[dict]) -> pd.DataFrame:
    """Build a unified DataFrame of window-level predictions."""
    frames: list[pd.DataFrame] = []
    for exp in experiments:
        csv_path = find_latest_csv(exp["path"], "window_level_predictions")
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        df["config"] = exp["config"]
        df["fold"] = exp["fold"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples."""
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) < 10:
        return np.nan, np.nan
    res = stats.wilcoxon(diff)
    return float(res.statistic), float(res.pvalue)


def permutation_test(
    a: np.ndarray, b: np.ndarray, n_perms: int = 10000
) -> tuple[float, float]:
    """Two-sided permutation test for paired mean difference."""
    diff = a - b
    observed = np.mean(diff)
    n = len(diff)
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=(n_perms, n))
    perm_means = np.mean(diff * signs, axis=1)
    count = int(np.sum(np.abs(perm_means) >= abs(observed)))
    p = (count + 1) / (n_perms + 1)
    return float(observed), float(p)


def corrected_paired_ttest(
    fold_a: np.ndarray,
    fold_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[float, float]:
    """Nadeau-Bengio corrected paired t-test for cross-validation.

    Corrects the variance estimate to account for training-set overlap between
    folds.  See Nadeau & Bengio (2003), *Machine Learning* 52(3).
    """
    diff = fold_a - fold_b
    k = len(diff)
    if k < 2:
        return np.nan, np.nan
    mean_d = np.mean(diff)
    var_d = np.var(diff, ddof=1)
    # Correction factor: (1/k + n_test/n_train)
    correction = 1.0 / k + n_test / n_train
    t_stat = mean_d / np.sqrt(correction * var_d) if var_d > 0 else np.nan
    if np.isnan(t_stat):
        return np.nan, np.nan
    p = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    return float(t_stat), float(p)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for mean(a - b)."""
    diff = a - b
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = np.mean(diff[indices], axis=1)
    alpha = 1 - ci_level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, float(np.mean(diff)), hi


def delong_test(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, float]:
    """DeLong's test for comparing two ROC AUCs on the same sample.

    Compact vectorized implementation using the structural-components
    approach from Sun & Xu (2014).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan

    def _structural_components(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # V10[i] = fraction of negatives scored below positive i
        # V01[j] = fraction of positives scored above negative j
        s_pos = scores[pos]
        s_neg = scores[neg]
        # (m, 1) vs (1, n) → (m, n), average over negatives (axis=1) → (m,)
        v10 = np.mean(s_pos[:, None] > s_neg[None, :], axis=1) + 0.5 * np.mean(s_pos[:, None] == s_neg[None, :], axis=1)
        # (n, 1) vs (1, m) → (n, m), average over positives (axis=1) → (n,)
        v01 = np.mean(s_neg[:, None] < s_pos[None, :], axis=1) + 0.5 * np.mean(s_neg[:, None] == s_pos[None, :], axis=1)
        return v10, v01

    v10_a, v01_a = _structural_components(scores_a)
    v10_b, v01_b = _structural_components(scores_b)

    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    # Covariance matrix of (AUC_A, AUC_B) via structural components
    s10 = np.cov(v10_a, v10_b)  # 2x2
    s01 = np.cov(v01_a, v01_b)  # 2x2

    S = s10 / m + s01 / n
    diff = auc_a - auc_b
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return np.nan, np.nan
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def mcnemar_test(
    pred_a: np.ndarray, pred_b: np.ndarray, gt: np.ndarray
) -> tuple[float, float]:
    """McNemar's test on paired binary predictions."""
    correct_a = pred_a == gt
    correct_b = pred_b == gt
    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c < 10:
        return np.nan, np.nan
    # Continuity-corrected
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(stat, df=1)
    return float(stat), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = a - b
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta (non-parametric effect size) with magnitude label."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan, "undefined"
    more = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    delta = (more - less) / (n_a * n_b)
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag


def holm_bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> list[tuple[float, bool]]:
    """Holm-Bonferroni step-down correction.

    Returns list of (adjusted_p, significant) tuples in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return [(adjusted[i], adjusted[i] <= alpha) for i in range(n)]


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def compute_inter_subject_variability(
    df: pd.DataFrame, metrics: list[str], config: str
) -> dict:
    """CV, Kruskal-Wallis across patient groups, descriptive stats."""
    sub = df[df["config"] == config].copy()
    result: dict = {"config": config, "metrics": {}}

    for metric in metrics:
        if metric not in sub.columns:
            continue
        vals = sub[metric].dropna()
        m_result: dict = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(vals.median()),
            "cv": float(vals.std() / vals.mean()) if vals.mean() != 0 else np.nan,
            "n": len(vals),
        }

        # Per-group descriptive stats
        if "group" in sub.columns:
            groups = sub.groupby("group")[metric].apply(list).to_dict()
            group_stats: dict = {}
            for g, g_vals in groups.items():
                arr = np.array([v for v in g_vals if not np.isnan(v)])
                if len(arr) == 0:
                    continue
                q25, q75 = np.percentile(arr, [25, 75])
                group_stats[g] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "iqr": float(q75 - q25),
                    "n": len(arr),
                }
            m_result["groups"] = group_stats

            # Kruskal-Wallis across patient groups
            group_arrays = [
                np.array([v for v in g_vals if not np.isnan(v)])
                for g_vals in groups.values()
            ]
            group_arrays = [g for g in group_arrays if len(g) >= 2]
            if len(group_arrays) >= 2:
                h_stat, h_p = stats.kruskal(*group_arrays)
                m_result["kruskal_wallis"] = {"H": float(h_stat), "p": float(h_p)}

                # Mann-Whitney U pairwise with Holm correction
                group_names = [
                    g for g, g_vals in groups.items()
                    if len([v for v in g_vals if not np.isnan(v)]) >= 2
                ]
                if len(group_names) >= 2:
                    pairs = list(itertools.combinations(group_names, 2))
                    raw_p = []
                    pair_results = []
                    for g1, g2 in pairs:
                        a = np.array([v for v in groups[g1] if not np.isnan(v)])
                        b = np.array([v for v in groups[g2] if not np.isnan(v)])
                        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                        raw_p.append(u_p)
                        pair_results.append(
                            {"groups": [g1, g2], "U": float(u_stat), "p": float(u_p)}
                        )
                    corrected = holm_bonferroni_correction(raw_p)
                    for pr, (adj_p, sig) in zip(pair_results, corrected):
                        pr["p_corrected"] = float(adj_p)
                        pr["significant"] = sig
                    m_result["mann_whitney_pairwise"] = pair_results

        result["metrics"][metric] = m_result

    return result


def compute_crossfold_consistency(
    df: pd.DataFrame, metrics: list[str], config: str
) -> dict:
    """Per-fold aggregates, mean +/- std, Kruskal-Wallis across folds."""
    sub = df[(df["config"] == config) & df["fold"].notna()].copy()
    result: dict = {"config": config, "metrics": {}}

    if sub["fold"].nunique() < 2:
        return result

    for metric in metrics:
        if metric not in sub.columns:
            continue
        fold_means = sub.groupby("fold")[metric].mean()
        m_result: dict = {
            "fold_means": {str(k): float(v) for k, v in fold_means.items()},
            "cv_mean": float(fold_means.mean()),
            "cv_std": float(fold_means.std()),
            "n_folds": int(fold_means.count()),
        }

        # Kruskal-Wallis across folds
        fold_arrays = [
            group[metric].dropna().values
            for _, group in sub.groupby("fold")
        ]
        fold_arrays = [f for f in fold_arrays if len(f) >= 2]
        if len(fold_arrays) >= 2:
            h_stat, h_p = stats.kruskal(*fold_arrays)
            m_result["kruskal_wallis"] = {"H": float(h_stat), "p": float(h_p)}

        result["metrics"][metric] = m_result

    return result


def compare_configs(
    df: pd.DataFrame,
    metrics: list[str],
    alpha: float = 0.05,
    window_df: pd.DataFrame | None = None,
) -> dict:
    """Pairwise config comparisons with all applicable tests."""
    configs = sorted(df["config"].unique())
    if len(configs) < 2:
        return {"pairs": [], "note": "Need >= 2 configs for comparison"}

    all_results: list[dict] = []
    all_p_values: list[float] = []
    p_value_indices: list[tuple[int, str]] = []  # (result_idx, key in result)

    for cfg_a, cfg_b in itertools.combinations(configs, 2):
        pair_result: dict = {"config_a": cfg_a, "config_b": cfg_b, "tests": {}}

        for metric in metrics:
            if metric not in df.columns:
                continue

            df_a = df[df["config"] == cfg_a].groupby("patient_id")[metric].mean().reset_index()
            df_b = df[df["config"] == cfg_b].groupby("patient_id")[metric].mean().reset_index()

            # Inner join on patient_id for paired tests
            merged = df_a.merge(df_b, on="patient_id", suffixes=("_a", "_b"))
            a_vals = merged[f"{metric}_a"].values
            b_vals = merged[f"{metric}_b"].values

            n_overlap = len(merged)
            n_a = len(df_a["patient_id"].unique())
            n_b = len(df_b["patient_id"].unique())

            test_results: dict = {
                "n_paired": n_overlap,
                "n_a": n_a,
                "n_b": n_b,
                "mean_a": float(a_vals.mean()) if len(a_vals) > 0 else np.nan,
                "mean_b": float(b_vals.mean()) if len(b_vals) > 0 else np.nan,
            }

            if n_overlap < 2:
                test_results["warning"] = "Too few paired samples"
                pair_result["tests"][metric] = test_results
                continue

            if n_overlap < n_a * 0.5 or n_overlap < n_b * 0.5:
                test_results["warning"] = (
                    f"Low patient overlap: {n_overlap}/{n_a} vs {n_overlap}/{n_b}"
                )

            result_idx = len(all_results)

            # Wilcoxon signed-rank
            w_stat, w_p = wilcoxon_signed_rank(a_vals, b_vals)
            test_results["wilcoxon"] = {"statistic": w_stat, "p": w_p}
            if not np.isnan(w_p):
                all_p_values.append(w_p)
                p_value_indices.append((result_idx, f"{metric}.wilcoxon"))

            # Permutation test
            perm_diff, perm_p = permutation_test(a_vals, b_vals)
            test_results["permutation"] = {"mean_diff": perm_diff, "p": perm_p}
            all_p_values.append(perm_p)
            p_value_indices.append((result_idx, f"{metric}.permutation"))

            # Nadeau-Bengio corrected t-test (per-fold aggregates)
            # Use raw df (before patient aggregation) to preserve fold information
            raw_a = df[(df["config"] == cfg_a) & df["fold"].notna()]
            raw_b = df[(df["config"] == cfg_b) & df["fold"].notna()]
            if metric in raw_a.columns and metric in raw_b.columns:
                folds_a = raw_a.groupby("fold")[metric].mean()
                folds_b = raw_b.groupby("fold")[metric].mean()
                common_folds = sorted(set(folds_a.index) & set(folds_b.index))
                if len(common_folds) >= 2:
                    fa = np.array([folds_a[f] for f in common_folds])
                    fb = np.array([folds_b[f] for f in common_folds])
                    # Estimate n_train/n_test from fold sizes (assumes balanced folds)
                    k = len(common_folds)
                    total_patients = len(
                        set(raw_a["patient_id"].unique()) | set(raw_b["patient_id"].unique())
                    )
                    n_test_est = max(1, total_patients // k)
                    n_train_est = max(1, total_patients - n_test_est)
                    t_stat, t_p = corrected_paired_ttest(fa, fb, n_train_est, n_test_est)
                    test_results["nadeau_bengio"] = {"t": t_stat, "p": t_p, "n_folds": k}
                    if not np.isnan(t_p):
                        all_p_values.append(t_p)
                        p_value_indices.append((result_idx, f"{metric}.nadeau_bengio"))

            # Bootstrap CI
            lo, mean_diff, hi = bootstrap_ci(a_vals, b_vals)
            test_results["bootstrap_ci"] = {"lo": lo, "mean": mean_diff, "hi": hi}

            # Cohen's d
            test_results["cohens_d"] = cohens_d(a_vals, b_vals)

            # Cliff's delta
            cd, cd_mag = cliffs_delta(a_vals, b_vals)
            test_results["cliffs_delta"] = {"delta": cd, "magnitude": cd_mag}

            pair_result["tests"][metric] = test_results

        # Window-level tests (DeLong, McNemar)
        if window_df is not None and not window_df.empty:
            wdf_a = window_df[window_df["config"] == cfg_a]
            wdf_b = window_df[window_df["config"] == cfg_b]
            if not wdf_a.empty and not wdf_b.empty:
                pair_result["window_level"] = _window_level_tests(
                    wdf_a, wdf_b, cfg_a, cfg_b, all_p_values, p_value_indices, len(all_results)
                )

        all_results.append(pair_result)

    # Holm-Bonferroni correction across all p-values
    if all_p_values:
        corrected = holm_bonferroni_correction(all_p_values, alpha)
        for i, (adj_p, sig) in enumerate(corrected):
            result_idx, key_path = p_value_indices[i]
            _set_correction(all_results[result_idx], key_path, adj_p, sig)

    return {"pairs": all_results, "alpha": alpha, "n_comparisons": len(all_p_values)}


def _window_level_tests(
    wdf_a: pd.DataFrame,
    wdf_b: pd.DataFrame,
    cfg_a: str,
    cfg_b: str,
    all_p_values: list[float],
    p_value_indices: list[tuple[int, str]],
    result_idx: int,
) -> dict:
    """DeLong and McNemar tests on window-level predictions."""
    result: dict = {}

    # Align on patient_id + file_name + global_window_idx for exact match
    join_cols = ["patient_id", "file_name", "global_window_idx"]
    available = [c for c in join_cols if c in wdf_a.columns and c in wdf_b.columns]
    if not available:
        available = ["patient_id"]

    required_cols = ["prob", "pred", "gt"]
    missing_a = [c for c in required_cols if c not in wdf_a.columns]
    missing_b = [c for c in required_cols if c not in wdf_b.columns]
    if missing_a or missing_b:
        missing = sorted(set(missing_a) | set(missing_b))
        result["warning"] = f"Missing required columns: {', '.join(missing)}"
        return result

    merged = wdf_a[available + required_cols].merge(
        wdf_b[available + required_cols],
        on=available,
        suffixes=("_a", "_b"),
    )
    if len(merged) < 20:
        result["warning"] = f"Too few aligned windows ({len(merged)})"
        return result

    result["n_aligned_windows"] = len(merged)

    gt_a = merged["gt_a"].values
    gt_b = merged["gt_b"].values
    if not np.array_equal(gt_a, gt_b):
        n_mismatch = int(np.sum(gt_a != gt_b))
        result["warning"] = f"Ground truth mismatch: {n_mismatch}/{len(gt_a)} samples differ"
        return result
    gt = gt_a

    # DeLong
    z, p = delong_test(gt, merged["prob_a"].values, merged["prob_b"].values)
    result["delong"] = {"z": z, "p": p}
    if not np.isnan(p):
        all_p_values.append(p)
        p_value_indices.append((result_idx, "window_level.delong"))

    # McNemar
    chi2, p = mcnemar_test(
        merged["pred_a"].values.astype(int),
        merged["pred_b"].values.astype(int),
        gt.astype(int),
    )
    result["mcnemar"] = {"chi2": chi2, "p": p}
    if not np.isnan(p):
        all_p_values.append(p)
        p_value_indices.append((result_idx, "window_level.mcnemar"))

    return result


def _set_correction(result: dict, key_path: str, adj_p: float, sig: bool) -> None:
    """Inject corrected p-value into nested result dict.

    *key_path* is like "f1.wilcoxon" or "window_level.delong".
    """
    parts = key_path.split(".")
    target = result
    if parts[0] == "window_level":
        target = target.get("window_level", {})
        key = parts[1]
    else:
        target = target.get("tests", {}).get(parts[0], {})
        key = parts[1]
    if isinstance(target, dict) and key in target and isinstance(target[key], dict):
        target[key]["p_corrected"] = adj_p
        target[key]["significant"] = sig


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def format_summary_table(
    variability: list[dict],
    crossfold: list[dict],
    comparison: dict,
    metrics: list[str],
    alpha: float,
) -> str:
    """Build a human-readable summary string."""
    lines: list[str] = []

    # --- Per-config summaries ---
    for var in variability:
        cfg = var["config"]
        lines.append(f"\n{'='*70}")
        lines.append(f"Config: {cfg}")
        lines.append(f"{'='*70}")

        # Crossfold info
        cf = next((c for c in crossfold if c["config"] == cfg), None)

        for metric in metrics:
            m_var = var.get("metrics", {}).get(metric)
            if m_var is None:
                continue

            lines.append(f"\n  {metric}:")
            lines.append(
                f"    Overall:  mean={m_var['mean']:.4f}  std={m_var['std']:.4f}  "
                f"median={m_var['median']:.4f}  CV={m_var['cv']:.3f}  (n={m_var['n']})"
            )

            # Per-group
            for g, gs in m_var.get("groups", {}).items():
                lines.append(
                    f"    Group {g}: mean={gs['mean']:.4f}  std={gs['std']:.4f}  "
                    f"median={gs['median']:.4f}  IQR={gs['iqr']:.4f}  (n={gs['n']})"
                )

            kw = m_var.get("kruskal_wallis")
            if kw:
                sig = "*" if kw["p"] < alpha else ""
                lines.append(
                    f"    Kruskal-Wallis (groups): H={kw['H']:.3f}  p={kw['p']:.4f} {sig}"
                )

            # Cross-fold
            if cf:
                m_cf = cf.get("metrics", {}).get(metric)
                if m_cf:
                    lines.append(
                        f"    Cross-fold: {m_cf['cv_mean']:.4f} +/- {m_cf['cv_std']:.4f}  "
                        f"({m_cf['n_folds']} folds)"
                    )
                    kw_f = m_cf.get("kruskal_wallis")
                    if kw_f:
                        sig = "*" if kw_f["p"] < alpha else ""
                        lines.append(
                            f"    Kruskal-Wallis (folds): H={kw_f['H']:.3f}  p={kw_f['p']:.4f} {sig}"
                        )

    # --- Pairwise comparisons ---
    pairs = comparison.get("pairs", [])
    if pairs:
        lines.append(f"\n{'='*70}")
        lines.append("Pairwise Configuration Comparisons")
        lines.append(f"{'='*70}")
        lines.append(
            f"(Holm-Bonferroni corrected, {comparison.get('n_comparisons', 0)} total tests, alpha={alpha})"
        )

        for pair in pairs:
            lines.append(f"\n  {pair['config_a']}  vs  {pair['config_b']}")
            lines.append(f"  {'-'*60}")

            for metric in metrics:
                tests = pair.get("tests", {}).get(metric)
                if tests is None:
                    continue

                mean_a = tests.get("mean_a", np.nan)
                mean_b = tests.get("mean_b", np.nan)
                lines.append(
                    f"\n    {metric}:  A={mean_a:.4f}  B={mean_b:.4f}  "
                    f"(n_paired={tests.get('n_paired', 0)})"
                )

                if "warning" in tests:
                    lines.append(f"      WARNING: {tests['warning']}")

                # Wilcoxon
                w = tests.get("wilcoxon", {})
                if w and not np.isnan(w.get("p", np.nan)):
                    sig = _sig_marker(w, alpha)
                    lines.append(
                        f"      Wilcoxon:      W={w['statistic']:.2f}  "
                        f"p={w['p']:.4f}  p_corr={w.get('p_corrected', np.nan):.4f} {sig}"
                    )

                # Permutation
                perm = tests.get("permutation", {})
                if perm:
                    sig = _sig_marker(perm, alpha)
                    lines.append(
                        f"      Permutation:   diff={perm['mean_diff']:.4f}  "
                        f"p={perm['p']:.4f}  p_corr={perm.get('p_corrected', np.nan):.4f} {sig}"
                    )

                # Nadeau-Bengio
                nb = tests.get("nadeau_bengio", {})
                if nb and not np.isnan(nb.get("p", np.nan)):
                    sig = _sig_marker(nb, alpha)
                    lines.append(
                        f"      Nadeau-Bengio: t={nb['t']:.3f}  "
                        f"p={nb['p']:.4f}  p_corr={nb.get('p_corrected', np.nan):.4f} "
                        f"({nb.get('n_folds', '?')} folds) {sig}"
                    )

                # Bootstrap
                bs = tests.get("bootstrap_ci", {})
                if bs:
                    lines.append(
                        f"      Bootstrap CI:  [{bs['lo']:.4f}, {bs['hi']:.4f}]  "
                        f"mean_diff={bs['mean']:.4f}"
                    )

                # Effect sizes
                cd = tests.get("cohens_d")
                if cd is not None:
                    lines.append(f"      Cohen's d:     {cd:.3f}")
                cld = tests.get("cliffs_delta", {})
                if cld:
                    lines.append(
                        f"      Cliff's delta: {cld.get('delta', np.nan):.3f} ({cld.get('magnitude', '?')})"
                    )

            # Window-level
            wl = pair.get("window_level", {})
            if wl:
                lines.append(f"\n    Window-level tests (n={wl.get('n_aligned_windows', '?')})")
                dl = wl.get("delong", {})
                if dl and not np.isnan(dl.get("p", np.nan)):
                    sig = _sig_marker(dl, alpha)
                    lines.append(
                        f"      DeLong:  z={dl['z']:.3f}  p={dl['p']:.4f}  "
                        f"p_corr={dl.get('p_corrected', np.nan):.4f} {sig}"
                    )
                mc = wl.get("mcnemar", {})
                if mc and not np.isnan(mc.get("p", np.nan)):
                    sig = _sig_marker(mc, alpha)
                    lines.append(
                        f"      McNemar: chi2={mc['chi2']:.3f}  p={mc['p']:.4f}  "
                        f"p_corr={mc.get('p_corrected', np.nan):.4f} {sig}"
                    )

    return "\n".join(lines)


def _sig_marker(test_dict: dict, alpha: float) -> str:
    sig = test_dict.get("significant")
    if sig is True:
        return "*"
    elif sig is False:
        return ""
    # Fallback to raw p
    p = test_dict.get("p", np.nan)
    return "*" if (not np.isnan(p) and p < alpha) else ""


def save_results_csv(comparison: dict, path: Path) -> None:
    """Write one row per (config_a, config_b, metric, test)."""
    rows: list[dict] = []
    for pair in comparison.get("pairs", []):
        cfg_a, cfg_b = pair["config_a"], pair["config_b"]
        for metric, tests in pair.get("tests", {}).items():
            for test_name in ["wilcoxon", "permutation", "nadeau_bengio"]:
                t = tests.get(test_name, {})
                if not t or (isinstance(t, dict) and np.isnan(t.get("p", np.nan))):
                    continue
                row = {
                    "config_a": cfg_a,
                    "config_b": cfg_b,
                    "metric": metric,
                    "test": test_name,
                    "statistic": t.get("statistic", t.get("t", t.get("mean_diff"))),
                    "p_value": t.get("p"),
                    "p_corrected": t.get("p_corrected"),
                    "significant": t.get("significant"),
                    "mean_a": tests.get("mean_a"),
                    "mean_b": tests.get("mean_b"),
                    "mean_diff": tests.get("bootstrap_ci", {}).get("mean"),
                    "ci_lo": tests.get("bootstrap_ci", {}).get("lo"),
                    "ci_hi": tests.get("bootstrap_ci", {}).get("hi"),
                    "cohens_d": tests.get("cohens_d"),
                    "cliffs_delta": tests.get("cliffs_delta", {}).get("delta"),
                    "cliffs_magnitude": tests.get("cliffs_delta", {}).get("magnitude"),
                    "n_paired": tests.get("n_paired"),
                }
                rows.append(row)

        # Window-level tests
        wl = pair.get("window_level", {})
        for test_name in ["delong", "mcnemar"]:
            t = wl.get(test_name, {})
            if not t or np.isnan(t.get("p", np.nan)):
                continue
            rows.append({
                "config_a": cfg_a,
                "config_b": cfg_b,
                "metric": "window_level",
                "test": test_name,
                "statistic": t.get("z", t.get("chi2")),
                "p_value": t.get("p"),
                "p_corrected": t.get("p_corrected"),
                "significant": t.get("significant"),
                "n_paired": wl.get("n_aligned_windows"),
            })

    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def save_results_json(
    variability: list[dict],
    crossfold: list[dict],
    comparison: dict,
    path: Path,
) -> None:
    """Write full structured results to JSON."""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    data = {
        "inter_subject_variability": variability,
        "crossfold_consistency": crossfold,
        "config_comparison": comparison,
    }
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)


def generate_plots(
    df: pd.DataFrame,
    comparison: dict,
    metrics: list[str],
    output_dir: Path,
) -> None:
    """Generate comparison boxplots, fold bar charts, and paired difference scatter."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots.")
        return

    try:
        import seaborn as sns
        _HAS_SEABORN = True
    except ImportError:
        _HAS_SEABORN = False

    configs = sorted(df["config"].unique())
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Boxplots of per-subject metrics across configs
    for metric in metrics:
        if metric not in df.columns:
            continue
        plot_df = df[["config", metric]].dropna()
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(configs)), 5))
        if _HAS_SEABORN:
            sns.boxplot(data=plot_df, x="config", y=metric, ax=ax)
            sns.stripplot(
                data=plot_df, x="config", y=metric, ax=ax,
                color="0.3", alpha=0.4, size=4, jitter=True,
            )
        else:
            grouped = [g[metric].values for _, g in plot_df.groupby("config", sort=True)]
            ax.boxplot(grouped, patch_artist=True)
            ax.set_xticklabels(configs)
        ax.set_title(f"{metric} by configuration")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / f"boxplot_{metric}.png", dpi=150)
        plt.close(fig)

    # 2. Fold-level bar charts (per config)
    folds = sorted(df[df["fold"].notna()]["fold"].unique())
    if len(folds) >= 2:
        for metric in metrics:
            if metric not in df.columns:
                continue
            fold_means = (
                df[df["fold"].notna()]
                .groupby(["config", "fold"])[metric]
                .mean()
                .reset_index()
            )
            if fold_means.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 2 * len(configs)), 5))
            if _HAS_SEABORN:
                sns.barplot(data=fold_means, x="fold", y=metric, hue="config", ax=ax)
            else:
                n_cfg = len(configs)
                width = 0.8 / n_cfg
                x = np.arange(len(folds))
                for i, cfg in enumerate(configs):
                    cfg_data = fold_means[fold_means["config"] == cfg]
                    vals = [
                        cfg_data[cfg_data["fold"] == f][metric].values[0]
                        if len(cfg_data[cfg_data["fold"] == f]) > 0
                        else 0
                        for f in folds
                    ]
                    ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cfg)
                ax.set_xticks(x)
                ax.set_xticklabels(folds)
            ax.set_title(f"{metric} per fold")
            ax.legend(title="Config", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"folds_{metric}.png", dpi=150)
            plt.close(fig)

    # 3. Paired difference scatter for each pair
    for pair in comparison.get("pairs", []):
        cfg_a, cfg_b = pair["config_a"], pair["config_b"]
        for metric in metrics:
            tests = pair.get("tests", {}).get(metric)
            if tests is None or tests.get("n_paired", 0) < 2:
                continue

            df_a = df[df["config"] == cfg_a][["patient_id", metric]].dropna()
            df_b = df[df["config"] == cfg_b][["patient_id", metric]].dropna()
            merged = df_a.merge(df_b, on="patient_id", suffixes=("_a", "_b"))
            if len(merged) < 2:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            diff = merged[f"{metric}_b"] - merged[f"{metric}_a"]
            ax.scatter(range(len(diff)), diff.values, alpha=0.6, edgecolors="k", linewidths=0.5)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.axhline(diff.mean(), color="blue", linestyle="-", linewidth=1, label=f"mean={diff.mean():.4f}")
            ax.set_xlabel("Patient (index)")
            ax.set_ylabel(f"Diff ({metric}): B - A")
            ax.set_title(f"{metric}: {cfg_b} - {cfg_a}")
            ax.legend()
            fig.tight_layout()
            safe_a = re.sub(r"[^\w\-]", "_", cfg_a)
            safe_b = re.sub(r"[^\w\-]", "_", cfg_b)
            fig.savefig(output_dir / f"paired_{metric}_{safe_a}_vs_{safe_b}.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical significance analysis for HBIOT pipeline outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/statistical_analysis.py --logs-dir logs/
  python scripts/statistical_analysis.py --logs-dir logs/ --filter "BIOT*" --plots
  python scripts/statistical_analysis.py --logs-dir logs/ --list-configs
""",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root directory to scan for experiment outputs (default: logs/)",
    )
    p.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help="Glob patterns to select specific configs (matched against relative path)",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help=f"Metrics to analyze (default: {' '.join(DEFAULT_METRICS)})",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for CSV/JSON/plots (default: <logs-dir>/statistical_results/)",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Generate comparison figures",
    )
    p.add_argument(
        "--window-level",
        action="store_true",
        help="Load window-level CSVs for DeLong & McNemar tests",
    )
    p.add_argument(
        "--fold-pattern",
        type=str,
        default=DEFAULT_FOLD_PATTERN,
        help=f'Regex for fold extraction, must have one capture group (default: "{DEFAULT_FOLD_PATTERN}")',
    )
    p.add_argument(
        "--list-configs",
        action="store_true",
        help="Print discovered configs and exit (dry-run)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logs_dir = args.logs_dir.resolve()

    if not logs_dir.is_dir():
        print(f"ERROR: --logs-dir {logs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Validate fold pattern
    try:
        re.compile(args.fold_pattern)
    except re.error as e:
        print(f"ERROR: Invalid --fold-pattern: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Discover ---
    print(f"Scanning {logs_dir} ...")
    experiments = discover_experiments(logs_dir, args.fold_pattern, args.filter)

    if not experiments:
        print("No experiments found (no patient_metrics_*.csv files).", file=sys.stderr)
        sys.exit(1)

    configs = sorted(set(e["config"] for e in experiments))
    print(f"Found {len(experiments)} experiment(s) across {len(configs)} config(s):\n")
    for cfg in configs:
        folds = sorted(set(e["fold"] for e in experiments if e["config"] == cfg and e["fold"] is not None))
        fold_str = f"  folds: {', '.join(folds)}" if folds else "  (no fold detected)"
        print(f"  {cfg}{fold_str}")

    if args.list_configs:
        return

    # --- Load ---
    print("\nLoading patient metrics ...")
    df = load_patient_metrics(experiments)
    if df.empty:
        print("ERROR: No patient metrics loaded.", file=sys.stderr)
        sys.exit(1)

    available_metrics = [m for m in args.metrics if m in df.columns]
    missing = [m for m in args.metrics if m not in df.columns]
    if missing:
        print(f"WARNING: Metrics not found in data: {', '.join(missing)}")
    if not available_metrics:
        print("ERROR: None of the requested metrics found in the data.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded {len(df)} patient-metric rows, "
        f"{df['patient_id'].nunique()} unique patients, "
        f"metrics: {', '.join(available_metrics)}"
    )

    window_df: pd.DataFrame | None = None
    if args.window_level:
        print("Loading window-level predictions ...")
        window_df = load_window_predictions(experiments)
        if window_df.empty:
            print("WARNING: No window-level predictions found.")
            window_df = None
        else:
            print(f"Loaded {len(window_df)} window-level rows.")

    # --- Analyze ---
    print("\nRunning analyses ...")

    # Suppress scipy warnings for small samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 1. Inter-subject variability
        variability = [
            compute_inter_subject_variability(df, available_metrics, cfg)
            for cfg in configs
        ]

        # 2. Cross-fold consistency
        crossfold = [
            compute_crossfold_consistency(df, available_metrics, cfg)
            for cfg in configs
        ]

        # 3. Pairwise config comparison
        comparison = compare_configs(df, available_metrics, args.alpha, window_df)

    # --- Output ---
    summary = format_summary_table(variability, crossfold, comparison, available_metrics, args.alpha)
    print(summary)

    output_dir = args.output or (logs_dir / "statistical_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "statistical_results.csv"
    json_path = output_dir / "statistical_results.json"

    save_results_csv(comparison, csv_path)
    save_results_json(variability, crossfold, comparison, json_path)
    print(f"\nResults saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    if args.plots:
        plots_dir = output_dir / "plots"
        print(f"Generating plots in {plots_dir} ...")
        generate_plots(df, comparison, available_metrics, plots_dir)
        print("Plots saved.")


if __name__ == "__main__":
    main()
