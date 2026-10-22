"""Evaluation metrics for single runs and Monte Carlo batches."""

from __future__ import annotations

import numpy as np


def position_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Horizontal position error (m) per step. ``est`` (K, 4), ``truth`` (K, 2)."""
    return np.linalg.norm(est[:, :2] - truth, axis=1)


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def position_nees(est: np.ndarray, cov: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared on position, per step.

    For a consistent filter its Monte Carlo average is ≈ 2 (the position dimension).
    """
    e = est[:, :2] - truth
    p = cov[:, :2, :2]
    return np.einsum("ki,kij,kj->k", e, np.linalg.inv(p), e)


def convergence_time(errors: np.ndarray, t: np.ndarray, threshold: float) -> float:
    """First time after which the error stays below ``threshold`` (NaN if it never does)."""
    above = np.nonzero(errors >= threshold)[0]
    if above.size == 0:
        return float(t[0])
    last = above[-1]
    return float(t[last + 1]) if last + 1 < len(t) else float("nan")


def summarize(errors: np.ndarray, t: np.ndarray, threshold: float) -> dict[str, float]:
    """Summary over Monte Carlo runs. ``errors`` has shape (runs, K)."""
    errors = np.atleast_2d(errors)
    conv = np.array([convergence_time(e, t, threshold) for e in errors])
    return {
        "runs": int(errors.shape[0]),
        "rmse_m": rmse(errors),
        "final_error_mean_m": float(errors[:, -1].mean()),
        "final_error_std_m": float(errors[:, -1].std()),
        "converged_fraction": float(np.mean(~np.isnan(conv))),
        "convergence_time_mean_s": float(np.nanmean(conv)) if np.any(~np.isnan(conv)) else None,
    }
