"""Evaluation metrics for single runs and Monte Carlo batches.

Definitions used throughout the report:

- **converged**: the position error stays below ``threshold`` (50 m) from the
  convergence time to the end of the flight, and that holds for at least the final
  ``hold`` seconds (100 s) — so the tracking window never contains acquisition.
- **false lock**: final error > 200 m while the filter's own 2σ position radius,
  2·√λ_max(P_pos), is < 50 m — the filter is confidently wrong.
- **tracking RMSE**: RMSE over the final ``hold`` seconds of converged runs only.
- **NEES / ANEES**: normalised estimation error squared on position (dimension 2);
  ANEES is its average over M runs, with the two-sided 95 % interval
  χ²_{2M}(0.025)/M … χ²_{2M}(0.975)/M for a consistent filter.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def position_error(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Horizontal position error (m) per step. ``est`` (K, 4), ``truth`` (K, 2)."""
    return np.linalg.norm(est[:, :2] - truth, axis=1)


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def position_nees(est: np.ndarray, cov: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared on position, per step.

    For a consistent filter its Monte Carlo average is ≈ 2 (the position dimension).
    ``cov`` may be (K, 4, 4) or (K, 2, 2).
    """
    e = est[:, :2] - truth
    p = cov[:, :2, :2]
    return np.einsum("ki,kij,kj->k", e, np.linalg.inv(p), e)


def two_sigma_radius(cov_pos: np.ndarray) -> np.ndarray:
    """2·√λ_max of the 2×2 position covariance, (…, 2, 2) → (…)."""
    return 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(cov_pos)[..., -1], 0.0))


def convergence_time(errors: np.ndarray, t: np.ndarray, threshold: float) -> float:
    """First time after which the error stays below ``threshold`` (NaN if it never does)."""
    above = np.nonzero(errors >= threshold)[0]
    if above.size == 0:
        return float(t[0])
    last = above[-1]
    return float(t[last + 1]) if last + 1 < len(t) else float("nan")


def is_converged(
    errors: np.ndarray, t: np.ndarray, threshold: float = 50.0, hold: float = 100.0
) -> bool:
    """Error below ``threshold`` throughout at least the final ``hold`` seconds."""
    tc = convergence_time(errors, t, threshold)
    return bool(np.isfinite(tc) and tc <= t[-1] - hold + 1e-9)


def is_false_lock(
    final_error: float,
    final_cov_pos: np.ndarray,
    error_min: float = 200.0,
    radius_max: float = 50.0,
) -> bool:
    return bool(final_error > error_min and two_sigma_radius(final_cov_pos) < radius_max)


def tracking_window(t: np.ndarray, hold: float = 100.0) -> np.ndarray:
    """Boolean mask of the final ``hold`` seconds."""
    return t > t[-1] - hold + 1e-9


# ---------------------------------------------------------------------- intervals
def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + conf / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int = 2000,
    conf: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over the first axis of ``values`` (runs)."""
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([statistic(values[i]) for i in idx])
    a = (1.0 - conf) / 2.0
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def rmse_from_mse(mse_per_run: np.ndarray) -> float:
    """Pooled RMSE over runs from each run's mean squared error."""
    return float(np.sqrt(np.mean(mse_per_run))) if len(mse_per_run) else float("nan")


def anees_bounds(n_runs: int, dim: int = 2, conf: float = 0.95) -> tuple[float, float]:
    """Interval that contains the ANEES of a consistent filter with probability ``conf``."""
    a = (1.0 - conf) / 2.0
    return (
        float(stats.chi2.ppf(a, dim * n_runs) / n_runs),
        float(stats.chi2.ppf(1 - a, dim * n_runs) / n_runs),
    )


def nees_in_bounds_fraction(nees: np.ndarray, dim: int = 2, conf: float = 0.95) -> np.ndarray:
    """Per step: fraction of runs whose NEES lies inside the single-run χ²_dim interval."""
    a = (1.0 - conf) / 2.0
    lo, hi = stats.chi2.ppf(a, dim), stats.chi2.ppf(1 - a, dim)
    nees = np.atleast_2d(nees)
    return np.mean((nees >= lo) & (nees <= hi), axis=0)


# ---------------------------------------------------------------------- paired tests
def mcnemar(success_a: np.ndarray, success_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes (same runs, two filters).

    Only discordant pairs carry information: n01 = A failed & B succeeded,
    n10 = A succeeded & B failed. Under H0 (equal success rates) n01 ~ Bin(n01 + n10, ½).
    """
    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p_value": p}


def paired_bootstrap_rmse_diff(
    mse_a: np.ndarray, mse_b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000
) -> dict[str, float]:
    """RMSE(A) − RMSE(B) over the same runs, with a paired percentile-bootstrap 95 % CI."""
    pair = np.column_stack([mse_a, mse_b])

    def diff(x: np.ndarray) -> float:
        return float(np.sqrt(x[:, 0].mean()) - np.sqrt(x[:, 1].mean()))

    lo, hi = bootstrap_ci(pair, diff, rng, n_boot)
    return {"diff": diff(pair), "lo": lo, "hi": hi}


# ---------------------------------------------------------------------- legacy summary
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
