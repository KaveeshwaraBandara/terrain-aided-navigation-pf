"""Resampling schemes for the particle filter.

Every function takes normalised weights ``w`` (N,) and a NumPy ``Generator`` and
returns an integer index array ``idx`` (N,) such that ``particles[idx]`` is the
resampled set.

Reference: J. D. Hol, T. B. Schön, F. Gustafsson, "On resampling algorithms for
particle filters", IEEE Nonlinear Statistical Signal Processing Workshop, 2006.

TODO(you): implement these — they are graded and worth comparing in the report
(variance of the offspring counts, runtime, effect on RMSE).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def effective_sample_size(w: np.ndarray) -> float:
    """N_eff = 1 / sum(w_i^2) for normalised weights.

    Equals N for uniform weights and 1 when one particle has all the weight.
    """
    raise NotImplementedError("resampling.effective_sample_size")


def multinomial(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw N indices i.i.d. from Categorical(w). Highest variance of the four."""
    raise NotImplementedError("resampling.multinomial")


def systematic(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniform u ~ U[0, 1/N); pointers u + i/N, i = 0..N-1, against cumsum(w)."""
    raise NotImplementedError("resampling.systematic")


def stratified(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One independent uniform per stratum: (i + u_i) / N, u_i ~ U[0, 1)."""
    raise NotImplementedError("resampling.stratified")


def residual(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep floor(N w_i) copies deterministically, fill the rest by multinomial on the residuals."""
    raise NotImplementedError("resampling.residual")


RESAMPLERS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "multinomial": multinomial,
    "systematic": systematic,
    "stratified": stratified,
    "residual": residual,
}
