"""Resampling schemes for the particle filter.

Every function takes normalised weights ``w`` (N,) and a NumPy ``Generator`` and
returns an integer index array ``idx`` (N,) such that ``particles[idx]`` is the
resampled set.

All four are *unbiased*: the expected number of copies of particle i is N w_i.
They differ in the variance of that number, which is noise the resampling step
adds to the estimate. Ordered from most to least variance in practice:
multinomial > residual > stratified ≳ systematic.

Reference: J. D. Hol, T. B. Schön, F. Gustafsson, "On resampling algorithms for
particle filters", IEEE Nonlinear Statistical Signal Processing Workshop, 2006.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def effective_sample_size(w: np.ndarray) -> float:
    """N_eff = 1 / sum(w_i^2) for normalised weights.

    Equals N for uniform weights and 1 when one particle has all the weight.
    """
    w = np.asarray(w, dtype=float)
    return float(1.0 / np.sum(w * w))


def _cdf(w: np.ndarray) -> np.ndarray:
    """Cumulative weights with the last entry pinned to exactly 1.

    Floating-point round-off can leave ``cumsum(w)[-1]`` at 0.9999999999999998;
    a pointer above that would index past the end of the array.
    """
    c = np.cumsum(np.asarray(w, dtype=float))
    c[-1] = 1.0
    return c


def _invert(cdf: np.ndarray, pointers: np.ndarray) -> np.ndarray:
    """Inverse-CDF lookup: for each pointer u in [0, 1), the index i with
    cdf[i-1] <= u < cdf[i].

    ``side="right"`` matters: a particle with zero weight has an empty interval
    and must never be selected, even when a pointer lands exactly on a boundary.
    """
    return np.searchsorted(cdf, pointers, side="right")


def multinomial(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw N indices i.i.d. from Categorical(w). Highest variance of the four.

    Offspring count of particle i ~ Binomial(N, w_i), so var = N w_i (1 - w_i).
    """
    n = len(w)
    return _invert(_cdf(w), rng.random(n))


def systematic(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniform u ~ U[0, 1/N); pointers u + i/N, i = 0..N-1, against cumsum(w).

    The pointers are evenly spaced, so particle i gets either floor(N w_i) or
    ceil(N w_i) copies — the minimum possible spread. O(N), one random number.
    """
    n = len(w)
    pointers = (rng.random() + np.arange(n)) / n
    return _invert(_cdf(w), pointers)


def stratified(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One independent uniform per stratum: (i + u_i) / N, u_i ~ U[0, 1).

    Same idea as systematic, but each of the N strata [i/N, (i+1)/N) gets its own
    random offset. Variance is never higher than multinomial's.
    """
    n = len(w)
    pointers = (rng.random(n) + np.arange(n)) / n
    return _invert(_cdf(w), pointers)


def residual(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep floor(N w_i) copies deterministically, fill the rest by multinomial on the residuals.

    Only the R = N - sum(floor(N w_i)) leftover slots are random, so the variance
    is lower than plain multinomial.
    """
    w = np.asarray(w, dtype=float)
    n = len(w)
    scaled = n * w
    copies = np.floor(scaled).astype(np.int64)
    idx = np.repeat(np.arange(n), copies)

    r = n - idx.size
    if r > 0:
        res = scaled - copies
        res /= res.sum()
        idx = np.concatenate([idx, _invert(_cdf(res), rng.random(r))])
    return idx


RESAMPLERS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "multinomial": multinomial,
    "systematic": systematic,
    "stratified": stratified,
    "residual": residual,
}
