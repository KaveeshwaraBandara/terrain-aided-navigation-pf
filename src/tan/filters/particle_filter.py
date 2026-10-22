"""Sequential importance resampling (SIR) particle filter for terrain-aided navigation.

Main contribution of the project. The bookkeeping (initialisation, weight
normalisation, the resampling decision, logging) is written; the estimation
maths is left as TODOs for you to implement and derive in the report.

Models (state x = [px, py, bx, by]):

    motion       p_k = p_{k-1} + dt (u_k - b_{k-1}) + w_p,   w_p ~ N(0, σ_p² I)
                 b_k = b_{k-1} + w_b,                         w_b ~ N(0, σ_b² I)
    measurement  z_k = h(p_k) + v_k,                          v_k ~ N(0, σ_z²)
                 h(p) = DEM height at p (bilinear)
"""

from __future__ import annotations

import numpy as np

from ..dem import DEM
from .base import Estimator
from .resampling import RESAMPLERS, effective_sample_size


class ParticleFilter(Estimator):
    name = "particle_filter"

    def __init__(
        self,
        dem: DEM,
        n_particles: int = 5000,
        meas_sigma: float = 3.6,
        process_sigma_pos: float = 2.0,
        process_sigma_bias: float = 0.002,
        resampling: str = "systematic",
        ess_threshold: float = 0.5,
        roughening: float = 0.0,
        snapshot_every: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if resampling not in RESAMPLERS:
            raise ValueError(f"unknown resampling '{resampling}', choose from {list(RESAMPLERS)}")
        self.dem = dem
        self.n = int(n_particles)
        self.meas_sigma = float(meas_sigma)
        self.sigma_pos = float(process_sigma_pos)
        self.sigma_bias = float(process_sigma_bias)
        self.resampling = resampling
        self.resample_fn = RESAMPLERS[resampling]
        self.ess_threshold = float(ess_threshold)
        self.roughening = float(roughening)
        self.snapshot_every = int(snapshot_every)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.particles = np.zeros((self.n, 4))  # (N, 4) = [px, py, bx, by]
        self.log_weights = np.full(self.n, -np.log(self.n))
        self.history: dict[str, list] = {"ess": [], "resampled": []}
        self.snapshots: list[tuple[int, np.ndarray]] = []  # (step, (N, 2) positions)
        self._k = 0

    # ------------------------------------------------------------------ provided
    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.particles = self.rng.multivariate_normal(mean, cov, size=self.n)
        self.log_weights = np.full(self.n, -np.log(self.n))
        self.history = {"ess": [], "resampled": []}
        self.snapshots = [(0, self.particles[:, :2].copy())] if self.snapshot_every else []
        self._k = 0

    @property
    def weights(self) -> np.ndarray:
        """Normalised weights from the log-weights (max-shift for numerical safety)."""
        lw = self.log_weights - np.max(self.log_weights)
        w = np.exp(lw)
        return w / w.sum()

    def step(self, u: np.ndarray, dt: float, z: float) -> tuple[np.ndarray, np.ndarray]:
        self.predict(u, dt)
        self.update(z)

        ess = effective_sample_size(self.weights)
        resampled = ess < self.ess_threshold * self.n
        if resampled:
            idx = self.resample_fn(self.weights, self.rng)
            self.particles = self.particles[idx]
            self.log_weights = np.full(self.n, -np.log(self.n))
            if self.roughening > 0:
                self.roughen()

        self._k += 1
        self.history["ess"].append(float(ess))
        self.history["resampled"].append(bool(resampled))
        if self.snapshot_every and self._k % self.snapshot_every == 0:
            self.snapshots.append((self._k, self.particles[:, :2].copy()))
        return self.estimate()

    # ------------------------------------------------------------------ TODO(you)
    def predict(self, u: np.ndarray, dt: float) -> None:
        """Propagate every particle through the motion model (vectorised, no Python loop).

        Use ``self.rng`` for the process noise and ``self.sigma_pos`` / ``self.sigma_bias``.
        """
        raise NotImplementedError("ParticleFilter.predict — implement the motion model")

    def update(self, z: float) -> None:
        """Re-weight particles by the terrain likelihood p(z | p_i).

        Work in log space: ``self.log_weights += log N(z; h(p_i), σ_z²)``, with
        h(p_i) from ``self.dem.height``. Normalisation is done by :attr:`weights`.
        Later experiment: a Student-t likelihood for radar-altimeter outliers.
        """
        raise NotImplementedError("ParticleFilter.update — implement the likelihood weighting")

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean (4,) and weighted covariance (4, 4) of the particle cloud."""
        raise NotImplementedError("ParticleFilter.estimate — implement the weighted mean/cov")

    def roughen(self) -> None:
        """Optional (Gordon et al., 1993): add small jitter after resampling to fight
        sample impoverishment. Scale by ``self.roughening``."""
        raise NotImplementedError("ParticleFilter.roughen — optional")
