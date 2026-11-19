"""Sequential importance resampling (SIR) particle filter for terrain-aided navigation.

Main contribution of the project. This is the bootstrap filter of Gordon, Salmond
and Smith (1993): the motion model is the proposal, so the importance weight of a
particle is just the measurement likelihood.

Models (state x = [px, py, bx, by]):

    motion       p_k = p_{k-1} + dt (u_k - b_{k-1}) + w_p,   w_p ~ N(0, σ_p² I)
                 b_k = b_{k-1} + w_b,                         w_b ~ N(0, σ_b² I)
    measurement  z_k = h(p_k) + v_k,                          v_k ~ N(0, σ_z²)
                 h(p) = DEM height at p (bilinear)

One filter step for particles {x_i, w_i}, i = 1..N:

    1. predict    x_i ~ p(x_k | x_i, u_k)                (sample the motion model)
    2. update     w_i ∝ w_i · p(z_k | x_i)              (weight by the terrain likelihood)
    3. resample   if N_eff = 1 / Σ w_i² < threshold · N  (fight weight degeneracy)
    4. estimate   x̂ = Σ w_i x_i,  P = Σ w_i (x_i − x̂)(x_i − x̂)ᵀ
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

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
        likelihood: str = "gaussian",
        nu: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if resampling not in RESAMPLERS:
            raise ValueError(f"unknown resampling '{resampling}', choose from {list(RESAMPLERS)}")
        if likelihood not in ("gaussian", "student_t"):
            raise ValueError(f"unknown likelihood '{likelihood}' (gaussian | student_t)")
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
        self.likelihood = likelihood
        self.nu = float(nu)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.particles = np.zeros((self.n, 4))  # (N, 4) = [px, py, bx, by]
        self.log_weights = np.full(self.n, -np.log(self.n))
        self.history: dict[str, list] = self._empty_history()
        self.snapshots: list[tuple[int, np.ndarray]] = []  # (step, (N, 2) positions)
        self._k = 0

    # ------------------------------------------------------------------ provided
    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.particles = self.rng.multivariate_normal(mean, cov, size=self.n)
        self.log_weights = np.full(self.n, -np.log(self.n))
        self.history = self._empty_history()
        self.snapshots = [(0, self.particles[:, :2].copy())] if self.snapshot_every else []
        self._k = 0

    @staticmethod
    def _empty_history() -> dict[str, list]:
        return {"ess": [], "resampled": [], "log_evidence": [], "nis": []}

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
        self.history["log_evidence"].append(self._last_log_evidence)
        self.history["nis"].append(self._last_nis)
        if self.snapshot_every and self._k % self.snapshot_every == 0:
            self.snapshots.append((self._k, self.particles[:, :2].copy()))
        return self.estimate()

    # ------------------------------------------------------------------ filter maths
    def predict(self, u: np.ndarray, dt: float) -> None:
        """Sample every particle from the motion model p(x_k | x_{k-1}, u_k).

        Each particle carries its own bias hypothesis b_i, so it integrates the INS
        velocity corrected by *its own* bias. Particles whose bias is wrong drift
        away from the terrain that matches the measurements and lose weight; that
        is how the filter estimates the bias without measuring it directly.
        """
        n = self.n
        bias = self.particles[:, 2:]  # b_{k-1}, read before the bias is perturbed below
        self.particles[:, :2] += dt * (np.asarray(u, dtype=float) - bias) + self.rng.normal(
            0.0, self.sigma_pos, size=(n, 2)
        )
        self.particles[:, 2:] += self.rng.normal(0.0, self.sigma_bias, size=(n, 2))

    def log_likelihood(self, residual: np.ndarray) -> np.ndarray:
        """log p(z | x_i) as a function of the residual z − h(p_i), with its constant.

        gaussian:   −½ log(2π σ²) − ½ (r/σ)²
        student_t:  log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(ν π σ²) − (ν+1)/2 · log(1 + (r/σ)²/ν)

        The constants cancel when the weights are normalised; they are kept so the
        log evidence log p(z_k | z_{1:k-1}) is comparable between the two models.
        The Student-t has heavy tails: a 100 m outlier costs it only a few units of
        log-likelihood instead of hundreds, so one bad altimeter sample cannot wipe
        out the particles near the truth.
        """
        s = self.meas_sigma
        r = residual / s
        if self.likelihood == "gaussian":
            return -0.5 * np.log(2.0 * np.pi * s * s) - 0.5 * r * r
        nu = self.nu
        const = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi * s * s)
        return const - 0.5 * (nu + 1.0) * np.log1p(r * r / nu)

    def update(self, z: float) -> None:
        """Re-weight particles by the terrain likelihood p(z | p_i).

        Log space: log w_i += log p(z | p_i) (Gaussian: −½ ((z − h(p_i)) / σ_z)² plus a
        constant). Working with logs avoids underflow: one 50 m misfit at σ_z = 3.6 m
        is a factor e^-96, which is already near the limit of float64.

        Particles outside the DEM get zero weight: the map is chosen to cover the
        mission area, so a hypothesis off the map is impossible. (Letting them read
        the clamped edge height would compare the measurement with fake flat
        terrain.) If every surviving particle is off the map, the clamped
        likelihood is used instead so the filter cannot collapse to all-zero weights.

        Also recorded, for divergence monitoring (no ground truth needed):
        - log evidence  log p(z_k | z_{1:k-1}) ≈ log Σ_i w_i p(z_k | x_i)
        - NIS  (z − ẑ)² / S  with ẑ = Σ w_i h(p_i),  S = Σ w_i (h(p_i) − ẑ)² + σ_z²
        """
        z = float(z)
        px, py = self.particles[:, 0], self.particles[:, 1]
        h = self.dem.height(px, py)
        loglik = self.log_likelihood(z - h)

        w_prev = self.weights
        z_hat = float(w_prev @ h)
        s_pred = float(w_prev @ (h - z_hat) ** 2) + self.meas_sigma**2
        self._last_nis = (z - z_hat) ** 2 / s_pred

        new = self.log_weights + np.where(self.dem.contains(px, py), loglik, -np.inf)
        evidence = logsumexp(new)
        if not np.isfinite(evidence):
            new = self.log_weights + loglik
            evidence = logsumexp(new)
        self._last_log_evidence = float(evidence)
        self.log_weights = new - evidence  # keep Σ w_i = 1 in log form

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean (MMSE estimate) and weighted covariance of the particle cloud.

        When the posterior is multimodal (several places match the measured terrain
        profile) the mean can sit between the modes, where there are no particles.
        That is expected behaviour, not a bug, and is worth showing in the report.
        """
        w = self.weights
        mean = w @ self.particles
        d = self.particles - mean
        cov = (d * w[:, None]).T @ d
        return mean, cov

    def roughen(self) -> None:
        """Roughening (Gordon, Salmond and Smith, 1993, Sec. 4).

        After resampling, many particles are exact copies. Add independent jitter
        with per-component std σ_j = K · E_j · N^(−1/d), where E_j is the spread
        (max − min) of component j, d = 4 is the state dimension and
        K = ``self.roughening`` (the paper used K = 0.2).
        """
        d = self.particles.shape[1]
        spread = np.ptp(self.particles, axis=0)
        sigma = self.roughening * spread * self.n ** (-1.0 / d)
        self.particles += self.rng.normal(0.0, 1.0, size=self.particles.shape) * sigma
