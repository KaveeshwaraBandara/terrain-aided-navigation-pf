"""Rao-Blackwellised (marginalised) particle filter — Schön, Gustafsson and Nordlund (2005).

The bias b enters the model linearly and the measurement does not depend on it::

    p_k = p_{k-1} + dt (u_k − b_{k-1}) + w_p,    w_p ~ N(0, σ_p² I)   (nonlinear part)
    b_k = b_{k-1} + w_b,                          w_b ~ N(0, σ_b² I)   (linear part)
    z_k = h(p_k) + v_k

So, conditioned on a particle's position history p_{0:k}, the bias is Gaussian and a
Kalman filter tracks it exactly. Particles carry only positions; each carries the
Kalman mean b̂_i of its bias. The Kalman *covariance* P depends only on the model
matrices, which are the same for every particle, so all particles share one P and one
Riccati recursion. One step:

1. sample   p_k ~ N(p_{k-1} + dt (u − b̂), dt² P + σ_p² I)        (bias marginalised out)
2. the sampled increment η = p_k − p_{k-1} − dt u = −dt b_{k-1} + w_p is a linear
   measurement of the bias:   S = dt² P + σ_p² I,  L = −dt P S⁻¹,
   b̂ ← b̂ + L (η + dt b̂),   P ← P − L S Lᵀ
3. bias time update          P ← P + σ_b² I
4. weights                    w_i ∝ w_i p(z_k | p_i)   (unchanged: z does not involve b)

Particles are stored as (N, 4) = [px, py, b̂x, b̂y] so the parent class's resampling and
weighting work unchanged. The state covariance adds the shared P to the spread of the
bias means.
"""

from __future__ import annotations

import numpy as np

from .particle_filter import ParticleFilter


class RaoBlackwellizedPF(ParticleFilter):
    name = "rbpf"

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("roughening", None)  # not used: positions are re-sampled every step
        super().__init__(*args, roughening=0.0, **kwargs)
        if self.hb_on or self.monitor_on:
            raise ValueError("RBPF supports neither the height-bias state nor the monitor")
        self.P_bias = np.eye(2)

    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=float)
        pos = self.rng.multivariate_normal(mean[:2], cov[:2, :2], size=self.n)
        self.particles = np.column_stack([pos, np.tile(mean[2:], (self.n, 1))])
        self.P_bias = np.array(cov[2:, 2:], dtype=float)
        self._reset_bookkeeping(mean, cov)

    def predict(self, u: np.ndarray, dt: float) -> None:
        u = np.asarray(u, dtype=float)
        P = self.P_bias
        S = dt * dt * P + self.sigma_pos**2 * np.eye(2)
        # 1. sample positions with the bias marginalised out
        chol = np.linalg.cholesky(S)
        b_hat = self.particles[:, 2:]
        mean = self.particles[:, :2] + dt * (u - b_hat)
        new_pos = mean + self.rng.standard_normal((self.n, 2)) @ chol.T
        # 2. the increment is a linear measurement of b_{k-1}
        eta = new_pos - self.particles[:, :2] - dt * u
        L = -dt * P @ np.linalg.inv(S)
        b_hat = b_hat + (eta + dt * b_hat) @ L.T
        P = P - L @ S @ L.T
        # 3. bias random walk
        P = P + self.sigma_bias**2 * np.eye(2)
        self.particles = np.column_stack([new_pos, b_hat])
        self.P_bias = 0.5 * (P + P.T)

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        mean, cov = super().estimate()  # spread of positions and of bias means
        cov[2:, 2:] += self.P_bias
        return mean, cov
