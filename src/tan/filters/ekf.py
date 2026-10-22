"""Extended Kalman filter baseline — linearises the terrain around the current estimate.

Included to *show* why TAN needs a particle filter: with a large initial error
or rugged terrain the local slope says little about where the UAV really is,
and the EKF diverges or locks onto the wrong hill.

Same models as the particle filter (x = [px, py, bx, by]):

    x_k = F x_{k-1} + G u_k + w,   F = [[I, -dt I], [0, I]],  G = [[dt I], [0]]
    z_k = h(p_k) + v,              H_k = [∂h/∂px, ∂h/∂py, 0, 0] at the predicted state
"""

from __future__ import annotations

import numpy as np

from ..dem import DEM
from .base import Estimator


class ExtendedKalmanFilter(Estimator):
    name = "ekf"

    def __init__(
        self,
        dem: DEM,
        meas_sigma: float = 3.6,
        process_sigma_pos: float = 2.0,
        process_sigma_bias: float = 0.002,
    ) -> None:
        self.dem = dem
        self.R = float(meas_sigma) ** 2
        self.Q = np.diag([process_sigma_pos**2] * 2 + [process_sigma_bias**2] * 2)
        self.x = np.zeros(4)
        self.P = np.eye(4)

    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.x = np.array(mean, dtype=float)
        self.P = np.array(cov, dtype=float)

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()

    # ------------------------------------------------------------------ TODO(you)
    def predict(self, u: np.ndarray, dt: float) -> None:
        """x⁻ = F x + G u,  P⁻ = F P Fᵀ + Q."""
        raise NotImplementedError("ExtendedKalmanFilter.predict")

    def update(self, z: float) -> None:
        """Linearise h with ``self.dem.gradient`` at the predicted position, then the usual
        K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹ update. Use the Joseph form for P to keep it symmetric PSD."""
        raise NotImplementedError("ExtendedKalmanFilter.update")
