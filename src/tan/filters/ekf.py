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

    # ------------------------------------------------------------------ filter maths
    def predict(self, u: np.ndarray, dt: float) -> None:
        """x⁻ = F x + G u,  P⁻ = F P Fᵀ + Q.

        The motion model is linear, so this half of the EKF is an exact Kalman
        prediction. Only the measurement update needs linearising.
        """
        F = np.eye(4)
        F[:2, 2:] = -dt * np.eye(2)
        self.x[:2] += dt * (np.asarray(u, dtype=float) - self.x[2:])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: float) -> None:
        """Linearised measurement update.

            H = [∂h/∂px, ∂h/∂py, 0, 0] at x⁻     (local terrain slope from the DEM)
            y = z − h(x⁻)                          (innovation)
            S = H P⁻ Hᵀ + R,   K = P⁻ Hᵀ / S
            x⁺ = x⁻ + K y
            P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ   (Joseph form: stays symmetric PSD)

        The weakness is H: it is the slope over one 30 m DEM cell, but the position
        uncertainty can be hundreds of metres. Over that range the terrain is nothing
        like a plane, so a large innovation gets "explained" by the wrong slope.
        """
        px, py = self.x[0], self.x[1]
        h = float(self.dem.height(px, py))
        gx, gy = self.dem.gradient(px, py)
        H = np.array([[float(gx), float(gy), 0.0, 0.0]])

        S = float((H @ self.P @ H.T)[0, 0]) + self.R
        K = (self.P @ H.T) / S  # (4, 1)
        self.x = self.x + K[:, 0] * (float(z) - h)

        A = np.eye(4) - K @ H
        self.P = A @ self.P @ A.T + self.R * (K @ K.T)
        self.P = 0.5 * (self.P + self.P.T)  # remove round-off asymmetry
