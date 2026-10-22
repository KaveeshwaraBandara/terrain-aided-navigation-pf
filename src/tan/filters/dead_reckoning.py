"""INS-only dead reckoning — the "no terrain information" baseline.

Implemented on purpose: it exercises the whole pipeline end to end before the
real filters exist, and it is the lower bar every filter must beat.
"""

from __future__ import annotations

import numpy as np

from .base import Estimator


class DeadReckoning(Estimator):
    name = "dead_reckoning"

    def __init__(self, process_sigma_pos: float = 0.0) -> None:
        self.q_pos = float(process_sigma_pos) ** 2
        self.x = np.zeros(4)
        self.P = np.eye(4)

    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.x = np.array(mean, dtype=float)
        self.P = np.array(cov, dtype=float)

    def predict(self, u: np.ndarray, dt: float) -> None:
        # Integrate the INS velocity, corrected by the (never-updated) bias guess.
        self.x[:2] += dt * (np.asarray(u) - self.x[2:])
        F = np.eye(4)
        F[:2, 2:] = -dt * np.eye(2)
        self.P = F @ self.P @ F.T
        self.P[:2, :2] += self.q_pos * np.eye(2)

    def update(self, z: float) -> None:
        """No-op: dead reckoning ignores the altimeter."""

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.copy(), self.P.copy()
