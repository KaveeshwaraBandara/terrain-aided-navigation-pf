"""Common interface for every estimator in the comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

STATE_DIM = 4  # [px, py, bx, by]


class Estimator(ABC):
    """Recursive Bayesian estimator for x = [px, py, bx, by].

    The experiment runner only calls :meth:`initialize` once and then :meth:`step`
    for every time step, so each filter is free to organise its internals.
    """

    name: str = "estimator"

    @abstractmethod
    def initialize(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """Set the prior belief p(x_0) = N(mean, cov)."""

    @abstractmethod
    def predict(self, u: np.ndarray, dt: float) -> None:
        """Time update with INS velocity ``u`` (2,) over ``dt`` seconds."""

    @abstractmethod
    def update(self, z: float) -> None:
        """Measurement update with terrain-height measurement ``z`` (m)."""

    @abstractmethod
    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current state estimate (4,) and its covariance (4, 4)."""

    def step(self, u: np.ndarray, dt: float, z: float) -> tuple[np.ndarray, np.ndarray]:
        self.predict(u, dt)
        self.update(z)
        return self.estimate()
