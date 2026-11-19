"""Posterior Cramér–Rao bound (PCRB) for terrain-aided navigation.

Tichavský, Muravchik and Nehorai (1998) give a recursion for the Bayesian information
matrix J_k of the state x_k, whose inverse bounds the MSE matrix of *any* estimator:
E[(x̂_k − x_k)(x̂_k − x_k)ᵀ] ⪰ J_k⁻¹. For linear dynamics with additive Gaussian noise
x_k = F x_{k-1} + G u_k + w_k, w_k ~ N(0, Q), and a measurement z_k = h(x_k) + v_k,
v_k ~ N(0, R), it reduces to

    J_k = (Q + F J_{k-1}⁻¹ Fᵀ)⁻¹ + E[H_kᵀ R⁻¹ H_k],     J_0 = P_0⁻¹,
    H_k = ∂h/∂x at the true state = [∇h(p_k)ᵀ, 0, 0].

The first term is the Kalman prediction of the information; the second is the
terrain information the altimeter adds, averaged over the distribution of the true
state. Because it only involves the *slope at the truth*, the bound ignores
ambiguity (other places with the same height profile): it is the error a filter
would reach if it already knew which hill it was over. It is therefore expected to
be tight in the tracking phase and loose during acquisition.

Two versions are used in the report:

- ensemble bound: the expectation is the average over the Monte Carlo ensemble of
  random flights. With uniformly random flight placement the filters' prior
  N(guess, σ₀²) is the exact conditional distribution of the true start, so this
  bounds the ensemble MSE that the experiments report.
- per-flight bound: H at one fixed flight path — the local (small-error) bound for
  that path, compared with the tracking error of runs that fly it.

The bound uses the *true* noise of the simulation (:func:`true_noise`), not the
filters' tuned process noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dem import DEM


def transition(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[:2, 2:] = -dt * np.eye(2)
    return F


def true_noise(sim_cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Process-noise covariance Q and measurement variance R of the *simulated* system.

    From :mod:`tan.simulate`: p_k = p_{k-1} + dt (u_k − b_{k-1}) − dt (Δb_k + n_k) and
    b_k = b_{k-1} + Δb_k, so w = [−dt (Δb + n); Δb] with, per axis,
    var(w_p) = dt² (q_b + q_n), cov(w_p, w_b) = −dt q_b, var(w_b) = q_b,
    where q_b = σ_rw² dt and q_n = σ_n². The altimeter difference has R = σ_baro² + σ_radar².
    """
    dt = float(sim_cfg["dt"])
    ins = sim_cfg["ins"]
    q_b = float(ins["bias_rw_sigma"]) ** 2 * dt
    q_n = float(ins["noise_sigma"]) ** 2
    i2 = np.eye(2)
    Q = np.block([[dt * dt * (q_b + q_n) * i2, -dt * q_b * i2], [-dt * q_b * i2, q_b * i2]])
    R = float(sim_cfg["baro_sigma"]) ** 2 + float(sim_cfg["radar_sigma"]) ** 2
    return Q, R


def slope_outer(dem: DEM, truth: np.ndarray) -> np.ndarray:
    """∇h ∇hᵀ at true positions. ``truth`` (..., K, 2) → (..., K, 2, 2)."""
    gx, gy = dem.bilinear_gradient(truth[..., 0], truth[..., 1])  # gradient of the true h
    g = np.stack([gx, gy], axis=-1)
    return g[..., :, None] * g[..., None, :]


def pcrb(
    expected_slope_outer: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: float, dt: float
) -> np.ndarray:
    """Bound covariances J_k⁻¹ for k = 1..K. ``expected_slope_outer`` is (K, 2, 2)."""
    F = transition(dt)
    J = np.linalg.inv(P0)
    out = np.empty((len(expected_slope_outer), 4, 4))
    for k, G in enumerate(expected_slope_outer):
        J = np.linalg.inv(Q + F @ np.linalg.solve(J, F.T))
        J[:2, :2] += G / R
        out[k] = np.linalg.inv(J)
    return out


def position_rmse_bound(bound_cov: np.ndarray) -> np.ndarray:
    """√trace of the position block: lower bound on the position RMSE, per step."""
    return np.sqrt(bound_cov[:, 0, 0] + bound_cov[:, 1, 1])


def mismatched_kf_error(
    slopes: np.ndarray,
    P0: np.ndarray,
    Q_filter: np.ndarray,
    R_filter: float,
    Q_true: np.ndarray,
    R_true: float,
    dt: float,
) -> np.ndarray:
    """Actual error covariance of a Kalman filter that knows the true terrain slopes along the
    path (``slopes`` (K, 2)) but computes its gains with its own tuning (Q_filter, R_filter),
    while the world has (Q_true, R_true):

        believed:  P⁻ = F P Fᵀ + Q_f,   K = P⁻Hᵀ / (H P⁻ Hᵀ + R_f),   P = (I − KH) P⁻
        actual:    A⁻ = F A Fᵀ + Q_t,   A = (I − KH) A⁻ (I − KH)ᵀ + K R_t Kᵀ

    With matched models this is the PCRB recursion (Kalman covariance); with the filters'
    inflated process noise it is the small-error performance their tuning allows.
    """
    F = transition(dt)
    P = np.array(P0, dtype=float)
    A = np.array(P0, dtype=float)
    out = np.empty((len(slopes), 4, 4))
    eye = np.eye(4)
    for k, g in enumerate(slopes):
        H = np.array([[g[0], g[1], 0.0, 0.0]])
        P = F @ P @ F.T + Q_filter
        A = F @ A @ F.T + Q_true
        S = float((H @ P @ H.T)[0, 0]) + R_filter
        K = P @ H.T / S
        IKH = eye - K @ H
        P = IKH @ P
        A = IKH @ A @ IKH.T + R_true * (K @ K.T)
        out[k] = A
    return out
