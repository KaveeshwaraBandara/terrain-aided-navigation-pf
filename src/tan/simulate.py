"""Ground-truth flight and sensor simulation.

This is the *world*, not the estimator: it knows the true position and uses the
DEM directly. The filters only ever see ``v_ins`` (control input) and ``z``
(terrain-height measurement).

Timing convention: the true position at t=0 is ``p0_true``. Step k = 1..K covers
the interval (t_{k-1}, t_k]; arrays are indexed 0..K-1 for steps 1..K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dem import DEM


@dataclass
class SimResult:
    t: np.ndarray  # (K,)  time at the end of each step, s
    dt: float
    p0_true: np.ndarray  # (2,)  true position at t=0, m
    truth: np.ndarray  # (K, 2) true position, m
    bias: np.ndarray  # (K, 2) true INS velocity bias, m/s
    v_ins: np.ndarray  # (K, 2) INS velocity = true velocity + bias + noise (control input u_k)
    z: np.ndarray  # (K,)  baro altitude - radar altitude  ≈ terrain height under the UAV
    altitude_msl: float
    p0_guess: np.ndarray  # (2,)  initial position guess handed to the filters
    init_sigma: float  # std of the initial position error per axis, m
    bias_sigma0: float  # std of the initial INS bias per axis, m/s

    @property
    def n_steps(self) -> int:
        return len(self.t)


def simulate(
    sim_cfg: dict[str, Any], init_cfg: dict[str, Any], dem: DEM, rng: np.random.Generator
) -> SimResult:
    dt = float(sim_cfg["dt"])
    n_steps = int(round(float(sim_cfg["duration"]) / dt))
    t = dt * np.arange(1, n_steps + 1)

    # --- true trajectory: constant speed, weaving around the box diagonal
    xmin, xmax, ymin, ymax = dem.bounds
    p0_true = np.array([xmin + 0.2 * (xmax - xmin), ymin + 0.2 * (ymax - ymin)])
    heading0 = np.arctan2(ymax - ymin, xmax - xmin)
    amp = np.deg2rad(float(sim_cfg["weave_amplitude_deg"]))
    heading = heading0 + amp * np.sin(2.0 * np.pi * t / float(sim_cfg["weave_period_s"]))
    v_true = float(sim_cfg["speed"]) * np.column_stack([np.cos(heading), np.sin(heading)])
    truth = p0_true + np.cumsum(v_true * dt, axis=0)

    if not dem.contains(truth[:, 0], truth[:, 1], margin=2 * dem.res).all():
        raise ValueError(
            "Simulated trajectory leaves the DEM. Shorten simulation.duration or lower speed."
        )

    # --- INS: velocity with a slowly wandering bias → dead reckoning drifts
    ins = sim_cfg["ins"]
    bias0 = rng.normal(0.0, float(ins["bias_sigma0"]), size=2)
    bias_steps = rng.normal(0.0, float(ins["bias_rw_sigma"]) * np.sqrt(dt), size=(n_steps, 2))
    bias = bias0 + np.cumsum(bias_steps, axis=0)
    v_ins = v_true + bias + rng.normal(0.0, float(ins["noise_sigma"]), size=(n_steps, 2))

    # --- altimeters: barometric altitude (MSL) and radar altitude (above ground)
    altitude_msl = float(dem.elevation.max() + float(sim_cfg["clearance_m"]))
    terrain = dem.height(truth[:, 0], truth[:, 1])
    baro = altitude_msl + rng.normal(0.0, float(sim_cfg["baro_sigma"]), size=n_steps)
    radar = altitude_msl - terrain + rng.normal(0.0, float(sim_cfg["radar_sigma"]), size=n_steps)
    outliers = rng.random(n_steps) < float(sim_cfg.get("outlier_prob", 0.0))
    radar[outliers] += rng.normal(0.0, float(sim_cfg.get("outlier_sigma", 100.0)), outliers.sum())
    z = baro - radar

    init_sigma = float(init_cfg["position_sigma"])
    p0_guess = p0_true + rng.normal(0.0, init_sigma, size=2)

    return SimResult(
        t=t,
        dt=dt,
        p0_true=p0_true,
        truth=truth,
        bias=bias,
        v_ins=v_ins,
        z=z,
        altitude_msl=altitude_msl,
        p0_guess=p0_guess,
        init_sigma=init_sigma,
        bias_sigma0=float(ins["bias_sigma0"]),
    )


def initial_belief(sim: SimResult) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean (4,) and covariance (4, 4) every filter starts from."""
    mean = np.array([sim.p0_guess[0], sim.p0_guess[1], 0.0, 0.0])
    cov = np.diag([sim.init_sigma**2] * 2 + [sim.bias_sigma0**2] * 2)
    return mean, cov
