"""Ground-truth flight and sensor simulation.

This is the *world*, not the estimator: it knows the true position and uses the
DEM directly. The filters only ever see ``v_ins`` (control input) and ``z``
(terrain-height measurement).

Timing convention: the true position at t=0 is ``p0_true``. Step k = 1..K covers
the interval (t_{k-1}, t_k]; arrays are indexed 0..K-1 for steps 1..K.

Common random numbers
---------------------
Every source of randomness has its own named stream (:data:`STREAMS`), and every
draw has a fixed size and is scaled afterwards (``sigma * standard_normal``). So
for Monte Carlo run *i*, changing a parameter (σ₀, the outlier rate, the INS grade,
the filter, …) changes only what that parameter scales: the flight, the sensor
noise and the initial-guess direction stay identical. The random-flight margin is
3·σ₀,max for every configuration, so the flight never depends on σ₀.

True system (what the PCRB in :mod:`tan.pcrb` assumes)::

    v_ins,k = v_true,k + b_k + n_k,       n_k ~ N(0, σ_n² I)
    b_k     = b_{k-1} + Δb_k,             Δb_k ~ N(0, σ_rw² dt I)
    p_k     = p_{k-1} + dt v_true,k
            = p_{k-1} + dt (v_ins,k − b_{k-1}) − dt (Δb_k + n_k)
    z_k     = baro_k − radar_k = s(p_k) + e_b,k − e_r,k
    s(p)    = DEM height + map error (+ offset)       — the surface the radar sees
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from .dem import DEM

STREAMS = ("flight", "ins", "altimeter", "outliers", "init", "map_error")
FILTER_STREAM_BASE = 1000


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
    radar_range: np.ndarray = field(default_factory=lambda: np.empty(0))  # (K,) true range, m
    outliers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))  # (K,)
    map_error: np.ndarray = field(default_factory=lambda: np.empty(0))  # (K,) s(p) − DEM(p), m

    @property
    def n_steps(self) -> int:
        return len(self.t)


# ---------------------------------------------------------------------- random streams
def run_streams(seed: int, run: int, plan: int | None = None) -> dict[str, np.random.Generator]:
    """Independent named generators for Monte Carlo run ``run`` of root seed ``seed``.

    With ``plan`` set, the flight stream depends only on (seed, plan): every run then
    flies the same flight plan with fresh sensor noise and initial error (used for the
    per-flight PCRB in Experiment 5).
    """
    streams = {
        name: np.random.default_rng([int(seed), int(run), i]) for i, name in enumerate(STREAMS)
    }
    if plan is not None:
        streams["flight"] = np.random.default_rng([int(seed), 0, 500 + int(plan)])
    return streams


def filter_rng(seed: int, run: int, family: str) -> np.random.Generator:
    """Generator for one filter family in one run — identical across configurations."""
    code = sum((i + 1) * ord(c) for i, c in enumerate(family))  # stable, unlike hash()
    return np.random.default_rng([int(seed), int(run), FILTER_STREAM_BASE + code])


def _as_streams(rng) -> dict[str, np.random.Generator]:
    if isinstance(rng, dict):
        return rng
    return dict(zip(STREAMS, rng.spawn(len(STREAMS)), strict=True))


# ---------------------------------------------------------------------- flight
def _weave(p0, heading0, phase, sim_cfg, t, dt):
    amp = np.deg2rad(float(sim_cfg["weave_amplitude_deg"]))
    heading = heading0 + amp * np.sin(2.0 * np.pi * t / float(sim_cfg["weave_period_s"]) + phase)
    v_true = float(sim_cfg["speed"]) * np.column_stack([np.cos(heading), np.sin(heading)])
    return p0 + np.cumsum(v_true * dt, axis=0), v_true


def _fixed_flight(dem: DEM, sim_cfg, t, dt):
    """The original deterministic flight: weave along the box diagonal from 20 % in."""
    xmin, xmax, ymin, ymax = dem.bounds
    p0 = np.array([xmin + 0.2 * (xmax - xmin), ymin + 0.2 * (ymax - ymin)])
    heading0 = np.arctan2(ymax - ymin, xmax - xmin)
    truth, v_true = _weave(p0, heading0, 0.0, sim_cfg, t, dt)
    if not dem.contains(truth[:, 0], truth[:, 1], margin=2 * dem.res).all():
        raise ValueError(
            "Simulated trajectory leaves the DEM. Shorten simulation.duration or lower speed."
        )
    return p0, truth, v_true


def flight_margin(sim_cfg) -> float:
    """Distance (m) the whole random flight keeps from the DEM edge: 3 σ₀,max."""
    return 3.0 * float(sim_cfg.get("flight", {}).get("margin_sigma0", 2000.0))


def _random_flight(dem: DEM, sim_cfg, t, dt, rng: np.random.Generator):
    """Random start, heading and weave phase; redraw until the flight ± margin is on the map."""
    fl = sim_cfg.get("flight", {})
    margin = flight_margin(sim_cfg)
    xmin, xmax, ymin, ymax = dem.bounds
    lo = np.array([xmin + margin, ymin + margin])
    hi = np.array([xmax - margin, ymax - margin])
    if np.any(hi <= lo):
        raise ValueError(f"DEM too small for a {margin:.0f} m flight margin")
    for _ in range(int(fl.get("max_tries", 1000))):
        p0 = rng.uniform(lo, hi)
        heading0, phase = rng.uniform(0.0, 2.0 * np.pi, size=2)
        truth, v_true = _weave(p0, heading0, phase, sim_cfg, t, dt)
        if dem.contains(truth[:, 0], truth[:, 1], margin=margin).all():
            return p0, truth, v_true
    raise ValueError("could not place a random flight inside the DEM margin; enlarge the region")


# ---------------------------------------------------------------------- map error
def correlated_field(shape, res: float, corr_length: float, rng: np.random.Generator):
    """Zero-mean, unit-std Gaussian random field with covariance exp(−d² / (2 ℓ²)).

    White noise convolved (via FFT) with a Gaussian kernel of std ℓ/√2 has exactly
    this covariance shape; the result is normalised to unit sample std.
    """
    white = rng.standard_normal(shape)
    if corr_length <= 0:
        return white
    s = corr_length / np.sqrt(2.0) / res  # kernel std in pixels
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    gain = np.exp(-2.0 * (np.pi * s) ** 2 * (fx**2 + fy**2))
    f = np.fft.irfft2(np.fft.rfft2(white) * gain, s=shape)
    return (f - f.mean()) / f.std()


def _corridor_max(dem: DEM, truth: np.ndarray, p0: np.ndarray, halfwidth: float) -> float:
    cache = dem.__dict__.setdefault("_max_cache", {})
    r = max(1, int(np.ceil(halfwidth / dem.res)))
    if r not in cache:
        cache[r] = DEM(maximum_filter(dem.elevation, size=2 * r + 1), dem.x0, dem.y0, dem.res)
    pts = np.vstack([p0, truth])
    return float(cache[r].height(pts[:, 0], pts[:, 1]).max())


# ---------------------------------------------------------------------- simulate
def simulate(sim_cfg: dict[str, Any], init_cfg: dict[str, Any], dem: DEM, rng) -> SimResult:
    """Simulate one flight. ``rng`` is a dict of named streams (:func:`run_streams`) or a
    single Generator (split into the named streams)."""
    streams = _as_streams(rng)
    dt = float(sim_cfg["dt"])
    n_steps = int(round(float(sim_cfg["duration"]) / dt))
    t = dt * np.arange(1, n_steps + 1)

    # --- true trajectory
    mode = sim_cfg.get("flight", {}).get("mode", "fixed")
    if mode == "fixed":
        p0_true, truth, v_true = _fixed_flight(dem, sim_cfg, t, dt)
    elif mode == "random":
        p0_true, truth, v_true = _random_flight(dem, sim_cfg, t, dt, streams["flight"])
    else:
        raise ValueError(f"unknown simulation.flight.mode '{mode}'")

    # --- INS: velocity with a slowly wandering bias → dead reckoning drifts
    ins = sim_cfg["ins"]
    g = streams["ins"]
    bias0 = float(ins["bias_sigma0"]) * g.standard_normal(2)
    bias_steps = float(ins["bias_rw_sigma"]) * np.sqrt(dt) * g.standard_normal((n_steps, 2))
    ins_noise = float(ins["noise_sigma"]) * g.standard_normal((n_steps, 2))
    bias = bias0 + np.cumsum(bias_steps, axis=0)
    v_ins = v_true + bias + ins_noise

    # --- the surface the radar sees: DEM + optional correlated map error + constant offset
    me = sim_cfg.get("map_error", {}) or {}
    map_err = np.full(n_steps, float(me.get("offset_m", 0.0)))
    if float(me.get("std_m", 0.0)) > 0:
        f = correlated_field(dem.shape, dem.res, float(me["corr_length_m"]), streams["map_error"])
        err_dem = DEM(f * float(me["std_m"]), dem.x0, dem.y0, dem.res)
        map_err += err_dem.height(truth[:, 0], truth[:, 1])
    terrain = dem.height(truth[:, 0], truth[:, 1]) + map_err

    # --- altimeters: barometric altitude (MSL) and radar altitude (above the surface)
    clearance = float(sim_cfg["clearance_m"])
    if sim_cfg.get("altitude_reference", "box") == "corridor":
        hw = float(sim_cfg.get("corridor_halfwidth_m", 1000.0))
        altitude_msl = _corridor_max(dem, truth, p0_true, hw) + clearance
    else:
        altitude_msl = float(dem.elevation.max() + clearance)
    a = streams["altimeter"]
    baro = altitude_msl + float(sim_cfg["baro_sigma"]) * a.standard_normal(n_steps)
    radar_range = altitude_msl - terrain
    radar = radar_range + float(sim_cfg["radar_sigma"]) * a.standard_normal(n_steps)
    o = streams["outliers"]
    u, n_out = o.random(n_steps), o.standard_normal(n_steps)
    outliers = u < float(sim_cfg.get("outlier_prob", 0.0))
    radar = radar + np.where(outliers, float(sim_cfg.get("outlier_sigma", 100.0)) * n_out, 0.0)
    z = baro - radar

    init_sigma = float(init_cfg["position_sigma"])
    p0_guess = p0_true + init_sigma * streams["init"].standard_normal(2)

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
        radar_range=radar_range,
        outliers=outliers,
        map_error=map_err,
    )


def initial_belief(sim: SimResult) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean (4,) and covariance (4, 4) every filter starts from."""
    mean = np.array([sim.p0_guess[0], sim.p0_guess[1], 0.0, 0.0])
    cov = np.diag([sim.init_sigma**2] * 2 + [sim.bias_sigma0**2] * 2)
    return mean, cov
