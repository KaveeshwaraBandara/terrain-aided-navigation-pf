"""Build DEM + filters from a config and run single or Monte Carlo experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import REPO_ROOT
from .dem import DEM
from .filters import (
    DeadReckoning,
    Estimator,
    ExtendedKalmanFilter,
    ParticleFilter,
    RaoBlackwellizedPF,
)
from .metrics import position_error
from .simulate import SimResult, filter_rng, initial_belief, run_streams, simulate


def build_dem(cfg: dict[str, Any]) -> DEM:
    region = cfg["region"]
    if region == "synthetic":
        return DEM.synthetic(**cfg.get("synthetic", {}))
    path = REPO_ROOT / cfg["dem_dir"] / f"{region}_utm44n.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first: python scripts/download_dem.py --region {region}"
        )
    return DEM.from_geotiff(path)


def make_filter(name: str, cfg: dict[str, Any], dem: DEM, rng: np.random.Generator) -> Estimator:
    if name == "dead_reckoning":
        return DeadReckoning(**cfg.get("dead_reckoning", {}))
    if name == "particle_filter":
        return ParticleFilter(dem, rng=rng, **cfg["particle_filter"])
    if name == "ekf":
        return ExtendedKalmanFilter(dem, **cfg["ekf"])
    if name == "rbpf":
        return RaoBlackwellizedPF(dem, rng=rng, **cfg["particle_filter"])
    raise ValueError(f"unknown filter '{name}'")


@dataclass
class FilterRun:
    status: str  # "ok" or "todo"
    message: str = ""
    est: np.ndarray | None = None  # (K, 4)
    cov: np.ndarray | None = None  # (K, 4, 4)
    error: np.ndarray | None = None  # (K,)
    extras: dict[str, Any] = field(default_factory=dict)


def run_filter(f: Estimator, sim: SimResult) -> FilterRun:
    mean, cov0 = initial_belief(sim)
    f.initialize(mean, cov0)
    K = sim.n_steps
    est = np.empty((K, 4))
    cov = np.empty((K, 4, 4))
    try:
        for k in range(K):
            est[k], cov[k] = f.step(sim.v_ins[k], sim.dt, float(sim.z[k]))
    except NotImplementedError as exc:
        return FilterRun(status="todo", message=str(exc))
    run = FilterRun(status="ok", est=est, cov=cov, error=position_error(est, sim.truth))
    if isinstance(f, ParticleFilter):
        run.extras = {
            "ess": np.array(f.history["ess"]),
            "resampled": np.array(f.history["resampled"], dtype=bool),
            "log_evidence": np.array(f.history["log_evidence"]),
            "nis": np.array(f.history["nis"]),
            "alarms": np.array(f.history["alarms"], dtype=bool),
            "snapshots": f.snapshots,
            "n_particles": f.n,
        }
    return run


def run_once(cfg: dict[str, Any], dem: DEM, run: int) -> tuple[SimResult, dict[str, FilterRun]]:
    """One Monte Carlo run with common random numbers: the flight and noise come from
    the run's named streams, and each filter family has its own stream."""
    seed = int(cfg["seed"])
    sim = simulate(cfg["simulation"], cfg["init"], dem, run_streams(seed, run))
    runs = {}
    for name in cfg["experiment"]["filters"]:
        f = make_filter(name, cfg, dem, filter_rng(seed, run, name))
        runs[name] = run_filter(f, sim)
    return sim, runs


def run_monte_carlo(cfg: dict[str, Any], dem: DEM | None = None):
    """Runs 0..M-1 of ``cfg['seed']`` (reproducible; serial — see tan.runner for sweeps)."""
    dem = dem if dem is not None else build_dem(cfg)
    n_runs = int(cfg["experiment"]["monte_carlo_runs"])
    results = [run_once(cfg, dem, i) for i in range(n_runs)]
    return dem, results
