"""Parallel, resumable Monte Carlo runner with common random numbers.

An *experiment* (``configs/experiments/<name>.yaml``) has one or more *studies*. A
study is a grid of parameter values × regions × estimator variants × M runs.

- One **task** = one (study, configuration, run). It simulates the flight once and
  runs every estimator variant on that same flight (common random numbers, see
  :mod:`tan.simulate`). Run *i* uses the same seeds in every configuration.
- Each task writes ``results/<exp>/<study>/<config>/run_XXXX.npz`` atomically; a task
  whose file exists is skipped, so an interrupted sweep resumes where it stopped.
- Each npz holds the time series (float32) plus a JSON row per estimator with the
  scalar metrics; :func:`collect_summary` turns those into ``summary.csv``.

Experiment YAML keys (all optional except ``filters``)::

    name, description
    runs: 100                     # Monte Carlo runs per configuration
    regions: [hill_country, ...]
    set: {dotted.key: value}      # overrides for every study
    filters: {variant: {family: particle_filter, set: {...}}}
    grid: {dotted.key: [v1, v2]}  # Cartesian product
    quick: {runs: 3, set: {...}}  # --quick smoke-test mode
    studies: {study_name: {runs, regions, set, filters, grid}}   # override the above
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import REPO_ROOT, deep_update, dotted_to_nested, get_dotted, load_config, load_yaml
from .data import manifest_hash
from .dem import DEM
from .experiment import build_dem, make_filter, run_filter
from .metrics import (
    convergence_time,
    is_converged,
    is_false_lock,
    position_nees,
    tracking_window,
    two_sigma_radius,
)
from .simulate import filter_rng, run_streams, simulate
from .terrain import along_track_stats, roughness_dem

EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"
COMMON_FILE = EXPERIMENTS_DIR / "_common.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FAMILIES = ("dead_reckoning", "ekf", "particle_filter", "rbpf")
_UPPER = np.triu_indices(4)


# ---------------------------------------------------------------------- spec
@dataclass(frozen=True)
class Variant:
    key: str
    family: str
    overrides: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Configuration:
    study: str
    region: str
    params: dict[str, Any]
    cfg: dict[str, Any]
    variants: list[Variant]
    runs: int

    @property
    def id(self) -> str:
        parts = [self.region] + [f"{k}={_fmt(v)}" for k, v in self.params.items()]
        return "__".join(parts)

    def run_path(self, root: Path, run: int) -> Path:
        return root / self.study / self.id / f"run_{run:04d}.npz"


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).replace("/", "-").replace(" ", "")


def load_experiment(name: str) -> dict[str, Any]:
    path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment definition {path}")
    spec = load_yaml(path)
    spec.setdefault("name", name)
    return spec


def expand(spec: dict[str, Any], quick: bool = False) -> list[Configuration]:
    """All (study, region, grid point) configurations of an experiment."""
    common = load_yaml(COMMON_FILE) if COMMON_FILE.exists() else {}
    base_cfg = load_config()
    studies = spec.get("studies") or {"main": {}}
    out = []
    for study, sub in studies.items():
        sub = sub or {}
        runs = int(sub.get("runs", spec.get("runs", 100)))
        regions = sub.get("regions", spec.get("regions", ["synthetic"]))
        sets = [common.get("set", {}), spec.get("set", {}), sub.get("set", {})]
        if quick:
            q = deep_update(spec.get("quick", {}), sub.get("quick", {}) or {})
            runs = int(q.get("runs", 3))
            sets.append(common.get("quick", {}).get("set", {}))
            sets.append(q.get("set", {}))
        filters = sub.get("filters", spec.get("filters"))
        if not filters:
            raise ValueError(f"study '{study}' defines no filters")
        variants = []
        for key, v in filters.items():
            family = (v or {}).get("family", key)
            if family not in FAMILIES:
                raise ValueError(f"unknown filter family '{family}' for '{key}'")
            variants.append(Variant(key, family, dotted_to_nested((v or {}).get("set", {}))))
        grid = sub.get("grid", spec.get("grid", {})) or {}
        keys = list(grid)
        for region in regions:
            for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
                params = dict(zip(keys, values, strict=True))
                cfg = base_cfg
                for s in sets:
                    cfg = deep_update(cfg, dotted_to_nested(s or {}))
                cfg = deep_update(cfg, dotted_to_nested(params))
                cfg = deep_update(cfg, {"region": region})
                out.append(Configuration(study, region, params, cfg, variants, runs))
    return out


# ---------------------------------------------------------------------- one task
_DEMS: dict[str, DEM] = {}


def get_dem(cfg: dict[str, Any]) -> DEM:
    region = cfg["region"]
    if region == "synthetic":
        key = "synthetic:" + json.dumps(cfg.get("synthetic", {}), sort_keys=True)
    else:
        key = region
    if key not in _DEMS:
        _DEMS[key] = build_dem(cfg)
    return _DEMS[key]


def warm_caches(configs: list[Configuration]) -> None:
    """Load every DEM (and its roughness grid) once, before the worker processes fork,
    so they share the memory copy-on-write."""
    for c in configs:
        dem = get_dem(c.cfg)
        roughness_dem(dem)


def _summary_row(name, family, run, sim, est, cov, err, threshold, hold) -> dict[str, Any]:
    t = sim.t
    conv_t = convergence_time(err, t, threshold)
    converged = is_converged(err, t, threshold, hold)
    win = tracking_window(t, hold)
    final_cov = cov[-1, :2, :2]
    # along-/cross-track decomposition of the error w.r.t. the true direction of travel
    d = np.gradient(sim.truth, axis=0)
    tang = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    e = est[:, :2] - sim.truth
    e_along = np.sum(e * tang, axis=1)
    e_cross = e[:, 0] * (-tang[:, 1]) + e[:, 1] * tang[:, 0]
    try:
        nees = position_nees(est, cov, sim.truth)
    except np.linalg.LinAlgError:
        nees = np.full(len(t), np.nan)
    return {
        "filter": name,
        "family": family,
        "run": run,
        "final_error_m": float(err[-1]),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mse_m2": float(np.mean(err**2)),
        "convergence_time_s": conv_t,
        "converged": converged,
        "false_lock": is_false_lock(float(err[-1]), final_cov),
        "tracking_mse_m2": float(np.mean(err[win] ** 2)) if converged else float("nan"),
        "tracking_mse_along_m2": float(np.mean(e_along[win] ** 2)) if converged else float("nan"),
        "tracking_mse_cross_m2": float(np.mean(e_cross[win] ** 2)) if converged else float("nan"),
        "final_error_along_m": float(e_along[-1]),
        "final_error_cross_m": float(e_cross[-1]),
        "final_2sigma_radius_m": float(two_sigma_radius(final_cov)),
        "mean_nees_tracking": float(np.mean(nees[win])),
        "final_bias_error_mps": float(np.linalg.norm(est[-1, 2:] - sim.bias[-1])),
        "tracking_bias_rmse_mps": float(
            np.sqrt(np.mean(np.sum((est[win, 2:] - sim.bias[win]) ** 2, 1)))
        ),
    }


def run_task(conf: Configuration, run: int, out: Path) -> float:
    """Simulate run ``run`` of ``conf``, run all variants, save the npz. Returns seconds."""
    t0 = time.perf_counter()
    cfg = conf.cfg
    dem = get_dem(cfg)
    seed = int(cfg["seed"])
    plan = get_dotted(cfg, "simulation.flight.plan")
    sim = simulate(cfg["simulation"], cfg["init"], dem, run_streams(seed, run, plan))
    threshold = float(cfg["experiment"]["convergence_threshold_m"])
    hold = float(get_dotted(cfg, "experiment.tracking_window_s", 100.0))
    meas_sigma = float(cfg["particle_filter"]["meas_sigma"])

    arrays: dict[str, np.ndarray] = {
        "t": sim.t.astype(np.float32),
        "truth": sim.truth,
        "p0_true": sim.p0_true,
        "p0_guess": sim.p0_guess,
        "bias": sim.bias.astype(np.float32),
        "radar_range": sim.radar_range.astype(np.float32),
        "outliers": sim.outliers,
        "map_error": sim.map_error.astype(np.float32),
        "altitude_msl": np.array(sim.altitude_msl),
    }
    terrain = along_track_stats(dem, sim.truth, meas_sigma=meas_sigma)
    rows = []
    for v in conf.variants:
        vcfg = deep_update(cfg, v.overrides)
        f = make_filter(v.family, vcfg, dem, filter_rng(seed, run, v.family))
        tf = time.perf_counter()
        fr = run_filter(f, sim)
        elapsed = time.perf_counter() - tf
        if fr.status != "ok":
            raise RuntimeError(f"{v.key}: {fr.message}")
        k = v.key
        arrays[f"{k}__est"] = fr.est.astype(np.float32)
        arrays[f"{k}__cov"] = fr.cov[:, _UPPER[0], _UPPER[1]].astype(np.float32)
        arrays[f"{k}__err"] = fr.error.astype(np.float32)
        for name in ("ess", "resampled", "log_evidence", "nis", "alarms", "height_bias"):
            if name in fr.extras:
                arrays[f"{k}__{name}"] = np.asarray(fr.extras[name])
        row = _summary_row(k, v.family, run, sim, fr.est, fr.cov, fr.error, threshold, hold)
        row["sweep_runtime_s"] = elapsed  # informational only — see scripts/benchmark.py
        if "resampled" in fr.extras:
            row["n_resampled"] = int(np.sum(fr.extras["resampled"]))
        if "alarms" in fr.extras:
            row["n_alarms"] = int(np.sum(fr.extras["alarms"]))
        row.update(
            {
                "init_error_m": float(np.linalg.norm(sim.p0_guess - sim.p0_true)),
                "radar_range_max_m": float(sim.radar_range.max()),
                "n_outliers": int(sim.outliers.sum()),
                **{f"terrain_{a}": b for a, b in terrain.items()},
            }
        )
        rows.append(row)
    arrays["summary"] = np.array(json.dumps(rows))

    path = conf.run_path(out, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------- sweep
def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _work(item: tuple[int, int, str]) -> tuple[int, int, float]:
    ci, run, out = item
    return ci, run, run_task(_CONFIGS[ci], run, Path(out))


_CONFIGS: list[Configuration] = []


def results_dir(name: str, quick: bool) -> Path:
    return RESULTS_DIR / (f"{name}_quick" if quick else name)


def write_metadata(spec: dict[str, Any], configs: list[Configuration], out: Path, jobs: int):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout
    (meta / f"pip_freeze_{stamp}.txt").write_text(freeze)
    (meta / "experiment.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    resolved = {c.study + "/" + c.id: c.cfg for c in configs}
    (meta / "resolved_configs.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    info = {
        "started": stamp,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "data_manifest_sha256": manifest_hash(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jobs": jobs,
        "configurations": len(configs),
    }
    (meta / f"invocation_{stamp}.json").write_text(json.dumps(info, indent=2))
    return info


def sweep(name: str, quick: bool = False, jobs: int = 10, only: str | None = None) -> Path:
    """Run every unfinished task of experiment ``name``; returns the results directory."""
    import multiprocessing as mp

    global _CONFIGS
    spec = load_experiment(name)
    configs = expand(spec, quick)
    if only:
        configs = [c for c in configs if c.study == only]
    out = results_dir(name, quick)
    info = write_metadata(spec, configs, out, jobs)
    todo = [
        (ci, r, str(out))
        for ci, c in enumerate(configs)
        for r in range(c.runs)
        if not c.run_path(out, r).exists()
    ]
    total = sum(c.runs for c in configs)
    print(
        f"[{name}{' quick' if quick else ''}] {len(configs)} configurations, {total} tasks, "
        f"{total - len(todo)} already done, {len(todo)} to run on {jobs} workers "
        f"(commit {info['git_commit'][:8]}{' dirty' if info['git_dirty'] else ''})",
        flush=True,
    )
    if todo:
        _CONFIGS = configs
        warm_caches(configs)
        t0 = time.time()
        done = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(jobs, initializer=_init_worker) as pool:
            for _ci, _run, _sec in pool.imap_unordered(_work, todo, chunksize=1):
                done += 1
                if done % max(1, len(todo) // 40) == 0 or done == len(todo):
                    el = time.time() - t0
                    eta = el / done * (len(todo) - done)
                    print(
                        f"  {done}/{len(todo)} tasks  {el / 60:5.1f} min elapsed, "
                        f"ETA {eta / 60:5.1f} min",
                        flush=True,
                    )
    csv_path = collect_summary(name, quick)
    print(f"summary → {csv_path}", flush=True)
    return out


# ---------------------------------------------------------------------- collect
def collect_summary(name: str, quick: bool = False) -> Path:
    """Gather the per-run JSON rows of every finished task into ``summary.csv``."""
    import csv

    spec = load_experiment(name)
    configs = expand(spec, quick)
    out = results_dir(name, quick)
    rows: list[dict[str, Any]] = []
    for c in configs:
        for r in range(c.runs):
            p = c.run_path(out, r)
            if not p.exists():
                continue
            with np.load(p) as z:
                for row in json.loads(str(z["summary"])):
                    rows.append(
                        {"study": c.study, "region": c.region, "config": c.id, **c.params, **row}
                    )
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    path = out / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def load_runs(name: str, study: str, config_id: str, keys: list[str], quick: bool = False):
    """Stack arrays ``keys`` over all finished runs of one configuration → {key: (M, …)}."""
    folder = results_dir(name, quick) / study / config_id
    files = sorted(folder.glob("run_*.npz"))
    out: dict[str, list] = {k: [] for k in keys}
    for p in files:
        with np.load(p) as z:
            for k in keys:
                out[k].append(z[k])
    return {k: np.stack(v) if v else np.empty(0) for k, v in out.items()}


def unpack_cov(upper: np.ndarray) -> np.ndarray:
    """(…, 10) upper-triangle storage → (…, 4, 4) symmetric covariance."""
    cov = np.zeros(upper.shape[:-1] + (4, 4), dtype=np.float64)
    cov[..., _UPPER[0], _UPPER[1]] = upper
    cov[..., _UPPER[1], _UPPER[0]] = upper
    return cov
