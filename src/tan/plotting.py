"""Report figures. Uses matplotlib's object API (no pyplot), so it works headless.

Colours: fixed categorical slots per estimator so a filter keeps its colour in
every figure; ground truth is drawn in near-black ink.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure

from .dem import DEM
from .simulate import SimResult

INK = "#0b0b0b"
MUTED = "#52514e"
COLORS = {
    "particle_filter": "#2a78d6",  # blue
    "ekf": "#eb6834",  # orange
    "dead_reckoning": "#1baf7a",  # aqua
}
LABELS = {
    "particle_filter": "Particle filter",
    "ekf": "EKF",
    "dead_reckoning": "Dead reckoning (INS only)",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def _hillshade(ax, dem: DEM) -> None:
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(dem.elevation, vert_exag=2.0, dx=dem.res, dy=dem.res)
    ext = np.array(dem.extent) / 1000.0
    ax.imshow(shade, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.9)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_trajectories(dem: DEM, sim: SimResult, estimates: dict[str, np.ndarray], path) -> Path:
    fig = Figure(figsize=(7, 6.5))
    ax = fig.subplots()
    _hillshade(ax, dem)
    truth = np.vstack([sim.p0_true, sim.truth]) / 1000.0
    ax.plot(truth[:, 0], truth[:, 1], color=INK, lw=2.0, ls="--", label="Ground truth")
    for name, est in estimates.items():
        xy = est[:, :2] / 1000.0
        ax.plot(
            xy[:, 0], xy[:, 1], color=COLORS.get(name, MUTED), lw=2.0, label=LABELS.get(name, name)
        )
    ax.plot(
        *(sim.p0_guess / 1000.0),
        marker="x",
        ms=9,
        mew=2,
        color=INK,
        ls="none",
        label="Initial guess",
    )
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.set_title("Trajectories over terrain")
    return _save(fig, path)


def plot_errors(t: np.ndarray, errors: dict[str, np.ndarray], path) -> Path:
    """Position error vs time. ``errors[name]`` is (runs, K); band = 5–95th percentile."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    _style(ax)
    for name, err in errors.items():
        err = np.atleast_2d(err)
        c = COLORS.get(name, MUTED)
        ax.plot(t, np.median(err, axis=0), color=c, lw=2.0, label=LABELS.get(name, name))
        if err.shape[0] > 1:
            lo, hi = np.percentile(err, [5, 95], axis=0)
            ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title("Position error (median over runs, 5–95% band)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_ess(t: np.ndarray, ess: np.ndarray, n_particles: int, resampled: np.ndarray, path) -> Path:
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    _style(ax)
    ax.plot(t, ess / n_particles, color=COLORS["particle_filter"], lw=2.0)
    ax.plot(
        t[resampled],
        (ess / n_particles)[resampled],
        ls="none",
        marker="o",
        ms=4,
        color=INK,
        label="resampled",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("N_eff / N")
    ax.set_title("Particle filter effective sample size")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_particle_snapshots(
    dem: DEM, sim: SimResult, snapshots: list[tuple[int, np.ndarray]], path, max_panels: int = 4
) -> Path:
    """Particle cloud at a few instants — shows the multimodal posterior collapsing."""
    if not snapshots:
        raise ValueError("no snapshots recorded (set particle_filter.snapshot_every > 0)")
    pick = np.linspace(0, len(snapshots) - 1, min(max_panels, len(snapshots))).astype(int)
    fig = Figure(figsize=(3.2 * len(pick), 3.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(pick)))
    for j, (ax, i) in enumerate(zip(axes, pick, strict=True)):
        k, pts = snapshots[i]
        _hillshade(ax, dem)
        truth = sim.p0_true if k == 0 else sim.truth[k - 1]
        ax.scatter(
            pts[:, 0] / 1000,
            pts[:, 1] / 1000,
            s=2,
            color=COLORS["particle_filter"],
            alpha=0.4,
            linewidths=0,
        )
        ax.plot(*(truth / 1000), marker="+", ms=12, mew=2, color=INK)
        # zoom on the cloud (at least ±300 m) so a collapsed posterior is still visible
        centre = np.median(pts, axis=0)
        half = max(300.0, 1.2 * np.max(np.abs(np.percentile(pts, [1, 99], axis=0) - centre)))
        ax.set_xlim((centre[0] - half) / 1000, (centre[0] + half) / 1000)
        ax.set_ylim((centre[1] - half) / 1000, (centre[1] + half) / 1000)
        ax.set_title(f"t = {k * sim.dt:.0f} s")
        if j:
            ax.set_ylabel("")
    return _save(fig, path)
