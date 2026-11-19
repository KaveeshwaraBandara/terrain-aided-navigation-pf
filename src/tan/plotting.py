"""Report figures. Uses matplotlib's object API (no pyplot), so it works headless.

Colours: fixed categorical slots per estimator so a filter keeps its colour in
every figure; ground truth is drawn in near-black ink. The slots come from a
palette validated for colour-vision deficiency (worst adjacent CVD ΔE 9.2 in
OKLab ×100). Three of the hues are below 3:1 contrast on white, so every
estimator also has its own line style and figures always carry a legend.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.figure import Figure

from .dem import DEM
from .simulate import SimResult

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
_BLUE, _ORANGE, _AQUA, _VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
_MAGENTA, _YELLOW, _GREEN, _RED = "#e87ba4", "#eda100", "#008300", "#e34948"
COLORS = {
    "particle_filter": _BLUE,
    "pf_5000": _BLUE,
    "ekf": _ORANGE,
    "dead_reckoning": _AQUA,
    "pf_20000": _VIOLET,
    "rbpf": _MAGENTA,
    "pf_monitor": _YELLOW,
    "pf_student_t": _GREEN,
    # E4 remedies: σ_z inflation is a parameter of the same PF (blue, new line styles);
    # the offset-state PF is a different estimator (palette slot 8)
    "pf_sz_x2": _BLUE,
    "pf_sz_x4": _BLUE,
    "pf_hbias": _RED,
    # E6: roughening is a parameter of the PF (blue); the monitor keeps its colour
    "pf_rough": _BLUE,
    "pf_monitor_rough": _YELLOW,
}
LINESTYLES = {
    "particle_filter": "-",
    "pf_5000": "-",
    "ekf": "--",
    "dead_reckoning": ":",
    "pf_20000": "-.",
    "rbpf": (0, (5, 1, 1, 1)),
    "pf_monitor": (0, (3, 1)),
    "pf_student_t": (0, (1, 1)),
    "pf_sz_x2": (0, (4, 2)),
    "pf_sz_x4": (0, (1, 1.5)),
    "pf_hbias": "-.",
    "pf_rough": ":",
    "pf_monitor_rough": ":",
}
LABELS = {
    "particle_filter": "Particle filter",
    "pf_5000": "PF (N = 5 000)",
    "ekf": "EKF",
    "dead_reckoning": "Dead reckoning (INS only)",
    "pf_20000": "PF (N = 20 000)",
    "rbpf": "RBPF",
    "pf_monitor": "PF + divergence monitor",
    "pf_student_t": "PF, Student-t likelihood",
    "pf_sz_x2": r"PF, $\sigma_z \times 2$",
    "pf_sz_x4": r"PF, $\sigma_z \times 4$",
    "pf_hbias": "PF + offset state",
    "pf_rough": "PF + roughening",
    "pf_monitor_rough": "PF + monitor + roughening",
}
ELEVATION_CMAP = LinearSegmentedColormap.from_list("elev", ["#f6f1e7", "#c9a77c", "#5b3a1e"])

# Report figures: single-column article, text width ≈ 6.3 in; 9 pt text.
TEXT_WIDTH_IN = 6.3
REPORT_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 200,
}


@contextmanager
def report_style():
    """rcParams are read when artists are created, so build the whole figure inside this."""
    with mpl.rc_context(REPORT_RC):
        yield


def save_report(fig: Figure, stem: str | Path) -> list[Path]:
    """Save ``stem.pdf`` (vector, for LaTeX) and ``stem.png`` (preview)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else {})
        out.append(p)
    return out


def style_of(name: str) -> dict:
    """Colour, line style and label for an estimator key (falls back to muted grey)."""
    return {
        "color": COLORS.get(name, MUTED),
        "ls": LINESTYLES.get(name, "-"),
        "label": LABELS.get(name, name),
    }


def add_scale_bar(ax, length_km: float, units_per_km: float = 1.0, loc=(0.06, 0.06)) -> None:
    """Black/white scale bar in axes-fraction position ``loc`` (data units per km given)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    length = length_km * units_per_km
    h = 0.012 * (y1 - y0)
    ax.add_patch(mpl.patches.Rectangle((bx, by), length, h, color=INK, lw=0))
    ax.add_patch(
        mpl.patches.Rectangle((bx + length / 2, by), length / 2, h, color="white", ec=INK, lw=0.5)
    )
    ax.text(
        bx + length / 2,
        by + 2.2 * h,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=7,
        color=INK,
        bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1},
    )


def plot_region_map(elev, transform, boxes: dict, anchors: dict, stem) -> list[Path]:
    """Sri Lanka elevation (COP90, lat/lon) with the three study boxes."""
    t = transform
    h, w = elev.shape
    extent = (t.c, t.c + w * t.a, t.f + h * t.e, t.f)
    with report_style():
        fig = Figure(figsize=(0.55 * TEXT_WIDTH_IN, 0.55 * TEXT_WIDTH_IN * 1.45))
        ax = fig.subplots()
        ls = LightSource(azdeg=315, altdeg=45)
        filled = np.nan_to_num(elev, nan=0.0)
        shade = ls.hillshade(filled, vert_exag=0.02, dx=1.0, dy=1.0)
        im = ax.imshow(
            np.ma.masked_invalid(elev),
            cmap=ELEVATION_CMAP,
            extent=extent,
            vmin=0,
            vmax=np.nanmax(elev),
            interpolation="nearest",
        )
        ax.imshow(
            np.ma.masked_where(~np.isfinite(elev), shade),
            cmap="gray",
            alpha=0.25,
            extent=extent,
            interpolation="nearest",
        )
        for key, b in boxes.items():
            ax.add_patch(
                mpl.patches.Rectangle(
                    (b["west"], b["south"]),
                    b["east"] - b["west"],
                    b["north"] - b["south"],
                    fill=False,
                    ec=INK,
                    lw=1.2,
                )
            )
            town, lat, lon = anchors[key]
            ax.plot(lon, lat, marker="o", ms=3, color=INK)
            ax.text(
                b["east"] + 0.06,
                0.5 * (b["south"] + b["north"]),
                town,
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_xlim(79.5, 82.0)
        ax.set_ylim(5.85, 9.9)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(7.8)))
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("Elevation (m)")
        cb.outline.set_visible(False)
        return save_report(fig, stem)


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
