"""Visual extras: particle-cloud snapshots, trajectories with a zoom inset, an animation.

The illustrative run is chosen by a fixed rule (stated in the captions), not by eye:
the lowest-index E1 run on the hill-country region at σ₀ = 2000 m in which PF (N = 20 000)
converged with a convergence time of at least 60 s (so acquisition is visible) and the
EKF did not converge. Falls back to the lowest-index converged PF run if none qualifies.
The run is re-simulated exactly from its seeds with the particle cloud recorded.
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure

from ..config import deep_update
from ..experiment import build_dem, make_filter, run_filter
from ..plotting import INK, TEXT_WIDTH_IN, add_scale_bar, report_style, save_report, style_of
from ..runner import results_dir
from ..simulate import filter_rng, run_streams, simulate
from .common import REGION_LABELS, Output, load_summary
from .e5 import resolved_cfg

SIGMA = "init.position_sigma"


def pick_run(d1, region: str, sigma: float) -> tuple[int, str, str]:
    g = d1[(d1["region"] == region) & (d1[SIGMA] == sigma)]
    pf = g[g["filter"] == "pf_20000"].set_index("run")
    ekf = g[g["filter"] == "ekf"].set_index("run")
    ok = pf[pf["converged"] & (pf["convergence_time_s"] >= 60) & ~ekf.loc[pf.index, "converged"]]
    rule = "primary"
    if ok.empty:
        ok, rule = pf[pf["converged"]], "fallback"
    run = int(ok.index.min())
    return run, str(g["config"].iloc[0]), rule


def _shade(dem):
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(dem.elevation, vert_exag=2.0, dx=dem.res, dy=dem.res)


def make(exp: str = "visuals", quick: bool = False) -> None:
    e1 = exp.replace("visuals", "e1", 1)
    out = Output(exp, quick)
    d1 = load_summary(e1, quick)
    region = "hill_country" if "hill_country" in set(d1["region"]) else d1["region"].iloc[0]
    sigma = 2000 if 2000 in set(d1[SIGMA]) else d1[SIGMA].max()
    run, config, rule = pick_run(d1, region, sigma)
    cfg = resolved_cfg(e1, quick, "main", config)
    dem = build_dem(cfg)
    seed = int(cfg["seed"])
    sim = simulate(cfg["simulation"], cfg["init"], dem, run_streams(seed, run))
    pf_cfg = deep_update(cfg, {"particle_filter": {"n_particles": 20000, "snapshot_every": 1}})
    pf = run_filter(
        make_filter("particle_filter", pf_cfg, dem, filter_rng(seed, run, "particle_filter")), sim
    )
    ekf = run_filter(make_filter("ekf", cfg, dem, filter_rng(seed, run, "ekf")), sim)
    dr = run_filter(
        make_filter("dead_reckoning", cfg, dem, filter_rng(seed, run, "dead_reckoning")), sim
    )
    # the re-run must reproduce the saved run exactly (same seeds, same code path)
    saved = np.load(results_dir(e1, quick) / "main" / config / f"run_{run:04d}.npz")
    reproduced = bool(np.allclose(saved["pf_20000__err"], pf.error, atol=1e-2))
    snaps = pf.extras["snapshots"]
    shade = _shade(dem)
    ext = np.array(dem.extent) / 1e3
    info = {
        "region": region,
        "sigma0": float(sigma),
        "run": run,
        "rule": rule,
        "reproduced": reproduced,
        "convergence_time_s": float(
            d1[(d1["config"] == config) & (d1["run"] == run) & (d1["filter"] == "pf_20000")][
                "convergence_time_s"
            ].iloc[0]
        ),
    }
    (out.root / "figures").mkdir(parents=True, exist_ok=True)
    (out.root / "figures" / f"{exp}_run.json").write_text(json.dumps(info, indent=2))

    # ------------------------------------------------------------ snapshots
    times = [0, 5, 20, 60, 150, sim.n_steps]
    times = [k for k in times if k <= sim.n_steps]
    rng = np.random.default_rng(0)
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 4.6), layout="constrained")
        axes = fig.subplots(2, 3).ravel()
        for ax, k in zip(axes, times, strict=False):
            pts = snaps[k][1] / 1e3
            sub = pts[rng.choice(len(pts), size=min(4000, len(pts)), replace=False)]
            truth = (sim.p0_true if k == 0 else sim.truth[k - 1]) / 1e3
            centre = np.median(pts, axis=0)
            half = max(
                0.4,
                1.15 * float(np.max(np.abs(np.percentile(pts, [1, 99], 0) - centre))),
                1.15 * float(np.max(np.abs(truth - centre))),
            )
            ax.imshow(
                shade,
                cmap="gray",
                extent=ext,
                origin="upper",
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )
            ax.scatter(
                sub[:, 0], sub[:, 1], s=1.2, color=style_of("pf_20000")["color"], alpha=0.35, lw=0
            )
            ax.plot(*truth, marker="+", ms=11, mew=2, color=INK)
            if k > 0:
                ax.plot(
                    *(pf.est[k - 1, :2] / 1e3),
                    marker="x",
                    ms=7,
                    mew=1.6,
                    color=style_of("pf_20000")["color"],
                )
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_aspect("equal")
            ax.text(
                0.03,
                0.97,
                f"t = {k * sim.dt:.0f} s",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "lw": 0, "pad": 1.5},
            )
            ax.set_xticks([])
            ax.set_yticks([])
            add_scale_bar(ax, max(0.1, float(f"{half / 2.5:.1g}")))
        save_report(fig, out.figure("snapshots"))

    # ------------------------------------------------------------ trajectories + zoom inset
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 0.78 * TEXT_WIDTH_IN), layout="constrained")
        ax = fig.subplots()
        ax.imshow(shade, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.9)
        tr = np.vstack([sim.p0_true, sim.truth]) / 1e3
        lines = {"dead_reckoning": dr.est, "ekf": ekf.est, "pf_20000": pf.est}

        def draw(a, lw=1.6):
            a.plot(tr[:, 0], tr[:, 1], color=INK, lw=lw + 0.4, ls="-", label="Ground truth")
            for name, est in lines.items():
                st = style_of(name)
                a.plot(
                    est[:, 0] / 1e3,
                    est[:, 1] / 1e3,
                    color=st["color"],
                    ls=st["ls"],
                    lw=lw,
                    label=st["label"],
                )
            a.plot(
                *(sim.p0_guess / 1e3),
                marker="x",
                ms=8,
                mew=2,
                color=INK,
                ls="none",
                label="Initial guess",
            )

        draw(ax)
        allxy = np.vstack([tr] + [e[:, :2] / 1e3 for e in lines.values()])
        pad = 2.0
        ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
        ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km, UTM 44N)")
        ax.set_ylabel("Northing (km, UTM 44N)")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        # zoom on where the PF locks on: the true track from 20 s before to 20 s after the
        # convergence time (estimates far from the truth simply enter from outside the box)
        kc = int(info["convergence_time_s"] / sim.dt)
        zoom = tr[max(0, kc - 20) : min(len(tr), kc + 20)]
        c, half = zoom.mean(0), max(0.4, 0.6 * float(np.ptp(zoom, axis=0).max()))
        z0, z1 = c - half, c + half
        ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
        ins.imshow(shade, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1, alpha=0.9)
        draw(ins, lw=1.2)
        ins.set_xlim(z0[0], z1[0])
        ins.set_ylim(z0[1], z1[1])
        ins.set_aspect("equal")
        ins.set_xticks([])
        ins.set_yticks([])
        ax.indicate_inset_zoom(ins, edgecolor=INK)
        add_scale_bar(ax, 2.0)
        save_report(fig, out.figure("trajectory"))

    # ------------------------------------------------------------ animation (MP4 + GIF)
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames = list(range(0, min(sim.n_steps, 180) + 1, 2))
    with report_style():
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.imshow(shade, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=1)
        cloud = ax.scatter([], [], s=1.0, color=style_of("pf_20000")["color"], alpha=0.35, lw=0)
        (truth_pt,) = ax.plot([], [], marker="+", ms=12, mew=2, color=INK, ls="none")
        (track,) = ax.plot([], [], color=INK, lw=1.0)
        label = ax.text(
            0.02,
            0.97,
            "",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "lw": 0},
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        stem = out.figure("animation")
        for writer, suffix in (
            (FFMpegWriter(fps=12, bitrate=900), ".mp4"),
            (PillowWriter(fps=8), ".gif"),
        ):
            with writer.saving(
                fig, str(stem.with_suffix(suffix)), dpi=100 if suffix == ".mp4" else 60
            ):
                for k in frames if suffix == ".mp4" else frames[::2]:
                    pts = snaps[k][1] / 1e3
                    sub = pts[rng.choice(len(pts), size=min(3000, len(pts)), replace=False)]
                    truth = (sim.p0_true if k == 0 else sim.truth[k - 1]) / 1e3
                    centre = np.median(pts, axis=0)
                    half = max(
                        0.6,
                        1.2 * float(np.max(np.abs(np.percentile(pts, [1, 99], 0) - centre))),
                        1.2 * float(np.max(np.abs(truth - centre))),
                    )
                    cloud.set_offsets(sub)
                    truth_pt.set_data([truth[0]], [truth[1]])
                    track.set_data(tr[: k + 1, 0], tr[: k + 1, 1])
                    label.set_text(
                        f"{REGION_LABELS.get(region, region)}, σ₀ = {sigma:g} m\n"
                        f"t = {k * sim.dt:.0f} s, N = 20 000"
                    )
                    ax.set_xlim(centre[0] - half, centre[0] + half)
                    ax.set_ylim(centre[1] - half, centre[1] + half)
                    writer.grab_frame()
    print(f"[{exp}] run {run} ({rule} rule, reproduced={reproduced}) → {out.root}")
