"""Live, animated view of one simulated flight: the particle cloud, the truth and every filter's
estimate moving over the terrain, with the position errors updating beside it.

Usage:
    python scripts/live_demo.py                                  # hill country, sigma0 = 500 m
    python scripts/live_demo.py --sigma0 1000 --run 0            # a false lock (wrong terrain)
    python scripts/live_demo.py --region north_central_plains --sigma0 2000
    python scripts/live_demo.py --region synthetic --particles 20000 --fps 20
    python scripts/live_demo.py --run 7                          # another random flight
    python scripts/live_demo.py --save demo.mp4                  # write a video instead of a window

Needs a desktop session (a window opens); uses the same simulation, filters and random
streams as the experiments, so run N here is the same flight as run N of the experiments.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from tan.config import REPO_ROOT, deep_update, dotted_to_nested, load_config, load_yaml
from tan.experiment import build_dem, make_filter
from tan.plotting import COLORS, INK, LABELS
from tan.simulate import filter_rng, initial_belief, run_streams, simulate


def build(args):
    cfg = load_config()
    common = load_yaml(REPO_ROOT / "configs" / "experiments" / "_common.yaml")
    cfg = deep_update(cfg, dotted_to_nested(common.get("set", {})))
    over = {
        "region": args.region,
        "init": {"position_sigma": float(args.sigma0)},
        "particle_filter": {"n_particles": int(args.particles), "snapshot_every": 0},
    }
    if args.region == "synthetic":
        over["synthetic"] = {"size_m": 40000.0, "n_hills": 240}
    cfg = deep_update(cfg, over)
    try:
        dem = build_dem(cfg)
    except FileNotFoundError as exc:
        sys.exit(f"{exc}\nTip: use --region synthetic (no download needed).")
    seed = int(cfg["seed"])
    sim = simulate(cfg["simulation"], cfg["init"], dem, run_streams(seed, args.run))
    names = ["dead_reckoning", "ekf", "particle_filter"]
    filters = {n: make_filter(n, cfg, dem, filter_rng(seed, args.run, n)) for n in names}
    mean, cov = initial_belief(sim)
    for f in filters.values():
        f.initialize(mean, cov)
    return cfg, dem, sim, filters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--region",
        default="hill_country",
        help="hill_country | kandy_rolling | north_central_plains | synthetic",
    )
    ap.add_argument("--sigma0", type=float, default=500.0, help="initial position std (m)")
    ap.add_argument("--particles", type=int, default=5000)
    ap.add_argument(
        "--run", type=int, default=0, help="which random flight (Monte Carlo run index)"
    )
    ap.add_argument("--fps", type=float, default=15.0, help="animation frames per second")
    ap.add_argument("--save", help="write the animation to this .mp4/.gif instead of showing it")
    args = ap.parse_args()

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LightSource

    cfg, dem, sim, filters = build(args)
    pf = filters["particle_filter"]
    K = sim.n_steps
    truth = np.vstack([sim.p0_true, sim.truth]) / 1e3
    est = {n: [np.array(sim.p0_guess) / 1e3] for n in filters}
    err = {n: [] for n in filters}
    rng = np.random.default_rng(0)

    shade = LightSource(azdeg=315, altdeg=45).hillshade(
        dem.elevation, vert_exag=2.0, dx=dem.res, dy=dem.res
    )
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.canvas.manager.set_window_title("Terrain-aided navigation — live particle filter")
    ax.imshow(
        shade,
        cmap="gray",
        extent=np.array(dem.extent) / 1e3,
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    cloud = ax.scatter([], [], s=2, color=COLORS["particle_filter"], alpha=0.35, lw=0)
    (truth_line,) = ax.plot([], [], color=INK, lw=2.0, ls="--", label="Ground truth")
    (truth_pt,) = ax.plot([], [], marker="+", ms=14, mew=2.5, color=INK, ls="none")
    lines, dots = {}, {}
    for n in filters:
        (lines[n],) = ax.plot([], [], color=COLORS[n], lw=1.8, label=LABELS[n])
        (dots[n],) = ax.plot([], [], marker="o", ms=6, color=COLORS[n], ls="none")
    ax.plot(*truth[0], marker="x", ms=8, mew=2, color=INK, ls="none")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    info = ax.text(
        0.99,
        0.01,
        "",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "lw": 0},
    )

    err_lines = {}
    for n in filters:
        (err_lines[n],) = ax2.plot([], [], color=COLORS[n], lw=1.8, label=LABELS[n])
    ax2.axhline(50, color="#8a8984", lw=0.8, ls=":", label="50 m (converged)")
    ax2.set_xlim(0, K * sim.dt)
    ax2.set_ylim(1, 2e4)
    ax2.set_yscale("log")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (m)")
    ax2.grid(True, color="#e4e3df")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{args.region.replace('_', ' ')} — initial position error σ₀ = {args.sigma0:g} m, "
        f"{pf.n:,} particles, flight {args.run}".replace(",", " ")
    )
    fig.tight_layout()
    half = [max(2.5 * args.sigma0 / 1e3, 1.0)]

    def update(k):
        if k > 0:
            for n, f in filters.items():
                x, _ = f.step(sim.v_ins[k - 1], sim.dt, float(sim.z[k - 1]))
                est[n].append(x[:2] / 1e3)
                err[n].append(float(np.linalg.norm(x[:2] - sim.truth[k - 1])))
        pts = pf.particles[:, :2] / 1e3
        sub = pts[rng.choice(len(pts), size=min(4000, len(pts)), replace=False)]
        cloud.set_offsets(sub)
        truth_line.set_data(truth[: k + 1, 0], truth[: k + 1, 1])
        truth_pt.set_data([truth[k, 0]], [truth[k, 1]])
        for n in filters:
            e = np.array(est[n])
            lines[n].set_data(e[:, 0], e[:, 1])
            dots[n].set_data([e[-1, 0]], [e[-1, 1]])
            if err[n]:
                err_lines[n].set_data(sim.t[: len(err[n])], err[n])
        # follow the action: centre between truth and cloud, zoom to the cloud's spread
        centre = 0.5 * (truth[k] + np.median(pts, axis=0))
        spread = float(np.max(np.abs(np.percentile(pts, [2, 98], axis=0) - centre)))
        target = max(0.4, 1.3 * spread, 1.2 * float(np.max(np.abs(truth[k] - centre))))
        half[0] = 0.85 * half[0] + 0.15 * target
        ax.set_xlim(centre[0] - half[0], centre[0] + half[0])
        ax.set_ylim(centre[1] - half[0], centre[1] + half[0])
        ess = pf.history["ess"][-1] / pf.n if pf.history["ess"] else 1.0
        e = {n: (err[n][-1] if err[n] else float("nan")) for n in filters}
        info.set_text(
            f"t = {k * sim.dt:5.0f} s   N_eff/N = {ess:4.2f}\n"
            f"error  PF {e['particle_filter']:7.1f} m\n"
            f"      EKF {e['ekf']:7.1f} m\n"
            f"       DR {e['dead_reckoning']:7.1f} m"
        )
        return []

    anim = FuncAnimation(
        fig, update, frames=K + 1, interval=1000.0 / args.fps, blit=False, repeat=False
    )
    if args.save:
        anim.save(args.save, fps=args.fps, dpi=90)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
