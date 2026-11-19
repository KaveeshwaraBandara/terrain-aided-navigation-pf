"""Correctness evidence for the report: filters against the exact Kalman filter.

On a tilted plane h(p) = a·px + b·py + c the measurement is linear and bilinear
interpolation is exact, so the EKF *is* the Kalman filter and its posterior is exact.
A correct particle filter (and RBPF) must agree with it up to Monte Carlo error. Also
checks that the PCRB recursion reproduces the Kalman covariance.

Writes report/generated/validation.tex and report/tables/validation.{tex,csv}.
"""

from __future__ import annotations

import numpy as np

from tan.config import deep_update, load_config
from tan.dem import DEM
from tan.experiment import run_filter
from tan.filters import ExtendedKalmanFilter, ParticleFilter, RaoBlackwellizedPF
from tan.pcrb import pcrb, slope_outer
from tan.reporting import fmt, macro_name, write_macros, write_table
from tan.simulate import filter_rng, initial_belief, run_streams, simulate

SEEDS = 5


def plane() -> DEM:
    n, res = 668, 30.0
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    return DEM(0.05 * xx + 0.03 * yy + 200.0, x0=0.0, y0=float(xs[-1]), res=res)


def main() -> None:
    cfg = load_config()
    dem = plane()
    kw = {k: v for k, v in cfg["particle_filter"].items() if k not in ("n_particles",)}
    kw["snapshot_every"] = 0
    rows, csv_rows, macros = [], [], {}
    for name, cls, n in (("PF", ParticleFilter, 20000), ("RBPF", RaoBlackwellizedPF, 5000)):
        zpos, zbias, ratio = [], [], []
        for seed in range(SEEDS):
            sim = simulate(
                cfg["simulation"],
                deep_update(cfg["init"], {"position_sigma": 100.0}),
                dem,
                run_streams(7, seed),
            )
            kf = run_filter(ExtendedKalmanFilter(dem, **cfg["ekf"]), sim)
            f = run_filter(cls(dem, n_particles=n, rng=filter_rng(7, seed, name), **kw), sim)
            sd = np.sqrt(np.einsum("kii->ki", kf.cov))
            z = np.abs(f.est - kf.est) / sd
            zpos.append(z[:, :2].max())
            zbias.append(z[:, 2:].max())
            tr = np.einsum("kii->ki", f.cov)[:, :2].sum(1) / np.einsum("kii->ki", kf.cov)[
                :, :2
            ].sum(1)
            ratio.append((tr.min(), tr.max()))
        r = np.array(ratio)
        rows.append(
            [
                name,
                f"{n:,}".replace(",", "\\,"),
                f"{fmt(min(zpos), 2)}--{fmt(max(zpos), 2)}",
                f"{fmt(min(zbias), 2)}--{fmt(max(zbias), 2)}",
                f"{fmt(r[:, 0].min(), 2)}--{fmt(r[:, 1].max(), 2)}",
            ]
        )
        csv_rows.append(
            [name, n, min(zpos), max(zpos), min(zbias), max(zbias), r[:, 0].min(), r[:, 1].max()]
        )
        macros[macro_name("val", name, "pos max")] = fmt(max(zpos), 2)
        macros[macro_name("val", name, "bias max")] = fmt(max(zbias), 2)
        macros[macro_name("val", name, "trace lo")] = fmt(r[:, 0].min(), 2)
        macros[macro_name("val", name, "trace hi")] = fmt(r[:, 1].max(), 2)
    write_table(
        "validation",
        [
            "Filter",
            "$N$",
            "max $|\\hat x - \\hat x_{KF}|/\\sigma_{KF}$, position",
            "same, bias",
            "trace ratio $P/P_{KF}$ (position)",
        ],
        rows,
        align="lrrrr",
        csv_header=[
            "filter",
            "n",
            "pos_min",
            "pos_max",
            "bias_min",
            "bias_max",
            "trace_ratio_min",
            "trace_ratio_max",
        ],
        csv_rows=csv_rows,
    )

    # PCRB recursion = Kalman covariance on the plane (largest relative difference)
    sim = simulate(
        cfg["simulation"],
        deep_update(cfg["init"], {"position_sigma": 200.0}),
        dem,
        run_streams(7, 0),
    )
    ekf = ExtendedKalmanFilter(dem, **cfg["ekf"])
    run = run_filter(ekf, sim)
    _, P0 = initial_belief(sim)
    b = pcrb(slope_outer(dem, sim.truth), P0, ekf.Q, ekf.R, sim.dt)
    rel = float(np.max(np.abs(b - run.cov) / np.maximum(np.abs(run.cov), 1e-12)))
    macros["ValPcrbRelDiff"] = f"{rel:.1e}"
    macros["ValSeeds"] = str(SEEDS)
    write_macros("validation", macros)
    print(rows, f"PCRB vs KF max relative difference {rel:.2e}")


if __name__ == "__main__":
    main()
