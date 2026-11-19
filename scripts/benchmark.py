"""Runtime per filter step — the only source of runtime numbers in the report (A3).

Single process, one maths thread, pinned to one performance core, nothing else of ours
running. Each estimator runs the same 300-step flight ``--reps`` times after a warm-up;
the per-step time is the run's wall time / K, and the median over repetitions is
reported (with the interquartile range).

Outputs: report/tables/runtime.{csv,tex}, report/generated/runtime.tex,
results/benchmark/runtime.json (raw repetition times + machine state).

Usage: python scripts/benchmark.py [--reps 7] [--cpu 4] [--region hill_country]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from tan.config import REPO_ROOT, deep_update, load_config  # noqa: E402
from tan.experiment import build_dem, make_filter, run_filter  # noqa: E402
from tan.filters.resampling import RESAMPLERS  # noqa: E402
from tan.reporting import fmt, macro_name, write_macros, write_table  # noqa: E402
from tan.simulate import filter_rng, run_streams, simulate  # noqa: E402

PF_N = [500, 1000, 2000, 5000, 10000, 20000, 50000]
RBPF_N = [500, 1000, 2000, 5000, 10000, 20000]


def variants():
    yield "dead_reckoning", "dead_reckoning", {}, None
    yield "ekf", "ekf", {}, None
    for n in PF_N:
        yield f"pf_{n}", "particle_filter", {"particle_filter": {"n_particles": n}}, n
    for n in RBPF_N:
        yield f"rbpf_{n}", "rbpf", {"particle_filter": {"n_particles": n}}, n
    yield (
        "pf_student_t_5000",
        "particle_filter",
        {"particle_filter": {"n_particles": 5000, "likelihood": "student_t"}},
        5000,
    )
    yield (
        "pf_monitor_5000",
        "particle_filter",
        {"particle_filter": {"n_particles": 5000, "monitor": {"enabled": True}}},
        5000,
    )


def machine_state() -> dict:
    def read(p: str) -> str:
        try:
            return Path(p).read_text().strip()
        except OSError:
            return "unknown"

    return {
        "cpu_model": next(
            (
                ln.split(":", 1)[1].strip()
                for ln in read("/proc/cpuinfo").splitlines()
                if ln.startswith("model name")
            ),
            "unknown",
        ),
        "mains_power": read("/sys/class/power_supply/ADP0/online") == "1",
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "loadavg": read("/proc/loadavg"),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cpu", type=int, default=4, help="logical CPU to pin to (a P-core)")
    ap.add_argument("--region", default="hill_country")
    args = ap.parse_args()

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    cfg = load_config()
    cfg = deep_update(
        cfg,
        {
            "region": args.region,
            "particle_filter": {"snapshot_every": 0},
            "simulation": {"flight": {"mode": "random"}, "altitude_reference": "corridor"},
            "init": {"position_sigma": 1000.0},
        },
    )
    try:
        dem = build_dem(cfg)
    except FileNotFoundError:
        cfg = deep_update(cfg, {"region": "synthetic", "synthetic": {"size_m": 40000.0}})
        dem = build_dem(cfg)
    sim = simulate(cfg["simulation"], cfg["init"], dem, run_streams(int(cfg["seed"]), 0))
    K = sim.n_steps
    state_before = machine_state()

    results = {}
    for key, family, over, n in variants():
        vcfg = deep_update(cfg, over)
        times = []
        for rep in range(args.reps + 1):  # first repetition is a warm-up
            f = make_filter(family, vcfg, dem, filter_rng(int(cfg["seed"]), rep, family))
            t0 = time.perf_counter()
            run_filter(f, sim)
            if rep:
                times.append((time.perf_counter() - t0) / K)
        ms = 1e3 * np.array(times)
        q1, med, q3 = np.percentile(ms, [25, 50, 75])
        results[key] = {
            "family": family,
            "n": n,
            "median_ms": med,
            "q1_ms": q1,
            "q3_ms": q3,
            "reps_ms": ms.tolist(),
        }
        print(f"{key:20s} {med:8.3f} ms/step  (IQR {q1:.3f}–{q3:.3f})", flush=True)

    # resampling schemes alone: time per call
    rng = np.random.default_rng(0)
    resample = {}
    for n in (1000, 5000, 20000):
        w = rng.random(n) ** 4
        w /= w.sum()
        for name, fn in RESAMPLERS.items():
            reps = []
            for _ in range(200):
                t0 = time.perf_counter()
                fn(w, rng)
                reps.append(time.perf_counter() - t0)
            resample[f"{name}_{n}"] = 1e3 * float(np.median(reps))
    for k, v in resample.items():
        print(f"resample {k:18s} {v:7.4f} ms/call")

    out = REPO_ROOT / "results" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    raw = {
        "region": cfg["region"],
        "steps": K,
        "reps": args.reps,
        "cpu": args.cpu,
        "machine_before": state_before,
        "machine_after": machine_state(),
        "filters": results,
        "resampling_ms_per_call": resample,
    }
    (out / "runtime.json").write_text(json.dumps(raw, indent=2))

    label = {
        "dead_reckoning": "Dead reckoning",
        "ekf": "EKF",
        "particle_filter": "PF",
        "rbpf": "RBPF",
    }
    rows, csv_rows = [], []
    for key, r in results.items():
        name = label[r["family"]]
        if key.startswith("pf_student"):
            name = "PF, Student-t"
        elif key.startswith("pf_monitor"):
            name = "PF + monitor"
        rows.append(
            [
                name,
                "--" if r["n"] is None else f"{r['n']:,}".replace(",", "\\,"),
                fmt(r["median_ms"], 3),
                f"{fmt(r['q1_ms'], 3)}--{fmt(r['q3_ms'], 3)}",
            ]
        )
        csv_rows.append([key, r["family"], r["n"], r["median_ms"], r["q1_ms"], r["q3_ms"]])
    write_table(
        "runtime",
        ["Estimator", "$N$", "Median (ms/step)", "IQR (ms/step)"],
        rows,
        align="lrrr",
        csv_header=["key", "family", "n", "median_ms", "q1_ms", "q3_ms"],
        csv_rows=csv_rows,
    )
    macros = {macro_name("rt", k): fmt(r["median_ms"], 3) for k, r in results.items()}
    macros["RtRegion"] = cfg["region"].replace("_", "\\_")
    macros["RtReps"] = str(args.reps)
    for k, v in resample.items():
        macros[macro_name("rt resample", k)] = fmt(v, 3)
    write_macros("runtime", macros, header=f"region={cfg['region']}")


if __name__ == "__main__":
    main()
