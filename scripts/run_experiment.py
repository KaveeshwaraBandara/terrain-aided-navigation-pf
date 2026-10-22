"""Run an experiment and write figures + metrics to results/<timestamp>_<region>/.

Usage:
    python scripts/run_experiment.py                         # configs/default.yaml
    python scripts/run_experiment.py --region hill_country --runs 50
    python scripts/run_experiment.py --set particle_filter.n_particles=1000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import yaml

from tan.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from tan.experiment import run_monte_carlo
from tan.metrics import summarize
from tan.plotting import plot_errors, plot_ess, plot_particle_snapshots, plot_trajectories


def parse_set(items: list[str]) -> dict:
    """Turn ['a.b=1', 'c=x'] into {'a': {'b': 1}, 'c': 'x'} (values parsed as YAML)."""
    out: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--region", help="override cfg.region")
    ap.add_argument("--runs", type=int, help="override experiment.monte_carlo_runs")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. ekf.meas_sigma=5")
    args = ap.parse_args()

    overrides = parse_set(args.set)
    if args.region:
        overrides["region"] = args.region
    if args.runs:
        overrides.setdefault("experiment", {})["monte_carlo_runs"] = args.runs
    cfg = load_config(args.config, overrides)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / cfg["experiment"]["output_dir"] / f"{stamp}_{cfg['region']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dem, results = run_monte_carlo(cfg)
    sim0, runs0 = results[0]
    names = cfg["experiment"]["filters"]

    # --- metrics
    summary, errors = {}, {}
    for name in names:
        if runs0[name].status != "ok":
            summary[name] = {"status": "todo", "message": runs0[name].message}
            print(f"  {name:16s} skipped — {runs0[name].message}")
            continue
        errors[name] = np.stack([runs[name].error for _, runs in results])
        summary[name] = summarize(
            errors[name], sim0.t, cfg["experiment"]["convergence_threshold_m"]
        )
        s = summary[name]
        print(f"  {name:16s} RMSE {s['rmse_m']:8.1f} m   final {s['final_error_mean_m']:8.1f} m")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # --- figures (first run for spatial plots, all runs for error curves)
    ok = {n: runs0[n].est for n in names if runs0[n].status == "ok"}
    plot_trajectories(dem, sim0, ok, out / "trajectories.png")
    plot_errors(sim0.t, errors, out / "position_error.png")
    pf = runs0.get("particle_filter")
    if pf is not None and pf.status == "ok":
        plot_ess(
            sim0.t,
            pf.extras["ess"],
            pf.extras["n_particles"],
            pf.extras["resampled"],
            out / "pf_ess.png",
        )
        if pf.extras["snapshots"]:
            plot_particle_snapshots(dem, sim0, pf.extras["snapshots"], out / "pf_snapshots.png")

    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
