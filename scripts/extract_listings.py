"""Report material taken straight from the code and configs, so it never drifts.

- report/generated/code_<name>.py: code excerpts (docstrings removed to keep them short)
- report/tables/parameters.{tex,csv}: the simulation and filter parameters actually used
  (configs/default.yaml + configs/experiments/_common.yaml)
- report/generated/implementation.tex: number of unit tests, lines of code
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap

from tan.config import REPO_ROOT, deep_update, dotted_to_nested, load_config, load_yaml
from tan.filters import resampling
from tan.filters.ekf import ExtendedKalmanFilter
from tan.filters.particle_filter import ParticleFilter
from tan.filters.rbpf import RaoBlackwellizedPF
from tan.pcrb import pcrb
from tan.reporting import GENERATED, fmt, write_macros, write_table

EXCERPTS = {
    "pf_predict": ParticleFilter.predict,
    "pf_update": ParticleFilter.update,
    "pf_step": ParticleFilter.step,
    "systematic": resampling.systematic,
    "ekf_update": ExtendedKalmanFilter.update,
    "rbpf_predict": RaoBlackwellizedPF.predict,
    "pcrb": pcrb,
}


def strip_docstring(src: str) -> str:
    tree = ast.parse(src)
    fn = tree.body[0]
    lines = src.splitlines()
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        doc = fn.body[0]
        del lines[doc.lineno - 1 : doc.end_lineno]
    return "\n".join(lines) + "\n"


PARAMS = [
    ("Simulation", "simulation.dt", "Time step $\\Delta t$", "s"),
    ("Simulation", "simulation.duration", "Flight duration", "s"),
    ("Simulation", "simulation.speed", "Ground speed", "m/s"),
    ("Simulation", "simulation.weave_amplitude_deg", "Heading weave amplitude", "deg"),
    ("Simulation", "simulation.weave_period_s", "Heading weave period", "s"),
    ("Simulation", "simulation.clearance_m", "Clearance above highest terrain in corridor", "m"),
    ("Simulation", "simulation.corridor_halfwidth_m", "Corridor half-width", "m"),
    ("Simulation", "simulation.flight.margin_sigma0", "Flight margin from map edge / 3", "m"),
    ("Sensors", "simulation.ins.bias_sigma0", "INS initial bias std (per axis)", "m/s"),
    (
        "Sensors",
        "simulation.ins.bias_rw_sigma",
        "INS bias random walk",
        "m/s/$\\sqrt{\\mathrm{s}}$",
    ),
    ("Sensors", "simulation.ins.noise_sigma", "INS white velocity noise", "m/s"),
    ("Sensors", "simulation.baro_sigma", "Barometric altimeter noise", "m"),
    ("Sensors", "simulation.radar_sigma", "Radar altimeter noise", "m"),
    ("Sensors", "simulation.outlier_sigma", "Radar outlier std (E4)", "m"),
    ("Prior", "init.position_sigma", "Initial position std $\\sigma_0$ (default)", "m"),
    ("Particle filter", "particle_filter.n_particles", "Particles $N$ (default)", ""),
    ("Particle filter", "particle_filter.resampling", "Resampling scheme", ""),
    (
        "Particle filter",
        "particle_filter.ess_threshold",
        "Resample when $N_\\mathrm{eff} < \\tau N$, $\\tau$",
        "",
    ),
    (
        "Particle filter",
        "particle_filter.process_sigma_pos",
        "Process noise, position $\\sigma_p$",
        "m/step",
    ),
    (
        "Particle filter",
        "particle_filter.process_sigma_bias",
        "Process noise, bias $\\sigma_b$",
        "m/s/step",
    ),
    ("Particle filter", "particle_filter.meas_sigma", "Measurement std $\\sigma_z$", "m"),
    ("Particle filter", "particle_filter.roughening", "Roughening $K$ (default)", ""),
    ("Evaluation", "experiment.convergence_threshold_m", "Convergence threshold", "m"),
    ("Evaluation", "experiment.tracking_window_s", "Tracking window", "s"),
]


def resolved_defaults() -> dict:
    cfg = load_config()
    common = load_yaml(REPO_ROOT / "configs" / "experiments" / "_common.yaml")
    return deep_update(cfg, dotted_to_nested(common.get("set", {})))


def get(cfg: dict, key: str):
    node = cfg
    for part in key.split("."):
        node = node[part]
    return node


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, obj in EXCERPTS.items():
        src = textwrap.dedent(inspect.getsource(obj))
        out = GENERATED / f"code_{name}.py"
        out.write_text(strip_docstring(src))
        print(out)

    cfg = resolved_defaults()
    rows, csv_rows = [], []
    for group, key, label, unit in PARAMS:
        v = get(cfg, key)
        shown = f"{v:g}" if isinstance(v, float) else str(v)
        rows.append([group, label, shown.replace("_", "\\_"), unit])
        csv_rows.append([group, key, v, unit])
    write_table(
        "parameters",
        ["Group", "Parameter", "Value", "Unit"],
        rows,
        align="lllr",
        csv_header=["group", "key", "value", "unit"],
        csv_rows=csv_rows,
    )

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    n_tests = sum(1 for line in collected.splitlines() if "::" in line)
    loc = sum(len(f.read_text().splitlines()) for f in (REPO_ROOT / "src" / "tan").rglob("*.py"))
    write_macros("implementation", {"NumTests": str(n_tests), "LinesOfCode": fmt(loc, 0)})
    print(f"{n_tests} tests, {loc} lines in src/tan")


if __name__ == "__main__":
    main()
