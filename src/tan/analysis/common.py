"""Shared helpers for turning saved Monte Carlo results into report figures and tables.

Every function here reads only saved results (``results/<exp>/summary.csv`` and the
per-run ``.npz`` files); nothing re-runs a filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..metrics import bootstrap_ci, rmse_from_mse, wilson_interval
from ..reporting import fmt, macro_name, write_macros, write_table
from ..runner import load_runs, results_dir

REGION_ORDER = ["hill_country", "kandy_rolling", "north_central_plains", "synthetic"]
REGION_LABELS = {
    "hill_country": "Hill country",
    "kandy_rolling": "Kandy",
    "north_central_plains": "Dry-zone plains",
    "synthetic": "Synthetic",
}
REGION_SHORT = {
    "hill_country": "Hill",
    "kandy_rolling": "Kandy",
    "north_central_plains": "Plains",
    "synthetic": "Synth",
}


@dataclass
class Output:
    """Where an experiment's figures/tables/macros go. Quick runs never touch report/."""

    exp: str
    quick: bool

    @property
    def root(self) -> Path:
        if self.quick or "@" in self.exp:  # smoke / development runs never touch report/
            return results_dir(self.exp, self.quick) / "report"
        return REPO_ROOT / "report"

    def figure(self, name: str) -> Path:
        return self.root / "figures" / f"{self.exp}_{name}"

    def table(self, name: str, header, rows, **kw) -> None:
        from .. import reporting

        old = reporting.TABLES
        reporting.TABLES = self.root / "tables"
        try:
            write_table(f"{self.exp}_{name}", header, rows, **kw)
        finally:
            reporting.TABLES = old

    def macros(self, macros: dict[str, str]) -> None:
        from .. import reporting

        old = reporting.GENERATED
        reporting.GENERATED = self.root / "generated"
        try:
            write_macros(self.exp, macros)
        finally:
            reporting.GENERATED = old


def load_summary(exp: str, quick: bool = False) -> pd.DataFrame:
    path = results_dir(exp, quick) / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/sweep.py --experiment {exp}")
    df = pd.read_csv(path)
    for col in ("converged", "false_lock"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def regions_in(df: pd.DataFrame) -> list[str]:
    present = set(df["region"])
    return [r for r in REGION_ORDER if r in present] + sorted(present - set(REGION_ORDER))


def rate(df: pd.DataFrame, col: str) -> dict[str, float]:
    k, n = int(df[col].sum()), int(len(df))
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else np.nan, "lo": lo, "hi": hi}


def pooled_rmse(mse: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    mse = np.asarray(mse, dtype=float)
    mse = mse[np.isfinite(mse)]
    if mse.size == 0:
        return {"rmse": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    lo, hi = bootstrap_ci(mse, lambda x: float(np.sqrt(np.mean(x))), rng)
    return {"rmse": rmse_from_mse(mse), "lo": lo, "hi": hi, "n": int(mse.size)}


def series(exp: str, quick: bool, study: str, config: str, keys: list[str]) -> dict:
    return load_runs(exp, study, config, keys, quick)


def pct(x: float, digits: int = 0) -> str:
    return fmt(100.0 * x, digits)


__all__ = [
    "Output",
    "REGION_LABELS",
    "REGION_ORDER",
    "REGION_SHORT",
    "fmt",
    "load_summary",
    "macro_name",
    "pct",
    "pooled_rmse",
    "rate",
    "regions_in",
    "series",
]


# ---------------------------------------------------------------------- plotting helpers
def axes_style(ax) -> None:
    from ..plotting import GRID

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def metric_table(
    df: pd.DataFrame, by: list[str], rng: np.random.Generator
) -> dict[tuple, dict[str, dict[str, float]]]:
    """Convergence / false-lock rates (Wilson) and tracking + overall RMSE (bootstrap)
    for every group of ``by`` columns."""
    out = {}
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        out[key] = {
            "conv": rate(g, "converged"),
            "fl": rate(g, "false_lock"),
            "track": pooled_rmse(g.loc[g["converged"], "tracking_mse_m2"].to_numpy(), rng),
            "rmse": pooled_rmse(g["mse_m2"].to_numpy(), rng),
        }
    return out


def plot_vs(ax, xs, filters, stats, key: str, xkey=lambda x, f: (x, f), percent=None) -> None:
    """Draw one metric (``conv``/``fl`` as %, ``track``/``rmse`` in m) against ``xs`` for
    each filter, with its 95 % band. ``stats[xkey(x, f)][key]`` holds the numbers."""
    from ..plotting import style_of

    percent = key in ("conv", "fl") if percent is None else percent
    for f in filters:
        st = style_of(f)
        vals = [stats.get(xkey(x, f), {}).get(key) for x in xs]
        if not any(v is not None for v in vals):
            continue
        name = "rate" if percent else "rmse"
        y = np.array([np.nan if v is None else v[name] for v in vals], dtype=float)
        lo = np.array([np.nan if v is None else v["lo"] for v in vals], dtype=float)
        hi = np.array([np.nan if v is None else v["hi"] for v in vals], dtype=float)
        scale = 100.0 if percent else 1.0
        ax.fill_between(xs, scale * lo, scale * hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(xs, scale * y, color=st["color"], ls=st["ls"], marker="o", ms=3, label=st["label"])
    if percent:
        ax.set_ylim(-3, 103)


def ci(r: dict[str, float], percent: bool = True) -> str:
    """'97 [94, 99]' for a rate (``percent``) or '12.3 [10.1, 15.0]' for an RMSE."""
    if percent:
        return f"{pct(r['rate'])} [{pct(r['lo'])}, {pct(r['hi'])}]"
    return f"{fmt(r['rmse'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]"


def yerr(y, lo, hi) -> list[np.ndarray]:
    """Error-bar lengths for matplotlib, clipped at 0 (a bootstrap bound can sit a rounding
    error above the point estimate)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return [np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)]


def plain_log_ticks(ax, which: str, ticks=(1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500)) -> None:
    """Plain-number ticks on a log axis (instead of 2×10⁰ style labels)."""
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    t = [v for v in ticks if lo <= v <= hi]
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_ticks(t)
    axis.set_ticklabels([f"{v:g}" for v in t])
    axis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
