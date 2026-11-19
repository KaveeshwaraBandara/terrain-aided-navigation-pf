"""E2 — terrain informativeness: per-flight performance vs the terrain under the flight."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from ..plotting import TEXT_WIDTH_IN, report_style, save_report, style_of
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    axes_style,
    fmt,
    load_summary,
    macro_name,
    pct,
    rate,
    regions_in,
    series,
)

FILTERS = ["ekf", "pf_5000"]
MARKERS = {"hill_country": "o", "kandy_rolling": "s", "north_central_plains": "^", "synthetic": "D"}
HOLD = 100.0


def window_rms(exp: str, quick: bool, df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    """RMS position error over the final 100 s of every run (converged or not)."""
    parts = []
    for (study, config), _ in df.groupby(["study", "config"]):
        data = series(exp, quick, study, config, [f"{f}__err" for f in filters] + ["t"])
        t = data["t"][0]
        win = t > t[-1] - HOLD + 1e-9
        for f in filters:
            err = data[f"{f}__err"].astype(float)
            parts.append(
                pd.DataFrame(
                    {
                        "study": study,
                        "config": config,
                        "filter": f,
                        "run": data["_run"],
                        "window_rms_m": np.sqrt(np.mean(err[:, win] ** 2, 1)),
                    }
                )
            )
    return df.merge(pd.concat(parts), on=["study", "config", "filter", "run"], how="left")


def spearman_ci(x, y, rng, n_boot=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    return float(rho), float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))


def make(exp: str = "e2", quick: bool = False) -> None:
    rng = np.random.default_rng(0)
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    filters = [f for f in FILTERS if f in set(df["filter"])]
    regions = regions_in(df)
    df = window_rms(exp, quick, df, filters)
    macros: dict[str, str] = {"ETwoRuns": str(int(df.groupby(["region", "filter"]).size().min()))}

    # ------------------------------------------------------------ scatter vs roughness
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 2.9), layout="constrained")
        axes = fig.subplots(1, len(filters), sharex=True, sharey=True, squeeze=False)[0]
        for ax, f in zip(axes, filters, strict=True):
            axes_style(ax)
            st = style_of(f)
            g = df[df["filter"] == f]
            for region in regions:
                r = g[g["region"] == region]
                conv = r["converged"].to_numpy()
                x = r["terrain_roughness_1km_m"].to_numpy()
                y = r["window_rms_m"].to_numpy()
                ax.scatter(
                    x[conv],
                    y[conv],
                    s=9,
                    marker=MARKERS.get(region, "o"),
                    color=st["color"],
                    alpha=0.6,
                    lw=0,
                    label=f"{REGION_LABELS.get(region, region)}, converged",
                )
                ax.scatter(
                    x[~conv],
                    y[~conv],
                    s=9,
                    marker=MARKERS.get(region, "o"),
                    facecolor="none",
                    edgecolor=st["color"],
                    alpha=0.7,
                    lw=0.6,
                    label=f"{REGION_LABELS.get(region, region)}, not converged",
                )
            rho, lo, hi = spearman_ci(g["terrain_roughness_1km_m"], g["window_rms_m"], rng)
            ax.text(
                0.98,
                0.97,
                f"Spearman ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            macros[macro_name("e2 rho", f)] = fmt(rho, 2)
            macros[macro_name("e2 rho", f, "lo")] = fmt(lo, 2)
            macros[macro_name("e2 rho", f, "hi")] = fmt(hi, 2)
            ax.axhline(50.0, color="#8a8984", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Terrain roughness under the flight (m)")
            ax.set_title(st["label"])
        axes[0].set_ylabel("RMS error, final 100 s (m)")
        axes[-1].legend(loc="lower left", fontsize=6.5, ncol=1, markerscale=1.4)
        save_report(fig, out.figure("scatter"))

    # ------------------------------------------------------------ binned by information
    info = "terrain_fisher_total_per_m2"
    edges = np.quantile(df[info], np.linspace(0, 1, 9))
    df["info_bin"] = np.clip(np.searchsorted(edges, df[info], side="right") - 1, 0, 7)
    with report_style():
        fig = Figure(figsize=(0.62 * TEXT_WIDTH_IN, 2.6), layout="constrained")
        ax = fig.subplots()
        axes_style(ax)
        for f in filters:
            st = style_of(f)
            g = df[df["filter"] == f]
            xs, y, lo, hi = [], [], [], []
            for _, b in g.groupby("info_bin"):
                r = rate(b, "converged")
                xs.append(np.median(b[info]))
                y.append(100 * r["rate"])
                lo.append(100 * r["lo"])
                hi.append(100 * r["hi"])
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
            ax.plot(xs, y, color=st["color"], ls=st["ls"], marker="o", ms=3, label=st["label"])
        ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Terrain information along the flight, Σ|∇h|²/σ_z² (m⁻²)")
        ax.set_ylabel("Converged runs (%)")
        ax.legend(loc="lower right")
        save_report(fig, out.figure("binned"))

    # ------------------------------------------------------------ per-region table
    rows, csv_rows = [], []
    for region in regions:
        for f in filters:
            g = df[(df["region"] == region) & (df["filter"] == f)]
            c = rate(g, "converged")
            conv = g[g["converged"]]
            along = np.sqrt(conv["tracking_mse_along_m2"].mean()) if len(conv) else np.nan
            cross = np.sqrt(conv["tracking_mse_cross_m2"].mean()) if len(conv) else np.nan
            ia = g["terrain_fisher_along_per_m2"].median()
            ic = g["terrain_fisher_cross_per_m2"].median()
            rho_r = stats.spearmanr(g["terrain_roughness_1km_m"], g["window_rms_m"]).statistic
            rows.append(
                [
                    REGION_LABELS.get(region, region),
                    style_of(f)["label"],
                    f"{pct(c['rate'])} [{pct(c['lo'])}, {pct(c['hi'])}]",
                    fmt(g["terrain_roughness_1km_m"].median()),
                    fmt(along),
                    fmt(cross),
                    fmt(ia / ic if ic else np.nan, 2),
                    fmt(rho_r, 2),
                ]
            )
            csv_rows.append(
                [
                    region,
                    f,
                    c["rate"],
                    c["lo"],
                    c["hi"],
                    g["terrain_roughness_1km_m"].median(),
                    along,
                    cross,
                    ia,
                    ic,
                    rho_r,
                ]
            )
            base = ("e2", REGION_SHORT.get(region, region), f)
            macros[macro_name(*base, "conv")] = pct(c["rate"])
            macros[macro_name(*base, "along")] = fmt(along)
            macros[macro_name(*base, "cross")] = fmt(cross)
            macros[macro_name(*base, "rough")] = fmt(g["terrain_roughness_1km_m"].median())
            macros[macro_name(*base, "rho")] = fmt(rho_r, 2)
    out.table(
        "regions",
        [
            "Region",
            "Filter",
            "Conv.\\ (\\%) [95\\% CI]",
            "Rough.\\ (m)",
            "Along RMS (m)",
            "Cross RMS (m)",
            "$I_\\parallel/I_\\perp$",
            "$\\rho_S$",
        ],
        rows,
        align="llrrrrrr",
        csv_header=[
            "region",
            "filter",
            "conv_rate",
            "conv_lo",
            "conv_hi",
            "median_roughness_m",
            "tracking_rms_along_m",
            "tracking_rms_cross_m",
            "median_info_along",
            "median_info_cross",
            "spearman_rho",
        ],
        csv_rows=csv_rows,
    )
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
