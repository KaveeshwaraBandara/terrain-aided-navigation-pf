"""E1 — filter comparison vs initial uncertainty (figures, tables, macros)."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ..metrics import mcnemar
from ..plotting import GRID, TEXT_WIDTH_IN, report_style, save_report, style_of
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    fmt,
    load_summary,
    macro_name,
    pct,
    pooled_rmse,
    rate,
    regions_in,
    series,
)

SIGMA = "init.position_sigma"
FILTERS = ["ekf", "pf_5000", "pf_20000"]
ALL = ["dead_reckoning", *FILTERS]


def _axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def make(exp: str = "e1", quick: bool = False) -> None:
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    rng = np.random.default_rng(0)
    regions = regions_in(df)
    sigmas = sorted(df[SIGMA].unique())
    filters = [f for f in FILTERS if f in set(df["filter"])]
    M = int(df.groupby(["config", "filter"]).size().min())

    # ------------------------------------------------------------------ rates
    rates = {}
    for (region, s, f), g in df.groupby(["region", SIGMA, "filter"]):
        rates[(region, s, f)] = {
            "conv": rate(g, "converged"),
            "fl": rate(g, "false_lock"),
            "track": pooled_rmse(g.loc[g["converged"], "tracking_mse_m2"].to_numpy(), rng),
            "rmse": pooled_rmse(g["mse_m2"].to_numpy(), rng),
        }

    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 4.2), layout="constrained")
        axes = fig.subplots(2, len(regions), sharex=True, sharey="row", squeeze=False)
        for j, region in enumerate(regions):
            for i, key in enumerate(["conv", "fl"]):
                ax = axes[i, j]
                _axes_style(ax)
                for f in filters:
                    st = style_of(f)
                    r = [rates[(region, s, f)][key] for s in sigmas]
                    y = 100 * np.array([x["rate"] for x in r])
                    lo = 100 * np.array([x["lo"] for x in r])
                    hi = 100 * np.array([x["hi"] for x in r])
                    ax.fill_between(sigmas, lo, hi, color=st["color"], alpha=0.15, lw=0)
                    ax.plot(
                        sigmas,
                        y,
                        color=st["color"],
                        ls=st["ls"],
                        marker="o",
                        ms=3,
                        label=st["label"],
                    )
                ax.set_xscale("log")
                ax.set_xticks(sigmas)
                ax.set_xticklabels([f"{s:g}" for s in sigmas], rotation=45)
                ax.minorticks_off()
                ax.set_ylim(-3, 103)
                if i == 0:
                    ax.set_title(REGION_LABELS.get(region, region))
                if i == 1:
                    ax.set_xlabel("Initial position std σ₀ (m)")
            axes[0, 0].set_ylabel("Converged runs (%)")
            axes[1, 0].set_ylabel("False-lock runs (%)")
        axes[0, -1].legend(loc="lower left")
        save_report(fig, out.figure("rates"))

    # ------------------------------------------------------------------ error vs time
    show = [s for s in (500, 2000) if s in sigmas] or sigmas[:2]
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 2.1 * len(show) + 0.4), layout="constrained")
        axes = fig.subplots(len(show), len(regions), sharex=True, sharey=True, squeeze=False)
        for i, s in enumerate(show):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                _axes_style(ax)
                cfg_id = df.loc[(df["region"] == region) & (df[SIGMA] == s), "config"].iloc[0]
                keys = [f"{f}__err" for f in ALL if f in set(df["filter"])]
                data = series(exp, quick, "main", cfg_id, keys + ["t"])
                t = data["t"][0]
                for key in keys:
                    f = key.split("__")[0]
                    st = style_of(f)
                    err = data[key]
                    q1, med, q3 = np.percentile(err, [25, 50, 75], axis=0)
                    ax.fill_between(t, q1, q3, color=st["color"], alpha=0.15, lw=0)
                    ax.plot(t, med, color=st["color"], ls=st["ls"], label=st["label"])
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(REGION_LABELS.get(region, region))
                if j == 0:
                    ax.set_ylabel(f"σ₀ = {s:g} m\nposition error (m)")
                if i == len(show) - 1:
                    ax.set_xlabel("Time (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        save_report(fig, out.figure("error_vs_time"))

    # ------------------------------------------------------------------ tables + tests
    rows, csv_rows, macros = [], [], {}
    trows = []
    test_rows, test_csv = [], []
    for region in regions:
        for s in sigmas:
            row = [REGION_LABELS.get(region, region), f"{s:g}"]
            trow = [REGION_LABELS.get(region, region), f"{s:g}"]
            for f in filters:
                r = rates[(region, s, f)]
                c, fl, tr = r["conv"], r["fl"], r["track"]
                row.append(f"{pct(c['rate'])} [{pct(c['lo'])}, {pct(c['hi'])}]")
                row.append(pct(fl["rate"]))
                trow.append(f"{fmt(tr['rmse'])} [{fmt(tr['lo'])}, {fmt(tr['hi'])}]")
                csv_rows.append(
                    [
                        region,
                        s,
                        f,
                        c["k"],
                        c["n"],
                        c["rate"],
                        c["lo"],
                        c["hi"],
                        fl["k"],
                        fl["rate"],
                        tr["rmse"],
                        tr["lo"],
                        tr["hi"],
                        tr["n"],
                        r["rmse"]["rmse"],
                    ]
                )
                base = ("e1", REGION_SHORT.get(region, region), f"s{s:g}", f)
                macros[macro_name(*base, "conv")] = pct(c["rate"])
                macros[macro_name(*base, "conv lo")] = pct(c["lo"])
                macros[macro_name(*base, "conv hi")] = pct(c["hi"])
                macros[macro_name(*base, "fl")] = pct(fl["rate"])
                macros[macro_name(*base, "track")] = fmt(tr["rmse"])
                macros[macro_name(*base, "rmse")] = fmt(r["rmse"]["rmse"])
            rows.append(row)
            trows.append(trow)
            # paired tests on the same runs (common random numbers)
            g = df[(df["region"] == region) & (df[SIGMA] == s)]
            piv = g.pivot(index="run", columns="filter", values="converged")
            for a, b in (("ekf", "pf_5000"), ("pf_5000", "pf_20000")):
                if a in piv and b in piv:
                    t = mcnemar(piv[a].to_numpy(), piv[b].to_numpy())
                    test_rows.append(
                        [
                            REGION_LABELS.get(region, region),
                            f"{s:g}",
                            f"{style_of(a)['label']} vs {style_of(b)['label']}",
                            str(t["a_only"]),
                            str(t["b_only"]),
                            "$<$0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3),
                        ]
                    )
                    test_csv.append([region, s, a, b, t["a_only"], t["b_only"], t["p_value"]])
                    macros[
                        macro_name(
                            "e1", REGION_SHORT.get(region, region), f"s{s:g}", a, "vs", b, "p"
                        )
                    ] = "<0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3)

    head = ["Region", "$\\sigma_0$ (m)"]
    for f in filters:
        lab = style_of(f)["label"]
        head += [f"{lab} conv.\\ (\\%) [95\\% CI]", f"{lab} FL (\\%)"]
    out.table(
        "convergence",
        head,
        rows,
        align="ll" + "rr" * len(filters),
        csv_header=[
            "region",
            "sigma0_m",
            "filter",
            "conv_k",
            "n",
            "conv_rate",
            "conv_lo",
            "conv_hi",
            "false_lock_k",
            "false_lock_rate",
            "tracking_rmse_m",
            "tracking_lo",
            "tracking_hi",
            "tracking_runs",
            "rmse_m",
        ],
        csv_rows=csv_rows,
    )
    out.table(
        "tracking",
        ["Region", "$\\sigma_0$ (m)"] + [f"{style_of(f)['label']} (m)" for f in filters],
        trows,
        align="ll" + "r" * len(filters),
    )
    out.table(
        "tests",
        ["Region", "$\\sigma_0$ (m)", "Comparison", "Only A", "Only B", "$p$ (McNemar)"],
        test_rows,
        align="lllrrr",
        csv_header=["region", "sigma0_m", "a", "b", "a_only", "b_only", "p_value"],
        csv_rows=test_csv,
    )
    macros["EOneRuns"] = str(M)

    # radar-altimeter ranges implied by the corridor altitude (A10): the flights are identical
    # for every sigma0 (common random numbers), so one configuration per region suffices
    rr_rows, rr_csv = [], []
    for region in regions:
        g = df[(df["region"] == region) & (df[SIGMA] == sigmas[0]) & (df["filter"] == ALL[0])]
        r = g["radar_range_max_m"].to_numpy()
        q50, q95, qmax = np.percentile(r, [50, 95, 100])
        rr_rows.append([REGION_LABELS.get(region, region), fmt(q50, 0), fmt(q95, 0), fmt(qmax, 0)])
        rr_csv.append([region, q50, q95, qmax, len(r)])
        short = REGION_SHORT.get(region, region)
        macros[macro_name("e1 radar", short, "median")] = fmt(q50, 0)
        macros[macro_name("e1 radar", short, "max")] = fmt(qmax, 0)
    out.table(
        "radar_range",
        ["Region", "Median (m)", "95th percentile (m)", "Largest (m)"],
        rr_rows,
        align="lrrr",
        csv_header=["region", "median_max_range_m", "p95_max_range_m", "largest_m", "runs"],
        csv_rows=rr_csv,
    )
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
