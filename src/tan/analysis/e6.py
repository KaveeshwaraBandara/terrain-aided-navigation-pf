"""E6 — false-lock detection and recovery (divergence monitor).

Study ``dev`` (development seeds) chooses the monitor parameters with a rule fixed in
advance (report/findings.md): the highest pooled convergence rate at σ₀ = 2000 m among
the settings whose false-alarm rate at σ₀ = 250 m is at most 5 % (a false alarm = a run
with at least one alarm); ties go to the longer window. Study ``test`` (main seeds)
evaluates that choice.
"""

from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from ..metrics import mcnemar
from ..plotting import TEXT_WIDTH_IN, report_style, save_report, style_of
from ..runner import results_dir
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    axes_style,
    ci,
    fmt,
    load_summary,
    macro_name,
    pct,
    rate,
    regions_in,
    yerr,
)

SIGMA = "init.position_sigma"
WIN = "particle_filter.monitor.window"
PFA = "particle_filter.monitor.false_alarm_prob"
MAX_FALSE_ALARM = 0.05


def choose(dev) -> tuple[dict, list]:
    mon = dev[dev["filter"] == "pf_monitor"].copy()
    mon["alarmed"] = mon["n_alarms"] > 0
    rows = []
    for (w, p), g in mon.groupby([WIN, PFA]):
        hi = g[g[SIGMA] == g[SIGMA].max()]
        lo = g[g[SIGMA] == g[SIGMA].min()]
        rows.append(
            {
                "window": int(w),
                "false_alarm_prob": float(p),
                "conv_high": rate(hi, "converged"),
                "fl_high": rate(hi, "false_lock"),
                "false_alarm_low": rate(lo, "alarmed"),
            }
        )
    ok = [r for r in rows if r["false_alarm_low"]["rate"] <= MAX_FALSE_ALARM]
    pool = ok or rows
    best = max(pool, key=lambda r: (r["conv_high"]["rate"], r["window"]))
    return best, rows


def make(exp: str = "e6", quick: bool = False) -> None:
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    macros: dict[str, str] = {}

    dev = df[df["study"] == "dev"]
    if not dev.empty:
        best, rows = choose(dev)
        base = dev[(dev["filter"] == "pf_5000")]
        bh = rate(
            base[base[SIGMA] == base[SIGMA].max()].drop_duplicates(["region", "run"]), "converged"
        )
        trows = [
            [
                str(r["window"]),
                fmt(r["false_alarm_prob"], 3),
                f"{pct(r['conv_high']['rate'])} [{pct(r['conv_high']['lo'])}, "
                f"{pct(r['conv_high']['hi'])}]",
                pct(r["fl_high"]["rate"]),
                f"{pct(r['false_alarm_low']['rate'])} [{pct(r['false_alarm_low']['lo'])}, "
                f"{pct(r['false_alarm_low']['hi'])}]",
                "chosen" if r is best else "",
            ]
            for r in rows
        ]
        trows.append(
            [
                "no monitor",
                "--",
                f"{pct(bh['rate'])} [{pct(bh['lo'])}, {pct(bh['hi'])}]",
                "",
                "",
                "",
            ]
        )
        out.table(
            "dev",
            [
                "Window $L$",
                "$p_\\mathrm{FA}$",
                "Conv.\\ at 2 km (\\%)",
                "FL at 2 km (\\%)",
                "False alarms at 250 m (\\%)",
                "",
            ],
            trows,
            align="rrrrrl",
        )
        choice = {"window": best["window"], "false_alarm_prob": best["false_alarm_prob"]}
        path = results_dir(exp, quick) / "dev_choice.json"
        path.write_text(json.dumps(choice, indent=2))
        print(f"[{exp}] dev choice: {choice} → {path}")
        macros["ESixWindow"] = str(best["window"])
        macros["ESixPfa"] = fmt(best["false_alarm_prob"], 3)
        macros["ESixDevRuns"] = str(int(dev["run"].nunique()))

    test = df[df["study"] == "test"]
    if not test.empty:
        regions = regions_in(test)
        sigmas = sorted(test[SIGMA].unique())
        filters = ["pf_5000", "pf_monitor"]
        stats, rows, csv_rows = {}, [], []
        for (region, s, f), g in test.groupby(["region", SIGMA, "filter"]):
            stats[(region, s, f)] = {
                "conv": rate(g, "converged"),
                "fl": rate(g, "false_lock"),
                "alarm": rate(g.assign(a=g["n_alarms"] > 0), "a"),
            }
        with report_style():
            fig = Figure(figsize=(TEXT_WIDTH_IN, 2.7), layout="constrained")
            axes = fig.subplots(1, len(sigmas), sharey=True, squeeze=False)[0]
            for ax, s in zip(axes, sigmas, strict=True):
                axes_style(ax)
                for k, f in enumerate(filters):
                    st = style_of(f)
                    x = np.arange(len(regions)) + (k - 0.5) * 0.22
                    for key, mk, fill in (("conv", "o", True), ("fl", "x", False)):
                        v = [stats[(r, s, f)][key] for r in regions]
                        y = 100 * np.array([a["rate"] for a in v])
                        lo = 100 * np.array([a["lo"] for a in v])
                        hi = 100 * np.array([a["hi"] for a in v])
                        ax.errorbar(
                            x,
                            y,
                            yerr=yerr(y, lo, hi),
                            fmt=mk,
                            ms=4,
                            capsize=2,
                            color=st["color"],
                            mfc=st["color"] if fill else "none",
                            label=f"{st['label']}: "
                            + ("converged" if key == "conv" else "false lock"),
                        )
                ax.set_xticks(range(len(regions)))
                ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
                ax.set_title(f"σ₀ = {s:g} m")
                ax.set_ylim(-3, 103)
            axes[0].set_ylabel("Runs (%)")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=2)
            save_report(fig, out.figure("test"))
        for region in regions:
            for s in sigmas:
                g = test[(test["region"] == region) & (test[SIGMA] == s)]
                piv = g.pivot(index="run", columns="filter", values="converged")
                t = mcnemar(piv["pf_5000"].to_numpy(), piv["pf_monitor"].to_numpy())
                a, b = stats[(region, s, "pf_5000")], stats[(region, s, "pf_monitor")]
                rows.append(
                    [
                        REGION_LABELS.get(region, region),
                        f"{s:g}",
                        ci(a["conv"]),
                        ci(b["conv"]),
                        pct(a["fl"]["rate"]),
                        pct(b["fl"]["rate"]),
                        pct(b["alarm"]["rate"]),
                        f"{t['a_only']}/{t['b_only']}",
                        "$<$0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3),
                    ]
                )
                csv_rows.append(
                    [
                        region,
                        s,
                        a["conv"]["rate"],
                        b["conv"]["rate"],
                        a["fl"]["rate"],
                        b["fl"]["rate"],
                        b["alarm"]["rate"],
                        t["a_only"],
                        t["b_only"],
                        t["p_value"],
                    ]
                )
                base = ("e6", REGION_SHORT.get(region, region), f"s{s:g}")
                macros[macro_name(*base, "conv base")] = pct(a["conv"]["rate"])
                macros[macro_name(*base, "conv mon")] = pct(b["conv"]["rate"])
                macros[macro_name(*base, "fl base")] = pct(a["fl"]["rate"])
                macros[macro_name(*base, "fl mon")] = pct(b["fl"]["rate"])
                macros[macro_name(*base, "alarm")] = pct(b["alarm"]["rate"])
                macros[macro_name(*base, "p")] = (
                    "<0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3)
                )
        out.table(
            "test",
            [
                "Region",
                "$\\sigma_0$ (m)",
                "Conv.\\ PF (\\%)",
                "Conv.\\ PF+mon.\\ (\\%)",
                "FL PF",
                "FL PF+mon.",
                "Runs with alarm (\\%)",
                "Discordant",
                "$p$",
            ],
            rows,
            align="llrrrrrrr",
            csv_header=[
                "region",
                "sigma0_m",
                "conv_pf",
                "conv_monitor",
                "fl_pf",
                "fl_monitor",
                "alarm_rate_monitor",
                "pf_only",
                "monitor_only",
                "p_value",
            ],
            csv_rows=csv_rows,
        )
        macros["ESixTestRuns"] = str(int(test["run"].nunique()))
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
