"""E6 — false-lock detection and recovery (divergence monitor).

Study ``dev`` (development seeds) chooses the monitor parameters with a rule fixed in
advance (report/findings.md): the highest pooled convergence rate at σ₀ = 2000 m among
the settings whose false-alarm rate at σ₀ = 250 m is at most 5 % (a false alarm = a run
with at least one alarm); ties go to the longer window. If no setting meets the constraint
(which happened), the setting with the lowest false-alarm rate is taken (extension recorded in
report/findings.md before the test runs). Study ``test`` (main seeds) evaluates that choice,
together with roughening (K = 0.05, from E3) and the combination.
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
VARIANTS = ["pf_5000", "pf_monitor", "pf_rough", "pf_monitor_rough"]
MARKERS = {"pf_5000": "o", "pf_monitor": "D", "pf_rough": "s", "pf_monitor_rough": "^"}


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
    if ok:  # the pre-registered rule
        best = max(ok, key=lambda r: (r["conv_high"]["rate"], r["window"]))
    else:
        # The pre-registered rule did not cover "no setting meets the constraint". Extension
        # fixed on the dev data before any test run (report/findings.md): keep the rule's
        # intent — control the alarm frequency — and take the lowest false-alarm rate; ties
        # go to the higher convergence rate.
        best = min(rows, key=lambda r: (r["false_alarm_low"]["rate"], -r["conv_high"]["rate"]))
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
        macros["ESixDevMinFa"] = pct(min(r["false_alarm_low"]["rate"] for r in rows))
        macros["ESixDevBestFa"] = pct(best["false_alarm_low"]["rate"])
        macros["ESixDevBestConv"] = pct(best["conv_high"]["rate"])
        macros["ESixPfa"] = fmt(best["false_alarm_prob"], 3)
        macros["ESixDevRuns"] = str(int(dev["run"].nunique()))

    test = df[df["study"] == "test"]
    if not test.empty:
        regions = regions_in(test)
        sigmas = sorted(test[SIGMA].unique())
        variants = [v for v in VARIANTS if v in set(test["filter"])]
        stats = {}
        for (region, s0, f), g in test.groupby(["region", SIGMA, "filter"]):
            stats[(region, s0, f)] = {
                "conv": rate(g, "converged"),
                "fl": rate(g, "false_lock"),
                "alarm": rate(g.assign(a=g["n_alarms"] > 0), "a"),
            }
        with report_style():
            fig = Figure(figsize=(TEXT_WIDTH_IN, 4.4), layout="constrained")
            axes = fig.subplots(2, len(sigmas), sharex=True, sharey=True, squeeze=False)
            for j, s0 in enumerate(sigmas):
                for i, key in enumerate(("conv", "fl")):
                    ax = axes[i, j]
                    axes_style(ax)
                    for k, f in enumerate(variants):
                        st = style_of(f)
                        x = np.arange(len(regions)) + (k - (len(variants) - 1) / 2) * 0.17
                        v = [stats[(r, s0, f)][key] for r in regions]
                        y = 100 * np.array([a["rate"] for a in v])
                        lo = 100 * np.array([a["lo"] for a in v])
                        hi = 100 * np.array([a["hi"] for a in v])
                        ax.errorbar(
                            x,
                            y,
                            yerr=yerr(y, lo, hi),
                            fmt=MARKERS[f],
                            ms=4,
                            capsize=2,
                            color=st["color"],
                            label=st["label"],
                        )
                    ax.set_xticks(range(len(regions)))
                    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions])
                    ax.set_ylim(-3, 103)
                    if i == 0:
                        ax.set_title(f"σ₀ = {s0:g} m")
                axes[0, 0].set_ylabel("Converged runs (%)")
                axes[1, 0].set_ylabel("False-lock runs (%)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
            save_report(fig, out.figure("test"))

        rows, csv_rows, trows, tcsv = [], [], [], []
        for region in regions:
            for s0 in sigmas:
                g = test[(test["region"] == region) & (test[SIGMA] == s0)]
                piv = g.pivot(index="run", columns="filter", values="converged")
                row = [REGION_LABELS.get(region, region), f"{s0:g}"]
                for f in variants:
                    a = stats[(region, s0, f)]
                    row.append(f"{ci(a['conv'])} / {pct(a['fl']['rate'])}")
                    base = ("e6", REGION_SHORT.get(region, region), f"s{s0:g}", f)
                    macros[macro_name(*base, "conv")] = pct(a["conv"]["rate"])
                    macros[macro_name(*base, "fl")] = pct(a["fl"]["rate"])
                    if f in ("pf_monitor", "pf_monitor_rough"):
                        macros[macro_name(*base, "alarm")] = pct(a["alarm"]["rate"])
                    csv_rows.append(
                        [
                            region,
                            s0,
                            f,
                            a["conv"]["rate"],
                            a["conv"]["lo"],
                            a["conv"]["hi"],
                            a["fl"]["rate"],
                            a["alarm"]["rate"],
                        ]
                    )
                row.append(pct(stats[(region, s0, "pf_monitor")]["alarm"]["rate"]))
                rows.append(row)
                for f in [v for v in variants if v != "pf_5000"]:
                    t = mcnemar(piv["pf_5000"].to_numpy(), piv[f].to_numpy())
                    p = "$<$0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3)
                    trows.append(
                        [
                            REGION_LABELS.get(region, region),
                            f"{s0:g}",
                            f"PF vs {style_of(f)['label']}",
                            str(t["a_only"]),
                            str(t["b_only"]),
                            p,
                        ]
                    )
                    tcsv.append([region, s0, f, t["a_only"], t["b_only"], t["p_value"]])
                    macros[
                        macro_name("e6", REGION_SHORT.get(region, region), f"s{s0:g}", f, "p")
                    ] = "<0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3)
        out.table(
            "test",
            ["Region", "$\\sigma_0$ (m)"]
            + [f"{style_of(f)['label']}" for f in variants]
            + ["Alarm (\\%)"],
            rows,
            align="ll" + "r" * (len(variants) + 1),
            csv_header=[
                "region",
                "sigma0_m",
                "filter",
                "conv_rate",
                "conv_lo",
                "conv_hi",
                "false_lock_rate",
                "alarm_rate",
            ],
            csv_rows=csv_rows,
        )
        out.table(
            "test_mcnemar",
            ["Region", "$\\sigma_0$ (m)", "Comparison", "Only PF", "Only variant", "$p$"],
            trows,
            align="lllrrr",
            csv_header=["region", "sigma0_m", "variant", "pf_only", "variant_only", "p_value"],
            csv_rows=tcsv,
        )
        macros["ESixTestRuns"] = str(int(test["run"].nunique()))
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
