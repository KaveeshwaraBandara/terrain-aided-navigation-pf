"""E7 — Rao-Blackwellised PF vs plain PF at equal N and at equal runtime."""

from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from ..config import REPO_ROOT
from ..metrics import mcnemar, paired_bootstrap_rmse_diff
from ..plotting import TEXT_WIDTH_IN, report_style, save_report, style_of
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    axes_style,
    ci,
    fmt,
    load_summary,
    macro_name,
    plain_log_ticks,
    pooled_rmse,
    rate,
    regions_in,
)

FAMILIES = {"pf": "pf_5000", "rbpf": "rbpf"}  # style keys


def runtimes() -> dict[str, float]:
    path = REPO_ROOT / "results" / "benchmark" / "runtime.json"
    if not path.exists():
        return {}
    return {k: v["median_ms"] for k, v in json.loads(path.read_text())["filters"].items()}


def make(exp: str = "e7", quick: bool = False) -> None:
    rng = np.random.default_rng(0)
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    df["kind"] = df["filter"].str.split("_").str[0]
    df["N"] = df["filter"].str.split("_").str[1].astype(int)
    regions = regions_in(df)
    Ns = sorted(df["N"].unique())
    rt = runtimes()
    stats = {}
    for (region, kind, n), g in df.groupby(["region", "kind", "N"]):
        conv = g[g["converged"]]
        stats[(region, kind, n)] = {
            "conv": rate(g, "converged"),
            "track": pooled_rmse(conv["tracking_mse_m2"].to_numpy(), rng),
            "bias": pooled_rmse((conv["tracking_bias_rmse_mps"] ** 2).to_numpy(), rng),
            "bias_all": pooled_rmse((g["tracking_bias_rmse_mps"] ** 2).to_numpy(), rng),
        }

    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 5.2), layout="constrained")
        axes = fig.subplots(3, len(regions), sharex=True, sharey="row", squeeze=False)
        for j, region in enumerate(regions):
            for i, key in enumerate(["conv", "track", "bias"]):
                ax = axes[i, j]
                axes_style(ax)
                for kind, sk in FAMILIES.items():
                    st = style_of(sk)
                    label = "PF" if kind == "pf" else "RBPF"
                    v = [stats.get((region, kind, n), {}).get(key) for n in Ns]
                    if any(x is None for x in v):
                        continue
                    name = "rate" if key == "conv" else "rmse"
                    sc = 100.0 if key == "conv" else 1.0
                    y = sc * np.array([x[name] for x in v])
                    lo = sc * np.array([x["lo"] for x in v])
                    hi = sc * np.array([x["hi"] for x in v])
                    ax.fill_between(Ns, lo, hi, color=st["color"], alpha=0.15, lw=0)
                    ax.plot(Ns, y, color=st["color"], ls=st["ls"], marker="o", ms=3, label=label)
                ax.set_xscale("log")
                plain_log_ticks(ax, "x", ticks=(500, 1000, 2000, 5000))
                if i == 0:
                    ax.set_title(REGION_LABELS.get(region, region))
                    ax.set_ylim(-3, 103)
                if i == 2:
                    ax.set_xlabel("Number of particles N")
            axes[0, 0].set_ylabel("Converged runs (%)")
            axes[1, 0].set_ylabel("Tracking RMSE (m)")
            axes[2, 0].set_ylabel("Bias RMSE (m/s)")
        from ..config import load_config

        prior = float(np.sqrt(2.0) * load_config()["simulation"]["ins"]["bias_sigma0"])
        for ax in axes[2]:
            ax.axhline(prior, color="#8a8984", lw=0.8, ls=":", label="prior only")
        axes[0, -1].legend(loc="lower right")
        save_report(fig, out.figure("equal_n"))

    if rt:
        with report_style():
            fig = Figure(figsize=(TEXT_WIDTH_IN, 2.5), layout="constrained")
            axes = fig.subplots(1, len(regions), sharey=True, squeeze=False)[0]
            for ax, region in zip(axes, regions, strict=True):
                axes_style(ax)
                for kind, sk in FAMILIES.items():
                    st = style_of(sk)
                    xs = [rt.get(f"{kind}_{n}") for n in Ns]
                    if any(x is None for x in xs):
                        continue
                    v = [stats[(region, kind, n)]["conv"] for n in Ns]
                    y = 100 * np.array([a["rate"] for a in v])
                    lo = 100 * np.array([a["lo"] for a in v])
                    hi = 100 * np.array([a["hi"] for a in v])
                    ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
                    ax.plot(
                        xs,
                        y,
                        color=st["color"],
                        ls=st["ls"],
                        marker="o",
                        ms=3,
                        label="PF" if kind == "pf" else "RBPF",
                    )
                ax.set_xscale("log")
                plain_log_ticks(ax, "x", ticks=(0.2, 0.3, 0.5, 0.8))
                ax.set_ylim(-3, 103)
                ax.set_title(REGION_LABELS.get(region, region))
                ax.set_xlabel("Runtime per step (ms)")
            axes[0].set_ylabel("Converged runs (%)")
            axes[-1].legend(loc="lower right")
            save_report(fig, out.figure("equal_runtime"))

    rows, csv_rows, macros = [], [], {}
    for region in regions:
        for n in Ns:
            g = df[(df["region"] == region) & (df["N"] == n)]
            piv = g.pivot(index="run", columns="kind", values="converged")
            t = mcnemar(piv["pf"].to_numpy(), piv["rbpf"].to_numpy())
            a, b = stats[(region, "pf", n)], stats[(region, "rbpf", n)]
            rows.append(
                [
                    REGION_LABELS.get(region, region),
                    f"{n:,}".replace(",", "\\,"),
                    ci(a["conv"]),
                    ci(b["conv"]),
                    fmt(a["track"]["rmse"]),
                    fmt(b["track"]["rmse"]),
                    fmt(a["bias"]["rmse"], 3),
                    fmt(b["bias"]["rmse"], 3),
                    "$<$0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3),
                ]
            )
            csv_rows.append(
                [
                    region,
                    n,
                    a["conv"]["rate"],
                    b["conv"]["rate"],
                    a["track"]["rmse"],
                    b["track"]["rmse"],
                    a["bias"]["rmse"],
                    b["bias"]["rmse"],
                    t["a_only"],
                    t["b_only"],
                    t["p_value"],
                ]
            )
            # tracking RMSE difference PF - RBPF on runs where both converged (paired bootstrap)
            both = piv["pf"].to_numpy() & piv["rbpf"].to_numpy()
            mse = g.pivot(index="run", columns="kind", values="tracking_mse_m2").to_numpy()[both]
            if len(mse) > 2:
                d = paired_bootstrap_rmse_diff(mse[:, 0], mse[:, 1], rng)
                key = ("e7", REGION_SHORT.get(region, region), f"n{n}", "track diff")
                macros[macro_name(*key)] = fmt(d["diff"], 1)
                macros[macro_name(*key, "lo")] = fmt(d["lo"], 1)
                macros[macro_name(*key, "hi")] = fmt(d["hi"], 1)
            for kind, s in (("pf", a), ("rbpf", b)):
                base = ("e7", REGION_SHORT.get(region, region), kind, f"n{n}")
                macros[macro_name(*base, "bias all")] = fmt(s["bias_all"]["rmse"], 2)
                macros[macro_name(*base, "conv")] = fmt(100 * s["conv"]["rate"], 0)
                macros[macro_name(*base, "track")] = fmt(s["track"]["rmse"])
                macros[macro_name(*base, "bias")] = fmt(s["bias"]["rmse"], 3)
    out.table(
        "summary",
        [
            "Region",
            "$N$",
            "PF conv.\\ (\\%)",
            "RBPF conv.\\ (\\%)",
            "PF track (m)",
            "RBPF track (m)",
            "PF bias (m/s)",
            "RBPF bias (m/s)",
            "$p$",
        ],
        rows,
        align="lrrrrrrrr",
        csv_header=[
            "region",
            "n",
            "pf_conv",
            "rbpf_conv",
            "pf_track_m",
            "rbpf_track_m",
            "pf_bias_mps",
            "rbpf_bias_mps",
            "pf_only",
            "rbpf_only",
            "p_value",
        ],
        csv_rows=csv_rows,
    )
    macros["ESevenRuns"] = str(int(df.groupby(["config", "filter"]).size().min()))
    from ..config import load_config

    b0 = float(load_config()["simulation"]["ins"]["bias_sigma0"])
    macros["ESevenPriorBias"] = fmt(np.sqrt(2.0) * b0, 2)  # 2-D RMS of the prior
    hilly = [r[10] for r in csv_rows if r[0] != "north_central_plains"]
    macros["ESevenHillyMinP"] = fmt(min(hilly), 2)
    for r in csv_rows:
        macros[macro_name("e7", REGION_SHORT.get(r[0], r[0]), f"n{r[1]}", "p")] = (
            "<0.001" if r[10] < 1e-3 else fmt(r[10], 3)
        )
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
