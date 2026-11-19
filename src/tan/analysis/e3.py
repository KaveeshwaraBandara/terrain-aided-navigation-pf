"""E3 — particle filter design: N, resampling scheme, ESS threshold, roughening."""

from __future__ import annotations

import json

import numpy as np
from matplotlib.figure import Figure

from ..config import REPO_ROOT
from ..filters.resampling import RESAMPLERS, effective_sample_size
from ..metrics import mcnemar
from ..plotting import MUTED, TEXT_WIDTH_IN, report_style, save_report, style_of
from .common import (
    Output,
    axes_style,
    fmt,
    load_summary,
    macro_name,
    metric_table,
    pct,
    yerr,
)

SCHEMES = ["multinomial", "systematic", "stratified", "residual"]
WORDS = {
    500: "five hundred",
    1000: "one k",
    2000: "two k",
    5000: "five k",
    10000: "ten k",
    20000: "twenty k",
    50000: "fifty k",
}


def offspring_variance(n: int, reps: int, rng: np.random.Generator) -> dict[str, float]:
    """Mean per-particle variance of the offspring count, Var(N_i), for each scheme.

    Weights: log-normal with log-std s = √ln(1/0.4), so N_eff ≈ 0.4 N — a typical cloud at
    the moment resampling is triggered (threshold 0.5). Same weight vector for all schemes;
    variance estimated over ``reps`` independent resamplings.
    """
    s = np.sqrt(np.log(1 / 0.4))
    w = np.exp(s * rng.standard_normal(n))
    w /= w.sum()
    out = {"ess_fraction": effective_sample_size(w) / n}
    for name in SCHEMES:
        counts = np.stack([np.bincount(RESAMPLERS[name](w, rng), minlength=n) for _ in range(reps)])
        out[name] = float(counts.var(axis=0, ddof=1).mean())
    nw = n * w
    out["multinomial_theory"] = float(np.mean(n * w * (1 - w)))
    frac = nw - np.floor(nw)
    r = frac.sum()
    out["residual_theory"] = float(np.mean(r * (frac / r) * (1 - frac / r)))
    out["lower_bound"] = float(np.mean(frac * (1 - frac)))  # floor/ceil only (systematic ideal)
    return out


def runtime_ms() -> dict[int, float]:
    path = REPO_ROOT / "results" / "benchmark" / "runtime.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())["filters"]
    return {v["n"]: v["median_ms"] for k, v in data.items() if k.startswith("pf_") and v["n"]}


def make(exp: str = "e3", quick: bool = False) -> None:
    rng = np.random.default_rng(0)
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    macros: dict[str, str] = {}
    rows, csv_rows = [], []

    def add_rows(study, label_of, stats, keys):
        for key in keys:
            st = stats[key]
            c, tr = st["conv"], st["track"]
            rows.append(
                [
                    study.replace("_", " "),
                    label_of(key),
                    f"{pct(c['rate'])} [{pct(c['lo'])}, {pct(c['hi'])}]",
                    pct(st["fl"]["rate"]),
                    f"{fmt(tr['rmse'])} [{fmt(tr['lo'])}, {fmt(tr['hi'])}]",
                    fmt(st["rmse"]["rmse"]),
                ]
            )
            csv_rows.append(
                [
                    study,
                    label_of(key),
                    c["rate"],
                    c["lo"],
                    c["hi"],
                    st["fl"]["rate"],
                    tr["rmse"],
                    tr["lo"],
                    tr["hi"],
                    st["rmse"]["rmse"],
                ]
            )

    # ------------------------------------------------------------ particle count
    ns = df[df["study"] == "n_sweep"].copy()
    ns["N"] = ns["filter"].str.split("_").str[1].astype(int)
    nstats = metric_table(ns, ["N"], rng)
    Ns = sorted(ns["N"].unique())
    rt = runtime_ms()
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 2.6), layout="constrained")
        a1, a2 = fig.subplots(1, 2)
        for ax in (a1, a2):
            axes_style(ax)
        st = style_of("pf_5000")
        conv = [nstats[(n,)]["conv"] for n in Ns]
        a1.fill_between(
            Ns,
            [100 * c["lo"] for c in conv],
            [100 * c["hi"] for c in conv],
            color=st["color"],
            alpha=0.15,
            lw=0,
        )
        a1.plot(Ns, [100 * c["rate"] for c in conv], color=st["color"], marker="o", ms=3)
        a1.set_xscale("log")
        a1.set_ylim(-3, 103)
        a1.set_xlabel("Number of particles N")
        a1.set_ylabel("Converged runs (%)")
        tr = [nstats[(n,)]["track"] for n in Ns]
        if rt and all(n in rt for n in Ns):
            xs = [rt[n] for n in Ns]
            a2.set_xlabel("Runtime per step (ms, single core)")
        else:
            xs = Ns
            a2.set_xlabel("Number of particles N (benchmark missing)")
        y = np.array([t["rmse"] for t in tr])
        a2.errorbar(
            xs,
            y,
            yerr=yerr(y, [t["lo"] for t in tr], [t["hi"] for t in tr]),
            color=st["color"],
            marker="o",
            ms=3,
            capsize=2,
            lw=1.2,
        )
        for x, yy, n in zip(xs, y, Ns, strict=True):
            a2.annotate(
                f"{n:,}".replace(",", " "),
                (x, yy),
                textcoords="offset points",
                xytext=(3, 4),
                fontsize=7,
                color=MUTED,
            )
        a2.set_xscale("log")
        ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10) if min(xs) / 1.5 <= t <= max(xs) * 1.5]
        a2.set_xticks(ticks)
        a2.set_xticklabels([f"{t:g}" for t in ticks])
        a2.minorticks_off()
        a2.set_ylabel("Tracking RMSE, converged runs (m)")
        save_report(fig, out.figure("particles"))
    add_rows("N sweep", lambda k: f"N = {k[0]:,}".replace(",", "\\,"), nstats, [(n,) for n in Ns])
    for n in Ns:
        macros[macro_name("e3 n", WORDS.get(n, n), "conv")] = pct(nstats[(n,)]["conv"]["rate"])
        macros[macro_name("e3 n", WORDS.get(n, n), "track")] = fmt(nstats[(n,)]["track"]["rmse"])
        if n in rt:
            macros[macro_name("e3 n", WORDS.get(n, n), "ms")] = fmt(rt[n], 2)

    # ------------------------------------------------------------ design choices
    rs = df[df["study"] == "resampling"]
    rstats = metric_table(rs, ["particle_filter.resampling", "filter"], rng)
    es = df[df["study"] == "ess_threshold"]
    estats = metric_table(es, ["particle_filter.ess_threshold"], rng)
    ro = df[df["study"] == "roughening"]
    ostats = metric_table(ro, ["particle_filter.roughening"], rng)
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 3.9), layout="constrained")
        axes = fig.subplots(2, 3, sharey="row")
        for ax in axes.ravel():
            axes_style(ax)
        rfilters = sorted(rs["filter"].unique(), key=lambda f: int(f.split("_")[1]))
        for j, f in enumerate(rfilters):
            color = style_of("pf_5000")["color"] if j else MUTED
            off = (j - 0.5) * 0.18
            for i, key in enumerate(["conv", "track"]):
                vals = [rstats[(s, f)][key] for s in SCHEMES]
                name = "rate" if key == "conv" else "rmse"
                sc = 100.0 if key == "conv" else 1.0
                y = np.array([v[name] for v in vals]) * sc
                lo = np.array([v["lo"] for v in vals]) * sc
                hi = np.array([v["hi"] for v in vals]) * sc
                axes[i, 0].errorbar(
                    np.arange(4) + off,
                    y,
                    yerr=yerr(y, lo, hi),
                    fmt="o",
                    ms=3,
                    capsize=2,
                    color=color,
                    label=f"N = {int(f.split('_')[1]):,}".replace(",", " "),
                )
        axes[1, 0].set_xticks(range(4))
        axes[1, 0].set_xticklabels(["multin.", "system.", "strat.", "resid."])
        axes[0, 0].set_xticks(range(4))
        axes[0, 0].set_xticklabels([])
        axes[0, 0].legend(loc="lower left")
        for col, (stats, xs, xlabel) in enumerate(
            [
                (
                    estats,
                    sorted(es["particle_filter.ess_threshold"].unique()),
                    "ESS threshold (× N)",
                ),
                (
                    ostats,
                    sorted(ro["particle_filter.roughening"].unique()),
                    "Roughening constant K",
                ),
            ],
            start=1,
        ):
            for i, key in enumerate(["conv", "track"]):
                vals = [stats[(x,)][key] for x in xs]
                name = "rate" if key == "conv" else "rmse"
                sc = 100.0 if key == "conv" else 1.0
                y = np.array([v[name] for v in vals]) * sc
                lo = np.array([v["lo"] for v in vals]) * sc
                hi = np.array([v["hi"] for v in vals]) * sc
                c = style_of("pf_5000")["color"]
                axes[i, col].fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
                axes[i, col].plot(xs, y, color=c, marker="o", ms=3)
            axes[1, col].set_xlabel(xlabel)
        axes[1, 0].set_xlabel("Resampling scheme")
        axes[0, 0].set_ylabel("Converged runs (%)")
        axes[1, 0].set_ylabel("Tracking RMSE (m)")
        axes[0, 0].set_ylim(-3, 103)
        save_report(fig, out.figure("design"))
    add_rows(
        "resampling",
        lambda k: f"{k[0]}, N = {int(k[1].split('_')[1]):,}".replace(",", "\\,"),
        rstats,
        [(s, f) for f in rfilters for s in SCHEMES],
    )
    add_rows(
        "ESS threshold",
        lambda k: f"{k[0]:g} N",
        estats,
        [(x,) for x in sorted(es["particle_filter.ess_threshold"].unique())],
    )
    add_rows(
        "roughening",
        lambda k: f"K = {k[0]:g}",
        ostats,
        [(x,) for x in sorted(ro["particle_filter.roughening"].unique())],
    )
    out.table(
        "summary",
        [
            "Study",
            "Setting",
            "Conv.\\ (\\%) [95\\% CI]",
            "FL (\\%)",
            "Tracking RMSE (m) [95\\% CI]",
            "RMSE (m)",
        ],
        rows,
        align="llrrrr",
        csv_header=[
            "study",
            "setting",
            "conv_rate",
            "conv_lo",
            "conv_hi",
            "false_lock_rate",
            "tracking_rmse_m",
            "tracking_lo",
            "tracking_hi",
            "rmse_m",
        ],
        csv_rows=csv_rows,
    )

    # mean number of resampling steps per flight, by ESS threshold
    for x, g in es.groupby("particle_filter.ess_threshold"):
        macros[macro_name("e3 ess", fmt(x, 1).replace(".", " point "), "resamples")] = fmt(
            g["n_resampled"].mean(), 0
        )
        for key in ("conv", "track"):
            v = estats[(x,)][key]
            macros[macro_name("e3 ess", fmt(x, 1).replace(".", " point "), key)] = (
                pct(v["rate"]) if key == "conv" else fmt(v["rmse"])
            )
    for x in sorted(ro["particle_filter.roughening"].unique()):
        for key in ("conv", "track"):
            v = ostats[(x,)][key]
            macros[macro_name("e3 rough", fmt(x, 2).replace(".", " point "), key)] = (
                pct(v["rate"]) if key == "conv" else fmt(v["rmse"])
            )
    for f in rfilters:
        for s in SCHEMES:
            for key in ("conv", "track"):
                v = rstats[(s, f)][key]
                macros[macro_name("e3 res", s, f, key)] = (
                    pct(v["rate"]) if key == "conv" else fmt(v["rmse"])
                )

    # paired tests against the defaults (systematic, 0.5 N, K = 0) on the same runs
    tests, tcsv = [], []

    def paired(study, col, base, others, label, filt=None):
        g = df[df["study"] == study]
        if filt:
            g = g[g["filter"] == filt]
        piv = g.pivot(index="run", columns=col, values="converged")
        for o in others:
            if base in piv and o in piv:
                t = mcnemar(piv[base].to_numpy(), piv[o].to_numpy())
                tests.append(
                    [
                        label,
                        f"{base} vs {o}",
                        str(t["a_only"]),
                        str(t["b_only"]),
                        "$<$0.001" if t["p_value"] < 1e-3 else fmt(t["p_value"], 3),
                    ]
                )
                tcsv.append([label, base, o, t["a_only"], t["b_only"], t["p_value"]])

    for f in rfilters:
        paired(
            "resampling",
            "particle_filter.resampling",
            "systematic",
            [s for s in SCHEMES if s != "systematic"],
            f"resampling, N = {int(f.split('_')[1]):,}".replace(",", "\\,"),
            f,
        )
    paired(
        "ess_threshold",
        "particle_filter.ess_threshold",
        0.5,
        [x for x in sorted(es["particle_filter.ess_threshold"].unique()) if x != 0.5],
        "ESS threshold",
    )
    paired(
        "roughening",
        "particle_filter.roughening",
        0.0,
        [x for x in sorted(ro["particle_filter.roughening"].unique()) if x != 0.0],
        "roughening",
    )
    out.table(
        "tests",
        ["Study", "Comparison", "Only A", "Only B", "$p$ (McNemar)"],
        tests,
        align="llrrr",
        csv_header=["study", "a", "b", "a_only", "b_only", "p_value"],
        csv_rows=tcsv,
    )

    # ------------------------------------------------------------ offspring variance
    vrows, vcsv = [], []
    for n in (1000, 5000):
        v = offspring_variance(n, 400 if not quick else 50, rng)
        vrows.append(
            [f"{n:,}".replace(",", "\\,")]
            + [fmt(v[s], 3) for s in SCHEMES]
            + [
                fmt(v["multinomial_theory"], 3),
                fmt(v["residual_theory"], 3),
                fmt(v["lower_bound"], 3),
            ]
        )
        vcsv.append(
            [n, v["ess_fraction"]]
            + [v[s] for s in SCHEMES]
            + [v["multinomial_theory"], v["residual_theory"], v["lower_bound"]]
        )
        for s in SCHEMES:
            macros[macro_name("e3 var", s, WORDS[n])] = fmt(v[s], 3)
        macros[macro_name("e3 var floor", WORDS[n])] = fmt(v["lower_bound"], 3)
    out.table(
        "offspring_variance",
        [
            "$N$",
            "Multinomial",
            "Systematic",
            "Stratified",
            "Residual",
            "Multin.\\ theory",
            "Resid.\\ theory",
            "floor/ceil bound",
        ],
        vrows,
        align="rrrrrrrr",
        csv_header=[
            "n",
            "ess_fraction",
            *SCHEMES,
            "multinomial_theory",
            "residual_theory",
            "floor_ceil_bound",
        ],
        csv_rows=vcsv,
    )
    macros["EThreeRuns"] = str(int(df.groupby(["config", "filter"]).size().min()))
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
