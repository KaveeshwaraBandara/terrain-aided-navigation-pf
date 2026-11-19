"""E5 — consistency (ANEES) and the posterior Cramér–Rao bound."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from ..experiment import build_dem
from ..metrics import anees_bounds, nees_in_bounds_fraction
from ..pcrb import mismatched_kf_error, pcrb, position_rmse_bound, slope_outer, true_noise
from ..plotting import INK, MUTED, TEXT_WIDTH_IN, report_style, save_report, style_of
from ..runner import results_dir
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    axes_style,
    fmt,
    load_summary,
    macro_name,
    plain_log_ticks,
    regions_in,
    series,
)

FILTERS = ["ekf", "pf_5000", "pf_20000"]
SIGMA = "init.position_sigma"
HOLD = 100.0


def sci(x: float) -> str:
    """Large ANEES values (false locks) in scientific notation for the table."""
    if not np.isfinite(x):
        return "--"
    if abs(x) < 1e4:
        return fmt(x, 1)
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10**e:.1f}\\times10^{{{e}}}$"


def resolved_cfg(exp: str, quick: bool, study: str, config: str) -> dict:
    path = results_dir(exp, quick) / "meta" / "resolved_configs.yaml"
    return yaml.safe_load(path.read_text())[f"{study}/{config}"]


def nees_pos(est: np.ndarray, cov_upper: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Position NEES for stacked runs: est (M,K,4), cov_upper (M,K,10), truth (M,K,2)."""
    e = est[..., :2].astype(float) - truth
    pxx, pxy, pyy = (cov_upper[..., i].astype(float) for i in (0, 1, 4))
    det = np.maximum(pxx * pyy - pxy * pxy, 1e-12)
    return (pyy * e[..., 0] ** 2 - 2 * pxy * e[..., 0] * e[..., 1] + pxx * e[..., 1] ** 2) / det


def bound_for(cfg: dict, truth: np.ndarray, dem) -> np.ndarray:
    """PCRB position-RMSE bound per step; ``truth`` (M,K,2): expectation over the M paths."""
    Q, R = true_noise(cfg["simulation"])
    s0 = float(cfg["init"]["position_sigma"])
    b0 = float(cfg["simulation"]["ins"]["bias_sigma0"])
    P0 = np.diag([s0**2, s0**2, b0**2, b0**2])
    G = slope_outer(dem, truth).mean(axis=0)
    return position_rmse_bound(pcrb(G, P0, Q, R, float(cfg["simulation"]["dt"])))


def tuned_floor(cfg: dict, path: np.ndarray, dem) -> np.ndarray:
    """Actual RMSE (per step) of a Kalman filter that knows the true slopes along ``path``
    (K, 2) but uses the particle filter's tuning (inflated sigma_p, sigma_z): what that tuning
    allows in the small-error regime (see tan.pcrb.mismatched_kf_error)."""
    Qt, Rt = true_noise(cfg["simulation"])
    pf = cfg["particle_filter"]
    sp, sb = float(pf["process_sigma_pos"]), float(pf["process_sigma_bias"])
    Qf = np.diag([sp**2, sp**2, sb**2, sb**2])
    Rf = float(pf["meas_sigma"]) ** 2
    s0 = float(cfg["init"]["position_sigma"])
    b0 = float(cfg["simulation"]["ins"]["bias_sigma0"])
    P0 = np.diag([s0**2, s0**2, b0**2, b0**2])
    gx, gy = dem.gradient(path[:, 0], path[:, 1])
    dt = float(cfg["simulation"]["dt"])
    A = mismatched_kf_error(np.column_stack([gx, gy]), P0, Qf, Rf, Qt, Rt, dt)
    return position_rmse_bound(A)


def make(exp: str = "e5", quick: bool = False) -> None:
    out = Output(exp, quick)
    macros: dict[str, str] = {}
    e1 = exp.replace("e5", "e1", 1)
    d1 = load_summary(e1, quick)
    regions = regions_in(d1)
    filters = [f for f in FILTERS if f in set(d1["filter"])]
    sigmas = sorted(d1[SIGMA].unique())
    show = [s for s in (500, 2000) if s in sigmas] or sigmas[:2]
    dems = {}

    # ------------------------------------------------------------ ensemble: E1 runs
    ens, cons = {}, {}
    trows, tcsv = [], []
    for region in regions:
        for s in sigmas:
            config = d1.loc[(d1["region"] == region) & (d1[SIGMA] == s), "config"].iloc[0]
            cfg = resolved_cfg(e1, quick, "main", config)
            if region not in dems:
                dems[region] = build_dem(cfg)
            keys = ["truth", "t"] + [f"{f}__{k}" for f in filters for k in ("est", "cov", "err")]
            data = series(e1, quick, "main", config, keys)
            t = data["t"][0]
            truth = data["truth"]
            ens[(region, s, "bound")] = bound_for(cfg, truth, dems[region])
            ens[(region, s, "t")] = t
            win = t > t[-1] - HOLD + 1e-9
            M = truth.shape[0]
            lo_b, hi_b = anees_bounds(M)
            summ = d1[(d1["config"] == config)]
            for f in filters:
                err = data[f"{f}__err"].astype(float)
                ens[(region, s, f)] = np.sqrt(np.mean(err**2, axis=0))
                nees = nees_pos(data[f"{f}__est"], data[f"{f}__cov"], truth)
                conv = summ[summ["filter"] == f].set_index("run").loc[data["_run"], "converged"]
                conv = conv.to_numpy()
                ens[(region, s, f, "conv")] = (
                    np.sqrt(np.mean(err[conv] ** 2, axis=0))
                    if conv.any()
                    else np.full(len(t), np.nan)
                )
                cons[(region, s, f, "all")] = nees.mean(axis=0)
                cons[(region, s, f, "conv")] = (
                    nees[conv].mean(axis=0) if conv.any() else np.full(len(t), np.nan)
                )
                inside = (
                    nees_in_bounds_fraction(nees[conv][:, win]).mean() if conv.any() else np.nan
                )
                an_all = float(np.mean(nees[:, win]))
                an_conv = float(np.mean(nees[conv][:, win])) if conv.any() else np.nan
                # all runs: a valid comparison (the bound applies to the ensemble MSE), so the
                # ratio must be >= 1 at every step; ratio_min checks that
                bnd = ens[(region, s, "bound")]
                ratio_all = float(np.mean(ens[(region, s, f)][win] / bnd[win]))
                ratio_min = float(np.min(ens[(region, s, f)] / bnd))
                # converged runs only: descriptive, *conditional* on error < 50 m, so not bounded
                # (truncation can push it below 1 where the bound approaches the threshold)
                ratio = float(np.mean(ens[(region, s, f, "conv")][win] / bnd[win]))
                lo_c, hi_c = anees_bounds(int(conv.sum())) if conv.any() else (np.nan, np.nan)
                trows.append(
                    [
                        REGION_LABELS.get(region, region),
                        f"{s:g}",
                        style_of(f)["label"],
                        sci(an_all),
                        fmt(an_conv, 2),
                        f"[{fmt(lo_c, 2)}, {fmt(hi_c, 2)}]",
                        f"{fmt(100 * inside, 0)}",
                        sci(ratio_all),
                        fmt(ratio, 2),
                    ]
                )
                tcsv.append(
                    [
                        region,
                        s,
                        f,
                        M,
                        int(conv.sum()),
                        an_all,
                        an_conv,
                        lo_c,
                        hi_c,
                        inside,
                        ratio_all,
                        ratio_min,
                        ratio,
                    ]
                )
                base = ("e5", REGION_SHORT.get(region, region), f"s{s:g}", f)
                macros[macro_name(*base, "anees conv")] = fmt(an_conv, 2)
                macros[macro_name(*base, "inside")] = fmt(100 * inside, 0)
                macros[macro_name(*base, "bound ratio")] = fmt(ratio, 2)
                macros[macro_name(*base, "bound ratio all")] = fmt(ratio_all, 1)
            macros[macro_name("e5", REGION_SHORT.get(region, region), f"s{s:g}", "bound")] = fmt(
                float(np.mean(ens[(region, s, "bound")][win])), 1
            )
            macros["EFiveAneesLo"], macros["EFiveAneesHi"] = fmt(lo_b, 2), fmt(hi_b, 2)
    out.table(
        "consistency",
        [
            "Region",
            "$\\sigma_0$ (m)",
            "Filter",
            "ANEES all",
            "ANEES conv.",
            "95\\% interval",
            "In $\\chi^2_2$ bounds (\\%)",
            "RMSE / PCRB",
            "RMSE$_{\\mathrm{conv}}$ / PCRB",
        ],
        trows,
        align="lllrrrrrr",
        csv_header=[
            "region",
            "sigma0_m",
            "filter",
            "runs",
            "converged_runs",
            "anees_all_tracking",
            "anees_converged_tracking",
            "anees_lo",
            "anees_hi",
            "fraction_inside_chi2",
            "rmse_all_over_pcrb_tracking",
            "rmse_all_over_pcrb_min_any_step",
            "rmse_converged_over_pcrb_tracking",
        ],
        csv_rows=tcsv,
    )

    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 2.1 * len(show) + 0.4), layout="constrained")
        axes = fig.subplots(len(show), len(regions), sharex=True, sharey=True, squeeze=False)
        for i, s in enumerate(show):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                axes_style(ax)
                t = ens[(region, s, "t")]
                for f in filters:
                    st = style_of(f)
                    ax.plot(
                        t,
                        ens[(region, s, f)],
                        color=st["color"],
                        ls=st["ls"],
                        label=f"{st['label']}, all runs",
                    )
                    ax.plot(
                        t,
                        ens[(region, s, f, "conv")],
                        color=st["color"],
                        ls=st["ls"],
                        lw=0.8,
                        alpha=0.7,
                        label=f"{st['label']}, converged runs",
                    )
                ax.plot(
                    t, ens[(region, s, "bound")], color=INK, ls=(0, (6, 2)), lw=1.2, label="PCRB"
                )
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(REGION_LABELS.get(region, region))
                if j == 0:
                    ax.set_ylabel(f"σ₀ = {s:g} m\nposition error (m)")
                if i == len(show) - 1:
                    ax.set_xlabel("Time (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
        save_report(fig, out.figure("bound"))

    s = show[0]
    with report_style():
        fig = Figure(figsize=(TEXT_WIDTH_IN, 4.0), layout="constrained")
        axes = fig.subplots(2, len(regions), sharex=True, sharey="row", squeeze=False)
        for j, region in enumerate(regions):
            t = ens[(region, s, "t")]
            for i, which in enumerate(["all", "conv"]):
                ax = axes[i, j]
                axes_style(ax)
                M = int(d1[(d1["region"] == region) & (d1[SIGMA] == s)]["run"].nunique())
                lo, hi = anees_bounds(M)
                ax.axhspan(lo, hi, color=MUTED, alpha=0.18, lw=0, label="95 % interval (M runs)")
                ax.axhline(2.0, color=MUTED, lw=0.8)
                for f in filters:
                    st = style_of(f)
                    ax.plot(
                        t,
                        cons[(region, s, f, which)],
                        color=st["color"],
                        ls=st["ls"],
                        label=st["label"],
                    )
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(REGION_LABELS.get(region, region))
                if j == 0:
                    ax.set_ylabel(
                        ("All runs" if which == "all" else "Converged runs") + "\nposition ANEES"
                    )
                if i == 1:
                    ax.set_xlabel("Time (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        axes[1, 0].set_ylim(0.3, 300)
        save_report(fig, out.figure("anees"))
    macros["EFiveAneesSigma"] = f"{s:g}"

    # ------------------------------------------------------------ per-flight plans (E5 runs)
    try:
        d5 = load_summary(exp, quick)
    except FileNotFoundError:
        d5 = None
    if d5 is not None:
        prows, pcsv = [], []
        pts = {f: [] for f in filters}
        for (region, config), g in d5.groupby(["region", "config"]):
            cfg = resolved_cfg(exp, quick, "main", config)
            if region not in dems:
                dems[region] = build_dem(cfg)
            data = series(
                exp, quick, "main", config, ["truth", "t"] + [f"{f}__err" for f in filters]
            )
            t = data["t"][0]
            win = t > t[-1] - HOLD + 1e-9
            truth = data["truth"]
            assert np.allclose(truth, truth[0]), "plan runs must share the flight"
            bound = bound_for(cfg, truth[:1], dems[region])
            b = float(np.sqrt(np.mean(bound[win] ** 2)))
            bf = tuned_floor(cfg, truth[0], dems[region])
            b_filt = float(np.sqrt(np.mean(bf[win] ** 2)))
            plan = int(g["simulation.flight.plan"].iloc[0])
            for f in filters:
                if f"{f}__err" not in data or data[f"{f}__err"].size == 0:
                    continue
                conv = g[g["filter"] == f].set_index("run").loc[data["_run"], "converged"]
                conv = conv.to_numpy()
                err = data[f"{f}__err"][conv][:, win].astype(float)
                r = float(np.sqrt(np.mean(err**2))) if conv.any() else np.nan
                pts[f].append((b, r, region))
                prows.append(
                    [
                        REGION_LABELS.get(region, region),
                        str(plan),
                        style_of(f)["label"],
                        str(int(conv.sum())),
                        fmt(b, 1),
                        fmt(b_filt, 1),
                        fmt(r, 1),
                        fmt(r / b, 2),
                        fmt(r / b_filt, 2),
                    ]
                )
                pcsv.append([region, plan, f, int(conv.sum()), b, b_filt, r, r / b, r / b_filt])
        out.table(
            "plans",
            [
                "Region",
                "Plan",
                "Filter",
                "Conv.\\ runs",
                "PCRB (m)",
                "Tuned KF (m)",
                "Tracking RMSE (m)",
                "RMSE/PCRB",
                "RMSE/tuned KF",
            ],
            prows,
            align="lllrrrrrr",
            csv_header=[
                "region",
                "plan",
                "filter",
                "converged_runs",
                "pcrb_tracking_m",
                "tuned_kf_rmse_tracking_m",
                "tracking_rmse_m",
                "ratio",
                "ratio_tuned_kf",
            ],
            csv_rows=pcsv,
        )
        with report_style():
            fig = Figure(figsize=(0.6 * TEXT_WIDTH_IN, 2.9), layout="constrained")
            ax = fig.subplots()
            axes_style(ax)
            allv = [v for f in filters for (b, r, _) in pts[f] for v in (b, r) if np.isfinite(v)]
            lim = (0.8 * min(allv), 1.25 * max(allv)) if allv else (1, 100)
            ax.plot(lim, lim, color=INK, lw=0.8, ls=(0, (6, 2)), label="RMSE = bound")
            markers = {"hill_country": "o", "kandy_rolling": "s", "north_central_plains": "^"}
            for f in filters:
                st = style_of(f)
                for b, r, region in pts[f]:
                    ax.scatter(b, r, s=14, color=st["color"], marker=markers.get(region, "D"), lw=0)
                ax.scatter([], [], s=14, color=st["color"], label=st["label"])
            for region, m in markers.items():
                if any(rr == region for f in filters for (_, _, rr) in pts[f]):
                    ax.scatter(
                        [], [], s=14, color=MUTED, marker=m, label=REGION_LABELS.get(region, region)
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            plain_log_ticks(ax, "x")
            plain_log_ticks(ax, "y")
            ax.set_xlabel("Per-flight PCRB, final 100 s (m)")
            ax.set_ylabel("Tracking RMSE, final 100 s (m)")
            ax.legend(fontsize=6.5, loc="upper left")
            save_report(fig, out.figure("plans"))
        for f in filters:
            ratios = [r / b for (b, r, _) in pts[f] if np.isfinite(r)]
            if ratios:
                macros[macro_name("e5 plan ratio median", f)] = fmt(float(np.median(ratios)), 2)
        # medians over the five flight plans, per region and filter
        pc = pd.DataFrame(
            pcsv,
            columns=[
                "region",
                "plan",
                "filter",
                "conv",
                "pcrb",
                "tuned",
                "rmse",
                "ratio",
                "ratio_tuned",
            ],
        )
        for (region, f), g in pc.groupby(["region", "filter"]):
            base = ("e5 plan", REGION_SHORT.get(region, region), f)
            macros[macro_name(*base, "pcrb")] = fmt(float(g["pcrb"].median()), 1)
            macros[macro_name(*base, "tuned")] = fmt(float(g["tuned"].median()), 1)
            macros[macro_name(*base, "rmse")] = fmt(float(g["rmse"].median()), 1)
            macros[macro_name(*base, "ratio")] = fmt(float(g["ratio"].median()), 2)
            macros[macro_name(*base, "ratio tuned")] = fmt(float(g["ratio_tuned"].median()), 2)
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
