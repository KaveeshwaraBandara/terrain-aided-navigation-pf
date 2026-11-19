"""E4 — robustness to model mismatch: outliers, map error, INS grade."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ..plotting import TEXT_WIDTH_IN, report_style, save_report, style_of
from .common import (
    REGION_LABELS,
    REGION_SHORT,
    Output,
    axes_style,
    fmt,
    load_summary,
    macro_name,
    metric_table,
    pct,
    plot_vs,
    regions_in,
    series,
)

STUDIES = {
    "outliers": ("simulation.outlier_prob", "Outlier probability per measurement", 100.0, "\\%"),
    "map_error": ("simulation.map_error.offset_m", "Constant map / altimeter offset (m)", 1.0, ""),
    "ins_grade": ("simulation.ins.bias_sigma0", "Initial INS bias std (m/s)", 1.0, ""),
}


def make(exp: str = "e4", quick: bool = False) -> None:
    rng = np.random.default_rng(0)
    df = load_summary(exp, quick)
    out = Output(exp, quick)
    macros: dict[str, str] = {}
    rows, csv_rows = [], []
    for study, (col, xlabel, xscale, unit) in STUDIES.items():
        d = df[df["study"] == study]
        if d.empty:
            continue
        regions = regions_in(d)
        filters = [
            f
            for f in [
                "dead_reckoning",
                "ekf",
                "pf_5000",
                "pf_student_t",
                "pf_sz_x2",
                "pf_sz_x4",
                "pf_hbias",
            ]
            if f in set(d["filter"])
        ]
        # dead reckoning never converges; in the map-error figure the EKF is left out because
        # its orange cannot be told apart from the offset-state PF's red (table keeps it)
        shown = [
            f
            for f in filters
            if f != "dead_reckoning" and not (study == "map_error" and f == "ekf")
        ]
        stats = metric_table(d, ["region", col, "filter"], rng)
        xs = sorted(d[col].unique())
        with report_style():
            fig = Figure(figsize=(TEXT_WIDTH_IN, 3.9), layout="constrained")
            axes = fig.subplots(2, len(regions), sharex=True, sharey="row", squeeze=False)
            for j, region in enumerate(regions):
                for i, key in enumerate(["conv", "track"]):
                    ax = axes[i, j]
                    axes_style(ax)
                    plot_vs(
                        ax,
                        [x * xscale for x in xs],
                        shown,
                        {
                            (x * xscale, f): stats[(region, x, f)]
                            for x in xs
                            for f in shown
                            if (region, x, f) in stats
                        },
                        key,
                    )
                axes[0, j].set_title(REGION_LABELS.get(region, region))
                axes[1, j].set_xlabel(xlabel)
            axes[0, 0].set_ylabel("Converged runs (%)")
            axes[1, 0].set_ylabel("Tracking RMSE (m)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(5, len(labels)),
                handlelength=3.5,
            )
            save_report(fig, out.figure(study))

        for region in regions:
            for x in xs:
                for f in filters:
                    st = stats.get((region, x, f))
                    if st is None:
                        continue
                    c, tr = st["conv"], st["track"]
                    rows.append(
                        [
                            study.replace("_", " "),
                            REGION_LABELS.get(region, region),
                            f"{x * xscale:g}{unit}",
                            style_of(f)["label"],
                            f"{pct(c['rate'])} [{pct(c['lo'])}, {pct(c['hi'])}]",
                            pct(st["fl"]["rate"]),
                            f"{fmt(tr['rmse'])} [{fmt(tr['lo'])}, {fmt(tr['hi'])}]",
                            fmt(st["rmse"]["rmse"]),
                        ]
                    )
                    csv_rows.append(
                        [
                            study,
                            region,
                            x,
                            f,
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
                    base = (
                        "e4",
                        study,
                        REGION_SHORT.get(region, region),
                        fmt(x * xscale, 1 if x * xscale % 1 else 0).replace(".", " point "),
                        f,
                    )
                    macros[macro_name(*base, "conv")] = pct(c["rate"])
                    macros[macro_name(*base, "track")] = fmt(tr["rmse"])
                    macros[macro_name(*base, "rmse")] = fmt(st["rmse"]["rmse"])

    out.table(
        "summary",
        [
            "Study",
            "Region",
            "Setting",
            "Filter",
            "Conv.\\ (\\%) [95\\% CI]",
            "FL (\\%)",
            "Tracking RMSE (m) [95\\% CI]",
            "RMSE (m)",
        ],
        rows,
        align="llllrrrr",
        csv_header=[
            "study",
            "region",
            "setting",
            "filter",
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
    # one table per study as well (the combined one is too long for a report page)
    for study in STUDIES:
        srows = [r[1:] for r in rows if r[0] == study.replace("_", " ")]
        if srows:
            out.table(
                f"summary_{study}",
                [
                    "Region",
                    "Setting",
                    "Filter",
                    "Conv.\\ (\\%) [95\\% CI]",
                    "FL (\\%)",
                    "Tracking RMSE (m) [95\\% CI]",
                    "RMSE (m)",
                ],
                srows,
                align="lllrrrr",
            )

    # height-offset state: how well is the true offset recovered?
    d = df[(df["study"] == "map_error") & (df["filter"] == "pf_hbias")]
    hrows, hcsv = [], []
    for (region, config), g in d.groupby(["region", "config"]):
        data = series(exp, quick, "map_error", config, ["pf_hbias__height_bias", "map_error"])
        if data["pf_hbias__height_bias"].size == 0:
            continue
        c_hat = data["pf_hbias__height_bias"][:, -100:].mean(axis=1)
        offset = float(g["simulation.map_error.offset_m"].iloc[0])
        conv = g.set_index("run").loc[data["_run"], "converged"].to_numpy()
        err = c_hat[conv] - offset
        hrows.append(
            [
                REGION_LABELS.get(region, region),
                f"{offset:g}",
                str(int(conv.sum())),
                fmt(np.mean(c_hat[conv]) if conv.any() else np.nan, 1),
                fmt(np.std(err) if conv.any() else np.nan, 1),
            ]
        )
        hcsv.append(
            [
                region,
                offset,
                int(conv.sum()),
                float(np.mean(c_hat[conv])) if conv.any() else np.nan,
                float(np.std(err)) if conv.any() else np.nan,
            ]
        )
        macros[macro_name("e4 hb", REGION_SHORT.get(region, region), f"o{offset:g}", "mean")] = fmt(
            np.mean(c_hat[conv]) if conv.any() else np.nan, 1
        )
    if hrows:
        out.table(
            "height_bias",
            [
                "Region",
                "True offset (m)",
                "Converged runs",
                "Mean estimate (m)",
                "Std of error (m)",
            ],
            hrows,
            align="lrrrr",
            csv_header=["region", "offset_m", "converged_runs", "mean_estimate_m", "std_error_m"],
            csv_rows=hcsv,
        )
    macros["EFourRuns"] = str(int(df.groupby(["config", "filter"]).size().min()))
    out.macros(macros)
    print(f"[{exp}] figures, tables and macros written to {out.root}")
