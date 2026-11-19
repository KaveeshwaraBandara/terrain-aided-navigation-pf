"""Terrain statistics and overview figure for the study regions (+ synthetic reference).

Outputs:
    report/tables/terrain_stats.{csv,tex}
    report/figures/terrain_overview.{pdf,png}
    report/generated/terrain.tex     (macros used in the report text)

Definitions are in src/tan/terrain.py.
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib.colors import LightSource
from matplotlib.figure import Figure

from tan.config import load_config, load_regions
from tan.experiment import build_dem
from tan.plotting import (
    ELEVATION_CMAP,
    INK,
    TEXT_WIDTH_IN,
    add_scale_bar,
    report_style,
    save_report,
)
from tan.reporting import FIGURES, fmt, macro_name, write_macros, write_table
from tan.terrain import terrain_stats

TITLES = {
    "hill_country": "Hill country (Nuwara Eliya)",
    "kandy_rolling": "Kandy (mid-country)",
    "north_central_plains": "Dry-zone plains (Anuradhapura)",
    "synthetic": "Synthetic (reference)",
}
SHORT = {
    "hill_country": "Hill",
    "kandy_rolling": "Kandy",
    "north_central_plains": "Plains",
    "synthetic": "Synthetic",
}
ROWS = [
    ("elev_min_m", "Min.\\ elevation (m)", 0),
    ("elev_max_m", "Max.\\ elevation (m)", 0),
    ("elev_range_m", "Elevation range (m)", 0),
    ("elev_std_m", "Elevation std (m)", 0),
    ("slope_mean_pct", "Mean slope (\\%)", 1),
    ("slope_std_pct", "Slope std (\\%)", 1),
    ("roughness_1km_m", "Roughness, 1 km window (m)", 1),
    ("fisher_density_per_m2", "Fisher density ($10^{-3}$\\,m$^{-2}$)", 2),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-only", action="store_true", help="for testing without data")
    args = ap.parse_args()

    cfg = load_config()
    meas_sigma = float(cfg["particle_filter"]["meas_sigma"])
    names = [] if args.synthetic_only else list(load_regions())
    names.append("synthetic")
    dems = {n: build_dem({**cfg, "region": n}) for n in names}
    stats = {n: terrain_stats(d, meas_sigma=meas_sigma) for n, d in dems.items()}

    header = ["Statistic"] + [TITLES[n].split(" (")[0] for n in names]
    rows, csv_rows = [], []
    for key, label, digits in ROWS:
        scale = 1e3 if key == "fisher_density_per_m2" else 1.0
        rows.append([label] + [fmt(stats[n][key] * scale, digits) for n in names])
        csv_rows.append([key] + [stats[n][key] for n in names])
    rows.append(
        ["Grid (cells)"] + [f"{d.shape[0]}\\,$\\times$\\,{d.shape[1]}" for d in dems.values()]
    )
    csv_rows.append(["grid_cells"] + [f"{d.shape[0]}x{d.shape[1]}" for d in dems.values()])
    write_table("terrain_stats", header, rows, csv_header=["statistic", *names], csv_rows=csv_rows)

    macros = {}
    for n in names:
        for key, _, digits in ROWS:
            scale = 1e3 if key == "fisher_density_per_m2" else 1.0
            macros[macro_name("terr", SHORT[n], key)] = fmt(stats[n][key] * scale, digits)
        # information in the exact (bilinear) gradient relative to the smoothed slopes the EKF
        # uses (central differences, interpolated) at the same 200 000 random positions
        d = dems[n]
        rs = np.random.default_rng(0)
        b = d.bounds
        x = rs.uniform(b[0] + 100, b[1] - 100, 200_000)
        y = rs.uniform(b[2] + 100, b[3] - 100, 200_000)
        ex, ey = d.bilinear_gradient(x, y)
        sx, sy = d.gradient(x, y)
        macros[macro_name("terr", SHORT[n], "grad ratio")] = fmt(
            float(np.mean(ex**2 + ey**2) / np.mean(sx**2 + sy**2)), 2
        )
        xmin, xmax, ymin, ymax = dems[n].extent
        macros[macro_name("terr", SHORT[n], "width km")] = fmt((xmax - xmin) / 1e3, 1)
        macros[macro_name("terr", SHORT[n], "height km")] = fmt((ymax - ymin) / 1e3, 1)
    # region boxes (as requested from OpenTopography) and the processed grids
    if not args.synthetic_only:
        import json

        from tan.config import REPO_ROOT

        manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text())
        regions = load_regions()
        brows, bcsv = [], []
        for n in names[:-1]:
            b, m = regions[n], manifest[n]
            w = (m["bounds_utm_m"]["right"] - m["bounds_utm_m"]["left"]) / 1e3
            h = (m["bounds_utm_m"]["top"] - m["bounds_utm_m"]["bottom"]) / 1e3
            brows.append(
                [
                    TITLES[n],
                    f"{b['south']:.2f}--{b['north']:.2f}",
                    f"{b['west']:.2f}--{b['east']:.2f}",
                    f"{w:.1f} $\\times$ {h:.1f}",
                    f"{m['shape'][0]}\\,$\\times$\\,{m['shape'][1]}",
                ]
            )
            bcsv.append(
                [
                    n,
                    b["south"],
                    b["north"],
                    b["west"],
                    b["east"],
                    w,
                    h,
                    *m["shape"],
                    m["elevation_sha256"],
                ]
            )
        write_table(
            "regions",
            [
                "Region",
                "Latitude ($^\\circ$N)",
                "Longitude ($^\\circ$E)",
                "Size (km)",
                "Grid (30 m cells)",
            ],
            brows,
            align="lrrrr",
            csv_header=[
                "region",
                "south",
                "north",
                "west",
                "east",
                "width_km",
                "height_km",
                "rows",
                "cols",
                "elevation_sha256",
            ],
            csv_rows=bcsv,
        )
    write_macros("terrain", macros)

    # --- overview figure: hillshade + elevation colour, one panel per region
    with report_style():
        n = len(names)
        ncol = 2 if n > 1 else 1
        nrow = int(np.ceil(n / ncol))
        fig = Figure(figsize=(TEXT_WIDTH_IN, 3.0 * nrow), layout="constrained")
        axes = np.atleast_1d(fig.subplots(nrow, ncol)).ravel()
        ls = LightSource(azdeg=315, altdeg=45)
        for ax, name in zip(axes, names, strict=False):
            dem = dems[name]
            h = dem.elevation
            xmin, xmax, ymin, ymax = dem.extent
            ext = (0.0, (xmax - xmin) / 1e3, 0.0, (ymax - ymin) / 1e3)
            # elevation colour ramp + translucent grey hillshade (no colour-shifting blend)
            sm = ax.imshow(
                h, cmap=ELEVATION_CMAP, extent=ext, origin="upper", interpolation="nearest"
            )
            shade = ls.hillshade(h, vert_exag=2.0, dx=dem.res, dy=dem.res)
            ax.imshow(
                shade,
                cmap="gray",
                extent=ext,
                origin="upper",
                alpha=0.35,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
            cb.set_label("Elevation (m)")
            cb.outline.set_visible(False)
            ax.set_xlabel("Easting from SW corner (km)")
            ax.set_ylabel("Northing (km)")
            ax.text(
                0.02,
                0.98,
                TITLES[name],
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color=INK,
                bbox={"facecolor": "white", "alpha": 0.8, "lw": 0, "pad": 1.5},
            )
            ax.set_aspect("equal")
            add_scale_bar(ax, 10.0 if ext[1] > 25 else 5.0)
        for ax in axes[n:]:
            ax.set_visible(False)
        stem = FIGURES / (
            "terrain_overview_synthetic" if args.synthetic_only else "terrain_overview"
        )
        for p in save_report(fig, stem):
            print(p)
    for n in names:
        s = stats[n]
        print(
            f"{n:22s} range {s['elev_range_m']:7.0f} m  slope {s['slope_mean_pct']:5.1f} %  "
            f"rough {s['roughness_1km_m']:6.1f} m  fisher {1e3 * s['fisher_density_per_m2']:.2f}e-3"
        )


if __name__ == "__main__":
    main()
