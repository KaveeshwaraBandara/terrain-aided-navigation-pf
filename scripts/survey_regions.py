"""Choose the three study regions by measured terrain roughness (pre-registered rule).

One COP90 request covers the whole island (data/raw/srilanka_cop90.tif, cached). For every
candidate box of BOX_DEG × BOX_DEG degrees whose corner lies on a GRID_DEG grid and that
contains the class's anchor town at least MARGIN_DEG inside its edges, compute the mean 1 km
roughness (local elevation std) and the no-data/sea fraction. Then:

    rugged    (anchor Nuwara Eliya) : the candidate with the HIGHEST mean roughness
    flat      (anchor Anuradhapura) : the candidate with the LOWEST mean roughness
    moderate  (anchor Kandy)        : among candidates that overlap neither box above, the one
                                      whose roughness is closest (in log) to the geometric
                                      mean of the rugged and flat choices
    all boxes: no-data/sea fraction < 1 %

The rule uses terrain statistics only, never filter results, and was written down in
report/findings.md before the survey ran.

Usage:
    python scripts/survey_regions.py            # survey, print, write table + map figure
    python scripts/survey_regions.py --write    # ... and write configs/regions.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import yaml

from tan.config import REGIONS_FILE, REPO_ROOT
from tan.data import DownloadError, fetch_globaldem, load_api_key
from tan.terrain import local_std

ISLAND = {"south": 5.85, "north": 9.90, "west": 79.50, "east": 82.00}
BOX_DEG, GRID_DEG, MARGIN_DEG = 0.36, 0.02, 0.06
# approximate town-centre coordinates (lat, lon), only used to anchor the candidate boxes
ANCHORS = {
    "hill_country": ("Nuwara Eliya", 6.950, 80.789),
    "kandy_rolling": ("Kandy", 7.291, 80.634),
    "north_central_plains": ("Anuradhapura", 8.311, 80.404),
}
DESCRIPTIONS = {
    "hill_country": "Central highlands around Nuwara Eliya — rugged relief (informative terrain)",
    "kandy_rolling": "Mid-country around Kandy — intermediate relief",
    "north_central_plains": "Dry-zone plains around Anuradhapura — flat (uninformative terrain)",
}


def load_survey():
    import rasterio

    api_key = load_api_key()
    try:
        raw = fetch_globaldem("srilanka", ISLAND, "COP90", api_key)
    except DownloadError as exc:
        sys.exit(str(exc))
    with rasterio.open(raw) as src:
        band = src.read(1, masked=True).astype(np.float64)
        t = src.transform
    elev = band.filled(np.nan)
    bad = ~np.isfinite(elev) | (elev <= 0.0)  # no-data or sea
    return np.where(bad, 0.0, elev), bad, t


def integral(a: np.ndarray) -> np.ndarray:
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s


def box_mean(s: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    total = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    return float(total / ((r1 - r0) * (c1 - c0)))


def candidates(lat: float, lon: float):
    offsets = np.arange(MARGIN_DEG, BOX_DEG - MARGIN_DEG + 1e-9, GRID_DEG)
    s0 = np.floor((lat - offsets) / GRID_DEG + 0.5) * GRID_DEG
    w0 = np.floor((lon - offsets) / GRID_DEG + 0.5) * GRID_DEG
    for s in np.unique(np.round(s0, 2)):
        for w in np.unique(np.round(w0, 2)):
            if s + MARGIN_DEG <= lat <= s + BOX_DEG - MARGIN_DEG and (
                w + MARGIN_DEG <= lon <= w + BOX_DEG - MARGIN_DEG
            ):
                yield {
                    "south": round(float(s), 2),
                    "north": round(float(s) + BOX_DEG, 2),
                    "west": round(float(w), 2),
                    "east": round(float(w) + BOX_DEG, 2),
                }


def overlaps(a: dict, b: dict) -> bool:
    return not (
        a["north"] <= b["south"]
        or b["north"] <= a["south"]
        or a["east"] <= b["west"]
        or b["east"] <= a["west"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write configs/regions.yaml")
    args = ap.parse_args()

    elev, bad, t = load_survey()
    px_deg = abs(t.a)
    window = int(round(1000.0 / (px_deg * 111_320.0)))  # ≈ 1 km in pixels (≈ 11 at 3")
    window += window % 2 == 0
    rough = local_std(elev, window)
    rough[bad] = 0.0
    s_rough, s_bad = integral(rough), integral(bad.astype(float))

    def idx(box):
        r0 = int(round((t.f - box["north"]) / px_deg))
        r1 = int(round((t.f - box["south"]) / px_deg))
        c0 = int(round((box["west"] - t.c) / px_deg))
        c1 = int(round((box["east"] - t.c) / px_deg))
        return r0, r1, c0, c1

    rows = []
    for key, (town, lat, lon) in ANCHORS.items():
        for box in candidates(lat, lon):
            r0, r1, c0, c1 = idx(box)
            frac_bad = box_mean(s_bad, r0, r1, c0, c1)
            land = 1.0 - frac_bad
            mean_rough = box_mean(s_rough, r0, r1, c0, c1) / max(land, 1e-9)
            rows.append(
                {
                    "region": key,
                    "anchor": town,
                    **box,
                    "roughness_1km_m": mean_rough,
                    "nodata_or_sea": frac_bad,
                }
            )

    def pick(key, choose, exclude=()):
        ok = [
            r
            for r in rows
            if r["region"] == key
            and r["nodata_or_sea"] < 0.01
            and not any(overlaps(r, e) for e in exclude)
        ]
        if not ok:
            sys.exit(f"no valid candidate for {key}")
        return choose(ok)

    rugged = pick("hill_country", lambda c: max(c, key=lambda r: r["roughness_1km_m"]))
    flat = pick("north_central_plains", lambda c: min(c, key=lambda r: r["roughness_1km_m"]))
    target = np.sqrt(rugged["roughness_1km_m"] * flat["roughness_1km_m"])
    moderate = pick(
        "kandy_rolling",
        lambda c: min(c, key=lambda r: abs(np.log(r["roughness_1km_m"] / target))),
        exclude=(rugged, flat),
    )
    chosen = {"hill_country": rugged, "kandy_rolling": moderate, "north_central_plains": flat}
    for r in rows:
        r["chosen"] = any(r is c for c in chosen.values())

    tables = REPO_ROOT / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "region_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"candidates: {len(rows)}; geometric-mean target roughness {target:.1f} m")
    for key, r in chosen.items():
        n = sum(1 for x in rows if x["region"] == key)
        vals = [x["roughness_1km_m"] for x in rows if x["region"] == key]
        print(
            f"  {key:22s} S{r['south']:.2f} N{r['north']:.2f} W{r['west']:.2f} "
            f"E{r['east']:.2f}  roughness {r['roughness_1km_m']:6.1f} m "
            f"(candidates {n}: {min(vals):.1f}–{max(vals):.1f} m)  "
            f"no-data/sea {r['nodata_or_sea']:.2%}"
        )

    from tan.plotting import plot_region_map

    plot_region_map(
        np.where(bad, np.nan, elev),
        t,
        chosen,
        ANCHORS,
        REPO_ROOT / "report" / "figures" / "region_map",
    )

    if args.write:
        out = {
            k: {
                "description": DESCRIPTIONS[k],
                **{e: chosen[k][e] for e in ("south", "north", "west", "east")},
            }
            for k in chosen
        }
        header = (
            "# Study regions (WGS84 lat/lon boxes, 0.36° ≈ 40 km × 40 km), chosen by\n"
            "# scripts/survey_regions.py with the pre-registered roughness rule described there.\n"
            "# Every region is reprojected to UTM zone 44N (EPSG:32644).\n\n"
        )
        REGIONS_FILE.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
        print(f"wrote {REGIONS_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
