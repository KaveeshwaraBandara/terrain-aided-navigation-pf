"""Download Copernicus GLO-30 DEM tiles from OpenTopography and reproject to UTM 44N.

Usage:
    python scripts/download_dem.py --region hill_country
    python scripts/download_dem.py --all
    python scripts/download_dem.py --bbox 6.85 7.03 80.68 80.86 --name my_box

Needs a free OpenTopography API key in the environment or in .env as OPENTOPO_API_KEY.
Endpoint: OpenTopography Global DEM API (/API/globaldem, demtype=COP30 = Copernicus 30 m DSM).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from tan.config import REPO_ROOT, load_regions
from tan.dem import UTM_44N, reproject_to_metric

API_URL = "https://portal.opentopography.org/API/globaldem"
ATTRIBUTION = (
    "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
    "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved"
)


def download(
    name: str,
    south: float,
    north: float,
    west: float,
    east: float,
    api_key: str,
    demtype: str,
    force: bool,
) -> Path:
    raw = REPO_ROOT / "data" / "raw" / f"{name}_{demtype.lower()}.tif"
    out = REPO_ROOT / "data" / "processed" / f"{name}_utm44n.tif"
    raw.parent.mkdir(parents=True, exist_ok=True)

    if raw.exists() and not force:
        print(f"[{name}] using cached {raw.relative_to(REPO_ROOT)}")
    else:
        params = {
            "demtype": demtype,
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
        print(f"[{name}] requesting {demtype} for S{south} N{north} W{west} E{east} ...")
        resp = requests.get(API_URL, params=params, timeout=120)
        if resp.status_code != 200:
            sys.exit(f"[{name}] OpenTopography returned {resp.status_code}: {resp.text[:300]}")
        if not resp.content.startswith((b"II*\x00", b"MM\x00*")):  # TIFF magic bytes
            sys.exit(f"[{name}] response is not a GeoTIFF: {resp.text[:300]}")
        raw.write_bytes(resp.content)
        print(f"[{name}] saved {raw.relative_to(REPO_ROOT)} ({len(resp.content) / 1e6:.1f} MB)")

    reproject_to_metric(raw, out, dst_crs=UTM_44N, res=30.0)
    print(f"[{name}] reprojected → {out.relative_to(REPO_ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--region", help="key in configs/regions.yaml")
    group.add_argument("--all", action="store_true", help="every region in configs/regions.yaml")
    group.add_argument("--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"))
    ap.add_argument("--name", help="output name when using --bbox")
    ap.add_argument("--demtype", default="COP30", help="OpenTopography demtype (default COP30)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENTOPO_API_KEY", "").strip()
    if not api_key:
        sys.exit("Set OPENTOPO_API_KEY in .env (copy .env.example) or your environment.")

    regions = load_regions()
    if args.bbox:
        if not args.name:
            sys.exit("--name is required with --bbox")
        jobs = {args.name: dict(zip(("south", "north", "west", "east"), args.bbox, strict=True))}
    elif args.all:
        jobs = regions
    else:
        if args.region not in regions:
            sys.exit(f"unknown region '{args.region}'. Options: {', '.join(regions)}")
        jobs = {args.region: regions[args.region]}

    for name, box in jobs.items():
        download(
            name,
            box["south"],
            box["north"],
            box["west"],
            box["east"],
            api_key,
            args.demtype,
            args.force,
        )
    print(f"\nData attribution (keep in the report): {ATTRIBUTION}")


if __name__ == "__main__":
    main()
