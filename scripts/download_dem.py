"""Download Copernicus GLO-30 DEM tiles from OpenTopography and reproject to UTM 44N.

Usage:
    python scripts/download_dem.py --region hill_country
    python scripts/download_dem.py --all
    python scripts/download_dem.py --bbox 6.85 7.03 80.68 80.86 --name my_box

Needs a free OpenTopography API key in the environment or in .env as OPENTOPO_API_KEY.
Downloads are cached in data/raw; processed rasters go to data/processed and every
region is recorded (bounding box, CRS, shape, resolution, SHA-256) in data/manifest.json.
API details and limits: see the docstring of src/tan/data.py.
"""

from __future__ import annotations

import argparse
import sys

from tan.config import load_regions
from tan.data import ATTRIBUTION, DownloadError, load_api_key, prepare_region, update_manifest


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

    api_key = load_api_key()
    entries = {}
    try:
        for name, box in jobs.items():
            entries[name] = prepare_region(name, box, api_key, args.demtype, force=args.force)
    except DownloadError as exc:
        sys.exit(str(exc))
    finally:
        if entries:
            print(f"manifest: {update_manifest(entries)}")
    print(f"\nData attribution (keep in the report): {ATTRIBUTION}")


if __name__ == "__main__":
    main()
