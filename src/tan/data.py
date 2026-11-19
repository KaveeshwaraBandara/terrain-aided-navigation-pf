"""Download Copernicus DEM tiles from OpenTopography, reproject, crop, and record provenance.

API facts below were checked against OpenTopography's own OpenAPI specification
(https://portal.opentopography.org/apidocs/openapi.json, fetched 2026-11-19) and the
developer page (https://opentopography.org/developers):

- ``GET https://portal.opentopography.org/API/globaldem``
- required query parameters: ``demtype`` (``COP30`` = "Copernicus Global DSM 30m",
  ``COP90`` = "Copernicus Global DSM 90m"), ``south``, ``north``, ``west``, ``east``
  (WGS 84 degrees), ``API_Key``; optional ``outputFormat`` (``GTiff`` is the default)
- responses: 200 OK, 204 No Data, 400 Bad request, 401 Unauthorized, 500 Internal error
- limits: 450,000 km² per request for 30 m data (4,050,000 km² for COP90); a free key
  allows 50 calls per 24 h (200 for academic users)

The API key is a secret. It is sent as a query parameter, so any error message that
contains the request URL would leak it; every message printed here goes through
:func:`redact` first.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT
from .dem import UTM_44N, crop_to_valid, reproject_to_metric

API_URL = "https://portal.opentopography.org/API/globaldem"
MANIFEST = REPO_ROOT / "data" / "manifest.json"
ATTRIBUTION = (
    "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
    "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved"
)


class DownloadError(RuntimeError):
    """A download failed; the message is safe to print (the key is redacted)."""


def load_api_key() -> str:
    """Read OPENTOPO_API_KEY from the environment or ``.env``. Never print the result."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return os.environ.get("OPENTOPO_API_KEY", "").strip()


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_path(name: str, demtype: str) -> Path:
    return REPO_ROOT / "data" / "raw" / f"{name}_{demtype.lower()}.tif"


def processed_path(name: str) -> Path:
    return REPO_ROOT / "data" / "processed" / f"{name}_utm44n.tif"


def fetch_globaldem(
    name: str, box: dict[str, float], demtype: str, api_key: str, force: bool = False
) -> Path:
    """Download one bounding box (cached in ``data/raw``). One attempt, no retry loop."""
    import requests

    raw = raw_path(name, demtype)
    if raw.exists() and not force:
        print(f"[{name}] using cached {raw.relative_to(REPO_ROOT)}")
        return raw
    if not api_key:
        raise DownloadError("OPENTOPO_API_KEY is not set (put it in .env; see .env.example).")
    params = {
        "demtype": demtype,
        "south": box["south"],
        "north": box["north"],
        "west": box["west"],
        "east": box["east"],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    print(
        f"[{name}] requesting {demtype} S{box['south']} N{box['north']} "
        f"W{box['west']} E{box['east']} ..."
    )
    try:
        resp = requests.get(API_URL, params=params, timeout=300)
    except requests.RequestException as exc:  # str(exc) contains the URL, hence the key
        raise DownloadError(f"[{name}] request failed: {redact(str(exc), api_key)}") from None
    body = redact(resp.text[:300], api_key) if resp.status_code != 200 else ""
    if resp.status_code == 401:
        raise DownloadError(f"[{name}] OpenTopography rejected the API key (401). {body}")
    if resp.status_code == 204:
        raise DownloadError(f"[{name}] OpenTopography has no data for this box (204).")
    if resp.status_code != 200:
        raise DownloadError(f"[{name}] OpenTopography returned {resp.status_code}: {body}")
    if not resp.content.startswith((b"II*\x00", b"MM\x00*")):  # TIFF magic bytes
        raise DownloadError(
            f"[{name}] response is not a GeoTIFF: {redact(resp.text[:300], api_key)}"
        )
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(resp.content)
    print(f"[{name}] saved {raw.relative_to(REPO_ROOT)} ({len(resp.content) / 1e6:.1f} MB)")
    return raw


def nodata_fraction(path: str | Path, sea_level_is_nodata: bool = True) -> float:
    """Fraction of cells that are no-data (or ≤ 0 m, i.e. sea, if requested)."""
    import rasterio

    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
    bad = np.ma.getmaskarray(band) | ~np.isfinite(band.filled(np.nan))
    if sea_level_is_nodata:
        bad |= band.filled(0.0) <= 0.0
    return float(bad.mean())


def prepare_region(
    name: str,
    box: dict[str, float],
    api_key: str,
    demtype: str = "COP30",
    res: float = 30.0,
    force: bool = False,
    max_nodata: float = 0.01,
) -> dict[str, Any]:
    """Download → check no-data/sea fraction → reproject to UTM 44N → crop to valid data.

    Returns the manifest entry for the region.
    """
    import rasterio

    raw = fetch_globaldem(name, box, demtype, api_key, force)
    frac = nodata_fraction(raw)
    if frac > max_nodata:
        raise DownloadError(
            f"[{name}] {frac:.2%} of the box is no-data or sea (limit {max_nodata:.0%}); "
            "move the box inland."
        )
    out = processed_path(name)
    tmp = out.with_suffix(".uncropped.tif")
    reproject_to_metric(raw, tmp, dst_crs=UTM_44N, res=res)
    crop = crop_to_valid(tmp, out)
    tmp.unlink()
    with rasterio.open(out) as src:
        elev = src.read(1).astype(np.float32)
        t = src.transform
        entry = {
            "demtype": demtype,
            "request_bbox_wgs84": {k: float(box[k]) for k in ("south", "north", "west", "east")},
            "raw_file": str(raw.relative_to(REPO_ROOT)),
            "raw_sha256": sha256_file(raw),
            "raw_nodata_or_sea_fraction": frac,
            "processed_file": str(out.relative_to(REPO_ROOT)),
            "processed_sha256": sha256_file(out),
            "elevation_sha256": hashlib.sha256(elev.tobytes()).hexdigest(),
            "crs": src.crs.to_string(),
            "shape": [int(src.height), int(src.width)],
            "res_m": float(t.a),
            "bounds_utm_m": {
                "left": float(src.bounds.left),
                "right": float(src.bounds.right),
                "bottom": float(src.bounds.bottom),
                "top": float(src.bounds.top),
            },
            "crop": crop,
        }
    print(f"[{name}] reprojected + cropped → {entry['processed_file']} {entry['shape']}")
    return entry


def update_manifest(entries: dict[str, dict[str, Any]]) -> Path:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest.update(entries)
    manifest = dict(sorted(manifest.items()))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return MANIFEST


def manifest_hash() -> str | None:
    """SHA-256 of the manifest file — stored with every experiment result."""
    return sha256_file(MANIFEST) if MANIFEST.exists() else None
