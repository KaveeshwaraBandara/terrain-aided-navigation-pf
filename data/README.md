# Data

Nothing in `raw/` or `processed/` is committed — regenerate with `make data` (needs a free
OpenTopography API key in `.env`, see `.env.example`). What *is* committed is
`manifest.json`, which records exactly which data every result was computed from.

## What is downloaded, and how

| File | Contents |
|---|---|
| `raw/<region>_cop30.tif` | Copernicus GLO-30 (`demtype=COP30`, "Copernicus Global DSM 30m"), WGS 84 lat/lon, one request per region box in `configs/regions.yaml` |
| `raw/srilanka_cop90.tif` | Copernicus GLO-90 (`COP90`) over the whole island — only used to choose the regions (`scripts/survey_regions.py`) and for the region map |
| `processed/<region>_utm44n.tif` | The GLO-30 box reprojected (bilinear) to a 30 m grid in UTM zone 44N (EPSG:32644), then cropped to the largest rectangle without no-data on its border — what the filters load |
| `manifest.json` | Per region: requested box, CRS, grid shape, resolution, UTM bounds, crop, no-data/sea fraction and SHA-256 of the raw file, the processed file and the elevation array |

Source: the OpenTopography Global DEM API, `GET https://portal.opentopography.org/API/globaldem`
with `demtype`, `south`, `north`, `west`, `east`, `outputFormat=GTiff` and `API_Key`, checked
against OpenTopography's OpenAPI specification (fetched 2026-11-19). Limits on a free key:
450 000 km² per request for 30 m data; 50 requests per 24 h (200 for academic users). `make data`
uses 4 requests; downloads are cached and never retried in a loop. The key is never printed —
also not inside request-error messages (see `src/tan/data.py`).

## Study regions

Chosen with a rule fixed before any data was looked at (`report/findings.md`, 2026-11-19):
candidate 0.36° × 0.36° boxes (≈ 40 km × 40 km) around Nuwara Eliya, Kandy and Anuradhapura are
scored by their mean 1 km roughness on the COP90 survey raster; the roughest box around Nuwara
Eliya, the flattest around Anuradhapura, and around Kandy the one closest to the geometric mean of
the two (no Kandy box comes close: all 144 candidates are rougher than the target, so the least
rough one was chosen). All three boxes are entirely on land (0 % no-data/sea).

## Accuracy and caveats

From the Copernicus DEM Product Handbook, v5.0 (29 Nov 2022, GEO.2018-1988-2), Table 1:
absolute vertical accuracy < 4 m and relative vertical accuracy < 2 m (slopes ≤ 20 %) or < 4 m
(slopes > 20 %), all as 90 % linear errors (relative: point-to-point within 1° × 1°).

The Copernicus DEM is a **digital surface model**: it "represents the surface of the Earth
including buildings, infrastructure and vegetation". Over forest a radar altimeter may measure to
the ground or partly into the canopy, not to the canopy top the map shows — a real, spatially
varying measurement bias, studied in Experiment 4. Water bodies are flattened in the edited
source model.

## Attribution

produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space
GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
Data accessed through OpenTopography.
