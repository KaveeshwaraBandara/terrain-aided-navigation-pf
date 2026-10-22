# Data

Nothing in `raw/` or `processed/` is committed — regenerate with `make data`.

| Folder | Contents |
|---|---|
| `raw/` | Copernicus GLO-30 GeoTIFFs as downloaded from OpenTopography (WGS84 lat/lon) |
| `processed/` | The same tiles reprojected to UTM 44N (EPSG:32644), 30 m grid — what the filters load |

**Note for the report:** the Copernicus DEM is a *surface* model (includes buildings and
vegetation), so over forest a radar altimeter measures to the canopy, not the ground.
That is a real, modelable measurement bias.

## Attribution

produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space
GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
Data accessed through OpenTopography.
