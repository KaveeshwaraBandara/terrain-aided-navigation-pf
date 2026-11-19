"""Digital elevation model (DEM) with vectorised bilinear lookup.

The filters call :meth:`DEM.height` for thousands of particles per step, so every
lookup here is vectorised (no Python loops).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

UTM_44N = "EPSG:32644"  # UTM zone 44N covers all of Sri Lanka (78°E–84°E)


@dataclass
class DEM:
    """Elevation grid on a regular, north-up, projected (metric) grid.

    Attributes
    ----------
    elevation : (H, W) array of heights in metres. Row 0 is the northern edge.
    x0, y0 : easting / northing (m) of the *centre* of pixel [0, 0] (top-left).
    res : grid spacing in metres (square pixels).
    crs : coordinate reference system string, or None for synthetic terrain.
    """

    elevation: np.ndarray
    x0: float
    y0: float
    res: float
    crs: str | None = None
    _dhdx: np.ndarray = field(init=False, repr=False)
    _dhdy: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.elevation = np.asarray(self.elevation, dtype=np.float64)
        d_row, d_col = np.gradient(self.elevation, self.res)
        self._dhdx = d_col
        self._dhdy = -d_row  # row index grows southward, northing grows northward

    # ------------------------------------------------------------------ geometry
    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape  # type: ignore[return-value]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the pixel centres, in metres."""
        h, w = self.shape
        return (self.x0, self.x0 + (w - 1) * self.res, self.y0 - (h - 1) * self.res, self.y0)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) of the pixel edges — for matplotlib ``imshow``."""
        xmin, xmax, ymin, ymax = self.bounds
        r = self.res / 2
        return (xmin - r, xmax + r, ymin - r, ymax + r)

    def contains(self, x, y, margin: float = 0.0) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.bounds
        x, y = np.asarray(x), np.asarray(y)
        return (
            (x >= xmin + margin)
            & (x <= xmax - margin)
            & (y >= ymin + margin)
            & (y <= ymax - margin)
        )

    # ------------------------------------------------------------------ lookups
    def _interp(self, grid: np.ndarray, x, y) -> np.ndarray:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        rows = (self.y0 - y) / self.res
        cols = (x - self.x0) / self.res
        coords = np.vstack([rows.ravel(), cols.ravel()])
        # order=1 → bilinear; mode="nearest" clamps queries outside the grid to the edge.
        values = map_coordinates(grid, coords, order=1, mode="nearest")
        return values.reshape(x.shape)

    def height(self, x, y) -> np.ndarray:
        """Bilinearly interpolated terrain height (m) at easting ``x``, northing ``y``."""
        return self._interp(self.elevation, x, y)

    def gradient(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Terrain slope (dh/dx, dh/dy) in m/m — the EKF measurement Jacobian needs this."""
        return self._interp(self._dhdx, x, y), self._interp(self._dhdy, x, y)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_geotiff(cls, path: str | Path) -> DEM:
        """Load a projected, north-up GeoTIFF (e.g. the output of ``scripts/download_dem.py``)."""
        import rasterio

        with rasterio.open(path) as src:
            if src.crs is None or src.crs.is_geographic:
                raise ValueError(
                    f"{path} is not in a projected (metric) CRS. "
                    "Reproject it first — scripts/download_dem.py does this for you."
                )
            t = src.transform
            if t.b != 0 or t.d != 0 or not np.isclose(abs(t.a), abs(t.e)):
                raise ValueError("DEM must be north-up with square pixels.")
            elev = src.read(1, masked=True).astype(np.float64).filled(np.nan)
            crs = src.crs.to_string()
            res = float(t.a)
            x0 = t.c + t.a / 2
            y0 = t.f + t.e / 2

        missing = np.isnan(elev)
        if missing.any():
            frac = float(missing.mean())
            if frac > 0.05:
                raise ValueError(f"{path}: {frac:.1%} of the DEM is no-data; pick another box.")
            warnings.warn(
                f"{path}: filling {frac:.2%} no-data cells with nearest values", stacklevel=2
            )
            idx = distance_transform_edt(missing, return_distances=False, return_indices=True)
            elev = elev[tuple(idx)]
        return cls(elev, x0=x0, y0=y0, res=res, crs=crs)

    @classmethod
    def synthetic(
        cls,
        size_m: float = 20_000.0,
        res: float = 30.0,
        n_hills: int = 60,
        relief: float = 600.0,
        base: float = 200.0,
        seed: int = 0,
    ) -> DEM:
        """Random sum-of-Gaussians terrain, for tests and offline runs (no download needed)."""
        rng = np.random.default_rng(seed)
        n = int(size_m // res) + 1
        xs = np.arange(n) * res
        xx, yy = np.meshgrid(xs, xs[::-1])  # row 0 = north
        h = np.full(xx.shape, base, dtype=np.float64)
        for _ in range(n_hills):
            cx, cy = rng.uniform(0.0, size_m, size=2)
            s = rng.uniform(0.01, 0.12) * size_m
            a = rng.uniform(0.2, 1.0) * relief
            h += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s**2))
        return cls(h, x0=0.0, y0=float(xs[-1]), res=res, crs=None)


def reproject_to_metric(
    src_path: str | Path, dst_path: str | Path, dst_crs: str = UTM_44N, res: float = 30.0
) -> Path:
    """Reproject a geographic (lat/lon) GeoTIFF to a metric CRS with bilinear resampling."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=res
        )
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": nodata,
            "compress": "deflate",
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
    return dst_path


def valid_window(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Rows/cols ``(r0, r1, c0, c1)`` (half-open) of a sub-rectangle whose border is all valid.

    Reprojecting a lat/lon box to UTM rotates it slightly, which leaves thin
    no-data wedges along the edges of the output grid. This greedily trims the
    edge row or column with the most invalid cells until the border is clean.
    Interior voids (if any) are left for :meth:`DEM.from_geotiff` to fill.
    """
    valid = np.asarray(valid, dtype=bool)
    r0, r1, c0, c1 = 0, valid.shape[0], 0, valid.shape[1]
    while r1 - r0 > 2 and c1 - c0 > 2:
        bad = {
            "top": int((~valid[r0, c0:c1]).sum()),
            "bottom": int((~valid[r1 - 1, c0:c1]).sum()),
            "left": int((~valid[r0:r1, c0]).sum()),
            "right": int((~valid[r0:r1, c1 - 1]).sum()),
        }
        worst = max(bad, key=bad.get)
        if bad[worst] == 0:
            break
        if worst == "top":
            r0 += 1
        elif worst == "bottom":
            r1 -= 1
        elif worst == "left":
            c0 += 1
        else:
            c1 -= 1
    return r0, r1, c0, c1


def crop_to_valid(src_path: str | Path, dst_path: str | Path) -> dict[str, int]:
    """Write the largest clean-bordered sub-rectangle of ``src_path`` to ``dst_path``."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        band = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(band.filled(np.nan))
        r0, r1, c0, c1 = valid_window(valid)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        profile = src.profile.copy()
        profile.update(
            width=c1 - c0,
            height=r1 - r0,
            transform=src.window_transform(window),
            compress="deflate",
        )
        data = src.read(1, window=window)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)
    return {
        "rows_removed": int(valid.shape[0] - (r1 - r0)),
        "cols_removed": int(valid.shape[1] - (c1 - c0)),
        "interior_nodata_cells": int((~valid[r0:r1, c0:c1]).sum()),
    }
