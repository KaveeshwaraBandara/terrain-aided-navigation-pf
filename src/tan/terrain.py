"""Terrain statistics: how much position information does a patch of terrain carry?

Used to choose the study regions, to describe them in the report, and to relate
per-flight filter performance to the terrain under the flight (Experiment 2).

Definitions (every one is stated in the report):

- elevation range / std: over all grid cells of the region.
- slope: |∇h| in percent grade (100 · m/m), from central differences on the grid.
- roughness: standard deviation of elevation inside a moving square window of
  ``window_m`` metres (default 1 km), averaged over the region. Insensitive to a
  constant tilt over large distances, sensitive to hills and valleys at the scale
  the filter has to resolve.
- Fisher information density: E[|∇h|²] / σ_z² (1/m²). For a locally linear terrain
  one altimeter measurement adds |∇h|²/σ_z² to the position information along the
  gradient direction, so this is the average information per measurement and the
  quantity that drives the PCRB (Experiment 5).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .dem import DEM


def local_std(elevation: np.ndarray, window_px: int) -> np.ndarray:
    """Standard deviation of ``elevation`` in a ``window_px`` × ``window_px`` moving window.

    Computed as sqrt(E[h²] − E[h]²) with box filters, after removing the global
    mean to limit floating-point cancellation. Edges use reflection.
    """
    window_px = max(1, int(window_px))
    h = np.asarray(elevation, dtype=np.float64)
    h = h - h.mean()
    m1 = uniform_filter(h, size=window_px, mode="reflect")
    m2 = uniform_filter(h * h, size=window_px, mode="reflect")
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def window_px_for(res: float, window_m: float) -> int:
    """Odd window size in pixels that best matches ``window_m`` metres."""
    n = max(1, int(round(window_m / res)))
    return n if n % 2 == 1 else n + 1


def slope_magnitude(dem: DEM) -> np.ndarray:
    """|∇h| in m/m at every grid cell."""
    return np.hypot(dem._dhdx, dem._dhdy)


def terrain_stats(dem: DEM, meas_sigma: float = 3.6, window_m: float = 1000.0) -> dict[str, float]:
    """Summary statistics of a whole DEM (see the module docstring for definitions)."""
    h = dem.elevation
    g = slope_magnitude(dem)
    rough = local_std(h, window_px_for(dem.res, window_m))
    return {
        "elev_min_m": float(h.min()),
        "elev_max_m": float(h.max()),
        "elev_range_m": float(np.ptp(h)),
        "elev_std_m": float(h.std()),
        "slope_mean_pct": float(100.0 * g.mean()),
        "slope_std_pct": float(100.0 * g.std()),
        "roughness_1km_m": float(rough.mean()),
        "fisher_density_per_m2": float(np.mean(g * g) / meas_sigma**2),
    }


def along_track_stats(
    dem: DEM, xy: np.ndarray, meas_sigma: float = 3.6, window_m: float = 1000.0
) -> dict[str, float]:
    """Terrain statistics *under one flight path* ``xy`` (K, 2) — for Experiment 2.

    - profile std: std of the terrain heights the altimeter actually sees;
    - roughness: mean 1 km local std along the path (bilinear lookup);
    - Fisher information: Σ_k |∇h(p_k)|² / σ_z² (1/m²), the total terrain
      information collected along the path, and its split along/across track.
    """
    x, y = xy[:, 0], xy[:, 1]
    heights = dem.height(x, y)
    gx, gy = dem.gradient(x, y)
    rough_grid = local_std(dem.elevation, window_px_for(dem.res, window_m))
    rough = DEM(rough_grid, x0=dem.x0, y0=dem.y0, res=dem.res).height(x, y)

    # unit vectors along and across the local direction of travel
    d = np.gradient(xy, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    g_along = gx * t[:, 0] + gy * t[:, 1]
    g_cross = -gx * t[:, 1] + gy * t[:, 0]
    s2 = meas_sigma**2
    return {
        "profile_std_m": float(np.std(heights)),
        "roughness_1km_m": float(np.mean(rough)),
        "fisher_total_per_m2": float(np.sum(gx * gx + gy * gy) / s2),
        "fisher_along_per_m2": float(np.sum(g_along**2) / s2),
        "fisher_cross_per_m2": float(np.sum(g_cross**2) / s2),
    }
