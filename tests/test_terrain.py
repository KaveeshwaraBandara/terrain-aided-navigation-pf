import numpy as np

from tan.dem import DEM
from tan.terrain import along_track_stats, local_std, terrain_stats, window_px_for


def plane(a=0.04, b=-0.03, n=201, res=30.0):
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    return DEM(a * xx + b * yy + 500.0, x0=0.0, y0=float(xs[-1]), res=res)


def test_local_std_of_constant_is_zero():
    assert np.allclose(local_std(np.full((40, 40), 1234.5), 7), 0.0, atol=1e-9)


def test_local_std_of_checkerboard():
    h = np.indices((60, 60)).sum(0) % 2 * 10.0  # 0/10 alternating → std 5 (odd window ≈ 5)
    s = local_std(h, 21)[20:40, 20:40]
    assert np.allclose(s, 5.0, atol=0.02)


def test_window_is_odd():
    assert window_px_for(30.0, 1000.0) == 33
    assert window_px_for(90.0, 1000.0) == 11


def test_plane_stats():
    a, b, sz = 0.04, -0.03, 3.6
    s = terrain_stats(plane(a, b), meas_sigma=sz)
    assert np.isclose(s["slope_mean_pct"], 100 * np.hypot(a, b))
    assert np.isclose(s["slope_std_pct"], 0.0, atol=1e-9)
    assert np.isclose(s["fisher_density_per_m2"], (a * a + b * b) / sz**2)


def test_along_track_split_on_plane():
    a, b, sz = 0.04, -0.03, 3.6
    dem = plane(a, b)
    xy = np.column_stack([np.linspace(1000, 4000, 100), np.full(100, 3000.0)])  # due east
    s = along_track_stats(dem, xy, meas_sigma=sz)
    assert np.isclose(s["fisher_along_per_m2"], 100 * a * a / sz**2)
    assert np.isclose(s["fisher_cross_per_m2"], 100 * b * b / sz**2)
    assert np.isclose(s["fisher_total_per_m2"], s["fisher_along_per_m2"] + s["fisher_cross_per_m2"])
