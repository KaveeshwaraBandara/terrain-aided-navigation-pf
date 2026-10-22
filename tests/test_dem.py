import numpy as np
import pytest

from tan.dem import DEM, UTM_44N, reproject_to_metric


def plane_dem(a=0.03, b=-0.02, c=150.0, n=101, res=30.0):
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    return DEM(a * xx + b * yy + c, x0=0.0, y0=float(xs[-1]), res=res), (a, b, c)


def test_bilinear_is_exact_on_a_plane(rng):
    dem, (a, b, c) = plane_dem()
    xmin, xmax, ymin, ymax = dem.bounds
    x = rng.uniform(xmin, xmax, 1000)
    y = rng.uniform(ymin, ymax, 1000)
    np.testing.assert_allclose(dem.height(x, y), a * x + b * y + c, atol=1e-6)


def test_gradient_on_a_plane(rng):
    dem, (a, b, _) = plane_dem()
    x = rng.uniform(100, 2900, 50)
    y = rng.uniform(100, 2900, 50)
    gx, gy = dem.gradient(x, y)
    np.testing.assert_allclose(gx, a, atol=1e-9)
    np.testing.assert_allclose(gy, b, atol=1e-9)


def test_broadcasting_and_clamping():
    dem, _ = plane_dem()
    assert dem.height(np.zeros((3, 4)), 0.0).shape == (3, 4)
    xmin, xmax, ymin, ymax = dem.bounds
    # outside the grid → value at the nearest edge
    assert np.isclose(dem.height(xmax + 1e4, ymax), dem.height(xmax, ymax))


def test_synthetic_has_relief(synthetic_dem):
    assert np.ptp(synthetic_dem.elevation) > 100


@pytest.mark.filterwarnings("ignore:.*no-data cells")
def test_reproject_geographic_to_utm(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    # small lat/lon raster near Kandy: elevation ramps 1000 m per degree of longitude
    west, north, step, n = 80.50, 7.40, 0.0003, 300
    lons = west + step * (np.arange(n) + 0.5)
    elev = np.tile(100.0 + 1000.0 * (lons - west), (n, 1)).astype("float32")
    src = tmp_path / "geo.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, step, step),
    ) as dst:
        dst.write(elev, 1)

    out = reproject_to_metric(src, tmp_path / "utm.tif")
    dem = DEM.from_geotiff(out)
    assert dem.crs is not None and "32644" in dem.crs

    lon, lat = 80.545, 7.355
    (x,), (y,) = warp_transform("EPSG:4326", UTM_44N, [lon], [lat])
    assert abs(float(dem.height(x, y)) - (100.0 + 1000.0 * (lon - west))) < 1.0
