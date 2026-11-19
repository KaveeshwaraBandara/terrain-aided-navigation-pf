import numpy as np
import pytest

from tan import data
from tan.dem import DEM, crop_to_valid, valid_window

FAKE_KEY = "not-a-real-key-0123456789abcdef"


def test_valid_window_trims_corner_wedges():
    valid = np.ones((50, 60), dtype=bool)
    for i in range(4):  # small triangular no-data wedges in two corners
        valid[i, : 4 - i] = False
        valid[-1 - i, -(4 - i) :] = False
    r0, r1, c0, c1 = valid_window(valid)
    inner = valid[r0:r1, c0:c1]
    assert inner.all()
    assert inner.size > 0.8 * valid.size


def test_valid_window_keeps_clean_raster():
    assert valid_window(np.ones((10, 12), dtype=bool)) == (0, 10, 0, 12)


def test_crop_to_valid_writes_clean_raster(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    elev = np.arange(40 * 30, dtype="float32").reshape(40, 30) + 100.0
    elev[:3, :5] = -9999.0
    src = tmp_path / "in.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=30,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:32644",
        transform=from_origin(500_000, 800_000, 30, 30),
        nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)
    info = crop_to_valid(src, tmp_path / "out.tif")
    dem = DEM.from_geotiff(tmp_path / "out.tif")
    assert info["interior_nodata_cells"] == 0
    assert np.all(dem.elevation >= 100.0)
    # geolocation is preserved: the pixel at a given easting/northing keeps its value
    x, y = 500_000 + 15 + 10 * 30, 800_000 - 15 - 20 * 30
    assert np.isclose(float(dem.height(x, y)), elev[20, 10])


def test_redact():
    assert data.redact(f"url?API_Key={FAKE_KEY}&x=1", FAKE_KEY) == "url?API_Key=***&x=1"
    assert data.redact("nothing", "") == "nothing"


def test_request_errors_never_contain_the_key(monkeypatch, tmp_path):
    requests = pytest.importorskip("requests")

    def boom(url, params, timeout):
        q = "&".join(f"{k}={v}" for k, v in params.items())
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}?{q}")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(data, "raw_path", lambda name, demtype: tmp_path / "x.tif")
    box = {"south": 7.0, "north": 7.1, "west": 80.0, "east": 80.1}
    with pytest.raises(data.DownloadError) as exc:
        data.fetch_globaldem("t", box, "COP30", FAKE_KEY)
    assert FAKE_KEY not in str(exc.value)
    assert "***" in str(exc.value)


def test_http_errors_never_contain_the_key(monkeypatch, tmp_path):
    requests = pytest.importorskip("requests")

    class Resp:
        status_code = 401
        text = f"Invalid key {FAKE_KEY}"
        content = b""

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Resp())
    monkeypatch.setattr(data, "raw_path", lambda name, demtype: tmp_path / "x.tif")
    box = {"south": 7.0, "north": 7.1, "west": 80.0, "east": 80.1}
    with pytest.raises(data.DownloadError) as exc:
        data.fetch_globaldem("t", box, "COP30", FAKE_KEY)
    assert FAKE_KEY not in str(exc.value) and "401" in str(exc.value)
