import numpy as np
import pytest

from tan.config import load_config
from tan.dem import DEM


@pytest.fixture
def todo():
    """Call ``fn``; skip the test (instead of failing) while it is still a TODO stub."""

    def _call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            pytest.skip(f"TODO not implemented yet: {exc}")

    return _call


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synthetic_dem():
    return DEM.synthetic(size_m=20_000, res=30, n_hills=60, relief=600, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
