"""End-to-end checks of the estimators on synthetic and planar terrain."""

import numpy as np
import pytest

from tan.config import deep_update
from tan.dem import DEM
from tan.experiment import make_filter, run_filter
from tan.filters import ExtendedKalmanFilter, ParticleFilter
from tan.simulate import simulate


def _sim(cfg, dem, rng, **init):
    return simulate(cfg["simulation"], deep_update(cfg["init"], init), dem, rng)


def test_dead_reckoning_drifts(cfg, synthetic_dem, rng):
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=1.0)
    run = run_filter(make_filter("dead_reckoning", cfg, synthetic_dem, rng), sim)
    assert run.status == "ok"
    # uncorrected INS bias → error grows over the flight
    assert run.error[-1] > run.error[:10].mean()


def _require(run):
    if run.status == "todo":
        pytest.skip(f"TODO not implemented yet: {run.message}")


def test_particle_filter_converges(cfg, synthetic_dem, rng):
    pf_cfg = deep_update(cfg, {"particle_filter": {"n_particles": 3000, "snapshot_every": 0}})
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=300.0)
    run = run_filter(make_filter("particle_filter", pf_cfg, synthetic_dem, rng), sim)
    _require(run)
    assert run.error[-30:].mean() < 60.0, "PF should lock on over informative synthetic terrain"


def test_ekf_covariance_stays_valid(cfg, synthetic_dem, rng):
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=30.0)
    run = run_filter(make_filter("ekf", cfg, synthetic_dem, rng), sim)
    _require(run)
    P = run.cov[-1]
    np.testing.assert_allclose(P, P.T, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_pf_matches_kalman_filter_on_planar_terrain(cfg, rng):
    """Linear-Gaussian sanity check.

    On a tilted plane h(p) = a·px + b·py + c the measurement is linear (bilinear
    interpolation and the slope are exact), so the EKF *is* the Kalman filter and
    gives the exact posterior. A correct particle filter must converge to it as N
    grows; with 20 000 particles it should agree to a small fraction of a sigma.
    """
    n, res = 668, 30.0
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    dem = DEM(0.05 * xx + 0.03 * yy + 200.0, x0=0.0, y0=float(xs[-1]), res=res)
    sim = _sim(cfg, dem, rng, position_sigma=100.0)

    kf = run_filter(ExtendedKalmanFilter(dem, **cfg["ekf"]), sim)
    pf_kwargs = {k: v for k, v in cfg["particle_filter"].items() if k != "n_particles"}
    pf = run_filter(ParticleFilter(dem, n_particles=20_000, rng=rng, **pf_kwargs), sim)
    _require(kf)
    _require(pf)

    kf_sd = np.sqrt(np.einsum("kii->ki", kf.cov))
    z = np.abs(pf.est - kf.est) / kf_sd  # mean difference in KF standard deviations
    assert z[:, :2].max() < 0.25, "PF position mean drifted from the exact KF solution"
    assert z[:, 2:].max() < 0.35, "PF bias mean drifted from the exact KF solution"

    trace_ratio = np.einsum("kii->ki", pf.cov)[:, :2].sum(1) / np.einsum("kii->ki", kf.cov)[
        :, :2
    ].sum(1)
    assert 0.75 < trace_ratio.min() and trace_ratio.max() < 1.35


def test_roughening_breaks_up_duplicates(synthetic_dem, rng):
    pf = ParticleFilter(synthetic_dem, n_particles=1000, roughening=0.2, rng=rng)
    pf.initialize(np.array([5000.0, 5000.0, 0.0, 0.0]), np.diag([100.0**2] * 2 + [0.1**2] * 2))
    pf.particles = pf.particles[np.repeat(np.arange(10), 100)]  # 10 unique particles
    pf.roughen()
    assert np.unique(pf.particles[:, 0]).size == 1000


def test_off_map_particles_get_zero_weight(synthetic_dem, rng):
    pf = ParticleFilter(synthetic_dem, n_particles=200, rng=rng)
    pf.initialize(np.array([5000.0, 5000.0, 0.0, 0.0]), np.diag([10.0**2] * 2 + [0.1**2] * 2))
    pf.particles[:50, 0] = -5000.0  # first 50 particles are 5 km west of the map
    pf.update(float(synthetic_dem.height(5000.0, 5000.0)))
    w = pf.weights
    assert np.all(w[:50] == 0.0)
    assert np.isclose(w.sum(), 1.0)
