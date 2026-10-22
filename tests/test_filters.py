"""End-to-end checks on synthetic terrain. Filter tests skip until the TODOs are done."""

import numpy as np
import pytest

from tan.config import deep_update
from tan.experiment import make_filter, run_filter
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
