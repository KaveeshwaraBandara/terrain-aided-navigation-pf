import numpy as np

from tan.config import deep_update
from tan.simulate import initial_belief, simulate


def test_shapes_and_bounds(cfg, synthetic_dem, rng):
    sim = simulate(cfg["simulation"], cfg["init"], synthetic_dem, rng)
    K = sim.n_steps
    assert sim.truth.shape == (K, 2) and sim.v_ins.shape == (K, 2) and sim.z.shape == (K,)
    assert synthetic_dem.contains(sim.truth[:, 0], sim.truth[:, 1]).all()


def test_noise_free_measurement_equals_terrain(cfg, synthetic_dem, rng):
    quiet = deep_update(cfg["simulation"], {"baro_sigma": 0.0, "radar_sigma": 0.0})
    sim = simulate(quiet, cfg["init"], synthetic_dem, rng)
    terrain = synthetic_dem.height(sim.truth[:, 0], sim.truth[:, 1])
    np.testing.assert_allclose(sim.z, terrain, atol=1e-9)


def test_initial_belief(cfg, synthetic_dem, rng):
    sim = simulate(cfg["simulation"], cfg["init"], synthetic_dem, rng)
    mean, cov = initial_belief(sim)
    assert mean.shape == (4,) and cov.shape == (4, 4)
    np.testing.assert_allclose(mean[:2], sim.p0_guess)
