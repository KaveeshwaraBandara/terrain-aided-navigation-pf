import numpy as np
import pytest

from tan.config import deep_update
from tan.simulate import (
    correlated_field,
    filter_rng,
    flight_margin,
    initial_belief,
    run_streams,
    simulate,
)


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


# ---------------------------------------------------------------- random flights + CRN
def _random_cfg(cfg, **extra):
    sim = deep_update(
        cfg["simulation"],
        {"flight": {"mode": "random", "margin_sigma0": 1000.0}, "duration": 120.0, **extra},
    )
    return sim


def test_random_flight_stays_inside_margin(cfg, synthetic_dem):
    sim_cfg = _random_cfg(cfg)
    margin = flight_margin(sim_cfg)
    starts = []
    for run in range(10):
        sim = simulate(sim_cfg, cfg["init"], synthetic_dem, run_streams(7, run))
        assert synthetic_dem.contains(sim.truth[:, 0], sim.truth[:, 1], margin=margin).all()
        starts.append(sim.p0_true)
    assert np.unique(np.round(starts), axis=0).shape[0] == 10  # really random


def test_crn_flight_and_noise_do_not_depend_on_sigma0(cfg, synthetic_dem):
    sim_cfg = _random_cfg(cfg)
    a = simulate(sim_cfg, {"position_sigma": 100.0}, synthetic_dem, run_streams(3, 5))
    b = simulate(sim_cfg, {"position_sigma": 2000.0}, synthetic_dem, run_streams(3, 5))
    np.testing.assert_array_equal(a.truth, b.truth)
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(a.v_ins, b.v_ins)
    # same direction of the initial error, scaled by σ₀
    np.testing.assert_allclose((b.p0_guess - b.p0_true) / 2000.0, (a.p0_guess - a.p0_true) / 100.0)


def test_crn_outlier_rate_does_not_shift_other_streams(cfg, synthetic_dem):
    lo = _random_cfg(cfg, outlier_prob=0.01)
    hi = _random_cfg(cfg, outlier_prob=0.10)
    a = simulate(lo, cfg["init"], synthetic_dem, run_streams(3, 2))
    b = simulate(hi, cfg["init"], synthetic_dem, run_streams(3, 2))
    np.testing.assert_array_equal(a.p0_guess, b.p0_guess)
    np.testing.assert_array_equal(a.v_ins, b.v_ins)
    assert np.all(b.outliers[a.outliers])  # nested: every 1 % outlier is also a 10 % outlier
    same = ~b.outliers
    np.testing.assert_array_equal(a.z[same], b.z[same])


def test_filter_streams_are_stable_and_distinct():
    a = filter_rng(42, 3, "particle_filter").random(3)
    b = filter_rng(42, 3, "particle_filter").random(3)
    c = filter_rng(42, 3, "rbpf").random(3)
    d = filter_rng(42, 4, "particle_filter").random(3)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c) and not np.allclose(a, d)


def test_corridor_altitude_is_lower_than_box_altitude(cfg, synthetic_dem):
    box = simulate(_random_cfg(cfg), cfg["init"], synthetic_dem, run_streams(1, 0))
    cor = simulate(
        _random_cfg(cfg, altitude_reference="corridor"),
        cfg["init"],
        synthetic_dem,
        run_streams(1, 0),
    )
    assert cor.altitude_msl <= box.altitude_msl
    assert cor.radar_range.min() >= cfg["simulation"]["clearance_m"] - 1e-6


def test_map_error_field_statistics(rng):
    f = correlated_field((400, 400), 30.0, 600.0, rng)
    assert f.std() == pytest.approx(1.0) and abs(f.mean()) < 1e-9
    # correlation at one correlation length ≈ exp(−½) for the squared-exponential model
    lag = 20  # 20 px × 30 m = 600 m
    c = np.mean(f[:, :-lag] * f[:, lag:])
    assert c == pytest.approx(np.exp(-0.5), abs=0.12)


def test_map_error_offset_shifts_measurements(cfg, synthetic_dem):
    base = _random_cfg(cfg)
    off = _random_cfg(cfg, map_error={"offset_m": 20.0})
    a = simulate(base, cfg["init"], synthetic_dem, run_streams(9, 1))
    b = simulate(off, cfg["init"], synthetic_dem, run_streams(9, 1))
    np.testing.assert_allclose(b.z - a.z, 20.0)


def test_flight_plan_fixes_the_flight_but_not_the_noise(cfg, synthetic_dem):
    sim_cfg = _random_cfg(cfg)
    a = simulate(sim_cfg, cfg["init"], synthetic_dem, run_streams(3, 0, plan=2))
    b = simulate(sim_cfg, cfg["init"], synthetic_dem, run_streams(3, 1, plan=2))
    np.testing.assert_array_equal(a.truth, b.truth)
    assert not np.allclose(a.z, b.z) and not np.allclose(a.p0_guess, b.p0_guess)
