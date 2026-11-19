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


@pytest.mark.parametrize("likelihood", ["gaussian", "student_t"])
def test_likelihood_is_a_normalised_density(synthetic_dem, likelihood):
    pf = ParticleFilter(synthetic_dem, n_particles=10, likelihood=likelihood, nu=4.0)
    r = np.linspace(-2000.0, 2000.0, 400_001)
    area = np.trapezoid(np.exp(pf.log_likelihood(r)), r)
    assert area == pytest.approx(1.0, abs=2e-3)


def test_student_t_survives_a_gross_outlier(synthetic_dem, rng):
    """One 300 m altimeter outlier: the Gaussian PF throws away the particles at the
    truth, the Student-t PF keeps them."""
    x0 = np.array([8000.0, 9000.0, 0.0, 0.0])
    h_true = float(synthetic_dem.height(x0[0], x0[1]))
    near = {}
    for lik in ("gaussian", "student_t"):
        pf = ParticleFilter(synthetic_dem, n_particles=4000, likelihood=lik, rng=rng)
        pf.initialize(x0, np.diag([300.0**2] * 2 + [0.01**2] * 2))
        pf.update(h_true + 300.0)
        d = np.linalg.norm(pf.particles[:, :2] - x0[:2], axis=1)
        near[lik] = pf.weights[d < 30.0].sum()
    assert near["student_t"] > 10 * near["gaussian"]


def test_pf_logs_evidence_and_nis(cfg, synthetic_dem, rng):
    pf_cfg = deep_update(cfg, {"particle_filter": {"n_particles": 2000, "snapshot_every": 0}})
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=50.0)
    run = run_filter(make_filter("particle_filter", pf_cfg, synthetic_dem, rng), sim)
    assert run.extras["log_evidence"].shape == (sim.n_steps,)
    assert np.all(np.isfinite(run.extras["log_evidence"]))
    # once locked on, the normalised innovation squared averages ≈ 1 (χ²₁ mean)
    assert 0.5 < np.mean(run.extras["nis"][-100:]) < 2.0


def test_rbpf_matches_kalman_filter_on_planar_terrain(cfg, rng):
    """A11: on a plane the RBPF must reproduce the exact Kalman filter, bias included."""
    from tan.filters import RaoBlackwellizedPF

    n, res = 668, 30.0
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    dem = DEM(0.05 * xx + 0.03 * yy + 200.0, x0=0.0, y0=float(xs[-1]), res=res)
    sim = _sim(cfg, dem, rng, position_sigma=100.0)
    kf = run_filter(ExtendedKalmanFilter(dem, **cfg["ekf"]), sim)
    kw = {k: v for k, v in cfg["particle_filter"].items() if k != "n_particles"}
    rb = run_filter(RaoBlackwellizedPF(dem, n_particles=5000, rng=rng, **kw), sim)
    _require(rb)
    kf_sd = np.sqrt(np.einsum("kii->ki", kf.cov))
    z = np.abs(rb.est - kf.est) / kf_sd
    assert z[:, :2].max() < 0.25
    assert z[:, 2:].max() < 0.25
    ratio = np.einsum("kii->ki", rb.cov) / np.einsum("kii->ki", kf.cov)
    assert 0.75 < ratio.min() and ratio.max() < 1.35


def _monitor_cfg(cfg, **mon):
    return deep_update(
        cfg,
        {
            "particle_filter": {
                "n_particles": 3000,
                "snapshot_every": 0,
                "monitor": {"enabled": True, **mon},
            }
        },
    )


def test_monitor_stays_quiet_when_tracking(cfg, synthetic_dem, rng):
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=30.0)
    run = run_filter(make_filter("particle_filter", _monitor_cfg(cfg), synthetic_dem, rng), sim)
    assert run.extras["alarms"].sum() <= 1
    assert run.error[-50:].mean() < 30.0


def test_monitor_recovers_from_forced_false_lock(cfg, synthetic_dem, rng):
    from tan.simulate import initial_belief

    sim = _sim(cfg, synthetic_dem, rng, position_sigma=800.0)
    pf = make_filter("particle_filter", _monitor_cfg(cfg, fraction=0.5), synthetic_dem, rng)
    mean, cov = initial_belief(sim)
    pf.initialize(mean, cov)
    # force a confident, wrong cloud 2.5 km from the truth
    wrong = sim.p0_true + np.array([2500.0, -1500.0])
    pf.particles[:, :2] = wrong + rng.normal(0.0, 5.0, size=(pf.n, 2))
    est = [pf.step(sim.v_ins[k], sim.dt, float(sim.z[k]))[0] for k in range(sim.n_steps)]
    err = np.linalg.norm(np.array(est)[:, :2] - sim.truth, axis=1)
    assert any(pf.history["alarms"])
    assert err[-30:].mean() < 60.0


def test_height_bias_state_absorbs_a_constant_offset(cfg, synthetic_dem, rng):
    """A7: with a 20 m constant measurement offset, the filter with an offset state
    estimates the offset and tracks; the estimate is returned as a 4-D state."""
    sim = _sim(cfg, synthetic_dem, rng, position_sigma=200.0)
    sim.z = sim.z + 20.0
    c = deep_update(
        cfg,
        {"particle_filter": {"n_particles": 4000, "snapshot_every": 0, "height_bias_sigma0": 30.0}},
    )
    run = run_filter(make_filter("particle_filter", c, synthetic_dem, rng), sim)
    assert run.est.shape == (sim.n_steps, 4)
    assert abs(run.extras["height_bias"][-50:].mean() - 20.0) < 3.0
    assert run.error[-50:].mean() < 40.0
