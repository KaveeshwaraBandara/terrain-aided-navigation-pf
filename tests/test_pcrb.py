import numpy as np
import pytest

from tan.config import deep_update
from tan.dem import DEM
from tan.experiment import run_filter
from tan.filters import ExtendedKalmanFilter
from tan.pcrb import pcrb, position_rmse_bound, slope_outer, transition, true_noise
from tan.simulate import initial_belief, run_streams, simulate


def plane_dem(a=0.05, b=0.03, n=668, res=30.0):
    xs = np.arange(n) * res
    xx, yy = np.meshgrid(xs, xs[::-1])
    return DEM(a * xx + b * yy + 200.0, x0=0.0, y0=float(xs[-1]), res=res)


def test_pcrb_equals_kalman_covariance_for_linear_model():
    """For a linear-Gaussian model the PCRB is the Kalman filter covariance."""
    dt, R = 1.0, 13.0
    Q = np.diag([4.0, 4.0, 1e-5, 1e-5])
    P0 = np.diag([500.0**2] * 2 + [0.3**2] * 2)
    g = np.array([0.05, 0.03])
    G = np.tile(np.outer(g, g), (200, 1, 1))
    bound = pcrb(G, P0, Q, R, dt)

    F, H = transition(dt), np.array([[g[0], g[1], 0.0, 0.0]])
    P = P0.copy()
    for k in range(200):
        P = F @ P @ F.T + Q
        S = (H @ P @ H.T)[0, 0] + R
        K = P @ H.T / S
        P = P - K @ H @ P
        np.testing.assert_allclose(bound[k], P, rtol=1e-6, atol=1e-9)


def test_pcrb_matches_ekf_on_planar_terrain(cfg):
    dem = plane_dem()
    sim = simulate(
        cfg["simulation"],
        deep_update(cfg["init"], {"position_sigma": 200.0}),
        dem,
        run_streams(0, 0),
    )
    ekf = ExtendedKalmanFilter(dem, **cfg["ekf"])
    run = run_filter(ekf, sim)
    _, P0 = initial_belief(sim)
    bound = pcrb(slope_outer(dem, sim.truth), P0, ekf.Q, ekf.R, sim.dt)
    np.testing.assert_allclose(bound, run.cov, rtol=1e-6, atol=1e-6)


def test_true_noise_matches_simulator(cfg, synthetic_dem):
    """Q of tan.pcrb.true_noise is the covariance of the simulator's actual process noise."""
    sim_cfg = deep_update(cfg["simulation"], {"ins": {"noise_sigma": 0.05, "bias_rw_sigma": 0.02}})
    Q, R = true_noise(sim_cfg)
    ws = []
    for run in range(40):
        sim = simulate(sim_cfg, cfg["init"], synthetic_dem, run_streams(5, run))
        dt = sim.dt
        p = np.vstack([sim.p0_true, sim.truth])
        b = np.vstack([sim.bias[0] - np.nan, sim.bias])  # b_0 unknown → drop first step
        w_p = p[1:] - p[:-1] - dt * (sim.v_ins - b[:-1])
        w_b = b[1:] - b[:-1]
        ws.append(np.column_stack([w_p, w_b])[1:])
    W = np.vstack(ws)
    emp = np.cov(W.T)
    np.testing.assert_allclose(np.diag(emp), np.diag(Q), rtol=0.06)
    assert emp[0, 2] == pytest.approx(Q[0, 2], rel=0.1)
    assert R == pytest.approx(cfg["simulation"]["baro_sigma"] ** 2 + 9.0)


def test_rmse_bound_shape():
    b = np.tile(np.diag([9.0, 16.0, 0.0, 0.0]), (3, 1, 1))
    np.testing.assert_allclose(position_rmse_bound(b), 5.0)


def test_mismatched_kf_equals_pcrb_when_models_match():
    from tan.pcrb import mismatched_kf_error

    Q = np.diag([0.01, 0.01, 1e-6, 1e-6])
    P0 = np.diag([200.0**2] * 2 + [0.3**2] * 2)
    slopes = np.tile([0.05, 0.03], (100, 1))
    G = slopes[:, :, None] * slopes[:, None, :]
    a = mismatched_kf_error(slopes, P0, Q, 13.0, Q, 13.0, 1.0)
    np.testing.assert_allclose(a, pcrb(G, P0, Q, 13.0, 1.0), rtol=1e-8, atol=1e-10)


def test_mistuned_kf_is_worse_than_the_bound():
    from tan.pcrb import mismatched_kf_error

    Qt = np.diag([0.0025, 0.0025, 4e-6, 4e-6])
    Qf = np.diag([4.0, 4.0, 4e-6, 4e-6])
    P0 = np.diag([200.0**2] * 2 + [0.3**2] * 2)
    rng = np.random.default_rng(0)
    slopes = rng.normal(0.0, 0.2, size=(200, 2))
    G = slopes[:, :, None] * slopes[:, None, :]
    a = position_rmse_bound(mismatched_kf_error(slopes, P0, Qf, 13.0, Qt, 13.0, 1.0))
    b = position_rmse_bound(pcrb(G, P0, Qt, 13.0, 1.0))
    assert np.all(a[50:] >= b[50:] - 1e-9) and a[-1] > 1.2 * b[-1]
