import numpy as np
import pytest

from tan.metrics import (
    anees_bounds,
    bootstrap_ci,
    convergence_time,
    is_converged,
    is_false_lock,
    mcnemar,
    nees_in_bounds_fraction,
    paired_bootstrap_rmse_diff,
    position_nees,
    two_sigma_radius,
    wilson_interval,
)


def test_wilson_known_values():
    # closed forms: 0/10 → upper z²/(n + z²) = 0.2775; 5/10 → 0.2366 … 0.7634
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(1 - 0.2775, abs=1e-4)


def test_anees_bounds_match_wilson_hilferty():
    # independent check: χ²_k quantile ≈ k (1 − 2/(9k) + z √(2/(9k)))³ (Wilson–Hilferty)
    m, k = 100, 200

    def wh(z):
        return k * (1 - 2 / (9 * k) + z * np.sqrt(2 / (9 * k))) ** 3 / m

    lo, hi = anees_bounds(m, dim=2)
    assert lo == pytest.approx(wh(-1.959964), abs=2e-3)
    assert hi == pytest.approx(wh(1.959964), abs=2e-3)


def test_consistent_filter_has_anees_near_dim(rng):
    m, k = 400, 50
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    L = np.linalg.cholesky(P)
    e = rng.standard_normal((m, k, 2)) @ L.T
    cov = np.broadcast_to(P, (k, 2, 2))
    nees = np.stack([position_nees(np.c_[ei, np.zeros((k, 2))], cov, np.zeros((k, 2))) for ei in e])
    assert nees.mean() == pytest.approx(2.0, rel=0.05)
    assert nees_in_bounds_fraction(nees).mean() == pytest.approx(0.95, abs=0.02)


def test_two_sigma_radius():
    assert two_sigma_radius(np.diag([9.0, 4.0])) == pytest.approx(6.0)


def test_convergence_definitions():
    t = np.arange(1, 301, dtype=float)
    err = np.full(300, 10.0)
    err[:150] = 500.0  # converged at t = 151 s, holds for the final 150 s
    assert convergence_time(err, t, 50.0) == 151.0
    assert is_converged(err, t, 50.0, hold=100.0)
    late = err.copy()
    late[:250] = 500.0  # converges at 251 s: only 50 s of tracking left
    assert not is_converged(late, t, 50.0, hold=100.0)
    never = np.full(300, 500.0)
    assert not is_converged(never, t)


def test_false_lock():
    tight = np.diag([10.0**2, 5.0**2])  # 2σ radius 20 m
    loose = np.diag([100.0**2, 5.0**2])  # 2σ radius 200 m
    assert is_false_lock(3000.0, tight)
    assert not is_false_lock(3000.0, loose)  # wrong, but honest about it
    assert not is_false_lock(20.0, tight)  # right


def test_mcnemar():
    a = np.array([1] * 30 + [0] * 70, dtype=bool)
    r = mcnemar(a, a)
    assert r["p_value"] == 1.0 and r["a_only"] == 0 and r["b_only"] == 0
    b = a.copy()
    b[30:45] = True  # B succeeds on 15 runs where A failed, never the other way
    r = mcnemar(a, b)
    assert r["b_only"] == 15 and r["a_only"] == 0
    assert r["p_value"] == pytest.approx(2 * 0.5**15)


def test_bootstrap_ci_covers_mean(rng):
    x = rng.normal(10.0, 1.0, 400)
    lo, hi = bootstrap_ci(x, np.mean, rng)
    assert lo < 10.0 < hi and hi - lo < 0.3


def test_paired_bootstrap_detects_difference(rng):
    base = rng.exponential(100.0, 200)
    r = paired_bootstrap_rmse_diff(base * 1.5, base, rng)
    assert r["lo"] > 0
