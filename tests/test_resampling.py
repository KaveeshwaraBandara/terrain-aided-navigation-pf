import numpy as np
import pytest

from tan.filters.resampling import RESAMPLERS, effective_sample_size

SCHEMES = sorted(RESAMPLERS)


def test_ess_bounds(todo):
    n = 100
    assert np.isclose(todo(effective_sample_size, np.full(n, 1 / n)), n)
    one_hot = np.zeros(n)
    one_hot[7] = 1.0
    assert np.isclose(todo(effective_sample_size, one_hot), 1.0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_output_is_valid_index_array(scheme, todo, rng):
    w = rng.random(500)
    w /= w.sum()
    idx = todo(RESAMPLERS[scheme], w, rng)
    assert idx.shape == (500,)
    assert np.issubdtype(idx.dtype, np.integer)
    assert idx.min() >= 0 and idx.max() < 500


@pytest.mark.parametrize("scheme", SCHEMES)
def test_degenerate_weights(scheme, todo, rng):
    w = np.zeros(50)
    w[13] = 1.0
    assert np.all(todo(RESAMPLERS[scheme], w, rng) == 13)


@pytest.mark.parametrize("scheme", ["systematic", "stratified", "residual"])
def test_uniform_weights_keep_every_particle(scheme, todo, rng):
    n = 64
    idx = todo(RESAMPLERS[scheme], np.full(n, 1 / n), rng)
    assert np.array_equal(np.sort(idx), np.arange(n))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_zero_weight_particles_never_selected(scheme, todo, rng):
    w = rng.random(1000)
    w[::3] = 0.0
    w /= w.sum()
    for _ in range(20):
        idx = todo(RESAMPLERS[scheme], w, rng)
        assert np.all(w[idx] > 0)


def test_systematic_offspring_is_floor_or_ceil(todo, rng):
    """Evenly spaced pointers → particle i gets floor(N w_i) or ceil(N w_i) copies."""
    n = 1000
    w = rng.random(n) ** 3
    w /= w.sum()
    for _ in range(20):
        counts = np.bincount(todo(RESAMPLERS["systematic"], w, rng), minlength=n)
        assert np.all(counts >= np.floor(n * w) - 1e-9)
        assert np.all(counts <= np.ceil(n * w) + 1e-9)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_unbiased_offspring_counts(scheme, todo, rng):
    w = np.array([0.5, 0.3, 0.15, 0.05])
    n_draws = 2000
    counts = np.zeros(4)
    for _ in range(n_draws):
        rep = np.repeat(np.arange(4), 25)  # 100 particles, 25 per hypothesis
        ww = w[rep] / w[rep].sum()
        counts += np.bincount(rep[todo(RESAMPLERS[scheme], ww, rng)], minlength=4)
    np.testing.assert_allclose(counts / counts.sum(), w, atol=0.01)
