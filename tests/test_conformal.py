"""Tests of mdfm/eval/conformal.py (B4.11): split conformal quantile and coverage, ellipsoid factor,
degenerate predictives, band width and the energy score against its closed form."""
import math

import numpy as np
import pytest
from scipy import stats

from mdfm.eval import conformal as C


def test_rank_and_quantile_follow_the_finite_sample_rule():
    s = np.arange(1.0, 10.0)                      # 9 scores: k = ceil(10 x 0.9) = 9, the largest
    assert C.rank(9, 0.9) == 9 and C.fit(s, 0.9) == 9.0
    assert C.rank(8, 0.9) == 9 and C.fit(s[:8], 0.9) == math.inf
    assert C.rank(99, 0.9) == 90 and C.fit(np.arange(1.0, 100.0), 0.9) == 90.0
    with pytest.raises(ValueError):
        C.fit([1.0, float("nan")], 0.9)
    with pytest.raises(ValueError):
        C.fit([], 0.9)
    with pytest.raises(ValueError):
        C.fit([1.0], 1.0)


def test_coverage_is_090_within_001_on_100000_exchangeable_draws():
    rng = np.random.default_rng(0)
    q = C.fit(rng.gamma(2.0, size=10000), 0.90)
    cov = C.coverage(rng.gamma(2.0, size=100000), q)
    assert abs(cov - 0.90) <= 0.01, cov


def test_marginal_coverage_of_small_calibration_sets_is_k_over_n_plus_1():
    rng = np.random.default_rng(1)
    n, reps = 19, 20000                           # k = 18, expected coverage 18 / 20 = 0.90
    cal = rng.standard_normal((reps, n))
    new = rng.standard_normal(reps)
    q = np.array([C.fit(c, 0.9) for c in cal])
    assert abs(np.mean(new <= q) - 0.90) <= 0.01


def test_ellipsoid_factor_rescales_a_too_narrow_gaussian():
    rng = np.random.default_rng(2)
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.02, 0.0], [0.0, 0.0, 0.01]])
    L = np.linalg.cholesky(2.0 * cov)             # truths spread twice the predictive variance
    mean = np.array([0.1, -0.2, 0.3])
    r, bad = C.mahalanobis(mean + rng.standard_normal((20000, 3)) @ L.T, mean, cov)
    assert not bad.any()
    c = C.ellipsoid_factor(r, 3, 0.9)
    assert abs(c - math.sqrt(2.0)) < 0.03, c
    r2, _ = C.mahalanobis(mean + rng.standard_normal((100000, 3)) @ L.T, mean, cov)
    assert abs(np.mean(C.covered(r2, c, 3, 0.9)) - 0.90) <= 0.01
    # the unscaled ellipsoid of the right covariance covers 0.90
    r3, _ = C.mahalanobis(mean + rng.standard_normal((100000, 3)) @ np.linalg.cholesky(cov).T, mean, cov)
    assert abs(np.mean(C.covered(r3, 1.0, 3, 0.9)) - 0.90) <= 0.01


def test_degenerate_predictives_never_cover_and_are_flagged():
    y = np.zeros((4, 3))
    same = np.ones((64, 3)) * 0.5
    mean, cov = C.moment_match(same)
    assert np.all(cov == 0) and C.degenerate(cov)
    r, bad = C.mahalanobis(y, mean, cov)
    assert bad.all() and np.isinf(r).all() and not C.covered(r, 1e9).any()
    line = np.outer(np.linspace(-1, 1, 64), [1.0, 2.0, 0.5])         # rank one: degenerate as well
    assert C.degenerate(C.moment_match(line)[1])
    assert not C.degenerate(np.eye(3) * 1e-6)
    assert C.degenerate(np.full((3, 3), np.nan))
    # per-frame predictives: one degenerate frame among good ones
    draws = np.random.default_rng(3).standard_normal((3, 64, 3))
    draws[1] = 0.0
    m, c = C.moment_match(draws)
    r, bad = C.mahalanobis(np.zeros((3, 3)), m, c)
    assert bad.tolist() == [False, True, False] and np.isinf(r[1]) and np.isfinite(r[[0, 2]]).all()
    # too few calibration scores give c = inf, which covers every finite radius and still no degenerate one
    c_inf = C.ellipsoid_factor([1.0, 2.0, 3.0])
    assert c_inf == math.inf
    assert C.covered(r, c_inf).tolist() == [True, False, True]


def test_moment_match_and_band_width():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 3))
    m, c = C.moment_match(x)
    assert np.allclose(m, x.mean(0)) and np.allclose(c, np.cov(x, rowvar=False, ddof=1))
    z = stats.norm.ppf(0.95)
    w = C.axis_widths(np.diag([0.01, 0.04, 0.09]), 2.0, 0.9)
    assert np.allclose(w, 2 * z * 2.0 * np.array([0.1, 0.2, 0.3]))
    assert np.isclose(C.band_width(np.diag([0.01, 0.04, 0.09]), 2.0, 0.9), 4 * z * (0.1 * 0.2 * 0.3) ** (1 / 3))
    assert C.band_width(np.zeros((3, 3))) == 0.0
    with pytest.raises(ValueError):
        C.moment_match(x[:1])


def test_energy_score_of_a_gaussian_matches_its_closed_form():
    rng = np.random.default_rng(5)
    for dim, sigma, y in ((3, 0.05, np.array([0.02, -0.03, 0.01])), (3, 0.1, np.zeros(3)),
                          (3, 0.02, np.array([0.2, 0.0, 0.0])), (1, 0.3, np.array([0.4]))):
        draws = rng.standard_normal((4000, dim)) * sigma
        es = C.energy_score(draws, y)
        exact = C.energy_score_gaussian(np.zeros(dim), sigma, y if dim == 3 else y[0], dim)
        assert abs(es - exact) <= 0.02 * abs(exact) + 1e-4 * sigma, (dim, sigma, es, exact)
    # the closed form is continuous at lam = 0 and grows with the distance
    e = [C.energy_score_gaussian(np.zeros(3), 1.0, np.array([d, 0.0, 0.0])) for d in (0.0, 1e-9, 1e-6, 0.5, 2.0)]
    assert abs(e[0] - e[1]) < 1e-9 and abs(e[1] - e[2]) < 1e-9 and e[2] < e[3] < e[4]


def test_energy_score_is_zero_for_the_truth_and_broadcasts_a_shared_draw_set():
    y = np.array([0.1, 0.2, 0.3])
    assert C.energy_score(np.tile(y, (64, 1)), y) == 0.0
    rng = np.random.default_rng(6)
    draws = rng.standard_normal((64, 3))
    ys = rng.standard_normal((5, 3))
    shared = C.energy_score(draws, ys)
    each = C.energy_score(np.broadcast_to(draws, (5, 64, 3)), ys)
    assert shared.shape == (5,) and np.allclose(shared, each)
    assert np.allclose(shared, [C.energy_score(draws, v) for v in ys])
