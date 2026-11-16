"""mdfm.neural.stats on synthetic inputs only (task T29): null p-values, Holm, BH and BY, the enumerated cluster
bootstrap, tau_int in both conventions, the null counts, and the draft flags of the OPEN defaults."""
import inspect
import itertools
import math
import re

import numpy as np
import pytest
from scipy import stats as sstats

from mdfm.neural import descriptors as D
from mdfm.neural import stats as S


# --- Phipson-Smyth (NR-1.7, NR-1.8) ------------------------------------------------------------------------------

def test_phipson_smyth_counts_ties_and_sidedness():
    r = S.phipson_smyth_p(5.0, [1.0, 2.0, 5.0, 6.0], sided="greater")
    assert (r.b, r.k) == (2, 4)
    assert r.p == pytest.approx(3 / 5)
    assert r.p_min == pytest.approx(1 / 5)
    assert not r.enters_pass_decision
    r2 = S.phipson_smyth_p(-5.0, [1.0, -2.0, 5.0, -6.0], sided="two-sided")
    assert (r2.b, r2.k) == (2, 4)
    r3 = S.phipson_smyth_p(-5.0, [1.0, -2.0, 5.0, -6.0], sided="greater")
    assert r3.b == 3 and r3.p == pytest.approx(4 / 5)  # -6 is the one null below -5


def test_phipson_smyth_smallest_p_and_null_floor():
    r = S.phipson_smyth_p(10.0, np.zeros(19), sided="greater")
    assert r.b == 0 and r.p == pytest.approx(1 / 20) and not r.enters_pass_decision
    r = S.phipson_smyth_p(10.0, np.zeros(20), sided="greater")
    assert r.p == pytest.approx(1 / 21) and r.enters_pass_decision
    assert S.enters_pass_decision(20) and not S.enters_pass_decision(19)


def test_phipson_smyth_refusals():
    with pytest.raises(ValueError):
        S.phipson_smyth_p(1.0, [], sided="greater")
    with pytest.raises(ValueError):
        S.phipson_smyth_p(1.0, [0.0, np.nan], sided="greater")
    with pytest.raises(ValueError):
        S.phipson_smyth_p(1.0, [0.0], sided="less")
    with pytest.raises(TypeError):
        S.phipson_smyth_p(1.0, [0.0])  # sidedness has no default (NR-1.7)


def test_draw_null_indices_enumerates_or_draws_without_replacement():
    e = S.draw_null_indices(50)
    assert e.enumerated and e.k == 50 and np.array_equal(e.indices, np.arange(50))
    assert e.draft == ("n_draw (OPEN-33)",)
    d = S.draw_null_indices(5000, seed=3)
    assert not d.enumerated and d.k == 999 and np.unique(d.indices).size == 999
    assert d.indices.min() >= 0 and d.indices.max() < 5000
    assert np.array_equal(d.indices, S.draw_null_indices(5000, seed=3).indices)
    with pytest.raises(ValueError):
        S.draw_null_indices(5000)
    assert S.draw_null_indices(5000, seed=1, n_draw=100).draft == ()


# --- Holm (NR-7.3) ------------------------------------------------------------------------------------------------

def test_holm_worked_example():
    r = S.holm([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4p, 3p, 2p, 1p = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    assert np.allclose(r.p_adj, [0.03, 0.06, 0.06, 0.02])
    assert r.reject.tolist() == [True, False, False, True]
    assert r.draft == ("alpha (OPEN-01)",) and r.level == 0.05
    assert S.holm([0.01, 0.04], alpha=0.1).draft == ()


def test_holm_step_down_equals_adjusted_p_rule():
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = int(rng.integers(1, 9))
        p = rng.uniform(0, 0.1, m) ** rng.uniform(0.5, 2)
        r = S.holm(p, alpha=0.05)
        assert np.array_equal(r.reject, r.p_adj <= 0.05)
        assert np.all(r.p_adj >= r.p_raw - 1e-15) and np.all(r.p_adj <= 1)
        srt = r.p_adj[np.argsort(p, kind="stable")]
        assert np.all(np.diff(srt) >= 0)


def test_holm_not_run_primary_enters_with_p_one():
    r = S.holm([0.001, 1.0, 0.004, 0.2, 0.03, 0.009])
    assert r.m == 6 and r.p_adj[1] == 1.0


# --- BH and BY (NR-8.2) -------------------------------------------------------------------------------------------

P8 = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]


def test_bh_worked_example():
    r = S.benjamini_hochberg(P8)
    assert r.reject.tolist() == [True, True] + [False] * 6
    assert np.allclose(r.p_adj, [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 8 * 0.074 / 7, 0.205])
    assert r.draft == ("q (OPEN-58)",)


def test_bh_is_step_up():
    r = S.benjamini_hochberg([0.04, 0.045], q=0.05)
    assert r.reject.tolist() == [True, True]  # p(1) misses 0.025 but p(2) meets 0.05
    assert np.allclose(r.p_adj, [0.045, 0.045])


def test_by_worked_example_and_relation_to_bh():
    c8 = sum(1 / i for i in range(1, 9))
    r = S.benjamini_yekutieli(P8)
    assert r.reject.tolist() == [True] + [False] * 7
    assert np.allclose(r.p_adj, np.minimum(1, S.benjamini_hochberg(P8).p_adj * c8))
    rng = np.random.default_rng(1)
    for _ in range(300):
        p = rng.uniform(0, 0.2, int(rng.integers(1, 12)))
        bh, by = S.benjamini_hochberg(p, q=0.1), S.benjamini_yekutieli(p, q=0.1)
        assert np.array_equal(bh.reject, bh.p_adj <= 0.1)
        assert np.array_equal(by.reject, by.p_adj <= 0.1)
        assert np.all(by.reject <= bh.reject)


def test_fdr_dispatch_and_flags():
    assert S.fdr(P8).method == "bh"
    assert S.fdr(P8).draft == ("method (OPEN-58)", "q (OPEN-58)")
    assert S.fdr(P8, method="by", q=0.05).draft == ()
    with pytest.raises(ValueError):
        S.fdr(P8, method="holm")
    with pytest.raises(ValueError):
        S.benjamini_hochberg([0.5, 1.2])


# --- Enumerated cluster bootstrap (NR-1.5, NR-1.6) ----------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (3, 10), (4, 35), (5, 126), (6, 462), (7, 1716)])
def test_enumeration_size_and_weights(n, expected):
    assert S.n_cluster_resamples(n) == expected == math.comb(2 * n - 1, n)
    counts, w = S.enumerate_cluster_resamples(n)
    assert counts.shape == (expected, n)
    assert np.all(counts.sum(axis=1) == n) and np.all(counts >= 0)
    assert len({tuple(r) for r in counts}) == expected
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    pmf = sstats.multinomial(n, np.full(n, 1.0 / n)).pmf(counts)
    assert np.allclose(w, pmf, rtol=1e-10)


def test_four_patients_give_35_weighted_resamples():
    counts, w = S.enumerate_cluster_resamples(4)
    assert counts.shape[0] == 35
    rows = {tuple(r): wi for r, wi in zip(counts.tolist(), w)}
    assert rows[(4, 0, 0, 0)] == pytest.approx(1 / 256)
    assert rows[(1, 1, 1, 1)] == pytest.approx(24 / 256)
    assert rows[(2, 1, 1, 0)] == pytest.approx(12 / 256)


def test_cluster_bootstrap_enumerates_at_four_patients():
    vals = np.array([1.0, 2.0, 3.0, 10.0])
    r = S.cluster_bootstrap(S.cluster_mean(vals), 4)
    assert r.enumerated and r.n_resamples == 35
    assert r.estimate == pytest.approx(4.0)
    # a linear statistic: the weighted mean of the resampled values equals the sample mean
    assert float(np.dot(r.values, r.weights)) == pytest.approx(4.0)
    assert 1.0 <= r.lo <= r.estimate <= r.hi <= 10.0
    assert set(r.draft) == {"replicates (OPEN-04)", "level (OPEN-01)", "method (OPEN-04)"}


@pytest.mark.parametrize("level,lo,hi", [(0.95, 1.5, 8.0), (0.8, 1.75, 6.25)])
def test_enumerated_interval_bounds(level, lo, hi):
    """The bounds NR-7.10 reads, pinned two ways: values from the 35 weighted resamples of [1, 2, 3, 10], and an
    oracle over the 4^4 = 256 equally likely ordered draws, the bound being the smallest mean whose share of draws
    at or below it reaches a / 2 (lower) or 1 - a / 2 (upper)."""
    vals = np.array([1.0, 2.0, 3.0, 10.0])
    r = S.cluster_bootstrap(S.cluster_mean(vals), 4, level=level)
    assert r.enumerated and (r.lo, r.hi) == (pytest.approx(lo), pytest.approx(hi))
    means = np.sort([np.mean(vals[list(d)]) for d in itertools.product(range(4), repeat=4)])
    a = 1.0 - level
    assert r.lo == pytest.approx(means[math.ceil(256 * a / 2 - 1e-9) - 1])
    assert r.hi == pytest.approx(means[math.ceil(256 * (1 - a / 2) - 1e-9) - 1])


def test_cluster_bootstrap_switches_at_the_replicate_count():
    stat = S.cluster_mean([0.0, 1.0, 2.0, 3.0])
    assert S.cluster_bootstrap(stat, 4, replicates=35).enumerated
    drawn = S.cluster_bootstrap(stat, 4, replicates=34, seed=0)
    assert not drawn.enumerated and drawn.n_resamples == 34
    with pytest.raises(ValueError):
        S.cluster_bootstrap(stat, 4, replicates=34)  # a drawn bootstrap needs a seed
    big = S.cluster_bootstrap(S.cluster_mean(np.arange(10.0)), 10, seed=5)
    assert not big.enumerated and big.n_resamples == 2000
    again = S.cluster_bootstrap(S.cluster_mean(np.arange(10.0)), 10, seed=5)
    assert (big.lo, big.hi) == (again.lo, again.hi)


def test_cluster_bootstrap_undefined_statistic():
    def stat(counts):
        return float("nan") if np.count_nonzero(counts) < 2 else float(np.count_nonzero(counts))
    with pytest.raises(ValueError):
        S.cluster_bootstrap(stat, 4)
    r = S.cluster_bootstrap(stat, 4, nan_policy="drop")
    assert r.undefined_weight == pytest.approx(4 / 256) and r.n_resamples == 31


def test_weighted_quantile_is_the_inverted_cdf():
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([0.5, 0.25, 0.25])
    assert S.weighted_quantile(v, w, 0.25) == 1.0
    assert S.weighted_quantile(v, w, 0.26) == 2.0
    assert S.weighted_quantile(v, w, 0.5) == 2.0
    assert S.weighted_quantile(v, w, 0.51) == 3.0


# --- tau_int and the null counts (NR-5.6 a, NR-5.7, OPEN-31, OPEN-33) --------------------------------------------

def ar1(phi, n, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    x = np.empty(n)
    x[0] = w[0] / math.sqrt(1 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + w[i]
    return x


def test_tau_int_recovers_ar1_in_both_conventions():
    phi, dt = 0.8, 0.01
    x = ar1(phi, 200_000, 5)
    d = S.tau_int(x, dt)
    s = S.tau_int(x, dt, convention="sokal")
    true_draft = (1 + phi) / (1 - phi)  # 9 samples
    assert d.tau_steps == pytest.approx(true_draft, rel=0.08)
    assert s.tau_steps == pytest.approx(true_draft / 2, rel=0.08)
    assert d.window == s.window and d.window_found
    assert d.tau_steps == pytest.approx(2 * s.tau_steps, rel=1e-12)
    assert d.tau_s == pytest.approx(d.tau_steps * dt)
    assert d.window >= 6.0 * d.tau_sokal_steps
    assert set(d.draft) == {"convention (OPEN-31)", "c_sokal (OPEN-31)", "estimator (OPEN-31)"}


def test_autocorrelation_estimators_on_a_short_series():
    """On 30 samples the two estimators differ; the draft one divides the lag-j autocovariance by n - j, and tau_int
    sums it up to Sokal's window."""
    x = ar1(0.7, 30, 11)
    n, d = x.size, x - x.mean()
    acov = np.array([np.dot(d[: n - j], d[j:]) for j in range(n)])
    unbiased = (acov / (n - np.arange(n))) / (acov[0] / n)
    biased = acov / acov[0]
    assert np.allclose(S.autocorrelation(x), unbiased) and np.allclose(S.autocorrelation(x, estimator="biased"), biased)
    assert not np.allclose(unbiased[1:6], biased[1:6], rtol=0.05)
    t = S.tau_int(x, 1.0)
    window = next(m for m in range(1, n) if m >= 6.0 * (0.5 + unbiased[1: m + 1].sum()))
    assert t.window == window and t.window_found
    assert t.tau_sokal_steps == pytest.approx(0.5 + unbiased[1: window + 1].sum())
    assert t.tau_sokal_steps != pytest.approx(S.tau_int(x, 1.0, estimator="biased").tau_sokal_steps)


def test_tau_int_white_noise_and_constant_series():
    w = np.random.default_rng(2).normal(size=100_000)
    assert S.tau_int(w, 1.0).tau_steps == pytest.approx(1.0, abs=0.05)
    assert S.tau_int(w, 1.0, convention="sokal").tau_steps == pytest.approx(0.5, abs=0.025)
    with pytest.raises(ValueError):
        S.tau_int(np.ones(100), 1.0)


def test_tau_int_pair_takes_each_series_alone():
    a = ar1(0.9, 50_000, 1)
    b = np.random.default_rng(3).normal(size=50_000)
    p = S.tau_int_pair(a, b, 0.25)
    assert p.which == "max" and p.tau_s == max(p.neural.tau_s, p.body.tau_s) == p.neural.tau_s
    assert S.tau_int_pair(a, b, 0.25, which="body").tau_s == p.body.tau_s
    assert "which (OPEN-31)" in p.draft


def test_s_min_and_k_shift():
    assert S.s_min(0.09, 0.01) == 27  # 3 * 0.09 / 0.01 is 26.999999999999996 in floats
    assert S.s_min(0.0905, 0.01) == 28
    assert S.s_min(-0.5, 0.01) == 1
    assert S.k_shift(1000, 27) == 947
    assert S.k_shift(40, 27) == 0
    shifts = S.admissible_shifts(1000, 27)
    assert shifts.size == S.k_shift(1000, 27)
    assert shifts[0] == 27 and shifts[-1] == 1000 - 27
    spaced = S.admissible_shifts(1000, 27, spacing="tau_int", tau_steps=9.0)
    assert np.all(np.diff(spaced) == 9) and spaced[0] == 27 and spaced[-1] <= 973
    assert spaced.size == (1000 - 2 * 27) // 9 + 1
    assert S.k_swap(5) == 4 and S.k_stride(4) == 23


def test_k_shift_on_an_ar1_record_matches_the_rule():
    dt = 0.25  # the 4 Hz grid of the slow regime
    x = ar1(0.8, 20_000, 9)
    t = S.tau_int(x, dt)
    sm = S.s_min(t.tau_s, dt)
    assert sm == math.ceil(3 * t.tau_steps - 1e-9)
    assert S.k_shift(x.size, sm) == x.size - 2 * sm + 1


# --- draft flags ---------------------------------------------------------------------------------------------------

def public_functions():
    for mod in (S, D):
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ == mod.__name__ and not name.startswith("_"):
                yield mod.__name__, name, obj


def test_every_draft_default_names_an_open_item():
    seen = 0
    for _, _, fn in public_functions():
        for arg, dft in S.draft_parameters(fn).items():
            assert re.fullmatch(r"OPEN-\d\d", dft.open_id), (fn.__name__, arg)
            assert dft.note, (fn.__name__, arg)
            seen += 1
    assert seen >= 30


def test_no_bare_float_default_in_the_rule_functions():
    """Every numeric default of a public function is a Draft, except tolerances and loop bounds listed here."""
    allowed = {("phipson_smyth_p", "atol")}
    for _, name, fn in public_functions():
        for arg, prm in inspect.signature(fn).parameters.items():
            if isinstance(prm.default, float) or (isinstance(prm.default, int) and not isinstance(prm.default, bool)):
                assert (name, arg) in allowed, (name, arg, prm.default)
