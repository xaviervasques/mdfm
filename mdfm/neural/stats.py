"""Statistics of the neural rule file (docs/rules/NEURAL_RULES_DRAFT.md, DRAFT).

Status: DRAFT. The rule file is not hashed, and every default below that settles one of its OPEN items is a Draft
value: the argument is explicit, its default names the OPEN item it fills, and every result lists the draft defaults
it used (field ``draft``). A caller that passes its own value removes that entry.

Contents and rules:

- phipson_smyth_p, draw_null_indices: null-based p-values, p = (1 + b) / (k + 1) (NR-1.7), the 20-null floor of
  NR-1.8, nulls drawn without replacement when they are not enumerated (draw count OPEN-33).
- holm: Holm step-down and the adjusted p-values in Wright's form (NR-7.3; alpha OPEN-01).
- benjamini_hochberg, benjamini_yekutieli, fdr: the within-family false discovery rate of NR-8.2 (q and the choice
  of BH or BY: OPEN-58).
- n_cluster_resamples, enumerate_cluster_resamples, cluster_bootstrap: the subject-clustered bootstrap of NR-1.5 and
  NR-1.6, all C(2n - 1, n) weighted resamples enumerated when they are not more than the replicate count (35 for 4
  patients), drawn otherwise (replicates and interval method OPEN-04, level OPEN-01).
- autocorrelation, tau_int, tau_int_pair, s_min, k_shift, admissible_shifts, k_swap, k_stride: the circular-shift
  null of NR-5.6 (a) and the null counts of NR-5.7, tau_int in both conventions of OPEN-31.

No function here reads data; every input is an array the caller gives.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# Draft defaults
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Draft:
    """A default that fills an OPEN item of the neural rule file; a draft until the owner settles that item.

    ``value`` is the value used, ``open_id`` the OPEN item (OPEN-nn), ``note`` why this value was proposed.
    """
    value: Any
    open_id: str
    note: str = ""

    def __repr__(self):
        return f"Draft({self.value!r}, {self.open_id})"


def resolve(arg, name: str, used: list | None = None):
    """The value of an argument; a Draft default is unwrapped and recorded in ``used`` as 'name (OPEN-nn)'."""
    if isinstance(arg, Draft):
        if used is not None:
            tag = f"{name} ({arg.open_id})"
            if tag not in used:
                used.append(tag)
        return arg.value
    return arg


ALPHA = Draft(0.05, "OPEN-01", "family-wise alpha of the Holm family; the design reports 95 percent intervals only")
LEVEL = Draft(0.95, "OPEN-01", "interval level; the only interval of the design is a 95 percent one (D9 G14)")
FDR_Q = Draft(0.05, "OPEN-58", "false discovery level within a family")
FDR_METHOD = Draft("bh", "OPEN-58", "Benjamini-Hochberg; 'by' is the Benjamini-Yekutieli alternative")
BOOT_REPLICATES = Draft(2000, "OPEN-04", "replicate count; enumeration takes over when C(2n - 1, n) is not larger")
BOOT_METHOD = Draft("percentile", "OPEN-04", "weighted percentile interval, the only method implemented")
NULL_DRAWS = Draft(999, "OPEN-33", "draw count; 1 / (k + 1) = 0.001 lies under alpha / m = 0.05 / 6 of NR-7.5")
SHIFT_SPACING = Draft("one_sample", "OPEN-33",
                      "distinct shifts one sample apart (the k_shift formula of NR-5.7); 'tau_int' spaces them")
TAU_CONVENTION = Draft("draft", "OPEN-31", "dt * (1 + 2 sum rho), the draft definition of NR-5.6 (a)")
TAU_C = Draft(6.0, "OPEN-31", "Sokal's window constant, stated in his convention (about 4 for a pure exponential, "
                              "at least 6 otherwise); the draft convention uses c / 2, the same window")
TAU_SERIES = Draft("max", "OPEN-31", "tau_int of the larger of the neural and body series")
ACF_ESTIMATOR = Draft("unbiased", "OPEN-31", "sample autocovariance divided by n - j; the register does not name the "
                                             "estimator, a proposed addition to OPEN-31")

MIN_ADMISSIBLE_NULLS = 20  # NR-1.8, fixed by the design (A2.5), not OPEN


def _float_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} holds a non-finite value")
    return a


# ---------------------------------------------------------------------------------------------------------------
# Null-based p-values (NR-1.7, NR-1.8)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NullP:
    """p = (1 + b) / (k + 1) with its counts. ``enters_pass_decision`` is False when k < 20 (NR-1.8)."""
    p: float
    b: int
    k: int
    p_min: float
    sided: str
    enters_pass_decision: bool


def phipson_smyth_p(observed: float, null_stats: Sequence[float], *, sided: str, atol: float = 0.0) -> NullP:
    """Null-based p-value of NR-1.7 (Phipson and Smyth 2010).

    ``null_stats`` holds the statistic of each of the k distinct admissible non-identity nulls evaluated (the caller
    leaves the identity out). b counts the nulls at least as extreme as ``observed``: null >= observed - atol when
    ``sided`` is 'greater' (a statistic whose rule names larger values as stronger evidence), |null| >= |observed| -
    atol when ``sided`` is 'two-sided' (a signed statistic with no pre-registered sign). NR-1.7 asks every test to
    state its sidedness, so the argument has no default; a statistic where smaller is stronger is negated by the
    caller. ``atol`` (0 by default) absorbs float noise in ties.
    """
    if sided not in ("greater", "two-sided"):
        raise ValueError("sided must be 'greater' or 'two-sided' (NR-1.7)")
    obs = float(observed)
    if not math.isfinite(obs):
        raise ValueError("observed statistic is not finite")
    null = _float_array(null_stats, "null_stats").ravel()
    k = int(null.size)
    if k == 0:
        raise ValueError("no admissible null: k = 0")
    if sided == "greater":
        b = int(np.sum(null >= obs - atol))
    else:
        b = int(np.sum(np.abs(null) >= abs(obs) - atol))
    return NullP(p=(1 + b) / (k + 1), b=b, k=k, p_min=1.0 / (k + 1), sided=sided,
                 enters_pass_decision=k >= MIN_ADMISSIBLE_NULLS)


@dataclass(frozen=True)
class NullDraw:
    """Indices into the admissible null set: all of them (enumerated) or a draw without replacement."""
    indices: np.ndarray
    enumerated: bool
    k_admissible: int
    k: int
    draft: tuple = ()


def draw_null_indices(k_admissible: int, *, seed: int | None = None, n_draw=NULL_DRAWS) -> NullDraw:
    """Which admissible nulls to evaluate (NR-1.7, NR-5.7).

    When k_admissible is at most the draw count every null is evaluated; otherwise ``n_draw`` distinct nulls are drawn
    without replacement (a seed is then required) and k is the number drawn.
    """
    used: list = []
    n = int(resolve(n_draw, "n_draw", used))
    if k_admissible < 0 or n < 1:
        raise ValueError("k_admissible must be >= 0 and n_draw >= 1")
    if k_admissible <= n:
        return NullDraw(np.arange(k_admissible), True, int(k_admissible), int(k_admissible), tuple(used))
    if seed is None:
        raise ValueError("drawing nulls needs an explicit seed")
    idx = np.sort(np.random.default_rng(seed).choice(k_admissible, size=n, replace=False))
    return NullDraw(idx, False, int(k_admissible), n, tuple(used))


# ---------------------------------------------------------------------------------------------------------------
# Multiplicity: Holm (NR-7.3), Benjamini-Hochberg and Benjamini-Yekutieli (NR-8.2)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Correction:
    """Raw and adjusted p-values and the rejections, in the order of the input."""
    method: str
    level: float
    m: int
    p_raw: np.ndarray
    p_adj: np.ndarray
    reject: np.ndarray
    draft: tuple = ()


def _check_p(pvalues) -> np.ndarray:
    p = _float_array(pvalues, "pvalues").ravel()
    if p.size == 0:
        raise ValueError("no p-value")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")
    return p


def holm(pvalues, *, alpha=ALPHA) -> Correction:
    """Holm step-down over m p-values (NR-7.3).

    Order p(1) <= ... <= p(m); reject p(1) to p(j - 1), j the smallest i with p(i) > alpha / (m - i + 1), all m when
    no such i exists. Adjusted p-values as Wright 1992 gives them for Holm:
    p_adj(i) = max over l <= i of min(1, (m - l + 1) * p(l)). A primary that is not run enters with p = 1 (NR-7.4,
    the caller's job). Ties keep their input order (stable sort).
    """
    used: list = []
    a = float(resolve(alpha, "alpha", used))
    p = _check_p(pvalues)
    m = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order]
    factors = (m - np.arange(m)).astype(float)  # m - i + 1 for i = 1..m
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, factors * ps))
    above = np.nonzero(ps > a / factors)[0]
    j = int(above[0]) if above.size else m
    reject_sorted = np.arange(m) < j
    p_adj = np.empty(m)
    reject = np.empty(m, dtype=bool)
    p_adj[order] = adj_sorted
    reject[order] = reject_sorted
    return Correction("holm", a, m, p, p_adj, reject, tuple(used))


def _step_up(p: np.ndarray, q: float, c: float, method: str, used: list) -> Correction:
    m = p.size
    order = np.argsort(p, kind="stable")
    ps = p[order]
    i = np.arange(1, m + 1, dtype=float)
    ok = np.nonzero(ps <= i * q / (m * c))[0]
    istar = int(ok[-1]) + 1 if ok.size else 0
    reject_sorted = np.arange(m) < istar
    adj_sorted = np.minimum(1.0, np.minimum.accumulate((m * c * ps / i)[::-1])[::-1])
    p_adj = np.empty(m)
    reject = np.empty(m, dtype=bool)
    p_adj[order] = adj_sorted
    reject[order] = reject_sorted
    return Correction(method, q, m, p, p_adj, reject, tuple(used))


def benjamini_hochberg(pvalues, *, q=FDR_Q) -> Correction:
    """Benjamini-Hochberg step-up (NR-8.2): reject p(1) to p(i*), i* the largest i with p(i) <= i * q / m.
    Adjusted p(i) = min over l >= i of min(1, m * p(l) / l)."""
    used: list = []
    qq = float(resolve(q, "q", used))
    return _step_up(_check_p(pvalues), qq, 1.0, "bh", used)


def benjamini_yekutieli(pvalues, *, q=FDR_Q) -> Correction:
    """Benjamini-Yekutieli step-up for arbitrary dependence (NR-8.2, OPEN-58): the BH rule at q / c(m),
    c(m) = sum over i = 1..m of 1 / i."""
    used: list = []
    qq = float(resolve(q, "q", used))
    p = _check_p(pvalues)
    c = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    return _step_up(p, qq, c, "by", used)


def fdr(pvalues, *, q=FDR_Q, method=FDR_METHOD) -> Correction:
    """The within-family correction of NR-8.2: 'bh' or 'by' (OPEN-58)."""
    used: list = []
    meth = resolve(method, "method", used)
    qq = resolve(q, "q", used)
    if meth == "bh":
        out = benjamini_hochberg(pvalues, q=qq)
    elif meth == "by":
        out = benjamini_yekutieli(pvalues, q=qq)
    else:
        raise ValueError("method must be 'bh' or 'by'")
    return Correction(out.method, out.level, out.m, out.p_raw, out.p_adj, out.reject, tuple(used))


# ---------------------------------------------------------------------------------------------------------------
# Subject-clustered bootstrap (NR-1.5, NR-1.6)
# ---------------------------------------------------------------------------------------------------------------


def n_cluster_resamples(n: int) -> int:
    """Number of distinct cluster resamples of n clusters: C(2n - 1, n) (35 for n = 4)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return math.comb(2 * n - 1, n)


ENUMERATION_LIMIT = 2_000_000


def enumerate_cluster_resamples(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every distinct resample of n clusters drawn n times with replacement, and its probability.

    Returns (counts, weights): counts (C, n) of integers c_i >= 0 summing to n, one row per resample, in
    lexicographic order of the bar positions; weights (C,) = n! / (prod c_i!) / n^n, the multinomial probability
    (NR-1.5), summing to 1.
    """
    total = n_cluster_resamples(n)
    if total > ENUMERATION_LIMIT:
        raise ValueError(f"{total} resamples exceed the enumeration limit {ENUMERATION_LIMIT}")
    rows = []
    for bars in itertools.combinations(range(2 * n - 1), n - 1):
        edges = (-1,) + bars + (2 * n - 1,)
        rows.append([edges[i + 1] - edges[i] - 1 for i in range(n)])
    counts = np.asarray(rows, dtype=np.int64).reshape(total, n)
    fact_n = math.factorial(n)
    nn = n ** n
    weights = np.array([fact_n // math.prod(math.factorial(int(c)) for c in row) / nn for row in counts])
    return counts, weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, prob: float) -> float:
    """Inverted weighted CDF: the smallest value v with sum of the weights of values <= v at least prob."""
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cdf = np.cumsum(w) / np.sum(w)
    i = int(np.searchsorted(cdf, prob - 1e-12, side="left"))
    return float(v[min(i, v.size - 1)])


def cluster_mean(values_per_cluster) -> Callable[[np.ndarray], float]:
    """Statistic for cluster_bootstrap: the mean of per-cluster values weighted by the resample counts."""
    v = _float_array(values_per_cluster, "values_per_cluster").ravel()

    def stat(counts: np.ndarray) -> float:
        return float(np.dot(counts, v) / np.sum(counts))
    return stat


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate (all clusters once), interval, and the resampled distribution with its weights."""
    estimate: float
    lo: float
    hi: float
    level: float
    method: str
    enumerated: bool
    n_clusters: int
    n_resamples: int
    values: np.ndarray
    weights: np.ndarray
    undefined_weight: float
    draft: tuple = ()


def cluster_bootstrap(statistic: Callable[[np.ndarray], float], n_clusters: int, *, seed: int | None = None,
                      replicates=BOOT_REPLICATES, level=LEVEL, method=BOOT_METHOD,
                      nan_policy: str = "raise") -> BootstrapResult:
    """Bootstrap that resamples the unit of independence (NR-1.6), enumerated when NR-1.5 says so.

    ``statistic(counts)`` takes the integer vector of how many times each cluster is drawn (length n_clusters, sum
    n_clusters) and returns a float; a statistic over records expands it with np.repeat(cluster_ids, counts). When
    C(2n - 1, n) is not larger than ``replicates`` every resample is enumerated with its multinomial weight;
    otherwise ``replicates`` resamples are drawn with equal weights (a seed is then required). The interval is the
    weighted percentile interval at ``level``. A statistic undefined on a resample (NaN) raises under
    nan_policy='raise'; under 'drop' that resample leaves the distribution and its weight is reported.
    """
    used: list = []
    reps = int(resolve(replicates, "replicates", used))
    lev = float(resolve(level, "level", used))
    meth = resolve(method, "method", used)
    if meth != "percentile":
        raise NotImplementedError("only the percentile interval is implemented (OPEN-04)")
    if not 0 < lev < 1:
        raise ValueError("level must lie in (0, 1)")
    if nan_policy not in ("raise", "drop"):
        raise ValueError("nan_policy must be 'raise' or 'drop'")
    n = int(n_clusters)
    estimate = float(statistic(np.ones(n, dtype=np.int64)))
    total = n_cluster_resamples(n)
    if total <= reps:
        counts, weights = enumerate_cluster_resamples(n)
        enumerated = True
    else:
        if seed is None:
            raise ValueError("a drawn bootstrap needs an explicit seed")
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=reps)
        weights = np.full(reps, 1.0 / reps)
        enumerated = False
    values = np.array([float(statistic(c)) for c in counts])
    bad = ~np.isfinite(values)
    undefined = float(np.sum(weights[bad]))
    if bad.any():
        if nan_policy == "raise":
            raise ValueError(f"statistic undefined on {int(bad.sum())} resamples (weight {undefined:.4g})")
        values, weights = values[~bad], weights[~bad]
    a = 1.0 - lev
    lo = weighted_quantile(values, weights, a / 2)
    hi = weighted_quantile(values, weights, 1 - a / 2)
    return BootstrapResult(estimate, lo, hi, lev, meth, enumerated, n, int(values.size), values, weights, undefined,
                           tuple(used))


# ---------------------------------------------------------------------------------------------------------------
# Integrated autocorrelation time and null counts (NR-5.6 a, NR-5.7, OPEN-31, OPEN-33)
# ---------------------------------------------------------------------------------------------------------------


def autocorrelation(x, *, max_lag: int | None = None, estimator=ACF_ESTIMATOR) -> np.ndarray:
    """Sample autocorrelation rho(0..max_lag) of one series (mean removed), by FFT.

    estimator 'unbiased': autocovariance at lag j divided by n - j; 'biased': divided by n. rho = acov / acov(0).
    """
    est = resolve(estimator, "estimator")
    a = _float_array(x, "x").ravel()
    n = a.size
    if n < 3:
        raise ValueError("series too short")
    if max_lag is None:
        max_lag = n - 1
    max_lag = int(min(max_lag, n - 1))
    d = a - a.mean()
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(d, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    if est == "unbiased":
        acov = acov / (n - np.arange(max_lag + 1))
    elif est == "biased":
        acov = acov / n
    else:
        raise ValueError("estimator must be 'unbiased' or 'biased'")
    if not acov[0] > 0:
        raise ValueError("constant series: autocorrelation undefined")
    return acov / acov[0]


@dataclass(frozen=True)
class TauInt:
    """Integrated autocorrelation time of one series.

    tau_s in seconds and tau_steps in samples in the chosen convention; tau_sokal_steps = 1/2 + sum rho(j) and
    tau_draft_steps = 1 + 2 sum rho(j) over j = 1..window, at the same window, so tau_draft_steps = 2 *
    tau_sokal_steps exactly. ``window`` is the smallest M >= 1 with M >= c_sokal * tau_sokal_steps(M); when no M
    below n satisfies it, window = n - 1 and window_found is False.
    """
    tau_s: float
    tau_steps: float
    tau_sokal_steps: float
    tau_draft_steps: float
    convention: str
    window: int
    window_found: bool
    c_sokal: float
    dt: float
    n: int
    draft: tuple = ()


def tau_int(x, dt: float, *, convention=TAU_CONVENTION, c_sokal=TAU_C, estimator=ACF_ESTIMATOR) -> TauInt:
    """tau_int of NR-5.6 (a) on a single series, in either convention of OPEN-31.

    'draft': tau_int = dt * (1 + 2 * sum_{j=1..M} rho(j)); 'sokal': dt * (1/2 + sum_{j=1..M} rho(j)) (Sokal 1997,
    eq. 2.16), half the draft value. The window M is Sokal's automatic window (his eqs 3.16 and 3.17) with constant
    ``c_sokal`` stated in his convention; under the draft convention the same condition reads M >= (c_sokal / 2) *
    tau_draft(M) / dt, so both conventions use one window and differ by exactly a factor 2. Never call this on a
    cross-correlation (NR-5.6).
    """
    used: list = []
    conv = resolve(convention, "convention", used)
    c = float(resolve(c_sokal, "c_sokal", used))
    est = resolve(estimator, "estimator", used)
    if conv not in ("draft", "sokal"):
        raise ValueError("convention must be 'draft' or 'sokal'")
    if not dt > 0 or not c > 0:
        raise ValueError("dt and c_sokal must be positive")
    a = np.asarray(x, dtype=float).ravel()
    rho = autocorrelation(a, estimator=est)
    n = a.size
    s = np.cumsum(rho[1:])  # s[M - 1] = sum_{j=1..M} rho(j)
    lags = np.arange(1, n)
    ok = np.nonzero(lags >= c * (0.5 + s))[0]
    found = bool(ok.size)
    m = int(lags[ok[0]]) if found else n - 1
    sokal = 0.5 + float(s[m - 1])
    draft = 2.0 * sokal
    steps = draft if conv == "draft" else sokal
    return TauInt(steps * dt, steps, sokal, draft, conv, m, found, c, float(dt), n, tuple(used))


@dataclass(frozen=True)
class TauIntPair:
    """tau_int of the neural and the body series, each alone, and the one the null uses (OPEN-31)."""
    neural: TauInt
    body: TauInt
    which: str
    tau_s: float
    draft: tuple = ()


def tau_int_pair(neural, body, dt: float, *, which=TAU_SERIES, convention=TAU_CONVENTION, c_sokal=TAU_C,
                 estimator=ACF_ESTIMATOR) -> TauIntPair:
    """tau_int of each series alone (never of a cross-correlation) and the chosen one: 'neural', 'body' or 'max'."""
    used: list = []
    w = resolve(which, "which", used)
    conv = resolve(convention, "convention", used)
    c = resolve(c_sokal, "c_sokal", used)
    est = resolve(estimator, "estimator", used)
    tn = tau_int(neural, dt, convention=conv, c_sokal=c, estimator=est)
    tb = tau_int(body, dt, convention=conv, c_sokal=c, estimator=est)
    if w == "neural":
        t = tn.tau_s
    elif w == "body":
        t = tb.tau_s
    elif w == "max":
        t = max(tn.tau_s, tb.tau_s)
    else:
        raise ValueError("which must be 'neural', 'body' or 'max'")
    return TauIntPair(tn, tb, w, t, tuple(used))


def s_min(tau_s: float, dt: float) -> int:
    """s_min = ceil(3 * tau_int / dt) (NR-5.7 a), at least 1 so that the identity shift is never admissible.

    A ratio within 1e-9 of an integer is taken as that integer (3 * 0.09 / 0.01 is 26.999999999999996 in floats).
    """
    if not dt > 0:
        raise ValueError("dt must be positive")
    r = 3.0 * float(tau_s) / float(dt)
    k = round(r)
    v = int(k) if abs(r - k) < 1e-9 else math.ceil(r)
    return max(1, v)


def k_shift(n: int, smin: int) -> int:
    """Distinct admissible circular shifts of NR-5.7 (a): N - 2 * s_min + 1 (0 when negative)."""
    return max(0, int(n) - 2 * int(smin) + 1)


def admissible_shifts(n: int, smin: int, *, spacing=SHIFT_SPACING, tau_steps: float | None = None) -> np.ndarray:
    """Circular shifts s with s_min <= s <= N - s_min (both circular distances to the identity at least s_min).

    spacing 'one_sample' (every such shift; the count is k_shift) or 'tau_int' (shifts at least one tau_int apart,
    starting at s_min, step ceil(tau_steps)): OPEN-33.
    """
    sp = resolve(spacing, "spacing")
    if sp == "one_sample":
        step = 1
    elif sp == "tau_int":
        if tau_steps is None or not tau_steps > 0:
            raise ValueError("spacing 'tau_int' needs tau_steps > 0")
        step = max(1, math.ceil(tau_steps - 1e-9))
    else:
        raise ValueError("spacing must be 'one_sample' or 'tau_int'")
    if int(n) - 2 * int(smin) + 1 <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(int(smin), int(n) - int(smin) + 1, step, dtype=np.int64)


def k_swap(r_s: int) -> int:
    """Cross-record swaps of NR-5.7 (b): R_s - 1 for a subject with R_s admissible records."""
    return max(0, int(r_s) - 1)


def k_stride(n_strides: int) -> int:
    """Stride-cycle shuffles of NR-5.7 (c): n_strides! - 1 (drawn, never enumerated)."""
    return math.factorial(int(n_strides)) - 1


def enters_pass_decision(k: int) -> bool:
    """NR-1.8: a record with fewer than 20 admissible nulls enters no pass decision."""
    return int(k) >= MIN_ADMISSIBLE_NULLS


def draft_parameters(func) -> dict:
    """{argument: Draft} of the defaults of ``func`` that fill an OPEN item."""
    import inspect
    return {name: prm.default for name, prm in inspect.signature(func).parameters.items()
            if isinstance(prm.default, Draft)}


__all__ = [
    "Draft", "resolve", "ALPHA", "LEVEL", "FDR_Q", "FDR_METHOD", "BOOT_REPLICATES", "BOOT_METHOD", "NULL_DRAWS",
    "SHIFT_SPACING", "TAU_CONVENTION", "TAU_C", "TAU_SERIES", "ACF_ESTIMATOR", "MIN_ADMISSIBLE_NULLS",
    "NullP", "phipson_smyth_p", "NullDraw", "draw_null_indices", "Correction", "holm", "benjamini_hochberg",
    "benjamini_yekutieli", "fdr", "n_cluster_resamples", "enumerate_cluster_resamples", "weighted_quantile",
    "cluster_mean", "BootstrapResult", "cluster_bootstrap", "autocorrelation", "TauInt", "tau_int", "TauIntPair",
    "tau_int_pair", "s_min", "k_shift", "admissible_shifts", "k_swap", "k_stride", "enters_pass_decision",
    "draft_parameters",
]
