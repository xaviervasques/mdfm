"""Split conformal calibration and the predictive scores of the gates (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 2.7 and 6).

Split conformal (Vovk, Gammerman and Shafer 2005; Lei et al. 2018): with n exchangeable calibration
scores, the finite-sample quantile q is the k-th smallest score, k = ceil((n + 1) level); a new
exchangeable score falls at or below q with probability at least level (and below level + 1 / (n + 1)
without ties). k > n gives q = inf (too few scores for the level).

Ellipsoids. A predictive given as m draws (m, d) is moment-matched: mean and covariance with ddof 1.
The score of a truth y is its Mahalanobis radius r = sqrt((y - mean)^T cov^-1 (y - mean)); the
chi-square ellipsoid at `level` holds r <= sqrt(chi2_d(level)). ellipsoid_factor returns the c that
scales that ellipsoid so that it covers `level` of the calibration truths: c = fit(r) /
sqrt(chi2_d(level)). A covariance that is non-finite, whose smallest eigenvalue is at most
DEGENERATE_TOL times its largest, or whose largest is at most DEGENERATE_ABS (a zero covariance
included) is degenerate: its radius is inf, so it never covers, and callers count it (a degenerate
predictive is reported as such, never as covered).
Band width: the per-axis central `level` widths of the scaled Gaussian, 2 z c sqrt(cov_ii) with z the
(1 + level) / 2 normal quantile, and their geometric mean over the d axes.

Energy score (Gneiting and Raftery 2007, beta = 1): ES(P, y) = E|X - y| - E|X - X'| / 2, estimated
from m draws by the V-statistic mean_i |x_i - y| - sum_ij |x_i - x_j| / (2 m^2), which is 0 when every
draw equals y. Units are those of the draws (metres in the G1 harness). All arithmetic in float64
numpy.
"""
import math

import numpy as np
from scipy import special, stats

LEVEL = 0.90
DEGENERATE_TOL = 1e-9          # smallest over largest covariance eigenvalue at or below it: degenerate
DEGENERATE_ABS = 1e-24         # largest eigenvalue at or below it (squared unit of the draws): degenerate
_K_EPS = 1e-9                  # ceil((n + 1) level) computed on (n + 1) level - _K_EPS against rounding


def _check_level(level):
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    return float(level)


def rank(n, level=LEVEL):
    """k = ceil((n + 1) level), the rank of the split conformal quantile among n scores."""
    return int(math.ceil((n + 1) * _check_level(level) - _K_EPS))


def fit(scores, level=LEVEL):
    """The split conformal quantile of scores (n,) at level: the k-th smallest score, k = rank(n,
    level); inf when k > n. inf scores (degenerate predictives) are allowed; NaN raises ValueError."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("no calibration score")
    if np.isnan(s).any():
        raise ValueError("NaN calibration score")
    k = rank(s.size, level)
    if k > s.size:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def coverage(scores, q):
    """Share of scores at or below q."""
    s = np.asarray(scores, dtype=np.float64)
    return float(np.mean(s <= q))


def chi2_radius(dof=3, level=LEVEL):
    """sqrt of the chi-square quantile with dof degrees of freedom at level: the Mahalanobis radius of
    the Gaussian ellipsoid that holds level of the mass."""
    return float(math.sqrt(stats.chi2.ppf(_check_level(level), dof)))


def moment_match(draws, ddof=1):
    """draws (..., m, d) -> mean (..., d), covariance (..., d, d) with ddof (1: unbiased)."""
    x = np.asarray(draws, dtype=np.float64)
    m = x.shape[-2]
    if m - ddof < 1:
        raise ValueError(f"{m} draws cannot give a covariance with ddof {ddof}")
    mean = x.mean(axis=-2)
    xc = x - mean[..., None, :]
    cov = np.einsum("...ni,...nj->...ij", xc, xc) / (m - ddof)
    return mean, cov


def degenerate(cov, tol=DEGENERATE_TOL):
    """cov (..., d, d) -> (...) bool: non-finite, smallest eigenvalue at most tol times the largest, or
    largest eigenvalue at most DEGENERATE_ABS (identical draws leave rounding residue near 1e-33)."""
    c = np.asarray(cov, dtype=np.float64)
    finite = np.isfinite(c).all(axis=(-1, -2))
    w = np.linalg.eigvalsh(np.where(finite[..., None, None], c, 0.0))
    return ~finite | (w[..., 0] <= tol * np.maximum(w[..., -1], 0.0)) | (w[..., -1] <= DEGENERATE_ABS)


def mahalanobis(y, mean, cov, tol=DEGENERATE_TOL):
    """Radius (...) of truths y (..., d) under Gaussians (mean (..., d), cov (..., d, d)); all three
    broadcast (one shared predictive against many truths included). Returns (radius, degenerate): the
    radius is inf where the covariance is degenerate."""
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    bad = degenerate(c, tol)
    w, V = np.linalg.eigh(np.where(bad[..., None, None], np.eye(c.shape[-1]), c))
    z = np.einsum("...i,...ij->...j", y - mean, V)
    r = np.sqrt(np.sum(z * z / w, axis=-1))
    bad = np.broadcast_to(bad, r.shape)
    return np.where(bad, np.inf, r), bad.copy()


def ellipsoid_factor(radius, dof=3, level=LEVEL):
    """c such that the chi-square dof ellipsoid at level, scaled by c, covers level of the calibration
    truths whose radii (n,) are given: fit(radius, level) / chi2_radius(dof, level)."""
    return fit(radius, level) / chi2_radius(dof, level)


def covered(radius, c, dof=3, level=LEVEL):
    """(...) bool: radius at or below c x chi2_radius(dof, level); an inf radius (a degenerate
    predictive) is never covered, not even by c = inf (fit on too few scores)."""
    r = np.asarray(radius, dtype=np.float64)
    return np.isfinite(r) & (r <= c * chi2_radius(dof, level))


def axis_widths(cov, c=1.0, level=LEVEL):
    """Per-axis central level widths (..., d) of the Gaussian with covariance c^2 cov (..., d, d):
    2 z c sqrt(cov_ii), z the (1 + level) / 2 standard normal quantile."""
    z = float(stats.norm.ppf((1 + _check_level(level)) / 2))
    var = np.diagonal(np.asarray(cov, dtype=np.float64), axis1=-2, axis2=-1)
    return 2.0 * z * c * np.sqrt(np.maximum(var, 0.0))


def band_width(cov, c=1.0, level=LEVEL):
    """Geometric mean (...) of axis_widths over the d axes (0 when an axis has zero width)."""
    w = axis_widths(cov, c, level)
    with np.errstate(divide="ignore"):
        return np.exp(np.mean(np.log(w), axis=-1))


def energy_score(draws, y):
    """Energy score (...) of the draws (..., m, d) at truths y (..., d), V-statistic (module
    docstring). The draws broadcast against y: one shared draw set (m, d) against truths (N, d) gives
    (N,), with the pairwise term computed once."""
    x = np.asarray(draws, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d1 = np.linalg.norm(x - y[..., None, :], axis=-1).mean(axis=-1)
    pw = np.linalg.norm(x[..., :, None, :] - x[..., None, :, :], axis=-1).mean(axis=(-1, -2))
    return d1 - 0.5 * pw


def energy_score_gaussian(mean, sigma, y, dim=3):
    """Closed-form energy score (...) of the isotropic Gaussian N(mean, sigma^2 I_dim) at y, for dim 1
    (the CRPS, Gneiting and Raftery 2007) or 3: with lam = |y - mean| / sigma, E|X - y| = sigma
    (sqrt(2 / pi) exp(-lam^2 / 2) + (lam + 1 / lam) erf(lam / sqrt 2)) (the mean of a noncentral chi
    with 3 degrees of freedom; 2 sqrt(2 / pi) sigma at lam = 0) and E|X - X'| = 4 sigma / sqrt(pi)."""
    mean = np.asarray(mean, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if dim == 1:
        z = (y - mean) / s
        return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
    if dim != 3:
        raise ValueError(f"closed form for dim 1 or 3, got {dim}")
    lam = np.linalg.norm(y - mean, axis=-1) / s
    safe = np.where(lam > 1e-8, lam, 1.0)
    far = math.sqrt(2 / math.pi) * np.exp(-lam * lam / 2) + (safe + 1 / safe) * special.erf(safe / math.sqrt(2))
    near = math.sqrt(2 / math.pi) * (2 + lam * lam / 3)
    e1 = s * np.where(lam > 1e-8, far, near)
    return e1 - 2 * s / math.sqrt(math.pi)
