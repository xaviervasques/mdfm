"""Tests of the motion and shape priors (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15):
causality and the 8-patch span (changing z_p leaves the prior of patches up to p and beyond p + 8
unchanged, bit for bit); log_prob equals the brute-force product of the Gaussians of each patch, each
computed from the truncated past alone; sample then log_prob is finite and padding stays zero; kl equals
the analytic Gaussian KL with free bits as a floor per dimension; condition dropout only in training;
the empty session slot adds nothing; the shape prior equals reports/body_store/priors.json and scipy's
multivariate normal, its KL the closed form, bit for bit the same under a bf16 autocast; N(0, I) at
initialization; the pilot prior holds about 3.2M parameters. Runs on the CPU in float64 where a tolerance
is tight.
"""
import json
import math

import numpy as np
import pytest
import torch

from mdfm.model.encoder import n_parameters
from mdfm.model.prior import (COND_DROP, PRIORS_PATH, SPAN, TASK_UNKNOWN, TASKS, Cond, MotionPrior, ShapePrior,
                              gaussian_kl, patch_mask)

F64 = torch.float64


def _randomized(prior, seed=0, std=0.08):
    """The prior with every weight drawn at random (the output layer is zero at initialization; LayerNorm
    gains around 1), float64, eval mode."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in prior.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
        for mod in prior.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.add_(1.0)
    return prior.double().eval().requires_grad_(False)


def _cond(W, seed=1):
    g = torch.Generator().manual_seed(seed)
    return Cond(torch.randint(0, 3, (W,), generator=g), torch.randint(0, len(TASKS), (W,), generator=g),
                torch.randn(W, 16, generator=g, dtype=F64))


@pytest.fixture(scope="module")
def prior():
    return _randomized(MotionPrior(d=32, layers=2, heads=4))


def test_prior_is_causal_within_its_span(prior):
    W, P = 3, 30
    g = torch.Generator().manual_seed(2)
    z = torch.randn(W, P, 32, generator=g, dtype=F64)
    c = _cond(W)
    mu, ls = prior.params(z, c)
    for p0 in (0, 7, 13, 21):
        z2 = z.clone()
        z2[:, p0] += torch.randn(W, 32, generator=g, dtype=F64)
        mu2, ls2 = prior.params(z2, c)
        same = [p for p in range(P) if p <= p0 or p > p0 + SPAN]
        moved = [p for p in range(p0 + 1, min(P, p0 + SPAN + 1))]
        assert torch.equal(mu[:, same], mu2[:, same]) and torch.equal(ls[:, same], ls2[:, same])
        for p in moved:
            assert float((mu[:, p] - mu2[:, p]).abs().max()) > 1e-8, (p0, p)


def test_padding_patches_are_never_read(prior):
    W, P = 2, 20
    g = torch.Generator().manual_seed(3)
    z = torch.randn(W, P, 32, generator=g, dtype=F64)
    n = torch.tensor([12, 20])
    m = patch_mask(n, P)
    c = _cond(W)
    z2 = z.clone()
    z2[0, 12:] = 100.0 * torch.randn(8, 32, generator=g, dtype=F64)
    assert torch.equal(prior.log_prob(z, c, m), prior.log_prob(z2, c, m))
    mu, _ = prior.params(z, c, m)
    mu2, _ = prior.params(z2, c, m)
    assert torch.equal(mu[0, :12], mu2[0, :12])


def test_log_prob_equals_the_brute_force_gaussian_product(prior):
    W, P = 2, 14
    g = torch.Generator().manual_seed(4)
    z = torch.randn(W, P, 32, generator=g, dtype=F64)
    c = _cond(W)
    total = torch.zeros(W, dtype=F64)
    for p in range(P):
        mu, ls = prior.params(z[:, :p + 1], c)                # the prediction of patch p from its past alone
        d = torch.distributions.Normal(mu[:, p], torch.exp(ls[:, p]))
        total = total + d.log_prob(z[:, p]).sum(-1)
    assert torch.allclose(prior.log_prob(z, c), total, rtol=0, atol=1e-10)


def test_sample_then_log_prob_is_finite(prior):
    W = 3
    c = _cond(W)
    n = torch.tensor([5, 9, 1])
    g = torch.Generator().manual_seed(5)
    x = prior.sample(c, n, n=4, generator=g)
    assert tuple(x.shape) == (4, W, 9, 32)
    m = patch_mask(n, 9)
    assert torch.equal(x[:, ~m], torch.zeros_like(x[:, ~m]))
    for i in range(4):
        lp = prior.log_prob(x[i], c, m)
        assert torch.isfinite(lp).all()


def test_kl_is_the_analytic_gaussian_kl_with_free_bits(prior):
    W, P = 2, 10
    g = torch.Generator().manual_seed(6)
    z = torch.randn(W, P, 32, generator=g, dtype=F64)
    mu_q = torch.randn(W, P, 32, generator=g, dtype=F64)
    lv_q = 0.3 * torch.randn(W, P, 32, generator=g, dtype=F64)
    c = _cond(W)
    mu_p, ls_p = prior.params(z, c)
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu_q, torch.exp(0.5 * lv_q)),
                                            torch.distributions.Normal(mu_p, torch.exp(ls_p)))
    assert torch.allclose(gaussian_kl(mu_q, lv_q, mu_p, ls_p), ref, atol=1e-12)
    assert torch.allclose(prior.kl(mu_q, lv_q, z, c, free_bits=0.0), ref.sum((1, 2)), atol=1e-9)
    # q equal to the prior: zero KL, and exactly the free-bits floor with free bits
    kl0 = prior.kl(mu_p, 2 * ls_p, z, c, free_bits=0.0)
    assert float(kl0.abs().max()) < 1e-9
    m = patch_mask(torch.tensor([10, 6]), P)
    klf = prior.kl(mu_p, 2 * ls_p, z, c, free_bits=0.02, mask=m)
    assert torch.allclose(klf, torch.tensor([0.02 * 10 * 32, 0.02 * 6 * 32], dtype=F64), atol=1e-9)


def test_condition_dropout_only_in_training():
    prior = _randomized(MotionPrior(d=32, layers=1, heads=4))
    W, P = 4, 6
    z = torch.randn(W, P, 32, dtype=F64, generator=torch.Generator().manual_seed(7))
    a, b = _cond(W, 8), _cond(W, 9)
    mu_a, _ = prior.params(z, a)
    mu_b, _ = prior.params(z, b)
    assert float((mu_a - mu_b).abs().max()) > 1e-6            # eval: the condition is read
    prior.train()
    prior.drop = 1.0
    ta, _ = prior.params(z, a, generator=torch.Generator().manual_seed(0))
    tb, _ = prior.params(z, b, generator=torch.Generator().manual_seed(0))
    unknown = Cond(torch.full((W,), 2), torch.full((W,), TASK_UNKNOWN), torch.zeros(W, 16, dtype=F64))
    tu, _ = prior.params(z, unknown, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(ta, tb, atol=1e-12) and torch.allclose(ta, tu, atol=1e-12)
    ev, _ = prior.params(z, a, dropout=False)                 # a density evaluation never drops
    assert torch.allclose(ev, mu_a, atol=1e-12)
    assert COND_DROP == 0.2 and TASKS[TASK_UNKNOWN] == "unknown"


def test_the_empty_session_slot_adds_nothing(prior):
    W, P = 2, 5
    z = torch.randn(W, P, 32, dtype=F64, generator=torch.Generator().manual_seed(10))
    c = _cond(W)
    with_zeros = Cond(c.sex, c.task, c.beta, torch.zeros(W, 16, dtype=F64))
    assert torch.equal(prior.params(z, c)[0], prior.params(z, with_zeros)[0])


def test_initialization_is_standard_normal():
    prior = MotionPrior(d=32, layers=1, heads=4).eval().requires_grad_(False)
    z = torch.randn(2, 7, 32)
    mu, ls = prior.params(z, Cond(torch.tensor([0, 1]), torch.tensor([0, 3]), torch.zeros(2, 16)))
    assert float(mu.abs().max()) == 0.0
    assert float(ls.abs().max()) < 1e-6


def test_pilot_prior_holds_about_3_2m_parameters():
    n = n_parameters(MotionPrior())
    assert 3_100_000 <= n <= 3_300_000, n


def test_shape_prior_equals_the_report_and_scipy():
    from scipy.stats import multivariate_normal
    sp = ShapePrior()
    with open(PRIORS_PATH, encoding="utf-8") as f:
        rep = json.load(f)["shape_prior"]
    g = torch.Generator().manual_seed(11)
    beta = torch.randn(6, 16, generator=g)
    gender = torch.tensor([0, 1, 2, 2, 1, 0])
    lp = sp.log_prob(beta, gender)
    for i, name in enumerate(("female", "male", "neutral", "neutral", "male", "female")):
        m, C = np.array(rep[name]["mean"]), np.array(rep[name]["cov"])
        assert np.allclose(sp.mean[gender[i]].numpy(), m, atol=1e-6)
        ref = multivariate_normal(m, C).logpdf(beta[i].double().numpy())
        assert abs(float(lp[i]) - ref) < 1e-3
    # KL of a diagonal Gaussian against the prior, closed form in numpy
    mu = torch.randn(3, 16, generator=g)
    lv = 0.2 * torch.randn(3, 16, generator=g)
    kl = sp.kl(mu, lv, torch.tensor([0, 1, 2]))
    for i, name in enumerate(("female", "male", "neutral")):
        m, C = np.array(rep[name]["mean"]), np.array(rep[name]["cov"])
        Ci = np.linalg.inv(C)
        d = m - mu[i].double().numpy()
        v = np.exp(lv[i].double().numpy())
        ref = 0.5 * (np.sum(np.diag(Ci) * v) + d @ Ci @ d - 16 + np.linalg.slogdet(C)[1] - np.log(v).sum())
        assert abs(float(kl[i]) - ref) < 1e-3
    draws = sp.sample(torch.tensor([1]), n=20000, generator=torch.Generator().manual_seed(12))
    assert float((draws.mean(0)[0] - sp.mean[1]).abs().max()) < 0.05
    assert torch.equal(sp.logvar0(2), torch.log(torch.diagonal(sp.cov[2])))
    with pytest.raises(ValueError):
        sp.log_prob(beta[:1], torch.tensor([3]))


def test_shape_prior_is_float32_under_bf16_autocast():
    sp = ShapePrior()
    g = torch.Generator().manual_seed(13)
    mu, lv = 2.0 * torch.randn(5, 16, generator=g), 0.3 * torch.randn(5, 16, generator=g)
    gender = torch.tensor([0, 1, 2, 1, 0])
    ref_kl, ref_lp = sp.kl(mu, lv, gender), sp.log_prob(mu, gender)
    ref_s = sp.sample(gender, n=3, generator=torch.Generator().manual_seed(14))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        kl, lp = sp.kl(mu, lv, gender), sp.log_prob(mu, gender)
        s = sp.sample(gender, n=3, generator=torch.Generator().manual_seed(14))
    assert kl.dtype == lp.dtype == s.dtype == torch.float32
    assert torch.equal(kl, ref_kl) and torch.equal(lp, ref_lp) and torch.equal(s, ref_s)


def test_shape_prior_buffers_are_statistics_in_the_state_dict():
    sd = ShapePrior().state_dict()
    assert sorted(sd) == ["chol", "cov", "mean"]
    assert tuple(sd["mean"].shape) == (3, 16) and tuple(sd["cov"].shape) == (3, 16, 16)
    assert math.isfinite(float(sd["chol"].sum()))
