"""Tests of the posterior heads (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15): pooling of
a lone window is the identity and pooling two windows of one person is the product of their Gaussians;
gradients reach mu and logvar through the reparameterized sample; log-variances stay inside [-10, 4] and
keep a gradient everywhere; the heads return float32 under bf16 autocast; the shape head starts at the
pooled shape prior.
"""
import numpy as np
import torch

from mdfm.model.posterior import (LOGVAR_MAX, LOGVAR_MIN, LatentHead, ShapeHead, group_ids, sample, soft_clamp,
                                  soft_clamp_inverse, subject_pool)
from mdfm.model.prior import ShapePrior


def test_pooling_a_lone_window_is_the_identity():
    g = torch.Generator().manual_seed(0)
    mu, lv = torch.randn(4, 16, generator=g, dtype=torch.float64), torch.randn(4, 16, generator=g, dtype=torch.float64)
    pm, pl = subject_pool(mu, lv, torch.tensor([3, 1, 7, 5]))
    assert torch.allclose(pm, mu, atol=1e-12) and torch.allclose(pl, lv, atol=1e-12)
    pm, pl = subject_pool(mu, lv, np.array(["a", "b", "c", "d"]))
    assert torch.allclose(pm, mu, atol=1e-12) and torch.allclose(pl, lv, atol=1e-12)


def test_pooling_one_person_is_the_product_of_gaussians():
    g = torch.Generator().manual_seed(1)
    mu, lv = torch.randn(3, 16, generator=g, dtype=torch.float64), torch.randn(3, 16, generator=g, dtype=torch.float64)
    pm, pl = subject_pool(mu, lv, ["p", "q", "p"])
    p0, p2 = torch.exp(-lv[0]), torch.exp(-lv[2])
    assert torch.allclose(pm[0], (mu[0] * p0 + mu[2] * p2) / (p0 + p2), atol=1e-12)
    assert torch.allclose(pl[0], -torch.log(p0 + p2), atol=1e-12)
    assert torch.equal(pm[0], pm[2]) and torch.allclose(pm[1], mu[1], atol=1e-12)
    assert group_ids(["p", "q", "p"]).tolist() == [0, 1, 0]


def test_gradients_reach_mu_and_logvar():
    mu = torch.zeros(2, 5, 32, requires_grad=True)
    lv = torch.zeros(2, 5, 32, requires_grad=True)
    z = sample(mu, lv, n=3, generator=torch.Generator().manual_seed(2))
    assert tuple(z.shape) == (3, 2, 5, 32)
    (z ** 2).sum().backward()
    assert float(mu.grad.abs().sum()) > 0 and float(lv.grad.abs().sum()) > 0
    pm, pl = subject_pool(mu[:, 0], lv[:, 0], torch.tensor([0, 0]))
    mu.grad = lv.grad = None
    (pm.sum() + pl.sum()).backward()
    assert float(mu.grad.abs().sum()) > 0 and float(lv.grad.abs().sum()) > 0


def test_logvars_stay_bounded_with_a_gradient():
    x = torch.linspace(-1e4, 1e4, 20001, dtype=torch.float64, requires_grad=True)
    y = soft_clamp(x)
    assert float(y.detach().min()) >= LOGVAR_MIN and float(y.detach().max()) <= LOGVAR_MAX
    y.sum().backward()
    assert bool((x.grad >= 0).all()) and float(x.grad[(x.detach() > -8) & (x.detach() < 2)].min()) > 0.8
    inside = torch.linspace(-6.0, 0.0, 61, dtype=torch.float64)
    assert float((soft_clamp(inside) - inside).abs().max()) < 0.02
    y = torch.linspace(-9.9, 3.9, 139, dtype=torch.float64)
    assert float((soft_clamp(soft_clamp_inverse(y)) - y).abs().max()) < 1e-12
    # float32 raw values near 1e6: the two softplus terms would cancel above the upper bound
    g = torch.Generator().manual_seed(0)
    big = soft_clamp(1e6 * torch.randn(200000, generator=g))
    assert float(big.min()) >= LOGVAR_MIN and float(big.max()) <= LOGVAR_MAX
    torch.manual_seed(0)
    head = LatentHead(d=16)
    with torch.no_grad():
        head.proj.weight.mul_(1e6)
    with torch.no_grad():
        _, lv = head(torch.randn(3, 4, 16, generator=g))
    assert float(lv.min()) >= LOGVAR_MIN and float(lv.max()) <= LOGVAR_MAX


def test_heads_return_float32_under_bf16_autocast():
    lat, shp = LatentHead(d=16), ShapeHead(d=16)
    h = torch.randn(2, 5, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mu, lv = lat(h.to(torch.bfloat16))
        mb, lb = shp(h[:, 0].to(torch.bfloat16))
    assert {mu.dtype, lv.dtype, mb.dtype, lb.dtype} == {torch.float32}
    assert tuple(mu.shape) == (2, 5, 32) and tuple(mb.shape) == (2, 16)


def test_shape_head_starts_at_the_pooled_prior():
    sp = ShapePrior()
    head = ShapeHead(d=16, mu0=sp.mean[2], logvar0=sp.logvar0(2)).requires_grad_(False)
    with torch.no_grad():
        head.proj.weight.zero_()
    mb, lb = head(torch.randn(3, 16))
    assert torch.allclose(mb, sp.mean[2].expand(3, 16))
    assert float((lb - sp.logvar0(2)).abs().max()) < 1e-5
    lat = LatentHead(d=16).requires_grad_(False)
    with torch.no_grad():
        lat.proj.weight.zero_()
    mu, lv = lat(torch.randn(2, 3, 16))
    assert float(mu.abs().max()) == 0.0 and float(lv.abs().max()) < 1e-6
