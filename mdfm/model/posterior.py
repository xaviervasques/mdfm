"""Posterior heads of the stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15).

LatentHead reads the encoder output at the P latent positions of each window and gives a diagonal
Gaussian over a D_Z = 32 dimensional latent per 0.25 s patch: mu_z, logvar_z (W, P, 32). ShapeHead reads
the subject position and gives a diagonal Gaussian over the 16 SMPL+H shape coefficients: mu_b, logvar_b
(W, 16); PhenotypeHead gives the same over the 6-dimensional phenotype latent of the open body (W, 6).
Log-variances lie in [LOGVAR_MIN, LOGVAR_MAX] = [-10, 4] through soft_clamp, which is smooth,
bounded and close to the identity inside the range, so no raw value loses its gradient. The heads run in
float32 whatever autocast the caller opened (spec 2.0: float32 outputs under bf16 autocast).

subject_pool combines the shape posteriors of the windows of one person in a micro-step as a product of
Gaussians: precision-weighted mean, precisions summed; a lone window is left unchanged. sample() draws
reparameterized samples, so gradients reach mu and logvar.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_Z = 32
N_BETAS = 16
LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


def soft_clamp(x, lo=LOGVAR_MIN, hi=LOGVAR_MAX):
    """lo + softplus(x - lo) - softplus(x - hi): smooth, inside [lo, hi], within 0.02 of x for x in [lo + 4,
    hi - 4], gradient positive everywhere (until it underflows far outside). Above the midpoint it is
    computed as the equal hi + softplus(lo - x) - softplus(hi - x): the first form cancels two large terms
    there, and float32 rounding of x - lo and x - hi would put a raw value near 1e6 above hi."""
    upper = hi + F.softplus(lo - x) - F.softplus(hi - x)
    lower = lo + F.softplus(x - lo) - F.softplus(x - hi)
    return torch.where(x > 0.5 * (lo + hi), upper, lower)


def soft_clamp_inverse(y, lo=LOGVAR_MIN, hi=LOGVAR_MAX, iters=60):
    """x with soft_clamp(x, lo, hi) = y for y inside (lo, hi), by Newton steps in float64 from x = y."""
    y = torch.as_tensor(y, dtype=torch.float64)
    x = y.clone()
    for _ in range(iters):
        x = x - (soft_clamp(x, lo, hi) - y) / (torch.sigmoid(x - lo) - torch.sigmoid(x - hi))
    return x


def _no_autocast(x):
    return torch.autocast(device_type=x.device.type, enabled=False)


class GaussianHead(nn.Module):
    """Linear(d, 2 k) in float32 -> mu (..., k), logvar (..., k) soft-clamped to [-10, 4]. mu0 and
    logvar0 (k,) (default 0) are the outputs at a zero input: the bias holds mu0 and the soft_clamp
    inverse of logvar0; the weight starts small."""

    def __init__(self, d, k, mu0=None, logvar0=None, init_std=0.02):
        super().__init__()
        self.k = int(k)
        self.proj = nn.Linear(d, 2 * self.k)
        nn.init.normal_(self.proj.weight, std=init_std)
        with torch.no_grad():
            self.proj.bias.zero_()
            if mu0 is not None:
                self.proj.bias[:self.k] = torch.as_tensor(mu0, dtype=torch.float32)
            lv0 = torch.zeros(self.k) if logvar0 is None else torch.as_tensor(logvar0, dtype=torch.float64)
            self.proj.bias[self.k:] = soft_clamp_inverse(lv0.expand(self.k)).float()

    def forward(self, h):
        with _no_autocast(h):
            dt = self.proj.weight.dtype
            out = F.linear(h.to(dt), self.proj.weight, self.proj.bias)
            mu, raw = out[..., :self.k], out[..., self.k:]
            return mu, soft_clamp(raw)


class LatentHead(GaussianHead):
    """LatentHead(d=512, d_z=32): h (W, P, d) at the latent positions -> mu_z, logvar_z (W, P, d_z);
    logvar 0 at a zero input (the unit variance of the motion prior at initialization)."""

    def __init__(self, d=512, d_z=D_Z):
        super().__init__(d, d_z)


class ShapeHead(GaussianHead):
    """ShapeHead(d=512, n_betas=16): h (W, d) at the subject position -> mu_b, logvar_b (W, 16); mu0 and
    logvar0 (16,) default to 0 (pass the pooled shape prior's mean and log variances)."""

    def __init__(self, d=512, n_betas=N_BETAS, mu0=None, logvar0=None):
        super().__init__(d, n_betas, mu0, logvar0)


class PhenotypeHead(GaussianHead):
    """PhenotypeHead(d=512): h (W, d) at the subject position -> mu_u, logvar_u (W, 6), a diagonal Gaussian over
    the open body's phenotype latent u (mdfm/body/openbody.py: phenotype = lo + (hi - lo) sigmoid(u), the
    logit-normal of plan section 3.1); mu0 and logvar0 (6,) default to 0 (pass the pooled PhenotypePrior's mean
    and log variances). The open body's counterpart of ShapeHead."""

    def __init__(self, d=512, mu0=None, logvar0=None):
        from ..body.openbody import N_PHENO
        super().__init__(d, N_PHENO, mu0, logvar0)


def group_ids(person):
    """Integer group ids (W,) int64 of person keys: an int64 tensor, a list or array of strings or ints."""
    if torch.is_tensor(person):
        return torch.unique(person, return_inverse=True)[1]
    _, inv = np.unique(np.asarray(person).astype(str), return_inverse=True)
    return torch.as_tensor(inv.reshape(-1), dtype=torch.int64)


def subject_pool(mu_b, logvar_b, person_hash):
    """Product of the Gaussians of the windows of each person in the micro-step, back on every window:
    mu (W, 16) = sum_i mu_i / s_i^2 / sum_i 1 / s_i^2 over the windows i of the person, logvar (W, 16) =
    -log sum_i 1 / s_i^2. person_hash (W,): int64 tensor or string keys. A lone window is returned
    unchanged (to rounding)."""
    gid = group_ids(person_hash).to(mu_b.device)
    G = int(gid.max()) + 1 if gid.numel() else 0
    prec = torch.exp(-logvar_b)
    P = torch.zeros(G, mu_b.shape[-1], dtype=mu_b.dtype, device=mu_b.device).index_add(0, gid, prec)
    M = torch.zeros_like(P).index_add(0, gid, prec * mu_b)
    return (M / P)[gid], -torch.log(P)[gid]


def sample(mu, logvar, n=None, generator=None):
    """Reparameterized draws mu + exp(logvar / 2) eps: (n, ...) for an int n, mu's shape for n None.
    generator: a torch.Generator on mu's device (None: the global one)."""
    shape = tuple(mu.shape) if n is None else (int(n),) + tuple(mu.shape)
    eps = torch.randn(shape, dtype=mu.dtype, device=mu.device, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps
