"""Priors of the stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5 and departures 3 and 6 of
section 1.5, task B4.15): the healthy motion prior over the latent patches and the shape prior.

MotionPrior(d_z=32, d=256, layers=4, heads=8, span=8): an autoregressive Gaussian over the 0.25 s latent
patches z (W, P, 32) of a window. Patch p gets mu_p and log sigma_p (diagonal) from the SPAN = 8 patches
before it, z_(p-8) to z_(p-1) (2 s), and the condition, never from z_p or later patches, so log_prob is an
exact chain-rule density and kl an exact Gaussian KL per patch given the sampled past. Architecture: one
query per patch, the condition vector plus a learned vector, refined by `layers` blocks of cross-attention
and MLP (pre-norm); the keys and values of every block are fixed memory tokens (the patch latents,
Linear(32, d) and LayerNorm, and the condition token), never updated by attention, so the span stays 8
patches whatever the depth (stacked self-attention over a sliding window would reach layers x 8). Query p
sees the memory tokens of patches j with 1 <= p - j <= span inside the window's valid patches, under the
rotary positions of model/rope.py at the patch centres (the score depends on the lag t_p - t_j only), and
the condition token, scored with the unrotated query (a rotation common to query and key cancels), which
also gives patch 0 a key. Output: LayerNorm, then Linear(d, 2 x 32) with zero weights, mu_p = out and
log sigma_p = 0.5 soft_clamp(out): the bias holds 0 for mu and the soft_clamp inverse of 0 for log sigma, so
the prior is N(0, I) at initialization; log sigma_p lies in [-5, 2] (the posterior's logvar range [-10, 4]
halved).

Condition (Cond): sex (0 female, 1 male, 2 unknown), task (the ids of docs/rules/task_vocabulary.yaml,
whose last name is unknown), beta (16) and a session slot of 16 reserved for stage 4, projected without
bias so that the zeros of stage 1 add nothing. No age band (departure 3). In training mode each condition
of each window is dropped to unknown with probability COND_DROP = 0.2 [cfg] (sex 2, the task unknown, beta
replaced by a learned unknown vector), so the prior also serves an unconditioned query.

ShapePrior: the per-gender Gaussian over the 16 betas of reports/body_store/priors.json (statistics of the
AMASS fits of the train persons, one vector per person; codes 0 female, 1 male, 2 neutral, the pooled prior
of an unknown sex). Its buffers are persistent (a checkpoint carries the prior it was trained with); they are
statistics, not SMPL+H arrays. The v1 body's prior.

PhenotypePrior (the open body, plan docs/MODEL_OPEN_PORT_PLAN.md section 3.1, step M2): the same per-sex Gaussian,
over the 6-dimensional phenotype latent u of mdfm/body/openbody.py (phenotype = lo + (hi - lo) sigmoid(u), a
logit-normal on Anny's phenotype box), rows 0 female, 1 male, 2 unknown. Without a file it holds DESIGN_PHENOTYPE_PRIOR,
a weak default [design, provisional]: the gender value (Anny's male 0, female 1) centred on 0.15 for a man, 0.85 for
a woman and 0.5 for an unknown sex with a latent standard deviation of 1, every other value centred on its box
centre with a latent standard deviation of 1.5. Step M4 replaces it by PhenotypePrior.fit on the open body store's
persons (one latent vector per person), written to a file whose SHA-256 the run's provenance records.

Precision: the trunk runs under the caller's autocast (bf16 in training); the output head, log_prob and kl
run in float32 (float64 for a float64 module); ShapePrior runs in float32 under any autocast.
"""
import json
import math
import os
from typing import NamedTuple

import torch
import torch.nn as nn

from .. import paths
from ..data.windows import task_vocabulary
from . import rope
from .posterior import D_Z, N_BETAS, soft_clamp, soft_clamp_inverse

PRIORS_PATH = os.path.join(paths.TREE, "reports", "body_store", "priors.json")
PATCH_S = 0.25
SPAN = 8                    # patches of past latents, 2 s
COND_DROP = 0.2             # [cfg] condition dropout per condition and window, training mode
FREE_BITS = 0.02            # [cfg] nats per latent dimension and patch
N_SEX = 3                   # 0 female, 1 male, 2 unknown
SEX_UNKNOWN = 2
N_SESSION = 16              # stage-4 session residual slot, zeros in stage 1
TASKS = tuple(task_vocabulary()["tasks"])
TASK_UNKNOWN = TASKS.index("unknown")
GENDERS = ("female", "male", "neutral")
LOG_2PI = math.log(2 * math.pi)


class Cond(NamedTuple):
    """Condition of the motion prior for W windows: sex (W,) int64 (0 female, 1 male, 2 unknown), task (W,)
    int64 task ids, beta (W, 16) shape coefficients, session (W, 16) or None (zeros)."""
    sex: torch.Tensor
    task: torch.Tensor
    beta: torch.Tensor
    session: torch.Tensor = None

    def repeat(self, n):
        """The condition of n copies of the W windows, (n W,) in copy-major order."""
        rep = lambda x: None if x is None else x.repeat((n,) + (1,) * (x.dim() - 1))  # noqa: E731
        return Cond(rep(self.sex), rep(self.task), rep(self.beta), rep(self.session))


def patch_mask(n_patches, P, device=None):
    """(W, P) bool, True on the first n_patches[w] patches of each window."""
    n = torch.as_tensor(n_patches, device=device)
    return torch.arange(P, device=n.device)[None] < n[:, None]


def _mask(mask, z):
    if mask is None:
        return torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
    return mask.to(device=z.device, dtype=torch.bool)


def gaussian_kl(mu_q, logvar_q, mu_p, log_sigma_p):
    """KL(N(mu_q, exp(logvar_q)) || N(mu_p, exp(2 log_sigma_p))) per element."""
    return (log_sigma_p - 0.5 * logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2)
            / (2 * torch.exp(2 * log_sigma_p)) - 0.5)


class _Block(nn.Module):
    """Pre-norm cross-attention of the patch queries to the fixed memory, then an MLP."""

    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.h = heads
        self.nq = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, mem, cmem, cos, sin, allowed):
        """x (W, P, d) queries, mem (W, P, d) patch tokens, cmem (W, 1, d) condition token, cos and sin
        (1, 1, P, dh / 2) of the patch centres, allowed (W, 1, P, P) bool query p may see patch j."""
        W, P, d = x.shape
        dh = d // self.h
        split = lambda y: y.view(W, -1, self.h, dh).transpose(1, 2)  # noqa: E731
        q = split(self.q(self.nq(x)))
        k, v = split(self.k(mem)), split(self.v(mem))
        kc, vc = split(self.k(cmem)), split(self.v(cmem))
        scale = 1.0 / math.sqrt(dh)
        s = rope.rotate(q, cos, sin) @ rope.rotate(k, cos, sin).transpose(-1, -2) * scale     # (W, H, P, P)
        s = s.float().masked_fill(~allowed, float("-inf"))
        sc = ((q * kc).sum(-1, keepdim=True) * scale).float()                                  # (W, H, P, 1)
        a = torch.softmax(torch.cat([sc, s], -1), -1)
        out = a[..., :1].to(vc.dtype) * vc + a[..., 1:].to(v.dtype) @ v
        x = x + self.o(out.transpose(1, 2).reshape(W, P, d))
        return x + self.mlp(self.n2(x))


class MotionPrior(nn.Module):
    """The autoregressive Gaussian motion prior over latent patches (module docstring)."""

    def __init__(self, d_z=D_Z, d=256, layers=4, heads=8, span=SPAN, mlp_ratio=4, n_tasks=len(TASKS),
                 n_betas=N_BETAS, n_session=N_SESSION, drop=COND_DROP, patch_s=PATCH_S):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError(f"width {d} with {heads} heads needs an even head dimension")
        self.d_z, self.d, self.span, self.drop, self.patch_s = int(d_z), int(d), int(span), float(drop), float(patch_s)
        self.heads, self.n_tasks = int(heads), int(n_tasks)
        self.z_in = nn.Linear(self.d_z, d)
        self.nm = nn.LayerNorm(d)
        self.sex = nn.Embedding(N_SEX, d)
        self.task = nn.Embedding(self.n_tasks, d)
        self.beta = nn.Linear(n_betas, d)
        self.beta_unknown = nn.Parameter(torch.zeros(d))
        self.session = nn.Linear(n_session, d, bias=False)
        self.query = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(_Block(d, heads, mlp_ratio) for _ in range(layers))
        self.nout = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * self.d_z)
        for p in (self.sex.weight, self.task.weight, self.beta_unknown, self.query):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[self.d_z:] = soft_clamp_inverse(torch.zeros(self.d_z)).float()

    def condition(self, cond, generator=None, dropout=None):
        """Condition vectors (W, d); with dropout (default: in training mode) each condition dropped to
        unknown with probability self.drop (a torch.Generator on the module's device may be given)."""
        dev = self.query.device
        sex = cond.sex.to(dev, torch.int64)
        task = cond.task.to(dev, torch.int64)
        W = sex.shape[0]
        keep_beta = torch.ones(W, 1, dtype=torch.bool, device=dev)
        if (self.training if dropout is None else dropout) and self.drop > 0:
            u = torch.rand(W, 3, device=dev, generator=generator)
            sex = torch.where(u[:, 0] < self.drop, torch.full_like(sex, SEX_UNKNOWN), sex)
            task = torch.where(u[:, 1] < self.drop, torch.full_like(task, TASK_UNKNOWN), task)
            keep_beta = (u[:, 2] >= self.drop)[:, None]
        beta = cond.beta.to(dev, self.beta.weight.dtype)
        c = self.sex(sex) + self.task(task) + torch.where(keep_beta, self.beta(beta), self.beta_unknown)
        if cond.session is not None:
            c = c + self.session(cond.session.to(dev, self.session.weight.dtype))
        return c

    def params(self, z, cond, mask=None, generator=None, dropout=None):
        """mu_p and log sigma_p (W, P, d_z) of every patch of z (W, P, d_z) given the SPAN patches before it
        and the condition; mask (W, P) bool marks the valid patches (padding is never a key). float32
        outputs (float64 for a float64 module) under any autocast."""
        W, P, _ = z.shape
        dev = z.device
        m = _mask(mask, z)
        c = self.condition(cond, generator, dropout)
        mem = self.nm(self.z_in(z.to(self.z_in.weight.dtype)))
        cmem = self.nm(c)[:, None]
        t = ((torch.arange(P, device=dev, dtype=torch.float64) + 0.5) * self.patch_s)[None]
        if self.query.dtype != torch.float64:
            t = t.float()
        cos, sin = rope.cos_sin(t, self.d // self.heads)
        lag = torch.arange(P, device=dev)[:, None] - torch.arange(P, device=dev)[None]      # p - j
        allowed = ((lag >= 1) & (lag <= self.span))[None, None] & m[:, None, None, :]
        x = (self.query + c)[:, None].expand(W, P, self.d)
        for blk in self.blocks:
            x = blk(x, mem, cmem, cos, sin, allowed)
        with torch.autocast(device_type=dev.type, enabled=False):
            dt = self.out.weight.dtype
            out = self.out(self.nout(x.to(dt)))
            return out[..., :self.d_z], 0.5 * soft_clamp(out[..., self.d_z:])

    def log_prob(self, z, cond, mask=None, generator=None, dropout=None):
        """log p(z | cond) (W,): sum over valid patches and dimensions of the Gaussian log densities of
        params()."""
        mu, ls = self.params(z, cond, mask, generator, dropout)
        zz = z.to(mu.dtype)
        lp = -0.5 * ((zz - mu) * torch.exp(-ls)) ** 2 - ls - 0.5 * LOG_2PI
        return (lp.sum(-1) * _mask(mask, z)).sum(-1)

    def kl(self, mu_q, logvar_q, z_sample, cond, free_bits=FREE_BITS, mask=None, generator=None, dropout=None):
        """KL (W,) of q(z) = N(mu_q, exp(logvar_q)) (W, P, d_z) against the prior of each patch given the
        sampled past z_sample (W, P, d_z), analytic per patch and dimension. Free bits per dimension: the
        KL of dimension k summed over the n_w valid patches of a window counts at least free_bits x n_w
        (below it, no gradient pulls q toward the prior)."""
        mu_p, ls_p = self.params(z_sample, cond, mask, generator, dropout)
        m = _mask(mask, z_sample).to(mu_p.dtype)
        kl = gaussian_kl(mu_q.to(mu_p.dtype), logvar_q.to(mu_p.dtype), mu_p, ls_p) * m[..., None]
        per_dim = kl.sum(1)                                                  # (W, d_z)
        floor = free_bits * m.sum(1, keepdim=True)
        return torch.maximum(per_dim, floor).sum(-1)

    def sample(self, cond, n_patches, n=1, generator=None):
        """n ancestral draws (n, W, P, d_z) patch by patch; n_patches an int or (W,) int64 (patches past a
        window's own count are 0). generator: a torch.Generator on the module's device."""
        dev = self.query.device
        W = cond.sex.shape[0]
        npw = torch.as_tensor(n_patches, device=dev).expand(W) if not torch.is_tensor(n_patches) or \
            n_patches.dim() == 0 else n_patches.to(dev)
        P = int(npw.max())
        c = cond.repeat(n)
        m = patch_mask(npw.repeat(n), P, dev)
        zs = []
        dt = self.out.weight.dtype
        for p in range(P):
            past = torch.stack(zs + [torch.zeros(n * W, self.d_z, device=dev, dtype=dt)], 1)
            mu, ls = self.params(past, c, m[:, :p + 1], generator, dropout=False)
            eps = torch.randn(n * W, self.d_z, device=dev, dtype=dt, generator=generator)
            zs.append((mu[:, p] + torch.exp(ls[:, p]) * eps) * m[:, p, None])
        return torch.stack(zs, 1).view(n, W, P, self.d_z)


class ShapePrior(nn.Module):
    """The per-gender Gaussian shape prior (module docstring): buffers mean (3, 16), cov (3, 16, 16) and its
    Cholesky factor chol (3, 16, 16), rows in GENDERS order (0 female, 1 male, 2 neutral)."""

    def __init__(self, mean=None, cov=None, path=PRIORS_PATH):
        super().__init__()
        if mean is None or cov is None:
            with open(path, encoding="utf-8") as f:
                sp = json.load(f)["shape_prior"]
            mean = [sp[g]["mean"] for g in GENDERS]
            cov = [sp[g]["cov"] for g in GENDERS]
        mean = torch.as_tensor(mean, dtype=torch.float64)
        cov = torch.as_tensor(cov, dtype=torch.float64)
        if tuple(mean.shape) != (3, N_BETAS) or tuple(cov.shape) != (3, N_BETAS, N_BETAS):
            raise ValueError(f"shape prior must be mean (3, 16) and cov (3, 16, 16), got {tuple(mean.shape)} "
                             f"and {tuple(cov.shape)}")
        self._set(mean, cov)

    def _set(self, mean, cov):
        self.k = int(mean.shape[-1])
        self.register_buffer("mean", mean.float())
        self.register_buffer("cov", cov.float())
        self.register_buffer("chol", torch.linalg.cholesky(cov).float())

    def logvar0(self, gender=2):
        """log of the diagonal of one gender's covariance (16,) (the pooled prior by default)."""
        return torch.log(torch.diagonal(self.cov[gender]))

    def _rows(self, gender, like):
        g = torch.as_tensor(gender, dtype=torch.int64, device=self.mean.device)
        if bool(((g < 0) | (g > 2)).any()):
            raise ValueError("gender codes must be 0 (female), 1 (male) or 2 (neutral)")
        dt = like.dtype
        return self.mean[g].to(dt), self.chol[g].to(dt)

    def _full(self):
        """A disabled autocast: the matrix products of log_prob, kl and sample run in float32 (float64 for
        float64 inputs), never in the caller's bf16."""
        return torch.autocast(device_type=self.mean.device.type, enabled=False)

    def log_prob(self, beta, gender):
        """log N(beta; mean_g, cov_g) (W,) for beta (W, 16) and gender codes (W,)."""
        with self._full():
            beta = beta.float() if beta.dtype in (torch.float16, torch.bfloat16) else beta
            m, L = self._rows(gender, beta)
            y = torch.linalg.solve_triangular(L, (beta - m)[..., None], upper=False)[..., 0]
            logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            return -0.5 * (y * y).sum(-1) - logdet - 0.5 * self.k * LOG_2PI

    def kl(self, mu_b, logvar_b, gender):
        """KL(N(mu_b, diag exp(logvar_b)) || N(mean_g, cov_g)) (W,)."""
        with self._full():
            mu_b = mu_b.float() if mu_b.dtype in (torch.float16, torch.bfloat16) else mu_b
            m, L = self._rows(gender, mu_b)
            Sinv = torch.cholesky_inverse(L)
            d = mu_b - m
            tr = (torch.diagonal(Sinv, dim1=-2, dim2=-1) * torch.exp(logvar_b.to(mu_b.dtype))).sum(-1)
            maha = (d[..., None, :] @ Sinv @ d[..., :, None])[..., 0, 0]
            logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            return 0.5 * (tr + maha - self.k + logdet - logvar_b.to(mu_b.dtype).sum(-1))

    def sample(self, gender, n=None, generator=None):
        """Draws (W, 16) or (n, W, 16) for gender codes (W,)."""
        with self._full():
            m, L = self._rows(gender, self.mean)
            shape = (m.shape[0], self.k) if n is None else (int(n), m.shape[0], self.k)
            eps = torch.randn(shape, dtype=m.dtype, device=m.device, generator=generator)
            return m + (L @ eps[..., None])[..., 0]


# [design, provisional until step M4 fits the prior on the open body store] (module docstring)
DESIGN_PHENOTYPE_PRIOR = {"gender_value": {0: 0.85, 1: 0.15, 2: 0.5}, "gender_sd": 1.0, "other_sd": 1.5}


class PhenotypePrior(ShapePrior):
    """The per-sex Gaussian over the phenotype latent u (6,) of mdfm/body/openbody.py (module docstring): buffers
    mean (3, 6), cov (3, 6, 6), chol, rows 0 female, 1 male, 2 unknown. mean and cov given, or read from `path`
    (a JSON {'phenotype_prior': {'female' | 'male' | 'neutral': {'mean', 'cov'}}}), else DESIGN_PHENOTYPE_PRIOR."""

    def __init__(self, mean=None, cov=None, path=None):
        nn.Module.__init__(self)
        from ..body import openbody as BO
        k = BO.N_PHENO
        if (mean is None or cov is None) and path is not None:
            with open(path, encoding="utf-8") as f:
                sp = json.load(f)["phenotype_prior"]
            mean = [sp[g]["mean"] for g in GENDERS]
            cov = [sp[g]["cov"] for g in GENDERS]
        if mean is None or cov is None:
            mean, cov = self.design_default()
        mean = torch.as_tensor(mean, dtype=torch.float64)
        cov = torch.as_tensor(cov, dtype=torch.float64)
        if tuple(mean.shape) != (3, k) or tuple(cov.shape) != (3, k, k):
            raise ValueError(f"phenotype prior must be mean (3, {k}) and cov (3, {k}, {k}), got {tuple(mean.shape)} "
                             f"and {tuple(cov.shape)}")
        self._set(mean, cov)

    @staticmethod
    def design_default():
        """(mean (3, 6), cov (3, 6, 6)) float64 of DESIGN_PHENOTYPE_PRIOR in the latent u."""
        from ..body import openbody as BO
        d = DESIGN_PHENOTYPE_PRIOR
        gi = BO.PHENOTYPE_KEYS.index("gender")
        mean = torch.zeros(3, BO.N_PHENO, dtype=torch.float64)
        var = torch.full((BO.N_PHENO,), d["other_sd"] ** 2, dtype=torch.float64)
        var[gi] = d["gender_sd"] ** 2
        for code, g in d["gender_value"].items():
            p = torch.tensor([g if i == gi else 0.5 * sum(BO.DOMAIN[key]) for i, key in enumerate(BO.PHENOTYPE_KEYS)],
                             dtype=torch.float64)
            mean[code] = BO.to_latent(p)
        return mean, torch.diag_embed(var).expand(3, -1, -1).clone()

    @classmethod
    def fit(cls, u, sex, shrink=1e-3):
        """The prior fitted on one latent vector per person: u (N, 6), sex (N,) codes (0 female, 1 male, 2
        unknown). Rows 0 and 1 from the persons of that sex, row 2 from all of them (the pooled prior of an
        unknown sex); covariances with `shrink` added to the diagonal. A sex with fewer than 7 persons takes the
        pooled row."""
        u = torch.as_tensor(u, dtype=torch.float64)
        sex = torch.as_tensor(sex, dtype=torch.int64)

        def moments(x):
            m = x.mean(0)
            c = torch.cov(x.T) if x.shape[0] > 1 else torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype)
            return m, c + shrink * torch.eye(x.shape[1], dtype=x.dtype)
        pooled = moments(u)
        rows = []
        for code in (0, 1):
            sel = u[sex == code]
            rows.append(moments(sel) if sel.shape[0] > u.shape[1] else pooled)
        rows.append(pooled)
        return cls(torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows]))
