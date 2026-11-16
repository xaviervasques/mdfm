"""Continuous-time rotary positions of the encoder (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5,
task B4.14).

A token carries a time t in seconds from its window start (float32 after the window start is
subtracted, spec 2.0): latent and measurement tokens at their patch centres, subject and nuisance tokens
at the window centre. Each head splits its dimension into D/2 pairs (x_i, x_{i + D/2}); pair i turns by
the angle omega_i t, with omega_i = 2 pi / period_i and the periods geometric from MIN_PERIOD_S (0.25 s,
one patch) to MAX_PERIOD_S (64 s). The score of a rotated query at t_q with a rotated key at t_k then
depends on t_k - t_q only, never on the absolute time or on the sample rate, so a constant shift of every
time in a window leaves the attention scores unchanged. No parameter: the frequencies are a function of
the head dimension.

Precision: angles, cosines and the rotation run in float32 (float64 for float64 input) and the result
is cast back to the input dtype, so bf16 queries and keys lose no phase to the bf16 rounding of t x
omega (at 10 s, the fastest pair turns by about 250 rad).
"""
import math

import torch

MIN_PERIOD_S = 0.25
MAX_PERIOD_S = 64.0


def periods(head_dim, min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S):
    """(head_dim // 2,) float64 periods in s, geometric from min_period_s to max_period_s."""
    if head_dim % 2 or head_dim < 2:
        raise ValueError(f"rotary positions need an even head dimension, got {head_dim}")
    n = head_dim // 2
    if n == 1:
        return torch.tensor([float(min_period_s)], dtype=torch.float64)
    e = torch.arange(n, dtype=torch.float64) / (n - 1)
    return min_period_s * (max_period_s / min_period_s) ** e


def frequencies(head_dim, min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S):
    """(head_dim // 2,) float64 angular frequencies omega_i = 2 pi / period_i in rad/s."""
    return 2.0 * math.pi / periods(head_dim, min_period_s, max_period_s)


def _compute_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def cos_sin(t, head_dim, min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S):
    """Token times t (R, L) s -> cos, sin (R, 1, L, head_dim // 2) of the angles omega_i t, float32
    (float64 for float64 t), broadcasting over the heads of (R, H, L, head_dim) queries and keys."""
    dt = _compute_dtype(t)
    with torch.autocast(device_type=t.device.type, enabled=False):
        omega = frequencies(head_dim, min_period_s, max_period_s).to(device=t.device, dtype=dt)
        ang = t.to(dt)[:, None, :, None] * omega
        return torch.cos(ang), torch.sin(ang)


def rotate(x, cos, sin):
    """x (R, H, L, D) turned pair by pair, (x_i, x_{i + D/2}) by the angles of cos, sin (R, 1, L, D/2)
    (cos_sin); computed in the dtype of cos and returned in the dtype of x."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x1, x2 = x.to(cos.dtype).chunk(2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


def apply(q, k, t, min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S):
    """Rotary positions on queries and keys q, k (R, H, L, D) at token times t (R, L) s -> (q, k)
    rotated, in their own dtypes."""
    if torch.float64 in (q.dtype, k.dtype):
        t = t.double()
    cos, sin = cos_sin(t, q.shape[-1], min_period_s, max_period_s)
    return rotate(q, cos, sin), rotate(k, cos, sin)
