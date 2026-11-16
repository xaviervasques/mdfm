"""The local 10 s encoder (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.14): the dense
pre-norm transformer measured by scripts/bench_encoder.py (reports/bench/bench_encoder.md), 12 layers of
width 512 with 8 heads and an MLP ratio of 4, 37,829,632 parameters, with continuous-time rotary
positions (model/rope.py) on queries and keys and one boolean typed mask per micro-step
(model/masks.py).

Block (as the bench): x + O(attn(LN1(x))), then + MLP(LN2(x)), MLP = Linear(d, 4d), GELU, Linear(4d, d);
a final LayerNorm. LayerNorm only, never BatchNorm, so no statistic is shared across records.

Attention runs under sdpa_kernel(BACKENDS, set_priority=True): cuDNN first, then the memory-efficient
kernel, then math. Without set_priority the list only enables backends and torch's default order picks
the efficient kernel (torch 2.11: sdpa_kernel(backends, set_priority=False)); the bench timed cuDNN at
11.69 ms against 17.86 for the efficient backend and 88.81 for math under the typed mask, and the flash
backend refuses a mask. cuDNN takes bf16 or float16 only: float32 queries fall through to the efficient
kernel on the GPU, and the CPU runs math. Precision: the caller opens the bf16 autocast (spec 2.0); the
rotary angles stay float32 (rope.py). The backends attribute can be set to force one backend (tests).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import rope

BACKENDS = (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)
PILOT = {"d": 512, "layers": 12, "heads": 8, "mlp_ratio": 4}
PILOT_PARAMETERS = 37_829_632            # reports/bench/bench_encoder.md, 12 x 512


class Block(nn.Module):
    """One pre-norm block with rotary positions on q and k."""

    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        if d % heads:
            raise ValueError(f"width {d} is not a multiple of {heads} heads")
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, cos, sin, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope.rotate(q, cos, sin), rope.rotate(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    """Encoder(d=512, layers=12, heads=8, mlp_ratio=4). forward(x (R, L, d) float32 or bf16, t (R, L) s,
    mask (R, 1, L, L) bool from masks.build_mask) -> (R, L, d)."""

    def __init__(self, d=512, layers=12, heads=8, mlp_ratio=4, backends=BACKENDS):
        super().__init__()
        self.d, self.heads = d, heads
        self.blocks = nn.ModuleList(Block(d, heads, mlp_ratio) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.backends = list(backends)

    def forward(self, x, t, mask):
        R, L, d = x.shape
        if d != self.d or tuple(t.shape) != (R, L):
            raise ValueError(f"x {tuple(x.shape)} and t {tuple(t.shape)} for an encoder of width {self.d}")
        if mask.dtype != torch.bool or tuple(mask.shape) != (R, 1, L, L):
            raise ValueError(f"mask must be bool (R, 1, L, L) = ({R}, 1, {L}, {L}), "
                             f"got {mask.dtype} {tuple(mask.shape)}")
        t = t.to(device=x.device, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
        cos, sin = rope.cos_sin(t, d // self.heads)
        with sdpa_kernel(self.backends, set_priority=True):
            for blk in self.blocks:
                x = blk(x, cos, sin, mask)
        return self.norm(x)


def n_parameters(module):
    """Number of parameters of a module."""
    return sum(p.numel() for p in module.parameters())
