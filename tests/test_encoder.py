"""Tests of the typed masks and the encoder (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.14):
37,829,632 parameters at 12 x 512 (the configuration of scripts/bench_encoder.py); the one-way test of the
bench, moved here: through the 12 layers, the gradient of every body output with respect to every nuisance
and reserved input is exactly 0.0 with the typed mask and non-zero without it, in float32 and under bf16
autocast (a row of 1,024 tokens on the GPU, 512 on a CPU-only machine, over 100 tokens of each type); the
same wall from the nuisance descriptor through tokenizer and encoder; the bench's weights load
strictly and give its outputs when every time is 0; the mask rules; no NaN from padding and no leak between
packed windows; a 2 x 64 CPU forward under 1 s; on the GPU, the mask of 4 rows of 4,096 tokens within 2.5 times
its own size, backend parity within 2e-2 in bf16 and a profiler trace naming the cuDNN attention kernel under
set_priority=True.
"""
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mdfm.data import windows as W
from mdfm.model import encoder as E
from mdfm.model import masks as M
from mdfm.model import tokenize as T

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- fixtures
def _packed_row(row, gen, types=4):
    """seg, ttype (1, row) of windows packed as in scripts/bench_encoder.py (80 percent of 40 to 120 tokens,
    15 percent of 200 to 600, 5 percent of 1,000 to 1,500) with a padding tail, token types drawn uniformly
    among the first `types`."""
    seg = torch.full((row,), -1, dtype=torch.int64)
    pos, w = 0, 0
    while True:
        u = torch.rand(1, generator=gen).item()
        lo, hi = (40, 120) if u < 0.80 else (200, 600) if u < 0.95 else (1000, 1500)
        n = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        if pos + n > row:
            break
        seg[pos:pos + n] = w
        pos, w = pos + n, w + 1
    ttype = torch.randint(0, types, (row,), generator=gen)
    return seg[None], ttype[None]


def _batch(R, L, d, gen, types=4, device=DEVICE, dtype=torch.float32):
    segs, tts = zip(*[_packed_row(L, gen, types) for _ in range(R)])
    seg = torch.cat(segs)
    seg = torch.where(seg >= 0, seg + 1000 * torch.arange(R)[:, None], seg)     # windows unique across rows
    return {"x": torch.randn(R, L, d, generator=gen, dtype=dtype).to(device),
            "t": (10 * torch.rand(R, L, generator=gen)).to(device),
            "seg": seg.to(device), "ttype": torch.cat(tts).to(device)}


def _imu(fps, dur, sites=(2, 7, 12), seed=0):
    rng = np.random.default_rng(seed)
    t = (rng.uniform(0, 1 / fps) + np.arange(int(dur * fps) + 1) / fps)
    t = t[t < dur]
    v = np.zeros((len(t), len(sites), 9), np.float32)
    for g in range(len(sites)):
        for c in range(6):
            v[:, g, c] = (1.0 if c == 1 else 0.0) + 0.3 * np.sin(2 * np.pi * (1.1 + 0.5 * c + g) * t + c)
    v[..., 3:6] *= 100.0
    cm = np.zeros((len(sites), 9), bool)
    cm[:, :6] = True
    return W.Stream("imu", "input", float(fps), t.astype(np.float32), v, cm, np.ones((len(t), len(sites)), bool),
                    meta={"site_ids": list(sites), "acc_content": "with_gravity", "layout": "g1_six",
                          "device": "xsens13"})


def _window(streams, dur):
    return W.Window("synthetic", "person", "key", True, 0.0, dur, streams, {"sex": 2, "age_band": 8, "task": 0})


# ---------------------------------------------------------------- size and structure
def test_pilot_encoder_has_the_bench_parameter_count():
    enc = E.Encoder(**E.PILOT)
    assert E.n_parameters(enc) == E.PILOT_PARAMETERS == 37_829_632
    d = 512
    per_block = 2 * d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (4 * d * d + 4 * d) + (4 * d * d + d)
    assert all(E.n_parameters(b) == per_block for b in enc.blocks) and len(enc.blocks) == 12
    assert E.n_parameters(enc.norm) == 2 * d
    assert not list(enc.buffers())                      # the rotary frequencies are not state


class _BenchBlock(torch.nn.Module):
    """The block of scripts/bench_encoder.py, verbatim (the script runs its benchmark on import)."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.o = torch.nn.Linear(d, d)
        self.n2 = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


def test_encoder_is_the_bench_encoder_with_rotary_positions():
    """The bench's state_dict loads strictly, and with every time at 0 (rotations the identity) the outputs equal
    the bench encoder's; relative times then change them."""
    torch.manual_seed(0)
    d, h, layers = 64, 4, 3
    blocks = torch.nn.ModuleList(_BenchBlock(d, h) for _ in range(layers))
    norm = torch.nn.LayerNorm(d)
    sd = {f"blocks.{k}": v for k, v in blocks.state_dict().items()} | {f"norm.{k}": v for k, v in norm.state_dict().items()}
    enc = E.Encoder(d=d, layers=layers, heads=h)
    enc.load_state_dict(sd, strict=True)
    enc, blocks, norm = enc.double(), blocks.double(), norm.double()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 40, d, generator=g, dtype=torch.float64)
    seg = torch.tensor([[0] * 25 + [1] * 12 + [-1] * 3, [2] * 40])
    mask = M.build_mask(seg, torch.randint(0, 4, (2, 40), generator=g))
    ref = x
    for blk in blocks:
        ref = blk(ref, mask)
    ref = norm(ref)
    with torch.no_grad():
        assert torch.allclose(enc(x, torch.zeros(2, 40, dtype=torch.float64), mask), ref, rtol=0, atol=1e-12)
        t = torch.rand(2, 40, generator=g, dtype=torch.float64) * 10
        assert (enc(x, t, mask) - ref).abs().max() > 1e-2


def test_mask_rules_on_every_type_pair():
    seg = torch.tensor([[0, 0, 0, 0, 1, 1, -1, -1]])
    tt = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1]])
    m = M.build_mask(seg, tt)[0, 0]
    allowed = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 0), (3, 3)}
    for q in range(8):
        for k in range(8):
            same = seg[0, q] == seg[0, k] and seg[0, q] >= 0
            want = q == k or (bool(same) and (int(tt[0, q]), int(tt[0, k])) in allowed)
            assert bool(m[q, k]) == want, (q, k)
    ctrl = M.build_mask(seg, tt, typed=False)[0, 0]
    assert ctrl[0, 1] and ctrl[0, 2] and not ctrl[0, 4] and not ctrl[6, 7] and ctrl[6, 6]
    assert M.build_mask(seg, tt).shape == (1, 1, 8, 8) and M.build_mask(seg, tt).dtype == torch.bool
    with pytest.raises(ValueError):
        M.build_mask(seg, torch.full_like(tt, 4))
    # packed rows of random types against the rule written out
    g = torch.Generator().manual_seed(2)
    seg = torch.cat([_packed_row(300, g)[0] for _ in range(2)])
    tt = torch.randint(0, 4, seg.shape, generator=g)
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] >= 0)
    rule = (tt[:, None, :] == M.BODY) | (tt[:, :, None] == tt[:, None, :])
    assert torch.equal(M.build_mask(seg, tt)[:, 0], (same & rule) | torch.eye(300, dtype=torch.bool))


@needs_cuda
def test_mask_of_a_micro_step_builds_without_an_integer_pair_tensor():
    """4 rows of 4,096 tokens on the GPU: the extra memory stays at twice the (R, L, L) bool mask (the pair
    lookup of the type table would allocate (R, L, L) int64 indices, 18 times the mask)."""
    g = torch.Generator().manual_seed(3)
    seg = torch.cat([_packed_row(4096, g)[0] for _ in range(4)]).cuda()
    tt = torch.randint(0, 4, seg.shape, generator=g).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    m = M.build_mask(seg, tt)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base <= 2.5 * m.numel()


# ---------------------------------------------------------------- the one-way rule
@pytest.mark.parametrize("autocast", [False, True])
def test_one_way_gradient_is_exactly_zero_through_12_layers(autocast):
    if autocast and not torch.cuda.is_available():
        pytest.skip("bf16 cuDNN attention needs CUDA")
    torch.manual_seed(0)
    enc = E.Encoder(**E.PILOT).to(DEVICE)
    # CPU row of 512 tokens: 256 held only 82 body tokens (seed 3), below the precondition
    L = 1024 if DEVICE == "cuda" else 512
    batch = _batch(1, L, 512, torch.Generator().manual_seed(3))
    real = batch["seg"] >= 0
    per_type = [int(((batch["ttype"] == k) & real).sum()) for k in range(len(M.TYPES))]
    assert min(per_type) > 100, per_type                # body, nuisance and both reserved types
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=autocast):
        typed = M.one_way_grad(enc, batch, typed=True)
        control = M.one_way_grad(enc, batch, typed=False)
    assert typed == 0.0
    assert control > 0.0


def test_nuisance_descriptors_never_reach_body_outputs():
    """Tokenizer and encoder together: body outputs do not move when the nuisance descriptors change and
    carry exactly zero gradient to the nuisance embedding tables; without the type rule they do."""
    torch.manual_seed(0)
    tok = T.Tokenizer(d=64, hidden=64).double()
    enc = E.Encoder(d=64, layers=3, heads=4).double()
    ws = [_window([_imu(100.0, 3.0, seed=1)], 3.0), _window([_imu(128.0, 2.2, (0, 3), seed=2)], 2.2)]
    batch = T.tokenize(ws, row_len=256)
    other = T.tokenize(ws, row_len=256)
    sizes = torch.tensor([len(v) + 2 for v in T.NUIS_FIELDS.values()])
    other.nuis_desc = (other.nuis_desc + 1) % sizes                               # every descriptor changed
    body = (batch.ttype == M.BODY) & (batch.seg >= 0)
    nuis = batch.kind == T.NUISANCE
    y0 = enc(tok(batch), batch.t, batch.mask())
    y1 = enc(tok(other), other.t, other.mask())
    assert torch.equal(y0[body], y1[body])
    assert (y0[nuis] - y1[nuis]).abs().max() > 1e-3
    for typed in (True, False):
        tok.zero_grad()
        y = enc(tok(batch), batch.t, batch.mask(typed))
        y[body].sum().backward()
        g = max(float(tab.weight.grad.abs().max()) for tab in tok.nuisance.tables)
        assert (g == 0.0) if typed else (g > 0.0)


# ---------------------------------------------------------------- padding and packing
def test_padding_gives_no_nan_and_packed_windows_do_not_leak():
    torch.manual_seed(0)
    enc = E.Encoder(d=64, layers=2, heads=4).double()
    g = torch.Generator().manual_seed(0)
    F64 = torch.float64
    a, b = torch.randn(1, 30, 64, generator=g, dtype=F64), torch.randn(1, 20, 64, generator=g, dtype=F64)
    ta, tb = torch.rand(1, 30, generator=g, dtype=F64) * 5, torch.rand(1, 20, generator=g, dtype=F64) * 5
    ya = enc(a, ta, M.build_mask(torch.zeros(1, 30, dtype=torch.int64), torch.zeros(1, 30, dtype=torch.int64)))
    x = torch.cat([a, b, torch.full((1, 14, 64), 1e3, dtype=torch.float64)], 1)       # 14 padding tokens
    t = torch.cat([ta, tb, torch.zeros(1, 14, dtype=torch.float64)], 1)
    seg = torch.tensor([[0] * 30 + [1] * 20 + [-1] * 14])
    y = enc(x, t, M.build_mask(seg, torch.zeros_like(seg)))
    assert torch.isfinite(y).all()
    assert torch.allclose(y[:, :30], ya, rtol=0, atol=1e-12)
    # a constant time shift of one window changes nothing (relative positions only)
    y2 = enc(x, torch.cat([ta + 3.25, tb, t[:, 50:]], 1), M.build_mask(seg, torch.zeros_like(seg)))
    assert torch.allclose(y2, y, rtol=0, atol=1e-10)


def test_cpu_forward_of_a_2_by_64_encoder_under_one_second():
    torch.manual_seed(0)
    enc = E.Encoder(d=64, layers=2, heads=4)
    batch = _batch(4, 512, 64, torch.Generator().manual_seed(5), device="cpu")
    mask = M.build_mask(batch["seg"], batch["ttype"])
    with torch.no_grad():
        enc(batch["x"], batch["t"], mask)
        t0 = time.perf_counter()
        y = enc(batch["x"], batch["t"], mask)
        dt = time.perf_counter() - t0
    assert y.shape == (4, 512, 64) and torch.isfinite(y).all()
    assert dt < 1.0, dt


# ---------------------------------------------------------------- GPU backends
@needs_cuda
def test_backend_parity_in_bf16():
    torch.manual_seed(0)
    enc = E.Encoder(**E.PILOT).cuda()
    batch = _batch(2, 512, 512, torch.Generator().manual_seed(7), types=2)
    mask = M.build_mask(batch["seg"], batch["ttype"])
    real = batch["seg"] >= 0
    outs = {}
    for be in (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        enc.backends = [be]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs[be] = enc(batch["x"], batch["t"], mask).float()[real]
    ref = outs[SDPBackend.MATH]
    assert torch.isfinite(ref).all() and ref.std() > 0.5
    for be in (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        assert (outs[be] - ref).abs().max() < 2e-2, be


def _profile_step(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        fn()
        torch.cuda.synchronize()
    ops = {e.name for e in prof.events()}
    kernels = {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    return ops, kernels


@needs_cuda
def test_profiler_names_the_cudnn_attention_kernel():
    torch.manual_seed(0)
    enc = E.Encoder(**E.PILOT).cuda()
    batch = _batch(2, 512, 512, torch.Generator().manual_seed(9))
    mask = M.build_mask(batch["seg"], batch["ttype"])

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = enc(batch["x"], batch["t"], mask)
        y.float().pow(2).mean().backward()

    step()
    ops, kernels = _profile_step(step)
    assert "aten::_scaled_dot_product_cudnn_attention" in ops
    assert "aten::_scaled_dot_product_cudnn_attention_backward" in ops
    assert not any("efficient_attention" in o for o in ops)
    assert any("cudnn" in k.lower() and "sdpa" in k.lower() for k in kernels), sorted(kernels)[:20]
    # the review note of the spec: without set_priority the list only enables backends, and torch 2.11
    # picks the efficient kernel
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)

    def plain():
        with sdpa_kernel(list(E.BACKENDS)):
            F.scaled_dot_product_attention(q, q, q, attn_mask=mask)

    ops, _ = _profile_step(plain)
    assert "aten::_scaled_dot_product_efficient_attention" in ops
    assert "aten::_scaled_dot_product_cudnn_attention" not in ops
