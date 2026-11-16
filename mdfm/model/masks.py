"""Typed attention masks of the encoder and the one-way test (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md,
2.5, task B4.14; design section 4: body positions never attend to nuisance tokens).

Token types (ttype): BODY 0 (latent, subject and the kinematic-band measurement tokens), NUISANCE 1 (one
per IMU site, camera or pose3d stream), NEURAL 2 and HIGHBAND 3 (the 20 to 250 Hz measurement tokens of
stage 2), both reserved. build_mask(seg, ttype) -> (R, 1, L, L) bool, True where a query may attend to
a key: both in one window (seg >= 0 and equal) and the key type allowed for the query type (ALLOWED):
body queries see body keys only; nuisance queries see body and nuisance keys; a reserved type sees body
keys and its own type. Every position sees itself, so no softmax row is empty and padding (seg -1) sees
only itself: a masked row never turns into NaN. With the typed rule, nothing written into a nuisance or
reserved position reaches a body output through any number of layers; one_way_grad measures it.
"""
import torch

BODY, NUISANCE, NEURAL, HIGHBAND = 0, 1, 2, 3
TYPES = ("body", "nuisance", "neural", "highband")
# ALLOWED[q, k]: a query of type q may attend to a key of type k of its own window
ALLOWED = torch.tensor([[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 1, 0],
                        [1, 0, 0, 1]], dtype=torch.bool)


def build_mask(seg, ttype, typed=True):
    """seg (R, L) int window index (-1 padding), ttype (R, L) int token type -> (R, 1, L, L) bool on
    seg's device (module docstring). typed=False keeps the window blocks and the diagonal only (the
    control of the one-way test)."""
    seg = torch.as_tensor(seg)
    ttype = torch.as_tensor(ttype, device=seg.device)
    if seg.dim() != 2 or ttype.shape != seg.shape:
        raise ValueError(f"seg {tuple(seg.shape)} and ttype {tuple(ttype.shape)} must both be (R, L)")
    real = seg >= 0
    if bool(((ttype < 0) | (ttype >= len(TYPES)))[real].any()):
        raise ValueError(f"token types outside 0 to {len(TYPES) - 1}")
    R, L = seg.shape
    m = (seg[:, :, None] == seg[:, None, :]) & real[:, :, None]
    if typed:
        tt = torch.where(real, ttype, torch.zeros_like(ttype)).long()
        # key_ok[r, a, k]: a query of type a may see key k; gathered by the query types through an expanded
        # (stride 0) index, so no (R, L, L) integer tensor is built (1.2 GB at 4 x 4,096 with pair indexing)
        key_ok = ALLOWED.to(seg.device)[:, tt].permute(1, 0, 2)                  # (R, n_types, L)
        m &= torch.gather(key_ok, 1, tt[:, :, None].expand(R, L, L))
    m |= torch.eye(L, dtype=torch.bool, device=seg.device)
    return m[:, None]


def _field(batch, name):
    return batch[name] if isinstance(batch, dict) else getattr(batch, name)


def one_way_grad(model, batch, typed=True, seed=0):
    """max |d body outputs / d non-body inputs|: model(x, t, mask) -> (R, L, D) (an Encoder), batch with
    x (R, L, D) token inputs, t (R, L) s, seg and ttype (R, L) (a dict or an object with those fields).
    The body outputs (ttype BODY, seg >= 0) are contracted with a seeded Gaussian cotangent, so no
    cancellation can hide a path; the result is the largest gradient entry on the real non-body inputs
    (0.0 when there are none). Exactly 0.0 with the typed mask; non-zero with typed=False."""
    x = _field(batch, "x").detach().clone().requires_grad_(True)
    t, seg, ttype = (_field(batch, k) for k in ("t", "seg", "ttype"))
    y = model(x, t, build_mask(seg, ttype, typed=typed))
    body = ((ttype == BODY) & (seg >= 0)).to(y.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = y[body]
    cot = torch.randn(out.shape, generator=gen, dtype=torch.float32).to(device=out.device, dtype=out.dtype)
    out.backward(cot)
    other = ((ttype != BODY) & (seg >= 0)).to(y.device)
    if not bool(other.any()):
        return 0.0
    return float(x.grad[other].abs().max())
