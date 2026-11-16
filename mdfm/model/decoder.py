"""Time-query body decoder of the stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task
B4.15; design sections 1 and 4; amended by reports/bench/phaseA_diagnosis.md, fixes 1 to 5, and by
reports/bench/root_motion_diagnosis.md, fix 1: the integrated root).

BodyDecoder(d=512, layers=4, heads=8, d_z=32, n_fourier=32 up to 20 Hz) turns the latent patches of a
window into knots of the 100 Hz body clock (knot k at t = k / rate s from the window start). Keys and
values: the P latent samples of the window, Linear(32, d) and LayerNorm, at the patch centres (p + 0.5) x
0.25 s; padding patches (beyond n_patches) are never keys. Queries: one per requested knot (window w,
knot k), Fourier features sin and cos of 2 pi f t at n_fourier frequencies geometric from FOURIER_MIN_HZ
= 0.05 Hz (one period spans 20 s, twice a window) to 20 Hz [cfg], through Linear(2 n_fourier, d), plus a
learned vector, FiLM-modulated by the window's betas with their gradient stopped (Linear(16, 2 d),
zero-initialized: h (1 + gamma) + shift): FiLM conditions the decoder on the shape and never trains the
shape (diagnosis C4). Each of the `layers` blocks: pre-norm cross-attention from the queries to the
window's latent keys, then an MLP. No attention among queries: a query's output depends on its own time,
its window's latents and betas only, so a sparse set of knots equals the same knots decoded densely.
Queries are grouped per window into a padded (W, Q_max, d) block for the attention; the padding slots are
discarded. checkpoint_blocks (False by default; the training loop sets it, train/loop.Trainer): under
autograd each block runs through torch.utils.checkpoint, which keeps only the block's input and recomputes
its activations in the backward pass; the values and the gradients are the same, the float32 activations of
the padded queries are not stored (phase B peaked above the 14.5 GB cap without it).

Cross-attention (bounded and local, diagnosis C2). Queries and keys are unit vectors per head (F.normalize
before the rotary positions, which keep the norm), the queries times g_h = QK_SCALE_MAX sigmoid(scale_raw_h)
(QK_SCALE0 at initialization), so that every content logit lies within +-QK_SCALE_MAX whatever the weights.
A fixed distance bias -|t_query - t_patch| / tau_h per head (tau: LOCAL_TAU_S for 8 heads, default_tau
otherwise; None: no bias) is added to the logits as the float attention mask of scaled_dot_product_attention
(scale 1.0), -inf at padding patches. The bias is not learned, so locality cannot saturate, and the bounded
content logits keep a gradient. The rotary periods of the decoder start at DEC_ROPE_MIN_PERIOD_S = 0.5 s,
twice the patch spacing (the fastest pair of rope.MIN_PERIOD_S = 0.25 s takes one phase at every patch
centre). attention_stats() recomputes the attention by hand: entropy, logit gap, mass near the query and
the residual norm per block.

Precision (diagnosis C1). The whole decoder runs in the dtype of its weights (float32 in training; float64
for a float64 module), whatever autocast the caller holds: bf16 rounding in the trunk becomes knot-to-knot
jitter that the Hermite second derivative (1 / h^2 = 1e4 at 100 Hz) amplifies about 1e4 times. The encoder,
the stems and the prior stay under the caller's bf16 autocast; only z (32 values per patch) crosses.

Output: LayerNorm and a zero-initialized Linear(d, 95) y, columns N_ROT = 65 rotation residuals, C_TR the
horizontal residual and vertical translation, C_CT contact logits, C_LV pos_logvar, C_RT the root rates.

Integrated root (root-motion diagnosis C1 and C2). The window frame puts the pelvis heading on +z and the
pelvis x and z at 0 at the centre knot c = L // 2 (body/frames.py, data/body_store.py), so the root yaw and
the horizontal root at any knot are integrals from c of the yaw rate and of the heading-frame velocity. The
decoder outputs these local rates and a fixed operator integrates them. The root grid of a window holds the
knots c + GRID_KNOTS i (grid_of: 0.1 s, from the first grid point at or before knot 0 to the first at or after
knot L - 1; 101 points for a 10 s window), queried in the same trunk call as the requested knots; at a grid
point y[C_RT] is the yaw rate omega (rad/s) and y[C_RT + 1:C_RT + 3] the heading-frame pelvis velocity u =
(lateral, forward) (m/s), both the mean rates over the grid interval centred on the point (the targets of
train/losses.truth_root_grid). cumtrap_from_centre integrates omega into psi (the yaw about the centre
heading, 0 at c) and then yaw_rotate(u, psi) into X (the horizontal displacement from the centre knot, 0 at c)
by the cumulative trapezoid; hermite_on_grid reads psi and X at every knot by the cubic Hermite through the
grid values with their tangents (omega and yaw_rotate(u, psi)), so the path is C1 in time and its velocity at
a grid point is the decoded one. With interval-mean rates the trapezoid telescopes: the integral at a grid
point is the local mean of the path over one grid interval, with no drift. Instantaneous rates (central
differences) alias the within-step motion and the mocap jitter into a drift that a 0.25 s grid does not
survive; the 0.25 s grid of reports/bench/root_motion_diagnosis.md failed its discretization check with either
rate (reports/bench/root_grid.md, scripts/check_root_grid.py), and the diagnosis' fallback, 0.1 s, is used.
Nothing in the path depends on a heading far from the knot: a query sees its own time and the latents, and
the path is a fixed function of the local rates. The grid depends on z, beta and the lengths only, so knots
decoded one at a time still equal the same knots decoded together.

Per knot:
- rot6d (M, 22, 6) with orthonormal columns (residual_rotations; phase A diagnosis C3: no output has a free
  scale): the root rot_y(psi) mean_R[0] exp([a, 0, c]) with psi the integrated yaw and (a, c) = (y0, y1) the
  tilt; joint j >= 1 mean_R[j] exp(omega_j) with omega_j = y[3j - 1:3j + 2] (axis-angle); mean_R the
  rotations of mean_pose (mean_rotations), the training mean of reports/body_store/priors.json (mean 6D value
  per joint over the train knots, root in the heading frame) with the root's own yaw removed, so that the
  heading of the decoded root is psi to second order in the tilt; mean_pose is saved with the weights and
  mean_R, not saved, is rebuilt from a loaded mean_pose that differs;
- transl (M, 3) m, the root translation in the window frame: horizontal anchor_xz + X + yaw_rotate(r, psi) -
  r_c, r = RES_MAX tanh(y[[C_TR, C_TR + 2]] / RES_MAX) a heading-frame residual bounded to 5 cm per component
  for the motion within a grid interval that the grid cannot carry and r_c its value at the window's centre
  grid point, so the centre knot sits on the anchor exactly; anchor_xz the mean window-frame root at the centre
  knot of train windows (reports/body_store/window_priors.json transl_centre_mean,
  scripts/build_window_priors.py; the window frame puts the pelvis, not the root, at 0 there); vertical
  transl0[1] + y[C_TR + 1], transl0 the mean root translation of train windows (phase A diagnosis C5a: with 0
  the initial feet stood 1.11 m below the floor);
- contact_logit (M, 2) [left, right] (0 at initialization);
- pos_logvar (M, 22) log m^2, the isotropic variance of each pelvis-relative joint position in the heading
  frame: soft_clamp(POS_LOGVAR0 + residual, -14, 0), 2 log 0.05 m = -5.991 [cfg] at initialization (-5.994
  after the clamp), within 0.9 mm to 1 m.
BodyOut.root (RootGrid) holds the grid: psi, omega, u, X at every grid point and whether the point lies in
its window (the supervision and the dev numbers read it). At initialization (y = 0) every knot is the mean
pose standing still at (anchor_x, transl0_y, anchor_z), yaw 0.

scatter_dense(values, w, k, W, T, fill) places decoded knots into (W, T, ...) buffers for
body/kinematics.KnotBatch and body/hermite.hermite_gather; knots that were not decoded hold `fill` (the
mean pose, transl0) and are never read when the decoded knots cover the Hermite supports of the queries.
"""
import json
import math
import os
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .. import paths
from ..body.rotations import axis_angle_to_matrix, matrix_to_rot6d, rot6d_to_matrix, rot_y, yaw_angle
from ..body.joints import N_BODY
from . import rope
from .posterior import D_Z, N_BETAS, soft_clamp
from .prior import PRIORS_PATH

RATE = 100.0
PATCH_S = 0.25
N_FOURIER = 32
FOURIER_MIN_HZ = 0.05       # [cfg]
FOURIER_MAX_HZ = 20.0
POS_LOGVAR0 = 2 * math.log(0.05)   # [cfg] 5 cm at initialization
POS_LOGVAR_MIN, POS_LOGVAR_MAX = -14.0, 0.0
GRID_KNOTS = 10                     # [cfg] knots between root-grid points: 0.1 s at 100 Hz (module docstring)
RES_MAX = 0.05                      # [cfg] m, bound of the per-knot heading-frame residual of the horizontal root
N_ROT = 2 + (N_BODY - 1) * 3        # 65: root tilt (x, z), axis-angle residuals of joints 1 to 21
C_TR = N_ROT                        # 3: lateral residual, vertical translation, forward residual
C_CT = C_TR + 3                     # 2: contact logits
C_LV = C_CT + 2                     # 22: pos_logvar
C_RT = C_LV + N_BODY                # 3: yaw rate (rad/s), heading-frame pelvis velocity (lateral, forward) m/s
N_OUT = C_RT + 3                    # 95 (93 before reports/bench/root_motion_diagnosis.md, 159 before phase A's)
QK_SCALE_MAX = 5.0                  # [cfg] content logits within +-5: a gap of at most 10 between two keys
QK_SCALE0 = 2.0                     # [cfg] at initialization
DEC_ROPE_MIN_PERIOD_S = 0.5         # [cfg] twice the patch spacing (rope.MIN_PERIOD_S is 0.25 s)
LOCAL_TAU_S = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, None)   # [cfg] distance scale per head (s), None: no bias
TAU_RANGE_S = (0.125, 8.0)          # default_tau for another head count
NEAR_S = 0.5                        # attention_stats: the mass within this distance of the query (s)
WINDOW_PRIORS_PATH = os.path.join(paths.TREE, "reports", "body_store", "window_priors.json")


class Grid(NamedTuple):
    """Root grid of W windows (grid_of): w, k, i (G,) int64 window, knot and index from the centre point; c, lo,
    n, start (W,) int64 the centre knot L // 2, the first index, the number of points, the flat offset."""
    w: torch.Tensor
    k: torch.Tensor
    i: torch.Tensor
    c: torch.Tensor
    lo: torch.Tensor
    n: torch.Tensor
    start: torch.Tensor


class RootGrid(NamedTuple):
    """Root at the grid points (G,) of a body (module docstring): w, k int64; psi (rad, yaw about the centre
    heading), omega (rad/s), u (G, 2) m/s heading-frame pelvis velocity (lateral, forward), X (G, 2) m
    horizontal pelvis displacement (x, z) from the centre knot in the window frame; valid (G,) bool, k within
    [0, L - 1]."""
    w: torch.Tensor
    k: torch.Tensor
    psi: torch.Tensor
    omega: torch.Tensor
    u: torch.Tensor
    X: torch.Tensor
    valid: torch.Tensor


class BodyOut(NamedTuple):
    """Decoded knots: rot6d (M, 22, 6), transl (M, 3) m, contact_logit (M, 2), pos_logvar (M, 22) log m^2,
    all float32 (float64 for a float64 decoder); w, k (M,) int64 the window and knot of each row; root the
    RootGrid of the windows (None for a body not decoded by BodyDecoder)."""
    rot6d: torch.Tensor
    transl: torch.Tensor
    contact_logit: torch.Tensor
    pos_logvar: torch.Tensor
    w: torch.Tensor
    k: torch.Tensor
    root: RootGrid = None


def yaw_rotate(xz, psi):
    """rot_y(psi) applied to horizontal vectors xz (..., 2) = (x, z) (rot_y(psi) e_z = (sin psi, 0, cos psi)):
    a heading-frame (lateral, forward) vector into the frame psi is measured in; yaw_rotate(v, -psi) goes back."""
    c, s = torch.cos(psi), torch.sin(psi)
    return torch.stack([c * xz[..., 0] + s * xz[..., 1], c * xz[..., 1] - s * xz[..., 0]], -1)


def grid_of(lengths, step=GRID_KNOTS):
    """The root grid (Grid) of windows of L = lengths (W,) knots: knots c + step i with c = L // 2 (the
    window-frame knot of data/body_store.py), i from -ceil(c / step) to ceil((L - 1 - c) / step), at least two
    points per window."""
    L = torch.as_tensor(lengths).long()
    c = torch.div(L, 2, rounding_mode="floor")
    lo = -torch.div(c + step - 1, step, rounding_mode="floor")
    hi = torch.div(L - 1 - c + step - 1, step, rounding_mode="floor")
    n = torch.clamp(hi - lo + 1, min=2)
    start = torch.cumsum(n, 0) - n
    w = torch.repeat_interleave(torch.arange(len(L), device=L.device), n)
    i = torch.arange(int(n.sum()), device=L.device) - start[w] + lo[w]
    return Grid(w, c[w] + step * i, i, c, lo, n, start)


def cumtrap_from_centre(x, g, h):
    """Cumulative trapezoid (G, ...) of grid values x (G, ...) at spacing h (s), 0 at each window's centre
    point, backwards for negative indices."""
    W, n_max = len(g.n), int(g.n.max())
    col = g.i - g.lo[g.w]
    buf = x.new_zeros((W, n_max) + tuple(x.shape[1:])).index_put((g.w, col), x)
    cum = torch.cat([torch.zeros_like(buf[:, :1]), torch.cumsum(0.5 * h * (buf[:, 1:] + buf[:, :-1]), 1)], 1)
    cum = cum - cum[torch.arange(W, device=x.device), -g.lo][:, None]
    return cum[g.w, col]


def hermite_on_grid(P, D, g, w, k, h, step=GRID_KNOTS):
    """Values (M, ...) at knots (w, k) (M,) of the cubic Hermite through the grid values P (G, ...) with
    tangents D (G, ...) per second, grid spacing h (s): exact for a polynomial of degree 3 with exact tangents."""
    d = k - g.c[w]
    j = torch.div(d, step, rounding_mode="floor")
    j = torch.minimum(torch.maximum(j, g.lo[w]), g.lo[w] + g.n[w] - 2)
    s = ((d - j * step).to(P.dtype) / step).reshape((-1,) + (1,) * (P.dim() - 1))
    a = g.start[w] + (j - g.lo[w])
    h00, h10, h01, h11 = (1 + 2 * s) * (1 - s) ** 2, s * (1 - s) ** 2, s * s * (3 - 2 * s), s * s * (s - 1)
    return h00 * P[a] + h10 * h * D[a] + h01 * P[a + 1] + h11 * h * D[a + 1]


def integrate_root(omega, u, g, h):
    """(psi (G,), X (G, 2), V (G, 2)): the yaw about the centre heading, the horizontal displacement from the
    centre knot and the window-frame velocity at the grid points of the rates omega (G,) rad/s and u (G, 2) m/s
    (module docstring)."""
    psi = cumtrap_from_centre(omega, g, h)
    V = yaw_rotate(u, psi)
    return psi, cumtrap_from_centre(V, g, h), V


def bounded_residual(v):
    """RES_MAX tanh(v / RES_MAX): the per-knot horizontal residual within +-RES_MAX (m) per component."""
    return RES_MAX * torch.tanh(v / RES_MAX)


def mean_pose_from_report(path=PRIORS_PATH):
    """The training mean pose (22, 6) float64 of reports/body_store/priors.json."""
    with open(path, encoding="utf-8") as f:
        mp = json.load(f)["mean_pose"]
    x = torch.tensor(mp["rot6d"], dtype=torch.float64)
    if tuple(x.shape) != (N_BODY, 6):
        raise ValueError(f"{path}: mean_pose.rot6d must be (22, 6), got {tuple(x.shape)}")
    return x


def transl0_from_report(path=WINDOW_PRIORS_PATH):
    """The mean root translation (3,) float64 m of train windows in the window frame,
    reports/body_store/window_priors.json (scripts/build_window_priors.py)."""
    with open(path, encoding="utf-8") as f:
        x = torch.tensor(json.load(f)["transl_mean"], dtype=torch.float64)
    if tuple(x.shape) != (3,) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{path}: transl_mean must be 3 finite values, got {x.tolist()}")
    return x


def anchor_from_report(path=WINDOW_PRIORS_PATH):
    """The anchor (2,) float64 m: x and z of the mean window-frame root translation at the centre knot L // 2 of
    train windows, reports/body_store/window_priors.json transl_centre_mean (scripts/build_window_priors.py; the
    window frame puts the pelvis at 0 there, so this is minus the rest pelvis offset, the same within a few mm
    for every subject)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if "transl_centre_mean" not in doc:
        raise ValueError(f"{path}: no transl_centre_mean (run scripts/build_window_priors.py)")
    x = torch.tensor(doc["transl_centre_mean"], dtype=torch.float64)
    if tuple(x.shape) != (3,) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{path}: transl_centre_mean must be 3 finite values, got {x.tolist()}")
    return x[[0, 2]]


def mean_rotations(mean_pose):
    """(22, 3, 3) rotations of a mean pose (22, 6) with the root's own yaw removed (rot_y(-yaw(R0)) R0), so
    that the decoded root rot_y(psi) R0 exp(tilt) has the heading psi (module docstring)."""
    R = rot6d_to_matrix(mean_pose)
    R0 = rot_y(-yaw_angle(R[0])) @ R[0]
    return torch.cat([R0[None], R[1:]], 0)


def default_tau(heads):
    """Distance scales (s) of the heads: LOCAL_TAU_S for len(LOCAL_TAU_S) heads; otherwise heads - 1 scales
    geometric over TAU_RANGE_S and one head without bias (None)."""
    heads = int(heads)
    if heads == len(LOCAL_TAU_S):
        return LOCAL_TAU_S
    if heads == 1:
        return (None,)
    lo, hi = TAU_RANGE_S
    n = heads - 1
    return tuple(lo * (hi / lo) ** (i / max(n - 1, 1)) for i in range(n)) + (None,)


def residual_rotations(y, mean_R, psi):
    """rot6d (M, 22, 6) with orthonormal columns from the raw rotation outputs y (M, 65), the mean rotations
    mean_R (22, 3, 3) (mean_rotations: the root without a yaw of its own) and the root yaw psi (M,) rad (the
    integrated yaw of hermite_on_grid; unbounded, so a window that turns by 635 deg is representable): root
    rot_y(psi) mean_R[0] exp([a, 0, c]), (a, c) = y[:, 0], y[:, 1] the tilt; joint j >= 1 mean_R[j]
    exp(omega_j), omega_j = y[:, 3j - 1:3j + 2]. No output has a free scale (phase A diagnosis C3); y = 0 and
    psi = 0 give the mean pose, and the heading of the root is psi to second order in the tilt."""
    M = y.shape[0]
    tilt = torch.stack([y[:, 0], torch.zeros_like(y[:, 0]), y[:, 1]], -1)
    R0 = rot_y(psi) @ mean_R[0] @ axis_angle_to_matrix(tilt)
    Rj = mean_R[1:] @ axis_angle_to_matrix(y[:, 2:N_ROT].reshape(M, N_BODY - 1, 3))
    return matrix_to_rot6d(torch.cat([R0[:, None], Rj], 1))


def knot_pairs(knots):
    """(w, k) int64 (M,) from a (w, k) pair of tensors or an (M, 2) tensor."""
    if torch.is_tensor(knots):
        if knots.dim() != 2 or knots.shape[1] != 2:
            raise ValueError(f"knots must be (M, 2) or a (w, k) pair, got {tuple(knots.shape)}")
        return knots[:, 0].long(), knots[:, 1].long()
    w, k = knots
    return torch.as_tensor(w).long(), torch.as_tensor(k).long()


def all_knots(lengths, device=None):
    """(w, k) (M,) int64 of every knot k < lengths[w] of every window, window-major."""
    lengths = torch.as_tensor(lengths, dtype=torch.int64, device=device)
    w = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
    start = torch.cumsum(lengths, 0) - lengths
    k = torch.arange(int(lengths.sum()), device=lengths.device) - start[w]
    return w, k


def group_slots(w, W):
    """Slot (M,) int64 of each query within its window's group, in input order, and the largest group size;
    queries of window i fill slots 0 to count_i - 1 of row i of a (W, Q_max) block."""
    M = w.numel()
    order = torch.argsort(w, stable=True)
    counts = torch.bincount(w, minlength=W)
    start = torch.cumsum(counts, 0) - counts
    slot = torch.empty(M, dtype=torch.int64, device=w.device)
    slot[order] = torch.arange(M, device=w.device) - start[w[order]]
    return slot, int(counts.max()) if M else 0


def scatter_dense(values, w, k, W, T, fill=None):
    """(W, T, ...) buffer holding values (M, ...) at (w, k) and `fill` (broadcastable to (...)) elsewhere
    (zeros by default); differentiable in values and fill."""
    shape = (W, T) + tuple(values.shape[1:])
    if fill is None:
        base = torch.zeros(shape, dtype=values.dtype, device=values.device)
    else:
        base = torch.as_tensor(fill, dtype=values.dtype, device=values.device).expand(shape).clone()
    return base.index_put((w.to(values.device), k.to(values.device)), values)


class _XBlock(nn.Module):
    """Pre-norm bounded cross-attention of the knot queries to the window's latent keys (module docstring),
    then an MLP."""

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
        u = QK_SCALE0 / QK_SCALE_MAX
        self.scale_raw = nn.Parameter(torch.full((heads,), math.log(u / (1 - u))))

    def scale(self):
        """g_h = QK_SCALE_MAX sigmoid(scale_raw_h) (H,), the bound of head h's content logits."""
        return QK_SCALE_MAX * torch.sigmoid(self.scale_raw)

    def qk(self, x, mem, qcs, kcs):
        """Unit queries and keys per head, rotated; the queries times g_h, so that every content logit lies
        within +-QK_SCALE_MAX whatever the weights (diagnosis C2). x (W, Q, d), mem (W, P, d) -> q (W, H, Q,
        dh), k (W, H, P, dh)."""
        W, Q, d = x.shape
        split = lambda y: y.view(W, -1, self.h, d // self.h).transpose(1, 2)  # noqa: E731
        q = rope.rotate(F.normalize(split(self.q(self.nq(x))), dim=-1), *qcs)
        k = rope.rotate(F.normalize(split(self.k(mem)), dim=-1), *kcs)
        return q * self.scale().to(q.dtype)[None, :, None, None], k

    def forward(self, x, mem, qcs, kcs, bias):
        """x (W, Q, d) queries, mem (W, P, d), qcs and kcs the rotary cos and sin of the query times (W, 1, Q,
        dh / 2) and of the patch centres (1, 1, P, dh / 2), bias (W, H, Q, P) float: -|t_query - t_patch| /
        tau_h, -inf at padding patches."""
        W, Q, d = x.shape
        q, k = self.qk(x, mem, qcs, kcs)
        v = self.v(mem).view(W, -1, self.h, d // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        x = x + self.o(a.transpose(1, 2).reshape(W, Q, d))
        return x + self.mlp(self.n2(x))


class _Setup(NamedTuple):
    w: torch.Tensor
    k: torch.Tensor
    slot: torch.Tensor
    x: torch.Tensor
    mem: torch.Tensor
    qcs: tuple
    kcs: tuple
    bias: torch.Tensor
    tq: torch.Tensor
    tp: torch.Tensor
    valid: torch.Tensor


class BodyDecoder(nn.Module):
    """The time-query decoder (module docstring). tau: one distance scale (s) or None per head (default
    default_tau(heads))."""

    def __init__(self, d=512, layers=4, heads=8, d_z=D_Z, n_betas=N_BETAS, n_fourier=N_FOURIER,
                 max_hz=FOURIER_MAX_HZ, min_hz=FOURIER_MIN_HZ, mlp_ratio=4, rate=RATE, patch_s=PATCH_S,
                 mean_pose=None, transl0=None, tau=None, anchor=None):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError(f"width {d} with {heads} heads needs an even head dimension")
        tau = default_tau(heads) if tau is None else tuple(tau)
        if len(tau) != heads or any(t is not None and not float(t) > 0 for t in tau):
            raise ValueError(f"tau needs one positive scale or None per head ({heads}), got {tau}")
        self.d, self.heads, self.rate, self.patch_s = int(d), int(heads), float(rate), float(patch_s)
        self.tau = tuple(None if t is None else float(t) for t in tau)
        mp = mean_pose_from_report() if mean_pose is None else torch.as_tensor(mean_pose, dtype=torch.float64)
        mp = mp.reshape(N_BODY, 6)
        self.register_buffer("mean_pose", mp.float())
        self.register_buffer("mean_R", mean_rotations(mp).float(), persistent=False)
        tr0 = transl0_from_report() if transl0 is None else torch.as_tensor(transl0, dtype=torch.float64)
        self.register_buffer("transl0", tr0.reshape(3).float())
        an = anchor_from_report() if anchor is None else torch.as_tensor(anchor, dtype=torch.float64)
        self.register_buffer("anchor_xz", an.reshape(2).float())
        self.register_buffer("inv_tau", torch.tensor([0.0 if t is None else 1.0 / t for t in self.tau]),
                             persistent=False)
        e = torch.arange(n_fourier, dtype=torch.float64) / max(n_fourier - 1, 1)
        self.register_buffer("freqs", (min_hz * (max_hz / min_hz) ** e).float(), persistent=False)
        self.z_in = nn.Linear(d_z, d)
        self.nm = nn.LayerNorm(d)
        self.t_in = nn.Linear(2 * n_fourier, d)
        self.query = nn.Parameter(torch.zeros(d))
        self.film = nn.Linear(n_betas, 2 * d)
        self.blocks = nn.ModuleList(_XBlock(d, heads, mlp_ratio) for _ in range(layers))
        self.checkpoint_blocks = False
        self.nout = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_OUT)
        nn.init.normal_(self.query, std=0.02)
        for lin in (self.film, self.out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """nn.Module's load, then mean_R (not saved) rebuilt by mean_rotations from the loaded mean_pose when it
        differs from the decoder's own, so that a checkpoint written under another reports/body_store/priors.json
        decodes around its own mean pose; an equal mean_pose leaves mean_R bit for bit."""
        before = self.mean_pose.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if not torch.equal(self.mean_pose.detach().to(before.device), before):
            with torch.no_grad():
                self.mean_R.copy_(mean_rotations(self.mean_pose.detach().double()).to(self.mean_R))

    def fourier(self, t):
        """(M, 2 n_fourier) sin and cos of 2 pi f t for knot times t (M,) s, in t's dtype."""
        ph = (2 * math.pi) * t[:, None] * self.freqs.to(t.dtype)
        return torch.cat([torch.sin(ph), torch.cos(ph)], -1)

    def _setup(self, z, beta, knots, n_patches):
        """Queries, keys, rotary angles and the distance bias of a call (the part before the blocks)."""
        w, k = knot_pairs(knots)
        W, P, _ = z.shape
        dev = z.device
        w, k = w.to(dev), k.to(dev)
        wdt = self.out.weight.dtype
        tdt = torch.float64 if wdt == torch.float64 else torch.float32
        npw = torch.full((W,), P, dtype=torch.int64, device=dev) if n_patches is None else \
            torch.as_tensor(n_patches, device=dev).long()
        key_mask = (torch.arange(P, device=dev)[None] < npw[:, None])[:, None, None, :]      # (W, 1, 1, P)
        mem = self.nm(self.z_in(z.to(wdt)))
        dh = self.d // self.heads
        tp = ((torch.arange(P, device=dev, dtype=tdt) + 0.5) * self.patch_s)[None]           # (1, P)
        kcs = rope.cos_sin(tp, dh, min_period_s=DEC_ROPE_MIN_PERIOD_S)
        t = k.to(tdt) / self.rate
        h = self.t_in(self.fourier(t).to(wdt)) + self.query
        gb = self.film(beta.detach().to(wdt))
        gamma, shift = gb[:, :self.d], gb[:, self.d:]
        h = h * (1 + gamma[w]) + shift[w]
        slot, Q = group_slots(w, W)
        x = torch.zeros(W, Q, self.d, dtype=h.dtype, device=dev).index_put((w, slot), h)
        tq = torch.zeros(W, Q, dtype=tdt, device=dev).index_put((w, slot), t)
        valid = torch.zeros(W, Q, dtype=torch.bool, device=dev).index_put(
            (w, slot), torch.ones_like(w, dtype=torch.bool))
        qcs = rope.cos_sin(tq, dh, min_period_s=DEC_ROPE_MIN_PERIOD_S)
        dist = (tq[:, :, None] - tp[:, None, :]).abs()                                       # (W, Q, P)
        bias = -dist[:, None] * self.inv_tau.to(tdt)[None, :, None, None]                    # (W, H, Q, P)
        bias = bias.masked_fill(~key_mask, float("-inf")).to(wdt)
        return _Setup(w, k, slot, x, mem, qcs, kcs, bias, tq, tp[0], valid)

    def _raw(self, z, beta, knots, n_patches):
        s = self._setup(z, beta, knots, n_patches)
        x = s.x
        recompute = self.checkpoint_blocks and torch.is_grad_enabled() and x.requires_grad
        for blk in self.blocks:
            if recompute:
                x = checkpoint(blk, x, s.mem, s.qcs, s.kcs, s.bias, use_reentrant=False)
            else:
                x = blk(x, s.mem, s.qcs, s.kcs, s.bias)
        return self.out(self.nout(x[s.w, s.slot])), s.w, s.k

    def _decode(self, z, beta, knots, n_patches, lengths):
        """BodyOut of the knots and the root grid of the windows from one trunk call (module docstring)."""
        w, k = knot_pairs(knots)
        dev = z.device
        w, k = w.to(dev), k.to(dev)
        L = torch.as_tensor(lengths, device=dev).long()
        if L.shape != (z.shape[0],):
            raise ValueError(f"lengths {tuple(L.shape)} for {z.shape[0]} windows")
        g = grid_of(L)
        y_all, _, _ = self._raw(z, beta, (torch.cat([w, g.w]), torch.cat([k, g.k])), n_patches)
        M = w.numel()
        y, yg = y_all[:M], y_all[M:]
        wdt = y.dtype
        h = GRID_KNOTS / self.rate
        omega, u = yg[:, C_RT], yg[:, C_RT + 1:C_RT + 3]
        psi_g, X_g, V = integrate_root(omega, u, g, h)
        psi = hermite_on_grid(psi_g, omega, g, w, k, h)
        X = hermite_on_grid(X_g, V, g, w, k, h)
        r = bounded_residual(y[:, [C_TR, C_TR + 2]])
        r_c = bounded_residual(yg[g.start - g.lo][:, [C_TR, C_TR + 2]])                  # (W, 2) at the centre
        xz = self.anchor_xz.to(wdt) + X + yaw_rotate(r, psi) - r_c[w]
        transl = torch.stack([xz[:, 0], self.transl0[1].to(wdt) + y[:, C_TR + 1], xz[:, 1]], -1)
        rot6d = residual_rotations(y[:, :N_ROT], self.mean_R.to(wdt), psi)
        contact = y[:, C_CT:C_CT + 2]
        pos_logvar = soft_clamp(POS_LOGVAR0 + y[:, C_LV:C_LV + N_BODY], POS_LOGVAR_MIN, POS_LOGVAR_MAX)
        root = RootGrid(g.w, g.k, psi_g, omega, u, X_g, (g.k >= 0) & (g.k < L[g.w]))
        return BodyOut(rot6d, transl, contact, pos_logvar, w, k, root)

    def forward(self, z, beta, knots, n_patches=None, lengths=None):
        """z (W, P, d_z) latent samples, beta (W, 16), knots: (w, k) (M,) int64 or (M, 2), n_patches (W,)
        valid patches per window (default P), lengths (W,) the knots of each window (required: the root is
        integrated from the centre knot lengths // 2) -> BodyOut of the M knots with the root grid of the
        windows (module docstring). The decoder runs whole in the dtype of its weights (float32 in training),
        whatever autocast the caller holds: bf16 rounding in the trunk becomes knot-to-knot jitter that the
        Hermite second derivative amplifies about 1e4 times (reports/bench/phaseA_diagnosis.md, C1)."""
        if lengths is None:
            raise ValueError("the decoder needs the window lengths: the root is integrated from the centre knot "
                             "lengths // 2 (reports/bench/root_motion_diagnosis.md, fix 1)")
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._decode(z, beta, knots, n_patches, lengths)

    def raw(self, z, beta, knots, n_patches=None):
        """(y (M, N_OUT), w, k): the raw outputs before the heads (rotation residuals in rad), autocast off."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._raw(z, beta, knots, n_patches)

    @torch.no_grad()
    def attention_stats(self, z, beta, knots, n_patches=None, max_queries=256, near_s=NEAR_S):
        """Per block, a dict over at most max_queries of the knots (evenly spaced in the given order), each
        head's attention recomputed by hand from qk() and the bias: entropy (nats, mean over queries and heads;
        uniform over 40 keys is 3.69), entropy_min_head (the lowest head mean), gap_max and gap_median (the
        difference of the two largest logits, content plus bias), near_mass (attention mass on patches within
        near_s of the query), residual_norm (mean norm of the query state after the block), scale (the mean
        content bound g_h)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            w, k = knot_pairs(knots)
            M = w.numel()
            if M > int(max_queries):
                idx = torch.linspace(0, M - 1, int(max_queries)).round().long()
                w, k = w[idx], k[idx]
            s = self._setup(z, beta, (w, k), n_patches)
            x = s.x
            vq = s.valid[:, None].expand(-1, self.heads, -1)                                 # (W, H, Q)
            near = (s.tq[:, :, None] - s.tp[None, None, :]).abs() <= near_s                   # (W, Q, P)
            rows = []
            for blk in self.blocks:
                q, kk = blk.qk(x, s.mem, s.qcs, s.kcs)
                logit = q @ kk.transpose(-1, -2) + s.bias                                     # (W, H, Q, P)
                p = torch.softmax(logit, -1)
                ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)                           # (W, H, Q)
                top = logit.topk(min(2, logit.shape[-1]), -1).values
                gap = (top[..., 0] - top[..., -1])[vq]
                gap = gap[torch.isfinite(gap)]
                mass = (p * near[:, None].to(p.dtype)).sum(-1)
                x = blk(x, s.mem, s.qcs, s.kcs, s.bias)
                per_head = torch.stack([ent[:, h][s.valid].mean() for h in range(self.heads)])
                rows.append({"entropy": float(ent[vq].mean()), "entropy_min_head": float(per_head.min()),
                             "gap_max": float(gap.max()) if gap.numel() else 0.0,
                             "gap_median": float(gap.median()) if gap.numel() else 0.0,
                             "near_mass": float(mass[vq].mean()),
                             "residual_norm": float(x[s.valid].norm(dim=-1).mean()),
                             "scale": float(blk.scale().mean())})
        return rows
