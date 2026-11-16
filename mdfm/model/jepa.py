"""JEPA auxiliary loss on the encoder (docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2, experiment E1; task
J2a): masks, the context pass, the predictor, the EMA target and the collapse statistics. The loss reduction
lives in train/losses.py (jepa_per_window) and the training step in train/loop.py; the whole module is attached
to the model only when the run's config enables it (mfm.MFM.attach_jepa, config section jepa, off by default).

Eligible windows (section 2.2; eligible_windows): at least min_patches latent patches (14 by default, a window
of more than 3.25 s), real windows only (the synthetic flag of the micro-batch, train/losses.is_synthetic, never
the encoder input), and no pose3d window by default (pose3d false): the tokenizer turns every pose3d token of a
window into the hip-heading frame of the sample nearest the window centre (tokenize.pose3d_heading), so a kept
token would read the hips inside a masked block through that frame (section 2.3, pose3d heading leak). The
remedy chosen is to leave pose3d windows without a JEPA term; a window is pose3d when it holds a nuisance token
of kind pose3d, read from the encoder input.

Masks (section 2.3; draw_blocks, block_mask, draw_masks): per eligible window of P patches of 0.25 s, a budget
K = floor(mask_max P) patches and a target count n_t drawn uniformly among the integers from ceil(mask_min P) to K; blocks
of block_min to block_max patches, each length drawn uniformly from block_min to min(block_max, K minus the
patches already masked), each placed uniformly among the starts that overlap no previous block; at most
max_blocks blocks; stop when the union reaches n_t, or when no block of block_min patches fits the budget or a
block of the drawn length fits no free space. The generator is numpy's, seeded by (micro-step seed, MASK_SALT),
so the sampler's data order, the renders and the refinement draws of the micro-step are not shifted. A masked
patch hides every stream at its time: its latent token and every measurement token of the patch (token_mask);
the subject and the nuisance tokens are never masked.

Context pass (section 2.4; context_mask, run_encoder): the online encoder on the tokenizer output of the main
pass, on the rows that hold a window with masked patches, under the typed mask (masks.build_mask) with every
masked key removed: no query other than j itself attends to a masked position j, so no attention row is empty.
The stems act token by token and a removed key contributes nothing, so the outputs at the kept positions are
those of the window with the masked tokens deleted; the outputs at the masked positions are discarded. Edge
leak, accepted by the proposal: a measurement token reads the native samples of a 0.5 s Hann span centred on
its patch (tokenize.SPAN_S), so the kept tokens beside a block read the samples within half a patch inside the
block edge. run_encoder is Encoder.forward with, optionally (recompute), each block run through
torch.utils.checkpoint (non-reentrant; the sdpa backends re-entered in the recomputation; the autocast state is
restored by checkpoint): only the block inputs are kept and the block activations are recomputed in the
backward pass, as the decoder does (spec section 1.5, departure 20). Same values and gradients.

Target (section 2.5): an EMA copy of the tokenizer and the encoder (JEPA.target_tokenizer, target_encoder),
without gradient, run on the same rows under the standard typed mask and the caller's autocast. Target vectors:
the final hidden state (after the encoder's own final LayerNorm) at the latent token of each masked patch,
through a parameter-free layer norm over its features, in float32. The copy is synchronized with the online
weights (sync) before the first phase B step, or when a state dict without JEPA entries is loaded into the
model (mfm.MFM.attach_jepa: the target then copies the loaded tokenizer and encoder); after every applied phase
B optimizer step, update_target: theta_target <- m theta_target + (1 - m) theta_online, m = ema_decay(s, N)
= end - (end - start) (1 + cos(pi s / N)) / 2 for phase B step s of N (0.996 at the start, 1.0 at the end).

Predictor (section 2.6; Predictor): Linear(d, width) on the context outputs at the kept body-typed positions of
the window (latent, subject and measurement tokens outside the blocks), one learned mask vector of the width at
the latent position of each masked patch (at the patch centre time), then depth pre-norm blocks of the encoder's
kind (encoder.Block: width, heads, MLP ratio 4, rotary time positions with periods 0.25 to 64 s, rope.py)
attending within the window over those positions only (masked measurement tokens, nuisance tokens and padding
excluded), a final LayerNorm and Linear(width, d), read at the mask positions. The output layer keeps its default
initialization: a zero output would give std_pred 0 at step 0 and disable the collapse alarm (section 2.8).
4 x 256 with 4 heads holds 3,422,720 parameters (PREDICTOR_PARAMETERS, the count of the proposal's [J1, C]).

Collapse statistics (section 2.8; spread): the mean over the features of the standard deviation (unbiased) of
the target vectors, and of the predictions, across the masked patches of one micro-step; None with fewer than
two masked patches. JEPA.ref_std holds the first finite pair (the step-0 references, saved with the model).
"""
import copy
import math
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel
from torch.utils.checkpoint import checkpoint

from . import masks as MK
from . import rope
from . import tokenize as TK
from .encoder import BACKENDS, Block

MASK_SALT = 0x4A455041                     # the mask generator's seed is (micro-step seed, MASK_SALT)
KIND_COL = TK.FIELD_NAMES.index("kind")
POSE3D_KIND = TK.NUIS_FIELDS["kind"].index("pose3d")
PREDICTOR_PARAMETERS = 3_422_720           # Predictor(512, 256, 4, 4), docs/proposals/JEPA_EXPLORATION_DRAFT.md 2.6


@dataclass(frozen=True)
class MaskRule:
    """Eligibility and mask rule of the module docstring (the fields of the config's jepa section of the same
    names)."""
    min_patches: int = 14
    real_only: bool = True
    pose3d: bool = False
    mask_min: float = 0.15
    mask_max: float = 0.30
    block_min: int = 4
    block_max: int = 8
    max_blocks: int = 4

    @classmethod
    def of(cls, section):
        """The MaskRule of an object carrying these fields (train/config.JepaSection)."""
        return cls(**{f.name: getattr(section, f.name) for f in fields(cls)})


# ---------------------------------------------------------------- eligibility and masks
def pose3d_windows(tokens):
    """(W,) bool numpy: windows holding a pose3d input stream (a nuisance token of kind pose3d)."""
    out = np.zeros(tokens.n_windows, bool)
    if len(tokens.nuis_win):
        kind = tokens.nuis_desc[:, KIND_COL].cpu().numpy()
        out[tokens.nuis_win.cpu().numpy()[kind == POSE3D_KIND]] = True
    return out


def eligible_windows(tokens, synthetic, rule=MaskRule()):
    """(W,) bool numpy: the windows that carry a JEPA term (module docstring): at least rule.min_patches patches,
    not synthetic when rule.real_only (synthetic (W,) bool, train/losses.is_synthetic), not pose3d unless
    rule.pose3d."""
    ok = tokens.n_patches.cpu().numpy() >= int(rule.min_patches)
    if rule.real_only:
        ok &= ~np.asarray(synthetic.cpu() if torch.is_tensor(synthetic) else synthetic, bool)
    if not rule.pose3d:
        ok &= ~pose3d_windows(tokens)
    return ok


def draw_blocks(P, rng, rule=MaskRule()):
    """[(start, length)] in patches: the blocks of one eligible window of P patches in the order they were placed
    (module docstring), drawn with the numpy Generator rng."""
    P = int(P)
    m = np.zeros(P, bool)
    K = int(math.floor(float(rule.mask_max) * P + 1e-9))
    lo = min(int(math.ceil(float(rule.mask_min) * P - 1e-9)), K)
    bmin, bmax = int(rule.block_min), int(rule.block_max)
    out = []
    if K < bmin:
        return out
    nt = int(rng.integers(lo, K + 1))
    while int(m.sum()) < nt and len(out) < int(rule.max_blocks):
        room = K - int(m.sum())
        if room < bmin:
            break
        L = int(rng.integers(bmin, min(bmax, room) + 1))
        c = np.concatenate([[0], np.cumsum(m)])
        starts = np.nonzero(c[L:] - c[:-L] == 0)[0]
        if not len(starts):
            break
        a = int(starts[int(rng.integers(len(starts)))])
        m[a:a + L] = True
        out.append((a, L))
    return out


def block_mask(P, rng, rule=MaskRule()):
    """(P,) bool: the masked patches of one eligible window of P patches, the union of draw_blocks."""
    m = np.zeros(int(P), bool)
    for a, L in draw_blocks(P, rng, rule):
        m[a:a + L] = True
    return m


def mask_generator(seed):
    """The numpy Generator of a micro-step's masks: seeded by (seed mod 2^63, MASK_SALT)."""
    return np.random.default_rng([int(seed) % 2 ** 63, MASK_SALT])


def draw_masks(n_patches, eligible, seed, rule=MaskRule()):
    """(W, P_max) bool numpy: the masked patches of every window, drawn in window order for the eligible ones
    (block_mask) from mask_generator(seed); False elsewhere."""
    n_p = np.asarray(n_patches.cpu() if torch.is_tensor(n_patches) else n_patches, np.int64)
    out = np.zeros((len(n_p), int(n_p.max()) if len(n_p) else 0), bool)
    rng = mask_generator(seed)
    for w in np.nonzero(np.asarray(eligible, bool))[0]:
        out[w, :n_p[w]] = block_mask(n_p[w], rng, rule)
    return out


def token_patch(tokens):
    """(R, L) int64: the patch index of every token, floor(t / PATCH_S) (latent and measurement tokens sit at
    their patch centres)."""
    return torch.floor(tokens.t / TK.PATCH_S).long()


def token_mask(tokens, patches):
    """(R, L) bool on the tokens' device: the tokens hidden by the masked patches (W, P_max) bool (array or
    tensor): the latent token and every measurement token of each masked patch."""
    pm = torch.as_tensor(np.asarray(patches) if not torch.is_tensor(patches) else patches, dtype=torch.bool,
                         device=tokens.seg.device)
    seg, kind = tokens.seg, tokens.kind
    if not pm.numel():
        return torch.zeros_like(seg, dtype=torch.bool)
    p = token_patch(tokens).clamp(0, pm.shape[1] - 1)
    hide = pm[seg.clamp_min(0), p]
    return (seg >= 0) & ((kind == TK.LATENT) | (kind == TK.MEASUREMENT)) & hide


def context_mask(seg, ttype, hidden):
    """(R, 1, L, L) bool: the typed mask of the rows (masks.build_mask) with every hidden key (R, L) removed from
    every query but itself."""
    m = MK.build_mask(seg, ttype, typed=True)
    L = seg.shape[1]
    m = m & ~hidden.to(m.device)[:, None, None, :]
    return m | torch.eye(L, dtype=torch.bool, device=m.device)


def window_mask(seg, members):
    """(R, 1, L, L) bool: queries and keys of one window, both in members (R, L) bool, plus the diagonal."""
    L = seg.shape[1]
    m = (seg[:, :, None] == seg[:, None, :]) & members[:, :, None] & members[:, None, :]
    return (m | torch.eye(L, dtype=torch.bool, device=seg.device))[:, None]


def select_rows(tokens, rows):
    """TokenBatch of the rows `rows` (1-D int64, increasing) of a TokenBatch: rows, stems and nuisance tokens
    kept, their row indices renumbered; lat_idx and subj_idx renumbered, -1 for a window outside the rows;
    n_patches and the window indices (seg, nuis_win) unchanged."""
    R, L = tokens.t.shape
    dev = tokens.seg.device
    rows = torch.as_tensor(rows, dtype=torch.int64, device=dev)
    remap = torch.full((R,), -1, dtype=torch.int64, device=dev)
    remap[rows] = torch.arange(len(rows), device=dev)
    stems = {}
    for s, si in tokens.stems.items():
        r = remap.to(si.row.device)[si.row]
        k = r >= 0
        stems[s] = TK.StemInput(vals=si.vals[k], tau=si.tau[k], quad=si.quad[k], smask=si.smask[k], nyq=si.nyq[k],
                                site=si.site[k], row=r[k], pos=si.pos[k])
    nr = remap.to(tokens.nuis_row.device)[tokens.nuis_row]
    nk = nr >= 0

    def flat(idx):
        r = remap.to(idx.device)[torch.div(idx.clamp_min(0), L, rounding_mode="floor")]
        return torch.where((idx >= 0) & (r >= 0), r * L + idx.clamp_min(0) % L, torch.full_like(idx, -1))
    return TK.TokenBatch(t=tokens.t[rows], seg=tokens.seg[rows], ttype=tokens.ttype[rows], kind=tokens.kind[rows],
                         stems=stems, nuis_desc=tokens.nuis_desc[nk], nuis_row=nr[nk], nuis_pos=tokens.nuis_pos[nk],
                         nuis_win=tokens.nuis_win[nk], nuis_stream=tokens.nuis_stream[nk],
                         nuis_site=tokens.nuis_site[nk], lat_idx=flat(tokens.lat_idx), subj_idx=flat(tokens.subj_idx),
                         n_patches=tokens.n_patches)


# ---------------------------------------------------------------- encoder pass with optional recomputation
def run_encoder(encoder, x, t, mask, recompute=False):
    """encoder.Encoder.forward(x, t, mask), each block run through torch.utils.checkpoint when recompute is true
    and autograd records (module docstring); the same values and gradients."""
    R, L, d = x.shape
    if d != encoder.d or tuple(t.shape) != (R, L):
        raise ValueError(f"x {tuple(x.shape)} and t {tuple(t.shape)} for an encoder of width {encoder.d}")
    if mask.dtype != torch.bool or tuple(mask.shape) != (R, 1, L, L):
        raise ValueError(f"mask must be bool (R, 1, L, L) = ({R}, 1, {L}, {L}), got {mask.dtype} {tuple(mask.shape)}")
    t = t.to(device=x.device, dtype=torch.float64 if x.dtype == torch.float64 else torch.float32)
    cos, sin = rope.cos_sin(t, d // encoder.heads)
    if not (recompute and torch.is_grad_enabled()):
        with sdpa_kernel(encoder.backends, set_priority=True):
            for blk in encoder.blocks:
                x = blk(x, cos, sin, mask)
        return encoder.norm(x)
    for blk in encoder.blocks:
        x = checkpoint(_BlockRun(blk, encoder.backends), x, cos, sin, mask, use_reentrant=False,
                       preserve_rng_state=False)
    return encoder.norm(x)


class _BlockRun:
    """A block called under its sdpa backends (re-entered when checkpoint recomputes it)."""

    def __init__(self, blk, backends):
        self.blk, self.backends = blk, backends

    def __call__(self, x, cos, sin, mask):
        with sdpa_kernel(self.backends, set_priority=True):
            return self.blk(x, cos, sin, mask)


# ---------------------------------------------------------------- predictor and target
class Predictor(nn.Module):
    """Predictor(d=512, width=256, depth=4, heads=4) of the module docstring. forward(h (R, L, d) context
    outputs, keep (R, L) bool kept positions, query (R, L) bool mask positions, t (R, L) s, mask (R, 1, L, L)
    bool) -> (R, L, d), read at the query positions."""

    def __init__(self, d=512, width=256, depth=4, heads=4, mlp_ratio=4):
        super().__init__()
        self.d, self.width, self.heads = int(d), int(width), int(heads)
        self.inp = nn.Linear(self.d, self.width)
        self.mask_token = nn.Parameter(torch.zeros(self.width))
        self.blocks = nn.ModuleList(Block(self.width, self.heads, mlp_ratio) for _ in range(int(depth)))
        self.norm = nn.LayerNorm(self.width)
        self.out = nn.Linear(self.width, self.d)
        self.backends = list(BACKENDS)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, h, keep, query, t, mask):
        z = self.inp(h).float()
        z = torch.where(keep[..., None], z, torch.zeros((), dtype=z.dtype, device=z.device))
        z = torch.where(query[..., None], self.mask_token.to(z.dtype), z)
        cos, sin = rope.cos_sin(t.to(device=z.device, dtype=torch.float32), self.width // self.heads)
        with sdpa_kernel(self.backends, set_priority=True):
            for blk in self.blocks:
                z = blk(z, cos, sin, mask)
        return self.out(self.norm(z))


def ema_decay(step, n_steps, start=0.996, end=1.0):
    """EMA decay of the target at phase B step `step` of n_steps: end - (end - start) (1 + cos(pi u)) / 2 with u
    = step / n_steps clipped to [0, 1] (start at u = 0, end at u = 1)."""
    u = min(max(float(step) / max(int(n_steps), 1), 0.0), 1.0)
    return float(end) - (float(end) - float(start)) * (1.0 + math.cos(math.pi * u)) / 2.0


def spread(v):
    """Mean over the features of the unbiased standard deviation of the rows of v (N, d) across N, as a float;
    None for fewer than two rows."""
    if v is None or v.shape[0] < 2:
        return None
    return float(v.detach().float().std(0).mean())


@dataclass
class JepaForward:
    """One context and target pass: eligible (W,) bool numpy windows with a term; patches (W, P_max) bool numpy
    masked patches; hidden (R, L) bool hidden tokens; rows (Rs,) int64 the rows passed, or None when no window
    holds a masked patch; x_in (Rs, L, d) the context input (a leaf requiring grad when asked); pred (N, d)
    float32 predictions and target (N, d) float32 target vectors at the N masked patches; window (N,) int64
    their windows."""
    eligible: np.ndarray
    patches: np.ndarray
    hidden: torch.Tensor = None
    rows: torch.Tensor = None
    x_in: torch.Tensor = None
    pred: torch.Tensor = None
    target: torch.Tensor = None
    window: torch.Tensor = None

    @property
    def n_masked(self):
        return 0 if self.window is None else int(self.window.numel())


class JEPA(nn.Module):
    """The JEPA module of a model (module docstring): JEPA(tokenizer, encoder, width, depth, heads) copies the
    tokenizer and the encoder into the target (no gradient) and builds the predictor. Buffers: ready (1 once the
    target copies the online weights to start from), ref_std (2,) the step-0 references (std_target, std_pred),
    NaN until measured."""

    def __init__(self, tokenizer, encoder, width=256, depth=4, heads=4):
        super().__init__()
        self.target_tokenizer = copy.deepcopy(tokenizer)
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_parameters():
            p.requires_grad_(False)
            p.grad = None
        self.predictor = Predictor(encoder.d, width, depth, heads)
        self.register_buffer("ready", torch.zeros((), dtype=torch.int64))
        self.register_buffer("ref_std", torch.full((2,), float("nan")))

    def target_parameters(self):
        return list(self.target_tokenizer.parameters()) + list(self.target_encoder.parameters())

    def _pairs(self, tokenizer, encoder):
        tp, op = self.target_parameters(), list(tokenizer.parameters()) + list(encoder.parameters())
        tb = list(self.target_tokenizer.buffers()) + list(self.target_encoder.buffers())
        ob = list(tokenizer.buffers()) + list(encoder.buffers())
        if len(tp) != len(op) or len(tb) != len(ob) or any(a.shape != b.shape for a, b in zip(tp + tb, op + ob)):
            raise ValueError("the JEPA target does not match the tokenizer and encoder it follows")
        return tp, op, tb, ob

    @torch.no_grad()
    def sync(self, tokenizer, encoder):
        """Target <- the online tokenizer and encoder (parameters and buffers); ready set to 1."""
        tp, op, tb, ob = self._pairs(tokenizer, encoder)
        for a, b in zip(tp + tb, op + ob):
            a.copy_(b.detach())
        self.ready.fill_(1)

    def is_ready(self):
        return bool(int(self.ready) == 1)

    @torch.no_grad()
    def update_target(self, tokenizer, encoder, decay):
        """theta_target <- decay theta_target + (1 - decay) theta_online for every parameter; buffers copied."""
        tp, op, tb, ob = self._pairs(tokenizer, encoder)
        torch._foreach_lerp_(tp, [p.detach() for p in op], 1.0 - float(decay))
        for a, b in zip(tb, ob):
            a.copy_(b)

    @torch.no_grad()
    def set_references(self, std_target, std_pred):
        """ref_std <- (std_target, std_pred) when both are finite and the references are still unset; returns
        True when set."""
        if std_target is None or std_pred is None or not bool(torch.isnan(self.ref_std).any()):
            return False
        if not (math.isfinite(std_target) and math.isfinite(std_pred)):
            return False
        self.ref_std.copy_(torch.tensor([std_target, std_pred], dtype=self.ref_std.dtype))
        return True

    def references(self):
        """(std_target, std_pred) step-0 references, None while unset."""
        r = [float(x) for x in self.ref_std.detach().cpu()]
        return (None, None) if any(math.isnan(x) for x in r) else (r[0], r[1])

    @torch.no_grad()
    def target_vectors(self, tokens, query):
        """(N, d) float32: the target pass on a TokenBatch (the selected rows) under the standard typed mask, at
        the flat positions query (N,), through a parameter-free layer norm."""
        x = self.target_tokenizer(tokens)
        h = self.target_encoder(x, tokens.t.to(x.device), tokens.mask().to(x.device))
        v = h.reshape(-1, h.shape[-1])[query]
        with torch.autocast(device_type=v.device.type, enabled=False):
            return F.layer_norm(v.float(), (v.shape[-1],))

    def forward_pass(self, encoder, tokens, x, synthetic, seed, rule=MaskRule(), recompute=True, grad=True):
        """JepaForward of a micro-step (module docstring): tokens the micro-step's TokenBatch, x (R, L, d) the
        tokenizer output of the main pass (its rows are detached into the context input, a leaf requiring grad
        when grad), synthetic (W,) bool, seed the micro-step seed; the online encoder runs the context pass
        (recompute: activation recomputation), the predictor the predictions, the target the target vectors."""
        elig = eligible_windows(tokens, synthetic, rule)
        pm = draw_masks(tokens.n_patches, elig, seed, rule)
        out = JepaForward(eligible=elig, patches=pm)
        has = pm.any(1) if pm.size else np.zeros(len(elig), bool)
        if not has.any():
            return out
        dev = x.device
        seg = tokens.seg.to(dev)
        hidden = token_mask(tokens, pm).to(dev)
        has_t = torch.as_tensor(has, device=dev)
        member = (seg >= 0) & has_t[seg.clamp_min(0)]
        rows = torch.nonzero(member.any(1))[:, 0]
        sub = select_rows(tokens, rows)
        seg_s, ttype_s, kind_s = sub.seg.to(dev), sub.ttype.to(dev), sub.kind.to(dev)
        hid_s, mem_s = hidden[rows], member[rows]
        query = mem_s & hid_s & (kind_s == TK.LATENT)
        q_flat = torch.nonzero(query.reshape(-1))[:, 0]
        target = self.target_vectors(sub, q_flat)
        x_in = x.detach()[rows]
        if grad:
            x_in.requires_grad_(True)
        h = run_encoder(encoder, x_in, sub.t.to(dev), context_mask(seg_s, ttype_s, hid_s), recompute=recompute)
        body = (ttype_s == MK.BODY) & ((kind_s == TK.LATENT) | (kind_s == TK.SUBJECT) | (kind_s == TK.MEASUREMENT))
        keep = mem_s & body & ~hid_s
        y = self.predictor(h, keep, query, sub.t.to(dev), window_mask(seg_s, keep | query))
        pred = y.reshape(-1, y.shape[-1])[q_flat].float()
        out.hidden, out.rows, out.x_in = hidden, rows, x_in
        out.pred, out.target, out.window = pred, target, seg_s.reshape(-1)[q_flat]
        return out
