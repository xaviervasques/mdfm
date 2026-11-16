"""Model and forward passes of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25).

G12Model(arm, mfm_config, head): tokenizer (model/tokenize.Tokenizer) and encoder (model/encoder.Encoder) of the
body model's sizes, built first and in mfm.MFM's order, so that under one seed both models start from the same
tokenizer and encoder weights (tests/test_g12_baseline.py); then, for the arm 'mfr', the feature head (FeatureHead),
for the arm 'jepa', the JEPA module of model/jepa.py (JEPA: the EMA target copy of the tokenizer and encoder, and
its Predictor) registered as 'jepa', which train/loop.make_ema leaves out of the model EMA. At the pilot's size the
tokenizer and the encoder are those of the body model (encoder.PILOT_PARAMETERS = 37,829,632 for the encoder).

Masks (draw_masks): model/jepa.draw_masks on the eligible windows (jepa.eligible_windows with the rule of
config.MaskSection: every window of at least min_patches patches, synthetic or real; pose3d windows when
pose3d_anchor is true), from the same generator (jepa.mask_generator of the micro-step seed) in the same window
order; a window with a pose3d anchor (data.heading_patch) redraws its blocks, from the same generator, until the
anchor patch is free, at most max_redraws draws, and stays unmasked after the last failure. Windows without an
anchor draw exactly what jepa.draw_masks draws. A masked patch hides its latent token and every measurement token of
the patch (jepa.token_mask); subject and nuisance tokens are never masked.

Context pass (both arms; context): the rows holding a window with masked patches (jepa.select_rows), the tokenizer
on them, the encoder under the typed mask with every hidden key removed from every query but itself
(jepa.context_mask, jepa.run_encoder): the outputs at the kept positions are those of the windows with the hidden
tokens deleted, as in E1. The kept body-typed positions (latent, subject and measurement tokens outside the
blocks) feed the predictor; nuisance tokens and padding never do.

Arm mfr (forward_mfr). Queries: the hidden measurement tokens. FeatureHead: Linear(d, width) on the kept positions;
at each query position a learned mask vector plus a learned embedding of the token's stem and of its site (the
group to reconstruct); depth pre-norm blocks of the encoder's kind (encoder.Block, rotary time positions)
attending within the window over kept and query positions (jepa.window_mask); a final LayerNorm; per stem a
Linear(width, C x 24) read at the query positions. Target: the stem features of the hidden token,
ContinuousStem.features of its own samples (parameter-free, float32, no gradient): per channel the Hann-weighted
projections at the 11 frequencies (real parts) and 10 non-zero ones (imaginary parts), the weighted standard
deviation, slope and coverage (24 values); scored on the channels holding a valid sample in the span and on the
projections kept under 0.8 x the stream's Nyquist (the stem zeroes the others). Loss of a token: the mean squared
error over its scored values; of a window: the mean over its hidden tokens with at least one scored value.

Arm jepa (forward_jepa). Queries: the latent token of each masked patch. Prediction: the JEPA module's Predictor.
Target: JEPA.target_vectors, the EMA target tokenizer and encoder on the same rows under the standard typed mask,
the final hidden state at the query, through a parameter-free layer norm, in float32. Loss: train/losses.
jepa_per_window (smooth L1, beta 1, mean over features, then over the window's masked patches). The target follows
the online tokenizer and encoder after every applied optimizer step (JEPA.update_target, decay jepa.ema_decay over
the whole run); collapse statistics jepa.spread of the targets and predictions.

Gradients reach the tokenizer, the encoder and the predictor or head; never the JEPA target. Every forward pass
reads the TokenStep's tokens and anchors, the micro-step seed and nothing else.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel

from ..model import jepa as JP
from ..model import masks as MK
from ..model import rope
from ..model import tokenize as TK
from ..model.encoder import BACKENDS, Block, Encoder
from ..train import losses as LS
from .config import ARMS, HeadSection


# ---------------------------------------------------------------- masks
def draw_masks(n_patches, eligible, anchors, seed, rule=JP.MaskRule(), max_redraws=16):
    """(W, P_max) bool numpy masked patches (module docstring): jepa.draw_masks with the anchored redraws."""
    n_p = np.asarray(n_patches.cpu() if torch.is_tensor(n_patches) else n_patches, np.int64)
    anc = np.asarray(anchors.cpu() if torch.is_tensor(anchors) else anchors, np.int64)
    out = np.zeros((len(n_p), int(n_p.max()) if len(n_p) else 0), bool)
    rng = JP.mask_generator(seed)
    for w in np.nonzero(np.asarray(eligible, bool))[0]:
        a = int(anc[w])
        for _ in range(int(max_redraws)):
            m = JP.block_mask(n_p[w], rng, rule)
            if a < 0 or not m[a]:
                out[w, :n_p[w]] = m
                break
    return out


def micro_masks(ts, seed, rule, max_redraws=16):
    """(eligible (W,) bool, masked patches (W, P_max) bool) of a TokenStep's windows for the micro-step seed."""
    elig = JP.eligible_windows(ts.tokens, np.zeros(ts.n_windows, bool), rule)
    return elig, draw_masks(ts.tokens.n_patches, elig, ts.anchors, seed, rule, max_redraws)


# ---------------------------------------------------------------- the feature head
def feature_layout(stem):
    """(per-channel feature count, frequency count) of a ContinuousStem."""
    return int(stem.per_channel), int(stem.n_freqs)


class FeatureHead(nn.Module):
    """FeatureHead(d, per_channel {stem: n}, width, depth, heads) of the module docstring. forward(h (R, L, d),
    keep (R, L) bool, query (R, L) bool, stem_id and site_id (R, L) int64 (-1 off the measurement tokens), t (R, L)
    s, mask (R, 1, L, L) bool) -> (R, L, width); predict(z (N, width), stem) -> (N, C, per_channel)."""

    def __init__(self, d=512, per_channel=None, width=256, depth=4, heads=4, mlp_ratio=4):
        super().__init__()
        per_channel = per_channel or {s: 24 for s in TK.STEMS}
        self.d, self.width, self.heads = int(d), int(width), int(heads)
        self.per_channel = {s: int(per_channel[s]) for s in TK.STEMS}
        self.inp = nn.Linear(self.d, self.width)
        self.mask_token = nn.Parameter(torch.zeros(self.width))
        self.q_stem = nn.Embedding(len(TK.STEMS), self.width)
        self.q_site = nn.Embedding(TK.N_SITES + 1, self.width)
        self.blocks = nn.ModuleList(Block(self.width, self.heads, mlp_ratio) for _ in range(int(depth)))
        self.norm = nn.LayerNorm(self.width)
        self.out = nn.ModuleDict({s: nn.Linear(self.width, TK.STEM_CHANNELS[s] * self.per_channel[s])
                                  for s in TK.STEMS})
        self.backends = list(BACKENDS)
        for p in (self.mask_token, self.q_stem.weight, self.q_site.weight):
            nn.init.normal_(p, std=0.02)

    def forward(self, h, keep, query, stem_id, site_id, t, mask):
        z = self.inp(h).float()
        z = torch.where(keep[..., None], z, torch.zeros((), dtype=z.dtype, device=z.device))
        q = self.mask_token.float() + self.q_stem(stem_id.clamp_min(0)).float() + self.q_site(site_id.clamp_min(0)).float()
        z = torch.where(query[..., None], q, z)
        cos, sin = rope.cos_sin(t.to(device=z.device, dtype=torch.float32), self.width // self.heads)
        with sdpa_kernel(self.backends, set_priority=True):
            for blk in self.blocks:
                z = blk(z, cos, sin, mask)
        return self.norm(z)

    def predict(self, z, stem):
        C = TK.STEM_CHANNELS[stem]
        return self.out[stem](z).float().view(z.shape[0], C, self.per_channel[stem])


# ---------------------------------------------------------------- the model
class G12Model(nn.Module):
    """G12Model(arm, d, layers, heads, mlp_ratio, stem_hidden, head) of the module docstring."""

    def __init__(self, arm="mfr", d=512, layers=12, heads=8, mlp_ratio=4, stem_hidden=1024, head=HeadSection()):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r}: one of {ARMS}")
        self.arm = arm
        self.tokenizer = TK.Tokenizer(d, stem_hidden)
        self.encoder = Encoder(d, layers, heads, mlp_ratio)
        if arm == "mfr":
            per = {s: feature_layout(self.tokenizer.stems[s])[0] for s in TK.STEMS}
            self.head = FeatureHead(d, per, int(head.width), int(head.depth), int(head.heads), mlp_ratio)
            self.add_module("jepa", None)
        else:
            self.head = None
            self.jepa = JP.JEPA(self.tokenizer, self.encoder, int(head.width), int(head.depth), int(head.heads))

    @classmethod
    def of(cls, arm, mcfg, head=HeadSection()):
        """The model of an mfm.MFMConfig's tokenizer and encoder sizes."""
        return cls(arm, int(mcfg.d), int(mcfg.layers), int(mcfg.heads), int(mcfg.mlp_ratio), int(mcfg.stem_hidden),
                   head)

    @property
    def device(self):
        return self.tokenizer.latent.device

    def parameter_counts(self):
        """{tokenizer, encoder, predictor (the head or the JEPA predictor), target (JEPA arm)} parameter counts."""
        n = lambda m: int(sum(p.numel() for p in m.parameters()))   # noqa: E731
        out = {"tokenizer": n(self.tokenizer), "encoder": n(self.encoder)}
        if self.arm == "mfr":
            out["predictor"] = n(self.head)
        else:
            out["predictor"] = n(self.jepa.predictor)
            out["target"] = int(sum(p.numel() for p in self.jepa.target_parameters()))
        return out


# ---------------------------------------------------------------- forward passes
@dataclass
class ArmOut:
    """One forward pass of an arm: term (W,) float32 per-window loss (0 without a term, with gradient), has (W,)
    bool numpy windows with a term, masked (masked patches), patches (patches of the windows with masked patches),
    hidden (hidden tokens), queries (scored query tokens), rows (rows passed), std_target and std_pred (jepa arm:
    jepa.spread of the targets and predictions; None otherwise or with fewer than two queries)."""
    term: torch.Tensor
    has: np.ndarray
    masked: int = 0
    patches: int = 0
    hidden: int = 0
    queries: int = 0
    rows: int = 0
    std_target: float = None
    std_pred: float = None


@dataclass
class Context:
    """The context pass of the rows with masked patches: sub (TokenBatch of the rows), h (Rs, L, d) encoder
    outputs, keep and member (Rs, L) bool, hid (Rs, L) bool hidden tokens, seg, kind (Rs, L)."""
    sub: object
    h: torch.Tensor
    keep: torch.Tensor
    member: torch.Tensor
    hid: torch.Tensor
    seg: torch.Tensor
    kind: torch.Tensor


def _empty(W, device):
    return ArmOut(torch.zeros(W, dtype=torch.float32, device=device), np.zeros(W, bool))


def context(model, tokens, patches, recompute=False):
    """Context of the module docstring for a TokenBatch on the model's device and masked patches (W, P_max); None
    when no window holds a masked patch."""
    has = patches.any(1) if patches.size else np.zeros(tokens.n_windows, bool)
    if not has.any():
        return None
    dev = model.device
    seg = tokens.seg.to(dev)
    hidden = JP.token_mask(tokens, patches).to(dev)
    has_t = torch.as_tensor(has, device=dev)
    member = (seg >= 0) & has_t[seg.clamp_min(0)]
    rows = torch.nonzero(member.any(1))[:, 0]
    sub = JP.select_rows(tokens, rows)
    seg_s, ttype_s, kind_s = sub.seg.to(dev), sub.ttype.to(dev), sub.kind.to(dev)
    hid_s, mem_s = hidden[rows], member[rows]
    x = model.tokenizer(sub)
    h = JP.run_encoder(model.encoder, x, sub.t.to(dev), JP.context_mask(seg_s, ttype_s, hid_s), recompute=recompute)
    body = (ttype_s == MK.BODY) & ((kind_s == TK.LATENT) | (kind_s == TK.SUBJECT) | (kind_s == TK.MEASUREMENT))
    keep = mem_s & body & ~hid_s
    return Context(sub, h, keep, mem_s, hid_s, seg_s, kind_s)


def stem_site_ids(sub, device):
    """(stem_id, site_id) (R, L) int64: the stem index (tokenize.STEMS) and site id of every measurement token,
    -1 elsewhere."""
    R, L = sub.t.shape
    stem_id = torch.full((R * L,), -1, dtype=torch.int64, device=device)
    site_id = torch.full((R * L,), -1, dtype=torch.int64, device=device)
    for k, s in enumerate(TK.STEMS):
        si = sub.stems[s]
        if not len(si.row):
            continue
        flat = si.row.to(device) * L + si.pos.to(device)
        stem_id[flat] = k
        site_id[flat] = si.site.to(device)
    return stem_id.view(R, L), site_id.view(R, L)


def feature_targets(stem_module, vals, tau, smask, nyq, quad):
    """(target (N, C, per_channel) float32, scored (N, C, per_channel) bool) of the spans (module docstring)."""
    N, _, C = vals.shape
    with torch.no_grad():
        f = stem_module.features(vals, tau, smask, nyq, quad).float()
    per = stem_module.per_channel
    target = f[:, :C * per].view(N, C, per)
    freqs = stem_module.freqs.to(f.device, torch.float32)
    keep = freqs[None] <= TK.NYQ_KEEP * nyq.to(f.device, torch.float32)[:, None] * (1.0 + 1e-6)   # (N, K)
    feat = torch.cat([keep, keep[:, stem_module.nonzero.to(f.device)],
                      torch.ones(N, per - keep.shape[1] - int(stem_module.nonzero.sum()), dtype=torch.bool,
                                 device=f.device)], 1)                                         # (N, per)
    chan = smask.to(torch.bool).any(1).to(f.device)                                             # (N, C)
    return target, chan[:, :, None] & feat[:, None, :]


def forward_mfr(model, tokens, patches, recompute=False):
    """ArmOut of the arm mfr (module docstring) on a TokenBatch already on the model's device."""
    W = tokens.n_windows
    dev = model.device
    ctx = context(model, tokens, patches, recompute)
    if ctx is None:
        return _empty(W, dev)
    sub = ctx.sub
    R, L = sub.t.shape
    query = ctx.member & ctx.hid & (ctx.kind == TK.MEASUREMENT)
    stem_id, site_id = stem_site_ids(sub, dev)
    z = model.head(ctx.h, ctx.keep, query, stem_id, site_id, sub.t.to(dev), JP.window_mask(ctx.seg, ctx.keep | query))
    zf = z.reshape(R * L, -1)
    qf = query.reshape(-1)
    segf = ctx.seg.reshape(-1)
    sums = torch.zeros(W, dtype=torch.float32, device=dev)
    counts = torch.zeros(W, dtype=torch.float32, device=dev)
    n_q = 0
    with torch.autocast(device_type=dev.type, enabled=False):
        for s in TK.STEMS:
            si = sub.stems[s]
            if not len(si.row):
                continue
            flat = si.row.to(dev) * L + si.pos.to(dev)
            q = qf[flat]
            if not bool(q.any()):
                continue
            target, scored = feature_targets(model.tokenizer.stems[s], si.vals.to(dev)[q], si.tau.to(dev)[q],
                                             si.smask.to(dev)[q], si.nyq.to(dev)[q], si.quad.to(dev)[q])
            pred = model.head.predict(zf[flat[q]], s)
            sw = scored.float()
            n = sw.sum((1, 2))
            err = (((pred - target) ** 2) * sw).sum((1, 2)) / n.clamp_min(1.0)
            ok = n > 0
            win = segf[flat[q]]
            sums = sums.index_add(0, win[ok], err[ok])
            counts = counts.index_add(0, win[ok], torch.ones_like(err[ok]))
            n_q += int(ok.sum())
    term = sums / counts.clamp_min(1.0)
    has = (counts > 0).cpu().numpy()
    hm = patches.any(1)
    return ArmOut(term, has, int(patches.sum()), int(tokens.n_patches.cpu().numpy()[hm].sum()), int(ctx.hid.sum()),
                  n_q, int(R))


def forward_jepa(model, tokens, patches, recompute=False):
    """ArmOut of the arm jepa (module docstring) on a TokenBatch already on the model's device."""
    W = tokens.n_windows
    dev = model.device
    jm = model.jepa
    has = patches.any(1) if patches.size else np.zeros(W, bool)
    if not has.any():
        return _empty(W, dev)
    ctx = context(model, tokens, patches, recompute)
    sub = ctx.sub
    query = ctx.member & ctx.hid & (ctx.kind == TK.LATENT)
    q_flat = torch.nonzero(query.reshape(-1))[:, 0]
    target = jm.target_vectors(sub, q_flat)
    y = jm.predictor(ctx.h, ctx.keep, query, sub.t.to(dev), JP.window_mask(ctx.seg, ctx.keep | query))
    pred = y.reshape(-1, y.shape[-1])[q_flat].float()
    window = ctx.seg.reshape(-1)[q_flat]
    term = LS.jepa_per_window(pred, target, window, W)
    got = torch.zeros(W, dtype=torch.bool, device=dev)
    got[window] = True
    return ArmOut(term, got.cpu().numpy(), int(patches.sum()), int(tokens.n_patches.cpu().numpy()[has].sum()),
                  int(ctx.hid.sum()), int(q_flat.numel()), int(sub.t.shape[0]), JP.spread(target), JP.spread(pred))


def forward(model, tokens, patches, recompute=False):
    """forward_mfr or forward_jepa by the model's arm."""
    return (forward_mfr if model.arm == "mfr" else forward_jepa)(model, tokens, patches, recompute)
