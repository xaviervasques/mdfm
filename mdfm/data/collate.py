"""Collate of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.2 and section 3 step 4, task
B4.17): the windows of one micro-step to the model's mdfm.model.mfm.Batch and the losses' Targets.

- Tokens: model/tokenize.window_tokens of every window, packed first-fit decreasing into rows of one bucket
  (data/sampler.BucketBatcher: the smallest row length that holds the longest window, or the bucket given),
  tokenize.tokenize with that placement. The encoder input (TokenBatch) carries stream samples, masks,
  times and nuisance descriptors only: no covariate, label, source, person, key, truth or synthetic flag.
- Kernel points: per stream-window (the one input stream of each input modality and the force and EMG
  targets, as mfm.batch_from_windows reads them), segments of `segment` consecutive band-copy samples:
  `segments` of them at distinct uniform starts, overlaps merged; a stream of at most segments x segment
  samples is tiled whole. At most max_points points per micro-step: segments are dropped, uniformly among the
  stream-windows holding more than one, never a stream-window's last, until the points fit; so every
  stream-window keeps at least one segment and every weight stays finite. A point's weight is n_total /
  n_kept of its stream-window (all its band-copy samples over the kept ones, as mfm.batch_from_windows
  weighs them; invalid samples add 0 to the likelihood). Segments at uniform starts reach the first and last
  segment - 1 samples of a stream less often than the others, which the weight ignores. The points, the
  per-window kernel fields and the Hermite supports are built by mfm.batch_from_windows on the windows cut
  to their kept samples, so the model reads the fields its own builder writes; the weights are then set per
  stream-window.
- Knots: the union of the Hermite supports of the points (float32 and float64, mfm.support_knots) and, for
  each window with a body truth (synthetic AMASS windows), supervision_knots distinct random knots of its
  clock; at most max_knots per micro-step, the decoder's root-grid points (model/decoder.grid_of: 101 per 10 s
  window at the 0.1 s grid, queried in the same decoder call; reports/bench/root_motion_diagnosis.md, fix 2)
  included: when the supports and the grid exceed it, segments are dropped as above; the supervision knots
  then share what is left, the same number per truth window.
- Truth (Batch.truth, a KnotBody over every window): the body of each window that carries one (rot6d, transl,
  betas, gender, padded with its last knot to the longest clock) and, for a window without one (a real
  window), identity rotations, zero translation and zero betas; Targets.has_truth marks the windows with a
  truth, and stage1_losses must receive the Targets (losses.truth_mask then keeps supervision off the filler).
  Targets come from losses.targets_from_windows (contacts, rendered nuisance, render pairs).
- Covariates stay outside the encoder input: Batch.sex, task, gender, source_id, person (ids of person_id
  within the micro-step). The HSIC stratum of each window (Window.hsic_stratum, attached by a stage-2 window store
  opened with a strata table; task S2.8, spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7) reaches the losses
  only, as the int64 field Targets.hsic_stratum (losses.targets_from_windows; None when every window carries the
  default stratum 0, as every stage-1 batch does); no token reads it.
- Refusals: a CODY source or person, by its 'cody' prefix or a CODY namespace of
  reports/manifest_tiers/cody_denylist.json ('adult:P01' included; train/provenance.is_cody_id), with
  canonical.Refused unless allow_eval, which only an evaluation stage passes (CODY never trains); and, when a
  guard is given (a callable on the windows, the sampler's MixtureSampler.guard), whatever it refuses.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..body.rotations import matrix_to_rot6d
from ..model import mfm as M
from ..model.decoder import grid_of
from ..model import tokenize as TK
from .canonical import Refused
from .sampler import BucketBatcher
from . import windows as W

INPUT = {"imu": "input", "pose3d": "input", "pose2d": "input", "force": "target", "emg": "target"}


@dataclass(frozen=True)
class CollateConfig:
    """Budgets of a micro-step [cfg] (train/config.CollateSection plus the bucket rows and the knot rate)."""
    segments: int = 4
    segment: int = 16
    max_points: int = 32768
    max_knots: int = 32768
    supervision_knots: int = 64
    rows: tuple = (128, 640, 1536)
    rate: float = 100.0


def collate_config(run_cfg):
    """CollateConfig of a train.config.RunConfig (its collate section and mixture rows)."""
    c = run_cfg.collate
    return CollateConfig(int(c.segments), int(c.segment), int(c.max_points), int(c.max_knots),
                         int(c.supervision_knots), tuple(int(r) for r in run_cfg.mixture.rows))


@dataclass
class MicroBatch:
    """One micro-step: batch (mfm.Batch), targets (losses.Targets), row_len and n_rows of the packing, useful
    and padded tokens, points and knots (batch.knots), segments dropped to fit the budgets, supervision knots
    added, grid_knots the decoder's root-grid points of the windows (knots + grid_knots <= max_knots)."""
    batch: M.Batch
    targets: object
    row_len: int
    n_rows: int
    useful: int
    padded: int
    points: int
    knots: int
    segments_dropped: int
    supervision_knots: int
    grid_knots: int = 0


def _stream(window, modality, i):
    s = [x for x in window.streams if x.modality == modality and x.role == INPUT[modality]]
    if len(s) > 1:
        raise ValueError(f"window {i} holds {len(s)} {modality} streams; the pilot model takes one per window")
    return s[0] if s and len(s[0].t) else None


def _segments(n, cfg, rng):
    """Start and stop (k, 2) of the segments of a stream of n samples (module docstring)."""
    seg, k = int(cfg.segment), int(cfg.segments)
    if n <= seg * k:
        st = np.arange(0, n, seg)
    else:
        st = np.sort(rng.choice(n - seg + 1, size=k, replace=False))
    return np.stack([st, np.minimum(st + seg, n)], 1)


def _indices(segs):
    return np.unique(np.concatenate([np.arange(a, b) for a, b in segs]))


def _cut_stream(s, idx):
    return W.Stream(modality=s.modality, role=s.role, fps=s.fps, t=s.t[idx], values=s.values[idx],
                    chan_mask=s.chan_mask, valid=s.valid[idx], meta=W.slice_meta(s.meta, idx), truth=s.truth)


class _Pts:
    def __init__(self, w, t):
        self.w, self.t = w, t


def _lengths(windows, rate):
    """Knots per window as mfm.batch_from_windows counts them."""
    out = np.zeros(len(windows), np.int64)
    for i, wd in enumerate(windows):
        if wd.body is not None:
            out[i] = int(wd.body["lengths"])
        else:
            t_last = max([float(np.max(s.t)) for s in wd.streams if len(s.t)] + [0.0])
            out[i] = int(np.ceil(max(wd.duration, t_last) * rate - 1e-6)) + 1
    return out


def _support_count(sel, windows, lengths, rate):
    """Distinct knots that the kept samples of sel {(window, modality): (stream, segments)} read."""
    ws, ts = [], []
    for (i, _), (s, segs) in sel.items():
        idx = _indices(segs)
        ws.append(np.full(len(idx), i, np.int64))
        ts.append(np.asarray(s.t, np.float32)[idx])
    if not ws:
        return 0
    pts = {"all": _Pts(torch.from_numpy(np.concatenate(ws)), torch.from_numpy(np.concatenate(ts)))}
    return int(M.support_knots(pts, torch.from_numpy(lengths), rate)[0].numel())


def _drop(sel, keys, rng):
    """Drop one segment, uniformly, of a stream-window drawn uniformly among keys holding more than one;
    returns its key, or None when every stream-window holds one segment."""
    cand = [k for k in keys if len(sel[k][1]) > 1]
    if not cand:
        return None
    k = cand[int(rng.integers(len(cand)))]
    s, segs = sel[k]
    sel[k] = (s, np.delete(segs, int(rng.integers(len(segs))), axis=0))
    return k


def collate(windows, bucket=None, cfg=None, gen=None, sources=M.SOURCES, families=None, guard=None,
            allow_eval=False, nuis_fields=None):
    """MicroBatch of windows.Window objects (module docstring). bucket: the row length (default the smallest
    of cfg.rows that holds the longest window); cfg: CollateConfig; gen: numpy Generator or seed of the point,
    segment and knot draws; sources and families as mfm.batch_from_windows; guard: a callable run on the
    windows first; nuis_fields: the nuisance descriptor vocabularies of the tokens (tokenize.nuisance_fields; None:
    tokenize.NUIS_FIELDS), those of the model that reads the batch (a run's name tables, train/loop.py, task S2.7)."""
    from ..train.losses import targets_from_windows
    from ..train.provenance import is_cody_id
    cfg = CollateConfig() if cfg is None else cfg
    rng = gen if isinstance(gen, np.random.Generator) else np.random.default_rng(gen)
    wins = list(windows)
    if not wins:
        raise ValueError("no window to collate")
    for i, wd in enumerate(wins):
        if not allow_eval and is_cody_id(wd.source, wd.person_id):
            raise Refused(f"window {i}: CODY is evaluation-only ({wd.source} {wd.person_id})")
    if guard is not None:
        guard(wins)
    rate = float(cfg.rate)
    # tokens, packed by bucket
    wts = [TK.window_tokens(w, nuis_fields=nuis_fields) for w in wins]
    batcher = BucketBatcher(cfg.rows, max_windows=max(len(wins), 1))
    packing = batcher.pack([w.n for w in wts], row_len=bucket)
    tokens = TK.tokenize(wts, row_len=packing.row_len, placement=packing.placement, nuis_fields=nuis_fields)
    # kernel segments per stream-window
    sel = {}
    for i, wd in enumerate(wins):
        for m in M.MODALITIES:
            s = _stream(wd, m, i)
            if s is not None:
                sel[(i, m)] = (s, _segments(len(s.t), cfg, rng))
    keys = sorted(sel)
    dropped = 0
    size = {k: len(_indices(segs)) for k, (_, segs) in sel.items()}
    n_points = sum(size.values())
    while n_points > int(cfg.max_points):
        k = _drop(sel, keys, rng)
        if k is None:
            raise ValueError(f"{n_points} points with one segment per stream-window exceed max_points "
                             f"{cfg.max_points}")
        dropped += 1
        n_points -= size[k]
        size[k] = len(_indices(sel[k][1]))
        n_points += size[k]
    lengths = _lengths(wins, rate)
    n_sup = int(sum(wd.body is not None for wd in wins))
    n_grid = int(grid_of(torch.from_numpy(lengths)).w.numel())
    budget = int(cfg.max_knots) - n_grid
    n_knots = _support_count(sel, wins, lengths, rate)
    while n_knots > budget:
        for _ in range(max(1, (n_knots - budget) // (int(cfg.segment) + 4))):
            if _drop(sel, keys, rng) is None:
                raise ValueError(f"{n_knots} support knots with one segment per stream-window and {n_grid} root-grid "
                                 f"points exceed max_knots {cfg.max_knots}")
            dropped += 1
        n_knots = _support_count(sel, wins, lengths, rate)
    # windows cut to their kept samples, through the model's builder
    cut, ratio = [], {}
    for i, wd in enumerate(wins):
        streams = []
        for m in M.MODALITIES:
            if (i, m) in sel:
                s, segs = sel[(i, m)]
                idx = _indices(segs)
                streams.append(_cut_stream(s, idx))
                ratio[(i, m)] = len(s.t) / len(idx)
        cut.append(replace(wd, streams=streams))
    b0 = M.batch_from_windows(cut, sources=sources, points_per_stream=None, extra_knots=0, gen=rng, rate=rate,
                              families=families, allow_eval=allow_eval, nuis_fields=nuis_fields)
    if not np.array_equal(b0.lengths.numpy(), lengths):
        raise RuntimeError("knot counts of the cut windows differ from the full windows'")
    points = {}
    for m, p in b0.points.items():
        r = torch.zeros(len(wins), dtype=torch.float32)
        for (i, mm), v in ratio.items():
            if mm == m:
                r[i] = float(v)
        points[m] = p.select(torch.arange(p.w.numel()), r[p.w])
    # knots: supports plus supervision knots of the truth windows
    kw, kk = b0.knots
    T = int(lengths.max())
    keyset = kw * T + kk
    per = 0
    if n_sup:
        per = min(int(cfg.supervision_knots), max(0, (budget - int(keyset.numel())) // n_sup))
    extra = []
    if per:
        for i, wd in enumerate(wins):
            if wd.body is not None:
                L = int(lengths[i])
                extra.append(i * T + rng.choice(L, size=min(per, L), replace=False))
    if extra:
        keyset = torch.unique(torch.cat([keyset, torch.from_numpy(np.concatenate(extra).astype(np.int64))]))
    knots = (keyset // T, keyset % T)
    truth = _truth(wins, lengths, b0.gender) if n_sup else None
    batch = replace(b0, tokens=tokens, points=points, knots=knots, truth=truth)
    targets = targets_from_windows(wins, T=T, allow_eval=allow_eval)
    n_pts = sum(int(p.w.numel()) for p in points.values())
    return MicroBatch(batch, targets, packing.row_len, packing.n_rows, packing.useful, packing.padded, n_pts,
                      int(keyset.numel()), dropped, int(sum(len(e) for e in extra)), n_grid)


def _truth(wins, lengths, gender):
    """KnotBody over every window: the bodies of the windows that carry one, filler elsewhere (module
    docstring)."""
    T = int(lengths.max())
    W_ = len(wins)
    eye = matrix_to_rot6d(torch.eye(3, dtype=torch.float32))
    rot = eye.expand(W_, T, 22, 6).clone()
    tr = torch.zeros(W_, T, 3)
    # the shape vector's size from the windows' bodies: 16 betas (SMPL+H, v1) or the 6 phenotype latents (open body)
    k = next((len(np.asarray(wd.body["betas"])) for wd in wins if wd.body is not None), 16)
    betas = torch.zeros(W_, k)
    for i, wd in enumerate(wins):
        if wd.body is None:
            continue
        L = int(lengths[i])
        r6 = torch.as_tensor(np.asarray(wd.body["rot6d"][:L]), dtype=torch.float32)
        t = torch.as_tensor(np.asarray(wd.body["transl"][:L]), dtype=torch.float32)
        rot[i, :L], tr[i, :L] = r6, t
        rot[i, L:], tr[i, L:] = r6[-1], t[-1]
        betas[i] = torch.as_tensor(np.asarray(wd.body["betas"]), dtype=torch.float32)
    return M.KnotBody(rot, tr, None, betas, gender.clone(), torch.from_numpy(np.asarray(lengths, np.int64)).clone())
