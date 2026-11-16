"""Data path of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25): the body run's sampler, unchanged
(data/sampler.MixtureSampler with its guard), and the token part of its collate.

collate_tokens(windows, rows) is the token half of data/collate.collate, line for line: the CODY refusal
(train/provenance.is_cody_id on the source and the person, canonical.Refused unless allow_eval, which no training
passes), the guard when given, model/tokenize.window_tokens of every window, data/sampler.BucketBatcher's packing
(the smallest row of rows that holds the longest window, first-fit decreasing) and tokenize.tokenize with that
placement; tests/test_g12_baseline.py checks that its TokenBatch equals collate's. The kernel points, knots and
body truth that collate adds for the body model are not built. train/losses.assert_label_free runs on the windows,
as targets_from_windows does for the body model.

TokenStep carries the encoder input (the TokenBatch, which holds stream samples, masks, times and nuisance
descriptors only), the packing counts and one integer per window, its pose3d anchor (heading_patch): no covariate,
label, source, person, key, truth or synthetic flag. A baseline step reads nothing else.

heading_patch(window): the patch of the sample whose hips set the heading frame of the window's pose3d tokens
(tokenize.patchify: pose3d_heading at the sample nearest duration / 2 among the samples with both hips valid, x
and z present; body/frames.hips_heading), -1 for a window without a pose3d input stream or without such a sample
(the frame is then fixed). A pose3d token of the window depends on that sample's hips through the frame; the
anchored masks of model.py keep its patch visible, so that no kept token reads a masked sample through the frame
(docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2.3, pose3d heading leak).

produce(sampler, rows) is train/loop.produce with collate_tokens in place of collate: the train/loop.Item of the
sampler's next micro-step (host seconds of sample, render and collate).
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..body.frames import C15_LEFT_HIP, C15_RIGHT_HIP
from ..data.canonical import Refused
from ..data.sampler import BucketBatcher
from ..factory import layouts as LY
from ..model import tokenize as TK
from ..train import losses as LS
from ..train import loop as LP
from ..train.provenance import is_cody_id


@dataclass
class TokenStep:
    """One micro-step of a baseline (module docstring): tokens (TokenBatch), anchors (W,) int64, the packing's
    row_len and n_rows, useful and padded tokens."""
    tokens: TK.TokenBatch
    anchors: torch.Tensor
    row_len: int
    n_rows: int
    useful: int
    padded: int

    @property
    def n_windows(self):
        return self.tokens.n_windows


def heading_patch(window):
    """The patch of a window's pose3d heading sample, or -1 (module docstring)."""
    d = float(window.duration)
    P = LY.n_patches(d)
    for s in window.streams:
        if s.role != "input" or s.modality != "pose3d":
            continue
        t = np.asarray(s.t, dtype=np.float64)
        if not len(t):
            return -1
        valid = np.asarray(s.valid, dtype=bool)
        cm = np.asarray(s.chan_mask, dtype=bool)
        ok = valid & cm[:, 0] & cm[:, 2]                                   # what patchify hands pose3d_heading
        hips = ok[:, C15_LEFT_HIP] & ok[:, C15_RIGHT_HIP]
        if not hips.any():
            return -1
        c = int(np.argmin(np.abs(t - d / 2)))
        frames = np.flatnonzero(hips)
        k = int(frames[int(np.argmin(np.abs(frames - c)))])
        p = int(np.floor(t[k] / TK.PATCH_S))
        return p if 0 <= p < P else -1
    return -1


def collate_tokens(windows, rows=(128, 640, 1536), bucket=None, guard=None, allow_eval=False):
    """TokenStep of windows.Window objects (module docstring)."""
    wins = list(windows)
    if not wins:
        raise ValueError("no window to collate")
    for i, wd in enumerate(wins):
        if not allow_eval and is_cody_id(wd.source, wd.person_id):
            raise Refused(f"window {i}: CODY is evaluation-only ({wd.source} {wd.person_id})")
    if guard is not None:
        guard(wins)
    LS.assert_label_free(wins)
    wts = [TK.window_tokens(w) for w in wins]
    batcher = BucketBatcher(tuple(int(r) for r in rows), max_windows=max(len(wins), 1))
    packing = batcher.pack([w.n for w in wts], row_len=bucket)
    tokens = TK.tokenize(wts, row_len=packing.row_len, placement=packing.placement)
    anchors = torch.tensor([heading_patch(w) for w in wins], dtype=torch.int64)
    return TokenStep(tokens, anchors, int(packing.row_len), int(packing.n_rows), int(packing.useful),
                     int(packing.padded))


def produce(sampler, rows):
    """train/loop.Item of the sampler's next micro-step, collated by collate_tokens (module docstring)."""
    t0 = time.perf_counter()
    r0 = LP._render_seconds(sampler.synth)
    step = sampler.next()
    t1 = time.perf_counter()
    render = LP._render_seconds(sampler.synth) - r0
    ts = collate_tokens(step.windows, rows)
    t2 = time.perf_counter()
    return LP.Item(step, ts, LP.snapshot(sampler), {"sample": t1 - t0 - render, "render": render, "collate": t2 - t1})
