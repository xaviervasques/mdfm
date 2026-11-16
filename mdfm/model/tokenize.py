"""Tokenizer of the stage-1 encoder (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5 and the collate
fields of 2.2, task B4.14): the input streams of windows.Window band copies to the token rows the encoder
reads.

Tokens of a window of duration d (factory/layouts.window_tokens counts them the same way):
- P = min(40, ceil(d / 0.25)) latent tokens at the patch centres (p + 0.5) x 0.25 s, one learned vector;
- one subject token at the window centre, one learned vector;
- one measurement token per channel group and 0.25 s patch of each input stream that holds at least one
  sample in the patch (patchify): an IMU site gives an acc group and a gyro group when those channels
  are present, a pose stream one group of its 15 joints; at the patch centre;
- one nuisance token per IMU site and per pose stream, at the window centre, typed NUISANCE.
Order within a window: latent, subject, measurement (stream, group, patch), nuisance (stream, site).
Target streams (force, EMG) carry no token; an input stream of another modality is refused.

Stem inputs (patchify): the native samples of the band copy inside the 0.5 s span centred on the
patch, [c - 0.25, c + 0.25) s (half overlap with both neighbours), at most N_MAX = 100: band copies run
at 100 to 199 Hz after decimation (data/window_store.band_copy) and cameras below, so a stream at 200 Hz
or more is not a band copy and is refused (the 20 to 250 Hz band is stage 2). Fixed scalings: acc / 1 g,
gyro / 200 deg/s; pose in canonical torso units, pose3d turned about +y into the hip-heading frame at the
window centre (body.frames.hips_heading at the sample nearest d / 2, both hips valid with x and z present;
left unturned when no sample has both), pose2d as recorded (image y up). vals (n, C): channels joint-major for pose (C = 45
for pose3d; 30 for pose2d, x and y), tau (n,) s from the patch centre, quad (n,) s the quadrature weights
(quadrature), smask (n, C) the sample and channel validity (the stream's valid and chan_mask); vals holds
0 wherever smask is False.

ContinuousStem(c_in, d=512, freqs_hz=(0, 4, ..., 40), hidden=1024). Per channel, with the weights w_i =
h(tau_i) x quad_i of the valid samples, h(tau) = cos^2(pi tau / 0.5 s) the Hann window and quad_i the
sample's quadrature weight in s (quadrature: its sampling cell, one period inside a run, times a smooth
taper to 0 over the first and last EDGE_S = 0.25 s of the window, so that a span cut by the window's start
or end still integrates a smooth weight): the projections sum_i w_i x_i exp(-2 pi j f_k tau_i) / sum_i w_i
(real parts at the 11 frequencies, imaginary parts at the 10 non-zero ones; the f = 0 real part is the
masked mean), the weighted standard deviation, the weighted least-squares slope times 0.25 s, and the
coverage sum_i w_i / 0.25 s (1 for a full span away from the window's edges, 0 for an absent channel). Per
token, one flag per frequency above 0.8 x the stream's Nyquist, whose projections are zeroed: a band copy
at 100 Hz keeps 40 Hz, exactly 0.8 x its 50 Hz Nyquist; a 30 fps camera keeps 0 to 12 Hz. The flags only
fire below 100 Hz (cameras), so the rate reaches a measurement token only as the band it can carry. Then
Linear, GELU, Linear. Hann-weighted sums over native samples approximate the same integrals at any rate,
so per-sample fps needs no resampling; an absent channel or an invalid sample contributes nothing,
whatever value it stores. The features run in float32 outside autocast, the Linear layers under the
caller's autocast.

Embeddings: a measurement token adds the stem embedding (imu_acc, imu_gyro, pose2d, pose3d: modality,
sensor type and dim) and the site embedding (the 14 IMU sites, one row for pose). NuisanceEmbed: one table
per descriptor field (NUIS_FIELDS: kind, layout family, device family, rate bucket, acc_content, dim,
camera family, site), summed; each table holds the field's vocabulary plus 'none' (the field does not
apply) and 'unknown'. The layout, device and camera vocabularies (NAMED_FIELDS) can be a run's own name tables
(train/names.py, task S2.3): window_tokens, tokenize, nuisance_descriptors, NuisanceEmbed and Tokenizer take
nuis_fields (nuisance_fields checks it), None giving NUIS_FIELDS, the stage-1 vocabularies. The descriptor comes
from the stream meta (layout, device, camera, acc_content, site_ids, dim) and the band-copy rate; a real stream's
device family is its layout family's (factory/layouts.FAMILY_RULES). The nuisance tokens are the only place a
nuisance value enters the encoder input, and masks.build_mask walls them off from every body position. Nothing
else of a window reaches the input: no covariate (cov), label, source, person, key, truth or synthetic flag;
TokenBatch has none of those fields.

TokenBatch (tokenize): the encoder input of a micro-step, windows placed in rows of L tokens: t, seg,
ttype, kind (R, L); one StemInput per stem type; the nuisance descriptors; the gathers lat_idx (W, P_max)
and subj_idx (W,) as flat indices row x L + position (-1 beyond a window's P), and per nuisance token
its window, stream and site. Placement: one window per row by default, windows in order packed into rows
of row_len, or explicit (row, offset) pairs (the collate of B4.17 packs by bucket).
"""
import math
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn

from ..body.frames import hips_heading
from ..factory import layouts as LY
from ..factory.imu_render import SITES
from . import masks

PATCH_S = LY.PATCH_S                     # 0.25 s
SPAN_S = 0.5                             # Hann span centred on the patch
N_MAX = 100                              # samples per span of a band copy (below 200 Hz)
MAX_FPS = 200.0                          # a stream at this rate or above is not a band copy
EDGE_S = 0.25                            # window-edge taper of the quadrature weights (quadrature)
FREQS_HZ = tuple(float(f) for f in range(0, 41, 4))
NYQ_KEEP = 0.8
ACC_SCALE_G = 1.0
GYRO_SCALE_DPS = 200.0
N_SITES = len(SITES)                     # site id N_SITES: none (pose)
STEMS = ("imu_acc", "imu_gyro", "pose2d", "pose3d")
STEM_CHANNELS = {"imu_acc": 3, "imu_gyro": 3, "pose2d": 30, "pose3d": 45}
INPUT_MODALITIES = ("imu", "pose2d", "pose3d")
# token kinds (TokenBatch.kind)
LATENT, SUBJECT, MEASUREMENT, NUISANCE, PADDING = range(5)
KINDS = ("latent", "subject", "measurement", "nuisance", "padding")
# nuisance descriptor vocabularies; a code equal to len(vocabulary) is 'none', len + 1 'unknown'
RATE_EDGES_HZ = (30.0, 60.0, 100.0, 150.0)
NUIS_FIELDS = {
    "kind": ("imu_site", "pose2d", "pose3d"),
    "layout": ("xsens13", "xsens13_insoles", "nonan3", "lower_back", "g1_six", "wrist_L", "wrist_acc",
               "random_sites", "camera1", "mocap3d", "mocap3d_plate"),
    "device": ("generic", "xsens13"),
    "rate": ("below_30", "30_to_60", "60_to_100", "100_to_150", "150_to_200"),
    "acc_content": ("with_gravity", "linear_no_gravity"),
    "dim": (2, 3),
    "camera": ("generic",),
    "site": tuple(SITES),
}
FIELD_NAMES = tuple(NUIS_FIELDS)
# fields whose vocabularies a run's name tables set (train/names.py, NUIS_TABLES); the others are fixed here
NAMED_FIELDS = ("layout", "device", "camera")


def nuisance_fields(nuis_fields=None):
    """The nuisance descriptor vocabularies of a run: NUIS_FIELDS itself when nuis_fields is None or NUIS_FIELDS;
    otherwise {field: tuple of names} with the fields of NUIS_FIELDS in its order, each vocabulary non-empty
    without a repeated name, and every field outside NAMED_FIELDS equal to NUIS_FIELDS's (the kind, rate, dim and
    site codes are read by name elsewhere: mfm.nuisance_from_hidden, model/jepa.py, rate_bucket); ValueError
    otherwise. A table's 'none' and 'unknown' codes are len(vocabulary) and len + 1 of the vocabulary given
    (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 1 item 5 and 4.1)."""
    if nuis_fields is None or nuis_fields is NUIS_FIELDS:
        return NUIS_FIELDS
    out = {str(k): tuple(v) for k, v in dict(nuis_fields).items()}
    if tuple(out) != FIELD_NAMES:
        raise ValueError(f"nuisance fields {list(out)}, expected {list(FIELD_NAMES)} in this order")
    for k, v in out.items():
        if not v or len(set(v)) != len(v):
            raise ValueError(f"nuisance field {k}: a non-empty vocabulary without repeats, got {list(v)}")
        if k not in NAMED_FIELDS and v != NUIS_FIELDS[k]:
            raise ValueError(f"nuisance field {k} is fixed ({list(NUIS_FIELDS[k])}), got {list(v)}")
    return out


# ---------------------------------------------------------------- host side: one window
@dataclass
class Group:
    """Measurement tokens of one channel group of a stream: stem name, site id (N_SITES for pose), patch
    (n_p,) patch indices, vals (n_p, N_MAX, C) float32, tau (n_p, N_MAX) float32 s, quad (n_p, N_MAX)
    float32 s, smask (n_p, N_MAX, C) bool, nyq (Hz) of the stream."""
    stem: str
    site: int
    patch: np.ndarray
    vals: np.ndarray
    tau: np.ndarray
    quad: np.ndarray
    smask: np.ndarray
    nyq: float


def _check_stream(stream):
    if stream.modality not in INPUT_MODALITIES:
        raise ValueError(f"no input stem for a {stream.modality} stream (inputs: {INPUT_MODALITIES})")
    fps = float(stream.fps)
    if not (math.isfinite(fps) and 0 < fps < MAX_FPS):
        raise ValueError(f"{stream.modality} stream at {fps:g} Hz is not a band copy (below {MAX_FPS:g} Hz); "
                         f"pass it through data.window_store.band_copy")
    return fps


def pose3d_heading(values, valid, t, duration):
    """Heading (rad, about +y from +z) of a canonical 3D pose stream at its sample nearest duration / 2
    (body.frames.hips_heading); 0.0 when no sample has both hips valid."""
    if not len(t):
        return 0.0
    c = int(np.argmin(np.abs(np.asarray(t, dtype=np.float64) - duration / 2)))
    try:
        return float(hips_heading(torch.as_tensor(np.asarray(values, dtype=np.float64)),
                                  torch.as_tensor(np.asarray(valid, dtype=bool)), centre=c))
    except ValueError:
        return 0.0


def quadrature(t, fps, duration, edge_s=EDGE_S):
    """(T,) quadrature weight of each sample in s: its sampling cell, from the midpoint with the previous
    sample to the midpoint with the next, each half at most one period (a sample beside a hole in the
    time axis does not stand for the hole) and inside [0, duration); the first cell reaches half a period
    before the first sample, the last half a period after the last. One period in a regular run; a
    jittered camera frame gets its own cell. Times the window-edge taper
    sin^2(pi / 2 min(1, t / edge_s)) sin^2(pi / 2 min(1, (duration - t) / edge_s)). The taper takes the
    weights smoothly to 0 at the window's start and end, where the spans of the first and last patches
    are cut: a cut where the Hann weight is still large makes the sums depend on where the samples fall
    relative to it: without the taper, the edge tokens of one analytic IMU signal sampled at 100 and 128 Hz
    differ by more than the 2 percent that tests/test_tokenize.py allows."""
    t = np.asarray(t, dtype=np.float64)
    if not len(t):
        return np.zeros(0)
    h = 1.0 / float(fps)
    left, right = np.empty_like(t), np.empty_like(t)
    mid = (t[1:] + t[:-1]) / 2
    left[0] = max(0.0, t[0] - h / 2)
    left[1:] = np.maximum(mid, t[1:] - h)
    right[-1] = min(float(duration), t[-1] + h / 2)
    right[:-1] = np.minimum(mid, t[:-1] + h)
    quad = np.clip(right - left, 0.0, None)
    if edge_s > 0:
        a = np.clip(t / edge_s, 0.0, 1.0)
        b = np.clip((float(duration) - t) / edge_s, 0.0, 1.0)
        quad = quad * np.sin(np.pi / 2 * a) ** 2 * np.sin(np.pi / 2 * b) ** 2
    return quad


def _turn_y(values, cm, theta):
    """values (T, 15, 3) turned about +y by -theta (heading theta to +z); a joint without both x and z
    in cm (15, 3) keeps neither."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])       # rot_y(-theta)
    out = np.asarray(values, dtype=np.float64) @ R.T
    cm = np.asarray(cm, dtype=bool).copy()
    both = cm[:, 0] & cm[:, 2]
    cm[:, 0] = cm[:, 2] = both
    return out, cm


def patchify(stream, duration, patch_s=PATCH_S):
    """[Group] of an input stream (windows.Stream band copy, times from the window start) in a window of
    `duration` s (module docstring); one group per IMU site and channel kind present, one per pose stream,
    each with the patches p < P that hold at least one sample."""
    fps = _check_stream(stream)
    t = np.asarray(stream.t, dtype=np.float64)
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{stream.modality} stream times are not strictly increasing")
    values = np.asarray(stream.values)
    valid = np.asarray(stream.valid, dtype=bool)
    cm = np.asarray(stream.chan_mask, dtype=bool)
    P = LY.n_patches(duration, patch_s)
    pk = np.floor(t / patch_s).astype(np.int64)
    patch = np.unique(pk[(pk >= 0) & (pk < P)])
    if not len(patch):
        return []
    c = (patch + 0.5) * patch_s
    a = np.searchsorted(t, c - SPAN_S / 2, side="left")
    b = np.searchsorted(t, c + SPAN_S / 2, side="left")
    n = b - a
    if n.max() > N_MAX:
        raise ValueError(f"{stream.modality} stream at {fps:g} Hz puts {n.max()} samples in a {SPAN_S} s span "
                         f"(at most {N_MAX})")
    j = np.arange(N_MAX)
    inside = j[None] < n[:, None]                                    # (n_p, N_MAX)
    idx = np.minimum(a[:, None] + j[None], len(t) - 1)
    tau = np.where(inside, t[idx] - c[:, None], 0.0).astype(np.float32)
    quad = np.where(inside, quadrature(t, fps, duration)[idx], 0.0).astype(np.float32)
    nyq = fps / 2.0
    out = []
    if stream.modality == "imu":
        sites = list(stream.meta["site_ids"])
        for g, site in enumerate(sites):
            ok = inside & valid[idx, g]
            for stem, sl, scale in (("imu_acc", slice(0, 3), ACC_SCALE_G), ("imu_gyro", slice(3, 6), GYRO_SCALE_DPS)):
                chan = cm[g, sl]
                if not chan.any():
                    continue
                sm = ok[..., None] & chan[None, None]
                v = np.where(sm, values[idx, g, sl] / scale, 0.0).astype(np.float32)
                out.append(Group(stem, int(site), patch, v, tau, quad, sm, nyq))
        return out
    if stream.modality == "pose3d":
        # masked or invalid entries set to 0 before the turn, which would spread a stored NaN over the other
        # coordinates of its joint; the heading reads hips whose x and z are both present
        keep = valid[..., None] & cm[None]
        values = np.where(keep, values, 0.0)
        theta = pose3d_heading(values, valid & cm[:, 0] & cm[:, 2], t, duration)
        x, cm = _turn_y(values, cm, theta)
        C = 3
        stem = "pose3d"
    else:
        x, cm, C, stem = values[..., :2], cm[:, :2], 2, "pose2d"
    sm = inside[..., None, None] & valid[idx][..., None] & cm[None, None]   # (n_p, N_MAX, 15, C)
    v = np.where(sm, x[idx], 0.0).astype(np.float32)
    k = x.shape[1] * C
    out.append(Group(stem, N_SITES, patch, v.reshape(len(patch), N_MAX, k), tau, quad, sm.reshape(len(patch), N_MAX, k),
                     nyq))
    return out


def _code(field, value, vocabs=NUIS_FIELDS):
    vocab = vocabs[field]
    if value is None:
        return len(vocab)
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab) + 1


def rate_bucket(fps):
    """Name of the rate bucket of a stream rate in Hz (RATE_EDGES_HZ)."""
    return NUIS_FIELDS["rate"][int(np.searchsorted(RATE_EDGES_HZ, float(fps), side="right"))]


def nuisance_descriptors(stream, nuis_fields=None):
    """(n, len(FIELD_NAMES)) int64 descriptor codes of the nuisance tokens of an input stream: one row per
    IMU site, one for a pose stream (module docstring); nuis_fields: the run's vocabularies (nuisance_fields;
    None: NUIS_FIELDS)."""
    vocabs = nuisance_fields(nuis_fields)
    fps = _check_stream(stream)
    m = stream.meta or {}
    layout = m.get("layout")
    rule = LY.FAMILY_RULES.get(layout, {})
    row = {"layout": _code("layout", layout, vocabs), "rate": _code("rate", rate_bucket(fps), vocabs)}
    if stream.modality == "imu":
        device = m.get("device") or rule.get("device")
        row.update(kind=_code("kind", "imu_site", vocabs), device=_code("device", device or "unknown", vocabs),
                   acc_content=_code("acc_content", m.get("acc_content") or "with_gravity", vocabs),
                   dim=_code("dim", None, vocabs), camera=_code("camera", None, vocabs))
        rows = []
        for site in m["site_ids"]:
            r = dict(row, site=_code("site", SITES[int(site)], vocabs))
            rows.append([r[f] for f in FIELD_NAMES])
        return np.asarray(rows, dtype=np.int64).reshape(-1, len(FIELD_NAMES))
    dim = 2 if stream.modality == "pose2d" else 3
    camera = (m.get("camera") or "unknown") if dim == 2 else None
    row.update(kind=_code("kind", stream.modality, vocabs), device=_code("device", None, vocabs),
               acc_content=_code("acc_content", None, vocabs), dim=_code("dim", dim, vocabs),
               camera=_code("camera", camera, vocabs), site=_code("site", None, vocabs))
    return np.asarray([[row[f] for f in FIELD_NAMES]], dtype=np.int64)


@dataclass
class WindowTokens:
    """Tokens of one window, host side: t (n,) float32 s, ttype (n,) and kind (n,) int64, P latent tokens
    first and the subject token at position P; groups [(Group, first position)]; nuis (Nn, F) int64
    descriptors with their positions nuis_pos, stream indices nuis_stream (into Window.streams) and site
    ids nuis_site (N_SITES for pose)."""
    n: int
    P: int
    t: np.ndarray
    ttype: np.ndarray
    kind: np.ndarray
    groups: list
    nuis: np.ndarray
    nuis_pos: np.ndarray
    nuis_stream: np.ndarray
    nuis_site: np.ndarray


def window_tokens(window, nuis_fields=None):
    """WindowTokens of a windows.Window (module docstring). Reads the input streams' times, values, masks,
    fps and meta (layout, device, camera, acc_content, site_ids) and the duration, nothing else. nuis_fields:
    the vocabularies of the nuisance descriptors (nuisance_fields; None: NUIS_FIELDS)."""
    vocabs = nuisance_fields(nuis_fields)
    d = float(window.duration)
    P = LY.n_patches(d)
    t = [(np.arange(P) + 0.5) * PATCH_S, [d / 2]]
    kind = [np.full(P, LATENT), [SUBJECT]]
    pos = P + 1
    groups, nuis, nuis_stream, nuis_site = [], [], [], []
    inputs = [(k, s) for k, s in enumerate(window.streams) if s.role == "input"]
    for k, s in inputs:
        for g in patchify(s, d):
            groups.append((g, pos))
            t.append((g.patch + 0.5) * PATCH_S)
            kind.append(np.full(len(g.patch), MEASUREMENT))
            pos += len(g.patch)
    for k, s in inputs:
        desc = nuisance_descriptors(s, vocabs)
        nuis.append(desc)
        nuis_stream.append(np.full(len(desc), k))
        nuis_site.append(np.asarray(s.meta["site_ids"], dtype=np.int64) if s.modality == "imu" else [N_SITES])
    nn_ = sum(len(x) for x in nuis)
    t.append(np.full(nn_, d / 2))
    kind.append(np.full(nn_, NUISANCE))
    kind = np.concatenate([np.asarray(x, dtype=np.int64) for x in kind])
    ttype = np.where(kind == NUISANCE, masks.NUISANCE, masks.BODY).astype(np.int64)
    F_ = len(FIELD_NAMES)
    times = np.concatenate([np.asarray(x, dtype=np.float64) for x in t]).astype(np.float32)
    return WindowTokens(n=len(kind), P=P, t=times, ttype=ttype, kind=kind, groups=groups,
                        nuis=np.concatenate(nuis) if nuis else np.zeros((0, F_), np.int64),
                        nuis_pos=np.arange(pos, pos + nn_, dtype=np.int64),
                        nuis_stream=np.concatenate(nuis_stream).astype(np.int64) if nuis else np.zeros(0, np.int64),
                        nuis_site=np.concatenate(nuis_site).astype(np.int64) if nuis else np.zeros(0, np.int64))


# ---------------------------------------------------------------- batch
@dataclass
class StemInput:
    """Measurement tokens of one stem type in a micro-step: vals (N, N_MAX, C) float32 in stem units, tau
    (N, N_MAX) float32 s from the patch centre, quad (N, N_MAX) float32 s the samples' quadrature weights
    (quadrature), smask (N, N_MAX, C) bool, nyq (N,) float32 Hz, site (N,) int64 (N_SITES for pose), row and
    pos (N,) int64, the token's place."""
    vals: torch.Tensor
    tau: torch.Tensor
    quad: torch.Tensor
    smask: torch.Tensor
    nyq: torch.Tensor
    site: torch.Tensor
    row: torch.Tensor
    pos: torch.Tensor

    def to(self, device):
        return StemInput(*(getattr(self, f.name).to(device) for f in fields(self)))


@dataclass
class TokenBatch:
    """Encoder input of a micro-step (module docstring): t (R, L) float32 s from each window's start, seg
    (R, L) int64 window index or -1, ttype and kind (R, L) int64, stems {name: StemInput}, nuis_desc (Nn,
    F) int64 with nuis_row, nuis_pos, nuis_win, nuis_stream, nuis_site (Nn,) int64, lat_idx (W, P_max)
    and subj_idx (W,) flat indices row x L + position, n_patches (W,) int64."""
    t: torch.Tensor
    seg: torch.Tensor
    ttype: torch.Tensor
    kind: torch.Tensor
    stems: dict
    nuis_desc: torch.Tensor
    nuis_row: torch.Tensor
    nuis_pos: torch.Tensor
    nuis_win: torch.Tensor
    nuis_stream: torch.Tensor
    nuis_site: torch.Tensor
    lat_idx: torch.Tensor
    subj_idx: torch.Tensor
    n_patches: torch.Tensor

    @property
    def shape(self):
        return tuple(self.t.shape)

    @property
    def n_windows(self):
        return int(self.subj_idx.shape[0])

    def mask(self, typed=True):
        """masks.build_mask of the rows, (R, 1, L, L) bool on the batch's device."""
        return masks.build_mask(self.seg, self.ttype, typed=typed)

    def to(self, device):
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = {k: s.to(device) for k, s in v.items()} if f.name == "stems" else v.to(device)
        return TokenBatch(**kw)


def place(lengths, row_len=None):
    """(row, offset) per window: one window per row when row_len is None, else windows in order packed
    into rows of row_len tokens (a new row when the next window does not fit)."""
    if row_len is None:
        return [(i, 0) for i in range(len(lengths))]
    out, r, o = [], 0, 0
    for n in lengths:
        if n > row_len:
            raise ValueError(f"a window of {n} tokens does not fit a row of {row_len}")
        if o + n > row_len:
            r, o = r + 1, 0
        out.append((r, o))
        o += n
    return out


def tokenize(windows, row_len=None, placement=None, nuis_fields=None):
    """TokenBatch of windows (windows.Window or WindowTokens), placed by `placement` [(row, offset)] in rows
    of row_len tokens, or by place(lengths, row_len); with neither, one window per row of the longest
    window's length. nuis_fields: the descriptor vocabularies of the Window objects' tokens (window_tokens;
    a WindowTokens carries the codes it was built with)."""
    wts = [w if isinstance(w, WindowTokens) else window_tokens(w, nuis_fields) for w in windows]
    if not wts:
        raise ValueError("no window to tokenize")
    lengths = [w.n for w in wts]
    if placement is None:
        placement = place(lengths, row_len)
    if len(placement) != len(wts):
        raise ValueError(f"{len(placement)} placements for {len(wts)} windows")
    L = int(row_len) if row_len is not None else max(o + n for (_, o), n in zip(placement, lengths))
    R = max(r for r, _ in placement) + 1
    t = np.zeros((R, L), np.float32)
    seg = np.full((R, L), -1, np.int64)
    ttype = np.zeros((R, L), np.int64)
    kind = np.full((R, L), PADDING, np.int64)
    Pmax = max(w.P for w in wts)
    lat = np.full((len(wts), Pmax), -1, np.int64)
    subj = np.zeros(len(wts), np.int64)
    parts = {s: [] for s in STEMS}
    nuis = {k: [] for k in ("desc", "row", "pos", "win", "stream", "site")}
    for i, (w, (r, o)) in enumerate(zip(wts, placement)):
        if o < 0 or o + w.n > L or r < 0:
            raise ValueError(f"window {i} ({w.n} tokens) at row {r} offset {o} leaves a row of {L}")
        if (seg[r, o:o + w.n] >= 0).any():
            raise ValueError(f"window {i} overlaps another window in row {r}")
        sl = slice(o, o + w.n)
        t[r, sl], seg[r, sl], ttype[r, sl], kind[r, sl] = w.t, i, w.ttype, w.kind
        lat[i, :w.P] = r * L + o + np.arange(w.P)
        subj[i] = r * L + o + w.P
        for g, p0 in w.groups:
            m = len(g.patch)
            parts[g.stem].append((g, np.full(m, r), o + p0 + np.arange(m)))
        nuis["desc"].append(w.nuis)
        nuis["row"].append(np.full(len(w.nuis), r))
        nuis["pos"].append(o + w.nuis_pos)
        nuis["win"].append(np.full(len(w.nuis), i))
        nuis["stream"].append(w.nuis_stream)
        nuis["site"].append(w.nuis_site)
    stems = {}
    for s in STEMS:
        C = STEM_CHANNELS[s]
        ps = parts[s]
        if not ps:
            stems[s] = StemInput(torch.zeros(0, N_MAX, C), torch.zeros(0, N_MAX), torch.zeros(0, N_MAX),
                                 torch.zeros(0, N_MAX, C, dtype=torch.bool), torch.zeros(0),
                                 torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64),
                                 torch.zeros(0, dtype=torch.int64))
            continue
        cat = lambda xs: torch.from_numpy(np.ascontiguousarray(np.concatenate(xs)))  # noqa: E731
        stems[s] = StemInput(vals=cat([g.vals for g, _, _ in ps]), tau=cat([g.tau for g, _, _ in ps]),
                             quad=cat([g.quad for g, _, _ in ps]), smask=cat([g.smask for g, _, _ in ps]),
                             nyq=cat([np.full(len(g.patch), g.nyq, np.float32) for g, _, _ in ps]),
                             site=cat([np.full(len(g.patch), g.site, np.int64) for g, _, _ in ps]),
                             row=cat([rr.astype(np.int64) for _, rr, _ in ps]),
                             pos=cat([pp.astype(np.int64) for _, _, pp in ps]))
    ni = lambda k: torch.from_numpy(np.concatenate(nuis[k]).astype(np.int64))  # noqa: E731
    return TokenBatch(t=torch.from_numpy(t), seg=torch.from_numpy(seg), ttype=torch.from_numpy(ttype),
                      kind=torch.from_numpy(kind), stems=stems,
                      nuis_desc=torch.from_numpy(np.concatenate(nuis["desc"]).astype(np.int64)
                                                 .reshape(-1, len(FIELD_NAMES))),
                      nuis_row=ni("row"), nuis_pos=ni("pos"), nuis_win=ni("win"), nuis_stream=ni("stream"),
                      nuis_site=ni("site"), lat_idx=torch.from_numpy(lat), subj_idx=torch.from_numpy(subj),
                      n_patches=torch.tensor([w.P for w in wts], dtype=torch.int64))


# ---------------------------------------------------------------- modules
def hann(tau, span_s=SPAN_S):
    """Hann weights cos^2(pi tau / span) inside |tau| < span / 2, 0 outside."""
    return torch.where(tau.abs() < span_s / 2, torch.cos(math.pi * tau / span_s) ** 2, torch.zeros_like(tau))


class ContinuousStem(nn.Module):
    """ContinuousStem(c_in, d=512, freqs_hz=FREQS_HZ, hidden=1024): Hann-weighted projections and moments
    of each channel over a span of native samples (module docstring), then Linear, GELU, Linear."""

    def __init__(self, c_in, d=512, freqs_hz=FREQS_HZ, hidden=1024):
        super().__init__()
        self.c_in = int(c_in)
        f = torch.tensor([float(x) for x in freqs_hz], dtype=torch.float32)
        self.register_buffer("freqs", f, persistent=False)
        self.register_buffer("nonzero", f > 0, persistent=False)
        self.n_freqs = len(f)
        self.per_channel = len(f) + int((f > 0).sum()) + 3
        self.n_features = self.c_in * self.per_channel + len(f)
        self.mlp = nn.Sequential(nn.Linear(self.n_features, hidden), nn.GELU(), nn.Linear(hidden, d))

    def features(self, vals, tau, smask, nyq, quad=None):
        """(N, n_features) float32 (float64 for a float64 stem) of the spans vals (N, n, C), tau (N, n) s,
        smask (N, n, C), nyq (N,) Hz and quad (N, n) s (quadrature; None: one period 1 / (2 nyq) per
        sample, no taper): per channel (C, per_channel) flattened channel-major, then the Nyquist flags
        (N, n_freqs)."""
        N, n, C = vals.shape
        if C != self.c_in:
            raise ValueError(f"stem of {self.c_in} channels fed {C}")
        dt = torch.float64 if self.mlp[0].weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=vals.device.type, enabled=False):
            sm = smask.to(torch.bool)
            zero = torch.zeros((), device=vals.device, dtype=dt)
            x = torch.where(sm, vals.to(dt), zero)
            tau, nyq = tau.to(dt), nyq.to(dt)
            freqs = self.freqs.to(dt)
            quad = (1.0 / (2.0 * nyq))[:, None].expand_as(tau) if quad is None else quad.to(dt)
            wm = (hann(tau) * quad)[..., None] * sm                     # (N, n, C) s
            sw = wm.sum(1)                                              # (N, C) s
            inv = torch.where(sw > 0, 1.0 / sw.clamp_min(1e-30), torch.zeros_like(sw))
            ph = (2.0 * math.pi) * tau[..., None] * freqs               # (N, n, K)
            wx = wm * x
            re = torch.einsum("nik,nic->nck", torch.cos(ph), wx) * inv[..., None]
            im = -torch.einsum("nik,nic->nck", torch.sin(ph), wx) * inv[..., None]
            keep = freqs[None] <= NYQ_KEEP * nyq[:, None] * (1.0 + 1e-6)   # (N, K)
            re = re * keep[:, None]
            im = (im * keep[:, None])[..., self.nonzero]
            mean = wx.sum(1) * inv
            dx = torch.where(sm, x - mean[:, None], zero)
            var = (wm * dx * dx).sum(1) * inv
            std = torch.where(var > 0, var.clamp_min(1e-30).sqrt(), torch.zeros_like(var))
            tbar = (wm * tau[..., None]).sum(1) * inv
            dtau = torch.where(sm, tau[..., None] - tbar[:, None], zero)
            stt = (wm * dtau * dtau).sum(1)
            stx = (wm * dtau * dx).sum(1)
            ok = stt > 1e-9 * sw * (SPAN_S / 2) ** 2
            slope = torch.where(ok, stx / stt.clamp_min(1e-30), torch.zeros_like(stt)) * PATCH_S
            cover = sw / (SPAN_S / 2)
            per = torch.cat([re, im, std[..., None], slope[..., None], cover[..., None]], -1)   # (N, C, per_channel)
            return torch.cat([per.reshape(N, -1), (~keep).to(dt)], -1)

    def forward(self, vals, tau, smask, nyq, quad=None):
        """(N, d) tokens of N spans (features)."""
        return self.mlp(self.features(vals, tau, smask, nyq, quad))


class NuisanceEmbed(nn.Module):
    """One embedding table per descriptor field (NUIS_FIELDS, or the run's vocabularies nuis_fields, see
    nuisance_fields; plus 'none' and 'unknown'), summed: desc (N, F) int64 -> (N, d). fields: the vocabularies
    of the tables."""

    def __init__(self, d=512, nuis_fields=None):
        super().__init__()
        self.fields = nuisance_fields(nuis_fields)
        self.tables = nn.ModuleList(nn.Embedding(len(v) + 2, d) for v in self.fields.values())

    def forward(self, desc):
        if desc.shape[-1] != len(self.tables):
            raise ValueError(f"descriptor of {desc.shape[-1]} fields, expected {len(self.tables)}")
        return sum(tab(desc[:, f]) for f, tab in enumerate(self.tables))


class Tokenizer(nn.Module):
    """Stems, embeddings and learned token vectors: forward(TokenBatch) -> x (R, L, d) float32 (float64 for
    a float64 module), zeros at padding (module docstring). LayerNorm or none, never BatchNorm. nuis_fields:
    the nuisance descriptor vocabularies (NuisanceEmbed; None: NUIS_FIELDS)."""

    def __init__(self, d=512, hidden=1024, freqs_hz=FREQS_HZ, nuis_fields=None):
        super().__init__()
        self.d = d
        self.stems = nn.ModuleDict({s: ContinuousStem(STEM_CHANNELS[s], d, freqs_hz, hidden) for s in STEMS})
        self.stem_embed = nn.Embedding(len(STEMS), d)
        self.site_embed = nn.Embedding(N_SITES + 1, d)
        self.nuisance = NuisanceEmbed(d, nuis_fields)
        self.latent = nn.Parameter(torch.zeros(d))
        self.subject = nn.Parameter(torch.zeros(d))
        for p in [self.stem_embed.weight, self.site_embed.weight, self.latent, self.subject] + \
                [tab.weight for tab in self.nuisance.tables]:
            nn.init.normal_(p, std=0.02)

    def forward(self, batch):
        R, L = batch.t.shape
        dev = self.latent.device
        dt = torch.float64 if self.latent.dtype == torch.float64 else torch.float32
        kind = batch.kind.reshape(-1).to(dev)
        x = torch.zeros(R * L, self.d, device=dev, dtype=dt)
        lat = torch.nonzero(kind == LATENT)[:, 0]
        x = x.index_copy(0, lat, self.latent.to(dt).expand(len(lat), -1))
        sub = torch.nonzero(kind == SUBJECT)[:, 0]
        x = x.index_copy(0, sub, self.subject.to(dt).expand(len(sub), -1))
        for k, s in enumerate(STEMS):
            si = batch.stems.get(s)
            if si is None or not len(si.row):
                continue
            h = self.stems[s](si.vals.to(dev), si.tau.to(dev), si.smask.to(dev), si.nyq.to(dev), si.quad.to(dev))
            h = h.to(dt) + self.stem_embed.weight[k].to(dt) + self.site_embed(si.site.to(dev)).to(dt)
            x = x.index_copy(0, si.row.to(dev) * L + si.pos.to(dev), h)
        if len(batch.nuis_row):
            h = self.nuisance(batch.nuis_desc.to(dev)).to(dt)
            x = x.index_copy(0, batch.nuis_row.to(dev) * L + batch.nuis_pos.to(dev), h)
        return x.view(R, L, self.d)
