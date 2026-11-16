"""Tests of the tokenizer stems and the rotary positions (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task
B4.14): one analytic signal sampled at 100 and 128 Hz gives tokens within 2 percent; content above 0.8 x the
Nyquist is masked and flagged; a stream at 200 Hz or more (not a band copy) or with times not strictly increasing
is refused and a band copy puts at most 100 samples in a span; a constant time shift leaves attention scores unchanged; absent channels and invalid
samples give the same tokens as zeros; no BatchNorm in mdfm.model; token counts and order equal
factory/layouts.window_tokens on synthetic windows of the layout families (82, 122, 164, 527 and 1,094 tokens
for 10 s) and on real band copies (weargait_pd, gait120); no covariate, label, source or truth reaches the
tokens; pose3d tokens do not depend on the lab heading; placement and gathers.

Synthetic windows: a procedural walk (sinusoidal hips, knees and arms, 1 m/s) rendered by factory/synth.py, so
no test needs the body store or an AMASS take; real windows: the first admitted recording of each source through
mdfm.data.sources (canonical.load with its flags at False), windows.cut and window_store.band_copy.
"""
import dataclasses
import math
import os
import re

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.data import windows as W
from mdfm.data import window_store as WS
from mdfm.factory import layouts as L
from mdfm.model import masks as M
from mdfm.model import rope
from mdfm.model import tokenize as T

F64 = torch.float64
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
HAVE_INDEX = os.path.exists(os.path.join(paths.CACHE, "manifest", "record_index.csv.gz")) and \
    os.path.isdir(f"{paths.MOVEATLAS}/datasets")
needs_index = pytest.mark.skipif(not HAVE_INDEX, reason="record index or MoveAtlas not found")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(paths.TREE, "mdfm", "model")


# ---------------------------------------------------------------- analytic streams
def _times(fps, dur, phase):
    t = phase + np.arange(int(math.floor((dur - phase) * fps)) + 1) / fps
    return t[t < dur]


def _imu(fps, dur=10.0, phase=0.0, sites=(2, 7), freqs=(1.3, 4.8, 8.7), meta=None):
    """IMU stream of analytic signals in contract units (acc g with 1 g on y, gyro deg/s)."""
    t = _times(fps, dur, phase)
    v = np.zeros((len(t), len(sites), 9), np.float32)
    for g in range(len(sites)):
        for c in range(6):
            s = 0.1 * t / dur + (1.0 if c == 1 else 0.0)
            for k, f in enumerate(freqs):
                s = s + (0.3 / (k + 1)) * np.sin(2 * np.pi * f * t + g + c + k)
            v[:, g, c] = s * (150.0 if c >= 3 else 1.0)
    cm = np.zeros((len(sites), 9), bool)
    cm[:, :6] = True
    m = {"site_ids": list(sites), "acc_content": "with_gravity", "layout": "g1_six", "device": "xsens13"}
    m.update(meta or {})
    return W.Stream("imu", "input", float(fps), t.astype(np.float32), v, cm, np.ones((len(t), len(sites)), bool),
                    meta=m)


# canonical 15 joints in torso units, y up, facing +z (camera.CANONICAL_15 order)
_REST15 = np.array([[0, 0, 0], [0, 1, 0], [0, 1.35, 0.05], [0.35, 0.95, 0], [-0.35, 0.95, 0], [0.4, 0.45, 0],
                    [-0.4, 0.45, 0], [0.42, 0.0, 0.05], [-0.42, 0.0, 0.05], [0.2, -0.05, 0], [-0.2, -0.05, 0],
                    [0.2, -0.95, 0.02], [-0.2, -0.95, 0.02], [0.2, -1.85, 0], [-0.2, -1.85, 0]], dtype=np.float64)


def _pose(fps, dur=10.0, phase=0.0, dim=3, heading=0.0):
    """Canonical pose stream of a walking-like motion (limbs swinging at 0.9 Hz), turned by `heading` about +y."""
    t = _times(fps, dur, phase)
    x = np.repeat(_REST15[None], len(t), 0).copy()
    s = np.sin(2 * np.pi * 0.9 * t)[:, None]
    for j, sign in ((11, 1), (13, 1), (12, -1), (14, -1), (5, -1), (7, -1), (6, 1), (8, 1)):
        x[:, j, 2] += sign * 0.3 * s[:, 0] * (1.5 if j in (13, 14, 7, 8) else 1.0)
    x[:, :, 1] += 0.03 * np.sin(2 * np.pi * 1.8 * t)[:, None]
    x[:, 0] = 0.0
    c, sn = math.cos(heading), math.sin(heading)
    R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])                      # rot_y(heading)
    x = x @ R.T
    cm = np.ones((15, 3), bool)
    if dim == 2:
        x[..., 2] = 0.0
        cm[:, 2] = False
    meta = {"dim": dim, "space": "3d_meter" if dim == 3 else "2d_pixel", "scale": 0.5, "layout": "mocap3d" if dim == 3
            else "camera1", "unit_to_si": 1.0 if dim == 3 else None}
    if dim == 2:
        meta["camera"] = "generic"
    return W.Stream(f"pose{dim}d", "input", float(fps), t.astype(np.float32), x.astype(np.float32), cm,
                    np.ones((len(t), 15), bool), meta=meta)


def _window(streams, dur=10.0, **kw):
    w = W.Window("synthetic", "person", "key", True, 0.0, dur, streams, {"sex": 2, "age_band": 8, "task": 0})
    return dataclasses.replace(w, **kw)


def _tokens(tok, windows):
    with torch.no_grad():
        b = T.tokenize(windows)
        return b, tok(b)


@pytest.fixture(scope="module")
def tok():
    torch.manual_seed(0)
    return T.Tokenizer()


# ---------------------------------------------------------------- rate independence and the band
@pytest.mark.parametrize("kind", ["imu", "pose3d", "pose2d"])
def test_one_signal_at_100_and_128_hz_gives_tokens_within_2_percent(tok, kind):
    make = {"imu": _imu, "pose3d": lambda fps, dur, phase: _pose(fps, dur, phase, 3),
            "pose2d": lambda fps, dur, phase: _pose(fps, dur, phase, 2)}[kind]
    for dur, pa, pb in ((10.0, 0.0031, 0.0062), (7.93, 0.0087, 0.0011)):
        a, b = _window([make(100.0, dur, pa)], dur), _window([make(128.0, dur, pb)], dur)
        ba, xa = _tokens(tok, [a])
        bb, xb = _tokens(tok, [b])
        assert torch.equal(ba.kind, bb.kind) and torch.equal(ba.t, bb.t)
        meas = ba.kind == T.MEASUREMENT
        assert int(meas.sum()) > 30
        rel = (xa[meas] - xb[meas]).norm(dim=-1) / xa[meas].norm(dim=-1)
        assert rel.max() < 0.02, (kind, dur, float(rel.max()))


def test_rate_independence_holds_across_the_band_before_the_mlp(tok):
    """The stem features themselves (before the MLP and the embeddings, which could hide a difference) of content
    from 11 to 38 Hz, the band a 100 Hz copy carries: 100 Hz against 128 and 199 Hz within 2 percent over the whole
    feature vector, and the projections other than the mean within 2 percent of the channel's projection norm."""
    K = len(T.FREQS_HZ)
    band = (11.0, 19.0, 27.0, 33.0, 38.0)
    for dur, pa, pb in ((10.0, 0.0031, 0.0062), (7.93, 0.0087, 0.0011)):
        ref = T.tokenize([_window([_imu(100.0, dur, pa, sites=(2,), freqs=band)], dur)])
        for fps in (128.0, 199.0):
            other = T.tokenize([_window([_imu(fps, dur, pb, sites=(2,), freqs=band)], dur)])
            for name in ("imu_acc", "imu_gyro"):
                stem = tok.stems[name]
                fa, fb = (stem.features(s.vals.double(), s.tau.double(), s.smask, s.nyq.double(), s.quad.double())
                          for s in (ref.stems[name], other.stems[name]))
                rel = (fa - fb).norm(dim=-1) / fa.norm(dim=-1)
                assert rel.max() < 0.02, (name, dur, fps, float(rel.max()))
                pa_, pb_ = (f[:, :-K].reshape(len(f), stem.c_in, stem.per_channel)[..., 1:2 * K - 1] for f in (fa, fb))
                rel = (pa_ - pb_).norm(dim=-1) / pa_.norm(dim=-1)
                assert rel.max() < 0.02, (name, dur, fps, float(rel.max()))
                # the top of the band is kept at every rate: away from the window edges, the 40 Hz projection of the
                # 38 Hz tone (amplitude a) has the modulus a / 4 of the Hann response one bin (2 Hz) off its peak,
                # within the 33 Hz leak (about 2 percent); the 36 Hz one within the larger 33 Hz leak (about 43
                # percent). pa_ holds the real parts at 4 to 40 Hz (0 to 9), then the imaginary parts (10 to 19).
                a = 0.3 / 5 * (150.0 / T.GYRO_SCALE_DPS if name == "imu_gyro" else 1.0 / T.ACC_SCALE_G)
                for s, f in ((ref.stems[name], pa_), (other.stems[name], pb_)):
                    patch = s.pos - s.pos.min()                   # one site: the stem's tokens are patches 0 to P - 1
                    inner = (patch >= 4) & (patch < patch.max() - 3)
                    m40 = torch.sqrt(f[inner][..., 9] ** 2 + f[inner][..., 19] ** 2) / (a / 4)
                    m36 = torch.sqrt(f[inner][..., 8] ** 2 + f[inner][..., 18] ** 2) / (a / 4)
                    assert (m40 - 1).abs().max() < 0.05, (name, dur, fps, float((m40 - 1).abs().max()))
                    assert ((m36 > 0.5) & (m36 < 1.5)).all(), (name, dur, fps, float(m36.min()), float(m36.max()))


def test_content_above_the_nyquist_band_is_masked_and_flagged(tok):
    K = len(T.FREQS_HZ)
    for fps, kept in ((100.0, K), (128.0, K), (30.0, 4), (25.0, 3), (60.0, 7)):
        w = _window([_pose(fps, 4.0, 0.0, 2)], 4.0)
        si = T.tokenize([w]).stems["pose2d"]
        stem = tok.stems["pose2d"]
        f = stem.features(si.vals, si.tau, si.smask, si.nyq, si.quad)
        flags = f[:, -K:]
        want = torch.tensor([0.0] * kept + [1.0] * (K - kept))
        assert torch.equal(flags, want.expand_as(flags)), fps
        per = f[:, :-K].reshape(len(f), stem.c_in, stem.per_channel)
        re, im = per[..., :K], per[..., K:K + K - 1]
        assert (re[..., kept:] == 0).all() and (im[..., kept - 1:] == 0).all()
        if kept < K:
            assert (re[..., :kept].abs().sum() > 0) and (im[..., :kept - 1].abs().sum() > 0)
    # 40 Hz is exactly 0.8 x the 50 Hz Nyquist of a 100 Hz band copy and is kept
    assert T.NYQ_KEEP * 50.0 == max(T.FREQS_HZ)


def test_streams_that_are_not_band_copies_are_refused():
    with pytest.raises(ValueError, match="not a band copy"):
        T.window_tokens(_window([_imu(200.0, 2.0)], 2.0))
    with pytest.raises(ValueError, match="no input stem"):
        s = _imu(100.0, 2.0)
        T.window_tokens(_window([dataclasses.replace(s, modality="force")], 2.0))
    # unsorted or repeated sample times would break the span search: refused
    s = _imu(100.0, 2.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        T.patchify(dataclasses.replace(s, t=s.t[::-1].copy()), 2.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        T.patchify(dataclasses.replace(s, t=np.concatenate([s.t[:1], s.t[:-1]])), 2.0)
    # a 199 Hz copy fills a span with at most 100 samples
    g = T.patchify(_imu(199.0, 3.0, phase=0.0001), 3.0)
    assert 99 <= max(int(grp.smask[..., 0].sum(1).max()) for grp in g) <= T.N_MAX
    # a 500 Hz record passes after the band copy (k = 5, 100 Hz)
    raw = _imu(500.0, 3.0)
    copy = WS.band_copy(raw)
    assert copy.fps == 100.0
    assert T.window_tokens(_window([copy], 3.0)).n == L.window_tokens(_window([copy], 3.0))


def test_quadrature_weights():
    t = _times(100.0, 10.0, 0.004)
    q = T.quadrature(t, 100.0, 10.0)
    inner = (t > T.EDGE_S) & (t < 10.0 - T.EDGE_S)
    assert np.allclose(q[inner], 0.01, rtol=0, atol=1e-12)
    assert q[0] < 2e-5 and q[-1] < 2e-5         # the taper falls to 0 at both edges (samples 4 and 6 ms from them)
    # the taper integrates to EDGE_S / 2 at each end
    assert abs(q.sum() - (10.0 - T.EDGE_S)) < 1e-4
    # jittered frames tile the window with their own cells
    tj = t + np.random.default_rng(0).uniform(-0.002, 0.002, len(t))
    qj = T.quadrature(np.sort(tj), 100.0, 10.0, edge_s=0.0)
    assert qj.max() < 0.015 and qj.min() > 0.005 and abs(qj.sum() - 10.0) < 0.002
    # a sample beside a hole in the time axis stands for at most one period of it
    th = np.concatenate([t[t < 3.0], t[t > 4.0]])
    qh = T.quadrature(th, 100.0, 10.0, edge_s=0.0)
    k = int(np.flatnonzero(th < 3.0)[-1])
    assert abs(qh[k] - 0.015) < 1e-9 and abs(qh[k + 1] - 0.015) < 1e-9


# ---------------------------------------------------------------- rotary positions
def test_a_constant_time_shift_leaves_attention_scores_unchanged():
    g = torch.Generator().manual_seed(0)
    R, H, Ln, D = 2, 4, 50, 64
    q = torch.randn(R, H, Ln, D, generator=g, dtype=F64)
    k = torch.randn(R, H, Ln, D, generator=g, dtype=F64)
    t = torch.rand(R, Ln, generator=g, dtype=F64) * 10
    s0 = torch.einsum("rhqd,rhkd->rhqk", *rope.apply(q, k, t))
    s1 = torch.einsum("rhqd,rhkd->rhqk", *rope.apply(q, k, t + 3.7))
    assert torch.allclose(s0, s1, rtol=0, atol=1e-10)
    qf, kf = q.float(), k.float()
    s2 = torch.einsum("rhqd,rhkd->rhqk", *rope.apply(qf, kf, t.float()))
    s3 = torch.einsum("rhqd,rhkd->rhqk", *rope.apply(qf, kf, t.float() + 3.7))
    assert (s2 - s3).abs().max() < 1e-3 * s2.abs().max()
    # norms kept, positions matter (scores move with relative time), the periods span 0.25 to 64 s
    qr, _ = rope.apply(q, k, t)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), rtol=1e-12)
    s4 = torch.einsum("rhqd,rhkd->rhqk", *rope.apply(q, k, t * 1.5))
    assert (s4 - s0).abs().max() > 1e-2
    p = rope.periods(D)
    assert p[0] == pytest.approx(0.25) and p[-1] == pytest.approx(64.0) and len(p) == D // 2
    # bf16 queries rotate in float32 and come back in bf16
    qb, kb = rope.apply(q.bfloat16(), k.bfloat16(), t.float())
    assert qb.dtype == kb.dtype == torch.bfloat16


# ---------------------------------------------------------------- masks and absent channels
def test_absent_channels_and_invalid_samples_give_the_same_tokens_as_zeros(tok):
    base = _imu(100.0, 4.0, sites=(2, 7, 12))
    cm = base.chan_mask.copy()
    cm[1, 3:6] = False                                   # no gyro at the second site
    valid = base.valid.copy()
    valid[50:80, 0] = False                              # a gap at the first site
    zero = base.values.copy()
    zero[:, 1, 3:6] = 0.0
    zero[~valid] = 0.0
    junk = base.values.copy()
    junk[:, 1, 3:6] = np.nan
    junk[50:80, 0] = 1e6
    a = _window([dataclasses.replace(base, values=zero, chan_mask=cm, valid=valid)], 4.0)
    b = _window([dataclasses.replace(base, values=junk, chan_mask=cm, valid=valid)], 4.0)
    ba, xa = _tokens(tok, [a])
    bb, xb = _tokens(tok, [b])
    assert torch.equal(xa, xb) and torch.isfinite(xa).all()
    # the site without gyro has no gyro group: 3 acc groups and 2 gyro groups
    assert len(ba.stems["imu_acc"].row) == 3 * 16 and len(ba.stems["imu_gyro"].row) == 2 * 16
    # a missing pose joint likewise
    p = _pose(100.0, 3.0)
    v = p.valid.copy()
    v[:, 5] = False
    pz, pj = p.values.copy(), p.values.copy()
    pz[:, 5] = 0.0
    pj[:, 5] = -7.0
    xa = _tokens(tok, [_window([dataclasses.replace(p, values=pz, valid=v)], 3.0)])[1]
    xb = _tokens(tok, [_window([dataclasses.replace(p, values=pj, valid=v)], 3.0)])[1]
    assert torch.equal(xa, xb)
    # pose3d: a NaN in one masked coordinate of an otherwise present joint, and junk in a masked hip, do not
    # spread through the heading turn (which mixes x and z) nor steer the heading
    p = _pose(100.0, 3.0, heading=0.7)
    cm = p.chan_mask.copy()
    cm[5, 1] = False                                     # joint 5 without y
    cm[9] = False                                        # left hip absent
    pz, pj = p.values.copy(), p.values.copy()
    pz[:, 5, 1], pz[:, 9] = 0.0, 0.0
    pj[:, 5, 1], pj[:, 9] = np.nan, (5.0, -3.0, 40.0)
    xa = _tokens(tok, [_window([dataclasses.replace(p, values=pz, chan_mask=cm)], 3.0)])[1]
    xb = _tokens(tok, [_window([dataclasses.replace(p, values=pj, chan_mask=cm)], 3.0)])[1]
    assert torch.isfinite(xb).all() and torch.equal(xa, xb)


def test_no_batchnorm_in_mdfm_model():
    from mdfm.model.encoder import Encoder
    for m in (T.Tokenizer(d=64, hidden=64), Encoder(d=64, layers=1, heads=4)):
        assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
    for name in os.listdir(MODEL_DIR):
        if name.endswith(".py") and not name.startswith("._"):
            with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as f:
                text = f.read()
            assert "BatchNorm" not in text.replace("never BatchNorm", "").replace("not BatchNorm", ""), name


# ---------------------------------------------------------------- what reaches the tokens
FORBIDDEN_FIELDS = re.compile(r"^(label|group|cohort|severity|diagnosis|phase|manifest|tier|lockbox|scope|lane|"
                              r"notes|aux|cov|sex|age|task|source|person|key|truth|body|synthetic)", re.I)


def test_no_covariate_label_source_or_truth_reaches_the_tokens(tok):
    s = _imu(100.0, 5.0)
    p = _pose(100.0, 5.0)
    a = _window([s, p], 5.0)
    truth = {"mount": np.eye(3), "label": "LEAK"}
    b = _window([dataclasses.replace(s, truth=truth, meta=dict(s.meta, source="weargait_pd")),
                 dataclasses.replace(p, meta=dict(p.meta, source="gait120"))], 5.0,
                source="weargait_pd", person_id="weargait_pd:X", key="other", synthetic=False,
                cov={"sex": 1, "age_band": 3, "task": 5}, body={"betas": np.ones(16)})
    ba, xa = _tokens(tok, [a])
    bb, xb = _tokens(tok, [b])
    assert torch.equal(xa, xb)
    for f in dataclasses.fields(ba):
        if f.name == "stems":
            for name in T.STEMS:
                for g in dataclasses.fields(T.StemInput):
                    assert torch.equal(getattr(ba.stems[name], g.name), getattr(bb.stems[name], g.name)), (name, g.name)
        else:
            assert torch.equal(getattr(ba, f.name), getattr(bb, f.name)), f.name
    names = [f.name for f in dataclasses.fields(T.TokenBatch)] + [f.name for f in dataclasses.fields(T.StemInput)]
    assert not [n for n in names if FORBIDDEN_FIELDS.match(n)], names


def test_pose3d_tokens_do_not_depend_on_the_lab_heading(tok):
    xs = [_tokens(tok, [_window([_pose(100.0, 4.0, 0.0, 3, heading=h)], 4.0)])[1] for h in (0.0, 1.1, -2.5)]
    for x in xs[1:]:
        assert torch.allclose(x, xs[0], rtol=0, atol=1e-5 * xs[0].abs().max())


# ---------------------------------------------------------------- layout, counts and placement
def test_placement_gathers_and_types():
    ws = [_window([_imu(100.0, 10.0)], 10.0), _window([_imu(128.0, 2.3, sites=(3,))], 2.3),
          _window([_pose(30.0, 6.0, 0.01, 2)], 6.0)]
    n = [T.window_tokens(w).n for w in ws]
    assert n == [L.window_tokens(w) for w in ws] == [40 + 1 + 4 * 40 + 2, 10 + 1 + 2 * 10 + 1, 24 + 1 + 24 + 1]
    b = T.tokenize(ws, row_len=256)
    assert b.shape == (2, 256) and b.n_windows == 3
    kind, seg, ttype = b.kind.reshape(-1), b.seg.reshape(-1), b.ttype.reshape(-1)
    for i, w in enumerate(ws):
        P = int(b.n_patches[i])
        lat = b.lat_idx[i, :P]
        assert (kind[lat] == T.LATENT).all() and (seg[lat] == i).all() and (b.lat_idx[i, P:] == -1).all()
        assert kind[b.subj_idx[i]] == T.SUBJECT and seg[b.subj_idx[i]] == i
        assert (b.t.reshape(-1)[lat] == torch.arange(P) * 0.25 + 0.125).all()
        assert abs(float(b.t.reshape(-1)[b.subj_idx[i]]) - w.duration / 2) < 1e-6
        assert int((seg == i).sum()) == n[i]
    nuis = b.nuis_row * 256 + b.nuis_pos
    assert (kind[nuis] == T.NUISANCE).all() and (ttype[nuis] == M.NUISANCE).all()
    assert ((ttype == M.NUISANCE) == (kind == T.NUISANCE)).all() and (kind[seg < 0] == T.PADDING).all()
    assert b.nuis_site.tolist() == [2, 7, 3, T.N_SITES] and b.nuis_win.tolist() == [0, 0, 1, 2]
    for name, si in b.stems.items():
        flat = si.row * 256 + si.pos
        assert (kind[flat] == T.MEASUREMENT).all() and len(set(flat.tolist())) == len(flat)
    # explicit placement, overlap and oversize refused
    b2 = T.tokenize(ws, row_len=400, placement=[(0, 0), (0, 203), (1, 10)])
    assert b2.shape == (2, 400)
    with pytest.raises(ValueError, match="overlaps"):
        T.tokenize(ws, row_len=400, placement=[(0, 0), (0, 100), (1, 0)])
    with pytest.raises(ValueError):
        T.tokenize(ws, row_len=100)
    # the typed mask of the batch: body tokens never see a nuisance token
    m = b.mask()[:, 0]
    body_q = (b.ttype == M.BODY) & (b.seg >= 0)
    nuis_k = b.ttype == M.NUISANCE
    assert not (m & body_q[:, :, None] & nuis_k[:, None, :]).any()


def test_nuisance_vocabularies_cover_the_factory():
    from mdfm.factory import camera_render as CR
    assert T.NUIS_FIELDS["layout"] == tuple(L.FAMILY_RULES)
    assert set(L.device_families()) <= set(T.NUIS_FIELDS["device"])
    assert set(CR.FAMILIES) <= set(T.NUIS_FIELDS["camera"])
    assert T.NUIS_FIELDS["site"] == tuple(L.SITES)
    assert [T.rate_bucket(f) for f in (25.0, 30.0, 60.0, 100.0, 128.0, 199.0)] == \
        ["below_30", "30_to_60", "60_to_100", "100_to_150", "100_to_150", "150_to_200"]


def _walk(K=1000):
    """BodyWindows of one procedural walk of K knots at 100 Hz in the window frame (1 m/s along +z)."""
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    from mdfm.data.body_store import BodyWindows
    t = torch.arange(K, dtype=F64) / 100.0
    ph = 2 * math.pi * 0.9 * t
    aa = torch.zeros(K, 22, 3, dtype=F64)
    aa[:, 1, 0], aa[:, 2, 0] = 0.4 * torch.sin(ph), -0.4 * torch.sin(ph)
    aa[:, 4, 0], aa[:, 5, 0] = 0.6 * (1 - torch.cos(ph)) / 2, 0.6 * (1 + torch.cos(ph)) / 2
    aa[:, 16, 2], aa[:, 17, 2] = -1.2, 1.2
    aa[:, 18, 0], aa[:, 19, 0] = -0.3 * torch.sin(ph), 0.3 * torch.sin(ph)
    rot6d = matrix_to_rot6d(axis_angle_to_matrix(aa)).float()
    transl = torch.stack([torch.zeros_like(t), 0.92 + 0.02 * torch.sin(2 * ph), t - t[K // 2]], -1).float()
    zero = torch.zeros(1, dtype=torch.int64)
    return BodyWindows(rot6d[None], transl[None], torch.zeros(1, K, 2, dtype=torch.bool),
                       torch.ones(1, K, dtype=torch.bool), torch.tensor([K]), torch.zeros(1, 16), torch.tensor([2]),
                       np.array(["test:walk"]), zero, zero.clone(), zero.clone())


@pytest.fixture(scope="module")
def synthetic():
    """{family: Window} rendered from the procedural walk, band-copied."""
    from mdfm.body.smplh import SkeletonBank
    from mdfm.factory.synth import SynthSampler
    layouts = L.load_layouts()
    sampler = SynthSampler(layouts=layouts, skel=SkeletonBank(torch.float32), dtype=torch.float32, device=DEVICE)
    body = _walk()
    out = {}
    for k, name in enumerate(("xsens13", "xsens13_insoles", "g1_six", "nonan3", "lower_back", "wrist_L", "wrist_acc",
                              "camera1", "mocap3d", "mocap3d_plate", "random_sites")):
        lay = L.concrete(layouts[name], 11)
        out[name] = (lay, sampler.render(body, [lay], 100 + k)[0])
    out["nonan3_raw"] = (L.concrete(layouts["nonan3"], 11),
                         sampler.render(body, [L.concrete(layouts["nonan3"], 11)], 5, band=False)[0])
    return out


@needs_ma
def test_token_counts_and_order_on_synthetic_windows(tok, synthetic):
    want = {"xsens13": 1094, "xsens13_insoles": 1094, "g1_six": 527, "nonan3": 164, "lower_back": 122, "wrist_L": 122,
            "wrist_acc": 82, "camera1": 82, "mocap3d": 82, "mocap3d_plate": 82}
    for name, (lay, w) in synthetic.items():
        if name == "nonan3_raw":
            continue
        W.validate(w)
        wt = T.window_tokens(w)
        assert wt.n == L.window_tokens(w) == L.token_count(lay, w.duration), name
        if name in want:
            assert wt.n == want[name], name
        P = wt.P
        assert (wt.kind[:P] == T.LATENT).all() and wt.kind[P] == T.SUBJECT
        nn_ = len(wt.nuis)
        assert (wt.kind[wt.n - nn_:] == T.NUISANCE).all() and (wt.kind[P + 1:wt.n - nn_] == T.MEASUREMENT).all()
        assert (T.nuisance_descriptors(w.streams[0])[:, T.FIELD_NAMES.index("layout")] ==
                T.NUIS_FIELDS["layout"].index(name)).all()
        unknown = np.array([len(v) + 1 for v in T.NUIS_FIELDS.values()])
        assert not (wt.nuis == unknown).any(), name
    x = _tokens(tok, [w for n, (_, w) in synthetic.items() if n != "nonan3_raw"])[1]
    assert torch.isfinite(x).all()
    # a synthetic nonan3 render before band_copy runs at 200 Hz and is refused
    raw = synthetic["nonan3_raw"][1]
    assert raw.streams[0].fps == 200.0
    with pytest.raises(ValueError, match="not a band copy"):
        T.window_tokens(raw)


@pytest.fixture(scope="module")
def real():
    """{source: band-copied first window} of the first admitted recording of weargait_pd and gait120."""
    from mdfm.data import sources as S
    out = {}
    for source in ("weargait_pd", "gait120"):
        w = W.cut(next(S.paired(source, limit=1)))[0]
        out[source] = dataclasses.replace(w, streams=[WS.band_copy(s) for s in w.streams])
    return out


@needs_index
def test_real_band_copies_tokenize_like_the_layouts(tok, real):
    for source, w in real.items():
        W.validate(w)
        wt = T.window_tokens(w)
        assert wt.n == L.window_tokens(w), source
        d = dict(zip(T.FIELD_NAMES, wt.nuis[0]))
        if source == "weargait_pd":
            assert wt.n == 1094 and len(wt.nuis) == 13
            assert d["layout"] == T.NUIS_FIELDS["layout"].index("xsens13")
            assert d["device"] == T.NUIS_FIELDS["device"].index("xsens13")
            assert d["rate"] == T.NUIS_FIELDS["rate"].index("100_to_150")
            assert d["acc_content"] == T.NUIS_FIELDS["acc_content"].index("with_gravity")
        else:
            assert len(wt.nuis) == 1 and d["layout"] == T.NUIS_FIELDS["layout"].index("mocap3d")
            assert d["dim"] == T.NUIS_FIELDS["dim"].index(3) and d["kind"] == T.NUIS_FIELDS["kind"].index("pose3d")
            assert not any(g.stem != "pose3d" for g, _ in wt.groups)
    x = _tokens(tok, list(real.values()))[1]
    assert torch.isfinite(x).all()
