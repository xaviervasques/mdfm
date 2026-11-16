"""Inference adapters of the stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 3 step 7,
sections 2.7 and 6, plan task B4.19): a checkpoint of mdfm/train/checkpoint.py loaded into the assembled model
of mdfm/model/mfm.py, bodies inferred on real or rendered windows on the 100 Hz clock, and the predictors that
the G1 and G3 harnesses call (mdfm/gates/g1.py, mdfm/gates/g3.py).

Loading (load_checkpoint): checkpoint.load (the file SHA-256 against its sidecar, the config and person-list
hashes, the records and lockbox hashes of the reports, the person guard on the stored lists, the state_dict
audit of the model, the EMA and the optimizer state), then MFM(the run's model config: config.model_config, so a
stage-2 checkpoint gets the name tables of its run, task S2.11) with the chosen weights ('ema' by default, or
'model') loaded strictly and audited again (provenance.audit); the optimizer state is dropped; the model is put
in eval mode with every parameter frozen. Loaded carries the checkpoint's SHA-256, step, phase and phase lengths
(phase_a_complete: the checkpoint closes phase A, the simulator-only model of G3, or lies in phase B or in the
stage-2 phase C), its training person lists and the hashes that moved since it was written (drift), and its tier
(checkpoint.load's ck['tier']: RELEASE for a checkpoint written before task S2.4, which recorded none) with the
licence record of its provenance (empty for those checkpoints); describe() reports both. load_checkpoint(...,
tier='RELEASE') refuses an INTERNAL checkpoint; by default no tier is asked, so an INTERNAL checkpoint is read for
evaluation with its tier stated.

Windows. An inference window is a windows.Window whose streams are band copies (data/window_store.band_copy),
as in training: record_windows(record) cuts a canonical IMU record (a G3 model input: magnetometer masked) into
windows of WIN_S s every HOP_S s from its first sample, the last one aligned on the record's end so that every
sample lies in a full window (a record shorter than WIN_S gives one window, one under MIN_S none); the band
copy runs on the whole record before the cut. Each window lasts n / fps s for its n samples (so n knots are
returned, knot k at the time of sample k), its stream carries the layout of its source (data/sources.py:
xsens13 for weargait_pd) and its covariates come from windows.covariates (sex from the source table, task from
the record key). g1_windows(gset, i) builds the three G1 inputs of one window of a G1Set (the inputs are
already band copies): 'six' (the six IMUs, layout g1_six), 'wrist' (wrist_L alone, layout wrist_L) and
'camera' (the 2D pose stream, layout camera1), source 'synthetic' (the reference renders of AMASS bodies, whose
per-source joint offsets are the synthetic row), sex the take's recorded gender, task from the window table.
A window, record or row whose source, subject or person is a CODY name is refused (canonical.Refused) unless
allow_eval is passed (an evaluation stage; nothing in this module trains).

Inference (infer). Windows in chunks of batch_windows: mfm.batch_from_windows (every sample a kernel point; the
nuisance codes under the model's own vocabularies when they are not stage 1's, batch_nuis_fields), MFM.encode
under a bf16 autocast on CUDA (float32 on the CPU), then refine_steps stopped-gradient refinement steps
(model/refine.refine; STEPS_EVAL = 20 and POINTS_EVAL = 256 points per stream-window and step at
evaluation, design section 4), the objective before the first and before the last step kept. The shape means
are then pooled over the windows of each group (posterior.subject_pool, the product of the Gaussians of the
refined means with the encoder variances; one group per trial or take, as training pools per person in a
micro-step; groups None: each window alone). The point body decodes the posterior means (z = mu_z, beta the
pooled mean) at every knot, the root integrated from each window's centre knot lengths // 2 of its own chunk
(model/decoder.py); the skeleton is SkeletonBank of the window's sex (neutral when unknown). Optional
draws: samples latent draws (z and beta from the refined posterior) decoded at every knot, each with
decoder_draws isotropic draws of the chosen joint from pos_logvar, the joint taken relative to the pelvis in
the pelvis-heading frame of its own body (the G1b frame: mdfm.gates.g1.heading_points). Randomness: one
torch.Generator on the model's device seeded by `seed` per call (refinement points and draws).

Predictors.
- G3Predictor(loaded): the callable of g3.dev_run and g3.read_lockbox, (trial_id, record) -> g3.TrialBody:
  record_windows, infer with one group per trial, contacts sigmoid(contact_logit) and joints of every window
  stitched on the trial clock (g3.stitched_body: each frame from the window where it is most central; strides
  and pass speeds read inside one window frame). Bodies are cached by trial id when a cache dict is given.
  zupt_mode (F2 lever 5a, task L5A; mdfm/eval/zupt_refine.py): 'off' by default, the predictor as it always was;
  'anchor' adds the inertial anchor of lever 5 to each chunk's refinement objective, its targets the version-2 ZUPT
  strides of the record's own foot sensors (no reference value), its constants fixed (20 steps); 'report' computes
  the same targets and the foot report without touching the objective. Either adds the option's fields and its
  per-target rows (zupt_rows) to the trial's log entry.
- G1Predictor(loaded): predict(gset, i) -> {joints_six, joints_camera, wrist, wrist_six} of mdfm.gates.g1: the
  point bodies of the six-IMU and camera inputs, the wrist draws of the single-wrist and of the six-IMU inputs
  (G1_SAMPLES latent samples x G1_DECODER_DRAWS decoder draws = 64, rule section 4). Windows run in blocks of
  whole takes (the take of the window asked and the takes that follow it, up to block_windows windows), one
  shape group per take, and a block's results are cached until a window outside it is asked. conformal: the
  factors of the two wrist predictives, fitted on the calib archives with fit_conformal (one pass over the
  calib windows for both). sha256, train_persons and train_persons_sha256 come from the checkpoint (the amass
  list, whose disjointness from the scored persons g1.run asserts).

Saved conformal factors (plan task B4.20, opt-in). conformal_record and save_conformal write the factors of a
fit on every calib window to the checkpoint's run folder, conformal_path: cache/runs/<run>/conformal_<first 12
hex digits of the checkpoint SHA-256>.json (format CONFORMAL_FORMAT: the checkpoint SHA-256, the settings that
define the two wrist predictives, the calib archives and their manifest SHA-256, the single-wrist and six-IMU
factors; conformal_text is the file's exact text, so its SHA-256 can be stated before the write). read_conformal
checks a file against a checkpoint SHA-256 (another checkpoint's file is refused), the settings of
CONFORMAL_SETTINGS and the calib manifest. G1Predictor.use_conformal_file sets c_wrist and c_six from
such a file; nothing in this module calls it, so a predictor reads a file only when its caller asks
(scripts/eval_g1_dev.py --conformal-file).
"""
import hashlib
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..body.rotations import rot6d_to_matrix, yaw_angle
from ..body.joints import fk
from ..body.sex import gender_code
from ..data import windows as W
from ..data.canonical import Refused, is_cody, namespace
from ..data.window_store import band_copy
from ..gates import g1 as G1
from ..gates import g3 as G3
from ..model import mfm as M
from ..model import tokenize as TK
from ..model.decoder import all_knots
from ..model.posterior import sample, subject_pool
from ..model.refine import POINTS_EVAL, STEPS_EVAL, refine

RATE = 100.0
WIN_S = 10.0                    # s, window length (spec 2.7: 10 s windows)
HOP_S = 5.0                     # s, window hop (spec 2.7: 5 s hop)
MIN_S = 1.0                     # s, a record shorter than this gives no window (windows.cut)
BATCH_WINDOWS = 8               # windows per chunk of infer [cfg]
WEIGHTS = ("ema", "model")
PELVIS = 0
WRIST_JOINT = G1.WRIST_JOINT    # 20, left wrist
G1_SAMPLES = G1.PARAMS["latent_samples"]         # 16
G1_DECODER_DRAWS = G1.PARAMS["decoder_draws"]    # 4
G1_CONFIGS = ("six", "wrist", "camera")
G1_SOURCE = "synthetic"
G1_LAYOUTS = {"six": "g1_six", "wrist": "wrist_L", "camera": "camera1"}


# ---------------------------------------------------------------- refusals
def _cody_names(*names):
    """True when a name is CODY: its prefix (canonical.is_cody), the namespace of a namespaced id among the
    CODY namespaces of canonical and of reports/manifest_tiers (provenance.is_cody_id)."""
    from ..data.canonical import CODY_NAMESPACES
    from ..train.provenance import is_cody_id
    names = [n for n in names if isinstance(n, str)]
    return is_cody(*names) or any(namespace(n) in CODY_NAMESPACES for n in names) or is_cody_id(*names)


def refuse_cody(*names, allow_eval=False):
    """canonical.Refused when a name (source, subject_ns, person id) is CODY and allow_eval is not set."""
    if not allow_eval and _cody_names(*names):
        raise Refused(f"CODY is evaluation-only; inference refuses {[n for n in names if isinstance(n, str)]} "
                      "without allow_eval")


# ---------------------------------------------------------------- checkpoints
@dataclass
class Loaded:
    """A checkpoint loaded for inference (module docstring). model: MFM in eval mode, parameters frozen;
    phase: the schedule's phase at the save; phase_steps: optimizer steps of phases A and B of the run;
    persons: the stored training person lists {source: ids}; hashes: the provenance hashes of the save."""
    model: object
    path: str
    name: str
    sha256: str
    weights: str
    step: int
    phase: str
    phase_steps: tuple
    config_json: str
    config_sha256: str
    persons: dict
    persons_sha256: dict
    hashes: dict
    drift: list
    tier: str = "RELEASE"
    licences: dict = field(default_factory=dict)

    @property
    def phase_a_complete(self):
        """True when the checkpoint closes phase A (or lies in phase B, or in the stage-2 phase C, which starts from
        a phase B end)."""
        if self.phase in ("B", "C"):
            return True
        return self.phase == "A" and bool(self.phase_steps) and self.step >= int(self.phase_steps[0])

    @property
    def device(self):
        return self.model.device

    def trained_persons(self):
        """frozenset of every stored training person id, all sources."""
        return frozenset(p for ids in self.persons.values() for p in ids)

    def describe(self):
        """JSON-ready provenance of the loaded checkpoint for the reports."""
        from ..gates import common
        n_par = int(sum(p.numel() for p in self.model.parameters()))
        keep = ("tiers_version", "records_sha256", "lockbox_sha256", "moveatlas_head", "code_sha256", "config_sha256")
        return {"path": common.rel(self.path), "name": self.name, "sha256": self.sha256, "weights": self.weights,
                "step": self.step, "phase": self.phase, "phase_steps": list(self.phase_steps),
                "phase_a_complete": self.phase_a_complete, "config_sha256": self.config_sha256,
                "parameters": n_par, "persons": {s: len(v) for s, v in sorted(self.persons.items())},
                "persons_sha256": self.persons_sha256, "saved_hashes": {k: self.hashes.get(k) for k in keep},
                "drift": list(self.drift), "tier": self.tier,
                "licence_matrix_sha256": (self.licences or {}).get("matrix_sha256")}


def checkpoint_name(path):
    """'<run folder>_<file stem>' of a checkpoint path, the name of its report folders."""
    p = os.path.abspath(str(path))
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"


def load_checkpoint(path, weights="ema", device="cpu", cfg=None, tier=None, records_state=None):
    """Loaded of a checkpoint (module docstring); weights 'ema' or 'model'; cfg: the current RunConfig, compared
    with the stored one (drift); tier: the tier asked of the checkpoint (checkpoint.load: 'RELEASE' refuses an
    INTERNAL checkpoint; None asks none, even with cfg); records_state: the records_sha256 of an archived records
    state the checkpoint is read under (checkpoint.load; the v1 checkpoints for the port's comparison). Raises what
    checkpoint.load raises (ProvenanceError, canonical.Refused)."""
    from ..train import checkpoint as CK
    from ..train import config as CF
    from ..train import provenance as PV
    if weights not in WEIGHTS:
        raise ValueError(f"weights {weights!r}, expected one of {WEIGHTS}")
    p = os.path.abspath(str(path))
    # an INTERNAL reader admits both tiers: with tier None the cfg (drift only) asks no tier of the checkpoint
    ck = CK.load(p, map_location="cpu", cfg=cfg, tier=PV.INTERNAL if tier is None else tier,
                 records_state=records_state)
    side = p + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            sha = f.read().split()[0]
    else:
        sha = PV.sha256_file(p)
    if ck.get(weights) is None:
        raise ValueError(f"{p} holds no {weights} state")
    rc = CF.from_json(ck["config"])
    model = M.MFM(CF.model_config(rc))
    model.load_state_dict(ck[weights], strict=True)
    PV.audit(model)
    sched = ck.get("sched") or {}
    prov = ck["provenance"]
    out = Loaded(model=None, path=p, name=checkpoint_name(p), sha256=sha, weights=weights, step=int(ck["step"]),
                 phase=str(sched.get("in_phase", (prov.get("extra") or {}).get("phase", ""))),
                 phase_steps=tuple(int(x) for x in sched.get("steps", ())), config_json=ck["config"],
                 config_sha256=ck["config_sha256"], persons={s: list(v) for s, v in prov["persons"].items()},
                 persons_sha256=dict(ck["persons_sha256"]), hashes=dict(prov.get("hashes") or {}),
                 drift=list(ck["drift"]), tier=ck["tier"], licences=dict(prov.get("licences") or {}))
    del ck
    model.eval()
    model.requires_grad_(False)
    out.model = model.to(torch.device(device))
    return out


def batch_nuis_fields(model):
    """The nuis_fields argument of mfm.batch_from_windows for `model`: None (the stage-1 call, unchanged) when the
    model's nuisance vocabularies are stage 1's (tokenize.NUIS_FIELDS), else the model's own (a stage-2 model of
    other name tables: MFM refuses a batch coded under other vocabularies)."""
    f = model.nuis_fields
    return None if f is TK.NUIS_FIELDS or dict(f) == dict(TK.NUIS_FIELDS) else f


def train_persons_sha256(loaded):
    """SHA-256 of every stored training person of a checkpoint ('all' of provenance.list_hashes)."""
    return loaded.persons_sha256.get("all", "")


# ---------------------------------------------------------------- windows
def spans(n, fps, win_s=WIN_S, hop_s=HOP_S, min_s=MIN_S):
    """(start, length) sample spans of a stream of n samples at fps (module docstring): windows of
    round(win_s fps) samples every round(hop_s fps), the last aligned on the end; one window of n samples when
    n is shorter than a window; none under round(min_s fps) samples."""
    win, hop = int(round(win_s * fps)), int(round(hop_s * fps))
    if hop < 1 or win < 1:
        raise ValueError(f"window {win_s} s and hop {hop_s} s at {fps} Hz")
    if n < max(1, int(round(min_s * fps))):
        return []
    if n <= win:
        return [(0, int(n))]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(int(s), win) for s in starts]


def _window(source, person_id, key, stream, t64, start, n, cov):
    """Window of samples [start, start + n) of a band-copied stream on native times t64 (float64 s)."""
    t0 = float(t64[start])
    fps = float(stream.fps)
    sel = slice(start, start + n)
    s = W.Stream(modality=stream.modality, role=stream.role, fps=fps, t=(t64[sel] - t0).astype(np.float32),
                 values=np.ascontiguousarray(stream.values[sel]), chan_mask=stream.chan_mask.copy(),
                 valid=np.ascontiguousarray(stream.valid[sel]), meta=W.slice_meta(stream.meta, sel))
    return W.Window(source=source, person_id=str(person_id), key=f"{key}@{start}", synthetic=False, t0=t0,
                    duration=n / fps, streams=[s], cov=dict(cov))


def record_windows(record, win_s=WIN_S, hop_s=HOP_S, min_s=MIN_S, layout=None, allow_eval=False):
    """(windows, spans, fps) of a canonical IMU record (module docstring): the band copy of the whole record
    cut into spans(n, fps); layout default: the source's (data/sources.SOURCES). A CODY record is refused
    without allow_eval."""
    from ..data import sources as S
    source = str(record.get("source") or "")
    refuse_cody(source, record.get("subject_ns"), record.get("person_id"), allow_eval=allow_eval)
    rec = W.whitelist(record)
    fps = W.record_fps(rec)
    values, cm, valid, meta = W.arrays("imu", rec)
    lay = layout if layout is not None else (S.SOURCES[source].layout if source in S.SOURCES else None)
    if lay:
        meta["layout"] = lay
    meta["source"] = source
    n0 = values.shape[0]
    t64 = np.arange(n0, dtype=np.float64) / fps
    full = W.Stream(modality="imu", role="input", fps=fps, t=t64, values=values, chan_mask=cm, valid=valid, meta=meta)
    b, info = band_copy(full, return_info=True)
    tb = t64[info.idx]
    cov = W.covariates(rec, source)
    sp = spans(len(tb), b.fps, win_s, hop_s, min_s)
    key = str(record.get("key") or "")
    wins = [W.validate(_window(source, record.get("person_id"), key, b, tb, s, n, cov)) for s, n in sp]
    return wins, sp, float(b.fps)


def g1_windows(gset, i, configs=G1_CONFIGS, allow_eval=False):
    """{config: Window} of the G1 inputs of window i of a G1Set (module docstring)."""
    from ..factory.imu_render import SITES
    r = gset.windows.iloc[int(i)]
    refuse_cody(r.get("archive"), r.get("subject_ns"), r.get("person_id"), r.get("take_id"), allow_eval=allow_eval)
    x = gset.inputs(i)
    n = int(r["n"])
    cov = {"sex": gender_code(r["gender"]), "age_band": W.AGE_UNKNOWN, "task": int(r["task"])}
    key = f"{r['take_id']}@{int(r['k'])}"
    out = {}
    dur = n / RATE
    t = np.arange(n, dtype=np.float64) / RATE
    for c in configs:
        if c in ("six", "wrist"):
            cols = list(range(len(G1.SITES))) if c == "six" else [G1.WRIST_COL]
            vals = np.zeros((n, len(cols), 9), np.float32)
            vals[..., :G1.N_CH] = x["imu"][:, cols]
            cm = np.zeros((len(cols), 9), bool)
            cm[:, :G1.N_CH] = True
            ok = np.asarray(x["imu_valid"], bool)[:, cols]
            vals[~ok] = 0.0
            meta = {"site_ids": [SITES.index(G1.SITES[j]) for j in cols], "acc_content": "with_gravity",
                    "layout": G1_LAYOUTS[c], "device": "xsens13", "source": G1_SOURCE}
            s = W.Stream("imu", "input", RATE, t.astype(np.float32), vals, cm, ok, meta)
        elif c == "camera":
            fps = float(x["cam_fps"])
            nc = int(x["cam"].shape[0])
            vals = np.zeros((nc, 15, 3), np.float32)
            ok = np.asarray(x["cam_valid"], bool)
            vals[..., :2] = np.where(ok[..., None], x["cam"], 0.0)
            cm = np.zeros((15, 3), bool)
            cm[:, :2] = True
            tc = (np.arange(nc, dtype=np.float64) / fps).astype(np.float32)
            meta = {"dim": 2, "scale": float(x["cam_scale"]), "unit_to_si": None, "camera": "generic",
                    "layout": G1_LAYOUTS[c], "source": G1_SOURCE}
            s = W.Stream("pose2d", "input", fps, tc, vals, cm, ok, meta)
        else:
            raise ValueError(f"unknown G1 input {c!r}, expected one of {G1_CONFIGS}")
        out[c] = W.validate(W.Window(G1_SOURCE, str(r["person_id"]), key, False, 0.0, dur, [s], dict(cov)))
    return out


# ---------------------------------------------------------------- inference
@dataclass
class WindowBody:
    """One window's inferred body on the 100 Hz clock, knot k at k / 100 s from the window start (module
    docstring): joints (n, 22, 3) m in the model's window frame, rot6d (n, 22, 6), transl (n, 3) m, contact (n,
    2) probabilities [left, right], pos_logvar (n, 22) log m^2, beta (16,) the pooled shape mean, gender (SkeletonBank
    code); draws (n, m, 3) m of the draw joint relative to the pelvis in the pelvis-heading frame, or None."""
    joints: np.ndarray
    rot6d: np.ndarray
    transl: np.ndarray
    contact: np.ndarray
    pos_logvar: np.ndarray
    beta: np.ndarray
    gender: int
    draws: np.ndarray = None


def seed_of(*parts):
    """A 63-bit seed from the SHA-256 of the parts joined by newlines."""
    h = hashlib.sha256("\n".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def _amp(device, amp):
    dev = torch.device(device)
    on = dev.type == "cuda" if amp is None else bool(amp)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=on)


def heading_points(x, pelvis, yaw):
    """Points x (..., 3) relative to the pelvis (..., 3) in the pelvis-heading frame, yaw (...) rad (torch form of
    mdfm.gates.g1.heading_points)."""
    v = x - pelvis
    c, s = torch.cos(yaw), -torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]], -1)


def _pose(model, rot6d, transl, beta, gender, w):
    """Joint positions (M, 22, 3) and global rotations (M, 22, 3, 3) of decoded knots, float32, autocast off."""
    with torch.autocast(device_type=rot6d.device.type, enabled=False):
        rest = model.skeleton.rest(beta.float(), gender)
        return fk(rot6d_to_matrix(rot6d.float()), transl.float(), rest[w])


def _group_ids(groups, n):
    if groups is None:
        return torch.arange(n, dtype=torch.int64)
    if len(groups) != n:
        raise ValueError(f"{len(groups)} groups for {n} windows")
    _, inv = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    return torch.as_tensor(inv.reshape(-1), dtype=torch.int64)


@torch.no_grad()
def infer(model, windows, refine_steps=STEPS_EVAL, points=POINTS_EVAL, groups=None, samples=0, decoder_draws=0,
          draw_joint=WRIST_JOINT, batch_windows=BATCH_WINDOWS, seed=0, amp=None, allow_eval=False, stats=None,
          zupt=None):
    """[WindowBody] of the windows (module docstring). refine_steps: refinement steps (20 at evaluation, 0 for
    the amortized posterior); points: kernel points per stream-window and step (None: all); groups: one key per
    window for the shape pooling (None: no pooling); samples, decoder_draws, draw_joint: the draws; stats: a
    dict that receives windows, chunks, refine_first and refine_last (sums over the chunks of the refinement
    objective -sum_w log_joint on the step's point draw, before the first and before the last step) and
    seconds. zupt: None (the default: the refinement as it always was), or the F2 lever-5a option of the windows
    (mdfm/eval/zupt_refine.TrialZupt): its objective(model, batch, first, count) gives the context under which a
    chunk is refined (the inertial anchor added to the objective; refine_first and refine_last then include it),
    or None for a chunk refined as without it."""
    wins = list(windows)
    if not wins:
        return []
    for wd in wins:
        refuse_cody(wd.source, wd.person_id, allow_eval=allow_eval)
    t_start = time.perf_counter()
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(int(seed) % (1 << 63))
    B = max(1, int(batch_windows))
    chunks = [wins[a:a + B] for a in range(0, len(wins), B)]
    kept, first, last = [], 0.0, 0.0
    for ci, ch in enumerate(chunks):
        batch = M.batch_from_windows(ch, sources=model.cfg.sources, families=model.heads.families, rate=model.rate,
                                     allow_eval=allow_eval, gen=0, nuis_fields=batch_nuis_fields(model)).to(dev)
        with _amp(dev, amp):
            post = model.encode(batch)
            if refine_steps:
                hist = []
                obj = None if zupt is None else zupt.objective(model, batch, ci * B, len(ch))
                if obj is None:
                    post = refine(model, post, batch, int(refine_steps), points_per_stream=points, generator=gen,
                                  history=hist)
                else:
                    with obj:
                        post = refine(model, post, batch, int(refine_steps), points_per_stream=points,
                                      generator=gen, history=hist)
                first += hist[0]
                last += hist[-1]
        kept.append((post.detach(), batch.lengths.to(dev), batch.tokens.n_patches.to(dev), batch.gender.to(dev)))
        del batch
    mu_b = torch.cat([k[0].mu_b for k in kept])
    lv_b = torch.cat([k[0].logvar_b for k in kept])
    gid = _group_ids(groups, len(wins)).to(dev)
    mu_p, lv_p = subject_pool(mu_b, lv_b, gid)
    out, a = [], 0
    for post, lengths, npw, gender in kept:
        Wc = post.mu_z.shape[0]
        mb, lb = mu_p[a:a + Wc], lv_p[a:a + Wc]
        a += Wc
        w, k = all_knots(lengths)
        mask = post.mask[..., None].to(post.mu_z.dtype)
        with _amp(dev, amp):
            bo = model.decode(post.mu_z * mask, mb, (w, k), npw, lengths)
        pos, _ = _pose(model, bo.rot6d, bo.transl, mb, gender, w)
        draws = None
        if samples:
            parts = []
            for _ in range(int(samples)):
                z = sample(post.mu_z, post.logvar_z, generator=gen) * mask
                beta = sample(mb, lb, generator=gen)
                with _amp(dev, amp):
                    ds = model.decode(z, beta, (w, k), npw, lengths)
                p, R = _pose(model, ds.rot6d, ds.transl, beta, gender, w)
                x = heading_points(p[:, draw_joint], p[:, PELVIS], yaw_angle(R[:, PELVIS]))
                if decoder_draws:
                    sd = torch.exp(0.5 * ds.pos_logvar[:, draw_joint].float())
                    eps = torch.randn((int(decoder_draws),) + tuple(x.shape), generator=gen, device=dev, dtype=x.dtype)
                    parts.append(x[None] + sd[None, :, None] * eps)
                else:
                    parts.append(x[None])
            draws = torch.cat(parts).permute(1, 0, 2).contiguous()          # (M, m, 3)
        cuts = torch.cumsum(lengths, 0).tolist()
        starts = [0] + cuts[:-1]
        cpu = {"joints": pos.cpu().numpy(), "rot6d": bo.rot6d.float().cpu().numpy(),
               "transl": bo.transl.float().cpu().numpy(),
               "contact": torch.sigmoid(bo.contact_logit.float()).cpu().numpy(),
               "pos_logvar": bo.pos_logvar.float().cpu().numpy(),
               "draws": None if draws is None else draws.cpu().numpy()}
        g = gender.cpu().numpy()
        beta_np = mb.float().cpu().numpy()
        for j, (s0, s1) in enumerate(zip(starts, cuts)):
            wd = wins[len(out)]
            n = min(int(round(wd.duration * model.rate)), s1 - s0)
            sl = slice(s0, s0 + n)
            out.append(WindowBody(joints=cpu["joints"][sl], rot6d=cpu["rot6d"][sl], transl=cpu["transl"][sl],
                                  contact=cpu["contact"][sl], pos_logvar=cpu["pos_logvar"][sl], beta=beta_np[j],
                                  gender=int(g[j]), draws=None if cpu["draws"] is None else cpu["draws"][sl]))
    if stats is not None:
        stats.update(windows=len(wins), chunks=len(chunks), refine_first=first if refine_steps else None,
                     refine_last=last if refine_steps else None, seconds=time.perf_counter() - t_start)
    return out


# ---------------------------------------------------------------- G3
@dataclass
class G3Predictor:
    """The model callable of mdfm.gates.g3 (module docstring): __call__(trial_id, record) -> g3.TrialBody. log:
    one dict per trial run (windows, refinement objective, seconds); cache: None (every call runs the model), or a
    dict that keeps {trial_id: TrialBody} (a stitched body holds (windows, T, 22, 3) float64 joints, tens of MB
    per trial)."""
    loaded: Loaded
    refine_steps: int = STEPS_EVAL
    points: int = POINTS_EVAL
    win_s: float = WIN_S
    hop_s: float = HOP_S
    batch_windows: int = BATCH_WINDOWS
    seed: int = 0
    pool: bool = True
    allow_eval: bool = False
    log: list = field(default_factory=list)
    cache: dict = None
    zupt_mode: str = "off"

    def __call__(self, trial_id, record):
        if self.cache is not None and trial_id in self.cache:
            return self.cache[trial_id]
        refuse_cody(trial_id, record.get("source"), record.get("subject_ns"), record.get("person_id"),
                    allow_eval=self.allow_eval)
        wins, sp, fps = record_windows(record, self.win_s, self.hop_s, allow_eval=self.allow_eval)
        if not wins:
            raise ValueError(f"{trial_id}: record too short for a window")
        st = {}
        zr = None
        if self.zupt_mode != "off":
            # F2 lever 5a (mdfm/eval/zupt_refine.py): targets from this record's own foot sensors, nothing else
            from . import zupt_refine as ZR
            zr = ZR.TrialZupt.of(record, wins, ZR.check_mode(self.zupt_mode), self.refine_steps,
                                 allow_eval=self.allow_eval)
        bodies = infer(self.loaded.model, wins, refine_steps=self.refine_steps, points=self.points,
                       groups=[trial_id] * len(wins) if self.pool else None, batch_windows=self.batch_windows,
                       seed=seed_of(self.seed, trial_id), allow_eval=self.allow_eval, stats=st, zupt=zr)
        rate = self.loaded.model.rate
        starts = [int(round(s / fps * rate)) for s, _ in sp]
        tb = G3.stitched_body(trial_id, [b.joints for b in bodies], [b.contact for b in bodies], None, starts, rate)
        entry = {"trial_id": trial_id, "person_id": str(record.get("person_id")),
                 "samples": int(np.asarray(record["signal"]).shape[0]), "fps": fps, "windows": len(wins),
                 "refine_first": st.get("refine_first"), "refine_last": st.get("refine_last"),
                 "seconds": st.get("seconds")}
        if zr is not None:
            fields_, rows = zr.summary(bodies, trial_id, str(record.get("person_id")), rate)
            entry.update(fields_)
            entry["zupt_rows"] = rows           # the per-target rows; the scripts move them to zupt_strides.csv
        self.log.append(entry)
        if self.cache is not None:
            self.cache[trial_id] = tb
        return tb


# ---------------------------------------------------------------- G1
class G1Predictor:
    """The model as a predictor of mdfm.gates.g1 (module docstring). configs: the inputs computed per window
    (every one of G1_CONFIGS for scoring; ('six', 'wrist') is enough for the conformal fit); block_windows: a
    window not in the cache runs with the windows of its take and of the takes that follow it in the set's order,
    whole takes, until block_windows windows (a longer take runs alone), in chunks of batch_windows, the shape
    pooled per take; subset: window indices a block may hold (None: all; the scripts' partial runs); c_wrist and
    c_six: the conformal factors of the single-wrist and six-IMU wrist predictives."""

    def __init__(self, loaded, refine_steps=STEPS_EVAL, points=POINTS_EVAL, samples=G1_SAMPLES,
                 decoder_draws=G1_DECODER_DRAWS, batch_windows=BATCH_WINDOWS, block_windows=None, seed=0, pool=True,
                 configs=G1_CONFIGS, allow_eval=False, name="mfm"):
        self.loaded = loaded
        self.refine_steps, self.points = int(refine_steps), points
        self.samples, self.decoder_draws = int(samples), int(decoder_draws)
        self.batch_windows, self.seed, self.pool = int(batch_windows), int(seed), bool(pool)
        self.block_windows = int(block_windows if block_windows is not None else batch_windows)
        self.configs = tuple(configs)
        self.allow_eval = allow_eval
        self.name = name
        self.c_wrist = None
        self.c_six = None
        self.subset = None
        self.log = []
        self._key, self._out = None, {}

    @property
    def sha256(self):
        return self.loaded.sha256

    @property
    def train_persons(self):
        return frozenset(self.loaded.persons.get("amass", ()))

    @property
    def train_persons_sha256(self):
        from ..data import amass_index
        return amass_index.person_list_sha256(sorted(self.train_persons)) if self.train_persons else ""

    @property
    def conformal(self):
        if self.c_wrist is None:
            raise ValueError("the model has no conformal factor: fit it on the calib archives (fit_conformal)")
        return {"wrist": float(self.c_wrist), "wrist_six": None if self.c_six is None else float(self.c_six)}

    def settings(self):
        """The settings that define the two wrist predictives (CONFORMAL_SETTINGS), as a saved conformal file
        records them."""
        return {"weights": self.loaded.weights, "refine_steps": self.refine_steps, "points_per_stream": self.points,
                "latent_samples": self.samples, "decoder_draws": self.decoder_draws,
                "shape_pooling": "per take" if self.pool else "none"}

    def use_conformal_file(self, path, calib=None):
        """Opt-in (module docstring): c_wrist and c_six from a saved conformal file, read by read_conformal
        against this predictor's checkpoint SHA-256 and settings, and against the calib set's manifest when calib
        is given (canonical.Refused on a mismatch). Returns the file's record; the file's path and SHA-256 are
        kept in conformal_file."""
        doc = read_conformal(path, self.sha256, self.settings(), calib)
        self.c_wrist = float(doc["factors"]["wrist"]["c"])
        self.c_six = float(doc["factors"]["wrist_six"]["c"])
        self.conformal_file = {"path": os.path.abspath(str(path)), "file_sha256": doc["file_sha256"]}
        return doc

    def block(self, gset, i):
        """Window indices of the block that window i opens (class docstring): its take's windows, then those of
        the following takes in the set's order while the block stays within block_windows."""
        takes = gset.windows["take_id"].to_numpy().astype(str)
        allowed = None if self.subset is None else set(int(j) for j in self.subset) | {int(i)}
        order = list(dict.fromkeys(takes.tolist()))
        members = {}
        for j, t in enumerate(takes):
            if allowed is None or j in allowed:
                members.setdefault(t, []).append(j)
        order = [t for t in order if t in members]
        pos = {t: k for k, t in enumerate(order)}
        k = pos[takes[int(i)]]
        idx = list(members[order[k]])
        for t in order[k + 1:]:
            if len(idx) + len(members[t]) > self.block_windows:
                break
            idx += members[t]
        return idx

    def _run(self, gset, idx):
        takes = [str(gset.windows["take_id"].iat[j]) for j in idx]
        wins = {c: [] for c in self.configs}
        for j in idx:
            for c, wd in g1_windows(gset, j, self.configs, self.allow_eval).items():
                wins[c].append(wd)
        res = {j: {} for j in idx}
        rec = {"split": gset.split, "first_take": takes[0], "takes": len(set(takes)), "windows": len(idx)}
        for c in self.configs:
            st = {}
            draws = c in ("six", "wrist")
            bodies = infer(self.loaded.model, wins[c], refine_steps=self.refine_steps, points=self.points,
                           groups=takes if self.pool else None,
                           samples=self.samples if draws else 0, decoder_draws=self.decoder_draws if draws else 0,
                           batch_windows=self.batch_windows, seed=seed_of(self.seed, gset.split, takes[0], c),
                           allow_eval=self.allow_eval, stats=st)
            for j, b in zip(idx, bodies):
                res[j][c] = b
            rec[c] = {k: st.get(k) for k in ("refine_first", "refine_last", "seconds")}
        self.log.append(rec)
        return res

    def predict(self, gset, i):
        """The prediction dict of mdfm.gates.g1 for window i of gset (module docstring); inputs not in configs
        read None."""
        i = int(i)
        if self._key != id(gset) or i not in self._out:
            self._out, self._key = {}, None
            self._out = self._run(gset, self.block(gset, i))
            self._key = id(gset)
        r = self._out[i]
        get = lambda c, a: getattr(r[c], a) if c in r else None  # noqa: E731
        return {"joints_six": get("six", "joints"), "joints_camera": get("camera", "joints"),
                "wrist": get("wrist", "draws"), "wrist_six": get("six", "draws")}


def fit_conformal(gset, predictor, windows=None, log=None):
    """The conformal factors of the single-wrist ('wrist') and six-IMU ('wrist_six') predictives of a predictor
    on the calib set, in one pass (mdfm.gates.g1.fit_conformal for both keys: every knot a Mahalanobis radius of
    its truth under the moment-matched draws, c = ellipsoid_factor of the radii): {key: {c, n_frames,
    n_degenerate, coverage_at_c}}. windows: a subset of window indices (default all). Refused on another split."""
    from . import conformal
    if gset.split != "calib":
        raise Refused(f"conformal factors are fitted on the calib archives only, not {gset.split}")
    log = log or (lambda *a: None)
    keys = ("wrist", "wrist_six")
    radii, bad = {k: [] for k in keys}, dict.fromkeys(keys, 0)
    idx = range(len(gset)) if windows is None else windows
    t0 = time.time()
    for j, i in enumerate(idx):
        J, yaw = gset.truth(i)
        y = G1.wrist_truth(J, yaw)
        pred = predictor.predict(gset, i)
        for k in keys:
            if pred[k] is None:
                raise ValueError(f"the predictor has no {k} predictive")
            mean, cov = conformal.moment_match(np.asarray(pred[k], dtype=np.float64), ddof=G1.PARAMS["draws_ddof"])
            r, b = conformal.mahalanobis(y, mean, cov)
            radii[k].append(r)
            bad[k] += int(np.broadcast_to(b, r.shape).sum())
        if (j + 1) % 100 == 0:
            log(f"calib: {j + 1} windows, {time.time() - t0:.0f} s")
    out = {}
    dof, level = G1.PARAMS["ellipsoid_dof"], G1.PARAMS["conformal_level"]
    for k in keys:
        r = np.concatenate(radii[k])
        c = conformal.ellipsoid_factor(r, dof, level)
        out[k] = {"c": c, "n_frames": int(r.size), "n_degenerate": bad[k],
                  "coverage_at_c": float(np.mean(conformal.covered(r, c, dof, level))) if math.isfinite(c) else 0.0}
    return out


# ---------------------------------------------------------------- saved conformal factors (B4.20, opt-in)
CONFORMAL_FORMAT = "mdfm-conformal/1"
CONFORMAL_KEYS = ("wrist", "wrist_six")
# the settings that define the two wrist predictives: a file fitted under other values is refused
CONFORMAL_SETTINGS = ("weights", "refine_steps", "points_per_stream", "latent_samples", "decoder_draws",
                      "shape_pooling")


def conformal_path(checkpoint_path, sha256):
    """The saved conformal file of a checkpoint: conformal_<first 12 hex digits of its SHA-256>.json in the
    checkpoint's run folder (cache/runs/<run>/)."""
    if not isinstance(sha256, str) or len(sha256) < 12:
        raise ValueError(f"checkpoint SHA-256 {sha256!r}")
    return os.path.join(os.path.dirname(os.path.abspath(str(checkpoint_path))), f"conformal_{sha256[:12]}.json")


def _calib_info(calib, windows):
    """{split, archives, windows, knots, persons, manifest_sha256, persons_sha256} of the calib windows used."""
    w = calib.windows.iloc[list(windows)]
    man = getattr(calib, "manifest", None) or {}
    return {"split": calib.split, "archives": sorted(str(a) for a in w["archive"].unique()), "windows": int(len(w)),
            "knots": int(w["n"].astype(int).sum()), "persons": int(w["person_id"].nunique()),
            "manifest_sha256": man.get("manifest_sha256"), "persons_sha256": man.get("persons_sha256")}


def conformal_record(predictor, fits, calib, windows=None, extra=None):
    """The record of a saved conformal file (format CONFORMAL_FORMAT): the checkpoint of the predictor (path,
    name, SHA-256, weights, step, phase), its settings (G1Predictor.settings, with batch_windows, block_windows
    and seed beside them), the calib set (_calib_info) and the factors {wrist, wrist_six: {c, n_frames,
    n_degenerate, coverage_at_c}} of fit_conformal; extra: more JSON-ready provenance (the script, its code
    SHA-256). Refused (ValueError): a set other than calib, a fit on a subset of the calib windows (the saved factor
    is the checkpoint's, fitted on every calib window), a missing or non-finite factor."""
    from ..gates import common
    if calib.split != "calib":
        raise ValueError(f"conformal factors are saved from the calib archives only, not {calib.split}")
    idx = range(len(calib)) if windows is None else sorted(set(int(i) for i in windows))
    if len(idx) != len(calib):
        raise ValueError(f"a fit on {len(idx)} of the {len(calib)} calib windows is not saved: the saved factor is "
                         "fitted on every calib window")
    factors = {}
    for k in CONFORMAL_KEYS:
        f = (fits or {}).get(k)
        if not isinstance(f, dict) or f.get("c") is None or not math.isfinite(float(f["c"])):
            raise ValueError(f"no finite {k} factor to save: {f!r}")
        factors[k] = {"c": float(f["c"]), "n_frames": int(f["n_frames"]), "n_degenerate": int(f["n_degenerate"]),
                      "coverage_at_c": float(f["coverage_at_c"])}
    ld = predictor.loaded
    settings = dict(predictor.settings(), batch_windows=int(predictor.batch_windows),
                    block_windows=int(predictor.block_windows), seed=int(predictor.seed))
    return {"format": CONFORMAL_FORMAT, "what": "conformal factors of the single-wrist and six-IMU wrist predictives, "
            "split conformal on every calib window (docs/rules/G1_RULE.md section 6, plan task B4.20)",
            "checkpoint": {"path": common.rel(ld.path), "name": ld.name, "sha256": ld.sha256, "weights": ld.weights,
                           "step": int(ld.step), "phase": ld.phase},
            "settings": settings, "calib": _calib_info(calib, idx), "factors": factors,
            "level": G1.PARAMS["conformal_level"], "ellipsoid_dof": G1.PARAMS["ellipsoid_dof"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "extra": dict(extra or {})}


def conformal_text(record):
    """The exact text save_conformal writes for a conformal record (ASCII JSON, indent 1, a final LF), so that a
    caller can state the SHA-256 of the file's bytes before the file is written."""
    import json
    got = record.get("format") if isinstance(record, dict) else None
    if got != CONFORMAL_FORMAT:
        raise ValueError(f"not a conformal record: format {got!r}")
    return json.dumps(record, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def save_conformal(path, record, overwrite=False):
    """Write a conformal record (conformal_record) to path, atomically (a temporary file in the same folder, then
    os.replace), UTF-8 with LF line ends, the text of conformal_text; an existing file is refused
    (FileExistsError) unless overwrite. Returns the path."""
    text = conformal_text(record)
    p = os.path.abspath(str(path))
    if os.path.exists(p) and not overwrite:
        raise FileExistsError(f"{p} exists: a saved conformal file is written once")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = os.path.join(os.path.dirname(p), f".{os.path.basename(p)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, p)
    return p


def read_conformal(path, sha256, settings=None, calib=None):
    """The record of a saved conformal file, checked (canonical.Refused on any failure): its format; its
    checkpoint SHA-256 equal to sha256 (a file of another checkpoint is refused); a finite factor for each of
    CONFORMAL_KEYS; when settings is given, its settings equal to them on every key of CONFORMAL_SETTINGS; when
    calib (a calib G1Set) is given, its calib manifest SHA-256 equal to the set's. The record gains file_sha256,
    the SHA-256 of the file's bytes."""
    import json
    p = os.path.abspath(str(path))
    with open(p, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise Refused(f"{p}: not a conformal file ({e})") from e
    if not isinstance(doc, dict) or doc.get("format") != CONFORMAL_FORMAT:
        raise Refused(f"{p}: format {doc.get('format') if isinstance(doc, dict) else None!r}, expected "
                      f"{CONFORMAL_FORMAT}")
    got = (doc.get("checkpoint") or {}).get("sha256")
    if got != sha256:
        raise Refused(f"{p}: fitted for checkpoint {str(got)[:12]}, not {str(sha256)[:12]}: a conformal file serves "
                      "its own checkpoint only")
    for k in CONFORMAL_KEYS:
        c = ((doc.get("factors") or {}).get(k) or {}).get("c")
        if not isinstance(c, (int, float)) or isinstance(c, bool) or not math.isfinite(float(c)):
            raise Refused(f"{p}: no finite {k} factor")
    if settings is not None:
        have = doc.get("settings") or {}
        diff = {k: [have.get(k), settings.get(k)] for k in CONFORMAL_SETTINGS if have.get(k) != settings.get(k)}
        if diff:
            what = ", ".join(f"{k} {v[0]} there, {v[1]} here" for k, v in diff.items())
            raise Refused(f"{p}: fitted under other settings ({what})")
    if calib is not None:
        mine = (getattr(calib, "manifest", None) or {}).get("manifest_sha256")
        theirs = (doc.get("calib") or {}).get("manifest_sha256")
        if calib.split != "calib" or mine != theirs:
            raise Refused(f"{p}: fitted on the calib inputs {str(theirs)[:12]}, not on {calib.split} "
                          f"{str(mine)[:12]}")
    doc["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc
