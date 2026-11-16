"""The G4 development estimate of a checkpoint (design section 9; rule docs/rules/G4_RULE.md, DRAFT; harness
mdfm/gates/g4.py; inference mdfm/eval/infer.py): ground reaction and leg EMG from pose, and the metric-3D round
trip, on development data only. Not a gate read: no lockbox person, row or record is read, mdfm.gates.g4.read_lockbox
is never called, and no reader flag is set.

Model (G4Predictor, defined here). Per recording, the pose3d record of a g4.paired_records pair (stroke rotated to y
up by sources.pair) gets the raw root track (mdfm.data.window_store.attach_raw_root: gait120 PELV_C, stroke PELO;
reader flags at False) and becomes one pose3d input stream as the window store builds it: the canonical coords
with their scale and unit and the root track in the meta, layout mocap3d, then window_store.band_copy (as recorded
for pose: every sample kept). The stream is cut into windows of infer.WIN_S s every infer.HOP_S s from its first
sample, the last aligned on the end (infer.spans; a record shorter than a window is one window, a record under
infer.MIN_S s gives none and is left out, below); the covariates come from windows.covariates. infer.infer runs
the windows of a recording with 20 refinement steps of 256 points per stream (STEPS_EVAL, POINTS_EVAL), the shape
pooled over the recording's windows, seed infer.seed_of(seed, kind, trial id). Each pose frame is read from the
window whose knot span holds it nearest its centre. At those frames the point body (decoded posterior means,
contact probabilities turned back into logits) goes through MFM.render on a batch of kernel points at the frame
times, the path of training:
- GRF: the force branch, ForceHead's per-foot vertical ground reaction in body weights [left, right]; a flight
  frame (both contact probabilities below 0.5, where training masks the force likelihood) predicts 0 on both
  feet, as the Newton floor does (g4.newton_floor). The predictor gives the sealed read's GRF contract, the BW
  before the flight rule and the contact probabilities at the frames (G4Predictor.grf_contact; the
  probabilities read flight exactly where the model's own flight flag does, checked per frame), and the rule is
  applied by the harness's code, g4.flight_predictor (G4_RULE.md sections 2 (a) and 4, item 11).
- EMG: the EMG branch (detached body, leg torques with ForceHead's force, joint angles, EMG decoder), the
  log-envelopes of the right tibialis anterior, gastrocnemius medialis, vastus lateralis and biceps femoris
  (g4.MUSCLES; a muscle the decoder does not model reads NaN).
- metric-3D: the true 15 joints of the trial (g4.world_joints: canonical coords x scale plus the root track,
  y up, floor at 0) at camera frames CAMERA_FPS (G1 default: 30 fps) from the first pose frame on, linear
  interpolation between pose frames (a joint missing at either neighbour is missing), projected by one static
  camera per trial, camera_render.sample_camera (G1 default family 'generic') aimed at the trial-mean pelvis,
  then camera_render.render_pose2d with the default TrackerProcess, from one torch generator seeded with the
  first 63 bits of SHA-256('<person_id>\\n<recording key>\\nG4v1') (the trial key: stroke recording keys repeat
  across persons). A missing truth joint is held from its nearest present frame for the render only and then
  marked undetected (a frame without its pelvis has no joint). The 2D stream (layout camera1, camera generic,
  source the trial's source) is band-copied, windowed and inferred as above; the pose3d branch of render gives
  the 15 canonical joints with the source's joint offsets at the pose frames. MPJPE: g4.mpjpe on pelvis-relative
  joints after one yaw about +y per window fitted in closed form (mdfm.gates.g1.yaw_fit: the model's window frame
  is not the lab frame); PA-MPJPE: g4.pa_mpjpe (per-frame similarity Procrustes). Per person pooled over its
  trials' frames, mean over persons. The camera (G4_RULE.md section 8 item 5) and this MPJPE frame (item 10) are
  the rule's, closed on 2026-10-05 in the DRAFT rule.

Rows. Primary, through g4.evaluate with the model beside the Newton floor and the template, with the rule's lines
as the sealed read applies them (docs/rules/G4_RULE.md, revisions of 2026-10-05): GRF on the development stroke
patients (g4.stroke_patients(lockbox=False); rule section 2, the development secondary row (a)), the body mass of
g4.grf_trials (C3D, else the MAT file; a patient with neither dropped and listed, no plate-derived mass); EMG on the
gait120 dev fold with the template built on the gait120 train fold, trials under g4.EMG_MIN_S (1 s) of pose left
out before any predictor is called, every predictor scored on the same frames of the same (trial, muscle) pairs,
and the line emg_r_gain (delta g4.EMG_GAIN) on which emg_r's pass depends. Secondary: GRF on the able-bodied
persons of the stroke_gait_vancriekinge dev fold (mdfm.data.sources.folds), the same g4 functions; and the GRF row
without the plate series (trial, foot) whose plate peak at the scored samples stays below g4.LOW_PEAK_BW (0.5 BW;
rule section 2 (b), computed by g4.evaluate with low_peak), the Newton floor recomputed on its own samples, the
series left out listed (low_peak_series.csv). metric-3D on every paired trial of the stroke dev patients, those of a
patient dropped from the GRF rows for want of a mass included (the line needs no mass). Every statistic is
g4's: per person, then the mean (GRF, metric-3D) or median (EMG) over persons with the person bootstrap interval
(g4.BOOT_N resamples, seed g4.SEED). GateResult rows (g4.gate_results with emg_gain) are dev estimates.

Stroke set (--stroke-set, spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md section 4.12, task S2.10). 'all' (the default, the
run above unchanged): the 40 non-lockbox stroke patients, which a stage-2 checkpoint trained on in part, so the
disjointness check below refuses every stage-2 checkpoint. 'fold': the proposed stroke dev fold of stage 2, the 8
patients of group stroke and fold dev in reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/folds_proposed.csv
(stroke_fold), their SHA-256 (sources.person_hash) asserted equal to the one recorded in summary.json beside it and
to STROKE_FOLD_SHA256, every one of them among the non-lockbox stroke patients; the GRF and metric-3D rows then score
these 8 (S047 has no pose and plate pair, so the GRF row scores one patient fewer), and a checkpoint trained on any of
them is refused by the same disjointness check. The gait120 and able-bodied rows are the same under both sets. The
default output folder of the fold carries the suffix _strokefold.

Records left out (LEFT_OUT, and g4.LEFT_OUT for the harness's). An input stream that gives no window, fewer than
round(infer.MIN_S x fps) samples (1 s, the window store's minimum: a recording under 1 s never reaches training,
window_store.MIN_S), is left out: the pose input of a GRF trial, or the camera render of a metric-3D trial; so is a
metric-3D trial whose truth pelvis is never valid. The harness leaves out an EMG trial under g4.EMG_MIN_S of pose
(the same 1 s; reason pose_shorter_than_min_s) before the model is called, and a GRF trial of a patient without a
recorded body mass (reason no_body_mass). Each one is listed in left_out.csv and counted per source and reason (and
per line) in g4.json and g4.md, never dropped silently. A GRF trial left out predicts NaN, so g4.grf_samples scores
none of its samples for any predictor, the Newton floor included. An EMG trial left out is removed from the trial
table for the template and the model alike (an EMG trial the predictor itself leaves out is removed after the
table, and the person table and the summary are recomputed by g4.emg_person_table and g4.summary on the kept
trials); the template's line over every paired trial is kept as a diagnostic. A person whose trials are all left
out is unscored and listed.

Refusals, before any record opens (check_sets): a CODY name, a lockbox person (reports/lockbox/lockbox_persons.csv,
the lockbox stroke patients and the gait120 lockbox persons of the index) in any set, a template person among the
scored persons. The scored persons are asserted disjoint from the training lists of the checkpoint
(persons/*.txt beside it, and the lists stored in it). The SHA-256 of reports/gates/lockbox_reads.csv is read
before and after the run and both are reported.

Writes <out> (default reports/gates/g4/dev_<checkpoint name>, suffix _model for the raw weights): g4.json, g4.md,
per_person_grf.csv, per_trial_grf.csv, per_person_emg.csv, per_trial_emg.csv, per_person_grf_able_bodied.csv,
per_person_grf_no_low_peak.csv, low_peak_series.csv, per_person_metric3d.csv, per_trial_metric3d.csv, trials.csv
(one row per inference call), left_out.csv, _provenance.json.

Run (mdfm environment; evaluation contract --checkpoint, --weights, --device, --out, --memory-gb):
  <conda>/envs/mdfm/python.exe scripts/eval_g4_dev.py --checkpoint <run>/ckpt_B_<step>.pt
  ... --quick --out <scratch folder>   (2 stroke patients, 2 gait120 dev, 5 train and 2 able-bodied persons)
  ... --stroke-set fold                (a stage-2 checkpoint: the stroke dev fold of 8)
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.body.hermite import hermite_gather  # noqa: E402
from mdfm.data import canonical, sources  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.data import windows as W  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.eval import infer as I  # noqa: E402
from mdfm.gates import common, g4  # noqa: E402
from mdfm.gates import g1 as G1  # noqa: E402
from mdfm.kernels import emg as emg_kernel  # noqa: E402
from mdfm.model import mfm as M  # noqa: E402

RUN = "g4_dev"
WHAT = "G4 development estimate of a checkpoint, not a gate read"
POSE_LAYOUT = "mocap3d"
CAMERA_LAYOUT = "camera1"
CAMERA_FAMILY = G1.PARAMS["camera_family"]            # 'generic', the G1c camera
CAMERA_FPS = G1.CAMERA_FPS                            # 30 fps, the G1c camera
CAMERA_TAG = "G4v1"
METRIC3D_STATUS = ("the camera (docs/rules/G4_RULE.md section 8 item 5, the G1c draw) and the MPJPE frame (item 10) "
                   "of the DRAFT rule, closed on 2026-10-05")
GRID_TOL_S = 1e-6                                     # pose times must sit on t[0] + i / fps within this
CONTACT_CLAMP = 1e-6                                  # contact probabilities clamped before the logit
QUICK = {"stroke": 2, "gait120_dev": 2, "gait120_train": 5, "able_bodied": 2}
SCORED_SETS = ("stroke_dev", "gait120_dev", "able_bodied_dev")
LEFT_OUT = {                                          # reason: meaning (module docstring, Records left out)
    "pose_shorter_than_a_window": f"pose input under {I.MIN_S:g} s (infer.MIN_S): no window",
    "camera_shorter_than_a_window": f"camera render under {I.MIN_S:g} s (infer.MIN_S): no window",
    "camera_no_pelvis": "truth pelvis never valid: no camera render",
}
LEFT_OUT_COLUMNS = ("line", "source", "person_id", "key", "stream", "reason", "samples", "fps", "seconds",
                    "min_samples")
STROKE_SETS = ("all", "fold")                         # --stroke-set (module docstring, Stroke set)
STROKE_FOLD_DIR = os.path.join(TREE, "reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06")
STROKE_FOLD_CSV = os.path.join(STROKE_FOLD_DIR, "folds_proposed.csv")
STROKE_FOLD_SUMMARY = os.path.join(STROKE_FOLD_DIR, "summary.json")
STROKE_FOLD_KEY = "stroke_gait_vancriekinge:stroke"   # the unit key the fold was drawn with (summary.json folds)
STROKE_FOLD_GROUP = "stroke"
STROKE_FOLD_SHA256 = "0604a51f5d82c40746b1931ae0ff4403180175e6aac4cd4a90b153763666a70c"   # spec sections 2 and 4.12
STROKE_FOLD_SUFFIX = "_strokefold"


# ---------------------------------------------------------------- sets and refusals
def able_bodied_dev():
    """Sorted person ids of the able-bodied stroke_gait_vancriekinge persons of the dev fold (sources.folds), outside
    the lockbox, with pose rows in the index (index rows only)."""
    f = sources.folds()
    dev = set(f.loc[(f["source"] == g4.STROKE) & (f["fold"] == "dev"), "person_id"])
    df = canonical.index()
    sel = ((df["source"] == g4.STROKE) & (df["group"] == "able_bodied") & (df["modality"] == "pose")
           & ~df["lockbox"].astype(bool))
    return sorted(dev & set(df.loc[sel, "person_id"]))


def lockbox_names():
    """Every lockbox person id the refusals use: reports/lockbox/lockbox_persons.csv (checked against its hash), the
    lockbox stroke patients and the gait120 lockbox persons of the index (index rows only)."""
    return frozenset(WS.lockbox_persons()) | frozenset(g4.stroke_patients(lockbox=True)) \
        | frozenset(g4.gait120_persons("lockbox"))


def check_sets(sets, template_key="gait120_template"):
    """Refused (canonical.Refused) before any record opens when a set ({name: person ids}) holds a CODY name or a
    lockbox person, or when the template persons meet a scored set; otherwise {name: count}."""
    locked = lockbox_names()
    for name, persons in sets.items():
        persons = [str(p) for p in persons]
        cody = [p for p in persons if canonical.is_cody(p)]
        if cody:
            raise Refused(f"CODY is evaluation-only and never a G4 set: {name} {cody[:3]}")
        bad = sorted(set(persons) & locked)
        if bad:
            raise Refused(f"lockbox persons in the development set {name}: {bad[:3]}; the lockbox is read by "
                          "mdfm.gates.g4.read_lockbox only")
    tmpl = set(sets.get(template_key, ()))
    for name, persons in sets.items():
        if name != template_key and tmpl & set(persons):
            raise Refused(f"template persons among the scored set {name}: {sorted(tmpl & set(persons))[:3]}")
    return {name: len(persons) for name, persons in sets.items()}


def stroke_fold(path=None, summary=None, expected=STROKE_FOLD_SHA256):
    """(sorted person ids, info) of the proposed stroke dev fold of stage 2 (module docstring, Stroke set): the rows
    of folds_proposed.csv with source stroke_gait_vancriekinge, group stroke and fold dev. ValueError when their
    SHA-256 differs from summary.json's sha256_dev of STROKE_FOLD_KEY or from `expected`; Refused when one is a CODY
    name or a lockbox person (names only, no record)."""
    path = path or STROKE_FOLD_CSV
    summary = summary or STROKE_FOLD_SUMMARY
    dev = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == g4.STROKE and r["group"] == STROKE_FOLD_GROUP and r["fold"] == "dev":
                dev.append(str(r["person_id"]))
    dev = sorted(dev)
    with open(summary, encoding="utf-8") as f:
        rec = json.load(f)["folds"][STROKE_FOLD_KEY]
    got = sources.person_hash(dev)
    if got != rec["sha256_dev"] or (expected is not None and got != expected):
        raise ValueError(f"the stroke dev fold of {common.rel(path)} hashes {got[:12]}, against "
                         f"{str(rec['sha256_dev'])[:12]} in {common.rel(summary)} and "
                         f"{str(expected)[:12]} expected")
    if len(dev) != int(rec["dev"]):
        raise ValueError(f"{len(dev)} stroke dev persons in {common.rel(path)}, {rec['dev']} in {common.rel(summary)}")
    check_sets({"stroke_fold": dev})
    return dev, {"set": "fold", "path": common.rel(path), "summary": common.rel(summary), "key": STROKE_FOLD_KEY,
                 "persons": len(dev), "sha256": got, "pool_persons": rec.get("persons"),
                 "pool_sha256": rec.get("sha256_pool")}


def stroke_set(name, fold_path=None, fold_summary=None):
    """(sorted stroke person ids, info) of a --stroke-set value: 'all', g4.stroke_patients(lockbox=False); 'fold',
    stroke_fold(), each of whose persons must be among those (ValueError otherwise)."""
    if name not in STROKE_SETS:
        raise ValueError(f"stroke set {name!r}, expected one of {STROKE_SETS}")
    pool = list(g4.stroke_patients(lockbox=False))
    if name == "all":
        return pool, {"set": "all", "persons": len(pool), "sha256": sources.person_hash(pool)}
    dev, info = stroke_fold(fold_path, fold_summary)
    outside = sorted(set(dev) - set(pool))
    if outside:
        raise ValueError(f"stroke fold persons outside the non-lockbox stroke patients: {outside[:3]}")
    return dev, info


def person_lists(checkpoint):
    """{source: frozenset of person ids} of the training lists persons/<source>.txt in the checkpoint's folder
    (mdfm/train/checkpoint.py writes them at every save); Refused when the folder or its lists are missing."""
    folder = os.path.join(os.path.dirname(os.path.abspath(str(checkpoint))), "persons")
    if not os.path.isdir(folder):
        raise Refused(f"no persons/ folder beside the checkpoint: {folder}")
    out = {}
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".txt") and not fn.startswith((".", "_")):
            with open(os.path.join(folder, fn), encoding="utf-8") as f:
                out[fn[:-4]] = frozenset(line.strip() for line in f if line.strip())
    if not out:
        raise Refused(f"no training list in {folder}")
    return out


def assert_disjoint(scored, lists):
    """Refused when a scored person ({set name: ids}) is in a training list ({list name: ids}); otherwise {set
    name: 0}, the overlaps checked."""
    trained = set().union(*[set(v) for v in lists.values()]) if lists else set()
    hits = {}
    for name, persons in scored.items():
        both = sorted(set(persons) & trained)
        if both:
            hits[name] = both
    if hits:
        raise Refused("scored persons among the checkpoint's training persons: "
                      + "; ".join(f"{k} {v[:3]}" for k, v in hits.items()))
    return {name: 0 for name in scored}


def camera_seed(person_id, key, tag=CAMERA_TAG):
    """The camera seed of a trial: the first 63 bits of SHA-256('<person_id>\\n<key>\\n<tag>')."""
    h = hashlib.sha256(f"{person_id}\n{key}\n{tag}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def sha256_or_none(path):
    return common.sha256_file(path) if os.path.exists(path) else None


# ---------------------------------------------------------------- streams and windows
def pose_stream(pr, layout=POSE_LAYOUT):
    """(band copy windows.Stream, times float64 (T,) on the recording clock) of the pose3d record of a PairedRecord
    with its root track attached (module docstring); t of the stream from the first sample. ValueError when the
    pose times are off the fps grid or the band copy drops a sample."""
    rec = W.whitelist(pr.streams["pose3d"])
    fps = W.record_fps(rec)
    values, cm, valid, meta = W.arrays("pose3d", rec)
    extra = (getattr(pr, "meta", None) or {}).get("pose3d") or {}
    meta.update({k: v for k, v in extra.items() if k in W.META_KEYS})
    meta["source"] = pr.source
    meta["layout"] = layout
    t64 = np.asarray(pr.t["pose3d"], dtype=np.float64)
    if t64.shape != (values.shape[0],):
        raise ValueError(f"{pr.key}: {t64.shape[0]} pose times for {values.shape[0]} samples")
    if len(t64) and np.abs(t64 - (t64[0] + np.arange(len(t64)) / fps)).max() > GRID_TOL_S:
        raise ValueError(f"{pr.person_id} {pr.key}: pose times off the {fps:g} Hz grid")
    W.check_root(meta, "pose3d", len(t64), f"{pr.key} pose3d")
    full = W.Stream("pose3d", "input", fps, t64 - t64[0], values, cm, valid, meta)
    b, info = WS.band_copy(full, return_info=True)
    if len(info.idx) != len(t64):
        raise ValueError(f"{pr.key}: the pose band copy kept {len(info.idx)} of {len(t64)} samples")
    return b, t64[info.idx]


def cut_windows(stream, t64, source, person_id, key, cov, win_s=I.WIN_S, hop_s=I.HOP_S, min_s=I.MIN_S):
    """(windows, spans) of a band-copied stream on times t64 (s, recording clock): infer.spans, each window from its
    first sample's time, lasting n / fps s, its meta (root track included) sliced with its samples."""
    sp = I.spans(len(t64), stream.fps, win_s, hop_s, min_s)
    out = []
    for s, n in sp:
        sel = slice(s, s + n)
        t0 = float(t64[s])
        st = W.Stream(modality=stream.modality, role=stream.role, fps=float(stream.fps),
                      t=(t64[sel] - t0).astype(np.float32), values=np.ascontiguousarray(stream.values[sel]),
                      chan_mask=stream.chan_mask.copy(), valid=np.ascontiguousarray(stream.valid[sel]),
                      meta=W.slice_meta(stream.meta, sel))
        out.append(W.validate(W.Window(source=source, person_id=str(person_id), key=f"{key}@{s}", synthetic=False,
                                       t0=t0, duration=n / float(stream.fps), streams=[st], cov=dict(cov))))
    return out, sp


def interp_frames(x, t, tq):
    """x (T, ...) at times t (T,) read at tq (Q,) by linear interpolation between the two nearest frames (clamped at
    the ends; a query on a frame reads that frame); NaN where a neighbour it weighs is NaN."""
    t = np.asarray(t, dtype=np.float64)
    tq = np.asarray(tq, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(t) == 1:
        return np.repeat(x[:1], len(tq), axis=0)
    i1 = np.clip(np.searchsorted(t, tq, side="right"), 1, len(t) - 1)
    i0 = i1 - 1
    a = np.clip((tq - t[i0]) / (t[i1] - t[i0]), 0.0, 1.0).reshape((-1,) + (1,) * (x.ndim - 1))
    with np.errstate(invalid="ignore"):
        out = (1.0 - a) * x[i0] + a * x[i1]
    return np.where(a == 0.0, x[i0], np.where(a == 1.0, x[i1], out))


def hold_fill(x):
    """x (Q, J, 3) with NaN rows filled per joint from the nearest frame where that joint is finite (a joint never
    finite takes joint 0's filled track); None when joint 0 is never finite."""
    x = np.array(x, dtype=np.float64)
    ok = np.isfinite(x).all(-1)
    if not ok[:, 0].any():
        return None
    order = [0] + [j for j in range(x.shape[1]) if j != 0]
    for j in order:
        good = np.flatnonzero(ok[:, j])
        if not len(good):
            x[:, j] = x[:, 0]
            continue
        k = np.clip(np.searchsorted(good, np.arange(len(x))), 1, len(good) - 1) if len(good) > 1 else None
        if k is None:
            near = np.full(len(x), good[0])
        else:
            lo, hi = good[k - 1], good[k]
            idx = np.arange(len(x))
            near = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
        x[~ok[:, j], j] = x[near[~ok[:, j]], j]
    return x


def camera_stream(world, t, seed, source, fps=CAMERA_FPS, family=CAMERA_FAMILY, tracker=None):
    """(band copy pose2d windows.Stream, camera times float64 (Q,) on the recording clock, info) of the metric-3D
    input of a trial (module docstring): world (T, 15, 3) m y up with NaN where invalid, t (T,) s; seed of the
    torch generator (sample_camera, then render_pose2d). None when the pelvis is never valid."""
    from mdfm.factory.camera_render import render_pose2d, sample_camera
    t = np.asarray(t, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    fin = np.isfinite(world).all(-1)
    if not fin[:, 0].any():
        return None
    tc = t[0] + np.arange(0.0, (t[-1] - t[0]) + 1e-9, 1.0 / fps)
    xc = interp_frames(world, t, tc)
    missing = ~np.isfinite(xc).all(-1)                                      # (Q, 15)
    filled = hold_fill(xc)
    if filled is None:
        return None
    target = np.nanmean(world[fin[:, 0], 0], axis=0)
    gen = torch.Generator().manual_seed(int(seed))
    cam = sample_camera(gen, torch.as_tensor(target, dtype=torch.float64), family=family, fps=fps)
    record, _ = render_pose2d(torch.as_tensor(filled, dtype=torch.float64), cam, gen, tracker)
    valid = np.asarray(record["valid_mask"], dtype=bool) & ~missing & ~missing[:, :1]
    vals = np.zeros(record["coords"].shape, dtype=np.float32)
    vals[..., :2] = np.where(valid[..., None], np.asarray(record["coords"])[..., :2], 0.0)
    cm = np.zeros((vals.shape[1], 3), dtype=bool)
    cm[:, :2] = True
    meta = {"dim": 2, "scale": float(record["scale"]), "unit_to_si": None, "camera": family, "layout": CAMERA_LAYOUT,
            "source": source}
    st = W.Stream("pose2d", "input", float(record["fps"]), tc - tc[0], vals, cm, valid, meta)
    b, info = WS.band_copy(st, return_info=True)
    params = torch.cat([cam.K.reshape(-1), cam.R.reshape(-1), cam.t.reshape(-1)]).tolist()
    return b, tc[info.idx], {"frames": int(len(tc)), "detected_share": float(valid.mean()) if valid.size else 0.0,
                             "truth_missing_share": float(missing.mean()) if missing.size else 0.0,
                             "scale_px": float(record["scale"]), "camera_KRt": params}


def assign_frames(t_frames, t0, knots, rate=I.RATE):
    """(window (T,) int64, -1 where no window covers the frame; rel (T,) s from that window's start): each frame goes
    to the window whose knot span [0, (K - 1) / rate] holds it (with half a knot of tolerance past the end, read
    at the end knot) nearest the span's centre, the earlier window on a tie."""
    t = np.asarray(t_frames, dtype=np.float64)
    win = np.full(len(t), -1, dtype=np.int64)
    rel = np.zeros(len(t))
    best = np.full(len(t), np.inf)
    for j, (s, k) in enumerate(zip(t0, knots)):
        r = t - float(s)
        end = (int(k) - 1) / float(rate)
        inside = (r >= -GRID_TOL_S) & (r <= end + 0.5 / float(rate))
        c = np.abs(r - end / 2.0)
        better = inside & (c < best)
        win[better] = j
        best[better] = c[better]
        rel[better] = np.clip(r[better], 0.0, end)
    return win, rel


def yaw_aligned(pred, truth, win_of):
    """Pelvis-relative predicted joints (T, 15, 3) turned about +y by one yaw per window (g1.yaw_fit over the entries
    finite in both, pelvis-relative), NaN where the prediction is missing; truth (T, 15, 3) in the lab frame."""
    P = np.asarray(pred, dtype=np.float64) - np.asarray(pred, dtype=np.float64)[:, :1]
    Q = np.asarray(truth, dtype=np.float64) - np.asarray(truth, dtype=np.float64)[:, :1]
    out = np.full(P.shape, np.nan)
    for j in np.unique(win_of[win_of >= 0]):
        sel = win_of == j
        ok = np.isfinite(P[sel]).all(-1) & np.isfinite(Q[sel]).all(-1)
        th = G1.yaw_fit(np.where(ok[..., None], Q[sel], 0.0), np.where(ok[..., None], P[sel], 0.0))
        out[sel] = G1.turn_y(P[sel], th)
    return out


# ---------------------------------------------------------------- the model as a G4 predictor
class G4Predictor:
    """The model as the GRF and EMG predictor of mdfm.gates.g4 and the metric-3D round trip (module docstring).

    grf(trial) -> (T, 2) BW at a g4.GRFTrial's pose frames: on the first trial of a person, the person's stroke
    pairs are read again (g4.paired_records, flags False), each run through the model, the predictions cached by
    (person, key), and the metric-3D trial run when the person is in metric3d_persons. grf_contact(trial) -> (bw,
    contact) of the same trial, the sealed read's GRF predictor contract (the flight rule left to
    g4.flight_predictor). metric3d_person(person): the metric-3D trials of a person without GRF trials (no
    recorded body mass). emg(trial) -> (T, M) log-
    envelopes at a g4.EMGTrial's pose frames from its record (keep_record). predict_pose(pr, outputs) and
    metric3d_trial(pr) work on one PairedRecord; frames(windows, bodies, times, outputs) reads per-frame outputs of
    inferred windows. log: one dict per inference call; echo: a callable given one line per call (None: silent).
    attach_root: the root reader (default window_store.attach_raw_root; a record whose pose3d meta already holds a
    root keeps it). left_out: one dict per input stream left out (module docstring, Records left out; leave_out):
    predict_pose and metric3d_trial return None for it, grf() and grf_contact() then read NaN at every frame
    (flight() None) and emg() NaN."""

    def __init__(self, model, refine_steps=I.STEPS_EVAL, points=I.POINTS_EVAL, win_s=I.WIN_S, hop_s=I.HOP_S,
                 min_s=I.MIN_S, batch_windows=I.BATCH_WINDOWS, seed=0, pool=True, metric3d_persons=(), attach_root=None,
                 camera_tag=CAMERA_TAG, echo=None):
        self.model = model
        self.echo = echo
        self.refine_steps, self.points = int(refine_steps), points
        self.win_s, self.hop_s, self.min_s = float(win_s), float(hop_s), float(min_s)
        self.batch_windows, self.seed, self.pool = max(1, int(batch_windows)), int(seed), bool(pool)
        self.metric3d_persons = set(metric3d_persons)
        self.attach_root = WS.attach_raw_root if attach_root is None else attach_root
        self.camera_tag = camera_tag
        self.log = []
        self.read_stats = collections.Counter()
        self.m3d = []
        self.left_out = []
        self._grf, self._done = {}, set()
        chans = [{"muscle": m, "side": g4.EMG_SIDE} for m in g4.MUSCLES]
        mus, side, modelled = emg_kernel.channel_table(chans)
        self.emg_muscle, self.emg_side, self.emg_modelled = mus, side, modelled

    # ------------------------------------------------------------ refusals and inputs
    @staticmethod
    def refuse(pr):
        """Refused for a CODY name or a lockbox person of a PairedRecord (names only, no read)."""
        I.refuse_cody(pr.source, pr.subject_ns, pr.person_id)
        if pr.person_id in WS.lockbox_persons() or pr.subject_ns in WS.lockbox_persons():
            raise Refused(f"lockbox person: {pr.source} {pr.person_id}; G4 dev runs never read the lockbox")

    def with_root(self, pr):
        """The record with its root track (attach_root unless its pose3d meta already holds one)."""
        self.refuse(pr)
        meta = ((getattr(pr, "meta", None) or {}).get("pose3d")) or {}
        if not any(k in meta for k in W.ROOT_KEYS):
            pr = self.attach_root(pr)
        return pr

    @staticmethod
    def trial_id(pr):
        return f"{pr.person_id}/{pr.key}"

    def leave_out(self, pr, stream, reason, samples, fps):
        """Records one input stream of a PairedRecord left out (module docstring, Records left out): stream 'pose'
        or 'camera', reason a key of LEFT_OUT, samples and fps of the stream."""
        if reason not in LEFT_OUT:
            raise ValueError(f"unknown reason {reason!r}, expected one of {sorted(LEFT_OUT)}")
        fps = float(fps)
        self.left_out.append({"source": pr.source, "person_id": pr.person_id, "key": pr.key, "stream": stream,
                              "reason": reason, "samples": int(samples), "fps": fps,
                              "seconds": round(int(samples) / fps, 4),
                              "min_samples": max(1, int(round(self.min_s * fps)))})
        if self.echo:
            self.echo(f"  {stream} {self.trial_id(pr)}: left out, {reason} ({int(samples)} samples at {fps:g} Hz)")

    # ------------------------------------------------------------ inference
    def infer(self, wins, tid, kind):
        """infer.infer of a recording's windows (shape pooled over them); one log row."""
        st = {}
        bodies = I.infer(self.model, wins, refine_steps=self.refine_steps, points=self.points,
                         groups=[tid] * len(wins) if self.pool else None, batch_windows=self.batch_windows,
                         seed=I.seed_of(self.seed, kind, tid), stats=st)
        self.log.append({"kind": kind, "trial_id": tid, "source": wins[0].source, "person_id": wins[0].person_id,
                         "windows": len(wins), "samples": int(sum(len(w.streams[0].t) for w in wins)),
                         "refine_first": st.get("refine_first"), "refine_last": st.get("refine_last"),
                         "seconds": st.get("seconds")})
        if self.echo:
            self.echo(f"  {kind} {tid}: {len(wins)} windows, {st.get('seconds') or 0.0:.1f} s")
        return bodies

    def _batch(self, wins, bodies, w, t, outputs):
        """(KnotBody, Batch) of a chunk of windows: the point bodies padded with their last knot, kernel points at
        (w, t) for the force, emg and pose3d branches of MFM.render."""
        dev = self.model.device
        T = max(len(b.rot6d) for b in bodies)

        def pad(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            n = x.shape[0]
            return torch.cat([x, x[-1:].expand((T - n,) + tuple(x.shape[1:]))]) if n < T else x

        logits = []
        for b in bodies:
            c = np.clip(np.asarray(b.contact, dtype=np.float64), CONTACT_CLAMP, 1.0 - CONTACT_CLAMP)
            logits.append(pad(np.log(c) - np.log1p(-c)))
        lengths = torch.as_tensor([len(b.rot6d) for b in bodies], dtype=torch.int64)
        gender = torch.as_tensor([int(b.gender) for b in bodies], dtype=torch.int64)
        body = M.KnotBody(torch.stack([pad(b.rot6d) for b in bodies]).to(dev),
                          torch.stack([pad(b.transl) for b in bodies]).to(dev), torch.stack(logits).to(dev),
                          torch.as_tensor(np.stack([b.beta for b in bodies]), dtype=torch.float32).to(dev),
                          gender.to(dev), lengths.to(dev))
        N = len(w)
        wt = torch.as_tensor(np.asarray(w), dtype=torch.int64, device=dev)
        tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=dev)
        one, zero = torch.ones(N, device=dev), torch.zeros(N, dtype=torch.int64, device=dev)

        def points(shape, extra):
            """Kernel points at (w, t); render reads w, t and extra only."""
            return M.Points(wt, tt, torch.zeros(shape, device=dev), torch.ones(shape[:2], dtype=torch.bool, device=dev),
                            one, zero, extra)

        pts = {}
        if "grf" in outputs:
            pts["force"] = points((N, 2), {})
        if "emg" in outputs:
            C = len(self.emg_muscle)
            per = {"muscle": self.emg_muscle, "side": self.emg_side}
            pts["emg"] = points((N, C), {k: torch.as_tensor(v, device=dev).expand(N, C).contiguous()
                                         for k, v in per.items()})
        if "joints15" in outputs:
            pts["pose3d"] = points((N, 15, 3), {})
        srcs = tuple(self.model.cfg.sources)
        batch = M.Batch(tokens=None, sex=torch.as_tensor([int(x.cov["sex"]) for x in wins], dtype=torch.int64),
                        task=torch.as_tensor([int(x.cov["task"]) for x in wins], dtype=torch.int64), gender=gender,
                        source_id=torch.as_tensor([srcs.index(x.source) for x in wins], dtype=torch.int64),
                        person=torch.zeros(len(wins), dtype=torch.int64), lengths=lengths, points=pts, win={},
                        knots=(torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)), truth=None,
                        sources=srcs, families=tuple(self.model.heads.families), rate=float(self.model.rate))
        return body, batch

    def contact(self, body, pts):
        """The model's contact probabilities (N, 2) at force kernel points, as MFM.render reads them for its flight
        flag: the body's contact logits interpolated at (w, t) (hermite_gather, the body's dtype, autocast off),
        then the sigmoid."""
        dt = torch.float64 if body.rot6d.dtype == torch.float64 else torch.float32
        with M._no_autocast(body.rot6d.device):
            lg = hermite_gather(body.contact_logit.to(dt), pts.w, pts.t.to(dt), float(self.model.rate), body.lengths)
            return torch.sigmoid(lg)

    @torch.no_grad()
    def frames(self, wins, bodies, t_frames, outputs):
        """Per-frame outputs at times t_frames (T,) (recording clock) of inferred windows ({name: array}): grf (T, 2)
        BW with flight frames at 0 and flight (T,) bool, and the pair the sealed read's GRF predictor gives, bw
        (T, 2) before the flight rule and contact (T, 2) probabilities (contact), whose flight frames
        (g4.apply_flight) must be the model's own flight flag (ValueError otherwise); emg (T, M) log-envelopes, NaN
        where the decoder does not model the muscle; joints15 (T, 15, 3) m in each frame's window frame; NaN where
        no window covers a frame. Also returns the window of each frame (T,)."""
        rate = float(self.model.rate)
        win_of, rel = assign_frames(t_frames, [w.t0 for w in wins], [len(b.rot6d) for b in bodies], rate)
        T = len(win_of)
        res = {}
        if "grf" in outputs:
            res["grf"] = np.full((T, 2), np.nan)
            res["flight"] = np.zeros(T, dtype=bool)
            res["bw"] = np.full((T, 2), np.nan)
            res["contact"] = np.full((T, 2), np.nan)
        if "emg" in outputs:
            res["emg"] = np.full((T, len(self.emg_muscle)), np.nan)
        if "joints15" in outputs:
            res["joints15"] = np.full((T, 15, 3), np.nan)
        B = self.batch_windows
        for a in range(0, len(wins), B):
            sel = np.flatnonzero((win_of >= a) & (win_of < a + B))
            if not len(sel):
                continue
            body, batch = self._batch(wins[a:a + B], bodies[a:a + B], win_of[sel] - a, rel[sel], outputs)
            pred = self.model.render(body, batch, M.Nuisance())
            if "force" in pred:
                bw = pred["force"]["bw"].float().cpu().numpy().astype(np.float64)
                fl = pred["force"]["flight"].cpu().numpy().astype(bool)
                p = self.contact(body, batch.points["force"]).float().cpu().numpy().astype(np.float64)
                _, fl_rule = g4.apply_flight(bw, p)
                if not np.array_equal(fl_rule, fl):
                    raise ValueError(f"the contact probabilities read flight at {int((fl_rule != fl).sum())} of "
                                     f"{len(fl)} frames where the model's own flight flag differs")
                res["grf"][sel] = np.where(fl[:, None], 0.0, bw)
                res["flight"][sel] = fl
                res["bw"][sel] = bw
                res["contact"][sel] = p
            if "emg" in pred:
                le = pred["emg"]["log_env"].float().cpu().numpy().astype(np.float64)
                ok = pred["emg"]["mask"].cpu().numpy().astype(bool)
                res["emg"][sel] = np.where(ok, le, np.nan)
            if "pose3d" in pred:
                res["joints15"][sel] = pred["pose3d"]["joints"].float().cpu().numpy()
        return res, win_of

    # ------------------------------------------------------------ one recording
    def predict_pose(self, pr, outputs=("grf", "emg")):
        """{output: per-frame array} at the pose frames pr.t['pose3d'] of a PairedRecord (module docstring), plus
        'windows' (the window of each frame); None when the record gives no window (left out, leave_out)."""
        pr = self.with_root(pr)
        stream, t64 = pose_stream(pr)
        cov = W.covariates(dict(pr.streams["pose3d"], key=pr.key), pr.source)
        wins, _ = cut_windows(stream, t64, pr.source, pr.person_id, pr.key, cov, self.win_s, self.hop_s, self.min_s)
        if not wins:
            self.leave_out(pr, "pose", "pose_shorter_than_a_window", len(t64), stream.fps)
            return None
        bodies = self.infer(wins, self.trial_id(pr), "pose")
        res, win_of = self.frames(wins, bodies, t64, outputs)
        if "flight" in res:
            self.log[-1]["flight_share"] = float(res["flight"].mean()) if len(res["flight"]) else None
        res["windows"] = win_of
        return res

    def metric3d_trial(self, pr):
        """The metric-3D round trip of one stroke trial (module docstring); appends {person, key, pred, truth,
        windows, camera info} to m3d and returns it; None when the trial is left out (pelvis never valid, or a
        camera stream without a window: leave_out)."""
        pr = self.with_root(pr)
        meta = pr.meta["pose3d"]
        world = g4.world_joints(pr.streams["pose3d"], np.asarray(meta["root"], dtype=np.float64),
                                np.asarray(meta["root_valid"], dtype=bool))
        t = np.asarray(pr.t["pose3d"], dtype=np.float64)
        cam = camera_stream(world, t, camera_seed(pr.person_id, pr.key, self.camera_tag), pr.source)
        if cam is None:
            self.leave_out(pr, "camera", "camera_no_pelvis", len(t), W.record_fps(W.whitelist(pr.streams["pose3d"])))
            return None
        stream, tc, info = cam
        cov = W.covariates(dict(pr.streams["pose3d"], key=pr.key), pr.source)
        wins, _ = cut_windows(stream, tc, pr.source, pr.person_id, pr.key, cov, self.win_s, self.hop_s, self.min_s)
        if not wins:
            self.leave_out(pr, "camera", "camera_shorter_than_a_window", len(tc), stream.fps)
            return None
        bodies = self.infer(wins, self.trial_id(pr), "camera")
        res, win_of = self.frames(wins, bodies, t, ("joints15",))
        self.log[-1]["detected_share"] = info["detected_share"]
        entry = {"person_id": pr.person_id, "key": pr.key, "pred": yaw_aligned(res["joints15"], world, win_of),
                 "truth": world - world[:, :1], "windows": len(wins), **info}
        self.m3d.append(entry)
        return entry

    # ------------------------------------------------------------ the g4 callables
    def _stroke_person(self, person):
        for pr in g4.paired_records(g4.STROKE, [person], stats=self.read_stats):
            t = np.asarray(pr.t["pose3d"], dtype=np.float64)
            res = self.predict_pose(pr, ("grf",))
            if res is None:                                   # left out: no sample of the trial is scored
                nan = np.full((len(t), 2), np.nan)
                self._grf[(pr.person_id, pr.key)] = {"t": t, "grf": nan, "flight": None, "bw": nan, "contact": nan}
            else:
                self._grf[(pr.person_id, pr.key)] = {"t": t, **{k: res[k] for k in ("grf", "flight", "bw", "contact")}}
            if pr.person_id in self.metric3d_persons:
                self.metric3d_trial(pr)
        self._done.add(person)

    def metric3d_person(self, person):
        """The metric-3D round trip of every paired stroke trial of a person whose GRF trials the harness dropped
        (no recorded body mass, g4.grf_trials): grf() never runs for them, and the metric-3D line needs no mass
        (G4_RULE.md sections 1, 3 and 4)."""
        for pr in g4.paired_records(g4.STROKE, [person], stats=self.read_stats):
            self.metric3d_trial(pr)

    def _trial(self, trial):
        """The cached predictions of a g4.GRFTrial (class docstring), checked against its pose frames."""
        k = (trial.person, trial.key)
        if k not in self._grf and trial.person not in self._done:
            self._stroke_person(trial.person)
        if k not in self._grf:
            raise KeyError(f"no model prediction for {k}: the trial did not pair on the second read")
        got = self._grf[k]
        if len(got["t"]) != len(trial.t) or not np.allclose(got["t"], trial.t, atol=GRID_TOL_S):
            raise ValueError(f"{k}: model frames differ from the trial's pose frames")
        return got

    def grf(self, trial):
        """(T, 2) BW at the pose frames of a g4.GRFTrial with the model's own flight flag applied (class
        docstring)."""
        return self._trial(trial)["grf"]

    def grf_contact(self, trial):
        """(bw, contact), each (T, 2) at the pose frames of a g4.GRFTrial: the model's per-foot BW before the
        flight rule and its contact probabilities, the GRF predictor contract of g4.read_lockbox. run_dev passes it
        through g4.flight_predictor, so the development row applies the sealed read's flight code (G4_RULE.md
        section 2 (a)). A trial left out reads NaN in both."""
        got = self._trial(trial)
        return got["bw"], got["contact"]

    def flight(self, trial):
        """(T,) bool: the frames of a GRF trial the model predicts in flight (after grf()); None for a trial left
        out."""
        return self._grf[(trial.person, trial.key)]["flight"]

    def emg(self, trial):
        """(T, M) log-envelopes at the pose frames of a g4.EMGTrial (its record, keep_record)."""
        if trial.record is None:
            raise ValueError(f"{trial.person} {trial.key}: the EMG trial carries no record (keep_record)")
        res = self.predict_pose(trial.record, ("emg",))
        if res is None:                                       # left out: run_dev drops the trial for every predictor
            return np.full((len(trial.t), len(self.emg_muscle)), np.nan)
        if res["emg"].shape[0] != len(trial.t):
            raise ValueError(f"{trial.person} {trial.key}: {res['emg'].shape[0]} model frames for {len(trial.t)}")
        return res["emg"]


# ---------------------------------------------------------------- tables
def r4(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if math.isfinite(v) else None


def grf_trial_table(trials, predictors):
    """Per (trial, foot): samples scored (plate valid, every prediction finite), plate peak, and per predictor the
    RMSE (%BW) and Pearson r."""
    import pandas as pd
    rows = []
    for tr in trials:
        preds = {k: np.asarray(f(tr)) for k, f in predictors.items()}
        for j, foot in enumerate(("L", "R")):
            ok = tr.obs_valid[:, j].copy()
            for p in preds.values():
                ok &= np.isfinite(p[:, j])
            o = tr.obs[ok, j]
            row = {"person_id": tr.person, "key": tr.key, "foot": foot, "samples": int(ok.sum()),
                   "obs_peak_bw": r4(o.max()) if ok.any() else None}
            for k, p in preds.items():
                row[f"{k}_rmse_pct_bw"] = r4(100 * np.sqrt(np.mean((p[ok, j] - o) ** 2))) if ok.any() else None
                row[f"{k}_r"] = r4(g4.pearson(p[ok, j], o))
            rows.append(row)
    return pd.DataFrame(rows)


def metric3d_tables(entries):
    """(per trial, per person) DataFrames of the metric-3D round trip: frames scored (truth pelvis valid and a
    prediction), joint entries, MPJPE and PA-MPJPE (mm, g4.mpjpe and g4.pa_mpjpe), detected 2D share; per person
    pooled over its trials' frames."""
    import pandas as pd
    trial_rows, by = [], collections.defaultdict(list)
    for e in entries:
        P, Q = e["pred"], e["truth"]
        both = np.isfinite(P).all(-1) & np.isfinite(Q).all(-1)
        frames = both.any(-1)
        trial_rows.append({"person_id": e["person_id"], "key": e["key"], "frames": int(frames.sum()),
                           "joints": int(both.sum()), "camera_frames": e["frames"],
                           "detected_share": r4(e["detected_share"]),
                           "mpjpe_mm": r4(g4.mpjpe(P[frames], Q[frames])) if frames.any() else None,
                           "pa_mpjpe_mm": r4(g4.pa_mpjpe(P[frames], Q[frames])) if frames.any() else None})
        if frames.any():
            by[e["person_id"]].append((P[frames], Q[frames], e))
    person_rows = []
    for person, items in sorted(by.items()):
        P = np.concatenate([x[0] for x in items])
        Q = np.concatenate([x[1] for x in items])
        person_rows.append({"person_id": person, "trials": len(items), "frames": int(len(P)),
                            "detected_share": float(np.mean([x[2]["detected_share"] for x in items])),
                            "mpjpe_mm": g4.mpjpe(P, Q), "pa_mpjpe_mm": g4.pa_mpjpe(P, Q)})
    cols = ["person_id", "trials", "frames", "detected_share", "mpjpe_mm", "pa_mpjpe_mm"]
    return pd.DataFrame(trial_rows), pd.DataFrame(person_rows, columns=cols)


def line_of(rec, stroke=(), able=()):
    """The line of a stream left out: metric3d (camera), emg (gait120 pose), grf (a stroke dev patient's pose),
    grf_able_bodied (an able-bodied dev person's pose), other."""
    if rec["stream"] == "camera":
        return "metric3d"
    if rec["source"] == g4.GAIT120:
        return "emg"
    if rec["person_id"] in set(stroke):
        return "grf"
    if rec["person_id"] in set(able):
        return "grf_able_bodied"
    return "other"


def left_out_report(records, stroke=(), able=()):
    """{min_s, reasons, n, counts {source: {reason: n}}, by_line {line: n}, records (LEFT_OUT_COLUMNS)} of the
    streams left out (module docstring, Records left out)."""
    rows = [{"line": line_of(r, stroke, able), **{k: r.get(k) for k in LEFT_OUT_COLUMNS if k != "line"}}
            for r in records]
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[r["source"]][r["reason"]] += 1
    return {"min_s": I.MIN_S, "reasons": {**g4.LEFT_OUT, **LEFT_OUT}, "n": len(rows),
            "counts": {s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
            "by_line": dict(sorted(collections.Counter(r["line"] for r in rows).items())), "records": rows}


def unscored(persons, table, col):
    """Sorted persons of a set without a finite value in column col of a per-person table."""
    if not len(table):
        return sorted(set(persons))
    ok = table.loc[np.isfinite(table[col].to_numpy(dtype=np.float64)), "person_id"]
    return sorted(set(persons) - set(ok))


def n_trials(table):
    """Distinct (person, key) trials of a per-trial table (gait120 keys repeat across persons)."""
    return int(table[["person_id", "key"]].drop_duplicates().shape[0]) if len(table) else 0


def mean_line(values):
    """{point, interval, n_persons}: the mean over persons and its person bootstrap interval (g4)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {"point": float(v.mean()) if len(v) else float("nan"), "interval": g4.person_bootstrap(v, np.mean),
            "n_persons": int(len(v))}


# ---------------------------------------------------------------- the development run
def run_dev(predictor, stroke, gait_dev, template, able=(), metric3d=True, log=None):
    """Both rows, the secondary GRF rows and the metric-3D round trip (module docstring), sets checked first. The
    model's GRF is predictor.grf_contact through g4.flight_predictor (the sealed read's flight code; stats counts
    grf_model_frames_plate and grf_model_flight_frames_plate); the metric-3D trials of a patient without a recorded
    mass run through predictor.metric3d_person.
    Returns a dict: gtr, person_grf, trial_grf, trial_emg (the EMG trials kept, column scored), person_emg,
    template, summary (g4.summary of the primary row on the kept trials, with emg_r_gain), emg_trials_all (EMG trials
    paired), emg_template_all_trials (the template's line over every paired EMG trial), stats, counts, low_peak (the
    secondary row without the low-peak plate series, g4.evaluate's grf_low_peak), grf_persons_no_mass, able
    (trials, person table, summary), metric3d (trial and person tables, mpjpe and pa_mpjpe lines), left_out
    (left_out_report of the predictor's and the harness's records)."""
    import pandas as pd
    log = log or (lambda *a: None)
    counts = check_sets({"stroke_dev": stroke, "gait120_dev": gait_dev, "gait120_template": template,
                         "able_bodied_dev": able})
    predictor.metric3d_persons = set(stroke) if metric3d else set()
    stats, info = collections.Counter(), {}
    t0 = time.time()
    grf_names, emg_names = ["newton", "model"], ["template", "model"]          # g4.evaluate's predictor order
    gtr, person_grf, trial_emg, person_emg, tmpl, summ = g4.evaluate(
        stroke, gait_dev, template, grf_predictors={"model": g4.flight_predictor(predictor.grf_contact, stats)},
        emg_predictors={"model": predictor.emg}, stats=stats, min_s=g4.EMG_MIN_S, same_samples=True,
        emg_gain=True, info=info, low_peak=True)
    if metric3d:                                      # the metric-3D line needs no mass: every paired trial
        for person in info["grf_persons_no_mass"]:
            predictor.metric3d_person(person)
    left_emg = {(r["person_id"], r["key"]) for r in getattr(predictor, "left_out", ())
                if r["stream"] == "pose" and r["source"] == g4.GAIT120}
    if left_emg and len(trial_emg):                   # the same trials for the template and the model
        drop = np.asarray([(p, k) in left_emg for p, k in zip(trial_emg["person_id"], trial_emg["key"])], dtype=bool)
        trial_emg = trial_emg.loc[~drop].reset_index(drop=True)
        person_emg = (g4.emg_person_table(trial_emg, emg_names) if len(trial_emg)
                      else pd.DataFrame(columns=["person_id", "pairs", "pairs_scored"]
                                        + [f"{k}_r" for k in emg_names]))
        summ = g4.summary(person_grf, person_emg, grf_names, emg_names, emg_gain=True)
    log(f"primary rows: {len(gtr)} GRF trials, {n_trials(trial_emg)} of {info['emg_trials_all']} EMG trials kept, "
        f"{time.time() - t0:.0f} s")
    trial_grf = grf_trial_table(gtr, {"newton": g4.newton_predict,
                                      "model": g4.flight_predictor(predictor.grf_contact)})
    out = {"gtr": gtr, "person_grf": person_grf, "trial_grf": trial_grf, "trial_emg": trial_emg,
           "person_emg": person_emg, "template": tmpl, "summary": summ, "stats": stats, "counts": counts,
           "emg_trials_all": info["emg_trials_all"], "emg_template_all_trials": info["emg_template_all_trials"],
           "low_peak": info["grf_low_peak"], "grf_persons_no_mass": info["grf_persons_no_mass"]}
    stats_ab, lo_ab = collections.Counter(), []
    if able:
        t0 = time.time()
        atr = g4.grf_trials(list(able), stats=stats_ab, left_out=lo_ab)
        person_ab = g4.grf_person_table(atr, {"newton": g4.newton_predict,
                                              "model": g4.flight_predictor(predictor.grf_contact, stats_ab)})
        summ_ab = g4.summary(person_ab, pd.DataFrame(), ["newton", "model"], [])
        log(f"secondary row: {len(atr)} able-bodied GRF trials, {time.time() - t0:.0f} s")
    else:
        atr, person_ab, summ_ab = [], pd.DataFrame(), None
    out["able"] = {"trials": atr, "person_grf": person_ab, "summary": summ_ab, "stats": stats_ab}
    m_trial, m_person = metric3d_tables(predictor.m3d)
    out["metric3d"] = {"trials": m_trial, "persons": m_person,
                       "mpjpe_mm": mean_line(m_person["mpjpe_mm"] if len(m_person) else []),
                       "pa_mpjpe_mm": mean_line(m_person["pa_mpjpe_mm"] if len(m_person) else []),
                       "detected_share": (float(m_trial["detected_share"].astype(float).mean()) if len(m_trial)
                                          else None)}
    out["left_out"] = left_out_report([*getattr(predictor, "left_out", ()), *info["left_out"], *lo_ab], stroke, able)
    return out


def gate_lines(summ, rule_sha256, checkpoint_sha256="", train_persons_sha256="", template_sha256=""):
    """The dev estimates of the gate lines (GateResult list): g4.gate_results of the model with the line
    emg_r_gain, on which emg_r's pass depends (G4_RULE.md sections 1 and 6, delta g4.EMG_GAIN)."""
    return g4.gate_results(summ, "model", rule_sha256, checkpoint_sha256, train_persons_sha256, template_sha256,
                           emg_gain=True)


# ---------------------------------------------------------------- report
def fmt(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.{nd}f}" if math.isfinite(v) else "n/a"


def fmt_iv(iv, nd=4):
    return "n/a" if not iv or iv[0] is None or not math.isfinite(iv[0]) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def md_table(df, cols=None):
    cols = list(df.columns) if cols is None else cols
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        out.append("| " + " | ".join("" if (v is None or (isinstance(v, float) and not math.isfinite(v))) else
                                     (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def clean(o):
    """JSON-ready copy: non-finite floats to None, numpy scalars to Python, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, common.GateResult):
        return clean(o.to_dict())
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def default_out(loaded, tree=TREE, stroke_set="all"):
    """reports/gates/g4/dev_<checkpoint name>, with the suffix _strokefold for the stroke fold and _model for the raw
    weights."""
    suffix = (STROKE_FOLD_SUFFIX if stroke_set == "fold" else "") + ("_model" if loaded.weights == "model" else "")
    return os.path.join(tree, "reports", "gates", "g4", f"dev_{loaded.name}{suffix}")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=I.WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=4.0, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--out", default=None)
    p.add_argument("--refine-steps", type=int, default=I.STEPS_EVAL)
    p.add_argument("--points", type=int, default=I.POINTS_EVAL, help="kernel points per stream-window and step")
    p.add_argument("--batch-windows", type=int, default=I.BATCH_WINDOWS)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a recording")
    p.add_argument("--no-secondary", action="store_true", help="no able-bodied GRF row")
    p.add_argument("--no-metric3d", action="store_true", help="no metric-3D round trip")
    p.add_argument("--quick", action="store_true", help="subsets (QUICK), written to the scratch folder of --out")
    p.add_argument("--stroke-set", default="all", choices=STROKE_SETS,
                   help="the scored stroke patients: all 40 non-lockbox (default) or the stage-2 dev fold of 8")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import pandas as pd
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    reads_before = sha256_or_none(common.READ_LOG)
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    out = a.out or default_out(loaded, stroke_set=a.stroke_set)
    if a.quick and (a.out is None or os.path.abspath(out) == os.path.abspath(default_out(loaded,
                                                                                         stroke_set=a.stroke_set))):
        raise SystemExit("--quick writes to a scratch folder: pass --out")
    paths.assert_not_moveatlas(out)
    t_load = time.time() - t_start
    print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step "
          f"{loaded.step}; load {t_load:.1f} s", flush=True)

    stroke, stroke_info = stroke_set(a.stroke_set)
    gait_dev = g4.gait120_persons("dev")
    template, able = g4.gait120_persons("train"), ([] if a.no_secondary else able_bodied_dev())
    if a.quick:
        stroke, gait_dev = stroke[:QUICK["stroke"]], gait_dev[:QUICK["gait120_dev"]]
        template, able = template[:QUICK["gait120_train"]], able[:QUICK["able_bodied"]]
    lists = person_lists(a.checkpoint)
    stored = {f"checkpoint:{s}": frozenset(v) for s, v in loaded.persons.items()}
    scored = {"stroke_dev": stroke, "gait120_dev": gait_dev, "able_bodied_dev": able}
    try:
        disjoint = assert_disjoint(scored, {**lists, **stored})
    except Refused as e:
        if a.stroke_set == "all":
            raise Refused(f"{e}; a stage-2 checkpoint is read with --stroke-set fold") from e
        raise

    pred = G4Predictor(loaded.model, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                       seed=a.seed, pool=not a.no_pool, echo=lambda s: print(s, flush=True))
    res = run_dev(pred, stroke, gait_dev, template, able, metric3d=not a.no_metric3d,
                  log=lambda s: print(s, flush=True))
    summ = res["summary"]

    with open(g4.RULE_PATH, "rb") as f:
        rule_sha = common.sha256_bytes(f.read())
    with open(g4.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())
    template_hash = sources.person_hash(template)
    results = gate_lines(summ, rule_sha, loaded.sha256, I.train_persons_sha256(loaded), template_hash)
    reads = [r for r in common.read_log() if r["gate"] == g4.GATE] if os.path.exists(common.READ_LOG) else []
    reads_after = sha256_or_none(common.READ_LOG)

    gtr, trial_grf, trial_emg = res["gtr"], res["trial_grf"], res["trial_emg"]
    person_grf, person_emg = res["person_grf"], res["person_emg"]
    flights = []
    for tr in gtr:
        f = pred.flight(tr)
        if f is not None:                                   # None: the trial was left out
            flights.append(f[tr.obs_valid.any(1)])
    flight_share = float(np.concatenate(flights).mean()) if flights and sum(len(x) for x in flights) else None
    by_muscle = (trial_emg.groupby("muscle")[["template_r", "model_r"]].median().reindex(list(g4.MUSCLES))
                 if len(trial_emg) else None)
    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    head = PV.moveatlas_head()
    wall = time.time() - t_start
    m3 = res["metric3d"]
    able_res = res["able"]
    trials = pd.DataFrame(pred.log)
    summ_ab = able_res["summary"]
    lo = res["left_out"]
    low = res["low_peak"]
    un = {"stroke": unscored(stroke, person_grf, "newton_rmse_pct_bw"),
          "gait120": unscored(gait_dev, person_emg, "template_r"),
          "able": unscored(able, able_res["person_grf"], "newton_rmse_pct_bw")}
    doc = {
        "run": RUN, "gate": g4.GATE, "what": WHAT, "quick": bool(a.quick),
        "checkpoint": loaded.describe(),
        "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": pred.win_s,
                      "hop_s": pred.hop_s, "batch_windows": a.batch_windows, "seed": a.seed,
                      "shape_pooling": "none" if a.no_pool else "per recording", "weights": a.weights,
                      "device": str(dev), "amp_bf16": dev.type == "cuda", "memory": mem,
                      "pose_input": f"pose3d, canonical coords with scale and the raw root track, layout {POSE_LAYOUT}",
                      "flight_rule": "both contact probabilities below 0.5: 0 BW on both feet (g4.flight_predictor, "
                                     "the sealed read's code, on the model's BW and contact probabilities)"},
        "rule": {"path": common.rel(g4.RULE_PATH), "sha256": rule_sha, "status": status[0],
                 "registered": common.registered(g4.GATE, rule_sha)},
        "lockbox": {"g4_reads_logged": len(reads), "read_lockbox_called": False,
                    "lockbox_reads_csv_sha256_before": reads_before, "lockbox_reads_csv_sha256_after": reads_after,
                    "unchanged": reads_before == reads_after},
        "sets": {"stroke_set": stroke_info,
                 "stroke_dev": len(stroke), "stroke_dev_sha256": sources.person_hash(stroke),
                 "stroke_scored": len(set(stroke)) - len(un["stroke"]), "stroke_trials": len(gtr),
                 "stroke_trials_left_out": lo["by_line"].get("grf", 0),
                 "stroke_samples": int(person_grf["samples"].sum()) if len(person_grf) else 0,
                 "stroke_unscored": un["stroke"], "stroke_no_body_mass": res["grf_persons_no_mass"],
                 "gait120_dev": len(gait_dev), "gait120_dev_sha256": sources.person_hash(gait_dev),
                 "gait120_dev_scored": len(set(gait_dev)) - len(un["gait120"]),
                 "gait120_dev_trials": n_trials(trial_emg), "gait120_dev_trials_all": res["emg_trials_all"],
                 "gait120_dev_unscored": un["gait120"],
                 "gait120_template": len(template), "gait120_template_sha256": template_hash,
                 "able_bodied_dev": len(able), "able_bodied_dev_sha256": sources.person_hash(able),
                 "able_bodied_scored": len(set(able)) - len(un["able"]),
                 "able_bodied_trials": len(able_res["trials"]),
                 "able_bodied_trials_left_out": lo["by_line"].get("grf_able_bodied", 0),
                 "able_bodied_unscored": un["able"],
                 "disjoint_from_training_lists": disjoint,
                 "training_lists": {k: len(v) for k, v in lists.items()}},
        "pairing": {"primary": dict(res["stats"]), "able_bodied": dict(able_res["stats"]),
                    "predictor_reads": dict(pred.read_stats)},
        "summary": {"primary": summ, "no_low_peak": low["summary"], "able_bodied": summ_ab},
        "lines": results,
        "low_peak": {"rule": "the primary GRF row without the plate series (trial, foot) whose plate peak at the "
                             "scored samples stays below min_peak_bw, every predictor and the Newton floor recomputed "
                             "on its own samples (G4_RULE.md section 2 (b)); it decides nothing",
                     "min_peak_bw": low["min_peak_bw"], "series_left_out": low["series_left_out"],
                     "samples_left_out": low["samples_left_out"], "patients": low["patients"],
                     "series_scored": int((trial_grf["samples"] > 0).sum()) if len(trial_grf) else 0},
        "left_out": {"rule": "an input stream without a window (under infer.MIN_S s, the window store's minimum), "
                             "an EMG trial under g4.EMG_MIN_S s of pose (left out by the harness before the model "
                             "is called), a GRF trial of a patient without a recorded body mass, or a metric-3D "
                             "trial without a valid pelvis; a GRF trial left out has no scored sample for any "
                             "predictor, an EMG trial left out is removed for the template and the model",
                     **lo},
        "metric3d": {"status": METRIC3D_STATUS, "camera": {"family": CAMERA_FAMILY, "fps": CAMERA_FPS,
                                                            "tracker": "default", "seed_tag": CAMERA_TAG,
                                                            "aim": "trial-mean pelvis"},
                     "mpjpe_mm": m3["mpjpe_mm"], "pa_mpjpe_mm": m3["pa_mpjpe_mm"],
                     "trials": int(len(m3["trials"])), "detected_share_mean_over_trials": m3["detected_share"],
                     "mpjpe_alignment": "pelvis-relative, one yaw about +y per window, g1.yaw_fit"},
        "diagnostics": {"model_flight_share_plate_frames": flight_share,
                        "emg_trials_kept": n_trials(trial_emg), "emg_trials_all": res["emg_trials_all"],
                        "emg_template_all_trials": res["emg_template_all_trials"],
                        "emg_median_r_by_muscle": ({m: {"template": r4(by_muscle.loc[m, "template_r"]),
                                                        "model": r4(by_muscle.loc[m, "model_r"])}
                                                    for m in g4.MUSCLES} if by_muscle is not None else None),
                        "emg_modelled": {m: bool(x) for m, x in zip(g4.MUSCLES, pred.emg_modelled)}},
        "config": {"grf_pass": g4.GRF_PASS, "grf_kill": g4.GRF_KILL, "r_gain": g4.R_GAIN, "emg_pass": g4.EMG_PASS,
                   "emg_kill": g4.EMG_KILL, "emg_gain": g4.EMG_GAIN, "emg_min_s": g4.EMG_MIN_S,
                   "low_peak_bw": g4.LOW_PEAK_BW, "boot_n": g4.BOOT_N, "seed": g4.SEED, "muscles": list(g4.MUSCLES)},
        "timing_s": {"load": round(t_load, 1), "all": round(wall, 1)},
        "moveatlas_head": head,
    }
    raw, doc = doc, clean(doc)
    text = report(raw, doc, person_grf, person_emg, able_res["person_grf"], m3["persons"], trials, a.quick,
                  person_low=low["person_grf"])
    common.write(doc, out, "g4", text)
    common.write_csv(person_grf, os.path.join(out, "per_person_grf.csv"))
    common.write_csv(trial_grf, os.path.join(out, "per_trial_grf.csv"))
    common.write_csv(person_emg, os.path.join(out, "per_person_emg.csv"))
    common.write_csv(trial_emg, os.path.join(out, "per_trial_emg.csv"))
    common.write_csv(able_res["person_grf"], os.path.join(out, "per_person_grf_able_bodied.csv"))
    common.write_csv(low["person_grf"], os.path.join(out, "per_person_grf_no_low_peak.csv"))
    common.write_csv(low["series"], os.path.join(out, "low_peak_series.csv"))
    common.write_csv(m3["persons"], os.path.join(out, "per_person_metric3d.csv"))
    common.write_csv(m3["trials"], os.path.join(out, "per_trial_metric3d.csv"))
    common.write_csv(trials, os.path.join(out, "trials.csv"))
    common.write_csv(pd.DataFrame(lo["records"], columns=list(LEFT_OUT_COLUMNS)), os.path.join(out, "left_out.csv"))
    code = [g4.__file__, I.__file__, M.__file__, os.path.abspath(__file__)]
    prov = {"script": "scripts/eval_g4_dev.py", "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "checkpoint": doc["checkpoint"]["path"], "checkpoint_sha256": loaded.sha256, "quick": bool(a.quick),
            "code_sha256": common.code_sha256(code), "code_files": [common.rel(c) for c in code],
            "inputs": ["MoveAtlas canonical pose, force and EMG records of stroke_gait_vancriekinge (dev patients, "
                       "able-bodied dev persons) and gait120 (dev and train folds), flags False",
                       "the raw root tracks of mdfm.data.mocap_raw (C3D and TRC of those persons)",
                       "configs/canonical_skeleton.yaml"]}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(text)
    print(f"left out: {lo['n']} streams, {lo['counts']}")
    print(f"written {common.rel(out)}")
    if reads_before != reads_after:
        print("WARNING: reports/gates/lockbox_reads.csv changed during the run (not by this script)", flush=True)


def report(raw, doc, person_grf, person_emg, person_ab, person_m3, trials, quick, person_low=None):
    """The markdown of g4.md from the result dict (raw: non-finite floats kept; doc: cleaned); person_low, the
    per-patient table of the row without the low-peak plate series."""
    import pandas as pd
    ck = doc["checkpoint"]
    summ = raw["summary"]["primary"]
    lb = doc["lockbox"]
    L = [f"# {WHAT}", "",
         "Written by scripts/eval_g4_dev.py (harness mdfm/gates/g4.py, inference mdfm/eval/infer.py); every number "
         "below is read from g4.json and the CSV files of this folder. Development data only: no lockbox person, row "
         "or record was read, and mdfm.gates.g4.read_lockbox was not called."
         + (" QUICK RUN (subsets)." if quick else ""),
         "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,}. Training persons: " + ", ".join(f"{s} {n}" for s, n in ck["persons"].items())
         + f" (SHA-256 {ck['persons_sha256'].get('all')}). Hashes that moved since the save: "
         f"{', '.join(ck['drift']) or 'none'}.", "",
         f"Rule {doc['rule']['path']}: status '{doc['rule']['status']}', SHA-256 {doc['rule']['sha256']} (registered: "
         f"{doc['rule']['registered']}). G4 lockbox reads logged: {lb['g4_reads_logged']}. "
         f"reports/gates/lockbox_reads.csv SHA-256 before the run {lb['lockbox_reads_csv_sha256_before']}, after "
         f"{lb['lockbox_reads_csv_sha256_after']} (unchanged: {lb['unchanged']}). MoveAtlas HEAD "
         f"{doc['moveatlas_head']}.",
         "",
         "## Gate lines (dev estimates, not gate reads)", "",
         "| line | unit | model | 95 % interval | pass | kill | trivial baseline | persons | verdict (dev estimate) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in raw["lines"]:
        L.append(f"| {r.line} | {r.unit} | {fmt(r.point)} | {fmt_iv(r.interval)} | {fmt(r.pass_line)} | "
                 f"{fmt(r.kill_line)} | {fmt(r.trivial_baseline)} | {r.n_persons} | {r.verdict} |")
    L += ["", "Trivial baselines: the Newton floor (GRF; the r gain is measured against it) and the EMG template "
          "(built on the gait120 train fold; the EMG r gain is measured against it, and emg_r passes only when "
          f"emg_r_gain reaches delta = {fmt(doc['config']['emg_gain'], 2)}).", "", "## Primary rows", ""]
    for key, label, stat in (("grf_rmse_pct_bw", "GRF RMSE, percent body weight", "mean"),
                             ("grf_r", "GRF Pearson r", "mean"),
                             ("grf_r_gain", "GRF r gain over the Newton floor", "mean"),
                             ("emg_r", "EMG Pearson r of log-envelopes", "median"),
                             ("emg_r_gain", "EMG r gain over the template (median of per-person differences)",
                              "median")):
        names = {"emg_r": ("template", "model"), "grf_r_gain": ("model",),
                 "emg_r_gain": ("model",)}.get(key, ("newton", "model"))
        L.append(f"- {label}: " + "; ".join(f"{n} {_line(summ.get(key, {}).get(n), stat)}" for n in names) + ".")
    d = doc["diagnostics"]
    L.append(f"- EMG rows on {d['emg_trials_kept']} of the {d['emg_trials_all']} paired dev trials, the same frames "
             f"of the same (trial, muscle) pairs for the template and the model (the trials under "
             f"{fmt(doc['config']['emg_min_s'], 1)} s left out, section Records left out); the template over "
             f"all {d['emg_trials_all']} trials: {_line(raw['diagnostics']['emg_template_all_trials'], 'median')}.")
    if d["emg_median_r_by_muscle"]:
        L.append("- Median r per muscle over the dev trials (template, model): " + ", ".join(
            f"{m} {fmt(v['template'])}, {fmt(v['model'])}" for m, v in d["emg_median_r_by_muscle"].items()) + ".")
    L.append(f"- Share of plate-valid frames the model reads in flight (0 BW on both feet): "
             f"{fmt(d['model_flight_share_plate_frames'])}. Muscles the EMG decoder models: "
             + ", ".join(m for m, x in d["emg_modelled"].items() if x) + ".")
    lp, nl = doc["low_peak"], raw["summary"]["no_low_peak"]
    L += ["", f"## Secondary row: the primary GRF row without the low-peak plate series (below {lp['min_peak_bw']:g} "
          "BW)", "",
          f"Rule section 2 (b), deciding nothing: {lp['series_left_out']} of the {lp['series_scored']} scored plate "
          f"series (trial, foot) left out, {lp['samples_left_out']} scored samples, from "
          f"{len(lp['patients'])} patients ({', '.join(lp['patients']) or 'none'}; low_peak_series.csv); the Newton "
          "floor and the model recomputed on the samples kept (per_person_grf_no_low_peak.csv).", ""]
    for key, label in (("grf_rmse_pct_bw", "GRF RMSE, percent body weight"), ("grf_r", "GRF Pearson r"),
                       ("grf_r_gain", "GRF r gain over the Newton floor")):
        names = ("model",) if key == "grf_r_gain" else ("newton", "model")
        L.append(f"- {label}: " + "; ".join(f"{n} {_line(nl.get(key, {}).get(n), 'mean')}" for n in names) + ".")
    ab = raw["summary"]["able_bodied"]
    L += ["", "## Secondary row: able-bodied stroke_gait_vancriekinge dev persons (GRF)", ""]
    if ab:
        for key, label in (("grf_rmse_pct_bw", "GRF RMSE, percent body weight"), ("grf_r", "GRF Pearson r"),
                           ("grf_r_gain", "GRF r gain over the Newton floor")):
            names = ("model",) if key == "grf_r_gain" else ("newton", "model")
            L.append(f"- {label}: " + "; ".join(f"{n} {_line(ab.get(key, {}).get(n), 'mean')}" for n in names) + ".")
    else:
        L.append("Not run (--no-secondary).")
    m3 = raw["metric3d"]
    L += ["", "## Metric-3D round trip", "",
          f"Status: {m3['status']}. One static camera per trial (family {m3['camera']['family']}, "
          f"{m3['camera']['fps']:g} fps, default tracker process, aimed at the trial-mean pelvis, seed "
          f"SHA-256(person, recording key, '{m3['camera']['seed_tag']}')), every paired trial of the stroke dev "
          "patients (a patient without a recorded body mass included: the line needs none); "
          f"{m3['trials']} trials, mean detected share of 2D joints {fmt(m3['detected_share_mean_over_trials'])}.", "",
          f"- MPJPE ({m3['mpjpe_alignment']}): {_line(m3['mpjpe_mm'], 'mean', nd=1)} mm.",
          f"- PA-MPJPE (per-frame similarity Procrustes): {_line(m3['pa_mpjpe_mm'], 'mean', nd=1)} mm.", ""]
    s = doc["sets"]
    L += ["## Sets", "",
          f"- Stroke dev patients (stroke set {s['stroke_set']['set']}"
          + (f", the stage-2 dev fold of {s['stroke_set']['path']}" if s["stroke_set"]["set"] == "fold" else "")
          + f"): {s['stroke_dev']} (SHA-256 {s['stroke_dev_sha256']}); scored {s['stroke_scored']}, "
          f"{s['stroke_trials']} trials ({s['stroke_trials_left_out']} left out), {s['stroke_samples']} scored "
          f"(frame, foot) samples; unscored: {', '.join(s['stroke_unscored']) or 'none'}; without a recorded body "
          f"mass (dropped from the GRF rows, kept in the metric-3D round trip): "
          f"{', '.join(s['stroke_no_body_mass']) or 'none'}.",
          f"- gait120 dev fold: {s['gait120_dev']} persons (SHA-256 {s['gait120_dev_sha256']}); scored "
          f"{s['gait120_dev_scored']}, {s['gait120_dev_trials']} of {s['gait120_dev_trials_all']} paired level-walking "
          f"trials scored; unscored: {', '.join(s['gait120_dev_unscored']) or 'none'}; template from the train fold, "
          f"{s['gait120_template']} persons (SHA-256 {s['gait120_template_sha256']}).",
          f"- Able-bodied stroke dev persons: {s['able_bodied_dev']} (SHA-256 {s['able_bodied_dev_sha256']}); scored "
          f"{s['able_bodied_scored']}, {s['able_bodied_trials']} trials ({s['able_bodied_trials_left_out']} left "
          f"out); unscored: {', '.join(s['able_bodied_unscored']) or 'none'}.",
          "- Disjoint from the checkpoint's training lists (persons/*.txt beside it and the lists stored in it), "
          "asserted: " + ", ".join(f"{k} {v} overlaps" for k, v in s["disjoint_from_training_lists"].items()) + ".",
          f"- Pairing counts: {doc['pairing']}.", ""]
    lo = doc["left_out"]
    L += ["## Records left out", "",
          f"An input stream without a window (under {lo['min_s']:g} s, infer.MIN_S, the window store's minimum), an "
          f"EMG trial under {fmt(doc['config']['emg_min_s'], 1)} s of pose (g4.EMG_MIN_S, left out by the harness "
          "before the model is called), a GRF trial of a patient without a recorded body mass, or a metric-3D trial "
          f"without a valid pelvis is left out and counted here: {lo['n']} streams. A GRF trial left out has no "
          "scored sample for any predictor (the Newton floor included); an EMG trial left out is removed for the "
          "template and the model alike."]
    if lo["n"]:
        L += ["", "| source | reason | streams |", "|---|---|---|"]
        L += [f"| {src} | {reason} | {n} |" for src, c in lo["counts"].items() for reason, n in c.items()]
        L += ["", "Per line: " + ", ".join(f"{k} {v}" for k, v in lo["by_line"].items()) + ".", "",
              md_table(pd.DataFrame(lo["records"], columns=list(LEFT_OUT_COLUMNS)))]
    L.append("")
    inf = doc["inference"]
    mem = inf["memory"]
    tr = trials
    L += ["## Inference", "",
          f"{inf['pose_input']}; {inf['window_s']:g} s windows every {inf['hop_s']:g} s, "
          f"{inf['refine_steps']} refinement steps of {inf['points_per_stream']} points per stream, shape pooled "
          f"{inf['shape_pooling']}, device {inf['device']}"
          + (f", peak allocated {mem['max_allocated_gb']:.2f} GB, reserved {mem['max_reserved_gb']:.2f} GB under a "
             f"{mem['cap_gb']} GB cap" if mem else "") + f". Flight rule: {inf['flight_rule']}.",
          (f"Inference calls: {len(tr)} ({', '.join(f'{k} {v}' for k, v in tr['kind'].value_counts().items())}), "
           f"{int(tr['windows'].sum())} windows, {float(tr['seconds'].sum()):.0f} s in the model. Time: load "
           f"{doc['timing_s']['load']} s, all {doc['timing_s']['all']} s." if len(tr) else "No inference call."), ""]
    L += ["## GRF per patient (primary)", "", md_table(person_grf) if len(person_grf) else "none", "",
          "## GRF per patient without the low-peak plate series", "",
          md_table(person_low) if person_low is not None and len(person_low) else "none", "",
          "## EMG per dev person", "", md_table(person_emg) if len(person_emg) else "none", "",
          "## GRF per able-bodied dev person", "", md_table(person_ab) if len(person_ab) else "none", "",
          "## Metric-3D per patient", "", md_table(person_m3) if len(person_m3) else "none", ""]
    return "\n".join(L) + "\n"


def _line(s, stat, nd=4):
    if not s:
        return "n/a"
    return f"{fmt(s['point'], nd)} {fmt_iv(s['interval'], nd)} ({stat} over {s['n_persons']} persons)"


if __name__ == "__main__":
    main()
