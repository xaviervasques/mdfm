"""T04 (weekend 2026-10-01): G3 dev diagnostics of a checkpoint on the 14 dev controls, the G3 learning-curve table of
a run, and the speed and cadence quantiles of the AMASS train walks against the weargait training-fold walkway. Dev
diagnostics only: no rule line changes (docs/rules/G3_RULE.md stays a DRAFT), no lockbox row or record is read, no
number here is a gate read.

Reused by import: mdfm.gates.g3 (dev_run for the targets, the population baseline and the shank-gyroscope detector;
fit_offsets, evaluate, ic_margin and the matching of strides, passes and ICs), mdfm.eval.infer (load_checkpoint and
G3Predictor, the inference of scripts/eval_g3_dev.py: 10 s windows every 5 s, refinement, shape pooled per trial,
bodies stitched on the trial clock), the helpers of scripts/eval_g3_dev.py (fit_offsets with its unfitted list,
fin, fmt, fmt_iv, clean) and the model_legs readout of scripts/eval_gait_insim.py (speed_readouts).

Default run (--checkpoint): reports/gates/g3/diag_<checkpoint name>/ (--ablations: diag_<name>_ablations/, so that
an ablations run never replaces the default diag.json; --weights model: diag_model_<name>[_ablations]/, a name
scripts/weekend_branch.py never merges with the EMA rows). Every predictor is scored by the G3 harness on the 14 dev
controls (g3.controls('dev')), tasks SelfPace and HurriedPace, against the walkway; each IC predictor has its own
offset fitted on the 14 (g3.fit_offsets). Signed errors are prediction minus reference everywhere.
- base: the model as eval_g3_dev.py runs it (contacts give the ICs, strides between ICs read on the stitched body,
  pass speeds over the walkway pass intervals).
- model_legs (eval_gait_insim.py, fix 7 of the root-motion diagnosis): the base ICs, strides and pass speeds from
  the stance-foot no-slip speed of the decoded pose (gait.stance_speed of each window's pelvis-relative joints and
  contacts, stitched: legs_speed_body reads both from the stitched trial body, which gives the values of
  eval_gait_insim.legs_speed); the root translation is not read.
- model_heel_strike (IC only): mdfm.readouts.gait.heel_strikes on the decoded joints. The stitched body holds each
  window in its own frame (a jump at every window change would bend the 1 s direction of progression), so the
  heel strikes are found in each window's own frame and each is kept from the window that owns its frame in the
  stitch (gait.stitch, the window where the frame is most central), or from a neighbour window when it lies within
  HS_BOUNDARY_S (0.25 s) of that window's frames (two windows can place one strike a few frames apart on either
  side of their boundary); per foot one strike per gait.HS_DISTANCE_S, the owned and more central candidate first
  (heel_strike_ics). Its own offset (dev fit), its IC median against the walkway and the insole references, and
  its margin against the shank-gyroscope detector on the reference ICs both match (g3.ic_margin; margin_ratio =
  model value over detector value, the G3 margin line 0.8).
- events.csv: one line per reference event of every scored row: kind stride (cm), speed (m/s, per walkway pass) or
  ic_walkway (ms on the trial clock, the prediction shifted by its offset); trial, person, task, foot, pass, the
  reference, the prediction, the signed error, matched.
- Signed stride and speed biases per person and per task (SelfPace, HurriedPace, all), and their means over persons.
- Spearman of the true speed against the signed speed error over the matched walkway passes of the persons of a
  subset (the unit is the pass: both tasks of every person), with a person-clustered bootstrap interval; the
  reading is on the 12 controls without WHC003 (no shank sensors) and WHC026 (shank sides exchanged).
- Rerun check: when a G3 dev report of the same checkpoint with EMA weights exists (reports/gates/g3_sim/dev_<name>/
  g3_sim.json, else reports/gates/g3/dev_<name>/g3_dev.json), the base row on all14 must match it within 0.5 cm of
  stride MAE and 0.01 m/s of speed MAE, and each person within 1 cm of stride MAE; otherwise the report states that
  a code path diverged. GPU inference is not bit-reproducible, so equality is not required. The inference settings
  of both runs are printed (batch_windows changes the refinement point draws).

--ablations adds the model rows (each a full pass of the model over the 28 dev trials, its own IC offset; a trial
whose input the ablation leaves unchanged reuses the base readouts of that trial, the same input and seed):
- lowpass12: acc and gyro low-passed at 12 Hz (Butterworth order 4, zero phase: sosfiltfilt) on each contiguous run
  of valid samples of each sensor before the model's band copy and tokenization (runs no longer than the band
  copy's padlen left as they are);
- sites6: every record cut to the sites of WHC003 (head, trunk, lower_back, wrist_L, wrist_R, foot_L);
- swap_whc026: the shank_L and shank_R sensors of WHC026 exchanged back (data, channel masks and validity);
- refine0 and refine60: 0 and 60 refinement steps instead of --refine-steps;
- family_lower_back, family_wrists, family_legs: every record cut to the sites of the family (lower_back; wrist_L
  and wrist_R; thigh, shank, ankle and foot of both sides). A trial without any site of the family is not run and
  its person drops out of that row. Each family row carries its transfer ratio, the family's stride MAE over the
  base row's on the same subset, set against the design's G10 lines (at most 1.3, kill above 1.6), labelled 'dev
  preview, not a G10 read': the design's G10 line splits weargait_pd by subject-id family (section 8 (c)); this
  sensor-family ratio is the preview the task asks for.
A cut record keeps its source's layout and device descriptors (weargait_pd: xsens13, the mount prior of those
devices), as WHC003's own six-site records do: fewer sites of the same devices, not another layout.

diag.json (the schema scripts/weekend_branch.py reads): rows {base, lowpass12, sites6, swap_whc026, refine0,
refine60, model_legs, family_lower_back, family_wrists, family_legs}; a row this run did not compute is null and
named in rows_not_run (null rows are never merged over another file's rows). Each row holds the subsets all14,
no_whc003 and no_whc003_whc026, each {stride_mae_cm, speed_mae_ms (m/s), stride_mape_pct, n_persons (persons with a
matched stride), per_person {id: {stride_cm, speed_ms, stride_mape_pct, selfpace_stride_cm}}, with intervals}; and
per row selfpace_stride_mae_cm {subset: {point, interval, n_persons, per_person {id: cm}}}, hurried_speed_bias_ms
{subset: {point, interval, n_persons, per_person}} (m/s), spearman_speed_error {subset: {rho, interval, n_passes,
n_persons}}, signed_bias, ic_walkway; the base row also heel_strike_ic {median_ms, offset_ms, margin_ratio, ...}.
The four values of the base row are repeated at the top level.

F2 lever 5a (task L5A; mdfm/eval/zupt_refine.py), off by default: --zupt-anchor runs the default diag with the inertial
anchor of lever 5 in every chunk's refinement objective (targets: the version-2 ZUPT strides of each record's own foot
sensors, no reference value; constants fixed in the module: weight 1.0, sd 0.1 m/s, edge 0.1 s, strides of at least
0.3 m, 20 refinement steps, so --refine-steps other than 20 and --ablations are refused); --zupt-report computes the
same targets and the foot report without touching the objective (the off arm's report under the same code). Either
needs --out, states the option in diag.json (inference.zupt_anchor, its run record, and the block zupt_anchor: trials,
targets, the term at its first and last refinement call, the foot report's medians), adds the option's columns to
trials.csv and writes zupt_strides.csv (one row per window and target: the foot report of zupt_refine.foot_report).
With --zupt-anchor the rerun check reads 'not applicable' (the reference reports run without the option) and the
label adds 'inertially anchored body (lever 5a, at inference)'.

--curve <run>: tabulates every G3 dev json (g3_dev.json or g3_sim.json) of the checkpoints of cache/runs/<run> found
in reports/gates/g3/curve_<run>_*/, reports/gates/g3/dev_<run>_*/ and reports/gates/g3_sim/dev_<run>_*/ (phase A of
phaseA_root_full: reports/gates/g3_sim/dev_phaseA_root_full_ckpt_A_010681/) into reports/gates/g3/curve_<run>/
curve.md and curve.json: per checkpoint step the model's stride MAE, speed MAE, stride MAPE, IC median (walkway), IC
margin (model over detector on the ICs both match) and IC offset, EMA weights; raw-weight reports listed apart. A
folder belongs to the run when its json's checkpoint name is <run>_ckpt_<A or B>_<step>. A step held by several
folders is read from the first of dev (g3), dev (g3_sim), curve (g3), curve (g3_sim); the others are listed.

--quantiles (CPU, no model): speed and cadence quantiles of the AMASS train walks (reports/amass_walks/takes.csv,
every subset's archive checked to be a train archive by mdfm.data.amass_index.check) against the walkway of the
weargait training-fold controls (g3.controls('train'), trials.csv of reports/weargait_targets read through g3.read_rows
with its lockbox guard; ww_speed_m_s and cadence_contacts per trial), SelfPace and HurriedPace: population quantiles
only, no person-level value is written. Output reports/gates/g3/speed_cadence_quantiles/ (quantiles.json,
quantiles.md, _provenance.json).

Every _provenance.json carries the stamp of scripts/moveatlas_stamp.py (read-only git rev-parse, diff, status).

Run (mdfm environment):
  <conda>/envs/mdfm/python.exe scripts/eval_g3_diag.py --checkpoint <run>/ckpt_<P>_<step>.pt [--ablations]
  <conda>/envs/mdfm/python.exe scripts/eval_g3_diag.py --curve <run>
  <conda>/envs/mdfm/python.exe scripts/eval_g3_diag.py --quantiles
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import numpy as np  # noqa: E402

ROWS = ("base", "lowpass12", "sites6", "swap_whc026", "refine0", "refine60", "model_legs", "family_lower_back",
        "family_wrists", "family_legs")
DEFAULT_ROWS = ("base", "model_legs")
SUBSETS = {"all14": (), "no_whc003": ("WHC003",), "no_whc003_whc026": ("WHC003", "WHC026")}
SUB12 = "no_whc003_whc026"
TASKS = ("SelfPace", "HurriedPace")
# the canonical IMU sites of configs/canonical_imu.yaml (mdfm.factory.imu_render.SITES), the 13 Xsens sites
XSENS_SITES = ("head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R", "shank_L", "shank_R",
               "ankle_L", "ankle_R", "foot_L", "foot_R")
SITE_ID = {s: i for i, s in enumerate(XSENS_SITES)}
WHC003_SITES = ("head", "trunk", "lower_back", "wrist_L", "wrist_R", "foot_L")
FAMILIES = {"family_lower_back": ("lower_back",), "family_wrists": ("wrist_L", "wrist_R"),
            "family_legs": ("thigh_L", "thigh_R", "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R")}
G10_PASS, G10_KILL = 1.3, 1.6
PREVIEW = "dev preview, not a G10 read"
LOWPASS_HZ, LOWPASS_ORDER = 12.0, 4
REFINE_ROWS = {"refine0": 0, "refine60": 60}
SWAP_PERSON = "WHC026"
RERUN_TOL = {"stride_mae_cm": 0.5, "speed_mae_ms": 0.01, "person_stride_cm": 1.0}
WHAT = "G3 dev diagnostics (T04), not a gate read"
ZUPT_LABEL = "inertially anchored body (lever 5a, at inference)"
ZUPT_CODE_FILES = ("mdfm/eval/zupt_refine.py", "mdfm/readouts/zupt.py", "mdfm/data/zupt_anchor.py",
                   "mdfm/train/losses.py")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
AMASS_WALKS_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
CKPT_NAME = re.compile(r"^(?P<run>.+)_ckpt_(?P<phase>[AB])_(?P<step>\d+)$")
EMPTY = "empty"                     # a transform's answer when the record holds none of the kept sites
HS_BOUNDARY_S = 0.25                # s: a window's heel strike this close to the frames it owns is a candidate


# ---------------------------------------------------------------- imports of the sibling scripts
def _sibling(name):
    """The module of scripts/<name>.py (loaded once, under a private name)."""
    key = f"_t04_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def EG():
    return _sibling("eval_g3_dev")


def INSIM():
    return _sibling("eval_gait_insim")


def stamp_moveatlas():
    """The read-only stamp of scripts/moveatlas_stamp.py."""
    return _sibling("moveatlas_stamp").stamp()


# ---------------------------------------------------------------- persons and subsets
def short_id(pid):
    """weargait:WHC003 -> WHC003; 'weargait:NLS901 (control)' -> NLS901 (scripts/weekend_branch.py short_id)."""
    s = str(pid).split(":")[-1].strip()
    return s.split(" ")[0] if s else s


def in_subset(pid, subset):
    return short_id(pid) not in SUBSETS[subset]


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mean_or_none(values):
    v = [x for x in (finite(y) for y in values) if x is not None]
    return float(np.mean(v)) if v else None


def boot_mean(values_by_person, n=None, seed=None):
    """Person bootstrap interval of the mean of one value per person (g3's seed and resamples), None under 2."""
    from mdfm.gates import common, g3
    ids = [p for p, v in sorted(values_by_person.items()) if finite(v) is not None]
    if len(ids) < 2:
        return None
    vals = np.array([float(values_by_person[p]) for p in ids])
    return list(common.cluster_bootstrap(vals, np.array(ids), n=n or g3.BOOT_N, seed=g3.SEED if seed is None else seed))


# ---------------------------------------------------------------- record transforms (ablations)
def site_ids(names):
    return frozenset(SITE_ID[n] for n in names)


def _loc(rec):
    return [int(x) for x in np.asarray(rec["location_ids"]).ravel()]


def cut_sites(rec, keep):
    """The record cut to the sensors whose location id is in `keep`: None when it holds no other sensor (unchanged),
    EMPTY when it holds none of them."""
    loc = _loc(rec)
    idx = [j for j, s in enumerate(loc) if s in keep]
    if len(idx) == len(loc):
        return None
    if not idx:
        return EMPTY
    out = dict(rec)
    out["signal"] = np.ascontiguousarray(np.asarray(rec["signal"])[:, idx])
    out["channel_mask"] = np.ascontiguousarray(np.asarray(rec["channel_mask"])[idx])
    out["location_ids"] = np.ascontiguousarray(np.asarray(rec["location_ids"]).ravel()[idx])
    if rec.get("sample_valid") is not None:
        out["sample_valid"] = np.ascontiguousarray(np.asarray(rec["sample_valid"])[:, idx])
    return out


def swap_shanks(rec):
    """The record with its shank_L and shank_R sensors exchanged (signal, channel mask, validity); None without both."""
    loc = _loc(rec)
    a, b = SITE_ID["shank_L"], SITE_ID["shank_R"]
    if a not in loc or b not in loc:
        return None
    ja, jb = loc.index(a), loc.index(b)
    out = dict(rec)
    for key, axis in (("signal", 1), ("channel_mask", 0), ("sample_valid", 1)):
        if rec.get(key) is None:
            continue
        x = np.array(rec[key], copy=True)
        sl_a = [slice(None)] * x.ndim
        sl_b = [slice(None)] * x.ndim
        sl_a[axis], sl_b[axis] = ja, jb
        tmp = x[tuple(sl_a)].copy()
        x[tuple(sl_a)] = x[tuple(sl_b)]
        x[tuple(sl_b)] = tmp
        out[key] = x
    return out


def lowpass(rec, hz=LOWPASS_HZ, order=LOWPASS_ORDER):
    """The record with acc and gyro (channels 0 to 5) low-passed at hz (Butterworth of `order`, zero phase) on each
    contiguous run of valid finite samples of each sensor; a run no longer than the padlen of the band copy
    (window_store._padlen) is left as it is."""
    from scipy.signal import butter, sosfiltfilt

    from mdfm.data import window_store as WS
    fps = float(rec["fps"])
    x0 = np.asarray(rec["signal"])
    sig = np.array(x0, dtype=np.float64, copy=True)
    cm = np.asarray(rec["channel_mask"], dtype=bool)
    sv = (np.asarray(rec["sample_valid"], dtype=bool) if rec.get("sample_valid") is not None
          else np.ones(sig.shape[:2], bool))
    sos = butter(order, hz, btype="lowpass", fs=fps, output="sos")
    padlen = WS._padlen(sos, fps, hz)
    for g in range(sig.shape[1]):
        ch = np.flatnonzero(cm[g, :6])
        if not len(ch):
            continue
        ok = sv[:, g] & np.isfinite(sig[:, g][:, ch]).all(axis=1)
        for a, b in WS.runs(ok):
            if b - a <= padlen:
                continue
            sig[a:b, g, ch] = sosfiltfilt(sos, sig[a:b, g, ch], axis=0, padlen=padlen)
    out = dict(rec)
    out["signal"] = sig.astype(x0.dtype)
    return out


@dataclass
class Variant:
    """A model row: transform(person_id, record) -> record, None (input unchanged: the base readouts are reused) or
    EMPTY (no input: the trial is not run); refine: refinement steps (None: --refine-steps)."""
    name: str
    transform: object = None
    refine: int = None
    what: str = ""


def variants(ablations):
    """{row: Variant} of a run: base, and with ablations the model rows of the module docstring."""
    out = {"base": Variant("base", what="the model as scripts/eval_g3_dev.py runs it")}
    if not ablations:
        return out
    out["lowpass12"] = Variant("lowpass12", lambda pid, rec: lowpass(rec),
                               what=f"acc and gyro low-passed at {LOWPASS_HZ:g} Hz (order {LOWPASS_ORDER}, zero phase)")
    out["sites6"] = Variant("sites6", lambda pid, rec: cut_sites(rec, site_ids(WHC003_SITES)),
                            what="every record cut to the sites of WHC003: " + ", ".join(WHC003_SITES))
    out["swap_whc026"] = Variant("swap_whc026",
                                 lambda pid, rec: swap_shanks(rec) if short_id(pid) == SWAP_PERSON else None,
                                 what=f"the shank_L and shank_R sensors of {SWAP_PERSON} exchanged back")
    for row, k in REFINE_ROWS.items():
        out[row] = Variant(row, refine=k, what=f"{k} refinement steps")
    for row, names in FAMILIES.items():
        out[row] = Variant(row, (lambda pid, rec, keep=site_ids(names): cut_sites(rec, keep)),
                           what="every record cut to " + ", ".join(names))
    return out


# ---------------------------------------------------------------- readouts of one stitched trial body
def window_ranges(tb):
    """[(start, stop)] frames of each window of a stitched body on the trial clock (its finite pelvis frames)."""
    J = np.asarray(tb.joints)
    out = []
    for w in range(J.shape[0]):
        idx = np.flatnonzero(np.isfinite(J[w, :, 0]).all(axis=1))
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
    return out


def legs_speed_body(tb):
    """(N,) m/s of a stitched trial body: gait.stance_speed of each window's pelvis-relative joints with the stitched
    contacts of its frames, each frame from the window where it is most central (gait.stitch); the values of
    scripts/eval_gait_insim.legs_speed on the window bodies (a frame's stitched contact is its owning window's)."""
    from mdfm.readouts import gait
    J = np.asarray(tb.joints, dtype=np.float64)
    c = np.asarray(tb.contact, dtype=np.float64)
    per, starts = [], []
    for w, (s, e) in enumerate(window_ranges(tb)):
        if e <= s:
            continue
        rel = J[w, s:e] - J[w, s:e, :1]
        per.append(gait.stance_speed(rel, c[s:e], tb.rate))
        starts.append(s)
    out = gait.stitch(per, None, starts)[0] if per else np.zeros(0)
    n = c.shape[0]
    if len(out) < n:
        out = np.concatenate([out, np.full(n - len(out), np.nan)])
    return out[:n]


def heel_strike_ics(tb):
    """{"L": times, "R": times} s: gait.heel_strikes in each window's own frame (module docstring). A strike of
    window w is a candidate when its frame is owned by w in the stitch or lies within HS_BOUNDARY_S of a frame w
    owns; per foot the candidates are taken in the order (distance to the window's owned frames, distance to the
    window's centre) and one is kept unless a kept strike of that foot lies within gait.HS_DISTANCE_S. Two windows
    that place one strike a few frames apart on either side of their ownership boundary thus give it once, never
    zero times nor twice."""
    from mdfm.readouts import gait
    J = np.asarray(tb.joints, dtype=np.float64)
    wof = np.asarray(tb.window_of_frame)
    rate = float(tb.rate)
    tol = int(round(HS_BOUNDARY_S * rate))
    cands = {f: [] for f in gait.FEET}
    for w, (s, e) in enumerate(window_ranges(tb)):
        own = np.flatnonzero(wof == w)
        if e - s < 3 or not len(own):
            continue
        centre = (s + e - 1) / 2.0
        hs = gait.heel_strikes(J[w, s:e], rate)
        for foot, times in hs.items():
            for t in times:
                f = s + int(round(float(t) * rate))
                d = int(np.min(np.abs(own - f)))
                if d <= tol:
                    cands[foot].append((d, abs(f - centre), f))
    gap = gait.HS_DISTANCE_S - 1e-9
    out = {}
    for foot, cs in cands.items():
        kept = []
        for _, _, f in sorted(cs):
            t = f / rate
            if any(abs(t - k) < gap for k in kept):
                continue
            kept.append(t)
        out[foot] = np.asarray(sorted(kept), dtype=np.float64)
    return out


def _ics_frame(tid, ics):
    import pandas as pd
    rows = [(tid, foot, float(t)) for foot in sorted(ics) for t in ics[foot]]
    return pd.DataFrame(rows, columns=["trial_id", "foot", "t_s"])


def trial_readouts(tb, targets, person_id, extras=True):
    """{name: (ics, strides, speeds)} DataFrames of one stitched trial body: 'model' (g3.body_predictions), and with
    extras 'model_legs' (base ICs, the strides and pass speeds of eval_gait_insim.speed_readouts on legs_speed_body)
    and 'model_heel_strike' (heel_strike_ics)."""
    from mdfm.gates import g3
    from mdfm.readouts import gait
    p = g3.body_predictions([tb], targets)
    out = {"model": (p.ics, p.strides, p.speeds)}
    if extras:
        ic = gait.initial_contacts(tb.contact, tb.rate)
        passes = targets.passes[targets.passes["trial_id"] == tb.trial_id]
        st, sp = INSIM().speed_readouts(tb.trial_id, person_id, legs_speed_body(tb), ic, passes, tb.rate)
        out["model_legs"] = (None, st[["trial_id", "foot", "t0", "t1", "length_m"]].copy(),
                             sp[["trial_id", "pass_id", "speed_m_s"]].copy())
        out["model_heel_strike"] = (_ics_frame(tb.trial_id, heel_strike_ics(tb)), None, None)
    return out


IC_COLS = ["trial_id", "foot", "t_s"]
STRIDE_COLS = ["trial_id", "foot", "t0", "t1", "length_m"]
SPEED_COLS = ["trial_id", "pass_id", "speed_m_s"]


def predictions(name, parts):
    """g3.Predictions of a name from per-trial (ics, strides, speeds) triples (a None member stays None when every
    trial has None there)."""
    import pandas as pd

    from mdfm.gates import g3

    def cat(k, cols):
        xs = [p[k] for p in parts if p[k] is not None]
        if not xs:
            return None if parts and all(p[k] is None for p in parts) else pd.DataFrame(columns=cols)
        return pd.concat(xs, ignore_index=True)
    return g3.Predictions(name, cat(0, IC_COLS), cat(1, STRIDE_COLS), cat(2, SPEED_COLS))


# ---------------------------------------------------------------- the model passes
def run_variants(pred_base, records, targets, vlist, log=print):
    """Run every variant over the records [(trial_id, person_id, record)]: {row: {trial_id: readouts}} with the
    trial log {row: [dict]}. The base variant runs first; a variant whose transform leaves a record unchanged reuses
    the base readouts of that trial (same input, same seed); EMPTY skips the trial."""
    from dataclasses import replace as dc_replace

    from mdfm.gates import g3
    out, tlog = {}, {}
    order = ["base"] + [k for k in vlist if k != "base"]
    for row in order:
        v = vlist[row]
        pred = pred_base if v.refine is None else dc_replace(pred_base, refine_steps=int(v.refine), log=[], cache=None)
        res, lg = {}, []
        t0 = time.time()
        for tid, pid, rec in records:
            x = rec if v.transform is None else v.transform(pid, rec)
            if isinstance(x, str) and x == EMPTY:
                lg.append({"row": row, "trial_id": tid, "person_id": pid, "status": "no input site"})
                continue
            if x is None and row != "base" and v.refine is None:
                res[tid] = {"model": out["base"][tid]["model"]}
                lg.append({"row": row, "trial_id": tid, "person_id": pid, "status": "unchanged: base reused"})
                continue
            tb = pred(tid, g3.model_input(rec if x is None else x))
            r = dict(pred.log[-1])
            res[tid] = trial_readouts(tb, targets, pid, extras=(row == "base"))
            del tb
            lg.append({"row": row, "status": "run", **r})
            log(f"  {row} {tid}: {r.get('windows')} windows, {float(r.get('seconds') or 0.0):.1f} s")
        out[row], tlog[row] = res, lg
        log(f"{row}: {sum(1 for x in lg if x['status'] == 'run')} trials run in {time.time() - t0:.1f} s")
    return out, tlog


# ---------------------------------------------------------------- scoring
def score(pred, targets):
    """(summary of the predictor, its tables, offsets {(name, reference): (offset_s, n)}, unfitted) by the harness:
    eval_g3_dev.fit_offsets (g3.fit_offsets on the dev controls) when the predictor has ICs, then g3.evaluate."""
    from mdfm.gates import g3
    offsets, unfitted = (EG().fit_offsets([pred], targets) if pred.ics is not None else ({}, []))
    summ, tables = g3.evaluate(targets, [pred], {k: v[0] for k, v in offsets.items()})
    return summ[pred.name], tables[pred.name], offsets, unfitted


def person_table(st, sp, persons):
    """{task: {person: {stride_cm, stride_mape_pct, stride_bias_cm, n_strides, speed_ms, speed_bias_ms, n_passes}}}
    for task 'all', SelfPace and HurriedPace, from the matched stride and pass tables of g3 (signed: prediction minus
    reference)."""
    k = st[st["matched"]] if st is not None else None
    q = sp[sp["matched"]] if sp is not None else None
    out = {}
    for task in ("all",) + TASKS:
        ks = None if k is None else (k if task == "all" else k[k["task"] == task])
        qs = None if q is None else (q if task == "all" else q[q["task"] == task])
        rows = {}
        for p in persons:
            a = None if ks is None else ks[ks["person_id"] == p]
            b = None if qs is None else qs[qs["person_id"] == p]
            n_s = 0 if a is None else int(len(a))
            n_p = 0 if b is None else int(len(b))
            rows[p] = {"stride_cm": float(np.abs(a["err_cm"]).mean()) if n_s else None,
                       "stride_mape_pct": float(np.abs(a["ape_pct"]).mean()) if n_s else None,
                       "stride_bias_cm": float(a["err_cm"].mean()) if n_s else None, "n_strides": n_s,
                       "speed_ms": float(np.abs(b["err_m_s"]).mean()) if n_p else None,
                       "speed_bias_ms": float(b["err_m_s"].mean()) if n_p else None, "n_passes": n_p}
        out[task] = rows
    return out


def spearman(st_speeds, subset, n_boot=None, seed=None):
    """Spearman of the true pass speed against the signed speed error over the matched passes of the persons of a
    subset, with a person-clustered bootstrap interval (None under 2 persons): {rho, interval, n_passes, n_persons,
    unit}."""
    from scipy.stats import spearmanr

    from mdfm.gates import g3
    q = st_speeds[st_speeds["matched"]] if st_speeds is not None else None
    if q is None:
        return {"rho": None, "interval": None, "n_passes": 0, "n_persons": 0, "unit": "walkway pass"}
    q = q[[in_subset(p, subset) for p in q["person_id"]]]
    persons = sorted(set(q["person_id"]))
    out = {"rho": None, "interval": None, "n_passes": int(len(q)), "n_persons": len(persons), "unit": "walkway pass"}
    if len(q) < 3 or q["speed_m_s"].nunique() < 2 or q["err_m_s"].nunique() < 2:
        return out
    x, y = q["speed_m_s"].to_numpy(float), q["err_m_s"].to_numpy(float)
    out["rho"] = float(spearmanr(x, y)[0])
    if len(persons) < 2:                    # a person bootstrap of one person is no interval (as boot_mean)
        return out
    by = {p: (q.loc[q["person_id"] == p, "speed_m_s"].to_numpy(float), q.loc[q["person_id"] == p, "err_m_s"].to_numpy(float))
          for p in persons}
    rng = np.random.default_rng(g3.SEED if seed is None else seed)
    boot = []
    for _ in range(int(n_boot or g3.BOOT_N)):
        pick = rng.integers(0, len(persons), size=len(persons))
        bx = np.concatenate([by[persons[i]][0] for i in pick])
        bb = np.concatenate([by[persons[i]][1] for i in pick])
        if len(bx) >= 3 and len(np.unique(bx)) > 1 and len(np.unique(bb)) > 1:
            boot.append(spearmanr(bx, bb)[0])
    boot = np.asarray([b for b in boot if np.isfinite(b)])
    if len(boot):
        out["interval"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return out


def subset_block(pt, persons, subset):
    """The subset entry of a row: {stride_mae_cm, speed_mae_ms, stride_mape_pct, n_persons, per_person, ...}."""
    ids = [p for p in persons if in_subset(p, subset)]
    al, sp = pt["all"], pt["SelfPace"]
    per = {p: {"stride_cm": al[p]["stride_cm"], "speed_ms": al[p]["speed_ms"],
               "stride_mape_pct": al[p]["stride_mape_pct"], "selfpace_stride_cm": sp[p]["stride_cm"]} for p in ids}
    s = {p: v["stride_cm"] for p, v in per.items()}
    v = {p: x["speed_ms"] for p, x in per.items()}
    m = {p: x["stride_mape_pct"] for p, x in per.items()}
    return {"stride_mae_cm": mean_or_none(s.values()), "speed_mae_ms": mean_or_none(v.values()),
            "stride_mape_pct": mean_or_none(m.values()),
            "n_persons": sum(1 for x in s.values() if finite(x) is not None),
            "n_persons_speed": sum(1 for x in v.values() if finite(x) is not None),
            "stride_mae_cm_interval": boot_mean(s), "speed_mae_ms_interval": boot_mean(v),
            "persons_without_stride": sorted(p for p, x in s.items() if finite(x) is None),
            "per_person": per}


def task_mean_block(pt, persons, task, key):
    """{per_person, subsets {subset: {point, interval, n_persons}}} of a per-person value of one task."""
    pp = {p: pt[task][p][key] for p in persons}
    subs = {}
    for s in SUBSETS:
        vals = {p: x for p, x in pp.items() if in_subset(p, s)}
        subs[s] = {"point": mean_or_none(vals.values()), "interval": boot_mean(vals),
                   "n_persons": sum(1 for x in vals.values() if finite(x) is not None)}
    return {"per_person": pp, "subsets": subs}


def row_entry(name, summ, tables, persons, what=""):
    """One diag.json row of a scored predictor with strides and speeds (module docstring)."""
    st, sp = tables.get("strides"), tables.get("speeds")
    pt = person_table(st, sp, persons)
    row = {s: subset_block(pt, persons, s) for s in SUBSETS}
    row["what"] = what
    sel = task_mean_block(pt, persons, "SelfPace", "stride_cm")
    row["selfpace_stride_mae_cm"] = {s: {**sel["subsets"][s], "per_person": {p: v for p, v in sel["per_person"].items()
                                                                               if in_subset(p, s)}} for s in SUBSETS}
    hb = task_mean_block(pt, persons, "HurriedPace", "speed_bias_ms")
    row["hurried_speed_bias_ms"] = {s: {**hb["subsets"][s], "per_person": {p: v for p, v in hb["per_person"].items()
                                                                              if in_subset(p, s)}} for s in SUBSETS}
    row["spearman_speed_error"] = {s: spearman(sp, s) for s in SUBSETS}
    row["signed_bias"] = {task: {"stride_cm": task_mean_block(pt, persons, task, "stride_bias_cm"),
                                 "speed_ms": task_mean_block(pt, persons, task, "speed_bias_ms")}
                          for task in ("all",) + TASKS}
    row["harness_lines_all14"] = {k: {"point": v.get("point"), "interval": v.get("interval"),
                                      "n_persons": v.get("n_persons"), "matched_share": v.get("matched_share"),
                                      "offset_ms": v.get("offset_ms")}
                                  for k, v in summ.items()}
    return row, pt


def ic_block(summ, tables, offsets, unfitted, shank_tables, name):
    """{median_ms, offset_ms, margin_ratio, ...}: the IC line of a predictor against the walkway (and the insole)
    reference, its dev offset and its margin against the shank-gyroscope detector on the ICs both match."""
    from mdfm.gates import g3
    out = {"reference": "walkway", "predictor": name, "offsets_unfitted": list(unfitted)}
    for ref in g3.REFERENCES:
        line = summ.get(f"ic_median_ms_{ref}", {})
        m = g = ok = None
        tab = tables.get(f"ic_{ref}")
        if tab is not None and shank_tables is not None and f"ic_{ref}" in shank_tables:
            m, g, ok = g3.ic_margin(tab, shank_tables[f"ic_{ref}"])
        off = offsets.get((name, ref), (float("nan"), 0))
        per = line.get("per_person", {}) or {}
        blk = {"median_ms": line.get("point"), "interval": line.get("interval"), "n_persons": line.get("n_persons"),
               "matched_share": line.get("matched_share"), "offset_ms": 1000.0 * off[0] if g3.offset_fitted(off[0])
               else None, "offset_n": int(off[1]), "margin_model_ms": m, "margin_detector_ms": g,
               "margin_ratio": (m / g) if finite(m) is not None and finite(g) not in (None, 0.0) else None,
               "margin_ok": ok, "margin_line": g3.IC_MARGIN,
               "subsets": {s: mean_or_none([v for p, v in per.items() if in_subset(p, s)])
                           if all(finite(v) is not None for p, v in per.items() if in_subset(p, s)) else None
                           for s in SUBSETS},
               "per_person": per}
        if ref == "walkway":
            out.update(blk)
        else:
            out["insole"] = blk
    return out


def transfer(row, base):
    """{subset: {ratio, pass_line, kill_line, reading}}: the family's stride MAE over the base row's."""
    out = {}
    for s in SUBSETS:
        a, b = finite(row[s]["stride_mae_cm"]), finite(base[s]["stride_mae_cm"])
        r = a / b if a is not None and b not in (None, 0.0) else None
        reading = ("n/a" if r is None else f"at most {G10_PASS}" if r <= G10_PASS else
                   f"above {G10_KILL}" if r > G10_KILL else f"between {G10_PASS} and {G10_KILL}")
        out[s] = {"ratio": r, "family_stride_mae_cm": a, "base_stride_mae_cm": b, "pass_line": G10_PASS,
                  "kill_line": G10_KILL, "reading": reading, "label": PREVIEW}
    return out


def events_frame(name, tables, targets):
    """events.csv rows of a scored predictor: strides, passes and walkway ICs (module docstring)."""
    import pandas as pd
    cols = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference",
            "prediction", "signed_error", "unit", "matched"]
    parts = []
    st = tables.get("strides")
    if st is not None and len(st):
        parts.append(pd.DataFrame({"row": name, "kind": "stride", "trial_id": st["trial_id"], "person_id": st["person_id"],
                                   "task": st["task"], "foot": st["foot"], "pass_id": st["pass_id"], "t0_s": st["t0"],
                                   "t1_s": st["t1"], "reference": 100.0 * st["length_m"],
                                   "prediction": 100.0 * st["pred_m"], "signed_error": st["err_cm"], "unit": "cm",
                                   "matched": st["matched"]}))
    sp = tables.get("speeds")
    if sp is not None and len(sp):
        parts.append(pd.DataFrame({"row": name, "kind": "speed", "trial_id": sp["trial_id"], "person_id": sp["person_id"],
                                   "task": sp["task"], "foot": "", "pass_id": sp["pass_id"], "t0_s": sp["t0"],
                                   "t1_s": sp["t1"], "reference": sp["speed_m_s"], "prediction": sp["pred_m_s"],
                                   "signed_error": np.where(sp["matched"], sp["err_m_s"], np.nan), "unit": "m/s",
                                   "matched": sp["matched"]}))
    ic = tables.get("ic_walkway")
    if ic is not None and len(ic):
        task = dict(zip(targets.trials["trial_id"], targets.trials["task"]))
        tasks = ic["task"] if "task" in ic else [task.get(t, "") for t in ic["trial_id"]]
        parts.append(pd.DataFrame({"row": name, "kind": "ic_walkway", "trial_id": ic["trial_id"],
                                   "person_id": ic["person_id"], "task": tasks,
                                   "foot": ic["foot"], "pass_id": "", "t0_s": ic["t_s"], "t1_s": np.nan,
                                   "reference": 1000.0 * ic["t_s"], "prediction": 1000.0 * ic["pred_t_s"],
                                   "signed_error": np.where(ic["matched"], 1000.0 * ic["diff_s"], np.nan),
                                   "unit": "ms", "matched": ic["matched"]}))
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------- rerun check
def reference_report(name, sha256, tree=TREE):
    """(path of the G3 dev json of the same checkpoint with EMA weights, doc) or (None, reason)."""
    for folder, stem in (("g3_sim", "g3_sim"), ("g3", "g3_dev")):
        p = os.path.join(tree, "reports", "gates", folder, f"dev_{name}", f"{stem}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        ck = doc.get("checkpoint") or {}
        if ck.get("sha256") != sha256:
            return None, (f"{os.path.relpath(p, tree).replace(os.sep, '/')} is of another checkpoint file (SHA-256 "
                          f"{str(ck.get('sha256'))[:12]})")
        if ck.get("weights") not in (None, "ema"):
            continue
        return p, doc
    return None, f"no G3 dev report of {name} with EMA weights"


def rerun_check(base_row, name, sha256, weights, inference, tree=TREE):
    """The rerun check of the module docstring on the base row of all14."""
    if weights != "ema":
        return {"evaluated": False, "reading": "not applicable: raw weights (the reference reports hold EMA weights)"}
    path, doc = reference_report(name, sha256, tree)
    if path is None:
        return {"evaluated": False, "reading": f"not evaluated: {doc}"}
    lines = ((doc.get("lines") or {}).get("dev") or {}).get("model") or {}
    ref_s = finite((lines.get("stride_mae_cm") or {}).get("point"))
    ref_v = finite((lines.get("speed_mae_m_s") or {}).get("point"))
    ref_pp = {}
    pp_csv = os.path.join(os.path.dirname(path), "per_person.csv")
    if os.path.exists(pp_csv):
        with open(pp_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["row"] == "dev" and r["predictor"] == "model" and r["line"] == "stride_mae_cm":
                    ref_pp[r["person_id"]] = finite(r["value"])
    sub = base_row["all14"]
    s, v = finite(sub["stride_mae_cm"]), finite(sub["speed_mae_ms"])
    dperson = {p: (finite(x["stride_cm"]) - ref_pp[p]) if finite(x["stride_cm"]) is not None and ref_pp.get(p) is not None
               else None for p, x in sub["per_person"].items()}
    ds = None if s is None or ref_s is None else s - ref_s
    dv = None if v is None or ref_v is None else v - ref_v
    fails = []
    if ds is None or abs(ds) > RERUN_TOL["stride_mae_cm"]:
        fails.append(f"stride MAE differs by {EG().fmt(ds, 3)} cm (tolerance {RERUN_TOL['stride_mae_cm']})")
    if dv is None or abs(dv) > RERUN_TOL["speed_mae_ms"]:
        fails.append(f"speed MAE differs by {EG().fmt(dv, 4)} m/s (tolerance {RERUN_TOL['speed_mae_ms']})")
    # a person without a matched stride in both runs agrees (the reference's per_person.csv leaves such a person out)
    neither = sorted(p for p, x in sub["per_person"].items() if finite(x["stride_cm"]) is None and ref_pp.get(p) is None)
    bad = sorted(p for p, d in dperson.items()
                 if p not in neither and (d is None or abs(d) > RERUN_TOL["person_stride_cm"]))
    if bad:
        fails.append(f"person stride MAE beyond {RERUN_TOL['person_stride_cm']} cm or missing: "
                     + ", ".join(f"{short_id(p)} ({EG().fmt(dperson[p], 2)})" for p in bad))
    ref_inf = doc.get("inference") or {}
    keys = ("refine_steps", "points_per_stream", "batch_windows", "seed", "shape_pooling", "weights")
    diff_inf = {k: [ref_inf.get(k), inference.get(k)] for k in keys if ref_inf.get(k) != inference.get(k)}
    md = max((abs(d) for d in dperson.values() if d is not None), default=None)
    return {"evaluated": True, "reference": os.path.relpath(path, tree).replace("\\", "/"), "passed": not fails,
            "stride_mae_cm": {"diag": s, "reference": ref_s, "difference": ds},
            "speed_mae_ms": {"diag": v, "reference": ref_v, "difference": dv},
            "person_stride_cm_difference": dperson, "max_person_stride_difference_cm": md,
            "persons_without_stride_in_both": neither,
            "tolerances": RERUN_TOL, "inference_differences": diff_inf,
            "reading": ("the base row matches the reference report within the tolerances" if not fails else
                        "a code path diverged: " + "; ".join(fails)
                        + (f" (inference settings differ, reference then this run: {diff_inf})" if diff_inf else ""))}


# ---------------------------------------------------------------- the dev context
def dev_context():
    """{dev, train, targets, train_targets, population, summ0, tables0, offsets0, records}: g3.dev_run without a
    model (walkway, population baseline and shank-gyroscope detector on the 14 dev controls) and the dev IMU records
    [(trial_id, person_id, record)] in g3.imu_records order (held in memory for the passes of the ablations)."""
    from mdfm.gates import g3
    summ0, tables0, offsets0, tg, tr, base, preds0 = g3.dev_run(None)
    recs = [(tid, str(rec["person_id"]), rec) for tid, rec in g3.imu_records(g3.controls("dev"), tg.trials)]
    return {"dev": g3.controls("dev"), "train": g3.controls("train"), "targets": tg, "train_targets": tr,
            "population": base, "summ0": summ0, "tables0": tables0, "offsets0": offsets0, "preds0": preds0,
            "records": recs}


def default_out(loaded, ablations, tree=TREE):
    raw = "model_" if loaded.weights == "model" else ""
    return os.path.join(tree, "reports", "gates", "g3",
                        f"diag_{raw}{loaded.name}" + ("_ablations" if ablations else ""))


def zupt_mode_of(a):
    """The lever-5a mode of the parsed arguments: 'anchor' (--zupt-anchor), 'report' (--zupt-report) or 'off'."""
    return "anchor" if getattr(a, "zupt_anchor", False) else "report" if getattr(a, "zupt_report", False) else "off"


def pop_zupt_rows(entries, row=None):
    """The per-target rows of the lever-5a option moved out of trial log entries (each entry loses its zupt_rows
    list), tagged with the diag row when given."""
    out = []
    for e in entries:
        for r in e.pop("zupt_rows", None) or []:
            out.append(dict(r, row=row) if row is not None else dict(r))
    return out


def diag(a, log=print):
    """The default and --ablations runs (module docstring); returns the output folder."""
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import pandas as pd
    import torch

    from mdfm import paths
    from mdfm.data import sources
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    eg = EG()
    t_start = time.time()
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    out = a.out or default_out(loaded, a.ablations)
    paths.assert_not_moveatlas(out)
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")
    ctx = dev_context()
    dev_ids, tg = list(ctx["dev"]), ctx["targets"]
    if set(dev_ids) & set(ctx["train"]):
        raise SystemExit("dev and training fold overlap")
    if set(dev_ids) & loaded.trained_persons():
        raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev_ids) & loaded.trained_persons())}")
    if tg.counts.get("lockbox_rows"):
        raise SystemExit("a lockbox row was kept")
    persons = sorted(set(tg.persons))
    zmode = zupt_mode_of(a)
    pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                         seed=a.seed, pool=not a.no_pool, zupt_mode=zmode)
    vlist = variants(a.ablations)
    t0 = time.time()
    res, tlog = run_variants(pred, ctx["records"], tg, vlist, log=log)
    t_model = time.time() - t0
    zrows = [r for row, lg in tlog.items() for r in pop_zupt_rows(lg, row)]
    shank_tables = ctx["tables0"].get("shank_gyro")

    rows, events, pts, offsets_all, unfitted_all = {}, [], {}, {}, {}

    def scored(name, key, variant_row):
        parts = [r[key] for r in res[variant_row].values() if key in r]
        p = predictions(name, parts)
        summ, tables, offs, unf = score(p, tg)
        offsets_all.update({f"{n}|{ref}": {"offset_ms": 1000.0 * o if g3.offset_fitted(o) else None, "n_matched": k}
                            for (n, ref), (o, k) in offs.items()})
        unfitted_all[name] = unf
        events.append(events_frame(name, tables, tg))
        return summ, tables, offs, unf

    for row in [r for r in ROWS if r in vlist or r == "model_legs"]:
        if row == "model_legs":
            summ, tables, _, _ = scored("model_legs", "model_legs", "base")
            what = "the base ICs, strides and pass speeds from the stance-foot no-slip speed (eval_gait_insim.py)"
        else:
            summ, tables, offs, unf = scored(row, "model", row)
            what = vlist[row].what
        rows[row], pts[row] = row_entry(row, summ, tables, persons, what)
        if row != "model_legs" and "ic_walkway" in tables:
            rows[row]["ic_walkway"] = ic_block(summ, tables, offs, unf, shank_tables, row)
            rows[row]["ic_walkway"].pop("per_person", None)
            rows[row]["ic_walkway"].get("insole", {}).pop("per_person", None)
        if row == "base":
            hs_summ, hs_tables, hs_offs, hs_unf = scored("model_heel_strike", "model_heel_strike", "base")
            rows[row]["heel_strike_ic"] = ic_block(hs_summ, hs_tables, hs_offs, hs_unf, shank_tables, "model_heel_strike")
        if row in FAMILIES:
            rows[row]["transfer_ratio"] = transfer(rows[row], rows["base"])
            rows[row]["label"] = PREVIEW
        rows[row]["trials_run"] = sum(1 for x in tlog.get(row if row != "model_legs" else "base", [])
                                      if x["status"] == "run")
        rows[row]["trials_reused"] = sum(1 for x in tlog.get(row, []) if x["status"].startswith("unchanged"))
        rows[row]["trials_without_input"] = sorted(x["trial_id"] for x in tlog.get(row, []) if x["status"] == "no input site")

    # shank detector and contact readout beside the heel strike
    det = ctx["summ0"].get("shank_gyro", {})
    detector = {ref: {"median_ms": det.get(f"ic_median_ms_{ref}", {}).get("point"),
                      "offset_ms": det.get(f"ic_median_ms_{ref}", {}).get("offset_ms"),
                      "matched_share": det.get(f"ic_median_ms_{ref}", {}).get("matched_share"),
                      "n_uncovered": det.get(f"ic_median_ms_{ref}", {}).get("n_uncovered")} for ref in g3.REFERENCES}
    inference = {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S, "hop_s": I.HOP_S,
                 "batch_windows": a.batch_windows, "seed": a.seed,
                 "shape_pooling": "per trial" if not a.no_pool else "none", "weights": a.weights,
                 "device": str(dev), "amp_bf16": dev.type == "cuda"}
    if zmode != "off":
        from mdfm.eval import zupt_refine as ZR
        inference["zupt_anchor"] = ZR.record_settings(zmode)
    if zmode == "anchor":
        check = {"evaluated": False, "reading": "not applicable: the inertial anchor of F2 lever 5a is in the "
                                                "refinement (--zupt-anchor), and the reference reports run without it"}
    else:
        check = rerun_check(rows["base"], loaded.name, loaded.sha256, loaded.weights, inference)
    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    inference["memory"] = mem
    base = rows["base"]
    res_json = {
        "what": WHAT, "task": "T04", "mode": "ablations" if a.ablations else "default",
        "checkpoint": loaded.describe(),
        "inference": inference,
        "sets": {"dev": dev_ids, "dev_sha256": sources.person_hash(dev_ids), "tasks": list(g3.TASKS),
                 "subsets": {s: [short_id(p) for p in persons if in_subset(p, s)] for s in SUBSETS},
                 "dev_counts": tg.counts},
        "signed_error": "prediction minus reference",
        "rows": {r: rows.get(r) for r in ROWS},
        "rows_not_run": [r for r in ROWS if r not in rows],
        "selfpace_stride_mae_cm": base["selfpace_stride_mae_cm"],
        "hurried_speed_bias_ms": base["hurried_speed_bias_ms"],
        "spearman_speed_error": base["spearman_speed_error"],
        "heel_strike_ic": {k: v for k, v in base["heel_strike_ic"].items() if k != "per_person"},
        "shank_gyro_detector": detector,
        "offsets_ms": offsets_all, "offsets_unfitted": unfitted_all,
        "rerun_check": check,
        "rule": {"path": "docs/rules/G3_RULE.md", "status": "DRAFT (no line changed by this script)"},
        "timing_s": {"model": round(t_model, 1), "all": round(time.time() - t_start, 1)},
    }
    if zmode != "off":
        res_json["what"] = WHAT + ("; " + ZUPT_LABEL if zmode == "anchor" else
                                   "; F2 lever 5a report only (the objective unchanged)")
        res_json["zupt_anchor"] = ZR.summary_block([x for x in tlog.get("base", []) if x.get("status") == "run"],
                                                   zmode)
        if zmode == "anchor":
            res_json["row_label"] = ZUPT_LABEL
    clean = eg.clean(res_json)
    text = markdown(clean, a.ablations)
    common.write(clean, out, "diag", text)
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    common.write_csv(ev, os.path.join(out, "events.csv"))
    common.write_csv(per_person_frame(pts, persons), os.path.join(out, "per_person.csv"))
    common.write_csv(pd.DataFrame([x for r in tlog.values() for x in r]), os.path.join(out, "trials.csv"))
    if zmode != "off":
        cols = ("row",) + ZR.ROW_FIELDS
        common.write_csv(pd.DataFrame(zrows, columns=list(cols)), os.path.join(out, "zupt_strides.csv"))
    code = [g3.__file__, g3.gait.__file__, I.__file__, os.path.abspath(__file__), eg.__file__, INSIM().__file__]
    names = ["mdfm/gates/g3.py", "mdfm/readouts/gait.py", "mdfm/eval/infer.py", "scripts/eval_g3_diag.py",
             "scripts/eval_g3_dev.py", "scripts/eval_gait_insim.py"]
    if zmode != "off":
        code += [os.path.join(TREE, *f.split("/")) for f in ZUPT_CODE_FILES]
        names += list(ZUPT_CODE_FILES)
    write_provenance(out, {"script": "scripts/eval_g3_diag.py", "mode": res_json["mode"],
                           "checkpoint": clean["checkpoint"]["path"], "checkpoint_sha256": loaded.sha256,
                           "torch": torch.__version__, "code_sha256": common.code_sha256(code), "code_files": names,
                           **({"zupt_anchor": zmode} if zmode != "off" else {})})
    log(text)
    log(f"written {common.rel(out)}")
    return out


def per_person_frame(pts, persons):
    import pandas as pd
    rows = []
    for row, pt in pts.items():
        for task, d in pt.items():
            for p in persons:
                rows.append({"row": row, "task": task, "person_id": p, **d[p]})
    return pd.DataFrame(rows)


def write_provenance(out, extra):
    from mdfm.gates import common
    prov = {**extra, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "moveatlas_stamp": stamp_moveatlas()}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- markdown of diag.json
def _f(x, nd=2):
    return EG().fmt(x, nd)


def markdown(r, ablations):
    ck = r["checkpoint"]
    L = [f"# G3 dev diagnostics: {ck['name']}", "",
         "Written by scripts/eval_g3_diag.py (task T04; harness mdfm/gates/g3.py, inference mdfm/eval/infer.py as in "
         "scripts/eval_g3_dev.py); every number below is read from diag.json, per_person.csv and events.csv of this "
         f"folder. {WHAT}: the 14 dev controls, no lockbox row or record, no rule line changed. Signed errors are "
         "prediction minus reference.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,}. Inference: {r['inference']['refine_steps']} refinement steps of "
         f"{r['inference']['points_per_stream']} points, batch_windows {r['inference']['batch_windows']}, seed "
         f"{r['inference']['seed']}, shape pooled {r['inference']['shape_pooling']}, device {r['inference']['device']}.",
         "", "Subsets: all14; no_whc003 (WHC003 holds no shank sensor); no_whc003_whc026 (WHC026's shank sides are "
         "exchanged), the 12.", ""]
    za = r.get("zupt_anchor")
    if za:
        tm = za.get("trial_medians") or {}
        L += ["## F2 lever 5a option", "",
              f"Mode {za['mode']} ({za['what']}): weight {za['weight']:g}, sd {za['sd_m_s']:g} m/s, edge {za['edge_s']:g}"
              f" s, strides of at least {za['min_length_m']:g} m, {za['steps']} steps; targets from {za['source']}. "
              f"{za['trials']} trials, {za['trials_with_targets']} with a target, {za['targets']} targets (window and "
              f"stride pairs) of {za['record_strides']} record strides. Medians over the trials: "
              + ("" if za["mode"] != "anchor" else
                 f"term {_f(tm.get('zupt_term_first'), 1)} at the first refinement call and "
                 f"{_f(tm.get('zupt_term_last'), 1)} at the last, decoded over ZUPT displacement "
                 f"{_f(tm.get('zupt_ratio_first'), 3)} and {_f(tm.get('zupt_ratio_last'), 3)}; ")
              + f"on the final bodies D over S {_f(tm.get('zupt_report_ratio_median'), 3)}, ankle speed at the "
              f"still instants {_f(tm.get('zupt_report_speed_median'), 3)} m/s, slide over the measured still run "
              f"{_f(tm.get('zupt_report_slide_share_median'), 3)} of S. Per target: zupt_strides.csv.", ""]
    L += ["## Rows", "", "| row | all14 stride MAE (cm) | speed MAE (m/s) | MAPE (%) | n | no_whc003 stride | speed | n | "
          "12 stride | 12 speed | 12 MAPE | n |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ROWS:
        row = r["rows"].get(name)
        if row is None:
            continue
        a, b, c = row["all14"], row["no_whc003"], row[SUB12]
        L.append(f"| {name} | {_f(a['stride_mae_cm'])} | {_f(a['speed_mae_ms'], 3)} | {_f(a['stride_mape_pct'], 1)} | "
                 f"{a['n_persons']} | {_f(b['stride_mae_cm'])} | {_f(b['speed_mae_ms'], 3)} | {b['n_persons']} | "
                 f"{_f(c['stride_mae_cm'])} | {_f(c['speed_mae_ms'], 3)} | {_f(c['stride_mape_pct'], 1)} | "
                 f"{c['n_persons']} |")
    if r["rows_not_run"]:
        L += ["", "Rows not run here (null in diag.json): " + ", ".join(r["rows_not_run"]) +
              ("." if ablations else "; scripts/eval_g3_diag.py --ablations computes them.")]
    L += ["", "Row definitions: " + "; ".join(f"{n}: {r['rows'][n]['what']}" for n in ROWS if r["rows"].get(n)) + ".",
          ""]
    base = r["rows"]["base"]
    L += ["## Speed error against speed (base)", "",
          "| subset | hurried speed bias (m/s) | 95 % interval | SelfPace stride MAE (cm) | Spearman(true speed, signed "
          "error) | 95 % interval | passes | persons |", "|---|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        hb, sp, sr = base["hurried_speed_bias_ms"][s], base["selfpace_stride_mae_cm"][s], base["spearman_speed_error"][s]
        L.append(f"| {s} | {_f(hb['point'], 3)} | {EG().fmt_iv(hb['interval'], 3)} | {_f(sp['point'])} | "
                 f"{_f(sr['rho'], 3)} | {EG().fmt_iv(sr['interval'], 3)} | {sr['n_passes']} | {sr['n_persons']} |")
    L += ["", "Signed biases of the base row, mean over persons (all14 / 12):", ""]
    for task in ("all",) + TASKS:
        sb = base["signed_bias"][task]
        L.append(f"- {task}: stride {_f(sb['stride_cm']['subsets']['all14']['point'])} / "
                 f"{_f(sb['stride_cm']['subsets'][SUB12]['point'])} cm; speed "
                 f"{_f(sb['speed_ms']['subsets']['all14']['point'], 3)} / "
                 f"{_f(sb['speed_ms']['subsets'][SUB12]['point'], 3)} m/s.")
    hs, ci = base["heel_strike_ic"], base.get("ic_walkway") or {}
    det = r["shank_gyro_detector"]["walkway"]
    L += ["", "## Initial contacts (walkway reference, all14)", "",
          "| predictor | IC median (ms) | 95 % interval | offset (ms) | matched share | margin model (ms) | margin "
          "detector (ms) | ratio (line 0.8) |", "|---|---|---|---|---|---|---|---|",
          f"| model (contact readout) | {_f(ci.get('median_ms'), 1)} | {EG().fmt_iv(ci.get('interval'), 1)} | "
          f"{_f(ci.get('offset_ms'), 1)} | {_f(ci.get('matched_share'), 3)} | {_f(ci.get('margin_model_ms'), 2)} | "
          f"{_f(ci.get('margin_detector_ms'), 2)} | {_f(ci.get('margin_ratio'), 3)} |",
          f"| model_heel_strike | {_f(hs.get('median_ms'), 1)} | {EG().fmt_iv(hs.get('interval'), 1)} | "
          f"{_f(hs.get('offset_ms'), 1)} | {_f(hs.get('matched_share'), 3)} | {_f(hs.get('margin_model_ms'), 2)} | "
          f"{_f(hs.get('margin_detector_ms'), 2)} | {_f(hs.get('margin_ratio'), 3)} |",
          f"| shank_gyro | {_f(det.get('median_ms'), 1)} |  | {_f(det.get('offset_ms'), 1)} | "
          f"{_f(det.get('matched_share'), 3)} |  |  |  |", "",
          "An IC median reads n/a when a person has more than half of its reference ICs unmatched (infinite, the G3 "
          "kill case). The heel strikes are found in each window's own frame (module docstring).", ""]
    fam = [n for n in FAMILIES if r["rows"].get(n)]
    if fam:
        L += [f"## Site families ({PREVIEW})", "", "Transfer ratio: the family's stride MAE over the base row's, "
              f"against the design's G10 lines (at most {G10_PASS}, kill above {G10_KILL}). The design's G10 line "
              "splits weargait_pd by subject-id family (section 8 (c)); this sensor-family ratio is a preview only.",
              "", "| family | all14 ratio | no_whc003 ratio | 12 ratio | reading on the 12 |", "|---|---|---|---|---|"]
        for n in fam:
            t = r["rows"][n]["transfer_ratio"]
            L.append(f"| {n} | {_f(t['all14']['ratio'], 3)} | {_f(t['no_whc003']['ratio'], 3)} | "
                     f"{_f(t[SUB12]['ratio'], 3)} | {t[SUB12]['reading']} |")
        L.append("")
    ck_ = r["rerun_check"]
    L += ["## Rerun check", "", f"{ck_['reading']}."]
    if ck_.get("evaluated"):
        L.append(f"Reference {ck_['reference']}: stride MAE {_f(ck_['stride_mae_cm']['diag'], 3)} against "
                 f"{_f(ck_['stride_mae_cm']['reference'], 3)} cm, speed MAE {_f(ck_['speed_mae_ms']['diag'], 4)} "
                 f"against {_f(ck_['speed_mae_ms']['reference'], 4)} m/s, largest person stride difference "
                 f"{_f(ck_['max_person_stride_difference_cm'], 3)} cm. Inference settings that differ (reference, "
                 f"this run): {ck_['inference_differences'] or 'none'}.")
    L += ["", "## Base row per person (all14)", "", "| person | stride MAE (cm) | speed MAE (m/s) | SelfPace stride (cm) "
          "| SelfPace stride bias (cm) | HurriedPace speed bias (m/s) |", "|---|---|---|---|---|---|"]
    for p, v in base["all14"]["per_person"].items():
        L.append(f"| {short_id(p)} | {_f(v['stride_cm'])} | {_f(v['speed_ms'], 3)} | {_f(v['selfpace_stride_cm'])} | "
                 f"{_f(base['signed_bias']['SelfPace']['stride_cm']['per_person'].get(p))} | "
                 f"{_f(base['signed_bias']['HurriedPace']['speed_ms']['per_person'].get(p), 3)} |")
    L += ["", f"Time: model passes {r['timing_s']['model']} s, all {r['timing_s']['all']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- --curve
def _num(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return finite(d)


def curve_entries(run, tree=TREE):
    """Every G3 dev json of the checkpoints of a run (module docstring), as dicts sorted by phase, step and source."""
    pats = [("g3", f"dev_{run}_*"), ("g3_sim", f"dev_{run}_*"), ("g3", f"curve_{run}_*"), ("g3_sim", f"curve_{run}_*")]
    seen, out = set(), []
    for rank, (folder, pat) in enumerate(pats):
        for d in sorted(glob.glob(os.path.join(tree, "reports", "gates", folder, pat))):
            if os.path.basename(d).startswith("._") or not os.path.isdir(d) or d in seen:
                continue
            seen.add(d)
            js = next((os.path.join(d, n) for n in ("g3_dev.json", "g3_sim.json") if os.path.exists(os.path.join(d, n))),
                      None)
            if js is None:
                continue
            with open(js, encoding="utf-8") as f:
                doc = json.load(f)
            ck = doc.get("checkpoint") or {}
            m = CKPT_NAME.match(str(ck.get("name", "")))
            if not m or m.group("run") != run:
                continue
            res = {x.get("line"): x for x in doc.get("results_dev", []) or [] if isinstance(x, dict)}
            ex = (res.get("ic_median_ms_walkway") or {}).get("extra") or {}
            mm, md = finite(ex.get("margin_model_ms")), finite(ex.get("margin_detector_ms"))
            model = ((doc.get("lines") or {}).get("dev") or {}).get("model") or {}
            off = (doc.get("offsets_ms") or {}).get("model|walkway") or {}
            out.append({"source": os.path.relpath(d, tree).replace("\\", "/"), "json": os.path.basename(js),
                        "rank": rank, "phase": m.group("phase"), "step": int(m.group("step")),
                        "weights": ck.get("weights", "ema"), "sha256": ck.get("sha256"),
                        "stride_mae_cm": _num(model, "stride_mae_cm", "point"),
                        "speed_mae_m_s": _num(model, "speed_mae_m_s", "point"),
                        "stride_mape_pct": _num(model, "stride_mape_pct", "point"),
                        "ic_median_ms_walkway": _num(model, "ic_median_ms_walkway", "point"),
                        "ic_offset_ms": finite(off.get("offset_ms")),
                        "margin_model_ms": mm, "margin_detector_ms": md,
                        "margin_ratio": mm / md if mm is not None and md not in (None, 0.0) else None,
                        "margin_ok": ex.get("margin_ok"),
                        "n_persons": (model.get("stride_mae_cm") or {}).get("n_persons"),
                        "secondary_row": "all68" in ((doc.get("lines") or {}))})
    out.sort(key=lambda e: (e["phase"], e["step"], e["weights"] != "ema", e["rank"], e["source"]))
    return out


def curve(run, tree=TREE, out=None, log=print):
    """--curve: curve.json and curve.md of a run (module docstring); returns the output folder."""
    from mdfm.gates import common
    entries = curve_entries(run, tree)
    steps, raw, dup = [], [], []
    keys = set()
    for e in entries:
        k = (e["phase"], e["step"], e["weights"])
        if k in keys:
            dup.append(e)
            continue
        keys.add(k)
        (steps if e["weights"] == "ema" else raw).append(e)
    out = out or os.path.join(tree, "reports", "gates", "g3", f"curve_{run}")
    res = {"what": f"G3 dev learning curve of {run} (T04), not a gate read", "run": run, "steps": steps,
           "raw_weights": raw, "duplicates": dup,
           "columns": ["stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "margin_ratio",
                       "ic_offset_ms"],
           "note": "values of the model on the 14 dev controls (row dev of each json); an IC median of null is "
                   "infinite (a person with most ICs unmatched) or missing"}
    res = EG().clean(res)
    L = [f"# G3 dev learning curve: {run}", "",
         "Written by scripts/eval_g3_diag.py --curve (task T04) from the G3 dev jsons of the run's checkpoints (row dev, "
         "the 14 dev controls; model, EMA weights unless listed apart). Every number is read from curve.json, which "
         "names the source folder of each step. Not a gate read.", "",
         "| phase | step | stride MAE (cm) | speed MAE (m/s) | stride MAPE (%) | IC median, walkway (ms) | margin ratio "
         "(model / detector, line 0.8) | IC offset (ms) | source |", "|---|---|---|---|---|---|---|---|---|"]
    for e in steps:
        L.append(f"| {e['phase']} | {e['step']:,} | {_f(e['stride_mae_cm'])} | {_f(e['speed_mae_m_s'], 3)} | "
                 f"{_f(e['stride_mape_pct'], 1)} | {_f(e['ic_median_ms_walkway'], 1)} | {_f(e['margin_ratio'], 3)} | "
                 f"{_f(e['ic_offset_ms'], 1)} | {e['source']} |")
    if not steps:
        L.append("| n/a | | | | | | | | no G3 dev json of this run found |")
    if raw:
        L += ["", "Raw weights (not on the curve): " + "; ".join(
            f"{e['phase']} {e['step']:,}: stride {_f(e['stride_mae_cm'])} cm, speed {_f(e['speed_mae_m_s'], 3)} m/s "
            f"({e['source']})" for e in raw) + "."]
    if dup:
        L += ["", "Other folders of a step already listed (not on the curve): " + ", ".join(e["source"] for e in dup) + "."]
    text = "\n".join(L) + "\n"
    common.write(res, out, "curve", text)
    write_provenance(out, {"script": "scripts/eval_g3_diag.py", "mode": "curve", "run": run,
                           "sources": [e["source"] for e in entries]})
    log(text)
    log(f"written {common.rel(out)}")
    return out


# ---------------------------------------------------------------- --quantiles
def _q(values):
    v = np.asarray([x for x in (finite(y) for y in values) if x is not None], dtype=np.float64)
    return {"n": int(len(v)), "quantiles": {f"{q:.2f}": (float(np.quantile(v, q)) if len(v) else None) for q in QUANTILES},
            "mean": float(v.mean()) if len(v) else None}


def amass_train_walks(path=AMASS_WALKS_CSV):
    """(rows {subset, speed_m_s, cadence_steps_min} of the takes whose archive is a train archive, {subset: split})."""
    from mdfm.data import amass_index
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    splits = {s: amass_index.check(s) for s in sorted({r["subset"] for r in rows})}
    keep = [{"subset": r["subset"], "speed_m_s": finite(r["speed_m_s"]), "cadence_steps_min": finite(r["cadence_steps_min"])}
            for r in rows if splits[r["subset"]] == "train"]
    return keep, splits


def weargait_train_walkway():
    """Rows {task, person_id, speed_m_s, cadence_steps_min} of the trials of the weargait training-fold controls,
    SelfPace and HurriedPace (g3.read_rows: no lockbox row)."""
    from mdfm.gates import g3
    train = g3.controls("train")
    rows = g3.read_rows(g3.TRIALS_CSV, train)
    return [{"task": r["task"], "person_id": r["person_id"], "speed_m_s": finite(r.get("ww_speed_m_s")),
             "cadence_steps_min": finite(r.get("cadence_contacts"))} for r in rows if r["task"] in TASKS], train


def share_at_or_below(values, x):
    v = [y for y in (finite(z) for z in values) if y is not None]
    return float(np.mean([y <= x for y in v])) if v and finite(x) is not None else None


def quantile_tables(amass, wg):
    """The quantile json: amass {speed, cadence}, weargait {task: {speed, cadence, n_trials, n_persons}} and, per task
    and quantity, the share of AMASS takes at or below each weargait quantile and inside its 5 to 95 percent range."""
    out = {"quantiles": [f"{q:.2f}" for q in QUANTILES],
           "amass_train": {"n_takes": len(amass), "subsets": sorted({r["subset"] for r in amass}),
                           "speed_m_s": _q(r["speed_m_s"] for r in amass),
                           "cadence_steps_min": _q(r["cadence_steps_min"] for r in amass)},
           "weargait_train_walkway": {}, "comparison": {}}
    for task in TASKS + ("both",):
        sel = [r for r in wg if task == "both" or r["task"] == task]
        out["weargait_train_walkway"][task] = {"n_trials": len(sel), "n_persons": len({r["person_id"] for r in sel}),
                                               "speed_m_s": _q(r["speed_m_s"] for r in sel),
                                               "cadence_steps_min": _q(r["cadence_steps_min"] for r in sel)}
        comp = {}
        for qty in ("speed_m_s", "cadence_steps_min"):
            wq = out["weargait_train_walkway"][task][qty]["quantiles"]
            av = [r[qty] for r in amass]
            lo, hi = wq.get("0.05"), wq.get("0.95")
            inside = [x for x in (finite(y) for y in av) if x is not None]
            comp[qty] = {"amass_share_at_or_below": {k: share_at_or_below(av, v) for k, v in wq.items()},
                         "amass_share_inside_weargait_5_95": (float(np.mean([lo <= x <= hi for x in inside]))
                                                              if inside and lo is not None and hi is not None else None),
                         "amass_share_above_weargait_95": (float(np.mean([x > hi for x in inside]))
                                                           if inside and hi is not None else None)}
        out["comparison"][task] = comp
    return out


def quantiles_report(tree=TREE, out=None, log=print):
    """--quantiles (module docstring); returns the output folder."""
    from mdfm.gates import common, g3
    amass, splits = amass_train_walks()
    wg, train = weargait_train_walkway()
    q = quantile_tables(amass, wg)
    q["what"] = "speed and cadence quantiles, AMASS train walks against the weargait training-fold walkway (T04)"
    q["sources"] = {"amass": "reports/amass_walks/takes.csv", "amass_splits": splits,
                    "weargait": "reports/weargait_targets/trials.csv (ww_speed_m_s, cadence_contacts)",
                    "weargait_persons": len(train), "tasks": list(TASKS)}
    q["definitions"] = {"amass": "per take, speed_m_s and cadence_steps_min of scripts/prepare_amass_walks.py",
                        "weargait": "per trial, the walkway speed (B3a speed rule over the trial's passes) and the "
                                    "walkway cadence (cadence_contacts) of scripts/build_weargait_targets.py"}
    q = EG().clean(q)
    out = out or os.path.join(tree, "reports", "gates", "g3", "speed_cadence_quantiles")
    A = q["amass_train"]
    L = ["# Speed and cadence quantiles: AMASS train walks against the weargait training-fold walkway", "",
         "Written by scripts/eval_g3_diag.py --quantiles (task T04, CPU, no model); every number is read from "
         "quantiles.json of this folder. Population quantiles only: no person-level value is written.", "",
         f"- AMASS train walks: {A['n_takes']} takes of reports/amass_walks/takes.csv (subsets "
         f"{', '.join(A['subsets'])}, each checked to be a train archive), per take speed and cadence.",
         f"- weargait training fold: the {q['sources']['weargait_persons']} non-lockbox training-fold controls, "
         "walkway speed and cadence per trial (reports/weargait_targets/trials.csv).", ""]
    for qty, unit, nd in (("speed_m_s", "m/s", 3), ("cadence_steps_min", "steps/min", 1)):
        L += [f"## {qty} ({unit})", "", "| population | n | " + " | ".join(f"q{float(x) * 100:g}" for x in q["quantiles"])
              + " |", "|---|---|" + "---|" * len(q["quantiles"])]
        L.append(f"| AMASS train | {A[qty]['n']} | " + " | ".join(_f(A[qty]['quantiles'][k], nd) for k in q["quantiles"])
                 + " |")
        for task in TASKS + ("both",):
            W_ = q["weargait_train_walkway"][task]
            L.append(f"| weargait {task} | {W_[qty]['n']} | " +
                     " | ".join(_f(W_[qty]['quantiles'][k], nd) for k in q["quantiles"]) + " |")
        L += ["", "Share of AMASS train takes at or below each weargait quantile, and inside the weargait 5 to 95 "
              "percent range:", ""]
        for task in TASKS:
            c = q["comparison"][task][qty]
            L.append(f"- {task}: " + ", ".join(f"q{float(k) * 100:g} {_f(v, 3)}" for k, v in
                                               c["amass_share_at_or_below"].items())
                     + f"; inside {_f(c['amass_share_inside_weargait_5_95'], 3)}, above the 95th "
                       f"{_f(c['amass_share_above_weargait_95'], 3)}.")
        L.append("")
    text = "\n".join(L) + "\n"
    common.write(q, out, "quantiles", text)
    from mdfm.train import provenance as PV
    write_provenance(out, {"script": "scripts/eval_g3_diag.py", "mode": "quantiles",
                           "inputs_sha256": {common.rel(p): PV.sha256_file(p) for p in
                                             (AMASS_WALKS_CSV, g3.TRIALS_CSV, g3.SUMMARY_JSON)},
                           "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__])})
    log(text)
    log(f"written {common.rel(out)}")
    return out


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", help="a checkpoint of mdfm/train/checkpoint.py (default and --ablations runs)")
    p.add_argument("--ablations", action="store_true", help="the model rows of the ablations")
    p.add_argument("--curve", metavar="RUN", help="tabulate the G3 dev reports of the run's checkpoints")
    p.add_argument("--quantiles", action="store_true", help="speed and cadence quantiles (CPU, no model)")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256, help="kernel points per stream-window and refinement step")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a trial")
    z = p.add_mutually_exclusive_group()
    z.add_argument("--zupt-anchor", action="store_true",
                   help="F2 lever 5a: the inertial anchor in the refinement (constants fixed; needs --out)")
    z.add_argument("--zupt-report", action="store_true",
                   help="F2 lever 5a report only: targets and foot report, objective unchanged (needs --out)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    modes = [bool(a.checkpoint), bool(a.curve), bool(a.quantiles)]
    if sum(modes) != 1:
        p.error("give exactly one of --checkpoint, --curve RUN and --quantiles")
    if a.ablations and not a.checkpoint:
        p.error("--ablations needs --checkpoint")
    if a.zupt_anchor or a.zupt_report:
        from mdfm.eval import zupt_refine as ZR
        if not a.checkpoint:
            p.error("--zupt-anchor and --zupt-report need --checkpoint")
        if not a.out:
            p.error("--zupt-anchor and --zupt-report need --out (never the folder of the run without the option)")
        if a.zupt_anchor and a.refine_steps != ZR.STEPS:
            p.error(f"--zupt-anchor runs {ZR.STEPS} refinement steps (a fixed constant of lever 5a)")
        if a.zupt_anchor and a.ablations:
            p.error("--zupt-anchor runs the default diag only (lever 5a)")
    return a


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.quantiles:
        quantiles_report(out=a.out)
    elif a.curve:
        curve(a.curve, out=a.out)
    else:
        diag(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
