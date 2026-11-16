"""CPL: the coupling diagnostic (rank 6 of docs/notes/STRATEGY_2026-10-15.md, section 4). Dev diagnostic, not a gate
read.

Question. docs/notes/SCALE_2026-10-15.md (test K, section 4) located the decoded body's short stride in the angular-rate
fit and proposed, untested, that the body takes its progression from the segment rotations through a coupling of
rotation and progression learned on synthetic (AMASS) bodies, under which the real controls' rotations read short.
This script measures that coupling in the data, with no model: how much the thigh, shank and foot rotate per metre
walked in the AMASS training walks and in the 54 training-fold weargait_pd controls, at equal stride, leg length and
cadence, and the stride at which an AMASS body of the walker's leg length and cadence would rotate its segments as the
real persons do (gamma_L).

Pre-registration: reports/bench/cpl/prediction.txt, its SHA-256 in reports/bench/cpl/prediction.sha256 and in
PRED_SHA256 below, asserted before any value is read (check_prediction).

Populations:
- real: train54 of reports/bench/g3_gain_check/results.json (f2_H1shape.load_sets: F1's lists and SHA-256 asserted,
  no person of the weargait_pd lockbox lists of mdfm.gates.g3 and mdfm.data.window_store), tasks SelfPace and
  HurriedPace; walkway strides of g3.load_targets (heel to heel of one foot, PKMAS cells); the gyroscopes of the
  stride side's thigh, shank, ankle and foot sensors of the canonical IMU records (g3.imu_records; locations
  thigh 5/6, shank 7/8, ankle 9/10, foot 11/12 of imu_render.SITES), NaN where sample_valid is false; stature from
  the demographic tables (f2_H1shape.read_demographics). Never a dev, lockbox, EVAL, CODY or care_pd record.
- AMASS train: the straight forward walks of the train body store (task 'walk'; styles of is_straight_forward:
  BMLrub normal_walk and treadmill_norm/slow/fast, KIT walking_slow/medium/fast, WalkingStraightForward(s), walking_N,
  walkingN, walk_Nm_straight_line; circles, backwards, obstacles, supports and handrails left out), each take whole on
  its knots (BodyStore.windows; treadmill takes in the belt frame of the store); the true body in float64 (KnotBatch
  kinematics: joints, velocities, global rotations, world angular velocities); contacts by frames.contact_labels with
  the exact velocities and the floor at y = 0, ICs by readouts.gait.initial_contacts, strides by gait.strides and
  scored by eval_gait_insim.steady_straight (the in-sim rule); stature of the rest mesh (f2_H1shape.stature_of).
  The ideal gyroscope of a site is the segment's angular velocity in the segment frame (R^T omega, as
  factory/imu_render.py computes it before the mount rotation, the bias and the white noise): segment joints of
  imu_render.SITE_RULES (thigh 1/2, shank and ankle 4/5, foot 7/8).
- AMASS dev: the forward walks of the dev body store (eval_gait_insim.select_takes, forward rows), the same code:
  the domain where the decoder reads the stride right (0.9655 in sim, v1_B).

Per stride and segment (the same code on both populations, segment_table):
- the gyroscope low-passed at LP_HZ (zero-phase Butterworth of order LP_ORDER on each finite run; the store's bodies
  are low-passed at 12 Hz before they are stored);
- the sagittal axis: the principal axis of the uncentred second moment of the gyro samples inside the kept strides of
  that trial (take), side and segment; planarity = its share of the trace;
- ROM (deg): the excursion (max minus min) of the integrated sagittal angle over the stride (trapezoid at the record
  rate from t0 to t1), linearly detrended so that it ends where it starts (removes a constant bias); peak (deg/s): the
  largest absolute sagittal rate in the stride. A stride with a non-finite sample in its window has no value.
- S (m), T = t1 - t0 (s), cadence c = 120 / T (steps/min), stride speed v = S / T (m/s), leg l = LEG_RATIO x stature.
  Kept: T within T_RANGE, S at least S_MIN, every value finite (AMASS: scored strides only).

Quantities (persons weighted equally throughout; bootstrap: AMASS persons and real persons resampled together, the
AMASS relations refitted on each resample):
- relation of a segment s on AMASS: log ROM_s = a_s + b_s log(S / l) + d_s (log c - log 110); sigma_s^2 its residual
  variance; kappa_s = exp(median over persons of each person's median residual of the real strides): the real
  rotation over the AMASS rotation at equal stride over leg, and cadence (below 1: the real persons rotate less per
  metre);
- gamma_L (primary): per stride, delta = sum_s w_s b_s r_s / sum_s w_s b_s^2 over thigh, shank and foot (w_s =
  1 / sigma_s^2, r_s the residual): the log stride shift at which the AMASS relations best reproduce the three
  measured rotations, a likelihood-style inversion of the forward relation (the decoder fits rotations); gamma_L =
  exp(median over persons of each person's median delta): the coupling-implied stride over the walked stride;
- gamma_s: the same with one segment; gamma_R: the inverse regression log(S / l) on the three log ROMs and the
  cadence (a posterior-mean reading, shrunk toward the AMASS strides);
- kappa_v,s: log ROM_s on (1, log v, log l): rotation at equal speed and leg length;
- h = log gamma_L / log Q, Q = the train54 decoded stride gain over the in-sim gain (v1_B; reports/bench/scale/chain):
  the share of the real-specific gap the coupling accounts for;
- link: Pearson r over train54 persons of log gamma_p (person median delta) and log g_p (median decoded over walkway
  stride, F1's v1_B decode, row base, matched strides), person bootstrap with the relations fixed;
- method checks: M1 split-half (relations fitted on one half of the AMASS train persons, gamma_L read on the other
  half, both ways); M2 the AMASS dev forward walks read through the train relations.

Writes reports/bench/cpl/: results.json, summary.md, real_strides.csv, amass_train_strides.csv,
amass_dev_strides.csv, _provenance.json (MoveAtlas stamp, script and prediction SHA-256).

Run (mdfm environment, CPU only):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/cpl_coupling.py [--limit-amass N] [--out DIR]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # never the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "bench", "cpl")
PRED = os.path.join(OUT, "prediction.txt")
PRED_SHA_FILE = os.path.join(OUT, "prediction.sha256")
PRED_SHA256 = "ba7b3233fd31529a6a3ae4b37275c70569a2aff0442b04b7e8b54437a1f9eff0"   # prediction.txt, 21:17:58
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
F1_EVENTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "decode_phaseA_root_full_ckpt_B_032044",
                         "train_events.csv")
CHAIN = os.path.join(TREE, "reports", "bench", "scale", "chain", "results.json")
LABEL = "dev diagnostic, not a gate read"
TASK = "CPL"

RATE = 100.0
LP_HZ = 12.0
LP_ORDER = 4
T_RANGE = (0.6, 2.0)            # stride time, s (f2_H5data)
S_MIN = 0.2                     # stride length, m (f2_H5data)
LEG_RATIO = 0.530               # greater trochanter height over stature (Winter 2009; f2_H5data)
C_CENTRE = math.log(110.0)
SEGMENTS = ("thigh", "shank", "ankle", "foot")
PRIMARY = ("thigh", "shank", "foot")
REAL_SITES = {"thigh": {"L": 5, "R": 6}, "shank": {"L": 7, "R": 8}, "ankle": {"L": 9, "R": 10},
              "foot": {"L": 11, "R": 12}}
BODY_JOINTS = {"thigh": {"L": 1, "R": 2}, "shank": {"L": 4, "R": 5}, "ankle": {"L": 4, "R": 5},
               "foot": {"L": 7, "R": 8}}
SPEED_BINS = (0.4, 0.8, 1.0, 1.2, 1.4, 2.2)
BOOT_N = 2000
SEED = 20261016
LEVEL = 0.95
# registered reading of gamma_L (prediction.txt section 3)
EXPLAINS_POINT, EXPLAINS_UPPER = 0.935, 0.97
MATCHED_BAND, MATCHED_IV = (0.965, 1.035), (0.94, 1.06)
LONG_POINT, LONG_LOWER = 1.035, 1.0
M1_BAND = (0.97, 1.03)
M2_BAND = (0.95, 1.05)

STRAIGHT_FORWARD = (
    re.compile(r"^\d+_normal_walk\d*$"), re.compile(r"^\d+_treadmill_(norm|slow|fast)$"),
    re.compile(r"^walking_(slow|medium|fast)\d*$"), re.compile(r"^WalkingStraightForwards?\d+(_\d+)?$"),
    re.compile(r"^walking_?\d+$"), re.compile(r"^walk_\d+m_straight_line\d*$"))


# ---------------------------------------------------------------- helpers
def rel(path):
    try:
        r = os.path.relpath(os.path.abspath(path), TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


SCRIPT_SHA256_AT_START = sha256_file(os.path.abspath(__file__))


def _sibling(name):
    key = f"_cpl_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def check_prediction(path=PRED, sha_file=PRED_SHA_FILE, expected=None):
    """SHA-256 of prediction.txt, asserted equal to the first word of prediction.sha256 and to PRED_SHA256."""
    expected = PRED_SHA256 if expected is None else expected
    if not expected:
        raise AssertionError("PRED_SHA256 is not set: the prediction is not registered")
    got = sha256_file(path)
    with open(sha_file, encoding="utf-8") as f:
        listed = f.read().split()[0]
    if not (got == listed == expected):
        raise AssertionError(f"prediction {got}, hash file {listed}, expected {expected}")
    return got


def style_of(take_id):
    return str(take_id).rsplit("/", 1)[-1].replace("_poses.npz", "").replace("_stageii.npz", "")


def is_straight_forward(take_id):
    s = style_of(take_id)
    return any(p.match(s) for p in STRAIGHT_FORWARD)


def person_split(person_id):
    """0 or 1: the parity of the first 8 hex digits of the SHA-256 of the person id (method check M1)."""
    return int(hashlib.sha256(str(person_id).encode("utf-8")).hexdigest()[:8], 16) % 2


# ---------------------------------------------------------------- signal code (both populations)
def lowpass(x, fps, hz=LP_HZ, order=LP_ORDER):
    """x (N, ...) -> zero-phase Butterworth low-pass along axis 0, applied on each run of rows that are finite in
    every column; runs too short for filtfilt's padding are set to NaN."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, dtype=np.float64)
    b, a = butter(order, hz / (fps / 2.0))
    pad = 3 * max(len(a), len(b))
    flat = x.reshape(len(x), -1)
    ok = np.isfinite(flat).all(1)
    out = np.full_like(flat, np.nan)
    k = 0
    n = len(ok)
    while k < n:
        if not ok[k]:
            k += 1
            continue
        e = k
        while e < n and ok[e]:
            e += 1
        if e - k > pad:
            out[k:e] = filtfilt(b, a, flat[k:e], axis=0)
        k = e
    return out.reshape(x.shape)


def sagittal_axis(samples):
    """samples (M, 3) -> (unit axis (3,), planarity): the principal axis of the uncentred second moment and its share
    of the trace."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s).all(1)]
    if len(s) < 10:
        return None, float("nan")
    m = s.T @ s / len(s)
    w, v = np.linalg.eigh(m)
    tr = float(w.sum())
    return v[:, -1], (float(w[-1] / tr) if tr > 0 else float("nan"))


def stride_rom(rate_axis, i0, i1, fps):
    """rate_axis (N,) deg/s -> (ROM deg, peak deg/s) over samples i0..i1 inclusive, or (nan, nan) when a sample is
    not finite or the window is outside the signal."""
    if i0 < 0 or i1 >= len(rate_axis) or i1 - i0 < 2:
        return float("nan"), float("nan")
    w = np.asarray(rate_axis[i0:i1 + 1], dtype=np.float64)
    if not np.isfinite(w).all():
        return float("nan"), float("nan")
    th = np.concatenate([[0.0], np.cumsum((w[1:] + w[:-1]) * 0.5 / fps)])
    k = np.arange(len(th), dtype=np.float64) / (len(th) - 1)
    th = th - th[-1] * k
    return float(th.max() - th.min()), float(np.abs(w).max())


def segment_table(gyro, windows, fps):
    """gyro (N, 3) deg/s of one segment (already low-passed), windows [(i0, i1)] -> (rom (K,), peak (K,), axis,
    planarity): the axis from the samples of every window, then ROM and peak per window."""
    idx = np.concatenate([np.arange(a, b + 1) for a, b in windows if 0 <= a < b < len(gyro)]) if windows else \
        np.zeros(0, np.int64)
    axis, plan = sagittal_axis(gyro[idx]) if len(idx) else (None, float("nan"))
    if axis is None:
        nan = np.full(len(windows), np.nan)
        return nan, nan.copy(), None, plan
    r = gyro @ axis
    out = [stride_rom(r, a, b, fps) for a, b in windows]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), axis, plan


# ---------------------------------------------------------------- real strides
def load_sets():
    """(train54, locked): F1's sets through f2_H1shape.load_sets, and no person of the window store's lockbox list."""
    from mdfm.data import window_store as WS
    from mdfm.data.canonical import Refused
    sets, locked = _sibling("f2_H1shape").load_sets(F1_RESULTS)
    locked = set(locked) | set(WS.lockbox_persons())
    if set(sets["train54"]) & locked:
        raise Refused("train54: lockbox persons")
    return sorted(sets["train54"]), locked, sets


def guard(persons, locked, allowed, what):
    from mdfm.data.canonical import Refused
    ps = {str(p) for p in persons}
    if ps & set(locked):
        raise Refused(f"{what}: lockbox persons")
    if allowed is not None and ps - set(allowed):
        raise AssertionError(f"{what}: persons outside the allowed set")
    return ps


def real_strides(persons, locked, log=print):
    """DataFrame of the real strides (one row per walkway stride with every quantity of the module docstring)."""
    import pandas as pd
    from mdfm.gates import g3
    guard(persons, locked, persons, "real persons")
    demo = _sibling("f2_H1shape").read_demographics(list(persons), locked)
    tg = g3.load_targets(tuple(persons))
    if tg.counts.get("lockbox_rows"):
        raise AssertionError("lockbox rows in the targets")
    rows = []
    planar = []
    for tid, rec in g3.imu_records(tuple(persons), tg.trials):
        pid = str(rec.get("person_id"))
        guard([pid], locked, persons, tid)
        fps = float(rec["fps"])
        sig = np.asarray(rec["signal"], dtype=np.float64)
        loc = [int(x) for x in np.asarray(rec["location_ids"])]
        cm = np.asarray(rec["channel_mask"], dtype=bool)
        valid = np.asarray(rec["sample_valid"], dtype=bool) if rec.get("sample_valid") is not None \
            else np.ones(sig.shape[:2], bool)
        st = tg.strides[tg.strides["trial_id"] == tid]
        for foot in ("L", "R"):
            sf = st[st["foot"] == foot].sort_values("t0")
            if not len(sf):
                continue
            T = (sf["t1"] - sf["t0"]).to_numpy(np.float64)
            S = sf["length_m"].to_numpy(np.float64)
            keep = (T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (S >= S_MIN) & np.isfinite(T) & np.isfinite(S)
            sf = sf[keep]
            win = [(int(round(a * fps)), int(round(b * fps))) for a, b in zip(sf["t0"], sf["t1"])]
            seg = {}
            for name in SEGMENTS:
                site = REAL_SITES[name][foot]
                if site not in loc or not cm[loc.index(site), 3:6].all():
                    seg[name] = (np.full(len(win), np.nan), np.full(len(win), np.nan))
                    continue
                j = loc.index(site)
                g = sig[:, j, 3:6].copy()
                g[~valid[:, j]] = np.nan
                rom, peak, _, plan = segment_table(lowpass(g, fps), win, fps)
                seg[name] = (rom, peak)
                planar.append({"population": "real", "unit": tid, "foot": foot, "segment": name, "planarity": plan})
            h = demo.get(pid, {}).get("height_m", float("nan"))
            for k, r in enumerate(sf.itertuples(index=False)):
                row = {"person_id": pid, "unit": tid, "task": r.task, "foot": foot, "pass_id": r.pass_id,
                       "t0": float(r.t0), "t1": float(r.t1), "S": float(r.length_m), "stature_m": h}
                for name in SEGMENTS:
                    row[f"rom_{name}"] = seg[name][0][k]
                    row[f"peak_{name}"] = seg[name][1][k]
                rows.append(row)
    d = finish(pd.DataFrame(rows))
    d.attrs["planarity"] = planar
    d.attrs["counts"] = {"trials": int(tg.counts.get("trials", 0)), "walkway_strides": int(len(tg.strides)),
                         "persons_without_height": sorted(p for p in persons if not math.isfinite(
                             demo.get(p, {}).get("height_m", float("nan"))))}
    log(f"real: {len(d)} strides of {d['person_id'].nunique()} persons")
    return d


def finish(d):
    """Adds T, c, v, l to a stride table and keeps the rows with T, S and l finite and in bounds."""
    d = d.copy()
    d["T"] = d["t1"] - d["t0"]
    d["c"] = 120.0 / d["T"]
    d["v"] = d["S"] / d["T"]
    d["l"] = LEG_RATIO * d["stature_m"]
    ok = (d["T"] >= T_RANGE[0]) & (d["T"] <= T_RANGE[1]) & (d["S"] >= S_MIN) & np.isfinite(d["l"])
    return d[ok].reset_index(drop=True)


# ---------------------------------------------------------------- AMASS strides
def truth_rates(bodies, bank, n):
    """The true body of BodyWindows row 0 at its n knots, float64 on the CPU: joints (n, 22, 3) m, segment-frame
    angular velocities (n, 22, 3) deg/s (R^T omega), contacts (n, 2) bool (frames.contact_labels, exact velocities,
    floor y = 0), as eval_gait_insim.truth_on_clock with omega added."""
    import torch
    from mdfm.body import frames
    from mdfm.body.kinematics import KnotBatch
    f64 = torch.float64
    with torch.no_grad():
        rest = bank.rest(bodies.betas[:1].to(f64), bodies.gender[:1].to(torch.int64))
        kb = KnotBatch(bodies.rot6d[:1].to(f64), bodies.transl[:1].to(f64), rest, rate=RATE,
                       lengths=bodies.lengths[:1])
        t = torch.arange(n, dtype=f64) / RATE
        w = torch.zeros(n, dtype=torch.int64)
        P, V, Wb = [], [], []
        for a in range(0, n, 65536):
            kin = kb.kinematics(w[a:a + 65536], t[a:a + 65536])
            P.append(kin.p)
            V.append(kin.v)
            Wb.append(torch.einsum("njba,njb->nja", kin.R, kin.omega))
        J, V, Wb = torch.cat(P), torch.cat(V), torch.cat(Wb)
        contact = frames.contact_labels(J[None], RATE, vel=V[None], floor=0.0)[0]
    return J.numpy(), np.rad2deg(Wb.numpy()), contact.numpy()


def amass_takes(store, split):
    """Rows of store.takes that the check reads: train, task 'walk' and is_straight_forward; dev, the forward rows of
    eval_gait_insim.select_takes. Every archive is checked against its split (amass_index.check), CODY refused."""
    import pandas as pd
    from mdfm.data import amass_index
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    ins = _sibling("eval_gait_insim")
    tk = store.takes
    if set(tk["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused(f"a lockbox person in the {split} body store")
    if split == "dev":
        _, aw_dev = ins.amass_walks_rows()
        sel = ins.select_takes(tk, extra=aw_dev)
        sel = sel[sel["forward"]]
        idx = sel["i"].to_numpy(np.int64)
    else:
        m = (tk["task_name"] == "walk").to_numpy() & np.array([is_straight_forward(t) for t in tk["take_id"]])
        idx = np.flatnonzero(m)
    out = tk.iloc[idx].copy()
    out["i"] = idx
    for a in sorted(set(out["archive"])):
        if amass_index.check(a) != split:
            raise Refused(f"{a} is not a {split} archive")
    for r in out.itertuples():
        I.refuse_cody(str(r.archive), str(r.take_id), str(r.person_id), str(getattr(r, "subject_ns", "")))
    return pd.DataFrame(out).reset_index(drop=True)


def amass_strides(split, limit=None, log=print):
    """DataFrame of the scored AMASS strides of a split with every quantity of the module docstring."""
    import pandas as pd
    import torch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import body_store
    from mdfm.readouts import gait
    ins = _sibling("eval_gait_insim")
    h1 = _sibling("f2_H1shape")
    store = body_store.BodyStore(split=split)
    takes = amass_takes(store, split)
    if limit:
        takes = takes.iloc[:int(limit)]
    bank = SkeletonBank(torch.float64)
    rows, planar, stat = [], [], {}
    t_start = time.time()
    for k, tr in enumerate(takes.itertuples()):
        i = int(tr.i)
        n = int(tr.n_knots)
        bodies = store.windows([i], [0], win_s=n / RATE)
        J, Wb, contact = truth_rates(bodies, bank, n)
        ics = gait.initial_contacts(contact.astype(np.float64), RATE)
        st = gait.strides(J, ics, rate=RATE)
        if not len(st):
            continue
        pel = J[:, gait.PELVIS][:, [0, 2]]
        ps = ins.passes_from_ics(ics, pel)
        ins.steady_straight(st, ps, pel)
        g = int(store.gender[i])
        key = (tuple(np.round(store.betas[i], 6)), g)
        if key not in stat:
            s_, l_ = h1.stature_of(store.betas[i], g)
            stat[key] = (float(np.asarray(s_)), float(np.asarray(l_)))
        for foot in ("L", "R"):
            sf = st[(st["foot"] == foot) & st["scored"].astype(bool)].sort_values("t0")
            T = (sf["t1"] - sf["t0"]).to_numpy(np.float64)
            S = sf["length_m"].to_numpy(np.float64)
            sf = sf[(T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (S >= S_MIN)]
            if not len(sf):
                continue
            win = [(int(round(a * RATE)), int(round(b * RATE))) for a, b in zip(sf["t0"], sf["t1"])]
            seg = {}
            for name in SEGMENTS:
                gyro = lowpass(Wb[:, BODY_JOINTS[name][foot]], RATE)
                rom, peak, _, plan = segment_table(gyro, win, RATE)
                seg[name] = (rom, peak)
                planar.append({"population": f"amass_{split}", "unit": str(tr.take_id), "foot": foot,
                               "segment": name, "planarity": plan})
            for j, r in enumerate(sf.itertuples(index=False)):
                row = {"person_id": str(tr.person_id), "unit": str(tr.take_id), "archive": str(tr.archive),
                       "style": style_of(tr.take_id), "treadmill": int(tr.treadmill), "foot": foot,
                       "t0": float(r.t0), "t1": float(r.t1), "S": float(r.length_m), "stature_m": stat[key][0],
                       "mesh_leg_m": stat[key][1]}
                for name in SEGMENTS:
                    row[f"rom_{name}"] = seg[name][0][j]
                    row[f"peak_{name}"] = seg[name][1][j]
                rows.append(row)
        if (k + 1) % 100 == 0:
            log(f"  AMASS {split}: {k + 1} of {len(takes)} takes, {len(rows)} strides, {time.time() - t_start:.0f} s")
    d = finish(pd.DataFrame(rows))
    d.attrs["planarity"] = planar
    d.attrs["counts"] = {"takes": int(len(takes)), "takes_with_strides": int(d["unit"].nunique()) if len(d) else 0}
    log(f"AMASS {split}: {len(d)} strides of {d['person_id'].nunique()} persons, {d['unit'].nunique()} takes")
    return d


# ---------------------------------------------------------------- relations and readings
def design(d, kind="stride"):
    """Design matrix of a stride table: 'stride' (1, log(S / l), log c - log 110), 'speed' (1, log v, log l)."""
    if kind == "stride":
        return np.stack([np.ones(len(d)), np.log(d["S"].to_numpy() / d["l"].to_numpy()),
                         np.log(d["c"].to_numpy()) - C_CENTRE], 1)
    if kind == "speed":
        return np.stack([np.ones(len(d)), np.log(d["v"].to_numpy()), np.log(d["l"].to_numpy())], 1)
    raise ValueError(kind)


def blocks(pid, X, y):
    """{person: (X'WX, X'Wy, y'Wy)} with W = 1 / the person's row count (persons weighted equally)."""
    out = {}
    pid = np.asarray(pid)
    for p in np.unique(pid):
        m = pid == p
        w = 1.0 / m.sum()
        out[p] = (w * X[m].T @ X[m], w * X[m].T @ y[m], w * float(y[m] @ y[m]))
    return out


def solve(bl, persons):
    """(coefficients, residual variance: the mean over the listed persons of their mean squared residual)."""
    A = sum(bl[p][0] for p in persons)
    b = sum(bl[p][1] for p in persons)
    yy = sum(bl[p][2] for p in persons)
    beta = np.linalg.solve(A, b)
    sse = yy - 2 * beta @ b + beta @ A @ beta
    return beta, float(sse / len(persons))


class Relations:
    """Forward relations of the segments on an AMASS stride table (persons weighted equally), kind 'stride' or
    'speed', and the inverse relation log(S / l) on (1, log ROM_thigh, log ROM_shank, log ROM_foot, log c - log 110)."""

    def __init__(self, amass, kind="stride", segments=PRIMARY):
        self.kind, self.segments = kind, tuple(segments)
        self.blocks, self.persons = {}, {}
        X = design(amass, kind)
        for s in SEGMENTS:
            y = np.log(amass[f"rom_{s}"].to_numpy())
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            self.blocks[s] = blocks(amass["person_id"].to_numpy()[ok], X[ok], y[ok])
        Xi = np.column_stack([np.ones(len(amass))] + [np.log(amass[f"rom_{s}"].to_numpy()) for s in PRIMARY]
                             + [np.log(amass["c"].to_numpy()) - C_CENTRE])
        yi = np.log(amass["S"].to_numpy() / amass["l"].to_numpy())
        ok = np.isfinite(Xi).all(1) & np.isfinite(yi)
        self.blocks["inverse"] = blocks(amass["person_id"].to_numpy()[ok], Xi[ok], yi[ok])

    def fit(self, persons=None):
        """{segment: (beta, sigma2)} and 'inverse', on the given AMASS persons (with repetition) or all."""
        out = {}
        for k, bl in self.blocks.items():
            ps = sorted(bl) if persons is None else [p for p in persons if p in bl]
            out[k] = solve(bl, ps)
        return out


def residuals(real, fit, kind="stride"):
    """{segment: (N,) log ROM minus the relation} of a stride table."""
    X = design(real, kind)
    return {s: np.log(real[f"rom_{s}"].to_numpy()) - X @ fit[s][0] for s in SEGMENTS}


def deltas(real, fit, segments=PRIMARY):
    """(N,) delta of the module docstring over the given segments (NaN where one is missing)."""
    r = residuals(real, fit, "stride")
    num = np.zeros(len(real))
    den = np.zeros(len(real))
    for s in segments:
        b, s2 = fit[s][0][1], fit[s][1]
        num = num + (b / s2) * r[s]
        den = den + b * b / s2
    return num / den


def inverse_log_ratio(real, fit):
    """(N,) log(S_hat / S) of the inverse relation."""
    Xi = np.column_stack([np.ones(len(real))] + [np.log(real[f"rom_{s}"].to_numpy()) for s in PRIMARY]
                         + [np.log(real["c"].to_numpy()) - C_CENTRE])
    return Xi @ fit["inverse"][0] + np.log(real["l"].to_numpy()) - np.log(real["S"].to_numpy())


def person_median(pid, x):
    """{person: median of the finite values}."""
    out = {}
    pid = np.asarray(pid)
    x = np.asarray(x, dtype=np.float64)
    for p in np.unique(pid):
        v = x[(pid == p) & np.isfinite(x)]
        if len(v):
            out[p] = float(np.median(v))
    return out


def med_of_persons(pm, persons=None):
    keys = sorted(pm) if persons is None else [p for p in persons if p in pm]
    return float(np.median([pm[p] for p in keys])) if keys else float("nan")


def statistics(real, fit, segments=PRIMARY):
    """Point values on a stride table: gamma_L, gamma_s, gamma_R, kappa_s (exp of the median over persons)."""
    pid = real["person_id"].to_numpy()
    out = {"gamma_L": math.exp(med_of_persons(person_median(pid, deltas(real, fit, segments))))}
    r = residuals(real, fit, "stride")
    for s in SEGMENTS:
        out[f"kappa_{s}"] = math.exp(med_of_persons(person_median(pid, r[s])))
        out[f"gamma_{s}"] = math.exp(med_of_persons(person_median(pid, r[s] / fit[s][0][1])))
    out["gamma_R"] = math.exp(med_of_persons(person_median(pid, inverse_log_ratio(real, fit))))
    return out


class Boot:
    """Joint person bootstrap: AMASS persons (relations refitted) and the persons of a stride table."""

    def __init__(self, rel, real, kind="stride", segments=PRIMARY):
        self.rel, self.kind, self.segments = rel, kind, segments
        self.real = real
        self.pid = real["person_id"].to_numpy()
        self.persons = sorted(set(self.pid))
        self.rows = {p: np.flatnonzero(self.pid == p) for p in self.persons}
        self.X = design(real, kind)
        self.y = {s: np.log(real[f"rom_{s}"].to_numpy()) for s in SEGMENTS}
        self.Xi = np.column_stack([np.ones(len(real))] + [np.log(real[f"rom_{s}"].to_numpy()) for s in PRIMARY]
                                  + [np.log(real["c"].to_numpy()) - C_CENTRE])
        self.off = np.log(real["l"].to_numpy()) - np.log(real["S"].to_numpy())

    def values(self, fit):
        """Per-person medians of every quantity under a fit: {name: {person: value}}."""
        out = {}
        r = {s: self.y[s] - self.X @ fit[s][0] for s in SEGMENTS}
        if self.kind == "stride":
            num = sum((fit[s][0][1] / fit[s][1]) * r[s] for s in self.segments)
            den = sum(fit[s][0][1] ** 2 / fit[s][1] for s in self.segments)
            out["gamma_L"] = person_median(self.pid, num / den)
            for s in SEGMENTS:
                out[f"gamma_{s}"] = person_median(self.pid, r[s] / fit[s][0][1])
            out["gamma_R"] = person_median(self.pid, self.Xi @ fit["inverse"][0] + self.off)
        for s in SEGMENTS:
            out[f"kappa_{s}"] = person_median(self.pid, r[s])
        return out

    def run(self, n_boot=BOOT_N, seed=SEED):
        rng = np.random.default_rng(seed)
        am = sorted(set().union(*[set(b) for b in self.rel.blocks.values()]))
        draws = {}
        for _ in range(int(n_boot)):
            ap = [am[i] for i in rng.integers(0, len(am), len(am))]
            rp = [self.persons[i] for i in rng.integers(0, len(self.persons), len(self.persons))]
            try:
                fit = self.rel.fit(ap)
            except np.linalg.LinAlgError:
                continue
            vals = self.values(fit)
            for k, pm in vals.items():
                v = [pm[p] for p in rp if p in pm]
                if v:
                    draws.setdefault(k, []).append(math.exp(float(np.median(v))))
        a = (1 - LEVEL) / 2
        return {k: [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))] for k, v in draws.items()}


def reading(gamma, iv):
    """The registered reading of gamma_L (prediction.txt section 3)."""
    lo, hi = iv
    if gamma <= EXPLAINS_POINT and hi < EXPLAINS_UPPER:
        return "COUPLING EXPLAINS"
    if MATCHED_BAND[0] <= gamma <= MATCHED_BAND[1] and lo >= MATCHED_IV[0] and hi <= MATCHED_IV[1]:
        return "COUPLING MATCHED"
    if gamma >= LONG_POINT and lo > LONG_LOWER:
        return "COUPLING LONG"
    return "PARTLY"


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def decoded_gains(persons, locked, path=F1_EVENTS):
    """{person: median decoded over walkway stride} of F1's v1_B train54 decode (row base, matched strides); the person
    column is read and guarded first, rows of other persons dropped before a value column is converted."""
    import pandas as pd
    from mdfm.data.canonical import Refused
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    if set(who["person_id"].dropna()) & set(locked):
        raise Refused(f"{rel(path)}: lockbox persons")
    ev = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    ev = ev[ev["person_id"].isin(set(persons))]
    ev = ev[(ev["row"] == "base") & (ev["kind"] == "stride") & (ev["matched"].str.strip().str.lower() == "true")]
    ratio = pd.to_numeric(ev["prediction"]).to_numpy() / pd.to_numeric(ev["reference"]).to_numpy()
    return person_median(ev["person_id"].to_numpy(), ratio), {"rows": int(len(ev)), "sha256": sha256_file(path)}


def link(real, fit, gains, n_boot=BOOT_N, seed=SEED):
    """Pearson r over persons of log gamma_p and log g_p, person bootstrap with the relations fixed."""
    gp = person_median(real["person_id"].to_numpy(), deltas(real, fit))
    ps = sorted(set(gp) & set(gains))
    x = np.array([gp[p] for p in ps])
    y = np.log(np.array([gains[p] for p in ps]))
    r = pearson(x, y)
    rng = np.random.default_rng(seed + 1)
    bs = []
    for _ in range(int(n_boot)):
        k = rng.integers(0, len(ps), len(ps))
        v = pearson(x[k], y[k])
        if np.isfinite(v):
            bs.append(v)
    a = (1 - LEVEL) / 2
    slope = float(np.polyfit(x, y, 1)[0]) if len(ps) > 2 else float("nan")
    return {"r": r, "interval": [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))], "n_persons": len(ps),
            "slope_log_g_on_log_gamma": slope,
            "reading": "LINKED" if (np.isfinite(r) and r > 0 and np.quantile(bs, a) > 0) else "NOT LINKED"}


def describe(d, label):
    """Per population: persons, strides, medians of person medians of S, v, c, l, S/l, ROM, ROM/S, peaks."""
    pid = d["person_id"].to_numpy()
    out = {"population": label, "persons": int(d["person_id"].nunique()), "strides": int(len(d))}
    for col in ("S", "v", "c", "l", "T"):
        out[col] = med_of_persons(person_median(pid, d[col].to_numpy()))
    out["S_over_l"] = med_of_persons(person_median(pid, (d["S"] / d["l"]).to_numpy()))
    for s in SEGMENTS:
        out[f"rom_{s}"] = med_of_persons(person_median(pid, d[f"rom_{s}"].to_numpy()))
        out[f"rom_per_m_{s}"] = med_of_persons(person_median(pid, (d[f"rom_{s}"] / d["S"]).to_numpy()))
        out[f"rom_per_leg_{s}"] = med_of_persons(person_median(pid, (d[f"rom_{s}"] / (d["S"] / d["l"])).to_numpy()))
        out[f"peak_{s}"] = med_of_persons(person_median(pid, d[f"peak_{s}"].to_numpy()))
        out[f"finite_{s}"] = float(np.isfinite(d[f"rom_{s}"].to_numpy()).mean()) if len(d) else float("nan")
    return out


def speed_bins(d):
    """Per speed bin of the stride speed: persons, strides, median of person medians of ROM / S per segment."""
    out = []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        b = d[(d["v"] >= lo) & (d["v"] < hi)]
        row = {"bin": f"{lo:.1f}-{hi:.1f}", "persons": int(b["person_id"].nunique()), "strides": int(len(b))}
        if len(b):
            pid = b["person_id"].to_numpy()
            row["S_over_l"] = med_of_persons(person_median(pid, (b["S"] / b["l"]).to_numpy()))
            row["c"] = med_of_persons(person_median(pid, b["c"].to_numpy()))
            for s in SEGMENTS:
                row[f"rom_per_m_{s}"] = med_of_persons(person_median(pid, (b[f"rom_{s}"] / b["S"]).to_numpy()))
        out.append(row)
    return out


def complete(d, segments=PRIMARY):
    m = np.ones(len(d), bool)
    for s in segments:
        m &= np.isfinite(d[f"rom_{s}"].to_numpy()) & (d[f"rom_{s}"].to_numpy() > 0)
    return d[m].reset_index(drop=True)


def interval_row(rel_, real, n_boot, kind="stride", seed=SEED):
    """Point values and joint bootstrap intervals of a stride table against a Relations object."""
    fit = rel_.fit()
    if kind == "stride":
        point = statistics(real, fit)
    else:
        r = residuals(real, fit, "speed")
        point = {f"kappa_{s}": math.exp(med_of_persons(person_median(real["person_id"].to_numpy(), r[s])))
                 for s in SEGMENTS}
    iv = Boot(rel_, real, kind).run(n_boot, seed) if n_boot else {}
    return {"point": point, "interval": iv, "persons": int(real["person_id"].nunique()), "strides": int(len(real))}


def coefficients(fit):
    return {k: {"beta": [float(x) for x in v[0]], "sigma2": float(v[1])} for k, v in fit.items()}


# ---------------------------------------------------------------- main
def analyse(real, am_train, am_dev, gains, chain, n_boot=BOOT_N):
    """Every quantity of the module docstring on the three stride tables; returns the results dict."""
    res = {}
    rc, ac, dc = complete(real), complete(am_train), complete(am_dev)
    res["counts"] = {"real_strides": int(len(real)), "real_complete": int(len(rc)),
                     "real_persons_complete": int(rc["person_id"].nunique()),
                     "amass_train_strides": int(len(am_train)), "amass_train_complete": int(len(ac)),
                     "amass_train_persons": int(ac["person_id"].nunique()),
                     "amass_dev_strides": int(len(am_dev)), "amass_dev_complete": int(len(dc)),
                     "amass_dev_persons": int(dc["person_id"].nunique())}
    rel_all = Relations(ac)
    fit = rel_all.fit()
    res["relations"] = {"amass_train_stride": coefficients(fit)}
    # primary
    prim = interval_row(rel_all, rc, n_boot)
    g, iv = prim["point"]["gamma_L"], prim["interval"].get("gamma_L", [float("nan")] * 2)
    res["primary"] = {**prim, "reading": reading(g, iv)}
    Q = chain["train54_gain"] / chain["insim_gain"]
    res["size"] = {"Q": Q, "train54_gain": chain["train54_gain"], "insim_gain": chain["insim_gain"],
                   "h": math.log(g) / math.log(Q), "h_interval": sorted([math.log(x) / math.log(Q) for x in iv])}
    # method checks
    m1 = {}
    for half in (0, 1):
        fit_on = ac[[person_split(p) == half for p in ac["person_id"]]]
        read_on = ac[[person_split(p) != half for p in ac["person_id"]]].reset_index(drop=True)
        if fit_on["person_id"].nunique() < 3 or read_on["person_id"].nunique() < 3:
            m1[f"fit_half{half}"] = {"point": {"gamma_L": float("nan")}, "interval": {}, "persons": 0, "strides": 0,
                                     "note": "fewer than 3 persons in a half (smoke run)"}
            continue
        rh = Relations(fit_on.reset_index(drop=True))
        m1[f"fit_half{half}"] = interval_row(rh, read_on, n_boot // 4, seed=SEED + 10 + half)
    ok1 = all(M1_BAND[0] <= m1[k]["point"]["gamma_L"] <= M1_BAND[1] for k in m1)
    res["M1"] = {**m1, "band": M1_BAND, "passes": ok1}
    m2 = interval_row(rel_all, dc, n_boot // 2, seed=SEED + 20)
    res["M2"] = {**m2, "band": M2_BAND, "passes": bool(M2_BAND[0] <= m2["point"]["gamma_L"] <= M2_BAND[1])}
    res["method_valid"] = bool(ok1)
    # real over AMASS dev (secondary)
    res["real_over_dev_gamma_L"] = g / m2["point"]["gamma_L"]
    # link
    res["link"] = link(rc, fit, gains, n_boot)
    # secondary rows on the real strides
    sec = {}
    sec["speed_kappa"] = interval_row(Relations(ac, "speed"), rc, n_boot // 2, kind="speed", seed=SEED + 30)
    for task in ("SelfPace", "HurriedPace"):
        sub = rc[rc["task"] == task].reset_index(drop=True)
        sec[f"task_{task}"] = interval_row(rel_all, sub, 0)
    lo = {k: np.quantile(np.log(ac[k] / ac["l"]) if k == "S" else np.log(ac[k]), [0.025, 0.975]) for k in ("S", "c")}
    inside = rc[(np.log(rc["S"] / rc["l"]).between(*lo["S"])) & (np.log(rc["c"]).between(*lo["c"]))]
    sec["inside_amass_range"] = {**interval_row(rel_all, inside.reset_index(drop=True), n_boot // 4,
                                                seed=SEED + 40), "share": float(len(inside) / len(rc))}
    for name, m in (("overground", ac["treadmill"] == 0), ("treadmill", ac["treadmill"] != 0),
                    ("KIT", ac["archive"] == "KIT"), ("BMLrub", ac["archive"] == "BMLrub"),
                    ("BMLrub_overground", (ac["archive"] == "BMLrub") & (ac["treadmill"] == 0))):
        sub = ac[m.to_numpy()].reset_index(drop=True)
        if sub["person_id"].nunique() >= 5:
            sec[f"relation_{name}"] = interval_row(Relations(sub), rc, n_boot // 4, seed=SEED + 50)
            sec[f"relation_{name}"]["coefficients"] = coefficients(Relations(sub).fit())
    # AMASS subsets read through the pooled relation (content localisation)
    for name, m in (("overground", ac["treadmill"] == 0), ("treadmill", ac["treadmill"] != 0),
                    ("KIT", ac["archive"] == "KIT"), ("BMLrub_overground",
                                                     (ac["archive"] == "BMLrub") & (ac["treadmill"] == 0))):
        sub = ac[m.to_numpy()].reset_index(drop=True)
        if sub["person_id"].nunique() >= 3:
            sec[f"amass_{name}_through_pooled"] = interval_row(rel_all, sub, 0)
    over = ac[(ac["treadmill"] == 0).to_numpy()].reset_index(drop=True)
    tread = ac[(ac["treadmill"] != 0).to_numpy()].reset_index(drop=True)
    if over["person_id"].nunique() >= 5 and tread["person_id"].nunique() >= 3:
        sec["amass_treadmill_through_overground"] = interval_row(Relations(over), tread, 0)
    res["secondary"] = sec
    # descriptive
    res["describe"] = [describe(real, "real train54"), describe(am_train, "AMASS train straight forward"),
                       describe(am_train[am_train["treadmill"] == 0], "AMASS train overground"),
                       describe(am_train[am_train["treadmill"] != 0], "AMASS train treadmill"),
                       describe(am_train[am_train["archive"] == "KIT"], "AMASS train KIT"),
                       describe(am_dev, "AMASS dev forward")]
    res["speed_bins"] = {"real": speed_bins(real), "amass_train": speed_bins(am_train),
                         "amass_train_overground": speed_bins(am_train[am_train["treadmill"] == 0]),
                         "amass_dev": speed_bins(am_dev)}
    pid = rc["person_id"].to_numpy()
    gp = person_median(pid, deltas(rc, fit))
    lg = np.array(list(gp.values()))
    res["persons"] = {"gamma_p_quantiles": {str(q): float(math.exp(np.quantile(lg, q)))
                                            for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                      "between_person_sd_log_gamma": float(np.std(lg, ddof=1)),
                      "share_below_1": float((lg < 0).mean()),
                      "kappa_shank_p": {str(q): float(math.exp(np.quantile(
                          list(person_median(pid, residuals(rc, fit)["shank"]).values()), q)))
                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    return res


def planarity_summary(*tables):
    out = {}
    for d in tables:
        for r in d.attrs.get("planarity", []):
            out.setdefault((r["population"], r["segment"]), []).append(r["planarity"])
    return {f"{p}/{s}": float(np.nanmedian(v)) for (p, s), v in sorted(out.items())}


def chain_gains(path=CHAIN):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    return {"train54_gain": float(c["train54"]["v1_B"]["stride"]["gain"]),
            "insim_gain": float(c["insim"]["v1_B"]["stride_gain"]), "sha256": sha256_file(path)}


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def fiv(iv, nd=3):
    return "" if not iv else f"[{iv[0]:.{nd}f}, {iv[1]:.{nd}f}]"


def markdown(res):
    p, sz = res["primary"], res["size"]
    pt, iv = p["point"], p["interval"]
    L = ["# CPL: rotation per metre in the AMASS training walks against the training controls", "",
         "Dev diagnostic, not a gate read. Written by scripts/cpl_coupling.py; every number below is read from "
         "results.json of this folder. Prediction: reports/bench/cpl/prediction.txt, SHA-256 "
         f"{res['prediction_sha256']} (asserted before any value was read). No lockbox, EVAL, CODY, care_pd or dev "
         f"control record. MoveAtlas {res['moveatlas'].get('head', 'n/a')}.", "",
         f"## Reading: **{p['reading']}**; method check M1 {'passes' if res['M1']['passes'] else 'FAILS'}, "
         f"M2 {'passes' if res['M2']['passes'] else 'fails'}", "",
         f"- gamma_L (coupling-implied over walked stride, train54): {f(pt['gamma_L'])} {fiv(iv.get('gamma_L'))} over "
         f"{p['persons']} persons, {p['strides']} strides.",
         f"- Size: h = log gamma_L / log Q = {f(sz['h'], 2)} {fiv(sz['h_interval'], 2)}, Q = {f(sz['train54_gain'])} / "
         f"{f(sz['insim_gain'], 4)} = {f(sz['Q'], 4)} (the decoded train54 gain over the in-sim gain, v1_B).",
         f"- Real over AMASS dev gamma_L: {f(res['real_over_dev_gamma_L'])}.", "",
         "| quantity | real train54 | 95 % | AMASS dev (M2) | 95 % |", "|---|---|---|---|---|"]
    m2 = res["M2"]
    for k in ["gamma_L", "gamma_R"] + [f"gamma_{s}" for s in SEGMENTS] + [f"kappa_{s}" for s in SEGMENTS]:
        L.append(f"| {k} | {f(pt.get(k))} | {fiv(iv.get(k))} | {f(m2['point'].get(k))} | "
                 f"{fiv(m2['interval'].get(k))} |")
    sk = res["secondary"]["speed_kappa"]
    L += ["", "At equal stride speed and leg length (log ROM on 1, log v, log l): "
          + ", ".join(f"{s} {f(sk['point'][f'kappa_{s}'])} {fiv(sk['interval'].get(f'kappa_{s}'))}" for s in SEGMENTS)
          + ".", "", "## Method checks", ""]
    for k in ("fit_half0", "fit_half1"):
        r = res["M1"][k]
        L.append(f"- M1 {k}: gamma_L {f(r['point']['gamma_L'])} {fiv(r['interval'].get('gamma_L'))} over "
                 f"{r['persons']} AMASS persons (band {res['M1']['band']}).")
    L.append(f"- M2 AMASS dev forward: gamma_L {f(m2['point']['gamma_L'])} {fiv(m2['interval'].get('gamma_L'))} over "
             f"{m2['persons']} persons, {m2['strides']} strides (band {m2['band']}).")
    lk = res["link"]
    L += ["", f"## Link to the decoded gain: **{lk['reading']}**", "",
          f"Pearson r of log gamma_p and log g_p over {lk['n_persons']} persons: {f(lk['r'])} {fiv(lk['interval'])}; "
          f"slope {f(lk['slope_log_g_on_log_gamma'])}.", "", "## Populations (medians of person medians)", "",
          "| population | persons | strides | S m | v m/s | c steps/min | l m | S/l | ROM thigh | shank | ankle | foot "
          "| deg/m thigh | shank | ankle | foot |", "|---" * 16 + "|"]
    for d in res["describe"]:
        L.append(f"| {d['population']} | {d['persons']} | {d['strides']} | {f(d['S'])} | {f(d['v'])} | {f(d['c'], 1)} | "
                 f"{f(d['l'])} | {f(d['S_over_l'])} | " + " | ".join(f(d[f'rom_{s}'], 1) for s in SEGMENTS) + " | "
                 + " | ".join(f(d[f'rom_per_m_{s}'], 1) for s in SEGMENTS) + " |")
    L += ["", "## Rotation per metre by stride speed (deg/m, median of person medians; persons)", "",
          "| population | bin m/s | persons | strides | S/l | thigh | shank | ankle | foot |", "|---" * 9 + "|"]
    for pop, rows in res["speed_bins"].items():
        for r in rows:
            if r["strides"]:
                L.append(f"| {pop} | {r['bin']} | {r['persons']} | {r['strides']} | {f(r.get('S_over_l'))} | "
                         + " | ".join(f(r.get(f'rom_per_m_{s}'), 1) for s in SEGMENTS) + " |")
    L += ["", "## Secondary rows (no criterion)", "", "| row | gamma_L | 95 % | kappa thigh | shank | foot | persons | "
          "strides |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res["secondary"].items():
        if k == "speed_kappa":
            continue
        pt2, iv2 = r["point"], r["interval"]
        L.append(f"| {k} | {f(pt2.get('gamma_L'))} | {fiv(iv2.get('gamma_L'))} | {f(pt2.get('kappa_thigh'))} | "
                 f"{f(pt2.get('kappa_shank'))} | {f(pt2.get('kappa_foot'))} | {r['persons']} | {r['strides']} |")
    pr = res["persons"]
    L += ["", f"Persons (train54): gamma_p quantiles {pr['gamma_p_quantiles']}; between-person SD of log gamma_p "
          f"{f(pr['between_person_sd_log_gamma'])}; share below 1 {f(pr['share_below_1'], 2)}.", "",
          f"Planarity (median share of the gyro second moment on the sagittal axis): {res['planarity']}.", "",
          f"Counts: {res['counts']}. Run {res['run_started']}, {res['timing_s']:.0f} s."]
    text = "\n".join(L)
    assert not any(ch in text for ch in "\u2013\u2014")
    return text


def write_json(path, obj):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(clean(obj), fh, indent=1, ensure_ascii=True, allow_nan=False)
        fh.write("\n")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-amass", type=int, default=None, help="smoke run: first N takes of each split")
    ap.add_argument("--boot", type=int, default=BOOT_N)
    a = ap.parse_args(argv)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    pred = check_prediction()
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    stamp = _sibling("moveatlas_stamp").stamp()
    os.makedirs(a.out, exist_ok=True)
    train54, locked, sets = load_sets()
    real = real_strides(train54, locked)
    am_train = amass_strides("train", a.limit_amass)
    am_dev = amass_strides("dev", a.limit_amass)
    gains, gsrc = decoded_gains(train54, locked)
    chain = chain_gains()
    res = {"what": LABEL, "task": TASK, "prediction_sha256": pred, "run_started": started,
           "moveatlas": stamp, "sets": {"train54_n": len(train54)},
           "inputs": {"f1_events": {"path": rel(F1_EVENTS), **gsrc}, "chain": {"path": rel(CHAIN), **chain}},
           "real_counts": real.attrs.get("counts"), "amass_counts": {"train": am_train.attrs.get("counts"),
                                                                      "dev": am_dev.attrs.get("counts")},
           "limit_amass": a.limit_amass}
    res.update(analyse(real, am_train, am_dev, gains, chain, a.boot))
    res["planarity"] = planarity_summary(real, am_train, am_dev)
    res["timing_s"] = time.time() - t0
    write_json(os.path.join(a.out, "results.json"), res)
    for name, d in (("real_strides.csv", real), ("amass_train_strides.csv", am_train),
                    ("amass_dev_strides.csv", am_dev)):
        d.to_csv(os.path.join(a.out, name), index=False, encoding="utf-8", lineterminator="\n")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(res) + "\n")
    import torch
    write_json(os.path.join(a.out, "_provenance.json"), {
        "script": rel(os.path.abspath(__file__)), "script_sha256_at_start": SCRIPT_SHA256_AT_START,
        "script_sha256_at_write": sha256_file(os.path.abspath(__file__)), "prediction_sha256": pred,
        "moveatlas_stamp": stamp, "run_started": started, "label": LABEL, "task": TASK, "torch": torch.__version__,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                       "PYTHONDONTWRITEBYTECODE")},
        "argv": sys.argv[1:]})
    print(res["primary"]["reading"], json.dumps({k: round(v, 4) for k, v in res["primary"]["point"].items()}),
          f"{res['timing_s']:.0f} s")


if __name__ == "__main__":
    main()
