"""Sensor frames of the 13 weargait_pd Xsens IMUs on control participants (plan task B2a, gate G2).

Prepares the virtual IMU factory (plan task B2): how the 13 Xsens sensors of weargait_pd sit
on the body, so that rendered sensors can imitate them. Controls only; PD participants are
counted from the folder listing and the CSV headers, their signals are never read.

Reads, read-only, under <moveatlas>/datasets/imu/weargait_pd/raw/:
  Version 1/CONTROL PARTICIPANTS/CSV files/<id>_Balance.csv and <id>_SelfPace.csv (signals),
  every other per-trial CSV of both groups (header line only), the PKMAS walkway metrics,
  the control demographic table (height), Code/Quality Control/QcAllData.m (the gyroscope
  channel the source's own QC viewer shows per site). The site map (raw sensor prefix to the
  canonical site) is parsed as text from build_canonical_imu.py (WEARGAIT_LOCMAP); no
  MoveAtlas code is imported or run. Also reads reports/lockbox/lockbox_persons.csv of this tree.

Writes <out> (default reports/weargait_frames/): raw_layout.csv, raw_files.csv,
site_gravity.csv, site_gyro_axis.csv, site_summary.csv, walk_speed_cadence.csv,
summary.json, summary.md, _provenance.json.

Definitions
- Row 0 of every trial is dropped: its VelInc and OriInc hold the increment accumulated
  since the start of the recording (checked and counted, see first_row_artifact).
- A sample of a site is valid when its six Acc and Gyr values are finite and the specific
  force norm exceeds 1 m/s2 (a zero-filled dropout would read 0).
- up: unit vector of the mean specific force (Acc columns) over a window, in the sensor
  frame. At rest it points vertically up; the gravity direction is -up.
- Balance quiet window: the 5 s window (500 samples, starts every 0.1 s) with the lowest
  summed gyroscope energy (mean squared angular-rate norm) over the sites of the trial, among
  the windows where every one of those sites is valid (whole-body quiet window, one posture
  for all sites). When no such window exists, the site with the fewest valid windows leaves
  the common set, and each site outside it uses its own quietest window (window_kind site).
  Each site also gets its own quietest window as a sensitivity check (own_window_angle_deg).
  A second whole-body window restricted to the samples labelled EO_FeetShoWidth or
  EC_FeetShoWidth measures how much the stance condition moves the estimate
  (sw_window_angle_deg). It is 0 by construction when the whole-body window already lies in
  a shoulder-width condition, so the summaries also report it over the moved windows only
  (sw_window_start_s differs from window_start_s).
- static_ok: mean specific force norm within 0.5 m/s2 of 9.81, resultant length of the unit
  specific-force vectors in the window at least 0.998 (direction steady within about 3.6 deg
  RMS) and gyroscope RMS in the window at most 10 deg/s. Only static_ok rows enter the site
  summaries.
- Gyroscope principal axis (SelfPace): leading eigenvector of the uncentred second-moment
  matrix of the angular-rate vectors over the samples labelled Walk in GeneralEvent (all
  valid samples when the trial carries fewer than 200 Walk samples, walk_source all), at
  least 200 samples. The sign is set so that the largest component is positive (the sign is
  not physical here). Angles between an axis and up are folded to [0, 90] degrees.
- Swing axis (a sign for the walking axis): s = angular rate projected on the axis; when the
  99th percentile of s exceeds 1.25 times the magnitude of the 1st percentile, the larger
  peak is on the + side (swing_sign +1); the reverse gives -1; otherwise 0 (ambiguous).
  swing_axis is the signed sensor axis of swing_sign times the axis. This peak rule is a
  heuristic and is not used to name body axes.
- Body right (swing-phase cue, thigh, shank, ankle and foot sensors): mean angular rate
  projected on the walking axis over the Walk samples where the sensor's own foot is off the
  walkway and the other foot is on it (the sensor's leg is in swing), at least 50 samples.
  In swing the thigh and shank rotate forward (distal end forward) and the foot rotates from
  toe-down at toe-off to toe-up at heel strike; both are positive rotations about the body's
  rightward axis. swing_phase_sign is the sign of that mean; body_right_axis is the signed
  sensor axis of swing_phase_sign times the walking axis. peak_phase_agree compares it with
  the peak rule. The other sites are left uninterpreted.
- Site mean of up: normalised vector mean; spread: median (and 90th percentile) angle of the
  participants to it. Site mean of the gyroscope axis: axial mean (leading eigenvector of the
  sum of outer products); spread: median folded angle to it.
- Sets: all (every control with data), fit (controls outside the weargait_pd lockbox of
  reports/lockbox/lockbox_persons.csv, the set a renderer should be calibrated on), HC and
  WHC (controls by ID prefix, the two recording cohorts). The 3 NLS controls enter all and
  fit only.
- Cadence cross-check: walkway contact onsets (L and R Foot Contact columns of the SelfPace
  CSV, an onset counts after at least 50 ms without contact of that foot), merged and split
  into passes at gaps above 1.5 s; cadence = 60 * sum(onsets - 1) / sum(pass durations)
  over passes with at least 3 onsets.
"""
import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import parse_args, write_provenance  # noqa: E402

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--limit", type=int, default=0, help="first N control participants only (test runs)")
OPT, rest = pre.parse_known_args()
if "--out" not in rest:
    rest += ["--out", os.path.join(os.path.dirname(HERE), "reports", "weargait_frames")]
sys.argv = [sys.argv[0]] + rest
ARGS = parse_args(__doc__.splitlines()[0])
OUT = ARGS.out
T0 = time.time()

RAW = f"{ARGS.moveatlas}/datasets/imu/weargait_pd/raw"
V1 = f"{RAW}/Version 1"
GROUP_DIRS = {"control": f"{V1}/CONTROL PARTICIPANTS/CSV files", "pd": f"{V1}/PD PARTICIPANTS/CSV files"}
MAT_DIRS = {"control": f"{V1}/CONTROL PARTICIPANTS/MAT files", "pd": f"{V1}/PD PARTICIPANTS/MAT file"}
PKMAS = f"{V1}/Walkway-derived metrics/PKMAS Walkway Gait Metrics - HP+SP.csv"
DEMO = {"control": f"{V1}/CONTROLS - Demographic+Clinical - datasetV1.csv",
        "pd": f"{V1}/PD - Demographic+Clinical - datasetV1.csv"}
QC_M = f"{RAW}/Code/Quality Control/QcAllData.m"
DATACARD = f"{ARGS.moveatlas}/datasets/imu/weargait_pd/datacard.md"
BUILD_IMU = f"{ARGS.moveatlas}/build_canonical_imu.py"
LOCKBOX = os.path.join(os.path.dirname(HERE), "reports", "lockbox", "lockbox_persons.csv")

CANON_ORDER = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R",
               "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R"]
IMU_SUFFIXES = ["Acc_X", "Acc_Y", "Acc_Z", "FreeAcc_E", "FreeAcc_N", "FreeAcc_U", "Gyr_X", "Gyr_Y", "Gyr_Z",
                "Mag_X", "Mag_Y", "Mag_Z", "VelInc_X", "VelInc_Y", "VelInc_Z",
                "OriInc_q0", "OriInc_q1", "OriInc_q2", "OriInc_q3", "Roll", "Pitch", "Yaw"]
W = 500            # 5 s at 100 Hz
STEP = 10          # window start every 0.1 s
G0 = 9.81
STATIC_ACC_TOL = 0.5     # m/s2
STATIC_RESULTANT_MIN = 0.998
STATIC_GYRO_MAX = 10.0   # deg/s
SW_LABELS = ("EO_FeetShoWidth", "EC_FeetShoWidth")
SWING_RATIO = 1.25
WEAK_AXIS = 0.6       # median energy share under which a site has no dominant walking axis
SWING_SITES = ("thigh_L", "thigh_R", "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R")
MIN_WALK = 200
MIN_PHASE = 50        # swing-phase samples needed for the body-right sign
OUTLIER_DEG = 45.0
CONTACT_MIN_OFF = 5      # samples (50 ms)
HEEL_SAMPLES = 5         # first 50 ms of a contact locate the footfall on the walkway
PASS_GAP_S = 1.5
RAD2DEG = 180.0 / np.pi


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- sources of the site map
def read_locmap():
    """WEARGAIT_LOCMAP parsed as text from the MoveAtlas ingestion script (not imported)."""
    with open(BUILD_IMU, encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"WEARGAIT_LOCMAP\s*=\s*(\{.*?\})", txt, re.S)
    if not m:
        sys.exit("WEARGAIT_LOCMAP not found in build_canonical_imu.py")
    locmap = ast.literal_eval(m.group(1))
    inv = {v: k for k, v in locmap.items()}
    if sorted(inv) != sorted(CANON_ORDER):
        sys.exit(f"site map differs from the 13 canonical sites: {sorted(inv)}")
    return [(s, inv[s]) for s in CANON_ORDER]


def read_qc_channels():
    """Gyroscope channel per raw sensor shown by the source's QC viewer (QcAllData.m)."""
    with open(QC_M, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    out = {}
    for pre_, ax in re.findall(r"inputData\.(\w+?)_Gyr_([XYZ])", txt):
        out.setdefault(pre_, ax)
    return out


def read_datacard_statements():
    """The datacard lines that state the signal content and the known unit issues (read as text)."""
    out = {}
    if not os.path.isfile(DATACARD):
        return out
    with open(DATACARD, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^- (signal|known issues[^:]*):\s*(.*)$", line.rstrip("\n"))
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


SITES = read_locmap()
PREFIX = dict(SITES)
QC_AXIS = read_qc_channels()


def qc_axis_for(prefix):
    if prefix in QC_AXIS:
        return QC_AXIS[prefix]
    base = re.sub(r"^[LR]_", "", prefix)
    for k, v in QC_AXIS.items():
        if re.sub(r"^[LR]_", "", k) == base:
            return v
    return ""


# ---------------------------------------------------------------- small vector helpers
def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.full(3, np.nan)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def axang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def axis_token(v, signed=True):
    i = int(np.argmax(np.abs(v)))
    tok = "XYZ"[i]
    if signed:
        tok = ("+" if v[i] >= 0 else "-") + tok
    return tok, float(abs(v[i]))


def sign_fix(v):
    i = int(np.argmax(np.abs(v)))
    return v if v[i] >= 0 else -v


def axial_mean(V):
    M = np.zeros((3, 3))
    for v in V:
        M += np.outer(v, v)
    w, U = np.linalg.eigh(M)
    return sign_fix(U[:, -1]), float(w[-1] / w.sum())


def up_from_euler(roll_deg, pitch_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    return np.stack([-np.sin(p), np.sin(r) * np.cos(p), np.cos(r) * np.cos(p)], 1)


def q(vals, p):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


# ---------------------------------------------------------------- file listing and headers
ID_RE = re.compile(r"^((?:WHC|HC|NLS|WPD)\d+(?: \(control\))?)(?:_(.*))?$")


def participant_of(fname):
    stem = os.path.splitext(fname)[0]
    m = ID_RE.match(stem)
    if m:
        return m.group(1), m.group(2) or ""
    parts = stem.split("_")
    return parts[0], "_".join(parts[1:])


def count_all_folders():
    """Participants per group across every CSV and MAT folder of Version 1 (listing only)."""
    per = defaultdict(set)
    union = defaultdict(set)
    for root, dirs, fs in os.walk(V1):
        dirs[:] = sorted(d for d in dirs if not d.startswith("._"))
        base = os.path.basename(root)
        if base not in ("CSV files", "MAT files", "MAT file"):
            continue
        rel = "/" + os.path.relpath(root, V1).replace("\\", "/")
        group = "control" if "CONTROL PARTICIPANTS" in rel else ("pd" if "PD PARTICIPANTS" in rel else "other")
        context = rel.split("/CONTROL PARTICIPANTS")[0].split("/PD PARTICIPANTS")[0].strip("/") or "main"
        kind = "csv" if base == "CSV files" else "mat"
        for f in fs:
            if f.startswith("._"):
                continue
            if (kind == "csv" and f.endswith(".csv") and f != "manifest.csv") or (kind == "mat" and f.endswith(".mat")):
                pid = participant_of(f)[0]
            else:
                continue
            pid = base_id(pid)
            per[f"{context}|{group}|{kind}"].add(pid)
            union[group].add(pid)
    return {"per_folder": {k: len(v) for k, v in sorted(per.items())},
            "union_by_group": {k: len(v) for k, v in sorted(union.items())},
            "ids_only_outside_main_csv": {g: sorted(union[g] - per.get(f"main|{g}|csv", set())) for g in sorted(union)}}


def list_trials():
    rows = []
    for group, d in GROUP_DIRS.items():
        for f in sorted(os.listdir(d)):
            if f.startswith("._") or not f.endswith(".csv") or f == "manifest.csv":
                continue
            pid, trial = participant_of(f)
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                header = next(csv.reader([fh.readline()]))
            hs = set(header)
            present = [s for s, p in SITES if all(f"{p}_{k}" in hs for k in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"))]
            full22 = [s for s, p in SITES if all(f"{p}_{k}" in hs for k in IMU_SUFFIXES)]
            rows.append({"group": group, "participant": pid, "trial": trial, "file": f,
                         "n_columns": len(header), "n_imu_sites": len(present),
                         "n_imu_sites_all_22_columns": len(full22),
                         "missing_sites": ";".join(s for s, _ in SITES if s not in present),
                         "has_insole_cells": int("LPressure1" in hs or "RPressure1" in hs),
                         "has_insole_imu": int("Linsole:Acc_X" in hs or "Rinsole:Acc_X" in hs),
                         "has_walkway": int("Walkway_X" in hs),
                         "has_events": int("GeneralEvent" in hs),
                         "bytes": os.path.getsize(os.path.join(d, f)), "_header": header})
    return rows


# ---------------------------------------------------------------- trial loading
WANTED_SUF = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z",
              "VelInc_X", "VelInc_Y", "VelInc_Z", "OriInc_q0", "OriInc_q1", "OriInc_q2", "OriInc_q3",
              "Roll", "Pitch", "Yaw", "FreeAcc_U"]


def num(df, cols):
    if not all(c in df.columns for c in cols):
        return None
    return np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols], 1)


def load_trial(path, header):
    want = {"Time", "GeneralEvent", "L Foot Contact", "R Foot Contact", "Walkway_X", "WalkwayFoot"}
    for _, p in SITES:
        want |= {f"{p}_{s}" for s in WANTED_SUF}
    use = [c for c in header if c in want]
    df = pd.read_csv(path, usecols=use, encoding="utf-8", low_memory=False)
    t = pd.to_numeric(df["Time"].astype(str).str.replace("sec", "", regex=False).str.strip(), errors="coerce").to_numpy(float)
    ev = df["GeneralEvent"].astype(str).str.strip().to_numpy() if "GeneralEvent" in df.columns else np.array([""] * len(df))
    trial = {"t_all": t, "n_rows": len(df), "sites": {}, "first_row_artifact": 0}
    art = False
    for s, p in SITES:
        A = num(df, [f"{p}_Acc_{a}" for a in "XYZ"])
        G = num(df, [f"{p}_Gyr_{a}" for a in "XYZ"])
        if A is None or G is None:
            continue
        Q = num(df, [f"{p}_OriInc_q{i}" for i in range(4)])
        V = num(df, [f"{p}_VelInc_{a}" for a in "XYZ"])
        if Q is not None and len(Q) and np.isfinite(Q[0, 0]) and abs(Q[0, 0] - 1.0) > 1e-3:
            art = True
        if V is not None and len(V) and np.all(np.isfinite(V[0])) and np.linalg.norm(V[0]) > 1.0:
            art = True
        d = {"A": A[1:], "G": G[1:], "Q": None if Q is None else Q[1:], "V": None if V is None else V[1:]}
        M = num(df, [f"{p}_Mag_{a}" for a in "XYZ"])
        E = num(df, [f"{p}_{a}" for a in ("Roll", "Pitch", "Yaw")])
        FU = num(df, [f"{p}_FreeAcc_U"])
        d["M"] = None if M is None else M[1:]
        d["E"] = None if E is None else E[1:]
        d["FU"] = None if FU is None else FU[1:, 0]
        an = np.linalg.norm(d["A"], axis=1)
        fin = np.all(np.isfinite(d["A"]), 1) & np.all(np.isfinite(d["G"]), 1)
        d["valid"] = fin & (an > 1.0)
        d["n_nan"] = int((~fin).sum())
        d["n_zero"] = int((fin & (an <= 1.0)).sum())
        d["e"] = np.where(d["valid"], np.sum(np.nan_to_num(d["G"]) ** 2, 1), 0.0)
        trial["sites"][s] = d
    trial["first_row_artifact"] = int(art)
    trial["t"] = t[1:]
    trial["ev"] = ev[1:]
    trial["contacts"] = {}
    for side in ("L", "R"):
        c = f"{side} Foot Contact"
        if c in df.columns:
            trial["contacts"][side] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)[1:]
    for c, k in (("Walkway_X", "ww_x"), ("WalkwayFoot", "ww_foot")):
        trial[k] = df[c].astype(str).to_numpy()[1:] if c in df.columns else None
    dt = np.diff(t[np.isfinite(t)])
    trial["dt_median"] = float(np.median(dt)) if len(dt) else float("nan")
    trial["dt_irregular"] = int(np.sum(np.abs(dt - 0.01) > 1e-6)) if len(dt) else -1
    trial["dt_max"] = float(np.max(dt)) if len(dt) else float("nan")
    return trial


# ---------------------------------------------------------------- quiet windows
def window_stats(d, n, allowed=None):
    if n < W:
        return None
    ee = d["e"]
    ok = d["valid"] if allowed is None else (d["valid"] & allowed)
    bad = (~ok).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(ee)])
    cb = np.concatenate([[0], np.cumsum(bad)])
    starts = np.arange(0, n - W + 1, STEP)
    return starts, (cs[starts + W] - cs[starts]) / W, cb[starts + W] - cb[starts]


def quiet_windows(trial, allowed=None):
    n = len(trial["t"])
    stats = {}
    for s, d in trial["sites"].items():
        st = window_stats(d, n, allowed)
        if st is not None and (st[2] == 0).any():
            stats[s] = st
    own = {}
    for s, (starts, E, B) in stats.items():
        ok = np.flatnonzero(B == 0)
        own[s] = int(starts[ok[np.argmin(E[ok])]])
    common = sorted(stats, key=CANON_ORDER.index)
    choice = None
    while common:
        starts = stats[common[0]][0]
        tot = np.sum([stats[s][1] for s in common], axis=0)
        ok = np.all([stats[s][2] == 0 for s in common], axis=0)
        if ok.any():
            idx = np.flatnonzero(ok)
            choice = int(starts[idx[np.argmin(tot[idx])]])
            break
        common.remove(min(common, key=lambda s: (int((stats[s][2] == 0).sum()), CANON_ORDER.index(s))))
    return choice, set(common) if choice is not None else set(), own


def estimate(d, i0):
    A = d["A"][i0:i0 + W]
    G = d["G"][i0:i0 + W]
    fbar = A.mean(0)
    nrm = float(np.linalg.norm(fbar))
    up = fbar / nrm
    out = {"acc_norm_ms2": nrm, "up": up,
           "gyro_rms_dps": float(np.sqrt(np.mean(np.sum(G ** 2, 1)))) * RAD2DEG,
           "acc_dir_resultant": float(np.linalg.norm(np.mean(A / np.linalg.norm(A, axis=1, keepdims=True), 0))),
           "euler_up_angle_deg": float("nan"), "freeacc_u_mean": float("nan")}
    if d["E"] is not None:
        E = d["E"][i0:i0 + W]
        if np.all(np.isfinite(E[:, :2])):
            out["euler_up_angle_deg"] = ang(unit(up_from_euler(E[:, 0], E[:, 1]).mean(0)), up)
    if d["FU"] is not None:
        fu = d["FU"][i0:i0 + W]
        if np.all(np.isfinite(fu)):
            out["freeacc_u_mean"] = float(fu.mean())
    return out


def unit_ratios(d, dt):
    """Least-squares ratio of the OriInc vector part (2 q / dt) to Gyr, and of VelInc to Acc dt."""
    res = {"gyro_unit_ratio": float("nan"), "acc_unit_ratio": float("nan"), "mag_norm_median": float("nan")}
    v = d["valid"]
    if d["Q"] is not None:
        m = v & np.all(np.isfinite(d["Q"]), 1)
        if m.sum() > 100:
            qv = 2.0 * d["Q"][m, 1:] / dt
            g = d["G"][m]
            den = float(np.sum(g * g))
            if den > 0:
                res["gyro_unit_ratio"] = float(np.sum(qv * g) / den)
    if d["V"] is not None:
        m = v & np.all(np.isfinite(d["V"]), 1)
        if m.sum() > 100:
            a = d["A"][m] * dt
            res["acc_unit_ratio"] = float(np.sum(d["V"][m] * a) / np.sum(a * a))
    if d["M"] is not None:
        m = v & np.all(np.isfinite(d["M"]), 1)
        if m.sum() > 100:
            res["mag_norm_median"] = float(np.median(np.linalg.norm(d["M"][m], axis=1)))
    return res


# ---------------------------------------------------------------- contacts and cadence
def contact_onsets(c):
    b = (np.nan_to_num(c) > 0).astype(np.int8)
    dd = np.diff(b)
    on = np.flatnonzero(dd == 1) + 1
    off = np.flatnonzero(dd == -1) + 1
    keep = []
    for o in on:
        prev = off[off < o]
        if len(prev) == 0 or o - prev[-1] >= CONTACT_MIN_OFF:
            keep.append(o)
    if b[0] == 1:
        keep = [0] + keep
    return np.array(keep, int)


def contact_cadence(trial):
    if len(trial["contacts"]) < 2:
        return {"n_contact_onsets": 0, "n_passes": 0, "cadence_contacts_steps_min": float("nan")}
    t = trial["t"]
    ons = np.sort(np.concatenate([t[contact_onsets(c)] for c in trial["contacts"].values()]))
    ons = ons[np.isfinite(ons)]
    if len(ons) < 3:
        return {"n_contact_onsets": int(len(ons)), "n_passes": 0, "cadence_contacts_steps_min": float("nan")}
    cuts = np.flatnonzero(np.diff(ons) > PASS_GAP_S) + 1
    steps = 0
    dur = 0.0
    npass = 0
    for seg in np.split(ons, cuts):
        if len(seg) >= 3 and seg[-1] > seg[0]:
            steps += len(seg) - 1
            dur += seg[-1] - seg[0]
            npass += 1
    return {"n_contact_onsets": int(len(ons)), "n_passes": npass,
            "cadence_contacts_steps_min": 60.0 * steps / dur if dur > 0 else float("nan")}


def walkway_gait(trial):
    """Footfall positions along the walkway X axis (cell units) at each contact onset: centroid
    of the cells carrying that foot's label over the first HEEL_SAMPLES of the contact. Passes as
    for the cadence; speed = sum |x_last - x_first| / sum durations; step = |dx| of consecutive
    footfalls inside a pass."""
    out = {"n_footfalls_ww": 0, "n_passes_ww": 0, "speed_cells_s": float("nan"), "step_cells_mean": float("nan")}
    if trial.get("ww_x") is None or trial.get("ww_foot") is None or len(trial["contacts"]) < 2:
        return out
    t = trial["t"]
    ff = []
    for side, c in trial["contacts"].items():
        for i in contact_onsets(c):
            xs = []
            for j in range(i, min(i + HEEL_SAMPLES, len(t))):
                xv, fv = trial["ww_x"][j], trial["ww_foot"][j]
                if xv == "nan" or fv == "nan":
                    continue
                xl, fl = xv.split("|"), fv.split("|")
                if len(xl) != len(fl):
                    continue
                for a, b in zip(xl, fl):
                    if b.strip() == side:
                        try:
                            xs.append(float(a))
                        except ValueError:
                            pass
            if xs and np.isfinite(t[i]):
                ff.append((t[i], float(np.mean(xs))))
    if len(ff) < 3:
        out["n_footfalls_ww"] = len(ff)
        return out
    ff.sort()
    tt = np.array([a for a, _ in ff])
    xx = np.array([b for _, b in ff])
    cuts = np.flatnonzero(np.diff(tt) > PASS_GAP_S) + 1
    dist = dur = 0.0
    steps = []
    npass = 0
    for ts, xs in zip(np.split(tt, cuts), np.split(xx, cuts)):
        if len(ts) >= 3 and ts[-1] > ts[0]:
            dist += abs(xs[-1] - xs[0])
            dur += ts[-1] - ts[0]
            steps += list(np.abs(np.diff(xs)))
            npass += 1
    out.update({"n_footfalls_ww": len(ff), "n_passes_ww": npass,
                "speed_cells_s": dist / dur if dur > 0 else float("nan"),
                "step_cells_mean": float(np.mean(steps)) if steps else float("nan")})
    return out


# ---------------------------------------------------------------- side tables
def read_lockbox():
    out = {}
    if not os.path.isfile(LOCKBOX):
        return out
    with open(LOCKBOX, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "weargait_pd":
                out[r["person_id"].split(":", 1)[1]] = r["group"]
    return out


def base_id(pid):
    return re.sub(r"\s*\(control\)\s*$", "", pid).strip()


def cohort_of(pid):
    m = re.match(r"^(WHC|HC|NLS|WPD)", pid)
    return m.group(1) if m else "other"


def read_demographics(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Subject ID")
    h = [c.strip() for c in rows[hi]]
    out = {}
    for r in rows[hi + 1:]:
        if r and r[0].strip():
            out[r[0].strip()] = dict(zip(h, [c.strip() for c in r]))
    return out


def read_pkmas():
    with open(PKMAS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    h0, h1 = rows[0], rows[1]

    def col(name, sub=""):
        for i, (a, b) in enumerate(zip(h0, h1)):
            if a.strip() == name and b.strip() == sub:
                return i
        raise KeyError((name, sub))

    cols = {"velocity_cm_s": col("Velocity (cm./sec.)"), "cadence_steps_min": col("Cadence (steps/min.)"),
            "ambulation_time_s": col("Ambulation Time (sec.)"),
            "n_steps": col("Step Length (cm.)", "#Samples"), "step_length_cm": col("Step Length (cm.)", "Mean"),
            "stride_length_cm": col("Stride Length (cm.)", "Mean"), "step_time_s": col("Step Time (sec.)", "Mean"),
            "stride_width_cm": col("Stride Width (cm.)", "Mean")}
    out = defaultdict(list)
    for r in rows[2:]:
        if len(r) < 3 or not r[1].strip():
            continue
        rec = {"task_raw": r[0], "task": r[0].strip(), "participant": r[1].strip(), "pd_vs_control": r[2].strip()}
        for k, i in cols.items():
            try:
                rec[k] = float(r[i]) if i < len(r) and r[i].strip() != "" else float("nan")
            except ValueError:
                rec[k] = float("nan")
        out[(rec["task"], rec["participant"])].append(rec)
    return out, len(rows) - 2


# ---------------------------------------------------------------- main
def main():
    log("MoveAtlas:", ARGS.moveatlas, "out:", OUT)
    log("site map:", SITES)
    log("QC viewer gyroscope channels:", QC_AXIS)
    lockbox = read_lockbox()
    files = list_trials()
    log(f"CSV files: {len(files)} ({Counter(r['group'] for r in files)})")

    # --- raw_files.csv (header facts for both groups, no signal read for PD)
    rf_cols = ["group", "participant", "trial", "file", "n_columns", "n_imu_sites", "n_imu_sites_all_22_columns",
               "missing_sites", "has_insole_cells", "has_insole_imu", "has_walkway", "has_events", "bytes"]
    pd.DataFrame([{k: r[k] for k in rf_cols} for r in files]).to_csv(
        os.path.join(OUT, "raw_files.csv"), index=False, lineterminator="\n", encoding="utf-8")

    by = {(r["group"], r["participant"], r["trial"]): r for r in files}
    controls = sorted({r["participant"] for r in files if r["group"] == "control"})
    pds = sorted({r["participant"] for r in files if r["group"] == "pd"})
    if OPT.limit:
        controls = controls[:OPT.limit]
        log(f"--limit {OPT.limit}: only {controls}")

    grav_rows, gyro_rows, trial_facts = [], [], []
    labels = {"Balance": Counter(), "SelfPace": Counter()}
    up_bal = {}
    cad_contacts = {}
    for k, pid in enumerate(controls):
        for trial_name in ("Balance", "SelfPace"):
            r = by.get(("control", pid, trial_name))
            if r is None:
                trial_facts.append({"participant": pid, "trial": trial_name, "status": "no_file"})
                continue
            if r["n_imu_sites"] == 0:
                trial_facts.append({"participant": pid, "trial": trial_name, "status": "no_imu_columns"})
                continue
            tr = load_trial(os.path.join(GROUP_DIRS["control"], r["file"]), r["_header"])
            labels[trial_name].update(Counter(tr["ev"].tolist()))
            dt = tr["dt_median"] if np.isfinite(tr["dt_median"]) else 0.01
            choice, common, own = quiet_windows(tr)
            sw_choice, sw_common = None, set()
            if trial_name == "Balance":
                sw_choice, sw_common, _ = quiet_windows(tr, allowed=np.isin(tr["ev"], SW_LABELS))
            trial_facts.append({"participant": pid, "trial": trial_name, "status": "read", "n_rows": tr["n_rows"],
                                "n_sites": len(tr["sites"]), "dt_median": tr["dt_median"],
                                "dt_irregular": tr["dt_irregular"], "dt_max": tr["dt_max"],
                                "first_row_artifact": tr["first_row_artifact"],
                                "common_window": choice is not None, "n_common_sites": len(common)})
            win_up = {}
            for s, d in tr["sites"].items():
                if s in common:
                    i0, kind = choice, "common"
                elif s in own:
                    i0, kind = own[s], "site"
                else:
                    continue
                est = estimate(d, i0)
                est_own = estimate(d, own[s]) if s in own else None
                lab = Counter(tr["ev"][i0:i0 + W].tolist()).most_common(1)[0]
                static_ok = (abs(est["acc_norm_ms2"] - G0) <= STATIC_ACC_TOL
                             and est["acc_dir_resultant"] >= STATIC_RESULTANT_MIN
                             and est["gyro_rms_dps"] <= STATIC_GYRO_MAX)
                win_up[s] = (est["up"], static_ok)
                if trial_name == "Balance":
                    tok, cosv = axis_token(est["up"])
                    ur = unit_ratios(d, dt)
                    est_sw = estimate(d, sw_choice) if (sw_choice is not None and s in sw_common) else None
                    row = {"participant": pid, "cohort": cohort_of(pid), "in_lockbox": int(pid in lockbox),
                           "site": s, "raw_prefix": PREFIX[s], "window_kind": kind,
                           "window_start_s": float(tr["t"][i0]), "window_label": lab[0],
                           "window_label_frac": lab[1] / W, "n_valid_trial": int(d["valid"].sum()),
                           "n_nan_trial": d["n_nan"], "n_zero_trial": d["n_zero"],
                           "gyro_rms_dps": est["gyro_rms_dps"], "acc_norm_ms2": est["acc_norm_ms2"],
                           "static_ok": int(static_ok),
                           "up_x": est["up"][0], "up_y": est["up"][1], "up_z": est["up"][2],
                           "vertical_axis": tok, "vertical_cos": cosv,
                           "acc_dir_resultant": est["acc_dir_resultant"],
                           "euler_up_angle_deg": est["euler_up_angle_deg"], "freeacc_u_mean": est["freeacc_u_mean"],
                           "own_window_angle_deg": ang(est["up"], est_own["up"]) if est_own else float("nan"),
                           "own_window_gyro_rms_dps": est_own["gyro_rms_dps"] if est_own else float("nan"),
                           "sw_window_start_s": float(tr["t"][sw_choice]) if est_sw else float("nan"),
                           "sw_window_label": Counter(tr["ev"][sw_choice:sw_choice + W].tolist()).most_common(1)[0][0] if est_sw else "",
                           "sw_window_angle_deg": ang(est["up"], est_sw["up"]) if est_sw else float("nan")}
                    row.update(ur)
                    grav_rows.append(row)
                    if static_ok:
                        up_bal[(pid, s)] = est["up"]
            if trial_name == "SelfPace":
                walk = tr["ev"] == "Walk"
                src = "Walk"
                if walk.sum() < MIN_WALK:
                    walk = np.ones(len(tr["ev"]), bool)
                    src = "all"
                cad_contacts[pid] = contact_cadence(tr)
                cad_contacts[pid].update(walkway_gait(tr))
                swing_mask = None
                if "L" in tr["contacts"] and "R" in tr["contacts"]:
                    lc = np.nan_to_num(tr["contacts"]["L"]) > 0
                    rc = np.nan_to_num(tr["contacts"]["R"]) > 0
                    swing_mask = {"L": ~lc & rc, "R": lc & ~rc}
                for s, d in tr["sites"].items():
                    m = walk & d["valid"]
                    if m.sum() < MIN_WALK:
                        continue
                    g = d["G"][m]
                    M2 = g.T @ g / len(g)
                    w, U = np.linalg.eigh(M2)
                    v1 = sign_fix(U[:, -1])
                    C = np.cov(g.T)
                    wc, Uc = np.linalg.eigh(C)
                    c1 = Uc[:, -1]
                    tok, cosv = axis_token(v1, signed=False)
                    qa = qc_axis_for(PREFIX[s])
                    ub = up_bal.get((pid, s))
                    us = win_up.get(s)
                    us_ok = us is not None and us[1]
                    gn = np.linalg.norm(g, axis=1) * RAD2DEG
                    sp = (g @ v1) * RAD2DEG
                    hi, lo = float(np.percentile(sp, 99)), float(np.percentile(sp, 1))
                    ratio = hi / abs(lo) if lo != 0 else float("inf")
                    swing_sign = 1 if ratio >= SWING_RATIO else (-1 if ratio <= 1.0 / SWING_RATIO else 0)
                    swing_tok = axis_token(swing_sign * v1)[0] if swing_sign else ""
                    ph_n, ph_mean, ph_sign, ph_tok = 0, float("nan"), 0, ""
                    if s in SWING_SITES and swing_mask is not None:
                        mp = swing_mask[s[-1]] & walk & d["valid"]
                        ph_n = int(mp.sum())
                        if ph_n >= MIN_PHASE:
                            ph_mean = float(np.mean(d["G"][mp] @ v1)) * RAD2DEG
                            ph_sign = int(np.sign(ph_mean))
                            ph_tok = axis_token(ph_sign * v1)[0] if ph_sign else ""
                    gyro_rows.append({
                        "participant": pid, "cohort": cohort_of(pid), "in_lockbox": int(pid in lockbox),
                        "site": s, "raw_prefix": PREFIX[s], "walk_source": src, "n_walk_samples": int(m.sum()),
                        "gyro_rms_dps": float(np.sqrt(np.mean(gn ** 2))), "gyro_p99_dps": float(np.percentile(gn, 99)),
                        "pc1_x": v1[0], "pc1_y": v1[1], "pc1_z": v1[2], "pc1_energy_frac": float(w[-1] / w.sum()),
                        "pc1_axis": tok, "pc1_axis_cos": cosv,
                        "centered_vs_uncentered_deg": axang(v1, c1),
                        "proj_p99_dps": hi, "proj_p1_dps": lo, "peak_ratio": ratio, "swing_sign": swing_sign,
                        "swing_axis": swing_tok,
                        "swing_phase_n": ph_n, "swing_phase_mean_dps": ph_mean, "swing_phase_sign": ph_sign,
                        "body_right_axis": ph_tok,
                        "peak_phase_agree": (int(swing_sign == ph_sign) if (swing_sign and ph_sign) else float("nan")),
                        "qc_axis": qa, "qc_axis_cos": float(abs(v1["XYZ".index(qa)])) if qa else float("nan"),
                        "angle_to_up_balance_deg": axang(v1, ub) if ub is not None else float("nan"),
                        "angle_to_up_selfpace_deg": axang(v1, us[0]) if us_ok else float("nan"),
                        "up_selfpace_vs_balance_deg": ang(us[0], ub) if (us_ok and ub is not None) else float("nan"),
                        "selfpace_window_static_ok": int(us_ok)})
        log(f"{k + 1}/{len(controls)} {pid}: gravity rows {len(grav_rows)}, gyro rows {len(gyro_rows)}")

    grav = pd.DataFrame(grav_rows)
    gyro = pd.DataFrame(gyro_rows)
    facts = pd.DataFrame(trial_facts)

    # --- site means and per-row angles to them
    def up_mean(df):
        U = df[["up_x", "up_y", "up_z"]].to_numpy(float)
        s = U.sum(0)
        return unit(s), float(np.linalg.norm(s) / len(U))

    grav["angle_to_site_mean_deg"] = np.nan
    gyro["angle_to_site_mean_deg"] = np.nan
    for s in CANON_ORDER:
        gs = grav[(grav.site == s) & (grav.static_ok == 1)]
        if len(gs):
            mu, _ = up_mean(gs)
            for i in gs.index:
                grav.loc[i, "angle_to_site_mean_deg"] = ang(grav.loc[i, ["up_x", "up_y", "up_z"]].to_numpy(float), mu)
        ys = gyro[gyro.site == s]
        if len(ys):
            mu, _ = axial_mean(ys[["pc1_x", "pc1_y", "pc1_z"]].to_numpy(float))
            for i in ys.index:
                gyro.loc[i, "angle_to_site_mean_deg"] = axang(gyro.loc[i, ["pc1_x", "pc1_y", "pc1_z"]].to_numpy(float), mu)

    grav.to_csv(os.path.join(OUT, "site_gravity.csv"), index=False, lineterminator="\n", encoding="utf-8", float_format="%.6g")
    gyro.to_csv(os.path.join(OUT, "site_gyro_axis.csv"), index=False, lineterminator="\n", encoding="utf-8", float_format="%.6g")
    log("wrote site_gravity.csv", grav.shape, "site_gyro_axis.csv", gyro.shape)

    # --- site_summary.csv
    sets = {"all": lambda df: df, "fit": lambda df: df[df.in_lockbox == 0],
            "HC": lambda df: df[df.cohort == "HC"], "WHC": lambda df: df[df.cohort == "WHC"]}
    summ = []
    means = {}
    for set_name, fsel in sets.items():
        G_ = fsel(grav[grav.static_ok == 1])
        Y_ = fsel(gyro)
        for s in CANON_ORDER:
            gs = G_[G_.site == s]
            ys = Y_[Y_.site == s]
            row = {"set": set_name, "site": s, "raw_prefix": PREFIX[s], "n_gravity": len(gs)}
            if len(gs):
                mu, R = up_mean(gs)
                means[(set_name, s)] = mu
                angs = [ang(u, mu) for u in gs[["up_x", "up_y", "up_z"]].to_numpy(float)]
                tok, cosv = axis_token(mu)
                modal = Counter(gs.vertical_axis).most_common(1)[0]
                row.update({"up_mean_x": mu[0], "up_mean_y": mu[1], "up_mean_z": mu[2], "up_resultant": R,
                            "up_spread_median_deg": float(np.median(angs)), "up_spread_p90_deg": q(angs, 90),
                            "vertical_axis": tok, "vertical_cos": cosv, "vertical_axis_modal": modal[0],
                            "vertical_axis_modal_frac": modal[1] / len(gs),
                            "n_vertical_axis_tokens": len(set(gs.vertical_axis)),
                            "sw_window_angle_median_deg": q(gs.sw_window_angle_deg.to_numpy(float), 50),
                            "sw_window_angle_p90_deg": q(gs.sw_window_angle_deg.to_numpy(float), 90)})
                mv = gs[np.isfinite(gs.sw_window_angle_deg) & (gs.sw_window_start_s != gs.window_start_s)]
                row.update({"sw_moved_n": len(mv),
                            "sw_moved_angle_median_deg": q(mv.sw_window_angle_deg.to_numpy(float), 50),
                            "sw_moved_angle_p90_deg": q(mv.sw_window_angle_deg.to_numpy(float), 90)})
            row["n_gyro"] = len(ys)
            if len(ys):
                V = ys[["pc1_x", "pc1_y", "pc1_z"]].to_numpy(float)
                mu, _ = axial_mean(V)
                angs = [axang(v, mu) for v in V]
                tok, cosv = axis_token(mu, signed=False)
                modal = Counter(ys.pc1_axis).most_common(1)[0]
                a_up = ys.angle_to_up_balance_deg.to_numpy(float)
                row.update({"pc1_mean_x": mu[0], "pc1_mean_y": mu[1], "pc1_mean_z": mu[2],
                            "pc1_spread_median_deg": float(np.median(angs)), "pc1_spread_p90_deg": q(angs, 90),
                            "pc1_axis": tok, "pc1_axis_cos": cosv, "pc1_axis_modal": modal[0],
                            "pc1_axis_modal_frac": modal[1] / len(ys),
                            "pc1_energy_frac_median": float(ys.pc1_energy_frac.median()),
                            "n_angle_to_up": int(np.isfinite(a_up).sum()),
                            "angle_to_up_median_deg": q(a_up, 50), "angle_to_up_p25_deg": q(a_up, 25),
                            "angle_to_up_p75_deg": q(a_up, 75),
                            "angle_to_up_selfpace_median_deg": q(ys.angle_to_up_selfpace_deg.to_numpy(float), 50),
                            "up_selfpace_vs_balance_median_deg": q(ys.up_selfpace_vs_balance_deg.to_numpy(float), 50),
                            "up_selfpace_vs_balance_p90_deg": q(ys.up_selfpace_vs_balance_deg.to_numpy(float), 90),
                            "qc_axis": ys.qc_axis.iloc[0],
                            "qc_axis_agree_frac": float((ys.pc1_axis == ys.qc_axis).mean()),
                            "gyro_rms_walk_median_dps": float(ys.gyro_rms_dps.median()),
                            "walk_source_all_count": int((ys.walk_source == "all").sum()),
                            "peak_ratio_median": float(ys.peak_ratio.replace(np.inf, np.nan).median()),
                            "swing_ambiguous_count": int((ys.swing_sign == 0).sum())})
                sw = ys[ys.swing_sign != 0]
                if len(sw):
                    Vs = sw[["pc1_x", "pc1_y", "pc1_z"]].to_numpy(float) * sw.swing_sign.to_numpy(float)[:, None]
                    mu_s = unit(Vs.sum(0))
                    tok_s, cos_s = axis_token(mu_s)
                    modal_s = Counter(sw.swing_axis).most_common(1)[0]
                    row.update({"swing_mean_x": mu_s[0], "swing_mean_y": mu_s[1], "swing_mean_z": mu_s[2],
                                "swing_resultant": float(np.linalg.norm(Vs.sum(0)) / len(Vs)),
                                "swing_axis": tok_s, "swing_axis_cos": cos_s, "swing_axis_modal": modal_s[0],
                                "swing_axis_modal_frac": modal_s[1] / len(ys)})
                ph = ys[ys.swing_phase_sign != 0]
                if len(ph):
                    Vp = ph[["pc1_x", "pc1_y", "pc1_z"]].to_numpy(float) * ph.swing_phase_sign.to_numpy(float)[:, None]
                    mu_p = unit(Vp.sum(0))
                    tok_p, cos_p = axis_token(mu_p)
                    modal_p = Counter(ph.body_right_axis).most_common(1)[0]
                    both = ys[(ys.swing_sign != 0) & (ys.swing_phase_sign != 0)]
                    row.update({"phase_decided_count": len(ph),
                                "swing_phase_abs_mean_median_dps": float(ph.swing_phase_mean_dps.abs().median()),
                                "body_right_mean_x": mu_p[0], "body_right_mean_y": mu_p[1], "body_right_mean_z": mu_p[2],
                                "body_right_resultant": float(np.linalg.norm(Vp.sum(0)) / len(Vp)),
                                "body_right_axis": tok_p, "body_right_cos": cos_p,
                                "body_right_modal": modal_p[0], "body_right_modal_frac": modal_p[1] / len(ys),
                                "peak_phase_both_decided": len(both),
                                "peak_phase_agree_count": int((both.swing_sign == both.swing_phase_sign).sum())})
            summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "site_summary.csv"), index=False, lineterminator="\n", encoding="utf-8", float_format="%.6g")
    log("wrote site_summary.csv", summ.shape)

    cohort_angle = {}
    for s in CANON_ORDER:
        a, b = means.get(("HC", s)), means.get(("WHC", s))
        cohort_angle[s] = ang(a, b) if a is not None and b is not None else None
    fit_vs_all = {s: ang(means[("fit", s)], means[("all", s)]) for s in CANON_ORDER
                  if ("fit", s) in means and ("all", s) in means}

    # --- walk_speed_cadence.csv
    pk, n_pk_rows = read_pkmas()
    demo = read_demographics(DEMO["control"])
    demo_pd = read_demographics(DEMO["pd"])
    dup = {k: len(v) for k, v in pk.items() if len(v) > 1}
    wrows = []
    for pid in controls:
        b = base_id(pid)
        recs = pk.get(("SelfPace", b), [])
        rec = recs[0] if recs else {}
        h_in = demo.get(b, {}).get("Height (in)", "")
        try:
            h_m = float(h_in) * 0.0254
        except ValueError:
            h_m = float("nan")
        v = rec.get("velocity_cm_s", float("nan"))
        cc = cad_contacts.get(pid, {})
        wrows.append({"participant": pid, "cohort": cohort_of(pid), "in_lockbox": int(pid in lockbox),
                      "pkmas_rows": len(recs), "pkmas_task_raw": rec.get("task_raw", ""),
                      "pkmas_group": rec.get("pd_vs_control", ""),
                      "gait_speed_m_s": v / 100.0, "cadence_steps_min": rec.get("cadence_steps_min", float("nan")),
                      "step_length_cm": rec.get("step_length_cm", float("nan")),
                      "stride_length_cm": rec.get("stride_length_cm", float("nan")),
                      "step_time_s": rec.get("step_time_s", float("nan")),
                      "stride_width_cm": rec.get("stride_width_cm", float("nan")),
                      "n_steps_pkmas": rec.get("n_steps", float("nan")),
                      "ambulation_time_s": rec.get("ambulation_time_s", float("nan")),
                      "height_m": h_m, "speed_per_height_1_s": (v / 100.0) / h_m if np.isfinite(h_m) and h_m > 0 else float("nan"),
                      "n_contact_onsets": cc.get("n_contact_onsets", 0), "n_passes": cc.get("n_passes", 0),
                      "cadence_contacts_steps_min": cc.get("cadence_contacts_steps_min", float("nan")),
                      "n_footfalls_ww": cc.get("n_footfalls_ww", 0), "n_passes_ww": cc.get("n_passes_ww", 0),
                      "speed_cells_s": cc.get("speed_cells_s", float("nan")),
                      "step_cells_mean": cc.get("step_cells_mean", float("nan"))})
    walk = pd.DataFrame(wrows)
    # walkway cell size from the controls that carry both a PKMAS step length and walkway footfalls
    mm = walk[np.isfinite(walk.step_length_cm) & np.isfinite(walk.step_cells_mean) & (walk.step_cells_mean > 0)]
    cell_ratio = (mm.step_length_cm / mm.step_cells_mean).to_numpy(float)
    cell_cm = float(np.median(cell_ratio)) if len(cell_ratio) else float("nan")
    walk["walkway_step_length_cm"] = walk.step_cells_mean * cell_cm
    walk["walkway_speed_m_s"] = walk.speed_cells_s * cell_cm / 100.0
    has_pk = np.isfinite(walk.gait_speed_m_s)
    walk["speed_source"] = np.where(has_pk, "pkmas", np.where(np.isfinite(walk.walkway_speed_m_s), "walkway_cells", ""))
    walk["gait_speed_best_m_s"] = np.where(has_pk, walk.gait_speed_m_s, walk.walkway_speed_m_s)
    walk["cadence_best_steps_min"] = np.where(np.isfinite(walk.cadence_steps_min), walk.cadence_steps_min, walk.cadence_contacts_steps_min)
    walk["speed_best_per_height_1_s"] = walk.gait_speed_best_m_s / walk.height_m
    walk.to_csv(os.path.join(OUT, "walk_speed_cadence.csv"), index=False, lineterminator="\n", encoding="utf-8", float_format="%.6g")
    log("wrote walk_speed_cadence.csv", walk.shape)

    # --- raw_layout.csv with measured evidence
    G_ok = grav[grav.static_ok == 1]
    rd = facts[facts.status == "read"]
    all_headers = [r["_header"] for r in files]

    def n_files_with(pred):
        return {g: sum(1 for r in files if r["group"] == g and pred(set(r["_header"]))) for g in ("control", "pd")}

    ev_c = n_files_with(lambda h: "GeneralEvent" in h)
    ww_c = n_files_with(lambda h: "Walkway_X" in h)
    ins_c = n_files_with(lambda h: "LPressure1" in h)
    insi_c = n_files_with(lambda h: "Linsole:Acc_X" in h)
    cop_c = n_files_with(lambda h: "COP L X" in h)
    imu_c = n_files_with(lambda h: any(h2.endswith("_Acc_X") and ":" not in h2 for h2 in h))

    def med(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else float("nan")

    layout = [
        ("Time", "Time", 1, "s (string '<value> sec')", "sample time from the trial start",
         f"median dt {med(rd.dt_median):.4f} s over {len(rd)} control trials read; trials with any dt off 0.01 s: {int((rd.dt_irregular > 0).sum())}; max dt {np.nanmax(rd.dt_max):.3f} s"),
        ("events", "GeneralEvent; ClinicalEvent", 2, "label", "task phase (Balance: EO/EC stance conditions; SelfPace: Standing, Walk, Turn) and clinical event",
         f"files with the columns: control {ev_c['control']}, pd {ev_c['pd']}"),
        ("walkway", "L Foot Contact; R Foot Contact; L Foot Pressure; R Foot Pressure; Walkway_X; Walkway_Y; WalkwayPressureLevel; WalkwayFoot", 8,
         "0/1 contact; integer pressure level; pipe-separated cell lists", "PKMAS walkway contacts resampled on the IMU clock",
         f"files with the columns: control {ww_c['control']}, pd {ww_c['pd']}"),
        ("insole", "LPressure1..16; LTotalForce; LCoP_X; LCoP_Y (and R)", 38, "not stated in the raw folder",
         "Moticon insole cells, total force and centre of pressure", f"files with the columns: control {ins_c['control']}, pd {ins_c['pd']}; extra COP L/R X/Y columns: control {cop_c['control']}, pd {cop_c['pd']}"),
        ("insole_imu", "Linsole:Acc_X..Z; Linsole:Gyr_X..Z (and R)", 12, "not stated in the raw folder", "inertial unit inside each insole",
         f"files with the columns: control {insi_c['control']}, pd {insi_c['pd']}"),
    ]
    per_sensor = [
        ("imu:Acc", "<sensor>_Acc_X/Y/Z", 3, "m/s2", "specific force in the sensor frame, gravity included",
         f"median |mean Acc| in static quiet windows {med(G_ok.acc_norm_ms2):.3f} m/s2 (n={len(G_ok)}); VelInc/(Acc dt) least-squares ratio median {med(grav.acc_unit_ratio):.5f}, p1 {q(grav.acc_unit_ratio, 1):.5f}, p99 {q(grav.acc_unit_ratio, 99):.5f}"),
        ("imu:FreeAcc", "<sensor>_FreeAcc_E/N/U", 3, "m/s2", "acceleration with gravity removed, in the local earth frame East, North, Up (by the suffix names)",
         f"median |mean FreeAcc_U| in static quiet windows {med(np.abs(G_ok.freeacc_u_mean)):.4f} m/s2"),
        ("imu:Gyr", "<sensor>_Gyr_X/Y/Z", 3, "rad/s", "angular rate in the sensor frame",
         f"(2 OriInc vector / dt)/Gyr least-squares ratio median {med(grav.gyro_unit_ratio):.5f}, p1 {q(grav.gyro_unit_ratio, 1):.5f}, p99 {q(grav.gyro_unit_ratio, 99):.5f} (a deg/s channel would give 0.01745)"),
        ("imu:Mag", "<sensor>_Mag_X/Y/Z", 3, "normalized to the local field (a.u.)", "magnetic field in the sensor frame",
         f"per site-trial median norm: median {med(grav.mag_norm_median):.3f}, p5 {q(grav.mag_norm_median, 5):.3f}, p95 {q(grav.mag_norm_median, 95):.3f}"),
        ("imu:VelInc", "<sensor>_VelInc_X/Y/Z", 3, "m/s per sample", "velocity increment over one sample (strapdown integration output), sensor frame",
         "equals Acc times dt (ratio above); row 0 holds the increment accumulated before the trial"),
        ("imu:OriInc", "<sensor>_OriInc_q0..q3", 4, "unit quaternion, scalar first", "orientation increment over one sample",
         "vector part equals Gyr dt / 2 (ratio above); row 0 holds the increment accumulated before the trial"),
        ("imu:Euler", "<sensor>_Roll/Pitch/Yaw", 3, "deg", "sensor orientation in the earth frame, R = Rz(yaw) Ry(pitch) Rx(roll); up in the sensor frame = (-sin pitch, sin roll cos pitch, cos roll cos pitch)",
         f"median angle between the Euler up and the Acc up in static quiet windows {med(G_ok.euler_up_angle_deg):.3f} deg, p99 {q(G_ok.euler_up_angle_deg, 99):.3f} deg"),
    ]
    lay_rows = [{"column_group": a, "columns": b, "n_columns": c, "unit": d, "meaning": e, "evidence": f} for a, b, c, d, e, f in layout + per_sensor]
    lay_rows.append({"column_group": "imu:sensors", "columns": "; ".join(f"{p} -> {s}" for s, p in SITES), "n_columns": 22 * len(SITES),
                     "unit": "", "meaning": "13 Xsens sensors, 22 columns each, raw prefix -> canonical site (WEARGAIT_LOCMAP of build_canonical_imu.py)",
                     "evidence": f"files with IMU columns: control {imu_c['control']}, pd {imu_c['pd']}"})
    pd.DataFrame(lay_rows).to_csv(os.path.join(OUT, "raw_layout.csv"), index=False, lineterminator="\n", encoding="utf-8")

    # --- counts and checks for summary.json
    n_files_group = Counter(r["group"] for r in files)
    trials_group = {g: dict(Counter(r["trial"] for r in files if r["group"] == g)) for g in ("control", "pd")}
    mat_counts = {}
    for g, d in MAT_DIRS.items():
        mat_counts[g] = len([f for f in os.listdir(d) if f.endswith(".mat") and not f.startswith("._")]) if os.path.isdir(d) else None
    col_counts = {g: dict(Counter(r["n_columns"] for r in files if r["group"] == g)) for g in ("control", "pd")}
    lock_ctrl = sorted(p for p in controls if p in lockbox)
    lock_ctrl_group_pd = sorted(p for p in lock_ctrl if lockbox[p] not in ("HC", "control"))  # v1 labelled controls HC, v1.1 on "control"
    lock_pd = sorted(p for p in pds if p in lockbox)
    lock_unmatched = sorted(p for p in lockbox if p not in controls and p not in pds)
    pk_ctrl = {p for (t, p), v in pk.items() if t == "SelfPace" and v[0]["pd_vs_control"] == "Control"}
    ctrl_base = {base_id(p) for p in controls}
    wk = walk[np.isfinite(walk.gait_speed_m_s)]
    both = walk[np.isfinite(walk.cadence_steps_min) & np.isfinite(walk.cadence_contacts_steps_min)]
    cad_r = float(np.corrcoef(both.cadence_steps_min, both.cadence_contacts_steps_min)[0, 1]) if len(both) > 2 else float("nan")
    cad_mad = float(np.median(np.abs(both.cadence_steps_min - both.cadence_contacts_steps_min))) if len(both) else float("nan")
    qs = [10, 25, 50, 75, 90]

    def quant(df, c):
        return {f"p{p}": q(df[c].to_numpy(float), p) for p in qs}

    sp_both = walk[np.isfinite(walk.gait_speed_m_s) & np.isfinite(walk.walkway_speed_m_s)]
    sp_r = float(np.corrcoef(sp_both.gait_speed_m_s, sp_both.walkway_speed_m_s)[0, 1]) if len(sp_both) > 2 else float("nan")
    sp_mad = float(np.median(np.abs(sp_both.gait_speed_m_s - sp_both.walkway_speed_m_s))) if len(sp_both) else float("nan")
    sp_rel = float(np.median(np.abs(sp_both.walkway_speed_m_s / sp_both.gait_speed_m_s - 1.0))) if len(sp_both) else float("nan")
    folder_counts = count_all_folders()
    lateral = {}
    for s in SWING_SITES:
        r = summ[(summ.set == "all") & (summ.site == s)]
        tok = tx(r.iloc[0].get("body_right_axis")) if len(r) else ""
        if tok:
            lat = tok if s.endswith("_R") else ({"+": "-", "-": "+"}[tok[0]] + tok[1])
            lateral[s] = {"body_right_axis": tok, "lateral_axis": lat,
                          "body_right_modal_frac": float(r.iloc[0].get("body_right_modal_frac")),
                          "phase_decided_count": int(r.iloc[0].get("phase_decided_count")),
                          "peak_rule_swing_axis": tx(r.iloc[0].get("swing_axis")),
                          "peak_phase_both_decided": int(r.iloc[0].get("peak_phase_both_decided")),
                          "peak_phase_agree_count": int(r.iloc[0].get("peak_phase_agree_count"))}
            up_m = means.get(("all", s))
            right = np.array([r.iloc[0].get(f"body_right_mean_{a}") for a in "xyz"], float)
            if up_m is not None and np.all(np.isfinite(right)):
                left = unit(-right - np.dot(-right, up_m) * up_m)
                fwd = np.cross(left, up_m)
                lateral[s].update({"segment_up": up_m.tolist(), "segment_left": left.tolist(), "segment_forward": fwd.tolist(),
                                   "forward_axis": axis_token(fwd)[0], "forward_axis_cos": axis_token(fwd)[1],
                                   "right_vs_up_deg": ang(right, up_m)})
    weak_axis_sites = {s: float(v) for s, v in zip(summ[summ.set == "all"].site, summ[summ.set == "all"].pc1_energy_frac_median)
                       if np.isfinite(v) and v < WEAK_AXIS}
    grav_out = grav[(grav.static_ok == 1) & (grav.angle_to_site_mean_deg > OUTLIER_DEG)]
    gyro_out = gyro[gyro.angle_to_site_mean_deg > OUTLIER_DEG]
    sw_moved = G_ok[np.isfinite(G_ok.sw_window_angle_deg) & (G_ok.sw_window_start_s != G_ok.window_start_s)]
    bal_read = facts[(facts.trial == "Balance") & (facts.status == "read")]
    sp_read = facts[(facts.trial == "SelfPace") & (facts.status == "read")]
    summary = {
        "moveatlas_raw": RAW,
        "site_map": {s: p for s, p in SITES},
        "qc_viewer_gyro_channel": {p: qc_axis_for(p) for _, p in SITES},
        "datacard_statements": read_datacard_statements(),
        "unit_checks": {"gyro_unit_ratio_median": med(grav.gyro_unit_ratio), "acc_unit_ratio_median": med(grav.acc_unit_ratio),
                        "mag_norm_median": med(grav.mag_norm_median), "static_acc_norm_median_ms2": med(G_ok.acc_norm_ms2)},
        "files": {"csv_per_group": dict(n_files_group), "trials_per_group": trials_group, "mat_files": mat_counts,
                  "column_count_distribution": col_counts},
        "participants": {"control_with_csv": len(controls), "pd_with_csv": len(pds),
                         "control_demographic_rows": len(demo), "pd_demographic_rows": len(demo_pd),
                         "control_by_cohort": dict(Counter(cohort_of(p) for p in controls)),
                         "pd_by_cohort": dict(Counter(cohort_of(p) for p in pds))},
        "controls_read": {"balance_read": int(len(bal_read)), "selfpace_read": int(len(sp_read)),
                          "trial_status": {f"{a}:{b}": int(c) for (a, b), c in Counter(zip(facts.trial, facts.status)).items()},
                          "balance_no_imu": sorted(facts[(facts.trial == "Balance") & (facts.status != "read")].participant.tolist()),
                          "selfpace_no_imu": sorted(facts[(facts.trial == "SelfPace") & (facts.status != "read")].participant.tolist()),
                          "trials_with_first_row_artifact": int(rd.first_row_artifact.sum()),
                          "sampling_hz": 1.0 / med(rd.dt_median),
                          "trials_with_irregular_dt": int((rd.dt_irregular > 0).sum()),
                          "balance_common_window_found": int(bal_read.common_window.sum()),
                          "balance_sites_outside_common_window": int((grav.window_kind == "site").sum()),
                          "missing_sites_in_read_balance": int(13 * len(bal_read) - len(grav))},
        "gravity": {"rows": len(grav), "static_ok": int(grav.static_ok.sum()),
                    "not_static": grav[grav.static_ok == 0][["participant", "site", "acc_norm_ms2", "acc_dir_resultant", "gyro_rms_dps"]].to_dict("records"),
                    "window_labels": dict(Counter(grav[grav.window_kind == "common"].drop_duplicates(["participant"]).window_label)),
                    "gyro_rms_static_median_dps": med(G_ok.gyro_rms_dps),
                    "gyro_rms_static_p95_dps": q(G_ok.gyro_rms_dps, 95),
                    "own_window_angle_median_deg": med(G_ok.own_window_angle_deg),
                    "own_window_angle_p90_deg": q(G_ok.own_window_angle_deg, 90),
                    "sw_window_n": int(np.isfinite(G_ok.sw_window_angle_deg).sum()),
                    "sw_window_angle_median_deg": med(G_ok.sw_window_angle_deg),
                    "sw_window_angle_p90_deg": q(G_ok.sw_window_angle_deg, 90),
                    "sw_window_moved_n": int(len(sw_moved)),
                    "sw_window_moved_participants": int(sw_moved.participant.nunique()),
                    "sw_window_moved_angle_median_deg": med(sw_moved.sw_window_angle_deg),
                    "sw_window_moved_angle_p90_deg": q(sw_moved.sw_window_angle_deg, 90),
                    "static_rule_fail_counts": {
                        "acc_norm": int((np.abs(grav.acc_norm_ms2 - G0) > STATIC_ACC_TOL).sum()),
                        "resultant": int((grav.acc_dir_resultant < STATIC_RESULTANT_MIN).sum()),
                        "gyro_rms": int((grav.gyro_rms_dps > STATIC_GYRO_MAX).sum())},
                    "outliers_over_45deg": grav_out[["participant", "site", "vertical_axis", "angle_to_site_mean_deg"]].to_dict("records")},
        "gyro_axis": {"rows": len(gyro), "walk_source_all": int((gyro.walk_source == "all").sum()),
                      "centered_vs_uncentered_median_deg": med(gyro.centered_vs_uncentered_deg),
                      "centered_vs_uncentered_p90_deg": q(gyro.centered_vs_uncentered_deg, 90),
                      "outliers_over_45deg_count": int(len(gyro_out)),
                      "outliers_over_45deg_by_site": dict(Counter(gyro_out.site))},
        "cohort_mean_up_angle_deg": cohort_angle,
        "fit_vs_all_mean_up_angle_deg": fit_vs_all,
        "walk": {"controls": len(walk), "with_pkmas_selfpace": int(np.isfinite(walk.gait_speed_m_s).sum()),
                 "pkmas_rows_total": n_pk_rows, "pkmas_duplicate_keys": {f"{a}|{b}": c for (a, b), c in dup.items()},
                 "pkmas_selfpace_controls": len(pk_ctrl),
                 "pkmas_selfpace_controls_without_csv": sorted(pk_ctrl - ctrl_base),
                 "controls_without_pkmas_selfpace": sorted(walk[~np.isfinite(walk.gait_speed_m_s)].participant.tolist()),
                 "with_height": int(np.isfinite(walk.height_m).sum()),
                 "speed_m_s_all": quant(wk, "gait_speed_m_s"), "cadence_all": quant(wk, "cadence_steps_min"),
                 "speed_m_s_fit": quant(wk[wk.in_lockbox == 0], "gait_speed_m_s"),
                 "cadence_fit": quant(wk[wk.in_lockbox == 0], "cadence_steps_min"),
                 "n_fit": int((wk.in_lockbox == 0).sum()),
                 "cadence_crosscheck_n": int(len(both)), "cadence_crosscheck_pearson_r": cad_r,
                 "cadence_crosscheck_median_abs_diff": cad_mad,
                 "walkway_cell_cm_estimate": cell_cm, "walkway_cell_cm_n": int(len(cell_ratio)),
                 "walkway_cell_cm_p10": q(cell_ratio, 10), "walkway_cell_cm_p90": q(cell_ratio, 90),
                 "walkway_speed_check_n": int(len(sp_both)), "walkway_speed_check_pearson_r": sp_r,
                 "walkway_speed_check_median_abs_diff_m_s": sp_mad,
                 "walkway_speed_check_median_rel_diff": sp_rel,
                 "speed_source_counts": dict(Counter(walk.speed_source)),
                 "cadence_best_n": int(np.isfinite(walk.cadence_best_steps_min).sum()),
                 "speed_best_all": quant(walk, "gait_speed_best_m_s"), "cadence_best_all": quant(walk, "cadence_best_steps_min"),
                 "speed_best_fit": quant(walk[walk.in_lockbox == 0], "gait_speed_best_m_s"),
                 "cadence_best_fit": quant(walk[walk.in_lockbox == 0], "cadence_best_steps_min"),
                 "n_best_all": int(np.isfinite(walk.gait_speed_best_m_s).sum()),
                 "n_best_fit": int((np.isfinite(walk.gait_speed_best_m_s) & (walk.in_lockbox == 0)).sum()),
                 "by_cohort_best": {c: {"n": int(np.isfinite(walk[walk.cohort == c].gait_speed_best_m_s).sum()),
                                        "speed": quant(walk[walk.cohort == c], "gait_speed_best_m_s"),
                                        "cadence": quant(walk[walk.cohort == c], "cadence_best_steps_min")}
                                    for c in ("HC", "WHC")}},
        "participants_all_folders": folder_counts,
        "lateral_axis": lateral,
        "sites_without_dominant_walking_axis": weak_axis_sites,
        "lockbox": {"weargait_persons": len(lockbox), "controls_in_lockbox": lock_ctrl,
                    "controls_in_lockbox_labelled_non_HC": {p: lockbox[p] for p in lock_ctrl_group_pd},
                    "pd_in_lockbox_with_csv": len(lock_pd), "lockbox_ids_without_csv": lock_unmatched},
        "labels": {k: dict(v) for k, v in labels.items()},
        "thresholds": {"window_samples": W, "window_step_samples": STEP, "static_acc_tol_ms2": STATIC_ACC_TOL,
                       "static_resultant_min": STATIC_RESULTANT_MIN, "static_gyro_max_dps": STATIC_GYRO_MAX,
                       "shoulder_width_labels": list(SW_LABELS), "swing_ratio": SWING_RATIO, "weak_axis_energy": WEAK_AXIS, "min_walk_samples": MIN_WALK,
                       "min_swing_phase_samples": MIN_PHASE, "outlier_deg": OUTLIER_DEG,
                       "contact_min_off_samples": CONTACT_MIN_OFF, "pass_gap_s": PASS_GAP_S},
        "limit": OPT.limit,
        "runtime_s": round(time.time() - T0, 1),
    }
    summary = clean(summary)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    write_summary_md(summary, summ, grav, gyro, walk)
    write_provenance(ARGS, {"outputs": sorted(os.listdir(OUT)), "limit": OPT.limit, "runtime_s": summary["runtime_s"],
                            "thresholds": summary["thresholds"]})
    log("done")


def clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN and inf to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def tx(x):
    return x if isinstance(x, str) else ""


def f1(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(x) else f"{x:.{nd}f}"


def write_summary_md(S, summ, grav, gyro, walk):
    L = []
    P = S["participants"]
    C = S["controls_read"]
    L.append("# weargait_pd sensor frames on controls (plan task B2a, gate G2)")
    L.append("")
    L.append("Generated by scripts/probe_weargait_sensor_frames.py; every number below is read from summary.json and the CSVs of this directory. MoveAtlas commit in _provenance.json. Definitions in the script docstring.")
    if S["limit"]:
        L.append("")
        L.append(f"TEST RUN: --limit {S['limit']} (first {S['limit']} controls only).")
    L.append("")
    L.append("## Raw layout")
    L.append("")
    L.append(f"CSV files: control {S['files']['csv_per_group'].get('control', 0)}, PD {S['files']['csv_per_group'].get('pd', 0)} "
             f"(raw_files.csv, one row per file with its sites and column groups). Participants with CSVs: control {P['control_with_csv']} "
             f"({', '.join(f'{k} {v}' for k, v in sorted(P['control_by_cohort'].items()))}), PD {P['pd_with_csv']} "
             f"({', '.join(f'{k} {v}' for k, v in sorted(P['pd_by_cohort'].items()))}); demographic rows: control {P['control_demographic_rows']}, PD {P['pd_demographic_rows']}. "
             f"MAT files (not read): control {S['files']['mat_files'].get('control')}, PD {S['files']['mat_files'].get('pd')}.")
    Fc = S["participants_all_folders"]
    L.append("")
    L.append("Participants per folder (listing only, IDs without the ' (control)' suffix): " +
             "; ".join(f"{k.replace('|', ' ')} {v}" for k, v in Fc["per_folder"].items()) +
             ". Union over every folder: " + ", ".join(f"{k} {v}" for k, v in Fc["union_by_group"].items()) +
             ". IDs present only outside the main CSV folders: " +
             "; ".join(f"{g} {len(v)} ({', '.join(v)})" for g, v in Fc["ids_only_outside_main_csv"].items() if v) + ".")
    L.append("")
    L.append("Each of the 13 Xsens sensors carries 22 columns, <sensor>_<quantity>. Columns are addressed by name: their order differs between files, and some files lack whole sensors (raw_files.csv). Units and conventions, with the evidence measured here (raw_layout.csv):")
    L.append("")
    L.append("| group | columns | unit | meaning | evidence |")
    L.append("|---|---|---|---|---|")
    with open(os.path.join(OUT, "raw_layout.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            L.append(f"| {r['column_group']} | {r['columns']} | {r['unit']} | {r['meaning']} | {r['evidence']} |")
    L.append("")
    if S.get("datacard_statements"):
        L.append("The datacard (datasets/imu/weargait_pd/datacard.md) states, verbatim: " +
                 "; ".join(f"{k}: \"{v}\"" for k, v in S["datacard_statements"].items()) + "." +
                 f" Measured here on the raw CSVs: gyroscope over the OriInc-derived rate in rad/s, median ratio {f1(S['unit_checks']['gyro_unit_ratio_median'], 5)} (1 for rad/s, 0.01745 for deg/s); magnetometer norm median {f1(S['unit_checks']['mag_norm_median'], 3)}; sampling {f1(C['sampling_hz'], 1)} Hz."
                 " The deg/s gyroscope of the canonical records is a MoveAtlas conversion: a renderer that imitates the raw CSVs emits rad/s, one that imitates the canonical records emits deg/s.")
        L.append("")
    L.append(f"Sampling: {f1(C['sampling_hz'], 1)} Hz (1 / median dt of the Time column). Row 0 of a trial is an accumulation artefact in {C['trials_with_first_row_artifact']} of {C['balance_read'] + C['selfpace_read']} control trials read and is dropped everywhere.")
    L.append("")
    L.append("## Controls read")
    L.append("")
    L.append(f"Balance trials read: {C['balance_read']}; SelfPace trials read: {C['selfpace_read']}. Balance without IMU columns or file: {', '.join(C['balance_no_imu']) or 'none'}; SelfPace without: {', '.join(C['selfpace_no_imu']) or 'none'}. "
             f"Site slots missing in the Balance trials read (sensor absent from the file or never valid for 5 s): {C['missing_sites_in_read_balance']}. "
             f"Balance trials with a whole-body quiet window: {C['balance_common_window_found']}; site estimates outside it (own window): {C['balance_sites_outside_common_window']}.")
    Gv = S["gravity"]
    L.append("")
    L.append(f"Gravity estimates: {Gv['rows']}, static_ok {Gv['static_ok']}. Gyroscope RMS in the static windows: median {f1(Gv['gyro_rms_static_median_dps'], 2)} deg/s, p95 {f1(Gv['gyro_rms_static_p95_dps'], 2)} deg/s. "
             f"Angle between the whole-body window estimate and the site's own quietest window: median {f1(Gv['own_window_angle_median_deg'], 2)} deg, p90 {f1(Gv['own_window_angle_p90_deg'], 2)} deg. "
             f"Balance condition holding the whole-body window (per participant): {', '.join(f'{k} {v}' for k, v in sorted(Gv['window_labels'].items(), key=lambda kv: -kv[1]))}. "
             f"Angle between the whole-body window estimate and the quietest whole-body window inside the shoulder-width conditions (EO or EC_FeetShoWidth), n {Gv['sw_window_n']}: median {f1(Gv['sw_window_angle_median_deg'], 2)} deg, p90 {f1(Gv['sw_window_angle_p90_deg'], 2)} deg. "
             f"That median is 0 by construction wherever the whole-body window already lies in a shoulder-width condition; over the {Gv['sw_window_moved_n']} estimates ({Gv['sw_window_moved_participants']} participants) whose window moves, the shift is median {f1(Gv['sw_window_moved_angle_median_deg'], 2)} deg, p90 {f1(Gv['sw_window_moved_angle_p90_deg'], 2)} deg (per site in the tables below).")
    if Gv["not_static"]:
        L.append("")
        L.append(f"Rejected as not static ({len(Gv['not_static'])}; rule failures: " + ", ".join(f"{k} {v}" for k, v in Gv["static_rule_fail_counts"].items()) + "): " +
                 "; ".join(f"{r['participant']} {r['site']} (|f| {f1(r['acc_norm_ms2'], 2)} m/s2, resultant {f1(r['acc_dir_resultant'], 4)}, {f1(r['gyro_rms_dps'], 2)} deg/s)" for r in Gv["not_static"]) + ".")
    L.append("")
    L.append("## Gravity in the sensor frame (Balance, quiet standing)")
    L.append("")
    L.append("up is the direction of the mean specific force at rest (vertical up); gravity is -up. vertical axis: the sensor axis most aligned with up and its sign (for example +X: the sensor +X axis points up). Set all (every control):")
    L.append("")

    def site_table(set_name):
        rows = summ[summ.set == set_name]
        out = ["| site | raw | n | up mean (x, y, z) | resultant | spread median / p90 (deg) | vertical axis (cos) | modal axis (share) | shoulder-width shift median / p90 (deg) | moved windows: n, shift median / p90 (deg) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in rows.iterrows():
            if not r.get("n_gravity"):
                out.append(f"| {r.site} | {r.raw_prefix} | 0 | | | | | | | |")
                continue
            out.append(f"| {r.site} | {r.raw_prefix} | {int(r.n_gravity)} | ({f1(r.up_mean_x, 3)}, {f1(r.up_mean_y, 3)}, {f1(r.up_mean_z, 3)}) | {f1(r.up_resultant, 3)} | "
                       f"{f1(r.up_spread_median_deg)} / {f1(r.up_spread_p90_deg)} | {r.vertical_axis} ({f1(r.vertical_cos, 3)}) | {r.vertical_axis_modal} ({f1(r.vertical_axis_modal_frac, 2)}) | "
                       f"{f1(r.sw_window_angle_median_deg)} / {f1(r.sw_window_angle_p90_deg)} | {int(r.sw_moved_n)}, {f1(r.sw_moved_angle_median_deg)} / {f1(r.sw_moved_angle_p90_deg)} |")
        return out

    L += site_table("all")
    L.append("")
    L.append("Set fit (controls outside the lockbox, for calibrating the renderer):")
    L.append("")
    L += site_table("fit")
    L.append("")
    L.append("Cohorts: angle between the HC and the WHC mean up per site, and the vertical axis of each cohort:")
    L.append("")
    L.append("| site | HC axis | WHC axis | angle HC vs WHC (deg) | HC spread (deg) | WHC spread (deg) |")
    L.append("|---|---|---|---|---|---|")
    for s in CANON_ORDER:
        h = summ[(summ.set == "HC") & (summ.site == s)]
        w = summ[(summ.set == "WHC") & (summ.site == s)]
        if len(h) and len(w):
            h, w = h.iloc[0], w.iloc[0]
            L.append(f"| {s} | {tx(h.get('vertical_axis'))} | {tx(w.get('vertical_axis'))} | {f1(S['cohort_mean_up_angle_deg'].get(s))} | {f1(h.get('up_spread_median_deg'))} | {f1(w.get('up_spread_median_deg'))} |")
    if Gv["outliers_over_45deg"]:
        L.append("")
        L.append(f"Static estimates more than {int(S['thresholds']['outlier_deg'])} deg from their site mean (all set): " +
                 "; ".join(f"{r['participant']} {r['site']} {r['vertical_axis']} {f1(r['angle_to_site_mean_deg'])} deg" for r in Gv["outliers_over_45deg"]) + ".")
    L.append("")
    L.append("## Gyroscope principal axis in walking (SelfPace, Walk samples)")
    L.append("")
    Y = S["gyro_axis"]
    L.append(f"Rows: {Y['rows']} (trials without Walk labels, all samples used: {Y['walk_source_all']}). Centred against uncentred leading axis: median {f1(Y['centered_vs_uncentered_median_deg'], 2)} deg, p90 {f1(Y['centered_vs_uncentered_p90_deg'], 2)} deg. "
             f"Estimates more than {int(S['thresholds']['outlier_deg'])} deg from the site axial mean: {Y['outliers_over_45deg_count']} ({', '.join(f'{k} {v}' for k, v in sorted(Y['outliers_over_45deg_by_site'].items())) or 'none'}).")
    L.append("")
    L.append("Angle to up: folded angle between the walking axis and the Balance up of the same participant and site (90 deg: a horizontal axis). QC axis: the gyroscope channel the source's QC viewer (QcAllData.m) shows for that sensor. Up drift: angle between the SelfPace quiet-window up and the Balance up of the same sensor. Set all:")
    L.append("")
    L.append("| site | n | axis mean (x, y, z) | axis (cos) | modal (share) | spread median (deg) | energy share median | angle to up median [p25, p75] (deg) | QC axis, agree share | up drift median / p90 (deg) | gyro RMS median (deg/s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summ[summ.set == "all"].iterrows():
        if not r.get("n_gyro"):
            continue
        L.append(f"| {r.site} | {int(r.n_gyro)} | ({f1(r.pc1_mean_x, 3)}, {f1(r.pc1_mean_y, 3)}, {f1(r.pc1_mean_z, 3)}) | {r.pc1_axis} ({f1(r.pc1_axis_cos, 3)}) | {r.pc1_axis_modal} ({f1(r.pc1_axis_modal_frac, 2)}) | "
                 f"{f1(r.pc1_spread_median_deg)} | {f1(r.pc1_energy_frac_median, 2)} | {f1(r.angle_to_up_median_deg)} [{f1(r.angle_to_up_p25_deg)}, {f1(r.angle_to_up_p75_deg)}] | "
                 f"{r.qc_axis}, {f1(r.qc_axis_agree_frac, 2)} | {f1(r.up_selfpace_vs_balance_median_deg)} / {f1(r.up_selfpace_vs_balance_p90_deg)} | {f1(r.gyro_rms_walk_median_dps)} |")
    L.append("")
    L.append(f"Swing axis (peak rule, a heuristic): the signed axis carrying the larger of the two angular-rate peaks (99th against 1st percentile of the projection, ratio at least {S['thresholds']['swing_ratio']}). It is not used to name body axes. Set all:")
    L.append("")
    L.append("| site | n | peak ratio median | ambiguous | swing axis mean (x, y, z) | swing axis (cos) | resultant | modal (share of n) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in summ[summ.set == "all"].iterrows():
        if not r.get("n_gyro"):
            continue
        L.append(f"| {r.site} | {int(r.n_gyro)} | {f1(r.peak_ratio_median, 2)} | {int(r.swing_ambiguous_count)} | ({f1(r.get('swing_mean_x'), 3)}, {f1(r.get('swing_mean_y'), 3)}, {f1(r.get('swing_mean_z'), 3)}) | "
                 f"{tx(r.get('swing_axis'))} ({f1(r.get('swing_axis_cos'), 3)}) | {f1(r.get('swing_resultant'), 3)} | {tx(r.get('swing_axis_modal'))} ({f1(r.get('swing_axis_modal_frac'), 2)}) |")
    L.append("")
    L.append(f"Body right (swing-phase cue): the sign of the mean angular rate on the walking axis over the Walk samples where the sensor's own foot is off the walkway and the other foot is on it (the sensor's leg in swing, at least {S['thresholds']['min_swing_phase_samples']} samples). In swing the thigh and shank rotate forward and the foot rotates from toe-down to toe-up; both are positive rotations about the body's rightward axis. "
             "The last two columns compare the peak rule with this cue on the trials where both decide. Set all:")
    L.append("")
    L.append("| site | n | decided | swing-phase mean rate, median abs (deg/s) | body right mean (x, y, z) | body right (cos) | resultant | modal (share of n) | peak rule decided | peak rule agrees |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summ[summ.set == "all"].iterrows():
        if not r.get("n_gyro") or not tx(r.get("body_right_axis")):
            continue
        L.append(f"| {r.site} | {int(r.n_gyro)} | {int(r.phase_decided_count)} | {f1(r.swing_phase_abs_mean_median_dps)} | ({f1(r.body_right_mean_x, 3)}, {f1(r.body_right_mean_y, 3)}, {f1(r.body_right_mean_z, 3)}) | "
                 f"{r.body_right_axis} ({f1(r.body_right_cos, 3)}) | {f1(r.body_right_resultant, 3)} | {r.body_right_modal} ({f1(r.body_right_modal_frac, 2)}) | {int(r.peak_phase_both_decided)} | {int(r.peak_phase_agree_count)} |")
    L.append("")
    if S["lateral_axis"]:
        L.append("Lateral axis (pointing away from the body midline: body right for a right-side sensor, its opposite for a left-side sensor), from the swing-phase cue: " +
                 "; ".join(f"{s} {v['lateral_axis']}" for s, v in S["lateral_axis"].items()) + ".")
        L.append("")
        L.append("Segment frame of these sensors in sensor coordinates, from the site means of the all set: up (Balance), left (opposite of the body-right mean, made orthogonal to up), forward = left x up. These three vectors fix the mean orientation of each sensor on its segment; participants scatter around it by the spreads reported above.")
        L.append("")
        L.append("| site | up (x, y, z) | left (x, y, z) | forward (x, y, z) | forward axis (cos) | angle body right to up (deg) |")
        L.append("|---|---|---|---|---|---|")
        for s, v in S["lateral_axis"].items():
            if "segment_up" not in v:
                continue
            vv = lambda k: "(" + ", ".join(f1(c, 3) for c in v[k]) + ")"  # noqa: E731
            L.append(f"| {s} | {vv('segment_up')} | {vv('segment_left')} | {vv('segment_forward')} | {v['forward_axis']} ({f1(v['forward_axis_cos'], 3)}) | {f1(v['right_vs_up_deg'])} |")
        L.append("")
    if S["sites_without_dominant_walking_axis"]:
        L.append(f"Sites whose median energy share of the leading walking axis is below {S['thresholds']['weak_axis_energy']} (no single dominant rotation axis; the axis there is unstable across participants): " +
                 "; ".join(f"{s} {f1(v, 2)}" for s, v in S["sites_without_dominant_walking_axis"].items()) + ".")
        L.append("")
    L.append("## Gait speed and cadence (PKMAS walkway, SelfPace)")
    L.append("")
    Wk = S["walk"]
    L.append(f"Controls: {Wk['controls']}; with a PKMAS SelfPace row: {Wk['with_pkmas_selfpace']}; with height: {Wk['with_height']}. PKMAS SelfPace control IDs without a control CSV: {', '.join(Wk['pkmas_selfpace_controls_without_csv']) or 'none'}. Controls without a PKMAS SelfPace row: {', '.join(Wk['controls_without_pkmas_selfpace']) or 'none'}. Duplicate PKMAS keys: {len(Wk['pkmas_duplicate_keys'])}.")
    L.append("")
    src_txt = ", ".join(f"{k or 'none'} {v}" for k, v in Wk["speed_source_counts"].items())
    L.append(f"Cross-check of the join: cadence from the walkway contact columns of the SelfPace IMU CSV against PKMAS cadence, n {Wk['cadence_crosscheck_n']}, Pearson r {f1(Wk['cadence_crosscheck_pearson_r'], 3)}, median absolute difference {f1(Wk['cadence_crosscheck_median_abs_diff'], 2)} steps/min.")
    L.append("")
    L.append(f"Walkway footfalls from the Walkway_X cell lists of the same CSV give speed and step length in cell units. Cell size, median of PKMAS step length over walkway step length in cells, n {Wk['walkway_cell_cm_n']}: {f1(Wk['walkway_cell_cm_estimate'], 3)} cm (p10 {f1(Wk['walkway_cell_cm_p10'], 3)}, p90 {f1(Wk['walkway_cell_cm_p90'], 3)}). "
             f"Walkway speed against PKMAS velocity, n {Wk['walkway_speed_check_n']}: Pearson r {f1(Wk['walkway_speed_check_pearson_r'], 3)}, median absolute difference {f1(Wk['walkway_speed_check_median_abs_diff_m_s'], 3)} m/s, median relative difference {f1(100 * Wk['walkway_speed_check_median_rel_diff'] if Wk['walkway_speed_check_median_rel_diff'] is not None else None, 1)} percent. "
             f"Best speed (PKMAS, else walkway cells) sources: {src_txt}; best cadence (PKMAS, else contacts) for {Wk['cadence_best_n']} controls.")
    L.append("")
    L.append("| set | n | speed p10 / p25 / p50 / p75 / p90 (m/s) | cadence p10 / p25 / p50 / p75 / p90 (steps/min) |")
    L.append("|---|---|---|---|")
    for nm, n_, sp, cd in (("all, PKMAS only", Wk["with_pkmas_selfpace"], Wk["speed_m_s_all"], Wk["cadence_all"]),
                           ("fit, PKMAS only", Wk["n_fit"], Wk["speed_m_s_fit"], Wk["cadence_fit"]),
                           ("all, best", Wk["n_best_all"], Wk["speed_best_all"], Wk["cadence_best_all"]),
                           ("fit, best", Wk["n_best_fit"], Wk["speed_best_fit"], Wk["cadence_best_fit"]),
                           ("HC, best", Wk["by_cohort_best"]["HC"]["n"], Wk["by_cohort_best"]["HC"]["speed"], Wk["by_cohort_best"]["HC"]["cadence"]),
                           ("WHC, best", Wk["by_cohort_best"]["WHC"]["n"], Wk["by_cohort_best"]["WHC"]["speed"], Wk["by_cohort_best"]["WHC"]["cadence"])):
        L.append(f"| {nm} | {n_} | {' / '.join(f1(sp[k], 2) for k in ('p10', 'p25', 'p50', 'p75', 'p90'))} | {' / '.join(f1(cd[k], 1) for k in ('p10', 'p25', 'p50', 'p75', 'p90'))} |")
    L.append("")
    L.append("## Lockbox")
    L.append("")
    Lk = S["lockbox"]
    L.append(f"weargait_pd lockbox persons: {Lk['weargait_persons']}; controls among them: {len(Lk['controls_in_lockbox'])} ({', '.join(Lk['controls_in_lockbox'])}). They are measured here (the task asks for every control) and flagged in_lockbox in every CSV; the fit set leaves them out. "
             f"Lockbox rows whose group is neither HC nor control although the person sits in CONTROL PARTICIPANTS: {len(Lk['controls_in_lockbox_labelled_non_HC'])} ({', '.join(f'{k} {v}' for k, v in Lk['controls_in_lockbox_labelled_non_HC'].items()) or 'none'}). "
             f"Lockbox IDs with no CSV in either group: {', '.join(Lk['lockbox_ids_without_csv']) or 'none'}.")
    L.append("")
    L.append(f"Runtime: {S['runtime_s']} s.")
    txt = "\n".join(L) + "\n"
    assert "\u2014" not in txt
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
