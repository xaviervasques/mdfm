"""weargait_pd targets for gate G3 and the second half of G2 (plan task B3, CPU part, B3a).

Builds, for every per-trial CSV of the tasks SelfPace, HurriedPace, SelfPace_mat,
HurriedPace_mat, SelfPace_matTURN, TUG and TandemGait (controls and PD), the reference
quantities that the IMU-only body model is scored against in G3 (design section 9: stride
length MAE against the walkway, gait speed MAE, initial contact against insole events):
task phases, walkway footfall events and positions, insole events, PKMAS walkway metrics
and the presence of the Xsens orientation columns.

Reads, read-only, under <moveatlas>/datasets/imu/weargait_pd/raw/Version 1/:
  CONTROL PARTICIPANTS/CSV files and PD PARTICIPANTS/CSV files (<ID>_<task>.csv, 100 Hz,
  columns addressed by name), Walkway-derived metrics/PKMAS Walkway Gait Metrics - HP+SP.csv,
  CONTROLS - Demographic+Clinical - datasetV1.csv and PD - Demographic+Clinical -
  datasetV1.csv (group of each participant), the Real-world context tasks folder (listing
  only), the release manifest.csv of each main participant folder and the file names of its MAT
  subfolder (listing only). No MAT file is opened. Also reads reports/lockbox/lockbox_persons.csv of this tree.
  The raw sensor prefix to canonical site map is parsed as text from
  <moveatlas>/build_canonical_imu.py (WEARGAIT_LOCMAP); no MoveAtlas code is imported or run.

Writes <out> (default reports/weargait_targets/): trials.csv, events.csv, phases.csv,
pkmas_match.csv, summary.json, summary.md, _provenance.json, and a per-trial cache under
<out>/cache/ (gitignored: reports/**/cache/) that makes the run resumable, with the full
insole event list in cache/insole_events.csv.

Options: --workers N (parallel trial readers, default 4), --limit N (first N trials only, test
runs), --no-cache (recompute every trial).

Definitions
- Trial: one CSV file <ID>_<task>.csv of the two main CSV folders. trial_id is the file stem.
  person_id is "weargait:<ID>" with the ID exactly as in the file name (for example
  "weargait:NLS901 (control)"), the form used by reports/lockbox/lockbox_persons.csv.
- group: "control" when the ID without the " (control)" suffix is a Subject ID of the
  CONTROLS demographic table, "PD" when it is one of the PD table ("conflict" if both,
  "unknown" if neither). folder_group records the folder the file sits in.
- lockbox: person_id is a weargait_pd row of reports/lockbox/lockbox_persons.csv. Lockbox
  trials stay in every table (flag lockbox true); every statistic of summary.md is computed on
  non-lockbox trials only and lockbox rows are never summarised.
- Time: the Time column ('<value> sec'). A sample time that is not finite is replaced by the
  sample index times 0.01 s (counted as time_nonfinite).
- Phases: maximal runs of equal GeneralEvent labels (missing label: "(missing)"). start_s is
  the time of the first sample, end_s the time of the last sample plus 0.01 s, duration_s
  = samples times 0.01 s. walk_s, turn_s and tandemwalk_s sum the runs labelled Walk, Turn and
  TandemWalk. Walking labels (for passes and completeness): Walk and TandemWalk.
- Walkway contact of a foot: the "L Foot Contact" or "R Foot Contact" column > 0 (NaN reads
  as no contact and is counted). Contact runs separated by fewer than 5 samples (50 ms) are
  merged. A footfall is one contact run: initial contact (IC) at its first sample, foot-off
  (FO) at the first sample after it. An IC at sample 0 or an FO at the end of the trial is
  censored (not an event; the foot was already or still on the walkway).
- Footprint of a footfall: the cells (Walkway_X, Walkway_Y, pipe-separated lists aligned with
  WalkwayFoot) labelled with that foot over the samples of the contact run. Recorded: x_min,
  x_max, x centroid and y mean over the distinct cells, the number of distinct cells, and ic_x
  (mean x of that foot's cells over the first 5 samples of contact). Cells labelled "Other" or
  "Other 1" (not a foot) are counted per trial and otherwise ignored. X runs along the walkway.
- Partial footfall: the footprint touches x = 0 or the largest x seen in any footprint of
  any non-lockbox trial (the two ends of the walkway): part of the foot may lie off the walkway.
- Cadence passes: the non-censored walkway ICs of both feet sorted in time, split where two
  consecutive ICs are more than 1.5 s apart; cadence_contacts = 60 * sum(n - 1) / sum(duration)
  over passes with at least 3 ICs (the rule of probe_weargait_sensor_frames.py).
- Straight passes: the non-censored ICs with a footprint whose GeneralEvent label is a walking
  label, split where two consecutive ICs are more than 1.5 s apart or fall in different
  GeneralEvent runs. Direction d = sign of the x centroid of the last footfall minus that of
  the first. Heel position: x_min of the footprint when d > 0, x_max when d < 0 (the rear edge
  of the footprint along the direction of travel). Step: two consecutive footfalls of opposite
  feet in a pass; stride: two footfalls of the same foot with exactly one footfall of the other
  foot between them; both ends must be non-partial; length = |heel difference| along x, in
  cells. Speed: over passes with at least 3 non-partial footfalls, sum of |heel(last) -
  heel(first)| over the sum of their IC time differences (cells/s). Cell size: the median over
  non-lockbox SelfPace trials matched to PKMAS of PKMAS mean stride length (cm) over the
  walkway mean stride length (cells); it converts cells to cm and m/s. HurriedPace trials do
  not enter the fit and give the held-out agreement.
- PKMAS match: the PKMAS file holds one row per participant and task (trial means and
  #Samples, no per-step rows). Task names are stripped of spaces. A row matches the CSV trial
  whose task has the same name (SelfPace -> <ID>_SelfPace.csv, HurriedPace ->
  <ID>_HurriedPace.csv) and whose ID without " (control)" equals the PKMAS Participant ID.
  IDs with a "_2" suffix (a second visit) are not matched; tasks other than SelfPace and
  HurriedPace are out of scope. Evidence for the pairing: the walkway cadence of the matched
  trial against PKMAS cadence, and the same for the _mat alternatives. cadence_verified: the
  walkway cadence of the matched trial is within 1 steps/min of PKMAS cadence; agreement is
  reported over every matched pair and over the cadence-verified pairs.
- Insole missing samples (descriptive): per trial, the leading and trailing NaN blocks of
  TotalForce (longest of the two feet, s) and the interior NaN runs (summed over feet: all,
  single samples, longer than 10 samples).
- Orientation: for each of the 13 Xsens sensors, present when <prefix>_Roll, _Pitch and _Yaw
  are all columns of the file; finite fraction = share of samples with the three values
  finite; zero fraction = share of samples with the three values exactly 0. ori_ok when present
  and the finite fraction is at least 0.99.
- Insole to walkway cross-correlation lag (a sync diagnostic in the manner of the source QC
  function QcFx_EstimateSyncDelay_v2): TotalForce with missing samples filled forward then
  backward, walkway "<side> Foot Pressure" with NaN as 0; c(k) = sum over walkway-loaded
  samples i of P[i] * F[i + k], k in -100..100 samples; lag = argmax, in ms; positive: the
  insole profile comes later than the walkway profile.
- Source listing (names only): the Synapse manifest.csv of each main participant folder lists
  the CSV files of the release. A listed CSV of the seven tasks that is not on disk is counted
  per person; for a person of the demographic tables without any CSV trial, a MAT file of the
  folder's MAT subfolder named <ID>.mat is reported by name. No MAT file is opened.
- PKMAS duplicate rows: two PKMAS rows whose metric columns (every column after 'PD vs
  Control') hold identical text; each names the other in duplicate_metrics_of.
- Pooled-shift residual: the matched IC differences minus the signed median of the same set
  of matched ICs (one shift for the whole set, unlike the per-trial offset below).
- Cause of an insole_missing_near_event state: the leading NaN block, the trailing NaN block or
  an interior NaN run of that foot's TotalForce within 300 ms of the walkway event (checked in
  that order).
- Summary scope: summary.md and summary.json describe non-lockbox trials only, apart from file
  and row inventories (numbers of CSV files, PKMAS rows and their match status, group labels).

Insole event rule (fixed and written here before any insole against walkway agreement was
computed; the insole force unit is not stated in the raw folder, so the rule is relative)
- F = LTotalForce or RTotalForce of the trial. A foot is scorable when F has at least 200
  finite samples and F95 > F05, where F05 and F95 are the 5th and 95th percentiles of the
  finite F over the whole trial.
- Median stance force Fst = median of the finite F with F >= F05 + 0.5 * (F95 - F05).
- Threshold thr = F05 + 0.10 * (Fst - F05). A finite sample is loaded when F > thr.
- Cleaning, inside each run of finite samples: unloaded runs shorter than 10 samples (100 ms)
  with a loaded sample on both sides become loaded; then loaded runs shorter than 10 samples
  become unloaded.
- IC: a loaded sample whose previous sample is finite and unloaded; FO: an unloaded sample
  whose previous sample is finite and loaded. Transitions next to a missing sample or at the
  trial ends are not events. Event time: the Time value of that sample (same convention as
  the walkway: first sample in the new state).
- Pairing: each non-censored walkway IC (FO) is paired with the nearest insole IC (FO) of the
  same foot within 1 s; the pair is matched when the absolute difference is at most 300 ms.
  diff_ms = insole time - walkway time. States: matched; no_insole_event_within_300ms;
  insole_missing_near_event (a non-finite insole sample within 300 ms of the walkway event);
  insole_invalid (foot not scorable); no_insole_columns.
- Offset-removed difference: within a trial with at least 4 matched ICs, the matched IC
  differences minus their trial median (separates a per-trial clock offset from event timing).

Completeness of a trial's targets (per trial)
- c_phases: a walking label among the phases; c_walkway: at least 2 non-censored walkway ICs
  per foot; c_insole: at least 2 matched ICs per foot; c_orientation: 13 sensors ori_ok;
  c_pkmas (SelfPace and HurriedPace only): a matched PKMAS row with finite velocity, stride
  length and cadence. complete: every applicable component true.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import parse_args, write_provenance  # noqa: E402

RULE_VERSION = "b3a-1"      # the insole event rule and the walkway definitions
CACHE_VERSION = "b3a-cache-2"  # per-trial cache layout (adds the insole NaN structure)
CADENCE_CHECK = 1.0         # steps/min: a PKMAS pairing is cadence-verified when the walkway cadence is this close
TASKS = ["SelfPace", "HurriedPace", "SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN", "TUG", "TandemGait"]
PKMAS_TASKS = {"SelfPace": "SelfPace", "HurriedPace": "HurriedPace"}
PKMAS_ALT = {"SelfPace": ["SelfPace_mat", "SelfPace_matTURN"], "HurriedPace": ["HurriedPace_mat"]}
CANON_ORDER = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R",
               "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R"]
WALK_LABELS = ("Walk", "TandemWalk")
DT = 0.01
WW_MIN_GAP = 5
HEEL_SAMPLES = 5
PASS_GAP_S = 1.5
INS_MIN_FINITE = 200
INS_LO_PCT, INS_HI_PCT = 5, 95
INS_STANCE_FRAC = 0.5
INS_THR_FRAC = 0.10
INS_MIN_RUN = 10
MATCH_MS = 300
SEARCH_MS = 1000
MIN_MATCHED_FOR_OFFSET = 4
ORI_OK_FRAC = 0.99
XCORR_MAX_LAG = 100
MIN_IC_PER_FOOT = 2
ID_RE = re.compile(r"^((?:WHC|HC|NLS|WPD)\d+(?: \(control\))?)(?:_(.*))?$")
STR_COLS = ("Time", "GeneralEvent", "Walkway_X", "Walkway_Y", "WalkwayFoot")
BASE_COLS = ["Time", "GeneralEvent", "L Foot Contact", "R Foot Contact", "L Foot Pressure", "R Foot Pressure",
             "Walkway_X", "Walkway_Y", "WalkwayFoot", "LTotalForce", "RTotalForce"]


def insole_rule_text():
    d = __doc__
    a = d.index("Insole event rule")
    b = d.index("Completeness of a trial")
    return d[a:b]


# ---------------------------------------------------------------- small helpers
def base_id(pid):
    return re.sub(r"\s*\(control\)\s*$", "", pid).strip()


def cohort_of(pid):
    m = re.match(r"^(WHC|HC|NLS|WPD)", pid)
    return m.group(1) if m else "other"


def participant_of(fname):
    stem = os.path.splitext(fname)[0]
    m = ID_RE.match(stem)
    if m:
        return m.group(1), m.group(2) or ""
    parts = stem.split("_")
    return parts[0], "_".join(parts[1:])


def runs_of(b):
    """Runs of True in a boolean array as (start, end_exclusive)."""
    d = np.diff(np.concatenate([[0], np.asarray(b, np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def fnum(x, nd=4):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if np.isfinite(xf) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------- per-trial worker
def insole_events(F, t):
    out = {"valid": False, "n_finite": 0, "lo": None, "hi": None, "fst": None, "thr": None,
           "ic_i": [], "fo_i": [], "fin": None}
    if F is None:
        return out
    fin = np.isfinite(F)
    out["n_finite"] = int(fin.sum())
    out["fin"] = fin
    if fin.sum() < INS_MIN_FINITE:
        return out
    x = F[fin]
    lo, hi = np.percentile(x, [INS_LO_PCT, INS_HI_PCT])
    out["lo"], out["hi"] = float(lo), float(hi)
    if not hi > lo:
        return out
    st = x[x >= lo + INS_STANCE_FRAC * (hi - lo)]
    fst = float(np.median(st))
    thr = lo + INS_THR_FRAC * (fst - lo)
    out["fst"], out["thr"] = fst, float(thr)
    out["valid"] = True
    state = np.full(len(F), -1, np.int8)
    state[fin] = (F[fin] > thr).astype(np.int8)
    for a, b in runs_of(fin):
        s_ = state[a:b].copy()
        for p, q in runs_of(s_ == 0):
            if p > 0 and q < len(s_) and q - p < INS_MIN_RUN:
                s_[p:q] = 1
        for p, q in runs_of(s_ == 1):
            if q - p < INS_MIN_RUN:
                s_[p:q] = 0
        state[a:b] = s_
    prev, cur = state[:-1], state[1:]
    out["ic_i"] = (np.flatnonzero((prev == 0) & (cur == 1)) + 1).tolist()
    out["fo_i"] = (np.flatnonzero((prev == 1) & (cur == 0)) + 1).tolist()
    return out


def xcorr_lag_ms(F, P):
    if F is None or P is None:
        return None
    x = pd.Series(F).ffill().bfill().to_numpy(float)
    if not np.isfinite(x).all():
        return None
    y = np.nan_to_num(P)
    idx = np.flatnonzero(y > 0)
    if len(idx) < 50:
        return None
    lags = np.arange(-XCORR_MAX_LAG, XCORR_MAX_LAG + 1)
    c = np.empty(len(lags))
    for i, k in enumerate(lags):
        j = idx + k
        m = (j >= 0) & (j < len(x))
        c[i] = float(np.dot(y[idx[m]], x[j[m]]))
    return int(lags[int(np.argmax(c))]) * 10


def footprint(WX, WY, WF, s, e, side):
    cells = set()
    icx = []
    mism = 0
    for j in range(s, e):
        xv, yv, fv = WX[j], WY[j], WF[j]
        if not (isinstance(xv, str) and isinstance(yv, str) and isinstance(fv, str)):
            continue
        xl, yl, fl = xv.split("|"), yv.split("|"), fv.split("|")
        if not (len(xl) == len(yl) == len(fl)):
            mism += 1
            continue
        for a, b, c in zip(xl, yl, fl):
            if c.strip() != side:
                continue
            try:
                xx, yy = int(float(a)), int(float(b))
            except ValueError:
                continue
            cells.add((xx, yy))
            if j < s + HEEL_SAMPLES:
                icx.append(xx)
    if not cells:
        return {"ncells": 0, "xmin": None, "xmax": None, "xc": None, "ymean": None, "icx": None, "mism": mism}
    arr = np.array(sorted(cells), float)
    return {"ncells": len(cells), "xmin": int(arr[:, 0].min()), "xmax": int(arr[:, 0].max()),
            "xc": float(arr[:, 0].mean()), "ymean": float(arr[:, 1].mean()),
            "icx": float(np.mean(icx)) if icx else None, "mism": mism}


def process_trial(job):
    path, cache_path, header = job["path"], job["cache"], job["header"]
    st = os.stat(path)
    sig = {"size": st.st_size, "mtime": int(st.st_mtime), "rule": RULE_VERSION, "cache": CACHE_VERSION}
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("_sig") == sig:
                c["_from_cache"] = True
                return c
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    hs = set(header)
    prefixes = job["prefixes"]
    ori_cols = [f"{p}_{a}" for p in prefixes for a in ("Roll", "Pitch", "Yaw")]
    use = [c for c in BASE_COLS + ori_cols if c in hs]
    dtypes = {c: str for c in STR_COLS if c in hs}
    df = pd.read_csv(path, usecols=use, dtype=dtypes, encoding="utf-8", low_memory=False)
    n = len(df)
    res = {"_sig": sig, "trial_id": job["trial_id"], "n": n}
    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"].astype(str).str.replace("sec", "", regex=False).str.strip(), errors="coerce").to_numpy(float)
    else:
        t = np.full(n, np.nan)
    bad_t = ~np.isfinite(t)
    res["time_nonfinite"] = int(bad_t.sum())
    t = np.where(bad_t, np.arange(n) * DT, t)
    dt = np.diff(t)
    res["dt_median"] = float(np.median(dt)) if len(dt) else None
    res["dt_irregular"] = int(np.sum(np.abs(dt - DT) > 1e-6)) if len(dt) else 0
    res["duration_s"] = float(n * DT)

    # phases
    has_ev = "GeneralEvent" in df.columns
    res["has_events"] = has_ev
    ev = df["GeneralEvent"].fillna("(missing)").astype(str).str.strip().to_numpy() if has_ev else np.array(["(no column)"] * n)
    change = np.concatenate([[True], ev[1:] != ev[:-1]]) if n else np.array([], bool)
    run_id = np.cumsum(change) - 1
    starts = np.flatnonzero(change).tolist() + [n]
    res["phases"] = [[str(ev[a]), float(t[a]), float(t[b - 1] + DT), int(b - a)] for a, b in zip(starts[:-1], starts[1:])]

    # walkway
    has_ww = all(c in df.columns for c in ("L Foot Contact", "R Foot Contact"))
    res["has_walkway"] = has_ww
    has_cells = all(c in df.columns for c in ("Walkway_X", "Walkway_Y", "WalkwayFoot"))
    res["has_walkway_cells"] = has_cells
    WX = df["Walkway_X"].to_numpy(object) if has_cells else None
    WY = df["Walkway_Y"].to_numpy(object) if has_cells else None
    WF = df["WalkwayFoot"].to_numpy(object) if has_cells else None
    other = 0
    if has_cells:
        for fv in WF:
            if isinstance(fv, str) and "Other" in fv:
                other += 1
    res["other_cell_samples"] = other
    # insole
    has_ins = all(c in df.columns for c in ("LTotalForce", "RTotalForce"))
    res["has_insole"] = has_ins
    ins = {}
    res["insole"] = {}
    res["xcorr_lag_ms"] = {}
    res["ww_nan"] = {}
    res["footfalls"] = {}
    half = int(round(MATCH_MS / 1000 / DT))
    for side in ("L", "R"):
        F = pd.to_numeric(df[f"{side}TotalForce"], errors="coerce").to_numpy(float) if has_ins else None
        ie = insole_events(F, t)
        ins[side] = ie
        nan_s = {"lead": 0, "trail": 0, "interior_runs": 0, "interior_single": 0, "interior_long": 0}
        if F is not None and n:
            nr = runs_of(~np.isfinite(F))
            for a, b in nr:
                if a == 0:
                    nan_s["lead"] = b - a
                elif b == n:
                    nan_s["trail"] = b - a
                else:
                    nan_s["interior_runs"] += 1
                    nan_s["interior_single"] += int(b - a == 1)
                    nan_s["interior_long"] += int(b - a > INS_MIN_RUN)
        res["insole"][side] = {"valid": ie["valid"], "n_finite": ie["n_finite"], "lo": ie["lo"], "hi": ie["hi"], "nan": nan_s,
                               "fst": ie["fst"], "thr": ie["thr"],
                               "ic_i": ie["ic_i"], "fo_i": ie["fo_i"],
                               "ic_t": [float(t[i]) for i in ie["ic_i"]], "fo_t": [float(t[i]) for i in ie["fo_i"]],
                               "ic_label": [str(ev[i]) for i in ie["ic_i"]], "fo_label": [str(ev[i]) for i in ie["fo_i"]]}
        P = pd.to_numeric(df[f"{side} Foot Pressure"], errors="coerce").to_numpy(float) if f"{side} Foot Pressure" in df.columns else None
        res["xcorr_lag_ms"][side] = xcorr_lag_ms(F, P) if (has_ins and P is not None) else None
        ffs = []
        if has_ww:
            c = pd.to_numeric(df[f"{side} Foot Contact"], errors="coerce").to_numpy(float)
            res["ww_nan"][side] = int(np.isnan(c).sum())
            runs = []
            for s, e in runs_of(np.nan_to_num(c) > 0):
                if runs and s - runs[-1][1] < WW_MIN_GAP:
                    runs[-1] = (runs[-1][0], e)
                else:
                    runs.append((s, e))
            fin = ie["fin"]
            for s, e in runs:
                ff = {"s": int(s), "e": int(e), "t_ic": float(t[s]), "t_fo": float(t[e]) if e < n else None,
                      "ic_cens": s == 0, "fo_cens": e >= n, "label_ic": str(ev[s]),
                      "label_fo": str(ev[e]) if e < n else "", "run_ic": int(run_id[s])}
                if fin is not None:
                    a0, a1 = max(0, s - half), min(n, s + half + 1)
                    ff["ins_fin_ic"] = bool(fin[a0:a1].all())
                    if e < n:
                        b0, b1 = max(0, e - half), min(n, e + half + 1)
                        ff["ins_fin_fo"] = bool(fin[b0:b1].all())
                    else:
                        ff["ins_fin_fo"] = False
                else:
                    ff["ins_fin_ic"] = ff["ins_fin_fo"] = False
                if has_cells:
                    ff.update(footprint(WX, WY, WF, s, e, side))
                else:
                    ff.update({"ncells": 0, "xmin": None, "xmax": None, "xc": None, "ymean": None, "icx": None, "mism": 0})
                ffs.append(ff)
        res["footfalls"][side] = ffs
    # orientation
    ori = {}
    for site, p in zip(CANON_ORDER, prefixes):
        cols = [f"{p}_{a}" for a in ("Roll", "Pitch", "Yaw")]
        if not all(cc in df.columns for cc in cols):
            ori[site] = {"present": False, "finite": None, "zero": None}
            continue
        E = np.stack([pd.to_numeric(df[cc], errors="coerce").to_numpy(float) for cc in cols], 1)
        finr = np.all(np.isfinite(E), 1)
        zr = finr & np.all(E == 0, 1)
        ori[site] = {"present": True, "finite": float(finr.mean()) if n else 0.0, "zero": float(zr.mean()) if n else 0.0}
    res["ori"] = ori
    res["read_s"] = round(time.time() - t0, 2)
    res = jsonable(res)
    if cache_path:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f)
        os.replace(tmp, cache_path)
    res["_from_cache"] = False
    return res


# ---------------------------------------------------------------- side tables
def read_locmap(build_imu):
    with open(build_imu, encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"WEARGAIT_LOCMAP\s*=\s*(\{.*?\})", txt, re.S)
    if not m:
        sys.exit("WEARGAIT_LOCMAP not found in build_canonical_imu.py")
    locmap = ast.literal_eval(m.group(1))
    inv = {v: k for k, v in locmap.items()}
    if sorted(inv) != sorted(CANON_ORDER):
        sys.exit(f"site map differs from the 13 canonical sites: {sorted(inv)}")
    return [inv[s] for s in CANON_ORDER]


def read_demographics(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Subject ID")
    out = set()
    for r in rows[hi + 1:]:
        if r and r[0].strip():
            out.add(r[0].strip())
    return out


def read_lockbox(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "weargait_pd":
                out[r["person_id"]] = r["group"]
    return out


PKMAS_COLS = [("velocity_cm_s", "Velocity (cm./sec.)", ""),
              ("stride_length_cm_mean", "Stride Length (cm.)", "Mean"),
              ("stride_length_n", "Stride Length (cm.)", "#Samples"),
              ("step_length_cm_mean", "Step Length (cm.)", "Mean"),
              ("step_length_n", "Step Length (cm.)", "#Samples"),
              ("cadence_steps_min", "Cadence (steps/min.)", ""),
              ("ambulation_time_s", "Ambulation Time (sec.)", ""),
              ("step_time_s_mean", "Step Time (sec.)", "Mean"),
              ("stride_time_s_mean", "Stride Time (sec.)", "Mean"),
              ("stride_width_cm_mean", "Stride Width (cm.)", "Mean"),
              ("stride_velocity_cm_s_mean", "Stride Velocity (cm./sec.)", "Mean")]


def read_pkmas(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    h0, h1 = rows[0], rows[1]

    def col(name, sub):
        for i, (a, b) in enumerate(zip(h0, h1)):
            if a.strip() == name and b.strip() == sub:
                return i
        raise KeyError((name, sub))

    idx = {k: col(nm, sub) for k, nm, sub in PKMAS_COLS}
    recs = []
    blank = 0
    for ri, r in enumerate(rows[2:]):
        if len(r) < 3 or not r[1].strip():
            blank += 1
            continue
        pid = r[1].strip()
        m = re.match(r"^(.*?)_(\d+)$", pid)
        rec = {"pkmas_row": ri + 3, "task_raw": r[0], "task": r[0].strip(), "participant": pid,
               "participant_base": m.group(1) if m else pid, "visit": int(m.group(2)) if m else 1,
               "pd_vs_control": r[2].strip()}
        for k, i in idx.items():
            try:
                rec[k] = float(r[i]) if i < len(r) and r[i].strip() != "" else float("nan")
            except ValueError:
                rec[k] = float("nan")
        rec["_metrics_key"] = tuple(x.strip() for x in r[3:])
        recs.append(rec)
    by_key = defaultdict(list)
    for rec in recs:
        by_key[rec["_metrics_key"]].append(rec)
    for rec in recs:
        rec["duplicate_metrics_of"] = "; ".join(f"{o['task']} {o['participant']} (row {o['pkmas_row']})"
                                                for o in by_key[rec["_metrics_key"]] if o is not rec and any(rec["_metrics_key"]))
    for rec in recs:
        del rec["_metrics_key"]
    headers = sorted({(a.strip(), b.strip()) for a, b in zip(h0, h1)})
    return recs, blank, headers


# ---------------------------------------------------------------- geometry of a trial (main process)
def trial_geometry(res, x_edge_hi):
    """Passes, heel positions, steps, strides and speed in walkway cells. Returns per-footfall
    annotations keyed by (side, index) and trial aggregates."""
    ffs = []
    for side in ("L", "R"):
        for k, f in enumerate(res["footfalls"].get(side, [])):
            if not f["ic_cens"]:
                ffs.append((f["t_ic"], side, k, f))
    ffs.sort(key=lambda z: (z[0], z[1]))
    out = {"n_ic": len(ffs)}
    # cadence passes
    tt = np.array([z[0] for z in ffs])
    steps = dur = 0.0
    npass = 0
    if len(tt) >= 3:
        cuts = np.flatnonzero(np.diff(tt) > PASS_GAP_S) + 1
        for seg in np.split(tt, cuts):
            if len(seg) >= 3 and seg[-1] > seg[0]:
                steps += len(seg) - 1
                dur += seg[-1] - seg[0]
                npass += 1
    out["cadence_contacts"] = 60.0 * steps / dur if dur > 0 else None
    out["n_cadence_passes"] = npass
    # straight passes
    seq = [z for z in ffs if z[3]["label_ic"] in WALK_LABELS and z[3]["ncells"] > 0]
    passes, cur = [], []
    for z in seq:
        if cur and (z[0] - cur[-1][0] > PASS_GAP_S or z[3]["run_ic"] != cur[-1][3]["run_ic"]):
            passes.append(cur)
            cur = []
        cur.append(z)
    if cur:
        passes.append(cur)
    ann = {}
    strides, stepl = [], []
    dist = sdur = 0.0
    n_speed_pass = 0
    for pi, P in enumerate(passes):
        d = int(np.sign(P[-1][3]["xc"] - P[0][3]["xc"])) if len(P) >= 2 else 0
        heel = []
        for z in P:
            f = z[3]
            partial = bool(f["xmin"] <= 0 or f["xmax"] >= x_edge_hi)
            h = (f["xmin"] if d > 0 else f["xmax"]) if d != 0 else None
            heel.append(h)
            ann[(z[1], z[2])] = {"pass_id": pi, "direction": d, "heel_x": h, "partial": partial,
                                 "step_cells": None, "stride_cells": None}
        if d == 0:
            continue
        for i in range(1, len(P)):
            a, b = P[i - 1], P[i]
            if a[1] != b[1] and not ann[(a[1], a[2])]["partial"] and not ann[(b[1], b[2])]["partial"]:
                v = abs(heel[i] - heel[i - 1])
                ann[(b[1], b[2])]["step_cells"] = v
                stepl.append(v)
            if i >= 2:
                c0 = P[i - 2]
                if c0[1] == b[1] and a[1] != b[1] and not ann[(c0[1], c0[2])]["partial"] and not ann[(b[1], b[2])]["partial"]:
                    v = abs(heel[i] - heel[i - 2])
                    ann[(b[1], b[2])]["stride_cells"] = v
                    strides.append(v)
        full = [i for i, z in enumerate(P) if not ann[(z[1], z[2])]["partial"]]
        if len(full) >= 3:
            i0, i1 = full[0], full[-1]
            dd = P[i1][0] - P[i0][0]
            if dd > 0:
                dist += abs(heel[i1] - heel[i0])
                sdur += dd
                n_speed_pass += 1
    out["n_straight_passes"] = len(passes)
    out["n_speed_passes"] = n_speed_pass
    out["speed_cells_s"] = dist / sdur if sdur > 0 else None
    out["stride_cells_mean"] = float(np.mean(strides)) if strides else None
    out["n_strides"] = len(strides)
    out["step_cells_mean"] = float(np.mean(stepl)) if stepl else None
    out["n_steps"] = len(stepl)
    return ann, out


# ---------------------------------------------------------------- statistics helpers
def qstats(x, prefix=""):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"{prefix}n": 0, f"{prefix}median": None, f"{prefix}p90": None, f"{prefix}mean": None}
    return {f"{prefix}n": int(len(x)), f"{prefix}median": float(np.median(x)),
            f"{prefix}p90": float(np.percentile(x, 90)), f"{prefix}mean": float(np.mean(x))}


def pair_stats(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return {"n": int(len(a)), "mae": None, "median_abs": None, "p90_abs": None, "bias": None, "pearson_r": None}
    d = b - a
    return {"n": int(len(a)), "mae": float(np.mean(np.abs(d))), "median_abs": float(np.median(np.abs(d))),
            "p90_abs": float(np.percentile(np.abs(d), 90)), "bias": float(np.mean(d)),
            "pearson_r": float(np.corrcoef(a, b)[0, 1])}


def pct(x, p):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else None


# ---------------------------------------------------------------- main
def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--limit", type=int, default=0)
    pre.add_argument("--workers", type=int, default=4)
    pre.add_argument("--no-cache", action="store_true")
    opt, rest = pre.parse_known_args()
    if "--out" not in rest:
        rest += ["--out", os.path.join(os.path.dirname(HERE), "reports", "weargait_targets")]
    sys.argv = [sys.argv[0]] + rest
    args = parse_args(__doc__.splitlines()[0])
    out = args.out
    T0 = time.time()

    def log(*a):
        print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

    raw = f"{args.moveatlas}/datasets/imu/weargait_pd/raw"
    v1 = f"{raw}/Version 1"
    group_dirs = {"control": f"{v1}/CONTROL PARTICIPANTS/CSV files", "PD": f"{v1}/PD PARTICIPANTS/CSV files"}
    pkmas_path = f"{v1}/Walkway-derived metrics/PKMAS Walkway Gait Metrics - HP+SP.csv"
    demo_paths = {"control": f"{v1}/CONTROLS - Demographic+Clinical - datasetV1.csv",
                  "PD": f"{v1}/PD - Demographic+Clinical - datasetV1.csv"}
    rw_dir = f"{v1}/Real-world context tasks"
    lockbox_path = os.path.join(os.path.dirname(HERE), "reports", "lockbox", "lockbox_persons.csv").replace("\\", "/")
    cache_dir = os.path.join(out, "cache", "trials")
    os.makedirs(cache_dir, exist_ok=True)

    prefixes = read_locmap(f"{args.moveatlas}/build_canonical_imu.py")
    site_prefix = dict(zip(CANON_ORDER, prefixes))
    log("site map:", site_prefix)
    demo = {g: read_demographics(p) for g, p in demo_paths.items()}
    lockbox = read_lockbox(lockbox_path)
    log(f"demographic IDs: control {len(demo['control'])}, PD {len(demo['PD'])}; lockbox weargait persons {len(lockbox)}")

    def group_of(pid):
        b = base_id(pid)
        inc, inp = b in demo["control"], b in demo["PD"]
        return "conflict" if inc and inp else ("control" if inc else ("PD" if inp else "unknown"))

    # --- listing
    jobs = []
    skipped = Counter()
    dup_header_files = []
    for fg, d in group_dirs.items():
        for f in sorted(os.listdir(d)):
            if f.startswith("._") or not f.endswith(".csv") or f == "manifest.csv":
                continue
            pid, task = participant_of(f)
            if task not in TASKS:
                skipped[task] += 1
                continue
            p = os.path.join(d, f)
            with open(p, encoding="utf-8") as fh:
                header = next(csv.reader([fh.readline()]))
            if len(set(header)) != len(header):
                dup_header_files.append(f)
            stem = os.path.splitext(f)[0]
            jobs.append({"path": p, "file": f, "trial_id": stem, "participant": pid, "task": task,
                         "folder_group": fg, "header": header, "prefixes": prefixes,
                         "cache": None if opt.no_cache else os.path.join(cache_dir, stem + ".json")})
    log(f"trials listed: {len(jobs)}; other CSVs skipped by task: {dict(skipped)}")
    rw_counts = Counter()
    for root, dirs, fs in os.walk(rw_dir):
        dirs[:] = sorted(dd for dd in dirs if not dd.startswith("._"))
        if os.path.basename(root) != "CSV files":
            continue
        rel = os.path.relpath(root, rw_dir).replace("\\", "/")
        for f in fs:
            if f.startswith("._") or not f.endswith(".csv") or f == "manifest.csv":
                continue
            rw_counts[f"{rel.split('/')[0]} | {rel.split('/')[1]} | {participant_of(f)[1]}"] += 1
    # --- source listing: CSVs of the release manifest not on disk; MAT files (names only) of persons without CSVs
    manifest_missing = defaultdict(list)
    table_without_csv = {}
    for fg, d in group_dirs.items():
        gdir = os.path.dirname(d)
        on_disk = {f for f in os.listdir(d) if not f.startswith("._")}
        man = os.path.join(gdir, "manifest.csv")
        if os.path.isfile(man):
            with open(man, encoding="utf-8", newline="") as fh:
                for r_ in csv.DictReader(fh):
                    nm = (r_.get("name") or "").strip()
                    if not nm.endswith(".csv") or nm == "manifest.csv" or nm in on_disk:
                        continue
                    pid_, task_ = participant_of(nm)
                    if task_ in TASKS:
                        manifest_missing[pid_].append(task_)
        mats = set()
        for sub in sorted(os.listdir(gdir)):
            if sub.startswith("MAT") and os.path.isdir(os.path.join(gdir, sub)):
                mats |= {f for f in os.listdir(os.path.join(gdir, sub)) if f.endswith(".mat") and not f.startswith("._")}
        with_csv = {base_id(j["participant"]) for j in jobs if j["folder_group"] == fg}
        for pid_ in sorted(demo[fg] - with_csv):
            table_without_csv[pid_] = {"group": fg, "mat_file_present": f"{pid_}.mat" in mats,
                                       "lockbox": f"weargait:{pid_}" in lockbox or f"weargait:{pid_} (control)" in lockbox}
    manifest_missing = {p: sorted(v, key=TASKS.index) for p, v in sorted(manifest_missing.items())}
    log(f"release manifest CSVs of the seven tasks not on disk: {sum(len(v) for v in manifest_missing.values())} "
        f"({len(manifest_missing)} persons); table persons without CSV trials: {len(table_without_csv)}")
    if opt.limit:
        jobs = jobs[:opt.limit]
        log(f"--limit {opt.limit}: {len(jobs)} trials")

    # --- per-trial processing (parallel, cached)
    results = {}
    n_cached = 0
    with ProcessPoolExecutor(max_workers=max(1, opt.workers)) as ex:
        futs = {ex.submit(process_trial, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]
            r = fu.result()
            results[j["trial_id"]] = r
            n_cached += int(bool(r.get("_from_cache")))
            if k % 25 == 0 or k == len(jobs):
                log(f"trials done {k}/{len(jobs)} (from cache {n_cached}); last {j['trial_id']} read {r.get('read_s')} s")
    jobs.sort(key=lambda j: (j["folder_group"], j["participant"], TASKS.index(j["task"])))

    # --- global walkway extent (non-lockbox footprints only)
    lb_trial = {j["trial_id"]: f"weargait:{j['participant']}" in lockbox for j in jobs}
    res_nl = [r for tid_, r in results.items() if not lb_trial[tid_]]
    xmax_all = [f["xmax"] for r in res_nl for s in ("L", "R") for f in r["footfalls"].get(s, []) if f.get("xmax") is not None]
    xmin_all = [f["xmin"] for r in res_nl for s in ("L", "R") for f in r["footfalls"].get(s, []) if f.get("xmin") is not None]
    x_edge_hi = int(max(xmax_all)) if xmax_all else 10 ** 9
    log(f"walkway x extent over non-lockbox footprints: {min(xmin_all) if xmin_all else None} .. {x_edge_hi}")

    # --- PKMAS
    pk, pk_blank, pk_headers = read_pkmas(pkmas_path)
    trial_by_key = {(base_id(j["participant"]), j["task"]): j for j in jobs}
    geo = {}
    ann_all = {}
    for j in jobs:
        ann, g = trial_geometry(results[j["trial_id"]], x_edge_hi)
        geo[j["trial_id"]] = g
        ann_all[j["trial_id"]] = ann
    pk_rows = []
    pk_by_trial = {}
    for rec in pk:
        row = dict(rec)
        row["group_source"] = group_of(rec["participant_base"])
        row["person_id"] = ""
        row["lockbox"] = ""
        csv_task = PKMAS_TASKS.get(rec["task"])
        base_trial = trial_by_key.get((rec["participant_base"], csv_task)) if csv_task else None
        if csv_task is None:
            status, tj = "task_out_of_scope", None
        elif rec["visit"] != 1:
            status, tj = "second_visit_not_matched", None
        elif base_trial is None:
            status, tj = "no_csv_trial", None
        else:
            status, tj = "matched", base_trial
        row["match_status"] = status
        row["csv_task"] = csv_task or ""
        row["trial_id"] = tj["trial_id"] if tj else ""
        if base_trial is not None:
            row["person_id"] = f"weargait:{base_trial['participant']}"
            row["lockbox"] = row["person_id"] in lockbox
            row["base_trial_id"] = base_trial["trial_id"]
            row["base_trial_cadence_contacts"] = geo[base_trial["trial_id"]]["cadence_contacts"]
        else:
            row["base_trial_id"] = ""
            row["base_trial_cadence_contacts"] = None
            cands = [jj for (b, _t), jj in trial_by_key.items() if b == rec["participant_base"]]
            if cands:
                row["person_id"] = f"weargait:{cands[0]['participant']}"
                row["lockbox"] = row["person_id"] in lockbox
        for alt in PKMAS_ALT.get(rec["task"], []):
            aj = trial_by_key.get((rec["participant_base"], alt))
            row[f"alt_cadence_{alt}"] = geo[aj["trial_id"]]["cadence_contacts"] if aj else None
        if tj is not None:
            if tj["trial_id"] in pk_by_trial:
                row["match_status"] = "duplicate_for_trial"
            else:
                pk_by_trial[tj["trial_id"]] = row
        pk_rows.append(row)
    log(f"PKMAS rows {len(pk)} (blank {pk_blank}); matched trials {len(pk_by_trial)}")

    # --- cell size from non-lockbox SelfPace trials
    ratios = []
    for tid, row in pk_by_trial.items():
        if row["task"] == "SelfPace" and not row["lockbox"]:
            sc = geo[tid]["stride_cells_mean"]
            if sc and np.isfinite(row["stride_length_cm_mean"]) and sc > 0:
                ratios.append(row["stride_length_cm_mean"] / sc)
    cell_cm = float(np.median(ratios)) if ratios else float("nan")
    log(f"cell size from {len(ratios)} non-lockbox SelfPace trials: {cell_cm:.4f} cm")

    # --- per-trial tables, events, phases
    trial_rows, event_rows, phase_rows, ins_rows = [], [], [], []
    per_trial_offsets = {}
    gap_cause = Counter()
    half_w = int(round(MATCH_MS / 1000 / DT))
    for j in jobs:
        tid = j["trial_id"]
        r = results[tid]
        g = geo[tid]
        ann = ann_all[tid]
        pid = j["participant"]
        person = f"weargait:{pid}"
        grp = group_of(pid)
        lb = person in lockbox
        common = {"trial_id": tid, "person_id": person, "group": grp, "task": j["task"], "lockbox": lb}
        for k, (lab, a, b, ns) in enumerate(r["phases"]):
            phase_rows.append({**common, "run_index": k, "label": lab, "start_s": round(a, 2), "end_s": round(b, 2),
                               "duration_s": round(ns * DT, 2), "samples": ns})
        lab_s = Counter()
        for lab, a, b, ns in r["phases"]:
            lab_s[lab] += ns * DT
        # insole events (full list to the cache)
        for side in ("L", "R"):
            ii = r["insole"][side]
            for kind in ("ic", "fo"):
                for tt_, lab in zip(ii[f"{kind}_t"], ii[f"{kind}_label"]):
                    ins_rows.append({"trial_id": tid, "person_id": person, "lockbox": lb, "foot": side,
                                     "event": kind.upper(), "t_s": round(tt_, 2), "phase": lab})
        # walkway events with the nearest insole event
        matched_ic = {"L": 0, "R": 0}
        n_ic = {"L": 0, "R": 0}
        n_fo = {"L": 0, "R": 0}
        n_cens = 0
        diffs_trial = []
        rows_trial = []
        for side in ("L", "R"):
            ii = r["insole"][side]
            ic_t = np.array(ii["ic_t"], float)
            fo_t = np.array(ii["fo_t"], float)
            for k, f in enumerate(r["footfalls"].get(side, [])):
                n_cens += int(f["ic_cens"]) + int(f["fo_cens"])
                for kind, tw, cens, lab, finok, arr, idx in (("IC", f["t_ic"], f["ic_cens"], f["label_ic"], f.get("ins_fin_ic"), ic_t, f["s"]),
                                                             ("FO", f["t_fo"], f["fo_cens"], f["label_fo"], f.get("ins_fin_fo"), fo_t, f["e"])):
                    if cens or tw is None:
                        continue
                    if kind == "IC":
                        n_ic[side] += 1
                    else:
                        n_fo[side] += 1
                    near, diff = None, None
                    if r["has_insole"] and ii["valid"] and len(arr):
                        i = int(np.searchsorted(arr, tw))
                        cand = [q for q in (i - 1, i) if 0 <= q < len(arr)]
                        q = min(cand, key=lambda q: abs(arr[q] - tw))
                        if abs(arr[q] - tw) * 1000 <= SEARCH_MS + 1e-6:
                            near = float(arr[q])
                            diff = int(round((near - tw) * 1000))
                    if not r["has_insole"]:
                        state = "no_insole_columns"
                    elif not ii["valid"]:
                        state = "insole_invalid"
                    elif not finok:
                        state = "insole_missing_near_event"
                    elif diff is not None and abs(diff) <= MATCH_MS:
                        state = "matched"
                    else:
                        state = "no_insole_event_within_300ms"
                    if state == "insole_missing_near_event" and not lb:
                        na = ii["nan"]
                        w0, w1 = max(0, idx - half_w), min(r["n"], idx + half_w + 1)
                        cause = ("leading_block" if w0 < na["lead"] else
                                 "trailing_block" if na["trail"] and w1 > r["n"] - na["trail"] else "interior_run")
                        gap_cause[f"{kind}|{grp}|{cause}"] += 1
                    row = {**common, "foot": side, "event": kind, "walkway_t_s": round(tw, 2), "phase": lab,
                           "insole_t_s": round(near, 2) if near is not None else None, "diff_ms": diff, "state": state,
                           "offset_removed_ms": None}
                    if kind == "IC":
                        a = ann.get((side, k), {})
                        row.update({"pass_id": a.get("pass_id"), "direction": a.get("direction"),
                                    "heel_x_cells": a.get("heel_x"), "ic_x_cells": fnum(f.get("icx"), 1),
                                    "fp_xmin_cells": f.get("xmin"), "fp_xmax_cells": f.get("xmax"),
                                    "fp_y_cells": fnum(f.get("ymean"), 1), "fp_ncells": f.get("ncells"),
                                    "partial": a.get("partial"),
                                    "step_cells": a.get("step_cells"), "stride_cells": a.get("stride_cells"),
                                    "step_cm": fnum(a["step_cells"] * cell_cm, 2) if a.get("step_cells") is not None else None,
                                    "stride_cm": fnum(a["stride_cells"] * cell_cm, 2) if a.get("stride_cells") is not None else None})
                        if state == "matched":
                            matched_ic[side] += 1
                            diffs_trial.append(diff)
                    rows_trial.append(row)
        if len(diffs_trial) >= MIN_MATCHED_FOR_OFFSET:
            off = float(np.median(diffs_trial))
            per_trial_offsets[tid] = off
            for row in rows_trial:
                if row["event"] == "IC" and row["state"] == "matched":
                    row["offset_removed_ms"] = round(row["diff_ms"] - off, 1)
        rows_trial.sort(key=lambda z: (z["walkway_t_s"], z["foot"], z["event"]))
        event_rows += rows_trial

        pkr = pk_by_trial.get(tid)
        ori = r["ori"]
        n_ori_present = sum(1 for s in CANON_ORDER if ori[s]["present"])
        n_ori_ok = sum(1 for s in CANON_ORDER if ori[s]["present"] and (ori[s]["finite"] or 0) >= ORI_OK_FRAC)
        hs = set(j["header"])
        n_imu_hdr = sum(1 for p in prefixes if all(f"{p}_{k}" in hs for k in ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")))
        c_ph = any(lab in WALK_LABELS for lab, *_ in r["phases"])
        c_ww = n_ic["L"] >= MIN_IC_PER_FOOT and n_ic["R"] >= MIN_IC_PER_FOOT
        c_ins = matched_ic["L"] >= MIN_IC_PER_FOOT and matched_ic["R"] >= MIN_IC_PER_FOOT
        c_ori = n_ori_ok == len(CANON_ORDER)
        c_pk = None
        if j["task"] in PKMAS_TASKS:
            c_pk = bool(pkr is not None and all(np.isfinite(pkr[k]) for k in ("velocity_cm_s", "stride_length_cm_mean", "cadence_steps_min")))
        complete = c_ph and c_ww and c_ins and c_ori and (c_pk if c_pk is not None else True)
        ins = r["insole"]
        row = {**common, "participant": pid, "folder_group": j["folder_group"], "cohort": cohort_of(pid), "file": j["file"],
               "n_columns": len(j["header"]), "samples": r["n"], "duration_s": round(r["duration_s"], 2),
               "dt_irregular": r["dt_irregular"], "time_nonfinite": r["time_nonfinite"],
               "walk_s": round(lab_s.get("Walk", 0.0), 2), "turn_s": round(lab_s.get("Turn", 0.0), 2),
               "tandemwalk_s": round(lab_s.get("TandemWalk", 0.0), 2), "n_phase_runs": len(r["phases"]),
               "phase_labels": ";".join(sorted(lab_s)),
               "has_walkway": r["has_walkway"], "has_walkway_cells": r["has_walkway_cells"],
               "ww_ic_L": n_ic["L"], "ww_ic_R": n_ic["R"], "ww_fo_L": n_fo["L"], "ww_fo_R": n_fo["R"],
               "ww_censored_events": n_cens, "ww_contact_nan": sum(r["ww_nan"].values()) if r["ww_nan"] else None,
               "other_cell_samples": r["other_cell_samples"],
               "has_insole": r["has_insole"],
               "ins_valid_L": ins["L"]["valid"], "ins_valid_R": ins["R"]["valid"],
               "ins_finite_frac_L": fnum(ins["L"]["n_finite"] / r["n"] if r["n"] else None, 4),
               "ins_finite_frac_R": fnum(ins["R"]["n_finite"] / r["n"] if r["n"] else None, 4),
               "ins_nan_lead_s": round(max(ins[s]["nan"]["lead"] for s in "LR") * DT, 2) if r["has_insole"] else None,
               "ins_nan_trail_s": round(max(ins[s]["nan"]["trail"] for s in "LR") * DT, 2) if r["has_insole"] else None,
               "ins_nan_interior_runs": sum(ins[s]["nan"]["interior_runs"] for s in "LR") if r["has_insole"] else None,
               "ins_nan_interior_single": sum(ins[s]["nan"]["interior_single"] for s in "LR") if r["has_insole"] else None,
               "ins_nan_interior_long": sum(ins[s]["nan"]["interior_long"] for s in "LR") if r["has_insole"] else None,
               "ins_f05_L": fnum(ins["L"]["lo"], 2), "ins_f05_R": fnum(ins["R"]["lo"], 2),
               "ins_stance_median_L": fnum(ins["L"]["fst"], 2), "ins_stance_median_R": fnum(ins["R"]["fst"], 2),
               "ins_threshold_L": fnum(ins["L"]["thr"], 2), "ins_threshold_R": fnum(ins["R"]["thr"], 2),
               "ins_ic_L": len(ins["L"]["ic_t"]), "ins_ic_R": len(ins["R"]["ic_t"]),
               "ins_fo_L": len(ins["L"]["fo_t"]), "ins_fo_R": len(ins["R"]["fo_t"]),
               "ins_ic_walking_L": sum(1 for lab in ins["L"]["ic_label"] if lab in WALK_LABELS),
               "ins_ic_walking_R": sum(1 for lab in ins["R"]["ic_label"] if lab in WALK_LABELS),
               "matched_ic_L": matched_ic["L"], "matched_ic_R": matched_ic["R"],
               "ic_offset_median_ms": fnum(per_trial_offsets.get(tid), 1),
               "xcorr_lag_ms_L": r["xcorr_lag_ms"].get("L"), "xcorr_lag_ms_R": r["xcorr_lag_ms"].get("R"),
               "cadence_contacts": fnum(g["cadence_contacts"], 3), "ww_n_straight_passes": g["n_straight_passes"],
               "ww_n_speed_passes": g["n_speed_passes"],
               "ww_speed_cells_s": fnum(g["speed_cells_s"], 3),
               "ww_speed_m_s": fnum(g["speed_cells_s"] * cell_cm / 100 if g["speed_cells_s"] is not None else None, 4),
               "ww_stride_cells_mean": fnum(g["stride_cells_mean"], 3),
               "ww_stride_cm_mean": fnum(g["stride_cells_mean"] * cell_cm if g["stride_cells_mean"] is not None else None, 2),
               "ww_n_strides": g["n_strides"],
               "ww_step_cm_mean": fnum(g["step_cells_mean"] * cell_cm if g["step_cells_mean"] is not None else None, 2),
               "ww_n_steps": g["n_steps"],
               "pkmas_matched": pkr is not None,
               "pkmas_velocity_m_s": fnum(pkr["velocity_cm_s"] / 100 if pkr else None, 4),
               "pkmas_stride_length_cm": fnum(pkr["stride_length_cm_mean"] if pkr else None, 3),
               "pkmas_step_length_cm": fnum(pkr["step_length_cm_mean"] if pkr else None, 3),
               "pkmas_cadence_steps_min": fnum(pkr["cadence_steps_min"] if pkr else None, 3),
               "pkmas_stride_n": fnum(pkr["stride_length_n"] if pkr else None, 0),
               "pkmas_cadence_verified": (bool(abs(pkr["cadence_steps_min"] - g["cadence_contacts"]) <= CADENCE_CHECK)
                                          if pkr is not None and g["cadence_contacts"] is not None and np.isfinite(pkr["cadence_steps_min"]) else None),
               "pkmas_duplicate_metrics_of": (pkr["duplicate_metrics_of"] or None) if pkr is not None else None,
               "n_imu_sites_header": n_imu_hdr, "n_ori_present": n_ori_present, "n_ori_ok": n_ori_ok}
        for s in CANON_ORDER:
            row[f"ori_finite_{s}"] = fnum(ori[s]["finite"], 4) if ori[s]["present"] else None
        for s in CANON_ORDER:
            row[f"ori_zero_{s}"] = fnum(ori[s]["zero"], 4) if ori[s]["present"] else None
        row.update({"c_phases": c_ph, "c_walkway": c_ww, "c_insole": c_ins, "c_orientation": c_ori,
                    "c_pkmas": c_pk, "complete": complete})
        trial_rows.append(row)

    trials = pd.DataFrame(trial_rows)
    events = pd.DataFrame(event_rows)
    phases = pd.DataFrame(phase_rows)
    ev_cols = ["trial_id", "person_id", "group", "task", "lockbox", "foot", "event", "walkway_t_s", "phase", "insole_t_s",
               "diff_ms", "state", "offset_removed_ms", "pass_id", "direction", "heel_x_cells", "ic_x_cells", "fp_xmin_cells",
               "fp_xmax_cells", "fp_y_cells", "fp_ncells", "partial", "step_cells", "stride_cells", "step_cm", "stride_cm"]
    events = events.reindex(columns=ev_cols)
    for c in ("diff_ms", "pass_id", "direction", "heel_x_cells", "fp_xmin_cells", "fp_xmax_cells", "fp_ncells", "step_cells", "stride_cells"):
        events[c] = events[c].astype("Int64")
    trials.to_csv(os.path.join(out, "trials.csv"), index=False, lineterminator="\n", encoding="utf-8")
    events.to_csv(os.path.join(out, "events.csv"), index=False, lineterminator="\n", encoding="utf-8")
    phases.to_csv(os.path.join(out, "phases.csv"), index=False, lineterminator="\n", encoding="utf-8")
    pd.DataFrame(ins_rows).to_csv(os.path.join(out, "cache", "insole_events.csv"), index=False, lineterminator="\n", encoding="utf-8")
    pkdf = pd.DataFrame(pk_rows)
    pk_front = ["pkmas_row", "task_raw", "task", "participant", "participant_base", "visit", "pd_vs_control", "group_source",
                "person_id", "lockbox", "match_status", "csv_task", "trial_id"]
    pkdf = pkdf[pk_front + [c for c in pkdf.columns if c not in pk_front]]
    for tid_col, src, scale, nd in (("ww_cadence_contacts", "cadence_contacts", 1.0, 3),):
        pkdf[tid_col] = [fnum(geo[t][src] * scale, nd) if t and geo[t][src] is not None else None for t in pkdf.trial_id]
    pkdf["ww_speed_m_s"] = [fnum(geo[t]["speed_cells_s"] * cell_cm / 100, 4) if t and geo[t]["speed_cells_s"] is not None else None for t in pkdf.trial_id]
    pkdf["ww_stride_cm_mean"] = [fnum(geo[t]["stride_cells_mean"] * cell_cm, 2) if t and geo[t]["stride_cells_mean"] is not None else None for t in pkdf.trial_id]
    pkdf["ww_n_strides"] = [geo[t]["n_strides"] if t else None for t in pkdf.trial_id]
    pkdf["cadence_verified"] = [bool(abs(a - b) <= CADENCE_CHECK) if (st == "matched" and np.isfinite(a) and b is not None and np.isfinite(b)) else None
                                for st, a, b in zip(pkdf.match_status, pkdf.cadence_steps_min.astype(float), pkdf.ww_cadence_contacts)]
    pkdf.to_csv(os.path.join(out, "pkmas_match.csv"), index=False, lineterminator="\n", encoding="utf-8", float_format="%.6g")
    log(f"wrote trials.csv {trials.shape}, events.csv {events.shape}, phases.csv {phases.shape}, pkmas_match.csv {pkdf.shape}")

    # ================================================================ summary (non-lockbox only)
    nl = trials[~trials.lockbox]
    evn = events[~events.lockbox]
    groups = ["control", "PD"]
    lb_files = {j["file"] for j in jobs if lb_trial[j["trial_id"]]}
    S = {"rule_version": RULE_VERSION, "cell_cm": cell_cm}
    S["listing"] = {
        "trials": int(len(trials)), "trials_by_task_group": {f"{t}|{gg}": int(((trials.task == t) & (trials.group == gg)).sum()) for t in TASKS for gg in sorted(set(trials.group))},
        "persons_by_group": {gg: int(trials[trials.group == gg].person_id.nunique()) for gg in sorted(set(trials.group))},
        "folder_vs_group": {f"{a}|{b}": int(v) for (a, b), v in Counter(zip(trials.folder_group, trials.group)).items()},
        "skipped_other_tasks": dict(skipped),
        "real_world_context_csv": dict(sorted(rw_counts.items())),
        "manifest_csv_not_on_disk": manifest_missing,
        "table_persons_without_csv": table_without_csv,
        # content-derived checks: non-lockbox trials only
        "duplicate_header_files_non_lockbox": [f for f in dup_header_files if f not in lb_files],
        "trials_with_insole_columns": {gg: int(nl[(nl.group == gg)].has_insole.sum()) for gg in groups},
        "trials_with_walkway_columns": {gg: int(nl[(nl.group == gg)].has_walkway.sum()) for gg in groups},
        "trials_with_events_column": {gg: int(sum(results[t]["has_events"] for t in nl[nl.group == gg].trial_id)) for gg in groups},
        "dt_irregular_trials": int((nl.dt_irregular > 0).sum()), "time_nonfinite_trials": int((nl.time_nonfinite > 0).sum()),
        "ww_contact_nan_trials": int((nl.ww_contact_nan.fillna(0) > 0).sum()),
        "walkway_x_extent": [int(min(xmin_all)) if xmin_all else None, x_edge_hi],
        "footprint_list_mismatch_samples": int(sum(f.get("mism", 0) for r in res_nl for s in ("L", "R") for f in r["footfalls"].get(s, []))),
        "missing_task_files": {},
    }
    persons_main = sorted(set(j["participant"] for j in jobs))
    have = {(j["participant"], j["task"]) for j in jobs}
    S["listing"]["missing_task_files"] = {t: sorted(p for p in persons_main if (p, t) not in have) for t in TASKS}
    lb_person_trials = trials[trials.lockbox].groupby("person_id").size()
    S["lockbox"] = {"persons": len(lockbox), "persons_with_trials": int(len(lb_person_trials)), "trial_rows": int(trials.lockbox.sum()),
                    "persons_without_main_csv": sorted(p for p in lockbox if p not in set(trials.person_id)),
                    "group_label_differs_from_source_tables": {p: {"lockbox_group": lockbox[p], "source_group": group_of(p.split(":", 1)[1])}
                                                               for p in sorted(lockbox)
                                                               if group_of(p.split(":", 1)[1]).lower() != lockbox[p].lower()}}
    S["non_lockbox"] = {"trials": int(len(nl)), "persons_by_group": {gg: int(nl[nl.group == gg].person_id.nunique()) for gg in groups},
                        "trials_by_group": {gg: int((nl.group == gg).sum()) for gg in groups}}
    # phases
    S["phases"] = {}
    for t in TASKS:
        sub = nl[nl.task == t]
        S["phases"][t] = {"trials": int(len(sub)),
                          "labels": dict(Counter(lab for pl in sub.phase_labels for lab in pl.split(";") if lab)),
                          "walk_s_median": pct(sub.walk_s, 50), "turn_s_median": pct(sub.turn_s, 50),
                          "tandemwalk_s_median": pct(sub.tandemwalk_s, 50), "samples_median": pct(sub.samples, 50)}
    # walkway and insole counts
    S["events"] = {}
    for t in TASKS:
        for gg in groups:
            sub = nl[(nl.task == t) & (nl.group == gg)]
            if not len(sub):
                continue
            S["events"][f"{t}|{gg}"] = {
                "trials": int(len(sub)),
                "ww_ic_per_foot_median": pct(pd.concat([sub.ww_ic_L, sub.ww_ic_R]), 50),
                "ww_ic_total": int(sub.ww_ic_L.sum() + sub.ww_ic_R.sum()),
                "ww_censored_events": int(sub.ww_censored_events.sum()),
                "insole_trials": int(sub.has_insole.sum()),
                "insole_feet_valid": int(sub.ins_valid_L.sum() + sub.ins_valid_R.sum()),
                "ins_ic_per_foot_median": pct(pd.concat([sub.ins_ic_L[sub.ins_valid_L], sub.ins_ic_R[sub.ins_valid_R]]), 50),
                "ins_ic_total": int(sub.ins_ic_L.sum() + sub.ins_ic_R.sum()),
                "ins_ic_walking_total": int(sub.ins_ic_walking_L.sum() + sub.ins_ic_walking_R.sum())}
            es = evn[(evn.task == t) & (evn.group == gg) & (evn.event == "IC")]
            sc = es.stride_cm.astype(float)
            S["events"][f"{t}|{gg}"].update({"strides": int(np.isfinite(sc).sum()), "stride_trials": int(es[np.isfinite(sc)].trial_id.nunique()),
                                             "stride_cm_median": pct(sc, 50), "partial_ic": int(sum(1 for v in es.partial if isinstance(v, (bool, np.bool_)) and v))})
    feet = pd.concat([nl[["group", "ins_finite_frac_L", "ins_f05_L", "ins_stance_median_L", "ins_threshold_L", "ins_valid_L", "has_insole"]]
                      .set_axis(["group", "fin", "f05", "fst", "thr", "valid", "has"], axis=1),
                      nl[["group", "ins_finite_frac_R", "ins_f05_R", "ins_stance_median_R", "ins_threshold_R", "ins_valid_R", "has_insole"]]
                      .set_axis(["group", "fin", "f05", "fst", "thr", "valid", "has"], axis=1)])
    fh = feet[feet.has]
    S["insole_nan"] = {}
    for gg in groups:
        sub = nl[(nl.group == gg) & nl.has_insole]
        S["insole_nan"][gg] = {"trials": int(len(sub)), "lead_s_median": pct(sub.ins_nan_lead_s, 50), "lead_s_p90": pct(sub.ins_nan_lead_s, 90),
                               "trials_with_trail": int((sub.ins_nan_trail_s.astype(float) > 0).sum()),
                               "interior_runs_median": pct(sub.ins_nan_interior_runs, 50),
                               "interior_runs_total": int(sub.ins_nan_interior_runs.sum()),
                               "interior_single_total": int(sub.ins_nan_interior_single.sum()),
                               "interior_long_total": int(sub.ins_nan_interior_long.sum())}
    S["insole_missing_cause"] = {f"{kind}|{gg}": {c: int(gap_cause.get(f"{kind}|{gg}|{c}", 0))
                                                   for c in ("leading_block", "trailing_block", "interior_run")}
                                 for kind in ("IC", "FO") for gg in groups}
    S["insole_levels"] = {gg: {"feet": int((fh.group == gg).sum()), "valid": int(fh[fh.group == gg].valid.sum()),
                               "finite_frac_median": pct(fh[fh.group == gg].fin, 50), "finite_frac_p10": pct(fh[fh.group == gg].fin, 10),
                               "f05_median": pct(fh[fh.group == gg].f05, 50), "f05_p90": pct(fh[fh.group == gg].f05, 90),
                               "stance_median_median": pct(fh[fh.group == gg].fst, 50),
                               "stance_median_p10": pct(fh[fh.group == gg].fst, 10), "stance_median_p90": pct(fh[fh.group == gg].fst, 90),
                               "threshold_median": pct(fh[fh.group == gg].thr, 50)} for gg in groups}
    # agreement
    def agree(sub_ev, kind):
        e = sub_ev[sub_ev.event == kind]
        st = Counter(e.state)
        m = e[e.state == "matched"]
        scorable = st.get("matched", 0) + st.get("no_insole_event_within_300ms", 0)
        with_ins = len(e) - st.get("no_insole_columns", 0)
        d = {"walkway_events": int(len(e)), "states": dict(st),
             "matched_share_of_scorable": (st.get("matched", 0) / scorable) if scorable else None,
             "missing_near_event_share_of_events": (st.get("insole_missing_near_event", 0) / len(e)) if len(e) else None,
             "missing_near_event_share_with_insole_columns": (st.get("insole_missing_near_event", 0) / with_ins) if with_ins else None,
             "trials": int(m.trial_id.nunique()), "persons": int(m.person_id.nunique())}
        d.update(qstats(np.abs(m.diff_ms.astype(float)), "abs_"))
        d["signed_median"] = pct(m.diff_ms.astype(float), 50)
        d["signed_p10"] = pct(m.diff_ms.astype(float), 10)
        d["signed_p90"] = pct(m.diff_ms.astype(float), 90)
        if kind == "IC":
            orr = m.offset_removed_ms.astype(float)
            d.update(qstats(np.abs(orr), "offset_removed_abs_"))
            d["offset_removed_trials"] = int(m[np.isfinite(orr)].trial_id.nunique())
            if d["signed_median"] is not None:
                pooled = np.abs(m.diff_ms.astype(float) - d["signed_median"])
                d.update(qstats(pooled, "pooled_removed_abs_"))
                d["pooled_removed_share_abs_le_30"] = float((pooled <= 30).mean())
            d["share_abs_le_30"] = float((np.abs(m.diff_ms.astype(float)) <= 30).mean()) if len(m) else None
            d["share_abs_le_50"] = float((np.abs(m.diff_ms.astype(float)) <= 50).mean()) if len(m) else None
        return d
    S["agreement"] = {}
    for kind in ("IC", "FO"):
        for gg in groups + ["all"]:
            sub = evn if gg == "all" else evn[evn.group == gg]
            S["agreement"][f"{kind}|{gg}|all_tasks"] = agree(sub, kind)
            for t in TASKS:
                S["agreement"][f"{kind}|{gg}|{t}"] = agree(sub[sub.task == t], kind)
    # IC agreement restricted to walking phases
    for gg in groups + ["all"]:
        sub = evn if gg == "all" else evn[evn.group == gg]
        S["agreement"][f"IC|{gg}|walking_phase_only"] = agree(sub[sub.phase.isin(WALK_LABELS)], "IC")
    # per-trial offsets and xcorr lags
    off = nl.ic_offset_median_ms.astype(float)
    S["offsets"] = {gg: {"trials": int(np.isfinite(off[nl.group == gg]).sum()),
                         "median": pct(off[nl.group == gg], 50), "p10": pct(off[nl.group == gg], 10), "p90": pct(off[nl.group == gg], 90),
                         "abs_p90": pct(np.abs(off[nl.group == gg]), 90)} for gg in groups}
    lag = pd.concat([nl[["group", "xcorr_lag_ms_L", "ic_offset_median_ms"]].set_axis(["group", "lag", "off"], axis=1),
                     nl[["group", "xcorr_lag_ms_R", "ic_offset_median_ms"]].set_axis(["group", "lag", "off"], axis=1)])
    lag = lag.astype({"lag": float, "off": float})
    S["xcorr"] = {}
    for gg in groups:
        L_ = lag[lag.group == gg]
        ok = np.isfinite(L_.lag) & np.isfinite(L_.off)
        S["xcorr"][gg] = {"feet": int(np.isfinite(L_.lag).sum()), "median": pct(L_.lag, 50), "p10": pct(L_.lag, 10),
                          "p90": pct(L_.lag, 90), "abs_gt_60": int((np.abs(L_.lag) > 60).sum()),
                          "at_limit": int((np.abs(L_.lag) >= XCORR_MAX_LAG * 10).sum()),
                          "pearson_with_ic_offset": float(np.corrcoef(L_.lag[ok], L_.off[ok])[0, 1]) if ok.sum() >= 3 else None,
                          "n_with_offset": int(ok.sum())}
    # PKMAS
    pkn = pkdf.copy()
    S["pkmas"] = {"rows": len(pk), "blank_rows": pk_blank, "status": dict(Counter(pkn.match_status)),
                  "rows_by_task": dict(Counter(pkn.task)),
                  "units": [f"{nm} [{sub}]" if sub else nm for _, nm, sub in PKMAS_COLS if (nm, sub) in set(pk_headers)],
                  "task_raw_variants": dict(Counter(pkn.task_raw)),
                  "matched_trials_by_task_group_non_lockbox": {f"{t}|{gg}": int(((nl.task == t) & (nl.group == gg) & nl.pkmas_matched).sum())
                                                               for t in PKMAS_TASKS for gg in groups},
                  "trials_by_task_group_non_lockbox": {f"{t}|{gg}": int(((nl.task == t) & (nl.group == gg)).sum()) for t in PKMAS_TASKS for gg in groups},
                  "persons_without_match_non_lockbox": {gg: int((~nl[nl.group == gg].groupby("person_id").pkmas_matched.any()).sum()) for gg in groups},
                  "persons_non_lockbox": {gg: int(nl[nl.group == gg].person_id.nunique()) for gg in groups},
                  "duplicate_metric_rows": sorted(f"{a} {p} (row {r_}) = {o}" for a, p, r_, o in zip(pkn.task, pkn.participant, pkn.pkmas_row, pkn.duplicate_metrics_of) if o),
                  "no_csv_trial_ids": sorted(f"{p} {t}" for p, t in zip(pkn[pkn.match_status == "no_csv_trial"].participant,
                                                                         pkn[pkn.match_status == "no_csv_trial"].task)),
                  "second_visit_ids": sorted(set(pkn[pkn.match_status == "second_visit_not_matched"].participant.tolist())),
                  "out_of_scope": sorted(f"{a}|{b}" for a, b in zip(pkn[pkn.match_status == "task_out_of_scope"].task, pkn[pkn.match_status == "task_out_of_scope"].participant)),
                  "group_label_disagreements": int(sum(1 for a, b in zip(pkn.pd_vs_control, pkn.group_source)
                                                       if b in ("control", "PD") and a.lower() != b.lower())),
                  "group_label_disagreement_rows": sorted(f"{p} ({a} vs {b})" for p, a, b in zip(pkn.participant, pkn.pd_vs_control, pkn.group_source)
                                                          if b in ("control", "PD") and a.lower() != b.lower())}
    pnl = pkn[(pkn.match_status == "matched") & (pkn.lockbox != True)]  # noqa: E712
    ev_pair = {}
    for t in PKMAS_TASKS:
        cand = [t] + PKMAS_ALT[t]
        for ct in cand:
            col = "ww_cadence_contacts" if ct == t else f"alt_cadence_{ct}"
            sub = pnl[pnl.task == t]
            ev_pair[f"{t}|{ct}"] = pair_stats(sub.cadence_steps_min.astype(float), sub[col].astype(float))
    S["pkmas"]["pairing_cadence_evidence"] = ev_pair
    sv = pkn[pkn.match_status == "second_visit_not_matched"]
    first = pkn[(pkn.visit == 1) & pkn.participant_base.isin(sv.participant_base)]
    S["pkmas"]["second_visit_evidence"] = [
        {"participant": r_.participant, "task": r_.task, "pkmas_cadence": r_.cadence_steps_min,
         "csv_cadence_contacts": r_.base_trial_cadence_contacts} for r_ in pd.concat([first, sv]).sort_values(["participant_base", "task", "visit"]).itertuples()]
    S["pkmas"]["cell_fit"] = {"n": len(ratios), "cell_cm": cell_cm, "p10": pct(ratios, 10), "p90": pct(ratios, 90)}
    S["pkmas"]["cadence_not_verified_non_lockbox"] = sorted(
        f"{p} {t} (PKMAS {a:.1f}, walkway {b:.1f} steps/min)" for p, t, a, b, v in
        zip(pnl.participant, pnl.task, pnl.cadence_steps_min.astype(float), pnl.ww_cadence_contacts.astype(float), pnl.cadence_verified)
        if v is not True)
    S["pkmas"]["agreement"] = {}
    for t in PKMAS_TASKS:
        for gg, subset in [(x, "matched") for x in groups + ["all"]] + [("all", "cadence_verified")]:
            sub = pnl[pnl.task == t] if gg == "all" else pnl[(pnl.task == t) & (pnl.group_source == gg)]
            if subset == "cadence_verified":
                sub = sub[sub.cadence_verified == True]  # noqa: E712
            S["pkmas"]["agreement"][f"{t}|{gg}|{subset}"] = {
                "velocity_m_s": pair_stats(sub.velocity_cm_s.astype(float) / 100, sub.ww_speed_m_s.astype(float)),
                "stride_length_cm": pair_stats(sub.stride_length_cm_mean.astype(float), sub.ww_stride_cm_mean.astype(float)),
                "cadence_steps_min": pair_stats(sub.cadence_steps_min.astype(float), sub.ww_cadence_contacts.astype(float)),
                "strides_per_trial_pkmas_median": pct(sub.stride_length_n.astype(float), 50),
                "strides_per_trial_walkway_median": pct(sub.ww_n_strides.astype(float), 50)}
    # orientation
    S["orientation"] = {}
    for gg in groups:
        sub = nl[nl.group == gg]
        fin_present = sub[[f"ori_finite_{s}" for s in CANON_ORDER]].astype(float)
        S["orientation"][gg] = {"trials": int(len(sub)), "all_13_ok": int((sub.n_ori_ok == 13).sum()),
                                "all_13_present": int((sub.n_ori_present == 13).sum()),
                                "trials_with_a_present_sensor_below_ok": int((fin_present < ORI_OK_FRAC).any(axis=1).sum()),
                                "trials_with_a_present_sensor_below_half": int((fin_present < 0.5).any(axis=1).sum()),
                                "min_finite_share_present": pct(fin_present.to_numpy().ravel(), 0),
                                "sites": {s: {"present": int(sub[f"ori_finite_{s}"].notna().sum()),
                                              "ok": int((sub[f"ori_finite_{s}"].astype(float) >= ORI_OK_FRAC).sum()),
                                              "finite_median": pct(sub[f"ori_finite_{s}"].astype(float), 50),
                                              "zero_gt_1pct": int((sub[f"ori_zero_{s}"].astype(float) > 0.01).sum())} for s in CANON_ORDER},
                                "imu_sites_header_13": int((sub.n_imu_sites_header == 13).sum())}
    # completeness
    S["completeness"] = {}
    for gg in groups:
        for t in TASKS + ["all_tasks"]:
            sub = nl[nl.group == gg] if t == "all_tasks" else nl[(nl.group == gg) & (nl.task == t)]
            if not len(sub):
                continue
            d = {"trials": int(len(sub))}
            for c in ("c_phases", "c_walkway", "c_insole", "c_orientation", "complete"):
                d[c] = float(sub[c].astype(bool).mean())
            pkc = sub.c_pkmas.dropna()
            d["c_pkmas"] = float(pkc.astype(bool).mean()) if len(pkc) else None
            d["complete_n"] = int(sub.complete.astype(bool).sum())
            S["completeness"][f"{gg}|{t}"] = d
    S["runtime_s"] = round(time.time() - T0, 1)
    S["trials_from_cache"] = n_cached
    S["workers"] = opt.workers
    S["limit"] = opt.limit
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(S), f, indent=1)
    write_summary_md(out, S)
    rule_hash = hashlib.sha256(insole_rule_text().encode("utf-8")).hexdigest()
    write_provenance(args, {"rule_version": RULE_VERSION, "insole_rule_sha256": rule_hash, "trials": len(jobs),
                            "trials_from_cache": n_cached, "limit": opt.limit, "workers": opt.workers,
                            "mat_files_opened": 0, "runtime_s": S["runtime_s"],
                            "inputs": [pkmas_path, demo_paths["control"], demo_paths["PD"], lockbox_path,
                                       group_dirs["control"], group_dirs["PD"]]})
    log("done")


# ---------------------------------------------------------------- summary.md
def f1(x, nd=1):
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}"


def pc(x):
    return "n/a" if x is None else f"{100 * float(x):.1f}"


def write_summary_md(out, S):
    L = []
    Li = S["listing"]
    L.append("# weargait_pd targets for G3 and the second half of G2 (plan task B3a)")
    L.append("")
    L.append("Generated by scripts/build_weargait_targets.py; every number below is read from summary.json and the CSVs of this directory. MoveAtlas commit in _provenance.json. Definitions, the insole event rule (fixed before any agreement was computed) and the completeness rule are in the script docstring.")
    L.append("")
    L.append("## Inputs")
    L.append("")
    tbg = Li["trials_by_task_group"]
    L.append(f"Trials (CSV files of the seven tasks in the two main CSV folders): {Li['trials']}. Persons by group (source CONTROLS and PD tables): "
             + ", ".join(f"{k} {v}" for k, v in Li["persons_by_group"].items()) + ". Folder against group: "
             + ", ".join(f"{k} {v}" for k, v in sorted(Li["folder_vs_group"].items())) + ".")
    L.append("")
    L.append("| task | control trials | PD trials | persons without the file |")
    L.append("|---|---|---|---|")
    for t in TASKS:
        miss = Li["missing_task_files"].get(t, [])
        L.append(f"| {t} | {tbg.get(f'{t}|control', 0)} | {tbg.get(f'{t}|PD', 0)} | {len(miss)}{(' (' + ', '.join(miss) + ')') if miss else ''} |")
    L.append("")
    L.append(f"Other CSVs of the main folders, not read: {', '.join(f'{k} {v}' for k, v in Li['skipped_other_tasks'].items())}. Real-world context tasks (listed, out of scope for this build): "
             + "; ".join(f"{k} {v}" for k, v in Li["real_world_context_csv"].items()) + ". No MAT file was opened: the CSVs carry every column this build needs.")
    L.append("")
    mm, tw_ = Li["manifest_csv_not_on_disk"], Li["table_persons_without_csv"]
    L.append(f"The CSVs do not cover every person. CSV files of the seven tasks listed in the release manifest.csv of the main folders but absent on disk: {sum(len(v) for v in mm.values())} files of {len(mm)} persons"
             + (" (" + "; ".join(f"{p}: {', '.join(v)}" for p, v in mm.items()) + ")" if mm else "") + ". "
             f"Persons of the demographic tables without any CSV trial: {len(tw_)}"
             + (" (" + "; ".join(f"{p}, {v['group']}, {'MAT file present' if v['mat_file_present'] else 'no MAT file'}{', lockbox' if v['lockbox'] else ''}" for p, v in tw_.items()) + ")" if tw_ else "")
             + ". This build does not open MAT files, so these persons and the missing task files have no targets here.")
    L.append("")
    L.append(f"Column groups, non-lockbox trials: trials with the insole columns (LTotalForce, RTotalForce): control {Li['trials_with_insole_columns']['control']}, PD {Li['trials_with_insole_columns']['PD']}; "
             f"with the walkway contact columns: control {Li['trials_with_walkway_columns']['control']}, PD {Li['trials_with_walkway_columns']['PD']}; "
             f"with GeneralEvent: control {Li['trials_with_events_column']['control']}, PD {Li['trials_with_events_column']['PD']}. "
             f"Files with duplicated column names: {len(Li['duplicate_header_files_non_lockbox'])}. Trials with any sample interval off 0.01 s: {Li['dt_irregular_trials']}; with a non-finite Time value: {Li['time_nonfinite_trials']}; with NaN in a walkway contact column: {Li['ww_contact_nan_trials']}. "
             f"Walkway x extent over every non-lockbox footprint: {Li['walkway_x_extent'][0]} to {Li['walkway_x_extent'][1]} cells. Samples whose cell lists differ in length: {Li['footprint_list_mismatch_samples']}. "
             "The insole force unit is not stated in the raw folder; the insole rule is relative to each trial's own force levels.")
    L.append("")
    Lk = S["lockbox"]
    L.append("## Lockbox")
    L.append("")
    L.append(f"weargait_pd persons in reports/lockbox/lockbox_persons.csv: {Lk['persons']}; with trials here: {Lk['persons_with_trials']} ({Lk['trial_rows']} trial rows, flagged lockbox true in every CSV and not summarised below). "
             f"Lockbox persons with no trial in the main CSV folders: {', '.join(Lk['persons_without_main_csv']) or 'none'}. "
             f"Lockbox rows whose group label differs from the source tables: {len(Lk['group_label_differs_from_source_tables'])}"
             + (" (" + "; ".join(f"{p}: lockbox {v['lockbox_group']}, source {v['source_group']}" for p, v in Lk["group_label_differs_from_source_tables"].items()) + ")" if Lk["group_label_differs_from_source_tables"] else "") + ".")
    L.append("")
    NL = S["non_lockbox"]
    L.append(f"Everything below: non-lockbox trials only, {NL['trials']} trials (control {NL['trials_by_group']['control']}, PD {NL['trials_by_group']['PD']}) of control {NL['persons_by_group']['control']} and PD {NL['persons_by_group']['PD']} persons.")
    L.append("")
    L.append("## Phases")
    L.append("")
    L.append("| task | trials | labels (trials carrying each) | Walk s median | Turn s median | TandemWalk s median |")
    L.append("|---|---|---|---|---|---|")
    for t in TASKS:
        P = S["phases"][t]
        L.append(f"| {t} | {P['trials']} | {', '.join(f'{k} {v}' for k, v in sorted(P['labels'].items()))} | {f1(P['walk_s_median'])} | {f1(P['turn_s_median'])} | {f1(P['tandemwalk_s_median'])} |")
    L.append("")
    L.append("## Walkway and insole events")
    L.append("")
    L.append("| task, group | trials | walkway IC per foot, median | walkway IC total | censored walkway events | partial footfalls (IC) | walkway strides (trials), stride cm median | trials with insole columns | scorable insole feet | insole IC per foot, median | insole IC total (in walking phases) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, E in S["events"].items():
        L.append(f"| {k.replace('|', ', ')} | {E['trials']} | {f1(E['ww_ic_per_foot_median'], 0)} | {E['ww_ic_total']} | {E['ww_censored_events']} | {E['partial_ic']} | {E['strides']} ({E['stride_trials']}), {f1(E['stride_cm_median'], 1)} | {E['insole_trials']} | {E['insole_feet_valid']} | {f1(E['ins_ic_per_foot_median'], 0)} | {E['ins_ic_total']} ({E['ins_ic_walking_total']}) |")
    L.append("")
    L.append("Walkway strides are counted in straight passes (ICs under a walking label, inside one GeneralEvent run, at most 1.5 s apart) between non-partial footfalls; in the _mat tasks, TUG and TandemGait they are secondary targets (turns and transfers happen on the walkway).")
    L.append("")
    L.append("Insole force levels per foot and trial (unit not stated):")
    L.append("")
    L.append("| group | feet with columns | scorable | finite share median (p10) | F05 median (p90) | median stance force median [p10, p90] | threshold median |")
    L.append("|---|---|---|---|---|---|---|")
    for gg, I in S["insole_levels"].items():
        L.append(f"| {gg} | {I['feet']} | {I['valid']} | {f1(I['finite_frac_median'], 3)} ({f1(I['finite_frac_p10'], 3)}) | {f1(I['f05_median'])} ({f1(I['f05_p90'])}) | {f1(I['stance_median_median'])} [{f1(I['stance_median_p10'])}, {f1(I['stance_median_p90'])}] | {f1(I['threshold_median'])} |")
    L.append("")
    L.append("Missing insole samples (NaN in LTotalForce or RTotalForce): " + "; ".join(
        f"{gg}: {N['trials']} trials, leading NaN block median {f1(N['lead_s_median'], 2)} s (p90 {f1(N['lead_s_p90'], 2)} s), trials with a trailing block {N['trials_with_trail']}, "
        f"interior NaN runs per trial median {f1(N['interior_runs_median'], 0)} (total {N['interior_runs_total']}, of which single samples {N['interior_single_total']}, longer than 10 samples {N['interior_long_total']})"
        for gg, N in S["insole_nan"].items()) + ". The insole rule scores no event within 300 ms of a missing sample (state insole_missing_near_event below). "
        "Walkway ICs in that state, by the NaN block that lies within 300 ms: " + "; ".join(
        f"{gg}: leading block {C_['leading_block']}, trailing block {C_['trailing_block']}, interior run {C_['interior_run']}"
        for gg, C_ in ((g_, S["insole_missing_cause"][f"IC|{g_}"]) for g_ in ("control", "PD"))) + ". "
        "Share of walkway ICs in that state: " + "; ".join(
        f"{gg} {pc(S['agreement'][f'IC|{gg}|all_tasks']['missing_near_event_share_of_events'])} % of all walkway ICs, "
        f"{pc(S['agreement'][f'IC|{gg}|all_tasks']['missing_near_event_share_with_insole_columns'])} % of those in trials with insole columns" for gg in ("control", "PD")) + ".")
    L.append("")
    L.append("## Walkway against insole: initial contact and foot-off timing")
    L.append("")
    L.append("diff = insole time minus walkway time, same foot, nearest insole event within 1 s, matched when |diff| <= 300 ms. Scorable = matched plus no insole event within 300 ms (the insole was finite around the walkway event). 100 Hz: differences are multiples of 10 ms.")
    L.append("")
    L.append("| event, group, tasks | walkway events | matched share of scorable | trials | persons | abs median (ms) | abs p90 (ms) | signed median [p10, p90] (ms) | share abs <= 30 ms | share abs <= 50 ms | pooled-shift-removed abs median / p90 (ms), share <= 30 ms | per-trial-offset-removed abs median / p90 (ms) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = [f"IC|{gg}|all_tasks" for gg in ("control", "PD", "all")] + [f"IC|{gg}|walking_phase_only" for gg in ("control", "PD", "all")] + \
            [f"FO|{gg}|all_tasks" for gg in ("control", "PD", "all")]
    for k in order:
        A = S["agreement"][k]
        L.append(f"| {k.replace('|', ', ')} | {A['walkway_events']} | {pc(A['matched_share_of_scorable'])} % | {A['trials']} | {A['persons']} | {f1(A['abs_median'], 0)} | {f1(A['abs_p90'], 0)} | {f1(A['signed_median'], 0)} [{f1(A['signed_p10'], 0)}, {f1(A['signed_p90'], 0)}] | "
                 f"{pc(A.get('share_abs_le_30'))} {'%' if A.get('share_abs_le_30') is not None else ''} | {pc(A.get('share_abs_le_50'))} {'%' if A.get('share_abs_le_50') is not None else ''} | "
                 + (f"{f1(A.get('pooled_removed_abs_median'), 0)} / {f1(A.get('pooled_removed_abs_p90'), 0)}, {pc(A.get('pooled_removed_share_abs_le_30'))} %" if A.get("pooled_removed_abs_n") else "n/a")
                 + f" | {f1(A.get('offset_removed_abs_median'), 1)} / {f1(A.get('offset_removed_abs_p90'), 1)} |")
    L.append("")
    L.append("Initial contact per task:")
    L.append("")
    L.append("| task | group | matched IC | trials | abs median (ms) | abs p90 (ms) | signed median (ms) | matched share of scorable | offset-removed abs median / p90 (ms) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        for gg in ("control", "PD"):
            A = S["agreement"][f"IC|{gg}|{t}"]
            L.append(f"| {t} | {gg} | {A['abs_n']} | {A['trials']} | {f1(A['abs_median'], 0)} | {f1(A['abs_p90'], 0)} | {f1(A['signed_median'], 0)} | {pc(A['matched_share_of_scorable'])} % | {f1(A.get('offset_removed_abs_median'), 1)} / {f1(A.get('offset_removed_abs_p90'), 1)} |")
    L.append("")
    L.append("States of the walkway IC events (all tasks, non-lockbox): " + "; ".join(
        f"{gg}: " + ", ".join(f"{k} {v}" for k, v in sorted(S['agreement'][f'IC|{gg}|all_tasks']['states'].items())) for gg in ("control", "PD")) + ".")
    L.append("")
    L.append("Per-trial clock offset (median IC diff of a trial with at least 4 matched ICs) and the insole to walkway cross-correlation lag of the force profiles (sync diagnostic, lag range +-1000 ms):")
    L.append("")
    L.append("| group | trials with an offset | offset median [p10, p90] (ms) | abs offset p90 (ms) | feet with a lag | lag median [p10, p90] (ms) | abs lag > 60 ms | lag at the range limit | Pearson r lag vs trial offset (n) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for gg in ("control", "PD"):
        O, X = S["offsets"][gg], S["xcorr"][gg]
        L.append(f"| {gg} | {O['trials']} | {f1(O['median'], 0)} [{f1(O['p10'], 0)}, {f1(O['p90'], 0)}] | {f1(O['abs_p90'], 0)} | {X['feet']} | {f1(X['median'], 0)} [{f1(X['p10'], 0)}, {f1(X['p90'], 0)}] | {X['abs_gt_60']} | {X['at_limit']} | {f1(X['pearson_with_ic_offset'], 3)} ({X['n_with_offset']}) |")
    L.append("")
    A = S["agreement"]["IC|all|all_tasks"]
    Aw = S["agreement"]["IC|all|walking_phase_only"]
    Fo = S["agreement"]["FO|all|all_tasks"]
    L.append(f"What this means for the 30 ms line of G3 (IMU initial contact against insole events): where walkway and insole see the same footfall, their ICs differ by a median {f1(A['abs_median'], 0)} ms and a 90th percentile {f1(A['abs_p90'], 0)} ms "
             f"(all non-lockbox tasks, both groups; walking phases only: {f1(Aw['abs_median'], 0)} and {f1(Aw['abs_p90'], 0)} ms). Most of it is a signed shift: insole minus walkway, signed median {f1(A['signed_median'], 0)} ms at IC and {f1(Fo['signed_median'], 0)} ms at FO, "
             f"while the force-profile cross-correlation lag has median {f1(S['xcorr']['control']['median'], 0)} ms (control) and {f1(S['xcorr']['PD']['median'], 0)} ms (PD) and Pearson r {f1(S['xcorr']['control']['pearson_with_ic_offset'], 2)} (control) and {f1(S['xcorr']['PD']['pearson_with_ic_offset'], 2)} (PD) with the per-trial IC offset. "
             + ("A later IC and an earlier FO with aligned force profiles point to the event definition (the insole threshold at 10 percent of the stance force is crossed after the first walkway cell loads and before the last one unloads) rather than to a clock offset. "
                if (A["signed_median"] is not None and Fo["signed_median"] is not None and A["signed_median"] > 0 > Fo["signed_median"]
                    and all(S["xcorr"][gg]["median"] is not None and abs(S["xcorr"][gg]["median"]) <= 10 for gg in ("control", "PD")))
                else "The signs and lags do not single out one cause (event definition or clock offset). ")
             + f"The shift is not the same in every trial: the per-trial median offset has p10 {f1(S['offsets']['control']['p10'], 0)} and p90 {f1(S['offsets']['control']['p90'], 0)} ms (control), {f1(S['offsets']['PD']['p10'], 0)} and {f1(S['offsets']['PD']['p90'], 0)} ms (PD). "
             f"Removing one pooled shift ({f1(A['signed_median'], 0)} ms) leaves a residual of median {f1(A.get('pooled_removed_abs_median'), 0)} ms, p90 {f1(A.get('pooled_removed_abs_p90'), 0)} ms "
             f"(walking phases only: {f1(Aw.get('pooled_removed_abs_median'), 0)} and {f1(Aw.get('pooled_removed_abs_p90'), 0)} ms); removing each trial's own median offset leaves median {f1(A.get('offset_removed_abs_median'), 1)} ms, p90 {f1(A.get('offset_removed_abs_p90'), 1)} ms. "
             f"Consequence: a detector that found the first walkway contact exactly would show a median error of about {f1(abs(A['signed_median']) if A['signed_median'] is not None else None, 0)} ms against this insole reference. The 30 ms line can only separate detectors if the gate fixes which contact definition it scores and how the shift is handled: "
             f"a pooled shift leaves p90 {f1(A.get('pooled_removed_abs_p90'), 0)} ms of reference disagreement, a per-trial offset p90 {f1(A.get('offset_removed_abs_p90'), 0)} ms (the timing floor the two references resolve, slightly understated because each offset is fitted on the same events).")
    L.append("")
    P = S["pkmas"]
    L.append("## PKMAS walkway metrics")
    L.append("")
    L.append(f"The file holds {P['rows']} participant-task rows ({P['blank_rows']} blank rows), trial-level means and #Samples only, no per-step rows. Rows by task: {', '.join(f'{k} {v}' for k, v in sorted(P['rows_by_task'].items()))}; raw task spellings: {', '.join(f'{repr(k)} {v}' for k, v in sorted(P['task_raw_variants'].items()))}. "
             f"Columns used and their units as stated in the header: {'; '.join(P['units'])}.")
    L.append("")
    L.append(f"Match status: {', '.join(f'{k} {v}' for k, v in sorted(P['status'].items()))}. PKMAS IDs without a CSV trial: {', '.join(P['no_csv_trial_ids']) or 'none'}. Second-visit IDs (not matched): {', '.join(P['second_visit_ids']) or 'none'}. Out of scope: {', '.join(P['out_of_scope']) or 'none'}. "
             f"PKMAS 'PD vs Control' against the source tables: {P['group_label_disagreements']} rows disagree{(' (' + ', '.join(P['group_label_disagreement_rows']) + ')') if P['group_label_disagreement_rows'] else ''}.")
    L.append("")
    L.append("Non-lockbox trials:")
    L.append("")
    L.append("| task | control trials matched / present | PD trials matched / present |")
    L.append("|---|---|---|")
    mt, pt = P["matched_trials_by_task_group_non_lockbox"], P["trials_by_task_group_non_lockbox"]
    for t in PKMAS_TASKS:
        L.append(f"| {t} | {mt[f'{t}|control']} / {pt[f'{t}|control']} | {mt[f'{t}|PD']} / {pt[f'{t}|PD']} |")
    L.append("")
    L.append(f"Non-lockbox persons with no matched PKMAS row in either task: control {P['persons_without_match_non_lockbox']['control']} of {P['persons_non_lockbox']['control']}, PD {P['persons_without_match_non_lockbox']['PD']} of {P['persons_non_lockbox']['PD']}. "
             f"PKMAS rows whose metric columns are identical to another row's: {len(P['duplicate_metric_rows'])}"
             + (" (" + "; ".join(P["duplicate_metric_rows"]) + ")" if P["duplicate_metric_rows"] else "")
             + ". A duplicated row cannot hold two different recordings; trials.csv flags it in pkmas_duplicate_metrics_of.")
    L.append("")
    L.append("Pairing evidence (non-lockbox matched rows): walkway cadence of the CSV trial against PKMAS cadence, for the declared trial and the _mat alternatives of the same person:")
    L.append("")
    L.append("| PKMAS task | CSV task | n | median abs diff (steps/min) | MAE | Pearson r |")
    L.append("|---|---|---|---|---|---|")
    for k, E in P["pairing_cadence_evidence"].items():
        a, b = k.split("|")
        L.append(f"| {a} | {b} | {E['n']} | {f1(E['median_abs'], 2)} | {f1(E['mae'], 2)} | {f1(E['pearson_r'], 3)} |")
    L.append("")
    if P["second_visit_evidence"]:
        L.append("Second-visit rows against the CSV trial of the base ID (cadence, steps/min): " + "; ".join(
            f"{e['participant']} {e['task']} PKMAS {f1(e['pkmas_cadence'], 1)}, CSV {f1(e['csv_cadence_contacts'], 1)}" for e in P["second_visit_evidence"]) + ".")
        L.append("")
    C = P["cell_fit"]
    L.append(f"Walkway cell size (cells to cm): median of PKMAS stride length over walkway stride length in cells, non-lockbox SelfPace trials, n {C['n']}: {f1(C['cell_cm'], 4)} cm (p10 {f1(C['p10'], 4)}, p90 {f1(C['p90'], 4)}). HurriedPace trials are not in the fit.")
    L.append("")
    L.append(f"Matched non-lockbox pairings whose walkway cadence differs from PKMAS cadence by more than {CADENCE_CHECK} steps/min (the CSV walkway contacts may not be the recording PKMAS processed): "
             f"{len(P['cadence_not_verified_non_lockbox'])}{(' (' + '; '.join(P['cadence_not_verified_non_lockbox']) + ')') if P['cadence_not_verified_non_lockbox'] else ''}. "
             "The rows 'all, cadence_verified' leave them out.")
    L.append("")
    L.append("Walkway-derived against PKMAS (non-lockbox matched trials; diff = walkway minus PKMAS):")
    L.append("")
    L.append("| task | group | pairs | quantity | n | MAE | median abs | p90 abs | bias | Pearson r |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, Ag in P["agreement"].items():
        t, gg, subset = k.split("|")
        for q, nd in (("velocity_m_s", 3), ("stride_length_cm", 2), ("cadence_steps_min", 2)):
            E = Ag[q]
            L.append(f"| {t} | {gg} | {subset} | {q} | {E['n']} | {f1(E['mae'], nd)} | {f1(E['median_abs'], nd)} | {f1(E['p90_abs'], nd)} | {f1(E['bias'], nd)} | {f1(E['pearson_r'], 3)} |")
    L.append("")
    L.append("Strides per trial, median (PKMAS #Samples against walkway strides counted here): " + "; ".join(
        f"{k.split('|')[0]} {k.split('|')[1]} {f1(Ag['strides_per_trial_pkmas_median'], 0)} vs {f1(Ag['strides_per_trial_walkway_median'], 0)}"
        for k, Ag in P["agreement"].items() if k.split("|")[1] in ("control", "PD")) + ".")
    L.append("")
    L.append("## Xsens orientation columns (Roll, Pitch, Yaw)")
    L.append("")
    L.append("| site | control present / ok | control finite share median | control trials with >1 % zero triples | PD present / ok | PD finite share median | PD trials with >1 % zero triples |")
    L.append("|---|---|---|---|---|---|---|")
    Oc, Op = S["orientation"]["control"], S["orientation"]["PD"]
    for s in CANON_ORDER:
        a, b = Oc["sites"][s], Op["sites"][s]
        L.append(f"| {s} | {a['present']} / {a['ok']} | {f1(a['finite_median'], 4)} | {a['zero_gt_1pct']} | {b['present']} / {b['ok']} | {f1(b['finite_median'], 4)} | {b['zero_gt_1pct']} |")
    L.append("")
    L.append(f"Trials with all 13 sensors ok: control {Oc['all_13_ok']} of {Oc['trials']}, PD {Op['all_13_ok']} of {Op['trials']}. Trials whose header carries Acc and Gyr for all 13 sensors: control {Oc['imu_sites_header_13']}, PD {Op['imu_sites_header_13']}. "
             f"Trials with Roll, Pitch and Yaw for all 13 sensors: control {Oc['all_13_present']}, PD {Op['all_13_present']}. Trials with a present sensor below {ORI_OK_FRAC} finite: control {Oc['trials_with_a_present_sensor_below_ok']}, PD {Op['trials_with_a_present_sensor_below_ok']}; "
             f"below 0.5 finite: control {Oc['trials_with_a_present_sensor_below_half']}, PD {Op['trials_with_a_present_sensor_below_half']}. Lowest finite share of a present sensor: control {f1(Oc['min_finite_share_present'], 4)}, PD {f1(Op['min_finite_share_present'], 4)}.")
    L.append("")
    L.append("## Completeness of targets")
    L.append("")
    L.append("Share of non-lockbox trials (percent) meeting each component (definitions in the docstring); c_pkmas applies to SelfPace and HurriedPace only.")
    L.append("")
    L.append("| group | task | trials | c_phases | c_walkway | c_insole | c_orientation | c_pkmas | complete (n) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, D in S["completeness"].items():
        gg, t = k.split("|")
        L.append(f"| {gg} | {t} | {D['trials']} | {pc(D['c_phases'])} | {pc(D['c_walkway'])} | {pc(D['c_insole'])} | {pc(D['c_orientation'])} | {pc(D['c_pkmas'])} | {pc(D['complete'])} ({D['complete_n']}) |")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("trials.csv (one row per trial), events.csv (one row per non-censored walkway IC or FO with the nearest insole event, the per-step walkway positions and step and stride lengths of the ICs), phases.csv (GeneralEvent runs), pkmas_match.csv (one row per PKMAS row with its match), summary.json. cache/ (gitignored) holds the per-trial results and cache/insole_events.csv, every insole event of every trial.")
    L.append("")
    L.append(f"Runtime: {S['runtime_s']} s ({S['trials_from_cache']} trials from the cache, {S['workers']} workers{', limit ' + str(S['limit']) if S['limit'] else ''}).")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
