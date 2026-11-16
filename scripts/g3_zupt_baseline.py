"""G3 ZUPT baseline (wave 16, task ZB, 2026-10-15): the label-free foot-IMU stride readout of task SCALE kept as a
G3 dev baseline. Every output is labelled LABEL ('dev baseline, not a G3 read').

Prediction: reports/gates/g3/zupt_dev/prediction.txt, hashed before the first dev run of this script. Its SHA-256 sits
beside it in prediction.sha256 and here in PRED_SHA256; check_prediction asserts that the three agree before any
target or record is read, so a prediction edited after the fact stops the run.

Method, the version-2 ZUPT of task SCALE ported unchanged (scratchpad wave15/SCALE/zupt2.py, SHA-256
b0b399f37a9a8558d2e25747bf0cc834a454050d74a53e06cfae27f22c19c5e5, copy reports/bench/scale/code/zupt2.py;
docs/notes/SCALE_2026-10-15.md section 2, test Z). Its constants are the module constants below.
- Input: the foot_L and foot_R sensors (canonical sites 11 and 12, g3.FOOT_SITES) of each dev IMU record as G3 gives
  it to a model (g3.model_input, magnetometer masked); acc in g with gravity, gyro in deg/s, 100 Hz; a sample whose
  sample_valid is False is NaN; a foot whose acc or gyro channels are masked gives no stride.
- Still samples: |w| < W_STILL_DPS and | |f| - 1 g | < F_STILL_G, both on a centred moving median of STILL_SMOOTH
  samples; still runs: maximal runs of still samples at least MIN_STILL_S long.
- One stride per pair of consecutive still runs (a, b) of one foot, from m_a (minimum |w| inside run a) to m_b (same
  in run b), kept when m_b - m_a lies in [MIN_STRIDE_S, MAX_STRIDE_S].
- At m_a: the tilt from the mean specific force of the TILT_HALF samples on each side of m_a (inside run a), yaw 0, no
  gyro bias; orientation by the exact exponential of each sample pair's mean rate; world acceleration R f - g z;
  velocity by the trapezoid from 0, its value at m_b removed linearly in time; position by the trapezoid; stride
  length the horizontal norm of the position at m_b. Nothing is fitted, no walkway value enters the length.
- Event times (fixed in this script before its dev run; prediction.txt section 1): each ZUPT stride's ends for the G3
  matching are the first samples of its two still runs, t0 = run a start / fps and t1 = run b start / fps, the
  foot-flat onsets, the first instants at which the method itself holds the foot landed. Nothing is fitted to the
  walkway timing.
- Pass speed (fixed with the event times): per walkway pass, the sum of the lengths over the sum of (t1 - t0) of the
  ZUPT strides of both feet whose t0 and t1 lie inside the pass interval widened by g3.STRIDE_MATCH_S on each side;
  no such stride, no speed (the pass is unmatched).

Lines: mdfm/gates/g3.py, imported, never edited. g3.load_targets and g3.imu_records read the 14 dev controls (lockbox
refused by the harness and by check_persons); g3.match_strides (both ends within 0.15 s, the nearest by the larger
end difference) and g3.match_speeds match; g3.evaluate on g3.trial_sets (dev14, every dev trial; dev12, the 14 without
WHC003 and WHC026, equal on dev to the valid inputs of docs/rules/G3_RULE.md section 2) gives stride_mae_cm,
stride_mape_pct and speed_mae_m_s with matched shares and g3's person-clustered intervals (2000 resamples, seed
20260929), beside the trivial baselines of the rule's section 7 computed by the same code: the population baseline
(per task the person-weighted mean of the training fold's walkway targets, the only training-fold data read, as
g3.dev_run reads it), the shank-gyroscope detector (IC line, offsets fitted on the dev persons as g3.fit_offsets fits
them) and the walkway itself (harness check). Harness position: per floored line, common.verdict with the matched-share
floor (g3.matched_share_floor_ok), as if the ZUPT were the model; descriptive, it decides nothing (the rule judges no
baseline by the floor).

Same strides (descriptive): the body is v1's phase B end as the G3 diagnostic read it, the matched base stride and
speed rows of reports/gates/g3/diag_phaseA_root_full_ckpt_B_032044/events.csv; joined to the walkway strides on
(trial, foot, t0, t1 rounded to 0.01 s), with the walkway length asserted equal. On the strides that the body and
the ZUPT both match: the body raw, the body divided by F1's train54 stride gain g_S
(reports/bench/g3_gain_check/results.json, end v1_B: the 'F1 head') and the ZUPT; on the passes both match, the same
for speed with F1's g_V.

Reproduction of SCALE (asserted): SCALE selected its ZUPT strides with a walkway-keyed match (keyed_match, its
zupt2.match: start and end within KEY_LEAD_S of the walkway ICs, duration within KEY_DUR_TOL of the walkway stride,
the start nearest t0 + KEY_AIM_S). This script recomputes that selection on the same records and asserts (a) per dev
walkway stride the keyed ZUPT length equal to reports/bench/scale/zupt_v2/real_strides.csv within REPRO_TOL_M and the
same matched flags, (b) the counts and the dev12 points of reports/bench/scale/zupt_same_strides/results.json (table
SS) within REPRO_TOL_CM. Points are the harness's mean over persons of the per-person mean absolute error, so they do
not depend on the bootstrap; intervals here are g3's (2000 resamples, seed 20260929), SCALE's SS table used F1's
line_of (10000, seed 20261001), so its intervals differ in the second decimal.

Version 2 of this script (2026-10-15, after its first dev run; post hoc and descriptive, labelled so in the outputs):
the decomposition of the G3 line against SCALE's keyed readout (decomposition: the strides both selections match and
whether they pick the same ZUPT stride; the strides only the G3 matching takes, with their length ratio, end timing,
duration ratio and the end velocity the ZUPT removed, a quality figure of its own; the per-person table), the matched
ZUPT stride's removed end velocity and duration in strides.csv, and an infinite line value written as the string 'inf'
instead of null. No method, match, line or value of the first run changed (compared with the first run's outputs; see
reports/weekend_2026-10-01/tasks/ZB.md).

Never: the GPU (CUDA_VISIBLE_DEVICES=-1 is set before torch loads), a lockbox, EVAL, CODY or care_pd record. Dev
only: the records read are those of the 14 dev controls; a lockbox person anywhere in an input raises Refused.
Writes reports/gates/g3/zupt_dev/: results.json, summary.md, strides.csv (one row per dev walkway stride),
passes.csv (one row per dev walkway pass), zupt_strides.csv (every ZUPT stride formed) and _provenance.json
(scripts/moveatlas_stamp.py).
Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\
    <conda>/envs/mdfm/python.exe scripts/g3_zupt_baseline.py
"""
import hashlib
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(TREE, "scripts"), TREE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.gates import common, g3  # noqa: E402

OUT = os.path.join(TREE, "reports", "gates", "g3", "zupt_dev")
PRED_PATH = os.path.join(OUT, "prediction.txt")
PRED_SHA_PATH = os.path.join(OUT, "prediction.sha256")
PRED_SHA256 = "08cd4314e1ed50abaae1a40690590f4b2823cd22218db6531dd82f06ca97d7f7"
LABEL = "dev baseline, not a G3 read"
NAME = "zupt"
EVENTS = os.path.join(TREE, "reports", "gates", "g3", "diag_phaseA_root_full_ckpt_B_032044", "events.csv")
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
F1_END = "v1_B"
SCALE_STRIDES = os.path.join(TREE, "reports", "bench", "scale", "zupt_v2", "real_strides.csv")
SCALE_SS = os.path.join(TREE, "reports", "bench", "scale", "zupt_same_strides", "results.json")
SCALE_ZUPT2_SHA256 = "b0b399f37a9a8558d2e25747bf0cc834a454050d74a53e06cfae27f22c19c5e5"
SCALE_ZUPT2_COPY = os.path.join(TREE, "reports", "bench", "scale", "code", "zupt2.py")
EXCLUDED_FROM_12 = g3.EXCLUDED_FROM_12
RATE = 100.0
REPRO_TOL_M = 1e-9
REPRO_TOL_CM = 0.005                 # "within rounding": half a unit of the second decimal
KEY_ROUND = 2                        # events.csv writes t0_s and t1_s to 0.01 s

# the version-2 ZUPT of task SCALE (zupt2.py), constants unchanged
G0 = 9.80665
W_STILL_DPS = 45.0
F_STILL_G = 0.12
STILL_SMOOTH = 5
MIN_STILL_S = 0.05
MAX_STRIDE_S = 2.5
MIN_STRIDE_S = 0.6
TILT_HALF = 2
# SCALE's walkway-keyed selection (zupt2.match), used only by the reproduction
KEY_LEAD_S = (-0.10, 0.60)
KEY_DUR_TOL = 0.15
KEY_AIM_S = 0.2


# ---------------------------------------------------------------- prediction hash, guards
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(path=None, sha_path=None, expected=None):
    """The SHA-256 of the prediction file, asserted equal to the first word of the hash file and to the script
    constant (AssertionError otherwise, raised explicitly so that python -O keeps it). Defaults: the module constants
    at call time."""
    path = PRED_PATH if path is None else path
    sha_path = PRED_SHA_PATH if sha_path is None else sha_path
    expected = PRED_SHA256 if expected is None else expected
    sha = sha256_file(path)
    with open(sha_path, encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (sha == stored == expected):
        raise AssertionError(f"prediction hash mismatch: {os.path.basename(path)} {sha}, "
                             f"{os.path.basename(sha_path)} {stored or 'empty'}, script {expected}")
    return sha


def check_persons(persons, allowed, locked, what="persons"):
    """The set of person ids; Refused when one is a lockbox person, AssertionError when one lies outside `allowed`
    (None: no such check)."""
    ps = {str(p) for p in persons}
    hit = sorted(ps & {str(p) for p in locked})
    if hit:
        raise Refused(f"{what}: {len(hit)} lockbox persons; this baseline reads dev persons only")
    if allowed is not None:
        out = sorted(ps - {str(p) for p in allowed})
        if out:
            raise AssertionError(f"{what}: {len(out)} persons outside the dev controls ({', '.join(out[:3])})")
    return ps


def check_record(tid, rec, persons):
    """AssertionError unless the record carries lockbox False, no EVAL tier, and a person of `persons`."""
    lb = rec.get("lockbox")
    if lb is None or (lb is not False and str(lb).strip().lower() not in ("false", "0")):
        raise AssertionError(f"{tid}: record with lockbox {lb!r}")
    if str(rec.get("tier", "")) in ("EVAL", "EXCLUDED_V1"):
        raise AssertionError(f"{tid}: record of tier {rec.get('tier')}")
    if str(rec.get("person_id")) not in {str(p) for p in persons}:
        raise AssertionError(f"{tid}: person {rec.get('person_id')!r} outside the requested set")


# ---------------------------------------------------------------- ZUPT (SCALE zupt2.py, version 2)
def _movmed(x, k):
    if k <= 1:
        return x
    n = len(x)
    h = k // 2
    pad = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
    idx = np.arange(n)[:, None] + np.arange(k)[None, :]
    return np.median(pad[idx], axis=1)


def still_runs(acc_g, gyro_dps, fps, w_still=W_STILL_DPS, f_still=F_STILL_G, smooth=STILL_SMOOTH,
               min_still_s=MIN_STILL_S):
    """([(start, end_exclusive)] of the still runs, |w| per sample); a sample with a non-finite value is never
    still."""
    acc = np.asarray(acc_g, np.float64)
    gyr = np.asarray(gyro_dps, np.float64)
    ok = np.isfinite(acc).all(1) & np.isfinite(gyr).all(1)
    wn = np.where(ok, np.linalg.norm(np.nan_to_num(gyr), axis=1), np.inf)
    fn = np.where(ok, np.abs(np.linalg.norm(np.nan_to_num(acc), axis=1) - 1.0), np.inf)
    wn_s = _movmed(np.where(np.isfinite(wn), wn, 1e9), smooth)
    fn_s = _movmed(np.where(np.isfinite(fn), fn, 1e9), smooth)
    still = (wn_s < w_still) & (fn_s < f_still) & ok
    runs = []
    i, n = 0, len(still)
    m = max(1, int(round(min_still_s * fps)))
    while i < n:
        if still[i]:
            j = i
            while j < n and still[j]:
                j += 1
            if j - i >= m:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs, wn


def _exp_so3(phi):
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        return np.eye(3)
    k = phi / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _tilt(f_mean):
    """Rotation R (sensor to world, world z up) that maps the mean specific force to +z, yaw 0."""
    u = f_mean / np.linalg.norm(f_mean)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(np.dot(u, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def integrate(acc_g, gyro_dps, fps, run_a, run_b, wn):
    """One stride between still runs a and b (module docstring): dict, or None when a sample is not finite."""
    a0, a1 = run_a
    b0, b1 = run_b
    ma = a0 + int(np.argmin(wn[a0:a1]))
    mb = b0 + int(np.argmin(wn[b0:b1]))
    f = np.asarray(acc_g[ma:mb + 1], np.float64) * G0
    w = np.deg2rad(np.asarray(gyro_dps[ma:mb + 1], np.float64))
    if not (np.isfinite(f).all() and np.isfinite(w).all()):
        return None
    dt = 1.0 / fps
    bias = np.zeros(3)
    lo, hi = max(a0, ma - TILT_HALF), min(a1, ma + TILT_HALF + 1)
    ft = np.asarray(acc_g[lo:hi], np.float64)
    if not np.isfinite(ft).all():
        return None
    R = _tilt(ft.mean(0))
    n = len(f)
    aw = np.empty((n, 3))
    for i in range(n):
        aw[i] = R @ f[i]
        if i < n - 1:
            R = R @ _exp_so3((0.5 * (w[i] + w[i + 1]) - bias) * dt)
    aw[:, 2] -= G0
    v = np.zeros((n, 3))
    v[1:] = np.cumsum(0.5 * (aw[1:] + aw[:-1]) * dt, axis=0)
    t = np.arange(n) * dt
    T = t[-1]
    v_end = v[-1].copy()
    v = v - (t / T)[:, None] * v_end[None, :]
    p = np.zeros((n, 3))
    p[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)
    vh = np.linalg.norm(v[:, :2], axis=1)
    return {"m_a": ma, "m_b": mb, "t_a": ma / fps, "t_b": mb / fps, "duration_s": T,
            "length_m": float(np.linalg.norm(p[-1, :2])), "vertical_m": float(p[-1, 2]),
            "peak_speed_m_s": float(vh.max()), "swing_s": (b0 - a1) / fps, "v_end_m_s": float(np.linalg.norm(v_end))}


def zupt_strides(acc_g, gyro_dps, fps=RATE):
    """[dict] of every ZUPT stride between consecutive still runs whose duration m_b - m_a lies in [MIN_STRIDE_S,
    MAX_STRIDE_S] (SCALE's zupt2.strides), each with its still runs and its event times t0_s and t1_s (the starts
    of its two still runs, the G3 matching times of this script)."""
    runs, wn = still_runs(acc_g, gyro_dps, fps)
    out = []
    for ra, rb in zip(runs[:-1], runs[1:]):
        ma = ra[0] + int(np.argmin(wn[ra[0]:ra[1]]))
        mb = rb[0] + int(np.argmin(wn[rb[0]:rb[1]]))
        d = (mb - ma) / fps
        if d < MIN_STRIDE_S or d > MAX_STRIDE_S:
            continue
        r = integrate(acc_g, gyro_dps, fps, ra, rb, wn)
        if r is not None:
            r["run_a"], r["run_b"] = ra, rb
            r["t0_s"], r["t1_s"] = ra[0] / fps, rb[0] / fps
            out.append(r)
    return out


def keyed_match(zs, t0, t1, lead_s=KEY_LEAD_S, dur_tol=KEY_DUR_TOL, aim_s=KEY_AIM_S):
    """SCALE's walkway-keyed selection (zupt2.match), for the reproduction only: the ZUPT stride whose start t_a lies
    in [t0 + lead_s[0], t0 + lead_s[1]] and whose end t_b lies in [t1 + lead_s[0], t1 + lead_s[1]], its duration
    within dur_tol of t1 - t0 (relative); the one whose start is nearest t0 + aim_s when several qualify; None
    otherwise."""
    best, bd = None, None
    for z in zs:
        if not (t0 + lead_s[0] <= z["t_a"] <= t0 + lead_s[1]):
            continue
        if not (t1 + lead_s[0] <= z["t_b"] <= t1 + lead_s[1]):
            continue
        if abs(z["duration_s"] - (t1 - t0)) > dur_tol * (t1 - t0):
            continue
        d = abs(z["t_a"] - (t0 + aim_s))
        if bd is None or d < bd:
            best, bd = z, d
    return best


def foot_channels(record):
    """{foot: (acc (T, 3) g, gyro (T, 3) deg/s)} of the record as G3 gives it to a model (g3.model_input), float64,
    NaN where sample_valid is False; a foot whose acc or gyro channels are masked, or whose site is absent, is left
    out."""
    rec = g3.model_input(record)
    sig = np.asarray(rec["signal"], dtype=np.float64)
    loc = [int(x) for x in np.asarray(rec["location_ids"])]
    cm = np.asarray(rec["channel_mask"], dtype=bool)
    valid = np.asarray(rec["sample_valid"], dtype=bool) if rec.get("sample_valid") is not None \
        else np.ones(sig.shape[:2], bool)
    out = {}
    for foot, site in g3.FOOT_SITES.items():
        if site in loc:
            j = loc.index(site)
            if cm[j, g3.ACC].all() and cm[j, g3.GYRO].all():
                a = sig[:, j, g3.ACC].copy()
                g = sig[:, j, g3.GYRO].copy()
                a[~valid[:, j]] = np.nan
                g[~valid[:, j]] = np.nan
                out[foot] = (a, g)
    return out


# ---------------------------------------------------------------- predictors
ZCOLS = ["trial_id", "foot", "t0", "t1", "length_m", "t_a", "t_b", "duration_s", "peak_speed_m_s", "v_end_m_s"]


def zupt_frame(by_trial):
    """DataFrame (ZCOLS) of every ZUPT stride of {trial_id: {foot: [stride dicts]}}; t0 and t1 are the event times
    (still-run starts)."""
    import pandas as pd
    rows = [(tid, foot, z["t0_s"], z["t1_s"], z["length_m"], z["t_a"], z["t_b"], z["duration_s"], z["peak_speed_m_s"],
             z["v_end_m_s"]) for tid in sorted(by_trial) for foot in sorted(by_trial[tid]) for z in by_trial[tid][foot]]
    return pd.DataFrame(rows, columns=ZCOLS)


def pass_speeds(zf, passes, tol=None):
    """Per walkway pass (passes: trial_id, pass_id, t0, t1): the ZUPT pass speed of the module docstring, sum of the
    lengths over sum of (t1 - t0) of the ZUPT strides of both feet with t0 >= pass t0 - tol and t1 <= pass t1 + tol
    (tol default g3.STRIDE_MATCH_S). DataFrame(trial_id, pass_id, speed_m_s, n_strides) for the passes with at least
    one such stride."""
    import pandas as pd
    tol = g3.STRIDE_MATCH_S if tol is None else float(tol)
    by = {t: g for t, g in zf.groupby("trial_id")} if len(zf) else {}
    rows = []
    for r in passes.itertuples():
        g = by.get(r.trial_id)
        if g is None:
            continue
        k = g[(g["t0"] >= r.t0 - tol - 1e-9) & (g["t1"] <= r.t1 + tol + 1e-9)]
        d = float((k["t1"] - k["t0"]).sum())
        if len(k) and d > 0:
            rows.append((r.trial_id, r.pass_id, float(k["length_m"].sum()) / d, int(len(k))))
    return pd.DataFrame(rows, columns=["trial_id", "pass_id", "speed_m_s", "n_strides"])


def zupt_predictions(zf, targets, name=NAME):
    """g3.Predictions of the ZUPT: strides (trial_id, foot, t0, t1, length_m) at the event times, speeds per walkway
    pass (pass_speeds); no IC."""
    sp = pass_speeds(zf, targets.passes)
    return g3.Predictions(name, None, zf[["trial_id", "foot", "t0", "t1", "length_m"]].copy(),
                          sp[["trial_id", "pass_id", "speed_m_s"]].copy())


def nearest_ends(pred_strides, strides):
    """(t0, t1) of the predicted stride g3.match_strides pairs with each walkway stride (same trial and foot, the
    nearest by the larger end difference; NaN without one), in the order of `strides`."""
    by = {k: g for k, g in pred_strides.groupby(["trial_id", "foot"])} if len(pred_strides) else {}
    a, b = [], []
    for r in strides.itertuples():
        g = by.get((r.trial_id, r.foot))
        if g is None:
            a.append(np.nan)
            b.append(np.nan)
            continue
        d = np.maximum(np.abs(g["t0"].to_numpy() - r.t0), np.abs(g["t1"].to_numpy() - r.t1))
        k = int(np.argmin(d))
        a.append(float(g["t0"].iloc[k]))
        b.append(float(g["t1"].iloc[k]))
    return np.asarray(a), np.asarray(b)


def keyed_table(by_trial, strides):
    """The walkway strides with SCALE's keyed selection: keyed_m (length, NaN when none), keyed_matched, keyed_ta and
    keyed_tb (m_a and m_b of the selected stride, s)."""
    km, kt0, kt1 = [], [], []
    for r in strides.itertuples():
        zs = by_trial.get(r.trial_id, {}).get(r.foot)
        m = keyed_match(zs, float(r.t0), float(r.t1)) if zs is not None else None
        km.append(np.nan if m is None else m["length_m"])
        kt0.append(np.nan if m is None else m["t_a"])
        kt1.append(np.nan if m is None else m["t_b"])
    out = strides[["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1", "length_m"]].copy()
    out["keyed_m"] = km
    out["keyed_matched"] = np.isfinite(out["keyed_m"].to_numpy(np.float64))
    out["keyed_ta"], out["keyed_tb"] = kt0, kt1
    return out


# ---------------------------------------------------------------- the body (v1's B end, diag events)
def read_body_events(path, allowed, locked, row="base"):
    """(strides, speeds) of a G3 diag events.csv (scripts/eval_g3_diag.py columns): the persons of the file are read
    first and checked (check_persons: lockbox refused, every person a dev control), then the matched rows of `row`:
    strides with trial_id, person_id, foot, pass_id, k0, k1 (t0_s and t1_s rounded to KEY_ROUND), x (walkway, cm),
    y (body, cm); speeds with trial_id, person_id, pass_id, x (m/s), y (m/s)."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    check_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(path))
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[(ev["row"] == row) & (ev["matched"].str.strip().str.lower() == "true")]
    st = ev[ev["kind"] == "stride"]
    sp = ev[ev["kind"] == "speed"]
    if (st["unit"] != "cm").any() or (sp["unit"] != "m/s").any():
        raise ValueError(f"{os.path.basename(path)}: stride rows not in cm or speed rows not in m/s")
    strides = pd.DataFrame({"trial_id": st["trial_id"].to_numpy(), "person_id": st["person_id"].to_numpy(),
                            "foot": st["foot"].to_numpy(), "pass_id": st["pass_id"].to_numpy(),
                            "k0": pd.to_numeric(st["t0_s"]).round(KEY_ROUND).to_numpy(),
                            "k1": pd.to_numeric(st["t1_s"]).round(KEY_ROUND).to_numpy(),
                            "x": pd.to_numeric(st["reference"]).to_numpy(np.float64),
                            "y": pd.to_numeric(st["prediction"]).to_numpy(np.float64)})
    speeds = pd.DataFrame({"trial_id": sp["trial_id"].to_numpy(), "person_id": sp["person_id"].to_numpy(),
                           "pass_id": sp["pass_id"].to_numpy(),
                           "x": pd.to_numeric(sp["reference"]).to_numpy(np.float64),
                           "y": pd.to_numeric(sp["prediction"]).to_numpy(np.float64)})
    for name, d in (("stride", strides), ("speed", speeds)):
        if len(d) and not np.isfinite(d[["x", "y"]].to_numpy()).all():
            raise ValueError(f"{os.path.basename(path)}: a matched {name} row without a finite value")
    if strides.duplicated(["trial_id", "foot", "k0", "k1"]).any():
        raise ValueError(f"{os.path.basename(path)}: two stride rows share trial, foot and end times")
    return strides, speeds


def f1_gains(path=F1_RESULTS, end=F1_END):
    """(g_S, g_V) of F1's end `end` (stride and speed gains fitted on the 54 training-fold controls)."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    g = r["ends"][end]["gain"]
    return float(g["stride"]["gain"]), float(g["speed"]["gain"])


# ---------------------------------------------------------------- lines
def line_of(values, persons):
    """The harness's line statistic (g3._line: mean over persons of the per-person mean, person-clustered bootstrap,
    g3.BOOT_N resamples, seed g3.SEED), without the per-person table."""
    out = g3._line(np.asarray(values, np.float64), np.asarray(persons).astype(str))
    return {k: v for k, v in out.items() if k != "per_person"} | {"per_person": out.get("per_person", {})}


def same_stride_lines(table, persons, g_s):
    """Lines (cm) on the walkway strides of `persons` that the body and the ZUPT (G3 matching) both match, and on
    those that the body and SCALE's keyed selection both match. table: strides.csv rows (length_m, g3_matched,
    g3_pred_m, keyed_matched, keyed_m, body_matched, body_cm)."""
    t = table[table["person_id"].isin(set(persons))]
    out = {"walkway_strides": int(len(t)), "body_matched": int(t["body_matched"].sum()),
           "zupt_matched": int(t["g3_matched"].sum()), "keyed_matched": int(t["keyed_matched"].sum())}
    x = 100.0 * t["length_m"].to_numpy(np.float64)
    both = (t["body_matched"] & t["g3_matched"]).to_numpy()
    kb = (t["body_matched"] & t["keyed_matched"]).to_numpy()
    out["both_g3"], out["both_keyed"] = int(both.sum()), int(kb.sum())
    p = t["person_id"].to_numpy()
    y_b = t["body_cm"].to_numpy(np.float64)
    y_z = 100.0 * t["g3_pred_m"].to_numpy(np.float64)
    y_k = 100.0 * t["keyed_m"].to_numpy(np.float64)
    bm = t["body_matched"].to_numpy()
    km = t["keyed_matched"].to_numpy()
    out["lines_cm"] = {
        "body raw, every body-matched stride": line_of(np.abs(y_b[bm] - x[bm]), p[bm]),
        "body with F1 head, every body-matched stride": line_of(np.abs(y_b[bm] / g_s - x[bm]), p[bm]),
        "body raw, strides the ZUPT also matches (G3 matching)": line_of(np.abs(y_b[both] - x[both]), p[both]),
        "body with F1 head, same strides (G3 matching)": line_of(np.abs(y_b[both] / g_s - x[both]), p[both]),
        "ZUPT, same strides (G3 matching)": line_of(np.abs(y_z[both] - x[both]), p[both]),
        "body raw, strides SCALE's keyed ZUPT also matches": line_of(np.abs(y_b[kb] - x[kb]), p[kb]),
        "body with F1 head, same strides (keyed)": line_of(np.abs(y_b[kb] / g_s - x[kb]), p[kb]),
        "ZUPT keyed, same strides (keyed): SCALE's SS point": line_of(np.abs(y_k[kb] - x[kb]), p[kb]),
        "ZUPT keyed, every keyed-matched stride": line_of(np.abs(y_k[km] - x[km]), p[km])}
    return out


def same_pass_lines(passes, persons, g_v):
    """Lines (m/s) on the walkway passes of `persons` that the body and the ZUPT both match."""
    t = passes[passes["person_id"].isin(set(persons))]
    both = (t["body_matched"] & t["zupt_matched"]).to_numpy()
    x = t["speed_m_s"].to_numpy(np.float64)
    yb = t["body_m_s"].to_numpy(np.float64)
    yz = t["zupt_m_s"].to_numpy(np.float64)
    p = t["person_id"].to_numpy()
    return {"walkway_passes": int(len(t)), "body_matched": int(t["body_matched"].sum()),
            "zupt_matched": int(t["zupt_matched"].sum()), "both": int(both.sum()),
            "lines_m_s": {"body raw, same passes": line_of(np.abs(yb[both] - x[both]), p[both]),
                          "body with F1 head, same passes": line_of(np.abs(yb[both] / g_v - x[both]), p[both]),
                          "ZUPT, same passes": line_of(np.abs(yz[both] - x[both]), p[both])}}


def harness_position(stats):
    """{line: {point, matched_share, floor_ok, verdict}} of the ZUPT's floored lines, read as if the ZUPT were the
    model (common.verdict with g3.matched_share_floor_ok as its condition); descriptive, decides nothing."""
    out = {}
    for line in g3.FLOORED_LINES:
        st = stats.get(line, {})
        pl, kl = g3.LINES[line]
        share = st.get("matched_share", float("nan"))
        ok = g3.matched_share_floor_ok(share)
        out[line] = {"point": st.get("point"), "matched_share": share, "floor_ok": ok, "pass_line": pl,
                     "kill_line": kl, "verdict": common.verdict(st.get("point"), pl, kl, conditions=(ok,))}
    return out


def _q(v, qs=(0.1, 0.5, 0.9)):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return {f"p{int(round(100 * q)):02d}": (float(np.quantile(v, q)) if len(v) else float("nan")) for q in qs} | \
        {"n": int(len(v))}


def coverage(table, persons):
    """Where the G3 matching and SCALE's keyed selection differ, on the walkway strides of `persons`: counts, the
    signed end differences of the ZUPT event times against the walkway ICs on the G3-matched strides, and the end
    difference of the nearest ZUPT stride on the strides only the keyed selection matches."""
    t = table[table["person_id"].isin(set(persons))]
    g, k = t["g3_matched"].to_numpy(), t["keyed_matched"].to_numpy()
    nf = t["no_foot_sensor"].to_numpy()
    return {"walkway_strides": int(len(t)), "g3_matched": int(g.sum()), "keyed_matched": int(k.sum()),
            "both": int((g & k).sum()), "g3_only": int((g & ~k).sum()), "keyed_only": int((~g & k).sum()),
            "neither": int((~g & ~k).sum()), "no_foot_sensor": int(nf.sum()),
            "g3_matched_share": float(g.mean()) if len(t) else float("nan"),
            "keyed_matched_share": float(k.mean()) if len(t) else float("nan"),
            "signed_t0_diff_s_g3_matched": _q((t["zupt_t0"] - t["t0"]).to_numpy(np.float64)[g]),
            "signed_t1_diff_s_g3_matched": _q((t["zupt_t1"] - t["t1"]).to_numpy(np.float64)[g]),
            "end_diff_s_keyed_only": _q(t["g3_end_diff_s"].to_numpy(np.float64)[~g & k]),
            "length_ratio_g3_matched_median": float(np.median((t["g3_pred_m"] / t["length_m"]).to_numpy()[g]))
            if g.any() else float("nan")}


def _brief(values, persons):
    out = line_of(values, persons)
    out.pop("per_person", None)
    return out


def decomposition(table, persons):
    """Post hoc (version 2), descriptive: the ZUPT's G3 line against SCALE's keyed readout, on the walkway strides of
    `persons`. Groups: both (matched by the G3 matching and by the keyed selection), g3_only, keyed_only. Per group:
    the count and persons; the line (cm) of the G3-matched length (both, g3_only) and of the keyed length (both,
    keyed_only); for both, the share where the two select the same ZUPT length; for the G3-matched groups the median
    ZUPT over walkway length, the share of errors above 10 cm and the end velocity the ZUPT removed (its own quality
    figure). per_person: per person the counts and mean absolute errors of the G3-matched, keyed and g3_only strides."""
    t = table[table["person_id"].isin(set(persons))]
    g = t["g3_matched"].to_numpy(bool)
    k = t["keyed_matched"].to_numpy(bool)
    x = 100.0 * t["length_m"].to_numpy(np.float64)
    eg = np.abs(100.0 * t["g3_pred_m"].to_numpy(np.float64) - x)
    ek = np.abs(100.0 * t["keyed_m"].to_numpy(np.float64) - x)
    ve = t["zupt_v_end_m_s"].to_numpy(np.float64)
    p = t["person_id"].to_numpy().astype(str)
    out = {}
    for name, m in (("both", g & k), ("g3_only", g & ~k), ("keyed_only", ~g & k)):
        d = {"n": int(m.sum()), "persons": int(len(set(p[m])))}
        if name != "keyed_only":
            d["g3_line_cm"] = _brief(eg[m], p[m])
            d["median_ratio"] = float(np.median((100.0 * t["g3_pred_m"].to_numpy(np.float64) / x)[m])) if m.any() \
                else float("nan")
            d["share_error_above_10_cm"] = float(np.mean(eg[m] > 10.0)) if m.any() else float("nan")
            d["v_end_removed_m_s"] = _q(ve[m])
            d["signed_t0_diff_s"] = _q((t["zupt_t0"] - t["t0"]).to_numpy(np.float64)[m])
            d["signed_t1_diff_s"] = _q((t["zupt_t1"] - t["t1"]).to_numpy(np.float64)[m])
            d["event_over_walkway_duration"] = _q((t["zupt_event_duration_s"] / (t["t1"] - t["t0"]))
                                                  .to_numpy(np.float64)[m])
        if name != "g3_only":
            d["keyed_line_cm"] = _brief(ek[m], p[m])
        if name == "both":
            d["same_length_share"] = float(np.mean(np.isclose(eg[m], ek[m], rtol=0.0, atol=1e-9))) if m.any() \
                else float("nan")
        out[name] = d
    pp = {}
    for person in sorted(set(p)):
        w = p == person
        pp[person] = {"n_g3": int((w & g).sum()), "g3_mae_cm": float(eg[w & g].mean()) if (w & g).any() else None,
                      "n_keyed": int((w & k).sum()), "keyed_mae_cm": float(ek[w & k].mean()) if (w & k).any() else None,
                      "n_g3_only": int((w & g & ~k).sum()),
                      "g3_only_mae_cm": float(eg[w & g & ~k].mean()) if (w & g & ~k).any() else None}
    out["per_person"] = pp
    return out


# ---------------------------------------------------------------- reproduction of SCALE
def scale_reference(strides_path=SCALE_STRIDES, ss_path=SCALE_SS, dev=None, locked=()):
    """(dev rows of SCALE's zupt_v2/real_strides.csv, SCALE's SS results). The persons of the CSV are checked against
    the lockbox list before its rows are kept (it also holds training-fold rows, which are dropped unread)."""
    import pandas as pd
    who = pd.read_csv(strides_path, encoding="utf-8", usecols=["person_id"], dtype=str)
    check_persons(set(who["person_id"].dropna()), None, locked, os.path.basename(strides_path))
    z = pd.read_csv(strides_path, encoding="utf-8", dtype={"trial_id": str, "person_id": str, "foot": str,
                                                            "pass_id": str, "task": str})
    if dev is not None:
        z = z[z["person_id"].isin(set(dev))].reset_index(drop=True)
    z["matched"] = z["matched"].astype(str).str.strip().str.lower() == "true"
    z["k0"], z["k1"] = z["t0"].round(KEY_ROUND), z["t1"].round(KEY_ROUND)
    with open(ss_path, encoding="utf-8") as f:
        ss = json.load(f)
    return z, ss


def reproduction(table, scale_rows, ss, lines12, tol_m=REPRO_TOL_M, tol_cm=REPRO_TOL_CM):
    """The checks of the module docstring (a) and (b): per stride (keyed length and flag against SCALE's CSV), counts
    and points against SCALE's SS table (dev12). Returns a dict with ok and every compared value; raises nothing (the
    caller asserts ok)."""
    a = table.assign(k0=table["t0"].round(KEY_ROUND), k1=table["t1"].round(KEY_ROUND))
    j = a.merge(scale_rows[["trial_id", "foot", "k0", "k1", "zupt_m", "matched"]], on=["trial_id", "foot", "k0", "k1"],
                how="outer", indicator=True)
    unpaired = int((j["_merge"] != "both").sum())
    jb = j[j["_merge"] == "both"]
    flags_equal = bool((jb["keyed_matched"].astype(bool).to_numpy() == jb["matched"].astype(bool).to_numpy()).all())
    m = jb["keyed_matched"].astype(bool) & jb["matched"].astype(bool)
    diff = np.abs(jb.loc[m, "keyed_m"].to_numpy(np.float64) - jb.loc[m, "zupt_m"].to_numpy(np.float64))
    max_diff = float(diff.max()) if len(diff) else float("nan")
    want = ss["lines_cm"]
    got_ss = lines12["lines_cm"]["ZUPT keyed, same strides (keyed): SCALE's SS point"]["point"]
    got_all = lines12["lines_cm"]["ZUPT keyed, every keyed-matched stride"]["point"]
    got_braw = lines12["lines_cm"]["body raw, strides SCALE's keyed ZUPT also matches"]["point"]
    got_bhead = lines12["lines_cm"]["body with F1 head, same strides (keyed)"]["point"]
    pts = {"ZUPT, strides the body also matched": (got_ss, want["ZUPT, strides the body also matched"]["point"]),
           "ZUPT, all ZUPT-matched strides": (got_all, want["ZUPT, all ZUPT-matched strides"]["point"]),
           "body raw, strides the ZUPT also matched": (got_braw, want["body raw, strides the ZUPT also matched"]["point"]),
           "body with F1 head, strides the ZUPT also matched":
               (got_bhead, want["body with F1 head, strides the ZUPT also matched"]["point"])}
    counts = {"walkway strides": (lines12["walkway_strides"], ss["dev12_walkway_strides"]),
              "body matched": (lines12["body_matched"], ss["body_matched"]),
              "keyed matched": (lines12["keyed_matched"], ss["zupt_matched"]),
              "both": (lines12["both_keyed"], ss["both_matched"])}
    points_ok = all(abs(float(g) - float(w)) <= tol_cm for g, w in pts.values())
    counts_ok = all(int(g) == int(w) for g, w in counts.values())
    per_stride_ok = unpaired == 0 and flags_equal and (not len(diff) or max_diff <= tol_m)
    return {"ok": bool(points_ok and counts_ok and per_stride_ok), "per_stride_ok": bool(per_stride_ok),
            "points_ok": bool(points_ok), "counts_ok": bool(counts_ok), "unpaired_rows": unpaired,
            "matched_flags_equal": flags_equal, "strides_compared": int(m.sum()), "max_abs_length_diff_m": max_diff,
            "tol_m": tol_m, "tol_cm": tol_cm,
            "points_cm": {k: {"here": float(g), "scale": float(w), "diff": float(g) - float(w)} for k, (g, w) in pts.items()},
            "counts": {k: {"here": int(g), "scale": int(w)} for k, (g, w) in counts.items()}}


# ---------------------------------------------------------------- the pipeline
def run(tg_dev, base, records, dev_persons, locked, body=None, gains=None, shank=True, log=print):
    """The pipeline on given dev targets, population baseline ({task: (stride_m, speed_m_s)}), record iterable of
    (trial_id, record) and, optionally, the body's matched rows (strides, speeds of read_body_events) with F1's
    gains (g_S, g_V). main passes the real ones, the tests synthetic ones. Returns (results dict, tables dict of
    DataFrames); writes nothing."""
    import pandas as pd
    t_start = time.time()
    dev_persons = sorted(dev_persons)
    check_persons(dev_persons, None, locked, "dev persons")
    check_persons(tg_dev.trials["person_id"], dev_persons, locked, "dev targets")
    if tg_dev.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg_dev.counts['lockbox_rows']} lockbox target rows kept")
    by_trial, gyros, feet_missing, n_rec = {}, {}, [], 0
    for tid, rec in records:
        check_record(tid, rec, dev_persons)
        n_rec += 1
        fc = foot_channels(rec)
        by_trial[tid] = {foot: zupt_strides(a, g, float(rec.get("fps", RATE))) for foot, (a, g) in fc.items()}
        feet_missing += [(tid, f) for f in ("L", "R") if f not in fc]
        if shank:
            gyros[tid] = g3.shank_gyro(rec)
    log(f"records {n_rec}, ZUPT strides {sum(len(v) for d in by_trial.values() for v in d.values())}, "
        f"{round(time.time() - t_start, 1)} s", flush=True)
    zf = zupt_frame(by_trial)
    zp = zupt_predictions(zf, tg_dev)
    preds = [g3.walkway_predictions(tg_dev), g3.population_predictions(base, tg_dev), zp]
    if shank:
        preds.append(g3.shank_predictions(gyros))
    offsets = g3.fit_offsets(preds, tg_dev)
    sets = g3.trial_sets(tg_dev)
    summ, tables = g3.evaluate(tg_dev, preds, {k: v[0] for k, v in offsets.items()}, sets=sets)
    # per walkway stride
    m = tables[NAME]["strides"]
    kt = keyed_table(by_trial, tg_dev.strides)
    if not (kt[["trial_id", "foot", "t0", "t1"]].reset_index(drop=True)
            .equals(m[["trial_id", "foot", "t0", "t1"]].reset_index(drop=True))):
        raise AssertionError("the keyed table and the G3 match table list different walkway strides")
    zt0, zt1 = nearest_ends(zp.strides, tg_dev.strides)
    st = kt.copy()
    st["g3_pred_m"] = np.where(m["matched"].to_numpy(), m["pred_m"].to_numpy(np.float64), np.nan)
    st["g3_end_diff_s"] = m["end_diff_s"].to_numpy(np.float64)
    st["g3_matched"] = m["matched"].to_numpy(bool)
    st["g3_err_cm"] = m["err_cm"].to_numpy(np.float64)
    st["zupt_t0"], st["zupt_t1"] = zt0, zt1
    zx = {(r.trial_id, r.foot, r.t0, r.t1): (r.v_end_m_s, r.t1 - r.t0) for r in zf.itertuples()}
    q = [zx.get((r.trial_id, r.foot, r.zupt_t0, r.zupt_t1), (np.nan, np.nan)) if g else (np.nan, np.nan)
         for r, g in zip(st.itertuples(), st["g3_matched"])]
    st["zupt_v_end_m_s"] = [a for a, _ in q]
    st["zupt_event_duration_s"] = [b for _, b in q]
    st["no_foot_sensor"] = [(t, f) in set(feet_missing) for t, f in zip(st["trial_id"], st["foot"])]
    st["in_dev12"] = ~st["person_id"].isin(set(EXCLUDED_FROM_12))
    near_ok = np.allclose(np.where(st["g3_matched"], m["pred_m"], 0.0),
                          np.where(st["g3_matched"], [_len_at(zf, r) for r in st.itertuples()], 0.0))
    if not near_ok:
        raise AssertionError("nearest_ends disagrees with g3.match_strides")
    # per walkway pass
    sp = tables[NAME]["speeds"]
    npass = pass_speeds(zf, tg_dev.passes)
    ps = sp[["trial_id", "person_id", "task", "pass_id", "t0", "t1", "speed_m_s"]].copy()
    ps["zupt_m_s"] = sp["pred_m_s"].to_numpy(np.float64)
    ps["zupt_matched"] = sp["matched"].to_numpy(bool)
    nmap = {(r.trial_id, str(r.pass_id)): r.n_strides for r in npass.itertuples()}
    ps["zupt_n_strides"] = [nmap.get((t, str(p)), 0) for t, p in zip(ps["trial_id"], ps["pass_id"])]
    ps["in_dev12"] = ~ps["person_id"].isin(set(EXCLUDED_FROM_12))
    res = {"what": LABEL, "dev_persons": dev_persons, "excluded_from_12": list(EXCLUDED_FROM_12),
           "records": n_rec, "feet_without_sensor": [list(x) for x in feet_missing],
           "zupt_strides_formed": int(len(zf)), "lines": _strip(summ),
           "harness_position": {s: harness_position({ln: v["sets"][s] for ln, v in summ[NAME].items()})
                                for s in sets},
           "offsets": {f"{k[0]}|{k[1]}": {"offset_s": v[0], "n": v[1]} for k, v in offsets.items()},
           "population_baseline": {t: {"stride_m": s, "speed_m_s": v} for t, (s, v) in base.items()},
           "coverage": {s: coverage(st, _persons_of(tg_dev, trials)) for s, trials in sets.items()},
           "decomposition": {s: decomposition(st, _persons_of(tg_dev, trials)) for s, trials in sets.items()}}
    persons_of_set = {s: _persons_of(tg_dev, trials) for s, trials in sets.items()}
    if body is not None:
        bst, bsp = body
        g_s, g_v = gains
        check_persons(bst["person_id"], dev_persons, locked, "body strides")
        st["k0"], st["k1"] = st["t0"].round(KEY_ROUND), st["t1"].round(KEY_ROUND)
        j = st.merge(bst[["trial_id", "foot", "k0", "k1", "x", "y"]], on=["trial_id", "foot", "k0", "k1"], how="left",
                     validate="one_to_one")
        if len(j) != len(st):
            raise AssertionError("the body join changed the number of walkway strides")
        hit = j["x"].notna().to_numpy()
        if hit.sum() != len(bst):
            raise AssertionError(f"{len(bst) - int(hit.sum())} matched body strides found no walkway stride")
        if not np.allclose(j.loc[hit, "x"].to_numpy() / 100.0, j.loc[hit, "length_m"].to_numpy(), atol=1e-6):
            raise AssertionError("the body join pairs different walkway strides")
        st["body_cm"] = j["y"].to_numpy(np.float64)
        st["body_matched"] = hit
        bmap = {(r.trial_id, str(r.pass_id)): (r.x, r.y) for r in bsp.itertuples()}
        bx = [bmap.get((t, str(p))) for t, p in zip(ps["trial_id"], ps["pass_id"])]
        if sum(b is not None for b in bx) != len(bsp):
            raise AssertionError("a matched body pass found no walkway pass")
        if not all(b is None or abs(b[0] - v) < 1e-6 for b, v in zip(bx, ps["speed_m_s"])):
            raise AssertionError("the body join pairs different walkway passes")
        ps["body_m_s"] = [np.nan if b is None else b[1] for b in bx]
        ps["body_matched"] = [b is not None for b in bx]
        res["f1_gains"] = {"g_S": g_s, "g_V": g_v, "end": F1_END}
        res["same_strides"] = {s: same_stride_lines(st, p, g_s) for s, p in persons_of_set.items()}
        res["same_passes"] = {s: same_pass_lines(ps, p, g_v) for s, p in persons_of_set.items()}
    res["timing_s"] = round(time.time() - t_start, 1)
    return res, {"strides": st, "passes": ps, "zupt_strides": zf}


def _len_at(zf, r):
    """Length of the ZUPT stride whose event times are (r.zupt_t0, r.zupt_t1) on r's trial and foot (NaN if none)."""
    if not np.isfinite(r.zupt_t0):
        return np.nan
    k = zf[(zf["trial_id"] == r.trial_id) & (zf["foot"] == r.foot) & (zf["t0"] == r.zupt_t0) & (zf["t1"] == r.zupt_t1)]
    return float(k["length_m"].iloc[0]) if len(k) else np.nan


def _persons_of(tg, trials):
    t = tg.trials
    return sorted(set(t.loc[t["trial_id"].isin(set(trials)), "person_id"]))


def _strip(summ):
    """The summary of g3.evaluate without the trial lists of its sets (kept: n_trials) and with per-person tables."""
    out = {}
    for pred, lines in summ.items():
        out[pred] = {}
        for line, st in lines.items():
            s = {k: v for k, v in st.items() if k != "sets"}
            s["sets"] = {n: {k: v for k, v in ss.items() if k != "trials"} for n, ss in (st.get("sets") or {}).items()}
            out[pred][line] = s
    return out


# ---------------------------------------------------------------- report
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, float):
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


SET_ROLE = {"dev12": "dev12 (valid inputs, the primary dev row)", "dev14": "dev14 (all trials)"}


def summary_md(res):
    """The markdown report of a results dict (as written to results.json)."""
    L = [f"# G3 ZUPT baseline on the dev controls ({LABEL})", "",
         "Written by scripts/g3_zupt_baseline.py under reports/gates/g3/zupt_dev/prediction.txt (SHA-256 "
         f"{res['prediction']['sha256']}, asserted before any target or record was read). Every number below is read "
         "from results.json of this folder. The method is the version-2 ZUPT of task SCALE (no model, nothing fitted, no "
         "walkway value in the length), its strides timed at the foot-flat onsets (the starts of its still runs) and "
         "matched by the G3 harness (mdfm/gates/g3.py: both ends within 0.15 s). Lines, intervals (2000 resamples, seed "
         "20260929) and matched shares are the harness's own. No lockbox, EVAL, CODY or care_pd record (asserted). "
         f"MoveAtlas {res['moveatlas'].get('head') or 'n/a'}.", "",
         "## G3 dev lines", "",
         "| set | predictor | stride MAE (cm) | 95 % | stride MAPE (%) | speed MAE (m/s) | 95 % | IC walkway (ms) | "
         "strides matched | passes matched | persons |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("dev12", "dev14"):
        for pred in (NAME, "population", "shank_gyro", "walkway"):
            lines = res["lines"].get(pred)
            if lines is None:
                continue
            g = {ln: lines[ln]["sets"][s] for ln in lines if s in lines[ln].get("sets", {})}
            sm, mp, sp, ic = (g.get(k, {}) for k in ("stride_mae_cm", "stride_mape_pct", "speed_mae_m_s",
                                                    "ic_median_ms_walkway"))
            npers = sm.get("n_persons", sp.get("n_persons", ic.get("n_persons", "n/a")))
            L.append(f"| {SET_ROLE[s]} | {pred} | {fmt(sm.get('point'))} | {fmt_iv(sm.get('interval'))} | "
                     f"{fmt(mp.get('point'), 1)} | {fmt(sp.get('point'), 3)} | {fmt_iv(sp.get('interval'), 3)} | "
                     f"{fmt(ic.get('point'))} | "
                     + (f"{fmt(sm.get('matched_share'), 3)} of {sm.get('n_events')}" if sm else "") + " | "
                     + (f"{fmt(sp.get('matched_share'), 3)} of {sp.get('n_events')}" if sp else "") + f" | {npers} |")
    L.append("")
    for s in ("dev12", "dev14"):
        un = res["lines"][NAME]["stride_mae_cm"]["sets"][s].get("persons_unmatched") or []
        L.append(f"- ZUPT persons without a G3-matched stride on {s}: {', '.join(un) if un else 'none'}.")
    L += ["", "G3 lines for context (docs/rules/G3_RULE.md, DRAFT): stride pass 5 cm, kill above 10; speed pass 0.08 m/s, "
          "kill above 0.15; stride MAPE (simulator-only line) pass 25, kill above 40; a floored line passes only with a "
          "matched share of at least 0.9. The population row is the rule's trivial baseline (training-fold means); the "
          "shank-gyroscope detector is the IC baseline (no stride or speed); the walkway row is the reference itself.", "",
          "Harness position of the ZUPT, read as if it were the model (descriptive, decides nothing; the rule judges no "
          "baseline by the floor):", "", "| set | line | point | matched share | floor ok | verdict |",
          "|---|---|---|---|---|---|"]
    for s in ("dev12", "dev14"):
        for ln, v in res["harness_position"][s].items():
            L.append(f"| {s} | {ln} | {fmt(v['point'], 3)} | {fmt(v['matched_share'], 3)} | {v['floor_ok']} | "
                     f"{v['verdict']} |")
    if res.get("same_strides"):
        L += ["", "## Same strides as the body (v1's phase B end, diag events; descriptive)", "",
              f"F1 gains of end {res['f1_gains']['end']}: g_S {fmt(res['f1_gains']['g_S'], 4)}, g_V "
              f"{fmt(res['f1_gains']['g_V'], 4)} (fitted on the 54 training-fold controls).", "",
              "| set | line | cm | 95 % | persons |", "|---|---|---|---|---|"]
        for s in ("dev12", "dev14"):
            ss = res["same_strides"][s]
            for k, v in ss["lines_cm"].items():
                L.append(f"| {s} | {k} | {fmt(v['point'])} | {fmt_iv(v['interval'])} | {v['n_persons']} |")
        for s in ("dev12", "dev14"):
            ss = res["same_strides"][s]
            L.append("")
            L.append(f"{s}: walkway strides {ss['walkway_strides']}; body matched {ss['body_matched']}; ZUPT matched "
                     f"(G3) {ss['zupt_matched']}; both {ss['both_g3']}; keyed ZUPT matched {ss['keyed_matched']}; body "
                     f"and keyed both {ss['both_keyed']}.")
        L += ["", "| set | line | m/s | 95 % | persons |", "|---|---|---|---|---|"]
        for s in ("dev12", "dev14"):
            sp = res["same_passes"][s]
            for k, v in sp["lines_m_s"].items():
                L.append(f"| {s} | {k} | {fmt(v['point'], 3)} | {fmt_iv(v['interval'], 3)} | {v['n_persons']} |")
        L += [""] + [f"{s}: walkway passes {res['same_passes'][s]['walkway_passes']}; body matched "
                     f"{res['same_passes'][s]['body_matched']}; ZUPT matched {res['same_passes'][s]['zupt_matched']}; "
                     f"both {res['same_passes'][s]['both']}." for s in ("dev12", "dev14")]
    L += ["", "## Coverage: the G3 matching against SCALE's walkway-keyed selection", "",
          "| set | walkway strides | G3 matched | keyed matched | both | G3 only | keyed only | neither | no foot sensor | "
          "ZUPT over walkway (median, G3 matched) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("dev12", "dev14"):
        c = res["coverage"][s]
        L.append(f"| {s} | {c['walkway_strides']} | {c['g3_matched']} ({fmt(c['g3_matched_share'], 3)}) | "
                 f"{c['keyed_matched']} ({fmt(c['keyed_matched_share'], 3)}) | {c['both']} | {c['g3_only']} | "
                 f"{c['keyed_only']} | {c['neither']} | {c['no_foot_sensor']} | "
                 f"{fmt(c['length_ratio_g3_matched_median'], 4)} |")
    L += ["", "| set | quantity | p10 | p50 | p90 | n |", "|---|---|---|---|---|---|"]
    for s in ("dev12", "dev14"):
        c = res["coverage"][s]
        for k in ("signed_t0_diff_s_g3_matched", "signed_t1_diff_s_g3_matched", "end_diff_s_keyed_only"):
            q = c[k]
            L.append(f"| {s} | {k} | {fmt(q['p10'], 3)} | {fmt(q['p50'], 3)} | {fmt(q['p90'], 3)} | {q['n']} |")
    dc = res.get("decomposition")
    if dc:
        L += ["", "## Why the G3 line differs from SCALE's keyed readout (post hoc, descriptive; version 2)", "",
              "Groups of walkway strides: both (the G3 matching and SCALE's keyed selection match it), G3 only, keyed "
              "only. Lines in cm (harness statistic); the removed end velocity is the ZUPT's own quality figure (the "
              "velocity left at the end still run, removed linearly; large when that run is not a stance).", "",
              "| set | group | strides | persons | G3-matched length | keyed length | same ZUPT stride | ZUPT over "
              "walkway (median) | errors above 10 cm | removed end velocity p50 (p90), m/s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in ("dev12", "dev14"):
            for gname in ("both", "g3_only", "keyed_only"):
                d = dc[s][gname]
                gl, kl = d.get("g3_line_cm"), d.get("keyed_line_cm")
                ve = d.get("v_end_removed_m_s")
                L.append(f"| {s} | {gname} | {d['n']} | {d['persons']} | "
                         + (f"{fmt(gl['point'])} {fmt_iv(gl['interval'])}" if gl else "") + " | "
                         + (f"{fmt(kl['point'])} {fmt_iv(kl['interval'])}" if kl else "") + " | "
                         + (fmt(d.get('same_length_share'), 3) if "same_length_share" in d else "") + " | "
                         + (fmt(d.get('median_ratio'), 4) if gl else "") + " | "
                         + (fmt(d.get('share_error_above_10_cm'), 3) if gl else "") + " | "
                         + (f"{fmt(ve['p50'], 3)} ({fmt(ve['p90'], 3)})" if ve else "") + " |")
        L += ["", "Timing of the G3-matched ZUPT strides by group (ZUPT event time minus walkway IC, s; event duration "
              "over walkway stride duration):", "", "| set | group | t0 difference p10, p50, p90 | t1 difference p10, "
              "p50, p90 | duration ratio p10, p50, p90 |", "|---|---|---|---|---|"]
        for s in ("dev12", "dev14"):
            for gname in ("both", "g3_only"):
                d = dc[s][gname]
                L.append(f"| {s} | {gname} | " + " | ".join(
                    ", ".join(fmt(d[k][q], 3) for q in ("p10", "p50", "p90"))
                    for k in ("signed_t0_diff_s", "signed_t1_diff_s", "event_over_walkway_duration")) + " |")
        L += ["", "Per person, dev12 (mean absolute error in cm over the person's strides of each kind):", "",
              "| person | G3 matched | G3 MAE | keyed matched | keyed MAE | G3 only | G3-only MAE |",
              "|---|---|---|---|---|---|---|"]
        for person, v in dc["dev12"]["per_person"].items():
            L.append(f"| {person} | {v['n_g3']} | {fmt(v['g3_mae_cm'])} | {v['n_keyed']} | {fmt(v['keyed_mae_cm'])} | "
                     f"{v['n_g3_only']} | {fmt(v['g3_only_mae_cm'])} |")
    rp = res.get("reproduction")
    if rp:
        L += ["", "## Reproduction of SCALE (asserted)", "",
              f"Per stride, SCALE's keyed selection recomputed here against reports/bench/scale/zupt_v2/real_strides.csv "
              f"(dev rows): {rp['strides_compared']} strides compared, largest length difference "
              f"{rp['max_abs_length_diff_m']:.3g} m (tolerance {rp['tol_m']:g}), unpaired rows {rp['unpaired_rows']}, "
              f"matched flags equal {rp['matched_flags_equal']}.", "",
              "| SCALE SS line (dev12) | here (cm) | SCALE (cm) | difference |", "|---|---|---|---|"]
        for k, v in rp["points_cm"].items():
            L.append(f"| {k} | {v['here']:.6f} | {v['scale']:.6f} | {v['diff']:.2e} |")
        L += ["", "| count (dev12) | here | SCALE |", "|---|---|---|"]
        for k, v in rp["counts"].items():
            L.append(f"| {k} | {v['here']} | {v['scale']} |")
        L += ["", f"Reproduction ok: {rp['ok']} (points within {rp['tol_cm']} cm, counts equal, per stride within the "
              "tolerance). SCALE's SS intervals used F1's bootstrap (10000 resamples, seed 20261001); the intervals of "
              "this folder are g3's."]
    L += ["", "## Counts and checks", "",
          f"Records read {res['records']} (dev controls only); feet without a foot sensor "
          f"{len(res['feet_without_sensor'])} ({', '.join(f'{t} {f}' for t, f in res['feet_without_sensor']) or 'none'}); "
          f"ZUPT strides formed {res['zupt_strides_formed']}.",
          f"Lockbox: dev persons in the weargait_pd lockbox list {res['lockbox']['persons_in_lockbox']}; lockbox target "
          f"rows kept {res['lockbox']['target_lockbox_rows']}; records with lockbox True {res['lockbox']['records_lockbox_true']} "
          "(each asserted 0; records read through canonical.load with allow_lockbox False).",
          f"Training fold: its walkway targets only, for the population baseline ({res['train_counts'].get('strides')} "
          f"strides, {res['train_counts'].get('passes')} passes); no training-fold record is read.",
          f"Time: {res['timing_s']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def _stamp():
    import moveatlas_stamp
    return moveatlas_stamp.stamp()


def below_normal():
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "BelowNormal"
        p.nice(10)
        return "nice 10"
    except Exception as e:  # noqa: BLE001
        return f"unchanged ({type(e).__name__})"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    prio = below_normal()
    import torch
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible: run with CUDA_VISIBLE_DEVICES=-1")
    from mdfm import paths
    pred_sha = check_prediction()                                       # before any target or record
    if sha256_file(SCALE_ZUPT2_COPY) != SCALE_ZUPT2_SHA256:
        raise AssertionError("reports/bench/scale/code/zupt2.py is not the SCALE version 2 this script ports")
    dev, train = g3.controls("dev"), g3.controls("train")
    if len(dev) != 14 or len(train) != 54 or set(dev) & set(train):
        raise SystemExit(f"folds: dev {len(dev)}, train {len(train)}, overlap {sorted(set(dev) & set(train))}")
    locked = g3.lockbox_persons()
    check_persons(dev, None, locked, "dev controls")
    check_persons(train, None, locked, "training fold")
    tg_dev = g3.load_targets(dev)
    tg_train = g3.load_targets(train)
    for name, tg in (("dev", tg_dev), ("train", tg_train)):
        if tg.counts.get("lockbox_rows", 0):
            raise AssertionError(f"{name}: {tg.counts['lockbox_rows']} lockbox target rows kept")
    base = g3.population_baseline(tg_train)
    n_lockbox_records = 0

    def records():
        nonlocal n_lockbox_records
        for tid, rec in g3.imu_records(dev, tg_dev.trials):         # canonical.load, allow_lockbox False
            if str(rec.get("lockbox")).strip().lower() not in ("false", "0"):
                n_lockbox_records += 1
            yield tid, rec

    body = read_body_events(EVENTS, dev, locked)
    gains = f1_gains()
    res, tabs = run(tg_dev, base, records(), dev, locked, body=body, gains=gains)
    scale_rows, ss = scale_reference(dev=dev, locked=locked)
    rp = reproduction(tabs["strides"][tabs["strides"]["in_dev12"]],
                      scale_rows[~scale_rows["person_id"].isin(set(EXCLUDED_FROM_12))], ss, res["same_strides"]["dev12"])
    rp14 = reproduction(tabs["strides"], scale_rows, ss, res["same_strides"]["dev12"])
    res["reproduction"] = rp
    res["reproduction"]["per_stride_dev14"] = {k: rp14[k] for k in ("per_stride_ok", "unpaired_rows",
                                                                     "matched_flags_equal", "strides_compared",
                                                                     "max_abs_length_diff_m")}
    stamp = _stamp()
    res["prediction"] = {"path": common.rel(PRED_PATH), "sha256": pred_sha, "asserted_before_reads": True,
                         "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(PRED_PATH)))}
    res["run_started"] = run_started
    res["moveatlas"] = {"head": stamp.get("head") if isinstance(stamp, dict) else None}
    res["lockbox"] = {"persons_in_lockbox": len(set(dev) & set(locked)),
                      "target_lockbox_rows": int(tg_dev.counts.get("lockbox_rows", 0))
                      + int(tg_train.counts.get("lockbox_rows", 0)),
                      "records_lockbox_true": int(n_lockbox_records), "asserted": True}
    res["train_counts"] = {"strides": int(tg_train.counts.get("strides", 0)),
                           "passes": int(tg_train.counts.get("passes", 0))}
    res["dev_counts"] = dict(tg_dev.counts)
    if res["lockbox"]["persons_in_lockbox"] or res["lockbox"]["target_lockbox_rows"] or n_lockbox_records:
        raise AssertionError(f"lockbox data reached the run: {res['lockbox']}")
    out = clean(res)
    text = summary_md(out)
    if any(c in text for c in ("\u2013", "\u2014")):
        raise ValueError("summary.md holds an en or em dash")
    common.write_text(os.path.join(OUT, "results.json"), json.dumps(out, indent=1, ensure_ascii=False,
                                                                     allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), text)
    common.write_csv(tabs["strides"], os.path.join(OUT, "strides.csv"))
    common.write_csv(tabs["passes"], os.path.join(OUT, "passes.csv"))
    common.write_csv(tabs["zupt_strides"], os.path.join(OUT, "zupt_strides.csv"))
    inputs = {common.rel(p): common.sha256_file(p) for p in (EVENTS, F1_RESULTS, SCALE_STRIDES, SCALE_SS,
                                                              *g3.TARGET_FILES, g3.SUMMARY_JSON)}
    prov = {"script": "scripts/g3_zupt_baseline.py", "label": LABEL, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "prediction_sha256": pred_sha,
            "scale_zupt2_sha256": SCALE_ZUPT2_SHA256, "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": __import__("pandas").__version__, "torch": torch.__version__,
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__, common.__file__]),
            "code_files": ["scripts/g3_zupt_baseline.py", "mdfm/gates/g3.py", "mdfm/gates/common.py"],
            "script_sha256": sha256_file(os.path.abspath(__file__)), "inputs_sha256": inputs,
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                           "PYTHONDONTWRITEBYTECODE")} | {"priority": prio}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    for f in ("results.json", "summary.md", "_provenance.json"):
        if not os.path.getmtime(os.path.join(OUT, f)) > os.path.getmtime(PRED_PATH):
            raise AssertionError(f"{f} is not newer than prediction.txt")
    print(text)
    if not rp["ok"]:
        raise SystemExit("the reproduction of SCALE failed: see the section 'Reproduction of SCALE' of summary.md")
    print(f"written {common.rel(OUT)}")


if __name__ == "__main__":
    main()
