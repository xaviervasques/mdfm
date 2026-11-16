"""F2-H4root (2026-10-05): does the decoder's root path carry the real-sensor scale error of the G3 stride and speed
(hypothesis H4), or do the decoded legs share it? Dev diagnostic, not a G3 read. Pre-registered prediction and
criteria: reports/weekend_2026-10-01/tasks/F2-H4root.md, section Prediction (written before this script read any
stride, speed or IC value).

Facts of mdfm/readouts/gait.py this test rests on: the G3 stride is the horizontal ankle displacement between two
successive decoded initial contacts (ICs) of one foot in one window frame (root translation plus the pelvis-relative
ankle, the latter at the same gait phase at both ends), and the pass speed the horizontal pelvis displacement over
the walkway pass interval: both read the root path. The model_legs row of scripts/eval_g3_diag.py reads the same
strides and passes from the stance-foot no-slip speed of the pelvis-relative joints (gait.stance_speed), the root
translation not read: the legs' own reading.

Default run (CPU; CUDA_VISIBLE_DEVICES=-1 is set before anything loads torch), per end of ENDS:
- sets: dev14, dev12 and train54 of reports/bench/g3_gain_check/results.json, their SHA-256 recomputed and asserted,
  no person of them in the weargait_pd lockbox list (mdfm.gates.g3.lockbox_persons, asserted); the persons of every
  events file are read first and asserted inside dev14 (or train54 for a dump), then rows are filtered to the set
  before any value is read;
- test A (read_test_a): on the dev12 strides matched in both rows base and model_legs of
  reports/gates/g3/diag_<name>/events.csv, g_root_real and g_legs_real (median of decoded over walkway stride); the
  in-sim gains g_root_sim and g_legs_sim = 1 + b / (100 t) of the forward row of
  reports/gates/g3_sim/insim_<name>/insim.json (rows model and model_legs); D_root = ln(g_root_real / g_root_sim),
  D_legs likewise, f = 1 - D_legs / D_root with a person bootstrap interval (BOOT_N, SEED; in-sim gains fixed);
  reading SUPPORTED (f >= 0.5 and lower bound > 0), NOT SUPPORTED (upper bound < 0.5), PARTLY otherwise;
- test B (read_test_b): per matched base stride of dev12 with both walkway ICs matched (ic_walkway rows), s decoded
  over walkway stride, d decoded over walkway stride duration, v decoded over walkway speed of its pass, k = s / (v d);
  medians against the bands of the prediction;
- with a dump folder present (--dump output, below), test A on train54 from the dump and the stance-foot slip
  reading.
Writes reports/bench/f2_scale/H4root/results.json, summary.md, per_person.csv and _provenance.json.

--dump (GPU, queued through scripts/queue_job.py; --print-job prints the command): the inference of
scripts/eval_g3_dev.py reused by import (mdfm.eval.infer.load_checkpoint and G3Predictor, the defaults of the F1
decode) on the SelfPace and HurriedPace records of the dev14 and train54 controls; per trial the base and model_legs
readouts of scripts/eval_g3_diag.py (trial_readouts), matched to the walkway by mdfm.gates.g3, written as
<out>/events.csv in the columns of eval_g3_diag.events_frame, and the per-stride decomposition of every decoded
stride (stride_decomposition: pelvis chord, ankle chord, pelvis-relative ankle change, legs integral, root, legs
and stance-foot world velocity along the progression, leg length), matched to the walkway strides
(<out>/decomposition.csv). Never a lockbox, EVAL or CODY record (check_record on every record).

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/f2_H4root.py
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__" and "--dump" not in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the CPU analysis never sees the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm import paths  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.readouts import gait  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H4root")
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
REPORT = "reports/weekend_2026-10-01/tasks/F2-H4root.md"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-H4root"

BOOT_N = 10000
SEED = 20261005
LEVEL = 0.95
F_SUPPORT = 0.5                    # test A: f at least this, lower bound above 0
K_BAND = (0.97, 1.03)              # test B: median k (stride follows the root path)
D_BAND = (0.98, 1.02)              # test B: median d (cadence right)
SV_BAND = (0.97, 1.03)             # test B: median s / v (speed error equals stride error)
SLIP_FRACTION = 0.5                # dump: |median slip| at least this x (1 - g_root / g_legs) x walkway speed
N_DEV, N_TRAIN = 14, 54
HIPS, KNEES = {"L": 1, "R": 2}, {"L": 4, "R": 5}


class End:
    def __init__(self, key, name, checkpoint, verdict):
        self.key, self.name, self.checkpoint, self.verdict = key, name, checkpoint, verdict


ENDS = (End("v1_B", "phaseA_root_full_ckpt_B_032044", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", True),
        End("s2_B", "pilot_v1_s2_ckpt_B_032044", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", True),
        End("v1_A", "phaseA_root_full_ckpt_A_010681", "cache/runs/phaseA_root_full/ckpt_A_010681.pt", False))
DUMP_END = ENDS[0]
JOB = {"name": "f2_H4root_dump_v1_B", "priority": 74, "est_hours": 0.5, "memory_gb": "12", "group": "F2",
       "batch_windows": 32}


def diag_events(name, tree=None):
    return os.path.join(tree or TREE, "reports", "gates", "g3", f"diag_{name}", "events.csv")


def insim_json(name, tree=None):
    return os.path.join(tree or TREE, "reports", "gates", "g3_sim", f"insim_{name}", "insim.json")


def dump_dir(name, out=None):
    return os.path.join(out or OUT, f"dump_{name}")


def _sibling(name):
    key = f"_f2h4_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def F1():
    return _sibling("g3_gain_check")


def rel(path):
    return F1().rel(path)


# ---------------------------------------------------------------- sets and guards
def load_sets(path=None, locked=None):
    """{dev14, dev12, train54} of F1's results.json, each SHA-256 recomputed and asserted, disjoint dev and train,
    dev12 inside dev14; Refused when a set holds a lockbox person (locked: g3.lockbox_persons() by default)."""
    from mdfm.data import sources
    with open(path or F1_RESULTS, encoding="utf-8") as f:
        s = json.load(f)["sets"]
    if locked is None:
        from mdfm.gates import g3
        locked = g3.lockbox_persons()
    sets = {k: sorted(s[k]) for k in ("dev14", "dev12")}
    from mdfm.gates import g3
    sets["train54"] = sorted(g3.controls("train"))
    for k in ("dev14", "dev12", "train54"):
        if sources.person_hash(sets[k]) != s[f"{k}_sha256"]:
            raise AssertionError(f"{k}: SHA-256 differs from {rel(path or F1_RESULTS)}")
        F1().guard_persons(sets[k], None, locked, k)
    if len(sets["dev14"]) != N_DEV or len(sets["train54"]) != N_TRAIN:
        raise AssertionError("set sizes differ from 14 and 54")
    if set(sets["dev14"]) & set(sets["train54"]) or not set(sets["dev12"]) <= set(sets["dev14"]):
        raise AssertionError("dev14, dev12 and train54 are not nested and disjoint as expected")
    return sets, sorted(locked)


def read_rows(path, allowed, locked, keep):
    """The rows of an events.csv (eval_g3_diag columns) of the persons `keep`: the person column is read and
    guarded first (every person inside `allowed`, none in `locked`), then the file, filtered to `keep`."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    F1().guard_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(path))
    F1().guard_persons(keep, allowed, locked, "kept persons")
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[ev["person_id"].isin(set(keep))].copy()
    ev["matched"] = ev["matched"].str.strip().str.lower() == "true"
    for c in ("t0_s", "t1_s", "reference", "prediction"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    return ev.reset_index(drop=True)


# ---------------------------------------------------------------- statistics
def median_ratio(y, x):
    y, x = np.asarray(y, np.float64), np.asarray(x, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return float(np.median(y[ok] / x[ok])) if ok.any() else float("nan")


def boot_persons(persons, stat, n=None, seed=None):
    """Person bootstrap: stat(index array) over resamples of the persons with replacement -> array of values."""
    p = np.asarray(persons).astype(str)
    uniq = np.unique(p)
    idx = {u: np.flatnonzero(p == u) for u in uniq}
    rng = np.random.default_rng(SEED if seed is None else seed)
    out = np.empty(BOOT_N if n is None else n)
    for b in range(len(out)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        out[b] = stat(np.concatenate([idx[u] for u in pick]))
    return out


def interval(v, level=LEVEL):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return [None, None]
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return [float(lo), float(hi)]


def f_share(g_root_real, g_legs_real, g_root_sim, g_legs_sim):
    """(D_root, D_legs, f = 1 - D_legs / D_root); f NaN when D_root is 0 or a gain is not positive."""
    vals = (g_root_real, g_legs_real, g_root_sim, g_legs_sim)
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return float("nan"), float("nan"), float("nan")
    d_root = math.log(g_root_real / g_root_sim)
    d_legs = math.log(g_legs_real / g_legs_sim)
    return d_root, d_legs, (1.0 - d_legs / d_root) if d_root != 0 else float("nan")


def reading_a(f, lo, hi):
    if f is None or not np.isfinite(f) or lo is None:
        return "NOT EVALUATED"
    if f >= F_SUPPORT and lo > 0:
        return "SUPPORTED"
    if hi < F_SUPPORT:
        return "NOT SUPPORTED"
    return "PARTLY"


ORDER = {"NOT SUPPORTED": 0, "PARTLY": 1, "SUPPORTED": 2}


def weaker(readings):
    rs = [r for r in readings if r in ORDER]
    return min(rs, key=ORDER.get) if rs else "NOT EVALUATED"


def in_band(x, band):
    return bool(np.isfinite(x) and band[0] <= x <= band[1])


# ---------------------------------------------------------------- in-sim gains
def insim_gains(path, expected_sha256=None):
    """{root, legs, true_stride_m, bias_cm}: 1 + b / (100 t) of the forward row (rows model and model_legs)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sha = (d.get("checkpoint") or {}).get("sha256")
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{rel(path)} holds checkpoint {str(sha)[:12]}, expected {expected_sha256[:12]}")
    t = float(d["counts"]["forward"]["true_stride_m"])
    b = {r: float(d["lines"]["forward"][r]["stride_bias_cm"]["point"]) for r in ("model", "model_legs")}
    return {"path": rel(path), "checkpoint_sha256": sha, "true_stride_m": t, "bias_cm": b,
            "root": 1.0 + b["model"] / (100.0 * t), "legs": 1.0 + b["model_legs"] / (100.0 * t)}


# ---------------------------------------------------------------- test A
KEY = ["trial_id", "foot", "t0_s", "t1_s"]


def paired_strides(ev):
    """Matched strides of rows base and model_legs joined on trial, foot and walkway times: person_id, task, x
    (walkway cm), root (base cm), legs (model_legs cm)."""
    st = ev[(ev["kind"] == "stride") & ev["matched"]]
    a = st[st["row"] == "base"][KEY + ["person_id", "task", "pass_id", "reference", "prediction"]]
    b = st[st["row"] == "model_legs"][KEY + ["prediction"]]
    m = a.merge(b, on=KEY, suffixes=("_root", "_legs"), validate="one_to_one")
    m = m.rename(columns={"reference": "x", "prediction_root": "root", "prediction_legs": "legs"})
    return m[np.isfinite(m[["x", "root", "legs"]].to_numpy()).all(axis=1)].reset_index(drop=True)


def test_a(pairs, sim, n_boot=None, seed=None):
    """Test A on paired strides (paired_strides frame) against the in-sim gains (insim_gains)."""
    x, r, lg = (pairs[c].to_numpy(np.float64) for c in ("x", "root", "legs"))
    g_root, g_legs = median_ratio(r, x), median_ratio(lg, x)
    d_root, d_legs, f = f_share(g_root, g_legs, sim["root"], sim["legs"])

    def stat(i):
        return f_share(median_ratio(r[i], x[i]), median_ratio(lg[i], x[i]), sim["root"], sim["legs"])[2]
    boots = boot_persons(pairs["person_id"], stat, n_boot, seed)
    lo, hi = interval(boots)
    gr_b = boot_persons(pairs["person_id"], lambda i: median_ratio(r[i], x[i]), n_boot, seed)
    gl_b = boot_persons(pairs["person_id"], lambda i: median_ratio(lg[i], x[i]), n_boot, seed)
    return {"strides": int(len(pairs)), "persons": int(pairs["person_id"].nunique()),
            "g_root_real": g_root, "g_root_real_ci": interval(gr_b),
            "g_legs_real": g_legs, "g_legs_real_ci": interval(gl_b),
            "g_root_sim": sim["root"], "g_legs_sim": sim["legs"],
            "root_over_legs_real": median_ratio(r, lg), "root_over_legs_sim": sim["root"] / sim["legs"],
            "D_root": d_root, "D_legs": d_legs, "f": f, "f_ci": [lo, hi], "reading": reading_a(f, lo, hi)}


def exploratory_mae(pairs, sim, n_boot=None, seed=None):
    """Not pre-registered (added after test A was read): the stride MAE (cm, mean over persons of each person's
    mean absolute error) of four label-free readouts on the paired strides: root raw, root / g_root_sim, legs raw,
    legs / g_legs_sim; and the in-sim-calibrated gains g_real / g_sim. Person bootstrap intervals."""
    x = pairs["x"].to_numpy(np.float64)
    reads = {"root_raw": pairs["root"].to_numpy(np.float64),
             "root_over_insim_gain": pairs["root"].to_numpy(np.float64) / sim["root"],
             "legs_raw": pairs["legs"].to_numpy(np.float64),
             "legs_over_insim_gain": pairs["legs"].to_numpy(np.float64) / sim["legs"]}
    p = pairs["person_id"].to_numpy().astype(str)

    def mae(y, i):
        e = np.abs(y[i] - x[i])
        pp = p[i]
        return float(np.mean([e[pp == u].mean() for u in np.unique(pp)]))
    out = {}
    for k, y in reads.items():
        full = np.arange(len(x))
        out[k] = {"mae_cm": mae(y, full), "mae_cm_ci": interval(boot_persons(p, lambda i, y=y: mae(y, i),
                                                                              n_boot, seed)),
                  "gain": median_ratio(y, x)}
    return out


def per_person_a(pairs):
    import pandas as pd
    rows = []
    for p, g in pairs.groupby("person_id", sort=True):
        x = g["x"].to_numpy(np.float64)
        rows.append({"person_id": p, "strides": int(len(g)), "walkway_stride_cm": float(np.mean(x)),
                     "g_root": median_ratio(g["root"], x), "g_legs": median_ratio(g["legs"], x),
                     "root_over_legs": median_ratio(g["root"], g["legs"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test B
def stride_ratios(ev, row="base"):
    """Per matched stride of `row` whose two walkway ICs are matched: s, d, v and k = s / (v d)."""
    import pandas as pd
    ic = ev[(ev["row"] == "base") & (ev["kind"] == "ic_walkway") & ev["matched"]]
    icmap = {(t, f, round(float(r), 1)): float(p) for t, f, r, p in
             zip(ic["trial_id"], ic["foot"], ic["reference"], ic["prediction"])}
    sp = ev[(ev["row"] == row) & (ev["kind"] == "speed") & ev["matched"]]
    vmap = {(t, str(p)): (float(y) / float(x) if float(x) > 0 else np.nan)
            for t, p, x, y in zip(sp["trial_id"], sp["pass_id"], sp["reference"], sp["prediction"])}
    st = ev[(ev["row"] == row) & (ev["kind"] == "stride") & ev["matched"]]
    rows = []
    for r in st.itertuples():
        p0 = icmap.get((r.trial_id, r.foot, round(1000.0 * r.t0_s, 1)))
        p1 = icmap.get((r.trial_id, r.foot, round(1000.0 * r.t1_s, 1)))
        v = vmap.get((r.trial_id, str(r.pass_id)), np.nan)
        s = r.prediction / r.reference if r.reference > 0 else np.nan
        d = ((p1 - p0) / (1000.0 * (r.t1_s - r.t0_s))) if p0 is not None and p1 is not None else np.nan
        rows.append((r.person_id, r.trial_id, r.foot, s, d, v))
    df = pd.DataFrame(rows, columns=["person_id", "trial_id", "foot", "s", "d", "v"])
    df["k"] = df["s"] / (df["v"] * df["d"])
    df["s_over_v"] = df["s"] / df["v"]
    return df


def test_b(df, n_boot=None, seed=None):
    ok = df[np.isfinite(df[["s", "d", "v", "k"]].to_numpy()).all(axis=1)].reset_index(drop=True)
    out = {"strides": int(len(ok)), "strides_without_both_ics": int(len(df) - len(ok)),
           "persons": int(ok["person_id"].nunique())}
    for c in ("s", "d", "v", "k", "s_over_v"):
        a = ok[c].to_numpy(np.float64)
        out[f"median_{c}"] = float(np.median(a)) if len(a) else float("nan")
        out[f"median_{c}_ci"] = interval(boot_persons(ok["person_id"], lambda i, a=a: float(np.median(a[i])),
                                                      n_boot, seed)) if len(a) else [None, None]
    out["stride_follows_root"] = in_band(out["median_k"], K_BAND)
    out["cadence_right"] = in_band(out["median_d"], D_BAND)
    out["speed_error_equals_stride_error"] = in_band(out["median_s_over_v"], SV_BAND)
    return out


# ---------------------------------------------------------------- per-stride decomposition (--dump)
def _window_of_stride(J, wof, f0, f1, a):
    """The window gait.strides reads a stride in (the midpoint's, else the nearest holding both ends), or None."""
    w = int(wof[(f0 + f1) // 2])
    if np.isfinite(J[w, f0, a]).all() and np.isfinite(J[w, f1, a]).all():
        return w
    both = np.flatnonzero(np.isfinite(J[:, f0, a]).all(axis=1) & np.isfinite(J[:, f1, a]).all(axis=1))
    return int(both[np.argmin(np.abs(both - w))]) if len(both) else None


def stride_decomposition(joints, contact, window_of_frame, rate=gait.RATE, thr=gait.CONTACT_THR):
    """Per decoded stride (successive decoded ICs of one foot) of a stitched body (joints (W, T, 22, 3) on the
    trial clock, contact (T, 2), window_of_frame (T,)), read in the window gait.strides reads it in:
    ankle_chord_m (the G3 stride), pelvis_chord_m, rel_ankle_along_m (change of the pelvis-relative ankle along the
    pelvis chord direction u), legs_m (mean gait.stance_speed over the stride x its duration, the model_legs
    stride), root_along_m_s (mean pelvis velocity along u), legs_along_m_s (mean no-slip velocity along u over the
    frames with a foot in contact), slip_along_m_s (mean world velocity along u of the foot points in contact: the
    no-slip residual, negative when the stance foot slides backwards, the root lagging the legs), stance_frames,
    leg_length_m (mean hip-knee plus knee-ankle length of both legs). Velocities: central differences inside the
    window's finite frames."""
    import pandas as pd
    J = np.asarray(joints, dtype=np.float64)
    c = np.asarray(contact, dtype=np.float64)
    wof = np.asarray(window_of_frame, dtype=np.int64)
    ics = gait.initial_contacts(c, rate)
    rows = []
    for foot in gait.FEET:
        t = np.sort(np.asarray(ics.get(foot, ()), dtype=np.float64))
        a = gait.ANKLES[foot]
        for t0, t1 in zip(t[:-1], t[1:]):
            f0, f1 = int(gait.frame_of(t0, rate)), int(gait.frame_of(t1, rate))
            if not 0 <= f0 < f1 < J.shape[1]:
                continue
            w = _window_of_stride(J, wof, f0, f1, a)
            if w is None:
                continue
            fin = np.flatnonzero(np.isfinite(J[w]).all(axis=(1, 2)))
            s, e = int(fin[0]), int(fin[-1]) + 1
            if not (s <= f0 and f1 < e) or e - s < 3:
                continue
            Jw = J[w, s:e]
            P = Jw[:, gait.PELVIS][:, [0, 2]]
            A = Jw[:, a][:, [0, 2]]
            i0, i1 = f0 - s, f1 - s
            chord = P[i1] - P[i0]
            n = float(np.hypot(*chord))
            u = chord / n if n > 0 else np.array([0.0, 1.0])
            relA = (A - P)
            vP = np.gradient(P, 1.0 / rate, axis=0)
            rel = Jw - Jw[:, :1]
            legs = gait.stance_speed(rel, c[s:e], rate, thr)
            seg = legs[i0:i1 + 1]
            seg = seg[np.isfinite(seg)]
            legs_m = float(seg.mean()) * (t1 - t0) if len(seg) else float("nan")
            slip, legs_v = [], []
            for j, (fa, fb) in enumerate(gait.FOOT_POINTS):
                q = 0.5 * (Jw[:, fa] + Jw[:, fb])[:, [0, 2]]
                vq = np.gradient(q, 1.0 / rate, axis=0)
                vr = np.gradient(q - P, 1.0 / rate, axis=0)
                on = np.isfinite(c[s:e, j]) & (c[s:e, j] >= thr)
                on[:i0] = False
                on[i1 + 1:] = False
                slip += list(vq[on] @ u)
                legs_v += list(-(vr[on] @ u))
            leg = []
            for side in gait.FEET:
                h, k, an = HIPS[side], KNEES[side], gait.ANKLES[side]
                leg.append(np.linalg.norm(Jw[i0:i1 + 1, h] - Jw[i0:i1 + 1, k], axis=1)
                           + np.linalg.norm(Jw[i0:i1 + 1, k] - Jw[i0:i1 + 1, an], axis=1))
            rows.append({"foot": foot, "t0": float(t0), "t1": float(t1), "window": w,
                         "ankle_chord_m": float(np.hypot(*(A[i1] - A[i0]))), "pelvis_chord_m": n,
                         "rel_ankle_along_m": float((relA[i1] - relA[i0]) @ u), "legs_m": legs_m,
                         "root_along_m_s": float((vP[i0:i1 + 1] @ u).mean()),
                         "legs_along_m_s": float(np.mean(legs_v)) if legs_v else float("nan"),
                         "slip_along_m_s": float(np.mean(slip)) if slip else float("nan"),
                         "stance_frames": int(len(slip)), "leg_length_m": float(np.mean(np.concatenate(leg)))})
    cols = ["foot", "t0", "t1", "window", "ankle_chord_m", "pelvis_chord_m", "rel_ankle_along_m", "legs_m",
            "root_along_m_s", "legs_along_m_s", "slip_along_m_s", "stance_frames", "leg_length_m"]
    return pd.DataFrame(rows, columns=cols).sort_values(["t0", "foot"], kind="stable").reset_index(drop=True)


def match_decomposition(dec, walkway_strides, passes, tol=None):
    """Walkway strides (trial_id, person_id, task, foot, pass_id, t0, t1, length_m) with the nearest decoded stride
    of dec (trial_id + stride_decomposition columns) by the larger end difference (mdfm.gates.g3.match_strides),
    matched within tol (g3.STRIDE_MATCH_S), and the walkway pass speed of the stride's pass."""
    import pandas as pd
    if tol is None:
        from mdfm.gates import g3
        tol = g3.STRIDE_MATCH_S
    by = {k: g.reset_index(drop=True) for k, g in dec.groupby(["trial_id", "foot"])} if len(dec) else {}
    vmap = {(t, str(p)): float(v) for t, p, v in zip(passes["trial_id"], passes["pass_id"], passes["speed_m_s"])}
    out = []
    for r in walkway_strides.itertuples():
        base = {"trial_id": r.trial_id, "person_id": r.person_id, "task": r.task, "foot": r.foot,
                "pass_id": r.pass_id, "walk_t0": r.t0, "walk_t1": r.t1, "walk_length_m": r.length_m,
                "walk_speed_m_s": vmap.get((r.trial_id, str(r.pass_id)), np.nan)}
        g = by.get((r.trial_id, r.foot))
        if g is None or not len(g):
            out.append({**base, "matched": False})
            continue
        d = np.maximum(np.abs(g["t0"].to_numpy() - r.t0), np.abs(g["t1"].to_numpy() - r.t1))
        k = int(np.argmin(d))
        row = {c: g[c].iloc[k] for c in g.columns if c not in ("trial_id", "foot")}
        out.append({**base, **row, "end_diff_s": float(d[k]), "matched": bool(d[k] <= tol + 1e-9)})
    return pd.DataFrame(out)


def dump(a, log=print):
    """--dump (module docstring): returns the output folder."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    import pandas as pd
    import torch

    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    diag = F1().DIAG()
    t_start = time.time()
    sets, locked = load_sets()
    persons = sorted(set(sets["dev14"]) | set(sets["train54"]))
    if a.limit_persons:
        persons = persons[:a.limit_persons]
    F1().guard_persons(persons, set(sets["dev14"]) | set(sets["train54"]), locked, "dump persons")
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    out = a.out or dump_dir(loaded.name)
    paths.assert_not_moveatlas(out)
    tg = g3.load_targets(persons)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg.counts['lockbox_rows']} lockbox target rows kept")
    pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                         seed=a.seed, pool=True)
    pid_of = dict(zip(tg.trials["trial_id"], tg.trials["person_id"]))
    parts, legs_parts, decs, done = [], [], [], []
    for tid, rec in g3.imu_records(persons, tg.trials):
        if a.limit_trials and len(done) >= a.limit_trials:
            break
        F1().check_record(tid, rec, persons)
        tb = pred(tid, g3.model_input(rec))
        ro = diag.trial_readouts(tb, tg, pid_of[tid])
        parts.append(ro["model"])
        legs_parts.append(ro["model_legs"])
        d = stride_decomposition(tb.joints, tb.contact, tb.window_of_frame, tb.rate)
        d.insert(0, "trial_id", tid)
        decs.append(d)
        r = pred.log[-1]
        log(f"  {tid}: {r['windows']} windows, {float(r.get('seconds') or 0.0):.1f} s, {len(d)} strides")
        del tb, rec
        done.append(tid)
    if not parts:
        raise ValueError("no trial was decoded")
    from dataclasses import replace as dc_replace
    keep = lambda df: df[df["trial_id"].isin(set(done))].reset_index(drop=True)  # noqa: E731
    tgs = dc_replace(tg, trials=keep(tg.trials), strides=keep(tg.strides), passes=keep(tg.passes),
                     walkway_ic=keep(tg.walkway_ic))
    evs = []
    for name, ps in (("base", parts), ("model_legs", [(None, p[1], p[2]) for p in legs_parts])):
        P = diag.predictions(name, ps)
        evs.append(diag.events_frame(name, {"strides": g3.match_strides(P, tgs), "speeds": g3.match_speeds(P, tgs)},
                                     tgs))
    ev = pd.concat(evs, ignore_index=True)
    common.write_csv(ev, os.path.join(out, "events.csv"))
    dec = pd.concat(decs, ignore_index=True)
    common.write_csv(match_decomposition(dec, tgs.strides, tgs.passes), os.path.join(out, "decomposition.csv"))
    common.write_csv(pd.DataFrame(pred.log), os.path.join(out, "trials.csv"))
    mem = {}
    if device.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9, "cap_gb": cap}
    doc = {"what": LABEL, "task": TASK, "mode": "dump", "checkpoint": loaded.describe(),
           "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "batch_windows": a.batch_windows,
                         "seed": a.seed, "weights": a.weights, "device": str(device), "memory": mem},
           "persons": persons, "trials": len(pred.log), "limit_trials": a.limit_trials,
           "timing_s": round(time.time() - t_start, 1)}
    common.write_text(os.path.join(out, "dump.json"),
                      json.dumps(F1().clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    prov = {"script": "scripts/f2_H4root.py", "mode": "dump", "label": LABEL, "checkpoint": rel(loaded.path),
            "checkpoint_sha256": loaded.sha256, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "torch": torch.__version__, "moveatlas_stamp": F1().stamp_moveatlas(),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__, gait.__file__, I.__file__,
                                               diag.__file__])}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(F1().clean(prov), indent=1) + "\n")
    log(f"dumped {len(pred.log)} trials to {rel(out)}")
    return out


# ---------------------------------------------------------------- dump reading
def read_dump(folder, sets, locked, sim, n_boot=None, seed=None, speed_sim=None):
    """Test A on train54 from a dump (its events.csv, rows base and model_legs) and the slip reading on its
    decomposition.csv (train54 and dev12); None when the folder holds no decomposition.csv. With speed_sim
    (insim_speed_gains), beside: test A on the pass speeds (test_a_speed, not pre-registered)."""
    import pandas as pd
    dpath = os.path.join(folder, "decomposition.csv")
    if not os.path.exists(dpath):
        return None
    allowed = set(sets["dev14"]) | set(sets["train54"])
    out = {"path": rel(folder)}
    ev = read_rows(os.path.join(folder, "events.csv"), allowed, locked, sets["train54"])
    pairs = paired_strides(ev)
    out["test_a_train54"] = test_a(pairs, sim, n_boot, seed)
    who = pd.read_csv(dpath, encoding="utf-8", usecols=["person_id"], dtype=str)
    F1().guard_persons(set(who["person_id"].dropna()), allowed, locked, "decomposition.csv")
    dec = pd.read_csv(dpath, encoding="utf-8", dtype={"person_id": str, "trial_id": str, "pass_id": str})
    dec = dec[dec["matched"].astype(str).str.lower() == "true"]
    for name in ("train54", "dev12"):
        d = dec[dec["person_id"].isin(set(sets[name]))]
        if not len(d):
            continue
        r = {"strides": int(len(d)), "persons": int(d["person_id"].nunique())}
        for c in ("slip_along_m_s", "root_along_m_s", "legs_along_m_s", "rel_ankle_along_m", "leg_length_m"):
            r[f"median_{c}"] = float(np.nanmedian(d[c]))
        r["median_ankle_over_pelvis_chord"] = median_ratio(d["ankle_chord_m"], d["pelvis_chord_m"])
        r["median_root_over_legs_m"] = median_ratio(d["pelvis_chord_m"], d["legs_m"])
        r["median_walk_speed_m_s"] = float(np.nanmedian(d["walk_speed_m_s"]))
        out[f"decomposition_{name}"] = r
    t = out.get("decomposition_train54")
    ta = out["test_a_train54"]
    if t:
        need = SLIP_FRACTION * (1.0 - ta["g_root_real"] / ta["g_legs_real"]) * t["median_walk_speed_m_s"]
        slip = t["median_slip_along_m_s"]
        out["slip_reading"] = {"required_backward_m_s": need, "median_slip_along_m_s": slip,
                               "supports_H4": bool(need > 0 and slip <= -need)}
    if speed_sim is not None:
        out["test_a_train54_speed_divisor"] = test_a_speed(ev, speed_sim, n_boot, seed)
    # descriptive (task F2D): the real gain the root would read if it lost on real input only what the legs lose
    # (g_root_sim x exp(D_legs)), on each divisor: the ceiling of a lever that removes the root-specific part alone
    ceil = {"stride": ta["g_root_sim"] * math.exp(ta["D_legs"]) if np.isfinite(ta["D_legs"]) else float("nan"),
            "stride_now": ta["g_root_real"]}
    q = out.get("test_a_train54_speed_divisor")
    if q:
        ceil["speed"] = q["g_root_speed_sim"] * math.exp(q["D_legs"]) if np.isfinite(q["D_legs"]) else float("nan")
        ceil["speed_now"] = q["g_root_speed_real"]
    out["root_specific_ceiling_train54"] = ceil
    return out


def insim_speed_gains(name, tree=None):
    """{root, legs}: the in-sim pass-speed gains of the forward row used by F2-H3angles and F2-CHAL (A): root, F1's
    median over the forward takes of model over true speed (g3_gain_check.insim_gain); legs, 1 + bv / tv of the
    model_legs row of insim.json (bv its speed bias, tv the mean scored true speed)."""
    folder = os.path.dirname(insim_json(name, tree))
    root = F1().insim_gain(folder)["speed_gain"]
    with open(insim_json(name, tree), encoding="utf-8") as f:
        d = json.load(f)
    bv = float(d["lines"]["forward"]["model_legs"]["speed_bias_m_s"]["point"])
    tv = float(d["counts"]["forward"]["true_speed_m_s"])
    return {"root": float(root), "legs": 1.0 + bv / tv, "path": rel(folder)}


def test_a_speed(ev, sim, n_boot=None, seed=None):
    """Not pre-registered (the second divisor of F2-CHAL, A, named before the dump was read): test A on the matched
    pass speeds, each row pooled over its own matched passes (as F2-H3angles' speed_gains), divided by the in-sim
    pass-speed gains: D = ln(real / sim), f = 1 - D_legs / D_root, person bootstrap."""
    sp = ev[(ev["kind"] == "speed") & ev["matched"]]
    rows = {r: sp[sp["row"] == r][["person_id", "reference", "prediction"]].reset_index(drop=True)
            for r in ("base", "model_legs")}
    persons = sorted(set(rows["base"]["person_id"]) | set(rows["model_legs"]["person_id"]))
    ratios = {}
    for r, d in rows.items():
        x, y = d["reference"].to_numpy(np.float64), d["prediction"].to_numpy(np.float64)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        p = d["person_id"].astype(str).to_numpy()
        ratios[r] = {q: (y[ok & (p == q)] / x[ok & (p == q)]) for q in persons}

    def gains(pick):
        g = {}
        for r in rows:
            a = np.concatenate([ratios[r][q] for q in pick]) if pick else np.array([])
            g[r] = float(np.median(a)) if len(a) else float("nan")
        return g
    g = gains(persons)
    d_root, d_legs, f = f_share(g["base"], g["model_legs"], sim["root"], sim["legs"])
    rng = np.random.default_rng(SEED if seed is None else seed)
    boots = []
    for _ in range(BOOT_N if n_boot is None else n_boot):
        pick = list(rng.choice(persons, size=len(persons), replace=True))
        gb = gains(pick)
        boots.append(f_share(gb["base"], gb["model_legs"], sim["root"], sim["legs"])[2])
    lo, hi = interval(boots)
    return {"what": "not pre-registered: the pass-speed divisor of F2-CHAL A, read on train54",
            "passes_root": int(len(rows["base"])), "passes_legs": int(len(rows["model_legs"])),
            "persons": len(persons), "g_root_speed_real": g["base"], "g_legs_speed_real": g["model_legs"],
            "g_root_speed_sim": sim["root"], "g_legs_speed_sim": sim["legs"], "sim_path": sim.get("path"),
            "D_root": d_root, "D_legs": d_legs, "f": f, "f_ci": [lo, hi], "reading_if_registered": reading_a(f, lo, hi)}


def scaled_lines(pairs, dump, n_boot=None, seed=None):
    """Descriptive (task F2D): dev12 S (cm, mean over persons of each person's mean absolute stride error) with the
    decoded root stride multiplied by a uniform factor taken from the train54 dump, no dev value used to set it:
    'root_specific_removed' = ceiling / now (the root losing on real input only what the legs lose), 'sim_gain' =
    g_root_sim / g_root_real (the whole sim-to-real loss of the root removed); 'raw' = 1. An expected effect of a
    lever under a pure-gain assumption, not a measurement of the lever."""
    c = dump["root_specific_ceiling_train54"]
    ta = dump["test_a_train54"]
    factors = {"raw": 1.0, "root_specific_removed": c["stride"] / c["stride_now"],
               "sim_gain": ta["g_root_sim"] / ta["g_root_real"]}
    x = pairs["x"].to_numpy(np.float64)
    p = pairs["person_id"].to_numpy().astype(str)

    def mae(y, i):
        e = np.abs(y[i] - x[i])
        pp = p[i]
        return float(np.mean([e[pp == u].mean() for u in np.unique(pp)]))
    out = {}
    for k, fac in factors.items():
        y = pairs["root"].to_numpy(np.float64) * fac
        out[k] = {"factor": fac, "S_cm": mae(y, np.arange(len(x))),
                  "S_cm_ci": interval(boot_persons(p, lambda i, y=y: mae(y, i), n_boot, seed)),
                  "gain": median_ratio(y, x)}
    return out


def assert_same_constants(res, previous=None):
    """AssertionError unless the constants of res equal those stored by the first run
    (reports/bench/f2_scale/H4root/results.json, written 08:47:50 before the dump existed)."""
    with open(previous or os.path.join(OUT, "results.json"), encoding="utf-8") as f:
        old = json.load(f)
    now = json.loads(json.dumps(F1().clean(res["constants"])))
    if now != old["constants"] or res["prediction"] != old["prediction"]:
        raise AssertionError(f"constants differ from the first run: {old['constants']} against {now}")
    return True


def prediction_section_sha256(report=None):
    """SHA-256 of the section '## Prediction' of the report (heading line to the next '## ' heading, LF), recorded
    with a rerun (the script held no hash before)."""
    import hashlib
    with open(report or os.path.join(TREE, REPORT), encoding="utf-8") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    a = next(i for i, x in enumerate(lines) if x.strip() == "## Prediction")
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return hashlib.sha256(("\n".join(lines[a:b]).rstrip("\n") + "\n").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the analysis
def analyse(out=None, n_boot=None, seed=None, log=print, results_out=None):
    """The analysis (module docstring). out: the folder the dumps are read from (default OUT) and, without
    results_out, the folder written; results_out: a new folder for results.json, summary.md, per_person.csv and
    _provenance.json (a rerun: the constants are asserted equal to those of the first run's results.json, and the
    Prediction section of the report is hashed into the results)."""
    import pandas as pd
    t_start = time.time()
    sets, locked = load_sets()
    res = {"what": LABEL, "task": TASK, "prediction": REPORT, "ends": {}, "constants": {
        "boot_n": BOOT_N if n_boot is None else n_boot, "seed": SEED if seed is None else seed, "level": LEVEL,
        "f_support": F_SUPPORT, "k_band": K_BAND, "d_band": D_BAND, "sv_band": SV_BAND, "slip_fraction": SLIP_FRACTION}}
    pp = []
    for e in ENDS:
        sim = insim_gains(insim_json(e.name))
        ev = read_rows(diag_events(e.name), set(sets["dev14"]), locked, sets["dev14"])
        r = {"checkpoint": e.checkpoint, "events": rel(diag_events(e.name)), "insim": sim, "verdict_end": e.verdict}
        for s in ("dev12", "dev14"):
            evs = ev[ev["person_id"].isin(set(sets[s]))]
            pairs = paired_strides(evs)
            r[f"test_a_{s}"] = test_a(pairs, sim, n_boot, seed)
            r[f"exploratory_{s}"] = exploratory_mae(pairs, sim, n_boot, seed)
            r[f"test_b_{s}"] = test_b(stride_ratios(evs, "base"), n_boot, seed)
            r[f"test_b_legs_{s}"] = test_b(stride_ratios(evs, "model_legs"), n_boot, seed)
            if s == "dev12":
                t = per_person_a(pairs)
                t.insert(0, "end", e.key)
                pp.append(t)
        dfolder = dump_dir(e.name, out)
        r["dump"] = (read_dump(dfolder, sets, locked, sim, n_boot, seed, speed_sim=insim_speed_gains(e.name))
                     if os.path.isdir(dfolder) else None)
        if r["dump"]:
            r["dump"]["dev12_S_if_scaled"] = scaled_lines(
                paired_strides(ev[ev["person_id"].isin(set(sets["dev12"]))]), r["dump"], n_boot, seed)
        res["ends"][e.key] = r
        log(f"{e.key}: test A dev12 f {r['test_a_dev12']['f']:.3f} {r['test_a_dev12']['reading']}")
    verdict_ends = [e.key for e in ENDS if e.verdict]
    res["reading_test_a"] = weaker(res["ends"][k]["test_a_dev12"]["reading"] for k in verdict_ends)
    res["sets"] = {k: sets[k] for k in ("dev14", "dev12")} | {"n_train54": len(sets["train54"])}
    from mdfm.data import sources
    res["sets"].update({f"{k}_sha256": sources.person_hash(sets[k]) for k in ("dev14", "dev12", "train54")})
    res["lockbox"] = {"persons_in_sets": len((set(sets["dev14"]) | set(sets["train54"])) & set(locked)),
                      "asserted": True, "list_size": len(locked)}
    res["timing_s"] = round(time.time() - t_start, 1)
    o = results_out or out or OUT
    paths.assert_not_moveatlas(o)
    if results_out:
        assert_same_constants(res)
        res["rerun"] = {"first_run": rel(os.path.join(OUT, "results.json")), "constants_equal_to_first_run": True,
                        "prediction_section_sha256": prediction_section_sha256(), "results_folder": rel(results_out)}
    from mdfm.gates import common
    common.write_text(os.path.join(o, "results.json"),
                      json.dumps(F1().clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_csv(pd.concat(pp, ignore_index=True), os.path.join(o, "per_person.csv"))
    common.write_text(os.path.join(o, "summary.md"), summary_md(F1().clean(res)))
    prov = {"script": "scripts/f2_H4root.py", "mode": "analyse", "label": LABEL,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "moveatlas_stamp": F1().stamp_moveatlas(),
            "inputs": [rel(diag_events(e.name)) for e in ENDS] + [rel(insim_json(e.name)) for e in ENDS]
            + [rel(F1_RESULTS)],
            "code_sha256": common.code_sha256([os.path.abspath(__file__), gait.__file__])}
    common.write_text(os.path.join(o, "_provenance.json"), json.dumps(F1().clean(prov), indent=1) + "\n")
    return res


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _iv(v, nd=3):
    return "" if not v or v[0] is None else f"[{_f(v[0], nd)}, {_f(v[1], nd)}]"


def summary_md(r):
    L = ["# F2-H4root: root path against legs (dev diagnostic, not a G3 read)", "",
         "Written by scripts/f2_H4root.py; every number below is read from results.json of this folder. Prediction and "
         f"criteria: {REPORT}. No lockbox person, EVAL or CODY record was read (asserted: "
         f"{r['lockbox']['persons_in_sets']} set persons in the lockbox list of {r['lockbox']['list_size']}).", "",
         f"**Test A reading (weaker of v1_B and s2_B on dev12): {r['reading_test_a']}**", "",
         "## Test A: sim-to-real loss of the root reading and of the legs reading", "",
         "| end | set | strides / persons | g_root real | 95 % | g_legs real | 95 % | g_root sim | g_legs sim | "
         "root/legs real | root/legs sim | D_root | D_legs | f | 95 % | reading |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        rows = [("dev12", e["test_a_dev12"]), ("dev14", e["test_a_dev14"])]
        if e.get("dump"):
            rows.append(("train54 (dump)", e["dump"]["test_a_train54"]))
        for s, a in rows:
            L.append(f"| {k} | {s} | {a['strides']} / {a['persons']} | {_f(a['g_root_real'])} | {_iv(a['g_root_real_ci'])} | "
                     f"{_f(a['g_legs_real'])} | {_iv(a['g_legs_real_ci'])} | {_f(a['g_root_sim'])} | {_f(a['g_legs_sim'])} | "
                     f"{_f(a['root_over_legs_real'])} | {_f(a['root_over_legs_sim'])} | {_f(a['D_root'])} | "
                     f"{_f(a['D_legs'])} | {_f(a['f'])} | {_iv(a['f_ci'])} | {a['reading']} |")
    L += ["", "g = median of decoded over walkway stride (real: strides matched in both rows); sim: 1 + b / (100 t) of the "
          "forward row of insim.json; D = ln(real / sim); f = 1 - D_legs / D_root; intervals: person bootstrap, sim "
          "fixed. v1_A is reported beside and enters no reading.", "",
          "## Test B: does the stride follow the root path, is cadence right", "",
          "| end | row | set | strides / persons | median s | median d | 95 % | median v | median k | 95 % | "
          "median s/v | 95 % | k in band | d in band | s/v in band |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        for row, pre in (("base", "test_b_"), ("model_legs", "test_b_legs_")):
            for s in ("dev12", "dev14"):
                b = e[f"{pre}{s}"]
                L.append(f"| {k} | {row} | {s} | {b['strides']} / {b['persons']} | {_f(b['median_s'])} | "
                         f"{_f(b['median_d'])} | {_iv(b['median_d_ci'])} | {_f(b['median_v'])} | {_f(b['median_k'])} | "
                         f"{_iv(b['median_k_ci'])} | {_f(b['median_s_over_v'])} | {_iv(b['median_s_over_v_ci'])} | "
                         f"{b['stride_follows_root']} | {b['cadence_right']} | {b['speed_error_equals_stride_error']} |")
    L += ["", f"s decoded / walkway stride; d decoded / walkway stride duration (both walkway ICs matched); v decoded / "
          f"walkway speed of the stride's pass; k = s / (v d). Bands: k {list(K_BAND)}, d {list(D_BAND)}, s/v "
          f"{list(SV_BAND)}.", "",
          "## Exploratory, not pre-registered: label-free readouts on the paired strides", "",
          "Added after test A was read. Each readout divided by its own in-sim gain (no walkway value used); stride MAE "
          "is the mean over persons of each person's mean absolute error (cm), person bootstrap.", "",
          "| end | set | readout | gain | MAE (cm) | 95 % |", "|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        for s in ("dev12", "dev14"):
            for name, x in e[f"exploratory_{s}"].items():
                L.append(f"| {k} | {s} | {name} | {_f(x['gain'])} | {_f(x['mae_cm'], 2)} | {_iv(x['mae_cm_ci'], 2)} |")
    L += ["", "## Dump (GPU, per-stride decomposition)", ""]
    any_dump = False
    for k, e in r["ends"].items():
        d = e.get("dump")
        if not d:
            continue
        any_dump = True
        for s in ("train54", "dev12"):
            x = d.get(f"decomposition_{s}")
            if x:
                L.append(f"- {k} {s}: {x['strides']} strides, {x['persons']} persons; median slip along progression "
                         f"{_f(x['median_slip_along_m_s'])} m/s, root {_f(x['median_root_along_m_s'])}, legs "
                         f"{_f(x['median_legs_along_m_s'])}, walkway {_f(x['median_walk_speed_m_s'])} m/s; ankle over "
                         f"pelvis chord {_f(x['median_ankle_over_pelvis_chord'])}; pelvis chord over legs "
                         f"{_f(x['median_root_over_legs_m'])}; leg length {_f(x['median_leg_length_m'])} m.")
        if d.get("slip_reading"):
            sr = d["slip_reading"]
            L.append(f"- {k} slip reading: required backward slip {_f(sr['required_backward_m_s'])} m/s, median "
                     f"{_f(sr['median_slip_along_m_s'])} m/s, supports H4 {sr['supports_H4']}.")
        if d.get("test_a_train54_speed_divisor"):
            q = d["test_a_train54_speed_divisor"]
            L.append(f"- {k} train54, not pre-registered (the pass-speed divisor of F2-CHAL A): g_root speed real "
                     f"{_f(q['g_root_speed_real'])} ({q['passes_root']} passes), g_legs speed real "
                     f"{_f(q['g_legs_speed_real'])} ({q['passes_legs']}), in-sim {_f(q['g_root_speed_sim'])} and "
                     f"{_f(q['g_legs_speed_sim'])}; D_root {_f(q['D_root'])}, D_legs {_f(q['D_legs'])}, f "
                     f"{_f(q['f'])} {_iv(q['f_ci'])} (the test A rule would read {q['reading_if_registered']}).")
        c = d.get("root_specific_ceiling_train54")
        if c:
            L.append(f"- {k} train54, descriptive: the root gain if it lost on real input only what the legs lose "
                     f"(g_root sim x exp(D_legs)): stride {_f(c['stride'])} (now {_f(c['stride_now'])})"
                     + (f", pass speed {_f(c['speed'])} (now {_f(c['speed_now'])})" if "speed" in c else "") + ".")
        s = d.get("dev12_S_if_scaled")
        if s:
            L.append(f"- {k} dev12, descriptive (a uniform factor from train54 on the decoded root stride, a pure-gain "
                     "assumption): " + "; ".join(
                         f"{name} factor {_f(v['factor'])}: S {_f(v['S_cm'], 2)} cm {_iv(v['S_cm_ci'], 2)}, gain "
                         f"{_f(v['gain'])}" for name, v in s.items()) + ".")
    if not any_dump:
        L.append("No dump folder yet (the --dump job is queued; rerun this script after it).")
    L += ["", f"Sets: dev14 {r['sets']['dev14_sha256']}, dev12 {r['sets']['dev12_sha256']}, train54 "
          f"{r['sets']['train54_sha256']} (SHA-256, equal to reports/bench/g3_gain_check/results.json, asserted).",
          f"Time: {r['timing_s']} s.", ""]
    if r.get("rerun"):
        L += [f"Rerun of {r['rerun']['first_run']}: constants equal to the first run (asserted); Prediction section "
              f"SHA-256 at the rerun {r['rerun']['prediction_section_sha256']}.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- CLI
def job_command():
    out = f"reports/bench/f2_scale/H4root/dump_{DUMP_END.name}"
    return ["scripts/queue_job.py", "--name", JOB["name"], "--cls", "1", "--priority", str(JOB["priority"]),
            "--est-hours", str(JOB["est_hours"]), "--group", JOB["group"], "--requires", DUMP_END.checkpoint,
            "--outputs", f"{out}/decomposition.csv",
            "--note", "F2-H4root: per-stride root, ankle, legs and stance-slip dump of dev14 and train54 by v1_B",
            "--", "scripts/f2_H4root.py", "--dump", "--checkpoint", DUMP_END.checkpoint, "--batch-windows",
            str(JOB["batch_windows"]), "--memory-gb", JOB["memory_gb"], "--out", out]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", action="store_true", help="decode dev14 and train54 and dump the decomposition (GPU)")
    p.add_argument("--print-job", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--limit-persons", type=int, default=0, help="--dump smoke test: the first n persons")
    p.add_argument("--limit-trials", type=int, default=0, help="--dump smoke test: the first n trials")
    p.add_argument("--boot", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--results-out", default=None,
                   help="analysis: a new folder for results.json, summary.md, per_person.csv and _provenance.json "
                        "(dumps still read under --out or reports/bench/f2_scale/H4root/); asserts the constants of "
                        "the first run")
    a = p.parse_args(argv)
    if a.dump and not a.checkpoint:
        p.error("--dump needs --checkpoint")
    return a


def main(argv=None):
    a = parse(argv)
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.print_job:
        print(" ".join(job_command()))
        return 0
    if a.dump:
        dump(a)
        return 0
    try:
        analyse(a.out, n_boot=a.boot, results_out=os.path.abspath(a.results_out) if a.results_out else None)
    except Refused as e:
        print(f"refused: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
