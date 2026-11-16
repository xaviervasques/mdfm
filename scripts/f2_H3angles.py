"""F2-H3angles (2026-10-05): is the short decoded stride of the real weargait data (F1: gain 0.847) a leg excursion
error (H3: too little hip and knee sagittal excursion, cadence right)? Every output is labelled 'dev diagnostic, not a
G3 read'. The prediction and its criteria were written before any value was read: reports/weekend_2026-10-01/tasks/
F2-H3angles.md, section Prediction (2026-10-05T08:40:27).

Default run (CPU, CUDA_VISIBLE_DEVICES=-1), Part A on existing files, at the ends of ENDS (the F1 ends: v1_B, s2_B,
v1_A beside):
- persons: dev14, dev12 and train54 of mdfm.gates.g3 (the F1 sets; their SHA-256 asserted equal to
  reports/bench/g3_gain_check/results.json), checked against the weargait_pd lockbox list before any file is read
  (g3_gain_check.check_sets); every events file is guarded on its persons (the dev set) before its values are read;
- cadence (C1): per walkway stride whose two end ICs have a matched decoded IC (ic_walkway rows of row base of
  reports/gates/g3/diag_<name>/events.csv; the walkway stride ends are the t0_s and t1_s of the stride rows and are
  found among the walkway IC times), r_T = decoded duration over walkway duration (the run's IC offset cancels);
  r_T of a set is the median over persons of the person medians; g_S on the same strides (pooled median of decoded
  over walkway, base row, matched strides); s_T = log(r_T) / log(g_S);
- legs (C2): the model_legs stride gain of the same file (pooled median ratio over its matched strides) over the
  in-sim model_legs gain of the same checkpoint (1 + b / (100 t), the forward row of reports/gates/g3_sim/insim_<name>/
  insim.json, as F1 computes the in-sim model gain): D_L; the base stride gain over F1's in-sim stride gain: D_R;
- intervals: person bootstrap, BOOT_N resamples, seed SEED, the same draws for every statistic of an end and set;
- the readings C1, C2 and Part A of the prediction (reading_c1, reading_c2, reading_a).
Part B, when the dump of B_END exists (reports/bench/f2_scale/H3angles/dump_<name>/): hip sagittal and knee flexion
ranges per walkway stride (real) and per scored true stride (in-sim, truth and decoded), the expected range at the
walkway speed from the least squares line of the true in-sim range on the true stride speed, rho_H and rho_K, and the
reading of the prediction (reading_b). Written: results.json, summary.md, _provenance.json under OUT.

--dump (GPU, queued as one job; --print-job prints the scripts/queue_job.py command): the inference of
scripts/eval_g3_dev.py (mdfm.eval.infer.G3Predictor, by import) on the SelfPace and HurriedPace records of train54 and
dev14, and the in-sim renders of scripts/eval_gait_insim.py (AMASS dev walking takes, xsens13, by import) with their
true bodies; per stride: stride_features of the decoded body (and of the truth in-sim), the legs stride (stance-foot
no-slip speed integrated over the stride) and the walkway or true stride. Writes <out>/strides_real.csv,
strides_insim.csv, dump.json and _provenance.json. Never a lockbox, EVAL or CODY record.

Angles (stride_features), from the joint positions alone, over the frames of the stride and on the side of its foot:
pelvis lateral axis l = left hip minus right hip, up axis u = spine1 minus pelvis made orthogonal to l, forward
f = l x u (SMPL faces +z with +x left and +y up); hip angle atan2(thigh . f, -thigh . u) (flexion positive, thigh =
knee minus hip); knee flexion = the angle between thigh and shank (shank = ankle minus knee), 0 when straight; thigh
inclination atan2(thigh . f_h, -thigh_y) with f_h the horizontal forward; ranges are max minus min over the stride.
A constant offset of a definition (the rest tilt of spine1) leaves every range unchanged.

Run (mdfm environment):
  CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/f2_H3angles.py
  <conda>/envs/mdfm/python.exe scripts/f2_H3angles.py --print-job
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

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H3angles")
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
REPORT = "reports/weekend_2026-10-01/tasks/F2-H3angles.md"
PREDICTION_AT = "2026-10-05T08:40:27"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-H3angles"
BOOT_N = 10000
SEED = 20261001
LEVEL = 0.95
RATE = 100.0
T_KEY_DECIMALS = 4                  # s, the rounding that pairs a walkway stride end with its walkway IC row
# criteria of the prediction (section Prediction of the report), never changed after a read
C1_BAND = (0.97, 1.03)
C1_SPATIAL_SHARE = 0.2
C1_TEMPORAL_SHARE = 0.5
C2_LEGS_MAX = 0.92
C2_AGREE = 0.04
C2_ROOT_LEGS_MIN = 0.96
C2_ROOT_MAX = 0.92
B_SUPPORT_MAX = 0.92
B_NOT_MIN = 0.96
B_UPPER_BELOW = 1.0
B_END = "v1_B"
EPS = 1e-9                          # floating tolerance of the comparisons with the criteria
# context only (no criterion): approximate normal sagittal arcs at free walking speed
PUBLISHED = {"source": "Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function, 2nd ed. SLACK, 2010 "
                       "(chapters on the hip and the knee; free walking speed about 82 m/min)",
             "speed_m_s": 1.37, "hip_arc_deg": 40.0, "knee_arc_deg": 60.0,
             "note": "hip arc about 30 deg flexion to 10 deg extension, knee about 0 to 60 deg; clinical definitions, "
                     "not the joint-position angles of this script"}
JOINT = {"pelvis": 0, "hip_L": 1, "hip_R": 2, "spine1": 3, "knee_L": 4, "knee_R": 5, "ankle_L": 7, "ankle_R": 8}
FEATURES = ("hip_range_deg", "hip_max_deg", "hip_min_deg", "knee_range_deg", "knee_max_deg", "thigh_range_deg",
            "leg_length_m", "ankle_fa_range_m", "ankle_m", "root_m")
JOB = {"name": "f2_h3angles_dump_v1_B", "priority": 74, "est_hours": 0.6, "memory_gb": "12", "batch_windows": 32}


def _sibling(name):
    """scripts/<name>.py of the tree, loaded once under a private name."""
    key = f"_f2h3_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def GG():
    return _sibling("g3_gain_check")


def ends():
    return GG().ENDS


def rel(path):
    return GG().rel(path)


def dump_dir(name, out=None):
    return os.path.join(out or OUT, f"dump_{name}")


# ---------------------------------------------------------------- sets and guards
def get_sets():
    """(sets {dev14, dev12, train54}, locked): the F1 sets, checked (sizes, disjoint, no lockbox person) and their
    SHA-256 asserted equal to F1's results.json."""
    from mdfm.data import sources
    from mdfm.gates import g3
    gg = GG()
    train, dev = sorted(g3.controls("train")), sorted(g3.controls("dev"))
    locked = g3.lockbox_persons()
    sets = {"dev14": dev, "dev12": [p for p in dev if p not in gg.EXCLUDED_FROM_12], "train54": train}
    gg.check_sets(sets, locked)
    if os.path.exists(F1_RESULTS):
        with open(F1_RESULTS, encoding="utf-8") as f:
            f1 = json.load(f)["sets"]
        for k in ("dev14", "train54"):
            if sources.person_hash(sets[k]) != f1[f"{k}_sha256"]:
                raise AssertionError(f"{k} differs from the F1 set")
    return sets, locked


def read_events(path, allowed, locked):
    """The events.csv of scripts/eval_g3_diag.py as a frame (text columns as str), read after its persons were
    guarded: Refused for a lockbox person, AssertionError for a person outside `allowed`."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    GG().guard_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(path))
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    for c in ("t0_s", "t1_s", "reference", "prediction"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev["is_matched"] = ev["matched"].str.strip().str.lower() == "true"
    return ev


# ---------------------------------------------------------------- Part A: cadence and legs
def stride_durations(ev, row="base"):
    """Per walkway stride of `row`: person_id, trial_id, foot, t0, t1, T_wk, T_dec (difference of the matched decoded
    ICs paired with the stride's two walkway ICs, NaN when one is missing or unmatched or they are not increasing),
    r_T = T_dec / T_wk, x (walkway cm), y (decoded cm), base_matched. attrs: ends_found (share of stride ends found
    among the walkway IC rows)."""
    import pandas as pd
    ic = ev[(ev["row"] == row) & (ev["kind"] == "ic_walkway")]
    key = lambda tid, foot, t: (tid, foot, round(float(t), T_KEY_DECIMALS))  # noqa: E731
    look = {}
    for r in ic.itertuples():
        look[key(r.trial_id, r.foot, r.t0_s)] = (float(r.prediction) / 1000.0, bool(r.is_matched))
    st = ev[(ev["row"] == row) & (ev["kind"] == "stride")]
    rows, found, ends = [], 0, 0
    for r in st.itertuples():
        a, b = look.get(key(r.trial_id, r.foot, r.t0_s)), look.get(key(r.trial_id, r.foot, r.t1_s))
        found += (a is not None) + (b is not None)
        ends += 2
        T_wk = float(r.t1_s - r.t0_s)
        T_dec = float("nan")
        if a is not None and b is not None and a[1] and b[1] and math.isfinite(a[0]) and math.isfinite(b[0]) \
                and b[0] > a[0]:
            T_dec = b[0] - a[0]
        rows.append((r.person_id, r.trial_id, r.task, r.foot, float(r.t0_s), float(r.t1_s), T_wk, T_dec,
                     T_dec / T_wk if T_wk > 0 else float("nan"), float(r.reference), float(r.prediction),
                     bool(r.is_matched)))
    df = pd.DataFrame(rows, columns=["person_id", "trial_id", "task", "foot", "t0", "t1", "T_wk", "T_dec", "r_T", "x",
                                     "y", "base_matched"])
    df.attrs["ends_found"] = found / ends if ends else float("nan")
    return df


def _person_arrays(df, col_fn, persons):
    """{person: sorted array} of the finite values col_fn(df) of each person of `persons` (persons without one left
    out)."""
    v = np.asarray(col_fn(df), np.float64)
    p = df["person_id"].astype(str).to_numpy()
    out = {}
    for q in persons:
        a = v[(p == q) & np.isfinite(v)]
        if len(a):
            out[q] = np.sort(a)
    return out


def _ratio(df):
    x, y = df["x"].to_numpy(np.float64), df["y"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(x) & (x > 0), y / x, np.nan)


def median_of_person_medians(arrs, ids):
    m = [float(np.median(arrs[i])) for i in ids if i in arrs]
    return float(np.median(m)) if m else float("nan")


def pooled_median(arrs, ids):
    parts = [arrs[i] for i in ids if i in arrs]
    return float(np.median(np.concatenate(parts))) if parts else float("nan")


def draws(persons, n_boot=None, seed=None):
    """(n_boot, P) person indices drawn with replacement (seed); the same draws serve every statistic of a set."""
    rng = np.random.default_rng(SEED if seed is None else seed)
    return rng.integers(0, len(persons), size=(BOOT_N if n_boot is None else int(n_boot), len(persons)))


def interval(v, level=LEVEL):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return [float(lo), float(hi)]


def part_a_set(dur, legs, base, legs_sim, base_sim, persons, n_boot=None, seed=None):
    """Part A statistics of one set of persons: r_T (median of person medians), g_S on the same strides (pooled),
    s_T, the legs and base stride gains on all their matched strides (pooled), D_L, D_R and D_L - D_R, each with the
    person bootstrap interval (one set of draws)."""
    persons = sorted(persons)
    same = dur[np.isfinite(dur["r_T"]) & dur["base_matched"]]
    rT = _person_arrays(same, lambda d: d["r_T"].to_numpy(), persons)
    gS = _person_arrays(same, _ratio, persons)
    gL = _person_arrays(legs, _ratio, persons)
    gB = _person_arrays(base, _ratio, persons)

    def stats(ids):
        r, g = median_of_person_medians(rT, ids), pooled_median(gS, ids)
        L, B = pooled_median(gL, ids), pooled_median(gB, ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            sT = math.log(r) / math.log(g) if r > 0 and g > 0 and g != 1 else float("nan")
        DL, DR = L / legs_sim, B / base_sim
        return {"r_T": r, "g_S_same": g, "s_T": sT, "g_L": L, "g_S": B, "D_L": DL, "D_R": DR, "D_L_minus_D_R": DL - DR}

    point = stats(persons)
    D = draws(persons, n_boot, seed)
    boot = [stats([persons[j] for j in row]) for row in D]
    out = {"n_persons": len(persons), "n_strides_r_T": int(len(same[same["person_id"].isin(persons)])),
           "n_persons_r_T": len(rT), "n_strides_legs": int(sum(len(v) for v in gL.values())),
           "n_strides_base": int(sum(len(v) for v in gB.values()))}
    for k, v in point.items():
        out[k] = {"point": v, "interval": interval([b[k] for b in boot])}
    out["in_sim"] = {"g_L_sim": legs_sim, "g_S_sim": base_sim}
    return out


def reading_c1(s):
    """SPATIAL, TEMPORAL or MIXED (the prediction's C1) from a part_a_set block."""
    iv, sT = s["r_T"]["interval"], s["s_T"]["point"]
    lo, hi = C1_BAND
    if iv is None or not math.isfinite(sT):
        return "MIXED"
    if lo - EPS <= iv[0] and iv[1] <= hi + EPS and abs(sT) <= C1_SPATIAL_SHARE + EPS:
        return "SPATIAL"
    if (iv[1] < lo or iv[0] > hi) and sT >= C1_TEMPORAL_SHARE - EPS:
        return "TEMPORAL"
    return "MIXED"


def reading_c2(s):
    """LEGS, ROOT or BETWEEN (the prediction's C2) from a part_a_set block (points)."""
    DL, DR = s["D_L"]["point"], s["D_R"]["point"]
    if not (math.isfinite(DL) and math.isfinite(DR)):
        return "BETWEEN"
    if DL <= C2_LEGS_MAX + EPS and abs(DL - DR) <= C2_AGREE + EPS:
        return "LEGS"
    if DL >= C2_ROOT_LEGS_MIN - EPS and DR <= C2_ROOT_MAX + EPS:
        return "ROOT"
    return "BETWEEN"


def reading_a(per_end):
    """The Part A reading of the prediction over the B ends: {end: (C1, C2)} -> (reading, text)."""
    c1 = {k: v[0] for k, v in per_end.items()}
    c2 = {k: v[1] for k, v in per_end.items()}
    if all(v == "TEMPORAL" for v in c1.values()):
        return "not_supported", "temporal at every B end: the stride shortfall follows the decoded rhythm"
    if all(v == "ROOT" for v in c2.values()):
        return "not_supported", "root at every B end: the decoded legs sweep the walkway stride, the root falls short"
    if all(v == "SPATIAL" for v in c1.values()) and all(v == "LEGS" for v in c2.values()):
        return ("compatible", "spatial and legs at every B end: the decoded legs sweep a short stride at the walkway "
                "rhythm, compatible with H3 or with a short decoded body (H1); the angles (Part B) decide")
    return "partly", "; ".join(f"{k}: {c1[k]}, {c2[k]}" for k in per_end)


def insim_legs_gain(folder, expected_sha256=None):
    """The in-sim model_legs gains of an eval_gait_insim.py folder, forward row: stride 1 + b / (100 t), speed
    1 + bv / tv (mean based, as F1's in-sim stride gain)."""
    with open(os.path.join(folder, "insim.json"), encoding="utf-8") as f:
        d = json.load(f)
    sha = (d.get("checkpoint") or {}).get("sha256")
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{rel(folder)} holds checkpoint {str(sha)[:12]}, expected {expected_sha256[:12]}")
    lg = d["lines"]["forward"]["model_legs"]
    c = d["counts"]["forward"]
    b, t = float(lg["stride_bias_cm"]["point"]), float(c["true_stride_m"])
    bv, tv = float(lg["speed_bias_m_s"]["point"]), float(c["true_speed_m_s"])
    return {"path": rel(folder), "checkpoint_sha256": sha, "stride_gain": 1.0 + b / (100.0 * t),
            "speed_gain": 1.0 + bv / tv, "stride_bias_cm": b, "true_stride_m": t, "speed_bias_m_s": bv,
            "true_speed_m_s": tv}


def part_a(end, sets, locked, n_boot=None, seed=None, tree=None):
    """Part A of one end: {checkpoint, files, ends_found, sets {dev12, dev14}: part_a_set, readings, speeds}."""
    gg = GG()
    dd = gg.diag_dir(end.name, tree)
    with open(os.path.join(dd, "diag.json"), encoding="utf-8") as f:
        ck = json.load(f).get("checkpoint") or {}
    sha = ck.get("sha256")
    ev = read_events(os.path.join(dd, "events.csv"), sets["dev14"], locked)
    dur = stride_durations(ev)
    st = ev[(ev["kind"] == "stride") & ev["is_matched"]]
    legs = st[st["row"] == "model_legs"].rename(columns={"reference": "x", "prediction": "y"})
    base = st[st["row"] == "base"].rename(columns={"reference": "x", "prediction": "y"})
    sp = ev[(ev["kind"] == "speed") & ev["is_matched"]].rename(columns={"reference": "x", "prediction": "y"})
    ins = gg.insim_gain(gg.insim_dir(end.name, tree), sha)
    insl = insim_legs_gain(gg.insim_dir(end.name, tree), sha)
    res = {"key": end.key, "name": end.name, "checkpoint": {"path": end.checkpoint, "sha256": sha},
           "events": rel(os.path.join(dd, "events.csv")), "ends_found": dur.attrs["ends_found"],
           "insim": {"model": ins, "model_legs": insl}, "sets": {}, "speed_gains": {}}
    for name in ("dev12", "dev14"):
        res["sets"][name] = part_a_set(dur, legs, base, insl["stride_gain"], ins["stride_gain"], sets[name],
                                       n_boot, seed)
        res["sets"][name]["C1"] = reading_c1(res["sets"][name])
        res["sets"][name]["C2"] = reading_c2(res["sets"][name])
        ps = sorted(sets[name])
        res["speed_gains"][name] = {
            row: pooled_median(_person_arrays(sp[sp["row"] == row], _ratio, ps), ps) for row in ("base", "model_legs")}
        # secondary (no criterion): the same ratios on the pass speeds, the in-sim legs speed gain as divisor
        sg = res["speed_gains"][name]
        sg["D_L_speed"] = sg["model_legs"] / insl["speed_gain"]
        sg["D_R_speed"] = sg["base"] / ins["speed_gain"]
    # stride durations of all walkway strides with both decoded ICs, whatever the stride match (no selection by it)
    allr = dur[np.isfinite(dur["r_T"])]
    ps = sorted(sets["dev12"])
    a12 = allr[allr["person_id"].isin(ps)]
    pm = a12.groupby("person_id")["r_T"].mean()
    res["r_T_descriptive_dev12"] = {
        "mean_of_person_means": float(pm.mean()) if len(pm) else float("nan"),
        "stride_p10": float(np.quantile(a12["r_T"], 0.1)) if len(a12) else float("nan"),
        "stride_p90": float(np.quantile(a12["r_T"], 0.9)) if len(a12) else float("nan"),
        "share_within_10ms": float((np.abs(a12["T_dec"] - a12["T_wk"]) <= 0.010 + 1e-9).mean()) if len(a12)
        else float("nan"), "strides": int(len(a12))}
    res["r_T_all_strides_dev12"] = median_of_person_medians(_person_arrays(allr, lambda d: d["r_T"].to_numpy(), ps),
                                                            ps)
    res["share_with_r_T"] = float(np.isfinite(dur["r_T"]).mean()) if len(dur) else float("nan")
    return res


# ---------------------------------------------------------------- angles
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def angle_series(J, side):
    """J (n, 22, 3) m of one body frame, side "L" or "R" -> {hip_deg, knee_deg, thigh_deg (n,), leg_length_m (n,),
    ankle_fa_m (n,)} (module docstring)."""
    J = np.asarray(J, np.float64)
    h, k, a = J[:, JOINT[f"hip_{side}"]], J[:, JOINT[f"knee_{side}"]], J[:, JOINT[f"ankle_{side}"]]
    pel = J[:, JOINT["pelvis"]]
    lat = _unit(J[:, JOINT["hip_L"]] - J[:, JOINT["hip_R"]])
    up0 = J[:, JOINT["spine1"]] - pel
    up = _unit(up0 - (up0 * lat).sum(-1, keepdims=True) * lat)
    fwd = np.cross(lat, up)
    thigh, shank = k - h, a - k
    hip = np.degrees(np.arctan2((thigh * fwd).sum(-1), -(thigh * up).sum(-1)))
    c = (thigh * shank).sum(-1) / (np.linalg.norm(thigh, axis=-1) * np.linalg.norm(shank, axis=-1))
    knee = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    fh = fwd.copy()
    fh[:, 1] = 0.0
    fh = _unit(fh)
    thigh_incl = np.degrees(np.arctan2((thigh * fh).sum(-1), -thigh[:, 1]))
    leg = np.linalg.norm(thigh, axis=-1) + np.linalg.norm(shank, axis=-1)
    fa = ((a - pel) * fh).sum(-1)
    return {"hip_deg": hip, "knee_deg": knee, "thigh_deg": thigh_incl, "leg_length_m": leg, "ankle_fa_m": fa}


def _segment(J, wof, f0, f1):
    """Frames f0..f1 of one window of a body: J (T, 22, 3) one frame, or (W, T, 22, 3) with window_of_frame (T,);
    the window of the midpoint frame, else the window holding every frame nearest to it; None without one."""
    J = np.asarray(J, np.float64)
    if J.ndim == 3:
        seg = J[f0:f1 + 1]
        return seg if np.isfinite(seg).all() else None
    w = int(np.asarray(wof)[(f0 + f1) // 2])
    ok = np.isfinite(J[:, f0:f1 + 1]).all(axis=(1, 2, 3))
    if not ok[w]:
        cand = np.flatnonzero(ok)
        if not len(cand):
            return None
        w = int(cand[np.argmin(np.abs(cand - w))])
    return J[w, f0:f1 + 1]


def stride_features(J, wof, foot, t0, t1, rate=RATE):
    """{FEATURES} of one stride (foot "L" or "R", times t0 < t1 s) of a body (module docstring); NaN values when the
    stride lies outside the clock or no window holds it whole. ankle_m: horizontal ankle displacement between the end
    frames (the G3 stride readout at these ICs); root_m: the horizontal pelvis displacement."""
    from mdfm.readouts import gait
    f0, f1 = int(gait.frame_of(t0, rate)), int(gait.frame_of(t1, rate))
    T = np.asarray(J).shape[-3]
    nan = {k: float("nan") for k in FEATURES}
    if not 0 <= f0 < f1 < T:
        return nan
    seg = _segment(J, wof, f0, f1)
    if seg is None:
        return nan
    s = angle_series(seg, foot)
    hor = lambda d: float(np.hypot(d[0], d[2]))  # noqa: E731
    a = JOINT[f"ankle_{foot}"]
    return {"hip_range_deg": float(np.ptp(s["hip_deg"])), "hip_max_deg": float(s["hip_deg"].max()),
            "hip_min_deg": float(s["hip_deg"].min()), "knee_range_deg": float(np.ptp(s["knee_deg"])),
            "knee_max_deg": float(s["knee_deg"].max()), "thigh_range_deg": float(np.ptp(s["thigh_deg"])),
            "leg_length_m": float(np.mean(s["leg_length_m"])), "ankle_fa_range_m": float(np.ptp(s["ankle_fa_m"])),
            "ankle_m": hor(seg[-1, a] - seg[0, a]), "root_m": hor(seg[-1, 0] - seg[0, 0])}


def legs_length(speed, t0, t1, rate=RATE):
    """The legs stride (m): the mean stance no-slip speed over [t0, t1] times t1 - t0 (eval_gait_insim)."""
    return _sibling("eval_gait_insim").interval_mean(speed, t0, t1, rate) * (t1 - t0)


# ---------------------------------------------------------------- Part B: the dump read
def fit_line(x, y):
    """(a, b, n) of the least squares line y = a + b x over the finite pairs."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(a), float(b), int(ok.sum())


def reading_b(ratio, upper):
    """SUPPORTED, NOT_SUPPORTED or PARTLY (the prediction's Part B) from rho_real / rho_sim and the bootstrap upper
    bound of rho_real."""
    if not (ratio is not None and math.isfinite(ratio)):
        return "NOT_EVALUATED"
    if ratio <= B_SUPPORT_MAX + EPS and upper is not None and upper < B_UPPER_BELOW:
        return "SUPPORTED"
    if ratio >= B_NOT_MIN - EPS:
        return "NOT_SUPPORTED"
    return "PARTLY"


def part_b_descriptive(real, sim, sets, n_boot=None, seed=None):
    """Descriptive only, added by task F2D on 2026-10-05 after the prediction and before the dump was read; no
    criterion. On the real strides of train54 and the scored in-sim strides:
    - thigh inclination range read as the registered joints are (rho_thigh real against sim);
    - speed dependence: least squares slope of the range on the stride speed, real decoded (walkway speed), in-sim
      true and in-sim decoded (true speed), and their ratios (real decoded over in-sim true, in-sim decoded over
      in-sim true): a real slope below the in-sim one is the compression of the amplitude-to-speed relation;
    - persons: Pearson, over train54 persons, of the person median of H_dec / H_expected with the person median of
      the decoded over walkway stride (row ankle_m at the walkway ICs): does a person whose decoded hip arc is short
      also get a short stride;
    - decoded leg length: real median, in-sim decoded and true medians."""
    tr = sorted(sets["train54"])
    rtr = real[real["person_id"].isin(tr)]
    out = {"note": "descriptive, no criterion (task F2D, 2026-10-05)"}
    feats = ("hip_range_deg", "knee_range_deg", "thigh_range_deg", "ankle_fa_range_m", "leg_length_m")
    need_real = {"walkway_speed_m_s", "walkway_m", "dec_ankle_m"} | {f"dec_{k}" for k in feats}
    need_sim = {"true_speed_m_s", "true_stride_m"} | {f"{s}_{k}" for s in ("true", "dec") for k in feats}
    missing = sorted((need_real - set(real.columns)) | (need_sim - set(sim.columns)))
    if missing:
        return {**out, "skipped": missing}
    a, b, n = fit_line(sim["true_speed_m_s"], sim["true_thigh_range_deg"])
    sim_r = sim.assign(r=sim["dec_thigh_range_deg"] / sim["true_thigh_range_deg"])
    sp = sorted(set(sim_r["person_id"]))
    rho_sim = median_of_person_medians(_person_arrays(sim_r, lambda d: d["r"].to_numpy(), sp), sp)
    rr = rtr.assign(r=rtr["dec_thigh_range_deg"].to_numpy(np.float64)
                    / (a + b * rtr["walkway_speed_m_s"].to_numpy(np.float64)))
    arr = _person_arrays(rr, lambda d: d["r"].to_numpy(), tr)
    pt = median_of_person_medians(arr, tr)
    bt = [median_of_person_medians(arr, [tr[i] for i in row]) for row in draws(tr, n_boot, seed)]
    out["thigh"] = {"truth_line": {"a": a, "b": b, "n": n}, "rho_sim": rho_sim, "rho_real_train54": pt,
                    "interval": interval(bt), "ratio_to_sim": pt / rho_sim if rho_sim else float("nan")}
    slopes = {}
    for j in ("hip", "knee", "thigh"):
        col = f"{j}_range_deg"
        _, b_real, n_real = fit_line(rtr["walkway_speed_m_s"], rtr[f"dec_{col}"])
        _, b_true, _ = fit_line(sim["true_speed_m_s"], sim[f"true_{col}"])
        _, b_dec, _ = fit_line(sim["true_speed_m_s"], sim[f"dec_{col}"])
        slopes[j] = {"real_decoded_deg_per_m_s": b_real, "n_real": n_real, "sim_true_deg_per_m_s": b_true,
                     "sim_decoded_deg_per_m_s": b_dec,
                     "real_decoded_over_sim_true": b_real / b_true if b_true else float("nan"),
                     "sim_decoded_over_sim_true": b_dec / b_true if b_true else float("nan")}
    out["speed_slopes"] = slopes
    ah, bh, _ = fit_line(sim["true_speed_m_s"], sim["true_hip_range_deg"])
    q = rtr.assign(rh=rtr["dec_hip_range_deg"] / (ah + bh * rtr["walkway_speed_m_s"]),
                   rs=rtr["dec_ankle_m"] / rtr["walkway_m"])
    pm = q.groupby("person_id")[["rh", "rs"]].median().dropna()
    out["persons_hip_ratio_vs_stride_ratio"] = {
        "n_persons": int(len(pm)), "pearson": GG().pearson(pm["rh"], pm["rs"]),
        "median_stride_ratio": float(pm["rs"].median()) if len(pm) else float("nan")}
    out["leg_length_m"] = {"real_decoded_train54": float(np.nanmedian(rtr["dec_leg_length_m"])),
                           "sim_decoded": float(np.nanmedian(sim["dec_leg_length_m"])),
                           "sim_true": float(np.nanmedian(sim["true_leg_length_m"]))}
    sp_r, sp_s = rtr["walkway_speed_m_s"].to_numpy(np.float64), sim["true_speed_m_s"].to_numpy(np.float64)
    # the expectation on the stride instead of the speed: the least squares line of the true in-sim range (and the
    # pelvis-relative ankle sweep) on the true stride, evaluated at the walkway stride; and how the real stride at a
    # speed compares with the in-sim one (the speed-based expectation assumes the same stride at the same speed)
    stride_based = {}
    for j in ("hip_range_deg", "knee_range_deg", "thigh_range_deg", "ankle_fa_range_m"):
        a_, b_, n_ = fit_line(sim["true_stride_m"], sim[f"true_{j}"])
        sim_r = sim.assign(r=sim[f"dec_{j}"] / sim[f"true_{j}"])
        sps = sorted(set(sim_r["person_id"]))
        rs = median_of_person_medians(_person_arrays(sim_r, lambda d: d["r"].to_numpy(), sps), sps)
        q = rtr.assign(r=rtr[f"dec_{j}"].to_numpy(np.float64) / (a_ + b_ * rtr["walkway_m"].to_numpy(np.float64)))
        arr = _person_arrays(q, lambda d: d["r"].to_numpy(), tr)
        ids = [p for p in tr if p in arr]
        pt = median_of_person_medians(arr, ids)
        bt = [median_of_person_medians(arr, [ids[i] for i in row]) for row in draws(ids, n_boot, seed)] if ids else []
        stride_based[j] = {"line": {"a": a_, "b": b_, "n": n_}, "rho_real": pt, "interval": interval(bt),
                           "rho_sim": rs, "ratio_to_sim": pt / rs if rs else float("nan")}
    out["stride_based_expectation"] = stride_based
    a_s, b_s, n_s = fit_line(sim["true_speed_m_s"], sim["true_stride_m"])
    ratio = rtr["walkway_m"].to_numpy(np.float64) / (a_s + b_s * rtr["walkway_speed_m_s"].to_numpy(np.float64))
    arr = _person_arrays(rtr.assign(r=ratio), lambda d: d["r"].to_numpy(), tr)
    out["real_stride_over_insim_stride_at_equal_speed"] = {
        "insim_line": {"a": a_s, "b": b_s, "n": n_s}, "median_of_person_medians": median_of_person_medians(arr, tr),
        "interval": interval([median_of_person_medians(arr, [tr[i] for i in row]) for row in draws(tr, n_boot, seed)])}
    # sensitivity of the registered ratio: what rho_real / rho_sim a decoded body would read if its arcs followed
    # the truth line at its own decoded stride speed (dec_ankle_m over the stride time) with the in-sim decoded bias,
    # that is H3 in its plain form at the measured stride shortfall
    if {"t0", "t1"} <= set(rtr.columns):
        v_dec = rtr["dec_ankle_m"].to_numpy(np.float64) / (rtr["t1"] - rtr["t0"]).to_numpy(np.float64)
        v_wk = rtr["walkway_speed_m_s"].to_numpy(np.float64)
        sens = {}
        for j in ("hip", "knee"):
            a_, b_, _ = fit_line(sim["true_speed_m_s"], sim[f"true_{j}_range_deg"])
            arr = _person_arrays(rtr.assign(r=(a_ + b_ * v_dec) / (a_ + b_ * v_wk)), lambda d: d["r"].to_numpy(), tr)
            sens[j] = median_of_person_medians(arr, tr)
        out["ratio_if_arcs_followed_the_decoded_speed"] = sens
    lo, hi = float(np.nanquantile(sp_s, 0.1)), float(np.nanquantile(sp_s, 0.9))
    out["stride_speed_m_s"] = {"real_train54_median": float(np.nanmedian(sp_r)),
                               "real_train54_p90": float(np.nanquantile(sp_r, 0.9)),
                               "sim_median": float(np.nanmedian(sp_s)), "sim_p10": lo, "sim_p90": hi,
                               "real_share_above_sim_p90": float(np.mean(sp_r > hi))}
    # the registered rho_H and rho_K restricted to the real strides inside the in-sim speed range (no extrapolation
    # of the truth line)
    band = rtr[(rtr["walkway_speed_m_s"] >= lo) & (rtr["walkway_speed_m_s"] <= hi)]
    out["in_sim_speed_band"] = {"n_strides": int(len(band)), "n_persons": int(band["person_id"].nunique())}
    for j in ("hip", "knee"):
        col = f"{j}_range_deg"
        a_, b_, _ = fit_line(sim["true_speed_m_s"], sim[f"true_{col}"])
        sim_r = sim.assign(r=sim[f"dec_{col}"] / sim[f"true_{col}"])
        sps = sorted(set(sim_r["person_id"]))
        rs = median_of_person_medians(_person_arrays(sim_r, lambda d: d["r"].to_numpy(), sps), sps)
        bb = band.assign(r=band[f"dec_{col}"].to_numpy(np.float64)
                         / (a_ + b_ * band["walkway_speed_m_s"].to_numpy(np.float64)))
        arr = _person_arrays(bb, lambda d: d["r"].to_numpy(), tr)
        ids = [p for p in tr if p in arr]
        pt = median_of_person_medians(arr, ids)
        bt = [median_of_person_medians(arr, [ids[i] for i in row]) for row in draws(ids, n_boot, seed)] if ids else []
        out["in_sim_speed_band"][j] = {"rho_real": pt, "interval": interval(bt),
                                       "ratio_to_sim": pt / rs if rs else float("nan")}
    return out


def prediction_section_sha256(report=None):
    """SHA-256 of the section '## Prediction' of the report (heading line to the next '## ' heading, LF): recorded
    with a rerun so that the reading is tied to the text it was read under (the script held no hash before)."""
    import hashlib
    with open(report or os.path.join(TREE, REPORT), encoding="utf-8") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    a = next(i for i, x in enumerate(lines) if x.strip() == "## Prediction")
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return hashlib.sha256(("\n".join(lines[a:b]).rstrip("\n") + "\n").encode("utf-8")).hexdigest()


def assert_same_criteria(res, previous=None):
    """AssertionError unless the criteria and the bootstrap of res equal those stored by the first run
    (reports/bench/f2_scale/H3angles/results.json, written before the dump existed)."""
    with open(previous or os.path.join(OUT, "results.json"), encoding="utf-8") as f:
        old = json.load(f)
    for k in ("criteria", "bootstrap", "prediction"):
        if json.loads(json.dumps(res[k])) != old[k]:
            raise AssertionError(f"{k} differ from those of the first run: {old[k]} against {res[k]}")
    return True


def part_b(folder, sets, locked, n_boot=None, seed=None):
    """Part B from a dump folder: the in-sim truth lines of the hip and knee ranges on the true stride speed, rho_sim,
    rho_real per set (train54 primary, dev12, dev14) with intervals, the readings, and descriptive ratios."""
    import pandas as pd
    real_p, sim_p = os.path.join(folder, "strides_real.csv"), os.path.join(folder, "strides_insim.csv")
    if not (os.path.exists(real_p) and os.path.exists(sim_p)):
        return {"evaluated": False, "reason": f"no dump in {rel(folder)}"}
    who = pd.read_csv(real_p, encoding="utf-8", usecols=["person_id"], dtype=str)
    GG().guard_persons(set(who["person_id"]), set(sets["dev14"]) | set(sets["train54"]), locked, "strides_real.csv")
    real = pd.read_csv(real_p, encoding="utf-8", dtype={"person_id": str, "trial_id": str, "foot": str})
    sim = pd.read_csv(sim_p, encoding="utf-8", dtype={"person_id": str, "take_id": str, "foot": str})
    sim = sim[sim["scored"].astype(str).str.lower() == "true"]
    out = {"evaluated": True, "dump": rel(folder), "n_sim_strides": int(len(sim)),
           "n_sim_persons": int(sim["person_id"].nunique()), "published_context": PUBLISHED, "joints": {}}
    for j in ("hip", "knee"):
        col = f"{j}_range_deg"
        a, b, n = fit_line(sim["true_speed_m_s"], sim[f"true_{col}"])
        sim_r = sim.assign(r=sim[f"dec_{col}"] / sim[f"true_{col}"])
        sp = sorted(set(sim_r["person_id"]))
        rho_sim = median_of_person_medians(_person_arrays(sim_r, lambda d: d["r"].to_numpy(), sp), sp)
        exp = a + b * real["walkway_speed_m_s"].to_numpy(np.float64)
        rr = real.assign(r=real[f"dec_{col}"].to_numpy(np.float64) / exp)
        blk = {"truth_line": {"a": a, "b": b, "n": n}, "rho_sim": rho_sim, "sets": {}}
        for sname in ("train54", "dev12", "dev14"):
            ps = sorted(sets[sname])
            arr = _person_arrays(rr, lambda d: d["r"].to_numpy(), ps)
            pt = median_of_person_medians(arr, ps)
            bt = [median_of_person_medians(arr, [ps[i] for i in row]) for row in draws(ps, n_boot, seed)]
            iv = interval(bt)
            ratio = pt / rho_sim if rho_sim and math.isfinite(rho_sim) else float("nan")
            blk["sets"][sname] = {"rho_real": pt, "interval": iv, "ratio_to_sim": ratio,
                                  "reading": reading_b(ratio, iv[1] if iv else None), "n_persons": len(arr),
                                  "median_decoded_deg": float(np.nanmedian(rr.loc[rr["person_id"].isin(ps),
                                                                                  f"dec_{col}"])),
                                  "median_expected_deg": float(np.nanmedian(exp[rr["person_id"].isin(ps).to_numpy()]))}
        blk["sim_medians"] = {"true_deg": float(np.nanmedian(sim[f"true_{col}"])),
                              "dec_deg": float(np.nanmedian(sim[f"dec_{col}"]))}
        out["joints"][j] = blk
    out["descriptive_dumps_2026_10_05"] = part_b_descriptive(real, sim, sets, n_boot, seed)
    desc = {}
    for sname in ("train54", "dev12"):
        q = real[real["person_id"].isin(sets[sname])]
        desc[sname] = {k: float(np.nanmedian(q[f"dec_{k}"] / q["walkway_m"])) for k in ("ankle_m", "root_m")}
        desc[sname]["legs_m"] = float(np.nanmedian(q["legs_m"] / q["walkway_m"]))
        desc[sname]["dec_leg_length_m"] = float(np.nanmedian(q["dec_leg_length_m"]))
    desc["insim"] = {"dec_over_true_leg_length": float(np.nanmedian(sim["dec_leg_length_m"] / sim["true_leg_length_m"])),
                     "true_leg_length_m": float(np.nanmedian(sim["true_leg_length_m"])),
                     "legs_over_true_stride": float(np.nanmedian(sim["legs_m"] / sim["true_ankle_m"]))}
    out["descriptive"] = desc
    out["reading"] = out["joints"]["hip"]["sets"]["train54"]["reading"]
    return out


# ---------------------------------------------------------------- summary
def fx(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:.{nd}f}" if math.isfinite(v) else "n/a"


def fiv(iv, nd=3):
    return f"[{fx(iv[0], nd)}, {fx(iv[1], nd)}]" if iv else "n/a"


def summary_md(r):
    L = [f"# F2-H3angles: leg excursion and cadence of the decoded walk ({LABEL})", "",
         f"Written by scripts/f2_H3angles.py; every number below is read from results.json of this folder. Prediction "
         f"and criteria: {REPORT}, section Prediction, written {PREDICTION_AT} before any value was read. No lockbox "
         "person, EVAL or CODY record was read (asserted).", "",
         f"## Part A reading: {r['reading_a']['reading']}", "", r["reading_a"]["text"], "",
         "| end | set | r_T (decoded over walkway stride duration) | 95 % | g_S same strides | s_T | 95 % | C1 | g_L real "
         "(legs) | g_L sim | D_L | 95 % | g_S real | g_S sim | D_R | 95 % | D_L - D_R | 95 % | C2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["part_a"].items():
        for sname, s in e["sets"].items():
            L.append(f"| {k} | {sname} | {fx(s['r_T']['point'])} | {fiv(s['r_T']['interval'])} | "
                     f"{fx(s['g_S_same']['point'])} | {fx(s['s_T']['point'])} | {fiv(s['s_T']['interval'])} | {s['C1']} | "
                     f"{fx(s['g_L']['point'])} | {fx(s['in_sim']['g_L_sim'])} | {fx(s['D_L']['point'])} | "
                     f"{fiv(s['D_L']['interval'])} | {fx(s['g_S']['point'])} | {fx(s['in_sim']['g_S_sim'])} | "
                     f"{fx(s['D_R']['point'])} | {fiv(s['D_R']['interval'])} | {fx(s['D_L_minus_D_R']['point'])} | "
                     f"{fiv(s['D_L_minus_D_R']['interval'])} | {s['C2']} |")
    L += ["", "Counts and checks per end:", ""]
    for k, e in r["part_a"].items():
        s12 = e["sets"]["dev12"]
        L.append(f"- {k} ({e['checkpoint']['path']}, SHA-256 {e['checkpoint']['sha256']}): {e['events']}; walkway "
                 f"stride ends found among the walkway IC rows {fx(e['ends_found'])}; strides with a decoded duration "
                 f"{fx(e['share_with_r_T'])}; dev12: {s12['n_strides_r_T']} strides of {s12['n_persons_r_T']} persons "
                 f"for r_T, {s12['n_strides_legs']} legs and {s12['n_strides_base']} base matched strides; r_T over "
                 f"every walkway stride with two decoded ICs (no stride-match selection), dev12: "
                 f"{fx(e['r_T_all_strides_dev12'])}; pass speed gains dev12 base {fx(e['speed_gains']['dev12']['base'])}"
                 f", model_legs {fx(e['speed_gains']['dev12']['model_legs'])}; in-sim model_legs speed gain "
                 f"{fx(e['insim']['model_legs']['speed_gain'])}, in-sim model speed gain "
                 f"{fx(e['insim']['model']['speed_gain'])}; secondary (no criterion), dev12 pass speeds: D_L_speed "
                 f"{fx(e['speed_gains']['dev12']['D_L_speed'])}, D_R_speed {fx(e['speed_gains']['dev12']['D_R_speed'])}; "
                 f"r_T descriptive, dev12: mean of person means {fx(e['r_T_descriptive_dev12']['mean_of_person_means'], 4)}"
                 f", stride 10th and 90th percentiles {fx(e['r_T_descriptive_dev12']['stride_p10'])} and "
                 f"{fx(e['r_T_descriptive_dev12']['stride_p90'])}, share of strides whose decoded duration is within "
                 f"10 ms of the walkway one {fx(e['r_T_descriptive_dev12']['share_within_10ms'])} "
                 f"({e['r_T_descriptive_dev12']['strides']} strides).")
    b = r["part_b"]
    L += ["", f"## Part B (angle dump of {B_END})", ""]
    if not b.get("evaluated"):
        L.append(f"Not evaluated: {b.get('reason')}. The dump is queued as job {JOB['name']} (scripts/queue_job.py, "
                 "group F2); rerun this script once it has run.")
    else:
        L += [f"Reading (hip, train54): {b['reading']}.", "",
              "| joint | set | rho_real | 95 % | rho_sim | ratio | reading | decoded median deg | expected median deg |",
              "|---|---|---|---|---|---|---|---|---|"]
        for j, blk in b["joints"].items():
            for sname, s in blk["sets"].items():
                L.append(f"| {j} | {sname} | {fx(s['rho_real'])} | {fiv(s['interval'])} | {fx(blk['rho_sim'])} | "
                         f"{fx(s['ratio_to_sim'])} | {s['reading']} | {fx(s['median_decoded_deg'], 1)} | "
                         f"{fx(s['median_expected_deg'], 1)} |")
        L += ["", "Truth lines (in-sim, range on true stride speed): " + "; ".join(
            f"{j}: a {fx(blk['truth_line']['a'], 2)} deg, b {fx(blk['truth_line']['b'], 2)} deg per m/s, n "
            f"{blk['truth_line']['n']}" for j, blk in b["joints"].items()) + ".",
              f"Descriptive: {json.dumps(b['descriptive'], sort_keys=True)}.",
              f"Context (no criterion): {PUBLISHED['source']}: hip arc about {fx(PUBLISHED['hip_arc_deg'], 0)} deg, "
              f"knee arc about {fx(PUBLISHED['knee_arc_deg'], 0)} deg at about {fx(PUBLISHED['speed_m_s'], 2)} m/s."]
        d = b.get("descriptive_dumps_2026_10_05")
        if d and d.get("skipped"):
            L += ["", f"Descriptive block of task F2D skipped: columns absent ({', '.join(d['skipped'])})."]
        elif d:
            L += ["", "### Descriptive, no criterion (task F2D, 2026-10-05)", "",
                  f"- Thigh inclination range, train54: rho_real {fx(d['thigh']['rho_real_train54'])} "
                  f"{fiv(d['thigh']['interval'])}, rho_sim {fx(d['thigh']['rho_sim'])}, ratio "
                  f"{fx(d['thigh']['ratio_to_sim'])}.",
                  "- Speed slopes of the ranges (deg per m/s): " + "; ".join(
                      f"{j}: real decoded {fx(v['real_decoded_deg_per_m_s'], 2)}, in-sim true "
                      f"{fx(v['sim_true_deg_per_m_s'], 2)}, in-sim decoded {fx(v['sim_decoded_deg_per_m_s'], 2)} "
                      f"(real decoded over in-sim true {fx(v['real_decoded_over_sim_true'])}, in-sim decoded over "
                      f"in-sim true {fx(v['sim_decoded_over_sim_true'])})" for j, v in d["speed_slopes"].items()) + ".",
                  f"- Inside the in-sim stride speed range [{fx(d['stride_speed_m_s']['sim_p10'], 2)}, "
                  f"{fx(d['stride_speed_m_s']['sim_p90'], 2)}] m/s ({d['in_sim_speed_band']['n_strides']} train54 "
                  f"strides, {d['in_sim_speed_band']['n_persons']} persons): hip rho_real "
                  f"{fx(d['in_sim_speed_band']['hip']['rho_real'])} {fiv(d['in_sim_speed_band']['hip']['interval'])}, "
                  f"ratio {fx(d['in_sim_speed_band']['hip']['ratio_to_sim'])}; knee rho_real "
                  f"{fx(d['in_sim_speed_band']['knee']['rho_real'])} {fiv(d['in_sim_speed_band']['knee']['interval'])}, "
                  f"ratio {fx(d['in_sim_speed_band']['knee']['ratio_to_sim'])}. Real train54 stride speed median "
                  f"{fx(d['stride_speed_m_s']['real_train54_median'], 2)} m/s, share above the in-sim 90th percentile "
                  f"{fx(d['stride_speed_m_s']['real_share_above_sim_p90'])}.",
                  f"- Persons (train54, {d['persons_hip_ratio_vs_stride_ratio']['n_persons']}): Pearson of the hip "
                  f"ratio with the decoded over walkway stride {fx(d['persons_hip_ratio_vs_stride_ratio']['pearson'])} "
                  f"(median stride ratio {fx(d['persons_hip_ratio_vs_stride_ratio']['median_stride_ratio'])}).",
                  f"- Decoded leg length (m): real train54 {fx(d['leg_length_m']['real_decoded_train54'])}, in-sim "
                  f"decoded {fx(d['leg_length_m']['sim_decoded'])}, in-sim true {fx(d['leg_length_m']['sim_true'])}.",
                  f"- Real walkway stride over the in-sim true stride at the same stride speed (in-sim line a "
                  f"{fx(d['real_stride_over_insim_stride_at_equal_speed']['insim_line']['a'])} m, b "
                  f"{fx(d['real_stride_over_insim_stride_at_equal_speed']['insim_line']['b'])} s), train54: "
                  f"{fx(d['real_stride_over_insim_stride_at_equal_speed']['median_of_person_medians'])} "
                  f"{fiv(d['real_stride_over_insim_stride_at_equal_speed']['interval'])}.",
                  "- Expectation on the walkway stride instead of its speed (true in-sim line on the true stride), "
                  "train54: " + "; ".join(
                      f"{j}: rho_real {fx(v['rho_real'])} {fiv(v['interval'])}, rho_sim {fx(v['rho_sim'])}, ratio "
                      f"{fx(v['ratio_to_sim'])}" for j, v in d["stride_based_expectation"].items()) + "."]
            if d.get("ratio_if_arcs_followed_the_decoded_speed"):
                s = d["ratio_if_arcs_followed_the_decoded_speed"]
                L.append(f"- Sensitivity of the registered ratio: a decoded body whose arcs followed the truth line at "
                         f"its own decoded stride speed (with the in-sim decoded bias) would read rho_real / rho_sim "
                         f"{fx(s['hip'])} (hip) and {fx(s['knee'])} (knee), train54.")
    rr = r.get("rerun")
    if rr:
        L += ["", f"Rerun of {rr['first_run']}: criteria equal to the first run (asserted); Prediction section SHA-256 "
              f"at the rerun {rr['prediction_section_sha256']}."]
    L += ["", "## Checks", "",
          f"- Sets: dev14 SHA-256 {r['sets']['dev14_sha256']}, train54 SHA-256 {r['sets']['train54_sha256']} (equal to "
          "F1's); persons of the sets in the weargait_pd lockbox list: 0 (asserted).",
          f"- MoveAtlas HEAD {r['moveatlas']['head']}. Run started {r['run_started']}, {fx(r['seconds'], 1)} s."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the analysis run
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_analysis(sets, locked, out=None, n_boot=None, seed=None, tree=None, stamp=None):
    from mdfm.data import sources
    t = time.time()
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    pa = {e.key: part_a(e, sets, locked, n_boot, seed, tree) for e in ends()}
    b_ends = {k: (v["sets"]["dev12"]["C1"], v["sets"]["dev12"]["C2"]) for k, v in pa.items()
              if k in GG().GAIN_ENDS}
    ra, txt = reading_a(b_ends)
    end_b = next(e for e in ends() if e.key == B_END)
    pb = part_b(dump_dir(end_b.name, out), sets, locked, n_boot, seed)
    if (set(sets["dev14"]) | set(sets["train54"])) & set(locked):
        raise AssertionError("lockbox persons in the sets")
    res = {"what": LABEL, "task": TASK, "prediction": {"report": REPORT, "written": PREDICTION_AT},
           "criteria": {"C1_band": C1_BAND, "C1_spatial_share": C1_SPATIAL_SHARE,
                        "C1_temporal_share": C1_TEMPORAL_SHARE, "C2_legs_max": C2_LEGS_MAX, "C2_agree": C2_AGREE,
                        "C2_root_legs_min": C2_ROOT_LEGS_MIN, "C2_root_max": C2_ROOT_MAX,
                        "B_support_max": B_SUPPORT_MAX, "B_not_min": B_NOT_MIN, "B_upper_below": B_UPPER_BELOW},
           "bootstrap": {"n": BOOT_N if n_boot is None else n_boot, "seed": SEED if seed is None else seed},
           "part_a": pa, "reading_a": {"reading": ra, "text": txt, "per_end_dev12": b_ends}, "part_b": pb,
           "sets": {"dev14_sha256": sources.person_hash(sets["dev14"]),
                    "train54_sha256": sources.person_hash(sets["train54"]), "n_dev14": len(sets["dev14"]),
                    "n_dev12": len(sets["dev12"]), "n_train54": len(sets["train54"])},
           "lockbox": {"persons_in_lockbox": 0, "asserted": True},
           "moveatlas": {"head": (stamp or {}).get("head")}, "run_started": run_started,
           "seconds": time.time() - t}
    return clean(res)


# ---------------------------------------------------------------- --dump (GPU)
def dump_real(pred, persons, train, locked, limit=None, log=print):
    """Rows of strides_real.csv: every walkway stride of the records of `persons` (train54 and dev14)."""
    from mdfm.gates import g3
    gg = GG()
    diag = gg.DIAG()
    gg.guard_persons(persons, None, locked, "dump persons")
    tg = g3.load_targets(persons)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg.counts['lockbox_rows']} lockbox target rows kept")
    rows, n = [], 0
    for tid, rec in g3.imu_records(persons, tg.trials):
        gg.check_record(tid, rec, persons)
        if limit is not None and n >= int(limit):
            break
        n += 1
        tb = pred(tid, g3.model_input(rec))
        legs = diag.legs_speed_body(tb)
        for r in tg.strides[tg.strides["trial_id"] == tid].itertuples():
            f = stride_features(tb.joints, tb.window_of_frame, r.foot, r.t0, r.t1, tb.rate)
            rows.append({"trial_id": tid, "person_id": r.person_id, "set": "train54" if r.person_id in train else "dev14",
                         "task": r.task, "pass_id": str(r.pass_id), "foot": r.foot, "t0": r.t0, "t1": r.t1,
                         "walkway_m": r.length_m, "walkway_speed_m_s": r.length_m / (r.t1 - r.t0),
                         "legs_m": legs_length(legs, r.t0, r.t1, tb.rate), **{f"dec_{k}": v for k, v in f.items()}})
        log(f"  {tid}: {pred.log[-1]['windows']} windows")
        del tb, rec
    return rows


def dump_insim(model, trained, device, seed=20260929, refine_steps=20, points=256, batch_windows=16, limit=None,
               log=print):
    """Rows of strides_insim.csv: every true stride of the AMASS dev walking takes of scripts/eval_gait_insim.py
    (render, inference and truth as its run_take), with the truth's and the decoded body's features at the true ICs,
    scored (steady and straight) and forward flags."""
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index, body_store
    from mdfm.eval import infer as I
    from mdfm.factory import layouts as LY
    from mdfm.factory import synth as SY
    from mdfm.gates import g3
    from mdfm.readouts import gait
    import torch
    ins = _sibling("eval_gait_insim")
    store = body_store.BodyStore(split="dev")
    _, aw_dev = ins.amass_walks_rows()
    takes = ins.select_takes(store.takes, extra=aw_dev)
    if limit:
        takes = takes.iloc[:int(limit)].reset_index(drop=True)
    if set(takes["person_id"]) & set(trained):
        raise Refused("dev persons among the checkpoint's training persons")
    if set(takes["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person among the dev takes")
    layout = ins.site_layout(LY.load_layouts()[ins.LAYOUT], None)
    sampler = SY.SynthSampler(store=store, device=torch.device(device))
    bank = SkeletonBank(torch.float64)
    rows = []
    for _, take in takes.iterrows():
        tid, pid = str(take["take_id"]), str(take["person_id"])
        I.refuse_cody(tid, pid)
        bodies, win = ins.render_take(sampler, store, int(take["i"]), layout, I.seed_of(seed, "render", tid))
        raw = win.streams[0]
        wins, sp, fps, tb_t, idx = ins.model_windows(raw, pid, tid, win.cov)
        wb = I.infer(model, wins, refine_steps=refine_steps, points=points, groups=[tid] * len(wins),
                     batch_windows=batch_windows, seed=I.seed_of(seed, tid))
        starts = [int(round(s / fps * RATE)) for s, _ in sp]
        body = g3.stitched_body(tid, [b.joints for b in wb], [b.contact for b in wb], None, starts, RATE)
        J, _, contact = ins.truth_on_clock(bodies, bank, tb_t)
        ics_t = gait.initial_contacts(contact.astype(np.float64), RATE)
        st_t = gait.strides(J, ics_t, rate=RATE)
        pel = J[:, gait.PELVIS][:, [0, 2]]
        ps = ins.passes_from_ics(ics_t, pel)
        ins.steady_straight(st_t, ps, pel)
        legs = ins.legs_speed(wb, starts, RATE)
        for r in st_t.itertuples():
            ft = stride_features(J, None, r.foot, r.t0, r.t1, RATE)
            fd = stride_features(body.joints, body.window_of_frame, r.foot, r.t0, r.t1, RATE)
            rows.append({"take_id": tid, "person_id": pid, "forward": bool(take["forward"]),
                         "scored": bool(getattr(r, "scored", False)), "foot": r.foot, "t0": r.t0, "t1": r.t1,
                         "true_stride_m": r.length_m, "true_speed_m_s": r.length_m / (r.t1 - r.t0),
                         "legs_m": legs_length(legs, r.t0, r.t1, RATE), **{f"true_{k}": v for k, v in ft.items()},
                         **{f"dec_{k}": v for k, v in fd.items()}})
        log(f"  {tid}: {len(wins)} windows, {len(st_t)} strides")
    return rows, {"takes": int(len(takes)), "persons": int(takes["person_id"].nunique()),
                  "forward_takes": int(takes["forward"].sum())}


def dump(a, log=print):
    """--dump (module docstring): returns the output folder."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    import pandas as pd
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.gates import common
    from mdfm.train import loop as LP
    gg = GG()
    t_start = time.time()
    sets, locked = get_sets()
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    out = a.out or dump_dir(loaded.name)
    paths.assert_not_moveatlas(out)
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")
    doc = {"what": LABEL, "task": TASK, "mode": "dump", "checkpoint": loaded.describe(),
           "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S,
                         "hop_s": I.HOP_S, "batch_windows": a.batch_windows, "seed": a.seed,
                         "shape_pooling": "per trial", "weights": a.weights, "device": str(device),
                         "amp_bf16": device.type == "cuda"},
           "sets": {"train54_sha256": sources.person_hash(sets["train54"]),
                    "dev14_sha256": sources.person_hash(sets["dev14"])}}
    if not a.no_real:
        pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                             seed=a.seed, pool=True)
        persons = sorted(sets["train54"]) + sorted(sets["dev14"])
        rows = dump_real(pred, persons, set(sets["train54"]), locked, a.limit_trials, log)
        common.write_csv(pd.DataFrame(rows), os.path.join(out, "strides_real.csv"))
        doc["real"] = {"trials": len(pred.log), "strides": len(rows)}
    if not a.no_insim:
        rows, info = dump_insim(loaded.model, loaded.trained_persons(), a.device, seed=a.seed,
                                refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                                limit=a.limit_takes, log=log)
        common.write_csv(pd.DataFrame(rows), os.path.join(out, "strides_insim.csv"))
        doc["insim"] = {**info, "strides": len(rows)}
    if device.type == "cuda":
        doc["inference"]["memory"] = {"max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
                                      "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9, "cap_gb": cap}
    doc["timing_s"] = round(time.time() - t_start, 1)
    common.write_text(os.path.join(out, "dump.json"), json.dumps(clean(doc), indent=1, ensure_ascii=False) + "\n")
    prov = {"script": "scripts/f2_H3angles.py", "mode": "dump", "label": LABEL, "checkpoint": rel(loaded.path),
            "checkpoint_sha256": loaded.sha256, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "moveatlas_stamp": gg.stamp_moveatlas(),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), gg.__file__, I.__file__]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    log(f"written {rel(out)}")
    return out


def job_command():
    """The scripts/queue_job.py argument list of the one dump job (cls 1, explicit --out under OUT)."""
    e = next(x for x in ends() if x.key == B_END)
    out = f"reports/bench/f2_scale/H3angles/dump_{e.name}"
    return ["scripts/queue_job.py", "--name", JOB["name"], "--cls", "1", "--priority", str(JOB["priority"]),
            "--est-hours", str(JOB["est_hours"]), "--group", "F2", "--requires", e.checkpoint,
            "--outputs", f"{out}/strides_real.csv", f"{out}/strides_insim.csv",
            "--note", f"F2-H3angles: hip and knee ranges, leg length and legs stride per stride, real (train54, dev14) "
                      f"and in-sim, decoded by {e.name}",
            "--", "scripts/f2_H3angles.py", "--dump", "--checkpoint", e.checkpoint, "--batch-windows",
            str(JOB["batch_windows"]), "--memory-gb", JOB["memory_gb"], "--out", out]


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", action="store_true", help="the angle dump (GPU, queued)")
    p.add_argument("--print-job", action="store_true", help="print the queue_job.py command of the dump job")
    p.add_argument("--checkpoint")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--limit-trials", type=int, default=None, help="--dump: real trials read, at most (tests)")
    p.add_argument("--limit-takes", type=int, default=None, help="--dump: in-sim takes read, at most (tests)")
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--no-insim", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--results-out", default=None,
                   help="analysis: the folder of results.json, summary.md and _provenance.json (default "
                        "reports/bench/f2_scale/H3angles/; dumps are still read there); a rerun names a new folder "
                        "and asserts the criteria of the first run")
    a = p.parse_args(argv)
    if a.dump and not a.checkpoint:
        p.error("--dump needs --checkpoint")
    return a


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_job:
        py = "<conda>/envs/mdfm/python.exe"
        print(" ".join([py] + [f'"{x}"' if " " in x else x for x in job_command()]))
        return 0
    if a.dump:
        dump(a)
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the analysis runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    from mdfm.gates import common
    o = os.path.abspath(a.results_out) if a.results_out else OUT
    paths.assert_not_moveatlas(o)
    sets, locked = get_sets()
    stamp = GG().stamp_moveatlas()
    res = run_analysis(sets, locked, stamp=stamp)
    if a.results_out:
        # a rerun: the criteria must be those of the first run, and the prediction text is hashed with the reading
        assert_same_criteria(res)
        res["rerun"] = {"first_run": rel(os.path.join(OUT, "results.json")), "criteria_equal_to_first_run": True,
                        "prediction_section_sha256": prediction_section_sha256(), "results_folder": rel(o)}
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    if chr(0x2013) in text:
        raise AssertionError("summary.md holds an en-dash")
    common.write_text(os.path.join(o, "results.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(o, "summary.md"), text)
    prov = {"script": "scripts/f2_H3angles.py", "mode": "analysis", "label": LABEL, "run_at": res["run_started"],
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "numpy": np.__version__,
            "inputs": [v["events"] for v in res["part_a"].values()] + [rel(F1_RESULTS)],
            "code_sha256": common.code_sha256([os.path.abspath(__file__), GG().__file__]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(o, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    print(text)
    print(f"written {rel(o)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
