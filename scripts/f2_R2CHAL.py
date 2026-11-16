"""F2-R2CHAL: adversarial challenge of F2 round 2 (dev diagnostic, not a G3 read)

Task F2-R2CHAL (report reports/weekend_2026-10-01/tasks/F2-R2CHAL.md, whose section '## Prediction' was hashed into
reports/bench/f2_scale/R2CHAL/prediction.sha256 before any stride, cadence or speed value was computed; the hash is
asserted here before any value is read).

Questions (Prediction section):
- Q1 reproduction: H5's C1 and C2 points with least squares written here (not imported from f2_H5data.py), a spot
  check of H5data/amass_strides.csv against a fresh read of 12 takes of the train body store, and H6's early.json
  from the two probe warps.json files;
- Q2 the cadence spread of the real passes at fixed walkway stride and leg length;
- Q3 the tempo coefficient b_c of the decoded stride on natural walks, in sim (strides_insim.csv of the H3 dump,
  forward scored strides) and on real input (strides_real.csv, train54), one estimator for both;
- Q4 speed and cadence ranges of the populations;
- Q5 the numbers of the lever-3 draft (insim.json, the H4 dumps), the seed differences, and the ends' in-sim legs
  and root biases beside their dev12 decoded-over-walkway level.
Added after the hash (no criterion, labelled in the outputs): Q6, the floor of the legs readout on the true body
of the AMASS train walks (gait.stance_speed with the true contacts, integrated over true strides, over the true
stride), and the share of walkway stride rows that repeat one value per pass (the walkway stride definition).

Data: weargait_pd train54, dev12 (and dev14 rows of strides_real.csv dropped) controls of
reports/bench/g3_gain_check/results.json, through f2_H1shape.load_sets (SHA-256 asserted, lockbox asserted absent),
the person column of every file checked before a value column is read; AMASS train walks through
mdfm.data.body_store (split train, which refuses the lockbox); derived files of earlier tasks. Never an EVAL, CODY or
care_pd record. CPU only.

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
    <conda>/envs/mdfm/python.exe scripts/f2_R2CHAL.py
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # never the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "R2CHAL")
REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2-R2CHAL.md")
PREDICTION_SHA = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "6882d994dbbe01a38067e43c3d6311051c9daafd8d9fded90df45959e23be764"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-R2CHAL"

F1_DIR = os.path.join(TREE, "reports", "bench", "g3_gain_check")
F1_RESULTS = os.path.join(F1_DIR, "results.json")
H5_DIR = os.path.join(TREE, "reports", "bench", "f2_scale", "H5data")
H5_REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2-H5data.md")
H6_DIR = os.path.join(TREE, "reports", "bench", "f2_scale", "H6warp")
H3_DUMP = os.path.join(TREE, "reports", "bench", "f2_scale", "H3angles", "dump_phaseA_root_full_ckpt_B_032044")
H3_DUMPS = os.path.join(TREE, "reports", "bench", "f2_scale", "H3angles", "dumps_2026-10-05", "results.json")
H4_DUMPS = os.path.join(TREE, "reports", "bench", "f2_scale", "H4root", "dumps_2026-10-05", "results.json")
TAKES_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
RM_FILES = (os.path.join(TREE, "reports", "bench", "root_motion_phaseA_root_full_ckpt_B_032044_xsens13_warps.json"),
            os.path.join(TREE, "reports", "bench", "root_motion_phaseA_root_full_ckpt_B_032044_warp15_warps.json"))
LEVER2 = os.path.join(TREE, "reports", "bench", "f2_lever2", "results.json")
G3SIM = os.path.join(TREE, "reports", "gates", "g3_sim")
G3DIAG = os.path.join(TREE, "reports", "gates", "g3")

LEG_RATIO = 0.530
T_RANGE = (0.6, 2.0)
S_MIN = 0.2
RATE = 100.0
C_CENTRE = math.log(110.0)
BOOT_N = 2000
SEED = 20261006
LEVEL = 0.95
TOL = 1e-6
SIM_TEMPO_MIN = 0.15            # Q3 criterion
UNRESOLVED_WIDTH = 0.3
BETA_V = (0.4, 2.0)
QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)
# ends of Q5: name -> (in-sim folder, dev12 diag folder)
ENDS = {"A_010681": ("insim_phaseA_root_full_ckpt_A_010681", "diag_phaseA_root_full_ckpt_A_010681"),
        "v1_B": ("insim_phaseA_root_full_ckpt_B_032044", "diag_phaseA_root_full_ckpt_B_032044"),
        "s2_B": ("insim_pilot_v1_s2_ckpt_B_032044", "diag_pilot_v1_s2_ckpt_B_032044"),
        "f2l2_B": ("insim_pilot_f2l2_ckpt_B_032044", "diag_pilot_f2l2_ckpt_B_032044")}
EARLY = {"v1_B_004000": "insim_phaseA_root_full_ckpt_B_004000", "s2_B_004000": "insim_pilot_v1_s2_ckpt_B_004000",
         "f2l2_B_004000": "insim_pilot_f2l2_ckpt_B_004000"}
SPOT_TAKES = 6                  # per subset


class Refused(Exception):
    pass


# ---------------------------------------------------------------- helpers
def rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), TREE).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _sibling(name):
    key = f"_f2r2chal_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def own_section(text):
    """This task's hashed text: from the line '## Prediction' up to and including the newline before the next
    '## ' heading (so the blank line that closes the section is included: two final newlines). The convention of
    the hash recorded at 2026-10-06T16:29:03; the description line of prediction.sha256 first said 'one final
    newline', which did not describe the hashed bytes, and was corrected without touching the hash or its time."""
    t = text.replace("\r\n", "\n")
    i = t.index("## Prediction\n")
    j = t.index("\n## ", i + 5)
    return t[i:j + 1]


def prediction_section(text):
    """The H5 convention: heading line to the next '## ' heading, LF, one final newline."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        a = next(i for i, x in enumerate(lines) if x.strip() == "## Prediction")
    except StopIteration:
        return None
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip("\n") + "\n"


def section_sha(report):
    with open(report, encoding="utf-8") as f:
        sec = prediction_section(f.read())
    if sec is None:
        raise AssertionError(f"{report}: no Prediction section")
    return hashlib.sha256(sec.encode("utf-8")).hexdigest()


def check_prediction(report=REPORT, sha_file=PREDICTION_SHA, expected=PREDICTION_SHA256):
    with open(report, encoding="utf-8") as f:
        sha = hashlib.sha256(own_section(f.read()).encode("utf-8")).hexdigest()
    with open(sha_file, encoding="utf-8") as f:
        recorded = f.read().split()[0]
    if not (sha == recorded == expected):
        raise AssertionError(f"Prediction section {sha}, hash file {recorded}, expected {expected}")
    return sha


def interval(v, level=LEVEL):
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if not len(v):
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


def ols(X, y, w=None):
    """Least squares coefficients of y on the columns of X (weights w, optional), by the normal equations."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    if w is None:
        w = np.ones(len(y))
    w = np.asarray(w, np.float64)
    A = (X * w[:, None]).T @ X
    b = (X * w[:, None]).T @ y
    return np.linalg.solve(A, b)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------- persons and guarded readers
def load_sets():
    """(sets, locked): F1's sets through f2_H1shape.load_sets (SHA-256 and lists asserted), lockbox lists of
    mdfm.gates.g3 and of mdfm.data.window_store, asserted disjoint from every set."""
    from mdfm.data import window_store as WS
    sets, locked = _sibling("f2_H1shape").load_sets(F1_RESULTS)
    locked = set(locked) | set(WS.lockbox_persons())
    for name in ("dev14", "dev12", "train54"):
        if set(sets[name]) & locked:
            raise Refused(f"{name}: lockbox persons")
    return sets, locked


def guarded_csv(path, allowed, locked, person_col="person_id"):
    """The rows of allowed persons of a CSV, as strings: the person column is read first (Refused for a lockbox
    person or a non-weargait id), the other rows are dropped before any value column is converted."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=[person_col], dtype=str)
    ps = set(who[person_col].dropna())
    if ps & set(locked):
        raise Refused(f"{rel(path)}: lockbox persons in the file")
    if any(not str(p).startswith("weargait:") for p in ps):
        raise Refused(f"{rel(path)}: a person outside weargait_pd")
    d = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    d = d[d[person_col].isin(set(allowed))].reset_index(drop=True)
    if set(d[person_col]) - set(allowed):
        raise AssertionError("a row outside the allowed persons")
    return d


def event_strides(path, allowed, locked):
    """Matched base stride rows: person_id, trial_id, pass_id, foot, t0, t1, S (m), D (m), c, kept by H5's bounds."""
    import pandas as pd
    ev = guarded_csv(path, allowed, locked)
    ev = ev[(ev["row"] == "base") & (ev["kind"] == "stride") & (ev["matched"].str.strip().str.lower() == "true")]
    if (ev["unit"] != "cm").any():
        raise ValueError("stride rows not in cm")
    d = pd.DataFrame({"person_id": ev["person_id"].to_numpy(), "trial_id": ev["trial_id"].to_numpy(),
                      "pass_id": ev["pass_id"].to_numpy(), "foot": ev["foot"].to_numpy(),
                      "t0": ev["t0_s"].astype(float).to_numpy(), "t1": ev["t1_s"].astype(float).to_numpy(),
                      "S": ev["reference"].astype(float).to_numpy() / 100.0,
                      "D": ev["prediction"].astype(float).to_numpy() / 100.0})
    T = d["t1"] - d["t0"]
    ok = (np.isfinite(d[["t0", "t1", "S", "D"]].to_numpy()).all(1) & (T >= T_RANGE[0]) & (T <= T_RANGE[1])
          & (d["S"] >= S_MIN) & (d["D"] > 0)).to_numpy()
    n_read = len(d)
    d = d[ok].reset_index(drop=True)
    d["c"] = 120.0 / (d["t1"] - d["t0"])
    d.attrs["counts"] = {"read": int(n_read), "kept": int(ok.sum())}
    return d


def heights(persons, locked):
    return _sibling("f2_H1shape").read_demographics(persons, locked)


# ---------------------------------------------------------------- Q1: H5 reproduction
def passes(d):
    g = d.groupby(["person_id", "trial_id", "pass_id"], sort=True)
    return g.agg(S=("S", "mean"), D=("D", "mean"), c=("c", "mean"), l=("l", "first")).reset_index()


def c2_points(p):
    X = np.stack([np.ones(len(p)), np.log(p["S"]), np.log(p["c"]), np.log(p["l"])], 1)
    b = ols(X, np.log(p["D"].to_numpy()))
    return {"intercept": b[0], "b_S": b[1], "b_c": b[2], "b_l": b[3]}


def relation(df):
    """alpha, beta of log(S / l) on (log c - log 110), persons weighted equally (weight 1 / strides of the person)."""
    n = df.groupby("person_id")["S"].transform("size").to_numpy(np.float64)
    X = np.stack([np.ones(len(df)), np.log(df["c"].to_numpy()) - C_CENTRE], 1)
    b = ols(X, np.log(df["S"].to_numpy() / df["l"].to_numpy()), 1.0 / n)
    return float(b[0]), float(b[1])


def med_of_person_medians(df, col):
    return float(df.groupby("person_id")[col].median().median())


def c1_points(real, amass_rel, g_sim):
    a, b = amass_rel
    r = real.copy()
    r["logDS"] = np.log(r["D"] / r["S"])
    r["logAS"] = np.log(r["l"]) + a + b * (np.log(r["c"]) - C_CENTRE) - np.log(r["S"])
    L_obs = med_of_person_medians(r, "logDS") - math.log(g_sim)
    L_H5 = med_of_person_medians(r, "logAS")
    return {"L_obs": L_obs, "L_H5": L_H5, "h": L_H5 / L_obs,
            "decoded_over_walkway": math.exp(med_of_person_medians(r, "logDS")),
            "amass_predicted_over_walkway": math.exp(L_H5)}


def amass_table(path=os.path.join(H5_DIR, "amass_strides.csv")):
    import pandas as pd
    from mdfm.data import amass_index
    d = pd.read_csv(path, encoding="utf-8")
    if set(d["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person among the AMASS strides")
    d["archive"] = d["take"].str.split("/").str[0]
    T = d["t1"] - d["t0"]
    ok = (d["scored"].astype(bool) & (T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (d["S"] >= S_MIN)).to_numpy()
    k = d[ok].copy().reset_index(drop=True)
    k["c"] = 120.0 / (k["t1"] - k["t0"])
    k["l"] = LEG_RATIO * k["stature_m"]
    return d, k


def compare(mine, theirs, tol=TOL):
    rows, ok = {}, True
    for key, v in mine.items():
        t = theirs.get(key)
        d = abs(v - t) if (t is not None and v is not None and np.isfinite(v)) else float("inf")
        rows[key] = {"mine": v, "file": t, "abs_diff": d, "equal": bool(d <= tol)}
        ok &= rows[key]["equal"]
    return ok, rows


def q1_h5(sets, locked, log=print):
    h5 = jload(os.path.join(H5_DIR, "results.json"))
    f1 = jload(F1_RESULTS)
    g_sim = float(f1["ends"]["v1_B"]["insim"]["stride_gain"])
    path = os.path.join(F1_DIR, "decode_phaseA_root_full_ckpt_B_032044", "train_events.csv")
    t54 = event_strides(path, sets["train54"], locked)
    hts = heights(sets["train54"] + sets["dev12"], locked)
    t54["l"] = [LEG_RATIO * hts.get(p, {}).get("height_m", float("nan")) for p in t54["person_id"]]
    t54 = t54[np.isfinite(t54["l"])].reset_index(drop=True)
    raw, am = amass_table()
    over = am[am["treadmill"] == 0]
    tread = am[am["treadmill"] == 1]
    rel_all, rel_over, rel_tread = relation(am), relation(over), relation(tread)
    p = passes(t54)
    c2 = c2_points(p)
    c1 = c1_points(t54, rel_all, g_sim)
    prim = h5["primary"]
    theirs = {"b_S": prim["C2"]["b_S"], "b_c": prim["C2"]["b_c"], "b_l": prim["C2"]["b_l"],
              "intercept": prim["C2"]["intercept"], "h": prim["C1"]["h"], "L_obs": prim["C1"]["L_obs"],
              "L_H5": prim["C1"]["L_H5"], "decoded_over_walkway": prim["C1"]["decoded_over_walkway"],
              "amass_predicted_over_walkway": prim["C1"]["amass_predicted_over_walkway"],
              "alpha_amass": h5["relations"]["AMASS train walks"]["coef"][0],
              "beta_amass": h5["relations"]["AMASS train walks"]["coef"][1],
              "alpha_over": h5["relations"]["AMASS overground"]["coef"][0],
              "beta_over": h5["relations"]["AMASS overground"]["coef"][1],
              "alpha_t54": h5["relations"]["weargait train54"]["coef"][0],
              "beta_t54": h5["relations"]["weargait train54"]["coef"][1]}
    rt = relation(t54)
    mine = {**c2, **c1, "alpha_amass": rel_all[0], "beta_amass": rel_all[1], "alpha_over": rel_over[0],
            "beta_over": rel_over[1], "alpha_t54": rt[0], "beta_t54": rt[1]}
    ok, rows = compare(mine, theirs)
    # the overground subset: who
    comp = {}
    for (arc, tm), g in am.groupby(["archive", "treadmill"]):
        comp[f"{arc} treadmill={int(tm)}"] = {"persons": int(g["person_id"].nunique()), "strides": int(len(g)),
                                             "takes": int(g["take"].nunique())}
    # treadmill-only relation, A/S of train54 against it (Q4 item)
    c1_tread = c1_points(t54, rel_tread, g_sim)
    c1_over = c1_points(t54, rel_over, g_sim)
    log(f"Q1 H5 points reproduced: {ok}")
    return {"reproduced": ok, "rows": rows, "n_passes": int(len(p)), "n_strides": int(len(t54)),
            "n_persons": int(t54["person_id"].nunique()), "counts": t54.attrs.get("counts"),
            "amass_composition_scored_in_bounds": comp, "relation_treadmill": list(rel_tread),
            "C1_against_treadmill_only": c1_tread, "C1_against_overground_only": c1_over}, t54, am, hts


def spot_check_amass(raw_path=os.path.join(H5_DIR, "amass_strides.csv"), per_subset=SPOT_TAKES, log=print):
    """Fresh read of per_subset KIT and per_subset BMLrub takes (evenly spaced in takes.csv order) from the train
    body store with the readouts of H5 (truth_on_clock, initial_contacts, strides, passes, steady_straight); their
    t0, t1, S and scored flag against amass_strides.csv."""
    import csv
    import pandas as pd
    import torch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index, body_store
    from mdfm.eval import infer as I
    from mdfm.readouts import gait
    ins = _sibling("eval_gait_insim")
    with open(TAKES_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for s in sorted({r["subset"] for r in rows}):
        if amass_index.check(s) != "train":
            raise Refused(f"{s} is not a train archive")
    pick = []
    for s in ("KIT", "BMLrub"):
        tk = [r["take"] for r in rows if r["subset"] == s]
        idx = np.linspace(0, len(tk) - 1, per_subset).round().astype(int)
        pick += [tk[i] for i in idx]
    cached = pd.read_csv(raw_path, encoding="utf-8")
    store = body_store.BodyStore(split="train")
    tk = store.takes
    if set(tk["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person in the train body store")
    pos = {t: i for i, t in enumerate(tk["take_id"])}
    bank = SkeletonBank(torch.float64)
    out, worst, n = [], 0.0, 0
    for t in pick:
        i = pos[t]
        r = tk.iloc[i]
        I.refuse_cody(str(r["archive"]), str(t), str(r["person_id"]), str(r.get("subject_ns", "")))
        nk = int(r["n_knots"])
        bodies = store.windows([i], [0], win_s=nk / RATE)
        J, _, contact = ins.truth_on_clock(bodies, bank, np.arange(nk, dtype=np.float64) / RATE)
        ics = gait.initial_contacts(contact.astype(np.float64), RATE)
        st = gait.strides(J, ics, rate=RATE)
        c = cached[cached["take"] == t].reset_index(drop=True)
        if len(st):
            pel = J[:, gait.PELVIS][:, [0, 2]]
            ps = ins.passes_from_ics(ics, pel)
            ins.steady_straight(st, ps, pel)
        same_n = len(st) == len(c)
        d = float("inf")
        if same_n and len(st):
            d = float(max(np.abs(st["length_m"].to_numpy() - c["S"].to_numpy()).max(),
                          np.abs(st["t0"].to_numpy() - c["t0"].to_numpy()).max(),
                          np.abs(st["t1"].to_numpy() - c["t1"].to_numpy()).max()))
            same_n = same_n and bool((st["scored"].astype(bool).to_numpy() == c["scored"].astype(bool).to_numpy()).all())
        elif same_n:
            d = 0.0
        out.append({"take": t, "strides_fresh": int(len(st)), "strides_cached": int(len(c)), "max_abs_diff": d,
                    "scored_equal": bool(same_n)})
        worst = max(worst, d)
        n += len(st)
    # the cached file is written with 6 decimals: the comparison tolerance is half a unit of the 6th decimal plus TOL
    tol = 5e-7 + TOL
    ok = all(o["scored_equal"] and o["max_abs_diff"] <= tol for o in out)
    log(f"Q1 AMASS spot check: {ok} (worst {worst:.2e} over {n} strides)")
    return {"reproduced": ok, "tolerance": tol, "takes": out, "strides": n}


def q1_early(log=print):
    """early.json recomputed: G, r, e, f*, P (linear in ln f, ln r, nearest segment outside), share."""
    import pandas as pd
    e = jload(os.path.join(H6_DIR, "early.json"))
    f1 = jload(F1_RESULTS)["ends"]["v1_B"]
    Q = float(f1["gain"]["stride"]["gain"]) / float(f1["insim"]["stride_gain"])
    rho = float(jload(H3_DUMPS)["part_b"]["descriptive_dumps_2026_10_05"]
                ["real_stride_over_insim_stride_at_equal_speed"]["median_of_person_medians"])
    t = pd.read_csv(TAKES_CSV, encoding="utf-8")
    v, c = t["speed_m_s"].astype(float).to_numpy(), t["cadence_steps_min"].astype(float).to_numpy()
    ok = np.isfinite(v) & np.isfinite(c) & (c > 0) & (v >= BETA_V[0]) & (v <= BETA_V[1])
    x, y = np.log(c[ok]), np.log(120.0 * v[ok] / c[ok])
    beta = float(np.polyfit(x, y, 1)[0])
    pts = {}
    for path in RM_FILES:
        rows = jload(path)["rows"]
        g = {float(r["factor"]): 1 + float(r["stride_bias_cm"]) / (100 * float(r["true_stride_m"])) for r in rows}
        for f, val in g.items():
            if f != 1.0 and f not in pts:
                pts[f] = val / g[1.0]
    fs = rho ** -(1 + 1 / beta)
    fac = sorted(pts)
    lo = min(fac)
    # f* below every factor: the segment [lowest factor, 1]
    if fs < lo:
        P = math.exp(math.log(pts[lo]) / math.log(lo) * math.log(fs))
    else:
        P = float("nan")
    share = math.log(P) / math.log(Q)
    mine = {"Q": Q, "rho": rho, "beta": beta, "f_star": fs, "P": P, "share": share,
            **{f"r_{f:g}": pts[f] for f in fac}}
    theirs = {"Q": e["Q"], "rho": e["rho"], "beta": e["beta_train_walks"]["beta"], "f_star": e["f_star"],
              "P": e["P"], "share": e["share"], **{f"r_{k}": v for k, v in e["r"].items() if k != "1"}}
    okc, rows = compare(mine, theirs, 1e-9)
    log(f"Q1 early.json reproduced: {okc}")
    return {"reproduced": okc, "rows": rows, "n_takes_beta": int(ok.sum())}


# ---------------------------------------------------------------- Q2: real cadence spread at fixed stride
def q2(t54, dev12):
    out = {}
    for name, d in (("train54", t54), ("dev12", dev12)):
        p = passes(d)
        X = np.stack([np.ones(len(p)), np.log(p["S"]), np.log(p["l"])], 1)
        y = np.log(p["c"].to_numpy())
        b = ols(X, y)
        res = y - X @ b
        out[name] = {"passes": int(len(p)), "sd_log_c_residual": float(res.std(ddof=X.shape[1])),
                     "q05_q95_factor": [float(math.exp(np.quantile(res, 0.05))),
                                        float(math.exp(np.quantile(res, 0.95)))],
                     "min_max_factor": [float(math.exp(res.min())), float(math.exp(res.max()))],
                     "sd_log_c_total": float(np.log(p["c"]).std(ddof=1)),
                     "share_of_cadence_variance_at_fixed_stride": float(res.var() / np.log(p["c"]).var())}
    return out


# ---------------------------------------------------------------- Q3: tempo coefficient on natural walks
def tempo_fit(df, y_col, s_col, l_col, persons_col="person_id", n_boot=BOOT_N, seed=SEED):
    d = df[np.isfinite(df[[y_col, s_col, l_col, "c"]].to_numpy()).all(1) & (df[y_col] > 0)].reset_index(drop=True)
    X = np.stack([np.ones(len(d)), np.log(d[s_col]), np.log(d["c"]), np.log(d[l_col])], 1)
    y = np.log(d[y_col].to_numpy())
    b = ols(X, y)
    persons = sorted(d[persons_col].unique())
    rows = {p: np.flatnonzero(d[persons_col].to_numpy() == p) for p in persons}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(len(persons), size=len(persons))
        idx = np.concatenate([rows[persons[i]] for i in pick])
        try:
            bs.append(ols(X[idx], y[idx]))
        except np.linalg.LinAlgError:
            continue
    bs = np.asarray(bs)
    # how much tempo the strides span at fixed stride and leg
    Xc = np.stack([np.ones(len(d)), np.log(d[s_col]), np.log(d[l_col])], 1)
    yc = np.log(d["c"].to_numpy())
    rc = yc - Xc @ ols(Xc, yc)
    return {"strides": int(len(d)), "persons": len(persons), "b_S": float(b[1]), "b_S_interval": interval(bs[:, 1]),
            "b_c": float(b[2]), "b_c_interval": interval(bs[:, 2]), "b_l": float(b[3]),
            "b_l_interval": interval(bs[:, 3]), "sd_log_c_at_fixed_stride": float(rc.std(ddof=3)),
            "q05_q95_tempo_factor": [float(math.exp(np.quantile(rc, 0.05))), float(math.exp(np.quantile(rc, 0.95)))]}


def q3_class(fit):
    lo, hi = fit["b_c_interval"]
    b = fit["b_c"]
    if hi - lo > UNRESOLVED_WIDTH:
        return "UNRESOLVED"
    if b >= SIM_TEMPO_MIN and lo > 0:
        return "SIM TEMPO"
    if hi < SIM_TEMPO_MIN:
        return "NO SIM TEMPO"
    return "UNRESOLVED"


def q3(sets, locked, hts, log=print):
    import pandas as pd
    sim = pd.read_csv(os.path.join(H3_DUMP, "strides_insim.csv"), encoding="utf-8")
    if any(not str(p).startswith("amass:person:") for p in sim["person_id"].unique()):
        raise Refused("strides_insim.csv: a person outside AMASS")
    sim = sim[(sim["forward"].astype(str) == "True") & (sim["scored"].astype(str) == "True")].reset_index(drop=True)
    T = sim["t1"] - sim["t0"]
    sim = sim[(T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (sim["true_stride_m"] >= S_MIN)].reset_index(drop=True)
    sim["c"] = 120.0 / (sim["t1"] - sim["t0"])
    res = {"sim_forward_scored": {}}
    for y in ("dec_root_m", "legs_m"):
        f = tempo_fit(sim, y, "true_stride_m", "true_leg_length_m")
        f["class"] = q3_class(f)
        res["sim_forward_scored"][y] = f
    res["sim_medians_over_true_stride"] = {y: float(np.nanmedian(sim[y] / sim["true_stride_m"]))
                                           for y in ("dec_root_m", "legs_m", "true_root_m", "true_ankle_m")}
    real = guarded_csv(os.path.join(H3_DUMP, "strides_real.csv"), sets["train54"], locked)
    real = real[real["set"] == "train54"].reset_index(drop=True)
    for col in ("t0", "t1", "walkway_m", "legs_m", "dec_root_m"):
        real[col] = pd.to_numeric(real[col], errors="coerce")
    T = real["t1"] - real["t0"]
    real = real[(T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (real["walkway_m"] >= S_MIN)].reset_index(drop=True)
    real["c"] = 120.0 / (real["t1"] - real["t0"])
    real["l"] = [LEG_RATIO * hts.get(p, {}).get("height_m", float("nan")) for p in real["person_id"]]
    res["real_train54"] = {y: tempo_fit(real, y, "walkway_m", "l") for y in ("dec_root_m", "legs_m")}
    res["label"] = res["sim_forward_scored"]["dec_root_m"]["class"]
    log(f"Q3 sim natural-walk tempo: {res['label']} (b_c {res['sim_forward_scored']['dec_root_m']['b_c']:.3f})")
    return res


# ---------------------------------------------------------------- Q4: ranges
def q4(t54, dev12, am):
    import pandas as pd
    def qs(x):
        return [float(np.quantile(x, q)) for q in QUANTS]
    sim = pd.read_csv(os.path.join(H3_DUMP, "strides_insim.csv"), encoding="utf-8")
    sim = sim[(sim["forward"].astype(str) == "True") & (sim["scored"].astype(str) == "True")]
    pops = {"train54": t54, "dev12": dev12, "AMASS pooled": am, "AMASS overground": am[am["treadmill"] == 0],
            "AMASS treadmill": am[am["treadmill"] == 1]}
    out = {k: {"strides": int(len(d)), "speed_m_s": qs(d["S"] * d["c"] / 120.0), "cadence": qs(d["c"]),
               "stride_m": qs(d["S"])} for k, d in pops.items()}
    c = 120.0 / (sim["t1"] - sim["t0"])
    out["in-sim dev forward scored"] = {"strides": int(len(sim)), "speed_m_s": qs(sim["true_speed_m_s"]),
                                        "cadence": qs(c), "stride_m": qs(sim["true_stride_m"])}
    return {"quantiles": list(QUANTS), "populations": out}


# ---------------------------------------------------------------- Q5: lever-3 numbers
def insim_line(folder):
    p = os.path.join(G3SIM, folder, "insim.json")
    if not os.path.exists(p):
        return None
    d = jload(p)
    fw = d["lines"]["forward"]
    t = float(d["counts"]["forward"]["true_stride_m"])
    root, legs = float(fw["model"]["stride_bias_cm"]["point"]), float(fw["model_legs"]["stride_bias_cm"]["point"])
    return {"path": rel(p), "true_stride_m": t, "root_bias_cm": root, "legs_bias_cm": legs,
            "slide_cm": legs - root, "g_root": 1 + root / (100 * t), "g_legs": 1 + legs / (100 * t),
            "root_over_legs": (1 + root / (100 * t)) / (1 + legs / (100 * t))}


def q5(sets, locked):
    out = {"ends": {}, "early": {}}
    for name, (sim_f, diag_f) in ENDS.items():
        row = insim_line(sim_f) or {}
        ev = os.path.join(G3DIAG, diag_f, "events.csv")
        if os.path.exists(ev):
            d = event_strides(ev, sets["dev12"], locked)
            d["r"] = d["D"] / d["S"]
            d["lr"] = np.log(d["r"])
            row["dev12_decoded_over_walkway_pooled_median"] = float(np.median(d["r"]))
            row["dev12_decoded_over_walkway_person_medians"] = math.exp(med_of_person_medians(d, "lr"))
            row["dev12_strides"] = int(len(d))
        out["ends"][name] = row
    for name, f in EARLY.items():
        out["early"][name] = insim_line(f)
    names = [n for n in ENDS if "dev12_decoded_over_walkway_pooled_median" in out["ends"][n]]
    real = np.array([out["ends"][n]["dev12_decoded_over_walkway_pooled_median"] for n in names])
    cross = {"ends": names, "what": "descriptive, 4 ends, not independent: Pearson r and rank order of the dev12 "
                                    "decoded-over-walkway level against the in-sim gains"}
    for key in ("g_root", "g_legs"):
        x = np.array([out["ends"][n][key] for n in names])
        cross[f"pearson_real_vs_{key}"] = float(np.corrcoef(x, real)[0, 1])
        cross[f"rank_{key}"] = [names[i] for i in np.argsort(-x)]
    cross["rank_real"] = [names[i] for i in np.argsort(-real)]
    b_ends = [n for n in names if n.endswith("_B")]
    rb = np.array([out["ends"][n]["dev12_decoded_over_walkway_pooled_median"] for n in b_ends])
    for key in ("g_root", "g_legs"):
        x = np.array([out["ends"][n][key] for n in b_ends])
        cross[f"pearson_B_ends_real_vs_{key}"] = float(np.corrcoef(x, rb)[0, 1])
    out["cross_end"] = cross
    v1, s2 = out["ends"]["v1_B"], out["ends"]["s2_B"]
    out["d_root_cm"] = abs(v1["root_bias_cm"] - s2["root_bias_cm"])
    out["d_legs_cm"] = abs(v1["legs_bias_cm"] - s2["legs_bias_cm"])
    h4 = jload(H4_DUMPS)["ends"]["v1_B"]["dump"]
    t = h4["test_a_train54"]
    out["h4_train54"] = {k: t[k] for k in ("g_root_real", "g_legs_real", "g_root_sim", "g_legs_sim",
                                           "root_over_legs_real", "root_over_legs_sim", "D_root", "D_legs")}
    out["h4_ceiling"] = h4["root_specific_ceiling_train54"]
    out["ceiling_recomputed"] = t["g_root_sim"] * math.exp(t["D_legs"])
    return out


# ---------------------------------------------------------------- Q6 (added after the hash): legs readout floor
def q6_floor(log=print):
    """The legs readout (gait.stance_speed of the pelvis-relative true joints with the TRUE contacts, mean over a true
    stride times its duration) over the true stride, on the scored strides of the AMASS train walks of takes.csv."""
    import csv
    import torch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index, body_store
    from mdfm.eval import infer as I
    from mdfm.readouts import gait
    ins = _sibling("eval_gait_insim")
    with open(TAKES_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for s in sorted({r["subset"] for r in rows}):
        if amass_index.check(s) != "train":
            raise Refused(f"{s} is not a train archive")
    store = body_store.BodyStore(split="train")
    tk = store.takes
    if set(tk["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person in the train body store")
    pos = {t: i for i, t in enumerate(tk["take_id"])}
    bank = SkeletonBank(torch.float64)
    per = {0: [], 1: []}
    persons = {0: {}, 1: {}}
    for n_done, r0 in enumerate(rows):
        t = r0["take"]
        if t not in pos:
            continue
        i = pos[t]
        r = tk.iloc[i]
        I.refuse_cody(str(r["archive"]), str(t), str(r["person_id"]), str(r.get("subject_ns", "")))
        nk = int(r["n_knots"])
        bodies = store.windows([i], [0], win_s=nk / RATE)
        J, _, contact = ins.truth_on_clock(bodies, bank, np.arange(nk, dtype=np.float64) / RATE)
        ics = gait.initial_contacts(contact.astype(np.float64), RATE)
        st = gait.strides(J, ics, rate=RATE)
        if not len(st):
            continue
        pel = J[:, gait.PELVIS][:, [0, 2]]
        ps = ins.passes_from_ics(ics, pel)
        ins.steady_straight(st, ps, pel)
        sp = gait.stance_speed(J - J[:, :1], contact.astype(np.float64), RATE)
        tm = int(r["treadmill"])
        for s in st.itertuples():
            T = s.t1 - s.t0
            if not (bool(s.scored) and T_RANGE[0] <= T <= T_RANGE[1] and s.length_m >= S_MIN):
                continue
            legs = ins.interval_mean(sp, s.t0, s.t1, RATE) * T
            if np.isfinite(legs):
                ratio = legs / s.length_m
                per[tm].append(ratio)
                persons[tm].setdefault(str(r["person_id"]), []).append(ratio)
        if (n_done + 1) % 100 == 0:
            log(f"  Q6 {n_done + 1} of {len(rows)} takes")
    out = {}
    for tm, name in ((0, "overground"), (1, "treadmill")):
        v = np.asarray(per[tm])
        pm = [float(np.median(x)) for x in persons[tm].values()]
        out[name] = {"strides": int(len(v)), "persons": len(pm), "median": float(np.median(v)) if len(v) else None,
                     "median_of_person_medians": float(np.median(pm)) if pm else None,
                     "q25_q75": [float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))] if len(v) else None}
    log(f"Q6 legs floor on the truth: {out}")
    return out


def insim_readouts():
    """Added after the hash: the forward scored strides of strides_insim.csv (true ICs): per readout the median and
    the mean of readout over true stride and the person-weighted mean error (cm); and the legs lines of the
    probe_root_motion run of v1_B (decoded legs over true speed and the same estimator on the truth)."""
    import pandas as pd
    s = pd.read_csv(os.path.join(H3_DUMP, "strides_insim.csv"), encoding="utf-8")
    s = s[(s["forward"].astype(str) == "True") & (s["scored"].astype(str) == "True")]
    out = {"strides": int(len(s))}
    for c in ("legs_m", "dec_root_m", "dec_ankle_m", "true_root_m"):
        r = s[c] / s["true_stride_m"]
        e = 100.0 * (s[c] - s["true_stride_m"])
        out[c] = {"median_over_true": float(np.nanmedian(r)), "mean_over_true": float(np.nanmean(r)),
                  "q10_q90_over_true": [float(np.nanquantile(r, 0.1)), float(np.nanquantile(r, 0.9))],
                  "person_weighted_mean_error_cm": float(e.groupby(s["person_id"]).mean().mean()),
                  "missing": int(s[c].isna().sum())}
    rm = jload(RM_FILES[0].replace("_warps.json", ".json"))["lines"]["legs"]
    out["probe_root_motion_xsens13_legs"] = rm
    return out


def walkway_repeat(sets, locked):
    """Share of walkway passes (train54, v1_B F1 events) whose matched base stride rows carry one reference value."""
    path = os.path.join(F1_DIR, "decode_phaseA_root_full_ckpt_B_032044", "train_events.csv")
    ev = guarded_csv(path, sets["train54"], locked)
    ev = ev[(ev["row"] == "base") & (ev["kind"] == "stride")]
    g = ev.groupby(["trial_id", "pass_id"])["reference"].nunique()
    n = ev.groupby(["trial_id", "pass_id"]).size()
    multi = n[n > 1].index
    return {"passes_with_2_or_more_strides": int(len(multi)),
            "share_with_one_reference_value": float((g.loc[multi] == 1).mean()) if len(multi) else None}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-store", action="store_true", help="skip the body-store parts (spot check, Q6)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    pred = check_prediction()                           # before any value is read
    hashes = {"H5": {"section": section_sha(H5_REPORT),
                     "recorded": open(os.path.join(H5_DIR, "prediction.sha256"), encoding="utf-8").read().split()[0]},
              "H6": {"file": sha256_file(os.path.join(H6_DIR, "prediction.txt")),
                     "recorded": open(os.path.join(H6_DIR, "prediction.sha256"), encoding="utf-8").read().split()[0]},
              "H6_amendment_1": {"file": sha256_file(os.path.join(H6_DIR, "prediction_amendment_1.txt")),
                                 "recorded": open(os.path.join(H6_DIR, "prediction_amendment_1.sha256"),
                                                  encoding="utf-8").read().split()[0]}}
    for v in hashes.values():
        v["equal"] = len(set(x for k, x in v.items() if k != "equal")) == 1
    sets, locked = load_sets()
    res = {"what": LABEL, "task": TASK, "prediction_sha256": pred, "run_started": started, "hashes_round2": hashes}
    r1, t54, am, hts = q1_h5(sets, locked)
    res["Q1_H5"] = r1
    dev12 = event_strides(os.path.join(G3DIAG, "diag_phaseA_root_full_ckpt_B_032044", "events.csv"),
                          sets["dev12"], locked)
    dev12["l"] = [LEG_RATIO * hts.get(p, {}).get("height_m", float("nan")) for p in dev12["person_id"]]
    dev12 = dev12[np.isfinite(dev12["l"])].reset_index(drop=True)
    res["Q1_early"] = q1_early()
    res["Q2"] = q2(t54, dev12)
    res["Q3"] = q3(sets, locked, hts)
    res["Q4"] = q4(t54, dev12, am)
    res["Q4"]["C1_against_treadmill_only"] = r1["C1_against_treadmill_only"]
    res["Q5"] = q5(sets, locked)
    res["added_after_hash"] = {"walkway_reference_per_pass": walkway_repeat(sets, locked),
                               "insim_readouts_true_ics": insim_readouts()}
    if not a.no_store:
        res["Q1_spot"] = spot_check_amass()
        fl = q6_floor()
        res["added_after_hash"]["Q6_legs_floor_truth"] = fl
        # the in-sim legs-minus-root difference of v1_B net of the legs readout's floor on the truth, if the floor of
        # the train walks (or of the probe's dev windows) holds on the forward dev strides
        v1 = res["Q5"]["ends"]["v1_B"]
        t = v1["true_stride_m"]
        floors = {"train_overground_median": fl["overground"]["median"],
                  "probe_dev_truth_median": res["added_after_hash"]["insim_readouts_true_ics"]
                  ["probe_root_motion_xsens13_legs"]["truth_legs_over_true_median"]}
        res["added_after_hash"]["v1_B_slide_net_of_floor"] = {
            k: {"floor": f, "floor_cm": 100 * t * (1 - f), "legs_bias_net_cm": v1["legs_bias_cm"] + 100 * t * (1 - f),
                "slide_net_cm": v1["slide_cm"] + 100 * t * (1 - f),
                "share_of_slide_that_is_floor": 100 * t * (1 - f) / -v1["slide_cm"]}
            for k, f in floors.items()}
    res["sets"] = {"lockbox_persons_in_sets": 0}
    from mdfm.data import sources
    res["sets"]["dev12_sha256"] = sources.person_hash(sets["dev12"])
    res["sets"]["train54_sha256"] = sources.person_hash(sets["train54"])
    stamp = _sibling("g3_gain_check").stamp_moveatlas()
    res["moveatlas_head"] = stamp.get("head") if isinstance(stamp, dict) else None
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(res), f, indent=1, ensure_ascii=False)
        f.write("\n")
    prov = {"script": rel(__file__), "script_sha256": sha256_file(__file__), "run_started": started,
            "moveatlas": stamp,
            "inputs_sha256": {rel(x): sha256_file(x) for x in
                              [F1_RESULTS, os.path.join(H5_DIR, "results.json"), os.path.join(H5_DIR, "amass_strides.csv"),
                               os.path.join(H6_DIR, "early.json"), H3_DUMPS, H4_DUMPS, TAKES_CSV, *RM_FILES,
                               RM_FILES[0].replace("_warps.json", ".json"),
                               os.path.join(H3_DUMP, "strides_insim.csv"), os.path.join(H3_DUMP, "strides_real.csv"),
                               os.path.join(F1_DIR, "decode_phaseA_root_full_ckpt_B_032044", "train_events.csv"),
                               REPORT, PREDICTION_SHA]
                              + [os.path.join(G3DIAG, d, "events.csv") for _, d in ENDS.values()]
                              + [os.path.join(G3SIM, s, "insim.json") for s, _ in ENDS.values()]}}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(prov), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(clean({k: res[k] for k in ("Q2",)}), indent=1))
    print(f"wrote {rel(os.path.join(a.out, 'results.json'))} in {res['seconds']:.1f} s")
    return res


if __name__ == "__main__":
    main()
