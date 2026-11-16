"""F2-H5data: is the decoded stride the AMASS stride at the walker's cadence? (dev diagnostic, not a G3 read)

Task F2-H5data (report reports/weekend_2026-10-01/tasks/F2-H5data.md, whose section '## Prediction' was hashed into
reports/bench/f2_scale/H5data/prediction.sha256 before any stride, cadence or speed value was read; the hash is
asserted here before any value is read).

Hypothesis H5: the model reads the cadence and infers the stride length from it through the stride-cadence relation
of its AMASS training walks, so real walkers who take longer strides at a given cadence are decoded short.

Inputs:
- sets: dev12 and train54 of reports/bench/g3_gain_check/results.json (scripts/f2_H1shape.py load_sets: SHA-256 and
  lists asserted, no person of the weargait_pd lockbox list of mdfm.gates.g3, nor of mdfm.data.window_store);
- real strides: the matched stride rows (row base) of the F1 decodes (train54) and of the G3 diag runs (dev), of the
  ends v1_B and s2_B; the person column of a file is read and guarded first, then only the rows of allowed persons
  are kept before a value column is converted;
- heights: the weargait_pd demographic tables through f2_H1shape.read_demographics (guarded, row by row);
- AMASS: the train walks of reports/amass_walks/takes.csv from the train body store (mdfm.data.body_store, which
  refuses the lockbox), true joints and contacts by scripts/eval_gait_insim.py truth_on_clock, strides by
  mdfm.readouts.gait, scored strides only (eval_gait_insim passes_from_ics and steady_straight); stature of each
  take's rest mesh by f2_H1shape.stature_of. Cached in <out>/amass_strides.csv (reused with --reuse-amass).

Quantities (Prediction section): the AMASS relation log(S / l) = alpha + beta log c (persons weighted equally), the
level share h (C1), the pass slope b_S of log D on (1, log S, log c, log l) (C2), the residual ratio R (C3), and the
reading; beside: dev12, s2_B, without weargait:HC134, prior-stature leg, quadratic AMASS relation, overground AMASS
takes only, real strides inside the AMASS cadence range.

Writes reports/bench/f2_scale/H5data/: results.json, summary.md, persons.csv, amass_strides.csv, _provenance.json.

Run (mdfm environment, CPU):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
    <conda>/envs/mdfm/python.exe scripts/f2_H5data.py
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
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm.data.canonical import Refused  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H5data")
REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2-H5data.md")
PREDICTION_SHA = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "a4123d10dadc75afe3a0dbfb201dd79f6f53c5bfd7230c6ee779a3d145c5e236"
F1_DIR = os.path.join(TREE, "reports", "bench", "g3_gain_check")
F1_RESULTS = os.path.join(F1_DIR, "results.json")
H1_RESULTS = os.path.join(TREE, "reports", "bench", "f2_scale", "H1shape", "results.json")
TAKES_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-H5data"
ENDS = {"v1_B": "phaseA_root_full_ckpt_B_032044", "s2_B": "pilot_v1_s2_ckpt_B_032044"}
PRIMARY_END, PRIMARY_SET = "v1_B", "train54"
SETS = ("train54", "dev12")
FLAGGED = "weargait:HC134"
LEG_RATIO = 0.530               # greater trochanter height over stature (Winter 2009, Fig. 4.1; f2_H1shape)
T_RANGE = (0.6, 2.0)            # stride time bounds, s
S_MIN = 0.2                     # stride length lower bound, m
RATE = 100.0
BOOT_N = 10000
SEED = 20261006
LEVEL = 0.95
C_CENTRE = math.log(110.0)      # centring of log cadence in the quadratic relation
CAD_RANGE_Q = (0.025, 0.975)
# criterion of the Prediction section
READ_POINT, READ_LOWER = 0.6, 0.3
CAD_POINT, CAD_UPPER = 0.3, 0.6
UNRESOLVED_WIDTH = 0.6
H_POINT, H_LOWER = 0.5, 0.25
H_NOT_UPPER = 0.25


# ---------------------------------------------------------------- helpers
def rel(path):
    try:
        r = os.path.relpath(os.path.abspath(path), TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def _sibling(name):
    key = f"_f2h5_{name}"
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


def prediction_section(text):
    """Text of the section '## Prediction' (heading line to the next '## ' heading), LF, one final newline; None
    without one."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        a = next(i for i, x in enumerate(lines) if x.strip() == "## Prediction")
    except StopIteration:
        return None
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip("\n") + "\n"


def check_prediction(report=None, sha_file=None, expected=None):
    """The SHA-256 of the report's Prediction section, asserted equal to the first word of the hash file and to the
    script constant."""
    expected = PREDICTION_SHA256 if expected is None else expected
    with open(report or REPORT, encoding="utf-8") as f:
        sec = prediction_section(f.read())
    if sec is None:
        raise AssertionError("no Prediction section")
    sha = hashlib.sha256(sec.encode("utf-8")).hexdigest()
    with open(sha_file or PREDICTION_SHA, encoding="utf-8") as f:
        recorded = f.read().split()[0]
    if not (sha == recorded == expected):
        raise AssertionError(f"Prediction section {sha}, hash file {recorded}, expected {expected}")
    return sha


def interval(values, level=LEVEL):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


# ---------------------------------------------------------------- real strides
def load_sets():
    """(sets {dev14, dev12, train54}, locked): f2_H1shape.load_sets (F1's sets, SHA-256 and lists asserted, no
    mdfm.gates.g3 lockbox person), and no person of the window store's lockbox list either."""
    from mdfm.data import window_store as WS
    sets, locked = _sibling("f2_H1shape").load_sets(F1_RESULTS)
    locked = set(locked) | set(WS.lockbox_persons())
    for name in ("dev14", "dev12", "train54"):
        if set(sets[name]) & locked:
            raise Refused(f"{name}: lockbox persons")
    return sets, locked


def read_strides(path, allowed, locked, row="base"):
    """Matched stride rows of an events.csv as a DataFrame person_id, trial_id, task, foot, pass_id, t0, t1, S, D
    (m). The person column is read and checked first (Refused for a lockbox person); the rows of persons outside
    `allowed` are dropped before any value column is converted."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    ps = set(who["person_id"].dropna())
    if ps & set(locked):
        raise Refused(f"{rel(path)}: lockbox persons in the file")
    ev = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    ev = ev[ev["person_id"].isin(set(allowed))]
    ev = ev[(ev["row"] == row) & (ev["kind"] == "stride") & (ev["matched"].str.strip().str.lower() == "true")]
    if (ev["unit"] != "cm").any():
        raise ValueError(f"{rel(path)}: stride rows not in cm")
    d = pd.DataFrame({"person_id": ev["person_id"].to_numpy(), "trial_id": ev["trial_id"].to_numpy(),
                      "task": ev["task"].to_numpy(), "foot": ev["foot"].to_numpy(),
                      "pass_id": ev["pass_id"].to_numpy(),
                      "t0": pd.to_numeric(ev["t0_s"]).to_numpy(np.float64),
                      "t1": pd.to_numeric(ev["t1_s"]).to_numpy(np.float64),
                      "S": pd.to_numeric(ev["reference"]).to_numpy(np.float64) / 100.0,
                      "D": pd.to_numeric(ev["prediction"]).to_numpy(np.float64) / 100.0})
    if set(d["person_id"]) - set(allowed):
        raise AssertionError("a stride outside the allowed persons")
    return d


def clean_strides(d):
    """(kept strides with c = 120 / (t1 - t0) steps/min, counts): strides with t1 - t0 in T_RANGE, S at least S_MIN,
    D positive and every value finite."""
    T = d["t1"] - d["t0"]
    ok = (np.isfinite(d[["t0", "t1", "S", "D"]].to_numpy()).all(1) & (T >= T_RANGE[0]) & (T <= T_RANGE[1])
          & (d["S"] >= S_MIN) & (d["D"] > 0)).to_numpy()
    k = d[ok].copy()
    k["c"] = 120.0 / (k["t1"] - k["t0"])
    return k.reset_index(drop=True), {"read": int(len(d)), "kept": int(ok.sum()), "dropped": int((~ok).sum())}


def events_path(end, which):
    name = ENDS[end]
    if which == "train54":
        return os.path.join(F1_DIR, f"decode_{name}", "train_events.csv")
    return os.path.join(TREE, "reports", "gates", "g3", f"diag_{name}", "events.csv")


def in_sim_gain(end, path=None):
    """The in-sim stride gain of an end from F1's results.json (ends.<end>.insim.stride_gain)."""
    with open(path or F1_RESULTS, encoding="utf-8") as f:
        r = json.load(f)
    return float(r["ends"][end]["insim"]["stride_gain"])


# ---------------------------------------------------------------- AMASS strides
def amass_takes():
    """Rows {subset, take} of reports/amass_walks/takes.csv, each subset asserted a train archive."""
    import csv
    from mdfm.data import amass_index
    with open(TAKES_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for s in sorted({r["subset"] for r in rows}):
        if amass_index.check(s) != "train":
            raise Refused(f"{s} is not a train archive")
    return rows


def amass_strides(limit=None, log=print):
    """Per scored true stride of the AMASS train walks: take, person_id, gender, treadmill, t0, t1, S (m), c
    (steps/min), stature_m, l (m)."""
    import pandas as pd
    import torch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index, body_store
    from mdfm.eval import infer as I
    from mdfm.readouts import gait
    ins = _sibling("eval_gait_insim")
    h1 = _sibling("f2_H1shape")
    rows = amass_takes()
    want = [r["take"] for r in rows]
    store = body_store.BodyStore(split="train")
    tk = store.takes
    if set(tk["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person in the train body store")
    idx = {t: i for i, t in enumerate(tk["take_id"])}
    missing = [t for t in want if t not in idx]
    if limit:
        want = [t for t in want if t in idx][:int(limit)]
    bank = SkeletonBank(torch.float64)
    out, stat = [], {}
    for n_done, t in enumerate(want):
        if t not in idx:
            continue
        i = idx[t]
        r = tk.iloc[i]
        I.refuse_cody(str(r["archive"]), str(t), str(r["person_id"]), str(r.get("subject_ns", "")))
        n = int(r["n_knots"])
        bodies = store.windows([i], [0], win_s=n / RATE)
        J, _, contact = ins.truth_on_clock(bodies, bank, np.arange(n, dtype=np.float64) / RATE)
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
            stat[key] = float(np.asarray(h1.stature_of(store.betas[i], g)[0]))
        for s in st.itertuples():
            out.append({"take": t, "person_id": str(r["person_id"]), "gender": g,
                        "treadmill": int(r["treadmill"]), "scored": bool(s.scored), "t0": s.t0, "t1": s.t1,
                        "S": float(s.length_m), "stature_m": stat[key]})
        if (n_done + 1) % 50 == 0:
            log(f"  AMASS {n_done + 1} of {len(want)} takes, {len(out)} strides")
    d = pd.DataFrame(out)
    d.attrs["missing_takes"] = missing
    return d


def amass_clean(d):
    T = d["t1"] - d["t0"]
    ok = (d["scored"].astype(bool) & (T >= T_RANGE[0]) & (T <= T_RANGE[1]) & (d["S"] >= S_MIN)).to_numpy()
    k = d[ok].copy()
    k["c"] = 120.0 / (k["t1"] - k["t0"])
    k["l"] = LEG_RATIO * k["stature_m"]
    return k.reset_index(drop=True), {"strides": int(len(d)), "kept": int(ok.sum()),
                                      "unscored_or_out_of_bounds": int((~ok).sum())}


# ---------------------------------------------------------------- relations
def design(logc, quad=False):
    x = np.asarray(logc, dtype=np.float64)
    cols = [np.ones_like(x), x - C_CENTRE]
    if quad:
        cols.append((x - C_CENTRE) ** 2)
    return np.stack(cols, 1)


def person_blocks(df, y, X):
    """{person: (X' W X, X' W y)} with weights 1 / the person's stride count."""
    out = {}
    pid = df["person_id"].to_numpy()
    for p in np.unique(pid):
        m = pid == p
        w = 1.0 / m.sum()
        Xp, yp = X[m], y[m]
        out[p] = (w * Xp.T @ Xp, w * Xp.T @ yp)
    return out


def solve_blocks(blocks, persons, counts=None):
    k = next(iter(blocks.values()))[0].shape[0]
    A, b = np.zeros((k, k)), np.zeros(k)
    for j, p in enumerate(persons):
        c = 1.0 if counts is None else counts[j]
        if c:
            A += c * blocks[p][0]
            b += c * blocks[p][1]
    return np.linalg.solve(A, b)


def fit_relation(df, quad=False):
    """Coefficients of log(S / l) on design(log c), persons weighted equally."""
    y = np.log(df["S"].to_numpy() / df["l"].to_numpy())
    X = design(np.log(df["c"].to_numpy()), quad)
    bl = person_blocks(df, y, X)
    return solve_blocks(bl, sorted(bl)), bl


def predict(coef, logc, leg):
    X = design(logc, len(coef) == 3)
    return np.asarray(leg) * np.exp(X @ coef)


# ---------------------------------------------------------------- statistics of the real strides
def padded(df, persons, col):
    """(P, Nmax) array of a column per person, NaN padded."""
    groups = [df.loc[df["person_id"] == p, col].to_numpy(np.float64) for p in persons]
    n = max(len(g) for g in groups)
    a = np.full((len(persons), n), np.nan)
    for i, g in enumerate(groups):
        a[i, :len(g)] = g
    return a


def mpm(per_person, rows=None):
    """Median over persons (rows of per_person, with repetition) of per-person values."""
    v = per_person if rows is None else per_person[rows]
    return float(np.nanmedian(v))


def pass_table(df):
    g = df.groupby(["person_id", "trial_id", "pass_id"], sort=True)
    t = g.agg(S=("S", "mean"), D=("D", "mean"), c=("c", "mean"), l=("l", "first"), n=("S", "size")).reset_index()
    return t


def pass_regression(passes, rows_by_person=None, persons=None):
    """Least squares of log D on (1, log S, log c, log l) over passes: (b_S, b_c, b_l, intercept)."""
    if rows_by_person is not None:
        idx = np.concatenate([rows_by_person[p] for p in persons])
        t = passes.iloc[idx]
    else:
        t = passes
    X = np.stack([np.ones(len(t)), np.log(t["S"]), np.log(t["c"]), np.log(t["l"])], 1)
    y = np.log(t["D"].to_numpy())
    if np.linalg.matrix_rank(X) < X.shape[1]:
        X = X[:, :3]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coef[1]), float(coef[2]), float("nan"), float(coef[0])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1]), float(coef[2]), float(coef[3]), float(coef[0])


def within_regression(passes):
    """Person fixed effects: log D on log S and log c after removing each person's means (b_S, b_c)."""
    _, inv = np.unique(passes["person_id"].to_numpy(), return_inverse=True)
    n = np.bincount(inv)
    cols = {}
    for c in ("D", "S", "c"):
        v = np.log(passes[c].to_numpy(np.float64))
        cols[c] = v - (np.bincount(inv, weights=v) / n)[inv]
    X = np.stack([cols["S"], cols["c"]], 1)
    coef, *_ = np.linalg.lstsq(X, cols["D"], rcond=None)
    return float(coef[0]), float(coef[1])


def c2_class(b, iv):
    lo, hi = iv
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo > UNRESOLVED_WIDTH:
        return "UNRESOLVED"
    if b >= READ_POINT and lo > READ_LOWER:
        return "READ"
    if b <= CAD_POINT and hi < CAD_UPPER:
        return "FROM CADENCE"
    return "BETWEEN"


def reading(c2, h, h_iv):
    """The verdict of the Prediction section."""
    if c2 == "FROM CADENCE" and h >= H_POINT and h_iv[0] > H_LOWER:
        return "SUPPORTED", "C2 FROM CADENCE and h at least 0.5 with its lower bound above 0.25"
    if c2 == "READ":
        return "NOT SUPPORTED", "C2 READ: the decoded pass stride follows the walkway stride at fixed cadence and leg"
    if h_iv[1] < H_NOT_UPPER and c2 != "FROM CADENCE":
        return "NOT SUPPORTED", f"the upper bound of h is below {H_NOT_UPPER} and C2 is {c2}"
    return "PARTLY", f"C2 {c2}, h {h:.3f} [{h_iv[0]:.3f}, {h_iv[1]:.3f}]: no clause of SUPPORTED or NOT SUPPORTED holds"


def analyse(real, amass, g_sim, n_boot=BOOT_N, seed=SEED, quad=False, leg_col="l"):
    """C1, C2, C3 and the real relation on one set of real strides (columns person_id, trial_id, pass_id, S, D, c,
    l) against the AMASS strides; joint person bootstrap of both populations."""
    real = real.copy()
    real["l"] = real[leg_col]
    persons = sorted(real["person_id"].unique())
    am_persons = sorted(amass["person_id"].unique())
    coefA, blA = fit_relation(amass, quad)
    logc = padded(real, persons, "c")
    logc = np.log(logc)
    logS = np.log(padded(real, persons, "S"))
    logD = np.log(padded(real, persons, "D"))
    logl = np.log(padded(real, persons, "l"))
    dS = np.nanmedian(logD - logS, axis=1)                      # per person, log D / S

    def a_over_s(coef):
        X = np.stack([np.ones_like(logc), logc - C_CENTRE] + ([(logc - C_CENTRE) ** 2] if len(coef) == 3 else []), -1)
        return np.nanmedian(logl + X @ coef - logS, axis=1)       # per person, log A / S

    aS = a_over_s(coefA)
    L_obs = mpm(dS) - math.log(g_sim)
    L_H5 = mpm(aS)
    h = L_H5 / L_obs
    # C3 per stride, pooled
    X = np.stack([np.ones_like(logc), logc - C_CENTRE] + ([(logc - C_CENTRE) ** 2] if quad else []), -1)
    logA = logl + X @ coefA

    def mad_ratio(rows):
        eS = (logD - logS)[rows].ravel()
        eA = (logD - logA)[rows].ravel()
        eS, eA = eS[np.isfinite(eS)], eA[np.isfinite(eA)]
        mS = np.median(np.abs(eS - np.median(eS)))
        mA = np.median(np.abs(eA - np.median(eA)))
        return float(mA / mS), float(mS), float(mA)

    R, madS, madA = mad_ratio(np.arange(len(persons)))
    passes = pass_table(real)
    rows_by_person = {p: np.flatnonzero(passes["person_id"].to_numpy() == p) for p in persons}
    bS, bc, bl, b0 = pass_regression(passes)
    wS, wc = within_regression(passes)
    # real relation, for description
    coefR, blR = fit_relation(real, quad=False)
    c_med = float(np.nanmedian(np.exp(logc)))
    coefA_lin, blA_lin = fit_relation(amass, quad=False)
    ratio_rel = float(np.exp(design([math.log(c_med)]) @ (coefR - coefA_lin))[0])
    rng = np.random.default_rng(seed)
    bh, bLH5, bLobs, bbS, bbc, bbl, bR, bwS, bratio = [], [], [], [], [], [], [], [], []
    P, Q = len(persons), len(am_persons)
    for _ in range(n_boot):
        rows = rng.integers(P, size=P)
        arows = rng.integers(Q, size=Q)
        cnt = np.bincount(arows, minlength=Q)
        cA = solve_blocks(blA, am_persons, cnt)
        aSb = a_over_s(cA)
        lo = mpm(dS, rows) - math.log(g_sim)
        lh = mpm(aSb, rows)
        bLobs.append(lo)
        bLH5.append(lh)
        bh.append(lh / lo if lo != 0 else float("nan"))
        ps_ = [persons[i] for i in rows]
        b = pass_regression(passes, rows_by_person, ps_)
        bbS.append(b[0])
        bbc.append(b[1])
        bbl.append(b[2])
        bR.append(mad_ratio(rows)[0])
        idx = np.concatenate([rows_by_person[p] for p in ps_])
        # within-person slope on the resampled persons (duplicates share a mean: same estimate as weights)
        t = passes.iloc[idx].copy()
        t["person_id"] = np.repeat(np.arange(len(ps_)), [len(rows_by_person[p]) for p in ps_])
        bwS.append(within_regression(t)[0])
        cR = solve_blocks(blR, ps_and_unique(ps_), counts_of(ps_))
        cAl = cA if len(cA) == 2 else solve_blocks(blA_lin, am_persons, cnt)
        bratio.append(float(np.exp(design([math.log(c_med)]) @ (cR - cAl))[0]))
    c2_iv = interval(bbS)
    return {
        "n_persons": P, "n_strides": int(len(real)), "n_passes": int(len(passes)),
        "g_sim": g_sim,
        "amass_relation": {"coef": coefA.tolist(), "quadratic": quad, "centre_log_c": C_CENTRE},
        "C1": {"h": h, "interval": interval(bh), "L_obs": L_obs, "L_obs_interval": interval(bLobs),
               "L_H5": L_H5, "L_H5_interval": interval(bLH5),
               "decoded_over_walkway": math.exp(mpm(dS)),
               "amass_predicted_over_walkway": math.exp(L_H5),
               "amass_predicted_over_walkway_interval": [math.exp(x) for x in interval(bLH5)]},
        "C2": {"b_S": bS, "interval": c2_iv, "class": c2_class(bS, c2_iv), "b_c": bc, "b_c_interval": interval(bbc),
               "b_l": bl, "b_l_interval": interval(bbl), "intercept": b0,
               "within_person_b_S": wS, "within_person_b_S_interval": interval(bwS), "within_person_b_c": wc},
        "C3": {"R": R, "interval": interval(bR), "mad_walkway": madS, "mad_amass": madA},
        "real_relation": {"coef": coefR.tolist(), "median_cadence": c_med,
                          "real_over_amass_at_median_cadence": ratio_rel,
                          "interval": interval(bratio)},
        "per_person": {p: {"log_D_over_S": float(dS[i]), "log_A_over_S": float(aS[i])} for i, p in enumerate(persons)},
    }


def ps_and_unique(ps):
    return sorted(set(ps))


def counts_of(ps):
    u = sorted(set(ps))
    return [ps.count(p) for p in u]


# ---------------------------------------------------------------- main
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fiv(iv, nd=3):
    return f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def write_summary(out, r):
    p = r["primary"]
    L = ["# F2-H5data: is the decoded stride the AMASS stride at the walker's cadence? (dev diagnostic, not a G3 "
         "read)", "",
         f"Written by scripts/f2_H5data.py; every number is read from results.json of this folder. Prediction "
         f"SHA-256 {r['prediction_sha256']} (asserted before any value was read). No lockbox person, EVAL, CODY or "
         f"care_pd record was read (asserted).", "",
         f"## Reading (v1_B, train54): **{r['reading']['verdict']}**", "",
         f"- {r['reading']['reason']}",
         f"- C1 level share h {fmt(p['C1']['h'])} {fiv(p['C1']['interval'])}: L_obs {fmt(p['C1']['L_obs'], 4)}, "
         f"L_H5 {fmt(p['C1']['L_H5'], 4)}; the AMASS relation predicts the real strides at "
         f"{fmt(p['C1']['amass_predicted_over_walkway'])} {fiv(p['C1']['amass_predicted_over_walkway_interval'])} "
         f"of their walked length; decoded over walkway {fmt(p['C1']['decoded_over_walkway'])}",
         f"- C2 pass slope b_S {fmt(p['C2']['b_S'])} {fiv(p['C2']['interval'])} ({p['C2']['class']}); b_c "
         f"{fmt(p['C2']['b_c'])} {fiv(p['C2']['b_c_interval'])}, b_l {fmt(p['C2']['b_l'])} "
         f"{fiv(p['C2']['b_l_interval'])}",
         f"- C3 (descriptive) R {fmt(p['C3']['R'])} {fiv(p['C3']['interval'])}", "",
         "## All rows", "",
         "| row | persons | strides | passes | h | 95 % | A/S | b_S | 95 % | C2 | b_c | within-person b_S | R | "
         "real/AMASS relation at median cadence |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in r["rows"].items():
        L.append(f"| {name} | {x['n_persons']} | {x['n_strides']} | {x['n_passes']} | {fmt(x['C1']['h'])} | "
                 f"{fiv(x['C1']['interval'])} | {fmt(x['C1']['amass_predicted_over_walkway'])} | "
                 f"{fmt(x['C2']['b_S'])} | {fiv(x['C2']['interval'])} | {x['C2']['class']} | {fmt(x['C2']['b_c'])} | "
                 f"{fmt(x['C2']['within_person_b_S'])} {fiv(x['C2']['within_person_b_S_interval'])} | "
                 f"{fmt(x['C3']['R'])} | {fmt(x['real_relation']['real_over_amass_at_median_cadence'])} "
                 f"{fiv(x['real_relation']['interval'])} |")
    a = r["amass"]
    L += ["", "## Relations log(S / l) = alpha + beta (log c - log 110), persons weighted equally", "",
          "| population | persons | strides | alpha | beta | median cadence (steps/min) | median S / l |",
          "|---|---|---|---|---|---|---|"]
    for name, x in r["relations"].items():
        L.append(f"| {name} | {x['n_persons']} | {x['n_strides']} | {fmt(x['coef'][0], 4)} | {fmt(x['coef'][1], 4)} | "
                 f"{fmt(x['median_cadence'], 1)} | {fmt(x['median_S_over_l'], 4)} |")
    L += ["", "## Counts", "",
          f"AMASS: {a['takes_listed']} takes listed, {a['takes_missing']} not in the train store, "
          f"{a['counts']['strides']} true strides, {a['counts']['kept']} scored and in bounds, {a['n_persons']} persons "
          f"({a['n_persons_overground']} with overground strides). Real (v1_B): "
          + "; ".join(f"{k} {v['kept']} of {v['read']} matched strides kept" for k, v in r["real_counts"].items())
          + f". MoveAtlas HEAD {r['moveatlas']['head']}. Time {fmt(r['timing_s'], 1)} s."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--reuse-amass", action="store_true", help="read <out>/amass_strides.csv instead of the store")
    ap.add_argument("--boot", type=int, default=BOOT_N)
    args = ap.parse_args(argv)
    import pandas as pd
    t_start = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    sys.stdout.reconfigure(encoding="utf-8")
    pred_sha = check_prediction()                      # before any value is read
    sets, locked = load_sets()
    allowed = {"train54": sets["train54"], "dev12": sets["dev12"]}
    os.makedirs(args.out, exist_ok=True)
    # AMASS
    am_csv = os.path.join(args.out, "amass_strides.csv")
    if args.reuse_amass and os.path.exists(am_csv):
        am_raw = pd.read_csv(am_csv, encoding="utf-8")
        missing = json.load(open(os.path.join(args.out, "amass_missing.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(args.out, "amass_missing.json")) else []
    else:
        print("reading AMASS train walks", flush=True)
        am_raw = amass_strides()
        missing = list(am_raw.attrs.get("missing_takes", []))
        am_raw.to_csv(am_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")
        with open(os.path.join(args.out, "amass_missing.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(missing, f)
    from mdfm.data import amass_index
    if set(am_raw["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person among the AMASS strides")
    amass, am_counts = amass_clean(am_raw)
    amass_over = amass[amass["treadmill"] == 0].reset_index(drop=True)
    # heights
    demo = _sibling("f2_H1shape").read_demographics(sets["train54"] + sets["dev12"], locked)
    with open(H1_RESULTS, encoding="utf-8") as f:
        prior = json.load(f)["prior_statures"]
    gname = _sibling("f2_H1shape").GENDER_NAME
    real, real_counts = {}, {}
    for end in ENDS:
        for s in SETS:
            d = read_strides(events_path(end, s), allowed[s], locked)
            d, cnt = clean_strides(d)
            d["l"] = [LEG_RATIO * demo.get(p, {}).get("height_m", float("nan")) for p in d["person_id"]]
            d["l_prior"] = [LEG_RATIO * prior[gname.get(demo.get(p, {}).get("sex", ""), "neutral")]["stature_m"]
                            for p in d["person_id"]]
            nh = d["l"].isna()
            cnt["no_height"] = int(nh.sum())
            real[(end, s)] = d[~nh].reset_index(drop=True)
            real_counts[f"{end}_{s}"] = cnt
    lo_c, hi_c = (float(np.quantile(amass["c"], q)) for q in CAD_RANGE_Q)
    rows = {}
    g = {e: in_sim_gain(e) for e in ENDS}
    print("analysing", flush=True)
    rows["v1_B train54 (primary)"] = analyse(real[("v1_B", "train54")], amass, g["v1_B"], args.boot)
    rows["v1_B dev12"] = analyse(real[("v1_B", "dev12")], amass, g["v1_B"], args.boot)
    rows["s2_B train54"] = analyse(real[("s2_B", "train54")], amass, g["s2_B"], args.boot)
    rows["s2_B dev12"] = analyse(real[("s2_B", "dev12")], amass, g["s2_B"], args.boot)
    t54 = real[("v1_B", "train54")]
    rows["v1_B train54 without HC134"] = analyse(t54[t54["person_id"] != FLAGGED].reset_index(drop=True), amass,
                                                 g["v1_B"], args.boot)
    rows["v1_B train54 prior-stature leg"] = analyse(t54, amass, g["v1_B"], args.boot, leg_col="l_prior")
    rows["v1_B train54 quadratic AMASS relation"] = analyse(t54, amass, g["v1_B"], args.boot, quad=True)
    rows["v1_B train54 overground AMASS only"] = analyse(t54, amass_over, g["v1_B"], args.boot)
    inr = t54[(t54["c"] >= lo_c) & (t54["c"] <= hi_c)].reset_index(drop=True)
    rows["v1_B train54 inside AMASS cadence range"] = analyse(inr, amass, g["v1_B"], args.boot)
    p = rows["v1_B train54 (primary)"]
    verdict, reason = reading(p["C2"]["class"], p["C1"]["h"], p["C1"]["interval"])
    rel_rows = {}
    for name, df in (("AMASS train walks", amass), ("AMASS overground", amass_over),
                     ("weargait train54", t54), ("weargait dev12", real[("v1_B", "dev12")])):
        coef = fit_relation(df)[0]
        rel_rows[name] = {"coef": coef.tolist(), "n_persons": int(df["person_id"].nunique()),
                          "n_strides": int(len(df)), "median_cadence": float(df["c"].median()),
                          "median_S_over_l": float((df["S"] / df["l"]).median())}
    stamp = _sibling("g3_gain_check").stamp_moveatlas()
    res = {"what": LABEL, "task": TASK, "prediction_sha256": pred_sha, "run_started": started,
           "reading": {"verdict": verdict, "reason": reason, "c2_class": p["C2"]["class"],
                       "criterion": {"READ": [READ_POINT, READ_LOWER], "FROM_CADENCE": [CAD_POINT, CAD_UPPER],
                                     "UNRESOLVED_WIDTH": UNRESOLVED_WIDTH, "H": [H_POINT, H_LOWER],
                                     "H_NOT_UPPER": H_NOT_UPPER}},
           "primary": {k: v for k, v in p.items() if k != "per_person"},
           "rows": {k: {kk: vv for kk, vv in v.items() if kk != "per_person"} for k, v in rows.items()},
           "relations": rel_rows,
           "amass": {"takes_listed": len(amass_takes()), "takes_missing": len(missing), "counts": am_counts,
                     "n_persons": int(amass["person_id"].nunique()),
                     "n_persons_overground": int(amass_over["person_id"].nunique()),
                     "cadence_range_2.5_97.5": [lo_c, hi_c]},
           "real_counts": real_counts, "g_sim": g,
           "sets": {"dev12_sha256": None, "train54_sha256": None, "lockbox_persons_in_sets": 0},
           "constants": {"leg_ratio": LEG_RATIO, "t_range_s": T_RANGE, "s_min_m": S_MIN, "boot_n": args.boot,
                         "seed": SEED, "level": LEVEL},
           "moveatlas": {"head": stamp.get("head") if isinstance(stamp, dict) else None},
           "timing_s": time.time() - t_start}
    from mdfm.data import sources
    res["sets"]["dev12_sha256"] = sources.person_hash(sets["dev12"])
    res["sets"]["train54_sha256"] = sources.person_hash(sets["train54"])
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(res), f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "persons.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("row,person_id,log_D_over_S,log_A_over_S\n")
        for name in ("v1_B train54 (primary)", "v1_B dev12"):
            for q, v in rows[name]["per_person"].items():
                f.write(f"{name},{q},{v['log_D_over_S']:.5f},{v['log_A_over_S']:.5f}\n")
    prov = {"script": rel(__file__), "script_sha256": sha256_file(__file__), "run_started": started,
            "moveatlas": stamp,
            "inputs_sha256": {rel(x): sha256_file(x) for x in
                              [F1_RESULTS, H1_RESULTS, TAKES_CSV, REPORT, PREDICTION_SHA]
                              + [events_path(e, s) for e in ENDS for s in SETS]}}
    with open(os.path.join(args.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(prov), f, indent=1, ensure_ascii=False)
    write_summary(args.out, clean(res))
    print(f"reading {verdict}: {reason}", flush=True)
    return res


if __name__ == "__main__":
    main()
