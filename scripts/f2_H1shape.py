"""F2-H1shape (2026-10-05): is the G3 scale error of the decoded stride (F1: about 0.847 of the walkway stride on
the 54 training-fold controls) a body-shape error, the model decoding a leg near the prior mean of its synthetic
training bodies so that taller persons are under-read more? Label of every output: dev diagnostic, not a G3 read.

Pre-registration: the section "## Prediction" of reports/weekend_2026-10-01/tasks/F2-H1shape.md. Its SHA-256 (the
section text from its heading to the next "## " heading, LF) is the constant PREDICTION_SHA256; check_prediction
asserts it before any stride, speed or height value is read, so a criterion edited after the fact stops the run.

Persons: dev14, dev12 and train54 of mdfm.gates.g3.controls, asserted equal (SHA-256) to the sets of
reports/bench/g3_gain_check/results.json, and disjoint from mdfm.gates.g3.lockbox_persons(). Demographic rows are
filtered on their Subject ID to those persons before any other column is read (read_demographics); a lockbox,
EVAL or CODY record is never opened.

Default run (CPU; CUDA_VISIBLE_DEVICES=-1 is set before torch loads):
- statures of the SMPL+H rest meshes (top minus bottom vertex, y up; stature_of) at the prior mean betas of each
  sex (reports/body_store/priors.json) and of the 360 AMASS train persons of the body store (mean betas of each
  person's takes, its gendered model); the leg (hip joint height above the sole) beside;
- reported height (Height (in) x 0.0254 m) and sex of the 68 persons;
- per person and end (v1_B primary, s2_B replication): r, the median over matched strides of decoded / walkway
  (row base, both paces; F1's events files read with scripts/g3_gain_check.read_events), and S, the mean walkway
  stride;
- the criteria C1, C2, C3 of the prediction and the verdict (verdict_of);
- when reports/bench/f2_scale/H1shape/dump_<end>/ exists (the GPU job below): D1 and D2.
Writes reports/bench/f2_scale/H1shape/results.json, persons.csv, amass_statures.csv, summary.md, _provenance.json.

--dump (GPU, one queued job, --print-job prints its scripts/queue_job.py command): the inference of
scripts/eval_g3_dev.py (mdfm.eval.infer: record_windows, one shape group per trial, seed_of(seed, trial_id), the
same refinement), reimplemented here (infer_two) so that the posterior of each trial is decoded twice: with the
pooled shape mean (base, equal to the official decode) and with beta_0 shifted so that the decoded rest-mesh
stature equals the person's reported height (cf; shift_beta0). Over the SelfPace and HurriedPace records of
train54 and dev14. A trial whose counterfactual needs a beta_0 shift beyond SHIFT_MAX keeps its base decode, gets no
counterfactual decode and is recorded (never clamped): skipped_cf.csv (trial, required shift, reported height) and
dump.json cf_skipped. Writes <out>/shape.csv (per trial: gender, pooled betas, decoded stature and leg, whether a
height exists, cf_status, shift, required shift, stature error after the shift), events_base.csv and events_cf.csv
(columns of scripts/eval_g3_diag.py, rows base and cf), skipped_cf.csv, trials.csv, dump.json, _provenance.json.

--results-out DIR (analysis): write results.json, summary.md, persons.csv, amass_statures.csv and _provenance.json
under DIR instead of reports/bench/f2_scale/H1shape/ (the dump is still read from there), so that a rerun never
rewrites the files of an earlier run. Post hoc, beside the registered criteria: the height plausibility screen
(reachable_statures, height_flags) and the criteria recomputed without the flagged persons.

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/f2_H1shape.py
  <conda>/envs/mdfm/python.exe scripts/f2_H1shape.py --print-job
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__" and "--dump" not in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the analysis never sees the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm import paths  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H1shape")
F1_DIR = os.path.join(TREE, "reports", "bench", "g3_gain_check")
F1_RESULTS = os.path.join(F1_DIR, "results.json")
REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2-H1shape.md")
PRIORS = os.path.join(TREE, "reports", "body_store", "priors.json")
PREDICTION_HEADING = "## Prediction"
PREDICTION_SHA256 = "ae46bbd0b545d0c5e447afb203ee6605b6928b5cd1aeae52c82d70349a79962e"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-H1shape"
BOOT_N = 10000
SEED = 20261005
LEVEL = 0.95
IN_TO_M = 0.0254
# Leg length from height: greater trochanter height 0.530 H (Winter DA, Biomechanics and Motor Control of Human
# Movement, 4th ed., Wiley 2009, Fig. 4.1, after Drillis and Contini 1966). Used for the leg column only; the
# criteria read statures, like for like with the rest meshes.
LEG_RATIO = 0.530
C_MIN = 0.5                      # C1 and C3: slope point at least this, lower bound above 0
F_MIN = 0.5                      # C2: share of the real-domain log shortfall that the stature ratio carries
D1_MAX = 0.5                     # D1: upper bound of the decoded-stature slope below this
D2_FRAC = 0.5                    # D2: the cf gain closes at least this share of the gap to the in-sim gain
STATURE_RANGE_M = (1.2, 2.2)     # sanity bounds of a rest-mesh stature
SHIFT_TOL_M = 1e-4
SHIFT_MAX = 6.0                  # largest |beta_0 shift| allowed
ENDS = {"v1_B": ("phaseA_root_full_ckpt_B_032044", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", 32),
        "s2_B": ("pilot_v1_s2_ckpt_B_032044", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", 16)}
PRIMARY = "v1_B"
DUMP_END = "v1_B"
# the first job of this name without _r2 failed on 2026-10-05 (a trial whose shift lay beyond SHIFT_MAX raised); a
# logged job name never reruns, so the job after the fix carries _r2
JOB = {"name": "f2_h1shape_dump_v1_B_r2", "cls": "1", "priority": "74", "est_hours": "0.25", "memory_gb": "12",
       "group": "F2D"}
SKIPPED_COLUMNS = ["trial_id", "person_id", "required_beta0_shift", "shift_max", "reported_height_m",
                   "reported_height_in", "stature_dec_m", "gender_code"]


def rel(path):
    try:
        r = os.path.relpath(os.path.abspath(path), TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def _sibling(name):
    key = f"_f2h1_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def GC():
    """scripts/g3_gain_check.py (F1): its guards, events reader and set checks."""
    return _sibling("g3_gain_check")


# ---------------------------------------------------------------- pre-registration
def prediction_section(text):
    """The text of the section '## Prediction' (its heading line to the next '## ' heading, excluded), LF line ends;
    None without one."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        a = next(i for i, x in enumerate(lines) if x.strip() == PREDICTION_HEADING)
    except StopIteration:
        return None
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[a:b]).rstrip("\n") + "\n"


def prediction_sha(text):
    s = prediction_section(text)
    return None if s is None else hashlib.sha256(s.encode("utf-8")).hexdigest()


def check_prediction(report=None, expected=None):
    """SHA-256 of the prediction section, asserted equal to `expected` (default PREDICTION_SHA256)."""
    report = REPORT if report is None else report
    expected = PREDICTION_SHA256 if expected is None else expected
    with open(report, encoding="utf-8") as f:
        sha = prediction_sha(f.read())
    if sha is None or sha != expected:
        raise AssertionError(f"prediction section of {os.path.basename(report)}: {sha}, expected {expected}")
    return sha


# ---------------------------------------------------------------- persons
def load_sets(f1_results=None):
    """(sets {dev14, dev12, train54}, locked): mdfm.gates.g3's folds, asserted equal to F1's sets and their SHA-256,
    checked by g3_gain_check.check_sets (sizes, disjointness, no lockbox person)."""
    from mdfm.data import sources
    from mdfm.gates import g3
    with open(f1_results or F1_RESULTS, encoding="utf-8") as f:
        f1 = json.load(f)["sets"]
    dev, train = sorted(g3.controls("dev")), sorted(g3.controls("train"))
    locked = g3.lockbox_persons()
    gc = GC()
    sets = {"dev14": dev, "dev12": [p for p in dev if p not in gc.EXCLUDED_FROM_12], "train54": train}
    gc.check_sets(sets, locked)
    for name in ("dev14", "dev12", "train54"):
        if sources.person_hash(sets[name]) != f1[f"{name}_sha256"]:
            raise AssertionError(f"{name} differs from the set of {rel(F1_RESULTS)}")
    if dev != f1["dev14"] or sets["dev12"] != f1["dev12"]:
        raise AssertionError("dev sets differ from F1's lists")
    return sets, locked


def table_id(person_id):
    """Subject ID of the weargait_pd demographic tables of a person id ('weargait:HC103' -> 'HC103',
    'weargait:NLS901 (control)' -> 'NLS901')."""
    s = str(person_id).replace(" (control)", "")
    if not s.startswith("weargait:"):
        raise ValueError(f"not a weargait person id: {person_id!r}")
    return s[len("weargait:"):].strip()


def demographic_files():
    from mdfm.data import sources
    out = []
    for pattern in sources.DEMOGRAPHIC_FILES["weargait_pd"]:
        files = sorted(x.replace("\\", "/") for x in glob.glob(sources._raw(pattern))
                       if not os.path.basename(x).startswith("._"))
        if len(files) != 1:
            raise FileNotFoundError(f"{pattern}: {len(files)} files")
        out.append(files[0])
    return out


def read_demographics(persons, locked, files=None, weight=False):
    """{person_id: {height_m, sex, file}} of the persons (guarded: Refused for a lockbox person). Each table row is
    kept or dropped on its Subject ID alone; only kept rows have their Height (in) and Sex read (and Weight (kg) as
    weight_kg when weight is True: the height plausibility check). AssertionError when a person appears twice.
    Persons absent from the tables are absent from the result."""
    from mdfm.data import sources
    want = {table_id(p): p for p in GC().guard_persons(persons, None, locked, "demographics")}
    out = {}
    for path in (files or demographic_files()):
        with open(path, encoding="utf-8-sig", newline="") as f:
            table = list(csv.reader(f))
        head = next(i for i, r in enumerate(table) if r and r[0].strip() == "Subject ID")
        cols = [c.strip() for c in table[head]]
        ih, isx = cols.index("Height (in)"), cols.index("Sex")
        iw = cols.index("Weight (kg)") if weight and "Weight (kg)" in cols else None
        for r in table[head + 1:]:
            if not r or r[0].strip() not in want:
                continue                                   # a row outside the persons: nothing else is read
            p = want[r[0].strip()]
            if p in out:
                raise AssertionError(f"{p} twice in the demographic tables")
            h = sources._num(r[ih]) if ih < len(r) else float("nan")
            out[p] = {"height_m": h * IN_TO_M if math.isfinite(h) else float("nan"),
                      "sex": sources._sex(r[isx]) if isx < len(r) else "", "file": rel(path)}
            if weight:
                out[p]["height_in"] = h
                out[p]["weight_kg"] = sources._num(r[iw]) if iw is not None and iw < len(r) else float("nan")
    return out


# ---------------------------------------------------------------- body shapes
_MESH = {}


def mesh(gender):
    """(v_template (V, 3), shapedirs (V, 3, 16), J_regressor rows 1 and 2 (2, V)) of a gendered SMPL+H model, float64,
    read from the MoveAtlas tree (paths.smplh_model) and kept in memory only."""
    if gender not in _MESH:
        m = np.load(paths.smplh_model(gender))
        _MESH[gender] = (np.asarray(m["v_template"], np.float64), np.asarray(m["shapedirs"], np.float64)[:, :, :16],
                         np.asarray(m["J_regressor"], np.float64)[1:3])
    return _MESH[gender]


GENDER_NAME = {0: "female", 1: "male", 2: "neutral", "F": "female", "M": "male", "": "neutral",
               "female": "female", "male": "male", "neutral": "neutral"}


def stature_of(betas, gender):
    """(stature m, leg m) of rest meshes: betas (..., 16); stature the top minus the bottom vertex along y, leg the
    mean height of the two hip joints above the bottom vertex."""
    vt, sd, jr = mesh(GENDER_NAME[gender])
    b = np.atleast_2d(np.asarray(betas, np.float64))
    v = vt[None] + np.einsum("vcs,bs->bvc", sd, b)
    y = v[..., 1]
    lo = y.min(1)
    hips = np.einsum("jv,bv->bj", jr, y)
    st, leg = y.max(1) - lo, hips.mean(1) - lo
    shape = np.asarray(betas).shape[:-1]
    return st.reshape(shape), leg.reshape(shape)


class ShiftOutOfRange(ValueError):
    """The beta_0 shift that sets the rest-mesh stature to the target lies beyond SHIFT_MAX. required: the shift the
    secant steps reached (None when they ran past SEARCH_MAX before converging); target_m: the target stature."""

    def __init__(self, required, target_m):
        self.required, self.target_m = required, target_m
        req = "beyond the search bound" if required is None else f"{required:.2f}"
        super().__init__(f"beta_0 shift {req} beyond {SHIFT_MAX} (target stature {target_m:.3f} m)")


SEARCH_MAX = 5 * SHIFT_MAX       # the secant steps may look this far, so that an out-of-range shift is reported


def shift_beta0(beta, gender, target_m, tol=SHIFT_TOL_M, iters=20):
    """(shifted betas, shift): beta_0 moved so that stature_of equals target_m (secant steps). ShiftOutOfRange (a
    ValueError) when the shift that reaches the target lies beyond SHIFT_MAX: never clamped. ValueError for a target
    that is not a positive number or without convergence."""
    b = np.array(beta, np.float64)
    if not (math.isfinite(target_m) and target_m > 0):
        raise ValueError(f"target stature {target_m}")
    d0, s0 = 0.0, float(stature_of(b, gender)[0]) - target_m
    d1 = 0.5 if s0 < 0 else -0.5
    for _ in range(iters):
        bb = b.copy()
        bb[0] += d1
        s1 = float(stature_of(bb, gender)[0]) - target_m
        if abs(s1) < tol:
            if abs(d1) > SHIFT_MAX:
                raise ShiftOutOfRange(float(d1), float(target_m))
            return bb, d1
        if s1 == s0:
            break
        d0, d1, s0 = d1, d1 - s1 * (d1 - d0) / (s1 - s0), s1
        if abs(d1) > SEARCH_MAX:
            raise ShiftOutOfRange(None, float(target_m))
    raise ValueError(f"no beta_0 shift reaches {target_m:.3f} m")


def counterfactual_shape(beta, gender, height_m):
    """The counterfactual shape of one trial: beta_0 shifted so that the rest-mesh stature equals the reported height.
    Returns {status, cf_beta, shift, stature_cf_m, required_shift}: status 'done' (cf_beta set), 'no_height' (the
    height is not a finite number) or 'skipped_shift' (ShiftOutOfRange: the trial gets no counterfactual decode, its
    required shift is kept, nothing is clamped)."""
    out = {"status": "no_height", "cf_beta": None, "shift": float("nan"), "stature_cf_m": float("nan"),
           "required_shift": float("nan")}
    if not (isinstance(height_m, (int, float, np.floating)) and math.isfinite(float(height_m))):
        return out
    try:
        cf_beta, shift = shift_beta0(beta, gender, float(height_m))
    except ShiftOutOfRange as e:
        return {**out, "status": "skipped_shift",
                "required_shift": float("nan") if e.required is None else float(e.required)}
    return {**out, "status": "done", "cf_beta": cf_beta, "shift": float(shift), "required_shift": float(shift),
            "stature_cf_m": float(stature_of(cf_beta, gender)[0])}


def reachable_statures(priors=None):
    """{sex: (low m, high m)}: the rest-mesh statures reached from the prior mean betas of the sex with |beta_0 shift|
    at most SHIFT_MAX (the dump's bound). Post hoc screen of the reported heights (heights outside cannot be reached
    by the counterfactual decode)."""
    with open(priors or PRIORS, encoding="utf-8") as f:
        sp = json.load(f)["shape_prior"]
    out = {}
    for g in ("female", "male"):
        m = np.asarray(sp[g]["mean"], np.float64)
        lo, hi = m.copy(), m.copy()
        lo[0] -= SHIFT_MAX
        hi[0] += SHIFT_MAX
        a, b = float(stature_of(lo, g)[0]), float(stature_of(hi, g)[0])
        out[g] = (min(a, b), max(a, b))
    return out


def height_flags(demo, persons, reach):
    """Persons of `persons` whose reported height lies outside the reachable statures of their sex (reach:
    reachable_statures), sorted."""
    out = []
    for p in sorted(persons):
        v = demo.get(p) or {}
        h, sx = v.get("height_m", float("nan")), GENDER_NAME.get(v.get("sex", ""), "neutral")
        if sx in reach and isinstance(h, float) and math.isfinite(h) and not reach[sx][0] <= h <= reach[sx][1]:
            out.append(p)
    return out


def amass_train_persons(root=None):
    """DataFrame person_id, gender, stature_m, leg_m of the train persons of the body store (mean betas over each
    person's takes, its gendered model)."""
    import pandas as pd
    from mdfm.data import body_store as BS
    root = root or BS.store_root()
    files = sorted(x for x in glob.glob(os.path.join(root, "train", "*", "takes.csv"))
                   if not os.path.basename(x).startswith("._"))
    if not files:
        raise FileNotFoundError(f"no takes.csv under {root}/train")
    t = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8",
                               usecols=["person_id", "gender"] + BS.BETAS) for f in files], ignore_index=True)
    for c in BS.BETAS:
        t[c] = t[c].astype(np.float64)
    g = t.groupby("person_id")
    genders = g["gender"].agg(lambda s: sorted(set(s)))
    if (genders.map(len) > 1).any():
        raise ValueError("a person with two genders in the store")
    mb = g[BS.BETAS].mean()
    rows = []
    for p in mb.index:
        gd = genders[p][0]
        st, leg = stature_of(mb.loc[p].to_numpy(), gd)
        rows.append((p, gd, float(st), float(leg)))
    return pd.DataFrame(rows, columns=["person_id", "gender", "stature_m", "leg_m"]), len(files)


def prior_statures(priors=None):
    """{sex: {stature_m, leg_m}} at the prior mean betas of reports/body_store/priors.json (female, male,
    neutral)."""
    with open(priors or PRIORS, encoding="utf-8") as f:
        sp = json.load(f)["shape_prior"]
    out = {}
    for g in ("female", "male", "neutral"):
        st, leg = stature_of(np.asarray(sp[g]["mean"], np.float64), g)
        out[g] = {"stature_m": float(st), "leg_m": float(leg), "n_persons": sp[g]["n_persons"]}
    return out


# ---------------------------------------------------------------- statistics
def ols(y, X):
    """Least-squares coefficients of y on [1, X] (X (n,) or (n, k)); NaN when the design is rank deficient."""
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    X = X[:, None] if X.ndim == 1 else X
    A = np.column_stack([np.ones(len(y)), X])
    if len(y) <= A.shape[1] or np.linalg.matrix_rank(A) < A.shape[1]:
        return np.full(A.shape[1], np.nan)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def r2(y, X):
    c = ols(y, X)
    if not np.isfinite(c).all():
        return float("nan")
    X = np.asarray(X, np.float64)
    X = X[:, None] if X.ndim == 1 else X
    res = np.asarray(y) - np.column_stack([np.ones(len(y)), X]) @ c
    tot = ((np.asarray(y) - np.mean(y)) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else float("nan")


def boot_coef(y, X, j, n_boot=None, seed=None, level=LEVEL):
    """{point, interval, resamples_used} of coefficient j (1 = first regressor) of ols, persons (rows) resampled."""
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    X = X[:, None] if X.ndim == 1 else X
    point = float(ols(y, X)[j])
    rng = np.random.default_rng(SEED if seed is None else seed)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = ols(y[i], X[i])[j]
    ok = vals[np.isfinite(vals)]
    a = (1 - level) / 2
    iv = [float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a))] if len(ok) else None
    return {"point": point, "interval": iv, "resamples_used": int(len(ok))}


def pearson(x, y):
    return GC().pearson(x, y)


def holds_slope(c, minimum=C_MIN):
    return bool(c["interval"] is not None and c["interval"][0] > 0 and np.isfinite(c["point"])
                and c["point"] >= minimum)


def verdict_of(c1, c2, c3):
    """The verdict of the prediction: supported when C2 and C3 hold; partly when C1 or C3 holds otherwise; not
    supported when neither C1 nor C3 holds."""
    if c2 and c3:
        return "supported"
    if c1 or c3:
        return "partly"
    return "not_supported"


# ---------------------------------------------------------------- per person
def person_table(strides, demo, prior):
    """DataFrame per person of the strides (person_id, x walkway cm, y decoded cm): r (median y / x), n strides,
    S_cm (mean walkway stride), height_m, sex, h_ref_m (prior stature of the person's sex), x = log(h_ref / height),
    leg_m = LEG_RATIO height."""
    import pandas as pd
    s = strides.assign(q=strides["y"] / strides["x"])
    g = s.groupby("person_id")
    t = pd.DataFrame({"r": g["q"].median(), "n_strides": g["q"].size(), "S_cm": g["x"].mean(),
                      "decoded_cm": g["y"].mean()}).reset_index()
    t["height_m"] = [demo.get(p, {}).get("height_m", float("nan")) for p in t["person_id"]]
    t["sex"] = [demo.get(p, {}).get("sex", "") for p in t["person_id"]]
    t["h_ref_m"] = [prior[GENDER_NAME[sx]]["stature_m"] for sx in t["sex"]]
    t["x"] = np.log(t["h_ref_m"] / t["height_m"])
    t["leg_m"] = LEG_RATIO * t["height_m"]
    return t


def criteria(t, g_s, g_sim, n_boot=None, seed=None):
    """C1, C2, C3 and descriptive rows on a person table (persons with a finite height, ratio and stride)."""
    ok = np.isfinite(t["x"]) & np.isfinite(t["r"]) & (t["r"] > 0) & np.isfinite(t["S_cm"])
    u = t[ok]
    y = np.log(u["r"].to_numpy())
    x = u["x"].to_numpy()
    ls = np.log(u["S_cm"].to_numpy())
    c1 = boot_coef(y, x, 1, n_boot, seed)
    c3 = boot_coef(y, np.column_stack([x, ls]), 1, n_boot, seed)
    c3_stride = boot_coef(y, np.column_stack([x, ls]), 2, n_boot, seed)
    stride_only = boot_coef(y, ls, 1, n_boot, seed)
    g_h1 = float(np.median(np.exp(x))) if len(x) else float("nan")
    denom = math.log(g_s / g_sim)
    f = math.log(g_h1) / denom if (np.isfinite(g_h1) and g_h1 > 0 and denom != 0) else float("nan")
    out = {"n_persons": int(len(u)), "n_dropped": int((~ok).sum()), "dropped": sorted(t.loc[~ok, "person_id"]),
           "C1": {**c1, "holds": holds_slope(c1)}, "C3": {**c3, "holds": holds_slope(c3)},
           "C3_log_stride_coef": c3_stride, "stride_only_slope": stride_only,
           "C2": {"G_H1": g_h1, "g_S": g_s, "g_sim": g_sim, "log_shortfall": denom, "f": f,
                  "holds": bool(np.isfinite(f) and f >= F_MIN)},
           "r2_x": r2(y, x), "r2_log_stride": r2(y, ls), "r2_both": r2(y, np.column_stack([x, ls])),
           "pearson_r_height": pearson(u["r"], u["height_m"]),
           "pearson_logS_x": pearson(ls, x),
           "median_r": float(np.median(u["r"])) if len(u) else float("nan"),
           "median_height_m": float(np.median(u["height_m"])) if len(u) else float("nan"),
           "median_h_ref_over_height": g_h1}
    for sx in ("F", "M"):
        w = u[u["sex"] == sx]
        out[f"within_sex_{sx}"] = {"n": int(len(w)), "pearson_r_height": pearson(w["r"], w["height_m"]),
                                  "median_height_m": float(np.median(w["height_m"])) if len(w) else float("nan")}
    out["verdict_inputs"] = {"C1": out["C1"]["holds"], "C2": out["C2"]["holds"], "C3": out["C3"]["holds"]}
    # post hoc, descriptive (not in the prediction): the ratio of a person as tall as the prior body of the sex
    a0 = boot_coef(y, x, 0, n_boot, seed)
    out["posthoc_ratio_at_prior_stature"] = {"point": math.exp(a0["point"]) if np.isfinite(a0["point"]) else None,
                                             "interval": [math.exp(v) for v in a0["interval"]] if a0["interval"]
                                             else None}
    out["posthoc_fit"] = {"a": float(ols(y, x)[0]), "b": float(ols(y, x)[1])}
    return out


def height_lever(dev_strides, persons, demo, prior, g_s, fit, n_boot=None, seed=None):
    """Post hoc, descriptive (not in the prediction): S on a dev set with the decoded stride divided by the constant
    train54 gain g_s, and by the height-aware gain exp(a + b x_i) fitted on the train54 persons (fit {a, b}); the
    difference paired by person (g3_gain_check.paired_of)."""
    gc = GC()
    u = gc.subset(dev_strides, persons)
    xs = {p: math.log(prior[GENDER_NAME[demo[p]["sex"]]]["stature_m"] / demo[p]["height_m"]) for p in set(u["person_id"])}
    gi = np.exp(fit["a"] + fit["b"] * u["person_id"].map(xs).to_numpy(np.float64))
    e = np.abs(u["y"].to_numpy(np.float64) / gi - u["x"].to_numpy(np.float64))
    h_pp = {str(k): float(v) for k, v in u.assign(_e=e).groupby("person_id")["_e"].mean().items()}
    c_pp = gc.person_abs_error(u, g_s)
    const, height = gc.line_of(c_pp, n_boot, seed), gc.line_of(h_pp, n_boot, seed)
    diff = gc.paired_of(c_pp, h_pp, n_boot, seed)
    return {"constant_gain_cm": const["point"], "constant_interval": const["interval"],
            "height_gain_cm": height["point"], "height_interval": height["interval"],
            "constant_minus_height_cm": diff["point"], "paired_interval": diff["interval"], "n_persons": const["n_persons"]}


def cpu_test_compare(folder, train_events, locked, train):
    """The one-trial CPU test of the dump (float32) against F1's GPU decode (bf16 autocast) of the same trial: mean
    decoded stride difference (cm) and the cf minus base difference; None without the folder."""
    import pandas as pd
    if not os.path.exists(os.path.join(folder, "events_base.csv")):
        return None
    gc = GC()
    b, _, _ = gc.read_events(os.path.join(folder, "events_base.csv"), train, locked, row="base")
    c, _, _ = gc.read_events(os.path.join(folder, "events_cf.csv"), train, locked, row="cf")
    f, _, _ = gc.read_events(train_events, train, locked, row="base")
    k = ["trial_id", "foot", "pass_id", "x"]
    f = f[f["trial_id"].isin(set(b["trial_id"]))]
    m = b.merge(f, on=k, suffixes=("_cpu", "_gpu")).merge(c[k + ["y"]].rename(columns={"y": "y_cf"}), on=k)
    sh = pd.read_csv(os.path.join(folder, "shape.csv"), encoding="utf-8")
    return {"trials": sorted(set(b["trial_id"])), "strides": int(len(m)),
            "mean_cpu_minus_gpu_cm": float((m["y_cpu"] - m["y_gpu"]).mean()),
            "max_abs_cpu_minus_gpu_cm": float((m["y_cpu"] - m["y_gpu"]).abs().max()),
            "mean_cf_minus_base_cm": float((m["y_cf"] - m["y_cpu"]).mean()),
            "mean_walkway_cm": float(m["x"].mean()), "mean_decoded_cpu_cm": float(m["y_cpu"].mean()),
            "mean_decoded_gpu_cm": float(m["y_gpu"].mean()),
            "beta0_shift": float(sh["beta0_shift"].iloc[0]), "stature_dec_m": float(sh["stature_dec_m"].iloc[0])}


# ---------------------------------------------------------------- inputs
def events_paths(end):
    name = ENDS[end][0]
    return (os.path.join(TREE, "reports", "gates", "g3", f"diag_{name}", "events.csv"),
            os.path.join(F1_DIR, f"decode_{name}", "train_events.csv"))


def f1_gains(end, f1_results=None):
    """(g_S train54 per stride, in-sim stride gain) of an end, from F1's results.json."""
    with open(f1_results or F1_RESULTS, encoding="utf-8") as f:
        e = json.load(f)["ends"][end]
    return float(e["gain"]["stride"]["gain"]), float(e["insim"]["stride_gain"])


# ---------------------------------------------------------------- the GPU dump
def infer_two(model, wins, heights_m, refine_steps, points, batch_windows, seed, allow_eval=False):
    """The posterior of mdfm.eval.infer.infer for the windows of one trial (one shape group), decoded twice: with the
    pooled shape mean (base) and with beta_0 shifted to the reported height (cf; None when heights_m is not finite or
    the shift it needs lies beyond SHIFT_MAX, counterfactual_shape). Returns (joints_base, contacts, joints_cf or
    None, beta (16,), gender code, cf) with cf the counterfactual_shape dict without its cf_beta."""
    import torch
    from mdfm.eval import infer as I
    from mdfm.model.decoder import all_knots
    from mdfm.model.posterior import subject_pool
    from mdfm.model.refine import refine
    from mdfm.model import mfm as M
    for wd in wins:
        I.refuse_cody(wd.source, wd.person_id, allow_eval=allow_eval)
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(int(seed) % (1 << 63))
    B = max(1, int(batch_windows))
    kept = []
    with torch.no_grad():
        for a in range(0, len(wins), B):
            ch = wins[a:a + B]
            batch = M.batch_from_windows(ch, sources=model.cfg.sources, families=model.heads.families,
                                         rate=model.rate, allow_eval=allow_eval, gen=0).to(dev)
            with I._amp(dev, None):
                post = model.encode(batch)
                if refine_steps:
                    post = refine(model, post, batch, int(refine_steps), points_per_stream=points, generator=gen,
                                  history=[])
            kept.append((post.detach(), batch.lengths.to(dev), batch.tokens.n_patches.to(dev), batch.gender.to(dev)))
            del batch
        mu_b = torch.cat([k[0].mu_b for k in kept])
        lv_b = torch.cat([k[0].logvar_b for k in kept])
        gid = torch.zeros(len(wins), dtype=torch.int64, device=dev)
        mu_p, _ = subject_pool(mu_b, lv_b, gid)
        genders = torch.cat([k[3] for k in kept]).cpu().numpy()
        if len(set(int(g) for g in genders)) != 1:
            raise ValueError("windows of one trial with two sex codes")
        gcode = int(genders[0])
        beta = mu_p[0].float().cpu().numpy().astype(np.float64)
        if not np.allclose(mu_p.float().cpu().numpy(), beta[None], atol=1e-5):
            raise AssertionError("pooled shape differs across the windows of one group")
        cf = counterfactual_shape(beta, gcode, heights_m)
        cf_beta = cf.pop("cf_beta")
        res = {"base": [], "cf": [], "contact": []}
        a = 0
        for post, lengths, npw, gender in kept:
            Wc = post.mu_z.shape[0]
            mb = mu_p[a:a + Wc]
            a += Wc
            w, k = all_knots(lengths)
            mask = post.mask[..., None].to(post.mu_z.dtype)
            variants = [("base", mb)]
            if cf_beta is not None:
                variants.append(("cf", torch.as_tensor(cf_beta, dtype=mb.dtype, device=dev)[None].expand_as(mb)))
            cuts = torch.cumsum(lengths, 0).tolist()
            starts = [0] + cuts[:-1]
            for key, bt in variants:
                with I._amp(dev, None):
                    bo = model.decode(post.mu_z * mask, bt, (w, k), npw, lengths)
                pos, _ = I._pose(model, bo.rot6d, bo.transl, bt, gender, w)
                pos = pos.cpu().numpy()
                con = torch.sigmoid(bo.contact_logit.float()).cpu().numpy()
                for j, (s0, s1) in enumerate(zip(starts, cuts)):
                    wd = wins[len(res[key])]
                    n = min(int(round(wd.duration * model.rate)), s1 - s0)
                    res[key].append(pos[s0:s0 + n])
                    if key == "base":
                        res["contact"].append(con[s0:s0 + n])
    return (res["base"], res["contact"], res["cf"] if cf_beta is not None else None, beta, gcode, cf)


def dump(a):
    """The GPU dump (module docstring)."""
    check_prediction()
    import pandas as pd
    import torch
    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    t_start = time.time()
    sets, locked = load_sets()
    persons = list(sets["train54"]) + list(sets["dev14"])
    if a.persons:
        persons = [p for p in persons if p in set(a.persons)]
    if not persons:
        raise ValueError("no person to decode")
    GC().guard_persons(persons, set(sets["train54"]) | set(sets["dev14"]), locked, "dump")
    demo = read_demographics(persons, locked)
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb) if device.type == "cuda" else None
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    out = a.out or os.path.join(OUT, f"dump_{loaded.name}")
    paths.assert_not_moveatlas(out)
    tg = g3.load_targets(persons)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError("lockbox target rows kept")
    rate = loaded.model.rate
    shape_rows, log, parts_b, parts_c, skipped = [], [], [], [], []
    n_done = 0
    for tid, rec in g3.imu_records(persons, tg.trials):
        GC().check_record(tid, rec, persons)
        if a.max_trials and n_done >= a.max_trials:
            break
        t0 = time.time()
        wins, sp, fps = I.record_windows(g3.model_input(rec))
        if not wins:
            continue
        p = str(rec.get("person_id"))
        h = demo.get(p, {}).get("height_m", float("nan"))
        jb, con, jc, beta, gcode, cf = infer_two(
            loaded.model, wins, h, a.refine_steps, a.points, a.batch_windows, I.seed_of(a.seed, tid))
        starts = [int(round(s / fps * rate)) for s, _ in sp]
        parts_b.append(g3.body_predictions([g3.stitched_body(tid, jb, con, None, starts, rate)], tg))
        if jc is not None:
            parts_c.append(g3.body_predictions([g3.stitched_body(tid, jc, con, None, starts, rate)], tg))
        st, leg = stature_of(beta, gcode)
        shape_rows.append({"trial_id": tid, "person_id": p, "gender_code": gcode, "windows": len(wins),
                           **{f"beta_{i}": float(beta[i]) for i in range(16)},
                           "stature_dec_m": float(st), "leg_dec_m": float(leg), "has_height": bool(math.isfinite(h)),
                           "cf_status": cf["status"], "beta0_shift": cf["shift"],
                           "beta0_shift_required": cf["required_shift"],
                           "cf_stature_error_m": cf["stature_cf_m"] - h})
        if cf["status"] == "skipped_shift":
            # recorded, never clamped: the trial keeps its base decode and gets no counterfactual one
            skipped.append({"trial_id": tid, "person_id": p, "required_beta0_shift": cf["required_shift"],
                            "shift_max": SHIFT_MAX, "reported_height_m": float(h),
                            "reported_height_in": float(h) / IN_TO_M, "stature_dec_m": float(st),
                            "gender_code": gcode})
        log.append({"trial_id": tid, "person_id": p, "windows": len(wins), "seconds": time.time() - t0,
                    "cf_status": cf["status"]})
        n_done += 1
        note = "" if cf["status"] != "skipped_shift" else (
            f"; counterfactual skipped: required beta_0 shift {fmt(cf['required_shift'], 2)} beyond {SHIFT_MAX} "
            f"(reported height {float(h):.3f} m)")
        print(f"  {tid}: {len(wins)} windows, beta_0 {beta[0]:+.2f}, stature {float(st):.3f} m{note}",
              flush=True)
        del rec
    if not parts_b:
        raise ValueError("no trial decoded")
    D = _sibling("eval_g3_diag")

    def events(parts, row):
        cat = lambda k: pd.concat([getattr(q, k) for q in parts], ignore_index=True)  # noqa: E731
        P = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds"))
        return D.events_frame(row, {"strides": g3.match_strides(P, tg), "speeds": g3.match_speeds(P, tg)}, tg)

    common.write_csv(events(parts_b, "base"), os.path.join(out, "events_base.csv"))
    if parts_c:
        common.write_csv(events(parts_c, "cf"), os.path.join(out, "events_cf.csv"))
    common.write_csv(pd.DataFrame(shape_rows), os.path.join(out, "shape.csv"))
    common.write_csv(pd.DataFrame(log), os.path.join(out, "trials.csv"))
    common.write_csv(pd.DataFrame(skipped, columns=SKIPPED_COLUMNS), os.path.join(out, "skipped_cf.csv"))
    doc = {"what": LABEL, "task": TASK, "mode": "dump", "checkpoint": loaded.describe(),
           "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "batch_windows": a.batch_windows,
                         "seed": a.seed, "shape_pooling": "per trial", "weights": a.weights, "device": str(device),
                         "cap_gb": cap},
           "prediction_sha256": PREDICTION_SHA256, "persons": len(persons),
           "sets_sha256": {"train54": sources.person_hash(sets["train54"]), "dev14": sources.person_hash(sets["dev14"])},
           "trials": len(log), "partial": bool(a.persons or a.max_trials),
           "cf_skipped": {"shift_max": SHIFT_MAX, "n_trials": len(skipped),
                          "trials": [{k: v for k, v in s.items() if k in ("trial_id", "person_id", "required_beta0_shift")}
                                     for s in skipped],
                          "file": "skipped_cf.csv",
                          "rule": "a trial whose counterfactual needs a beta_0 shift beyond shift_max keeps its base "
                                  "decode and gets no counterfactual decode; never clamped"},
           "timing_s": round(time.time() - t_start, 1)}
    common.write_text(os.path.join(out, "dump.json"), json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    prov = {"script": "scripts/f2_H1shape.py", "mode": "dump", "checkpoint": rel(loaded.path),
            "checkpoint_sha256": loaded.sha256, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "torch": torch.__version__, "moveatlas_stamp": GC().stamp_moveatlas(),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
    print(f"dumped {len(log)} trials to {rel(out)}")
    return out


def job_command():
    name, ck, bw = ENDS[DUMP_END]
    out = f"reports/bench/f2_scale/H1shape/dump_{name}"
    return ["scripts/queue_job.py", "--name", JOB["name"], "--cls", JOB["cls"], "--priority", JOB["priority"],
            "--est-hours", JOB["est_hours"], "--group", JOB["group"], "--requires", ck,
            "--outputs", f"{out}/shape.csv", f"{out}/events_base.csv", f"{out}/events_cf.csv",
            f"{out}/skipped_cf.csv",
            "--note", "F2-H1shape: decoded shape per trial and a decode with the stature set to the reported height, "
                      "train54 and dev14 controls; r2 of f2_h1shape_dump_v1_B (task F2D): a trial whose height needs a "
                      "beta_0 shift beyond 6 is skipped and recorded, never clamped",
            "--", "scripts/f2_H1shape.py", "--dump", "--checkpoint", ck, "--batch-windows", str(bw),
            "--memory-gb", JOB["memory_gb"], "--out", out]


# ---------------------------------------------------------------- the dump read (D1, D2)
def _d1(t, n_boot=None, seed=None):
    d1 = boot_coef(t["stature_dec_m"].to_numpy(), t["height_m"].to_numpy(), 1, n_boot, seed)
    d1["holds"] = bool(d1["interval"] is not None and d1["interval"][1] < D1_MAX)
    d1["n_persons"] = int(len(t))
    return d1


def read_dump(folder, sets, locked, demo, g_sim, n_boot=None, seed=None, flagged=()):
    """D1 and D2 from a dump folder (None when it does not exist; a status only when it is partial).

    D1 as registered (every train54 person with a reported height). D2 compares the two decodes of the same
    posteriors: g_cf over the train54 strides of the cf decode against g_base over the strides of the same trials
    (the reading; trials without a cf decode, cf_skipped of dump.json, are out of both sides), with g_base over every
    train54 trial of the base decode beside (the registered wording 'the dump's own base decode'). Post hoc beside:
    D1 without the persons of `flagged` (the height screen of the analysis)."""
    import pandas as pd
    if not os.path.exists(os.path.join(folder, "dump.json")):
        return None
    with open(os.path.join(folder, "dump.json"), encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("partial"):
        return {"status": "partial dump, not read"}
    allowed = set(sets["train54"]) | set(sets["dev14"])
    sh = pd.read_csv(os.path.join(folder, "shape.csv"), encoding="utf-8", dtype={"person_id": str, "trial_id": str})
    GC().guard_persons(set(sh["person_id"]), allowed, locked, "shape.csv")
    gc = GC()
    base, _, _ = gc.read_events(os.path.join(folder, "events_base.csv"), allowed, locked, row="base")
    cf, _, _ = gc.read_events(os.path.join(folder, "events_cf.csv"), allowed, locked, row="cf")
    tr = set(sets["train54"])
    if "cf_status" in sh.columns:
        cf_trials = set(sh.loc[sh["cf_status"] == "done", "trial_id"])
    else:                                              # a dump written before the skip rule: every trial with a height
        cf_trials = set(sh.loc[sh["has_height"].astype(str).str.lower() == "true", "trial_id"])
    skipped = list((doc.get("cf_skipped") or {}).get("trials") or [])
    sk_path = os.path.join(folder, "skipped_cf.csv")
    if os.path.exists(sk_path):
        sk = pd.read_csv(sk_path, encoding="utf-8", dtype={"person_id": str, "trial_id": str})
        GC().guard_persons(set(sk["person_id"]), allowed, locked, "skipped_cf.csv")
        if set(sk["trial_id"]) != {s["trial_id"] for s in skipped}:
            raise AssertionError("skipped_cf.csv and dump.json cf_skipped name different trials")
    pp = sh.groupby("person_id").agg(stature_dec_m=("stature_dec_m", "mean"), leg_dec_m=("leg_dec_m", "mean"),
                                     beta0_shift=("beta0_shift", "mean")).reset_index()
    pp["height_m"] = [demo.get(p, {}).get("height_m", float("nan")) for p in pp["person_id"]]
    t = pp[pp["person_id"].isin(tr) & np.isfinite(pp["height_m"])]
    d1 = _d1(t, n_boot, seed)
    bt = gc.subset(base, tr)
    ct = gc.subset(cf, tr)
    bp = bt[bt["trial_id"].isin(cf_trials)]
    gb_all = gc.median_ratio(bt["x"], bt["y"])
    gb = gc.median_ratio(bp["x"], bp["y"])
    gcf = gc.median_ratio(ct["x"], ct["y"])
    need = D2_FRAC * (g_sim - gb)
    need_all = D2_FRAC * (g_sim - gb_all)
    d2 = {"g_base": gb, "g_base_all_trials": gb_all, "g_cf": gcf, "g_sim": g_sim, "gain_change": gcf - gb,
          "needed": need, "holds": bool(np.isfinite(gcf) and gcf - gb >= need),
          "gain_change_against_all_trials": gcf - gb_all, "needed_against_all_trials": need_all,
          "holds_against_all_trials": bool(np.isfinite(gcf) and gcf - gb_all >= need_all),
          "strides_base_paired": int(len(bp)), "strides_base_all": int(len(bt)), "strides_cf": int(len(ct)),
          "mean_cf_minus_base_cm_paired": _paired_mean_diff(bp, ct)}
    out = {"status": "read", "D1": d1, "D2": d2, "n_train_persons": int(len(t)),
           "cf_skipped": {"n_trials": len(skipped), "trials": skipped,
                          "train54_trials": sorted(s["trial_id"] for s in skipped if s["person_id"] in tr)},
           "median_decoded_stature_over_height": float(np.median(t["stature_dec_m"] / t["height_m"])),
           "median_decoded_stature_m": float(np.median(t["stature_dec_m"])),
           "decoded_stature_sd_m": float(np.std(t["stature_dec_m"], ddof=1)) if len(t) > 1 else float("nan"),
           "height_sd_m": float(np.std(t["height_m"], ddof=1)) if len(t) > 1 else float("nan"),
           "median_beta0_shift": float(np.nanmedian(t["beta0_shift"]))}
    fl = sorted(set(flagged) & set(t["person_id"]))
    if fl:
        out["posthoc_D1_without_flagged"] = {**_d1(t[~t["person_id"].isin(fl)], n_boot, seed), "left_out": fl}
    return out


def _paired_mean_diff(base, cf):
    """Mean of cf minus base decoded stride (cm) over the strides present in both (trial, foot, pass, walkway)."""
    k = ["trial_id", "foot", "pass_id", "x"]
    m = base.merge(cf[k + ["y"]].rename(columns={"y": "y_cf"}), on=k)
    return float((m["y_cf"] - m["y"]).mean()) if len(m) else float("nan")


# ---------------------------------------------------------------- post hoc: the reported heights
def ft_in_reading(height_in):
    """The height in inches if a two-digit entry were feet and inches written without a separator (47 read as 4 ft
    7 in = 55 in); NaN when the entry is not of that form (an integer from 40 to 69 whose second digit is at most 9
    and first digit 4 to 6)."""
    if not (isinstance(height_in, (int, float)) and math.isfinite(height_in)) or height_in != int(height_in):
        return float("nan")
    ft, inch = divmod(int(height_in), 10)
    return float(12 * ft + inch) if 4 <= ft <= 6 else float("nan")


def _bmi(kg, m):
    return float(kg / m ** 2) if math.isfinite(kg) and math.isfinite(m) and m > 0 else float("nan")


def height_check(sets, locked, demo, prior, tables, dev_strides, n_boot=None, seed=None):
    """Post hoc (not in the prediction; it changes no registered reading): the reported heights screened against the
    statures the counterfactual decode can reach (reachable_statures, the dump's SHIFT_MAX); for each flagged person
    a unit or entry check (weight and BMI under the reported height and under a feet-and-inches reading of the entry,
    the person's x and walkway stride against the others of train54); then C1, C2, C3, the ratio at the prior
    stature and the height-aware lever recomputed without the flagged persons, at each end."""
    import pandas as pd
    persons = sorted(set(sets["dev14"]) | set(sets["train54"]))
    reach = reachable_statures()
    flagged = height_flags(demo, persons, reach)
    out = {"what": "post hoc, not in the prediction; changes no registered reading",
           "rule": (f"reported height outside the rest-mesh statures reached from the prior mean betas of the sex with "
                    f"|beta_0 shift| <= {SHIFT_MAX} (the dump's bound)"),
           "reachable_statures_m": {g: list(v) for g, v in reach.items()}, "n_screened": len(persons),
           "flagged": flagged, "persons": {}, "ends": {}}
    if not flagged:
        return out
    w = read_demographics(flagged, locked, weight=True)
    tab = pd.concat(tables, ignore_index=True)
    prim = tab[(tab["end"] == PRIMARY) & (tab["set"] == "train54")]
    for p in flagged:
        v = w[p]
        h_in, kg, sx = v["height_in"], v["weight_kg"], v["sex"]
        alt = ft_in_reading(h_in)
        same = [demo[q]["height_m"] for q in sets["train54"]
                if q != p and demo.get(q, {}).get("sex") == sx and math.isfinite(demo[q]["height_m"])]
        row = tab[(tab["person_id"] == p) & (tab["end"] == PRIMARY)]
        others = prim[~prim["person_id"].isin(flagged)]
        out["persons"][p] = {
            "set": "train54" if p in sets["train54"] else "dev14", "sex": sx, "reported_height_in": h_in,
            "reported_height_m": v["height_m"], "weight_kg": kg, "bmi_at_reported_height": _bmi(kg, v["height_m"]),
            "feet_inches_reading_in": alt, "feet_inches_reading_m": alt * IN_TO_M,
            "bmi_at_feet_inches_reading": _bmi(kg, alt * IN_TO_M),
            "reachable_statures_m": list(reach[GENDER_NAME[sx]]),
            "same_sex_train54_others": {"n": len(same), "min_m": float(min(same)) if same else float("nan"),
                                        "median_m": float(np.median(same)) if same else float("nan"),
                                        "max_m": float(max(same)) if same else float("nan")},
            "x": float(row["x"].iloc[0]) if len(row) else float("nan"),
            "x_range_of_the_others_train54": [float(others["x"].min()), float(others["x"].max())],
            "ratio_r_v1_B": float(row["r"].iloc[0]) if len(row) else float("nan"),
            "walkway_stride_cm": float(row["S_cm"].iloc[0]) if len(row) else float("nan"),
            "train54_median_walkway_stride_cm": float(prim["S_cm"].median())}
    for end in ENDS:
        g_s, g_sim = f1_gains(end)
        tt = tab[(tab["end"] == end) & (tab["set"] == "train54") & ~tab["person_id"].isin(flagged)]
        c = criteria(tt.drop(columns=["end", "set"]), g_s, g_sim, n_boot, seed)
        e = {"train54_without_flagged": c,
             "verdict_without_flagged": verdict_of(c["C1"]["holds"], c["C2"]["holds"], c["C3"]["holds"]),
             "height_lever_without_flagged": {
                 s: height_lever(dev_strides[end], [q for q in sets[s] if q not in flagged], demo, prior, g_s,
                                 c["posthoc_fit"], n_boot, seed) for s in ("dev12", "dev14")}}
        out["ends"][end] = e
    return out


# ---------------------------------------------------------------- the analysis
def run(n_boot=None, seed=None):
    import pandas as pd
    check_prediction()
    sets, locked = load_sets()
    persons = sorted(set(sets["dev14"]) | set(sets["train54"]))
    prior = prior_statures()
    amass, n_files = amass_train_persons()
    demo = read_demographics(persons, locked)
    gc = GC()
    res = {"what": LABEL, "task": TASK, "prediction_sha256": PREDICTION_SHA256,
           "prior_statures": prior, "leg_ratio": LEG_RATIO,
           "amass_train": {"persons": int(len(amass)), "shards": n_files,
                           **{g: {"n": int((amass["gender"] == g).sum()),
                                  "median_stature_m": float(amass.loc[amass["gender"] == g, "stature_m"].median()),
                                  "q10": float(amass.loc[amass["gender"] == g, "stature_m"].quantile(0.1)),
                                  "q90": float(amass.loc[amass["gender"] == g, "stature_m"].quantile(0.9)),
                                  "median_leg_m": float(amass.loc[amass["gender"] == g, "leg_m"].median())}
                              for g in ("female", "male")}},
           "demographics": {"persons_found": len(demo), "persons_asked": len(persons),
                            "with_height": int(sum(np.isfinite(v["height_m"]) for v in demo.values())),
                            "sex_counts": {s: int(sum(v["sex"] == s for v in demo.values())) for s in ("F", "M", "")}},
           "ends": {}}
    for sx, g in (("F", "female"), ("M", "male")):
        hs = np.array([v["height_m"] for p, v in demo.items() if v["sex"] == sx and p in set(sets["train54"])])
        hs = hs[np.isfinite(hs)]
        res["demographics"][f"train54_{g}_median_height_m"] = float(np.median(hs)) if len(hs) else float("nan")
        a_st = amass.loc[amass["gender"] == g, "stature_m"].to_numpy()
        res["demographics"][f"train54_{g}_share_above_amass_median"] = (
            float(np.mean(hs > np.median(a_st))) if len(hs) else float("nan"))
    tables, dev_strides = [], {}
    for end in ENDS:
        dev_path, train_path = events_paths(end)
        g_s, g_sim = f1_gains(end)
        dst, _, _ = gc.read_events(dev_path, sets["dev14"], locked)
        tst, _, _ = gc.read_events(train_path, sets["train54"], locked)
        tt = person_table(tst, demo, prior)
        dt = person_table(dst, demo, prior)
        res["ends"][end] = {"g_S": g_s, "g_sim": g_sim,
                            "train54": criteria(tt, g_s, g_sim, n_boot, seed),
                            "dev14": criteria(dt, g_s, g_sim, n_boot, seed)}
        fit = res["ends"][end]["train54"]["posthoc_fit"]
        res["ends"][end]["posthoc_height_lever"] = {
            s: height_lever(dst, sets[s], demo, prior, g_s, fit, n_boot, seed) for s in ("dev12", "dev14")}
        dev_strides[end] = dst
        if end == DUMP_END:
            res["cpu_test"] = cpu_test_compare(os.path.join(OUT, "cpu_test_dump_HC100"), train_path, locked,
                                               sets["train54"])
        tables += [tt.assign(end=end, set="train54"), dt.assign(end=end, set="dev14")]
    p = res["ends"][PRIMARY]["train54"]["verdict_inputs"]
    res["verdict"] = verdict_of(p["C1"], p["C2"], p["C3"])
    res["posthoc_height_check"] = height_check(sets, locked, demo, prior, tables, dev_strides, n_boot, seed)
    res["dump"] = read_dump(os.path.join(OUT, f"dump_{ENDS[DUMP_END][0]}"), sets, locked, demo,
                            f1_gains(DUMP_END)[1], n_boot, seed, flagged=res["posthoc_height_check"]["flagged"])
    n_lb = len(set(persons) & set(locked))
    if n_lb:
        raise AssertionError("lockbox persons in the sets")
    res["lockbox"] = {"persons_in_lockbox": 0, "asserted": True}
    return res, pd.concat(tables, ignore_index=True), amass


def fmt(v, d=3):
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not math.isfinite(x) else f"{x:.{d}f}"


def iv(c, d=3):
    return "n/a" if not c.get("interval") else f"[{fmt(c['interval'][0], d)}, {fmt(c['interval'][1], d)}]"


def summary_md(r):
    L = [f"# F2-H1shape: body shape and the G3 stride scale ({LABEL})", "",
         "Written by scripts/f2_H1shape.py; every number below is read from results.json of this folder. "
         f"Prediction: reports/weekend_2026-10-01/tasks/F2-H1shape.md, section SHA-256 {r['prediction_sha256']}, "
         "asserted before any value was read. No lockbox person, EVAL or CODY record was read (asserted).", "",
         f"**Verdict (v1_B, train54): {r['verdict']}**", "",
         "## Statures", "",
         "| body | stature (m) | leg, hip joint above sole (m) |", "|---|---|---|"]
    for g, v in r["prior_statures"].items():
        L.append(f"| prior mean betas, {g} ({v['n_persons']} persons) | {fmt(v['stature_m'])} | {fmt(v['leg_m'])} |")
    a = r["amass_train"]
    L += ["", f"AMASS train persons of the body store: {a['persons']}.", "",
          "| sex | persons | median stature | 10th pct | 90th pct | median leg |", "|---|---|---|---|---|---|"]
    for g in ("female", "male"):
        v = a[g]
        L.append(f"| {g} | {v['n']} | {fmt(v['median_stature_m'])} | {fmt(v['q10'])} | {fmt(v['q90'])} | "
                 f"{fmt(v['median_leg_m'])} |")
    d = r["demographics"]
    L += ["", f"weargait_pd persons found {d['persons_found']} of {d['persons_asked']}, with a height "
          f"{d['with_height']}; sex counts {d['sex_counts']}. train54 median height: female "
          f"{fmt(d['train54_female_median_height_m'])} m, male {fmt(d['train54_male_median_height_m'])} m; share "
          f"above the AMASS median stature of their sex: female {fmt(d['train54_female_share_above_amass_median'], 2)}, "
          f"male {fmt(d['train54_male_share_above_amass_median'], 2)}.", "",
          "## Criteria", "",
          "| end | set | persons | median r | C1 slope b1 | 95 % | C3 slope b3 (held for log S) | 95 % | "
          "log S coef | 95 % | G_H1 | f | C1 | C2 | C3 | R2 x | R2 log S | R2 both |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for end, e in r["ends"].items():
        for s in ("train54", "dev14"):
            c = e[s]
            L.append(f"| {end} | {s} | {c['n_persons']} | {fmt(c['median_r'])} | {fmt(c['C1']['point'])} | "
                     f"{iv(c['C1'])} | {fmt(c['C3']['point'])} | {iv(c['C3'])} | {fmt(c['C3_log_stride_coef']['point'])} | "
                     f"{iv(c['C3_log_stride_coef'])} | {fmt(c['C2']['G_H1'])} | {fmt(c['C2']['f'])} | "
                     f"{c['C1']['holds']} | {c['C2']['holds']} | {c['C3']['holds']} | {fmt(c['r2_x'])} | "
                     f"{fmt(c['r2_log_stride'])} | {fmt(c['r2_both'])} |")
    L += ["", "| end | set | Pearson r of ratio with height | within female (n, r) | within male (n, r) | "
          "Pearson log S with x | slope of log r on log S alone | 95 % |", "|---|---|---|---|---|---|---|---|"]
    for end, e in r["ends"].items():
        for s in ("train54", "dev14"):
            c = e[s]
            L.append(f"| {end} | {s} | {fmt(c['pearson_r_height'])} | {c['within_sex_F']['n']}, "
                     f"{fmt(c['within_sex_F']['pearson_r_height'])} | {c['within_sex_M']['n']}, "
                     f"{fmt(c['within_sex_M']['pearson_r_height'])} | {fmt(c['pearson_logS_x'])} | "
                     f"{fmt(c['stride_only_slope']['point'])} | {iv(c['stride_only_slope'])} |")
    L += ["", "## Post hoc, descriptive (not in the prediction; they change no criterion)", "",
          "Ratio of a person as tall as the prior body of the sex (exp of the intercept of log r on x), and S on the "
          "dev sets with the decoded stride divided by F1's constant train54 gain or by the height-aware gain "
          "exp(a + b x) fitted on the train54 persons.", "",
          "| end | ratio at prior stature (train54) | 95 % | set | S, constant gain (cm) | 95 % | S, height-aware "
          "gain (cm) | 95 % | constant minus height-aware | 95 % paired |", "|---|---|---|---|---|---|---|---|---|---|"]
    for end, e in r["ends"].items():
        rp = e["train54"]["posthoc_ratio_at_prior_stature"]
        for s, h in e["posthoc_height_lever"].items():
            L.append(f"| {end} | {fmt(rp['point'])} | {iv(rp)} | {s} | {fmt(h['constant_gain_cm'], 2)} | "
                     f"{iv({'interval': h['constant_interval']}, 2)} | {fmt(h['height_gain_cm'], 2)} | "
                     f"{iv({'interval': h['height_interval']}, 2)} | {fmt(h['constant_minus_height_cm'], 2)} | "
                     f"{iv({'interval': h['paired_interval']}, 2)} |")
    ct = r.get("cpu_test")
    if ct:
        L += ["", f"CPU test of the dump (cpu_test_dump_HC100, one trial {', '.join(ct['trials'])}, float32 on the "
              f"CPU) against F1's GPU decode (bf16 autocast) of the same trial, {ct['strides']} matched strides: mean "
              f"walkway {fmt(ct['mean_walkway_cm'], 2)} cm, decoded GPU {fmt(ct['mean_decoded_gpu_cm'], 2)} cm, "
              f"decoded CPU {fmt(ct['mean_decoded_cpu_cm'], 2)} cm (CPU minus GPU mean "
              f"{fmt(ct['mean_cpu_minus_gpu_cm'], 2)}, largest {fmt(ct['max_abs_cpu_minus_gpu_cm'], 2)} cm); stature set "
              f"to the height (beta_0 shift {fmt(ct['beta0_shift'], 2)}) moves the decoded stride by "
              f"{fmt(ct['mean_cf_minus_base_cm'], 2)} cm on average."]
    hc = r.get("posthoc_height_check")
    if hc:
        L += ["", "## Post hoc: reported heights the counterfactual decode cannot reach (not in the prediction)", "",
              f"Rule: {hc['rule']}. Reachable statures: " + "; ".join(
                  f"{g} [{fmt(v[0])}, {fmt(v[1])}] m" for g, v in hc["reachable_statures_m"].items())
              + f". Persons screened {hc['n_screened']}, flagged {len(hc['flagged'])}"
              + (f": {', '.join(hc['flagged'])}." if hc["flagged"] else ".")]
        for p, v in hc["persons"].items():
            s = v["same_sex_train54_others"]
            L += ["", f"- {p} ({v['set']}, sex {v['sex']}): reported {fmt(v['reported_height_in'], 0)} in = "
                  f"{fmt(v['reported_height_m'])} m, weight {fmt(v['weight_kg'], 0)} kg, BMI at the reported height "
                  f"{fmt(v['bmi_at_reported_height'], 1)}; read as feet and inches without a separator "
                  f"{fmt(v['feet_inches_reading_in'], 0)} in = {fmt(v['feet_inches_reading_m'])} m, BMI "
                  f"{fmt(v['bmi_at_feet_inches_reading'], 1)}; the other train54 persons of the sex: {s['n']}, heights "
                  f"{fmt(s['min_m'])} to {fmt(s['max_m'])} m (median {fmt(s['median_m'])}); x {fmt(v['x'])} against "
                  f"[{fmt(v['x_range_of_the_others_train54'][0])}, {fmt(v['x_range_of_the_others_train54'][1])}] for "
                  f"the others; ratio r {fmt(v['ratio_r_v1_B'])} (v1_B); walkway stride {fmt(v['walkway_stride_cm'], 1)} "
                  f"cm (train54 median {fmt(v['train54_median_walkway_stride_cm'], 1)})."]
        if hc["ends"]:
            L += ["", "Criteria without the flagged persons (train54):", "",
                  "| end | persons | C1 slope b1 | 95 % | C3 slope b3 | 95 % | log S coef | f | C1 | C2 | C3 | verdict | "
                  "ratio at prior stature | 95 % | dev12 S constant (cm) | dev12 S height-aware (cm) | paired | 95 % |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for end, e in hc["ends"].items():
                c, h = e["train54_without_flagged"], e["height_lever_without_flagged"]["dev12"]
                rp = c["posthoc_ratio_at_prior_stature"]
                L.append(f"| {end} | {c['n_persons']} | {fmt(c['C1']['point'])} | {iv(c['C1'])} | {fmt(c['C3']['point'])} | "
                         f"{iv(c['C3'])} | {fmt(c['C3_log_stride_coef']['point'])} | {fmt(c['C2']['f'])} | "
                         f"{c['C1']['holds']} | {c['C2']['holds']} | {c['C3']['holds']} | {e['verdict_without_flagged']} | "
                         f"{fmt(rp['point'])} | {iv(rp)} | {fmt(h['constant_gain_cm'], 2)} | {fmt(h['height_gain_cm'], 2)} | "
                         f"{fmt(h['constant_minus_height_cm'], 2)} | {iv({'interval': h['paired_interval']}, 2)} |")
    L += ["", "## Decoded shape (GPU dump)", ""]
    dmp = r.get("dump")
    if not dmp:
        L.append(f"Not read: the dump of the queued job ({JOB['name']}) does not exist yet.")
    elif dmp.get("status") != "read":
        L.append(f"Not read: {dmp.get('status')}.")
    else:
        d2 = dmp["D2"]
        L += [f"D1 slope of decoded stature on height {fmt(dmp['D1']['point'])} {iv(dmp['D1'])}, holds "
              f"{dmp['D1']['holds']} ({dmp['D1'].get('n_persons', dmp['n_train_persons'])} persons); median decoded "
              f"stature over height {fmt(dmp['median_decoded_stature_over_height'])}.",
              f"D2 stride gain base {fmt(d2['g_base'])} (trials with a counterfactual decode; every base trial "
              f"{fmt(d2['g_base_all_trials'])}), with the stature set to the height {fmt(d2['g_cf'])}, needed change "
              f"{fmt(d2['needed'])}, holds {d2['holds']} (against every base trial: {d2['holds_against_all_trials']}); "
              f"mean paired change of the decoded stride {fmt(d2['mean_cf_minus_base_cm_paired'], 2)} cm.",
              f"Trials without a counterfactual decode (shift beyond {SHIFT_MAX}): {dmp['cf_skipped']['n_trials']}"
              + (": " + ", ".join(f"{s['trial_id']} (required {fmt(s['required_beta0_shift'], 2)})"
                                  for s in dmp["cf_skipped"]["trials"]) if dmp["cf_skipped"]["trials"] else "") + "."]
        if dmp.get("posthoc_D1_without_flagged"):
            q = dmp["posthoc_D1_without_flagged"]
            L.append(f"Post hoc: D1 without {', '.join(q['left_out'])}: {fmt(q['point'])} {iv(q)}, holds {q['holds']}.")
    return "\n".join(L) + "\n"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", action="store_true", help="the GPU dump of the decoded shape (queued)")
    p.add_argument("--print-job", action="store_true", help="print the queue_job.py command of the dump")
    p.add_argument("--checkpoint", default=ENDS[DUMP_END][1])
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=12.0)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--batch-windows", type=int, default=ENDS[DUMP_END][2])
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--persons", nargs="*", default=None, help="--dump: a subset (a test run; marked partial)")
    p.add_argument("--max-trials", type=int, default=0, help="--dump: stop after this many trials (test; partial)")
    p.add_argument("--out", default=None)
    p.add_argument("--results-out", default=None,
                   help="analysis: the folder of results.json, summary.md, persons.csv, amass_statures.csv and "
                        "_provenance.json (default reports/bench/f2_scale/H1shape/); a rerun names a new folder")
    p.add_argument("--n-boot", type=int, default=BOOT_N)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_job:
        print(" ".join(["<conda>/envs/mdfm/python.exe"] + [f'"{x}"' if " " in x else x
                                                                             for x in job_command()]))
        return 0
    if a.dump:
        dump(a)
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the analysis runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    from mdfm.gates import common
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    o = os.path.abspath(a.results_out) if a.results_out else OUT
    paths.assert_not_moveatlas(o)
    res, table, amass = run(a.n_boot)
    res["run_started"] = started
    res["results_folder"] = rel(o)
    res["moveatlas"] = {"head": (GC().stamp_moveatlas() or {}).get("head")}
    res["timing_s"] = round(time.time() - t0, 1)
    res = clean(res)
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    if chr(0x2013) in text:
        raise AssertionError("summary.md holds an en-dash")
    common.write_text(os.path.join(o, "results.json"), json.dumps(res, indent=1, ensure_ascii=False,
                                                                   allow_nan=False) + "\n")
    # per person: the stride columns only (no reported height, sex or quantity derived from them; the flagged
    # persons of the post hoc height check carry theirs in results.json, the check's purpose)
    common.write_csv(table[["end", "set", "person_id", "r", "n_strides", "S_cm", "decoded_cm"]],
                     os.path.join(o, "persons.csv"))
    common.write_csv(amass, os.path.join(o, "amass_statures.csv"))
    common.write_text(os.path.join(o, "summary.md"), text)
    prov = {"script": "scripts/f2_H1shape.py", "mode": "analysis", "label": LABEL, "run_at": started,
            "prediction_sha256": PREDICTION_SHA256, "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_stamp": GC().stamp_moveatlas(), "python": sys.version.split()[0], "numpy": np.__version__,
            "inputs": {e: [rel(x) for x in events_paths(e)] for e in ENDS} | {"f1": rel(F1_RESULTS),
                                                                              "priors": rel(PRIORS)},
            "code_sha256": common.code_sha256([os.path.abspath(__file__)]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(o, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    print(text)
    print(f"written {rel(o)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
