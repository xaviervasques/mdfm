"""F2-H2sensor (2026-10-05): is the G3 scale error (F1: decoded stride about 0.85 of the walkway stride on real
weargait_pd input, 0.97 in the model's own synthetic domain) a sensor amplitude-scale error? Do the real IMU
signals carry a smaller amplitude than the synthetic renders at the same walking speed? Label of every output:
'dev diagnostic, not a G3 read'. CPU only (CUDA_VISIBLE_DEVICES=-1 is set before torch loads); no model is run.

Pre-registration: the Prediction section of reports/weekend_2026-10-01/tasks/F2-H2sensor.md, frozen in
reports/bench/f2_scale/H2sensor/prediction.txt with its SHA-256 in prediction.sha256 and here in PREDICTION_SHA256;
check_prediction asserts the three agree before any target or signal value is read.

Persons: train54 (primary) and dev14 (replication), g3.controls('train') and g3.controls('dev'), asserted equal to
the SHA-256 of reports/bench/g3_gain_check/results.json and disjoint from the weargait_pd lockbox list (g3 and the
window store's own list) before any row value is read. Every reader keeps a row only after its person id is
checked against the set; a kept row of a lockbox person or with a lockbox flag raises.

Real units (read_real): the weargait_pd IMU records of the window store (cache/window_store/weargait_pd/imu, the
band copy the model reads: acc in g with gravity, gyro in deg/s, 100 Hz), tasks SelfPace and HurriedPace. A unit is
a walkway pass of mdfm.gates.g3.load_targets: samples from its first to its last non-partial footfall (t0, t1 on
the record clock), its speed the pass speed, its cadence 120 / mean duration of the pass's walkway strides. Rest:
the Standing runs of reports/weargait_targets/phases.csv (read through g3.read_rows and its lockbox check).

Synthetic units (read_synth): the 745 AMASS train walks of reports/amass_walks/takes.csv rendered by
scripts/g2_renderer_realism.rendered_takes(None) (reference renderer imu_render.render, device model of
reports/g2_renderer/device_noise.json, 12 Hz body clock as in the body store, steady-walking frames only, the
others NaN), one unit per take, speed and cadence of takes.csv. Every feature is invariant to the sensor rotation,
so the mounting does not enter.

Features (unit_features) per site and unit, on the samples after lowpass_runs (4th-order Butterworth, zero phase,
on each valid run; a run shorter than its padlen is dropped) at CUTOFF_HZ (primary) and unfiltered (secondary):
G_pc half the 2.5 to 97.5 percentile range of the angular rate projected on its first principal axis (deg/s);
G_p95 the 95th percentile of the angular-rate norm (deg/s); A_dyn the RMS norm of the specific force minus its mean
vector (g); A_p95 and A_med the 95th percentile and median of the specific-force norm (g). A unit needs MIN_SAMPLES
samples at a site.

Matched speed (fit_power, ratios): per site and feature, log f = a + b log v (and, secondary, + c log cadence) by
least squares over the synthetic units with v in SYNTH_V; per real unit the amplitude ratio q = f / exp(fit) and
the implied speed ratio s = q^(1 / b) (NaN when b < MIN_B; with the cadence term the speed exponent b). Per person
and site the median over its units; per person the median over the LEG sites present; the median over persons
with a person bootstrap interval (BOOT_N, SEED). Primary units: pass speed within PRIMARY_V.

Rest gravity (rest_gravity): per person and site the median specific-force norm over the Standing samples with an
angular-rate norm below REST_GYRO_DPS (unfiltered band copy), then the median over persons. The renderer's rest
value is 1 g by construction.

Speed definition check (speed_check, secondary): for the KIT overground takes, the horizontal pelvis displacement
over the longest steady run divided by its duration (mdfm.data.prep.amass_to_body_clock translation), against the
speed of takes.csv (median of the pelvis speed relative to the stance ankle).

Reading (reading): the criterion of the Prediction section, applied to train54.

Writes reports/bench/f2_scale/H2sensor/: results.json, per_site.csv, per_person.csv, synth_fit.csv, summary.md,
_provenance.json. Run (mdfm environment):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 <conda>/envs/mdfm/python.exe scripts/f2_H2sensor.py
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mdfm  # noqa: E402,F401
from mdfm.data.canonical import Refused  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H2sensor")
PREDICTION_TXT = os.path.join(OUT, "prediction.txt")
PREDICTION_SHA = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "9283719c996d102fd52bc8193f85fccc47c16c5802d5b6b39615c44a453ad6b7"
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
PHASES_CSV = os.path.join(TREE, "reports", "weargait_targets", "phases.csv")
TAKES_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-H2sensor"
SITES13 = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R", "shank_L", "shank_R",
           "ankle_L", "ankle_R", "foot_L", "foot_R"]
LEG = ("thigh_L", "thigh_R", "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R")
FEATURES = ("G_pc", "G_p95", "A_dyn", "A_p95", "A_med")
RATE = 100.0
CUTOFF_HZ = 12.0
ORDER = 4
MIN_SAMPLES = 100
SYNTH_V = (0.4, 2.0)
PRIMARY_V = (0.5, 1.5)
MIN_B = 0.2
REST_GYRO_DPS = 10.0
BOOT_N = 10000
SEED = 20261005
TASKS = ("SelfPace", "HurriedPace")
# criterion of the Prediction section
SMALL_POINT, SMALL_UPPER, SAME_POINT = 0.92, 0.97, 0.97
GRAVITY_TOL = 0.03
AGREE_Q = 0.05
NEAR1 = (0.95, 1.05)
MOST_LEG = 5


# ---------------------------------------------------------------- guards
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(txt=PREDICTION_TXT, sha=PREDICTION_SHA, expected=PREDICTION_SHA256):
    """AssertionError unless the frozen prediction, its hash file and the constant agree."""
    got = sha256_file(txt)
    with open(sha, encoding="utf-8") as f:
        listed = f.read().split()[0]
    if not (got == listed == expected):
        raise AssertionError(f"prediction hash {got}, file {listed}, constant {expected}")
    return got


def person_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_sets(dev, train, locked, f1_sets):
    """AssertionError unless dev and train match the SHA-256 of F1's results.json, hold 14 and 54 persons and are
    disjoint; Refused when either holds a lockbox person."""
    dev, train = sorted(set(dev)), sorted(set(train))
    hit = (set(dev) | set(train)) & set(locked)
    if hit:
        raise Refused(f"{len(hit)} lockbox persons in the sets")
    if len(dev) != 14 or len(train) != 54 or set(dev) & set(train):
        raise AssertionError(f"sets of {len(dev)} dev and {len(train)} training-fold persons")
    if person_hash(dev) != f1_sets["dev14_sha256"] or person_hash(train) != f1_sets["train54_sha256"]:
        raise AssertionError("sets differ from reports/bench/g3_gain_check/results.json")
    return {"dev14": dev, "train54": train}


def keep_row(person, allowed, locked, flag=None):
    """True when person is in `allowed`; Refused when it is a lockbox person or its row carries a lockbox flag that
    is not False (checked only for the persons kept)."""
    if person not in allowed:
        return False
    if person in locked:
        raise Refused(f"lockbox person {person}")
    if flag is not None and str(flag).strip().lower() not in ("false", "0", ""):
        raise Refused(f"row of {person} with lockbox flag {flag!r}")
    return True


# ---------------------------------------------------------------- signal helpers
def runs(mask):
    """(start, stop) of the True runs of a boolean vector."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def lowpass_runs(x, valid, fps=RATE, cutoff=CUTOFF_HZ, order=ORDER):
    """x (T, C) float, valid (T,) bool -> float64 (T, C), each valid run filtered by a zero-phase Butterworth
    low-pass (sosfiltfilt); NaN outside the valid runs and on runs too short to filter (padlen). cutoff None: the
    valid samples as they are."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(x).all(1)
    out = np.full(x.shape, np.nan)
    if cutoff is None:
        out[valid] = x[valid]
        return out
    sos = butter(order, cutoff / (fps / 2.0), output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    for a, b in runs(valid):
        if b - a > padlen:
            out[a:b] = sosfiltfilt(sos, x[a:b], axis=0)
    return out


def unit_features(acc, gyro):
    """acc (n, 3) g and gyro (n, 3) deg/s (rows with a NaN dropped) -> dict of FEATURES, or None under
    MIN_SAMPLES samples. Every feature is invariant to a rotation of the sensor."""
    ok = np.isfinite(acc).all(1) & np.isfinite(gyro).all(1)
    acc, gyro = acc[ok], gyro[ok]
    if len(acc) < MIN_SAMPLES:
        return None
    g0 = gyro - gyro.mean(0)
    _, _, vt = np.linalg.svd(g0, full_matrices=False)
    p = gyro @ vt[0]
    lo, hi = np.percentile(p, [2.5, 97.5])
    an = np.linalg.norm(acc, axis=1)
    return {"G_pc": float((hi - lo) / 2.0), "G_p95": float(np.percentile(np.linalg.norm(gyro, axis=1), 95)),
            "A_dyn": float(np.sqrt((np.linalg.norm(acc - acc.mean(0), axis=1) ** 2).mean())),
            "A_p95": float(np.percentile(an, 95)), "A_med": float(np.median(an))}


# ---------------------------------------------------------------- matched speed
def fit_power(v, f, cad=None):
    """Least squares of log f on [1, log v] (and log cad): {'a', 'b', 'c' (None without cadence), 'n',
    'resid_sd', 'r2'}; rows with a non-finite or non-positive value dropped."""
    v, f = np.asarray(v, float), np.asarray(f, float)
    cols = [np.ones_like(v), np.log(np.where(v > 0, v, np.nan))]
    if cad is not None:
        cad = np.asarray(cad, float)
        cols.append(np.log(np.where(cad > 0, cad, np.nan)))
    X = np.stack(cols, 1)
    y = np.log(np.where(f > 0, f, np.nan))
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 2:
        return {"a": float("nan"), "b": float("nan"), "c": None, "n": int(len(y)), "resid_sd": float("nan"),
                "r2": float("nan")}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1.0 - float((res ** 2).sum()) / float(((y - y.mean()) ** 2).sum()) if len(y) > 1 else float("nan")
    return {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]) if cad is not None else None,
            "n": int(len(y)), "resid_sd": float(res.std(ddof=X.shape[1])), "r2": r2}


def ratios(fit, v, f, cad=None):
    """(q, s): amplitude ratio f / predicted and implied speed ratio q^(1 / b) (NaN when b < MIN_B)."""
    pred = fit["a"] + fit["b"] * math.log(v)
    if fit.get("c") is not None:
        pred += fit["c"] * math.log(cad)
    q = f / math.exp(pred)
    s = q ** (1.0 / fit["b"]) if fit["b"] >= MIN_B else float("nan")
    return q, s


def boot_median(person_values, n_boot=BOOT_N, seed=SEED):
    """Median over persons and the 95 % person-bootstrap percentile interval: {point, lo, hi, n}."""
    x = np.asarray([v for v in person_values if np.isfinite(v)], float)
    if len(x) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = np.median(x[idx], axis=1)
    return {"point": float(np.median(x)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": int(len(x))}


def classify(stat):
    """'reads smaller', 'reads same or larger' or 'in between' (Prediction criterion)."""
    if not np.isfinite(stat["point"]):
        return "not computed"
    if stat["point"] <= SMALL_POINT and stat["hi"] < SMALL_UPPER:
        return "reads smaller"
    if stat["point"] >= SAME_POINT:
        return "reads same or larger"
    return "in between"


def sensor_cause(gravity_by_site, q_by_site):
    """The sensor-cause test of the criterion. gravity_by_site {site: median rest norm g}; q_by_site {site:
    {feature: median q over persons}}. Returns {gravity_sites (leg sites beyond GRAVITY_TOL), gyro_kind,
    acc_kind (leg sites where that kind agrees within AGREE_Q at q <= SMALL_POINT while the other kind's A_dyn or
    G_pc lies in NEAR1), found}."""
    grav = [s for s in LEG if s in gravity_by_site and np.isfinite(gravity_by_site[s])
            and abs(gravity_by_site[s] - 1.0) > GRAVITY_TOL]
    gk, ak = [], []
    for s in LEG:
        q = q_by_site.get(s)
        if not q:
            continue
        g = (q["G_pc"], q["G_p95"])
        a = (q["A_dyn"], q["A_p95"])
        if abs(g[0] - g[1]) <= AGREE_Q and max(g) <= SMALL_POINT and NEAR1[0] <= q["A_dyn"] <= NEAR1[1]:
            gk.append(s)
        if abs(a[0] - a[1]) <= AGREE_Q and max(a) <= SMALL_POINT and NEAR1[0] <= q["G_pc"] <= NEAR1[1]:
            ak.append(s)
    return {"gravity_sites": grav, "gyro_kind_sites": gk, "acc_kind_sites": ak,
            "found": bool(grav or len(gk) >= MOST_LEG or len(ak) >= MOST_LEG)}


def reading(cls_g, cls_a, cause):
    """SUPPORTED, PARTLY or NOT SUPPORTED with its reason (Prediction criterion)."""
    both_small = cls_g == "reads smaller" and cls_a == "reads smaller"
    both_same = cls_g == "reads same or larger" and cls_a == "reads same or larger"
    if both_small and cause["found"]:
        return "SUPPORTED", "both features read smaller and a sensor cause is found"
    if both_same and not cause["gravity_sites"]:
        return "NOT SUPPORTED", "both features read same or larger and rest gravity is within 3 % of 1 g at every leg site"
    if both_small:
        return "PARTLY", "both features read smaller without a sensor cause"
    return "PARTLY", f"G_pc {cls_g}, A_dyn {cls_a}, sensor cause {'found' if cause['found'] else 'not found'}"


# ---------------------------------------------------------------- real data
def store_root():
    from mdfm.data import window_store as WS
    return WS.store_root()


def read_store_records(root, allowed, locked, tasks=TASKS):
    """Rows (dicts) of <root>/weargait_pd/records.csv for the imu stream, tasks and persons; a row is kept only
    after its person id passed keep_row."""
    out = []
    with open(os.path.join(root, "weargait_pd", "records.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["stream"] != "imu":
                continue
            if not keep_row(r["person_id"], allowed, locked):
                continue
            if r["key"] in tasks:
                out.append(r)
    return out


def store_block(root, r):
    """(values float32 (n, G, C), valid bool (n, G), times (n,)) of one records row."""
    folder = os.path.join(root, "weargait_pd", "imu")
    n, G, C = int(r["n"]), int(r["G"]), int(r["C"])
    v = np.memmap(os.path.join(folder, "values.f16"), dtype="<f2", mode="r")
    m = np.memmap(os.path.join(folder, "valid.u1"), dtype=np.uint8, mode="r")
    va, ma = int(r["values_at"]), int(r["valid_at"])
    vals = np.asarray(v[va:va + n * G * C], dtype=np.float32).reshape(n, G, C)
    valid = np.asarray(m[ma:ma + n * G], dtype=bool).reshape(n, G)
    del v, m
    t = float(r["offset"]) + np.arange(n) * (int(r["k"]) / float(r["fps_native"]))
    return vals, valid, t


def site_columns(r):
    meta = json.loads(r["meta"])
    return [SITES13[i] if i < len(SITES13) else None for i in meta["site_ids"]]


def read_real(root, persons, locked, cutoffs):
    """Per walkway pass and site the features at each cutoff. Returns (units list of dicts, rest list, counts)."""
    from mdfm.gates import g3
    allowed = set(persons)
    tg = g3.load_targets(tuple(sorted(allowed)))
    if tg.counts.get("lockbox_rows"):
        raise Refused("lockbox rows in the targets")
    passes, strides = tg.passes, tg.strides
    cad = {}
    for (tid, pid), g in strides.groupby(["trial_id", "pass_id"]):
        cad[(tid, str(pid))] = 120.0 / float((g["t1"] - g["t0"]).mean())
    phases = [r for r in g3.read_rows(PHASES_CSV, tuple(sorted(allowed))) if r["label"] == "Standing"
              and r["task"] in TASKS]
    recs = read_store_records(root, allowed, locked)
    units, rest = [], []
    counts = {"records": len(recs), "passes_targets": int(len(passes)), "passes_used": 0, "standing_runs": len(phases)}
    for r in recs:
        tid = g3.trial_id(r["person_id"], r["key"])
        vals, valid, t = store_block(root, r)
        cols = site_columns(r)
        filt = {}
        for c in cutoffs:
            filt[c] = {}
            for j, site in enumerate(cols):
                if site is None:
                    continue
                filt[c][site] = lowpass_runs(vals[:, j, :6], valid[:, j], cutoff=c)
        P = passes[passes["trial_id"] == tid]
        for p in P.itertuples(index=False):
            sel = (t >= p.t0) & (t <= p.t1)
            row = {"person": r["person_id"], "task": r["key"], "trial": tid, "pass": str(p.pass_id),
                   "speed": float(p.speed_m_s), "cadence": cad.get((tid, str(p.pass_id)), float("nan")),
                   "duration_s": float(p.t1 - p.t0), "feat": {}}
            for c in cutoffs:
                for site, x in filt[c].items():
                    fe = unit_features(x[sel, 0:3], x[sel, 3:6])
                    if fe is not None:
                        row["feat"][(c, site)] = fe
            units.append(row)
            counts["passes_used"] += 1
        raw = filt.get(None)
        if raw is None:
            raw = {site: lowpass_runs(vals[:, j, :6], valid[:, j], cutoff=None) for j, site in enumerate(cols) if site}
        for ph in phases:
            if ph["trial_id"] != tid:
                continue
            sel = (t >= float(ph["start_s"])) & (t < float(ph["end_s"]))
            for site, x in raw.items():
                y = x[sel]
                y = y[np.isfinite(y).all(1)]
                y = y[np.linalg.norm(y[:, 3:6], axis=1) < REST_GYRO_DPS]
                if len(y):
                    rest.append({"person": r["person_id"], "site": site, "n": int(len(y)),
                                 "norm_sum": float(np.linalg.norm(y[:, 0:3], axis=1).sum()),
                                 "norms": np.linalg.norm(y[:, 0:3], axis=1)})
    return units, rest, counts


def rest_gravity(rest):
    """{site: {'median_over_persons', 'n_persons', 'per_person' {person: median}}}."""
    by = {}
    for r in rest:
        by.setdefault(r["site"], {}).setdefault(r["person"], []).append(r["norms"])
    out = {}
    for site, d in by.items():
        pp = {p: float(np.median(np.concatenate(v))) for p, v in d.items()}
        out[site] = {"median_over_persons": float(np.median(list(pp.values()))), "n_persons": len(pp),
                     "q05": float(np.percentile(list(pp.values()), 5)),
                     "q95": float(np.percentile(list(pp.values()), 95)), "per_person": pp}
    return out


# ---------------------------------------------------------------- synthetic data
def read_synth(cutoffs):
    """Per AMASS take and site the features at each cutoff, with its speed and cadence (takes.csv)."""
    import g2_renderer_realism as g2
    takes = list(g2.rendered_takes(None))
    meta = {}
    with open(TAKES_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            meta[r["take"]] = r
    units = []
    for tk in takes:
        x = tk["x"]
        m = meta[tk["take"]]
        row = {"take": tk["take"], "person": tk["person"], "subset": m["subset"], "speed": float(m["speed_m_s"]),
               "cadence": float(m["cadence_steps_min"]), "steady_frames": int(np.isfinite(x).all(2).all(1).sum()),
               "steady_mask": np.isfinite(x).all(2).all(1), "feat": {}}
        for c in cutoffs:
            for s, site in enumerate(SITES13):
                y = lowpass_runs(x[:, s, :], np.isfinite(x[:, s, :]).all(1), cutoff=c)
                fe = unit_features(y[:, 0:3], y[:, 3:6])
                if fe is not None:
                    row["feat"][(c, site)] = fe
        units.append(row)
    return units, g2


def speed_check(synth_units):
    """KIT overground takes: horizontal pelvis displacement over the longest steady run / its duration, against
    takes.csv speed. Returns {n, median_ratio, q25, q75, per_take}."""
    import torch
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    out = []
    for u in synth_units:
        if u["subset"] != "KIT":
            continue
        rr = runs(u["steady_mask"])
        if not rr:
            continue
        a, b = max(rr, key=lambda ab: ab[1] - ab[0])
        if b - a < 100:
            continue
        seq = load_sequence("KIT", u["take"])
        _, tr, _ = amass_to_body_clock(seq, rate=RATE, cutoff=CUTOFF_HZ, dtype=torch.float64)
        tr = tr.numpy()
        i0, i1 = 50 + a, 50 + b - 1          # render clock t = 0.5 + i / 100 s, body clock k / 100 s
        if i1 >= len(tr):
            continue
        d = tr[i1] - tr[i0]
        v = float(np.hypot(d[0], d[2])) / ((i1 - i0) / RATE)
        out.append({"take": u["take"], "speed_takes_csv": u["speed"], "speed_displacement": v,
                    "ratio": v / u["speed"]})
    r = np.array([o["ratio"] for o in out])
    return {"n": len(out), "median_ratio": float(np.median(r)) if len(r) else float("nan"),
            "q25": float(np.percentile(r, 25)) if len(r) else float("nan"),
            "q75": float(np.percentile(r, 75)) if len(r) else float("nan"), "per_take": out}


# ---------------------------------------------------------------- analysis
def fit_all(synth_units, cutoffs):
    fits = {}
    for c in cutoffs:
        for site in SITES13:
            for k in FEATURES:
                v, f, cd = [], [], []
                for u in synth_units:
                    fe = u["feat"].get((c, site))
                    if fe is None or not (SYNTH_V[0] <= u["speed"] <= SYNTH_V[1]):
                        continue
                    v.append(u["speed"]), f.append(fe[k]), cd.append(u["cadence"])
                fits[(c, site, k, "v")] = fit_power(v, f)
                fits[(c, site, k, "vc")] = fit_power(v, f, cd)
    return fits


def unit_ratios(units, fits, cutoffs, speed_scale=1.0):
    """Adds row['q'][(c, site, k, model)] = (q, s). speed_scale multiplies the real speed before matching (the
    speed-definition sensitivity reading)."""
    for u in units:
        u.setdefault("q", {})
        for (c, site), fe in u["feat"].items():
            for k in FEATURES:
                for model in ("v", "vc"):
                    fit = fits[(c, site, k, model)]
                    if not np.isfinite(fit["a"]) or (model == "vc" and not np.isfinite(u["cadence"])):
                        continue
                    u["q"][(c, site, k, model, speed_scale)] = ratios(fit, u["speed"] * speed_scale, fe[k],
                                                                      u["cadence"])


def aggregate(units, persons, c, k, model, which, speed_range=PRIMARY_V, speed_scale=1.0, sites=LEG):
    """Per person the median over `sites` of the per-site median over the person's units of q (which=0) or s
    (which=1); returns ({person: value}, {site: {person: value}})."""
    per_site = {}
    for u in units:
        if u["person"] not in persons:
            continue
        if speed_range is not None and not (speed_range[0] <= u["speed"] <= speed_range[1]):
            continue
        for site in sites:
            val = u["q"].get((c, site, k, model, speed_scale))
            if val is None or not np.isfinite(val[which]):
                continue
            per_site.setdefault(site, {}).setdefault(u["person"], []).append(val[which])
    ps = {s: {p: float(np.median(v)) for p, v in d.items()} for s, d in per_site.items()}
    pp = {}
    for s, d in ps.items():
        for p, v in d.items():
            pp.setdefault(p, []).append(v)
    return {p: float(np.median(v)) for p, v in pp.items()}, ps


SPEED_BINS = ((0.0, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 9.9))


def within_exponent(units, persons, site, k, c=CUTOFF_HZ):
    """Least-squares slope of log f on log v after removing each person's means (real passes of `persons`)."""
    by = {}
    for u in units:
        fe = u["feat"].get((c, site))
        if u["person"] in persons and fe is not None and fe[k] > 0:
            by.setdefault(u["person"], []).append((math.log(u["speed"]), math.log(fe[k])))
    xs, ys = [], []
    for v in by.values():
        if len(v) < 2:
            continue
        a = np.asarray(v)
        xs.append(a[:, 0] - a[:, 0].mean()), ys.append(a[:, 1] - a[:, 1].mean())
    if not xs:
        return float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float((x * y).sum() / (x * x).sum())


def exploratory(units, fits, persons):
    """Post hoc (written after the reading): the leg-pooled implied speed ratio of G_pc and A_dyn by pass-speed
    bin, and per leg site the within-person speed exponent of the real passes over the synthetic exponent."""
    out = {"by_speed_bin": {}, "within_exponent": {}}
    for lo, hi in SPEED_BINS:
        for k in ("G_pc", "A_dyn"):
            pp, _ = aggregate(units, persons, CUTOFF_HZ, k, "v", 1, (lo, hi))
            n_units = sum(1 for u in units if u["person"] in persons and lo <= u["speed"] <= hi)
            out["by_speed_bin"][f"{lo}-{hi}|{k}"] = {**boot_median(list(pp.values())), "units": n_units}
    for site in LEG:
        for k in ("G_pc", "A_dyn"):
            w = within_exponent(units, persons, site, k)
            sb = fits[(CUTOFF_HZ, site, k, "v")]["b"]
            out["within_exponent"][f"{site}|{k}"] = {"real_within_b": w, "synth_b": sb, "ratio": w / sb}
    return out


def fmt(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fmt_stat(s, nd=3):
    return f"{fmt(s['point'], nd)} [{fmt(s['lo'], nd)}, {fmt(s['hi'], nd)}] (n {s['n']})"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def ckey(c):
    return "lp12" if c == CUTOFF_HZ else "band40"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-speed-check", action="store_true")
    args = ap.parse_args(argv)
    t_start = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    sys.stdout.reconfigure(encoding="utf-8")
    pred_sha = check_prediction()
    from mdfm.gates import g3
    from mdfm.data import window_store as WS
    with open(F1_RESULTS, encoding="utf-8") as f:
        f1 = json.load(f)
    locked = set(g3.lockbox_persons()) | set(WS.lockbox_persons())
    sets = check_sets(g3.controls("dev"), g3.controls("train"), locked, f1["sets"])
    persons = sets["dev14"] + sets["train54"]
    cutoffs = (CUTOFF_HZ, None)
    root = store_root()
    print("reading real units", flush=True)
    real, rest, counts = read_real(root, persons, locked, cutoffs)
    assert not ({u["person"] for u in real} - set(persons)), "a real unit outside the sets"
    print(f"real passes {len(real)}; reading synthetic units", flush=True)
    synth, g2 = read_synth(cutoffs)
    fits = fit_all(synth, cutoffs)
    unit_ratios(real, fits, cutoffs)
    sc = None if args.no_speed_check else speed_check(synth)
    if sc is not None and np.isfinite(sc["median_ratio"]):
        unit_ratios(real, fits, cutoffs, speed_scale=1.0 / sc["median_ratio"])
    grav = rest_gravity(rest)

    res = {"what": LABEL, "task": TASK, "prediction_sha256": pred_sha, "run_started": started,
           "sets": {"dev14_sha256": person_hash(sets["dev14"]), "train54_sha256": person_hash(sets["train54"]),
                    "lockbox_persons_in_sets": 0},
           "counts": {**counts, "real_units": len(real), "synth_units": len(synth),
                      "synth_persons": len({u["person"] for u in synth})},
           "constants": {"cutoff_hz": CUTOFF_HZ, "synth_v": SYNTH_V, "primary_v": PRIMARY_V, "min_b": MIN_B,
                         "rest_gyro_dps": REST_GYRO_DPS, "boot_n": BOOT_N, "seed": SEED, "leg": LEG},
           "rest_gravity": {s: {k: v for k, v in d.items() if k != "per_person"} for s, d in grav.items()},
           "speed_check": None if sc is None else {k: v for k, v in sc.items() if k != "per_take"},
           "fits": {f"{ckey(c)}|{s}|{k}|{m}": v for (c, s, k, m), v in fits.items()},
           "lines": {}, "per_site": {}}
    # lines: set x cutoff x feature x model x speed range
    for set_name in ("train54", "dev14"):
        P = set(sets[set_name])
        for c in cutoffs:
            for k in FEATURES:
                for model in ("v", "vc"):
                    for rng_name, rngv in (("primary", PRIMARY_V), ("all", None)):
                        for which, wname in ((0, "q"), (1, "s")):
                            pp, ps = aggregate(real, P, c, k, model, which, rngv)
                            key = f"{set_name}|{ckey(c)}|{k}|{model}|{rng_name}|{wname}"
                            res["lines"][key] = boot_median(list(pp.values()))
                            if model == "v" and rng_name == "primary":
                                res["per_site"][key] = {s: {"median": float(np.median(list(d.values()))),
                                                            "n_persons": len(d)} for s, d in ps.items()}
                            if model == "v" and rng_name == "primary" and set_name == "train54" and c == CUTOFF_HZ:
                                res.setdefault("per_person", {})[f"{k}|{wname}"] = pp
                        # non-leg sites, q only, for context
                    pp_all, ps_all = aggregate(real, P, c, k, model, 0, PRIMARY_V, sites=SITES13)
                    res["per_site"][f"{set_name}|{ckey(c)}|{k}|{model}|primary|q|all_sites"] = {
                        s: {"median": float(np.median(list(d.values()))), "n_persons": len(d)} for s, d in ps_all.items()}
        if sc is not None and np.isfinite(sc["median_ratio"]):
            scale = 1.0 / sc["median_ratio"]
            for k in ("G_pc", "A_dyn"):
                pp, _ = aggregate(real, P, CUTOFF_HZ, k, "v", 1, PRIMARY_V, speed_scale=scale)
                res["lines"][f"{set_name}|lp12|{k}|v|primary|s|speed_rescaled"] = boot_median(list(pp.values()))
    # reading on train54
    Lg = res["lines"]["train54|lp12|G_pc|v|primary|s"]
    La = res["lines"]["train54|lp12|A_dyn|v|primary|s"]
    qsite = {s: {k: res["per_site"][f"train54|lp12|{k}|v|primary|q"].get(s, {}).get("median", float("nan"))
                 for k in FEATURES} for s in LEG}
    cause = sensor_cause({s: d["median_over_persons"] for s, d in grav.items()}, qsite)
    cls_g, cls_a = classify(Lg), classify(La)
    verdict, why = reading(cls_g, cls_a, cause)
    res["reading"] = {"G_pc": cls_g, "A_dyn": cls_a, "sensor_cause": cause, "verdict": verdict, "reason": why}
    Lg14 = res["lines"]["dev14|lp12|G_pc|v|primary|s"]
    La14 = res["lines"]["dev14|lp12|A_dyn|v|primary|s"]
    res["replication_dev14"] = {"G_pc": classify(Lg14), "A_dyn": classify(La14)}
    # synthetic and real speed slopes (context): log f on log v over the real passes, pooled persons
    slopes = {}
    for set_name in ("train54",):
        P = set(sets[set_name])
        for site in LEG:
            for k in ("G_pc", "A_dyn"):
                v = [u["speed"] for u in real if u["person"] in P and (CUTOFF_HZ, site) in u["feat"]]
                f_ = [u["feat"][(CUTOFF_HZ, site)][k] for u in real if u["person"] in P and (CUTOFF_HZ, site) in u["feat"]]
                fr = fit_power(v, f_)
                slopes[f"{site}|{k}"] = {"real_b": fr["b"], "synth_b": fits[(CUTOFF_HZ, site, k, "v")]["b"],
                                         "real_n": fr["n"], "synth_n": fits[(CUTOFF_HZ, site, k, "v")]["n"]}
    res["speed_exponents"] = slopes
    res["exploratory"] = exploratory(real, fits, set(sets["train54"]))
    res["timing_s"] = round(time.time() - t_start, 1)

    os.makedirs(args.out, exist_ok=True)
    # provenance
    stamp = None
    try:
        import moveatlas_stamp
        stamp = moveatlas_stamp.stamp()
    except Exception as e:                     # the stamp is read-only git; report a failure, never hide it
        stamp = {"error": repr(e)}
    res["moveatlas"] = {"head": (stamp or {}).get("head")}
    files = [os.path.abspath(__file__), g3.__file__, WS.__file__, g2.__file__]
    prov = {"script": "scripts/f2_H2sensor.py", "run_started": started, "moveatlas_stamp": stamp,
            "code_sha256": {os.path.relpath(p, TREE).replace("\\", "/"): sha256_file(p) for p in files},
            "inputs_sha256": {"reports/bench/g3_gain_check/results.json": sha256_file(F1_RESULTS),
                              "reports/amass_walks/takes.csv": sha256_file(TAKES_CSV),
                              "reports/weargait_targets/phases.csv": sha256_file(PHASES_CSV),
                              "window_store/weargait_pd/records.csv":
                                  sha256_file(os.path.join(root, "weargait_pd", "records.csv")),
                              "reports/g2_renderer/device_noise.json":
                                  sha256_file(os.path.join(TREE, "reports", "g2_renderer", "device_noise.json"))},
            "synth_cache": os.path.basename(g2._mount_tag(None)) + ".npz"}
    with open(os.path.join(args.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(prov), f, indent=1)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(res), f, indent=1)
    write_tables(args.out, res, fits, grav, synth, sc)
    write_summary(args.out, res)
    print(json.dumps(clean(res["reading"]), indent=1))
    print(f"done in {res['timing_s']} s")


def write_tables(out, res, fits, grav, synth, sc):
    with open(os.path.join(out, "synth_fit.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["band", "site", "feature", "model", "a", "b", "c", "n", "resid_sd", "r2"])
        for (c, s, k, m), v in fits.items():
            w.writerow([ckey(c), s, k, m, fmt(v["a"], 5), fmt(v["b"], 5), "" if v["c"] is None else fmt(v["c"], 5),
                        v["n"], fmt(v["resid_sd"], 5), fmt(v["r2"], 4)])
    with open(os.path.join(out, "per_site.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["set", "band", "feature", "quantity", "site", "median_over_persons", "n_persons",
                    "rest_gravity_g"])
        for key, d in res["per_site"].items():
            parts = key.split("|")
            if parts[3] != "v":
                continue
            for s, v in d.items():
                w.writerow([parts[0], parts[1], parts[2], parts[5] + ("_all_sites" if len(parts) > 6 else ""), s,
                            fmt(v["median"], 4), v["n_persons"],
                            fmt(grav.get(s, {}).get("median_over_persons", float("nan")), 4)])
    with open(os.path.join(out, "per_person.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["person", "quantity", "value"])
        for q, d in res.get("per_person", {}).items():
            for p, v in sorted(d.items()):
                w.writerow([p, q, fmt(v, 4)])
    if sc is not None:
        with open(os.path.join(out, "speed_check.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["take", "speed_takes_csv", "speed_displacement", "ratio"])
            for o in sc["per_take"]:
                w.writerow([o["take"], fmt(o["speed_takes_csv"], 4), fmt(o["speed_displacement"], 4),
                            fmt(o["ratio"], 4)])


def write_summary(out, r):
    L = r["lines"]
    rd = r["reading"]
    md = [f"# F2-H2sensor: real against rendered IMU amplitude at equal walking speed ({LABEL})", "",
          "Written by scripts/f2_H2sensor.py; every number is read from results.json of this folder. Prediction "
          f"SHA-256 {r['prediction_sha256']} (asserted before any value was read). No lockbox person, EVAL or CODY "
          "record was read (asserted).", "",
          f"## Reading (train54, primary passes, 12 Hz features): **{rd['verdict']}**", "",
          f"- G_pc implied speed ratio: {fmt_stat(L['train54|lp12|G_pc|v|primary|s'])}: {rd['G_pc']}",
          f"- A_dyn implied speed ratio: {fmt_stat(L['train54|lp12|A_dyn|v|primary|s'])}: {rd['A_dyn']}",
          f"- sensor cause: gravity sites beyond 3 %: {rd['sensor_cause']['gravity_sites'] or 'none'}; gyro-kind "
          f"sites {rd['sensor_cause']['gyro_kind_sites'] or 'none'}; acc-kind sites "
          f"{rd['sensor_cause']['acc_kind_sites'] or 'none'}; found {rd['sensor_cause']['found']}",
          f"- reason: {rd['reason']}",
          f"- dev14 replication: G_pc {r['replication_dev14']['G_pc']}, A_dyn {r['replication_dev14']['A_dyn']}", "",
          "## Leg-pooled lines (median over persons of the per-person median over leg sites; 95 % person bootstrap)", "",
          "| set | band | feature | model | passes | amplitude ratio q | implied speed ratio s |", "|---|---|---|---|---|---|---|"]
    for set_name in ("train54", "dev14"):
        for band in ("lp12", "band40"):
            for k in FEATURES:
                for model in ("v", "vc"):
                    for rng in ("primary", "all"):
                        q = L.get(f"{set_name}|{band}|{k}|{model}|{rng}|q")
                        s = L.get(f"{set_name}|{band}|{k}|{model}|{rng}|s")
                        if q is None:
                            continue
                        if band == "band40" and (model == "vc" or rng == "all"):
                            continue
                        md.append(f"| {set_name} | {band} | {k} | {'speed' if model == 'v' else 'speed+cadence'} | "
                                  f"{rng} | {fmt_stat(q)} | {fmt_stat(s)} |")
    md += ["", "## Per leg site (train54, 12 Hz, matched speed, primary passes): median over persons of q", "",
           "| site | " + " | ".join(FEATURES) + " | rest gravity (g) |", "|---" * (len(FEATURES) + 2) + "|"]
    for s in SITES13:
        vals = [r["per_site"].get(f"train54|lp12|{k}|v|primary|q|all_sites", {}).get(s, {}).get("median")
                for k in FEATURES]
        g = r["rest_gravity"].get(s, {}).get("median_over_persons")
        md.append(f"| {s} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(g, 4)} |")
    sc = r.get("speed_check")
    if sc:
        md += ["", f"## Speed definition (KIT overground takes, n {sc['n']})", "",
               f"Displacement speed over takes.csv speed: median {fmt(sc['median_ratio'])}, quartiles "
               f"[{fmt(sc['q25'])}, {fmt(sc['q75'])}]. Implied speed ratios with the real speeds rescaled by its "
               f"inverse (train54, 12 Hz): G_pc {fmt_stat(L['train54|lp12|G_pc|v|primary|s|speed_rescaled'])}, "
               f"A_dyn {fmt_stat(L['train54|lp12|A_dyn|v|primary|s|speed_rescaled'])}."]
    md += ["", "## Speed exponents b (log feature on log speed): real train54 passes against synthetic takes", "",
           "| site | feature | real b | synthetic b |", "|---|---|---|---|"]
    for key, v in r["speed_exponents"].items():
        s, k = key.split("|")
        md.append(f"| {s} | {k} | {fmt(v['real_b'])} | {fmt(v['synth_b'])} |")
    ex = r.get("exploratory")
    if ex:
        md += ["", "## Exploratory, post hoc (written after the reading; no criterion reads it)", "",
               "Implied speed ratio s by pass-speed bin (train54, 12 Hz, leg-pooled; speed bin in m/s):", "",
               "| bin | feature | passes | s |", "|---|---|---|---|"]
        for key, v in ex["by_speed_bin"].items():
            b, k = key.split("|")
            md.append(f"| {b} | {k} | {v['units']} | {fmt_stat(v)} |")
        md += ["", "Within-person speed exponent of the real passes (train54) against the synthetic exponent:", "",
               "| site | feature | real within-person b | synthetic b | ratio |", "|---|---|---|---|---|"]
        for key, v in ex["within_exponent"].items():
            s, k = key.split("|")
            md.append(f"| {s} | {k} | {fmt(v['real_within_b'])} | {fmt(v['synth_b'])} | {fmt(v['ratio'])} |")
    c = r["counts"]
    md += ["", "## Counts", "",
           f"Real: {c['records']} records, {c['passes_used']} walkway passes ({c['passes_targets']} in the targets), "
           f"{c['standing_runs']} Standing runs. Synthetic: {c['synth_units']} takes of {c['synth_persons']} AMASS "
           f"performers. MoveAtlas HEAD {r['moveatlas']['head']}. Time {r['timing_s']} s.", ""]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
