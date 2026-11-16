"""WRS (wave 17, 2026-10-16; docs/notes/STRATEGY_2026-10-15.md section 4, rank 4): the wrist-only G3 bars on the dev
controls. What wrist-only stride and speed do a classical method, a supervised regressor, a frozen-token probe and the
decoded body give, under the G3 harness's sets, targets and statistics (mdfm/gates/g3.py, imported), and does the
decoded body show a premium over the best of them? Every output carries LABEL: dev diagnostics, not a G3 read, not a
G12 comparison.

Prediction and method: reports/bench/g3_wrist/prediction.txt, hashed before any run (prediction.sha256 beside it and
PREDICTION_SHA256 here); check_prediction asserts the three agree before any target, record or checkpoint is read, in
every mode, so a prediction edited after the fact stops the script.

Persons: the 54 training-fold controls fit (g3.controls('train')), the 14 dev controls and the 12 of g3.DEV_SETS are
scored; lockbox persons are refused by g3's readers and asserted absent; records come through g3.imu_records (reader
flags False) and are checked one by one (lockbox False, no EVAL tier, person of the set). Input of every row: the G3
model input (g3.model_input) cut to wrist_L and wrist_R (scripts/eval_g3_diag.py cut_sites, its family_wrists row).

Rows (prediction.txt section 2):
- a1, classical: wrist cadence (wrist_cadence: the signed sagittal gyroscope rate of each usable wrist, the
  autocorrelation peak at the stride period in STRIDE_LAG_S) times the training-fold mean step (task-blind);
- a2, classical: stride from one of FORMS (cadence, Weinberg amplitude, gyroscope RMS) by OLS on the 54, the form
  chosen by person cross-fitted stride line on the 54; speed = stride x stride frequency;
- b1, supervised ridge (mdfm/eval/probes.fit_probe) on per-pass wrist features (96 window features of
  scripts/g2_renderer_realism.py averaged over the pass windows of scripts/g3_supervised_baseline.py, and the three
  classical terms); b2, the gradient boosting of scripts/g3_supervised_baseline.py on the 96 wrist window features;
- token probes (mdfm/baselines/readouts.py, the G12R protocol unchanged) on wrist-only tokens of the body seeds v1_B,
  s2_B and of the G12 baselines MFR and JEPA (two seeds each): --tokens;
- decoded body, wrist-only (mdfm/eval/infer.G3Predictor, CPU float32): raw, and divided by one training-fold gain per
  line (scripts/g3_gain_check.py fit_gain on the 54): --decode.
Premium rule (section 3): per arm, line and set, the comparator is the bar with the lowest point; premium when the two
body seeds separate below it, the paired person bootstrap interval of the seed-mean difference
(mdfm/baselines/comparison.bootstrap_line) lies below 0 and the difference is at most -m, m = max(0.10 x the
comparator's point, 2 x the larger seed spread); PREMIUM on a line when it holds on dev12 and on dev14, LOSS for the
mirror (separation and interval), NO PREMIUM otherwise.

Modes (all CPU: CUDA_VISIBLE_DEVICES=-1 is set before torch loads):
  --decode KEY   (v1_B, s2_B) the wrist-only body on the 14 then the 54: per-trial predictions in
                 cache/g3_wrist/decode_<name>/trials/ (resumable: a trial file of the same checkpoint and settings is
                 kept), then reports/bench/g3_wrist/decode_<name>/ (ics.csv, strides.csv, speeds.csv, decode.json,
                 _provenance.json); no target value is written there, the read run matches.
  --tokens KEY   (the six ENDS) wrist-only token features to cache/g3_wrist/tokens_<name>/features.npz, the G12R probes,
                 reports/bench/g3_wrist/tokens_<name>/ (results.json, per_row.csv, _provenance.json).
  (default)      the read: classical and supervised rows, the decode and token outputs, every line, comparison and
                 prediction; reports/bench/g3_wrist/ results.json, summary.md, per_row.csv, _provenance.json.
  --print-jobs   the commands of a full run.
--limit-persons N (smoke) keeps the first N persons of each set and needs --out and --cache.

Run (mdfm environment; OMP_NUM_THREADS=2):
  python scripts/g3_wrist_bars.py --decode v1_B
  python scripts/g3_wrist_bars.py --tokens mfr_s1
  python scripts/g3_wrist_bars.py
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # every mode runs on the CPU
    os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
for _p in (HERE, TREE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm.data.canonical import Refused  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g3_wrist")
CACHE = os.path.join(TREE, "cache", "g3_wrist")
PREDICTION_PATH = os.path.join(OUT, "prediction.txt")
PREDICTION_SHA_PATH = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "01fec4d7b2c3f8f252f02b7f5160864ed488fd3343b54563102c1f2bd19ab1a5"
LABEL = "dev diagnostic, not a G3 read, not a G12 comparison"
TASK = "WRS"
WRIST_SITES = ("wrist_L", "wrist_R")
RATE = 100.0
SPAN_MARGIN_S = 1.0                 # s added on each side of the pass span (wrist features, a and b rows)
MIN_VALID = 0.8                     # share of valid span samples a wrist needs to be used
STRIDE_LAG_S = (0.66, 1.70)         # s, the stride-period search range of the wrist cadence
MIN_PAIRS = 100                     # samples: a lag with fewer valid pairs is not a candidate
FORMS = {"F1": ("f_step",), "F2": ("f_step", "A"), "F3": ("f_step", "G")}
CLASSICAL_TERMS = ("f_step", "A", "G")
SEED = 20260929                     # g3.SEED
BOOT_N = 2000                       # g3.BOOT_N
MARGIN_FRACTION = 0.10
LINES = ("stride_mae_cm", "speed_mae_m_s")
SETS = ("dev12", "dev14")
N_TRAIN, N_DEV = 54, 14
DECODE_SETTINGS = {"refine_steps": 20, "points": 256, "batch_windows": 32, "seed": 20260929, "weights": "ema",
                   "pool": True, "device": "cpu"}


@dataclass(frozen=True)
class End:
    """A checkpoint: key, report name, tree-relative path, kind (body or baseline), arm, seed, SHA-256 (as the G12R
    readout read it)."""
    key: str
    name: str
    path: str
    kind: str
    arm: str
    seed: int
    sha256: str


ENDS = (
    End("v1_B", "phaseA_root_full_ckpt_B_032044", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", "body", "body",
        20260928, "cffccdac4109dfbfb921ed4ebcb690ce27dd2d35045a3afdaaca7d38f321a677"),
    End("s2_B", "pilot_v1_s2_ckpt_B_032044", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", "body", "body",
        20261001, "5e97c4fe3b208f8f11850a5318ed827d26316f1e2fc867355d74f50a3d96e023"),
    End("mfr_s1", "g12_mfr_s20260928_ckpt_B_056774", "cache/runs/g12_mfr_s20260928/ckpt_B_056774.pt", "baseline",
        "mfr", 20260928, "f1b23a15147537b9ddbead491774a59ba9b41e001438173669e984e55d3c38a0"),
    End("mfr_s2", "g12_mfr_s20261001_ckpt_B_056774", "cache/runs/g12_mfr_s20261001/ckpt_B_056774.pt", "baseline",
        "mfr", 20261001, "ef9fc2b7b2c30806e48b7639b4b75bbef05bab90dca8fa4a84387e6f67dbbb4d"),
    End("jepa_s1", "g12_jepa_s20260928_ckpt_B_047617", "cache/runs/g12_jepa_s20260928/ckpt_B_047617.pt", "baseline",
        "jepa", 20260928, "18a4e382d6c3ee498a17c4a255dac5457afb8e648a53dc95e0d075a1b1b9d1a9"),
    End("jepa_s2", "g12_jepa_s20261001_ckpt_B_047617", "cache/runs/g12_jepa_s20261001/ckpt_B_047617.pt", "baseline",
        "jepa", 20261001, "c974dc55e3d7df9c0f2b6ce836b3f02fa4a3d24dd33dfc46adcb0cce6839e925"),
)
END = {e.key: e for e in ENDS}
BODY_KEYS = ("v1_B", "s2_B")
TOKEN_ARMS = {"tok_body": ("v1_B", "s2_B"), "tok_mfr": ("mfr_s1", "mfr_s2"), "tok_jepa": ("jepa_s1", "jepa_s2")}
GROUPS = {"classical": ("a1", "a2"), "supervised": ("b1", "b2"), "token": tuple(TOKEN_ARMS)}
ARMS = {"d_raw": "decoded body, raw", "d_gain": "decoded body, one training-fold gain per line"}


# ---------------------------------------------------------------- siblings, prediction, folders
def _sibling(name):
    key = f"_wrs_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def DIAG():
    return _sibling("eval_g3_diag")


def GAIN():
    return _sibling("g3_gain_check")


def SB():
    return _sibling("g3_supervised_baseline")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(path=PREDICTION_PATH, sha_path=PREDICTION_SHA_PATH, expected=PREDICTION_SHA256):
    """The SHA-256 of the prediction file, asserted equal to its sidecar and to PREDICTION_SHA256."""
    got = sha256_file(path)
    with open(sha_path, encoding="utf-8") as f:
        side = f.read().split()[0].strip()
    if not (got == side == expected):
        raise AssertionError(f"prediction hash mismatch: file {got}, sidecar {side}, script {expected}")
    return got


def rel(path):
    from mdfm.gates import common
    return common.rel(path)


def decode_dirs(end, out=None, cache=None):
    """(report folder, per-trial cache folder) of a decode."""
    return (os.path.join(out or OUT, f"decode_{end.name}"), os.path.join(cache or CACHE, f"decode_{end.name}", "trials"))


def token_dirs(end, out=None, cache=None):
    """(report folder, feature file) of a token readout."""
    return (os.path.join(out or OUT, f"tokens_{end.name}"), os.path.join(cache or CACHE, f"tokens_{end.name}",
                                                                          "features.npz"))


# ---------------------------------------------------------------- persons
def sets_of(limit=0):
    """(train, dev, dev12) person lists: g3's folds (54 and 14, disjoint, no lockbox person, asserted) and the 12 of
    g3.DEV_SETS; with limit, the first `limit` persons of each (smoke)."""
    from mdfm.eval import probes
    from mdfm.gates import g3
    train, dev = g3.controls("train"), g3.controls("dev")
    if len(train) != N_TRAIN or len(dev) != N_DEV or set(train) & set(dev):
        raise AssertionError(f"folds: train {len(train)}, dev {len(dev)}, overlap {sorted(set(train) & set(dev))}")
    locked = set(g3.lockbox_persons())
    if (set(train) | set(dev)) & locked:
        raise Refused("a lockbox person among the non-lockbox controls")
    if limit:
        train = train[:max(int(limit), 2 * probes.FOLDS)]
        dev = dev[:max(int(limit), 2)]
    dev12 = [p for p in dev if p not in set(g3.DEV_SETS["dev12"])]
    return train, dev, dev12


def wrist_input(rec):
    """The G3 model input of a record cut to the wrists (eval_g3_diag.cut_sites); None when it holds no wrist."""
    from mdfm.gates import g3
    D = DIAG()
    r = g3.model_input(rec)
    c = D.cut_sites(r, D.site_ids(WRIST_SITES))
    if c is None:
        return r
    if isinstance(c, str):
        return None
    return c


def wrist_records(persons, trials, log=None):
    """(trial_id, wrist-only model input) of the persons' records, each checked (readouts.check_record); a record
    without a wrist is skipped and counted in log['no_wrist'] when log is a dict."""
    from mdfm.baselines import readouts as RO
    from mdfm.gates import g3
    for tid, rec in g3.imu_records(persons, trials):
        RO.check_record(tid, rec, persons)
        r = wrist_input(rec)
        if r is None:
            if log is not None:
                log.setdefault("no_wrist", []).append(tid)
            continue
        yield tid, r


# ---------------------------------------------------------------- wrist signals and the classical terms
def wrist_arrays(rec):
    """{site: (T, 6) acc (g) and gyro (deg/s)} of the wrists of a record whose six channels are in the channel mask;
    a sample whose sample_valid is False or with a non-finite value is NaN."""
    D = DIAG()
    sig = np.asarray(rec["signal"], dtype=np.float64)
    loc = [int(x) for x in np.asarray(rec["location_ids"]).ravel()]
    cm = np.asarray(rec["channel_mask"], dtype=bool)
    sv = (np.asarray(rec["sample_valid"], dtype=bool) if rec.get("sample_valid") is not None
          else np.ones(sig.shape[:2], bool))
    out = {}
    for name in WRIST_SITES:
        lid = D.SITE_ID[name]
        if lid in loc:
            j = loc.index(lid)
            if cm[j, :6].all():
                x = sig[:, j, :6].copy()
                bad = ~sv[:, j] | ~np.isfinite(x).all(axis=1)
                x[bad] = np.nan
                out[name] = x
    return out


def sagittal(gyro):
    """(T,) the gyroscope (T, 3) minus its mean over the valid samples, projected on its first principal axis; NaN
    where a sample is not valid; None with fewer than 3 valid samples."""
    g = np.asarray(gyro, dtype=np.float64)
    ok = np.isfinite(g).all(axis=1)
    if ok.sum() < 3:
        return None
    c = g[ok] - g[ok].mean(axis=0)
    v = np.linalg.svd(c, full_matrices=False)[2][0]
    out = np.full(len(g), np.nan)
    out[ok] = c @ v
    return out


def autocorr(x, kmax):
    """(kmax + 1,) autocorrelation of x (NaN as missing): x minus its mean over the valid samples, missing samples as
    0, each lag's sum over the count of its valid pairs, divided by lag 0; NaN at a lag with fewer than MIN_PAIRS
    valid pairs (and everywhere when lag 0 is not positive)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    n = len(x)
    r = np.full(kmax + 1, np.nan)
    if ok.sum() < 2:
        return r
    a = np.where(ok, x - x[ok].mean(), 0.0)
    w = ok.astype(np.float64)
    for k in range(min(kmax, n - 1) + 1):
        cnt = float((w[:n - k] * w[k:]).sum())
        if cnt >= MIN_PAIRS or k == 0:
            r[k] = float((a[:n - k] * a[k:]).sum()) / max(cnt, 1.0)
    if not (np.isfinite(r[0]) and r[0] > 0):
        return np.full(kmax + 1, np.nan)
    return r / r[0]


def peak_lag(r, lo, hi):
    """(lag in samples, value) of the highest finite r in [lo, hi], refined by a parabola through its neighbours
    (shift clipped to half a sample); (NaN, NaN) without a finite value there."""
    r = np.asarray(r, dtype=np.float64)
    hi = min(int(hi), len(r) - 1)
    seg = r[int(lo):hi + 1]
    if not len(seg) or not np.isfinite(seg).any():
        return float("nan"), float("nan")
    k = int(lo) + int(np.nanargmax(seg))
    shift = 0.0
    if 0 < k < len(r) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
        den = r[k - 1] - 2.0 * r[k] + r[k + 1]
        if den != 0:
            shift = float(np.clip(0.5 * (r[k - 1] - r[k + 1]) / den, -0.5, 0.5))
    return k + shift, float(r[k])


def pass_terms(wr, t0, t1, rate=RATE, margin=SPAN_MARGIN_S):
    """The classical terms of one pass from the wrist arrays of its record (prediction section 2 (a)): {f_stride,
    f_step (1/s), lag_s, peak, A, G, n_wrists}; NaN terms without a usable wrist or a stride peak."""
    lo, hi = (int(round(s * rate)) for s in STRIDE_LAG_S)
    rs, As, Gs = [], [], []
    for name in sorted(wr):
        x = wr[name]
        a = max(int(round((t0 - margin) * rate)), 0)
        b = min(int(round((t1 + margin) * rate)) + 1, x.shape[0])
        seg = x[a:b]
        if not len(seg):
            continue
        ok = np.isfinite(seg).all(axis=1)
        if ok.mean() < MIN_VALID:
            continue
        s = sagittal(seg[:, 3:6])
        if s is None:
            continue
        rs.append(autocorr(s, hi))
        an = np.linalg.norm(seg[ok, :3], axis=1)
        As.append(float(an.max() - an.min()) ** 0.25)
        Gs.append(float(np.sqrt(np.nanmean(s * s))))
    nan = float("nan")
    out = {"f_stride": nan, "f_step": nan, "lag_s": nan, "peak": nan, "A": nan, "G": nan, "n_wrists": len(rs)}
    if not rs:
        return out
    with np.errstate(invalid="ignore"):
        r = np.nanmean(np.stack(rs), axis=0) if len(rs) > 1 else rs[0]
    lag, val = peak_lag(r, lo, hi)
    out.update(A=float(np.mean(As)), G=float(np.mean(Gs)))
    if np.isfinite(lag) and lag > 0:
        out.update(f_stride=rate / lag, f_step=2.0 * rate / lag, lag_s=lag / rate, peak=val)
    return out


def walkway_cadence(ics, t0, t1):
    """steps/s of the walkway ICs (both feet) inside [t0, t1]: (n - 1) / (last - first); NaN under 3 ICs."""
    t = np.sort(np.asarray(ics, dtype=np.float64))
    t = t[(t >= t0 - 1e-9) & (t <= t1 + 1e-9)]
    if len(t) < 3 or t[-1] <= t[0]:
        return float("nan")
    return float((len(t) - 1) / (t[-1] - t[0]))


# ---------------------------------------------------------------- per-pass wrist features (a and b rows)
def unit_features(records, units, persons, log=print):
    """Per pass of `units` (probes.pass_units): the classical terms (pass_terms) and the mean of the 96 wrist window
    features over its supervised-baseline windows; and the window rows of b2. records: (trial_id, wrist-only input).
    Returns {'terms' DataFrame (n_units, ...), 'win_mean' (n_units, 96), 'X_win' (N, 96), 'idx_win' (N,),
    'counts'}."""
    import pandas as pd
    S = SB()
    G2 = S.G2
    sites = [G2.SITES13.index(s) for s in WRIST_SITES]
    nfeat = len(sites) * 6 * (5 + len(G2.BANDS))
    by_trial = {t: g.index.to_numpy() for t, g in units.groupby("trial_id")}
    terms = [None] * len(units)
    win_mean = np.full((len(units), nfeat), np.nan)
    Xw, iw = [], []
    seen = set()
    n_rec = 0
    for tid, rec in records:
        if rec.get("person_id") not in set(persons):
            raise AssertionError(f"{tid}: person outside the set")
        seen.add(tid)
        n_rec += 1
        if tid not in by_trial:
            continue
        wr = wrist_arrays(rec)
        x13 = S.site_array(rec)
        T = x13.shape[0]
        for i in by_trial[tid]:
            t0, t1 = float(units.at[i, "t0"]), float(units.at[i, "t1"])
            terms[i] = pass_terms(wr, t0, t1)
            st = S.window_starts(t0, t1, T)
            if not st:
                continue
            F = np.empty((len(st), nfeat))
            with np.errstate(invalid="ignore", divide="ignore"):
                for k, s in enumerate(st):
                    F[k] = np.asarray(G2.features(x13[s:s + S.WIN], sites), np.float64)
            Xw.append(F)
            iw += [i] * len(st)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                win_mean[i] = np.nanmean(F, axis=0)
    empty = {"f_stride": np.nan, "f_step": np.nan, "lag_s": np.nan, "peak": np.nan, "A": np.nan, "G": np.nan,
             "n_wrists": 0}
    tf = pd.DataFrame([t if t is not None else empty for t in terms])
    Xw = np.concatenate(Xw) if Xw else np.zeros((0, nfeat))
    counts = {"records": n_rec, "passes": int(len(units)), "trials_without_record": sorted(set(units["trial_id"]) - seen),
              "passes_with_cadence": int(np.isfinite(tf["f_step"]).sum()), "windows": int(len(iw)),
              "wrists_used": {str(k): int(v) for k, v in tf["n_wrists"].value_counts().sort_index().items()}}
    log(f"  wrist features: {n_rec} records, {counts['passes_with_cadence']} of {len(units)} passes with a cadence, "
        f"{len(iw)} windows", flush=True)
    return {"terms": tf, "win_mean": win_mean, "X_win": Xw, "idx_win": np.asarray(iw, dtype=np.int64), "counts": counts}


# ---------------------------------------------------------------- classical and supervised rows
def ols_fit(X, y):
    """Least-squares coefficients of y on [1, X] over the rows with finite values."""
    X = np.asarray(X, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if ok.sum() <= X.shape[1] + 1:
        raise ValueError(f"{int(ok.sum())} finite rows for {X.shape[1] + 1} coefficients")
    A = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    return np.linalg.lstsq(A, y[ok], rcond=None)[0]


def ols_predict(coef, X):
    X = np.asarray(X, dtype=np.float64).reshape(-1, len(coef) - 1)
    return coef[0] + X @ coef[1:]


def classical_a1(terms, s_bar):
    """a1 pass predictions: stride s_bar where the cadence exists, speed s_bar x f_stride."""
    f = terms["f_stride"].to_numpy(np.float64)
    has = np.isfinite(f)
    return {"stride_m": np.where(has, s_bar, np.nan), "speed_m_s": np.where(has, s_bar * f, np.nan)}


def form_crossfit(terms, units, tg, form):
    """(stride line point on the fitting set, out-of-fold stride predictions) of an a2 form: OLS fitted on the other
    person folds (probes.cross_fit_splits), scored by g3.stride_lines."""
    from mdfm.eval import probes
    from mdfm.gates import g3
    X = terms[list(FORMS[form])].to_numpy(np.float64)
    y = units["stride_m"].to_numpy(np.float64)
    p = units["person_id"].astype(str).to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, te in probes.cross_fit_splits(p):
        probes.check_disjoint(p[tr], p[te])
        oof[te] = ols_predict(ols_fit(X[tr], y[tr]), X[te])
    pred = probes.pass_predictions(units, {"stride_m": oof, "speed_m_s": np.full(len(y), np.nan)}, tg, "a2_cf")
    return float(g3.stride_lines(pred, tg)[0]["stride_mae_cm"]["point"]), oof


def classical_a2(terms_tr, units_tr, tg_train, terms_dev):
    """({'chosen', 'crossfit' {form: point}, 'coef' {form: list}}, dev pass predictions) of a2 (module docstring)."""
    cf = {f: form_crossfit(terms_tr, units_tr, tg_train, f)[0] for f in FORMS}
    chosen = min(FORMS, key=lambda f: (not np.isfinite(cf[f]), cf[f] if np.isfinite(cf[f]) else 0.0))
    coef = {f: ols_fit(terms_tr[list(FORMS[f])].to_numpy(np.float64), units_tr["stride_m"].to_numpy(np.float64))
            for f in FORMS}
    stride = ols_predict(coef[chosen], terms_dev[list(FORMS[chosen])].to_numpy(np.float64))
    speed = stride * terms_dev["f_stride"].to_numpy(np.float64)
    return ({"chosen": chosen, "crossfit_stride_mae_cm": cf, "coef": {f: [float(c) for c in v] for f, v in coef.items()},
             "terms": {f: ["1"] + list(FORMS[f]) for f in FORMS}},
            {"stride_m": stride, "speed_m_s": speed})


def b1_matrix(feat):
    """(n_units, 99) b1 features: the 96 window-feature means and the three classical terms."""
    return np.column_stack([feat["win_mean"], feat["terms"][list(CLASSICAL_TERMS)].to_numpy(np.float64)])


def impute(X_fit, X_other):
    """(X_fit, X_other) with every non-finite value replaced by the column median of X_fit (0 for a column without a
    finite value in X_fit)."""
    X_fit = np.array(X_fit, dtype=np.float64, copy=True)
    X_other = np.array(X_other, dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(X_fit), X_fit, np.nan), axis=0) if len(X_fit) else np.zeros(X_fit.shape[1])
    med = np.where(np.isfinite(med), med, 0.0)
    for X in (X_fit, X_other):
        bad = ~np.isfinite(X)
        X[bad] = np.take(med, np.nonzero(bad)[1])
    return X_fit, X_other


def supervised_b1(feat_tr, units_tr, feat_dev):
    """({target: {'alpha'}}, dev pass predictions) of the ridge b1, fitted on the training units only."""
    from mdfm.eval import probes
    Xtr, Xdev = impute(b1_matrix(feat_tr), b1_matrix(feat_dev))
    p = units_tr["person_id"].astype(str).to_numpy()
    out, sel = {}, {}
    for t in ("stride_m", "speed_m_s"):
        m, s = probes.fit_probe(Xtr, units_tr[t].to_numpy(np.float64), p)
        out[t] = m.predict(Xdev)
        sel[t] = {"alpha": s["alpha"]}
    return sel, out


def supervised_b2(feat_tr, units_tr, feat_dev, n_dev):
    """Dev pass predictions of b2: the supervised baseline's gradient boosting on the wrist window rows."""
    return SB().fit_and_predict_units(feat_tr["X_win"], feat_tr["idx_win"], units_tr, feat_dev["X_win"],
                                      feat_dev["idx_win"], n_dev)


# ---------------------------------------------------------------- the decoded body (--decode)
def trial_record(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def decode_trial(pred, tid, rec, tg):
    """{ics, strides, speeds, log} of one trial: the wrist-only body (pred: infer.G3Predictor) read by
    g3.body_predictions on the targets' pass intervals."""
    from mdfm.gates import g3
    tb = pred(tid, rec)
    P = g3.body_predictions([tb], tg)
    return {"ics": [[str(r.foot), float(r.t_s)] for r in P.ics.itertuples()],
            "strides": [[str(r.foot), float(r.t0), float(r.t1), float(r.length_m)] for r in P.strides.itertuples()],
            "speeds": [[str(r.pass_id), float(r.speed_m_s)] for r in P.speeds.itertuples()],
            "log": pred.log[-1] if pred.log else {}}


def run_decode(key, limit=0, out=None, cache=None, log=print):
    """--decode KEY (module docstring)."""
    import pandas as pd

    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    check_prediction()
    end = END[key]
    if end.kind != "body":
        raise SystemExit(f"--decode reads a body checkpoint; {key} is a {end.kind}")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    train, dev, _ = sets_of(0)
    if limit:                                           # smoke: no fold is fitted here, so a few persons suffice
        train, dev = train[:int(limit)], dev[:int(limit)]
    rep, tdir = decode_dirs(end, out, cache)
    os.makedirs(tdir, exist_ok=True)
    ld = I.load_checkpoint(os.path.join(TREE, end.path), weights=DECODE_SETTINGS["weights"], device="cpu")
    if ld.sha256 != end.sha256:
        raise AssertionError(f"{end.path}: SHA-256 {ld.sha256}, expected {end.sha256}")
    trained = ld.trained_persons()
    if set(dev) & trained:
        raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev) & trained)}")
    pred = I.G3Predictor(ld, refine_steps=DECODE_SETTINGS["refine_steps"], points=DECODE_SETTINGS["points"],
                         batch_windows=DECODE_SETTINGS["batch_windows"], seed=DECODE_SETTINGS["seed"],
                         pool=DECODE_SETTINGS["pool"])
    settings = dict(DECODE_SETTINGS, window_s=I.WIN_S, hop_s=I.HOP_S, sites=list(WRIST_SITES))
    rows = {"ics": [], "strides": [], "speeds": []}
    counts = {}
    for set_name, persons in (("dev14", dev), ("train54", train)):
        tg = g3.load_targets(persons)
        if tg.counts["lockbox_rows"]:
            raise Refused("a lockbox target row was kept")
        info = {"run": 0, "kept": 0}
        n = len(tg.trials)
        for k, (tid, rec) in enumerate(wrist_records(persons, tg.trials, info)):
            path = os.path.join(tdir, f"{tid}.json")
            prev = trial_record(path)
            if prev and prev.get("checkpoint_sha256") == end.sha256 and prev.get("settings") == settings \
                    and prev.get("set") == set_name:
                d = prev
                info["kept"] += 1
            else:
                t0 = time.time()
                d = {"trial_id": tid, "person_id": str(rec.get("person_id")), "set": set_name,
                     "checkpoint_sha256": end.sha256, "settings": settings, **decode_trial(pred, tid, rec, tg)}
                d = clean(d)
                common.write_text(path, json.dumps(d) + "\n")
                info["run"] += 1
                log(f"  {set_name} {k + 1}/{n} {tid}: {len(d['strides'])} strides ({time.time() - t0:.0f} s)",
                    flush=True)
            for kind in rows:
                rows[kind] += [[set_name, tid, d["person_id"]] + list(r) for r in d[kind]]
        counts[set_name] = {"trials": n, **info, "persons": len(persons), "target_counts": tg.counts}
    cols = {"ics": ["foot", "t_s"], "strides": ["foot", "t0", "t1", "length_m"], "speeds": ["pass_id", "speed_m_s"]}
    files = {}
    for kind, c in cols.items():
        df = pd.DataFrame(rows[kind], columns=["set", "trial_id", "person_id"] + c)
        files[kind] = common.write_csv(df, os.path.join(rep, f"{kind}.csv"))
    res = {"what": LABEL, "task": TASK, "mode": "decode", "run_started": started, "partial": bool(limit),
           "checkpoint": {**ld.describe(), "key": end.key}, "settings": settings,
           "trained_persons_among_train54": len(set(train) & trained), "counts": counts,
           "files": {k: {"path": rel(v), "sha256": common.sha256_file(v)} for k, v in files.items()},
           "prediction_sha256": PREDICTION_SHA256, "moveatlas_head": PV.moveatlas_head(),
           "timing_s": round(time.time() - t_start, 1)}
    common.write_text(os.path.join(rep, "decode.json"), json.dumps(clean(res), indent=1) + "\n")
    common.write_text(os.path.join(rep, "_provenance.json"), json.dumps(clean(provenance(started, end)), indent=1) + "\n")
    log(f"decode {end.name}: written {rel(rep)} ({res['timing_s']} s)", flush=True)
    return res


# ---------------------------------------------------------------- token probes (--tokens)
def run_tokens(key, limit=0, out=None, cache=None, from_features=False, log=print):
    """--tokens KEY (module docstring): readouts.collect on wrist-only records, readouts.run_probes."""
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    check_prediction()
    end = END[key]
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    train, dev, _ = sets_of(limit)
    rep, feat_path = token_dirs(end, out, cache)
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise Refused("a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    skipped = {}
    if from_features:
        data, meta = RO.load_features(feat_path)
        if (meta.get("checkpoint") or {}).get("sha256") != end.sha256:
            raise SystemExit(f"{rel(feat_path)} holds another checkpoint's features")
        for s, u in (("train", u_tr), ("dev", u_dev)):
            f = data["units"][s]
            if list(f["trial_id"]) != list(u["trial_id"]) or list(f["pass_id"].astype(str)) != list(u["pass_id"].astype(str)):
                raise SystemExit(f"{rel(feat_path)}: {s} passes differ from the targets read now")
            data["units"][s] = u
        ck, train_in_ckpt, t_tok = meta["checkpoint"], meta["train_among_checkpoint_persons"], meta.get("tokens_s")
    else:
        kind, model, info = RO.load_any(os.path.join(TREE, end.path), weights="ema", device="cpu")
        if info["sha256"] != end.sha256:
            raise AssertionError(f"{end.path}: SHA-256 {info['sha256']}, expected {end.sha256}")
        if kind != end.kind:
            raise AssertionError(f"{end.path}: loaded as {kind}, expected {end.kind}")
        trained = info["trained_persons"]
        if set(dev) & trained:
            raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev) & trained)}")
        train_in_ckpt = len(set(train) & trained)
        ck = info["describe"]

        def encode(wins):
            return RO.encode_windows(model, wins, RO.BATCH_WINDOWS)

        t0 = time.time()
        data = {"units": {"train": u_tr, "dev": u_dev}}
        data["train"] = RO.collect(encode, wrist_records(train, tg_train.trials, skipped), u_tr, tg_train, train,
                                   log=lambda *a, **k: None)
        data["dev"] = RO.collect(encode, wrist_records(dev, tg_dev.trials, skipped), u_dev, tg_dev, dev,
                                 log=lambda *a, **k: None)
        t_tok = round(time.time() - t0, 1)
        meta = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": ck, "kind": kind, "task": TASK,
                "inference": {"device": "cpu", "amp_bf16": False, "sites": list(WRIST_SITES), "decoder": False},
                "train_among_checkpoint_persons": train_in_ckpt, "tokens_s": t_tok, "limit_persons": limit}
        RO.save_features(feat_path, data, meta)
        del model
        log(f"tokens {end.name}: features written ({t_tok} s)", flush=True)
    t0 = time.time()
    res0 = RO.run_probes(data, tg_train, tg_dev, log=lambda *a, **k: None)
    t_probe = round(time.time() - t0, 1)
    rows_path = common.write_csv(res0["rows"], os.path.join(rep, "per_row.csv"))
    res = {"what": LABEL, "task": TASK, "mode": "tokens", "run_started": started, "partial": bool(limit),
           "key": end.key, "arm": end.arm, "seed": end.seed, "kind": end.kind, "checkpoint": ck,
           "input": "G3 model input cut to wrist_L and wrist_R", "device": "cpu",
           "train_among_checkpoint_persons": train_in_ckpt, "sets": res0["sets"], "chosen": res0["chosen"],
           "crossfit": res0["crossfit"], "lines": res0["lines"], "per_person": res0["per_person"],
           "variants": res0["variants"], "rows": {"path": rel(rows_path), "sha256": common.sha256_file(rows_path),
                                                  "n_rows": int(len(res0["rows"])), "check": res0["rows_check"]},
           "counts": {"train": data["train"]["counts"], "dev": data["dev"]["counts"], "no_wrist": skipped.get("no_wrist", [])},
           "features": {"path": rel(feat_path), "sha256": common.sha256_file(feat_path)},
           "prediction_sha256": PREDICTION_SHA256, "moveatlas_head": PV.moveatlas_head(),
           "timing_s": {"tokens": t_tok, "probes": t_probe, "all": round(time.time() - t_start, 1)}}
    common.write_text(os.path.join(rep, "results.json"), json.dumps(clean(res), indent=1) + "\n")
    common.write_text(os.path.join(rep, "_provenance.json"), json.dumps(clean(provenance(started, end)), indent=1) + "\n")
    L = res0["lines"]
    log(f"tokens {end.name}: dev12 stride {L['dev12']['stride_mae_cm']['point']:.2f} cm "
        f"({res0['chosen']['stride_mae_cm']}), speed {L['dev12']['speed_mae_m_s']['point']:.3f} m/s; written {rel(rep)}",
        flush=True)
    return res


# ---------------------------------------------------------------- lines, rows and comparisons (read)
def row_frames(pred, tg, name):
    """(stats {line: g3 stats}, ROW_COLUMNS frame of both lines) of a Predictions on a Targets."""
    import pandas as pd

    from mdfm.baselines import readouts as RO
    from mdfm.gates import g3
    st, st_tab = g3.stride_lines(pred, tg)
    sp, sp_tab = g3.speed_line(pred, tg)
    frame = pd.concat([RO._pass_rows(st_tab, "stride_mae_cm", name, "err_cm", "pred_m"),
                       RO._pass_rows(sp_tab, "speed_mae_m_s", name, "err_m_s", "pred_m_s")], ignore_index=True)
    return {"stride_mae_cm": RO.strip(st["stride_mae_cm"]), "speed_mae_m_s": RO.strip(sp)}, frame


def set_stats(frame, persons, line):
    """The line's point on a set from a row frame (comparison.line_point) with g3's person bootstrap interval and the
    matched share of the set's units."""
    from mdfm.baselines import comparison as CMP
    from mdfm.gates import common
    f = CMP.of_line(frame, line)
    f = f[f["person_id"].astype(str).isin(set(persons))]
    ids, v = CMP.person_values(f, "mean")
    point = float(np.mean(v)) if len(ids) else float("nan")
    iv = (list(common.cluster_bootstrap(v, ids, n=BOOT_N, seed=SEED)) if len(ids) >= 2 and np.isfinite(v).all()
          else None)
    m = f["matched"].astype(bool) if len(f) else f["matched"]
    return {"point": point, "interval": iv, "n_persons": int(len(ids)), "matched_share": float(m.mean()) if len(f) else
            float("nan"), "n_events": int(len(f)), "per_person": {str(i): float(x) for i, x in zip(ids, v)}}


def pass_units_of(frame, line, persons):
    """Per-pass units (person_id, x walkway, y predicted) of a row frame: stride, the matched strides of a pass
    averaged; speed, each matched pass."""
    import pandas as pd
    from mdfm.baselines import comparison as CMP
    f = CMP.of_line(frame, line)
    f = f[f["matched"].astype(bool) & f["person_id"].astype(str).isin(set(persons))]
    if line == "speed_mae_m_s":
        return pd.DataFrame({"person_id": f["person_id"].astype(str), "x": f["target"].astype(float),
                             "y": f["prediction"].astype(float)}).reset_index(drop=True)
    g = f.groupby(["person_id", "trial_id", "pass_id"], sort=True)
    return g.agg(x=("target", "mean"), y=("prediction", "mean")).reset_index()[["person_id", "x", "y"]]


def spread(points):
    p = [float(x) for x in points]
    return float(max(p) - min(p)) if len(p) > 1 else 0.0


def premium_test(body_frames, comp_frames, persons, line):
    """The premium rule (module docstring) of body seeds against one bar's seeds on a line and set."""
    from mdfm.baselines import comparison as CMP
    bf = [CMP.of_line(f, line) for f in body_frames]
    cf = [CMP.of_line(f, line) for f in comp_frames]
    bp = [CMP.line_point(f, "mean", persons) for f in bf]
    cp = [CMP.line_point(f, "mean", persons) for f in cf]
    sep = CMP.separation(bp, cp)
    boot = CMP.bootstrap_line(bf, cf, persons, "mean", BOOT_N, SEED)
    iv = boot["interval"]
    bm, cm = CMP.seed_mean(bp), CMP.seed_mean(cp)
    diff = bm - cm
    d = max(spread(bp), spread(cp))
    m = max(MARGIN_FRACTION * cm, 2.0 * d) if np.isfinite(cm) else float("nan")
    premium = bool(sep == CMP.BODY and iv is not None and iv[1] < 0 and np.isfinite(diff) and diff <= -m)
    loss = bool(sep == "comparator" and iv is not None and iv[0] > 0)
    return {"body_points": bp, "comparator_points": cp, "body_mean": bm, "comparator_mean": cm, "difference": diff,
            "interval": iv, "undefined_resamples": boot["undefined"], "separation": sep, "seed_spread": d,
            "margin": m, "required_body_mean": cm - m if np.isfinite(m) else float("nan"), "premium": premium,
            "loss": loss}


def best_bar(bars, names, persons, line):
    """The bar of `names` with the lowest seed-mean point on the line and set (ties to the earlier name)."""
    from mdfm.baselines import comparison as CMP
    best, key = None, None
    for n in names:
        pts = [CMP.line_point(CMP.of_line(f, line), "mean", persons) for f in bars[n]]
        m = CMP.seed_mean(pts)
        k = (not np.isfinite(m), m if np.isfinite(m) else 0.0)
        if best is None or k < key:
            best, key = n, k
    return best


def comparisons(arms, bars, sets):
    """{arm: {line: {set: {'best': test with 'bar', group: test with 'bar'}}}} and {arm: {line: reading}}."""
    every = [n for g in GROUPS.values() for n in g]
    out, readings = {}, {}
    for arm, frames in arms.items():
        out[arm], readings[arm] = {}, {}
        for line in LINES:
            out[arm][line] = {}
            for s in SETS:
                persons = sets[s]
                block = {}
                for label, names in (("best", every), *GROUPS.items()):
                    b = best_bar(bars, names, persons, line)
                    block[label] = {"bar": b, **premium_test(frames, bars[b], persons, line)}
                out[arm][line][s] = block
            prem = all(out[arm][line][s]["best"]["premium"] for s in SETS)
            loss = all(out[arm][line][s]["best"]["loss"] for s in SETS)
            readings[arm][line] = "PREMIUM" if prem else "LOSS" if loss else "NO PREMIUM"
    return out, readings


# ---------------------------------------------------------------- the read
def read_decode(end, train, dev, out=None):
    """{set: Predictions} of a decode folder (dev14, train54), its decode.json; the persons of the file are checked
    against the sets before any value is used."""
    import pandas as pd

    from mdfm.gates import g3
    rep, _ = decode_dirs(end, out)
    with open(os.path.join(rep, "decode.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["checkpoint"]["sha256"] != end.sha256:
        raise AssertionError(f"{rep}: decode of another checkpoint")
    allowed = {"dev14": set(dev), "train54": set(train)}
    frames = {k: pd.read_csv(os.path.join(rep, f"{k}.csv"), encoding="utf-8", dtype={"set": str, "trial_id": str,
                             "person_id": str, "foot": str, "pass_id": str}) for k in ("ics", "strides", "speeds")}
    for k, df in frames.items():
        for s, g in df.groupby("set"):
            bad = set(g["person_id"]) - allowed.get(s, set())
            if bad:
                raise Refused(f"{k}.csv: {len(bad)} persons outside the {s} set")
    out_p = {}
    for s in ("dev14", "train54"):
        sel = {k: df[df["set"] == s] for k, df in frames.items()}
        out_p[s] = g3.Predictions(f"body_{end.key}", sel["ics"][["trial_id", "foot", "t_s"]].reset_index(drop=True),
                                  sel["strides"][["trial_id", "foot", "t0", "t1", "length_m"]].reset_index(drop=True),
                                  sel["speeds"][["trial_id", "pass_id", "speed_m_s"]].reset_index(drop=True))
    return out_p, meta


def body_gain(P_train, tg_train, train, dev):
    """{stride, speed}: F1's fit_gain on the matched strides and passes of the 54 (decoded over walkway)."""
    import pandas as pd

    from mdfm.gates import g3
    G = GAIN()
    locked = set(g3.lockbox_persons())
    ms = g3.match_strides(P_train, tg_train)
    ms = ms[ms["matched"]]
    mv = g3.match_speeds(P_train, tg_train)
    mv = mv[mv["matched"]]
    us = pd.DataFrame({"person_id": ms["person_id"].astype(str), "x": ms["length_m"], "y": ms["pred_m"]})
    uv = pd.DataFrame({"person_id": mv["person_id"].astype(str), "x": mv["speed_m_s"], "y": mv["pred_m_s"]})
    return {"stride": G.fit_gain(us, train, dev, locked), "speed": G.fit_gain(uv, train, dev, locked),
            "matched_train": {"strides": int(len(ms)), "of_strides": int(len(tg_train.strides)), "passes": int(len(mv)),
                              "of_passes": int(len(tg_train.passes))}}


def scaled(P, g_s, g_v, name):
    """Predictions with the strides divided by g_s and the speeds by g_v (same times: the matching is unchanged)."""
    from mdfm.gates import g3
    st = P.strides.copy()
    st["length_m"] = st["length_m"] / g_s
    sp = P.speeds.copy()
    sp["speed_m_s"] = sp["speed_m_s"] / g_v
    return g3.Predictions(name, P.ics, st, sp)


def body_ic(P, tg_dev, sets):
    """Descriptive: the body's wrist-only IC line (walkway) on each set, offset fitted on the 14 (g3's rule)."""
    from mdfm.eval import probes
    from mdfm.gates import g3
    from mdfm.baselines import readouts as RO
    p = g3.Predictions("body_ic", P.ics)
    off, n = g3.fit_offsets([p], tg_dev)[("body_ic", "walkway")]
    out = {}
    for s, persons in sets.items():
        tg = tg_dev if set(persons) == set(tg_dev.persons) else probes.subset_targets(tg_dev, persons)
        line, _ = g3.ic_line(p, g3.reference_ics(tg, "walkway"), off)
        out[s] = RO.strip(line)
    return {"offset_ms": 1000.0 * off if g3.offset_fitted(off) else None, "offset_n": int(n), "sets": out}


def token_rows(end, out=None):
    """(results dict, {line: chosen-variant frame of the 14}) of a token readout folder."""
    import pandas as pd
    rep, _ = token_dirs(end, out)
    with open(os.path.join(rep, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    if (res.get("checkpoint") or {}).get("sha256") != end.sha256:
        raise AssertionError(f"{rep}: tokens of another checkpoint")
    rows = pd.read_csv(os.path.join(rep, "per_row.csv"), encoding="utf-8", dtype={"pass_id": str, "trial_id": str,
                                                                                   "person_id": str, "foot": str})
    rows["matched"] = rows["matched"].astype(str).str.lower() == "true"
    rows["enters"] = rows["enters"].astype(str).str.lower() == "true"
    frames = []
    for line in LINES:
        v = res["chosen"][line]
        frames.append(rows[(rows["line"] == line) & (rows["variant"] == v)])
    return res, pd.concat(frames, ignore_index=True)


def context_rows(tree=TREE):
    """All-site rows of the same checkpoints and the GPU family_wrists rows, read from their reports (beside, no
    reading): {row: {set: {line: point}}, 'sources': {row: path}}."""
    out, src = {}, {}
    names = {"dev12": ("S12", "V12"), "dev14": ("S14", "V14")}
    p = os.path.join(tree, "reports", "bench", "g3_gain_check", "results.json")
    if os.path.exists(p):
        g = json.load(open(p, encoding="utf-8"))
        for k in BODY_KEYS:
            e = g["ends"].get(k) or {}
            rc = e.get("recalibration") or {}
            for arm, part in (("d_raw", "raw"), ("d_gain", "recalibrated")):
                out.setdefault(f"all_sites_{arm}_{k}", {})
                for s, (sk, vk) in names.items():
                    out[f"all_sites_{arm}_{k}"][s] = {"stride_mae_cm": (rc.get(part) or {}).get(sk, {}).get("point"),
                                                      "speed_mae_m_s": (rc.get(part) or {}).get(vk, {}).get("point")}
                src[f"all_sites_{arm}_{k}"] = rel(p)
    for e in ENDS:
        p = os.path.join(tree, "reports", "bench", "g12", "readouts", e.name, "results.json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            out[f"all_sites_tok_{e.key}"] = {s: {ln: (r["lines"].get(s) or {}).get(ln, {}).get("point") for ln in LINES}
                                             for s in SETS}
            src[f"all_sites_tok_{e.key}"] = rel(p)
    p = os.path.join(tree, "reports", "bench", "g3_supervised_baseline", "results.json")
    if os.path.exists(p):
        r = json.load(open(p, encoding="utf-8"))
        out["all_sites_supervised"] = {s: {ln: r["lines"][s]["supervised"][ln]["point"] for ln in LINES} for s in SETS}
        src["all_sites_supervised"] = rel(p)
    for k in BODY_KEYS:
        p = os.path.join(tree, "reports", "gates", "g3", f"diag_{END[k].name}_ablations_bw32", "diag.json")
        if os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))["rows"]["family_wrists"]
            out[f"gpu_family_wrists_{k}"] = {s: {"stride_mae_cm": r[sub]["stride_mae_cm"], "speed_mae_m_s": r[sub]["speed_mae_ms"]}
                                             for s, sub in (("dev14", "all14"), ("dev12", "no_whc003_whc026"))}
            src[f"gpu_family_wrists_{k}"] = rel(p)
    return {"rows": out, "sources": src}


def within(lo, hi, x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return bool(lo <= v <= hi) if math.isfinite(v) else None


def predictions_check(res):
    """P1 to P9 of prediction.txt section 4 from the results: {id: {'what', 'values', 'met'}}."""
    R = res["rows"]

    def pt(row, s, line):
        return ((R.get(row) or {}).get(s) or {}).get(line, {}).get("point")

    out = {}

    def add(pid, what, checks):
        vals = {k: v for k, (v, _) in checks.items()}
        oks = [ok for _, ok in checks.values()]
        out[pid] = {"what": what, "values": vals, "met": all(o is True for o in oks) if oks else None}

    add("P1", "CPU decode against the GPU family_wrists rows (d-raw)", {
        "v1_B dev12 stride in [22.4, 26.4]": (pt("d_raw_v1_B", "dev12", "stride_mae_cm"), within(22.4, 26.4, pt("d_raw_v1_B", "dev12", "stride_mae_cm"))),
        "s2_B dev12 stride in [21.7, 25.7]": (pt("d_raw_s2_B", "dev12", "stride_mae_cm"), within(21.7, 25.7, pt("d_raw_s2_B", "dev12", "stride_mae_cm"))),
        "v1_B dev14 stride in [24.7, 28.7]": (pt("d_raw_v1_B", "dev14", "stride_mae_cm"), within(24.7, 28.7, pt("d_raw_v1_B", "dev14", "stride_mae_cm"))),
        "s2_B dev14 stride in [23.8, 27.8]": (pt("d_raw_s2_B", "dev14", "stride_mae_cm"), within(23.8, 27.8, pt("d_raw_s2_B", "dev14", "stride_mae_cm"))),
        "v1_B dev12 speed in [0.21, 0.29]": (pt("d_raw_v1_B", "dev12", "speed_mae_m_s"), within(0.21, 0.29, pt("d_raw_v1_B", "dev12", "speed_mae_m_s"))),
        "s2_B dev12 speed in [0.21, 0.29]": (pt("d_raw_s2_B", "dev12", "speed_mae_m_s"), within(0.21, 0.29, pt("d_raw_s2_B", "dev12", "speed_mae_m_s")))})
    g = res.get("gains") or {}
    chk = {}
    for k in BODY_KEYS:
        gs = ((g.get(k) or {}).get("stride") or {}).get("gain")
        gv = ((g.get(k) or {}).get("speed") or {}).get("gain")
        chk[f"{k} g_S in [0.80, 0.90]"] = (gs, within(0.80, 0.90, gs))
        chk[f"{k} g_V in [0.78, 0.88]"] = (gv, within(0.78, 0.88, gv))
    add("P2", "wrist-only training-fold gains", chk)
    chk = {}
    for k in BODY_KEYS:
        chk[f"d_gain {k} dev12 stride in [6.5, 13.0]"] = (pt(f"d_gain_{k}", "dev12", "stride_mae_cm"), within(6.5, 13.0, pt(f"d_gain_{k}", "dev12", "stride_mae_cm")))
        chk[f"d_gain {k} dev12 speed in [0.05, 0.11]"] = (pt(f"d_gain_{k}", "dev12", "speed_mae_m_s"), within(0.05, 0.11, pt(f"d_gain_{k}", "dev12", "speed_mae_m_s")))
    add("P3", "decoded body with one training-fold gain", chk)
    add("P4", "a1", {"stride dev12 in [28, 36]": (pt("a1", "dev12", "stride_mae_cm"), within(28, 36, pt("a1", "dev12", "stride_mae_cm"))),
                     "speed dev12 in [0.10, 0.20]": (pt("a1", "dev12", "speed_mae_m_s"), within(0.10, 0.20, pt("a1", "dev12", "speed_mae_m_s")))})
    add("P5", "a2", {"stride dev12 in [9, 16]": (pt("a2", "dev12", "stride_mae_cm"), within(9, 16, pt("a2", "dev12", "stride_mae_cm"))),
                     "speed dev12 in [0.07, 0.15]": (pt("a2", "dev12", "speed_mae_m_s"), within(0.07, 0.15, pt("a2", "dev12", "speed_mae_m_s")))})
    add("P6", "b1 and b2", {
        "b1 stride dev12 in [8, 15]": (pt("b1", "dev12", "stride_mae_cm"), within(8, 15, pt("b1", "dev12", "stride_mae_cm"))),
        "b1 speed dev12 in [0.06, 0.12]": (pt("b1", "dev12", "speed_mae_m_s"), within(0.06, 0.12, pt("b1", "dev12", "speed_mae_m_s"))),
        "b2 stride dev12 in [8, 16]": (pt("b2", "dev12", "stride_mae_cm"), within(8, 16, pt("b2", "dev12", "stride_mae_cm"))),
        "b2 speed dev12 in [0.06, 0.13]": (pt("b2", "dev12", "speed_mae_m_s"), within(0.06, 0.13, pt("b2", "dev12", "speed_mae_m_s")))})
    chk = {}
    for arm, keys in TOKEN_ARMS.items():
        lo, hi = (9, 19) if arm == "tok_jepa" else (7, 15)
        for k in keys:
            v = pt(f"{arm}_{k}", "dev12", "stride_mae_cm")
            chk[f"{arm} {k} stride dev12 in [{lo}, {hi}]"] = (v, within(lo, hi, v))
            v = pt(f"{arm}_{k}", "dev12", "speed_mae_m_s")
            chk[f"{arm} {k} speed dev12 in [0.05, 0.13]"] = (v, within(0.05, 0.13, v))
    add("P7", "token probes, chosen variants", chk)
    c = (res.get("cadence") or {}).get("dev14") or {}
    add("P8", "cadence on the dev passes", {
        "median absolute relative error at most 0.03": (c.get("median_abs_rel_error"), None if c.get("median_abs_rel_error") is None else bool(c["median_abs_rel_error"] <= 0.03)),
        "share within 5 percent at least 0.90": (c.get("share_within_5pct"), None if c.get("share_within_5pct") is None else bool(c["share_within_5pct"] >= 0.90))})
    rd = res.get("readings") or {}
    add("P9", "readings", {
        "d_raw stride LOSS": ((rd.get("d_raw") or {}).get("stride_mae_cm"), (rd.get("d_raw") or {}).get("stride_mae_cm") == "LOSS"),
        "d_raw speed LOSS": ((rd.get("d_raw") or {}).get("speed_mae_m_s"), (rd.get("d_raw") or {}).get("speed_mae_m_s") == "LOSS"),
        "d_gain stride not PREMIUM": ((rd.get("d_gain") or {}).get("stride_mae_cm"), (rd.get("d_gain") or {}).get("stride_mae_cm") in ("NO PREMIUM", "LOSS")),
        "d_gain speed not PREMIUM": ((rd.get("d_gain") or {}).get("speed_mae_m_s"), (rd.get("d_gain") or {}).get("speed_mae_m_s") in ("NO PREMIUM", "LOSS"))})
    return out


def run_read(limit=0, out=None, log=print):
    """The default mode (module docstring)."""
    import pandas as pd

    from mdfm.eval import probes
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    check_prediction()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    rep = os.path.abspath(out or OUT)
    train, dev, dev12 = sets_of(limit)
    sets = {"dev12": dev12, "dev14": dev}
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise Refused("a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    log("wrist features (training fold, then dev)", flush=True)
    skipped = {}
    f_tr = unit_features(wrist_records(train, tg_train.trials, skipped), u_tr, train, log)
    f_dev = unit_features(wrist_records(dev, tg_dev.trials, skipped), u_dev, dev, log)
    preds, details = {}, {}
    s_bar = common.person_weighted_mean(tg_train.strides["length_m"], tg_train.strides["person_id"])
    preds["a1"] = classical_a1(f_dev["terms"], s_bar)
    details["a1"] = {"s_bar_m": s_bar, "what": "training-fold person-weighted mean stride over both tasks"}
    details["a2"], preds["a2"] = classical_a2(f_tr["terms"], u_tr, tg_train, f_dev["terms"])
    details["b1"], preds["b1"] = supervised_b1(f_tr, u_tr, f_dev)
    preds["b2"] = supervised_b2(f_tr, u_tr, f_dev, len(u_dev))
    details["b2"] = {"params": SB().HGB_PARAMS, "windows_train": int(len(f_tr["idx_win"])),
                     "windows_dev": int(len(f_dev["idx_win"]))}
    rows, frames = {}, {}
    for name, p in preds.items():
        P = probes.pass_predictions(u_dev, p, tg_dev, name)
        _, frames[name] = row_frames(P, tg_dev, name)
    base = g3.population_baseline(tg_train)
    _, frames["population"] = row_frames(g3.population_predictions(base, tg_dev), tg_dev, "population")
    # decoded body
    gains, ic_lines, decode_meta = {}, {}, {}
    for k in BODY_KEYS:
        P, meta = read_decode(END[k], train, dev, out)
        decode_meta[k] = {"settings": meta["settings"], "counts": meta["counts"], "timing_s": meta["timing_s"],
                          "moveatlas_head": meta["moveatlas_head"], "files": meta["files"]}
        gains[k] = body_gain(P["train54"], tg_train, train, dev)
        gs, gv = gains[k]["stride"]["gain"], gains[k]["speed"]["gain"]
        _, frames[f"d_raw_{k}"] = row_frames(P["dev14"], tg_dev, f"d_raw_{k}")
        _, frames[f"d_gain_{k}"] = row_frames(scaled(P["dev14"], gs, gv, f"d_gain_{k}"), tg_dev, f"d_gain_{k}")
        ic_lines[k] = body_ic(P["dev14"], tg_dev, sets)
    # token probes
    tok_meta = {}
    for arm, keys in TOKEN_ARMS.items():
        for k in keys:
            res_t, fr = token_rows(END[k], out)
            frames[f"{arm}_{k}"] = fr.assign(variant=f"{arm}_{k}:" + fr["variant"].astype(str))
            tok_meta[f"{arm}_{k}"] = {"chosen": res_t["chosen"], "crossfit": res_t["crossfit"],
                                      "ic": {s: res_t["lines"][s]["ic_median_ms"] for s in SETS},
                                      "counts": res_t["counts"], "timing_s": res_t["timing_s"],
                                      "moveatlas_head": res_t["moveatlas_head"], "rows": res_t["rows"]}
    # lines of every row on both sets
    for name, fr in frames.items():
        rows[name] = {s: {ln: set_stats(fr, sets[s], ln) for ln in LINES} for s in SETS}
    # g3's own points on the pass rows and the body rows equal the frame points (check)
    worst = 0.0
    for name in ("a1", "a2", "b1", "b2", "population") + tuple(f"d_raw_{k}" for k in BODY_KEYS):
        for s in SETS:
            tg = tg_dev if s == "dev14" else probes.subset_targets(tg_dev, sets[s])
            if name in preds:
                P = probes.pass_predictions(u_dev, preds[name], tg_dev, name)
            elif name == "population":
                P = g3.population_predictions(base, tg_dev)
            else:
                P = read_decode(END[name[len("d_raw_"):]], train, dev, out)[0]["dev14"]
            st, _ = row_frames(P, tg, name)
            for ln in LINES:
                a, b = float(st[ln]["point"]), float(rows[name][s][ln]["point"])
                if np.isnan(a) != np.isnan(b) or (not np.isnan(a) and abs(a - b) > 1e-9 * max(1.0, abs(a))):
                    raise AssertionError(f"{name} {s} {ln}: frame {b}, g3 {a}")
                if not np.isnan(a):
                    worst = max(worst, abs(a - b))
    bars = {"a1": [frames["a1"]], "a2": [frames["a2"]], "b1": [frames["b1"]], "b2": [frames["b2"]]}
    for arm, keys in TOKEN_ARMS.items():
        bars[arm] = [frames[f"{arm}_{k}"] for k in keys]
    arms = {a: [frames[f"{a}_{k}"] for k in BODY_KEYS] for a in ARMS}
    comp, readings = comparisons(arms, bars, sets)
    # post hoc, labelled as such: the gain arm against a1, the bar with the same label count (one number per line)
    equal_labels = {line: {s: premium_test(arms["d_gain"], bars["a1"], sets[s], line) for s in SETS} for line in LINES}
    # within-person slope, r and gain per pass (beside)
    G = GAIN()
    describe = {}
    for name, fr in frames.items():
        describe[name] = {s: {ln: G.describe(pass_units_of(fr, ln, sets[s]), n_boot=BOOT_N, seed=SEED) for ln in LINES}
                          for s in SETS}
    # cadence against the walkway cadence (descriptive)
    cad = {}
    ics = {t: g["t_s"].to_numpy() for t, g in tg_dev.walkway_ic.groupby("trial_id")}
    fs = f_dev["terms"]["f_step"].to_numpy(np.float64)
    wc = np.array([walkway_cadence(ics.get(r.trial_id, ()), r.t0, r.t1) for r in u_dev.itertuples()])
    for s in SETS:
        m = u_dev["person_id"].isin(set(sets[s])).to_numpy() & np.isfinite(fs) & np.isfinite(wc)
        e = np.abs(fs[m] - wc[m]) / wc[m]
        cad[s] = {"passes": int(m.sum()), "of_passes": int(u_dev["person_id"].isin(set(sets[s])).sum()),
                  "median_abs_rel_error": float(np.median(e)) if len(e) else None,
                  "share_within_5pct": float(np.mean(e < 0.05)) if len(e) else None,
                  "pearson_r": float(np.corrcoef(fs[m], wc[m])[0, 1]) if m.sum() > 2 else None}
    per_row = pd.concat([fr.assign(row=name) for name, fr in frames.items()], ignore_index=True)
    per_row["in_dev12"] = per_row["person_id"].isin(set(dev12))
    rows_path = common.write_csv(per_row, os.path.join(rep, "per_row.csv"))
    res = {"what": LABEL, "task": TASK, "run_started": started, "partial": bool(limit),
           "prediction": {"path": rel(PREDICTION_PATH), "sha256": PREDICTION_SHA256},
           "sets": {"train": train, "train_sha256": _person_hash(train), "dev14": dev, "dev14_sha256": _person_hash(dev),
                    "dev12": dev12, "dev12_sha256": _person_hash(dev12)},
           "rows": rows, "details": details, "gains": gains, "body_ic": ic_lines, "decode": decode_meta,
           "tokens": tok_meta, "comparisons": comp, "readings": readings,
           "post_hoc_equal_labels": {"what": "post hoc: d_gain against a1 (one training-fold number per line each), "
                                             "the premium test of the rule", "tests": equal_labels},
           "describe": describe, "cadence": cad,
           "features": {"train": f_tr["counts"], "dev": f_dev["counts"], "no_wrist": skipped.get("no_wrist", [])},
           "context": context_rows(), "frame_check_max_abs_difference": worst,
           "per_row": {"path": rel(rows_path), "sha256": common.sha256_file(rows_path), "n_rows": int(len(per_row))},
           "constants": {"boot_n": BOOT_N, "seed": SEED, "margin_fraction": MARGIN_FRACTION, "span_margin_s": SPAN_MARGIN_S,
                         "min_valid": MIN_VALID, "stride_lag_s": list(STRIDE_LAG_S), "min_pairs": MIN_PAIRS,
                         "forms": {k: list(v) for k, v in FORMS.items()}, "probe_seed": probes.SEED,
                         "probe_folds": probes.FOLDS},
           "lockbox": {"persons_in_lockbox": len((set(train) | set(dev)) & set(g3.lockbox_persons())),
                       "target_lockbox_rows": int(tg_train.counts["lockbox_rows"]) + int(tg_dev.counts["lockbox_rows"])},
           "moveatlas_head": PV.moveatlas_head()}
    res["predictions"] = predictions_check(res)
    res["timing_s"] = round(time.time() - t_start, 1)
    res = clean(res)
    text = summary_md(res)
    common.write_text(os.path.join(rep, "results.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(rep, "summary.md"), text)
    common.write_text(os.path.join(rep, "_provenance.json"), json.dumps(clean(provenance(started, None)), indent=1) + "\n")
    log(text)
    return res


def _person_hash(persons):
    from mdfm.data import sources as SRC
    return SRC.person_hash(sorted(persons))


# ---------------------------------------------------------------- output helpers
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if x in ("inf", "-inf") or (isinstance(x, float) and math.isinf(x)):
        return str(x)
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else ("inf" if o > 0 else "-inf" if o < 0 else None)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def provenance(started, end):
    from mdfm.gates import common
    st = DIAG().stamp_moveatlas()
    files = [os.path.abspath(__file__), os.path.join(HERE, "eval_g3_diag.py"), os.path.join(HERE, "g3_gain_check.py"),
             os.path.join(HERE, "g3_supervised_baseline.py"), os.path.join(HERE, "g2_renderer_realism.py"),
             os.path.join(TREE, "mdfm", "gates", "g3.py"), os.path.join(TREE, "mdfm", "eval", "probes.py"),
             os.path.join(TREE, "mdfm", "eval", "infer.py"), os.path.join(TREE, "mdfm", "baselines", "readouts.py"),
             os.path.join(TREE, "mdfm", "baselines", "comparison.py"), os.path.join(TREE, "mdfm", "readouts", "gait.py")]
    import torch
    return {"script": "scripts/g3_wrist_bars.py", "label": LABEL, "task": TASK, "run_at": started,
            "checkpoint": None if end is None else {"key": end.key, "path": end.path, "sha256": end.sha256},
            "prediction_sha256": PREDICTION_SHA256, "code_sha256": common.code_sha256(files),
            "code_files": [rel(f) for f in files], "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "moveatlas_stamp": st,
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}


ROW_TITLES = (("a1", "a1 classical: wrist cadence x training-fold mean step"),
              ("a2", "a2 classical: amplitude model fitted on the 54"),
              ("b1", "b1 supervised ridge, wrist features"),
              ("b2", "b2 supervised gradient boosting, wrist window features"),
              ("tok_mfr_mfr_s1", "token probe, MFR seed 20260928"), ("tok_mfr_mfr_s2", "token probe, MFR seed 20261001"),
              ("tok_jepa_jepa_s1", "token probe, JEPA seed 20260928"),
              ("tok_jepa_jepa_s2", "token probe, JEPA seed 20261001"),
              ("tok_body_v1_B", "token probe, body v1_B"), ("tok_body_s2_B", "token probe, body s2_B"),
              ("d_raw_v1_B", "decoded body v1_B, raw"), ("d_raw_s2_B", "decoded body s2_B, raw"),
              ("d_gain_v1_B", "decoded body v1_B, one training-fold gain"),
              ("d_gain_s2_B", "decoded body s2_B, one training-fold gain"),
              ("population", "population (g3's trivial baseline, per task)"))


def summary_md(r):
    """summary.md: every number read from the results dict as written to results.json."""
    L = [f"# Wrist-only G3 bars on the dev controls ({r['what']})", "",
         "Written by scripts/g3_wrist_bars.py (task WRS; strategy memo docs/notes/STRATEGY_2026-10-15.md, section 4, rank 4) "
         f"under {r['prediction']['path']} (SHA-256 {r['prediction']['sha256']}, asserted before any target, record or "
         "checkpoint was read). Every number below is read from results.json of this folder. Input of every row: the G3 "
         "model input cut to wrist_L and wrist_R. Fits on the 54 training-fold controls only; scored on dev12 and dev14 "
         "with the G3 harness's lines and person bootstrap (2,000 resamples, seed 20260929). No lockbox, EVAL, CODY or "
         f"care_pd record. MoveAtlas {r.get('moveatlas_head')}.", "",
         "## Rows", "",
         "| row | dev12 stride (cm) | 95 % | dev12 speed (m/s) | 95 % | dev14 stride (cm) | 95 % | dev14 speed (m/s) | 95 % | "
         "strides matched (dev14) | within-person slope, stride per pass (dev12) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, title in ROW_TITLES:
        x = r["rows"].get(key)
        if not x:
            continue
        a, b = x["dev12"], x["dev14"]
        ws = ((r["describe"].get(key) or {}).get("dev12") or {}).get("stride_mae_cm", {}).get("within_slope", {})
        L.append(f"| {title} | {fmt(a['stride_mae_cm']['point'])} | {fmt_iv(a['stride_mae_cm']['interval'])} | "
                 f"{fmt(a['speed_mae_m_s']['point'], 3)} | {fmt_iv(a['speed_mae_m_s']['interval'], 3)} | "
                 f"{fmt(b['stride_mae_cm']['point'])} | {fmt_iv(b['stride_mae_cm']['interval'])} | "
                 f"{fmt(b['speed_mae_m_s']['point'], 3)} | {fmt_iv(b['speed_mae_m_s']['interval'], 3)} | "
                 f"{fmt(b['stride_mae_cm']['matched_share'], 3)} | {fmt(ws.get('point'), 3)} {fmt_iv(ws.get('interval'), 3)} |")
    L += ["", "Pass rows (a, b, token probes, population) give every walkway stride of a pass its pass's prediction, so "
          "every stride matches; the decoded body is matched on its own strides (both ends within 0.15 s), as in G3.", ""]
    d = r["details"]
    L += ["## Classical and supervised details", "",
          f"- a1: training-fold mean stride {fmt(d['a1']['s_bar_m'], 4)} m (task-blind).",
          f"- a2: form chosen {d['a2']['chosen']} by person cross-fitted stride line on the 54: " +
          ", ".join(f"{f} {fmt(v)} cm" for f, v in d["a2"]["crossfit_stride_mae_cm"].items()) +
          f"; coefficients of the chosen form ({', '.join(d['a2']['terms'][d['a2']['chosen']])}): " +
          ", ".join(fmt(c, 4) for c in d["a2"]["coef"][d["a2"]["chosen"]]) + ".",
          f"- b1: ridge penalties stride {fmt(d['b1']['stride_m']['alpha'], 2)}, speed {fmt(d['b1']['speed_m_s']['alpha'], 2)}.",
          f"- b2: {d['b2']['windows_train']} training windows, {d['b2']['windows_dev']} dev windows.", ""]
    c = r["cadence"]
    L += ["Wrist cadence against the walkway cadence of the dev passes (descriptive): " +
          "; ".join(f"{s}: {c[s]['passes']} of {c[s]['of_passes']} passes, median absolute relative error "
                    f"{fmt(c[s]['median_abs_rel_error'], 4)}, within 5 percent {fmt(c[s]['share_within_5pct'], 3)}, Pearson "
                    f"{fmt(c[s]['pearson_r'], 3)}" for s in SETS) + ".", ""]
    L += ["## Decoded body: gains and IC (wrist-only input)", "",
          "| seed | g_S (54, per stride) | strides matched (54) | g_V (54, per pass) | IC walkway dev12 (ms) | dev14 (ms) | IC "
          "matched share dev14 | offset (ms) |", "|---|---|---|---|---|---|---|---|"]
    for k in BODY_KEYS:
        g = r["gains"][k]
        ic = r["body_ic"][k]
        L.append(f"| {k} | {fmt(g['stride']['gain'], 4)} | {g['matched_train']['strides']} of {g['matched_train']['of_strides']} "
                 f"| {fmt(g['speed']['gain'], 4)} | {fmt(ic['sets']['dev12'].get('point'), 1)} | "
                 f"{fmt(ic['sets']['dev14'].get('point'), 1)} | {fmt(ic['sets']['dev14'].get('matched_share'), 3)} | "
                 f"{fmt(ic.get('offset_ms'), 1)} |")
    L += ["", "Token IC probes (walkway, chosen variant, offset fitted on the 14; descriptive): " +
          "; ".join(f"{k} {fmt(v['ic']['dev12'].get('point'), 1)} / {fmt(v['ic']['dev14'].get('point'), 1)} ms"
                    for k, v in r["tokens"].items()) + " (dev12 / dev14).", ""]
    L += ["## Premium rule", "",
          "Comparator: the bar (a1, a2, b1, b2, token probes of body, MFR, JEPA; seed means) with the lowest point. "
          "Premium: both body seeds below every comparator seed, the paired person-bootstrap interval of the seed-mean "
          "difference below 0, and the difference at most -m, m = max(0.10 x comparator point, 2 x the larger seed spread). "
          "PREMIUM on a line needs it on dev12 and dev14; LOSS is the mirror (separation and interval); NO PREMIUM otherwise.", "",
          "| arm | line | set | comparator | body seed mean | comparator mean | difference | 95 % | margin m | body mean a "
          "premium needs (at most) | separation | premium | loss |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for line in LINES:
            nd = 2 if line == "stride_mae_cm" else 3
            for s in SETS:
                t = r["comparisons"][arm][line][s]["best"]
                L.append(f"| {arm} | {line} | {s} | {t['bar']} | {fmt(t['body_mean'], nd)} | {fmt(t['comparator_mean'], nd)} | "
                         f"{fmt(t['difference'], nd)} | {fmt_iv(t['interval'], nd)} | {fmt(t['margin'], nd)} | "
                         f"{fmt(t['required_body_mean'], nd)} | {t['separation']} | {t['premium']} | {t['loss']} |")
    L += ["", "Readings: " + "; ".join(f"{arm} {line}: **{r['readings'][arm][line]}**" for arm in ARMS for line in LINES) + ".",
          "", "Against each group's best bar (difference body minus bar, 95 % interval, premium / loss):", "",
          "| arm | line | set | classical | supervised | token |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for line in LINES:
            nd = 2 if line == "stride_mae_cm" else 3
            for s in SETS:
                b = r["comparisons"][arm][line][s]
                cells = [f"{b[g]['bar']}: {fmt(b[g]['difference'], nd)} {fmt_iv(b[g]['interval'], nd)} "
                         f"({'premium' if b[g]['premium'] else 'loss' if b[g]['loss'] else 'neither'})" for g in GROUPS]
                L.append(f"| {arm} | {line} | {s} | " + " | ".join(cells) + " |")
    L += ["", "Post hoc (not in prediction.txt): the gain arm against a1, the bar with the same label count (one training-fold "
          "number per line), by the same test:", "", "| line | set | d_gain mean | a1 | difference | 95 % | premium | loss |",
          "|---|---|---|---|---|---|---|---|"]
    for line in LINES:
        nd = 2 if line == "stride_mae_cm" else 3
        for s in SETS:
            t = r["post_hoc_equal_labels"]["tests"][line][s]
            L.append(f"| {line} | {s} | {fmt(t['body_mean'], nd)} | {fmt(t['comparator_mean'], nd)} | {fmt(t['difference'], nd)} "
                     f"| {fmt_iv(t['interval'], nd)} | {t['premium']} | {t['loss']} |")
    ctx = r["context"]["rows"]
    L += ["", "## Beside: all-site rows of the same checkpoints and the GPU family_wrists rows (read from their reports)", "",
          "| row | dev12 stride | dev12 speed | dev14 stride | dev14 speed | source |", "|---|---|---|---|---|---|"]
    for k, v in ctx.items():
        L.append(f"| {k} | {fmt((v.get('dev12') or {}).get('stride_mae_cm'))} | {fmt((v.get('dev12') or {}).get('speed_mae_m_s'), 3)} | "
                 f"{fmt((v.get('dev14') or {}).get('stride_mae_cm'))} | {fmt((v.get('dev14') or {}).get('speed_mae_m_s'), 3)} | "
                 f"{r['context']['sources'].get(k)} |")
    L += ["", "## Predictions (prediction.txt section 4)", "", "| id | what | met | values |", "|---|---|---|---|"]
    for pid, p in r["predictions"].items():
        vals = "; ".join(f"{k}: {fmt(v, 4) if fin(v) is not None else v}" for k, v in p["values"].items())
        L.append(f"| {pid} | {p['what']} | {p['met']} | {vals} |")
    L += ["", "## Checks", "",
          f"- Lockbox: persons of the sets in the lockbox list {r['lockbox']['persons_in_lockbox']}; target lockbox rows "
          f"{r['lockbox']['target_lockbox_rows']} (asserted 0).",
          f"- Row frames against g3's own points (a, b, population, decoded raw): largest difference "
          f"{r['frame_check_max_abs_difference']}.",
          f"- Records without a wrist: {len(r['features']['no_wrist'])}.",
          f"- Per-row file {r['per_row']['path']} (SHA-256 {r['per_row']['sha256']}, {r['per_row']['n_rows']} rows).",
          f"- Decodes: " + "; ".join(f"{k} {v['settings']['device']}, {v['timing_s']} s, MoveAtlas {v['moveatlas_head']}"
                                     for k, v in r["decode"].items()) + ".",
          f"- Token readouts: " + "; ".join(f"{k} {v['timing_s'].get('all')} s, MoveAtlas {v['moveatlas_head']}"
                                            for k, v in r["tokens"].items()) + ".",
          f"- Time of this read: {r.get('timing_s')} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def job_commands():
    py = "<conda>/envs/mdfm/python.exe"
    cmds = [f"{py} scripts/g3_wrist_bars.py --decode {k}" for k in BODY_KEYS]
    cmds += [f"{py} scripts/g3_wrist_bars.py --tokens {e.key}" for e in ENDS]
    cmds.append(f"{py} scripts/g3_wrist_bars.py")
    return cmds


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--decode", choices=BODY_KEYS)
    g.add_argument("--tokens", choices=tuple(END))
    g.add_argument("--print-jobs", action="store_true")
    p.add_argument("--from-features", action="store_true", help="--tokens: refit from the feature file")
    p.add_argument("--limit-persons", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--cache", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_jobs:
        print("\n".join(job_commands()))
        return 0
    if a.limit_persons and not (a.out and a.cache):
        raise SystemExit("--limit-persons needs --out and --cache (a smoke run never takes the default folders)")
    print(f"process priority: {SB().below_normal()} (the GPU training's data loader shares the CPU)", flush=True)
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if a.decode:
        run_decode(a.decode, a.limit_persons, a.out, a.cache)
    elif a.tokens:
        run_tokens(a.tokens, a.limit_persons, a.out, a.cache, a.from_features)
    else:
        run_read(a.limit_persons, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
