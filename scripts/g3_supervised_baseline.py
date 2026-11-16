"""G3 supervised baseline: a supervised lower bar for real-sensor gait readouts (weekend task T34). Every output
is labelled 'supervised baseline, dev only, not a G3 read'.

Rule: reports/bench/g3_supervised_baseline/rule.txt (version 1, written 2026-10-01 before any target was read).
Its SHA-256 sits beside it in rule.sha256 and here in RULE_SHA256; check_rule asserts the three agree before any
target is read (checked_targets), so a rule edited after the fact stops the run.

What the rule fixes, in short:
- Persons: fit on the 54 training-fold weargait_pd controls (mdfm.gates.g3.controls("train")), score on the 14 dev
  controls and on the 12 of them without WHC003 (no shank sensors) and WHC026 (shank sides that match the other
  foot's walkway ICs, docs/rules/G3_RULE.md). No lockbox person is read: the person sets are asserted disjoint
  from mdfm.gates.g3.lockbox_persons(), targets come from g3.load_targets (lockbox rows refused and counted, the
  count asserted 0) and records from g3.imu_records (canonical.load with every reader flag False), each asserted
  to carry lockbox False and to belong to the requested set.
- Input: the IMU records of SelfPace and HurriedPace as the model reads them in G3 (g3.model_input), acc and gyro
  of the 13 Xsens sites (g2_renderer_realism.to_sites13), invalid samples NaN.
- Units: walkway passes of the G3 targets; interval from the pass speed interval and the pass's walkway strides;
  targets the mean walkway stride of the pass (m) and the pass speed (m/s).
- Features: the window features of scripts/g2_renderer_realism.py (features, imported), on 2 s windows whose
  centres lie 1 s apart, symmetric about the pass midpoint, at least one per pass.
- Model: HistGradientBoostingRegressor with the fixed HGB_PARAMS, one per target, trained on windows labelled with
  their pass target; the pass prediction is the mean over its windows. A person-grouped 5-fold cross-validation
  inside the 54 is reported as a diagnostic and selects nothing.
- Scoring: the G3 lines (g3.stride_lines, g3.speed_line) with every walkway stride of a pass given its pass's
  predicted stride; person-clustered bootstrap intervals of g3 (2000 resamples, seed 20260929). Reference rows:
  the population baseline of g3 and the walkway itself.

Never: the GPU (CUDA_VISIBLE_DEVICES=-1 is set before torch loads), a lockbox, EVAL or CODY record. The process
runs at BelowNormal priority and pauses while cache/runs/queue/SLOW exists.

Writes reports/bench/g3_supervised_baseline/: results.json, summary.md and _provenance.json (scripts/
moveatlas_stamp.py), after rule.txt and rule.sha256 (the run asserts that both are older than its outputs).
Run:
  <conda>/envs/mdfm/python.exe scripts/g3_supervised_baseline.py
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
import g2_renderer_realism as G2  # noqa: E402
from mdfm.gates import common, g3  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g3_supervised_baseline")
RULE_PATH = os.path.join(OUT, "rule.txt")
RULE_SHA_PATH = os.path.join(OUT, "rule.sha256")
RULE_SHA256 = "0a34402c4c598430d1864eedd75fa1a7ea8724ddc828fe682e3bcf5cc80962ec"
LABEL = "supervised baseline, dev only, not a G3 read"
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
RATE = 100.0
WIN = G2.WIN                      # 200 samples, 2 s
HOP_S = 1.0
N_FEATURES = 13 * 6 * (5 + len(G2.BANDS))
HGB_PARAMS = {"loss": "squared_error", "learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
              "min_samples_leaf": 20, "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
CV_FOLDS = 5
TARGETS = ("stride_m", "speed_m_s")
NAME = "supervised"
UNIT_KEYS = ["trial_id", "person_id", "task", "pass_id"]


# ---------------------------------------------------------------- rule, machine discipline
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_rule(rule_path=RULE_PATH, sha_path=RULE_SHA_PATH, expected=RULE_SHA256):
    """The SHA-256 of the rule file, asserted equal to the first word of the hash file and to `expected`
    (AssertionError otherwise, raised explicitly so that python -O keeps it)."""
    sha = sha256_file(rule_path)
    with open(sha_path, encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (sha == stored == expected):
        raise AssertionError(f"rule hash mismatch: {os.path.basename(rule_path)} {sha}, "
                             f"{os.path.basename(sha_path)} {stored or 'empty'}, script {expected}")
    return sha


def checked_targets(sets, loader=None, rule_path=RULE_PATH, sha_path=RULE_SHA_PATH, expected=RULE_SHA256):
    """{name: Targets} of the person sets {name: persons}, read by `loader` (default g3.load_targets) only after
    check_rule passed; every set's kept lockbox rows are asserted to be 0. Returns (rule SHA-256, targets)."""
    sha = check_rule(rule_path, sha_path, expected)
    loader = g3.load_targets if loader is None else loader
    out = {}
    for name, persons in sets.items():
        tg = loader(persons)
        if tg.counts.get("lockbox_rows", 0):
            raise AssertionError(f"{name}: {tg.counts['lockbox_rows']} lockbox target rows kept")
        out[name] = tg
    return sha, out


def below_normal():
    """BelowNormal priority for this process (Windows); returns what was done."""
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


def wait_while_slow(path=SLOW, poll_s=60.0, log=print):
    """Sleep while the SLOW flag exists; returns the seconds waited."""
    t0 = time.time()
    told = False
    while os.path.exists(path):
        if not told:
            log(f"paused: {path} exists", flush=True)
            told = True
        time.sleep(poll_s)
    return time.time() - t0


# ---------------------------------------------------------------- lockbox guard
def assert_no_lockbox(sets, locked):
    """AssertionError when a person of any set {name: persons} is in `locked`."""
    locked = set(locked)
    for name, persons in sets.items():
        hit = sorted(set(persons) & locked)
        if hit:
            raise AssertionError(f"{name}: {len(hit)} lockbox persons requested")


def check_record(tid, rec, persons):
    """AssertionError unless the record carries lockbox False, no EVAL tier, and a person of `persons`."""
    lb = rec.get("lockbox")
    if lb is None or (lb is not False and str(lb).strip().lower() not in ("false", "0")):
        raise AssertionError(f"{tid}: record with lockbox {lb!r}")
    if str(rec.get("tier", "")) in ("EVAL", "EXCLUDED_V1"):
        raise AssertionError(f"{tid}: record of tier {rec.get('tier')}")
    if rec.get("person_id") not in set(persons):
        raise AssertionError(f"{tid}: person {rec.get('person_id')!r} outside the requested set")


# ---------------------------------------------------------------- units and windows
def pass_units(targets):
    """One row per walkway pass (trial_id, person_id, task, pass_id): t0, t1 (the span of the pass speed
    interval and of the pass's walkway strides), n_strides, stride_m (mean walkway stride, NaN without one),
    speed_m_s (pass speed, NaN without one)."""
    import pandas as pd
    s = targets.strides.copy()
    p = targets.passes.copy()
    s["pass_id"] = s["pass_id"].astype(str)
    p["pass_id"] = p["pass_id"].astype(str)
    if p.duplicated(UNIT_KEYS).any():
        raise ValueError("a pass listed twice in the targets")
    sg = (s.groupby(UNIT_KEYS, sort=True)
          .agg(s_t0=("t0", "min"), s_t1=("t1", "max"), stride_m=("length_m", "mean"), n_strides=("length_m", "size"))
          .reset_index()) if len(s) else pd.DataFrame(columns=UNIT_KEYS + ["s_t0", "s_t1", "stride_m", "n_strides"])
    pg = p[UNIT_KEYS + ["t0", "t1", "speed_m_s"]].rename(columns={"t0": "p_t0", "t1": "p_t1"})
    u = sg.merge(pg, on=UNIT_KEYS, how="outer")
    for c in ("s_t0", "s_t1", "p_t0", "p_t1", "stride_m", "speed_m_s"):
        u[c] = pd.to_numeric(u[c], errors="coerce").astype(float)
    u["t0"] = u[["s_t0", "p_t0"]].min(axis=1)
    u["t1"] = u[["s_t1", "p_t1"]].max(axis=1)
    u["n_strides"] = pd.to_numeric(u["n_strides"], errors="coerce").fillna(0).astype(int)
    u = u.sort_values(UNIT_KEYS, kind="stable").reset_index(drop=True)
    return u[UNIT_KEYS + ["t0", "t1", "n_strides", "stride_m", "speed_m_s"]]


def window_centres(t0, t1, hop_s=HOP_S):
    """floor((t1 - t0) / hop) + 1 centres (s), hop apart, symmetric about the midpoint of [t0, t1]."""
    n = int(math.floor((t1 - t0) / hop_s + 1e-9)) + 1
    mid = 0.5 * (t0 + t1)
    return [mid + (k - (n - 1) / 2.0) * hop_s for k in range(max(n, 1))]


def window_starts(t0, t1, n_samples, rate=RATE, win=WIN, hop_s=HOP_S):
    """First samples of the windows of a pass: each window holds `win` samples that start win // 2 before its
    centre (rounded to the sample grid), shifted inside [0, n_samples); none when the record is shorter."""
    if n_samples < win:
        return []
    out = []
    for c in window_centres(t0, t1, hop_s):
        s = int(round(c * rate)) - win // 2
        out.append(min(max(s, 0), n_samples - win))
    return out


def site_array(rec):
    """(T, 13, 6) float32: acc and gyro of the 13 sites (G2.SITES13 order) of the record as the model reads it in
    G3 (g3.model_input); a sample whose sample_valid is False, a channel whose channel_mask is False and a site the
    record lacks are NaN."""
    r = g3.model_input(rec)
    sig = np.array(r["signal"], dtype=np.float32, copy=True)
    cm = np.asarray(r["channel_mask"], dtype=bool)
    sig[:, ~cm] = np.nan
    valid = r.get("sample_valid")
    if valid is not None:
        sig[~np.asarray(valid, dtype=bool)] = np.nan
    return G2.to_sites13(sig, np.asarray(r["location_ids"]))


def window_features(x, starts):
    """(len(starts), N_FEATURES) float32: G2.features of each window x[s:s + WIN] (NaN where a channel has a
    non-finite sample in the window)."""
    out = np.empty((len(starts), N_FEATURES), np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, s in enumerate(starts):
            out[i] = np.asarray(G2.features(x[s:s + WIN]), np.float32)
    return out


def build_rows(records, units, persons, wait=None):
    """Window rows of the units: records iterable of (trial_id, record), each checked (check_record) against
    `persons`. Returns (X (N, N_FEATURES) float32, unit index (N,), counts dict)."""
    by_trial = {t: g.index.to_numpy() for t, g in units.groupby("trial_id")}
    X, idx = [], []
    n_rec = n_short = 0
    seen = set()
    for tid, rec in records:
        check_record(tid, rec, persons)
        seen.add(tid)
        n_rec += 1
        if tid not in by_trial:
            continue
        x = site_array(rec)
        T = x.shape[0]
        for i in by_trial[tid]:
            st = window_starts(units.at[i, "t0"], units.at[i, "t1"], T)
            if not st:
                n_short += 1
                continue
            X.append(window_features(x, st))
            idx += [i] * len(st)
        del x, rec
        if wait is not None:
            wait()
    X = np.concatenate(X) if X else np.empty((0, N_FEATURES), np.float32)
    idx = np.asarray(idx, dtype=np.int64)
    covered = np.zeros(len(units), bool)
    covered[np.unique(idx)] = True
    counts = {"records": n_rec, "trials_with_passes": int(units["trial_id"].nunique()),
              "trials_without_record": sorted(set(units["trial_id"]) - seen), "passes": int(len(units)),
              "passes_with_windows": int(covered.sum()), "passes_without_window": int((~covered).sum()),
              "windows": int(len(idx)), "windows_per_pass_median": float(np.median(np.bincount(idx)[np.unique(idx)]))
              if len(idx) else float("nan"),
              "passes_with_stride_target": int(units["stride_m"].notna().sum()),
              "passes_with_speed_target": int(units["speed_m_s"].notna().sum()),
              "window_feature_nan_share": float(np.isnan(X).mean()) if len(X) else float("nan")}
    return X, idx, counts


# ---------------------------------------------------------------- model
def make_model(params=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**(HGB_PARAMS if params is None else params))


def fit_predict(X_tr, y_tr, X_te, params=None):
    """Window predictions on X_te of a regressor fitted on (X_tr, y_tr)."""
    m = make_model(params)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def pass_means(values, idx, n_units):
    """Mean of window values per unit index (NaN for a unit without a window)."""
    s = np.bincount(idx, weights=values, minlength=n_units)
    n = np.bincount(idx, minlength=n_units)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def fit_and_predict_units(X_tr, idx_tr, units_tr, X_te, idx_te, n_te, params=None):
    """{target: (n_te,) pass predictions}: per target a regressor on the training windows whose pass has that
    target (label: the pass target), window predictions averaged per pass."""
    out = {}
    for t in TARGETS:
        y = units_tr[t].to_numpy(dtype=np.float64)[idx_tr]
        keep = np.isfinite(y)
        if not keep.any():
            raise ValueError(f"no training window with a {t} target")
        out[t] = pass_means(fit_predict(X_tr[keep], y[keep], X_te, params), idx_te, n_te) if len(X_te) \
            else np.full(n_te, np.nan)
    return out


def grouped_cv(X, idx, units, folds=CV_FOLDS, params=None):
    """Out-of-fold pass predictions {target: (n_units,)} of a person-grouped k-fold inside the training set."""
    from sklearn.model_selection import GroupKFold
    persons = units["person_id"].to_numpy()[idx]
    out = {t: np.full(len(units), np.nan) for t in TARGETS}
    for tr, te in GroupKFold(n_splits=folds).split(X, groups=persons):
        pred = fit_and_predict_units(X[tr], idx[tr], units, X[te], idx[te], len(units), params)
        te_units = np.unique(idx[te])
        for t in TARGETS:
            out[t][te_units] = pred[t][te_units]
    return out


# ---------------------------------------------------------------- scoring
def predictions_of(units, pred, targets, name=NAME):
    """g3.Predictions of pass predictions {target: (n_units,)}: every walkway stride of the targets receives the
    predicted stride of its pass with its own foot and end times; every pass its predicted speed. A stride or
    pass whose pass has no finite prediction is left out (unmatched in the lines)."""
    import pandas as pd
    key = list(zip(units["trial_id"], units["pass_id"].astype(str)))
    ps = {k: v for k, v in zip(key, pred["stride_m"]) if np.isfinite(v)}
    pv = {k: v for k, v in zip(key, pred["speed_m_s"]) if np.isfinite(v)}
    s = targets.strides
    sk = list(zip(s["trial_id"], s["pass_id"].astype(str)))
    st = s.loc[[k in ps for k in sk], ["trial_id", "foot", "t0", "t1"]].copy()
    st["length_m"] = [ps[k] for k in sk if k in ps]
    sp = pd.DataFrame([(t, p, v) for (t, p), v in pv.items()], columns=["trial_id", "pass_id", "speed_m_s"])
    return g3.Predictions(name, None, st.reset_index(drop=True), sp)


def subset_targets(tg, persons):
    """The Targets restricted to `persons` (every frame filtered on person_id)."""
    keep = set(persons)
    f = lambda d: d[d["person_id"].isin(keep)].reset_index(drop=True)  # noqa: E731
    trials = f(tg.trials)
    offsets = tg.offsets[tg.offsets["trial_id"].isin(set(trials["trial_id"]))].reset_index(drop=True)
    return g3.Targets(tuple(sorted(keep & set(tg.persons))), tg.lockbox, trials, f(tg.walkway_ic), f(tg.insole_ic),
                      f(tg.strides), f(tg.passes), offsets, {"strides": int(len(f(tg.strides))),
                                                             "passes": int(len(f(tg.passes)))})


def score(targets, predictions):
    """{name: {line: stats}} of the G3 stride and speed lines (g3.stride_lines, g3.speed_line) of each
    predictor on the targets."""
    out = {}
    for p in predictions:
        s = {}
        if p.strides is not None:
            lines, _ = g3.stride_lines(p, targets)
            s.update(lines)
        if p.speeds is not None:
            s["speed_mae_m_s"], _ = g3.speed_line(p, targets)
        out[p.name] = s
    return out


def run(tg_train, tg_dev, train_records, dev_records, train_persons, dev_persons, excluded=EXCLUDED_FROM_12,
        params=None, cv=True, wait=None, log=print):
    """The pipeline on given targets and record iterables (main passes the real ones, the tests synthetic
    ones). Returns the results dict (no file written)."""
    t_start = time.time()
    u_tr, u_dev = pass_units(tg_train), pass_units(tg_dev)
    X_tr, i_tr, c_tr = build_rows(train_records, u_tr, train_persons, wait)
    log(f"training windows {len(i_tr)} from {c_tr['passes_with_windows']} passes", flush=True)
    X_dev, i_dev, c_dev = build_rows(dev_records, u_dev, dev_persons, wait)
    log(f"dev windows {len(i_dev)} from {c_dev['passes_with_windows']} passes", flush=True)
    t_feat = time.time() - t_start
    pred = fit_and_predict_units(X_tr, i_tr, u_tr, X_dev, i_dev, len(u_dev), params)
    base = g3.population_baseline(tg_train)
    sets = {"dev14": sorted(dev_persons), "dev12": sorted(set(dev_persons) - set(excluded))}
    lines = {}
    for name, persons in sets.items():
        tg = tg_dev if name == "dev14" else subset_targets(tg_dev, persons)
        preds = [predictions_of(u_dev, pred, tg), g3.population_predictions(base, tg), g3.walkway_predictions(tg)]
        lines[name] = score(tg, preds)
    cv_lines = None
    if cv:
        oof = grouped_cv(X_tr, i_tr, u_tr, CV_FOLDS, params)
        cv_lines = score(tg_train, [predictions_of(u_tr, oof, tg_train, f"{NAME}_cv")])
    per_pass = u_dev.assign(pred_stride_m=pred["stride_m"], pred_speed_m_s=pred["speed_m_s"],
                            n_windows=np.bincount(i_dev, minlength=len(u_dev)) if len(i_dev) else 0)
    return {"label": LABEL, "sets": sets, "excluded_from_12": list(excluded),
            "counts": {"train": c_tr, "dev": c_dev}, "population_baseline": {t: {"stride_m": s, "speed_m_s": v}
                                                                              for t, (s, v) in base.items()},
            "lines": lines, "cv_train": cv_lines, "per_pass": per_pass,
            "timing_s": {"features": round(t_feat, 1), "all": round(time.time() - t_start, 1)}}


# ---------------------------------------------------------------- report
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def summary_md(res):
    """The markdown report of a results dict (as written to results.json)."""
    L = ["# G3 supervised baseline on real IMU gait (" + LABEL + ")", "",
         "Written by scripts/g3_supervised_baseline.py under the rule reports/bench/g3_supervised_baseline/rule.txt "
         f"(SHA-256 {res['rule']['sha256']}, asserted before any target was read). Every number below is read from "
         "results.json of this folder. A dev-only lower bar for G3, not a G3 read: no lockbox person was read "
         "(asserted).", "",
         "Model: HistGradientBoostingRegressor with fixed hyperparameters ("
         + ", ".join(f"{k} {v}" for k, v in res["model"]["params"].items()) + "), one per target, trained on "
         f"{res['counts']['train']['windows']} windows of {res['counts']['train']['passes_with_windows']} walkway passes "
         f"of the {len(res['sets']['train'])} training-fold controls; features: the {res['model']['n_features']} window "
         "features of scripts/g2_renderer_realism.py (2 s windows, 1 s apart, symmetric about the pass midpoint); the "
         "prediction of a pass is the mean over its windows. Targets: the mean walkway stride of the pass and the "
         "pass speed.", "",
         "Scoring: the G3 lines of mdfm/gates/g3.py (stride: every walkway stride gets its pass's predicted stride; "
         "absolute error, mean per person, mean over persons), 95 percent person-clustered bootstrap intervals "
         f"({g3.BOOT_N} resamples, seed {g3.SEED}). G3 lines for context: stride pass 5 cm, kill above 10; speed "
         "pass 0.08 m/s, kill above 0.15.", "",
         "## Dev lines", "",
         "| set | predictor | stride MAE (cm) | 95 % interval | speed MAE (m/s) | 95 % interval | stride MAPE (%) | "
         "persons | strides matched | passes matched |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sname in ("dev14", "dev12"):
        for pname in (NAME, "population", "walkway"):
            s = res["lines"][sname].get(pname, {})
            st, sp, mp = s.get("stride_mae_cm", {}), s.get("speed_mae_m_s", {}), s.get("stride_mape_pct", {})
            L.append(f"| {sname} ({len(res['sets'][sname])}) | {pname} | {fmt(st.get('point'))} | "
                     f"{fmt_iv(st.get('interval'))} | {fmt(sp.get('point'), 3)} | {fmt_iv(sp.get('interval'), 3)} | "
                     f"{fmt(mp.get('point'), 1)} | {st.get('n_persons', 'n/a')} | {fmt(st.get('matched_share'), 3)} of "
                     f"{st.get('n_events', 'n/a')} | {fmt(sp.get('matched_share'), 3)} of {sp.get('n_events', 'n/a')} |")
    L += ["", f"dev12 leaves out {', '.join(res['excluded_from_12'])}. The walkway row is the reference itself (zero "
          "error by construction); the population row is the trivial baseline of g3 (per task, the person-weighted "
          "mean of the training fold).", ""]
    cv = res.get("cv_train")
    if cv:
        c = cv[f"{NAME}_cv"]
        L += ["## Diagnostic: person-grouped 5-fold cross-validation inside the training fold", "",
              f"Selects nothing (the hyperparameters are fixed by the rule). Stride MAE {fmt(c['stride_mae_cm']['point'])} cm "
              f"{fmt_iv(c['stride_mae_cm']['interval'])}, speed MAE {fmt(c['speed_mae_m_s']['point'], 3)} m/s "
              f"{fmt_iv(c['speed_mae_m_s']['interval'], 3)}, over {c['stride_mae_cm']['n_persons']} persons.", ""]
    ct, cd = res["counts"]["train"], res["counts"]["dev"]
    lb = res["lockbox"]
    L += ["## Counts and checks", "",
          "| set | records | passes | with windows | windows | stride targets | speed targets | NaN share of features |",
          "|---|---|---|---|---|---|---|---|"]
    for nm, c in (("train", ct), ("dev", cd)):
        L.append(f"| {nm} | {c['records']} | {c['passes']} | {c['passes_with_windows']} | {c['windows']} | "
                 f"{c['passes_with_stride_target']} | {c['passes_with_speed_target']} | "
                 f"{fmt(c['window_feature_nan_share'], 3)} |")
    L += ["", f"Lockbox: training and dev persons in the weargait_pd lockbox list {lb['persons_in_lockbox']}; lockbox "
          f"target rows kept {lb['target_lockbox_rows']}; records read with lockbox True {lb['records_lockbox_true']} "
          "(each asserted 0; records read through canonical.load with allow_eval and allow_lockbox False).",
          f"Rule file times: rule.txt {res['rule']['rule_mtime']}, rule.sha256 {res['rule']['sha_mtime']}; this run "
          f"started {res['run_started']}.",
          f"MoveAtlas: HEAD {res['moveatlas'].get('head') or 'n/a'} (full stamp in _provenance.json).",
          f"Time: features {res['timing_s']['features']} s, all {res['timing_s']['all']} s.", ""]
    return "\n".join(L) + "\n"


def lines_of(lines):
    return {s: {n: {ln: {k: v for k, v in st.items() if k != "per_person"} | {"per_person": st.get("per_person", {})}
                    for ln, st in d.items()} for n, d in v.items()} for s, v in lines.items()}


# ---------------------------------------------------------------- main
def _stamp():
    import moveatlas_stamp
    return moveatlas_stamp.stamp()


def _mtime(path):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))


def stamp_head(stamp):
    """The MoveAtlas commit of a moveatlas_stamp.stamp() result (its first key that names a HEAD or commit)."""
    if not isinstance(stamp, dict):
        return None
    for k in ("head", "HEAD", "moveatlas_head", "commit", "rev"):
        if stamp.get(k):
            return stamp[k]
    return None


def finalize(res, rule_sha, train, dev, locked, train_counts, dev_counts, n_lockbox_records, stamp, run_started,
             rule_path=RULE_PATH, sha_path=RULE_SHA_PATH):
    """The JSON-ready results (non-finite numbers as null) from a run() dict and the run's checks."""
    from mdfm.data import sources
    res = dict(res)
    per_pass = res.pop("per_pass")
    out = {"what": LABEL, "run_started": run_started,
           "rule": {"path": common.rel(rule_path), "sha256": rule_sha, "sha_file": common.rel(sha_path),
                    "asserted_before_targets": True, "rule_mtime": _mtime(rule_path), "sha_mtime": _mtime(sha_path)},
           "model": {"class": "sklearn.ensemble.HistGradientBoostingRegressor", "params": HGB_PARAMS,
                     "n_features": N_FEATURES, "window_samples": WIN, "hop_s": HOP_S,
                     "features": "scripts/g2_renderer_realism.py features (imported)"},
           "lockbox": {"persons_in_lockbox": len((set(train) | set(dev)) & set(locked)),
                       "target_lockbox_rows": int(train_counts.get("lockbox_rows", 0)) + int(dev_counts.get("lockbox_rows", 0)),
                       "records_lockbox_true": int(n_lockbox_records), "asserted": True,
                       "reader_flags": {"allow_eval": False, "allow_lockbox": False}},
           **res,
           "target_counts": {"train": train_counts, "dev": dev_counts},
           "moveatlas": {"head": stamp_head(stamp)},
           "per_pass_dev": per_pass.to_dict("records")}
    out["sets"] = {**res["sets"], "train": sorted(train), "train_sha256": sources.person_hash(train),
                   "dev14_sha256": sources.person_hash(res["sets"]["dev14"]),
                   "dev12_sha256": sources.person_hash(res["sets"]["dev12"])}
    out["lines"] = lines_of(res["lines"])
    if res.get("cv_train"):
        out["cv_train"] = lines_of({"train": res["cv_train"]})["train"]
    if out["lockbox"]["persons_in_lockbox"] or out["lockbox"]["target_lockbox_rows"] or n_lockbox_records:
        raise AssertionError(f"lockbox data reached the run: {out['lockbox']}")
    return clean(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_run = time.time()
    prio = below_normal()
    import torch
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible: run with CUDA_VISIBLE_DEVICES=-1")
    from mdfm import paths
    print(f"priority {prio}", flush=True)
    train, dev = g3.controls("train"), g3.controls("dev")
    if len(train) != 54 or len(dev) != 14 or set(train) & set(dev):
        raise SystemExit(f"folds: train {len(train)}, dev {len(dev)}, overlap {sorted(set(train) & set(dev))}")
    missing = sorted(set(EXCLUDED_FROM_12) - set(dev))
    if missing:
        raise SystemExit(f"excluded persons not among the dev controls: {missing}")
    locked = g3.lockbox_persons()
    assert_no_lockbox({"train": train, "dev": dev}, locked)
    wait_while_slow()
    rule_sha, tgs = checked_targets({"train": train, "dev": dev})       # the rule hash first, then the targets
    tg_train, tg_dev = tgs["train"], tgs["dev"]
    print(f"rule {rule_sha[:12]} asserted; targets: train {tg_train.counts}, dev {tg_dev.counts}", flush=True)
    n_lockbox_records = 0

    def records(persons, tg):
        nonlocal n_lockbox_records
        for tid, rec in g3.imu_records(persons, tg.trials):         # canonical.load, allow_lockbox False
            if str(rec.get("lockbox")).strip().lower() not in ("false", "0"):
                n_lockbox_records += 1
            yield tid, rec

    res = run(tg_train, tg_dev, records(train, tg_train), records(dev, tg_dev), train, dev,
              wait=lambda: wait_while_slow())
    stamp = _stamp()
    res = finalize(res, rule_sha, train, dev, locked, tg_train.counts, tg_dev.counts, n_lockbox_records, stamp,
                   run_started)
    res["timing_s"]["wall"] = round(time.time() - t_run, 1)
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    payload = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    common.write_text(os.path.join(OUT, "results.json"), payload)
    common.write_text(os.path.join(OUT, "summary.md"), text)
    prov = {"script": "scripts/g3_supervised_baseline.py", "label": LABEL, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp,
            "python": sys.version.split()[0], "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__, "torch": torch.__version__,
            "rule_sha256": rule_sha,
            "code_sha256": common.code_sha256([os.path.abspath(__file__), G2.__file__, g3.__file__]),
            "code_files": ["scripts/g3_supervised_baseline.py", "scripts/g2_renderer_realism.py", "mdfm/gates/g3.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "priority": prio}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    rule_t = max(os.path.getmtime(RULE_PATH), os.path.getmtime(RULE_SHA_PATH))
    for f in ("results.json", "summary.md", "_provenance.json"):
        if not os.path.getmtime(os.path.join(OUT, f)) > rule_t:
            raise AssertionError(f"{f} is not newer than the rule files")
    print(text)
    print(f"written {common.rel(OUT)}")


if __name__ == "__main__":
    main()
