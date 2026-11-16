"""G3 ZUPT on PD gait, with label-free event times and one training-fold gain (wave 17, task ZPD, 2026-10-16):
ranks 2 and 3 of docs/notes/STRATEGY_2026-10-15.md section 4. Every output is labelled LABEL ('dev diagnostic, not a
G3 read').

The ZUPT is task ZB's (scripts/g3_zupt_baseline.py, imported and never edited; SCALE's version 2): strapdown
integration of each foot sensor (foot_L, foot_R) between consecutive still runs, nothing fitted, no walkway value in
the length. The predictors differ only in the event times handed to the G3 matching and in one gain:
- zupt_ff (task ZB, unchanged): each stride timed at its foot-flat onsets (the starts of its two still runs).
- zupt_seg (label-free segmentation): the same strides timed at the foot-flat onsets minus one lag L, a constant
  fitted on the 54 training-fold controls only (fit_lag: the median of event time minus walkway IC over the walkway
  ICs with an event within LAG_TOL_S, the statistic g3.fit_offset gives the IC detectors). At inference the event
  times come from the foot signal alone: no walkway keying, no per-person or per-group fit.
- zupt_ff_gain, zupt_seg_gain: zupt_ff and zupt_seg with their strides and pass speeds divided by one gain per line
  (g_S, g_V), each fitted on the 54 training-fold controls only as F1 fitted the body's (scripts/g3_gain_check.py
  fit_gain: the median of predicted over walkway over the training fold's matched strides, and matched passes).
Pass speed (every predictor): task ZB's, the sum of the lengths over the sum of the event-time durations of the ZUPT
strides of both feet inside the walkway pass widened by g3.STRIDE_MATCH_S (Z.pass_speeds), then divided by g_V for a
gain predictor.

Development, training fold only (--train-only): the lag, the gains and the in-sample lines on the 54 training-fold
controls, written to train_fit.json, train_fit.md and train_strides.csv; no dev, PD, lockbox, EVAL, CODY or
care_pd record or target is read. Its values were seen before the prediction was written; the full run refits them
and asserts them equal to the registered constants FIT_LAG_S, FIT_GAINS (within FIT_TOL).

Full run (after the prediction is hashed; check_prediction asserts it before any target or record is read):
- the training fit (refitted, asserted);
- dev controls (g3.controls('dev'), 14; sets dev14 and dev12 of g3.trial_sets): the four ZUPT predictors, the
  population baseline (training-fold means, g3.population_baseline) and the walkway (harness check) through
  g3.evaluate; zupt_ff asserted equal to task ZB's results.json (reports/gates/g3/zupt_dev/, the same code);
- PD persons: the persons of scripts/eval_g3_pd_dev.py (imported: pd_persons, t19_folds, guard, pd_sets), the 74
  scored non-lockbox weargait_pd PD persons and the 14 scored of T19's proposed PD dev fold; rows pd_all, pd_t19dev,
  pd_all_valid, pd_t19dev_valid (g3.validity_row on every record, asserted equal to the valid flags of the body's PD
  folder); the population baseline recomputed and asserted equal to the body's PD folder; every PD person is out of
  sample for every fit here (the lag and the gains come from controls only);
- the body: v1's phase B end as the PD half read it (reports/gates/g3/dev_pd_phaseA_root_full_ckpt_B_032044/:
  g3_pd_dev.json for its lines, its F1 gain row and population_pd; events.csv, row base, for its matched strides
  and passes) and as the G3 diagnostic read it on the dev controls (Z.EVENTS); F1's gains (Z.f1_gains), asserted
  equal to those of the PD folder;
- same strides: per set and ZUPT predictor, on the walkway strides that the body and the predictor both match, the
  body raw, the body divided by F1's g_S and the predictor, with the paired person difference (body with F1 gain
  minus ZUPT; person bootstrap, g3.BOOT_N resamples, seed g3.SEED); the same on the passes for speed;
- event timing: per group (train54, dev14, pd_all, pd_t19dev) the signed difference foot-flat onset minus walkway IC
  over the walkway ICs with an onset within LAG_TOL_S (the statistic of fit_lag, reported, fitted on nothing here),
  per stride and per person;
- the reading of the prediction (reading: HOLDS, HOLDS ON ITS STRIDES, DEGRADES) and each prediction met or missed.

Never: the GPU (CUDA_VISIBLE_DEVICES=-1 before torch loads), a lockbox, EVAL, CODY or care_pd record. A lockbox
person anywhere in an input raises Refused (Z.check_persons, eval_g3_pd_dev.guard); every record is checked
(Z.check_record: lockbox False, no EVAL tier, a person of the requested set).
Writes reports/gates/g3/zupt_pd/: results.json, summary.md, strides.csv (one row per walkway stride: dev controls and
PD), passes.csv, zupt_strides.csv (every ZUPT stride formed), train_fit.json, train_fit.md, train_strides.csv and
_provenance.json (scripts/moveatlas_stamp.py).
Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\
    <conda>/envs/mdfm/python.exe scripts/g3_zupt_pd.py --train-only
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\
    <conda>/envs/mdfm/python.exe scripts/g3_zupt_pd.py
"""
import argparse
import importlib.util
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

import g3_zupt_baseline as Z  # noqa: E402

TASK = "ZPD"
OUT = os.path.join(TREE, "reports", "gates", "g3", "zupt_pd")
PRED_PATH = os.path.join(OUT, "prediction.txt")
PRED_SHA_PATH = os.path.join(OUT, "prediction.sha256")
PRED_SHA256 = "c0ab4d4b1d0bb2f7289f44ef8c2f4eff62626e87c230223883b85ae935cbb322"
LABEL = "dev diagnostic, not a G3 read"
PD_DIR = os.path.join(TREE, "reports", "gates", "g3", "dev_pd_phaseA_root_full_ckpt_B_032044")
PD_JSON = os.path.join(PD_DIR, "g3_pd_dev.json")
PD_EVENTS = os.path.join(PD_DIR, "events.csv")
PD_VALIDITY = os.path.join(PD_DIR, "validity.csv")
ZB_RESULTS = os.path.join(TREE, "reports", "gates", "g3", "zupt_dev", "results.json")
FF, SEG, FF_GAIN, SEG_GAIN = "zupt_ff", "zupt_seg", "zupt_ff_gain", "zupt_seg_gain"
ZUPTS = (FF, SEG, FF_GAIN, SEG_GAIN)
RAW_OF = {FF: FF, SEG: SEG, FF_GAIN: FF, SEG_GAIN: SEG}
LAG_TOL_S = g3.IC_MATCH_S              # 0.3 s, the window of g3.fit_offset
FIT_TOL = 1e-9
# the training fit, registered from the --train-only run (prediction.txt section 1)
FIT_LAG_S = 0.14000000000000057
FIT_GAINS = {FF: (0.991758915759732, 0.999580762543954), SEG: (0.9907983410753659, 0.99801792789004)}
PD_ROWS = ("pd_all", "pd_t19dev", "pd_all_valid", "pd_t19dev_valid")
CTRL_ROWS = ("dev12", "dev14")
BODY_ROW_OF = {"dev12": "dev_valid", "dev14": "dev14", **{r: r for r in PD_ROWS}}
READ_SETS = ("pd_all", "pd_t19dev")    # the rows the reading reads (prediction.txt section 3)
PASS_STRIDE_CM, PASS_SPEED_M_S = g3.LINES["stride_mae_cm"][0], g3.LINES["speed_mae_m_s"][0]
WALKWAY_OFFSETS = {("walkway", r): 0.0 for r in g3.REFERENCES}   # the walkway predictor's ICs are the reference


# ---------------------------------------------------------------- small helpers
def _sibling(name):
    """The module scripts/<name>.py, loaded once under a private name."""
    key = f"_zpd_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def PDM():
    """scripts/eval_g3_pd_dev.py (the PD persons, T19's fold, the guard and the PD sets)."""
    return _sibling("eval_g3_pd_dev")


def _q(v, qs=(0.1, 0.5, 0.9)):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return {f"p{int(round(100 * q)):02d}": (float(np.quantile(v, q)) if len(v) else float("nan")) for q in qs} | \
        {"n": int(len(v))}


def person_line(per_person):
    """{point (mean over persons), interval (common.cluster_bootstrap, g3.BOOT_N, g3.SEED; None under 2 persons),
    n_persons} of {person: value}."""
    ids = sorted(per_person)
    if not ids:
        return {"point": float("nan"), "interval": None, "n_persons": 0}
    v = np.array([per_person[i] for i in ids], np.float64)
    iv = list(common.cluster_bootstrap(v, np.array(ids), n=g3.BOOT_N, seed=g3.SEED)) if len(ids) >= 2 else None
    return {"point": float(v.mean()), "interval": iv, "n_persons": len(ids)}


def per_person_mean(values, persons):
    import pandas as pd
    s = pd.Series(np.asarray(values, np.float64)).groupby(np.asarray(persons).astype(str)).mean()
    return {str(k): float(x) for k, x in s.items()}


def brief(st):
    return {k: v for k, v in st.items() if k not in ("per_person", "trials", "sets")}


# ---------------------------------------------------------------- ZUPT strides of records
def read_zupt(records, persons, trials, validity=False, log=print):
    """(DataFrame of every ZUPT stride (person_id, trial_id, foot, t0_ff, t1_ff, length_m, t_a, t_b, duration_s,
    peak_speed_m_s, v_end_m_s), feet without a foot sensor, validity rows or None, records read). Each record is
    checked (Z.check_record) against `persons`; the strides are Z.zupt_strides on Z.foot_channels."""
    t0 = time.time()
    by, missing, rows, n = {}, [], [], 0
    info = trials.set_index("trial_id")
    for tid, rec in records:
        Z.check_record(tid, rec, persons)
        n += 1
        fc = Z.foot_channels(rec)
        by[tid] = {foot: Z.zupt_strides(a, g, float(rec.get("fps", Z.RATE))) for foot, (a, g) in fc.items()}
        missing += [(tid, f) for f in ("L", "R") if f not in fc]
        if validity:
            rows.append(g3.validity_row(tid, info.at[tid, "person_id"], info.at[tid, "task"], rec))
    zf = Z.zupt_frame(by).rename(columns={"t0": "t0_ff", "t1": "t1_ff"})
    pmap = dict(zip(trials["trial_id"], trials["person_id"]))
    zf.insert(0, "person_id", [pmap[t] for t in zf["trial_id"]])
    log(f"  records {n}, ZUPT strides {len(zf)}, {round(time.time() - t0, 1)} s", flush=True)
    return zf, missing, (g3.validity_frame(rows) if validity else None), n


def timed(zf, lag_s):
    """The ZUPT strides with event times t0, t1 = the foot-flat onsets minus lag_s (0: task ZB's times)."""
    out = zf.copy()
    out["t0"] = zf["t0_ff"].to_numpy(np.float64) - float(lag_s)
    out["t1"] = zf["t1_ff"].to_numpy(np.float64) - float(lag_s)
    return out


def event_ics(zt):
    """DataFrame(trial_id, foot, t_s): the distinct event times (t0 and t1) of the timed ZUPT strides, per trial and
    foot: the ZUPT's own initial-contact estimates."""
    import pandas as pd
    a = zt[["trial_id", "foot", "t0"]].rename(columns={"t0": "t_s"})
    b = zt[["trial_id", "foot", "t1"]].rename(columns={"t1": "t_s"})
    return (pd.concat([a, b], ignore_index=True).drop_duplicates()
            .sort_values(["trial_id", "foot", "t_s"], kind="stable").reset_index(drop=True))


def onset_timing(zf, walkway_ic, tol=LAG_TOL_S):
    """The walkway ICs (a Targets.walkway_ic frame) with the nearest foot-flat onset (event time of zupt_ff) within
    tol: g3.ic_differences rows (diff_s = onset minus walkway IC, matched). Fitted on nothing."""
    return g3.ic_differences(event_ics(timed(zf, 0.0)), walkway_ic, 0.0, tol)


def timing_summary(d):
    """Per-stride and per-person quantiles of the matched diff_s of an onset_timing table, with the matched share."""
    k = d[d["matched"]]
    per = k.groupby("person_id")["diff_s"].median() if len(k) else []
    return {"median_s": float(np.median(k["diff_s"])) if len(k) else float("nan"), "n_matched": int(len(k)),
            "n_reference": int(len(d)), "matched_share": float(d["matched"].mean()) if len(d) else float("nan"),
            "per_ic": _q(k["diff_s"]), "per_person_median": _q(per), "n_persons": int(k["person_id"].nunique())}


def fit_lag(zf, targets, train, dev, locked, tol=LAG_TOL_S):
    """The lag L of zupt_seg (module docstring), fitted on training-fold persons only: Refused for a lockbox person,
    AssertionError for a dev person or a person outside `train`. Returns timing_summary with lag_s."""
    persons = {str(p) for p in targets.walkway_ic["person_id"]} | {str(p) for p in zf["person_id"]}
    Z.check_persons(persons, None, locked, "lag fit")
    on_dev = sorted(persons & {str(p) for p in dev})
    if on_dev:
        raise AssertionError(f"the lag is never fitted on dev persons: {len(on_dev)} in the fit")
    Z.check_persons(persons, train, locked, "lag fit")
    s = timing_summary(onset_timing(zf, targets.walkway_ic, tol))
    if not math.isfinite(s["median_s"]):
        raise ValueError("no foot-flat onset within the window of a walkway IC: no lag")
    return {"lag_s": s["median_s"], **s}


def fit_gains(tables, train, dev, locked):
    """(g_S, g_V) of one raw ZUPT predictor: the medians of predicted over walkway over its matched strides and
    passes of the training fold (scripts/g3_gain_check.py fit_gain), with counts; the same person checks as fit_lag."""
    s = tables["strides"]
    s = s[s["matched"].astype(bool)]
    p = tables["speeds"]
    p = p[p["matched"].astype(bool)]
    persons = {str(x) for x in s["person_id"]} | {str(x) for x in p["person_id"]}
    Z.check_persons(persons, None, locked, "gain fit")
    on_dev = sorted(persons & {str(x) for x in dev})
    if on_dev:
        raise AssertionError(f"the gain is never fitted on dev persons: {len(on_dev)} in the fit")
    Z.check_persons(persons, train, locked, "gain fit")
    rs = s["pred_m"].to_numpy(np.float64) / s["length_m"].to_numpy(np.float64)
    rv = p["pred_m_s"].to_numpy(np.float64) / p["speed_m_s"].to_numpy(np.float64)
    rs, rv = rs[np.isfinite(rs) & (rs > 0)], rv[np.isfinite(rv) & (rv > 0)]
    if not len(rs) or not len(rv):
        raise ValueError("no matched stride or pass to fit a gain on")
    return {"g_S": float(np.median(rs)), "g_V": float(np.median(rv)), "n_strides": int(len(rs)),
            "n_passes": int(len(rv)), "n_persons": int(len(persons))}


# ---------------------------------------------------------------- predictors
def zupt_predictions(zt, targets, name, g_s=1.0, g_v=1.0):
    """g3.Predictions of timed ZUPT strides: strides at the event times (length / g_s), pass speeds of Z.pass_speeds
    (/ g_v); no IC."""
    sp = Z.pass_speeds(zt, targets.passes)
    st = zt[["trial_id", "foot", "t0", "t1", "length_m"]].copy()
    st["length_m"] = st["length_m"].to_numpy(np.float64) / float(g_s)
    sp = sp[["trial_id", "pass_id", "speed_m_s"]].copy()
    sp["speed_m_s"] = sp["speed_m_s"].to_numpy(np.float64) / float(g_v)
    return g3.Predictions(name, None, st, sp)


def all_predictions(zf, targets, lag_s, gains, base=None):
    """[walkway, population (when base), zupt_ff, zupt_seg, and, when gains ({FF: (g_S, g_V), SEG: ...}), the two
    gain predictors]."""
    out = [g3.walkway_predictions(targets)]
    if base is not None:
        out.append(g3.population_predictions(base, targets))
    zt = {FF: timed(zf, 0.0), SEG: timed(zf, lag_s)}
    out += [zupt_predictions(zt[FF], targets, FF), zupt_predictions(zt[SEG], targets, SEG)]
    if gains:
        out += [zupt_predictions(zt[FF], targets, FF_GAIN, *gains[FF]),
                zupt_predictions(zt[SEG], targets, SEG_GAIN, *gains[SEG])]
    return out


def train_fit(tg_train, zf, train, dev, locked):
    """The training fit of the module docstring on the training fold's targets and ZUPT strides: (fit dict, summary
    and tables of the in-sample evaluation of the four predictors on the set train54)."""
    lag = fit_lag(zf, tg_train, train, dev, locked)
    sets = {"train54": frozenset(tg_train.trials["trial_id"])}
    preds = all_predictions(zf, tg_train, lag["lag_s"], None)
    summ, tables = g3.evaluate(tg_train, preds, WALKWAY_OFFSETS, sets=sets)
    gains = {v: fit_gains(tables[v], train, dev, locked) for v in (FF, SEG)}
    g = {v: (gains[v]["g_S"], gains[v]["g_V"]) for v in (FF, SEG)}
    summ2, tables2 = g3.evaluate(tg_train, all_predictions(zf, tg_train, lag["lag_s"], g), WALKWAY_OFFSETS, sets=sets)
    return {"lag": lag, "gains": gains}, summ2, tables2


def assert_fit(fit, lag_s=None, gains=None, tol=FIT_TOL):
    """AssertionError unless the refitted lag and gains equal the registered ones (FIT_LAG_S, FIT_GAINS) within tol."""
    lag_s = FIT_LAG_S if lag_s is None else lag_s
    gains = FIT_GAINS if gains is None else gains
    if lag_s is None or gains is None:
        raise AssertionError("the training fit is not registered (FIT_LAG_S, FIT_GAINS)")
    got = [fit["lag"]["lag_s"]] + [fit["gains"][v][k] for v in (FF, SEG) for k in ("g_S", "g_V")]
    want = [lag_s] + [gains[v][i] for v in (FF, SEG) for i in (0, 1)]
    bad = [(a, b) for a, b in zip(got, want) if not abs(float(a) - float(b)) <= tol]
    if bad:
        raise AssertionError(f"the training fit differs from the registered one: {bad}")
    return True


# ---------------------------------------------------------------- per-stride and per-pass tables
def stride_table(targets, tables, group):
    """One row per walkway stride: the stride, and per ZUPT predictor its matched flag, prediction (m) and end
    difference, plus the event times of the nearest stride of each raw predictor."""
    st = targets.strides[["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1", "length_m"]].copy()
    st.insert(0, "group", group)
    for v in ZUPTS:
        m = tables[v]["strides"]
        if not m[["trial_id", "foot", "t0", "t1"]].reset_index(drop=True).equals(
                targets.strides[["trial_id", "foot", "t0", "t1"]].reset_index(drop=True)):
            raise AssertionError(f"{v}: the match table lists other walkway strides")
        st[f"{v}_matched"] = m["matched"].to_numpy(bool)
        st[f"{v}_m"] = np.where(m["matched"].to_numpy(bool), m["pred_m"].to_numpy(np.float64), np.nan)
        if v in (FF, SEG):
            st[f"{v}_end_diff_s"] = m["end_diff_s"].to_numpy(np.float64)
    for v in (FF_GAIN, SEG_GAIN):
        if not (st[f"{v}_matched"] == st[f"{RAW_OF[v]}_matched"]).all():
            raise AssertionError(f"{v}: the gain changed which strides match")
    return st


def pass_table(targets, tables, group):
    ps = targets.passes[["trial_id", "person_id", "task", "pass_id", "t0", "t1", "speed_m_s"]].copy()
    ps.insert(0, "group", group)
    for v in ZUPTS:
        m = tables[v]["speeds"]
        ps[f"{v}_matched"] = m["matched"].to_numpy(bool)
        ps[f"{v}_m_s"] = np.where(m["matched"].to_numpy(bool), m["pred_m_s"].to_numpy(np.float64), np.nan)
    return ps


def evaluate_group(tg, zf, lag_s, gains, base, sets, group):
    """(g3.evaluate summary with sets, stride_table, pass_table) of the walkway, the population (when base) and the
    four ZUPT predictors on one group's targets and ZUPT strides."""
    summ, tabs = g3.evaluate(tg, all_predictions(zf, tg, lag_s, gains, base), WALKWAY_OFFSETS, sets=sets)
    return summ, stride_table(tg, tabs, group), pass_table(tg, tabs, group)


def per_person(st, trials, v):
    """Descriptive: per person of the walkway strides of `trials`, the strides, the share predictor v matches and its
    mean absolute error (cm) there; sorted by that error, highest first (NaN last)."""
    t = st[st["trial_id"].isin(set(trials))]
    rows = []
    for p, g in t.groupby("person_id"):
        m = g[f"{v}_matched"].to_numpy(bool)
        e = np.abs(100.0 * (g[f"{v}_m"].to_numpy(np.float64) - g["length_m"].to_numpy(np.float64)))[m]
        rows.append({"person_id": str(p), "strides": int(len(g)), "matched_share": float(m.mean()),
                     "mae_cm": float(e.mean()) if len(e) else float("nan")})
    return sorted(rows, key=lambda r: (-r["mae_cm"] if math.isfinite(r["mae_cm"]) else math.inf, r["person_id"]))


def join_body(st, ps, body):
    """st and ps with body_cm, body_matched and body_m_s, body_pass_matched from the body's matched rows (Z.read_body_events
    output), joined on (trial, foot, end times rounded to Z.KEY_ROUND) and (trial, pass); every body row must find its
    walkway stride or pass with the same reference value."""
    bst, bsp = body
    st = st.copy()
    st["k0"], st["k1"] = st["t0"].round(Z.KEY_ROUND), st["t1"].round(Z.KEY_ROUND)
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
    st = st.drop(columns=["k0", "k1"])
    ps = ps.copy()
    bmap = {(r.trial_id, str(r.pass_id)): (r.x, r.y) for r in bsp.itertuples()}
    bx = [bmap.get((t, str(p))) for t, p in zip(ps["trial_id"], ps["pass_id"])]
    if sum(b is not None for b in bx) != len(bsp):
        raise AssertionError("a matched body pass found no walkway pass")
    if not all(b is None or abs(b[0] - v) < 1e-6 for b, v in zip(bx, ps["speed_m_s"])):
        raise AssertionError("the body join pairs different walkway passes")
    ps["body_m_s"] = [np.nan if b is None else b[1] for b in bx]
    ps["body_matched"] = [b is not None for b in bx]
    return st, ps


def same_lines(st, ps, trials, g_s, g_v):
    """Per ZUPT predictor, on the walkway strides of `trials` that the body and the predictor both match: lines (cm)
    of the body raw, the body divided by g_s and the predictor (g3._line through Z.line_of), and the paired person
    difference body with gain minus predictor (person_line of the per-person mean differences); the same on the
    passes (m/s, g_v)."""
    t = st[st["trial_id"].isin(set(trials))]
    p = ps[ps["trial_id"].isin(set(trials))]
    out = {"walkway_strides": int(len(t)), "body_matched": int(t["body_matched"].sum()),
           "walkway_passes": int(len(p)), "body_passes_matched": int(p["body_matched"].sum()), "by": {}}
    x = 100.0 * t["length_m"].to_numpy(np.float64)
    yb = t["body_cm"].to_numpy(np.float64)
    pid = t["person_id"].to_numpy().astype(str)
    xv = p["speed_m_s"].to_numpy(np.float64)
    ybv = p["body_m_s"].to_numpy(np.float64)
    pidv = p["person_id"].to_numpy().astype(str)
    for v in ZUPTS:
        both = (t["body_matched"] & t[f"{v}_matched"]).to_numpy(bool)
        yz = 100.0 * t[f"{v}_m"].to_numpy(np.float64)
        eb, eg, ez = np.abs(yb - x), np.abs(yb / g_s - x), np.abs(yz - x)
        d = {"both": int(both.sum()), "persons": int(len(set(pid[both]))),
             "stride_cm": {"body raw": brief(Z.line_of(eb[both], pid[both])),
                           "body with F1 gain": brief(Z.line_of(eg[both], pid[both])),
                           v: brief(Z.line_of(ez[both], pid[both]))}}
        bg, bz = per_person_mean(eg[both], pid[both]), per_person_mean(ez[both], pid[both])
        pr = person_line({k: bg[k] - bz[k] for k in bg})
        pr["lower_above_0"] = bool(pr["interval"] is not None and pr["interval"][0] > 0)
        d["stride_paired_body_gain_minus_zupt_cm"] = pr
        bothv = (p["body_matched"] & p[f"{v}_matched"]).to_numpy(bool)
        yzv = p[f"{v}_m_s"].to_numpy(np.float64)
        ebv, egv, ezv = np.abs(ybv - xv), np.abs(ybv / g_v - xv), np.abs(yzv - xv)
        d["passes_both"] = int(bothv.sum())
        d["speed_m_s"] = {"body raw": brief(Z.line_of(ebv[bothv], pidv[bothv])),
                          "body with F1 gain": brief(Z.line_of(egv[bothv], pidv[bothv])),
                          v: brief(Z.line_of(ezv[bothv], pidv[bothv]))}
        bgv, bzv = per_person_mean(egv[bothv], pidv[bothv]), per_person_mean(ezv[bothv], pidv[bothv])
        prv = person_line({k: bgv[k] - bzv[k] for k in bgv})
        prv["lower_above_0"] = bool(prv["interval"] is not None and prv["interval"][0] > 0)
        d["speed_paired_body_gain_minus_zupt_m_s"] = prv
        out["by"][v] = d
    return out


def set_lines(summ, names, sets):
    """{set: {predictor: {line: brief stats}}} of a g3.evaluate summary with sets."""
    out = {}
    for s in sets:
        out[s] = {}
        for n in names:
            if n not in summ:
                continue
            out[s][n] = {ln: brief(st["sets"][s]) for ln, st in summ[n].items() if s in (st.get("sets") or {})}
    return out


# ---------------------------------------------------------------- reading (prediction.txt section 3)
def reading(lines, same, rows=READ_SETS, v=SEG):
    """The registered reading of the PD result on predictor v over the rows `rows` (module docstring and
    prediction.txt section 3): per row, the conditions (a) stride MAE at most the G3 pass line (5 cm), (b) speed
    MAE at most 0.08 m/s, (c) stride and pass matched shares at least g3.MATCHED_SHARE_FLOOR, (d) on the strides it
    shares with the body, the paired body with F1 gain minus ZUPT interval above 0, (e) the same on the passes.
    Per row: HOLDS when (a) to (e) hold; HOLDS ON ITS STRIDES when all but (c) hold; DEGRADES otherwise. Overall: the
    reading of pd_all, with pd_t19dev beside it."""
    out = {}
    for r in rows:
        sl = lines[r][v]
        sm, sp = sl["stride_mae_cm"], sl["speed_mae_m_s"]
        sb = same[r]["by"][v]
        c = {"a_stride_within_pass": bool(Z.fin(sm["point"]) is not None and sm["point"] <= PASS_STRIDE_CM),
             "b_speed_within_pass": bool(Z.fin(sp["point"]) is not None and sp["point"] <= PASS_SPEED_M_S),
             "c_shares_at_floor": bool(g3.matched_share_floor_ok(sm["matched_share"])
                                       and g3.matched_share_floor_ok(sp["matched_share"])),
             "d_stride_beats_body_with_gain": bool(sb["stride_paired_body_gain_minus_zupt_cm"]["lower_above_0"]),
             "e_speed_beats_body_with_gain": bool(sb["speed_paired_body_gain_minus_zupt_m_s"]["lower_above_0"])}
        core = c["a_stride_within_pass"] and c["b_speed_within_pass"] and c["d_stride_beats_body_with_gain"] \
            and c["e_speed_beats_body_with_gain"]
        out[r] = {"conditions": c, "reading": ("HOLDS" if core and c["c_shares_at_floor"] else
                                               "HOLDS ON ITS STRIDES" if core else "DEGRADES")}
    out["overall"] = out[rows[0]]["reading"]
    out["predictor"] = v
    return out


# ---------------------------------------------------------------- reports
def fmt(x, nd=2):
    return Z.fmt(x, nd)


def fmt_iv(iv, nd=2):
    return Z.fmt_iv(iv, nd)


def train_md(fit, tl):
    """train_fit.md: the training fit and the in-sample lines (descriptive)."""
    lag, g = fit["lag"], fit["gains"]
    L = [f"# ZPD training fit on the 54 training-fold controls ({LABEL})", "",
         "Written by scripts/g3_zupt_pd.py --train-only. Training-fold persons only: no dev, PD, lockbox, EVAL, CODY or "
         "care_pd record or target was read. The lines below are in-sample (the lag and the gains were fitted on these "
         "strides) and decide nothing.", "",
         "## Lag of zupt_seg", "",
         f"Foot-flat onset minus walkway IC over the walkway ICs with an onset within {LAG_TOL_S} s: median "
         f"{fmt(lag['lag_s'], 4)} s on {lag['n_matched']} of {lag['n_reference']} ICs (share "
         f"{fmt(lag['matched_share'], 3)}), {lag['n_persons']} persons; per IC p10 {fmt(lag['per_ic']['p10'], 3)}, p50 "
         f"{fmt(lag['per_ic']['p50'], 3)}, p90 {fmt(lag['per_ic']['p90'], 3)} s; per-person medians p10 "
         f"{fmt(lag['per_person_median']['p10'], 3)}, p50 {fmt(lag['per_person_median']['p50'], 3)}, p90 "
         f"{fmt(lag['per_person_median']['p90'], 3)} s.", "",
         "## Gains (median of predicted over walkway, matched strides and passes)", "",
         "| predictor | g_S | g_V | strides | passes | persons |", "|---|---|---|---|---|---|"]
    for v in (FF, SEG):
        L.append(f"| {v} | {fmt(g[v]['g_S'], 4)} | {fmt(g[v]['g_V'], 4)} | {g[v]['n_strides']} | {g[v]['n_passes']} | "
                 f"{g[v]['n_persons']} |")
    L += ["", "## In-sample lines, train54 (descriptive)", "",
          "| predictor | stride MAE (cm) | 95 % | MAPE (%) | strides matched | speed MAE (m/s) | passes matched |",
          "|---|---|---|---|---|---|---|"]
    for v in ZUPTS:
        s = tl[v]
        L.append(f"| {v} | {fmt(s['stride_mae_cm']['point'])} | {fmt_iv(s['stride_mae_cm']['interval'])} | "
                 f"{fmt(s['stride_mape_pct']['point'], 1)} | {fmt(s['stride_mae_cm']['matched_share'], 3)} of "
                 f"{s['stride_mae_cm']['n_events']} | {fmt(s['speed_mae_m_s']['point'], 3)} | "
                 f"{fmt(s['speed_mae_m_s']['matched_share'], 3)} of {s['speed_mae_m_s']['n_events']} |")
    return "\n".join(L) + "\n"


def summary_md(res):
    """summary.md of a results dict (as written to results.json)."""
    L = [f"# G3 ZUPT on PD gait, label-free event times, one training-fold gain ({LABEL})", "",
         "Written by scripts/g3_zupt_pd.py (task ZPD) under reports/gates/g3/zupt_pd/prediction.txt (SHA-256 "
         f"{res['prediction']['sha256']}, asserted before any target or record was read). Every number below is read "
         "from results.json of this folder. ZUPT: task ZB's (scripts/g3_zupt_baseline.py, SCALE's version 2), matched by "
         "the G3 harness (both stride ends within 0.15 s); lines, intervals (2000 resamples, seed 20260929) and matched "
         "shares are the harness's. The lag and the gains are fitted on the 54 training-fold controls only, so every "
         "dev control and every PD person is out of sample. The body is v1's phase B end (raw, and divided by F1's "
         "training-fold gains). No lockbox, EVAL, CODY or care_pd record (asserted). "
         f"MoveAtlas {res['moveatlas'].get('head') or 'n/a'}.", "",
         "Predictors: zupt_ff (task ZB: foot-flat onsets), zupt_seg (onsets minus the training-fold lag "
         f"{fmt(res['fit']['lag']['lag_s'], 4)} s), zupt_ff_gain and zupt_seg_gain (divided by their training-fold "
         "gains); population (training-fold control means); body rows from "
         "reports/gates/g3/dev_pd_phaseA_root_full_ckpt_B_032044/g3_pd_dev.json (model, its F1 gain row, population_pd).",
         "", "## Lines per row", "",
         "| row | predictor | stride MAE (cm) | 95 % | MAPE (%) | strides matched | speed MAE (m/s) | 95 % | passes "
         "matched | persons |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in CTRL_ROWS + PD_ROWS:
        for n in ZUPTS + ("population",):
            s = res["lines"][r].get(n)
            if not s:
                continue
            sm, mp, sp = s.get("stride_mae_cm", {}), s.get("stride_mape_pct", {}), s.get("speed_mae_m_s", {})
            L.append(f"| {r} | {n} | {fmt(sm.get('point'))} | {fmt_iv(sm.get('interval'))} | {fmt(mp.get('point'), 1)} "
                     f"| {fmt(sm.get('matched_share'), 3)} of {sm.get('n_events')} | {fmt(sp.get('point'), 3)} | "
                     f"{fmt_iv(sp.get('interval'), 3)} | {fmt(sp.get('matched_share'), 3)} of {sp.get('n_events')} | "
                     f"{sm.get('n_persons')} |")
        b = res["body_rows"].get(r)
        if b:
            for n, s in b["lines"].items():
                sm, mp, sp = s.get("stride_mae_cm", {}), s.get("stride_mape_pct", {}), s.get("speed_mae_m_s", {})
                L.append(f"| {r} | {n} | {fmt(sm.get('point'))} | {fmt_iv(sm.get('interval'))} | "
                         f"{fmt(mp.get('point'), 1)} | {fmt(sm.get('matched_share'), 3) if sm.get('matched_share') is not None else ''} | "
                         f"{fmt(sp.get('point'), 3)} | {fmt_iv(sp.get('interval'), 3)} | "
                         f"{fmt(sp.get('matched_share'), 3) if sp.get('matched_share') is not None else ''} | "
                         f"{sm.get('n_persons')} |")
    L += ["", f"Body rows: the folder's row {BODY_ROW_OF['dev12']} stands beside dev12 (the same trials, asserted) and "
          "dev14 beside dev14; 'body with F1 gain' is the folder's gain diagnostic (strides and passes divided by F1's "
          "g_S and g_V, fitted on the 54 training-fold controls); population_pd is the mean of T19's proposed PD train "
          "fold (in sample for its persons in pd_all, out of sample on pd_t19dev).", "",
          "G3 lines for context (docs/rules/G3_RULE.md, DRAFT; the PD half has no rule): stride pass 5 cm, kill above "
          "10; speed pass 0.08 m/s, kill above 0.15; a floored line passes only with a matched share of at least 0.9.",
          "", "## Same strides as the body (and same passes)", "",
          "Per row and ZUPT predictor, on the walkway strides the body and the predictor both match: the body raw, the "
          "body with F1's gain and the predictor; paired: body with F1 gain minus the predictor (per-person means, "
          "person bootstrap).", "",
          "| row | predictor | strides both | body raw (cm) | body with F1 gain (cm) | ZUPT (cm) | paired (cm), 95 % | "
          "passes both | body raw (m/s) | body with F1 gain (m/s) | ZUPT (m/s) | paired (m/s), 95 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in CTRL_ROWS + PD_ROWS:
        for v in ZUPTS:
            d = res["same"][r]["by"][v]
            s, p = d["stride_cm"], d["speed_m_s"]
            ps, pv = d["stride_paired_body_gain_minus_zupt_cm"], d["speed_paired_body_gain_minus_zupt_m_s"]
            L.append(f"| {r} | {v} | {d['both']} | {fmt(s['body raw']['point'])} | "
                     f"{fmt(s['body with F1 gain']['point'])} | {fmt(s[v]['point'])} | {fmt(ps['point'])} "
                     f"{fmt_iv(ps['interval'])} | {d['passes_both']} | {fmt(p['body raw']['point'], 3)} | "
                     f"{fmt(p['body with F1 gain']['point'], 3)} | {fmt(p[v]['point'], 3)} | {fmt(pv['point'], 3)} "
                     f"{fmt_iv(pv['interval'], 3)} |")
    L += ["", "## Foot-flat onset against the walkway IC, per group (fitted on nothing here)", "",
          f"Onset minus walkway IC over the walkway ICs with an onset within {LAG_TOL_S} s (the statistic of the lag fit; "
          "the lag itself comes from train54 only).", "",
          "| group | median (s) | per IC p10, p50, p90 | per-person median p10, p50, p90 | ICs with an onset | persons |",
          "|---|---|---|---|---|---|"]
    for gname, t in res["onset_timing"].items():
        L.append(f"| {gname} | {fmt(t['median_s'], 3)} | "
                 + ", ".join(fmt(t["per_ic"][k], 3) for k in ("p10", "p50", "p90")) + " | "
                 + ", ".join(fmt(t["per_person_median"][k], 3) for k in ("p10", "p50", "p90"))
                 + f" | {t['n_matched']} of {t['n_reference']} ({fmt(t['matched_share'], 3)}) | {t['n_persons']} |")
    rd = res["reading"]
    L += ["", f"## Reading (prediction.txt section 3; predictor {rd['predictor']})", "",
          "| row | stride within 5 cm | speed within 0.08 m/s | shares at the 0.9 floor | stride beats body with gain | "
          "speed beats body with gain | reading |", "|---|---|---|---|---|---|---|"]
    for r in READ_SETS:
        c = rd[r]["conditions"]
        L.append(f"| {r} | {c['a_stride_within_pass']} | {c['b_speed_within_pass']} | {c['c_shares_at_floor']} | "
                 f"{c['d_stride_beats_body_with_gain']} | {c['e_speed_beats_body_with_gain']} | {rd[r]['reading']} |")
    L += ["", f"Overall reading (pd_all): {rd['overall']}.", ""]
    pr = res.get("predictions_check")
    if pr:
        L += ["## Predictions (prediction.txt section 2)", "", "| item | value | interval | met |", "|---|---|---|---|"]
        for k, v in pr.items():
            if isinstance(v["value"], str):
                L.append(f"| {k} | {v['value']} | {v['lo']} | {v['met']} |")
            else:
                L.append(f"| {k} | {fmt(v['value'], 4)} | [{fmt(v['lo'], 4)}, {fmt(v['hi'], 4)}] | {v['met']} |")
        L.append("")
    pp = res.get("per_person_zupt_seg")
    if pp:
        L += ["## Per person, zupt_seg (descriptive, post hoc in kind: no prediction covers it)", "",
              "The ten persons with the highest mean absolute error of zupt_seg on its matched strides, per row.", "",
              "| row | person | strides | matched share | MAE (cm) |", "|---|---|---|---|---|"]
        for r, rows in pp.items():
            for x in rows[:10]:
                L.append(f"| {r} | {x['person_id']} | {x['strides']} | {fmt(x['matched_share'], 3)} | "
                         f"{fmt(x['mae_cm'])} |")
        L.append("")
    tf = res["fit"]
    L += ["## Training fit (54 training-fold controls)", "",
          f"Lag {fmt(tf['lag']['lag_s'], 4)} s ({tf['lag']['n_matched']} walkway ICs, {tf['lag']['n_persons']} persons). "
          + "; ".join(f"{v}: g_S {fmt(tf['gains'][v]['g_S'], 4)}, g_V {fmt(tf['gains'][v]['g_V'], 4)}" for v in (FF, SEG))
          + ". Registered in prediction.txt and asserted equal on this run (train_fit.md: the in-sample lines).", "",
          "## Counts and checks", ""]
    for k, v in res["checks"].items():
        L.append(f"- {k}: {v}")
    L += ["", f"Time: {res['timing_s']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def below_normal():
    return Z.below_normal()


def body_rows(doc):
    """{row: {lines: {body raw, body with F1 gain, population_pd}}} from the body's g3_pd_dev.json."""
    out = {}
    gd = (doc.get("gain_diagnostic") or {}).get("rows") or {}
    for r in CTRL_ROWS + PD_ROWS:
        br = BODY_ROW_OF[r]
        lines = doc["lines"].get(br) or {}
        d = {"body raw": {k: brief(lines["model"][k]) for k in ("stride_mae_cm", "stride_mape_pct", "speed_mae_m_s")}}
        if br in gd:
            d["body with F1 gain"] = {k: brief(gd[br][k]["recalibrated"]) for k in ("stride_mae_cm", "speed_mae_m_s")}
        if "population_pd" in lines:
            d["population_pd"] = {k: brief(lines["population_pd"][k])
                                  for k in ("stride_mae_cm", "stride_mape_pct", "speed_mae_m_s")}
        out[r] = {"source_row": br, "lines": d}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-only", action="store_true", help="the training fit only (54 training-fold controls)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    prio = below_normal()
    import pandas as pd
    import torch
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible: run with CUDA_VISIBLE_DEVICES=-1")
    from mdfm import paths
    pred_sha = None
    if not a.train_only:
        pred_sha = Z.check_prediction(PRED_PATH, PRED_SHA_PATH, PRED_SHA256)      # before any target or record
    dev, train = g3.controls("dev"), g3.controls("train")
    if len(dev) != 14 or len(train) != 54 or set(dev) & set(train):
        raise SystemExit(f"folds: dev {len(dev)}, train {len(train)}, overlap {sorted(set(dev) & set(train))}")
    locked = g3.lockbox_persons()
    Z.check_persons(dev, None, locked, "dev controls")
    Z.check_persons(train, None, locked, "training fold")

    # ---------------- training fold: the fit
    print("training fold (54 controls)", flush=True)
    tg_train = g3.load_targets(train)
    if tg_train.counts.get("lockbox_rows", 0):
        raise AssertionError("training fold: lockbox target rows kept")
    zf_train, miss_train, _, n_train = read_zupt(g3.imu_records(train, tg_train.trials), train, tg_train.trials)
    fit, summ_tr, tabs_tr = train_fit(tg_train, zf_train, train, dev, locked)
    tl = {v: {ln: brief(st["sets"]["train54"]) for ln, st in summ_tr[v].items()} for v in ZUPTS}
    st_tr = stride_table(tg_train, tabs_tr, "train54")
    zf_tr_out = zf_train.assign(group="train54")
    fit_doc = {"what": LABEL, "task": TASK, "fit": fit, "train54_lines": tl, "records": n_train,
               "feet_without_sensor": [list(x) for x in miss_train], "zupt_strides_formed": int(len(zf_train)),
               "run_started": run_started, "persons": len(train)}
    if a.train_only:
        stamp = Z._stamp()
        fit_doc["moveatlas"] = {"head": stamp.get("head") if isinstance(stamp, dict) else None}
        fit_doc["script_sha256"] = Z.sha256_file(os.path.abspath(__file__))
        common.write_text(os.path.join(OUT, "train_fit.json"),
                          json.dumps(Z.clean(fit_doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        common.write_text(os.path.join(OUT, "train_fit.md"), train_md(Z.clean(fit), Z.clean(tl)))
        common.write_csv(st_tr, os.path.join(OUT, "train_strides.csv"))
        print(train_md(Z.clean(fit), Z.clean(tl)))
        print(f"written {common.rel(OUT)} (train fit only), {round(time.time() - t_start, 1)} s")
        return
    assert_fit(fit)
    gains = {v: (fit["gains"][v]["g_S"], fit["gains"][v]["g_V"]) for v in (FF, SEG)}
    lag_s = fit["lag"]["lag_s"]
    base = g3.population_baseline(tg_train)

    # ---------------- dev controls
    print("dev controls (14)", flush=True)
    tg_dev = g3.load_targets(dev)
    if tg_dev.counts.get("lockbox_rows", 0):
        raise AssertionError("dev: lockbox target rows kept")
    zf_dev, miss_dev, _, n_dev = read_zupt(g3.imu_records(dev, tg_dev.trials), dev, tg_dev.trials)
    sets_dev = g3.trial_sets(tg_dev)
    summ_dev, st_dev, ps_dev = evaluate_group(tg_dev, zf_dev, lag_s, gains, base, sets_dev, "dev14")
    with open(ZB_RESULTS, encoding="utf-8") as f:
        zb = json.load(f)
    zb_check = {}
    for s in CTRL_ROWS:
        for ln in ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct"):
            want = zb["lines"]["zupt"][ln]["sets"][s]["point"]
            got = summ_dev[FF][ln]["sets"][s]["point"]
            zb_check[f"{s}|{ln}"] = {"zb": want, "here": got}
            if not abs(float(want) - float(got)) <= 1e-9:
                raise AssertionError(f"zupt_ff differs from task ZB on {s} {ln}: {got} against {want}")
    body_dev = Z.read_body_events(Z.EVENTS, dev, locked)
    g_s, g_v = Z.f1_gains()
    st_dev, ps_dev = join_body(st_dev, ps_dev, body_dev)

    # ---------------- PD persons
    print("PD persons", flush=True)
    PD = PDM()
    pdp = PD.pd_persons()
    t19dev, t19train, t19info = PD.t19_folds()
    scored = sorted(pdp.scored)
    PD.guard(scored)
    if set(scored) & (set(dev) | set(train)):
        raise SystemExit("a control among the PD persons")
    if set(scored) & set(locked):
        raise Refused("a lockbox person among the scored PD persons")
    tg_pd = g3.load_targets(scored)
    if tg_pd.counts.get("lockbox_rows", 0):
        raise AssertionError("PD: lockbox target rows kept")
    zf_pd, miss_pd, val_pd, n_pd = read_zupt(g3.imu_records(scored, tg_pd.trials), scored, tg_pd.trials,
                                             validity=True)
    sets_pd = PD.pd_sets(tg_pd.trials, t19dev, g3.valid_trials(val_pd))
    summ_pd, st_pd, ps_pd = evaluate_group(tg_pd, zf_pd, lag_s, gains, base, sets_pd, "pd")
    with open(PD_JSON, encoding="utf-8") as f:
        pd_doc = json.load(f)
    # checks against the body's PD folder: validity, population lines, F1 gains, persons
    vf = pd.read_csv(PD_VALIDITY, encoding="utf-8", dtype={"trial_id": str, "person_id": str, "group": str})
    Z.check_persons(set(vf["person_id"]), None, locked, os.path.basename(PD_VALIDITY))
    vf_valid = vf["valid"].astype(str).str.strip().str.lower() == "true"
    pd_valid_folder = frozenset(vf.loc[(vf["group"] == "PD") & vf_valid, "trial_id"])
    ctrl_valid_folder = frozenset(vf.loc[(vf["group"] == "control") & vf_valid, "trial_id"])
    if pd_valid_folder != g3.valid_trials(val_pd):
        raise AssertionError("the PD valid trials differ from the body's PD folder")
    if ctrl_valid_folder != sets_dev["dev12"]:
        raise AssertionError("the folder's dev_valid trials are not the dev12 trials")
    pop_check = {}
    for r in PD_ROWS:
        for ln in ("stride_mae_cm", "speed_mae_m_s"):
            want = pd_doc["lines"][r]["population"][ln]["point"]
            got = summ_pd["population"][ln]["sets"][r]["point"]
            pop_check[f"{r}|{ln}"] = {"folder": want, "here": got}
            if not abs(float(want) - float(got)) <= 1e-9:
                raise AssertionError(f"population line differs from the body's PD folder on {r} {ln}")
    fg = pd_doc["gain_diagnostic"]["gain"]
    if abs(fg["stride"] - g_s) > 1e-12 or abs(fg["speed"] - g_v) > 1e-12:
        raise AssertionError("F1's gains differ between g3_gain_check and the body's PD folder")
    if pd_doc["persons"]["scored_sha256"] != __import__("mdfm.data.sources", fromlist=["x"]).person_hash(scored):
        raise AssertionError("the scored PD persons differ from the body's PD folder")
    body_pd = Z.read_body_events(PD_EVENTS, scored, locked)
    st_pd, ps_pd = join_body(st_pd, ps_pd, body_pd)

    # ---------------- lines, same strides, timing, reading
    names = ZUPTS + ("population", "walkway")
    lines = set_lines(summ_dev, names, CTRL_ROWS) | set_lines(summ_pd, names, PD_ROWS)
    same = {s: same_lines(st_dev, ps_dev, sets_dev[s], g_s, g_v) for s in CTRL_ROWS}
    same |= {s: same_lines(st_pd, ps_pd, sets_pd[s], g_s, g_v) for s in PD_ROWS}
    t19set = set(t19dev)
    wic = tg_pd.walkway_ic
    onset = {"train54": timing_summary(onset_timing(zf_train, tg_train.walkway_ic)),
             "dev14": timing_summary(onset_timing(zf_dev, tg_dev.walkway_ic)),
             "pd_all": timing_summary(onset_timing(zf_pd, wic)),
             "pd_t19dev": timing_summary(onset_timing(zf_pd[zf_pd["person_id"].isin(t19set)],
                                                      wic[wic["person_id"].isin(t19set)]))}
    rd = reading(lines, same)
    res = {"what": LABEL, "task": TASK, "fit": fit, "lines": lines, "same": same, "onset_timing": onset,
           "reading": rd, "body_rows": body_rows(pd_doc), "f1_gains": {"g_S": g_s, "g_V": g_v, "end": Z.F1_END},
           "sets": {s: {"n_trials": len(t)} for s, t in (sets_dev | sets_pd).items()},
           "persons": {"dev": dev, "pd_scored": len(scored), "pd_t19dev_scored": len(t19set & set(scored)),
                       "t19": t19info, "pd_not_scored": pdp.not_scored},
           "checks": {"zupt_ff equals task ZB on dev12 and dev14 (stride, speed, MAPE)": True,
                      "PD valid trials equal the body folder's": True, "folder dev_valid equals dev12": True,
                      "population lines equal the body folder's (PD rows)": True,
                      "F1 gains equal the body folder's": True, "scored PD persons equal the body folder's": True,
                      "training fit equals the registered one": True,
                      "records read": {"train54": n_train, "dev": n_dev, "pd": n_pd},
                      "feet without a foot sensor": {"train54": len(miss_train), "dev": len(miss_dev),
                                                     "pd": len(miss_pd)},
                      "ZUPT strides formed": {"train54": int(len(zf_train)), "dev": int(len(zf_dev)),
                                              "pd": int(len(zf_pd))},
                      "lockbox persons in any input": 0},
           "consistency": {"zb": zb_check, "population": pop_check},
           "per_person_zupt_seg": {"dev12": per_person(st_dev, sets_dev["dev12"], SEG),
                                   "pd_all": per_person(st_pd, sets_pd["pd_all"], SEG)}}
    res["predictions_check"] = predictions_check(res)
    stamp = Z._stamp()
    res["prediction"] = {"path": common.rel(PRED_PATH), "sha256": pred_sha, "asserted_before_reads": True}
    res["run_started"] = run_started
    res["moveatlas"] = {"head": stamp.get("head") if isinstance(stamp, dict) else None}
    res["timing_s"] = round(time.time() - t_start, 1)
    out = Z.clean(res)
    text = summary_md(out)
    if any(c in text for c in ("\u2013", "\u2014")):
        raise ValueError("summary.md holds an en or em dash")
    common.write_text(os.path.join(OUT, "results.json"), json.dumps(out, indent=1, ensure_ascii=False,
                                                                     allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), text)
    common.write_text(os.path.join(OUT, "train_fit.json"),
                      json.dumps(Z.clean(fit_doc | {"moveatlas": res["moveatlas"]}), indent=1, ensure_ascii=False,
                                 allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "train_fit.md"), train_md(Z.clean(fit), Z.clean(tl)))
    common.write_csv(st_tr, os.path.join(OUT, "train_strides.csv"))
    common.write_csv(pd.concat([st_dev, st_pd], ignore_index=True), os.path.join(OUT, "strides.csv"))
    common.write_csv(pd.concat([ps_dev, ps_pd], ignore_index=True), os.path.join(OUT, "passes.csv"))
    common.write_csv(pd.concat([zf_tr_out, zf_dev.assign(group="dev14"), zf_pd.assign(group="pd")],
                               ignore_index=True), os.path.join(OUT, "zupt_strides.csv"))
    inputs = {common.rel(p): common.sha256_file(p) for p in (Z.EVENTS, Z.F1_RESULTS, ZB_RESULTS, PD_JSON, PD_EVENTS,
                                                              PD_VALIDITY, PDM().FOLDS_CSV, PDM().STAGE2_SUMMARY,
                                                              *g3.TARGET_FILES, g3.SUMMARY_JSON)}
    prov = {"script": "scripts/g3_zupt_pd.py", "task": TASK, "label": LABEL, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "prediction_sha256": pred_sha,
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__,
            "code_sha256": common.code_sha256([os.path.abspath(__file__), Z.__file__, PDM().__file__, g3.__file__,
                                               common.__file__]),
            "code_files": ["scripts/g3_zupt_pd.py", "scripts/g3_zupt_baseline.py", "scripts/eval_g3_pd_dev.py",
                           "mdfm/gates/g3.py", "mdfm/gates/common.py"],
            "script_sha256": Z.sha256_file(os.path.abspath(__file__)), "inputs_sha256": inputs,
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                           "PYTHONDONTWRITEBYTECODE")} | {"priority": prio}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(Z.clean(prov), indent=1) + "\n")
    for f in ("results.json", "summary.md", "_provenance.json"):
        if not os.path.getmtime(os.path.join(OUT, f)) > os.path.getmtime(PRED_PATH):
            raise AssertionError(f"{f} is not newer than prediction.txt")
    print(text)
    print(f"written {common.rel(OUT)}")


# ---------------------------------------------------------------- predictions (prediction.txt section 2)
PREDICTIONS = {
    "P1 dev12 zupt_seg stride matched share": (0.85, 0.97),
    "P1 dev14 zupt_seg stride matched share": (0.78, 0.95),
    "P1 dev12 zupt_seg stride MAE (cm)": (1.3, 2.5),
    "P1 dev12 zupt_seg speed MAE (m/s)": (0.010, 0.040),
    "P2 dev12 zupt_seg_gain stride MAE (cm)": (1.0, 2.4),
    "P3 pd_all onset minus walkway IC, median (s)": (0.08, 0.15),
    "P4 pd_all zupt_ff stride matched share": (0.40, 0.80),
    "P4 pd_all zupt_ff stride MAE (cm)": (1.5, 5.0),
    "P5 pd_all zupt_seg stride matched share": (0.80, 0.97),
    "P5 pd_all zupt_seg stride MAE (cm)": (1.5, 4.0),
    "P5 pd_all zupt_seg speed MAE (m/s)": (0.015, 0.060),
    "P5 pd_all zupt_seg passes matched share": (0.95, 1.0),
    "P5 pd_t19dev zupt_seg stride matched share": (0.75, 0.98),
    "P5 pd_t19dev zupt_seg stride MAE (cm)": (1.3, 4.5),
    "P6 pd_all zupt_seg_gain stride MAE (cm)": (1.2, 3.8),
    "P6 pd_all zupt_seg minus zupt_seg_gain stride MAE (cm)": (-0.2, 0.8),
    "P7 pd_all same strides, body with F1 gain (cm)": (5.0, 7.0),
    "P7 pd_all same strides, zupt_seg (cm)": (1.5, 4.0),
    "P7 pd_all paired body with F1 gain minus zupt_seg (cm)": (2.0, 5.0),
}
PREDICTED_READING = {"pd_all": "HOLDS", "pd_t19dev": "HOLDS"}


def prediction_values(res):
    """{item: value} of the items of PREDICTIONS, read from a results dict."""
    ln = res["lines"]

    def pt(row, pred, line, key="point"):
        return ln[row][pred][line][key]

    sm = res["same"]["pd_all"]["by"][SEG]
    return {
        "P1 dev12 zupt_seg stride matched share": pt("dev12", SEG, "stride_mae_cm", "matched_share"),
        "P1 dev14 zupt_seg stride matched share": pt("dev14", SEG, "stride_mae_cm", "matched_share"),
        "P1 dev12 zupt_seg stride MAE (cm)": pt("dev12", SEG, "stride_mae_cm"),
        "P1 dev12 zupt_seg speed MAE (m/s)": pt("dev12", SEG, "speed_mae_m_s"),
        "P2 dev12 zupt_seg_gain stride MAE (cm)": pt("dev12", SEG_GAIN, "stride_mae_cm"),
        "P3 pd_all onset minus walkway IC, median (s)": res["onset_timing"]["pd_all"]["median_s"],
        "P4 pd_all zupt_ff stride matched share": pt("pd_all", FF, "stride_mae_cm", "matched_share"),
        "P4 pd_all zupt_ff stride MAE (cm)": pt("pd_all", FF, "stride_mae_cm"),
        "P5 pd_all zupt_seg stride matched share": pt("pd_all", SEG, "stride_mae_cm", "matched_share"),
        "P5 pd_all zupt_seg stride MAE (cm)": pt("pd_all", SEG, "stride_mae_cm"),
        "P5 pd_all zupt_seg speed MAE (m/s)": pt("pd_all", SEG, "speed_mae_m_s"),
        "P5 pd_all zupt_seg passes matched share": pt("pd_all", SEG, "speed_mae_m_s", "matched_share"),
        "P5 pd_t19dev zupt_seg stride matched share": pt("pd_t19dev", SEG, "stride_mae_cm", "matched_share"),
        "P5 pd_t19dev zupt_seg stride MAE (cm)": pt("pd_t19dev", SEG, "stride_mae_cm"),
        "P6 pd_all zupt_seg_gain stride MAE (cm)": pt("pd_all", SEG_GAIN, "stride_mae_cm"),
        "P6 pd_all zupt_seg minus zupt_seg_gain stride MAE (cm)":
            pt("pd_all", SEG, "stride_mae_cm") - pt("pd_all", SEG_GAIN, "stride_mae_cm"),
        "P7 pd_all same strides, body with F1 gain (cm)": sm["stride_cm"]["body with F1 gain"]["point"],
        "P7 pd_all same strides, zupt_seg (cm)": sm["stride_cm"][SEG]["point"],
        "P7 pd_all paired body with F1 gain minus zupt_seg (cm)": sm["stride_paired_body_gain_minus_zupt_cm"]["point"],
    }


def predictions_check(res):
    """{item: {value, lo, hi, met}} of PREDICTIONS, and the reading items {predicted, read, met}."""
    vals = prediction_values(res)
    out = {}
    for k, (lo, hi) in PREDICTIONS.items():
        v = vals.get(k)
        ok = v is not None and math.isfinite(float(v)) and lo <= float(v) <= hi
        out[k] = {"value": v, "lo": lo, "hi": hi, "met": bool(ok)}
    for r, want in PREDICTED_READING.items():
        got = res["reading"][r]["reading"]
        out[f"P8 reading {r}"] = {"value": got, "lo": want, "hi": want, "met": got == want}
    return out


if __name__ == "__main__":
    main()
