"""Checks of the G12 readouts (task G12IC, 2026-10-13): defects that would flatter a readout's stride and speed lines,
and what the rule's IC probe reads in its tokens. Reads the outputs of scripts/g12_readout.py (reports/bench/g12/
readouts/<name>/: results.json, per_row.csv, per_pass.csv) and their feature files (cache/runs/g12_readouts/<name>/
features.npz), the G3 walkway targets of the 54 training-fold and the 14 dev weargait_pd controls (g3.load_targets,
dev reads only), the body's G3 dev reports that scripts/g12_compare.py reads, and the comparison it wrote
(reports/bench/g12/compare_g3_dev/g12.json). Dev only: no lockbox, EVAL, CODY or care_pd record, no sealed read, no
record or model pass (the checkpoint is opened for its provenance only, with --checkpoints). CPU, numpy.

Checks (each PASS or FAIL with its numbers; the exit code is 1 on any FAIL):
 1 sets: the fitting set is g3.controls('train') (54), dev14 g3.controls('dev') (14), disjoint, no lockbox person
   (g3.lockbox_persons, the id list), the body reports' dev set; dev12 is dev14 without g3.EXCLUDED_FROM_12; each
   readout's results.json states these sets.
 2 feature file: the persons of the fitting units and token rows are among the 54, those of the dev ones among the
   14, no lockbox person, two records (tasks) per person.
 3 checkpoint (--checkpoints): no dev or lockbox person among the checkpoint's training persons
   (train/checkpoint.load, its provenance).
 4 folds: probes.person_folds over the 54's passes puts every person in one fold, no dev person.
 5 targets: per variant, the stride, pass and IC rows of per_row.csv are the walkway strides (trial, foot, t0, t1,
   length_m), passes (trial, pass, speed_m_s) and walkway ICs (trial, foot, t_s) of g3.load_targets on the 14,
   exactly, as many as the body reports' n_events.
 6 units: stride value = 100 |prediction - target| (cm), target and prediction in metres (STRIDE_RANGE_M); speed value
   = |prediction - target| (m/s), both in SPEED_RANGE_M_S.
 7 one prediction per pass: per_pass.csv has one row per (trial, pass); every stride row of a pass carries the pass's
   stride prediction, every speed row the pass's speed prediction.
 8 refit from the fitting set alone: probes.fit_probe on the 54's pooled features and pass targets, applied to the
   14's features, gives per_pass.csv's predictions within REFIT_TOL (the feature file keeps the pooled features in
   float32; the readout fitted them in float64). No dev target enters (the fit takes none).
 9 lines: per variant, the stride and speed lines recomputed from per_row.csv (per person the mean, mean over
   persons; dev12 on the rows flagged in_dev12) equal results.json's; in_dev12 is exactly 'person in dev12'; a
   baseline's chosen points equal g12.json's (IC included).
10 body points: the body reports' per-person model rows give their g3_dev.json points on the 14 and g12.json's
   points on the 14 and the 12 (stride, speed).

Diagnostics (reported, no PASS or FAIL):
- span reference: a ridge on the pass span's duration alone (1 / duration, duration; the span is the targets' pass
  interval, through which every pass probe pools its token rows), fitted on the 54, scored by the G3 lines on the 14
  and the 12: what the pass timing alone gives (the token features do not receive the duration: they are the mean and
  standard deviation of the token rows inside the span).
- IC probe: per IC variant, the probe of readouts.ic_stage (probes.fit_probe on the labelled token rows of the 54)
  applied to the dev token rows; on the dev rows labelled by the same rule (probes.ic_targets with the dev pass spans),
  per foot: the AUC of the patch score (the sum of the foot's sub-patch scores, which probes.decode_ics thresholds at
  probes.IC_THRESHOLD) for 'the patch holds a walkway IC of this foot', the mean score of such patches and of the
  others, the hit and false-alarm shares at the threshold, the share of hit patches whose best sub-patch is the
  reference's (exact, within one); the IC line (g3.ic_line, offset fitted on the 14 by g3.fit_offsets) at the
  thresholds THRESHOLDS (sensitivity only: the rule's probe is J0's, threshold 0.5); and (unless --no-context) the
  same probe on the rows of patches p - 1, p, p + 1 concatenated, which tells whether contact timing sits in the
  neighbouring tokens rather than in the patch's own.

Writes, with --write, <out> (default reports/bench/g12/readout_checks): checks.json and checks.md. Prints the checks.

Usage (mdfm environment, CPU, from the tree root; CUDA_VISIBLE_DEVICES=-1):
  python scripts/check_g12_readouts.py [--readouts NAME ...] [--checkpoints] [--drift-log PATH] [--no-context] [--write]
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

import numpy as np  # noqa: E402

TASK = "G12IC"
LABEL = "G12 readout checks, dev only, not a G12 comparison"
READOUTS = os.path.join(TREE, "reports", "bench", "g12", "readouts")
FEATURES = os.path.join(TREE, "cache", "runs", "g12_readouts")
COMPARE = os.path.join(TREE, "reports", "bench", "g12", "compare_g3_dev", "g12.json")
OUT = os.path.join(TREE, "reports", "bench", "g12", "readout_checks")
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
REFIT_TOL = 1e-6
STRIDE_RANGE_M = (0.3, 2.5)
SPEED_RANGE_M_S = (0.2, 3.0)
PASS_LINES = ("stride_mae_cm", "speed_mae_m_s")
FOLD_SIZES = (54, 14)                 # scripts/g12_readout.py FOLD_SIZES: train, dev


def _compare_module():
    """scripts/g12_compare.py (beside this script) as a module: its BODY_REPORTS are the body reports read here."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g12_compare.py")
    spec = importlib.util.spec_from_file_location("g12_compare_for_checks", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(out, cid, readout, ok, detail):
    """Append one check {id, readout, ok, detail} to out and return ok."""
    out.append({"id": str(cid), "readout": readout, "ok": bool(ok), "detail": detail})
    return bool(ok)


def _float(x):
    if x is None:
        return float("nan")
    return float(x)


def _same(a, b, tol=1e-9):
    a, b = _float(a), _float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _bool(s):
    return s.astype(str).str.lower().isin(("true", "1"))


# ---------------------------------------------------------------- checks
def check_global_sets(out, train, dev, dev12, locked, body_sets, excluded, sizes=FOLD_SIZES):
    """Check 1 on the folds (module docstring). body_sets: {report: dev persons}; sizes: (fitting, dev) persons."""
    check(out, 1, "", len(train) == sizes[0] and len(dev) == sizes[1] and not set(train) & set(dev),
          f"folds: train {len(train)}, dev {len(dev)}, overlap {len(set(train) & set(dev))}")
    check(out, 1, "", not (set(train) | set(dev)) & set(locked),
          f"lockbox persons among the fitting and dev sets: {len((set(train) | set(dev)) & set(locked))}")
    check(out, 1, "", sorted(set(dev) - set(excluded)) == list(dev12) and len(dev12) == len(dev) - len(set(excluded)),
          f"dev12: {len(dev12)} persons, dev14 without {', '.join(sorted(excluded))}")
    for name, persons in body_sets.items():
        check(out, 1, "", sorted(persons) == list(dev), f"body report {name}: dev set of {len(persons)} = dev14")


def check_results_sets(out, name, res, train, dev, dev12):
    st = res.get("sets") or {}
    return check(out, 1, name, st.get("train") == list(train) and st.get("dev14") == list(dev)
                 and st.get("dev12") == list(dev12),
                 f"results.json sets: train {len(st.get('train') or [])}, dev14 {len(st.get('dev14') or [])}, "
                 f"dev12 {len(st.get('dev12') or [])}")


def check_features(out, name, data, train, dev, locked):
    """Check 2 (module docstring)."""
    for s, want in (("train", train), ("dev", dev)):
        pu = set(map(str, data["units"][s]["person_id"]))
        pt = set(np.asarray(data[s]["tokens"]["person_id"]).astype(str))
        rec = int((data[s].get("counts") or {}).get("records", -1))
        check(out, 2, name, pu <= set(want) and pt <= set(want) and not (pu | pt) & set(locked)
              and rec == 2 * len(pu),
              f"feature file {s}: unit persons {len(pu)}, token-row persons {len(pt)}, all among the {len(want)}, "
              f"lockbox {len((pu | pt) & set(locked))}, records {rec}")
    pt = set(np.asarray(data["train"]["tokens"]["person_id"]).astype(str)) | set(map(str, data["units"]["train"]
                                                                                    ["person_id"]))
    check(out, 2, name, not pt & set(dev), f"dev persons among the fitting units and token rows: {len(pt & set(dev))}")


def check_checkpoint(out, name, trained, train, dev, locked):
    """Check 3 (module docstring); trained: the checkpoint's training persons."""
    t = set(trained)
    check(out, 3, name, not t & set(dev) and not t & set(locked),
          f"checkpoint training persons {len(t)}: dev among them {len(t & set(dev))}, lockbox {len(t & set(locked))}, "
          f"fitting-set persons among them {len(t & set(train))} (windows seen without labels)")


def check_folds(out, name, u_tr, dev):
    """Check 4 (module docstring)."""
    from mdfm.eval import probes
    p = np.asarray(u_tr["person_id"]).astype(str)
    f = probes.person_folds(p)
    ids = np.unique(p)
    n_folds = np.array([len(np.unique(f[p == i])) for i in ids])
    sizes = np.bincount([int(f[np.flatnonzero(p == i)[0]]) for i in ids], minlength=probes.FOLDS).tolist()
    check(out, 4, name, n_folds.max() == 1 and not set(ids) & set(dev),
          f"person folds of the {len(ids)}: persons per fold {sizes}, every person in one fold, dev persons "
          f"{len(set(ids) & set(dev))}")


def _sorted_array(df, cols, by):
    return df[cols].sort_values(by, kind="stable").to_numpy()


def _equal_frames(a, b, n_str):
    if a.shape != b.shape:
        return False
    if not all((a[:, j].astype(str) == b[:, j].astype(str)).all() for j in range(n_str)):
        return False
    return bool(np.allclose(a[:, n_str:].astype(float), b[:, n_str:].astype(float), atol=1e-12, rtol=0.0))


def check_rows(out, name, rows, per_pass, tg_dev, body_n, variants):
    """Checks 5 (targets), 6 (units) and 7 (one prediction per pass) of every variant (module docstring). body_n:
    {report: {'stride': n, 'speed': n, 'ic': n}}."""
    S = tg_dev.strides.copy()
    S["pass_id"] = S["pass_id"].astype(str)
    P = tg_dev.passes.copy()
    P["pass_id"] = P["pass_id"].astype(str)
    W = tg_dev.walkway_ic
    pp = per_pass.copy()
    pp["pass_id"] = pp["pass_id"].astype(str)
    keys = list(zip(pp["trial_id"].astype(str), pp["pass_id"]))
    unique_pp = not pp.duplicated(["trial_id", "pass_id"]).any()
    for v in variants:
        sr = rows[(rows["line"] == "stride_mae_cm") & (rows["variant"] == v)].copy()
        sp = rows[(rows["line"] == "speed_mae_m_s") & (rows["variant"] == v)].copy()
        ic = rows[(rows["line"] == "ic_median_ms") & (rows["variant"] == v)].copy()
        sr["pass_id"] = sr["pass_id"].astype(str)
        sp["pass_id"] = sp["pass_id"].astype(str)
        ok_s = _equal_frames(_sorted_array(sr, ["trial_id", "foot", "t0", "t1", "target"], ["trial_id", "t0", "foot"]),
                             _sorted_array(S, ["trial_id", "foot", "t0", "t1", "length_m"], ["trial_id", "t0", "foot"]),
                             2)
        ok_p = _equal_frames(_sorted_array(sp, ["trial_id", "pass_id", "target"], ["trial_id", "pass_id"]),
                             _sorted_array(P, ["trial_id", "pass_id", "speed_m_s"], ["trial_id", "pass_id"]), 2)
        ok_i = True
        if len(ic):
            ok_i = _equal_frames(_sorted_array(ic, ["trial_id", "foot", "t_s"], ["trial_id", "foot", "t_s"]),
                                 _sorted_array(W, ["trial_id", "foot", "t_s"], ["trial_id", "foot", "t_s"]), 2)
        ok_n = all(len(sr) == n["stride"] and len(sp) == n["speed"] and (not len(ic) or len(ic) == n["ic"])
                   for n in body_n.values())
        check(out, 5, name, ok_s and ok_p and ok_i and ok_n,
              f"{v}: stride rows {len(sr)}, pass rows {len(sp)}, IC rows {len(ic)} are the targets of g3.load_targets "
              f"(strides {ok_s}, passes {ok_p}, ICs {ok_i}); body reports' n_events "
              + "; ".join(f"{k} {n['stride']}/{n['speed']}/{n['ic']}" for k, n in body_n.items()))
        lo, hi = STRIDE_RANGE_M
        a, b = SPEED_RANGE_M_S
        ok6 = (bool(np.allclose(sr["value"], 100.0 * np.abs(sr["prediction"] - sr["target"])))
               and bool(sr["target"].between(lo, hi).all()) and bool(sr["prediction"].between(lo, hi).all())
               and bool(np.allclose(sp["value"], np.abs(sp["prediction"] - sp["target"])))
               and bool(sp["target"].between(a, b).all()) and bool(sp["prediction"].between(a, b).all()))
        check(out, 6, name, ok6,
              f"{v}: stride value = 100 |prediction - target| (cm), target {sr['target'].min():.2f} to "
              f"{sr['target'].max():.2f} m, prediction {sr['prediction'].min():.2f} to {sr['prediction'].max():.2f} m; "
              f"speed value = |prediction - target| (m/s), target {sp['target'].min():.2f} to {sp['target'].max():.2f}, "
              f"prediction {sp['prediction'].min():.2f} to {sp['prediction'].max():.2f}")
        ks = dict(zip(keys, pp[f"pred_stride_m_{v}"].to_numpy(np.float64)))
        kv = dict(zip(keys, pp[f"pred_speed_m_s_{v}"].to_numpy(np.float64)))
        distinct = sr.groupby(["trial_id", "pass_id"])["prediction"].nunique()
        ps = np.array([ks.get(k, np.nan) for k in zip(sr["trial_id"].astype(str), sr["pass_id"])])
        pv = np.array([kv.get(k, np.nan) for k in zip(sp["trial_id"].astype(str), sp["pass_id"])])
        ok7 = (unique_pp and (distinct.max() if len(distinct) else 1) == 1
               and bool(np.allclose(sr["prediction"], ps)) and not sp.duplicated(["trial_id", "pass_id"]).any()
               and bool(np.allclose(sp["prediction"], pv)))
        check(out, 7, name, ok7,
              f"{v}: per_pass rows {len(pp)} (unique {unique_pp}); distinct stride predictions per pass at most "
              f"{int(distinct.max()) if len(distinct) else 0}; stride and speed rows carry per_pass's predictions")


def check_refit(out, name, data, u_tr, per_pass, variants, tol=REFIT_TOL):
    """Check 8 (module docstring)."""
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    p = np.asarray(u_tr["person_id"]).astype(str)
    for v in variants:
        worst = 0.0
        for tgt in RO.TARGETS:
            m, _ = probes.fit_probe(data["train"]["X"][v], u_tr[tgt].to_numpy(np.float64), p)
            d = np.abs(m.predict(data["dev"]["X"][v]) - per_pass[f"pred_{tgt}_{v}"].to_numpy(np.float64))
            worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else float("inf"))
        check(out, 8, name, worst <= tol, f"{v}: dev predictions refitted from the fitting set alone, largest "
              f"difference {worst:.2e} (tolerance {tol:g})")


def line_from_rows(g, persons):
    """Per person the mean of the entering values, then the mean over persons (g3._line's point for stride, speed)."""
    g = g[_bool(g["enters"]) & g["person_id"].isin(set(persons))]
    return float(g.groupby("person_id")["value"].mean().mean()) if len(g) else float("nan")


def check_lines(out, name, rows, res, dev, dev12, variants):
    """Check 9 on the readout's own lines (module docstring)."""
    flag = _bool(rows["in_dev12"])
    check(out, 9, name, set(rows.loc[flag, "person_id"]) == set(dev12) and not set(rows.loc[~flag, "person_id"]) & set(dev12),
          f"in_dev12 is exactly 'person in dev12' ({int(flag.sum())} of {len(rows)} rows)")
    for v in variants:
        for ln in PASS_LINES:
            g = rows[(rows["line"] == ln) & (rows["variant"] == v)]
            for sname, persons in (("dev14", dev), ("dev12", dev12)):
                got = line_from_rows(g if sname == "dev14" else g[_bool(g["in_dev12"])], persons)
                ref = _float(res["variants"]["pass"][v]["dev"][sname][ln]["point"])
                check(out, 9, name, _same(got, ref), f"{v} {ln} {sname}: from the rows {got:.4f}, results.json {ref:.4f}")


def check_compare(out, name, res, comp):
    """Check 9 on the comparison: a baseline's chosen points in g12.json (every line, IC included)."""
    if res.get("kind") != "baseline":
        return
    for sname in ("dev14", "dev12"):
        for ln in ("stride_mae_cm", "speed_mae_m_s", "ic_median_ms"):
            got = (((comp.get("points") or {}).get(sname) or {}).get(ln) or {}).get(res["arm"], {}).get(str(res["seed"]))
            ref = res["lines"][sname][ln]["point"]
            check(out, 9, name, got is not None and _same(got, ref),
                  f"g12.json {res['arm']} seed {res['seed']} {sname} {ln}: {got}, readout {ref} "
                  f"(variant {res['lines'][sname][ln]['variant']})")


def check_body_points(out, report, pb, g3_dev, comp, seed, dev12):
    """Check 10 (module docstring) for one body report."""
    model = g3_dev["lines"]["dev"]["model"]
    for ln in PASS_LINES:
        g = pb[(pb["row"] == "dev") & (pb["predictor"] == "model") & (pb["line"] == ln)]
        b14 = float(g["value"].mean())
        b12 = float(g[g["person_id"].isin(set(dev12))]["value"].mean())
        c14 = ((comp.get("points") or {}).get("dev14", {}).get(ln, {}).get("body", {}) or {}).get(str(seed))
        c12 = ((comp.get("points") or {}).get("dev12", {}).get(ln, {}).get("body", {}) or {}).get(str(seed))
        check(out, 10, report, _same(b14, model[ln]["point"]) and _same(b14, c14) and _same(b12, c12),
              f"body seed {seed} {ln}: per_person.csv dev14 {b14:.4f} (g3_dev.json {_float(model[ln]['point']):.4f}, "
              f"g12.json {_float(c14):.4f}), dev12 {b12:.4f} (g12.json {_float(c12):.4f})")


# ---------------------------------------------------------------- diagnostics
def span_reference(u_tr, u_dev, tg_dev):
    """{set: {'stride_mae_cm', 'speed_mae_m_s'}} of the duration-only ridge (module docstring)."""
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    from mdfm.gates import g3

    def feats(u):
        d = (u["t1"] - u["t0"]).to_numpy(np.float64)
        return np.stack([1.0 / d, d], 1)

    p = np.asarray(u_tr["person_id"]).astype(str)
    pred = {}
    for tgt in RO.TARGETS:
        m, _ = probes.fit_probe(feats(u_tr), u_tr[tgt].to_numpy(np.float64), p)
        pred[tgt] = m.predict(feats(u_dev))
    pd_ = probes.pass_predictions(u_dev, pred, tg_dev, "span_duration")
    _, tgs = RO.dev_sets(tg_dev)
    out = {}
    for s, tg in tgs.items():
        st, _ = g3.stride_lines(pd_, tg)
        sp, _ = g3.speed_line(pd_, tg)
        out[s] = {"stride_mae_cm": float(st["stride_mae_cm"]["point"]), "speed_mae_m_s": float(sp["point"])}
    return out


def auc(score, y):
    """Mann-Whitney AUC of score for y (ties averaged); NaN without both classes; non-finite scores left out."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    r = (start + (cnt + 1) / 2.0)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def context_rows(X, start, trial_ids, patch_s=0.25):
    """Rows of patches p - 1, p, p + 1 of the same trial concatenated (NaN where a neighbour is absent)."""
    X = np.asarray(X, dtype=np.float64)
    tid = np.asarray(trial_ids).astype(str)
    k = np.round(np.asarray(start, dtype=np.float64) / patch_s).astype(np.int64)
    where = {(t, int(i)): j for j, (t, i) in enumerate(zip(tid, k))}
    D = X.shape[1]
    out = np.full((len(X), 3 * D), np.nan)
    for j, (t, i) in enumerate(zip(tid, k)):
        for b, d in enumerate((-1, 0, 1)):
            m = where.get((t, int(i) + d))
            if m is not None:
                out[j, b * D:(b + 1) * D] = X[m]
    return out


def _patch_stats(S, Yd, labd, threshold):
    from mdfm.eval import probes
    B = probes.SUB_BINS
    out = {}
    for fi, foot in enumerate(probes.FEET):
        sl = slice(fi * B, (fi + 1) * B)
        sc = S[labd, sl]
        p = sc.sum(axis=1)
        y = Yd[labd, sl].sum(axis=1) > 0
        fin = np.isfinite(p)
        hit = fin & y & (p > threshold)
        bt = np.argmax(Yd[labd, sl], axis=1)
        bp = np.argmax(np.where(np.isfinite(sc), sc, -np.inf), axis=1)
        mean = lambda a: float(np.mean(a)) if len(a) else float("nan")  # noqa: E731
        out[foot] = {"auc": auc(p, y), "mean_score_ic": mean(p[fin & y]), "mean_score_other": mean(p[fin & ~y]),
                     "hit": mean(p[fin & y] > threshold), "false_alarm": mean(p[fin & ~y] > threshold),
                     "bin_exact": mean(bt[hit] == bp[hit]), "bin_within_one": mean(np.abs(bt[hit] - bp[hit]) <= 1),
                     "n_ic_patches": int((fin & y).sum()), "n_other_patches": int((fin & ~y).sum())}
    return out


def _ic_lines(S, dv, tg_dev, threshold, name):
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    from mdfm.gates import g3
    ics = probes.decode_ics(S, dv["start"], dv["trial_id"], threshold=threshold)
    pred = g3.Predictions(name, ics[["trial_id", "foot", "t_s"]].copy())
    off, n = g3.fit_offsets([pred], tg_dev)[(name, "walkway")]
    _, tgs = RO.dev_sets(tg_dev)
    res = {"threshold": threshold, "n_ics": int(len(ics)),
           "offset_ms": 1000.0 * off if g3.offset_fitted(off) else None, "offset_n": int(n)}
    for s, tg in tgs.items():
        line, _ = g3.ic_line(pred, g3.reference_ics(tg, "walkway"), off)
        res[s] = {"point": float(line["point"]), "matched_share": float(line["matched_share"]),
                  "persons_inf": sorted(p for p, x in (line.get("per_person") or {}).items() if not np.isfinite(x))}
    return res


def ic_diagnostic(data, tg_tr, tg_dev, u_tr, u_dev, variants=None, thresholds=THRESHOLDS, context=True):
    """{variant: {'alpha', 'patches' {foot: stats} at probes.IC_THRESHOLD, 'thresholds' [line at each threshold],
    'context' {...} or None}} plus 'labelled' counts (module docstring)."""
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    variants = RO.IC_VARIANTS if variants is None else variants
    tr, dv = data["train"]["tokens"], data["dev"]["tokens"]
    Y, lab = probes.ic_targets(tr["start"], tr["trial_id"], tg_tr.walkway_ic, u_tr[["trial_id", "t0", "t1"]])
    Yd, labd = probes.ic_targets(dv["start"], dv["trial_id"], tg_dev.walkway_ic, u_dev[["trial_id", "t0", "t1"]])
    p_tr = np.asarray(tr["person_id"]).astype(str)
    out = {"labelled": {"train": int(lab.sum()), "train_rows": int(len(lab)), "dev": int(labd.sum()),
                        "dev_rows": int(len(labd))}, "variants": {}}
    for v in variants:
        Xtr = np.asarray(tr[v], dtype=np.float64)
        Xdv = np.asarray(dv[v], dtype=np.float64)
        m, s = probes.fit_probe(Xtr[lab], Y[lab], p_tr[lab])
        S = m.predict(Xdv)
        r = {"alpha": float(s["alpha"]), "patches": _patch_stats(S, Yd, labd, probes.IC_THRESHOLD),
             "thresholds": [_ic_lines(S, dv, tg_dev, t, f"ic_{v}") for t in thresholds], "context": None}
        if context:
            Ctr = context_rows(Xtr, tr["start"], tr["trial_id"])
            Cdv = context_rows(Xdv, dv["start"], dv["trial_id"])
            mc, sc = probes.fit_probe(Ctr[lab], Y[lab], p_tr[lab])
            Sc = mc.predict(Cdv)
            r["context"] = {"alpha": float(sc["alpha"]), "patches": _patch_stats(Sc, Yd, labd, probes.IC_THRESHOLD),
                            "line": _ic_lines(Sc, dv, tg_dev, probes.IC_THRESHOLD, f"ic_{v}_context")}
        out["variants"][v] = r
    return out


# ---------------------------------------------------------------- report
def _f(x, nd=2):
    x = _float(x)
    return "n/a" if math.isnan(x) else ("inf" if math.isinf(x) else f"{x:.{nd}f}")


def report_md(res):
    """checks.md from the results dict (every number read from it)."""
    L = [f"# G12 readout checks ({LABEL})", "",
         f"Written {res['time']} by scripts/check_g12_readouts.py (task {TASK}; SHA-256 {res['script_sha256']}). "
         f"MoveAtlas HEAD now {res['moveatlas_head'] or 'n/a'}; the readouts state "
         f"{', '.join(sorted(set(r['moveatlas_head'] or 'n/a' for r in res['readouts'].values())))}. Every number "
         "below is read from checks.json of this folder. Dev only: no lockbox, EVAL, CODY or care_pd record was read.",
         "", f"## Checks: {res['n_pass']} pass, {res['n_fail']} fail", "",
         "| check | readout | result | detail |", "|---|---|---|---|"]
    for c in res["checks"]:
        L.append(f"| {c['id']} | {c['readout'] or 'all'} | {'PASS' if c['ok'] else 'FAIL'} | "
                 f"{c['detail'].replace('|', '/')} |")
    L += ["", "## Span reference (pass duration alone, a ridge fitted on the 54)", "",
          "| set | stride MAE (cm) | speed MAE (m/s) |", "|---|---|---|"]
    for s, v in res["span_reference"].items():
        L.append(f"| {s} | {_f(v['stride_mae_cm'])} | {_f(v['speed_mae_m_s'], 3)} |")
    L += ["", "## IC probe in the tokens (dev rows inside the pass spans)", "",
          "Patch score: the sum of a foot's sub-patch scores, thresholded at 0.5 by the rule's probe (J0's). AUC: "
          "patch score for 'the patch holds a walkway IC of this foot'. Hit and false alarm at 0.5. Bin: the share of "
          "hit patches whose best sub-patch is the reference's (exact, within one).", "",
          "| readout | variant | alpha | AUC L / R | mean score IC / other, L | hit L / R | false alarm L / R | bin exact "
          "/ within one, L | context AUC L / R | context hit L / R |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["ic"].items():
        for v, d in r["variants"].items():
            pL, pR = d["patches"]["L"], d["patches"]["R"]
            c = d.get("context") or {}
            cL, cR = (c.get("patches") or {}).get("L", {}), (c.get("patches") or {}).get("R", {})
            L.append(f"| {name} | {v} | {_f(d['alpha'], 3)} | {_f(pL['auc'], 3)} / {_f(pR['auc'], 3)} | "
                     f"{_f(pL['mean_score_ic'])} / {_f(pL['mean_score_other'])} | {_f(pL['hit'], 3)} / {_f(pR['hit'], 3)} | "
                     f"{_f(pL['false_alarm'], 3)} / {_f(pR['false_alarm'], 3)} | {_f(pL['bin_exact'])} / "
                     f"{_f(pL['bin_within_one'])} | {_f(cL.get('auc'), 3)} / {_f(cR.get('auc'), 3)} | "
                     f"{_f(cL.get('hit'), 3)} / {_f(cR.get('hit'), 3)} |")
    L += ["", "IC line (ms, g3.ic_line, walkway reference, offset fitted on the 14) when the probe's ICs are decoded at "
          "other thresholds (sensitivity only; the rule's probe decodes at 0.5); matched share in brackets.", "",
          "| readout | variant | " + " | ".join(f"threshold {t:g}: dev14 / dev12" for t in res["thresholds"])
          + " | context at 0.5: dev14 / dev12 |", "|---|---|" + "---|" * (len(res["thresholds"]) + 1)]
    for name, r in res["ic"].items():
        for v, d in r["variants"].items():
            cells = [f"{_f(t['dev14']['point'], 1)} ({_f(t['dev14']['matched_share'], 3)}) / {_f(t['dev12']['point'], 1)} "
                     f"({_f(t['dev12']['matched_share'], 3)})" for t in d["thresholds"]]
            c = (d.get("context") or {}).get("line")
            cells.append(f"{_f(c['dev14']['point'], 1)} ({_f(c['dev14']['matched_share'], 3)}) / "
                         f"{_f(c['dev12']['point'], 1)} ({_f(c['dev12']['matched_share'], 3)})" if c else "n/a")
            L.append(f"| {name} | {v} | " + " | ".join(cells) + " |")
    L += ["", "## Sources", ""]
    for name, r in res["readouts"].items():
        L.append(f"- {name}: results.json SHA-256 {r['results_sha256']}, per_row.csv {r['per_row_sha256']}, per_pass.csv "
                 f"{r['per_pass_sha256']}, features {r['features']} ({r['features_sha256']}); kind {r['kind']}, arm "
                 f"{r['arm']}, seed {r['seed']}")
    for name, r in res["body_reports"].items():
        L.append(f"- body report {name}: g3_dev.json SHA-256 {r['g3_dev_sha256']}, per_person.csv {r['per_person_sha256']}")
    L.append(f"- comparison {res['compare']['path']}: SHA-256 {res['compare']['sha256']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def checkpoint_persons(path, drift_log=None):
    """The training persons of a checkpoint (train/checkpoint.load provenance; a tolerated records drift is logged to
    drift_log, beside the checkpoint by default, as every checkpoint read does)."""
    from mdfm.train import checkpoint as CK
    ck = CK.load(path, map_location="cpu", drift_log=drift_log)
    out = {p for ids in ck["provenance"]["persons"].values() for p in ids}
    del ck
    return out


def run(names, features_root=FEATURES, readouts_root=READOUTS, compare_path=COMPARE, checkpoints=False,
        drift_log=None, context=True, log=print):
    """The results dict (module docstring)."""
    import pandas as pd

    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    t0 = time.time()
    CMPS = _compare_module()
    train, dev = g3.controls("train"), g3.controls("dev")
    dev12 = sorted(set(dev) - set(g3.EXCLUDED_FROM_12))
    locked = set(g3.lockbox_persons())
    with open(compare_path, encoding="utf-8") as f:
        comp = json.load(f)
    body, body_sets, body_n = {}, {}, {}
    for seed, rep in CMPS.BODY_REPORTS.items():
        d = os.path.join(TREE, rep)
        with open(os.path.join(d, "g3_dev.json"), encoding="utf-8") as f:
            gj = json.load(f)
        pb = pd.read_csv(os.path.join(d, "per_person.csv"), encoding="utf-8", dtype={"person_id": str})
        body[rep] = (seed, gj, pb)
        body_sets[rep] = sorted(gj["sets"]["dev"])
        m = gj["lines"]["dev"]["model"]
        body_n[rep] = {"stride": int(m["stride_mae_cm"]["n_events"]), "speed": int(m["speed_mae_m_s"]["n_events"]),
                       "ic": int(m["ic_median_ms_walkway"]["n_events"])}
    checks = []
    check_global_sets(checks, train, dev, dev12, locked, body_sets, g3.EXCLUDED_FROM_12)
    tg_tr, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_tr.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise SystemExit("a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_tr), probes.pass_units(tg_dev)
    check_folds(checks, "", u_tr, dev)
    readouts, ic = {}, {}
    for name in names:
        rd = os.path.join(readouts_root, name)
        fp = os.path.join(features_root, name, "features.npz")
        with open(os.path.join(rd, "results.json"), encoding="utf-8") as f:
            res = json.load(f)
        rows = pd.read_csv(os.path.join(rd, "per_row.csv"), encoding="utf-8",
                           dtype={"pass_id": str, "person_id": str, "trial_id": str, "foot": str})
        per_pass = pd.read_csv(os.path.join(rd, "per_pass.csv"), encoding="utf-8",
                               dtype={"pass_id": str, "person_id": str, "trial_id": str})
        data, meta = RO.load_features(fp)
        if (meta.get("checkpoint") or {}).get("name") != (res.get("checkpoint") or {}).get("name"):
            raise SystemExit(f"{common.rel(fp)} holds another checkpoint's features than {name}")
        variants = [v for v in RO.VARIANTS if v in res["variants"]["pass"]]
        log(f"== {name} ({res.get('kind')}, arm {res.get('arm')}, seed {res.get('seed')})", flush=True)
        check_results_sets(checks, name, res, train, dev, dev12)
        check_features(checks, name, data, train, dev, locked)
        if checkpoints:
            p = res["checkpoint"]["path"]
            check_checkpoint(checks, name, checkpoint_persons(p if os.path.isabs(p) else os.path.join(TREE, p),
                                                              drift_log), train, dev, locked)
        check_rows(checks, name, rows, per_pass, tg_dev, body_n, variants)
        check_refit(checks, name, data, u_tr, per_pass, variants)
        check_lines(checks, name, rows, res, dev, dev12, variants)
        check_compare(checks, name, res, comp)
        ic[name] = ic_diagnostic(data, tg_tr, tg_dev, u_tr, u_dev,
                                 [v for v in RO.IC_VARIANTS if v in res["variants"]["ic"]["variants"]],
                                 THRESHOLDS, context)
        readouts[name] = {"kind": res.get("kind"), "arm": res.get("arm"), "seed": res.get("seed"),
                          "moveatlas_head": res.get("moveatlas_head"),
                          "results_sha256": common.sha256_file(os.path.join(rd, "results.json")),
                          "per_row_sha256": common.sha256_file(os.path.join(rd, "per_row.csv")),
                          "per_pass_sha256": common.sha256_file(os.path.join(rd, "per_pass.csv")),
                          "features": common.rel(fp), "features_sha256": common.sha256_file(fp)}
    for rep, (seed, gj, pb) in body.items():
        check_body_points(checks, rep, pb, gj, comp, seed, dev12)
    span = span_reference(u_tr, u_dev, tg_dev)
    n_fail = sum(not c["ok"] for c in checks)
    return {"what": LABEL, "task": TASK, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "script_sha256": common.sha256_file(os.path.abspath(__file__)), "moveatlas_head": PV.moveatlas_head(),
            "sets": {"train": len(train), "dev14": dev, "dev12": dev12}, "thresholds": list(THRESHOLDS),
            "refit_tol": REFIT_TOL, "checkpoints_checked": bool(checkpoints), "checks": checks,
            "n_pass": len(checks) - n_fail, "n_fail": n_fail, "span_reference": span, "ic": ic, "readouts": readouts,
            "body_reports": {rep: {"seed": seed,
                                   "g3_dev_sha256": common.sha256_file(os.path.join(TREE, rep, "g3_dev.json")),
                                   "per_person_sha256": common.sha256_file(os.path.join(TREE, rep, "per_person.csv"))}
                             for rep, (seed, _, _) in body.items()},
            "compare": {"path": common.rel(compare_path), "sha256": common.sha256_file(compare_path)},
            "seconds": round(time.time() - t0, 1)}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [_clean(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else ("inf" if o > 0 else "-inf" if o < 0 else None)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return _clean(o.item())
    return o


def write(res, out):
    from mdfm.gates import common
    res = _clean(res)
    common.write_text(os.path.join(out, "checks.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(out, "checks.md"), report_md(res))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--readouts", nargs="*", default=None, help="readout folder names (default: every folder with "
                    "results.json under reports/bench/g12/readouts)")
    ap.add_argument("--readouts-root", default=READOUTS)
    ap.add_argument("--features-root", default=FEATURES)
    ap.add_argument("--compare", default=COMPARE)
    ap.add_argument("--checkpoints", action="store_true", help="check 3: open each checkpoint for its provenance")
    ap.add_argument("--drift-log", default=None, help="where checkpoint.load logs a tolerated records drift "
                    "(default: beside the checkpoint)")
    ap.add_argument("--no-context", action="store_true", help="skip the neighbouring-patch IC probe")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    names = a.readouts
    if not names:
        names = sorted(d for d in os.listdir(a.readouts_root) if not d.startswith("._")
                       and os.path.exists(os.path.join(a.readouts_root, d, "results.json")))
    res = run(names, a.features_root, a.readouts_root, a.compare, a.checkpoints, a.drift_log, not a.no_context)
    for c in res["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'} {c['id']} {c['readout'] or 'all'}: {c['detail']}")
    for s, v in res["span_reference"].items():
        print(f"span reference {s}: stride {v['stride_mae_cm']:.2f} cm, speed {v['speed_mae_m_s']:.3f} m/s")
    for name, r in res["ic"].items():
        for v, d in r["variants"].items():
            pl, pr = d["patches"]["L"], d["patches"]["R"]
            t5 = d["thresholds"][-1]
            print(f"IC {name} {v}: alpha {d['alpha']:g}, AUC {pl['auc']:.3f} / {pr['auc']:.3f}, hit {pl['hit']:.3f} / "
                  f"{pr['hit']:.3f}; at {t5['threshold']:g}: dev14 {t5['dev14']['point']:.1f} ms, dev12 "
                  f"{t5['dev12']['point']:.1f} ms")
    print(f"{res['n_pass']} pass, {res['n_fail']} fail ({res['seconds']} s)")
    if a.write:
        write(res, os.path.abspath(a.out))
        print(f"written {os.path.abspath(a.out)}")
    return 1 if res["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
