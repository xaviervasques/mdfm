"""Post hoc coverage of task ZPD's ZUPT per person (wave 17, task ZPD, review fix, 2026-10-16). Every output is
labelled LABEL: post hoc, descriptive, covered by no prediction, deciding nothing.

Why: summary.md of the ZPD run (reports/gates/g3/zupt_pd/, prediction SHA-256 c0ab4d4b...) lists the ten persons with
the highest zupt_seg error per row, and sorts a person whose error is undefined (no matched stride) last, so the
persons with the lowest coverage do not appear in it. This script lists every person by the share of its walkway
strides that zupt_seg matches, lowest first, and classifies each stride zupt_seg leaves unmatched.

Reads only files the ZPD run wrote, plus two fold files: strides.csv, passes.csv, zupt_strides.csv,
train_strides.csv, results.json and train_fit.json of reports/gates/g3/zupt_pd/; validity.csv of the body's PD
folder (reports/gates/g3/dev_pd_phaseA_root_full_ckpt_B_032044/: the dev12 trials are its valid control records, as
the run asserted); T19's proposed PD fold (scripts/eval_g3_pd_dev.py t19_folds) for pd_t19dev. No record, target
file, lockbox, EVAL, CODY or care_pd row is read; a lockbox person in any input raises Refused. Every row's trials,
pooled zupt_seg matched share and number of persons with a matched stride are asserted equal to the run's
results.json (train_fit.json for train54), and the end differences recomputed from zupt_strides.csv and the run's lag
are asserted equal to those of strides.csv: the tables below describe the run, nothing new is fitted or read.

Rows: train54 (the 54 training-fold controls, in sample for the lag and the gains, reported for reference; no body
and no pass columns), dev12, dev14, pd_all, pd_t19dev. Per row and person (person_table): walkway strides; matched
share of zupt_seg, zupt_ff and the body; zupt_seg's mean absolute stride error (cm; NaN when no stride matches) and
the body's on its own matched strides (raw, and divided by F1's training-fold g_S, the run's f1_gains);
the median end difference of zupt_seg (s: the larger of the two end differences to the nearest zupt_seg stride of
the same trial and foot, g3.match_strides) over all the person's strides and over its unmatched ones; its unmatched
strides by class; its walkway passes, the share zupt_seg gives a speed for and the mean absolute speed error there.

Class of an unmatched walkway stride (classify), against the zupt_seg strides of the same trial and foot (event times:
the foot-flat onsets minus the run's lag):
- no_foot: the trial and foot have no ZUPT stride at all (no foot sensor, or no two consecutive still runs
  Z.MIN_STRIDE_S to Z.MAX_STRIDE_S apart);
- gap: no ZUPT stride spans the walkway stride's midpoint (consecutive still runs there lie more than
  Z.MAX_STRIDE_S apart, so the stride between them was not formed, or the stride lies outside the first and last
  still runs);
- merged: the spanning ZUPT stride lasts at least MERGE_RATIO times the walkway stride (a stance with no still run
  detected: one ZUPT stride covers two or more walkway strides);
- split: it lasts at most 1 / MERGE_RATIO times the walkway stride (a still run detected inside the stride);
- shifted: it lasts about as long as the walkway stride and an end lies more than g3.STRIDE_MATCH_S away.
A constant offset between the foot sensor's clock and the walkway's leaves strides of the walkway's duration with
shifted ends; a stance the still detector misses leaves merged strides and gaps. For each distinct merged ZUPT stride,
merged_len_ratio is its length over (walkway pass speed times its integration time duration_s): near 1 when the
integral across the missed stance still measures the distance walked.

Run (mdfm environment, CPU, a few seconds):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\
    <conda>/envs/mdfm/python.exe scripts/g3_zupt_pd_coverage.py
Writes reports/gates/g3/zupt_pd/posthoc_coverage/: persons.csv, unmatched.csv, results.json, summary.md,
_provenance.json.
"""
import argparse
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
from mdfm.gates import common, g3  # noqa: E402

import g3_zupt_baseline as Z  # noqa: E402
import g3_zupt_pd as P  # noqa: E402

TASK = "ZPD"
LABEL = "post hoc, descriptive (no prediction covers it); dev diagnostic, not a G3 read"
RUN = P.OUT
OUT = os.path.join(RUN, "posthoc_coverage")
ROWS = ("train54", "dev12", "dev14", "pd_all", "pd_t19dev")
FULL_ROWS = ("dev12", "pd_all")          # every person listed in summary.md
LOWEST_N = 10                            # the other rows: the ten lowest
MERGE_RATIO = 1.5
CLASSES = ("no_foot", "gap", "merged", "split", "shifted")
END_TOL = 1e-6                           # recomputed against written end differences (CSV round trip)
SHARE_TOL = 1e-9


# ---------------------------------------------------------------- sets
def row_sets(strides, train_strides, valid_controls, t19dev):
    """{row: frozenset of trial ids}: train54 (train_strides.csv), dev14 (group dev14 of strides.csv), dev12 (dev14
    and the valid control trials), pd_all (group pd), pd_t19dev (pd_all trials of T19's dev persons)."""
    dev14 = frozenset(strides.loc[strides["group"] == "dev14", "trial_id"])
    pd_rows = strides[strides["group"] == "pd"]
    t19 = {str(p) for p in t19dev}
    return {"train54": frozenset(train_strides["trial_id"]), "dev12": dev14 & frozenset(valid_controls),
            "dev14": dev14, "pd_all": frozenset(pd_rows["trial_id"]),
            "pd_t19dev": frozenset(pd_rows.loc[pd_rows["person_id"].astype(str).isin(t19), "trial_id"])}


# ---------------------------------------------------------------- classification
def seg_strides(zupt, lag_s):
    """{(trial, foot): (a0, a1, length_m, duration_s, index)} of the ZUPT strides at zupt_seg's event times (foot-flat
    onsets minus lag_s), sorted by a0."""
    out = {}
    z = zupt.reset_index(drop=True)
    for (t, f), g in z.groupby(["trial_id", "foot"], sort=False):
        g = g.sort_values("t0_ff", kind="stable")
        out[(str(t), str(f))] = (g["t0_ff"].to_numpy(np.float64) - lag_s, g["t1_ff"].to_numpy(np.float64) - lag_s,
                                 g["length_m"].to_numpy(np.float64), g["duration_s"].to_numpy(np.float64),
                                 g.index.to_numpy())
    return out


def classify(walk, segs, merge_ratio=MERGE_RATIO):
    """DataFrame aligned with walk (trial_id, foot, t0, t1, zupt_seg_matched): end_diff_s (recomputed, the nearest
    zupt_seg stride by the larger end difference, inf without one), cls ('matched' or a class of CLASSES), span_s
    (duration of the spanning ZUPT stride, NaN without one), start_offset_s (its start minus the walkway t0),
    zupt_index (index of the spanning stride in the zupt frame, -1 without one), zupt_length_m, zupt_duration_s."""
    import pandas as pd
    rows = []
    for r in walk.itertuples(index=False):
        g = segs.get((str(r.trial_id), str(r.foot)))
        rec = {"end_diff_s": math.inf, "cls": "matched", "span_s": math.nan, "start_offset_s": math.nan,
               "zupt_index": -1, "zupt_length_m": math.nan, "zupt_duration_s": math.nan}
        if g is not None and len(g[0]):
            a0, a1, ln, du, ix = g
            d = np.maximum(np.abs(a0 - r.t0), np.abs(a1 - r.t1))
            rec["end_diff_s"] = float(d.min())
        if not bool(r.zupt_seg_matched):
            if g is None or not len(g[0]):
                rec["cls"] = "no_foot"
            else:
                mid, dur = 0.5 * (r.t0 + r.t1), r.t1 - r.t0
                cover = np.flatnonzero((a0 <= mid) & (a1 >= mid))
                if not len(cover):
                    rec["cls"] = "gap"
                else:
                    k = int(cover[np.argmin(d[cover])])
                    span = float(a1[k] - a0[k])
                    rec.update(span_s=span, start_offset_s=float(a0[k] - r.t0), zupt_index=int(ix[k]),
                               zupt_length_m=float(ln[k]), zupt_duration_s=float(du[k]))
                    rec["cls"] = ("merged" if span >= merge_ratio * dur else
                                  "split" if span <= dur / merge_ratio else "shifted")
        rows.append(rec)
    return pd.DataFrame(rows, index=walk.index)


def check_end_diffs(walk, cl, tol=END_TOL):
    """AssertionError unless the recomputed end differences equal strides.csv's zupt_seg_end_diff_s (both infinite,
    or within tol) and the recomputed match flags equal zupt_seg_matched."""
    want = walk["zupt_seg_end_diff_s"].to_numpy(np.float64)
    got = cl["end_diff_s"].to_numpy(np.float64)
    both_inf = np.isinf(want) & np.isinf(got)
    with np.errstate(invalid="ignore"):
        ok = both_inf | (np.abs(want - got) <= tol)
    if not ok.all():
        raise AssertionError(f"{int((~ok).sum())} end differences differ from the run's strides.csv")
    flags = got <= g3.STRIDE_MATCH_S + 1e-9
    if not (flags == walk["zupt_seg_matched"].to_numpy(bool)).all():
        raise AssertionError("the recomputed zupt_seg match flags differ from the run's strides.csv")
    return True


# ---------------------------------------------------------------- tables
def _median(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else math.nan


def _mean(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else math.nan


def person_table(walk, cl, passes, pass_speed, row, g_s=None):
    """[dict] per person of the row's walkway strides (walk, with cl from classify): the columns of the module
    docstring, sorted by zupt_seg matched share ascending, then person id. passes: the row's walkway passes (or None).
    With body columns, also the body's mean absolute stride error (cm) on its matched strides, raw and divided by g_s
    (F1's training-fold gain, the run's f1_gains; NaN when g_s is None)."""
    w = walk.join(cl)
    has_body = "body_matched" in w.columns
    out = []
    for p, g in w.groupby("person_id", sort=True):
        m = g["zupt_seg_matched"].to_numpy(bool)
        x = 100.0 * g["length_m"].to_numpy(np.float64)
        err = np.abs(100.0 * g["zupt_seg_m"].to_numpy(np.float64) - x)[m]
        un = g[~m]
        merged = un[un["cls"] == "merged"].drop_duplicates("zupt_index")
        ratio = [r.zupt_length_m / (pass_speed.get((r.trial_id, str(r.pass_id)), math.nan) * r.zupt_duration_s)
                 for r in merged.itertuples()] if pass_speed is not None else []
        if has_body:
            bm = g["body_matched"].astype(bool).to_numpy()
            yb = g["body_cm"].to_numpy(np.float64)
            body = {"body_share": float(bm.mean()), "body_mae_cm": _mean(np.abs(yb - x)[bm]),
                    "body_gain_mae_cm": _mean(np.abs(yb / g_s - x)[bm]) if g_s else math.nan}
        else:
            body = {"body_share": math.nan, "body_mae_cm": math.nan, "body_gain_mae_cm": math.nan}
        rec = {"row": row, "person_id": str(p), "strides": int(len(g)), "zupt_seg_share": float(m.mean()),
               "zupt_ff_share": float(g["zupt_ff_matched"].astype(bool).mean()), **body,
               "zupt_seg_mae_cm": float(err.mean()) if len(err) else math.nan,
               "end_diff_median_s": _median(g["end_diff_s"]), "unmatched": int(len(un)),
               "unmatched_end_diff_median_s": _median(un["end_diff_s"]),
               **{f"n_{c}": int((un["cls"] == c).sum()) for c in CLASSES},
               "merged_len_ratio_median": _median(ratio)}
        if passes is not None:
            q = passes[passes["person_id"] == p]
            pm = q["zupt_seg_matched"].astype(bool).to_numpy()
            e = np.abs(q["zupt_seg_m_s"].to_numpy(np.float64) - q["speed_m_s"].to_numpy(np.float64))[pm]
            rec |= {"passes": int(len(q)), "zupt_seg_pass_share": float(pm.mean()) if len(q) else math.nan,
                    "zupt_seg_speed_mae_m_s": float(e.mean()) if len(e) else math.nan}
        else:
            rec |= {"passes": 0, "zupt_seg_pass_share": math.nan, "zupt_seg_speed_mae_m_s": math.nan}
        out.append(rec)
    return sorted(out, key=lambda r: (r["zupt_seg_share"], r["person_id"]))


def row_summary(persons, walk, cl, pass_speed):
    """Per row: persons, strides, pooled zupt_seg share (and without the persons with no matched stride), the
    per-person share quantiles (min, p10, p25, p50), the persons individually below the G3 floor
    (g3.MATCHED_SHARE_FLOOR) and with no matched stride, the share of the row's unmatched strides those below the floor
    hold, the unmatched strides by class and the share of gap and merged among them (strides between still runs found
    too far apart), the median merged_len_ratio over distinct merged ZUPT strides and the median start offset of the
    shifted strides."""
    floor = g3.MATCHED_SHARE_FLOOR
    sh = np.array([r["zupt_seg_share"] for r in persons], np.float64)
    m = walk["zupt_seg_matched"].to_numpy(bool)
    w = walk.join(cl)
    un = w[~m]
    below = [r for r in persons if r["zupt_seg_share"] < floor]
    none = {r["person_id"] for r in persons if r["zupt_seg_share"] == 0}
    keep = ~walk["person_id"].astype(str).isin(none).to_numpy()
    merged = un[un["cls"] == "merged"].drop_duplicates("zupt_index")
    ratio = [r.zupt_length_m / (pass_speed.get((r.trial_id, str(r.pass_id)), math.nan) * r.zupt_duration_s)
             for r in merged.itertuples()] if pass_speed is not None else []
    n_gm = int(un["cls"].isin(("gap", "merged")).sum())
    return {"persons": len(persons), "strides": int(len(walk)), "pooled_share": float(m.mean()),
            "pooled_share_without_no_match": float(m[keep].mean()) if keep.any() else math.nan,
            "gap_or_merged_share": (n_gm / len(un)) if len(un) else math.nan,
            "persons_with_a_match": int(sum(r["zupt_seg_share"] > 0 for r in persons)),
            "share_min": float(sh.min()), "share_p10": float(np.quantile(sh, 0.1)),
            "share_p25": float(np.quantile(sh, 0.25)), "share_p50": float(np.quantile(sh, 0.5)),
            "persons_below_floor": len(below), "floor": floor,
            "persons_without_match": [r["person_id"] for r in persons if r["zupt_seg_share"] == 0],
            "unmatched": int(len(un)),
            "unmatched_held_below_floor": (float(sum(r["unmatched"] for r in below) / len(un)) if len(un) else
                                           math.nan),
            "unmatched_by_class": {c: int((un["cls"] == c).sum()) for c in CLASSES},
            "merged_zupt_strides": int(len(merged)), "merged_len_ratio_median": _median(ratio),
            "shifted_start_offset_median_s": _median(un.loc[un["cls"] == "shifted", "start_offset_s"])}


def check_against_run(summ, results, train_fit, sets):
    """AssertionError unless each row's trials, pooled zupt_seg share and persons with a matched stride equal the
    run's (results.json; train_fit.json for train54)."""
    for row, s in summ.items():
        if row == "train54":
            line = train_fit["train54_lines"][P.SEG]["stride_mae_cm"]
        else:
            line = results["lines"][row][P.SEG]["stride_mae_cm"]
            if results["sets"][row]["n_trials"] != len(sets[row]):
                raise AssertionError(f"{row}: {len(sets[row])} trials against the run's {results['sets'][row]}")
        if abs(float(line["matched_share"]) - s["pooled_share"]) > SHARE_TOL:
            raise AssertionError(f"{row}: pooled share {s['pooled_share']} against the run's {line['matched_share']}")
        if int(line["n_persons"]) != s["persons_with_a_match"]:
            raise AssertionError(f"{row}: {s['persons_with_a_match']} persons with a match against the run's "
                                 f"{line['n_persons']}")
        if int(line["n_events"]) != s["strides"]:
            raise AssertionError(f"{row}: {s['strides']} strides against the run's {line['n_events']}")
    return True


# ---------------------------------------------------------------- report
def fmt(x, nd=2):
    return Z.fmt(x, nd)


def _person_line(r):
    return (f"| {r['person_id']} | {r['strides']} | {fmt(r['zupt_seg_share'], 3)} | {fmt(r['zupt_ff_share'], 3)} | "
            f"{fmt(r['body_share'], 3)} | {fmt(r['zupt_seg_mae_cm'])} | {fmt(r['body_mae_cm'])} | "
            f"{fmt(r['body_gain_mae_cm'])} | {fmt(r['end_diff_median_s'], 2)} | "
            f"{fmt(r['unmatched_end_diff_median_s'], 2)} | "
            + ", ".join(str(r[f"n_{c}"]) for c in CLASSES)
            + f" | {fmt(r['merged_len_ratio_median'], 2)} | {fmt(r['zupt_seg_pass_share'], 3)} |")


def summary_md(res):
    s = res["rows"]
    L = [f"# ZPD: coverage of zupt_seg per person ({LABEL})", "",
         f"Written by scripts/g3_zupt_pd_coverage.py after the ZPD run and its review (run at {res.get('run_at')}). "
         "Post hoc and "
         "descriptive: no prediction covers these tables and they decide nothing. Read from the run's files "
         f"(reports/gates/g3/zupt_pd/, prediction SHA-256 {res['run']['prediction_sha256']}, run of "
         f"{res['run']['run_started']}, MoveAtlas {res['run']['moveatlas_head']}); each row's trials, pooled "
         "matched share and persons with a matched stride are asserted equal to the run's results.json, and the end "
         "differences recomputed from zupt_strides.csv with the run's lag "
         f"({fmt(res['run']['lag_s'], 4)} s) equal those of the run's strides.csv. No record or target was read.", "",
         "Why: the run's summary.md lists the ten persons with the highest zupt_seg error per row and sorts a person "
         "with no matched stride last, so it hides the persons with the lowest coverage. Here every person is sorted "
         "by its zupt_seg matched share, lowest first.", "",
         "## Per row", "",
         f"Persons below the floor: persons whose own matched share is below {res['floor']} (the G3 floor is a "
         "condition on a row's pooled share; it is applied here per person only to describe the spread). "
         "Unmatched held below the floor: the share of the row's unmatched strides that these persons hold. "
         "Classes of the unmatched strides (module docstring): no_foot, gap, merged, split, shifted.", "",
         "| row | persons | strides | pooled share | without persons with no match | per-person share min, p10, "
         "p25, p50 | persons below the floor | persons with no match | unmatched | held below the floor | no_foot, "
         "gap, merged, split, shifted | gap or merged | merged length ratio | shifted start offset (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in ROWS:
        x = s[row]
        L.append(f"| {row} | {x['persons']} | {x['strides']} | {fmt(x['pooled_share'], 3)} | "
                 f"{fmt(x['pooled_share_without_no_match'], 3)} | "
                 + ", ".join(fmt(x[k], 3) for k in ("share_min", "share_p10", "share_p25", "share_p50"))
                 + f" | {x['persons_below_floor']} | {', '.join(x['persons_without_match']) or 'none'} | "
                 f"{x['unmatched']} | {fmt(x['unmatched_held_below_floor'], 3)} | "
                 + ", ".join(str(x["unmatched_by_class"][c]) for c in CLASSES)
                 + f" | {fmt(x['gap_or_merged_share'], 3)} | {fmt(x['merged_len_ratio_median'], 2)} "
                 f"({x['merged_zupt_strides']} strides) | {fmt(x['shifted_start_offset_median_s'], 2)} |")
    L += ["", "train54 is in sample for the lag and the gains (reported for reference; train_strides.csv has no body "
          "and no pass columns, so its merged length ratio is n/a). Per person below: the body's errors are over the "
          f"body's own matched strides (raw, and divided by F1's training-fold g_S {fmt(res['run'].get('f1_g_S'), 4)}"
          "), the zupt_seg error over zupt_seg's matched strides.", ""]
    head = ("| person | strides | zupt_seg share | zupt_ff share | body share | zupt_seg MAE (cm) | body MAE (cm) | "
            "body with F1 gain MAE (cm) | end diff median (s) | unmatched end diff median (s) | no_foot, gap, merged, "
            "split, shifted | merged length ratio | zupt_seg pass share |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    for row in ROWS:
        rows = res["persons"][row]
        if row in FULL_ROWS:
            L += [f"## {row}: every person, lowest zupt_seg share first", "", head, sep]
            L += [_person_line(r) for r in rows]
        else:
            L += [f"## {row}: the {LOWEST_N} persons with the lowest zupt_seg share", "", head, sep]
            L += [_person_line(r) for r in rows[:LOWEST_N]]
        L.append("")
    L += [f"Time: {res['timing_s']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def load_run(run=RUN):
    import pandas as pd
    rd = {"strides": pd.read_csv(os.path.join(run, "strides.csv"), encoding="utf-8",
                                 dtype={"trial_id": str, "person_id": str, "foot": str, "pass_id": str}),
          "passes": pd.read_csv(os.path.join(run, "passes.csv"), encoding="utf-8",
                                dtype={"trial_id": str, "person_id": str, "pass_id": str}),
          "zupt": pd.read_csv(os.path.join(run, "zupt_strides.csv"), encoding="utf-8",
                              dtype={"trial_id": str, "person_id": str, "foot": str}),
          "train": pd.read_csv(os.path.join(run, "train_strides.csv"), encoding="utf-8",
                               dtype={"trial_id": str, "person_id": str, "foot": str, "pass_id": str})}
    with open(os.path.join(run, "results.json"), encoding="utf-8") as f:
        rd["results"] = json.load(f)
    with open(os.path.join(run, "train_fit.json"), encoding="utf-8") as f:
        rd["train_fit"] = json.load(f)
    return rd


def compute(rd, valid_controls, t19dev, locked):
    """{rows: row summaries, persons: person tables, unmatched: DataFrame} of a loaded run (load_run)."""
    import pandas as pd
    st, tr, zp, ps = rd["strides"], rd["train"], rd["zupt"], rd["passes"]
    for name, df in (("strides.csv", st), ("train_strides.csv", tr), ("zupt_strides.csv", zp),
                     ("passes.csv", ps)):
        Z.check_persons(set(df["person_id"]), None, locked, name)
    lag = float(rd["results"]["fit"]["lag"]["lag_s"])
    if abs(lag - P.FIT_LAG_S) > P.FIT_TOL:
        raise AssertionError(f"the run's lag {lag} is not the registered {P.FIT_LAG_S}")
    sets = row_sets(st, tr, valid_controls, t19dev)
    segs = seg_strides(zp, lag)
    pass_speed = {(r.trial_id, str(r.pass_id)): float(r.speed_m_s) for r in ps.itertuples()}
    g_s = (rd["results"].get("f1_gains") or {}).get("g_S")
    summ, tables, unm = {}, {}, []
    for row in ROWS:
        src = tr if row == "train54" else st
        walk = src[src["trial_id"].isin(sets[row])]
        cl = classify(walk, segs)
        check_end_diffs(walk, cl)
        pr = ps[ps["trial_id"].isin(sets[row])] if row != "train54" else None
        spd = pass_speed if row != "train54" else None
        tables[row] = person_table(walk, cl, pr, spd, row, g_s)
        summ[row] = row_summary(tables[row], walk, cl, spd)
        u = walk.join(cl)
        u = u[~u["zupt_seg_matched"].astype(bool)]
        unm.append(u.assign(row=row)[["row", "trial_id", "person_id", "foot", "pass_id", "t0", "t1", "length_m",
                                      "end_diff_s", "cls", "span_s", "start_offset_s", "zupt_length_m",
                                      "zupt_duration_s"]])
    check_against_run(summ, rd["results"], rd["train_fit"], sets)
    return {"rows": summ, "persons": tables, "unmatched": pd.concat(unm, ignore_index=True), "lag_s": lag,
            "sets": {r: len(t) for r, t in sets.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t_start = time.time()
    run_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    import pandas as pd
    from mdfm import paths
    Z.check_prediction(P.PRED_PATH, P.PRED_SHA_PATH, P.PRED_SHA256)
    locked = g3.lockbox_persons()
    rd = load_run()
    vf = pd.read_csv(P.PD_VALIDITY, encoding="utf-8", dtype={"trial_id": str, "person_id": str, "group": str})
    Z.check_persons(set(vf["person_id"]), None, locked, os.path.basename(P.PD_VALIDITY))
    valid = vf["valid"].astype(str).str.strip().str.lower() == "true"
    valid_controls = frozenset(vf.loc[(vf["group"] == "control") & valid, "trial_id"])
    t19dev, _, _ = P.PDM().t19_folds()
    Z.check_persons(set(t19dev), None, locked, "T19 dev fold")
    out = compute(rd, valid_controls, t19dev, locked)
    rs = rd["results"]
    res = {"what": LABEL, "task": TASK, "run_at": run_at, "floor": g3.MATCHED_SHARE_FLOOR, "merge_ratio": MERGE_RATIO,
           "run": {"path": common.rel(RUN), "prediction_sha256": rs["prediction"]["sha256"],
                   "run_started": rs["run_started"], "moveatlas_head": rs["moveatlas"]["head"], "lag_s": out["lag_s"],
                   "f1_g_S": rs["f1_gains"]["g_S"]},
           "sets": out["sets"], "rows": out["rows"], "persons": out["persons"],
           "checks": {"rows equal the run (trials, pooled share, persons with a match, strides)": True,
                      "end differences equal the run's strides.csv": True, "lockbox persons in any input": 0}}
    res["timing_s"] = round(time.time() - t_start, 1)
    clean = Z.clean(res)
    text = summary_md(clean)
    if any(c in text for c in ("\u2013", "\u2014")):
        raise ValueError("summary.md holds an en or em dash")
    common.write_text(os.path.join(OUT, "results.json"),
                      json.dumps(clean, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), text)
    common.write_csv(pd.DataFrame([r for row in ROWS for r in out["persons"][row]]), os.path.join(OUT, "persons.csv"))
    common.write_csv(out["unmatched"], os.path.join(OUT, "unmatched.csv"))
    inputs = [os.path.join(RUN, f) for f in ("strides.csv", "passes.csv", "zupt_strides.csv", "train_strides.csv",
                                             "results.json", "train_fit.json", "prediction.txt")]
    stamp = Z._stamp()
    prov = {"script": "scripts/g3_zupt_pd_coverage.py", "task": TASK, "label": LABEL, "run_at": run_at,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp,
            "measured_run_moveatlas_head": rs["moveatlas"]["head"],
            "inputs_sha256": {common.rel(p): common.sha256_file(p) for p in
                              inputs + [P.PD_VALIDITY, P.PDM().FOLDS_CSV, P.PDM().STAGE2_SUMMARY]},
            "script_sha256": common.sha256_file(os.path.abspath(__file__)),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), P.__file__, Z.__file__]),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                           "PYTHONDONTWRITEBYTECODE")}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(Z.clean(prov), indent=1) + "\n")
    print(text)
    print(f"written {common.rel(OUT)}")


if __name__ == "__main__":
    main()
