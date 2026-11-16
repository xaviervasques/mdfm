"""L5AN (2026-10-20): the dated section-10 note of the F2 lever-5a rule (reports/bench/f2_lever5a/rule.txt). Label of
every output: dev diagnostic, not a G3 read; every 5a row is labelled ROW_LABEL ('inertially anchored body (lever 5a,
at inference)').

Rule section 10: a criterion is never changed after a 5a number is seen; 'If a criterion proves badly posed, the output
says so and keeps both readings.' The reading of the rule (scripts/f2_lever5a.py: reports/bench/f2_lever5a/results.json
and summary.md) stays as it is, NOT EVALUATED (case 1): the code control's rerun check (scripts/eval_g3_diag.py
rerun_check, per-person stride tolerance 1 cm) failed on WHC003. This note changes no criterion, no output of the rule
and no verdict. It asserts the rule's SHA-256 (scripts/f2_lever5a.py check_rule, RULE_SHA256) before it reads anything,
reads existing files only, and computes:
a. R0: every run of v1's B end (cache/runs/phaseA_root_full/ckpt_B_032044.pt, EMA weights) at the settings of the
   rule's D1 under reports/ (V1_RUNS, each checked against its own JSON; the walk reads every *.json under reports/
   naming that checkpoint at its top level, lockbox, CODY and care_pd folders pruned: a JSON that records at least one
   D1 inference setting and none other is at D1 settings, and each such JSON not read is listed in EXCLUDED with its
   reason or reported as unlisted; a JSON that records no inference setting is listed apart): per-person all14
   stride MAE, per-person run-to-run SD and range, the run pairs with a person beyond 1 cm and with WHC003 beyond 1 cm,
   the same-code pairs (one script, one recorded code SHA-256), the rerun check of each run against v1's dev report
   (eval_g3_diag's three criteria, reproduced and asserted equal to the checks the diag runs store), the code control's
   WHC003 position among the other runs, and the per-trial refinement objective (refine_first, refine_last of
   trials.csv) of the control against the reference beside the same-code pairs.
b. Both readings: as written (results.json); with R0's per-person item set aside, the rule's own verdict function
   (scripts/f2_lever5a.py verdict) on the existing R1, R2 and T6; and the rule's lines with each v1 run at D1 settings
   in the place of the reference (R1 by the rule's paired bootstrap, S(14), V(14), the walkway IC and stride_mape_pct,
   SLIDE with the rule's thresholds unchanged), through the same verdict function.
c. The fragile lines: the walkway IC (per-person moves, exact arithmetic on the 0.5 ms grid of the per-person
   medians, asserted) and the SLIDE margin (legs rise minus half the base rise) per v1 reference with its mean;
   s2_B's beside.
d. The bar: the anchored body against the version-2 ZUPT on the dev12 strides both match (the join of
   scripts/f2_lever5.py decoded_over_zupt), and the label-free G3-matched ZUPT line (reports/gates/g3/zupt_dev).
e. The foot report, reported beside and descriptive (rule section 5: it reads nothing; rule section 8: no 5a dev number
   is confirmatory evidence of a mechanism): the 5a diag against the code control (zupt_strides.csv, row base, dev12),
   the added displacement D, the added slide over the measured still run, the slide share of the added displacement;
   the legs over base rise ratio of SLIDE per v1 reference.
f. The rule's route (section 9), as results.json prints it.
g. Beside the rule, outside its section 9 routes: a programme proposal to the owner, who decides.
Every sentence that states a conclusion is conditioned on the fields it rests on (a HarnessDefect when they do not
support it).

Writes reports/bench/f2_lever5a/note_section10/results.json, note.md and _provenance.json (with the MoveAtlas stamp, as
scripts/f2_lever5a.py writes it). The output folder is that folder or a folder outside this tree and outside the
MoveAtlas tree (a test's temporary folder); any other folder is refused (exit 6). Never over an existing results.json,
note.md or _provenance.json (exit 5) unless --force-rewrite-note is given, which only the note's own folder honours.
Exit codes: 0 the note written; 3 the rule is not hashed or its hash differs (nothing read); 4 a harness defect (a run
at other settings, a reproduction that disagrees with the file it reproduces, a conclusion its fields do not support);
5 an output exists; 6 the output folder is refused.
Data discipline: the dev14 and dev12 persons of F1, guarded against the lockbox list; two events files (the H4root dump,
the lever-2 probe) also hold training-fold rows, dropped by person before any value is used; no lockbox, EVAL, CODY or
care_pd record; CPU only (CUDA_VISIBLE_DEVICES=-1).

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever5a_note.py [--out DIR] [--force-rewrite-note]
"""
import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the note stays off the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")


def _load_reading():
    key = "_l5an_f2_lever5a"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, "f2_lever5a.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[key] = m
        spec.loader.exec_module(m)
    return sys.modules[key]


LA = _load_reading()                                  # scripts/f2_lever5a.py: the rule's constants and readers
TASK = "L5AN"
NOTE_DATE = "2026-10-20"
LABEL, ROW_LABEL = LA.LABEL, LA.ROW_LABEL
BENCH = LA.BENCH
NOTE_REL = os.path.join(BENCH, "note_section10")
RESULTS_REL = os.path.join(BENCH, "results.json")
SUMMARY_REL = os.path.join(BENCH, "summary.md")
V1_B, V1_SHA, G3 = LA.V1_B, LA.V1_SHA, LA.G3
REF_INFERENCE = LA.REF_INFERENCE
LEGS_ROW, MIN_ROW_PERSONS, TOL = LA.LEGS_ROW, LA.MIN_ROW_PERSONS, LA.TOL
FOCUS = "weargait:WHC003"
DEV_REF, RULE_REF, CONTROL = "g3_dev", "g3_diag", "code_control"
EXIT_WRITTEN, EXIT_NOT_HASHED, EXIT_HARNESS, EXIT_EXISTS, EXIT_REFUSED = 0, 3, 4, 5, 6
OUTPUT_NAMES = ("results.json", "note.md", "_provenance.json")
SENSITIVE = ("lockbox", "cody", "care_pd", "carepd")     # folder or file names the walk never enters or opens
INFERENCE_KEYS = tuple(k for k in REF_INFERENCE if k != "weights")   # the settings an inference block records
H4 = os.path.join("reports", "bench", "f2_scale", "H4root", f"dump_{V1_B}")
H1 = os.path.join("reports", "bench", "f2_scale", "H1shape", f"dump_{V1_B}")
P1 = os.path.join("reports", "bench", "f2_lever1", f"precision_{V1_B}")
P2 = os.path.join("reports", "bench", "f2_lever2", f"probe_{V1_B}")
S2_ABL = os.path.join(G3, f"diag_{LA.S2_B}_ablations_bw32")
ZUPT_V2_STRIDES = os.path.join("reports", "bench", "scale", "zupt_v2", "real_strides.csv")


class HarnessDefect(AssertionError):
    """A run at other settings, a person outside the sets, a reproduction that disagrees with its file."""


def _run(key, folder, kind, meta, what, events=None, row=None, prov="_provenance.json", pass_check=None):
    return {"key": key, "folder": folder, "kind": kind, "meta": meta, "what": what, "events": events, "row": row,
            "prov": prov, "pass_check": pass_check}


# The runs of v1's B end at the D1 settings (refine_steps 20, points_per_stream 256, batch_windows 32, seed 20260929,
# shape pooled per trial, EMA weights), oldest first; every one is checked against its own JSON (run_meta).
V1_RUNS = (
    _run(DEV_REF, os.path.join(G3, f"dev_{V1_B}"), "dev", "g3_dev.json",
         "v1's G3 dev report: the rule's dev reference and the rerun check's reference"),
    _run(RULE_REF, os.path.join(G3, f"diag_{V1_B}"), "diag", "diag.json", "v1's G3 diag report: the rule's reference",
         "events.csv", "base"),
    _run("g3_diag_ablations", os.path.join(G3, f"diag_{V1_B}_ablations"), "diag", "diag.json",
         "the G3 diag ablations run, row base", "events.csv", "base"),
    _run("H4root_dump", H4, "events", "dump.json",
         "the F2-H4root dump, row base (dump.json records no shape_pooling; scripts/f2_H4root.py builds G3Predictor "
         "with pool=True, one group per trial)", "events.csv", "base"),
    _run("lever1_bf16", P1, "events", "decode.json", "the F2 lever-1 precision probe, pass bf16 (dev12 only)",
         "events.csv", "bf16", "_provenance_decode.json", {"autocast_bf16": True}),
    _run("lever1_bf16_rerun", P1, "events", "decode.json",
         "the F2 lever-1 precision probe, pass bf16_rerun (dev12 only; its fp32 pass is at another precision, left "
         "out)", "events.csv", "bf16_rerun", "_provenance_decode.json", {"autocast_bf16": True}),
    _run("H1shape_dump", H1, "events", "dump.json", "the F2-H1shape dump, its base events", "events_base.csv",
         "base"),
    _run("lever2_base", P2, "events", "decode.json", "the F2 lever-2 probe, pass base (dev12 only)", "events.csv",
         "base", "_provenance_decode.json", {"columns_scaled": 0}),
    _run("lever2_base_rerun", P2, "events", "decode.json",
         "the F2 lever-2 probe, pass base_rerun (dev12 only; its other passes scale input channels, left out)",
         "events.csv", "base_rerun", "_provenance_decode.json", {"columns_scaled": 0}),
    _run("g3_diag_ablations_bw32", os.path.join(G3, f"diag_{V1_B}_ablations_bw32"), "diag", "diag.json",
         "the G3 diag ablations run at batch 32, row base (lever 5's d_base and d_legs read it)", "events.csv", "base"),
    _run("g3_diag_bw32_now", os.path.join(G3, f"diag_{V1_B}_bw32_now"), "diag", "diag.json",
         "a G3 diag rerun at batch 32, row base", "events.csv", "base"),
    _run("g3_dev_bw32_now", os.path.join(G3, f"dev_{V1_B}_bw32_now"), "dev", "g3_dev.json",
         "a G3 dev rerun at batch 32"),
    _run("inference_mode_ema_s20_pool", os.path.join("reports", "gates", "inference_mode", "ema_s20_pool", "g3_dev"),
         "dev", "g3_dev.json", "the inference-mode study, arm EMA, 20 steps, pooled"),
    _run(CONTROL, LA.OUT["off_diag"], "diag", "diag.json",
         "the rule's code control (--zupt-report under the 5a code): the run whose rerun check failed", "events.csv",
         "base"),
)
EXCLUDED = {
    f"{LA.OUT['a5_diag']}/diag.json": "the 5a arm itself (the option on), not a run without it",
    f"{LA.OUT['a5_dev']}/g3_dev.json": "the 5a arm itself (the option on), not a run without it",
    f"reports/bench/g3_gain_check/decode_{V1_B}/decode.json": "a train54 decode (train_events.csv): no dev14 stride",
    f"reports/bench/f2_scale/H3angles/dump_{V1_B}/dump.json":
        "a joint-angle dump (strides_real.csv, strides_insim.csv): no matched G3 stride prediction",
    "reports/bench/f2_scale/H1shape/cpu_test_dump_HC100/dump.json":
        "a CPU test dump of one training-fold person (HC100): no dev14 stride",
    f"reports/gates/g3/dev_pd_{V1_B}/g3_pd_dev.json": "the PD dev persons, not dev14",
}
EXCLUDED = {k.replace("\\", "/"): v for k, v in EXCLUDED.items()}
# JSONs of the checkpoint that record no inference setting but hold dev decodes: named with what they are (each must
# be among the walk's files without settings, else a HarnessDefect).
NOTED_WITHOUT_SETTINGS = {
    f"reports/bench/g3_wrist/decode_{V1_B}/decode.json":
        "the wrist-only CPU decode of task WRS (its settings block names the sites wrist_L and wrist_R): another input",
}


# ---------------------------------------------------------------- small helpers
def rel(path, tree=None):
    return LA.rel(path, tree)


def read_json(path):
    return LA.read_json(path)


def fin(x):
    return LA.fin(x)


def short(pid):
    s = str(pid).split(":")[-1].strip()
    return s.split(" ")[0] if s else s


def tol_of():
    """eval_g3_diag's rerun tolerances (stride MAE, speed MAE, person stride)."""
    return dict(LA._sibling("eval_g3_diag").RERUN_TOL)


def frac_text(f):
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"


# ---------------------------------------------------------------- pure functions (tests/test_f2_lever5a_note.py)
def person_mae(frame):
    """{person_id: mean |y - x|} of a frame with columns person_id, x, y."""
    if frame is None or not len(frame):
        return {}
    e = (frame["y"] - frame["x"]).abs()
    return {str(p): float(v) for p, v in e.groupby(frame["person_id"]).mean().items()}


def spread(values, persons):
    """Per person of `persons`: {n_runs, mean, sd (ddof 1), range, min, max} over the runs of `values` ({run: {person:
    value}}) that hold a value for the person."""
    out = {}
    for p in persons:
        x = np.array([v[p] for v in values.values() if v.get(p) is not None], dtype=np.float64)
        if not len(x):
            continue
        out[p] = {"n_runs": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                  "range": float(x.max() - x.min()), "min": float(x.min()), "max": float(x.max())}
    return out


def pairs(values, tol, focus):
    """Every pair (a, b) of runs in the order of `values`: over the persons both hold, b minus a per person, the
    largest |difference| and its person, the persons beyond tol and the focus person's difference (None when one run
    lacks it)."""
    out = []
    for a, b in itertools.combinations(list(values), 2):
        common = sorted(p for p in values[a] if values[a][p] is not None and values[b].get(p) is not None)
        if not common:
            continue
        d = {p: values[b][p] - values[a][p] for p in common}
        arg = max(common, key=lambda p: abs(d[p]))
        out.append({"a": a, "b": b, "n_persons": len(common), "max_abs": abs(d[arg]), "argmax": arg,
                    "beyond": {p: x for p, x in d.items() if abs(x) > tol + TOL}, "focus": d.get(focus)})
    return out


def pair_counts(prs, n_all, tol):
    """Counts over the pairs of `pairs`: every pair, the pairs holding all n_all persons, the pairs holding the focus
    person, and of each the pairs with a person (or the focus person) beyond tol."""
    full = [x for x in prs if x["n_persons"] == n_all]
    foc = [x for x in prs if x["focus"] is not None]
    return {"pairs": len(prs), "pairs_with_a_person_beyond": sum(1 for x in prs if x["beyond"]),
            "pairs_of_all_persons": len(full),
            "pairs_of_all_persons_with_a_person_beyond": sum(1 for x in full if x["beyond"]),
            "pairs_holding_the_focus_person": len(foc),
            "pairs_with_the_focus_person_beyond": sum(1 for x in foc if abs(x["focus"]) > tol + TOL)}


def same_code_pairs(meta):
    """The pairs (a, b) of runs, in the order of `meta` ({run: {script, code_sha256}}), that ran one script under one
    recorded code SHA-256."""
    keys = list(meta)
    return [(a, b) for a, b in itertools.combinations(keys, 2)
            if meta[a].get("code_sha256") and meta[a].get("code_sha256") == meta[b].get("code_sha256")
            and meta[a].get("script") == meta[b].get("script")]


def rerun_criteria(run_pp, run_s, run_v, ref_pp, ref_s, ref_v, tol):
    """eval_g3_diag rerun_check's criteria on a run's all14 values against the reference report's: |stride MAE
    difference| within tol['stride_mae_cm'], |speed MAE difference| within tol['speed_mae_ms'] and every person's
    stride MAE within tol['person_stride_cm'] (a person missing on one side fails, missing on both agrees). Not
    evaluated when the run lacks a reference person (a dev12-only run)."""
    lacking = sorted(p for p in ref_pp if run_pp.get(p) is None)
    if lacking:
        return {"evaluated": False, "passed": None, "reason": f"{len(lacking)} persons of the reference not held "
                f"({', '.join(short(p) for p in lacking)})"}
    persons = sorted(set(run_pp) | set(ref_pp))
    d = {p: (run_pp[p] - ref_pp[p]) if run_pp.get(p) is not None and ref_pp.get(p) is not None else None
         for p in persons if not (run_pp.get(p) is None and ref_pp.get(p) is None)}
    ds = None if run_s is None or ref_s is None else run_s - ref_s
    dv = None if run_v is None or ref_v is None else run_v - ref_v
    bad = sorted(p for p, x in d.items() if x is None or abs(x) > tol["person_stride_cm"])
    fails = []
    if ds is None or abs(ds) > tol["stride_mae_cm"]:
        fails.append("stride")
    if dv is None or abs(dv) > tol["speed_mae_ms"]:
        fails.append("speed")
    if bad:
        fails.append("person")
    md = max((abs(x) for x in d.values() if x is not None), default=None)
    return {"evaluated": True, "passed": not fails, "failed": fails, "stride_difference": ds, "speed_difference": dv,
            "persons_beyond": {p: d[p] for p in bad}, "max_person_difference": md}


def position(x, others):
    """The value x among `others`: their mean, SD (ddof 1), min and max, x's z and whether x lies inside their range."""
    o = np.asarray(others, dtype=np.float64)
    m, s = float(o.mean()), float(o.std(ddof=1))
    return {"value": float(x), "n_others": int(len(o)), "mean": m, "sd": s, "min": float(o.min()),
            "max": float(o.max()), "z": (float(x) - m) / s if s > 0 else None,
            "inside_range": bool(o.min() <= x <= o.max())}


def rel_diff(a, b):
    """{n, max, median} of |a - b| / |b| over the keys both mappings hold."""
    ks = sorted(set(a) & set(b))
    r = np.array([abs(a[k] - b[k]) / abs(b[k]) for k in ks if b[k]], dtype=np.float64)
    return {"n": int(len(r)), "max": float(r.max()) if len(r) else None,
            "median": float(np.median(r)) if len(r) else None}


def slide_margin(g_ref, g_a5, follow=0.5):
    """SLIDE's arithmetic against one reference: the base and legs rises of the 5a gains over the reference's, the
    margin legs rise minus `follow` times the base rise (negative: below half), and the legs over base rise ratio."""
    rb, rl = g_a5["base"] - g_ref["base"], g_a5[LEGS_ROW] - g_ref[LEGS_ROW]
    return {"base_rise": rb, "legs_rise": rl, "margin": rl - follow * rb, "legs_over_base": rl / rb if rb else None}


def snap(v, denom):
    """v as the Fraction k / denom nearest to it; HarnessDefect when v lies further than 1e-6 from it."""
    f = Fraction(int(round(float(v) * denom)), denom)
    if abs(float(f) - float(v)) > 1e-6:
        raise HarnessDefect(f"{v} is not on the 1/{denom} grid")
    return f


def range_of(values):
    """{min, max} of the finite values, None when there is none."""
    v = [float(x) for x in values if fin(x) is not None]
    return {"min": min(v), "max": max(v)} if v else None


def ic_arithmetic(ref_pp, a5_pp, s2_line, share=Fraction(1, 20), multiple=2):
    """The walkway IC line in exact arithmetic: the per-person medians lie on the 0.5 ms grid (the median of an even
    count of 10 ms errors), the line is their mean, the margin max(share x v1's line, multiple x d) with d = |v1's line
    minus s2's line| (lever 4's non-inferiority margin); the 5a minus v1 difference and the per-person moves."""
    persons = sorted(ref_pp)
    if set(a5_pp) != set(persons):
        raise HarnessDefect("the IC persons of the two reports differ")
    n = len(persons)
    r = {p: snap(ref_pp[p], 2) for p in persons}
    a = {p: snap(a5_pp[p], 2) for p in persons}
    line_r, line_a = sum(r.values()) / n, sum(a.values()) / n
    d = abs(line_r - snap(s2_line, 2 * n))
    margin = max(share * line_r, multiple * d)
    diff = line_a - line_r
    allv = list(r.values()) + list(a.values())
    den = 1
    for x in allv:
        den = den * x.denominator // math.gcd(den, x.denominator)
    g = 0
    for x in allv:
        g = math.gcd(g, int(x * den))
    return {"n_persons": n, "grid_ms": Fraction(g, den), "sum_v1_ms": sum(r.values()), "sum_a5_ms": sum(a.values()),
            "line_v1": line_r,
            "line_a5": line_a, "seed_d": d, "share": share, "multiple": multiple, "share_of_line": share * line_r,
            "multiple_of_d": multiple * d,
            "margin": margin, "difference": diff, "difference_equals_margin": diff == margin,
            "moves_ms": {p: a[p] - r[p] for p in persons if a[p] != r[p]}}


def bar_compare(j):
    """The anchored body and the ZUPT on the same strides (frame with person_id, x walkway cm, y body cm, z ZUPT cm):
    per-person mean absolute errors and their means over persons, the share of strides each is closer on, ties, the
    median of body over ZUPT."""
    eb, ez = (j["y"] - j["x"]).abs(), (j["z"] - j["x"]).abs()
    pb = eb.groupby(j["person_id"]).mean()
    pz = ez.groupby(j["person_id"]).mean()
    return {"strides": int(len(j)), "persons": int(j["person_id"].nunique()),
            "body_mae_person_mean_cm": float(pb.mean()), "zupt_mae_person_mean_cm": float(pz.mean()),
            "share_body_closer": float((eb < ez).mean()), "share_zupt_closer": float((ez < eb).mean()),
            "ties": int((eb == ez).sum()), "body_over_zupt_median": float(np.median(j["y"] / j["z"])),
            "per_person": {str(p): {"body_cm": float(pb[p]), "zupt_cm": float(pz[p])} for p in pb.index}}


def foot_report_pairs(a5, off, persons):
    """The foot report of the 5a run against the code control, paired per target (trial_id, window, foot, t_a, t_b to
    the millisecond): the added displacement dD = D_a5 - D_off between the still instants, the added slide over the
    measured still run on the targets whose run lies inside the window in both, and the slide share of dD; pooled over
    the targets and per person (ratio of the person means, mean of the person ratios); D over S and the slide share of
    S beside. a5, off: frames of zupt_strides.csv rows of one row."""
    key = ["trial_id", "window", "foot", "ka", "kb"]
    fr = []
    for z in (a5, off):
        z = z[z["person_id"].isin(set(persons))].copy()
        z["ka"], z["kb"] = z["t_a"].astype(float).round(3), z["t_b"].astype(float).round(3)
        z["inside"] = z["run_inside"].astype(str).str.strip().str.lower() == "true"
        fr.append(z)
    m = fr[0].merge(fr[1], on=key, suffixes=("_a", "_o"), how="inner")
    if len(m) != len(fr[0]) or len(m) != len(fr[1]):
        raise HarnessDefect(f"the foot reports do not pair one to one ({len(fr[0])}, {len(fr[1])}, {len(m)} paired)")
    if not np.allclose(m["S_a"].astype(float), m["S_o"].astype(float), rtol=0, atol=1e-12):
        raise HarnessDefect("the ZUPT targets differ between the 5a run and the code control")
    m["dD"] = m["D_a"].astype(float) - m["D_o"].astype(float)
    both = m["inside_a"] & m["inside_o"]
    m["dslide"] = np.where(both, m["slide_m_a"].astype(float) - m["slide_m_o"].astype(float), np.nan)
    mb = m[both]
    per = {}
    for pid, g in m.groupby("person_id_a"):
        gb = g[g["inside_a"] & g["inside_o"]]
        dd, ds = float(g["dD"].mean()), float(gb["dslide"].mean()) if len(gb) else None
        per[str(pid)] = {"targets": int(len(g)), "dD_cm": 100.0 * dd, "dslide_cm": None if ds is None else 100.0 * ds,
                         "slide_part_of_dD": None if ds is None or not dd else ds / dd,
                         "D_over_S_off": float((g["D_o"] / g["S_o"]).mean()),
                         "D_over_S_a5": float((g["D_a"] / g["S_a"]).mean())}
    pdd = np.array([x["dD_cm"] for x in per.values()])
    pds = np.array([x["dslide_cm"] for x in per.values() if x["dslide_cm"] is not None])
    prt = np.array([x["slide_part_of_dD"] for x in per.values() if x["slide_part_of_dD"] is not None])
    return {"targets": int(len(m)), "targets_inside_both": int(both.sum()), "persons": len(per),
            "mean_S_m": float(m["S_a"].astype(float).mean()), "mean_D_off_m": float(m["D_o"].astype(float).mean()),
            "mean_D_a5_m": float(m["D_a"].astype(float).mean()), "pooled_dD_m": float(m["dD"].mean()),
            "pooled_dD_inside_m": float(mb["dD"].mean()), "pooled_dslide_m": float(mb["dslide"].mean()),
            "pooled_slide_part_of_dD": float(mb["dslide"].mean() / mb["dD"].mean()),
            "slide_share_of_S_off": float((mb["slide_m_o"] / mb["S_o"]).mean()),
            "slide_share_of_S_a5": float((mb["slide_m_a"] / mb["S_a"]).mean()),
            "person_mean_dD_cm": float(pdd.mean()), "person_mean_dslide_cm": float(pds.mean()),
            "person_ratio_of_means": float(pds.mean() / pdd.mean()), "person_mean_of_ratios": float(prt.mean()),
            "per_person": per}


def set_aside_reading(res, rc, tol):
    """The rule's verdict function (scripts/f2_lever5a.py verdict) on the existing R1, R2 and T6 of results.json `res`
    with R0's per-person item set aside: asserted first that the code control's rerun check `rc` failed on that item
    alone (stride and speed MAE within their tolerances, no inference difference) and that every other R0 check
    holds."""
    r0 = res["R0"]
    if r0["holds"]:
        raise HarnessDefect("R0 holds: nothing to set aside")
    if not (rc.get("evaluated") and rc.get("passed") is False):
        raise HarnessDefect("the code control's rerun check is not a failed, evaluated check")
    if abs(rc["stride_mae_cm"]["difference"]) > tol["stride_mae_cm"] or \
            abs(rc["speed_mae_ms"]["difference"]) > tol["speed_mae_ms"] or rc.get("inference_differences"):
        raise HarnessDefect("the code control's rerun check failed on more than its per-person item")
    other = {k: x for k, x in r0["checks"].items() if k != "code_control_rerun_passed"}
    if not all(other.values()):
        raise HarnessDefect(f"another R0 check fails: {other}")
    item = [f for f in r0["fail"] if "rerun check" in f]
    if len(item) != 1 or "person stride MAE" not in item[0]:
        raise HarnessDefect(f"R0's failures are not the per-person item alone: {r0['fail']}")
    rest = [f for f in r0["fail"] if f not in item]
    v = LA.verdict({"fail": rest}, res["R1"], res["R2"], res["T6"])
    return {"set_aside": item[0], "other_R0_checks": other, "verdict": v}


def gset_from_pp(S, V, sets, source):
    """A G3 set in the form of weekend_branch.g3set_from_diag_row from per-person S (cm) and V (m/s): the 14-person
    subset when every dev14 person holds S and V, the 12-person subset when every dev12 person does."""
    wb = LA.WB()
    subs = {}
    for sub, persons in ((wb.SUB14, sets["dev14"]), (wb.SUB12, sets["dev12"])):
        if all(S.get(p) is not None and V.get(p) is not None for p in persons):
            s = {p: S[p] for p in persons}
            v = {p: V[p] for p in persons}
            subs[sub] = {"S": s, "V": v, "S_point": float(np.mean(list(s.values()))),
                         "V_point": float(np.mean(list(v.values()))), "n_persons": len(persons)}
    return {"source": source, "kind": "pp", "subsets": subs}


# ---------------------------------------------------------------- readers of the runs (live tree)
def sensitive(name):
    return any(w in name.lower() for w in SENSITIVE)


def find_d1_jsons(tree):
    """Every *.json under reports/ (AppleDouble files, this note's folder and any folder or file whose name holds
    lockbox, cody or care_pd left out; a file is parsed only when its text holds the checkpoint's SHA-256) whose top-level
    checkpoint is v1's B end, classified by its inference block: 'found' when it records at least one setting of
    INFERENCE_KEYS and every recorded setting (weights included) equals REF_INFERENCE ({rel path: [keys not
    recorded]}); 'no_settings' when it records none of INFERENCE_KEYS and no other weights; 'other_settings' when a
    recorded setting differs. 'walked' counts the JSON files opened, 'pruned' the folders left out by name."""
    out = {"found": {}, "no_settings": [], "other_settings": [], "walked": 0, "pruned": []}
    root = os.path.join(tree, "reports")
    note = os.path.normcase(os.path.abspath(os.path.join(tree, NOTE_REL)))
    for dp, dn, fn in os.walk(root):
        keep = []
        for d in sorted(dn):
            if d.startswith("._") or os.path.normcase(os.path.abspath(os.path.join(dp, d))) == note:
                continue
            if sensitive(d):
                out["pruned"].append(rel(os.path.join(dp, d), tree))
                continue
            keep.append(d)
        dn[:] = keep
        for f in sorted(fn):
            if f.startswith("._") or not f.endswith(".json") or sensitive(f):
                continue
            p = os.path.join(dp, f)
            out["walked"] += 1
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            if V1_SHA not in text:
                continue
            try:
                doc = json.loads(text)
            except ValueError:
                continue
            if not isinstance(doc, dict):
                continue
            ck = doc.get("checkpoint")
            if not isinstance(ck, dict) or ck.get("sha256") != V1_SHA:
                continue
            inf = doc.get("inference") if isinstance(doc.get("inference"), dict) else {}
            w = ck.get("weights") or inf.get("weights")
            inf = dict(inf, weights=w) if w is not None else inf
            r = rel(p, tree)
            if any(k in inf and inf[k] != x for k, x in REF_INFERENCE.items()):
                out["other_settings"].append(r)
            elif not any(k in inf for k in INFERENCE_KEYS):
                out["no_settings"].append(r)
            else:
                out["found"][r] = [k for k in REF_INFERENCE if k not in inf]
    return out


def run_meta(tree, run):
    """The run's own record: its JSON's checkpoint and settings (HarnessDefect on a departure), its provenance (run_at,
    script, code SHA-256, MoveAtlas head)."""
    p = os.path.join(tree, run["folder"], run["meta"])
    doc = read_json(p)
    if doc is None:
        raise HarnessDefect(f"missing: {rel(p, tree)}")
    ck = doc.get("checkpoint") or {}
    inf = doc.get("inference") or {}
    if ck.get("sha256") != V1_SHA:
        raise HarnessDefect(f"{rel(p, tree)}: checkpoint {str(ck.get('sha256'))[:12]}, expected {V1_SHA[:12]}")
    w = ck.get("weights") or inf.get("weights")
    if w != "ema":
        raise HarnessDefect(f"{rel(p, tree)}: {w} weights")
    seen = dict(inf, weights=w)
    bad = {k: seen.get(k) for k, x in REF_INFERENCE.items() if k in seen and seen[k] != x}
    if bad:
        raise HarnessDefect(f"{rel(p, tree)}: settings {bad} differ from D1's {REF_INFERENCE}")
    za = inf.get("zupt_anchor")
    if za is not None and not (run["key"] == CONTROL and za.get("mode") == "report" and not za.get("in_objective")):
        raise HarnessDefect(f"{rel(p, tree)}: the lever-5a option recorded ({za.get('mode')})")
    for k, x in (run.get("pass_check") or {}).items():
        got = ((doc.get("passes") or {}).get(run["row"]) or {}).get(k)
        if got != x:
            raise HarnessDefect(f"{rel(p, tree)}: pass {run['row']} has {k} {got}, expected {x}")
    pv = read_json(os.path.join(tree, run["folder"], run["prov"])) or {}
    ms = pv.get("moveatlas_stamp") or {}
    return {"key": run["key"], "folder": run["folder"].replace("\\", "/"), "file": rel(p, tree), "kind": run["kind"],
            "row": run["row"], "what": run["what"], "run_at": pv.get("run_at"), "script": pv.get("script"),
            "code_sha256": pv.get("code_sha256"), "moveatlas_head": ms.get("head") or pv.get("moveatlas_head"),
            "not_recorded": [k for k in REF_INFERENCE if k not in seen], "zupt_mode": (za or {}).get("mode")}


def _events_values(tree, run, sets, locked):
    import pandas as pd
    path = os.path.join(tree, run["folder"], run["events"])
    st, sp, _ = LA.GG().read_events(path, None, locked, row=run["row"])
    st = st[st["person_id"].isin(set(sets["dev14"]))]
    sp = sp[sp["person_id"].isin(set(sets["dev14"]))]
    return person_mae(pd.DataFrame(st)), person_mae(pd.DataFrame(sp))


def run_values(tree, run, sets, locked):
    """{S, V: {dev14 person: stride MAE cm, speed MAE m/s}, S14, V14 (the all14 means, None for a dev12-only run)} of a
    run: per_person.csv (row dev, predictor model) for a dev report, the base row of diag.json for a diag run (asserted
    equal to its events.csv), the matched events of the run's row for the others."""
    import csv
    dev14 = set(sets["dev14"])
    if run["kind"] == "dev":
        S, V = {}, {}
        with open(os.path.join(tree, run["folder"], "per_person.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["row"] == "dev" and r["predictor"] == "model":
                    if r["line"] == "stride_mae_cm":
                        S[r["person_id"]] = fin(r["value"])
                    elif r["line"] == "speed_mae_m_s":
                        V[r["person_id"]] = fin(r["value"])
        doc = read_json(os.path.join(tree, run["folder"], run["meta"]))
        model = ((doc.get("lines") or {}).get("dev") or {}).get("model") or {}
        s14, v14 = fin((model.get("stride_mae_cm") or {}).get("point")), fin((model.get("speed_mae_m_s") or {}).get(
            "point"))
    elif run["kind"] == "diag":
        doc = read_json(os.path.join(tree, run["folder"], run["meta"]))
        sub = doc["rows"]["base"]["all14"]
        S = {p: fin(x.get("stride_cm")) for p, x in sub["per_person"].items()}
        V = {p: fin(x.get("speed_ms")) for p, x in sub["per_person"].items()}
        s14, v14 = fin(sub["stride_mae_cm"]), fin(sub["speed_mae_ms"])
        es, ev = _events_values(tree, run, sets, locked)
        if set(es) != set(S) or max(abs(es[p] - S[p]) for p in S) > 1e-9 or max(abs(ev[p] - V[p]) for p in V) > 1e-9:
            raise HarnessDefect(f"{run['folder']}: events.csv does not reproduce the diag's per-person values")
    else:
        S, V = _events_values(tree, run, sets, locked)
        s14 = float(np.mean([S[p] for p in sets["dev14"]])) if all(p in S for p in sets["dev14"]) else None
        v14 = float(np.mean([V[p] for p in sets["dev14"]])) if all(p in V for p in sets["dev14"]) else None
    out14 = set(S) | set(V)
    if out14 - dev14 or out14 & set(locked):
        raise HarnessDefect(f"{run['folder']}: a person outside dev14")
    return {"S": S, "V": V, "S14": s14, "V14": v14}


def run_trials(tree, run, sets):
    """{trial_id: (refine_first, refine_last)} of the dev14 trials of the run's trials.csv (its row or pass), None when
    the file holds no refine columns."""
    import pandas as pd
    p = os.path.join(tree, run["folder"], "trials.csv")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p, encoding="utf-8", dtype={"trial_id": str, "person_id": str})
    if not {"refine_first", "refine_last"} <= set(t.columns):
        return None
    if "pass" in t.columns:
        t = t[t["pass"] == run["row"]]
    elif "row" in t.columns and run["kind"] == "diag":
        t = t[t["row"] == run["row"]]
    elif "row" in t.columns:
        t = t[t["row"] == "dev"]
    t = t[t["person_id"].isin(set(sets["dev14"]))]
    return {str(r.trial_id): (float(r.refine_first), float(r.refine_last)) for r in t.itertuples()}


def stride_gains(tree, folder, events, sets, locked, base_row="base"):
    """{base, model_legs: the dev12 stride gain (F1's median form) of the matched strides of the rows base_row and
    model_legs, None when the row is held by fewer than MIN_ROW_PERSONS dev14 persons}, the events read with the
    training-fold rows dropped."""
    gg = LA.GG()
    p = os.path.join(tree, folder, events)
    out = {}
    for key, row in (("base", base_row), (LEGS_ROW, LEGS_ROW)):
        st, _, _ = gg.read_events(p, None, locked, row=row)
        st = st[st["person_id"].isin(set(sets["dev14"]))]
        if st["person_id"].nunique() < MIN_ROW_PERSONS:
            out[key] = None
            continue
        s12 = gg.subset(st, sets["dev12"])
        out[key] = fin(gg.median_ratio(s12["x"], s12["y"]))
    return out


def ic_per_person(tree, folder):
    import csv
    out = {}
    with open(os.path.join(tree, folder, "per_person.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["row"] == "dev" and r["predictor"] == "model" and r["line"] == "ic_median_ms_walkway":
                out[r["person_id"]] = fin(r["value"])
    return out


def bar_join(tree, sets, locked):
    """The matched base-row dev12 strides of the 5a diag joined to the matched dev12 strides of the version-2 ZUPT
    readout on trial, foot and the two walkway instants (scripts/f2_lever5.py decoded_over_zupt's join): person_id,
    x (walkway cm), y (body cm), z (ZUPT cm)."""
    import pandas as pd
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(os.path.join(tree, LA.OUT["a5_diag"], "events.csv"), encoding="utf-8", dtype=text,
                     keep_default_na=False)
    ev = ev[(ev["row"] == "base") & (ev["kind"] == "stride") & (ev["matched"].str.strip().str.lower() == "true")
            & ev["person_id"].isin(set(sets["dev12"]))]
    if set(ev["person_id"]) & set(locked):
        raise HarnessDefect("a lockbox person in the events")
    ev = ev.assign(t0=pd.to_numeric(ev["t0_s"]).round(2), t1=pd.to_numeric(ev["t1_s"]).round(2),
                   y=pd.to_numeric(ev["prediction"]), x=pd.to_numeric(ev["reference"]))
    z = pd.read_csv(os.path.join(tree, ZUPT_V2_STRIDES), encoding="utf-8")
    z = z[z["person_id"].isin(set(sets["dev12"])) & (z["matched"].astype(str).str.lower() == "true")].copy()
    z["t0"], z["t1"] = z["t0"].round(2), z["t1"].round(2)
    j = ev.merge(z[["trial_id", "foot", "t0", "t1", "zupt_m", "walkway_m"]], on=["trial_id", "foot", "t0", "t1"],
                 how="inner")
    j = j.assign(z=100.0 * j["zupt_m"])
    if float((j["x"] - 100.0 * j["walkway_m"]).abs().max()) > 1e-6:
        raise HarnessDefect("the walkway stride of the events and of the ZUPT readout differ on a joined stride")
    return j[["person_id", "trial_id", "foot", "x", "y", "z"]].reset_index(drop=True)


def foot_frames(tree):
    import pandas as pd
    out = []
    for key in ("a5_diag", "off_diag"):
        z = pd.read_csv(os.path.join(tree, LA.OUT[key], "zupt_strides.csv"), encoding="utf-8",
                        dtype={"person_id": str, "trial_id": str, "row": str, "foot": str})
        out.append(z[z["row"] == "base"])
    return out


# ---------------------------------------------------------------- the note
def _pp_rows(meta, values, persons):
    return {k: {p: values[k]["S"].get(p) for p in persons if values[k]["S"].get(p) is not None} for k in meta}


def item_a(tree, sets, locked, meta, values, tol):
    """R0: the runs, their spread, pairs, same-code pairs, rerun checks, the control's position, the objective."""
    persons = sets["dev14"]
    S = _pp_rows(meta, values, persons)
    sp = spread(S, persons)
    prs = pairs(S, tol["person_stride_cm"], FOCUS)
    counts = pair_counts(prs, len(persons), tol["person_stride_cm"])
    sc = []
    by = {(x["a"], x["b"]): x for x in prs}
    for a, b in same_code_pairs({k: meta[k] for k in meta}):
        x = by[(a, b)]
        sc.append({"a": a, "b": b, "script": meta[a]["script"], "code_sha256": meta[a]["code_sha256"],
                   "run_at": [meta[a]["run_at"], meta[b]["run_at"]], "n_persons": x["n_persons"],
                   "max_abs": x["max_abs"], "argmax": x["argmax"], "beyond": x["beyond"], "focus": x["focus"]})
    ref = values[DEV_REF]
    checks, stored = {}, {}
    for k in meta:
        if k == DEV_REF:
            continue
        v = values[k]
        checks[k] = rerun_criteria(v["S"], v["S14"], v["V14"], ref["S"], ref["S14"], ref["V14"], tol)
        if meta[k]["kind"] == "diag":
            rc = (read_json(os.path.join(tree, meta[k]["folder"], "diag.json")) or {}).get("rerun_check") or {}
            if rc.get("evaluated"):
                same = (bool(rc.get("passed")) == checks[k]["passed"] and abs(
                    rc["max_person_stride_difference_cm"] - checks[k]["max_person_difference"]) < 1e-9)
                if not same:
                    raise HarnessDefect(f"{k}: the reproduced rerun check differs from the stored one")
                stored[k] = {"passed": bool(rc.get("passed")), "reading": rc.get("reading")}
    ev = [k for k, c in checks.items() if c["evaluated"]]
    others = [S[k][FOCUS] for k in meta if k != CONTROL and FOCUS in S[k]]
    pos = position(S[CONTROL][FOCUS], others)
    rank = 1 + sum(1 for x in others if x < S[CONTROL][FOCUS])
    trials = {k: run_trials(tree, r, sets) for k, r in ((r["key"], r) for r in V1_RUNS)}
    obj = {}
    comps = [(CONTROL, RULE_REF), (CONTROL, DEV_REF)] + [(a, b) for a, b in same_code_pairs(meta)]
    for a, b in comps:
        if trials.get(a) is None or trials.get(b) is None:
            continue
        ta, tb = trials[a], trials[b]
        obj[f"{a} vs {b}"] = {"same_code": (a, b) in same_code_pairs(meta),
                              "refine_first": rel_diff({t: x[0] for t, x in ta.items()},
                                                       {t: x[0] for t, x in tb.items()}),
                              "refine_last": rel_diff({t: x[1] for t, x in ta.items()},
                                                      {t: x[1] for t, x in tb.items()})}
    sc_keys = [f"{a} vs {b}" for a, b in same_code_pairs(meta) if f"{a} vs {b}" in obj]
    sc_first = [obj[k]["refine_first"]["max"] for k in sc_keys]
    sc_last = [obj[k]["refine_last"]["max"] for k in sc_keys]
    passed = sum(1 for k in ev if checks[k]["passed"])
    failed = [k for k in ev if not checks[k]["passed"]]
    n_sc_beyond = sum(1 for x in sc if x["beyond"])
    ctrl_first = (obj.get(f"{CONTROL} vs {RULE_REF}") or {}).get("refine_first", {}).get("max")
    conditions = {"same_code_pairs": len(sc), "same_code_pairs_with_a_person_beyond": n_sc_beyond,
                  "code_control_focus_inside_the_others_range": pos["inside_range"],
                  "runs_failing_the_check": len(failed), "code_control_fails": CONTROL in failed,
                  "control_refine_first_within_same_code": bool(sc_first and ctrl_first is not None
                                                                and ctrl_first <= max(sc_first))}
    if not (n_sc_beyond and pos["inside_range"] and CONTROL in failed and len(failed) > 1):
        raise HarnessDefect(f"the numbers do not show the per-person tolerance badly posed ({conditions}): the "
                            "reading of this note does not follow")
    if not conditions["control_refine_first_within_same_code"]:
        raise HarnessDefect("the control's refinement objective does not agree with the reference's as closely as the "
                            "same-code pairs agree: the caveat does not follow")
    return {"label": "R0 item: the run-to-run spread of v1's B end at the D1 settings (dev diagnostic, not a G3 read)",
            "runs": [dict(meta[k], S14=values[k]["S14"], V14=values[k]["V14"]) for k in meta],
            "per_person_stride_cm": S, "spread": sp, "pairs": prs, "pair_counts": counts,
            "same_code_pairs": sc, "same_code_pairs_with_a_person_beyond": n_sc_beyond,
            "rerun_check_against_the_dev_report": {"tolerances": tol, "runs": checks, "stored_checks_agree": stored,
                                                   "stored_passed": sum(1 for x in stored.values() if x["passed"]),
                                                   "stored_failed": sum(1 for x in stored.values() if not x["passed"]),
                                                   "evaluated": len(ev), "passed": passed, "failed": failed},
            "code_control_focus_position": dict(pos, person=FOCUS, rank_from_lowest=rank),
            "objective": {"comparisons": obj,
                          "same_code_refine_first_max": max(sc_first) if sc_first else None,
                          "same_code_refine_first_max_range": range_of(sc_first),
                          "same_code_refine_last_max_range": range_of(sc_last)},
            "reading_conditions": conditions,
            "reading": ("badly posed: the per-person tolerance is inside the run-to-run spread of one checkpoint "
                        f"({n_sc_beyond} of the {len(sc)} same-code pairs break it, {len(failed)} of the {len(ev)} "
                        "runs holding every person fail it against v1's own dev report, the code control's "
                        f"{short(FOCUS)} lies inside the other runs' range)"),
            "caveat": ("the GPU identity, bit for bit, of the report mode with the off path is not shown: the report "
                       "mode cannot touch the objective (mdfm/eval/zupt_refine.py TrialZupt.objective returns None in "
                       "mode 'report'; tests/test_zupt_refine.py finds report equal to off bit for bit on the CPU), "
                       "and the control's refinement objective at the first call agrees with the reference's as "
                       "closely as same-code reruns agree with each other")}


def item_b(tree, sets, locked, res, meta, values, tol):
    """Both readings: as written; R0's per-person item set aside; the rule's lines with each v1 run as the reference."""
    off = read_json(os.path.join(tree, LA.OUT["off_diag"], "diag.json"))
    aside = set_aside_reading(res, off.get("rerun_check") or {}, tol)
    m = res["margins"]
    a5, why = LA.diag_set_at(tree, LA.OUT["a5_diag"], sets, locked, V1_SHA)
    if a5 is None:
        raise HarnessDefect(f"the 5a diag base row: {why}")
    wb = LA.WB()
    g_a5 = stride_gains(tree, LA.OUT["a5_diag"], "events.csv", sets, locked)
    if any(abs(g_a5[r] - res["R2"]["SLIDE"]["a5"][r]) > 1e-12 for r in ("base", LEGS_ROW)):
        raise HarnessDefect("the 5a gains differ from the rule's reading")
    t = m["transfer"]
    d_base, d_legs = t["d_r"]["base"], t["d_legs"]
    per = {}
    for run in V1_RUNS:
        k = run["key"]
        v = values[k]
        gset = gset_from_pp(v["S"], v["V"], sets, k)
        r1 = LA.r1_reading(gset, a5, m, f"{k} holds no dev12 S and V" if wb.SUB12 not in gset["subsets"] else None)
        vals_ref = {}
        if wb.SUB14 in gset["subsets"]:
            vals_ref = {"S14": gset["subsets"][wb.SUB14]["S_point"], "V14": gset["subsets"][wb.SUB14]["V_point"]}
        reasons = {x: f"{k} holds no all14 value" for x in ("S14", "V14") if x not in vals_ref}
        if run["kind"] == "dev":
            for x in ("IC", "MAPE"):
                vals_ref[x], _ = LA.dev_line_at(tree, run["folder"], LA.G3_LINE_OF[x], V1_SHA)
            ic_src = k
        else:
            vals_ref["IC"], vals_ref["MAPE"] = res["R2"]["IC"]["v1"], res["R2"]["MAPE"]["v1"]
            ic_src = DEV_REF
        vals_a5 = {x: res["R2"][x]["a5"] for x in ("S14", "V14", "IC", "MAPE")}
        if run["events"] is not None and run["kind"] in ("diag", "events"):
            g = stride_gains(tree, run["folder"], run["events"], sets, locked, run["row"])
        else:
            g = {"base": None, LEGS_ROW: None}
        if g[LEGS_ROW] is None:
            slide = LA.slide_reading(None, g_a5, d_base, d_legs, f"{k} holds no {LEGS_ROW} row")
            mg = None
        else:
            slide = LA.slide_reading(g, g_a5, d_base, d_legs)
            mg = slide_margin(g, g_a5, LA.L5().FOLLOW_SHARE)
            if run["kind"] == "diag":
                gl, _ = LA.gains_at(tree, run["folder"], sets, locked)
                if any(abs(gl[r] - g[r]) > 1e-12 for r in ("base", LEGS_ROW)):
                    raise HarnessDefect(f"{k}: the gains differ from scripts/f2_lever5a.py gains_at")
        r2 = LA.r2_reading(vals_ref, vals_a5, m, reasons, slide)
        v_ = LA.verdict({"fail": []}, r1, r2, res["T6"])
        per[k] = {"run_at": meta[k]["run_at"], "R1": r1, "R2": r2, "SLIDE_margin": mg, "IC_MAPE_reference": ic_src,
                  "gains": g, "verdict": {"verdict": v_["verdict"], "case": v_["case"], "reasons": v_["reasons"]},
                  "label": ROW_LABEL}
    # the reproduction of the rule's own reading
    rr = per[RULE_REF]
    for L in LA.PRIMARY:
        if abs(rr["R1"][L]["improvement"]["point"] - res["R1"][L]["improvement"]["point"]) > 1e-9 or \
                rr["R1"][L]["improvement"]["interval"] != res["R1"][L]["improvement"]["interval"]:
            raise HarnessDefect(f"R1 {L} against the reference does not reproduce results.json")
    for x in ("S14", "V14", "IC", "MAPE"):
        if abs(rr["R2"][x]["difference"] - res["R2"][x]["difference"]) > 1e-9:
            raise HarnessDefect(f"R2 {x} against the reference does not reproduce results.json")
    if abs(rr["R2"]["SLIDE"]["base_rise"] - res["R2"]["SLIDE"]["base_rise"]) > 1e-12 or \
            rr["R2"]["SLIDE"]["status"] != res["R2"]["SLIDE"]["status"]:
        raise HarnessDefect("SLIDE against the reference does not reproduce results.json")
    cc = res["reported"]["against_the_code_control"]["SLIDE"]
    if per[CONTROL]["R2"]["SLIDE"]["status"] != cc["status"]:
        raise HarnessDefect("SLIDE against the code control does not reproduce results.json")
    if rr["verdict"]["verdict"] != aside["verdict"]["verdict"]:
        raise HarnessDefect("the reading against the reference differs from the set-aside reading")
    slide_refs = [k for k in per if per[k]["SLIDE_margin"] is not None]
    others = [k for k in slide_refs if k != RULE_REF]
    lines = {"R1_meets": {L: res["R1"][L]["meets"] for L in LA.PRIMARY},
             "R2_holds": {k: res["R2"][k]["holds"] for k in LA.R2_LINES},
             "IC_difference_minus_margin": res["R2"]["IC"]["difference"] - res["R2"]["IC"]["margin"],
             "SLIDE": res["R2"]["SLIDE"]["status"], "T6": res["T6"]["status"]}
    go_against = [k for k in per if per[k]["verdict"]["verdict"] == LA.GO]
    r1_ranges = {}
    for L in LA.PRIMARY:
        imp = [per[k]["R1"][L].get("improvement") or {} for k in per]
        r1_ranges[L] = {"point": range_of([x.get("point") for x in imp]),
                        "lower_bound_min": min(x["interval"][0] for x in imp if x.get("interval"))}
    without_legs = [k for k in per if per[k]["SLIDE_margin"] is None]
    counts = {}
    for k in without_legs:
        key = f"{per[k]['verdict']['verdict']} (case {per[k]['verdict']['case']})"
        counts[key] = counts.get(key, 0) + 1
    asw = aside["verdict"]["verdict"]
    if asw != LA.GO:
        statement = f"the rule's verdict stays {res['verdict']['verdict']}; the set-aside reading is {asw}"
    elif set(go_against) < set(slide_refs):
        statement = (f"the rule's verdict stays {res['verdict']['verdict']}; the set-aside GO holds against "
                     f"{', '.join(go_against) or 'none'} only and is not this rule's result")
    else:
        statement = (f"the rule's verdict stays {res['verdict']['verdict']}; the set-aside GO holds against every "
                     "reference with a model_legs row and is not this rule's result")
    return {"label": "both readings (rule section 10), dev diagnostic, not a G3 read",
            "as_written": {"verdict": res["verdict"]["verdict"], "case": res["verdict"]["case"],
                           "reasons": res["verdict"]["reasons"]},
            "R0_per_person_item_set_aside": dict(aside, lines=lines),
            "against_each_v1_run": per,
            "references_with_slide": slide_refs,
            "go_against": go_against,
            "go_survives_the_reference_spread": bool(go_against) and set(go_against) == set(slide_refs),
            "no_go_slide_against": [k for k in others if per[k]["verdict"]["verdict"] == LA.NO_GO
                                    and "SLIDE FAILS" in per[k]["verdict"]["reasons"]],
            "r1_meets_against": [k for k in per if all(per[k]["R1"][L].get("meets") for L in LA.PRIMARY)],
            "r1_ranges_over_the_references": r1_ranges,
            "without_legs_row": without_legs, "without_legs_row_verdicts": counts,
            "statement": statement,
            "thresholds": {"d_base": d_base, "d_legs": d_legs, "follow_share": LA.L5().FOLLOW_SHARE,
                           "seed_multiple": LA.L5().SEED_MULTIPLE}}


def item_c(tree, res, b, meta):
    """The fragile lines: the walkway IC in exact arithmetic, the IC of the other v1 dev runs; SLIDE per reference and
    s2_B's beside."""
    m = res["margins"]
    ref_ic = ic_per_person(tree, os.path.join(G3, f"dev_{V1_B}"))
    a5_ic = ic_per_person(tree, LA.OUT["a5_dev"])
    ic = ic_arithmetic(ref_ic, a5_ic, m["s2"]["IC"], Fraction(str(LA.F2().OTHER_FRACTION)),
                       int(LA.F2().SEED_MULTIPLE))
    if abs(float(ic["margin"]) - m["noninferiority"]["IC"]["margin"]) > 1e-9 or \
            abs(float(ic["difference"]) - res["R2"]["IC"]["difference"]) > 1e-9:
        raise HarnessDefect("the exact IC arithmetic does not reproduce the rule's margin and difference")
    import pandas as pd
    fps = set(pd.read_csv(os.path.join(tree, LA.OUT["a5_dev"], "trials.csv"), encoding="utf-8")["fps"].astype(float))
    if len(fps) != 1:
        raise HarnessDefect(f"the dev trials hold several frame rates: {sorted(fps)}")
    frame = Fraction(1000) / Fraction(fps.pop()).limit_denominator(1000)
    ic["net_move_ms"] = sum(ic["moves_ms"].values(), Fraction(0))
    ic["frame_ms"] = frame
    ic["net_frames"] = ic["net_move_ms"] / frame
    other_dev = {}
    for run in V1_RUNS:
        if run["kind"] != "dev" or run["key"] == DEV_REF:
            continue
        x = ic_per_person(tree, run["folder"])
        line = sum(snap(v, 2) for v in x.values()) / len(x)
        other_dev[run["key"]] = {"run_at": meta[run["key"]]["run_at"], "line": line,
                                 "minus_reference": line - ic["line_v1"],
                                 "minus_reference_in_margins": (line - ic["line_v1"]) / ic["margin"],
                                 "beyond_the_margin": line - ic["line_v1"] > ic["margin"],
                                 "a5_minus_this": ic["line_a5"] - line,
                                 "moves_ms": {p: snap(x[p], 2) - snap(ref_ic[p], 2) for p in sorted(x)
                                              if snap(x[p], 2) != snap(ref_ic[p], 2)}}
    per = b["against_each_v1_run"]
    sl = {k: dict(per[k]["SLIDE_margin"], status=per[k]["R2"]["SLIDE"]["status"], run_at=per[k]["run_at"])
          for k in b["references_with_slide"]}
    mg = [x["margin"] for x in sl.values()]
    mo = [x["margin"] for k, x in sl.items() if k != RULE_REF]
    lo = [x["legs_over_base"] for x in sl.values()]
    s2 = {}
    sets, locked = LA.load_sets(tree)
    g_s2a5 = stride_gains(tree, LA.OUT["s2_a5_diag"], "events.csv", sets, locked)
    t = m["transfer"]
    for key, folder in (("s2_bw32_reference", LA.REF["s2_diag_bw32"]), ("s2_ablations_bw32", S2_ABL)):
        g = stride_gains(tree, folder, "events.csv", sets, locked)
        s2[key] = dict(slide_margin(g, g_s2a5, LA.L5().FOLLOW_SHARE),
                       status=LA.slide_reading(g, g_s2a5, t["d_r"]["base"], t["d_legs"])["status"])
    held = [k for k, x in sl.items() if x["status"] == "HELD"]
    feet = [k for k, x in sl.items() if x["status"] == "FEET SLIDE"]
    mult = LA.L5().SEED_MULTIPLE
    return {"label": "the fragile lines (dev diagnostic, not a G3 read)",
            "one_5a_run": {"a5_diag": LA.OUT["a5_diag"].replace("\\", "/"), "a5_dev": LA.OUT["a5_dev"].replace(
                "\\", "/"), "note": "one 5a run per report: the arm's own run-to-run spread enters every SLIDE margin "
                                    "and the IC difference and is not measured"},
            "IC": dict(ic, label=ROW_LABEL, other_v1_dev_runs=other_dev),
            "SLIDE": {"per_reference": sl, "mean_margin_all": float(np.mean(mg)),
                      "mean_margin_others": float(np.mean(mo)),
                      "held": held, "feet_slide": feet, "undecided": bool(held and feet),
                      "legs_over_base_mean": float(np.mean(lo)), "legs_over_base_range": range_of(lo),
                      "label": ROW_LABEL, "seed_multiple": mult,
                      "legs_rise_above_multiple_of_d_legs_everywhere": all(x["legs_rise"] > mult * t["d_legs"]
                                                                           for x in sl.values()),
                      "s2_B_replication": s2, "s2_B_label": f"{ROW_LABEL}, s2_B"}}


def item_d(tree, sets, locked, res):
    j = bar_join(tree, sets, locked)
    bc = bar_compare(j)
    g, n = LA.L5().decoded_over_zupt(os.path.join(tree, LA.OUT["a5_diag"], "events.csv"),
                                     os.path.join(tree, ZUPT_V2_STRIDES), sets["dev12"], locked)
    if n != bc["strides"] or abs(g - bc["body_over_zupt_median"]) > 1e-12 or \
            abs(g - res["T6"]["ends"]["a5"]["decoded_over_zupt"]) > 1e-12:
        raise HarnessDefect("the bar's join does not reproduce decoded_over_zupt and T6")
    zd = read_json(os.path.join(tree, LA.ZUPT_DEV_RESULTS))["lines"]["zupt"]["stride_mae_cm"]["sets"]["dev12"]
    z2 = read_json(os.path.join(tree, LA.ZUPT_V2_RESULTS))["baseline_line_dev12"]
    return {"label": "the bar (dev diagnostic, not a G3 read)", "row_label": ROW_LABEL, "same_strides": bc,
            "zupt_v2_walkway_timed_dev12": z2,
            "zupt_g3_matched_label_free_dev12": {k: zd.get(k) for k in ("point", "interval", "n_persons",
                                                                         "matched_share", "n_events")},
            "a5_S12_cm": res["R1"]["S12"]["a5"]}


def item_e(tree, sets, c):
    a5, off = foot_frames(tree)
    fm = foot_report_pairs(a5, off, sets["dev12"])
    sl = c["SLIDE"]
    return {"label": "the foot report (reported beside, descriptive; dev diagnostic, not a G3 read; rule section 8: no "
                     "5a dev number here is confirmatory evidence of a mechanism; the foot report and T6 measure "
                     "convergence to the anchor, not accuracy)", "row_label": ROW_LABEL, "foot_report": fm,
            "legs_over_base": {k: x["legs_over_base"] for k, x in sl["per_reference"].items()},
            "legs_over_base_mean": sl["legs_over_base_mean"], "legs_over_base_range": sl["legs_over_base_range"]}


def proposal(b, c, d):
    """The programme proposal to the owner (beside the rule, outside its section 9 routes), built from the fields it
    rests on; HarnessDefect when they do not support it."""
    s = d["same_strides"]
    sl = c["SLIDE"]
    need = {"the set-aside reading is GO": b["R0_per_person_item_set_aside"]["verdict"]["verdict"] == LA.GO,
            "the set-aside GO does not hold against every reference": not b["go_survives_the_reference_spread"],
            "SLIDE is undecided across the references": sl["undecided"],
            "the ZUPT is closer on most strides": s["share_zupt_closer"] > 0.5,
            "the ZUPT's stride MAE is below the body's": s["zupt_mae_person_mean_cm"] < s["body_mae_person_mean_cm"]}
    if not all(need.values()):
        raise HarnessDefect(f"the proposal does not follow: {need}")
    text = ("Lever 5b (the training run of docs/proposals/F2_LEVER5_DRAFT.md) is not proposed; lever 5a is closed "
            "without a result under this rule: with R0's per-person item set aside its GO holds against "
            f"{', '.join(b['go_against'])} only of the {len(b['references_with_slide'])} v1 runs that hold a "
            "model_legs row, SLIDE's status changes with the v1 run taken as the reference, and on the "
            f"{s['strides']} dev12 strides both match the {ROW_LABEL} reads "
            f"{f2(s['body_mae_person_mean_cm'])} cm against {f2(s['zupt_mae_person_mean_cm'])} cm for the version-2 "
            f"ZUPT of the same foot sensors, which is closer on {f2(100 * s['share_zupt_closer'], 1)} percent of "
            "them; v1's B end stays the reference and no 5a row enters a G3 verdict; for future rules, a rerun "
            "tolerance is calibrated on the run-to-run spread of the reference checkpoint, or the reference is an off "
            "run in the same queue as the arm, under the same code.")
    return {"outside_the_rule": True, "decides": "the owner", "conditions": need,
            "rests_on": ["the set-aside reading (not this rule's result)",
                         "SLIDE with each v1 run in the place of the reference (this note, not the rule)",
                         "the bar against the version-2 ZUPT (rule section 7: a bar beside, reading nothing)"],
            "proposal": text}


def run(tree=None, log=print):
    """Every item of the note; (results, exit code). RuleNotHashed or AssertionError from the rule check first."""
    t0 = time.time()
    tree = tree or TREE
    rule_sha = LA.check_rule(tree)
    sets, locked = LA.load_sets(tree)
    res = read_json(os.path.join(tree, RESULTS_REL))
    if res is None or res.get("rule", {}).get("sha256") != rule_sha:
        raise HarnessDefect("results.json of the rule is missing or of another rule")
    if (res["verdict"]["verdict"], res["verdict"]["case"]) != (LA.NOT_EVALUATED, 1):
        raise HarnessDefect(f"the rule's reading is {res['verdict']['verdict']} (case {res['verdict']['case']}), not "
                            "NOT EVALUATED (case 1): this note is about that reading")
    tol = tol_of()
    walk = find_d1_jsons(tree)
    found = walk["found"]
    meta = {r["key"]: run_meta(tree, r) for r in V1_RUNS}
    listed = {m_["file"] for m_ in meta.values()}
    discovery = {"d1_settings": dict(REF_INFERENCE), "walked_json": walk["walked"], "pruned": walk["pruned"],
                 "found": found, "read": sorted(f for f in found if f in listed),
                 "excluded": {f: EXCLUDED[f] for f in sorted(found) if f in EXCLUDED and f not in listed},
                 "unlisted": sorted(f for f in found if f not in listed and f not in EXCLUDED),
                 "listed_not_found": sorted(f for f in listed if f not in found),
                 "no_settings": walk["no_settings"],
                 "no_settings_noted": {f: w for f, w in NOTED_WITHOUT_SETTINGS.items() if f in walk["no_settings"]},
                 "other_settings": walk["other_settings"]}
    if discovery["listed_not_found"]:
        raise HarnessDefect(f"a run of V1_RUNS is not found at D1 settings: {discovery['listed_not_found']}")
    if set(discovery["no_settings_noted"]) != set(NOTED_WITHOUT_SETTINGS):
        raise HarnessDefect("a JSON of NOTED_WITHOUT_SETTINGS is not among the walk's files without settings")
    values = {r["key"]: run_values(tree, r, sets, locked) for r in V1_RUNS}
    out = {"what": LABEL, "row_label": ROW_LABEL, "task": TASK, "date": NOTE_DATE,
           "rule": {"path": LA.RULE_REL.replace("\\", "/"), "sha256": rule_sha, "section": 10},
           "reading_of_the_rule": {"path": RESULTS_REL.replace("\\", "/"),
                                   "sha256": LA.sha256_file(os.path.join(tree, RESULTS_REL)),
                                   "summary_sha256": LA.sha256_file(os.path.join(tree, SUMMARY_REL)),
                                   "verdict": res["verdict"]["verdict"], "case": res["verdict"]["case"],
                                   "unchanged": True},
           "discovery": discovery}
    out["a"] = item_a(tree, sets, locked, meta, values, tol)
    out["b"] = item_b(tree, sets, locked, res, meta, values, tol)
    out["c"] = item_c(tree, res, out["b"], meta)
    out["d"] = item_d(tree, sets, locked, res)
    out["e"] = item_e(tree, sets, out["c"])
    if LA.route(res["verdict"]) != res["route"]:
        raise HarnessDefect("the route of results.json is not the rule's route of its verdict")
    out["f"] = {"rule_section": 9, "route": res["route"],
                "source": RESULTS_REL.replace("\\", "/") + " (route)"}
    out["g"] = proposal(out["b"], out["c"], out["d"])
    out["sets"] = {"dev14": sorted(sets["dev14"]), "dev12": sorted(sets["dev12"])}
    out["timing_s"] = round(time.time() - t0, 1)
    rc = out["a"]["rerun_check_against_the_dev_report"]
    log(f"note: {len(meta)} runs of v1_B at D1 settings; rerun check passed by {rc['passed']} of {rc['evaluated']}; "
        f"GO against {out['b']['go_against']}")
    return out, EXIT_WRITTEN


# ---------------------------------------------------------------- outputs
def f2(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def sgn(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:+.{nd}f}"


def iv(x, nd=2):
    return "n/a" if not x else f"[{f2(x[0], nd)}, {f2(x[1], nd)}]"


def sci(x):
    v = fin(x)
    return "n/a" if v is None else f"{v:.1e}"


def moves_text(moves):
    return ", ".join(f"{short(p)} {'+' if y > 0 else ''}{frac_text(y)}" for p, y in moves.items()) or "none"


def plus(x):
    return "+" if x >= 0 else ""


def settings_text(s):
    """The D1 settings of a REF_INFERENCE mapping, in words."""
    return (f"refine_steps {s['refine_steps']}, points_per_stream {s['points_per_stream']}, batch_windows "
            f"{s['batch_windows']}, seed {s['seed']}, shape pooled {s['shape_pooling']}, {s['weights']} weights")


def note_md(r):
    a, b, c, d, e = r["a"], r["b"], r["c"], r["d"], r["e"]
    rc = a["rerun_check_against_the_dev_report"]
    pc = a["pair_counts"]
    pos = a["code_control_focus_position"]
    runs = {x["key"]: x for x in a["runs"]}
    keys = list(runs)
    dev14, dev12 = r["sets"]["dev14"], r["sets"]["dev12"]
    n14 = len(dev14)
    fo = short(FOCUS)
    dis = r["discovery"]
    L = [f"# F2 lever 5a: the dated section-10 note of the rule, {r['date']} ({LABEL})", "",
         f"Written by scripts/f2_lever5a_note.py (task {TASK}) under reports/bench/f2_lever5a/rule.txt (SHA-256 "
         f"{r['rule']['sha256']}, asserted before anything was read), section 10: a criterion is never changed after a "
         "5a number is seen, and a criterion that proves badly posed is said so with both readings kept. Every 5a row "
         f"is labelled '{ROW_LABEL}'. No lockbox, EVAL, CODY or care_pd record was read; nothing here is a G3 read or "
         "enters a G3 verdict; every number below is computed by this script from the files it names.", "",
         f"The reading of the rule is unchanged: reports/bench/f2_lever5a/results.json (SHA-256 "
         f"{r['reading_of_the_rule']['sha256']}) and summary.md read **{r['reading_of_the_rule']['verdict']}** (case "
         f"{r['reading_of_the_rule']['case']}). This note changes no criterion, no output and no verdict.", "",
         "## a. R0: the per-person tolerance against the run-to-run spread of v1's B end", "",
         f"Every run of v1's B end ({LA.V1_CKPT}) at the D1 settings ({settings_text(dis['d1_settings'])}) found under "
         f"reports/: the walk opened {dis['walked_json']} JSON files (folders {', '.join(dis['pruned']) or 'none'} "
         "pruned by name); of those naming that checkpoint at their top level, "
         f"{len(dis['found'])} record D1 settings and no other, {len(dis['no_settings'])} record no inference setting "
         f"and {len(dis['other_settings'])} record other settings (each listed in results.json, discovery). Of the "
         f"{len(dis['found'])}: {len(keys)} runs read from {len(dis['read'])} of them, {len(dis['excluded'])} excluded "
         f"with a reason, {len(dis['unlisted'])} unlisted.", "",
         f"| run | folder | date (run_at) | script | code SHA-256 | persons | S({n14}) cm | {fo} cm | rerun check "
         "against the dev report |", "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        x, ck = runs[k], rc["runs"].get(k)
        pp = a["per_person_stride_cm"][k]
        if ck is None:
            st = "the reference"
        elif not ck["evaluated"]:
            st = f"not applicable: {ck['reason']}"
        elif ck["passed"]:
            st = f"passes (largest person {f2(ck['max_person_difference'])} cm)"
        else:
            st = "fails: " + ", ".join(f"{short(p)} {sgn(v)}" for p, v in ck["persons_beyond"].items())
        code = str(x["code_sha256"])[:12] if x["code_sha256"] else "not recorded"
        L.append(f"| {k} | {x['folder']} | {x['run_at']} | {x['script']} | {code} | {len(pp)} | {f2(x['S14'])} | "
                 f"{f2(pp.get(FOCUS))} | {st} |")
    nr = [f"{k} ({', '.join(runs[k]['not_recorded'])})" for k in keys if runs[k]["not_recorded"]]
    only14 = [short(p) for p in dev14 if p not in set(dev12)]
    L += ["", "What each run is: " + "; ".join(f"{k}, {runs[k]['what']}" for k in keys) + ".",
          "Settings not recorded in a run's JSON: " + ("; ".join(nr) or "none") + ".",
          "Excluded: " + ("; ".join(f"{f} ({w})" for f, w in dis["excluded"].items()) or "none") + ".",
          "Unlisted: " + (", ".join(dis["unlisted"]) or "none") + ".",
          "Without inference settings, named: " + ("; ".join(f"{f} ({w})" for f, w in dis["no_settings_noted"].items())
                                                   or "none") + ".", "",
          f"Per-person all{n14} stride MAE (cm) of each run (a dev12-only run holds no {' and '.join(only14)}):", "",
          "| run | " + " | ".join(short(p) for p in dev14) + " |", "|---|" + "---|" * len(dev14)]
    for k in keys:
        pp = a["per_person_stride_cm"][k]
        L.append(f"| {k} | " + " | ".join(f2(pp.get(p)) for p in dev14) + " |")
    L += ["", "Run-to-run spread per person over the runs that hold the person:", "",
          "| person | runs | mean cm | SD cm | range cm | min | max |", "|---|---|---|---|---|---|---|"]
    for p, x in sorted(a["spread"].items(), key=lambda kv: -(kv[1]["sd"] or 0)):
        L.append(f"| {short(p)} | {x['n_runs']} | {f2(x['mean'])} | {f2(x['sd'], 3)} | {f2(x['range'])} | "
                 f"{f2(x['min'])} | {f2(x['max'])} |")
    tol = rc["tolerances"]["person_stride_cm"]
    L += ["", f"Pairs of runs: {pc['pairs']} pairs, {pc['pairs_with_a_person_beyond']} with some person beyond "
          f"{tol} cm; of the {pc['pairs_of_all_persons']} pairs that hold all {n14} persons, "
          f"{pc['pairs_of_all_persons_with_a_person_beyond']} with some person beyond {tol} cm; {fo} beyond {tol} "
          f"cm in {pc['pairs_with_the_focus_person_beyond']} of the {pc['pairs_holding_the_focus_person']} pairs that "
          "hold it.", "", "Same-code pairs (one script, one recorded code SHA-256):", "",
          f"| pair | script | code SHA-256 | persons | largest person difference cm | persons beyond {tol} cm | {fo} |",
          "|---|---|---|---|---|---|---|"]
    for x in a["same_code_pairs"]:
        L.append(f"| {x['a']} / {x['b']} | {x['script']} | {x['code_sha256'][:12]} | {x['n_persons']} | "
                 f"{f2(x['max_abs'])} ({short(x['argmax'])}) | "
                 f"{', '.join(f'{short(p)} {sgn(v)}' for p, v in x['beyond'].items()) or 'none'} | "
                 f"{sgn(x['focus']) if x['focus'] is not None else 'not held'} |")
    L += ["", f"{a['same_code_pairs_with_a_person_beyond']} of the {len(a['same_code_pairs'])} same-code pairs hold a "
          f"person beyond {tol} cm.", "",
          f"Rerun check against v1's dev report (eval_g3_diag's criteria: all{n14} stride MAE within "
          f"{rc['tolerances']['stride_mae_cm']} cm, speed MAE within {rc['tolerances']['speed_mae_ms']} m/s, each "
          f"person within {tol} cm): {rc['passed']} of the {rc['evaluated']} runs that hold all {n14} persons pass; "
          f"failing: {', '.join(rc['failed'])}. The checks the {len(rc['stored_checks_agree'])} diag runs store agree "
          f"with this reproduction ({rc['stored_passed']} passed, {rc['stored_failed']} failed): "
          + "; ".join(f"{k} {'passed' if x['passed'] else 'failed'}" for k, x in rc["stored_checks_agree"].items())
          + ".", "",
          f"The code control's {fo} ({f2(pos['value'])} cm) among the other {pos['n_others']} runs that hold {fo}: "
          f"mean {f2(pos['mean'])}, SD {f2(pos['sd'])}, range {f2(pos['min'])} to {f2(pos['max'])}, z {sgn(pos['z'])}, "
          f"rank {pos['rank_from_lowest']} from the lowest, inside the range: {pos['inside_range']}.", "",
          "Refinement objective per dev trial (trials.csv), relative difference |a - b| / |b|:", "",
          "| comparison | same code | trials | refine_first max | median | refine_last max | median |",
          "|---|---|---|---|---|---|---|"]
    for k, x in a["objective"]["comparisons"].items():
        L.append(f"| {k} | {x['same_code']} | {x['refine_first']['n']} | {sci(x['refine_first']['max'])} | "
                 f"{sci(x['refine_first']['median'])} | {sci(x['refine_last']['max'])} | "
                 f"{sci(x['refine_last']['median'])} |")
    of, ol = a["objective"]["same_code_refine_first_max_range"], a["objective"]["same_code_refine_last_max_range"]
    L += ["", f"Over the same-code pairs, the largest relative difference reads {sci(of['min'])} to {sci(of['max'])} "
          f"at refine_first and {sci(ol['min'])} to {sci(ol['max'])} at refine_last.", ""]
    sa = b["R0_per_person_item_set_aside"]
    sal = sa["lines"]
    ic = c["IC"]
    eq = ("at equality" if ic["difference_equals_margin"] else
          "inside the margin" if sal["IC_difference_minus_margin"] <= 0 else "beyond the margin")
    L += [f"Reading: **{a['reading']}**. Caveat: {a['caveat']}.", "",
          "## b. Both readings", "",
          f"- As written, the rule's verdict: **{b['as_written']['verdict']}** (case {b['as_written']['case']}): "
          + "; ".join(b["as_written"]["reasons"]) + ".",
          "- With R0's per-person item set aside (the rule's verdict function on the existing R1, R2 and T6; the "
          "control's rerun check failed on that item alone, every other R0 check holds), the verdict function would "
          f"read {sa['verdict']['verdict']} (case {sa['verdict']['case']}) against the rule's reference; this is not "
          "this rule's result: " + "; ".join(sa["verdict"]["reasons"]) + "; R2 "
          + ", ".join(f"{k} {'HOLDS' if h else 'FAILS'}" for k, h in sal["R2_holds"].items())
          + f" (IC difference minus margin {sal['IC_difference_minus_margin']:.1e} ms, {eq}: section c; SLIDE "
          f"{sal['SLIDE']}); T6 {sal['T6']}.",
          f"- With each v1 run at D1 settings in the place of the reference (R1 by the rule's paired bootstrap; S({n14}) "
          f"and V({n14}) from the run; IC and stride_mape_pct from the run when it is a dev report, else from the rule's "
          "dev reference; SLIDE with the rule's thresholds unchanged, where the run holds a model_legs row; T6 as "
          "read), the rule's verdict function would read (none of these is this rule's result):", "",
          "| reference | S(12) v1 cm | improvement [95 %] | meets | V(12) v1 m/s | improvement [95 %] | meets | "
          "SLIDE margin | SLIDE | verdict function |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in b["against_each_v1_run"].items():
        s, v = x["R1"]["S12"], x["R1"]["V12"]
        si, vi = s.get("improvement") or {}, v.get("improvement") or {}
        mg = x["SLIDE_margin"]
        L.append(f"| {k} | {f2(s['v1'])} | {f2(si.get('point'))} {iv(si.get('interval'))} | {s['meets']} | "
                 f"{f2(v['v1'], 4)} | {f2(vi.get('point'), 4)} {iv(vi.get('interval'), 4)} | {v['meets']} | "
                 f"{sgn(mg['margin'], 6) if mg else 'no model_legs row'} | "
                 f"{x['R2']['SLIDE'].get('status') or 'missing'} | {x['verdict']['verdict']} (case "
                 f"{x['verdict']['case']}) |")
    rr = b["r1_ranges_over_the_references"]
    wl = "; ".join(f"{n} {v}" for v, n in b["without_legs_row_verdicts"].items()) or "none"
    L += ["", f"5a column: {ROW_LABEL}, S(12) {f2(d['a5_S12_cm'])} cm. R1 MEETS on both lines against "
          f"{len(b['r1_meets_against'])} of the {len(b['against_each_v1_run'])} v1 runs (S(12) improvement "
          f"{f2(rr['S12']['point']['min'])} to {f2(rr['S12']['point']['max'])} cm, lowest lower bound "
          f"{f2(rr['S12']['lower_bound_min'])} cm; V(12) {f2(rr['V12']['point']['min'], 4)} to "
          f"{f2(rr['V12']['point']['max'], 4)} m/s, lowest lower bound {f2(rr['V12']['lower_bound_min'], 4)} m/s). Of "
          f"the {len(b['references_with_slide'])} runs that hold a model_legs row, the verdict function reads GO "
          f"against {', '.join(b['go_against']) or 'none'} and NO-GO (SLIDE FAILS) against "
          f"{', '.join(b['no_go_slide_against']) or 'none'}; the {len(b['without_legs_row'])} runs without that row: "
          f"{wl}. "
          + ("The set-aside GO does not survive the run-to-run spread of the reference. " if b["go_against"]
             and not b["go_survives_the_reference_spread"] else "")
          + f"Statement: {b['statement']}.", "",
          "## c. The fragile lines", "",
          f"One 5a run per report ({c['one_5a_run']['a5_diag']}, {c['one_5a_run']['a5_dev']}): the arm's own "
          "run-to-run spread enters every SLIDE margin and the IC difference below and is not measured; the "
          "per-reference margins show the reference's spread only.", ""]
    mv = ", ".join(f"{short(p)} {plus(x)}{frac_text(x)} ms" for p, x in ic["moves_ms"].items())
    if ic["difference_equals_margin"]:
        holds = "the line HOLDS at equality"
    elif ic["difference"] <= ic["margin"]:
        holds = "the line HOLDS inside its margin"
    else:
        holds = "the line FAILS"
    nf = ic["net_frames"]
    L += [f"- Walkway IC ({ROW_LABEL} against v1's dev report): the per-person medians are multiples of "
          f"{frac_text(ic['grid_ms'])} ms; moves 5a minus v1: {mv}. Sums over the {ic['n_persons']} persons: v1 "
          f"{frac_text(ic['sum_v1_ms'])} ms, 5a {frac_text(ic['sum_a5_ms'])} ms; lines {frac_text(ic['line_v1'])} = "
          f"{float(ic['line_v1']):.4f} ms and "
          f"{frac_text(ic['line_a5'])} = {float(ic['line_a5']):.4f} ms; difference {frac_text(ic['difference'])} ms. "
          f"Margin max({float(ic['share']):g} x {frac_text(ic['line_v1'])}, {ic['multiple']} x "
          f"{frac_text(ic['seed_d'])}) = max({frac_text(ic['share_of_line'])}, {frac_text(ic['multiple_of_d'])}) = "
          f"{frac_text(ic['margin'])} ms ({float(ic['margin']):.4f}). The difference equals the margin exactly: "
          f"{ic['difference_equals_margin']}; {holds}, on a net move of {plus(ic['net_move_ms'])}"
          f"{frac_text(ic['net_move_ms'])} ms summed over the persons ({frac_text(nf)} "
          f"{'frame' if abs(nf) == 1 else 'frames'} of {frac_text(ic['frame_ms'])} ms).",
          "- The other v1 dev runs at D1 settings: " + "; ".join(
              f"{k} ({x['run_at']}) reads {frac_text(x['line'])} = {float(x['line']):.4f} ms, "
              f"{plus(x['minus_reference'])}{frac_text(x['minus_reference'])} ms against the reference "
              f"({frac_text(x['minus_reference_in_margins'])} margins) "
              f"(moves {moves_text(x['moves_ms'])}), "
              f"5a minus this run {plus(x['a5_minus_this'])}{frac_text(x['a5_minus_this'])} ms"
              + (": as an arm against v1's own report, this rerun of v1 would read IC FAILS" if x["beyond_the_margin"]
                 else "") for k, x in ic["other_v1_dev_runs"].items()) + ".", ""]
    sl = c["SLIDE"]
    fs = b["thresholds"]["follow_share"]
    L += [f"- SLIDE ({ROW_LABEL}; thresholds unchanged: d_base {f2(b['thresholds']['d_base'], 4)}, d_legs "
          f"{f2(b['thresholds']['d_legs'], 4)}): margin legs rise minus {fs:g} x the base rise per v1 reference:", "",
          "| reference | date | base rise | legs rise | margin | legs over base | SLIDE |",
          "|---|---|---|---|---|---|---|"]
    for k, x in sl["per_reference"].items():
        L.append(f"| {k} | {x['run_at']} | {sgn(x['base_rise'], 5)} | {sgn(x['legs_rise'], 5)} | "
                 f"{sgn(x['margin'], 6)} | {f2(x['legs_over_base'], 3)} | {x['status']} |")
    s2 = sl["s2_B_replication"]
    if sl["undecided"]:
        slide_end = "SLIDE is undecided within the run-to-run spread: its status changes with the v1 run taken as the " \
                    "reference."
    elif sl["held"]:
        slide_end = "SLIDE HOLDS against every reference."
    else:
        slide_end = "SLIDE FAILS against every reference."
    mult = sl["seed_multiple"]
    L += ["", f"Mean margin over the {len(sl['per_reference'])} references {sgn(sl['mean_margin_all'], 6)}, over the "
          f"{len(sl['per_reference']) - 1} other than the rule's {sgn(sl['mean_margin_others'], 6)}; HELD against "
          f"{', '.join(sl['held']) or 'none'}, FEET SLIDE against {', '.join(sl['feet_slide']) or 'none'}. The legs "
          f"rise exceeds {mult:g} x d_legs against every reference: {sl['legs_rise_above_multiple_of_d_legs_everywhere']}"
          + (f"; the binding criterion is the {fs:g} x base rise" if sl["legs_rise_above_multiple_of_d_legs_everywhere"]
             else "")
          + f". Beside, reading nothing, the replication on the second seed ({sl['s2_B_label']}): margin "
          + "; ".join(f"{sgn(x['margin'], 6)} against {k} ({x['status']})" for k, x in s2.items())
          + f". {slide_end}", "",
          "## d. The bar", ""]
    s = d["same_strides"]
    zg, z2 = d["zupt_g3_matched_label_free_dev12"], d["zupt_v2_walkway_timed_dev12"]
    L += [f"On the {s['strides']} dev12 strides ({s['persons']} persons) that the 5a diag and the version-2 ZUPT both "
          "match (the join of scripts/f2_lever5.py decoded_over_zupt; rule section 7: a bar beside, reading "
          "nothing):", "",
          "| row | stride MAE, mean over persons (cm) | strides it is closer on |", "|---|---|---|",
          f"| {ROW_LABEL} | {f2(s['body_mae_person_mean_cm'])} | {f2(100 * s['share_body_closer'], 1)} percent |",
          f"| version-2 ZUPT of the same foot sensors (walkway-timed segmentation) | "
          f"{f2(s['zupt_mae_person_mean_cm'])} | {f2(100 * s['share_zupt_closer'], 1)} percent |", "",
          f"Ties {s['ties']}; body over ZUPT, median {f2(s['body_over_zupt_median'], 3)} (T6). The ZUPT readout on all "
          f"its matched dev12 strides reads {f2(z2['point_cm'])} cm {iv(z2['interval'])} (matched share "
          f"{f2(z2['matched_share'], 3)}); its label-free G3-matched line (reports/gates/g3/zupt_dev) reads "
          f"{f2(zg['point'])} cm {iv(zg['interval'])} on dev12 at a matched share {f2(zg['matched_share'], 3)} "
          f"({zg['n_events']} walkway strides); the anchored body's S(12) is {f2(d['a5_S12_cm'])} cm on its own "
          "matched strides.", "", "## e. The foot report (reported beside, descriptive)", ""]
    fm = e["foot_report"]
    lr = e["legs_over_base_range"]
    L += ["Rule section 5 reports the foot report beside, reading nothing; rule section 8: no 5a dev number here is "
          "confirmatory evidence of a mechanism. The foot report and T6 measure the body's convergence to the anchor, "
          "not its accuracy: both are circular by construction, the targets being the anchor's own. The version-2 "
          "ZUPT of the targets is a disclosed post hoc change of method (reports/bench/scale/prediction_amendment_1.txt "
          "and reports/bench/scale/zupt_v2/summary.md: post hoc in its method, not in its criterion).", "",
          f"Foot report of the 5a diag ({ROW_LABEL}) against the code control (zupt_strides.csv, row base, dev12, "
          f"{fm['targets']} targets paired one to one, the same ZUPT strides S, {fm['targets_inside_both']} with the "
          f"measured still run inside the window in both): mean S {f2(fm['mean_S_m'], 4)} m; D between the still "
          f"instants {f2(fm['mean_D_off_m'], 4)} m (control) and {f2(fm['mean_D_a5_m'], 4)} m (5a): added D "
          f"{f2(100 * fm['pooled_dD_m'])} cm per target. On the targets with the run inside in both, the added D is "
          f"{f2(100 * fm['pooled_dD_inside_m'])} cm and the added slide over the measured still run "
          f"{f2(100 * fm['pooled_dslide_m'])} cm: the added slide is {f2(fm['pooled_slide_part_of_dD'], 3)} of the "
          f"added displacement (person means: added D {f2(fm['person_mean_dD_cm'])} cm, added slide "
          f"{f2(fm['person_mean_dslide_cm'])} cm, ratio of the means {f2(fm['person_ratio_of_means'], 3)}, mean of the "
          f"person ratios {f2(fm['person_mean_of_ratios'], 3)}). Slide over S: {f2(fm['slide_share_of_S_off'], 4)} "
          f"(control) and {f2(fm['slide_share_of_S_a5'], 4)} (5a).", "",
          f"Per person ({ROW_LABEL} against the code control):", "",
          "| person | targets | added D cm | added slide cm | slide part | D over S control | D over S 5a |",
          "|---|---|---|---|---|---|---|"]
    for p, x in fm["per_person"].items():
        L.append(f"| {short(p)} | {x['targets']} | {f2(x['dD_cm'])} | {f2(x['dslide_cm'])} | "
                 f"{f2(x['slide_part_of_dD'], 3)} | {f2(x['D_over_S_off'], 3)} | {f2(x['D_over_S_a5'], 3)} |")
    L += ["", f"The legs row's rise over the base row's ({ROW_LABEL}; SLIDE's two gains), per v1 reference: "
          + ", ".join(f"{k} {f2(x, 3)}" for k, x in e["legs_over_base"].items())
          + f"; mean {f2(e['legs_over_base_mean'], 3)}, range {f2(lr['min'], 3)} to {f2(lr['max'], 3)}.", "",
          "## f. The rule's route", "",
          f"Rule section {r['f']['rule_section']}, as {r['f']['source']} prints it: {r['f']['route']}", "",
          "## g. Beside the rule: a programme proposal to the owner", "",
          "Outside the rule's section 9 routes, a proposal to the owner, who decides. It rests on "
          + "; ".join(r["g"]["rests_on"]) + ". " + r["g"]["proposal"], ""]
    return "\n".join(L) + "\n"


def _clean(o):
    if isinstance(o, Fraction):
        return {"fraction": frac_text(o), "value": float(o)}
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _under(path, root):
    p = os.path.normcase(os.path.abspath(path))
    r = os.path.normcase(os.path.abspath(root))
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def out_dir_allowed(out_dir, tree=None):
    """None when out_dir may receive the note: the note's own folder, or a folder outside this tree and outside the
    MoveAtlas tree (a test's temporary folder); else why not."""
    tree = tree or TREE
    if is_note_folder(out_dir, tree):
        return None
    if _under(out_dir, tree):
        return (f"{rel(out_dir, tree)} is inside this tree: the note writes only {NOTE_REL.replace(os.sep, '/')} "
                "there")
    if paths.MOVEATLAS and _under(out_dir, paths.MOVEATLAS):
        return f"{out_dir} is inside the MoveAtlas tree"
    return None


def is_note_folder(out_dir, tree=None):
    return os.path.normcase(os.path.abspath(out_dir)) == os.path.normcase(os.path.abspath(os.path.join(
        tree or TREE, NOTE_REL)))


def existing_outputs(out_dir):
    """The names of OUTPUT_NAMES that exist in out_dir."""
    return [n for n in OUTPUT_NAMES if os.path.exists(os.path.join(out_dir, n))]


def write_outputs(out_dir, res, started, force=False, tree=None):
    """results.json, note.md and _provenance.json in out_dir; PermissionError for a refused folder; FileExistsError
    when one of the three exists unless force, which only the note's own folder honours."""
    from mdfm.gates import common
    tree = tree or TREE
    why = out_dir_allowed(out_dir, tree)
    if why:
        raise PermissionError(why)
    path = os.path.join(out_dir, "results.json")
    own = is_note_folder(out_dir, tree)
    there = existing_outputs(out_dir)
    if there and not (force and own):
        raise FileExistsError(os.path.join(out_dir, there[0]))
    os.makedirs(out_dir, exist_ok=True)
    text = note_md(res)
    for t, what in ((text, "note.md"), (json.dumps(_clean(res), ensure_ascii=False), "results.json")):
        common.assert_no_em_dash(t, what)
        if "\u2013" in t:
            raise ValueError(f"{what} holds an en-dash")
    common.write_text(path, json.dumps(LA.F2().clean(_clean(res)), indent=1, ensure_ascii=False, allow_nan=False)
                      + "\n")
    common.write_text(os.path.join(out_dir, "note.md"), text)
    prov = {"script": "scripts/f2_lever5a_note.py", "task": TASK, "date": NOTE_DATE, "label": LABEL,
            "row_label": ROW_LABEL, "run_at": started, "python": sys.version.split()[0],
            "moveatlas_root": paths.MOVEATLAS, "rule_sha256": res["rule"]["sha256"],
            "inputs_sha256": {RESULTS_REL.replace("\\", "/"): res["reading_of_the_rule"]["sha256"],
                              SUMMARY_REL.replace("\\", "/"): res["reading_of_the_rule"]["summary_sha256"]},
            "code_sha256": common.code_sha256([os.path.abspath(__file__)] + [os.path.join(SCRIPTS, f"{n}.py") for n in
                                                                            ("f2_lever5a", "f2_lever5", "f2_lever4",
                                                                             "f2_lever2", "weekend_branch",
                                                                             "g3_gain_check", "eval_g3_diag")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "forced": bool(force and own)}
    try:
        prov["moveatlas_stamp"] = LA.GG().stamp_moveatlas()
    except Exception as e:  # noqa: BLE001  (reported, never required)
        prov["moveatlas_stamp"] = {"error": str(e)}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(LA.F2().clean(prov), indent=1) + "\n")
    return path


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help=f"output folder (default {NOTE_REL}; else a folder outside this tree "
                                                "and outside the MoveAtlas tree)")
    p.add_argument("--force-rewrite-note", action="store_true",
                   help=f"rewrite the existing outputs; honoured in {NOTE_REL} only")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    out_dir = a.out or os.path.join(TREE, NOTE_REL)
    why = out_dir_allowed(out_dir)
    if why:
        print(f"refused: {why}")
        return EXIT_REFUSED
    there = existing_outputs(out_dir)
    if there and not (a.force_rewrite_note and is_note_folder(out_dir)):
        print(f"an output exists: {os.path.join(out_dir, there[0])}")
        return EXIT_EXISTS
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        res, code = run()
    except LA.RuleNotHashed as e:
        print(f"refused: {e}")
        return EXIT_NOT_HASHED
    except HarnessDefect as e:
        print(f"harness defect: {e}")
        return EXIT_HARNESS
    except AssertionError as e:                       # the rule's hash differs (check_rule)
        print(f"refused: {e}")
        return EXIT_NOT_HASHED if "rule hash" in str(e) else EXIT_HARNESS
    try:
        write_outputs(out_dir, res, started, force=a.force_rewrite_note)
    except FileExistsError as e:
        print(f"an output exists: {e}")
        return EXIT_EXISTS
    except PermissionError as e:
        print(f"refused: {e}")
        return EXIT_REFUSED
    except HarnessDefect as e:
        print(f"harness defect: {e}")
        return EXIT_HARNESS
    print(note_md(res))
    return code


if __name__ == "__main__":
    sys.exit(main())
