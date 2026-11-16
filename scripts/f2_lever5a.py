"""L5A (2026-10-20): F2 lever 5a, the inertial anchor of lever 5 inside the test-time refinement (label-free), read on
v1's B end against v1's B end without it. Label of every output: dev diagnostic, not a G3 read; every 5a row is
labelled ROW_LABEL.

Rule: reports/bench/f2_lever5a/rule.txt, final, hashed by the orchestrator before any 5a output exists (rule.sha256
beside it). check_rule asserts the file's SHA-256 equal to the first word of rule.sha256 (and to RULE_SHA256 when that
constant is filled) before any 5a value is read; without rule.sha256 the reading reads nothing (RuleNotHashed, exit
3). rule_numbers() formats every number the rule prints from the files that exist; tests/test_f2_lever5a.py asserts
each one in the rule's text.

The reading (default mode; the rule's sections 4 to 8):
- R0, integrity: every 5a output written after rule.sha256 (its _provenance.json run_at against the hash file's
  modification time); each output of the expected checkpoint (SHA-256), EMA weights and the reference's inference
  settings (REF_INFERENCE, also asserted of the reference reports); the run record of the option equal to the rule's
  (mode, weight, sd, edge, floor, steps, ZUPT version); every dev trial run, targets present; the two 5a runs agree
  (diag base row against the dev JSON, the rerun tolerances of scripts/eval_g3_diag.py); the code control (v1_B
  with --zupt-report under the same code) passed eval_g3_diag's rerun check against v1's dev report, and shares the
  5a diag's code SHA-256.
- R1, primary: S(12) and V(12), paired v1 minus 5a per person on dev12 (scripts/f2_lever4.py r1_reading, lever 4's
  margins and bootstrap); MEETS when the point is at least the margin and the lower bound above 0.
- R2, non-inferiority (5a minus v1, positive is worse): S(14), V(14), the walkway IC and stride_mape_pct at lever
  4's margins, and SLIDE, the stance-slide guard: FAILS (FEET SLIDE) when the dev12 base-row stride gain rises beyond
  2 x d_base and the model_legs row's rise is below half of it or below 2 x d_legs (scripts/f2_lever5.py t5_flag,
  lever 5's thresholds), the gains read from the default diag events of v1_B and of the 5a run.
- T6 BEYOND THE INPUT (qualifies a GO, scripts/f2_lever5.py decoded_over_zupt): the 5a dev12 decoded stride over the
  version-2 ZUPT stride, median over the strides both match, above 1.02.
- Verdict: (1) an R0 failure NOT EVALUATED; (2) an R2 line read and FAILS NO-GO; (3) S(12) and V(12) read and
  neither MEETS NO-GO; (4) a missing R1 or R2 input NOT EVALUATED; (5) both MEET: GO when T6 is read and HELD, 'GO,
  qualified' when T6 flags, 'GO, T6 not evaluated' when it is missing; (6) one MEETS: PARTIAL.
- Reported beside, reading nothing: R1 and R2 against the code control in place of the reference; the foot report
  (zupt_strides.csv) of the 5a run against the code control, per dev12 person (D over S, the ankle speed at the
  still instants, the slide over the measured still run), paired; the replication on s2_B (its matched bw32 diag
  against its 5a diag); the bars (the version-2 ZUPT readout of SCALE, the G3-matched ZUPT line, the F1 head, lever
  4's end); the term at its first and last refinement call; the all68 rows of the dev JSONs.

Modes: --rule-numbers (prints rule_numbers()); default: the reading, written to reports/bench/f2_lever5a/results.json,
summary.md and _provenance.json (never over an existing results.json: exit 5).
Exit codes: 0 a verdict; 3 NOT EVALUATED (an unhashed rule included); 4 a harness defect; 5 an output exists.
Data discipline: dev14 and dev12 of F1, guarded against the lockbox list; no lockbox, EVAL, CODY or care_pd record.

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever5a.py [--rule-numbers] [--out DIR]
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the reading stays off the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
BENCH = os.path.join("reports", "bench", "f2_lever5a")
RULE_REL = os.path.join(BENCH, "rule.txt")
RULE_SHA_REL = os.path.join(BENCH, "rule.sha256")
RULE_SHA256 = "361c81dd51c43ebe8108db8996c695cc2c0e66d61c6460d35f2fa82544282694"   # pinned 2026-10-20 when rule.sha256 was written (task L5A review)
LABEL = "dev diagnostic, not a G3 read"
ROW_LABEL = "inertially anchored body (lever 5a, at inference)"
TASK = "L5A"
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
F2L4_B = "pilot_f2l4_ckpt_B_032044"
V1_CKPT = "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
S2_CKPT = "cache/runs/pilot_v1_s2/ckpt_B_032044.pt"
V1_SHA = "cffccdac4109dfbfb921ed4ebcb690ce27dd2d35045a3afdaaca7d38f321a677"
S2_SHA = "5e97c4fe3b208f8f11850a5318ed827d26316f1e2fc867355d74f50a3d96e023"
G3 = os.path.join("reports", "gates", "g3")
OUT = {"a5_diag": os.path.join(BENCH, f"diag_l5a_{V1_B}"), "a5_dev": os.path.join(BENCH, f"dev_l5a_{V1_B}"),
       "off_diag": os.path.join(BENCH, f"diag_l5aoff_{V1_B}"), "s2_a5_diag": os.path.join(BENCH, f"diag_l5a_{S2_B}")}
REF = {"v1_diag": os.path.join(G3, f"diag_{V1_B}"), "v1_dev": os.path.join(G3, f"dev_{V1_B}"),
       "s2_diag_bw32": os.path.join(G3, f"diag_{S2_B}_bw32")}
REF_INFERENCE = {"refine_steps": 20, "points_per_stream": 256, "batch_windows": 32, "seed": 20260929,
                 "shape_pooling": "per trial", "weights": "ema"}
ZUPT_RECORD = {"mode": "anchor", "weight": 1.0, "sd_m_s": 0.1, "edge_s": 0.1, "min_length_m": 0.3, "steps": 20,
               "in_objective": True}
DEV_TRIALS = 28
RERUN_TOL = {"stride_mae_cm": 0.5, "speed_mae_ms": 0.01}
PRIMARY = ("S12", "V12")
R2_LINES = ("S14", "V14", "IC", "MAPE", "SLIDE")
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "IC": "ms", "MAPE": "percent", "SLIDE": "gain"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "IC": 2, "MAPE": 2, "SLIDE": 4}
G3_LINE_OF = {"IC": "ic_median_ms_walkway", "MAPE": "stride_mape_pct"}
TOL = 1e-9
BEYOND = 1.02
LEGS_ROW = "model_legs"
MIN_ROW_PERSONS = 10
NO_GO, GO, PARTIAL, NOT_EVALUATED = "NO-GO", "GO", "PARTIAL", "NOT EVALUATED"
GO_QUALIFIED, GO_T6_MISSING = "GO, qualified", "GO, T6 not evaluated"
EXIT_VERDICT, EXIT_NOT_EVALUATED, EXIT_HARNESS, EXIT_EXISTS = 0, 3, 4, 5
ZUPT_V2_RESULTS = os.path.join("reports", "bench", "scale", "zupt_v2", "results.json")
ZUPT_DEV_RESULTS = os.path.join(G3, "zupt_dev", "results.json")
ZUPT_DEV_STRIDES = os.path.join(G3, "zupt_dev", "zupt_strides.csv")
Q_RESULTS = os.path.join("reports", "bench", "scale", "q", "results.json")
F2L4_RESULTS = os.path.join("reports", "bench", "f2_lever4", "results.json")
F1_RESULTS = os.path.join("reports", "bench", "g3_gain_check", "results.json")
SMOKE_JSON = os.path.join(BENCH, "smoke", "smoke_train54.json")   # the CPU smoke of rule section 0 (train54)


class HarnessDefect(AssertionError):
    """A departure of the harness itself (a person outside the set, a reference report at other settings)."""


class RuleNotHashed(RuntimeError):
    """rule.sha256 does not exist: the rule is not registered, nothing is read."""


# ---------------------------------------------------------------- the sibling readers
def _sibling(name):
    key = f"_l5a_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


def L5():
    """scripts/f2_lever5.py (its margins: lever 4's with the transfer thresholds, t5_flag, decoded_over_zupt)."""
    return _sibling("f2_lever5")


def L4():
    return L5().L4()


def F2():
    return L4().F2()


def WB():
    return L4().WB()


def GG():
    return L4().GG()


def fin(x):
    return F2().fin(x)


def read_json(path):
    return F2().read_json(path)


def rel(path, tree=None):
    try:
        return os.path.relpath(os.path.abspath(path), tree or TREE).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path):
    return F2().sha256_file(path)


# ---------------------------------------------------------------- the rule
def check_rule(tree=None):
    """The SHA-256 of rule.txt, asserted equal to rule.sha256's first word (and to RULE_SHA256 when filled);
    RuleNotHashed when rule.sha256 is absent."""
    tree = tree or TREE
    p = os.path.join(tree, RULE_SHA_REL)
    if not os.path.exists(p):
        raise RuleNotHashed(f"{RULE_SHA_REL} does not exist: the lever-5a rule is not registered, nothing is read")
    got = sha256_file(os.path.join(tree, RULE_REL))
    with open(p, encoding="utf-8") as f:
        stored = (f.read().split() or [""])[0]
    if got != stored or (RULE_SHA256 is not None and got != RULE_SHA256):
        raise AssertionError(f"rule hash mismatch: rule.txt {got}, rule.sha256 {stored}, script {RULE_SHA256}")
    return got


def rule_hash_time(tree=None):
    """The local modification time of rule.sha256 (the moment the rule was registered), ISO seconds."""
    p = os.path.join(tree or TREE, RULE_SHA_REL)
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")


def load_sets(tree=None):
    return L5().load_sets(tree)


# ---------------------------------------------------------------- readers of a folder
def diag_set_at(tree, folder, sets, locked, sha):
    """(the base row of folder/diag.json as a G3 set, None; reason when None), persons guarded, checkpoint SHA-256 and
    EMA weights asserted."""
    path = os.path.join(tree, folder, "diag.json")
    doc = read_json(path)
    if doc is None:
        return None, f"missing: {rel(path, tree)}"
    issue = F2().checkpoint_issue(doc, None, sha, rel(path, tree))
    if issue:
        return None, issue
    s = WB().g3set_from_diag_row((doc.get("rows") or {}).get("base"), path)
    if s is None:
        return None, f"no base row in {rel(path, tree)}"
    F2().guard_set(s, sets, locked, rel(path, tree))
    return s, None


def dev_line_at(tree, folder, line, sha):
    """(point, None) of a results_dev line of folder/g3_dev.json, or (None, reason)."""
    path = os.path.join(tree, folder, "g3_dev.json")
    doc = read_json(path)
    if doc is None:
        return None, f"missing: {rel(path, tree)}"
    issue = F2().checkpoint_issue(doc, None, sha, rel(path, tree))
    if issue:
        return None, issue
    for r in doc.get("results_dev") or []:
        if isinstance(r, dict) and r.get("line") == line:
            v = fin(r.get("point"))
            return (v, None) if v is not None else (None, f"{line} not finite in {rel(path, tree)}")
    return None, f"no {line} line in {rel(path, tree)}"


def dev_model_line(doc, row, line):
    """The model's point of a line in row 'dev' or 'all68' of a g3_dev.json's lines block."""
    return fin(((((doc or {}).get("lines") or {}).get(row) or {}).get("model") or {}).get(line, {}).get("point"))


def gains_at(tree, folder, sets, locked):
    """{base, model_legs: dev12 stride gain (F1's median form) or None} of folder/events.csv, with the reasons."""
    out, why = {}, []
    for row in ("base", LEGS_ROW):
        g, n, n14, w = L4()._stride_gain(os.path.join(tree, folder, "events.csv"), sets["dev14"], locked,
                                         sets["dev12"], row=row, min_persons=MIN_ROW_PERSONS)
        out[row] = g
        if w:
            why.append(w)
    return out, why


# ---------------------------------------------------------------- numbers of the rule (attainability included)
def share_line(units, g, f):
    """Pure-gain arithmetic of scripts/probe_walkway_anchor.py gain_scan at one share f of the gap between the train54
    gain g and 1: (line, paired improvement over f = 0 {point, interval}) on v1's dev12 units."""
    gg, wb = GG(), WB()
    raw = gg.person_abs_error(units, 1.0)
    gf = g + f * (1.0 - g)
    sc = gg.person_abs_error(units, g / gf)
    p = wb.paired(raw, sc)
    return float(np.mean(list(sc.values()))), {"point": p["point"], "interval": p["interval"]}


def margins(tree, sets, locked):
    """Lever 4's margins with lever 5's transfer thresholds (scripts/f2_lever5.py margins)."""
    return L5().margins(tree, sets, locked)


def rule_numbers(tree=None, sets=None, locked=None, m=None):
    """({placeholder: the string the rule prints}, checks {name: bool}) from the files that exist."""
    tree = tree or TREE
    if sets is None or locked is None:
        sets, locked = load_sets(tree)
    m = m if m is not None else margins(tree, sets, locked)
    s, v, sd, ni, t = m["success"]["S12"], m["success"]["V12"], m["seed_diff"], m["noninferiority"], m["transfer"]
    P = L4()._sibling("probe_walkway_anchor")
    _, dev14, dev12, lb, _ = P.person_sets()
    att = P.attainability(dev14, dev12, set(lb))
    sc_s, sc_v = att["gain_scan"]["S12"], att["gain_scan"]["V12"]
    out = {"V1_SHA": V1_SHA, "S2_SHA": S2_SHA,
           "S12_LINE": f"{s['line']:.4f}", "S12_D": f"{sd['S12']:.4f}", "S12_M": f"{s['margin']:.4f}",
           "S12_NEED": f"{s['needed_line_at_most']:.4f}", "V12_LINE": f"{v['line']:.4f}", "V12_D": f"{sd['V12']:.4f}",
           "V12_M": f"{v['margin']:.4f}", "V12_NEED": f"{v['needed_line_at_most']:.4f}"}
    for k in ("S14", "V14", "IC", "MAPE"):
        out[f"{k}_M"], out[f"{k}_LINE"], out[f"{k}_D"] = (f"{ni[k]['margin']:.4f}", f"{ni[k]['line']:.4f}",
                                                          f"{ni[k]['seed_diff']:.4f}")
    out["D_BASE"], out["D_LEGS"] = f"{t['d_r']['base']:.4f}", f"{t['d_legs']:.4f}"
    gv1, _ = gains_at(tree, REF["v1_diag"], sets, locked)
    g4, _ = gains_at(tree, os.path.join(G3, f"diag_{F2L4_B}"), sets, locked)
    out["G_BASE_V1"], out["G_LEGS_V1"] = f"{gv1['base']:.4f}", f"{gv1[LEGS_ROW]:.4f}"
    rb4, rl4 = g4["base"] - gv1["base"], g4[LEGS_ROW] - gv1[LEGS_ROW]
    out["L4_BASE_RISE"], out["L4_LEGS_RISE"] = f"{rb4:+.4f}", f"{rl4:+.4f}"
    zc = os.path.join(tree, L5().ZUPT_V2_STRIDES)
    for key, name in (("T6_V1", V1_B), ("T6_L4", F2L4_B)):
        g, n = L5().decoded_over_zupt(os.path.join(tree, G3, f"diag_{name}", "events.csv"), zc, sets["dev12"], locked)
        out[key], out[key + "_N"] = f"{g:.3f}", f"{n}"
    z = read_json(os.path.join(tree, ZUPT_V2_RESULTS))
    b12, b14 = z["baseline_line_dev12"], z["baseline_line_dev14"]
    out.update({"ZK12": f"{b12['point_cm']:.2f}", "ZK12_IV": f"[{b12['interval'][0]:.2f}, {b12['interval'][1]:.2f}]",
                "ZK12_SHARE": f"{b12['matched_share']:.3f}", "ZK14": f"{b14['point_cm']:.2f}",
                "Z_TRAIN54": f"{z['Z1']['train54']['median']:.3f}"})
    zd = read_json(os.path.join(tree, ZUPT_DEV_RESULTS))
    zl = zd["lines"]["zupt"]["stride_mae_cm"]["sets"]["dev12"]
    out["ZG12"], out["ZG12_SHARE"] = f"{zl['point']:.2f}", f"{zl['matched_share']:.3f}"
    out["ZDEV_FORMED"], out["ZDEV_RECORDS"] = f"{zd['zupt_strides_formed']:,}", f"{zd['records']}"
    import pandas as pd
    zs = pd.read_csv(os.path.join(tree, ZUPT_DEV_STRIDES), encoding="utf-8")
    out["ZDEV_03"] = f"{int((zs['length_m'] >= 0.3).sum()):,}"
    out["HEAD_S"], out["HEAD_V"] = f"{sc_s['head_line']:.4f}", f"{sc_v['head_line']:.4f}"
    out["F_S"], out["F_V"] = (f"{sc_s['smallest_share_meeting']['f']:.2f}",
                              f"{sc_v['smallest_share_meeting']['f']:.2f}")
    r4 = read_json(os.path.join(tree, F2L4_RESULTS))["R1"]
    out["L4_S12"], out["L4_V12"] = f"{r4['S12']['f2l4']:.2f}", f"{r4['V12']['f2l4']:.4f}"
    q = read_json(os.path.join(tree, Q_RESULTS))["real"]
    for arm, k in (("q0w20", "Q0"), ("q20", "Q20"), ("q60", "Q60")):
        out[k], out[k + "_SLIDE"] = f"{q[arm]['gain']:.3f}", f"{q[arm]['slide']['median']:.3f}"
    out["Q20_IV"] = f"[{q['q20']['interval'][0]:.3f}, {q['q20']['interval'][1]:.3f}]"
    fq = (q["q20"]["gain"] - q["q0w20"]["gain"]) / (1.0 - q["q0w20"]["gain"])
    out["Q_SHARE"] = f"{fq:.2f}"
    f1 = read_json(os.path.join(tree, F1_RESULTS))["ends"]["v1_B"]["gain"]
    gs, gvv = float(f1["stride"]["gain"]), float(f1["speed"]["gain"])
    ev = os.path.join(tree, REF["v1_diag"], "events.csv")
    st, spd, _ = GG().read_events(ev, sets["dev14"], locked)
    ls, ps = share_line(GG().subset(st, sets["dev12"]), gs, fq)
    lv, pv = share_line(GG().subset(spd, sets["dev12"]), gvv, fq)
    out.update({"QS_S12": f"{ls:.2f}", "QS_S12_IMP": f"{ps['point']:.2f}", "QS_V12": f"{lv:.4f}",
                "QS_V12_IMP": f"{pv['point']:.4f}"})
    meets_q = bool(ps["point"] >= s["margin"] - TOL and ps["interval"][0] > TOL
                   and pv["point"] >= v["margin"] - TOL and pv["interval"][0] > TOL)
    sm = read_json(os.path.join(tree, SMOKE_JSON))
    on, rp, of = sm["modes"]["anchor"], sm["modes"]["report"], sm["modes"]["off"]
    out.update({"SMOKE_TRIAL": sm["trial"], "SMOKE_TERM_FIRST": f"{on['zupt_term_first']:,.0f}",
                "SMOKE_TERM_LAST": f"{on['zupt_term_last']:,.0f}", "SMOKE_RATIO_FIRST": f"{on['zupt_ratio_first']:.3f}",
                "SMOKE_RATIO_LAST": f"{on['zupt_ratio_last']:.3f}",
                "SMOKE_OFF_RATIO": f"{rp['zupt_report_ratio_median']:.3f}",
                "SMOKE_ON_RATIO": f"{on['zupt_report_ratio_median']:.3f}",
                "SMOKE_OFF_SLIDE": f"{rp['zupt_report_slide_share_median']:.3f}",
                "SMOKE_ON_SLIDE": f"{on['zupt_report_slide_share_median']:.3f}",
                "SMOKE_TARGETS": f"{on['zupt_targets']}", "SMOKE_WINDOWS": f"{on['zupt_windows_anchored']} of "
                                                                          f"{on['windows']}",
                "SMOKE_SECONDS": f"{on['seconds']:.0f}", "SMOKE_OFF_SECONDS": f"{of['seconds']:.0f}"})
    checks = {"margins_below_lines": bool(s["margin"] < s["line"] and v["margin"] < v["line"]),
              "head_meets_both": bool(sc_s["head_improvement"] >= s["margin"]
                                      and sc_v["head_improvement"] >= v["margin"]),
              "q_share_meets_both": meets_q,
              "lever4_default_diag_reads_feet_slide": bool(L5().t5_flag(rb4, rl4, t["d_r"]["base"], t["d_legs"])),
              "v1_t6_below_threshold": bool(float(out["T6_V1"]) <= BEYOND)}
    return out, checks


# ---------------------------------------------------------------- R0
def _prov(tree, folder):
    return read_json(os.path.join(tree, folder, "_provenance.json"))


def _settings_issues(doc, what, sha, zupt=None):
    out = []
    if doc is None:
        return [f"{what}: missing"]
    ck = doc.get("checkpoint") or {}
    if ck.get("sha256") != sha:
        out.append(f"{what}: checkpoint {str(ck.get('sha256'))[:12]}, expected {sha[:12]}")
    if ck.get("weights") not in (None, "ema"):
        out.append(f"{what}: {ck.get('weights')} weights")
    inf = doc.get("inference") or {}
    bad = {k: inf.get(k) for k, x in REF_INFERENCE.items() if inf.get(k) != x}
    if bad:
        out.append(f"{what}: inference settings {bad} differ from the reference's {REF_INFERENCE}")
    za = inf.get("zupt_anchor")
    if zupt == "none":
        if za is not None:
            out.append(f"{what}: the option is recorded in a reference report")
    elif zupt is not None:
        from mdfm.readouts import zupt as Z
        want = dict(ZUPT_RECORD, mode=zupt, in_objective=zupt == "anchor")
        got = {k: (za or {}).get(k) for k in want}
        if got != want or ((za or {}).get("zupt") or {}).get("version") != Z.VERSION:
            out.append(f"{what}: the option's run record {got} is not the rule's {want}")
    return out


def r0_reading(tree):
    """R0 of the rule (module docstring): {holds, fail [reasons], checks, values}."""
    fail, checks, values = [], {}, {}
    hashed_at = rule_hash_time(tree)
    values["rule_hashed_at"] = hashed_at
    docs = {"a5_diag": read_json(os.path.join(tree, OUT["a5_diag"], "diag.json")),
            "a5_dev": read_json(os.path.join(tree, OUT["a5_dev"], "g3_dev.json")),
            "off_diag": read_json(os.path.join(tree, OUT["off_diag"], "diag.json")),
            "v1_diag": read_json(os.path.join(tree, REF["v1_diag"], "diag.json")),
            "v1_dev": read_json(os.path.join(tree, REF["v1_dev"], "g3_dev.json"))}
    issues = []
    for key, zupt in (("a5_diag", "anchor"), ("a5_dev", "anchor"), ("off_diag", "report"), ("v1_diag", "none"),
                      ("v1_dev", "none")):
        issues += _settings_issues(docs[key], key, V1_SHA, zupt)
    checks["settings"] = not issues
    fail += issues
    late = []
    for key in ("a5_diag", "a5_dev", "off_diag"):
        pv = _prov(tree, OUT[key])
        run_at = (pv or {}).get("run_at")
        values[f"{key}_run_at"] = run_at
        if not run_at:
            late.append(f"{key}: no run_at in its _provenance.json")
        elif str(run_at) <= hashed_at:
            late.append(f"{key}: written at {run_at}, not after the rule's hash ({hashed_at})")
    checks["after_the_hash"] = not late
    fail += late
    tr = []
    for key in ("a5_diag", "off_diag"):
        za = (docs[key] or {}).get("zupt_anchor") or {}
        values[f"{key}_trials"], values[f"{key}_targets"] = za.get("trials"), za.get("targets")
        if za.get("trials") != DEV_TRIALS or not za.get("targets"):
            tr.append(f"{key}: {za.get('trials')} trials and {za.get('targets')} targets, expected {DEV_TRIALS} trials "
                      "with targets")
    zdev = ((docs["a5_dev"] or {}).get("zupt_anchor") or {}).get("dev") or {}
    if zdev.get("trials") != DEV_TRIALS or not zdev.get("targets"):
        tr.append(f"a5_dev: {zdev.get('trials')} dev trials and {zdev.get('targets')} targets")
    checks["trials_and_targets"] = not tr
    fail += tr
    pair = []
    if docs["a5_diag"] is not None and docs["a5_dev"] is not None:
        base = ((docs["a5_diag"].get("rows") or {}).get("base") or {}).get("all14") or {}
        ds = fin(base.get("stride_mae_cm"))
        dv = fin(base.get("speed_mae_ms"))
        es, ev = dev_model_line(docs["a5_dev"], "dev", "stride_mae_cm"), dev_model_line(docs["a5_dev"], "dev",
                                                                                        "speed_mae_m_s")
        values["a5_pair"] = {"diag_S14": ds, "dev_S14": es, "diag_V14": dv, "dev_V14": ev}
        if None in (ds, es, dv, ev) or abs(ds - es) > RERUN_TOL["stride_mae_cm"] + TOL \
                or abs(dv - ev) > RERUN_TOL["speed_mae_ms"] + TOL:
            pair.append(f"the 5a diag and dev runs disagree beyond the rerun tolerances: {values['a5_pair']}")
    checks["a5_pair_agrees"] = not pair and docs["a5_diag"] is not None and docs["a5_dev"] is not None
    fail += pair
    rc = (docs["off_diag"] or {}).get("rerun_check") or {}
    values["code_control_rerun_check"] = rc.get("reading")
    checks["code_control_rerun_passed"] = bool(rc.get("evaluated") and rc.get("passed"))
    if not checks["code_control_rerun_passed"]:
        fail.append(f"the code control's rerun check did not pass: {rc.get('reading')}")
    pa, po = _prov(tree, OUT["a5_diag"]) or {}, _prov(tree, OUT["off_diag"]) or {}
    checks["one_code_version"] = bool(pa.get("code_sha256") and pa.get("code_sha256") == po.get("code_sha256"))
    if not checks["one_code_version"]:
        fail.append("the 5a diag and the code control ran under different code (code_sha256)")
    return {"holds": not fail, "fail": fail, "checks": checks, "values": values}


# ---------------------------------------------------------------- R1, R2, T6
def renamed(o, new="a5"):
    if isinstance(o, dict):
        return {(new if k == "f2l4" else k): renamed(v, new) for k, v in o.items()}
    if isinstance(o, list):
        return [renamed(v, new) for v in o]
    return o


def r1_reading(ref_set, a5_set, m, why=None):
    out = renamed(L4().r1_reading(ref_set, a5_set, m, why))
    for x in out.values():
        x["label"] = ROW_LABEL
    return out


def slide_reading(g_ref, g_a5, d_base, d_legs, why=None):
    """The stance-slide guard SLIDE of R2 (module docstring)."""
    row = {"read": False, "v1": g_ref, "a5": g_a5, "difference": None, "margin": None, "holds": None,
           "unit": UNIT["SLIDE"], "reason": None, "label": ROW_LABEL, "d_base": d_base, "d_legs": d_legs}
    if any(x is None or x.get("base") is None or x.get(LEGS_ROW) is None for x in (g_ref, g_a5)):
        row["reason"] = why or "a dev12 gain of row base or model_legs is missing"
        return row
    rb, rl = g_a5["base"] - g_ref["base"], g_a5[LEGS_ROW] - g_ref[LEGS_ROW]
    flag = L5().t5_flag(rb, rl, d_base, d_legs)
    row.update(read=True, base_rise=rb, legs_rise=rl, difference=rb - rl, holds=not flag,
               status="FEET SLIDE" if flag else "HELD")
    return row


def r2_reading(vals_ref, vals_a5, m, reasons, slide):
    out = {}
    for k in ("S14", "V14", "IC", "MAPE"):
        a, b = vals_ref.get(k), vals_a5.get(k)
        margin = m["noninferiority"][k]["margin"]
        ok = a is not None and b is not None
        row = {"read": ok, "v1": a, "a5": b, "difference": None, "margin": margin, "holds": None, "unit": UNIT[k],
               "reason": None if ok else (reasons.get(k) or f"no {k} value"), "label": ROW_LABEL}
        if ok:
            row["difference"] = b - a
            row["holds"] = bool(b - a <= margin + TOL)
        out[k] = row
    out["SLIDE"] = slide
    return out


def t6_reading(tree, folder, sets, locked):
    zc = os.path.join(tree, L5().ZUPT_V2_STRIDES)
    out = {"line": "T6", "flag_name": "BEYOND THE INPUT", "label": ROW_LABEL, "threshold": BEYOND, "ends": {}}
    for key, f in (("v1_B", REF["v1_diag"]), ("a5", folder)):
        g, n = L5().decoded_over_zupt(os.path.join(tree, f, "events.csv"), zc, sets["dev12"], locked)
        out["ends"][key] = {"decoded_over_zupt": g, "strides": n}
    g = out["ends"]["a5"]["decoded_over_zupt"]
    if g is None:
        out.update(read=False, flag=None, status="missing", reason=f"no 5a diag events or no {L5().ZUPT_V2_STRIDES}")
        return out
    flag = bool(g > BEYOND + TOL)
    out.update(read=True, flag=flag, status="BEYOND THE INPUT" if flag else "HELD", reason=None)
    return out


# ---------------------------------------------------------------- verdict and route
def verdict(r0, r1, r2, t6):
    """The rule's verdict order (module docstring)."""
    if r0["fail"]:
        return {"verdict": NOT_EVALUATED, "case": 1, "reasons": list(r0["fail"]), "missing": list(r0["fail"])}
    failed = [k for k in R2_LINES if r2[k]["read"] and not r2[k]["holds"]]
    unread = [f"{k}: {r2[k]['reason']}" for k in R2_LINES if not r2[k]["read"]]
    if failed:
        return {"verdict": NO_GO, "case": 2, "reasons": [f"{k} FAILS" for k in failed], "missing": unread,
                "failed": failed}
    if all(r1[L]["read"] for L in PRIMARY) and not any(r1[L]["meets"] for L in PRIMARY):
        return {"verdict": NO_GO, "case": 3, "reasons": ["neither S(12) nor V(12) MEETS"], "missing": unread,
                "failed": []}
    missing = [f"{L}: {r1[L]['reason']}" for L in PRIMARY if not r1[L]["read"]] + unread
    if missing:
        return {"verdict": NOT_EVALUATED, "case": 4, "reasons": missing, "missing": missing}
    met = [L for L in PRIMARY if r1[L]["meets"]]
    if len(met) == 2:
        if not t6["read"]:
            v = GO_T6_MISSING
        else:
            v = GO_QUALIFIED if t6["flag"] else GO
        return {"verdict": v, "case": 5, "reasons": [f"{L} MEETS" for L in met] + ([t6["status"]] if t6.get("flag")
                                                                                     else []),
                "missing": [] if t6["read"] else [f"T6: {t6['reason']}"], "met": met,
                "flags": [t6["status"]] if t6.get("flag") else []}
    return {"verdict": PARTIAL, "case": 6, "reasons": [f"{L} MEETS" for L in met], "missing": [], "met": met,
            "flags": [t6["status"]] if t6.get("flag") else []}


def route(v):
    """The routes of the rule (section 9), as text."""
    x = v["verdict"]
    if x == GO:
        return ("GO: the existing decoder holds the input's scale at inference without slide and without a cost on "
                "the other lines; lever 5b (the training run of docs/proposals/F2_LEVER5_DRAFT.md, its DRAFT rule "
                "reports/bench/f2_lever5/rule.txt) is proposed to the owner, who decides it; the 5a rows are reported "
                f"as '{ROW_LABEL}' beside v1's, never in their place.")
    if x == GO_QUALIFIED:
        return ("GO, qualified (BEYOND THE INPUT): the body's stride exceeds its own input's integral; reported, and "
                "lever 5b is not proposed as drafted: the owner decides whether the term's weight is a new rule.")
    if x == GO_T6_MISSING:
        return ("GO, T6 not evaluated: reported; lever 5b is not proposed until T6 is read under this rule. Missing: "
                + "; ".join(v.get("missing") or []) + ".")
    if x == PARTIAL:
        return ("PARTIAL: reported; lever 5b is not proposed on this reading; the line that met: "
                + ", ".join(v.get("met") or []) + ".")
    if x == NO_GO:
        f = v.get("failed") or []
        if "SLIDE" in f:
            return ("NO-GO (stance slide): the scale came through the feet sliding in stance at inference, the failure "
                    "lever 5b expects; 5b is not proposed as drafted, and the upstream alternative of the draft (section "
                    "8 item 5) is the body track's next candidate after its own diagnostic.")
        if f:
            return ("NO-GO (" + ", ".join(f) + "): the anchor costs a line the body must hold; 5b is not proposed as "
                    "drafted.")
        return ("NO-GO (neither S(12) nor V(12)): the decoder cannot hold the scale at inference with the stillness "
                "part on; lever 5b, a training run, is the only route left on the body (draft section 8 item 4), and "
                "lever 1 stays the only correction with a measured effect on the readout.")
    return "NOT EVALUATED: nothing is proposed; missing: " + "; ".join(v.get("missing") or []) + "."


# ---------------------------------------------------------------- reported beside
def foot_rows(tree, folder, persons):
    """{person: {D_over_S, speed, slide_share}} means over the zupt_strides.csv rows of the dev row (diag: row base)
    of a folder, the persons in `persons` only; None when the file is absent."""
    import pandas as pd
    p = os.path.join(tree, folder, "zupt_strides.csv")
    if not os.path.exists(p):
        return None
    z = pd.read_csv(p, encoding="utf-8", dtype={"person_id": str, "trial_id": str, "row": str})
    z = z[z["row"].isin(("base", "dev")) & z["person_id"].isin(set(persons))]
    out = {}
    for pid, g in z.groupby("person_id"):
        inside = g[g["run_inside"].astype(str).str.lower() == "true"]
        out[str(pid)] = {"D_over_S": float(g["D_over_S"].mean()),
                         "speed": float(np.concatenate([g["v_a"].to_numpy(float), g["v_b"].to_numpy(float)]).mean()),
                         "slide_share": float(inside["slide_share"].mean()) if len(inside) else None,
                         "rows": int(len(g))}
    return out


def foot_report_beside(tree, sets):
    a = foot_rows(tree, OUT["a5_diag"], sets["dev12"])
    b = foot_rows(tree, OUT["off_diag"], sets["dev12"])
    out = {"label": "reported, reading nothing", "row_label": ROW_LABEL, "persons": "dev12"}
    if a is None or b is None:
        out["reason"] = "a zupt_strides.csv is missing"
        return out
    wb = WB()
    for k in ("D_over_S", "speed", "slide_share"):
        pa = {p: x[k] for p, x in a.items()}
        pb = {p: x[k] for p, x in b.items()}
        d = wb.paired(pa, pb)             # 5a minus the code control
        va = [x for x in pa.values() if x is not None]
        vb = [x for x in pb.values() if x is not None]
        out[k] = {"a5_mean": float(np.mean(va)) if va else None, "off_mean": float(np.mean(vb)) if vb else None,
                  "a5_minus_off": {kk: d.get(kk) for kk in ("n", "point", "interval")}}
    return out


def replication_s2(tree, sets, locked, m):
    ref, why_r = diag_set_at(tree, REF["s2_diag_bw32"], sets, locked, S2_SHA)
    a5, why_a = diag_set_at(tree, OUT["s2_a5_diag"], sets, locked, S2_SHA)
    out = {"label": "reported, reading nothing (the effect on the second seed's B end)", "row_label": ROW_LABEL}
    if ref is None or a5 is None:
        out["reason"] = why_r or why_a
        return out
    out["R1_form"] = r1_reading(ref, a5, m)
    return out


def bars(tree):
    out = {"label": "bars beside, reading nothing"}
    z = read_json(os.path.join(tree, ZUPT_V2_RESULTS))
    out["zupt_keyed_dev12_cm"] = z["baseline_line_dev12"]
    out["zupt_keyed_dev14_cm"] = z["baseline_line_dev14"]
    zd = read_json(os.path.join(tree, ZUPT_DEV_RESULTS))
    out["zupt_g3_matched_dev12"] = {k: zd["lines"]["zupt"][k]["sets"]["dev12"].get(x) for k in
                                    ("stride_mae_cm", "speed_mae_m_s") for x in ("point",)}
    out["zupt_g3_matched_dev12_share"] = zd["lines"]["zupt"]["stride_mae_cm"]["sets"]["dev12"]["matched_share"]
    r4 = read_json(os.path.join(tree, F2L4_RESULTS))["R1"]
    out["lever4_end"] = {"S12": r4["S12"]["f2l4"], "V12": r4["V12"]["f2l4"]}
    return out


# ---------------------------------------------------------------- the reading
def run(tree=None, sets=None, locked=None, m=None, log=print):
    """Every reading of the rule; (results, exit code). RuleNotHashed while rule.sha256 is absent."""
    t0 = time.time()
    tree = tree or TREE
    rule_sha = check_rule(tree)
    if sets is None or locked is None:
        sets, locked = load_sets(tree)
    m = m if m is not None else margins(tree, sets, locked)
    res = {"what": LABEL, "row_label": ROW_LABEL, "task": TASK, "rule": {"path": RULE_REL.replace("\\", "/"),
                                                                         "sha256": rule_sha},
           "asserted_before_any_5a_value": True, "margins": m}
    res["R0"] = r0_reading(tree)
    ref, why_ref = diag_set_at(tree, REF["v1_diag"], sets, locked, V1_SHA)
    if ref is None:
        raise HarnessDefect(f"v1's diag base row: {why_ref}")
    a5, why_a5 = diag_set_at(tree, OUT["a5_diag"], sets, locked, V1_SHA)
    res["R1"] = r1_reading(ref, a5, m, why_a5)
    wb = WB()
    vals_ref = {"S14": ref["subsets"][wb.SUB14]["S_point"], "V14": ref["subsets"][wb.SUB14]["V_point"]}
    vals_a5, reasons = {}, {}
    if a5 is not None:
        vals_a5.update({"S14": a5["subsets"].get(wb.SUB14, {}).get("S_point"),
                        "V14": a5["subsets"].get(wb.SUB14, {}).get("V_point")})
    for k in ("S14", "V14"):
        if vals_a5.get(k) is None:
            reasons[k] = why_a5 or f"no {k} in the 5a diag base row"
    for k in ("IC", "MAPE"):
        vals_ref[k], _ = dev_line_at(tree, REF["v1_dev"], G3_LINE_OF[k], V1_SHA)
        vals_a5[k], reasons[k] = dev_line_at(tree, OUT["a5_dev"], G3_LINE_OF[k], V1_SHA)
    t = m["transfer"]
    g_ref, w_ref = gains_at(tree, REF["v1_diag"], sets, locked)
    g_a5, w_a5 = gains_at(tree, OUT["a5_diag"], sets, locked)
    slide = slide_reading(g_ref, g_a5, t["d_r"]["base"], t["d_legs"], "; ".join(w_ref + w_a5) or None)
    res["R2"] = r2_reading(vals_ref, vals_a5, m, reasons, slide)
    res["T6"] = t6_reading(tree, OUT["a5_diag"], sets, locked)
    v = verdict(res["R0"], res["R1"], res["R2"], res["T6"])
    res["verdict"], res["route"] = v, route(v)
    rep = {"label": "reported, reading nothing", "row_label": ROW_LABEL}
    off, why_off = diag_set_at(tree, OUT["off_diag"], sets, locked, V1_SHA)
    if off is not None:
        rep["against_the_code_control"] = {"R1": r1_reading(off, a5, m, why_a5) if a5 is not None else None,
                                           "S12_code_control": off["subsets"][wb.SUB12]["S_point"],
                                           "V12_code_control": off["subsets"][wb.SUB12]["V_point"]}
        g_off, w_off = gains_at(tree, OUT["off_diag"], sets, locked)
        rep["against_the_code_control"]["SLIDE"] = slide_reading(g_off, g_a5, t["d_r"]["base"], t["d_legs"],
                                                                 "; ".join(w_off + w_a5) or None)
    else:
        rep["against_the_code_control"] = {"reason": why_off}
    rep["foot_report"] = foot_report_beside(tree, sets)
    rep["replication_s2_B"] = replication_s2(tree, sets, locked, m)
    rep["bars"] = bars(tree)
    rep["term"] = ((read_json(os.path.join(tree, OUT["a5_diag"], "diag.json")) or {}).get("zupt_anchor") or {})
    dv = read_json(os.path.join(tree, OUT["a5_dev"], "g3_dev.json"))
    rv = read_json(os.path.join(tree, REF["v1_dev"], "g3_dev.json"))
    rep["all68"] = {k: {"v1": dev_model_line(rv, "all68", k), "a5": dev_model_line(dv, "all68", k)}
                    for k in ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct")}
    res["reported"] = rep
    res["timing_s"] = round(time.time() - t0, 1)
    log(f"verdict of the rule: {v['verdict']} (case {v['case']})")
    return res, (EXIT_NOT_EVALUATED if v["verdict"] == NOT_EVALUATED else EXIT_VERDICT)


# ---------------------------------------------------------------- outputs
def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def summary_md(r):
    v = r["verdict"]
    L = [f"# F2 lever 5a: the inertial anchor in the refinement, on v1's B end ({LABEL})", "",
         f"Written by scripts/f2_lever5a.py (task {TASK}) under reports/bench/f2_lever5a/rule.txt (SHA-256 "
         f"{r['rule']['sha256']}, asserted before any 5a value was read). Every 5a row is a '{ROW_LABEL}' row. No "
         "lockbox, EVAL, CODY or care_pd record was read; nothing here is a G3 read or enters a G3 verdict.", "",
         f"## Verdict of the rule: **{v['verdict']}** (case {v['case']})", "",
         "Reasons: " + ("; ".join(v["reasons"]) or "none") + ".", "", "## Route", "", r["route"], "",
         "## R0", "", f"Holds: {r['R0']['holds']}. Failures: {'; '.join(r['R0']['fail']) or 'none'}.", "",
         "| check | ok |", "|---|---|"]
    L += [f"| {k} | {x} |" for k, x in r["R0"]["checks"].items()]
    L += ["", "## R1, primary lines (dev12, paired v1 minus 5a)", "",
          "| line | v1 | 5a | improvement | 95 % | persons | margin | meets | reason |", "|---|---|---|---|---|---|---|---|---|"]
    for k, x in r["R1"].items():
        p = x.get("improvement") or {}
        L.append(f"| {k} ({x['unit']}) | {fmt(x['v1'], ND[k])} | {fmt(x['a5'], ND[k])} | {fmt(p.get('point'), ND[k])} | "
                 f"{fmt_iv(p.get('interval'), ND[k])} | {p.get('n', 0)} | {fmt(x['margin'], ND[k])} | {x['meets']} | "
                 f"{x['reason'] or ''} |")
    L += ["", "## R2, non-inferiority (5a minus v1; positive is worse) and the stance-slide guard", "",
          "| line | v1 | 5a | difference | margin | holds | reason |", "|---|---|---|---|---|---|---|"]
    for k in ("S14", "V14", "IC", "MAPE"):
        x = r["R2"][k]
        L.append(f"| {k} ({x['unit']}) | {fmt(x['v1'], ND[k])} | {fmt(x['a5'], ND[k])} | {fmt(x['difference'], ND[k])} | "
                 f"{fmt(x['margin'], ND[k])} | {x['holds']} | {x['reason'] or ''} |")
    s = r["R2"]["SLIDE"]
    L += ["", f"SLIDE: base-row dev12 stride gain v1 {fmt((s['v1'] or {}).get('base'), 4)}, 5a "
          f"{fmt((s['a5'] or {}).get('base'), 4)} (rise {fmt(s.get('base_rise'), 4)}); model_legs v1 "
          f"{fmt((s['v1'] or {}).get(LEGS_ROW), 4)}, 5a {fmt((s['a5'] or {}).get(LEGS_ROW), 4)} (rise "
          f"{fmt(s.get('legs_rise'), 4)}); d_base {fmt(s['d_base'], 4)}, d_legs {fmt(s['d_legs'], 4)}: "
          f"{s.get('status') or 'missing'}" + (f" ({s['reason']})" if s.get("reason") else "") + ".", ""]
    t6 = r["T6"]
    L += [f"T6 BEYOND THE INPUT: 5a decoded over ZUPT {fmt(t6['ends']['a5']['decoded_over_zupt'], 3)} on "
          f"{t6['ends']['a5']['strides']} strides (v1 {fmt(t6['ends']['v1_B']['decoded_over_zupt'], 3)}); threshold "
          f"{BEYOND}: {t6['status']}.", ""]
    rep = r["reported"]
    L += ["## Reported beside, reading nothing", ""]
    cc = rep["against_the_code_control"]
    if cc.get("R1"):
        L.append(f"- Against the code control (v1_B, --zupt-report, the same code): S(12) control "
                 f"{fmt(cc['S12_code_control'])} cm, improvement {fmt(cc['R1']['S12']['improvement']['point'])} "
                 f"{fmt_iv(cc['R1']['S12']['improvement']['interval'])}; V(12) control {fmt(cc['V12_code_control'], 4)} "
                 f"m/s, improvement {fmt(cc['R1']['V12']['improvement']['point'], 4)}; SLIDE {cc['SLIDE'].get('status')}.")
    fr = rep["foot_report"]
    if "reason" not in fr:
        L.append("- Foot report (dev12 person means, 5a against the code control): " + "; ".join(
            f"{k} {fmt(fr[k]['a5_mean'], 3)} against {fmt(fr[k]['off_mean'], 3)} (paired "
            f"{fmt(fr[k]['a5_minus_off']['point'], 3)} {fmt_iv(fr[k]['a5_minus_off']['interval'], 3)})"
            for k in ("D_over_S", "speed", "slide_share")) + ".")
    rp = rep["replication_s2_B"]
    if rp.get("R1_form"):
        L.append(f"- s2_B replication: S(12) {fmt(rp['R1_form']['S12']['v1'])} to {fmt(rp['R1_form']['S12']['a5'])} cm, "
                 f"V(12) {fmt(rp['R1_form']['V12']['v1'], 4)} to {fmt(rp['R1_form']['V12']['a5'], 4)} m/s.")
    else:
        L.append(f"- s2_B replication: {rp.get('reason')}.")
    b = rep["bars"]
    L.append(f"- Bars: version-2 ZUPT readout dev12 {fmt(b['zupt_keyed_dev12_cm']['point_cm'])} cm (walkway-timed, "
             f"matched share {fmt(b['zupt_keyed_dev12_cm']['matched_share'], 3)}); G3-matched ZUPT line "
             f"{fmt(b['zupt_g3_matched_dev12']['stride_mae_cm'])} cm (share {fmt(b['zupt_g3_matched_dev12_share'], 3)});"
             f" lever 4's end S(12) {fmt(b['lever4_end']['S12'])} cm, V(12) {fmt(b['lever4_end']['V12'], 4)} m/s.")
    L.append("")
    return "\n".join(L) + "\n"


def write_outputs(out_dir, res, started):
    """results.json, summary.md and _provenance.json; never over an existing results.json (FileExistsError)."""
    from mdfm.gates import common
    path = os.path.join(out_dir, "results.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(path, json.dumps(F2().clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    prov = {"script": "scripts/f2_lever5a.py", "task": TASK, "label": LABEL, "row_label": ROW_LABEL, "run_at": started,
            "python": sys.version.split()[0], "moveatlas_root": paths.MOVEATLAS,
            "code_sha256": common.code_sha256([os.path.abspath(__file__)] + [os.path.join(SCRIPTS, f"{n}.py") for n in
                                                                            ("f2_lever5", "f2_lever4", "f2_lever2",
                                                                             "weekend_branch", "g3_gain_check")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    try:
        prov["moveatlas_stamp"] = GG().stamp_moveatlas()
    except Exception as e:  # noqa: BLE001  (reported, never required)
        prov["moveatlas_stamp"] = {"error": str(e)}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(F2().clean(prov), indent=1) + "\n")
    return path


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rule-numbers", action="store_true", help="print the numbers the rule prints")
    p.add_argument("--out", default=None, help=f"output folder (default {BENCH})")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.rule_numbers:
        nums, checks = rule_numbers()
        for k, x in nums.items():
            print(f"{k}\t{x}")
        print(json.dumps(checks, indent=1))
        return 0
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        res, code = run()
    except RuleNotHashed as e:
        print(f"NOT EVALUATED: {e}")
        return EXIT_NOT_EVALUATED
    except HarnessDefect as e:
        print(f"harness defect: {e}")
        return EXIT_HARNESS
    try:
        write_outputs(a.out or os.path.join(TREE, BENCH), res, started)
    except FileExistsError as e:
        print(f"an output exists: {e}")
        return EXIT_EXISTS
    print(summary_md(res))
    return code


if __name__ == "__main__":
    sys.exit(main())
