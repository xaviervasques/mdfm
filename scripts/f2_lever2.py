"""F2R (2026-10-05): the readings of the F2 lever-2 rule, the phase-B-only continuation of v1 with amplitude
randomization of the synthetic IMU renders at the lower back and both thighs (run cache/runs/pilot_f2l2). Label of
every output: dev diagnostic, not a G3 read.

Rule: reports/bench/f2_lever2/rule.txt (SHA-256 in rule.sha256 beside it and in RULE_SHA256 here) with
rule_amendment_1.txt (SHA-256 in rule_amendment_1.sha256 and AMEND_SHA256). check_rules asserts that the six agree
before any f2l2 value is read (run calls it first). Written before any input of the run existed.

Inputs (rule.txt section 4; folder names from the post-train templates of cache/runs/queue/post_train/ and the job
files of group F2L2), every path under --tree (default the tree root):
- f2l2_B (name pilot_f2l2_ckpt_B_032044): reports/gates/g3/diag_<name>/diag.json and events.csv (post_diag),
  reports/gates/g3/dev_<name>/g3_dev.json, reports/gates/g1/dev_<name>/g1.json and reports/gates/g3_sim/insim_<name>/
  (post_j1); cache/runs/pilot_f2l2/log.jsonl; reports/gpu_queue/job_log.csv;
- v1_B (name phaseA_root_full_ckpt_B_032044): the same report paths, and cache/runs/phaseA_root_full/log.jsonl;
- s2_B (pilot_v1_s2_ckpt_B_032044): g3_dev.json and g1.json, for the seed differences of the IC, G1a and
  stride_mape_pct margins; reports/gpu_queue/branch_seeds_20261002_211426.json for the others;
- early (reported only): reports/gates/g3/curve_<run>_ckpt_B_004000/ (post_curve), reports/gates/g1/
  dev_<run>_ckpt_B_004000_lt2/ and reports/gates/g3_sim/insim_<run>_ckpt_B_004000/ (the two F2L2 jobs), for run
  pilot_f2l2 and phaseA_root_full;
- beside: reports/bench/f2_scale/H4root/dumps_2026-10-05/results.json (the ceilings of rule section 7).
G3 and G1 folders are read with scripts/weekend_branch.py (g3_sources, g3_line, g3set_from_folder, g1_reads, paired,
the v2check bootstrap: 10,000 resamples in blocks of 1,000, numpy default_rng(20261001)) and scripts/g3_gain_check.py
(read_events, median_ratio, insim_gain), by import.

Margins (margins_from): computed from the v1 and s2 files as tests/test_f2_lever2_rule.py computes them, then
asserted equal to the numbers that rule.txt section 7 and amendment A1 print, at their printed precision
(rule_number_check); a departure is a harness defect (exit EXIT_HARNESS), so no criterion can move after an f2l2
number is seen.

Readings (rule section 8, amendment A1 and A2):
- R0 (r0_reading): the run's log and job row. Stop cases (the run stopped on an alarm, a non-finite stop of
  NONFINITE_RUN events at consecutive optimizer steps, or a peak reserved memory above PEAK_RESERVED_MAX_GB) against
  every other failure (no log, an init_from source or a start config other than section 1's, a render_aug entry other
  than section 1's, report_mismatches not empty, no phase B verify event at step 32,044, no job_log.csv row or a last
  exit other than 0). There must be at least one init_from event and every one must name the source of section 1
  (a crash resume logs 'resume', not 'init_from').
- R1 (r1_reading): S(12) and V(12), the diag base row (subset no_whc003_whc026), paired v1 minus f2l2 per person;
  MEETS when the point is at least m_L (within TOL) and the lower bound above 0 (beyond TOL).
- R2 (r2_reading): f2l2 minus v1 points of S(14), V(14) (diag base, all14), the walkway IC (g3_dev.json
  ic_median_ms_walkway), G1a (full G1 dev run) and, under the amendment, stride_mape_pct (g3_dev.json); HOLDS when
  at most the margin (within TOL).
- Verdict, both readings (rule.txt section 10 and the amendment's preamble): verdict_rule_only applies rule.txt
  section 8 with its four R2 lines; a case where its NO-GO and NOT EVALUATED conditions both apply is reported as
  unordered. verdict_amended applies the order of amendment A2 to the five R2 lines; it is the verdict of the rule.
  Where they differ, the output says so.
- Route of rule section 9 (route), with the input-scaling probe of v1_B beside it (probe_reading, probe_sentence:
  reports/bench/f2_lever2/probe_phaseA_root_full_ckpt_B_032044/results.json when it exists, its stated SHA-256
  values checked against probe_prediction.txt and probe_prediction_amendment_1.txt). The probe decides neither the
  verdict nor the route (probe_prediction_amendment_1.txt P1): under a GO, its NOT THE AMPLITUDE PATH names the
  mechanism only, and the GO route of section 9 and the owner's decision stand.
- Reported, reading nothing: in-sim stride and speed gains and the in-sim forward stride MAE of both ends, the dev12
  decoded-over-walkway gains, the distances of S(12) and V(12) to the pass and kill lines, stride_mape_pct (under
  rule.txt alone) and the insole IC line beside its seed difference, the seconds per phase B step (J1's definition:
  the sum of wall_s over the sum of optimizer_steps of the phase B log records) against v1's, the imu_nll of
  amendment A3 (median over the phase B log records per block of BLOCK_STEPS steps; dev.ema.terms.imu_nll at the dev
  events of every multiple of BLOCK_STEPS and the end), the early read at ckpt_B_004000, and v1's in-sim and dev12
  gains with the ceilings of the H4 dump.

Writes <out>/results.json, summary.md and _provenance.json (out: reports/bench/f2_lever2/ by default) when the rule's
verdict is reached or --write is given; it never rewrites an existing results.json (EXIT_EXISTS; pass --out for a new
folder). Exit codes: EXIT_VERDICT (0) GO, PARTIAL or NO-GO; EXIT_NOT_EVALUATED (3) an input or event is missing,
listed on the output; EXIT_HARNESS (4) a margin differs from the rule's printed number or a persons guard failed;
EXIT_EXISTS (5).

Data discipline: dev14 and dev12 from reports/bench/g3_gain_check/results.json, asserted equal to the SHA-256 that
rule.txt section 3 quotes and disjoint from mdfm.gates.g3.lockbox_persons(); the person keys of every diag row read
are guarded (dev14 only, none in the lockbox list). No lockbox, EVAL, CODY or care_pd record; CPU only
(CUDA_VISIBLE_DEVICES=-1 is set before torch can load).

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever2.py [--write] [--out DIR]
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the readings never see the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
BENCH = os.path.join("reports", "bench", "f2_lever2")
RULE_REL = os.path.join(BENCH, "rule.txt")
RULE_SHA_REL = os.path.join(BENCH, "rule.sha256")
AMEND_REL = os.path.join(BENCH, "rule_amendment_1.txt")
AMEND_SHA_REL = os.path.join(BENCH, "rule_amendment_1.sha256")
RULE_SHA256 = "f368c29e25c6a05fbe7a7fd811a86493ac6d93c27c4ddc8a2176adb5857efb49"
AMEND_SHA256 = "9ba9a6f3ecd2567381222a57c80aded4d3c0aef99834c7034735f8c7c645a43d"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2R"
RUN = "pilot_f2l2"
RUN_DIR = "cache/runs/pilot_f2l2"
V1_RUN_DIR = "cache/runs/phaseA_root_full"
F2L2_B = "pilot_f2l2_ckpt_B_032044"
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
F2L2_4000 = "pilot_f2l2_ckpt_B_004000"
V1_4000 = "phaseA_root_full_ckpt_B_004000"
CONFIG_SHA256 = "26503b1edfe1fe3f9f2161d0687ed482fc6d790fbe458a23af3d73a3f312ca6a"
SOURCE_PATH = "cache/runs/phaseA_root_full/ckpt_A_010681.pt"
SOURCE_SHA256 = "ad224565725e6ec973d9b4b2af0e67d2c658241d8644b7d879a8730d70fb5c10"
RENDER_AUG = {"enabled": True, "sites": ["lower_back", "thigh_L", "thigh_R"], "gyro_log_max": 0.5,
              "acc_log_max": 0.5}
PHASE_B_STEPS = 32044
PEAK_RESERVED_MAX_GB = 14.0
NONFINITE_RUN = 3
TOL = 1e-9
TARGET_FRACTION, OTHER_FRACTION, SEED_MULTIPLE = 0.10, 0.05, 2.0
BLOCK_STEPS = 4000
V1_S_PER_STEP_J1 = 2.416            # reports/weekend_2026-10-01/tasks/J1.md, output [A] (quoted by rule section 5)
DEV14_SHA256 = "31cd3cd41c548cc467275bc1556b0a9f8ba11fe6dc5a3fa3c4acd2bb7acd4267"
DEV12_SHA256 = "09f1173e34474ae9053a2e74efa1680c07ca38c664fbd30e7c0f01481355338a"
SEEDS_REL = os.path.join("reports", "gpu_queue", "branch_seeds_20261002_211426.json")
F1_RESULTS_REL = os.path.join("reports", "bench", "g3_gain_check", "results.json")
H4_DUMP_REL = os.path.join("reports", "bench", "f2_scale", "H4root", "dumps_2026-10-05", "results.json")
JOB_LOG_REL = os.path.join("reports", "gpu_queue", "job_log.csv")
# the input-scaling probe of v1_B (scripts/f2_input_scaling_probe.py), reported beside the route and never deciding it
# (probe_prediction_amendment_1.txt P1)
PROBE_RESULTS_REL = os.path.join(BENCH, "probe_phaseA_root_full_ckpt_B_032044", "results.json")
PROBE_PREDICTION_SHA256 = "740b8f2d2bed9a0bfe06b024e141dbf6a6d2d065c57448d5cd025c8a287622a4"
PROBE_AMEND_SHA256 = "7aa8afaa2e13cc2e57b21ca1bca6f9175ee8d81fbdf74bb538ede799b8166892"
PRIMARY = ("S12", "V12")
R2_RULE = ("S14", "V14", "IC", "G1a")
R2_AMENDED = R2_RULE + ("MAPE",)
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "IC": "ms", "G1a": "mm", "MAPE": "percent",
        "IC_insole": "ms"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "IC": 2, "G1a": 2, "MAPE": 2, "IC_insole": 2}
G3_LINE_OF = {"IC": "ic_median_ms_walkway", "MAPE": "stride_mape_pct", "IC_insole": "ic_median_ms_insole"}
EXIT_VERDICT, EXIT_NOT_EVALUATED, EXIT_HARNESS, EXIT_EXISTS = 0, 3, 4, 5
NO_GO, GO, PARTIAL, NOT_EVALUATED = "NO-GO", "GO", "PARTIAL", "NOT EVALUATED"


class HarnessDefect(AssertionError):
    """A departure of the harness itself: a margin that differs from the rule's printed number, a person outside
    the expected set."""


def _sibling(name):
    key = f"_f2r_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def WB():
    """scripts/weekend_branch.py: G3 and G1 folder readers, paired bootstrap, run-log helpers."""
    return _sibling("weekend_branch")


def GG():
    """scripts/g3_gain_check.py (F1): events reader, median ratio, in-sim gain, MoveAtlas stamp."""
    return _sibling("g3_gain_check")


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:                                # another drive (tests on a temporary folder)
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def fin(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_json(path):
    """The parsed file, or None when it is absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the rule
def check_rules(tree=None, expected=None, expected_amend=None):
    """The SHA-256 of rule.txt and rule_amendment_1.txt under `tree`, each asserted equal to the first word of its
    hash file and to the script constant (module values at call time). Returns (rule sha, amendment sha)."""
    tree = tree or TREE
    out = []
    for rp, sp, want in ((RULE_REL, RULE_SHA_REL, RULE_SHA256 if expected is None else expected),
                         (AMEND_REL, AMEND_SHA_REL, AMEND_SHA256 if expected_amend is None else expected_amend)):
        got = sha256_file(os.path.join(tree, rp))
        with open(os.path.join(tree, sp), encoding="utf-8") as f:
            words = f.read().split()
        stored = words[0] if words else ""
        if not (got == stored == want):
            raise AssertionError(f"rule hash mismatch: {os.path.basename(rp)} {got}, {os.path.basename(sp)} "
                                 f"{stored or 'empty'}, script {want}")
        out.append(got)
    return tuple(out)


def rule_texts(tree=None):
    tree = tree or TREE
    with open(os.path.join(tree, RULE_REL), encoding="utf-8") as f:
        rule = f.read()
    with open(os.path.join(tree, AMEND_REL), encoding="utf-8") as f:
        amend = f.read()
    return rule, amend


# ---------------------------------------------------------------- sets
def load_sets(f1_doc, locked):
    """{dev14, dev12} of F1's results.json, asserted equal to the SHA-256 that rule.txt section 3 quotes, dev12 equal
    to dev14 without weargait:WHC003 and weargait:WHC026, none in the lockbox list."""
    from mdfm.data import sources
    s = f1_doc["sets"]
    sets = {"dev14": sorted(s["dev14"]), "dev12": sorted(s["dev12"])}
    for k, want in (("dev14", DEV14_SHA256), ("dev12", DEV12_SHA256)):
        if sources.person_hash(sets[k]) != want:
            raise HarnessDefect(f"{k}: SHA-256 differs from rule.txt section 3")
    if set(sets["dev12"]) != set(sets["dev14"]) - set(GG().EXCLUDED_FROM_12):
        raise HarnessDefect("dev12 is not dev14 without weargait:WHC003 and weargait:WHC026")
    if (set(sets["dev14"]) | set(sets["dev12"])) & set(locked):
        raise HarnessDefect("a person of dev14 is in the lockbox list")
    return sets


def guard_set(gset, sets, locked, what):
    """HarnessDefect unless every person of a G3 set (WB.g3set_from_diag_row or g3set_from_folder) lies in dev14
    (the 12 subset in dev12) and none in the lockbox list."""
    if gset is None:
        return
    wb = WB()
    for sub, allowed in ((wb.SUB14, sets["dev14"]), (wb.SUB12, sets["dev12"])):
        s = gset["subsets"].get(sub) or {}
        persons = set(s.get("S") or {}) | set(s.get("V") or {})
        if persons & set(locked):
            raise HarnessDefect(f"{what}: a lockbox person in the {sub} rows")
        out = sorted(persons - set(allowed))
        if out:
            raise HarnessDefect(f"{what}: persons outside the expected set in the {sub} rows ({', '.join(out[:3])})")


# ---------------------------------------------------------------- readers
def diag_dir(tree, name):
    return os.path.join(tree, "reports", "gates", "g3", f"diag_{name}")


def g3dev_path(tree, name):
    return os.path.join(tree, "reports", "gates", "g3", f"dev_{name}", "g3_dev.json")


def insim_dir(tree, name):
    return os.path.join(tree, "reports", "gates", "g3_sim", f"insim_{name}")


def checkpoint_issue(doc, name, sha, what):
    """None, or why a report is not of the checkpoint `name` (EMA weights; SHA-256 `sha` when known)."""
    ck = (doc or {}).get("checkpoint") or {}
    if name is not None and ck.get("name") not in (None, name):
        return f"{what}: checkpoint {ck.get('name')}, expected {name}"
    weights = ck.get("weights") or ((doc or {}).get("inference") or {}).get("weights")
    if weights not in (None, "ema"):
        return f"{what}: {weights} weights, expected EMA"
    if sha and ck.get("sha256") and ck.get("sha256") != sha:
        return f"{what}: checkpoint SHA-256 {str(ck.get('sha256'))[:12]}, expected {sha[:12]}"
    return None


def diag_set(tree, name, sets, locked, sha=None):
    """(the diag base row of a checkpoint as a G3 set, or None; reason when None). The persons are guarded."""
    wb = WB()
    path = os.path.join(diag_dir(tree, name), "diag.json")
    doc = read_json(path)
    if doc is None:
        return None, f"missing: {rel(path, tree)}"
    issue = checkpoint_issue(doc, name, sha, rel(path, tree))
    if issue:
        return None, issue
    s = wb.g3set_from_diag_row(((doc.get("rows") or {}).get("base")), path)
    if s is None:
        return None, f"no base row in {rel(path, tree)}"
    guard_set(s, sets, locked, rel(path, tree))
    return s, None


def g3dev_line(tree, name, line, sha=None):
    """(point, None) of a results_dev line of g3_dev.json, or (None, reason)."""
    path = g3dev_path(tree, name)
    doc = read_json(path)
    if doc is None:
        return None, f"missing: {rel(path, tree)}"
    issue = checkpoint_issue(doc, name, sha, rel(path, tree))
    if issue:
        return None, issue
    for r in doc.get("results_dev") or []:
        if isinstance(r, dict) and r.get("line") == line:
            v = fin(r.get("point"))
            return (v, None) if v is not None else (None, f"{line} not finite in {rel(path, tree)}")
    return None, f"no {line} line in {rel(path, tree)}"


def g1a_of(tree, name, variant="full", sha=None):
    """(the G1a read of a checkpoint in a variant (WB.g1_reads), None; reason when None)."""
    g = WB().g1_reads(tree, name).get(variant)
    if g is None or g.get("point") is None:
        return None, f"no G1 dev run ({variant}) of {name} under reports/gates/g1/"
    doc = read_json(g["json"])
    issue = checkpoint_issue(doc, None, sha, rel(g["json"], tree)) if doc else None
    if issue:
        return None, issue
    return g, None


def run_log(tree, run_dir):
    """(records, unparsed lines) of <run_dir>/log.jsonl; (None, 0) when absent."""
    return WB().read_jsonl(os.path.join(tree, run_dir, "log.jsonl"))


def verify_sha(recs):
    v = WB().b_end_verify(recs or [])
    return (v or {}).get("sha256")


# ---------------------------------------------------------------- margins
def margins_from(tree, sets, locked):
    """Every margin of rule.txt section 7 and amendment A1, computed from the v1 and s2 files as
    tests/test_f2_lever2_rule.py computes them: {lines, seed_diff, success, noninferiority, insole, early}. A missing
    file raises (the margins exist before the run)."""
    wb = WB()
    v1, why = diag_set(tree, V1_B, sets, locked)
    if v1 is None:
        raise FileNotFoundError(why)
    lines = {"S12": v1["subsets"][wb.SUB12]["S_point"], "V12": v1["subsets"][wb.SUB12]["V_point"],
             "S14": v1["subsets"][wb.SUB14]["S_point"], "V14": v1["subsets"][wb.SUB14]["V_point"]}
    seeds = read_json(os.path.join(tree, SEEDS_REL))
    if seeds is None:
        raise FileNotFoundError(SEEDS_REL)
    sd = {k: float(seeds["seed_diff"]["B_end"][k]["abs_diff"]) for k in ("S12", "V12", "S14", "V14")}
    s2 = {}
    for k in ("IC", "MAPE", "IC_insole"):
        a, ea = g3dev_line(tree, V1_B, G3_LINE_OF[k])
        b, eb = g3dev_line(tree, S2_B, G3_LINE_OF[k])
        if a is None or b is None:
            raise FileNotFoundError(ea or eb)
        lines[k], s2[k], sd[k] = a, b, abs(a - b)
    ga, ea = g1a_of(tree, V1_B)
    gb, eb = g1a_of(tree, S2_B)
    if ga is None or gb is None:
        raise FileNotFoundError(ea or eb)
    lines["G1a"], s2["G1a"], sd["G1a"] = ga["point"], gb["point"], abs(ga["point"] - gb["point"])
    success = {}
    for k in PRIMARY:
        m = max(TARGET_FRACTION * lines[k], SEED_MULTIPLE * sd[k])
        success[k] = {"line": lines[k], "ten_percent": TARGET_FRACTION * lines[k], "two_seed": SEED_MULTIPLE * sd[k],
                      "margin": m, "needed_line_at_most": lines[k] - m}
    ni = {k: {"line": lines[k], "seed_diff": sd[k], "five_percent": OTHER_FRACTION * lines[k],
              "two_seed": SEED_MULTIPLE * sd[k], "margin": max(OTHER_FRACTION * lines[k], SEED_MULTIPLE * sd[k])}
          for k in R2_AMENDED}
    early = {}
    c4 = wb.g3set_from_folder(os.path.join(tree, "reports", "gates", "g3", f"curve_{V1_4000}"))
    for k, key in (("S12", "S"), ("V12", "V")):
        v = (c4 or {}).get("subsets", {}).get(wb.SUB12, {}).get(f"{key}_point")
        s4 = fin(((seeds["seed_diff"].get("ckpt_B_004000") or {}).get(k) or {}).get("abs_diff"))
        early[k] = {"line": v, "seed_diff": s4,
                    "margin": None if v is None or s4 is None else max(TARGET_FRACTION * v, SEED_MULTIPLE * s4)}
    return {"lines": lines, "s2": s2, "seed_diff": sd, "success": success, "noninferiority": ni,
            "insole": {"line": lines["IC_insole"], "seed_diff": sd["IC_insole"]}, "early": early,
            "sources": {"v1_diag": v1["source"], "seeds": SEEDS_REL.replace("\\", "/")}}


def rule_number_check(m, rule_text, amend_text):
    """The strings that rule.txt section 7 and amendment A1 print for the margins, formatted from m; returns the
    list of those missing from the texts (empty: the margins the script reads are the registered ones)."""
    nl = chr(10)
    sec7 = rule_text[rule_text.index(nl + "7. Margins"):rule_text.index(nl + "8. Readings")]
    a1 = amend_text[amend_text.index(nl + "A1. "):amend_text.index(nl + "A2. ")]
    s, v = m["success"]["S12"], m["success"]["V12"]
    want7 = [f"{s['line']:.2f} cm", f"{s['margin']:.3f} cm", f"{s['two_seed']:.3f} cm", f"{v['line']:.3f} m/s",
             f"{v['margin']:.4f} m/s", f"{v['two_seed']:.4f} m/s", f"{s['needed_line_at_most']:.2f} cm",
             f"{v['needed_line_at_most']:.3f} m/s"]
    for k, unit, nd in (("S14", "cm", 2), ("V14", "m/s", 4), ("IC", "ms", 2), ("G1a", "mm", 2)):
        n = m["noninferiority"][k]
        want7 += [f"{n['line']:.{nd}f} {unit}", f"{n['margin']:.{nd}f} {unit}", f"d {n['seed_diff']:.{nd}f} {unit}"]
    for k, unit, nd in (("S12", "cm", 2), ("V12", "m/s", 4)):
        e = m["early"][k]
        if e["line"] is not None and e["margin"] is not None:
            want7 += [f"{e['line']:.{nd}f} {unit}", f"{e['margin']:.{nd}f} {unit}"]
        else:
            want7.append(f"early {k} line (absent)")
    n = m["noninferiority"]["MAPE"]
    want_a1 = [f"v1 {n['line']:.2f} percent", f"d {n['seed_diff']:.2f} percent", f"margin {n['margin']:.2f} percent",
               f"v1 {m['insole']['line']:.2f} ms", f"d {m['insole']['seed_diff']:.2f} ms",
               f"d {m['noninferiority']['IC']['seed_diff']:.2f} ms on the walkway line"]
    return [x for x in want7 if x not in sec7] + [x for x in want_a1 if x not in a1]


# ---------------------------------------------------------------- R0
def r0_reading(tree, recs, unparsed, job_log=None):
    """R0 of rule section 8: {holds, stop [reasons], fail [reasons], checks, values}. stop: the run stopped on an
    alarm, a non-finite stop or the memory bound; fail: every other R0 failure."""
    wb = WB()
    stop, fail, checks, values = [], [], {}, {"unparsed_log_lines": unparsed}
    if recs is None:
        fail.append(f"no run log {RUN_DIR}/log.jsonl")
        return {"holds": False, "stop": stop, "fail": fail, "checks": checks, "values": values}
    inits = wb.log_events(recs, "init_from")
    starts = wb.log_events(recs, "start")
    values["init_from_events"] = len(inits)
    values["start_events"] = len(starts)
    bad_init = []
    for r in inits:
        src = r.get("source") or {}
        path = str(src.get("path") or "").replace("\\", "/")
        if not path.endswith(SOURCE_PATH) or src.get("sha256") != SOURCE_SHA256:
            bad_init.append(f"{path or 'no path'} ({str(src.get('sha256'))[:12]})")
    checks["init_from"] = bool(inits) and not bad_init
    if not inits:
        fail.append("no init_from event")
    elif bad_init:
        fail.append("init_from source other than section 1: " + "; ".join(bad_init))
    bad_cfg = sorted({str(s.get("config_sha256")) for s in starts if s.get("config_sha256") != CONFIG_SHA256})
    bad_aug = [s.get("render_aug") for s in starts if s.get("render_aug") != RENDER_AUG]
    checks["start_config"] = bool(starts) and not bad_cfg
    checks["render_aug_enabled"] = bool(starts) and not bad_aug
    if not starts:
        fail.append("no start event")
    if bad_cfg:
        fail.append("start event at another config SHA-256: " + ", ".join(x[:12] for x in bad_cfg))
    if bad_aug:
        fail.append(f"start event whose render_aug is not section 1's: {bad_aug[0]}")
    v = wb.b_end_verify(recs)
    values["verify"] = None if v is None else {"step": v.get("step"), "sha256": v.get("sha256"),
                                               "path": str(v.get("path")).replace("\\", "/")}
    checks["verify_B_32044"] = bool(v is not None and v.get("phase") == "B" and v.get("step") == PHASE_B_STEPS)
    if not checks["verify_B_32044"]:
        fail.append(f"no phase B verify event at step {PHASE_B_STEPS}")
    mism = [s.get("report_mismatches") for s in starts if s.get("report_mismatches")]
    if v is not None and v.get("report_mismatches") != []:
        mism.append(v.get("report_mismatches", "absent from the verify event"))
    checks["report_mismatches_empty"] = not mism
    if mism:
        fail.append(f"report_mismatches not empty: {mism[0]}")
    alarms = wb.log_events(recs, "alarm")
    values["alarm_events"] = len(alarms)
    checks["no_alarm"] = not alarms
    if alarms:
        stop.append(f"alarm ({len(alarms)} alarm events)")
    nfs = wb.nonfinite_stop(recs, NONFINITE_RUN)
    values["nonfinite_events"] = len(wb.log_events(recs, "nonfinite"))
    checks["no_nonfinite_stop"] = not nfs
    if nfs:
        stop.append(f"non-finite stop ({NONFINITE_RUN} nonfinite events at consecutive optimizer steps)")
    peak = wb.peak_reserved_gb(recs)
    values["peak_reserved_gb"] = peak
    checks["peak_reserved"] = peak is not None and peak <= PEAK_RESERVED_MAX_GB + TOL
    if peak is None:
        fail.append("no memory_gb.max_reserved in the log")
    elif peak > PEAK_RESERVED_MAX_GB + TOL:
        stop.append(f"memory bound (peak reserved {peak:.2f} GB above {PEAK_RESERVED_MAX_GB})")
    code, jpath = wb.job_exit(tree, job_log or os.path.join(tree, JOB_LOG_REL), RUN_DIR)
    values["job_exit"] = code
    values["job_log"] = rel(jpath, tree) if jpath else None
    checks["job_exit_0"] = code == 0
    if code is None:
        fail.append(f"no job_log.csv row of a scripts/train_stage1.py job on {RUN_DIR}")
    elif code != 0:
        fail.append(f"last train_stage1.py job on {RUN_DIR} exited {code}")
    return {"holds": not stop and not fail, "stop": stop, "fail": fail, "checks": checks, "values": values}


# ---------------------------------------------------------------- R1 and R2
def r1_reading(v1_set, f2_set, m, why_missing=None):
    """{S12, V12: {read, v1, f2l2, improvement {n, point, interval}, margin, meets, reason}}; improvement v1 minus
    f2l2 (WB.paired, the v2check bootstrap)."""
    wb = WB()
    out = {}
    for L, key in (("S12", "S"), ("V12", "V")):
        margin = m["success"][L]["margin"]
        a_pp, a_pt = wb.g3_line(v1_set, key, wb.SUB12)
        b_pp, b_pt = wb.g3_line(f2_set, key, wb.SUB12)
        row = {"read": False, "v1": a_pt, "f2l2": b_pt, "improvement": None, "margin": margin, "meets": None,
               "unit": UNIT[L], "reason": None}
        if a_pp is None or b_pp is None:
            row["reason"] = why_missing or f"no {L} of the diag base row"
        else:
            p = wb.paired(a_pp, b_pp)
            row["improvement"] = {k: p[k] for k in ("n", "point", "interval")}
            if p["point"] is None or p["interval"] is None:
                row["reason"] = p.get("reason") or "fewer than 2 paired persons"
            else:
                row["read"] = True
                row["point_at_least_margin"] = bool(p["point"] >= margin - TOL)
                row["lower_above_0"] = bool(p["interval"][0] > TOL)
                row["meets"] = row["point_at_least_margin"] and row["lower_above_0"]
        out[L] = row
    return out


def r2_reading(values_v1, values_f2, m, reasons):
    """{line: {read, v1, f2l2, difference f2l2 minus v1, margin, holds, reason}} for the lines of R2_AMENDED;
    values_* map line -> point or None, reasons line -> why the f2l2 value is missing."""
    out = {}
    for k in R2_AMENDED:
        a, b = values_v1.get(k), values_f2.get(k)
        margin = m["noninferiority"][k]["margin"]
        row = {"read": a is not None and b is not None, "v1": a, "f2l2": b, "difference": None, "margin": margin,
               "holds": None, "unit": UNIT[k], "reason": None if (a is not None and b is not None) else
               (reasons.get(k) or f"no {k} value")}
        if row["read"]:
            row["difference"] = b - a
            row["holds"] = bool(b - a <= margin + TOL)
        out[k] = row
    return out


# ---------------------------------------------------------------- verdicts
def verdict_amended(r0, r1, r2, lines=R2_AMENDED):
    """The verdict of the rule: amendment A2's order over R0, R1 and the R2 lines `lines`. Returns {verdict, case,
    reasons, missing}."""
    if r0["stop"]:
        return {"verdict": NO_GO, "case": 1, "reasons": list(r0["stop"]), "missing": []}
    if r0["fail"]:
        return {"verdict": NOT_EVALUATED, "case": 2, "reasons": list(r0["fail"]), "missing": list(r0["fail"])}
    failed = [k for k in lines if r2[k]["read"] and not r2[k]["holds"]]
    if failed:
        return {"verdict": NO_GO, "case": 3, "reasons": [f"{k} FAILS" for k in failed],
                "missing": [f"{k}: {r2[k]['reason']}" for k in lines if not r2[k]["read"]]}
    if all(r1[L]["read"] for L in PRIMARY) and not any(r1[L]["meets"] for L in PRIMARY):
        return {"verdict": NO_GO, "case": 4, "reasons": ["neither S(12) nor V(12) MEETS"],
                "missing": [f"{k}: {r2[k]['reason']}" for k in lines if not r2[k]["read"]]}
    missing = ([f"{L}: {r1[L]['reason']}" for L in PRIMARY if not r1[L]["read"]]
               + [f"{k}: {r2[k]['reason']}" for k in lines if not r2[k]["read"]])
    if missing:
        return {"verdict": NOT_EVALUATED, "case": 5, "reasons": missing, "missing": missing}
    met = [L for L in PRIMARY if r1[L]["meets"]]
    return {"verdict": GO if len(met) == 2 else PARTIAL, "case": 6, "reasons": [f"{L} MEETS" for L in met],
            "missing": [], "met": met}


def verdict_rule_only(r0, r1, r2, lines=R2_RULE):
    """rule.txt section 8 alone, with its four R2 lines: NO-GO and NOT EVALUATED conditions collected; when both
    apply, the case is unordered under rule.txt alone and is reported as such."""
    nogo, ne = list(r0["stop"]), list(r0["fail"])
    nogo += [f"{k} FAILS" for k in lines if r2[k]["read"] and not r2[k]["holds"]]
    if all(r1[L]["read"] for L in PRIMARY) and not any(r1[L]["meets"] for L in PRIMARY):
        nogo.append("neither S(12) nor V(12) MEETS")
    ne += ([f"{L}: {r1[L]['reason']}" for L in PRIMARY if not r1[L]["read"]]
           + [f"{k}: {r2[k]['reason']}" for k in lines if not r2[k]["read"]])
    if nogo and ne:
        return {"verdict": f"UNORDERED ({NO_GO} or {NOT_EVALUATED})", "unordered": True, "no_go": nogo,
                "not_evaluated": ne}
    if nogo:
        return {"verdict": NO_GO, "unordered": False, "no_go": nogo, "not_evaluated": []}
    if ne:
        return {"verdict": NOT_EVALUATED, "unordered": False, "no_go": [], "not_evaluated": ne}
    met = [L for L in PRIMARY if r1[L]["meets"]]
    return {"verdict": GO if len(met) == 2 else PARTIAL, "unordered": False, "no_go": [], "not_evaluated": [],
            "met": met}


def probe_reading(tree=None):
    """The input-scaling probe of v1_B, reported beside the route (probe_prediction_amendment_1.txt P1): its
    registered consequence, the consequence under the note's gate (P2) and its primary reading, from
    PROBE_RESULTS_REL when it exists; the SHA-256 values that results.json states are checked against the texts'.
    It decides neither the verdict nor the route."""
    tree = tree or TREE
    path = os.path.join(tree, PROBE_RESULTS_REL)
    doc = read_json(path)
    if doc is None:
        return {"present": False, "reason": f"absent at this reading: {rel(path, tree)}"}
    issues = []
    if doc.get("prediction_sha256") != PROBE_PREDICTION_SHA256:
        issues.append(f"prediction SHA-256 {str(doc.get('prediction_sha256'))[:12]}, expected "
                      f"{PROBE_PREDICTION_SHA256[:12]}")
    if doc.get("amendment_1_sha256") != PROBE_AMEND_SHA256:
        issues.append(f"amendment SHA-256 {str(doc.get('amendment_1_sha256'))[:12]}, expected "
                      f"{PROBE_AMEND_SHA256[:12]}")
    cons, ng, prim = doc.get("consequence") or {}, doc.get("consequence_note_gate") or {}, doc.get("primary") or {}
    return {"present": True, "path": rel(path, tree), "issues": issues, "consequence": cons.get("key"),
            "arms": list(cons.get("arms") or []), "note_gate": ng.get("key"), "note_gate_differs": ng.get("differs"),
            "primary_reading": prim.get("reading"), "primary_rise": fin(prim.get("point")),
            "primary_interval": prim.get("interval"), "T": fin(prim.get("T"))}


def probe_sentence(verdict, probe):
    """The probe beside the route (probe_prediction_amendment_1.txt P1), as text."""
    probe = probe or {"present": False, "reason": "not read"}
    if not probe.get("present"):
        state = str(probe.get("reason"))
    elif probe.get("issues"):
        state = "present but not written under the registered texts (" + "; ".join(probe["issues"]) + "), not cited"
    else:
        state = (str(probe.get("consequence")) + (f" ({', '.join(probe['arms'])})" if probe.get("arms") else "")
                 + (f"; under the note's gate {probe.get('note_gate')}" if probe.get("note_gate_differs") else ""))
    s = ("Beside the route, the input-scaling probe of v1_B (reports/bench/f2_lever2/probe_prediction.txt with "
         f"probe_prediction_amendment_1.txt P1): {state}. It decides neither this verdict nor this route.")
    if verdict == GO:
        s += (" Under a GO, its NOT THE AMPLITUDE PATH names the mechanism only (another than the amplitude path at "
              "these sites as v1_B reads it: the down-weighting of amendment A3 or another), and the owner's decision "
              "stands.")
    elif verdict == NO_GO:
        s += (" Under a NO-GO, AMPLITUDE PATH says the handling failed, not the relevance, and NOT THE AMPLITUDE PATH "
              "sends the next lever.")
    else:
        s += " The owner decides whether the lever continues."
    return s


def route(verdict, met=None, missing=None, probe=None):
    """The route of rule section 9, as text, with the probe beside it (probe_sentence)."""
    if verdict == GO:
        r = ("GO: the f2l2 B end is proposed to the owner as the body of reference in place of v1's B end (owner "
             "decision; its selected numbers labelled 'post-selection, optimistic'). Before any lockbox use: its own "
             "train54 decode and, if the owner adopts the calibration head of lever 1, its own head. The readout "
             "levers of the F2 note are refit on the f2l2 end. Its mechanism is not named from dev numbers "
             "(amendment A3: the probe of A3 runs first).")
    elif verdict == PARTIAL:
        r = (f"PARTIAL: reported; the f2l2 end is not proposed; the line that met: {', '.join(met or [])}. v1's B "
             "end stays the reference.")
    elif verdict == NO_GO:
        r = ("NO-GO: the lever drops as a retrain lever. Before the next one (root-to-stance coupling with the "
             "supervised root as the anchor), the input-scaling probe of rule section 0 is read on v1_B "
             "(reports/bench/f2_lever2/probe_prediction.txt, scripts/f2_input_scaling_probe.py): if scaling the real "
             "channels of these sites moves the train54 stride gain, the amplitude path carries the loss and its "
             "handling, not its relevance, failed; if it does not, the next lever follows. The A3 probe (site "
             "log_scale under the gains) runs with it.")
    else:
        r = "NOT EVALUATED: nothing is proposed; missing: " + "; ".join(missing or []) + "."
    return r + " " + probe_sentence(verdict, probe)


# ---------------------------------------------------------------- reported, reading nothing
def insim_block(tree, name, sha=None):
    """In-sim gains (g3_gain_check.insim_gain) and the forward stride MAE of a checkpoint's insim folder, or the
    reason it is absent."""
    folder = insim_dir(tree, name)
    if not os.path.exists(os.path.join(folder, "insim.json")):
        return {"present": False, "reason": f"missing: {rel(folder, tree)}"}
    try:
        ig = GG().insim_gain(folder, sha)
    except ValueError as e:
        return {"present": False, "reason": str(e)}
    doc = read_json(os.path.join(folder, "insim.json"))
    mae = fin((((doc.get("lines") or {}).get("forward") or {}).get("model") or {}).get("stride_mae_cm", {})
              .get("point"))
    return {"present": True, "stride_gain": fin(ig["stride_gain"]), "speed_gain": fin(ig["speed_gain"]),
            "forward_stride_mae_cm": mae, "path": rel(folder, tree)}


def dev12_gains(tree, name, sets, locked):
    """Median decoded over walkway of the matched dev12 strides and passes of the diag events.csv (row base)."""
    g = GG()
    path = os.path.join(diag_dir(tree, name), "events.csv")
    if not os.path.exists(path):
        return {"present": False, "reason": f"missing: {rel(path, tree)}"}
    st, sp, _ = g.read_events(path, sets["dev14"], locked)
    st, sp = g.subset(st, sets["dev12"]), g.subset(sp, sets["dev12"])
    return {"present": True, "stride_gain": fin(g.median_ratio(st["x"], st["y"])),
            "speed_gain": fin(g.median_ratio(sp["x"], sp["y"])), "strides": int(len(st)), "passes": int(len(sp))}


def seconds_per_step(recs):
    """J1's definition: the sum of wall_s over the sum of optimizer_steps of the phase B 'log' records."""
    logs = [r for r in WB().log_events(recs or [], "log", "B")
            if fin(r.get("wall_s")) is not None and fin(r.get("optimizer_steps"))]
    steps = sum(float(r["optimizer_steps"]) for r in logs)
    return sum(float(r["wall_s"]) for r in logs) / steps if steps else None


def imu_nll_blocks(recs, total=PHASE_B_STEPS, block=BLOCK_STEPS):
    """{block label: median of terms.imu_nll over the phase B log records of the block}: blocks of `block` steps,
    the last one up to `total` (steps 1 to 4,000, ..., 28,001 to 32,044)."""
    n_blocks = max(1, total // block)
    acc = {}
    for r in WB().log_events(recs or [], "log", "B"):
        s, v = fin(r.get("step")), fin((r.get("terms") or {}).get("imu_nll"))
        if s is None or v is None or s < 1:
            continue
        k = min(int((s - 1) // block), n_blocks - 1)
        acc.setdefault(k, []).append(v)
    out = {}
    for k in sorted(acc):
        hi = total if k == n_blocks - 1 else (k + 1) * block
        out[f"{k * block + 1}-{hi}"] = float(np.median(acc[k]))
    return out


def imu_nll_dev(recs, total=PHASE_B_STEPS, block=BLOCK_STEPS):
    """{step: dev.ema.terms.imu_nll} at the phase B dev events of every multiple of `block` and the end."""
    out = {}
    for r in WB().log_events(recs or [], "dev", "B"):
        s = r.get("step")
        if isinstance(s, int) and (s % block == 0 or s == total):
            v = fin((((r.get("dev") or {}).get("ema") or {}).get("terms") or {}).get("imu_nll"))
            if v is not None:
                out[str(s)] = v
    return out


def early_read(tree, sets, locked, m):
    """Reported only: paired S(12) and V(12) improvements v1 minus f2l2 at ckpt_B_004000 (curve folders), the
    --limit-takes 2 G1a of both, the in-sim gains at step 4,000."""
    wb = WB()
    a = wb.g3set_from_folder(os.path.join(tree, "reports", "gates", "g3", f"curve_{V1_4000}"))
    b = wb.g3set_from_folder(os.path.join(tree, "reports", "gates", "g3", f"curve_{F2L2_4000}"))
    for s, what in ((a, V1_4000), (b, F2L2_4000)):
        guard_set(s, sets, locked, f"curve_{what}")
    out = {"label": "early, reported only", "lines": {}}
    for L, key in (("S12", "S"), ("V12", "V")):
        pa, xa = wb.g3_line(a, key, wb.SUB12)
        pb, xb = wb.g3_line(b, key, wb.SUB12)
        p = wb.paired(pa, pb) if pa is not None and pb is not None else {"n": 0, "point": None, "interval": None}
        mg = m["early"][L]["margin"]
        out["lines"][L] = {"v1": xa, "f2l2": xb, "improvement": {k: p.get(k) for k in ("n", "point", "interval")},
                           "margin": mg, "would_meet": None if p.get("point") is None or mg is None else
                           bool(p["point"] >= mg - TOL and p["interval"][0] > TOL)}
    ga = wb.g1_reads(tree, V1_4000).get("lt2")
    gb = wb.g1_reads(tree, F2L2_4000).get("lt2")
    if ga and gb:
        p = wb.paired(ga["takes"], gb["takes"], ga["knots"], gb["knots"])
        out["G1a_lt2"] = {"v1": ga["point"], "f2l2": gb["point"], "improvement_by_take":
                          {k: p.get(k) for k in ("n", "point", "interval")}}
    else:
        out["G1a_lt2"] = {"v1": (ga or {}).get("point"), "f2l2": (gb or {}).get("point"),
                          "reason": "a --limit-takes 2 G1 run is missing"}
    out["insim"] = {"v1": insim_block(tree, V1_4000), "f2l2": insim_block(tree, F2L2_4000)}
    return out


def ceilings(tree):
    """The ceilings of rule section 7 from the H4 dump (stride and pass-speed gains of train54, dev12 S if scaled)."""
    d = read_json(os.path.join(tree, H4_DUMP_REL))
    if d is None:
        return {"present": False}
    e = d["ends"]["v1_B"]["dump"]
    c = e["root_specific_ceiling_train54"]
    s = e["dev12_S_if_scaled"]
    return {"present": True, "train54_stride_gain_now": c["stride_now"], "train54_stride_gain_ceiling": c["stride"],
            "train54_speed_gain_now": c["speed_now"], "train54_speed_gain_ceiling": c["speed"],
            "dev12_S_root_specific_removed_cm": s["root_specific_removed"]["S_cm"],
            "dev12_S_whole_root_loss_removed_cm": s["sim_gain"]["S_cm"], "source": H4_DUMP_REL.replace("\\", "/")}


# ---------------------------------------------------------------- the run
def run(tree=None, sets=None, locked=None, margins=None, check_numbers=True, job_log=None, log=print):
    """Every reading of the rule; returns (results dict, exit code). No file written."""
    t0 = time.time()
    tree = tree or TREE
    rule_sha, amend_sha = check_rules(tree)
    wb = WB()
    res = {"what": LABEL, "task": TASK, "rule": {"path": RULE_REL.replace("\\", "/"), "sha256": rule_sha},
           "amendment_1": {"path": AMEND_REL.replace("\\", "/"), "sha256": amend_sha},
           "asserted_before_any_f2l2_value": True}
    m = margins if margins is not None else margins_from(tree, sets, locked)
    res["margins"] = m
    if check_numbers:
        rule_text, amend_text = rule_texts(tree)
        miss = rule_number_check(m, rule_text, amend_text)
        res["margins_check"] = {"missing_from_rule_text": miss}
        if miss:
            raise HarnessDefect(f"margins differ from the numbers the rule prints: {miss}")
    else:
        res["margins_check"] = {"skipped": True}
    recs, unparsed = run_log(tree, RUN_DIR)
    v1_recs, _ = run_log(tree, V1_RUN_DIR)
    f2_sha, v1_sha = verify_sha(recs), verify_sha(v1_recs)
    res["R0"] = r0_reading(tree, recs, unparsed, job_log)
    v1_set, why_v1 = diag_set(tree, V1_B, sets, locked, v1_sha)
    if v1_set is None:
        raise HarnessDefect(f"v1's diag base row: {why_v1}")
    f2_set, why_f2 = diag_set(tree, F2L2_B, sets, locked, f2_sha)
    res["R1"] = r1_reading(v1_set, f2_set, m, why_f2)
    vals_v1 = {"S14": v1_set["subsets"][wb.SUB14]["S_point"], "V14": v1_set["subsets"][wb.SUB14]["V_point"]}
    vals_f2, reasons = {}, {}
    if f2_set is not None:
        vals_f2.update({"S14": f2_set["subsets"].get(wb.SUB14, {}).get("S_point"),
                        "V14": f2_set["subsets"].get(wb.SUB14, {}).get("V_point")})
    for k in ("S14", "V14"):
        if vals_f2.get(k) is None:
            reasons[k] = why_f2 or f"no {k} in the diag base row"
    for k in ("IC", "MAPE"):
        vals_v1[k], _ = g3dev_line(tree, V1_B, G3_LINE_OF[k], v1_sha)
        vals_f2[k], reasons[k] = g3dev_line(tree, F2L2_B, G3_LINE_OF[k], f2_sha)
    g1v1, _ = g1a_of(tree, V1_B, sha=v1_sha)
    g1f2, reasons["G1a"] = g1a_of(tree, F2L2_B, sha=f2_sha)
    vals_v1["G1a"], vals_f2["G1a"] = (g1v1 or {}).get("point"), (g1f2 or {}).get("point")
    res["R2"] = r2_reading(vals_v1, vals_f2, m, reasons)
    va, vr = verdict_amended(res["R0"], res["R1"], res["R2"]), verdict_rule_only(res["R0"], res["R1"], res["R2"])
    differ = va["verdict"] != vr["verdict"]
    res["verdict"] = {"rule": va["verdict"], "amended": va, "rule_txt_alone": vr, "differ": differ,
                      "statement": ("the verdict under rule.txt alone differs from the verdict of the rule (rule.txt "
                                    "with amendment 1)" if differ else
                                    "the verdict under rule.txt alone equals the verdict of the rule")}
    probe = probe_reading(tree)
    res["route"] = route(va["verdict"], va.get("met"), va.get("missing"), probe)
    # reported, reading nothing
    rep = {"label": "reported, reading nothing"}
    rep["insim"] = {"f2l2_B": insim_block(tree, F2L2_B, f2_sha), "v1_B": insim_block(tree, V1_B, v1_sha)}
    rep["dev12_gains"] = {"f2l2_B": dev12_gains(tree, F2L2_B, sets, locked), "v1_B": dev12_gains(tree, V1_B, sets,
                                                                                                   locked)}
    dist = {}
    for L, line in (("S12", "stride_mae_cm"), ("V12", "speed_mae_m_s")):
        for who in ("v1", "f2l2"):
            x = res["R1"][L][who]
            dist[f"{L} {who}"] = None if x is None else {
                "value": x, "minus_pass": x - wb.LINES[line]["pass"], "minus_kill": x - wb.LINES[line]["kill"]}
    rep["distances_to_g3_lines"] = dist
    mp = res["R2"]["MAPE"]
    rep["stride_mape_pct_under_rule_txt_alone"] = {"v1": mp["v1"], "f2l2": mp["f2l2"], "difference": mp["difference"]}
    ins_v1, _ = g3dev_line(tree, V1_B, G3_LINE_OF["IC_insole"], v1_sha)
    ins_f2, ins_why = g3dev_line(tree, F2L2_B, G3_LINE_OF["IC_insole"], f2_sha)
    rep["ic_insole"] = {"v1": ins_v1, "f2l2": ins_f2, "difference": None if ins_f2 is None or ins_v1 is None else
                        ins_f2 - ins_v1, "seed_diff": m["insole"]["seed_diff"], "reason": ins_why}
    rep["seconds_per_phase_b_step"] = {"f2l2": seconds_per_step(recs), "v1": seconds_per_step(v1_recs),
                                       "v1_quoted_J1": V1_S_PER_STEP_J1}
    rep["imu_nll_a3"] = {"train_median_per_block": {"f2l2": imu_nll_blocks(recs), "v1": imu_nll_blocks(v1_recs)},
                         "dev_ema": {"f2l2": imu_nll_dev(recs), "v1": imu_nll_dev(v1_recs)},
                         "note": "amendment A3: whether the likelihood paid for the gains; it does not separate the body "
                                 "path from the scale path"}
    res["reported"] = rep
    res["early"] = early_read(tree, sets, locked, m)
    res["beside"] = {"ceilings": ceilings(tree),
                     "v1_B_insim": rep["insim"]["v1_B"], "v1_B_dev12": rep["dev12_gains"]["v1_B"],
                     "input_scaling_probe": dict(probe, label="beside the route, deciding nothing "
                                                              "(probe_prediction_amendment_1.txt P1)")}
    res["timing_s"] = round(time.time() - t0, 1)
    code = EXIT_NOT_EVALUATED if va["verdict"] == NOT_EVALUATED else EXIT_VERDICT
    log(f"verdict of the rule: {va['verdict']} (case {va['case']}); rule.txt alone: {vr['verdict']}")
    return res, code


# ---------------------------------------------------------------- report
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


def summary_md(r):
    v = r["verdict"]
    L = [f"# F2 lever 2: the amplitude-randomized phase B continuation against v1 ({LABEL})", "",
         f"Written by scripts/f2_lever2.py (task {TASK}) under reports/bench/f2_lever2/rule.txt (SHA-256 "
         f"{r['rule']['sha256']}) and rule_amendment_1.txt (SHA-256 {r['amendment_1']['sha256']}), both asserted "
         "before any f2l2 value was read. Every number below is read from results.json of this folder. No lockbox, "
         "EVAL, CODY or care_pd record was read. Nothing here is a G3 read or enters a G3 verdict.", "",
         f"## Verdict of the rule: **{v['rule']}** (amendment A2, case {v['amended']['case']})", "",
         f"Under rule.txt alone: **{v['rule_txt_alone']['verdict']}**. {v['statement'][0].upper()}{v['statement'][1:]}.",
         "", "Reasons: " + ("; ".join(v["amended"]["reasons"]) or "none") + ".", "", "## Route", "", r["route"], "",
         "## R0, integrity of the run", "",
         f"Holds: {r['R0']['holds']}. Stop cases: {'; '.join(r['R0']['stop']) or 'none'}. Other failures: "
         f"{'; '.join(r['R0']['fail']) or 'none'}.", "",
         "| check | ok |", "|---|---|"]
    L += [f"| {k} | {x} |" for k, x in r["R0"]["checks"].items()]
    vals = r["R0"]["values"]
    L += ["", "Values: " + "; ".join(f"{k} {x}" for k, x in vals.items()) + ".", "",
          "## R1, primary lines (dev12, raw decoded body, paired v1 minus f2l2)", "",
          "| line | v1 | f2l2 | improvement | 95 % | persons | margin | meets | reason |", "|---|---|---|---|---|---|---|---|---|"]
    for k, x in r["R1"].items():
        nd = ND[k]
        p = x.get("improvement") or {}
        L.append(f"| {k} ({x['unit']}) | {fmt(x['v1'], nd)} | {fmt(x['f2l2'], nd)} | {fmt(p.get('point'), nd)} | "
                 f"{fmt_iv(p.get('interval'), nd)} | {p.get('n', 0)} | {fmt(x['margin'], nd)} | {x['meets']} | "
                 f"{x['reason'] or ''} |")
    L += ["", "## R2, non-inferiority (f2l2 minus v1; positive is worse)", "",
          "| line | v1 | f2l2 | difference | margin | holds | decides under | reason |", "|---|---|---|---|---|---|---|---|"]
    for k, x in r["R2"].items():
        nd = ND[k]
        L.append(f"| {k} ({x['unit']}) | {fmt(x['v1'], nd)} | {fmt(x['f2l2'], nd)} | {fmt(x['difference'], nd)} | "
                 f"{fmt(x['margin'], nd)} | {x['holds']} | {'amendment A1' if k == 'MAPE' else 'both readings'} | "
                 f"{x['reason'] or ''} |")
    rep = r["reported"]
    L += ["", "## Reported, reading nothing", ""]
    for who in ("f2l2_B", "v1_B"):
        ins = rep["insim"][who]
        dg = rep["dev12_gains"][who]
        L.append(f"- {who}: in-sim stride gain {fmt(ins.get('stride_gain'), 4)}, speed gain {fmt(ins.get('speed_gain'), 4)}, "
                 f"forward stride MAE {fmt(ins.get('forward_stride_mae_cm'), 2)} cm"
                 + ("" if ins.get("present") else f" ({ins.get('reason')})")
                 + f"; dev12 decoded over walkway: strides {fmt(dg.get('stride_gain'), 4)}, passes "
                 f"{fmt(dg.get('speed_gain'), 4)}" + ("" if dg.get("present") else f" ({dg.get('reason')})") + ".")
    for k, x in rep["distances_to_g3_lines"].items():
        if x:
            L.append(f"- {k}: {fmt(x['value'], 4)}; minus the pass line {x['minus_pass']:+.4f}, minus the kill line "
                     f"{x['minus_kill']:+.4f}.")
    mp, ic = rep["stride_mape_pct_under_rule_txt_alone"], rep["ic_insole"]
    L += [f"- stride_mape_pct (reported under rule.txt alone, R2 under amendment A1): v1 {fmt(mp['v1'])}, f2l2 "
          f"{fmt(mp['f2l2'])}, difference {fmt(mp['difference'])} percent.",
          f"- Insole IC (decides nothing, G3 rule section 6): v1 {fmt(ic['v1'])}, f2l2 {fmt(ic['f2l2'])} ms, difference "
          f"{fmt(ic['difference'])} ms beside the seed difference {fmt(ic['seed_diff'])} ms.",
          f"- Seconds per phase B step: f2l2 {fmt(rep['seconds_per_phase_b_step']['f2l2'], 3)}, v1 "
          f"{fmt(rep['seconds_per_phase_b_step']['v1'], 3)} (J1 quotes {V1_S_PER_STEP_J1})."]
    a3 = rep["imu_nll_a3"]
    L += ["", "Amendment A3, imu_nll (training, median per block; dev EMA at the dev events):", "",
          "| block | f2l2 train | v1 train |", "|---|---|---|"]
    tb = a3["train_median_per_block"]
    for blk in sorted(set(tb["f2l2"]) | set(tb["v1"]), key=lambda s: int(s.split("-")[0])):
        L.append(f"| {blk} | {fmt(tb['f2l2'].get(blk), 1)} | {fmt(tb['v1'].get(blk), 1)} |")
    L += ["", "| step | f2l2 dev | v1 dev |", "|---|---|---|"]
    dv = a3["dev_ema"]
    for s in sorted(set(dv["f2l2"]) | set(dv["v1"]), key=int):
        L.append(f"| {s} | {fmt(dv['f2l2'].get(s), 1)} | {fmt(dv['v1'].get(s), 1)} |")
    e = r["early"]
    L += ["", f"## Early read at ckpt_B_004000 ({e['label']})", ""]
    for k, x in e["lines"].items():
        nd = ND[k]
        p = x["improvement"]
        L.append(f"- {k}: v1 {fmt(x['v1'], nd)}, f2l2 {fmt(x['f2l2'], nd)}, improvement {fmt(p.get('point'), nd)} "
                 f"{fmt_iv(p.get('interval'), nd)} against the early margin {fmt(x['margin'], nd)}: would meet "
                 f"{x['would_meet']}.")
    g = e["G1a_lt2"]
    L.append(f"- G1a (--limit-takes 2): v1 {fmt(g.get('v1'))}, f2l2 {fmt(g.get('f2l2'))} mm"
             + (f"; paired by take {fmt((g.get('improvement_by_take') or {}).get('point'))} mm" if
                g.get("improvement_by_take") else f" ({g.get('reason')})") + ".")
    for who in ("v1", "f2l2"):
        ins = e["insim"][who]
        L.append(f"- in-sim at step 4,000, {who}: stride gain {fmt(ins.get('stride_gain'), 4)}, speed gain "
                 f"{fmt(ins.get('speed_gain'), 4)}" + ("" if ins.get("present") else f" ({ins.get('reason')})") + ".")
    c = r["beside"]["ceilings"]
    if c.get("present"):
        L += ["", "## Beside: the dumps' arithmetic (rule section 7)", "",
              f"train54 stride gain {fmt(c['train54_stride_gain_now'], 4)} now, root-specific ceiling "
              f"{fmt(c['train54_stride_gain_ceiling'], 4)}; pass speed {fmt(c['train54_speed_gain_now'], 4)} to "
              f"{fmt(c['train54_speed_gain_ceiling'], 4)}; dev12 S at the ceiling "
              f"{fmt(c['dev12_S_root_specific_removed_cm'])} cm, with the whole root loss removed "
              f"{fmt(c['dev12_S_whole_root_loss_removed_cm'])} cm ({c['source']})."]
    pb = r["beside"].get("input_scaling_probe") or {}
    L += ["", "## Beside: the input-scaling probe of v1_B (deciding nothing, probe_prediction_amendment_1.txt P1)", ""]
    if pb.get("present"):
        L.append(f"Consequence {pb.get('consequence')}"
                 + (f" ({', '.join(pb['arms'])})" if pb.get("arms") else "")
                 + f"; primary reading {pb.get('primary_reading')}, rise {fmt(pb.get('primary_rise'), 4)} "
                 f"{fmt_iv(pb.get('primary_interval'), 4)} against T {fmt(pb.get('T'), 4)}; under the note's gate "
                 f"{pb.get('note_gate')} ({pb.get('path')})."
                 + (" Not cited: " + "; ".join(pb["issues"]) + "." if pb.get("issues") else ""))
    else:
        L.append(f"Absent: {pb.get('reason')}.")
    L += ["", "## Margins", "", "Computed from the v1 and s2 files and checked against the numbers the rule prints: "
          + ("skipped (a test)" if r["margins_check"].get("skipped") else "equal") + ".", ""]
    lb = r.get("lockbox") or {}
    L += [f"- Lockbox persons in the sets: {lb.get('persons_in_lockbox', 'n/a')} (asserted 0).",
          f"- MoveAtlas: HEAD {(r.get('moveatlas') or {}).get('head') or 'n/a'}. Run {r.get('run_started', 'n/a')}, "
          f"{r.get('timing_s')} s.", ""]
    return "\n".join(L) + "\n"


def finalize(res, sets, locked, stamp, started):
    out = dict(res)
    out["run_started"] = started
    out["sets"] = {k: sorted(v) for k, v in sets.items()}
    n_lb = len(set(sets["dev14"]) & set(locked))
    if n_lb:
        raise HarnessDefect(f"{n_lb} lockbox persons in the sets")
    out["lockbox"] = {"persons_in_lockbox": n_lb, "asserted": True}
    out["moveatlas"] = {"head": (stamp or {}).get("head")}
    return clean(out)


def write_outputs(out_dir, res, tree, stamp, started, inputs):
    """results.json, summary.md and _provenance.json under out_dir; never over an existing results.json."""
    from mdfm.gates import common
    if os.path.exists(os.path.join(out_dir, "results.json")):
        raise FileExistsError(os.path.join(out_dir, "results.json"))
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(out_dir, "results.json"),
                      json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    prov = {"script": "scripts/f2_lever2.py", "task": TASK, "label": LABEL, "run_at": started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "numpy": np.__version__, "rule_sha256": res["rule"]["sha256"],
            "amendment_1_sha256": res["amendment_1"]["sha256"],
            "verdict": res["verdict"]["rule"],
            "inputs": {rel(p, tree): sha256_file(p) for p in inputs if os.path.isfile(p)},
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "weekend_branch.py"),
                                               os.path.join(SCRIPTS, "g3_gain_check.py")]),
            "code_files": ["scripts/f2_lever2.py", "scripts/weekend_branch.py", "scripts/g3_gain_check.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out_dir, "_provenance.json"),
                      json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    return text


def input_files(tree):
    """The files the readings may read (for the provenance)."""
    out = [os.path.join(tree, p) for p in (RULE_REL, AMEND_REL, SEEDS_REL, F1_RESULTS_REL, H4_DUMP_REL, JOB_LOG_REL,
                                           PROBE_RESULTS_REL)]
    out += [os.path.join(tree, d, "log.jsonl") for d in (RUN_DIR, V1_RUN_DIR)]
    for name in (F2L2_B, V1_B, S2_B):
        out += [os.path.join(diag_dir(tree, name), f) for f in ("diag.json", "events.csv")]
        out += [g3dev_path(tree, name), os.path.join(tree, "reports", "gates", "g1", f"dev_{name}", "g1.json"),
                os.path.join(insim_dir(tree, name), "insim.json")]
    return out


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="the tree whose report folders are read (default this tree)")
    p.add_argument("--out", default=None, help="output folder (default reports/bench/f2_lever2/ of the tree)")
    p.add_argument("--write", action="store_true", help="write the outputs also when the verdict is NOT EVALUATED")
    return p.parse_args(argv)


def main(argv=None, sets=None, locked=None, margins=None, check_numbers=True, stamp=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the readings run on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    tree = os.path.abspath(a.tree) if a.tree else TREE
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    check_rules(tree)
    try:
        if sets is None or locked is None:
            from mdfm.gates import g3
            locked = g3.lockbox_persons()
            sets = load_sets(read_json(os.path.join(tree, F1_RESULTS_REL)), locked)
        res, code = run(tree, sets, locked, margins, check_numbers)
    except HarnessDefect as e:
        print(f"HARNESS DEFECT: {e}")
        return EXIT_HARNESS
    if stamp is None:
        stamp = GG().stamp_moveatlas()
    out = finalize(res, sets, locked, stamp, started)
    out_dir = a.out or os.path.join(tree, BENCH)
    if code == EXIT_NOT_EVALUATED and not a.write:
        print(summary_md(out))
        print("NOT EVALUATED: nothing written (missing: " + "; ".join(out["verdict"]["amended"]["missing"]) + ")")
        return code
    try:
        text = write_outputs(out_dir, out, tree, stamp, started, input_files(tree))
    except FileExistsError as e:
        print(f"{e} exists: nothing rewritten; pass --out with a new folder")
        return EXIT_EXISTS
    print(text)
    print(f"written {rel(out_dir, tree)}; exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
