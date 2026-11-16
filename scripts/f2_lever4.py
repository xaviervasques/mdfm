"""L4C (2026-10-06): the readings of the F2 lever-4 rule, the phase-B-only continuation of v1 with the walkway anchor on
the training-fold controls (run cache/runs/pilot_f2l4, configs/stage1_pilot_f2l4.yaml), its anchor replay and its CPU
pre-flight. Label of every output: dev diagnostic, not a G3 read; every f2l4 row is labelled 'walkway-anchored body'.

Rule: reports/bench/f2_lever4/rule.txt, SHA-256 in rule.sha256 beside it and in RULE_SHA256 here; check_rule asserts
that the three agree before any f2l4 value is read (run calls it first). Written after the rule was hashed and before
any input of the run existed. Amendment 1 (task FIXL4, the review of L4C, hashed before any f2l4 number existed):
reports/bench/f2_lever4/rule_amendment_1.txt, SHA-256 in rule_amendment_1.sha256 and AMEND_SHA256, asserted by
check_amendment beside the rule's; it binds T2's producer (A2: the events of the three ends written by one version
of scripts/eval_g3_pd_dev.py, the v1_B and s2_B ones written before f2l4_B's; otherwise T2 is missing and its reading
under rule.txt alone is printed beside, with the verdict it would give), declares T3's person floor (A3) and pins the
anchor inputs at the run's launch (A5, --pin). Amendment 2 (task T2P, at its review, hashed before any f2l4 evaluation
existed): reports/bench/f2_lever4/rule_amendment_2.txt, SHA-256 in rule_amendment_2.sha256 and AMEND2_SHA256,
asserted by check_amendment_2; the provenance of an end's events is the producer's _provenance_events.json beside
events.csv, never the folder's _provenance.json (B1: script, script_sha256, run_at, events_sha256 equal to the
SHA-256 of events.csv, diag_sha256), the events layout's script scripts/eval_g3_diag.py is part of the one version
(B2: one diag_sha256 at the three ends; code_sha256 equality printed beside), the PD jobs run in class 1 (B3), the
producer and the layout's script are frozen until the f2l4 end's events exist (B4, FREEZE, asserted by
tests/test_f2_lever4.py) and the batch windows of the three ends are printed beside T2 (B5).

Readings (default mode; rule sections 4 to 8). Inputs, every path under --tree (default the tree root):
- f2l4_B (name pilot_f2l4_ckpt_B_032044) and v1_B (phaseA_root_full_ckpt_B_032044): reports/gates/g3/diag_<name>/
  (diag.json, events.csv), reports/gates/g3/dev_<name>/g3_dev.json, reports/gates/g1/dev_<name>/g1.json and
  reports/gates/g3_sim/insim_<name>/ (the post-train templates of the queue); cache/runs/<run>/log.jsonl;
  reports/gpu_queue/job_log.csv; s2_B (pilot_v1_s2_ckpt_B_032044) for the seed differences, through
  scripts/f2_lever2.py's margins_from (reused by import: the same lines, seed differences and margins as lever 2);
- transfer lines: T1, the train54 decode of each end, reports/bench/g3_gain_check/decode_<name>/train_events.csv
  (scripts/g3_gain_check.py --decode), with the dev12 strides of the diag events; T2, the PD dev fold's matched strides
  of each end in reports/gates/g3/dev_pd_<name>/events.csv (the events layout of scripts/eval_g3_diag.py; the file may
  hold any weargait_pd person of reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/folds_proposed.csv, and the
  persons of its proposed PD dev fold are read; amendment 1 A2 on its producer); T3, the dev12
  strides of rows base, sites6, family_legs and family_wrists of reports/gates/g3/diag_<name>_ablations_bw32/
  events.csv; T4, the in-sim stride gain of the f2l4 end (g3_gain_check.insim_gain). The thresholds d_gap,
  d_dev12_stride and d_insim come from reports/bench/g3_gain_check/results.json (F1), d_pd and each d_r from the
  v1_B and s2_B files of T2 and T3 when they exist (rule section 7); a missing input makes its T line missing.
- early read (reported only): reports/gates/g3/curve_<run>_ckpt_B_004000/, reports/gates/g1/
  dev_<run>_ckpt_B_004000_lt2/ and reports/gates/g3_sim/insim_<run>_ckpt_B_004000/ for pilot_f2l4 and
  phaseA_root_full.
Readings: R0 (the run's log, job row, start event: config, source, walkway_anchor section and table check), R1 (S(12),
V(12), paired v1 minus f2l4, MEETS at least the margin with the lower bound above 0), R2 (S(14), V(14), walkway IC,
G1a, stride_mape_pct, f2l4 minus v1 at most the margin), T1 to T4, the verdict in the order of rule section 8 and the
route of section 9. Reported, reading nothing: the walkway_anchor log entries per block of 4,000 steps (median term,
gain, chord ratio), hsic and hsic_active per block beside v1's, the in-sim and dev12 gains, the seconds per phase B
step. Until the run's evaluations exist the verdict reads NOT EVALUATED, naming what is missing; nothing is written
then unless --write is given.

--replay: v1's phase B plans (plan only, no window read or rendered: mdfm.data.sampler.MixtureSampler of the config,
seed loop.phase_seed, every micro-step) through mdfm.data.walkway_anchor (the anchor table of the config built on the
window store), per train54 person and in total: anchored windows, segments, anchored seconds, with the table and
without its clock exclusion; writes <out>/anchor_replay.json (rule section 7 (c) reads it).

--pin: writes <out>/anchor_inputs.json, the input pin of the run's job (mdfm.data.walkway_anchor.PIN_ENV;
amendment 1 A5): TABLE_INPUTS, the four input SHA-256 of rule section 1, after asserting the rule's hash and that the
files on disk still hash to them.

--preflight: the CPU gradient pre-flight (rule section 0): the run's trainer on the CPU (configs/stage1_pilot_f2l4.yaml,
init-from cache/runs/phaseA_root_full/ckpt_A_010681.pt into a scratch run folder, the load's drift line written into
that scratch folder, never into v1's run folder), the first phase B micro-steps of v1's
sampler rendered on the CPU until PREFLIGHT_ANCHORED of them hold anchored windows; on each, a sub-batch of every
anchored window and a uniform sample of PREFLIGHT_OTHERS other windows is collated and forwarded (amortized posterior,
no refinement), and the gradient norms over the encoder and over the decoder of: the anchor term, the rest of the
objective, sup_root_rate (whole, u and yaw parts), imu_nll, the weighted hsic and the full loss, each per-window value
weighted as in the full micro-step (an anchored window by 1 / N, a sampled other window by (N - n_a) / (m N)), and the
cosine of the anchor's and the hsic's encoder gradients; writes <out>/preflight.json. An estimate on a sub-batch: the
hsic of a sub-batch is not the micro-step's, and a scaled sample's gradient norm overstates its noise.

Exit codes: 0 a verdict (GO, its qualified forms, PARTIAL, NO-GO); 3 NOT EVALUATED; 4 a harness defect (a margin or
threshold that differs from the rule's printed number, a person outside the expected set); 5 results.json exists.
Data discipline: as scripts/f2_lever2.py; no lockbox, EVAL, CODY or care_pd record; the readings on the CPU
(CUDA_VISIBLE_DEVICES=-1 is set before torch can load, --preflight included).

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever4.py [--write] [--out DIR] [--replay] [--preflight]
        [--pin]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the readings, the replay and the pre-flight never see the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
BENCH = os.path.join("reports", "bench", "f2_lever4")
RULE_REL = os.path.join(BENCH, "rule.txt")
RULE_SHA_REL = os.path.join(BENCH, "rule.sha256")
RULE_SHA256 = "4e2683e927051a621f33b8f292e6514e14edeb2aa59b7442036dfca5ff9fe372"
AMEND_REL = os.path.join(BENCH, "rule_amendment_1.txt")
AMEND_SHA_REL = os.path.join(BENCH, "rule_amendment_1.sha256")
AMEND_SHA256 = "d6cf899106070018c41c8a448da4232cd112c1d403f54959c13787b88206581b"
AMEND2_REL = os.path.join(BENCH, "rule_amendment_2.txt")
AMEND2_SHA_REL = os.path.join(BENCH, "rule_amendment_2.sha256")
AMEND2_SHA256 = "fb1e770050f68684c8ab629ff6de8b1b52603bf59164e29feda1b6848b31ca7d"
PIN_NAME = "anchor_inputs.json"              # the input pin of the run's job (amendment 1 A5)
PD_PRODUCER = "scripts/eval_g3_pd_dev.py"    # T2's producer (amendment 1 A2)
PD_DIAG = "scripts/eval_g3_diag.py"          # the events layout, part of the producer (amendment 2 B2)
PD_EVENTS_PROV = "_provenance_events.json"   # the provenance of an end's events (amendment 2 B1)
PD_PROV_FIELDS = ("script", "script_sha256", "run_at", "events_sha256", "diag_sha256")
# amendment 2 B4: the producer and the layout's script as hashed, until the f2l4 end's events or the reading exist
FREEZE = {PD_PRODUCER: "6763ce128454d2509c0dcec646a3ed933a71ad3aba9caa6de7a2f83ab9a9435d",
          PD_DIAG: "e99e18e866a69b1fd776c240d26f5c7234acc85de708f00d164ca116b95166ad"}
LABEL = "dev diagnostic, not a G3 read"
ROW_LABEL = "walkway-anchored body"
TASK = "L4C"
RUN = "pilot_f2l4"
RUN_DIR = "cache/runs/pilot_f2l4"
V1_RUN_DIR = "cache/runs/phaseA_root_full"
F2L4_B = "pilot_f2l4_ckpt_B_032044"
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
F2L4_4000 = "pilot_f2l4_ckpt_B_004000"
V1_4000 = "phaseA_root_full_ckpt_B_004000"
CONFIG_REL = "configs/stage1_pilot_f2l4.yaml"
CONFIG_SHA256 = "f9296067811308de569d023fc5f572b0ce5056e6ede7e6ba56be9f27d0d4701c"
SOURCE_PATH = "cache/runs/phaseA_root_full/ckpt_A_010681.pt"
SOURCE_SHA256 = "ad224565725e6ec973d9b4b2af0e67d2c658241d8644b7d879a8730d70fb5c10"
TRAIN54_SHA256 = "293f98d02adc48b0fa266c2125be40c23201066403fcb110543d8c882c9c77d6"
WALKWAY_ANCHOR = {"enabled": True, "weight": 1.0, "sd": 0.1,
                  "tasks": ["SelfPace", "HurriedPace", "SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN"],
                  "persons_sha256": TRAIN54_SHA256}
# the anchor table at the hash (rule section 1): its input files, cell size and fit count
TABLE_INPUTS = {"reports/weargait_targets/events.csv": "f8283656435424d06fffd8fb1cb2bb70420009c9b4b252118cad8c5f473f2d8d",
                "reports/weargait_targets/trials.csv": "8d9d4083f31292ac750af5fd9ee0b7d1921f153c4185c16288490025092377cf",
                "reports/walkway_anchor/alignment_trials.csv":
                    "c313b56391b4c08a4b103ac67ceca934613b8bc52b8c24cb4cf6a6360f4d67ae",
                "reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/folds_proposed.csv":
                    "bb92c5a99a629eca01f4741f92c846d6ce18e848c60b6a678c6eef362cd2bfd8"}
CELL_CM = 1.2713577081207839
CELL_N = 42
PHASE_B_STEPS = 32044
PEAK_RESERVED_MAX_GB = 14.0
NONFINITE_RUN = 3
TOL = 1e-9
SEED_MULTIPLE = 2.0
FOLLOW_SHARE = 0.5                  # a row or the PD fold follows when its rise is at least this share of the reference's
OVERSHOOT_MULTIPLE = 2.0            # PD OVERSHOOT (reported): the PD rise above this multiple of the controls' rise
BLOCK_STEPS = 4000
MIN_ROW_PERSONS = 10                # a T3 row is read when at least this many dev14 persons hold a matched stride
T3_ROWS = ("base", "sites6", "family_legs", "family_wrists")
F1_RESULTS_REL = os.path.join("reports", "bench", "g3_gain_check", "results.json")
FOLDS_REL = os.path.join("reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06", "folds_proposed.csv")
JOB_LOG_REL = os.path.join("reports", "gpu_queue", "job_log.csv")
PRIMARY = ("S12", "V12")
R2_LINES = ("S14", "V14", "IC", "G1a", "MAPE")
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "IC": "ms", "G1a": "mm", "MAPE": "percent"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "IC": 2, "G1a": 2, "MAPE": 2}
G3_LINE_OF = {"IC": "ic_median_ms_walkway", "MAPE": "stride_mape_pct"}
EXIT_VERDICT, EXIT_NOT_EVALUATED, EXIT_HARNESS, EXIT_EXISTS = 0, 3, 4, 5
NO_GO, GO, PARTIAL, NOT_EVALUATED = "NO-GO", "GO", "PARTIAL", "NOT EVALUATED"
GO_DOMAIN, GO_UNTESTED = "GO, walkway-domain only", "GO, transfer not evaluated"
FLAGS = {"T1": "PERSON GAP", "T2": "CONTROL ONLY", "T3": "LAYOUT KEYED", "T4": "SIM OVERSHOOT"}
PREFLIGHT_ANCHORED = 2              # micro-steps with anchored windows read by the pre-flight
PREFLIGHT_MAX_MICRO = 8             # micro-steps drawn at most
PREFLIGHT_OTHERS = 12               # other windows sampled into a pre-flight sub-batch
PREFLIGHT_RUN_DIR = "cache/runs/preflight_f2l4"


class HarnessDefect(AssertionError):
    """A departure of the harness itself: a margin or threshold that differs from the rule's printed number, a person
    outside the expected set."""


def _sibling(name):
    key = f"_l4c_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def F2():
    """scripts/f2_lever2.py: the lever-2 readers and margins, reused."""
    return _sibling("f2_lever2")


def WB():
    return F2().WB()


def GG():
    return F2().GG()


def rel(path, tree=None):
    return F2().rel(path, tree)


def fin(x):
    return F2().fin(x)


def read_json(path):
    return F2().read_json(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the rule
def check_rule(tree=None, expected=None):
    """The SHA-256 of rule.txt under `tree`, asserted equal to the first word of rule.sha256 and to RULE_SHA256 (or
    expected). Returns it."""
    tree = tree or TREE
    want = RULE_SHA256 if expected is None else expected
    got = sha256_file(os.path.join(tree, RULE_REL))
    with open(os.path.join(tree, RULE_SHA_REL), encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (got == stored == want):
        raise AssertionError(f"rule hash mismatch: rule.txt {got}, rule.sha256 {stored or 'empty'}, script {want}")
    return got


def check_amendment(tree=None, expected=None):
    """The SHA-256 of rule_amendment_1.txt under `tree`, asserted equal to the first word of rule_amendment_1.sha256
    and to AMEND_SHA256 (or expected). Returns it."""
    tree = tree or TREE
    want = AMEND_SHA256 if expected is None else expected
    got = sha256_file(os.path.join(tree, AMEND_REL))
    with open(os.path.join(tree, AMEND_SHA_REL), encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (got == stored == want):
        raise AssertionError(f"amendment hash mismatch: rule_amendment_1.txt {got}, rule_amendment_1.sha256 "
                             f"{stored or 'empty'}, script {want}")
    return got


def check_amendment_2(tree=None, expected=None):
    """The SHA-256 of rule_amendment_2.txt under `tree`, asserted equal to the first word of rule_amendment_2.sha256
    and to AMEND2_SHA256 (or expected). Returns it."""
    tree = tree or TREE
    want = AMEND2_SHA256 if expected is None else expected
    got = sha256_file(os.path.join(tree, AMEND2_REL))
    with open(os.path.join(tree, AMEND2_SHA_REL), encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (got == stored == want):
        raise AssertionError(f"amendment hash mismatch: rule_amendment_2.txt {got}, rule_amendment_2.sha256 "
                             f"{stored or 'empty'}, script {want}")
    return got


def freeze_window(tree=None):
    """Amendment 2 B4: True while the freeze holds, that is while neither the f2l4 end's _provenance_events.json nor
    the reading (reports/bench/f2_lever4/results.json) exists under `tree`."""
    tree = tree or TREE
    ends = (os.path.join(tree, "reports", "gates", "g3", f"dev_pd_{F2L4_B}", PD_EVENTS_PROV),
            os.path.join(tree, BENCH, "results.json"))
    return not any(os.path.exists(p) for p in ends)


def rule_text(tree=None):
    with open(os.path.join(tree or TREE, RULE_REL), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- margins and thresholds
def transfer_thresholds(tree=None):
    """d_gap, d_dev12_stride and d_insim from F1's results.json (rule section 7): |v1_B minus s2_B| of the
    train54-minus-dev12 stride gain gap, of the dev12 stride gain and of the in-sim stride gain; with the end values."""
    ends = read_json(os.path.join(tree or TREE, F1_RESULTS_REL))["ends"]
    t = {}
    for end in ("v1_B", "s2_B"):
        e = ends[end]
        d12 = e["descriptive"]["dev12"]
        t[end] = {"train54_stride": float(e["gain"]["stride"]["gain"]),
                  "dev12_stride": float(d12["stride_per_stride"]["gain"]["point"]),
                  "insim_stride": float(e["insim"]["stride_gain"])}
        t[end]["gap_stride"] = t[end]["train54_stride"] - t[end]["dev12_stride"]
    return {"d_gap": abs(t["v1_B"]["gap_stride"] - t["s2_B"]["gap_stride"]),
            "d_dev12_stride": abs(t["v1_B"]["dev12_stride"] - t["s2_B"]["dev12_stride"]),
            "d_insim": abs(t["v1_B"]["insim_stride"] - t["s2_B"]["insim_stride"]), "ends": t}


def margins(tree, sets, locked):
    """The margins of rule section 7 (scripts/f2_lever2.py margins_from: the lines of v1, the seed differences, the
    success, non-inferiority and early margins) and the transfer thresholds."""
    m = F2().margins_from(tree, sets, locked)
    m["transfer"] = transfer_thresholds(tree)
    return m


def section(text, start, end):
    nl = chr(10)
    return text[text.index(nl + start):text.index(nl + end)]


def rule_number_check(m, text):
    """The strings that rule section 7 prints for the margins and thresholds, formatted from m; returns those missing
    from the text, whitespace runs read as one space (empty: the numbers the script reads are the registered ones)."""
    sec7 = " ".join(section(text, "7. Margins", "8. Readings").split())
    s, v = m["success"]["S12"], m["success"]["V12"]
    sd = m["seed_diff"]
    want = [f"S(12) v1 {s['line']:.4f} cm, d {sd['S12']:.4f}, m_S {s['margin']:.4f} cm",
            f"f2l4 at most {s['needed_line_at_most']:.4f} cm",
            f"V(12) v1 {v['line']:.4f} m/s, d {sd['V12']:.4f}, m_V {v['margin']:.4f} m/s",
            f"f2l4 at most {v['needed_line_at_most']:.4f} m/s"]
    for k, unit in (("S14", "cm"), ("V14", "m/s"), ("IC", "ms"), ("G1a", "mm"), ("MAPE", "percent")):
        n = m["noninferiority"][k]
        name = "stride_mape_pct" if k == "MAPE" else k.replace("14", "(14)")
        want.append(f"{name} {n['margin']:.4f} {unit} (line {n['line']:.4f}, d {n['seed_diff']:.4f})")
    e = m["early"]
    want.append(f"S(12) {e['S12']['line']:.2f} cm and V(12) {e['V12']['line']:.4f} m/s; early margins "
                f"{e['S12']['margin']:.2f} cm and {e['V12']['margin']:.4f} m/s")
    t = m["transfer"]
    want += [f"d_gap {t['d_gap']:.4f}", f"d_dev12_stride {t['d_dev12_stride']:.4f}", f"d_insim {t['d_insim']:.4f}"]
    return [x for x in want if x not in sec7]


# ---------------------------------------------------------------- R0
def r0_reading(tree, recs, unparsed, job_log=None):
    """R0 of rule section 8: lever 2's checks (source, config, verify at step 32,044, report mismatches, alarms,
    non-finite stop, memory bound, job exit) and the walkway anchor's: every start event carries the section of rule
    section 1 and a passed table check with the train54 SHA-256, 54 persons, the cell size and fit count of section 1
    and the input files' SHA-256 of section 1; render_aug and jepa absent; phase B log records carry a walkway_anchor
    entry at weight 1.0."""
    wb = WB()
    stop, fail, checks, values = [], [], {}, {"unparsed_log_lines": unparsed}
    if recs is None:
        fail.append(f"no run log {RUN_DIR}/log.jsonl")
        return {"holds": False, "stop": stop, "fail": fail, "checks": checks, "values": values}
    inits, starts = wb.log_events(recs, "init_from"), wb.log_events(recs, "start")
    values["init_from_events"], values["start_events"] = len(inits), len(starts)
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
    checks["start_config"] = bool(starts) and not bad_cfg
    if not starts:
        fail.append("no start event")
    if bad_cfg:
        fail.append("start event at another config SHA-256: " + ", ".join(x[:12] for x in bad_cfg))
    anchor_issues = []
    for s in starts:
        wa = s.get("walkway_anchor") or {}
        sec, tab = wa.get("section"), wa.get("table") or {}
        if sec != WALKWAY_ANCHOR:
            anchor_issues.append(f"section {sec}")
        if tab.get("check") != "passed" or tab.get("persons_sha256") != TRAIN54_SHA256 or tab.get("persons") != 54:
            anchor_issues.append("table check not passed on train54")
        if fin(tab.get("cell_cm")) is None or abs(float(tab["cell_cm"]) - CELL_CM) > TOL or tab.get("cell_n") != CELL_N:
            anchor_issues.append(f"cell size {tab.get('cell_cm')} cm on {tab.get('cell_n')} trials")
        if tab.get("inputs") != TABLE_INPUTS:
            anchor_issues.append("anchor table inputs other than section 1's")
        if s.get("render_aug") is not None or s.get("jepa") is not None:
            anchor_issues.append("render_aug or jepa present")
    checks["walkway_anchor_start"] = bool(starts) and not anchor_issues
    if anchor_issues:
        fail.append("walkway_anchor start entry: " + "; ".join(sorted(set(anchor_issues))))
    logs_b = wb.log_events(recs, "log", "B")
    bad_log = [r.get("step") for r in logs_b if (r.get("walkway_anchor") or {}).get("weight") != 1.0]
    checks["walkway_anchor_logged"] = bool(logs_b) and not bad_log
    if logs_b and bad_log:
        fail.append(f"phase B log records without the walkway_anchor entry at weight 1.0 (steps {bad_log[:3]})")
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
    values["job_exit"], values["job_log"] = code, (rel(jpath, tree) if jpath else None)
    checks["job_exit_0"] = code == 0
    if code is None:
        fail.append(f"no job_log.csv row of a scripts/train_stage1.py job on {RUN_DIR}")
    elif code != 0:
        fail.append(f"last train_stage1.py job on {RUN_DIR} exited {code}")
    return {"holds": not stop and not fail, "stop": stop, "fail": fail, "checks": checks, "values": values}


# ---------------------------------------------------------------- R1 and R2
def r1_reading(v1_set, f4_set, m, why_missing=None):
    """{S12, V12: {read, v1, f2l4, improvement {n, point, interval}, margin, meets, reason, label}}; improvement v1
    minus f2l4 (WB.paired, the v2check bootstrap)."""
    wb = WB()
    out = {}
    for L, key in (("S12", "S"), ("V12", "V")):
        margin = m["success"][L]["margin"]
        a_pp, a_pt = wb.g3_line(v1_set, key, wb.SUB12)
        b_pp, b_pt = wb.g3_line(f4_set, key, wb.SUB12)
        row = {"read": False, "v1": a_pt, "f2l4": b_pt, "improvement": None, "margin": margin, "meets": None,
               "unit": UNIT[L], "reason": None, "label": ROW_LABEL}
        if a_pp is None or b_pp is None:
            row["reason"] = why_missing or f"no {L} of the diag base row"
        else:
            p = wb.paired(a_pp, b_pp)
            row["improvement"] = {k: p[k] for k in ("n", "point", "interval")}
            if p["point"] is None or p["interval"] is None:
                row["reason"] = p.get("reason") or "fewer than 2 paired persons"
            else:
                row["read"] = True
                row["meets"] = bool(p["point"] >= margin - TOL and p["interval"][0] > TOL)
        out[L] = row
    return out


def r2_reading(values_v1, values_f4, m, reasons):
    """{line: {read, v1, f2l4, difference f2l4 minus v1, margin, holds, reason, label}} for R2_LINES."""
    out = {}
    for k in R2_LINES:
        a, b = values_v1.get(k), values_f4.get(k)
        margin = m["noninferiority"][k]["margin"]
        ok = a is not None and b is not None
        row = {"read": ok, "v1": a, "f2l4": b, "difference": None, "margin": margin, "holds": None, "unit": UNIT[k],
               "reason": None if ok else (reasons.get(k) or f"no {k} value"), "label": ROW_LABEL}
        if ok:
            row["difference"] = b - a
            row["holds"] = bool(b - a <= margin + TOL)
        out[k] = row
    return out


# ---------------------------------------------------------------- transfer lines
def _stride_gain(path, allowed, locked, persons=None, row="base", min_persons=None):
    """(gain, n strides, n dev persons) of the matched strides of `row` of an events.csv (g3_gain_check.read_events,
    guarded to `allowed`), restricted to `persons`: the median of decoded over walkway (F1's form); gain None when the
    file is absent or fewer than min_persons persons of `allowed` hold a matched stride."""
    g = GG()
    if not os.path.exists(path):
        return None, 0, 0, f"missing: {rel(path)}"
    st, _, _ = g.read_events(path, allowed, locked, row=row)
    n_all = int(st["person_id"].nunique())
    if min_persons is not None and n_all < min_persons:
        return None, 0, n_all, f"row {row}: {n_all} persons with a matched stride in {rel(path)}"
    if persons is not None:
        st = g.subset(st, persons)
    v = fin(g.median_ratio(st["x"], st["y"])) if len(st) else None
    return v, int(len(st)), n_all, None if v is not None else f"no matched stride in {rel(path)}"


def pd_dev_persons(tree):
    """The proposed PD dev fold of the reviewed stage-2 sources (weargait_pd persons of fold dev; source, person_id
    and fold columns only)."""
    out = []
    with open(os.path.join(tree, FOLDS_REL), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "weargait_pd" and r["fold"] == "dev":
                out.append(r["person_id"])
    return sorted(out)


def t1_reading(tree, sets, locked, thr):
    """T1 PERSON GAP: the train54-minus-dev12 stride gain gap of f2l4_B against v1_B's (each end's train54 decode and
    diag events), flagged when the rise exceeds 2 x d_gap."""
    from mdfm.gates import g3
    train = g3.controls("train")
    out = {"line": "T1", "flag_name": FLAGS["T1"], "threshold": thr["d_gap"], "label": ROW_LABEL, "ends": {}}
    for key, name in (("v1_B", V1_B), ("f2l4_B", F2L4_B)):
        tg, nt, _, wt = _stride_gain(os.path.join(tree, "reports", "bench", "g3_gain_check", f"decode_{name}",
                                                  "train_events.csv"), train, locked)
        dg, nd, _, wd = _stride_gain(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}", "events.csv"),
                                     sets["dev14"], locked, sets["dev12"])
        out["ends"][key] = {"train54_stride_gain": tg, "dev12_stride_gain": dg, "train54_strides": nt,
                            "dev12_strides": nd, "gap": None if tg is None or dg is None else tg - dg,
                            "missing": [w for w in (wt, wd) if w]}
    a, b = out["ends"]["v1_B"]["gap"], out["ends"]["f2l4_B"]["gap"]
    if a is None or b is None:
        out.update(read=False, flag=None, status="missing",
                   reason="; ".join(out["ends"]["v1_B"]["missing"] + out["ends"]["f2l4_B"]["missing"]))
        return out
    out["rise"] = b - a
    flag = bool(b - a > SEED_MULTIPLE * thr["d_gap"] + TOL)
    out.update(read=True, flag=flag, status=FLAGS["T1"] if flag else "HELD", reason=None)
    return out


def pd_pool_persons(tree):
    """Every weargait_pd person of the reviewed stage-2 folds (folds_proposed.csv; source and person_id columns): the
    persons a dev_pd events.csv may hold; a person outside them is a harness defect (amendment 1 A2)."""
    out = set()
    with open(os.path.join(tree, FOLDS_REL), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "weargait_pd":
                out.add(r["person_id"])
    return sorted(out)


def pd_producer_check(tree):
    """Amendment 1 A2 with amendment 2 B1 and B2, on T2's producer: {ok, reasons, ends {key: {script, script_sha256,
    run_at, events_sha256, diag_sha256, code_sha256, batch_windows, provenance}}, code_sha256_equal, batch_windows}.
    The provenance of an end's events is reports/gates/g3/dev_pd_<name>/_provenance_events.json (PD_EVENTS_PROV; the
    folder's _provenance.json is not read for T2, B1). ok when, at the three ends, it names scripts/eval_g3_pd_dev.py
    and carries script_sha256, run_at, events_sha256 and diag_sha256, events_sha256 is the SHA-256 of the folder's
    events.csv, script_sha256 (A2) and diag_sha256 (B2) are each one value over the three ends, and the v1_B and s2_B
    events were written (run_at) before f2l4_B's. Printed beside, not conditions: whether code_sha256 is one value over
    the three ends (B2) and each end's batch windows (B5)."""
    ends, why = {}, []
    for key, name in (("v1_B", V1_B), ("s2_B", S2_B), ("f2l4_B", F2L4_B)):
        folder = os.path.join(tree, "reports", "gates", "g3", f"dev_pd_{name}")
        p = os.path.join(folder, PD_EVENTS_PROV)
        doc = None
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        if not isinstance(doc, dict):
            why.append(f"{key}: no {rel(p, tree)}")
            ends[key] = {}
            continue
        e = {k: doc.get(k) for k in PD_PROV_FIELDS}
        e.update(code_sha256=doc.get("code_sha256"), batch_windows=(doc.get("inference") or {}).get("batch_windows"),
                 provenance=rel(p, tree))
        ends[key] = e
        if e["script"] != PD_PRODUCER:
            why.append(f"{key}: written by {e['script']!r}, not {PD_PRODUCER}")
        for k in PD_PROV_FIELDS[1:]:
            if not e[k]:
                why.append(f"{key}: no {k} in its {PD_EVENTS_PROV}")
        ev = os.path.join(folder, "events.csv")
        if e["events_sha256"] and (not os.path.exists(ev) or sha256_file(ev) != e["events_sha256"]):
            why.append(f"{key}: {rel(ev, tree)} is not the file its {PD_EVENTS_PROV} names (events_sha256)")
    if not why and len({ends[k]["script_sha256"] for k in ends}) != 1:
        why.append("the three ends were written by different versions of " + PD_PRODUCER)
    if not why and len({ends[k]["diag_sha256"] for k in ends}) != 1:
        why.append(f"the three ends' events were laid out by different versions of {PD_DIAG} (diag_sha256)")
    if not why:
        late = [k for k in ("v1_B", "s2_B") if not str(ends[k]["run_at"]) < str(ends["f2l4_B"]["run_at"])]
        if late:
            why.append(f"{', '.join(late)} written after f2l4_B's ({ends['f2l4_B']['run_at']})")
    codes = [ends[k].get("code_sha256") for k in ends]
    return {"ok": not why, "reasons": why, "ends": ends,
            "code_sha256_equal": None if not all(codes) else len(set(codes)) == 1,
            "batch_windows": {k: ends[k].get("batch_windows") for k in ends}}


def t2_reading(tree, sets, locked, thr):
    """T2 CONTROL ONLY: the PD dev fold's stride gain (dev_pd_<end>/events.csv, the PD dev fold's persons of a file of
    weargait_pd persons) and the dev12 controls' (diag events) at v1_B, s2_B and f2l4_B; d_pd from v1_B and s2_B;
    flagged when the controls rise beyond 2 x d_dev12_stride and the PD gain does not follow (its rise beyond 2 x d_pd
    and at least FOLLOW_SHARE of the controls' rise); PD OVERSHOOT reported when the PD rise exceeds
    OVERSHOOT_MULTIPLE x the controls' rise. Amendment 1 A2 with amendment 2 B1 and B2: read only when
    pd_producer_check holds; otherwise missing, with the reading under rule.txt alone in rule_txt_alone."""
    pd_dev, pool = pd_dev_persons(tree), pd_pool_persons(tree)
    out = {"line": "T2", "flag_name": FLAGS["T2"], "label": ROW_LABEL, "pd_dev_persons": len(pd_dev), "ends": {},
           "d_dev12_stride": thr["d_dev12_stride"]}
    miss = []
    for key, name in (("v1_B", V1_B), ("s2_B", S2_B), ("f2l4_B", F2L4_B)):
        pg, npd, _, wp = _stride_gain(os.path.join(tree, "reports", "gates", "g3", f"dev_pd_{name}", "events.csv"),
                                      pool, locked, pd_dev)
        cg, nc, _, wc = (None, 0, 0, None) if key == "s2_B" else _stride_gain(
            os.path.join(tree, "reports", "gates", "g3", f"diag_{name}", "events.csv"), sets["dev14"], locked,
            sets["dev12"])
        out["ends"][key] = {"pd_stride_gain": pg, "pd_strides": npd, "dev12_stride_gain": cg,
                            "ratio_pd_over_dev12": None if pg is None or not cg else pg / cg}
        miss += [w for w in (wp, wc) if w]
    e = out["ends"]
    prod = pd_producer_check(tree)
    out["producer"] = prod             # with the batch windows of the three ends (amendment 2 B5), printed beside T2
    need = (e["v1_B"]["pd_stride_gain"], e["s2_B"]["pd_stride_gain"], e["f2l4_B"]["pd_stride_gain"],
            e["v1_B"]["dev12_stride_gain"], e["f2l4_B"]["dev12_stride_gain"])
    if any(x is None for x in need):
        out.update(read=False, flag=None, status="missing", reason="; ".join(miss) or "a gain is not defined")
        return out
    d_pd = abs(e["v1_B"]["pd_stride_gain"] - e["s2_B"]["pd_stride_gain"])
    rc = e["f2l4_B"]["dev12_stride_gain"] - e["v1_B"]["dev12_stride_gain"]
    rp = e["f2l4_B"]["pd_stride_gain"] - e["v1_B"]["pd_stride_gain"]
    follows = bool(rp > SEED_MULTIPLE * d_pd + TOL and rp >= FOLLOW_SHARE * rc - TOL)
    flag = bool(rc > SEED_MULTIPLE * thr["d_dev12_stride"] + TOL and not follows)
    out.update(read=True, d_pd=d_pd, controls_rise=rc, pd_rise=rp, pd_follows=follows, flag=flag,
               status=FLAGS["T2"] if flag else "HELD", reason=None,
               pd_overshoot=bool(rc > 0 and rp > OVERSHOOT_MULTIPLE * rc))
    if not prod["ok"]:
        out["rule_txt_alone"] = {"read": True, "flag": flag, "status": out["status"], "reason": None}
        out.update(read=False, flag=None, status="missing",
                   reason="amendment 1 A2 and amendment 2 B1 and B2, the producer of the PD events: "
                          + "; ".join(prod["reasons"]))
    return out


def t3_reading(tree, sets, locked):
    """T3 LAYOUT KEYED: the dev12 stride gain of rows T3_ROWS of each end's ablations run (diag_<end>_ablations_bw32);
    d_r per row from v1_B and s2_B; flagged when the base row rises beyond 2 x d_base and no other row follows (its
    rise beyond 2 x d_r and at least FOLLOW_SHARE of the base row's rise). A row read by fewer than MIN_ROW_PERSONS
    dev14 persons at an end is not read; T3 is missing when the base row is not read at every end."""
    out = {"line": "T3", "flag_name": FLAGS["T3"], "label": ROW_LABEL, "rows": {}}
    miss = []
    for r in T3_ROWS:
        row = {}
        for key, name in (("v1_B", V1_B), ("s2_B", S2_B), ("f2l4_B", F2L4_B)):
            g, n, n14, why = _stride_gain(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}_ablations_bw32",
                                                       "events.csv"), sets["dev14"], locked, sets["dev12"], row=r,
                                          min_persons=MIN_ROW_PERSONS)
            row[key] = {"gain": g, "strides": n, "dev14_persons": n14}
            if why:
                miss.append(why)
        ok = all(row[k]["gain"] is not None for k in ("v1_B", "s2_B", "f2l4_B"))
        row["read"] = ok
        if ok:
            row["d_r"] = abs(row["v1_B"]["gain"] - row["s2_B"]["gain"])
            row["rise"] = row["f2l4_B"]["gain"] - row["v1_B"]["gain"]
        out["rows"][r] = row
    base = out["rows"]["base"]
    if not base["read"]:
        out.update(read=False, flag=None, status="missing", reason="; ".join(sorted(set(miss))))
        return out
    rb = base["rise"]
    for r in T3_ROWS[1:]:
        x = out["rows"][r]
        x["follows"] = bool(x["read"] and x["rise"] > SEED_MULTIPLE * x["d_r"] + TOL and x["rise"] >= FOLLOW_SHARE * rb
                            - TOL)
    flag = bool(rb > SEED_MULTIPLE * base["d_r"] + TOL and not any(out["rows"][r]["follows"] for r in T3_ROWS[1:]))
    out.update(read=True, flag=flag, status=FLAGS["T3"] if flag else "HELD", reason=None,
               rows_not_read=[r for r in T3_ROWS[1:] if not out["rows"][r]["read"]])
    return out


def t4_reading(tree, thr, sha=None):
    """T4 SIM OVERSHOOT: the in-sim stride gain of f2l4_B (g3_gain_check.insim_gain), flagged above 1 + 2 x d_insim."""
    blk = F2().insim_block(tree, F2L4_B, sha)
    out = {"line": "T4", "flag_name": FLAGS["T4"], "label": ROW_LABEL, "threshold": thr["d_insim"],
           "insim_stride_gain": blk.get("stride_gain"), "v1_B_insim_stride_gain": thr["ends"]["v1_B"]["insim_stride"]}
    if not blk.get("present") or blk.get("stride_gain") is None:
        out.update(read=False, flag=None, status="missing", reason=blk.get("reason") or "no in-sim stride gain")
        return out
    flag = bool(blk["stride_gain"] > 1.0 + SEED_MULTIPLE * thr["d_insim"] + TOL)
    out.update(read=True, flag=flag, status=FLAGS["T4"] if flag else "HELD", reason=None)
    return out


# ---------------------------------------------------------------- verdict and route
def verdict(r0, r1, r2, t):
    """Rule section 8: (1) a stop case NO-GO; (2) another R0 failure NOT EVALUATED; (3) an R2 line read and FAILS
    NO-GO; (4) both primary lines read and neither MEETS NO-GO; (5) a missing input of R1 or R2 NOT EVALUATED; (6)
    both MEET: GO when T1 to T4 are read and none flags, GO_DOMAIN when one flags, GO_UNTESTED when none flags and one
    is missing; (7) one MEETS: PARTIAL, its T flags beside."""
    if r0["stop"]:
        return {"verdict": NO_GO, "case": 1, "reasons": list(r0["stop"]), "missing": []}
    if r0["fail"]:
        return {"verdict": NOT_EVALUATED, "case": 2, "reasons": list(r0["fail"]), "missing": list(r0["fail"])}
    failed = [k for k in R2_LINES if r2[k]["read"] and not r2[k]["holds"]]
    unread = [f"{k}: {r2[k]['reason']}" for k in R2_LINES if not r2[k]["read"]]
    if failed:
        return {"verdict": NO_GO, "case": 3, "reasons": [f"{k} FAILS" for k in failed], "missing": unread}
    if all(r1[L]["read"] for L in PRIMARY) and not any(r1[L]["meets"] for L in PRIMARY):
        return {"verdict": NO_GO, "case": 4, "reasons": ["neither S(12) nor V(12) MEETS"], "missing": unread}
    missing = [f"{L}: {r1[L]['reason']}" for L in PRIMARY if not r1[L]["read"]] + unread
    if missing:
        return {"verdict": NOT_EVALUATED, "case": 5, "reasons": missing, "missing": missing}
    met = [L for L in PRIMARY if r1[L]["meets"]]
    flags = [t[k]["status"] for k in ("T1", "T2", "T3", "T4") if t[k]["read"] and t[k]["flag"]]
    t_missing = [f"{k}: {t[k]['reason']}" for k in ("T1", "T2", "T3", "T4") if not t[k]["read"]]
    if len(met) == 2:
        if flags:
            v = GO_DOMAIN
        elif t_missing:
            v = GO_UNTESTED
        else:
            v = GO
        return {"verdict": v, "case": 6, "reasons": [f"{L} MEETS" for L in met] + flags, "missing": t_missing,
                "met": met, "flags": flags}
    return {"verdict": PARTIAL, "case": 7, "reasons": [f"{L} MEETS" for L in met], "missing": t_missing, "met": met,
            "flags": flags}


def route(v):
    """The route of rule section 9, as text."""
    x = v["verdict"]
    if x == GO:
        return ("GO: the f2l4 B end is proposed to the owner as a walkway-anchored body beside v1's B end (not in its "
                "place until the owner rules on the claim change of section D), its selected numbers labelled "
                "'post-selection, optimistic'; every T line is printed with it. Before any lockbox use: its own "
                "train54 decode, and a calibration head only in the leave-one-person-out form.")
    if x == GO_DOMAIN:
        return ("GO, walkway-domain only (" + ", ".join(v.get("flags") or []) + "): reported as a source-specific "
                "gain on the weargait_pd walkway, the in-domain result that lever 1 already gives as a readout; the end "
                "is not proposed as a body for G4, the clinician view or another device.")
    if x == GO_UNTESTED:
        return ("GO, transfer not evaluated: reported; the end is not proposed until the missing T input exists and "
                "the qualified verdict is read under this rule. Missing: " + "; ".join(v.get("missing") or []) + ".")
    if x == PARTIAL:
        return ("PARTIAL: reported; the end is not proposed; the line that met: " + ", ".join(v.get("met") or [])
                + (". T flags: " + ", ".join(v["flags"]) if v.get("flags") else "") + ".")
    if x == NO_GO:
        return ("NO-GO: the lever drops at this weight. If the early read and the term's log show the term still large "
                "at the end, the pull was too weak and the JEPA-style lambda is a new rule, not a rerun; if train54 "
                "closed and dev12 did not (T1 PERSON GAP), the anchor learned persons.")
    return "NOT EVALUATED: nothing is proposed; missing: " + "; ".join(v.get("missing") or []) + "."


# ---------------------------------------------------------------- reported, reading nothing
def anchor_blocks(recs, total=PHASE_B_STEPS, block=BLOCK_STEPS):
    """Per block of `block` phase B steps: the median over the log records of walkway_anchor.loss, gain and
    chord_ratio, and of hsic and hsic_active."""
    n_blocks = max(1, total // block)
    acc = {}
    for r in WB().log_events(recs or [], "log", "B"):
        s = fin(r.get("step"))
        if s is None or s < 1:
            continue
        k = min(int((s - 1) // block), n_blocks - 1)
        wa = r.get("walkway_anchor") or {}
        for name, v in (("anchor_loss", wa.get("loss")), ("anchor_gain", wa.get("gain")),
                        ("anchor_chord_ratio", wa.get("chord_ratio")), ("hsic", r.get("hsic")),
                        ("hsic_active", r.get("hsic_active"))):
            if fin(v) is not None:
                acc.setdefault(k, {}).setdefault(name, []).append(float(v))
    out = {}
    for k in sorted(acc):
        hi = total if k == n_blocks - 1 else (k + 1) * block
        out[f"{k * block + 1}-{hi}"] = {n: float(np.median(v)) for n, v in sorted(acc[k].items())}
    return out


def early_read(tree, sets, locked, m):
    """Reported only: paired S(12) and V(12) improvements v1 minus f2l4 at ckpt_B_004000, the --limit-takes 2 G1a of
    both, the in-sim gains at step 4,000."""
    wb, f2 = WB(), F2()
    a = wb.g3set_from_folder(os.path.join(tree, "reports", "gates", "g3", f"curve_{V1_4000}"))
    b = wb.g3set_from_folder(os.path.join(tree, "reports", "gates", "g3", f"curve_{F2L4_4000}"))
    for s, what in ((a, V1_4000), (b, F2L4_4000)):
        f2.guard_set(s, sets, locked, f"curve_{what}")
    out = {"label": "early, reported only", "row_label": ROW_LABEL, "lines": {}}
    for L, key in (("S12", "S"), ("V12", "V")):
        pa, xa = wb.g3_line(a, key, wb.SUB12)
        pb, xb = wb.g3_line(b, key, wb.SUB12)
        p = wb.paired(pa, pb) if pa is not None and pb is not None else {"n": 0, "point": None, "interval": None}
        mg = m["early"][L]["margin"]
        out["lines"][L] = {"v1": xa, "f2l4": xb, "improvement": {k: p.get(k) for k in ("n", "point", "interval")},
                           "margin": mg, "would_meet": None if p.get("point") is None or mg is None else
                           bool(p["point"] >= mg - TOL and p["interval"][0] > TOL)}
    ga, gb = wb.g1_reads(tree, V1_4000).get("lt2"), wb.g1_reads(tree, F2L4_4000).get("lt2")
    out["G1a_lt2"] = {"v1": (ga or {}).get("point"), "f2l4": (gb or {}).get("point")}
    if ga and gb:
        p = wb.paired(ga["takes"], gb["takes"], ga["knots"], gb["knots"])
        out["G1a_lt2"]["improvement_by_take"] = {k: p.get(k) for k in ("n", "point", "interval")}
    else:
        out["G1a_lt2"]["reason"] = "a --limit-takes 2 G1 run is missing"
    out["insim"] = {"v1": f2.insim_block(tree, V1_4000), "f2l4": f2.insim_block(tree, F2L4_4000)}
    return out


# ---------------------------------------------------------------- the run
def run(tree=None, sets=None, locked=None, m=None, check_numbers=True, job_log=None, log=print):
    """Every reading of the rule; returns (results dict, exit code). No file written."""
    t0 = time.time()
    tree = tree or TREE
    rule_sha = check_rule(tree)
    amend_sha = check_amendment(tree)
    amend2_sha = check_amendment_2(tree)
    wb, f2 = WB(), F2()
    res = {"what": LABEL, "row_label": ROW_LABEL, "task": TASK,
           "rule": {"path": RULE_REL.replace("\\", "/"), "sha256": rule_sha,
                    "amendment_1": {"path": AMEND_REL.replace("\\", "/"), "sha256": amend_sha},
                    "amendment_2": {"path": AMEND2_REL.replace("\\", "/"), "sha256": amend2_sha}},
           "asserted_before_any_f2l4_value": True}
    m = m if m is not None else margins(tree, sets, locked)
    res["margins"] = m
    if check_numbers:
        miss = rule_number_check(m, rule_text(tree))
        res["margins_check"] = {"missing_from_rule_text": miss}
        if miss:
            raise HarnessDefect(f"margins differ from the numbers the rule prints: {miss}")
    else:
        res["margins_check"] = {"skipped": True}
    recs, unparsed = f2.run_log(tree, RUN_DIR)
    v1_recs, _ = f2.run_log(tree, V1_RUN_DIR)
    f4_sha, v1_sha = f2.verify_sha(recs), f2.verify_sha(v1_recs)
    res["R0"] = r0_reading(tree, recs, unparsed, job_log)
    v1_set, why_v1 = f2.diag_set(tree, V1_B, sets, locked, v1_sha)
    if v1_set is None:
        raise HarnessDefect(f"v1's diag base row: {why_v1}")
    f4_set, why_f4 = f2.diag_set(tree, F2L4_B, sets, locked, f4_sha)
    res["R1"] = r1_reading(v1_set, f4_set, m, why_f4)
    vals_v1 = {"S14": v1_set["subsets"][wb.SUB14]["S_point"], "V14": v1_set["subsets"][wb.SUB14]["V_point"]}
    vals_f4, reasons = {}, {}
    if f4_set is not None:
        vals_f4.update({"S14": f4_set["subsets"].get(wb.SUB14, {}).get("S_point"),
                        "V14": f4_set["subsets"].get(wb.SUB14, {}).get("V_point")})
    for k in ("S14", "V14"):
        if vals_f4.get(k) is None:
            reasons[k] = why_f4 or f"no {k} in the diag base row"
    for k in ("IC", "MAPE"):
        vals_v1[k], _ = f2.g3dev_line(tree, V1_B, G3_LINE_OF[k], v1_sha)
        vals_f4[k], reasons[k] = f2.g3dev_line(tree, F2L4_B, G3_LINE_OF[k], f4_sha)
    g1v1, _ = f2.g1a_of(tree, V1_B, sha=v1_sha)
    g1f4, reasons["G1a"] = f2.g1a_of(tree, F2L4_B, sha=f4_sha)
    vals_v1["G1a"], vals_f4["G1a"] = (g1v1 or {}).get("point"), (g1f4 or {}).get("point")
    res["R2"] = r2_reading(vals_v1, vals_f4, m, reasons)
    thr = m["transfer"]
    res["T"] = {"T1": t1_reading(tree, sets, locked, thr), "T2": t2_reading(tree, sets, locked, thr),
                "T3": t3_reading(tree, sets, locked), "T4": t4_reading(tree, thr, f4_sha)}
    v = verdict(res["R0"], res["R1"], res["R2"], res["T"])
    res["verdict"] = v
    res["route"] = route(v)
    alone = res["T"]["T2"].get("rule_txt_alone")
    if alone:                       # amendment 1 A2 made T2 missing: the verdict under rule.txt alone, printed beside
        res["verdict_rule_txt_alone"] = verdict(res["R0"], res["R1"], res["R2"],
                                                dict(res["T"], T2=dict(res["T"]["T2"], **alone)))
    rep = {"label": "reported, reading nothing", "row_label": ROW_LABEL}
    rep["blocks"] = {"f2l4": anchor_blocks(recs), "v1": anchor_blocks(v1_recs)}
    rep["insim"] = {"f2l4_B": f2.insim_block(tree, F2L4_B, f4_sha), "v1_B": f2.insim_block(tree, V1_B, v1_sha)}
    rep["dev12_gains"] = {"f2l4_B": f2.dev12_gains(tree, F2L4_B, sets, locked),
                          "v1_B": f2.dev12_gains(tree, V1_B, sets, locked)}
    rep["seconds_per_phase_b_step"] = {"f2l4": f2.seconds_per_step(recs), "v1": f2.seconds_per_step(v1_recs)}
    rep["decoded_stature"] = {"present": False, "reason": "no stature dump of the f2l4 end is queued (rule section 5)"}
    res["reported"] = rep
    res["early"] = early_read(tree, sets, locked, m)
    res["timing_s"] = round(time.time() - t0, 1)
    log(f"verdict of the rule: {v['verdict']} (case {v['case']})")
    return res, (EXIT_NOT_EVALUATED if v["verdict"] == NOT_EVALUATED else EXIT_VERDICT)


# ---------------------------------------------------------------- report
def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def summary_md(r):
    v = r["verdict"]
    L = [f"# F2 lever 4: the walkway-anchored phase B continuation against v1 ({LABEL})", "",
         f"Written by scripts/f2_lever4.py (task {TASK}) under reports/bench/f2_lever4/rule.txt (SHA-256 "
         f"{r['rule']['sha256']}), its amendment 1 (SHA-256 {r['rule']['amendment_1']['sha256']}) and its amendment 2 "
         f"(SHA-256 {r['rule']['amendment_2']['sha256']}), all three asserted before any f2l4 value was read. Every "
         f"f2l4 row is a '{ROW_LABEL}' row, "
         "not a label-free stage-1 row. No lockbox, EVAL, CODY or care_pd record was read. Nothing here is a G3 read "
         "or enters a G3 verdict.", "",
         f"## Verdict of the rule: **{v['verdict']}** (section 8, case {v['case']})", "",
         "Reasons: " + ("; ".join(v["reasons"]) or "none") + ".", "", "## Route", "", r["route"], "",
         "## R0, integrity of the run", "",
         f"Holds: {r['R0']['holds']}. Stop cases: {'; '.join(r['R0']['stop']) or 'none'}. Other failures: "
         f"{'; '.join(r['R0']['fail']) or 'none'}.", "", "| check | ok |", "|---|---|"]
    L += [f"| {k} | {x} |" for k, x in r["R0"]["checks"].items()]
    L += ["", "## R1, primary lines (dev12, paired v1 minus f2l4)", "",
          "| line | label | v1 | f2l4 | improvement | 95 % | persons | margin | meets | reason |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in r["R1"].items():
        nd = ND[k]
        p = x.get("improvement") or {}
        L.append(f"| {k} ({x['unit']}) | {x['label']} | {fmt(x['v1'], nd)} | {fmt(x['f2l4'], nd)} | "
                 f"{fmt(p.get('point'), nd)} | {fmt_iv(p.get('interval'), nd)} | {p.get('n', 0)} | "
                 f"{fmt(x['margin'], nd)} | {x['meets']} | {x['reason'] or ''} |")
    L += ["", "## R2, non-inferiority (f2l4 minus v1; positive is worse)", "",
          "| line | label | v1 | f2l4 | difference | margin | holds | reason |", "|---|---|---|---|---|---|---|---|"]
    for k, x in r["R2"].items():
        nd = ND[k]
        L.append(f"| {k} ({x['unit']}) | {x['label']} | {fmt(x['v1'], nd)} | {fmt(x['f2l4'], nd)} | "
                 f"{fmt(x['difference'], nd)} | {fmt(x['margin'], nd)} | {x['holds']} | {x['reason'] or ''} |")
    L += ["", "## Transfer lines (rule section 8; they qualify a GO and never make one)", "",
          "| line | label | status | reason |", "|---|---|---|---|"]
    for k, x in r["T"].items():
        L.append(f"| {k} {x['flag_name']} | {x['label']} | {x['status']} | {x.get('reason') or ''} |")
    prod = r["T"]["T2"].get("producer")
    if prod:
        bw = ", ".join(f"{k} {v if v is not None else 'n/a'}" for k, v in prod["batch_windows"].items())
        ce = prod["code_sha256_equal"]
        L += ["", f"T2's producer (amendment 2): batch windows {bw} (B5: a difference between v1_B and s2_B puts a "
              "change of the refinement draws into d_pd); code_sha256 one value over the three ends: "
              f"{'n/a' if ce is None else ce} (printed, not a condition)."]
    if r.get("verdict_rule_txt_alone"):
        va = r["verdict_rule_txt_alone"]
        L += ["", f"Under rule.txt alone (without amendment 1 A2 and amendment 2) T2 reads "
              f"{r['T']['T2']['rule_txt_alone']['status']} and the verdict would read {va['verdict']} (case "
              f"{va['case']}); the verdict of the rule is the one above."]
    rep = r["reported"]
    L += ["", f"## Reported, reading nothing ({ROW_LABEL} rows)", "",
          "| block | f2l4 anchor term | gain | chord ratio | f2l4 hsic | hsic_active | v1 hsic | v1 hsic_active |",
          "|---|---|---|---|---|---|---|---|"]
    bf, bv = rep["blocks"]["f2l4"], rep["blocks"]["v1"]
    for blk in sorted(set(bf) | set(bv), key=lambda s: int(s.split("-")[0])):
        a, b = bf.get(blk, {}), bv.get(blk, {})
        L.append(f"| {blk} | {fmt(a.get('anchor_loss'), 3)} | {fmt(a.get('anchor_gain'), 4)} | "
                 f"{fmt(a.get('anchor_chord_ratio'), 4)} | {fmt(a.get('hsic'), 4)} | {fmt(a.get('hsic_active'), 3)} | "
                 f"{fmt(b.get('hsic'), 4)} | {fmt(b.get('hsic_active'), 3)} |")
    for who in ("f2l4_B", "v1_B"):
        ins, dg = rep["insim"][who], rep["dev12_gains"][who]
        L.append(f"- {who}: in-sim stride gain {fmt(ins.get('stride_gain'), 4)}"
                 + ("" if ins.get("present") else f" ({ins.get('reason')})")
                 + f"; dev12 decoded over walkway strides {fmt(dg.get('stride_gain'), 4)}"
                 + ("" if dg.get("present") else f" ({dg.get('reason')})") + ".")
    sp = rep["seconds_per_phase_b_step"]
    L += [f"- Seconds per phase B step: f2l4 {fmt(sp['f2l4'], 3)}, v1 {fmt(sp['v1'], 3)}.",
          f"- Decoded stature: {rep['decoded_stature']['reason']}."]
    e = r["early"]
    L += ["", f"## Early read at ckpt_B_004000 ({e['label']}, {ROW_LABEL})", ""]
    for k, x in e["lines"].items():
        nd = ND[k]
        p = x["improvement"]
        L.append(f"- {k}: v1 {fmt(x['v1'], nd)}, f2l4 {fmt(x['f2l4'], nd)}, improvement {fmt(p.get('point'), nd)} "
                 f"{fmt_iv(p.get('interval'), nd)} against the early margin {fmt(x['margin'], nd)}: would meet "
                 f"{x['would_meet']}.")
    L += ["", "## Margins", "", "Computed from the v1 and s2 files and checked against the numbers the rule prints: "
          + ("skipped (a test)" if r["margins_check"].get("skipped") else "equal") + ".", ""]
    return "\n".join(L) + "\n"


def clean(o):
    return F2().clean(o)


def write_outputs(out_dir, res, tree, started, name="results.json", text=None):
    """results.json (or `name`), summary.md and _provenance.json under out_dir; never over an existing file."""
    from mdfm.gates import common
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(path, json.dumps(clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if text is not None:
        common.assert_no_em_dash(text, "summary.md")
        common.write_text(os.path.join(out_dir, "summary.md"), text)
    prov = {"script": "scripts/f2_lever4.py", "task": TASK, "label": LABEL, "row_label": ROW_LABEL, "run_at": started,
            "output": name, "moveatlas_root": paths.MOVEATLAS, "python": sys.version.split()[0],
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "f2_lever2.py"),
                                               os.path.join(SCRIPTS, "weekend_branch.py"),
                                               os.path.join(SCRIPTS, "g3_gain_check.py")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    try:
        prov["moveatlas_stamp"] = GG().stamp_moveatlas()
    except Exception as e:  # noqa: BLE001  (the stamp is reported, never required)
        prov["moveatlas_stamp"] = {"error": str(e)}
    common.write_text(os.path.join(out_dir, f"_provenance_{os.path.splitext(name)[0]}.json"
                                   if name != "results.json" else "_provenance.json"),
                      json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- --replay
def anchor_replay(tree=None, n_micro=None, log=print):
    """v1's phase B plans through the anchor of configs/stage1_pilot_f2l4.yaml (module docstring, --replay)."""
    from types import SimpleNamespace
    from mdfm.data import body_store as BS, sampler as S, walkway_anchor as WA, window_store as WS
    from mdfm.train import config as CF, loop as LP
    t0 = time.time()
    tree = tree or TREE
    cfg = CF.load(os.path.join(tree, CONFIG_REL))
    assert CF.sha256(cfg) == CONFIG_SHA256, "configs/stage1_pilot_f2l4.yaml is not the rule's"
    ws = WS.WindowStore()
    table = WA.build_anchor_table(ws, cfg.walkway_anchor)
    check = WA.assert_anchor_clean(table, ws.person_lists("train")[WA.SOURCE], WA.refused_persons(ws),
                                   cfg.walkway_anchor.persons_sha256)
    keep = WA.CLOCK_MAX_MS
    try:
        WA.CLOCK_MAX_MS = float("inf")
        table_all = WA.build_anchor_table(ws, cfg.walkway_anchor)
    finally:
        WA.CLOCK_MAX_MS = keep
    smp = S.MixtureSampler(BS.BodyStore(split="train"), ws, None, "B", LP.phase_seed(cfg, "B"), cfg.mixture,
                           guard=False)
    total = int(CF.step_counts(cfg)[1]) * int(cfg.schedule.micro_per_step)
    n = total if n_micro is None else min(int(n_micro), total)
    keys = {int(i): k for i, k in ws.recordings["key"].items()}
    who = {int(i): p for i, p in ws.recordings["person_id"].items()}
    train = ws.person_lists("train")[WA.SOURCE]
    per = {p: {"windows": 0, "anchored": 0, "segments": 0, "seconds": 0.0} for p in train}
    tot = {"weargait_windows": 0, "anchored_windows": 0, "segments": 0, "anchored_seconds": 0.0,
           "without_clock_exclusion": {"anchored_windows": 0, "segments": 0, "anchored_seconds": 0.0}}
    for _ in range(n):
        plan = smp.plan()
        recs, wins = [], []
        for it in plan.items:
            if it.source == WA.SOURCE:
                recs.append(int(it.recording))
                wins.append(SimpleNamespace(synthetic=False, source=WA.SOURCE, key=keys[int(it.recording)],
                                            t0=float(it.t0), duration=float(it.dur)))
                per[who[int(it.recording)]]["windows"] += 1
        tot["weargait_windows"] += len(wins)
        for tb, acc in ((table, tot), (table_all, tot["without_clock_exclusion"])):
            tg = WA.targets_for_windows(tb, recs, wins)
            if not tg.n:
                continue
            w = tg.window.numpy()
            sec = (tg.k_b - tg.k_a).double().numpy() / 100.0
            acc["anchored_windows"] += int(len(set(w.tolist())))
            acc["segments"] += int(tg.n)
            acc["anchored_seconds"] += float(sec.sum())
            if tb is table:
                for i in sorted(set(w.tolist())):
                    p = who[recs[i]]
                    per[p]["anchored"] += 1
                    per[p]["segments"] += int((w == i).sum())
                    per[p]["seconds"] += float(sec[w == i].sum())
    a = np.array([per[p]["anchored"] for p in train], dtype=np.float64)
    out = {"what": "anchor replay of v1's phase B plans under configs/stage1_pilot_f2l4.yaml (plan only)",
           "task": TASK, "config_sha256": CONFIG_SHA256, "micro_steps": n, "micro_steps_of_phase_b": total,
           "table": check, "totals": tot,
           "per_person": {"persons": len(train), "persons_anchored": int((a > 0).sum()),
                          "anchored_min": int(a.min()), "anchored_median": float(np.median(a)),
                          "anchored_max": int(a.max()), "rows": per},
           "seconds": round(time.time() - t0, 1)}
    log(f"replay: {n} micro-steps, {tot['anchored_windows']} anchored windows, persons anchored "
        f"{out['per_person']['persons_anchored']} of {len(train)}, min {out['per_person']['anchored_min']}, median "
        f"{out['per_person']['anchored_median']}")
    return out


# ---------------------------------------------------------------- --preflight
def _norms(grads, groups):
    out = {}
    for name, idx in groups.items():
        s = 0.0
        for i in idx:
            g = grads[i]
            if g is not None:
                s += float((g.double() ** 2).sum())
        out[name] = math.sqrt(s)
    return out


def _cos(ga, gb, idx):
    num = na = nb = 0.0
    for i in idx:
        a, b = ga[i], gb[i]
        if a is None or b is None:
            continue
        a, b = a.double().reshape(-1), b.double().reshape(-1)
        num += float((a * b).sum())
        na += float((a * a).sum())
        nb += float((b * b).sum())
    return num / math.sqrt(na * nb) if na > 0 and nb > 0 else None


class drift_log_in:
    """Context: mdfm.train.checkpoint.load appends a tolerated drift to <folder>/drift_log.jsonl instead of the file
    beside the loaded checkpoint (the review of L4C: a pre-flight wrote a line into v1's run folder)."""

    def __init__(self, folder):
        self.folder = folder

    def __enter__(self):
        from mdfm.train import checkpoint as CK
        self.CK, self.orig = CK, CK.load
        log = os.path.join(self.folder, CK.DRIFT_LOG)
        os.makedirs(self.folder, exist_ok=True)

        def load(path, map_location="cpu", cfg=None, drift_log=None):
            return self.orig(path, map_location=map_location, cfg=cfg, drift_log=drift_log or log)

        CK.load = load
        return log

    def __exit__(self, *exc):
        self.CK.load = self.orig
        return False


def preflight(tree=None, run_dir=None, n_anchored=PREFLIGHT_ANCHORED, max_micro=PREFLIGHT_MAX_MICRO,
              others=PREFLIGHT_OTHERS, log=print):
    """The CPU gradient pre-flight (module docstring, --preflight)."""
    import torch
    from mdfm.data import body_store as BS, collate as CO, walkway_anchor as WA, window_store as WS
    from mdfm.factory import synth as SY
    from mdfm.train import config as CF, losses as LS, loop as LP
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:                                  # pragma: no cover
        proc = None
    t0 = time.time()
    tree = tree or TREE
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2") or 2)))
    cfg = CF.load(os.path.join(tree, CONFIG_REL))
    assert CF.sha256(cfg) == CONFIG_SHA256
    cpu = torch.device("cpu")
    bs, ws = BS.BodyStore(split="train"), WS.WindowStore()
    synth = SY.SynthSampler(bs, device=cpu)
    rd = os.path.join(tree, run_dir or PREFLIGHT_RUN_DIR)
    tr = LP.Trainer(cfg, bs, ws, synth, cpu, rd, prefetch=0, dev_hook=None, echo=None)
    with drift_log_in(rd):                  # the load's drift line goes to the scratch folder, not to v1's run folder
        tr.init_from(os.path.join(tree, SOURCE_PATH))
    model, table, lc = tr.model, tr.anchor_table, tr.loss_cfg
    model.train()
    wB = LS.term_weights("B", 1.0, lc)
    enc = [i for i, (n, _) in enumerate(model.named_parameters()) if n.startswith("encoder.")]
    dec = [i for i, (n, _) in enumerate(model.named_parameters()) if n.startswith("decoder.")]
    params = [p for p in model.parameters()]
    groups = {"encoder": enc, "decoder": dec}
    tr._start_sampler("B")
    rng = np.random.default_rng(20261006)
    micro, done = [], 0
    for m_i in range(int(max_micro)):
        if done >= int(n_anchored):
            break
        it = tr.next_item()
        tg_full = WA.targets_for_step(table, it.step)
        rec = {"micro_step": m_i, "windows": len(it.step.windows), "segments": int(tg_full.n)}
        if not tg_full.n:
            micro.append(rec)
            continue
        items = list(it.step.plan.items)
        anch = sorted(set(tg_full.window.tolist()))
        rest = [i for i in range(len(it.step.windows)) if i not in set(anch)]
        pick = sorted(rng.choice(rest, size=min(int(others), len(rest)), replace=False).tolist()) if rest else []
        sub = anch + pick
        wins = [it.step.windows[i] for i in sub]
        recs = [int(items[i].recording) if not it.step.windows[i].synthetic else -1 for i in sub]
        tg = WA.targets_for_windows(table, recs, wins)
        mb = CO.collate(wins, cfg=tr.collate_cfg, gen=it.step.seed)
        batch, targets = mb.batch.to(cpu), mb.targets.to(cpu)
        N, n_a, m = len(it.step.windows), len(anch), len(pick)
        c = torch.tensor([1.0 / N] * n_a + [(N - n_a) / (max(m, 1) * N)] * m, dtype=torch.float32)
        gen = torch.Generator(device=cpu).manual_seed(int(it.step.seed) % 2 ** 63)
        post = LP.encode(model, batch)
        body, pred = LP.forward_draw(model, batch, post, gen, truth_windows=LS.truth_mask(batch, targets, cpu))
        out = LS.stage1_losses(model, batch, post, body, pred, "B", targets=targets, cfg=lc)
        ta = time.perf_counter()
        ap = LS.walkway_anchor_parts(body.out.root, batch.lengths, tg, batch.rate, float(cfg.walkway_anchor.sd))
        t_anchor = time.perf_counter() - ta
        wa = float(cfg.walkway_anchor.weight)
        objs = {"anchor": (c * wa * ap["term"]).sum(), "rest": (c * out["per_window"]).sum(),
                "sup_root_rate": (c * wB["sup_root_rate"] * out["sup_root_rate"]).sum(),
                "sup_root_rate_u": (c * wB["sup_root_rate"] * out["sup_root_rate_u"]).sum(),
                "sup_root_rate_yaw": (c * wB["sup_root_rate"] * out["sup_root_rate_yaw"]).sum(),
                "imu_nll": (c * wB["imu_nll"] * out["imu_nll"]).sum(),
                "hsic_weighted": (c * wB["hsic"] * out["hsic"]).sum()}
        objs["full"] = objs["rest"] + objs["anchor"]
        grads = {}
        for k, v in objs.items():
            grads[k] = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        norms = {k: _norms(g, groups) for k, g in grads.items()}
        ratio = {k: {grp: (norms["anchor"][grp] / norms[k][grp] if norms[k][grp] > 0 else None) for grp in groups}
                 for k in objs if k != "anchor"}
        rec.update({"sub_batch": {"anchored_windows": n_a, "other_windows_sampled": m, "other_windows": N - n_a,
                                  "tokens": int(mb.useful), "segments": int(tg.n)},
                    "anchor": {"term_sum": float(ap["term"].detach().sum()),
                               "per_window_log_mean_estimate": float(ap["term"].detach().sum()) / N,
                               "gain": float(ap["S"].detach().sum() / ap["D"].detach().sum()) if tg.n else None,
                               "chord_ratio": float(ap["chord"].detach().sum() / ap["S"].detach().sum())
                               if float(ap["S"].detach().sum()) > 0 else None, "seconds": t_anchor},
                    "values": {k: float(v.detach()) for k, v in objs.items()}, "grad_norms": norms,
                    "anchor_over": ratio,
                    "cosine_anchor_hsic_encoder": _cos(grads["anchor"], grads["hsic_weighted"], enc),
                    "rss_gb": proc.memory_info().rss / 1e9 if proc else None})
        micro.append(rec)
        done += 1
        del grads, objs, out, body, pred, post, mb, batch
        log(f"pre-flight micro-step {m_i}: {n_a} anchored windows of {N}; anchor over rest encoder "
            f"{ratio['rest']['encoder']}, decoder {ratio['rest']['decoder']}")
    tr._close_prefetch()
    return {"what": "CPU gradient pre-flight of the walkway anchor (estimate on sub-batches; module docstring)",
            "task": TASK, "config_sha256": CONFIG_SHA256, "source": SOURCE_PATH, "table": tr.anchor_check,
            "device": "cpu", "threads": torch.get_num_threads(), "refinement": "none (amortized posterior)",
            "weights_B": {k: wB[k] for k in ("sup_root_rate", "imu_nll", "hsic")},
            "walkway_anchor_weight": float(cfg.walkway_anchor.weight), "micro_steps": micro,
            "peak_rss_gb": max([r.get("rss_gb") or 0.0 for r in micro] or [0.0]), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- --pin
def input_pin(tree=None):
    """The input pin of the run's job (amendment 1 A5; mdfm.data.walkway_anchor.PIN_ENV): {inputs: TABLE_INPUTS, ...}
    after asserting the rule's hash and that every input file under `tree` hashes to its value."""
    tree = tree or TREE
    rule_sha = check_rule(tree)
    now = {k: sha256_file(os.path.join(tree, k)) for k in TABLE_INPUTS}
    bad = sorted(k for k in TABLE_INPUTS if now[k] != TABLE_INPUTS[k])
    if bad:
        raise HarnessDefect(f"input files differ from rule section 1: {bad}")
    return {"what": "input pin of the walkway-anchor run (mdfm/data/walkway_anchor.py, " + "MDFM_WALKWAY_ANCHOR_PIN)",
            "task": "FIXL4", "rule": {"path": RULE_REL.replace("\\", "/"), "sha256": rule_sha,
                                      "section": "1 (the anchor table at the hash)"},
            "inputs": dict(TABLE_INPUTS)}


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="the tree whose report folders are read (default this tree)")
    p.add_argument("--out", default=None, help="output folder (default reports/bench/f2_lever4/ of the tree)")
    p.add_argument("--write", action="store_true", help="write the outputs also when the verdict is NOT EVALUATED")
    p.add_argument("--replay", action="store_true", help="the anchor replay of v1's phase B plans")
    p.add_argument("--micro-steps", type=int, default=None, help="--replay: the first N micro-steps only")
    p.add_argument("--preflight", action="store_true", help="the CPU gradient pre-flight")
    p.add_argument("--pin", action="store_true", help="write the input pin of the run's job (anchor_inputs.json)")
    return p.parse_args(argv)


def main(argv=None, sets=None, locked=None, m=None, check_numbers=True):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("scripts/f2_lever4.py runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    tree = os.path.abspath(a.tree) if a.tree else TREE
    out_dir = a.out or os.path.join(tree, BENCH)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if a.pin:
        try:
            res = input_pin(tree)
        except HarnessDefect as e:
            print(f"HARNESS DEFECT: {e}")
            return EXIT_HARNESS
        try:
            print(f"written {rel(write_outputs(out_dir, res, tree, started, PIN_NAME), tree)}")
        except FileExistsError as e:
            print(f"{e} exists: nothing rewritten; pass --out with a new folder")
            return EXIT_EXISTS
        return 0
    if a.replay or a.preflight:
        res = anchor_replay(tree, a.micro_steps) if a.replay else preflight(tree)
        res["run_started"] = started
        name = "anchor_replay.json" if a.replay else "preflight.json"
        try:
            print(f"written {rel(write_outputs(out_dir, res, tree, started, name), tree)}")
        except FileExistsError as e:
            print(f"{e} exists: nothing rewritten; pass --out with a new folder")
            return EXIT_EXISTS
        return 0
    check_rule(tree)
    check_amendment(tree)
    check_amendment_2(tree)
    try:
        if sets is None or locked is None:
            from mdfm.gates import g3
            locked = g3.lockbox_persons()
            sets = F2().load_sets(read_json(os.path.join(tree, F1_RESULTS_REL)), locked)
        res, code = run(tree, sets, locked, m, check_numbers)
    except HarnessDefect as e:
        print(f"HARNESS DEFECT: {e}")
        return EXIT_HARNESS
    res = dict(res, run_started=started, sets={k: sorted(v) for k, v in sets.items()},
               lockbox={"persons_in_lockbox": len(set(sets["dev14"]) & set(locked)), "asserted": True})
    if res["lockbox"]["persons_in_lockbox"]:
        print("HARNESS DEFECT: lockbox persons in the sets")
        return EXIT_HARNESS
    text = summary_md(clean(res))
    if code == EXIT_NOT_EVALUATED and not a.write:
        print(text)
        print("NOT EVALUATED: nothing written (missing: " + "; ".join(res["verdict"]["missing"]) + ")")
        return code
    try:
        write_outputs(out_dir, res, tree, started, "results.json", text)
    except FileExistsError as e:
        print(f"{e} exists: nothing rewritten; pass --out with a new folder")
        return EXIT_EXISTS
    print(text)
    print(f"written {rel(out_dir, tree)}; exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
