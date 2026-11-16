"""L5 (2026-10-15): F2 lever 5, the inertial anchor (label-free), a phase-B-only continuation of v1 (run
cache/runs/pilot_f2l5, configs/pilot_f2l5.yaml, both DRAFT): its anchor table, replay, CPU pre-flight, input pin, cost,
smoke check and the readings of its DRAFT rule. Label of every output: dev diagnostic, not a G3 read; every f5 row is
labelled 'inertially anchored body'.

Rule: reports/bench/f2_lever5/rule.txt, a DRAFT (docs/proposals/F2_LEVER5_DRAFT.md, section 5; task L5). It binds once
the owner approves it and its SHA-256 is written in rule.sha256 beside it and in RULE_SHA256 here; until then
check_rule raises RuleDraft and the reading mode reads nothing (exit 3). Its section 7 numbers are those that
rule_numbers() formats from the files (tests/test_f2_lever5.py asserts each one in the text). The readings reuse
scripts/f2_lever4.py (a private instance whose end and labels are this lever's: R1, R2 and T1 to T4 exactly as lever
4 reads them) and add R0's zupt_anchor checks, T5 FEET SLIDE and T6 BEYOND THE INPUT, and the verdict over T1 to T6
(draft section 5): an alarm, a non-finite stop or the memory bound NO-GO; another R0 failure NOT EVALUATED; an R2
line FAILS NO-GO; neither primary line MEETS NO-GO; a missing input NOT EVALUATED; both MEET: GO when T1 to T6 are
read and none flags, 'GO, qualified' naming each flag otherwise, 'GO, transfer not evaluated' when an input is
missing; PARTIAL when exactly one MEETS.

Modes (every one CPU only: CUDA_VISIBLE_DEVICES=-1 is set before torch can load):
--table: the anchor table of configs/pilot_f2l5.yaml on the window store (mdfm.data.zupt_anchor.build_table: the
  version-2 ZUPT strides of the foot sensors of the weargait_pd training fold, from the store's band copy), its check
  summary, counts per task and person and the share of turning strides; beside it, the same ZUPT on the canonical raw
  IMU records of the same recordings (the input of test Z(v2), MoveAtlas through mdfm.data.canonical, training-fold
  persons only, lockbox asserted false), strides paired by foot and still instants (2 samples), the band-copy length
  over the raw one. Writes <out>/table.json.
--replay: v1's phase B plans (plan only: mdfm.data.sampler.MixtureSampler of the config, seed loop.phase_seed, no window
  read or rendered) through the table: anchored windows, strides and seconds per training person and per micro-step.
  Writes <out>/anchor_replay.json (rule section 7 (c)).
--preflight: the run's trainer on the CPU (configs/pilot_f2l5.yaml, init-from cache/runs/phaseA_root_full/
  ckpt_A_010681.pt into --run-dir, the load's drift line kept there), the first phase B micro-steps of v1's sampler
  until PREFLIGHT_ANCHORED hold anchored windows; on each, a sub-batch of every anchored window and PREFLIGHT_OTHERS
  other windows (amortized posterior, no refinement): the term's size (distance and still parts, per anchored window
  and per stride, gain, decoded ankle speed at the instants) beside the rest of the objective, the gradient norms over
  the encoder and the decoder of the term, the rest, sup_root_rate and imu_nll (each per-window value weighted as in
  the full micro-step), the seconds of the anchor part. Writes <out>/preflight.json.
--pin: <out>/anchor_inputs.json, the input pin of the run's job (mdfm.data.zupt_anchor.PIN_ENV): the table's inputs
  and content SHA-256 of <out>/table.json, after asserting that the store files hash to them now.
--cost: the GPU hours of lever 4 (reports/gpu_queue/job_log.csv: its training job and its post-train jobs) and the
  estimate of this lever's jobs from them. Writes <out>/cost.json.
--smoke-check RUN_DIR: the smoke run's log.jsonl (start record's zupt_anchor table check, the phase B zupt_anchor log
  entries, seconds per step, peak memory) beside v1's first phase B steps. Writes <out>/smoke.json.
--rule-numbers: prints the numbers of rule section 7 as rule_numbers() formats them.
(default): the readings of the rule (refused while it is a DRAFT).

Exit codes: 0 a verdict or a written mode output; 3 NOT EVALUATED (the DRAFT rule included); 4 a harness defect; 5 an
output exists. Data discipline: no lockbox, EVAL, CODY or care_pd record; training-fold persons only in --table,
--replay and --preflight; the canonical reads of --table refuse any person outside train54.

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever5.py [--table|--replay|--preflight|--pin|--cost|
        --smoke-check RUN_DIR|--rule-numbers] [--out DIR] [--run-dir DIR]
"""
import argparse
import csv
import datetime
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # every mode of this script stays off the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
BENCH = os.path.join("reports", "bench", "f2_lever5")
RULE_REL = os.path.join(BENCH, "rule.txt")
RULE_SHA_REL = os.path.join(BENCH, "rule.sha256")
RULE_SHA256 = None                           # written on the owner's approval (the rule is a DRAFT)
LABEL = "dev diagnostic, not a G3 read"
ROW_LABEL = "inertially anchored body"
TASK = "L5"
RUN = "pilot_f2l5"
RUN_DIR = "cache/runs/pilot_f2l5"
F5_B = "pilot_f2l5_ckpt_B_032044"
F5_4000 = "pilot_f2l5_ckpt_B_004000"
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
F2L4_B = "pilot_f2l4_ckpt_B_032044"
CONFIG_REL = "configs/pilot_f2l5.yaml"
CONFIG_SHA256 = "5a4483a2b60ba731bbb69c7789c9f899e7f8f152b26fecf6d28f1eedf3c3afd7"
SOURCE_PATH = "cache/runs/phaseA_root_full/ckpt_A_010681.pt"
SOURCE_SHA256 = "ad224565725e6ec973d9b4b2af0e67d2c658241d8644b7d879a8730d70fb5c10"
V1_RUN_DIR = "cache/runs/phaseA_root_full"
TRAIN54_SHA256 = "293f98d02adc48b0fa266c2125be40c23201066403fcb110543d8c882c9c77d6"
ZUPT_ANCHOR = {"enabled": True, "weight": 1.0, "sd": 0.1,
               "tasks": ["SelfPace", "HurriedPace", "SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN", "TUG",
                         "FreeWalk", "SelfPace_doorpat"],
               "persons_sha256": TRAIN54_SHA256, "edge_s": 0.1, "min_length_m": 0.3}
TABLE_JSON, REPLAY_JSON, PREFLIGHT_JSON, PIN_NAME, COST_JSON, SMOKE_JSON = (
    "table.json", "anchor_replay.json", "preflight.json", "anchor_inputs.json", "cost.json", "smoke.json")
PHASE_B_STEPS = 32044
TOL = 1e-9
SEED_MULTIPLE = 2.0
FOLLOW_SHARE = 0.5
BEYOND = 1.02                       # T6: the decoded stride over the ZUPT stride above this reads BEYOND
LEGS_ROW = "model_legs"             # T5: the dev12 legs readout (stance no-slip) row of the bw32 ablations run
T3_ROWS = ("base", "sites6", "family_legs", "family_wrists")
MIN_ROW_PERSONS = 10
FLAGS = {"T1": "PERSON GAP", "T2": "CONTROL ONLY", "T3": "LAYOUT KEYED", "T4": "SIM OVERSHOOT", "T5": "FEET SLIDE",
         "T6": "BEYOND THE INPUT"}
T_LINES = tuple(FLAGS)
NO_GO, GO, PARTIAL, NOT_EVALUATED = "NO-GO", "GO", "PARTIAL", "NOT EVALUATED"
GO_QUALIFIED, GO_UNTESTED = "GO, qualified", "GO, transfer not evaluated"
EXIT_VERDICT, EXIT_NOT_EVALUATED, EXIT_HARNESS, EXIT_EXISTS = 0, 3, 4, 5
ZUPT_V2_STRIDES = os.path.join("reports", "bench", "scale", "zupt_v2", "real_strides.csv")
CHAIN_RESULTS = os.path.join("reports", "bench", "scale", "chain", "results.json")
Q_RESULTS = os.path.join("reports", "bench", "scale", "q", "results.json")
F2L4_RESULTS = os.path.join("reports", "bench", "f2_lever4", "results.json")
ZUPT_V2_RESULTS = os.path.join("reports", "bench", "scale", "zupt_v2", "results.json")
JOB_LOG_REL = os.path.join("reports", "gpu_queue", "job_log.csv")
PREFLIGHT_ANCHORED = 2
PREFLIGHT_MAX_MICRO = 12
PREFLIGHT_OTHERS = 12
PREFLIGHT_RUN_DIR = "cache/runs/preflight_f2l5"
SMOKE_STEPS = 200                   # --cost: the smoke job's phase B steps
STARTUP_HOURS = 0.1                 # --cost: an allowance for a job's start (the table build, the init-from load)
PAIR_SAMPLES = 2                    # --table: a raw stride pairs a table stride when both instants lie within this


class HarnessDefect(AssertionError):
    """A departure of the harness itself (a number that differs from the rule's, a person outside the set)."""


class RuleDraft(RuntimeError):
    """The rule is a DRAFT: no rule.sha256, or no RULE_SHA256 in this script."""


# ---------------------------------------------------------------- lever 4's readers, on this lever's end
def L4():
    """A private instance of scripts/f2_lever4.py whose end, run and labels are this lever's (its R1, R2 and T1 to T4
    then read the f5 end exactly as lever 4 read its own)."""
    key = "_l5_f2_lever4"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, "f2_lever4.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    m.F2L4_B, m.F2L4_4000, m.RUN, m.RUN_DIR, m.ROW_LABEL, m.TASK = F5_B, F5_4000, RUN, RUN_DIR, ROW_LABEL, TASK
    return m


def renamed(o):
    """o with the keys 'f2l4' and 'f2l4_B' of lever 4's readers renamed 'f5' and 'f5_B'."""
    if isinstance(o, dict):
        return {{"f2l4": "f5", "f2l4_B": "f5_B"}.get(k, k): renamed(v) for k, v in o.items()}
    if isinstance(o, list):
        return [renamed(v) for v in o]
    return o


def sha256_file(path):
    return L4().sha256_file(path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel(path, tree=None):
    try:
        return os.path.relpath(os.path.abspath(path), tree or TREE).replace("\\", "/")
    except ValueError:                  # another drive
        return str(path).replace("\\", "/")


def check_rule(tree=None):
    """The SHA-256 of rule.txt, asserted equal to rule.sha256 and RULE_SHA256; RuleDraft while either is absent."""
    tree = tree or TREE
    p = os.path.join(tree, RULE_SHA_REL)
    if RULE_SHA256 is None or not os.path.exists(p):
        raise RuleDraft("the lever-5 rule is a DRAFT (no rule.sha256 or no RULE_SHA256): nothing is read")
    got = sha256_file(os.path.join(tree, RULE_REL))
    with open(p, encoding="utf-8") as f:
        stored = (f.read().split() or [""])[0]
    if not (got == stored == RULE_SHA256):
        raise AssertionError(f"rule hash mismatch: rule.txt {got}, rule.sha256 {stored}, script {RULE_SHA256}")
    return got


def load_sets(tree=None):
    from mdfm.gates import g3
    locked = g3.lockbox_persons()
    sets = L4().F2().load_sets(read_json(os.path.join(tree or TREE, L4().F1_RESULTS_REL)), locked)
    return sets, locked


# ---------------------------------------------------------------- margins and thresholds
def transfer_refs(tree, sets, locked):
    """The thresholds that lever 4 computed at its reading and the draft names (section 5), from v1_B and s2_B files
    only: d_pd (the PD dev fold's stride gain, dev_pd_<end>/events.csv), d_r of T3_ROWS and d_legs of LEGS_ROW (the
    dev12 stride gain of the bw32 ablations runs), each |v1_B minus s2_B|, with the end values; lever 4's own end
    beside."""
    l4 = L4()
    pd_dev, pool = l4.pd_dev_persons(tree), l4.pd_pool_persons(tree)
    out = {"ends": {}}
    for key, name in (("v1_B", V1_B), ("s2_B", S2_B), ("f2l4_B", F2L4_B)):
        e = {}
        e["pd_stride"] = l4._stride_gain(os.path.join(tree, "reports", "gates", "g3", f"dev_pd_{name}", "events.csv"),
                                         pool, locked, pd_dev)[0]
        for row in T3_ROWS + (LEGS_ROW,):
            e[row] = l4._stride_gain(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}_ablations_bw32",
                                                  "events.csv"), sets["dev14"], locked, sets["dev12"], row=row,
                                     min_persons=MIN_ROW_PERSONS)[0]
        out["ends"][key] = e
    a, b = out["ends"]["v1_B"], out["ends"]["s2_B"]
    out["d_pd"] = abs(a["pd_stride"] - b["pd_stride"])
    out["d_r"] = {r: abs(a[r] - b[r]) for r in T3_ROWS}
    out["d_legs"] = abs(a[LEGS_ROW] - b[LEGS_ROW])
    return out


def margins(tree, sets, locked):
    """Lever 4's margins and transfer thresholds (scripts/f2_lever4.py margins: the same lines, seed differences and
    margins) with transfer_refs."""
    m = L4().margins(tree, sets, locked)
    m["transfer"].update(transfer_refs(tree, sets, locked))
    return m


def reach_numbers(tree):
    """Section 7 (d): test Q's real stride gains and their intervals (reports/bench/scale/q/results.json, arms q0w20,
    q20, q60) and lever 4's dev12 stride gains at v1_B and its end (reports/bench/f2_lever4/results.json, T1)."""
    q = read_json(os.path.join(tree, Q_RESULTS))["real"]
    r4 = read_json(os.path.join(tree, F2L4_RESULTS))["T"]["T1"]["ends"]
    return {"q": {arm: {"gain": q[arm]["gain"], "interval": q[arm]["interval"], "slide": q[arm]["slide"]["median"]}
                  for arm in ("q0w20", "q20", "q60")},
            "f2l4_dev12": {"v1_B": r4["v1_B"]["dev12_stride_gain"], "f2l4_B": r4["f2l4_B"]["dev12_stride_gain"]}}


def rule_numbers(tree=None, sets=None, locked=None):
    """{placeholder: the string the rule prints} of its section 7 and of the table of its section 1, from the files
    (tests/test_f2_lever5.py asserts every one in the rule's text), and checks {name: bool}."""
    tree = tree or TREE
    if sets is None or locked is None:
        sets, locked = load_sets(tree)
    l4 = L4()
    P = l4._sibling("probe_walkway_anchor")
    m = margins(tree, sets, locked)
    train, dev14, dev12, lb, _ = P.person_sets()
    att = P.attainability(dev14, dev12, set(lb))
    tab = read_json(os.path.join(tree, BENCH, TABLE_JSON))
    rp = read_json(os.path.join(tree, BENCH, REPLAY_JSON))
    s, v, sd, ni, e, t = (m["success"]["S12"], m["success"]["V12"], m["seed_diff"], m["noninferiority"], m["early"],
                          m["transfer"])
    sc_s, sc_v = att["gain_scan"]["S12"], att["gain_scan"]["V12"]
    chk, cnt = tab["table"], tab["table"]["counts"]
    pp, tot = rp["per_person"], rp["totals"]
    out = {"CONFIG_SHA": CONFIG_SHA256, "TRAIN54": TRAIN54_SHA256, "TABLE_SHA": chk["table_sha256"],
           "REC": f"{cnt['recordings']}", "STRIDES": f"{cnt['strides']:,}",
           "SPP_MIN": f"{cnt['strides_per_person']['min']}", "SPP_MED": f"{cnt['strides_per_person']['median']:g}",
           "SPP_MAX": f"{cnt['strides_per_person']['max']}", "TURN20": f"{cnt['turn_over_20_deg']:,}",
           "S12_LINE": f"{s['line']:.4f}", "S12_D": f"{sd['S12']:.4f}", "S12_M": f"{s['margin']:.4f}",
           "S12_NEED": f"{s['needed_line_at_most']:.4f}", "V12_LINE": f"{v['line']:.4f}", "V12_D": f"{sd['V12']:.4f}",
           "V12_M": f"{v['margin']:.4f}", "V12_NEED": f"{v['needed_line_at_most']:.4f}",
           "HEAD_S": f"{sc_s['head_line']:.4f}", "HEAD_V": f"{sc_v['head_line']:.4f}",
           "F_S": f"{sc_s['smallest_share_meeting']['f']:.2f}", "F_V": f"{sc_v['smallest_share_meeting']['f']:.2f}",
           "E_S": f"{e['S12']['line']:.2f}", "E_V": f"{e['V12']['line']:.4f}", "E_S_M": f"{e['S12']['margin']:.2f}",
           "E_V_M": f"{e['V12']['margin']:.4f}",
           "D_GAP": f"{t['d_gap']:.4f}", "D_DEV12": f"{t['d_dev12_stride']:.4f}", "D_INSIM": f"{t['d_insim']:.4f}",
           "D_PD": f"{t['d_pd']:.4f}", "D_BASE": f"{t['d_r']['base']:.4f}", "D_SITES6": f"{t['d_r']['sites6']:.4f}",
           "D_LEGS_FAM": f"{t['d_r']['family_legs']:.4f}", "D_WRISTS": f"{t['d_r']['family_wrists']:.4f}",
           "D_LEGS": f"{t['d_legs']:.4f}", "LEGS_V1": f"{t['ends']['v1_B'][LEGS_ROW]:.4f}",
           "LEGS_S2": f"{t['ends']['s2_B'][LEGS_ROW]:.4f}",
           "L4_BASE_RISE": f"{t['ends']['f2l4_B']['base'] - t['ends']['v1_B']['base']:+.4f}",
           "L4_LEGS_RISE": f"{t['ends']['f2l4_B'][LEGS_ROW] - t['ends']['v1_B'][LEGS_ROW]:+.4f}",
           "C_ANCH": f"{pp['persons_anchored']}", "C_MIN": f"{pp['anchored_min']}",
           "C_MED": f"{pp['anchored_median']:g}",
           "C_WIN": f"{tot['anchored_windows']:,}", "C_STR": f"{tot['strides']:,}",
           "C_SEC": f"{tot['anchored_seconds']:,.1f}", "C_MICRO": f"{rp['micro_steps']:,}",
           "C_PER_MICRO": f"{tot['anchored_windows'] / rp['micro_steps']:.2f}",
           "C_STR_MICRO": f"{rp['per_micro_step']['strides_mean']:.2f}"}
    rn = reach_numbers(tree)
    for arm, k in (("q0w20", "Q0"), ("q20", "Q20"), ("q60", "Q60")):
        out[k] = f"{rn['q'][arm]['gain']:.3f}"
        out[k + "_SLIDE"] = f"{rn['q'][arm]['slide']:.3f}"
    out["Q20_IV"] = f"[{rn['q']['q20']['interval'][0]:.3f}, {rn['q']['q20']['interval'][1]:.3f}]"
    out["L4_V1_DEV12"] = f"{rn['f2l4_dev12']['v1_B']:.3f}"
    out["L4_END_DEV12"] = f"{rn['f2l4_dev12']['f2l4_B']:.3f}"
    ra = tab["raw_agreement"]
    out.update({"PAIR_N": f"{ra['table_strides']:,}", "PAIR_SHARE": f"{ra['paired_share']:.3f}",
                "BR_MED": f"{ra['band_over_raw_median']:.4f}", "BR_P05": f"{ra['band_over_raw_p05_p95'][0]:.4f}",
                "BR_P95": f"{ra['band_over_raw_p05_p95'][1]:.4f}"})
    z = read_json(os.path.join(tree, ZUPT_V2_RESULTS))
    out["Z_TRAIN54"] = f"{z['Z1']['train54']['median']:.3f}"
    out["Z_SIM"] = f"{z['Z2']['forward_scored']['median']:.3f}"
    zc = os.path.join(tree, ZUPT_V2_STRIDES)
    for key, name in (("T6_V1", V1_B), ("T6_L4", F2L4_B)):
        g, n = decoded_over_zupt(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}", "events.csv"), zc,
                                 sets["dev12"], locked)
        out[key], out[key + "_N"] = f"{g:.3f}", f"{n}"
    for k, key in (("S14", "S14"), ("V14", "V14"), ("IC", "IC"), ("G1A", "G1a"), ("MAPE", "MAPE")):
        out[k] = f"{ni[key]['margin']:.4f}"
    l4_base_rise = t["ends"]["f2l4_B"]["base"] - t["ends"]["v1_B"]["base"]
    l4_legs_rise = t["ends"]["f2l4_B"][LEGS_ROW] - t["ends"]["v1_B"][LEGS_ROW]
    checks = {"attainable": bool(att["attainable"] and s["margin"] < s["line"] and v["margin"] < v["line"]
                                 and pp["persons_anchored"] == pp["persons"] == 54),
              "head_meets_both": bool(sc_s["head_improvement"] >= s["margin"]
                                      and sc_v["head_improvement"] >= v["margin"]),
              "lever4_end_reads_feet_slide": bool(t5_flag(l4_base_rise, l4_legs_rise, t["d_r"]["base"], t["d_legs"]))}
    return out, checks


# ---------------------------------------------------------------- R0
def r0_reading(tree, recs, unparsed, job_log=None, table=None):
    """R0 of the rule: lever 4's checks (source, config, verify at step 32,044, report mismatches, alarms, non-finite
    stop, memory bound, job exit) with the inertial anchor's: every start event carries the section of rule section 1
    and a passed table check with 54 persons of the train54 SHA-256, the streams read imu alone, the inputs and the
    table SHA-256 of section 1 (`table`: table.json's check); walkway_anchor, render_aug and jepa absent; every phase B
    log record carries a zupt_anchor entry at weight 1.0."""
    l4 = L4()
    wb = l4.WB()
    stop, fail, checks, values = [], [], {}, {"unparsed_log_lines": unparsed}
    if recs is None:
        fail.append(f"no run log {RUN_DIR}/log.jsonl")
        return {"holds": False, "stop": stop, "fail": fail, "checks": checks, "values": values}
    inits, starts = wb.log_events(recs, "init_from"), wb.log_events(recs, "start")
    bad_init = [str((r.get("source") or {}).get("path")) for r in inits
                if not str((r.get("source") or {}).get("path") or "").replace("\\", "/").endswith(SOURCE_PATH)
                or (r.get("source") or {}).get("sha256") != SOURCE_SHA256]
    checks["init_from"] = bool(inits) and not bad_init
    if not inits or bad_init:
        fail.append("no init_from event" if not inits
                    else "init_from source other than section 1: " + "; ".join(bad_init))
    bad_cfg = sorted({str(s.get("config_sha256")) for s in starts if s.get("config_sha256") != CONFIG_SHA256})
    checks["start_config"] = bool(starts) and not bad_cfg
    if not starts:
        fail.append("no start event")
    if bad_cfg:
        fail.append("start event at another config SHA-256: " + ", ".join(x[:12] for x in bad_cfg))
    issues = []
    want = table or {}
    for s in starts:
        za = s.get("zupt_anchor") or {}
        sec, tab = za.get("section"), za.get("table") or {}
        if sec != ZUPT_ANCHOR:
            issues.append(f"section {sec}")
        if tab.get("check") != "passed" or tab.get("persons_sha256") != TRAIN54_SHA256 or tab.get("persons") != 54:
            issues.append("table check not passed on train54")
        if tab.get("streams_read") != ["imu"]:
            issues.append(f"streams read {tab.get('streams_read')}")
        if not want or tab.get("inputs") != want.get("inputs") or tab.get("table_sha256") != want.get("table_sha256"):
            issues.append("table inputs or content SHA-256 other than section 1's")
        for other in ("walkway_anchor", "render_aug", "jepa"):
            if s.get(other) is not None:
                issues.append(f"{other} present")
    checks["zupt_anchor_start"] = bool(starts) and not issues
    if issues:
        fail.append("zupt_anchor start entry: " + "; ".join(sorted(set(issues))))
    logs_b = wb.log_events(recs, "log", "B")
    bad_log = [r.get("step") for r in logs_b if (r.get("zupt_anchor") or {}).get("weight") != 1.0]
    checks["zupt_anchor_logged"] = bool(logs_b) and not bad_log
    if logs_b and bad_log:
        fail.append(f"phase B log records without the zupt_anchor entry at weight 1.0 (steps {bad_log[:3]})")
    v = wb.b_end_verify(recs)
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
    checks["no_alarm"] = not alarms
    if alarms:
        stop.append(f"alarm ({len(alarms)} alarm events)")
    if wb.nonfinite_stop(recs, l4.NONFINITE_RUN):
        stop.append("non-finite stop")
    checks["no_nonfinite_stop"] = not any("non-finite" in x for x in stop)
    peak = wb.peak_reserved_gb(recs)
    values["peak_reserved_gb"] = peak
    checks["peak_reserved"] = peak is not None and peak <= l4.PEAK_RESERVED_MAX_GB + TOL
    if peak is None:
        fail.append("no memory_gb.max_reserved in the log")
    elif peak > l4.PEAK_RESERVED_MAX_GB + TOL:
        stop.append(f"memory bound (peak reserved {peak:.2f} GB above {l4.PEAK_RESERVED_MAX_GB})")
    code, jpath = wb.job_exit(tree, job_log or os.path.join(tree, JOB_LOG_REL), RUN_DIR)
    values["job_exit"] = code
    checks["job_exit_0"] = code == 0
    if code is None:
        fail.append(f"no job_log.csv row of a scripts/train_stage1.py job on {RUN_DIR}")
    elif code != 0:
        fail.append(f"last train_stage1.py job on {RUN_DIR} exited {code}")
    return {"holds": not stop and not fail, "stop": stop, "fail": fail, "checks": checks, "values": values}


# ---------------------------------------------------------------- T5 and T6
def t5_flag(base_rise, legs_rise, d_base, d_legs):
    """FEET SLIDE (draft section 5): the base row's dev12 stride gain rises beyond 2 x d_base and the legs readout's
    rise is below half of it or below 2 x d_legs."""
    return bool(base_rise > SEED_MULTIPLE * d_base + TOL and
                (legs_rise < FOLLOW_SHARE * base_rise - TOL or legs_rise < SEED_MULTIPLE * d_legs - TOL))


def t5_reading(tree, sets, locked, thr):
    """T5 FEET SLIDE on the bw32 ablations runs of v1_B and the f5 end (rows base and LEGS_ROW, dev12)."""
    l4 = L4()
    out = {"line": "T5", "flag_name": FLAGS["T5"], "label": ROW_LABEL, "d_base": thr["d_r"]["base"],
           "d_legs": thr["d_legs"], "ends": {}}
    why = []
    for key, name in (("v1_B", V1_B), ("f5_B", F5_B)):
        e = {}
        for row in ("base", LEGS_ROW):
            g, n, n14, w = l4._stride_gain(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}_ablations_bw32",
                                                        "events.csv"), sets["dev14"], locked, sets["dev12"], row=row,
                                           min_persons=MIN_ROW_PERSONS)
            e[row] = g
            if w:
                why.append(w)
        out["ends"][key] = e
    a, b = out["ends"]["v1_B"], out["ends"]["f5_B"]
    if any(x is None for x in (a["base"], b["base"], a[LEGS_ROW], b[LEGS_ROW])):
        out.update(read=False, flag=None, status="missing", reason="; ".join(sorted(set(why))) or "a gain is missing")
        return out
    rb, rl = b["base"] - a["base"], b[LEGS_ROW] - a[LEGS_ROW]
    flag = t5_flag(rb, rl, thr["d_r"]["base"], thr["d_legs"])
    out.update(read=True, base_rise=rb, legs_rise=rl, flag=flag, status=FLAGS["T5"] if flag else "HELD", reason=None)
    return out


def decoded_over_zupt(events, zupt_csv, persons, locked):
    """(median of the decoded stride over the version-2 ZUPT stride, n strides) over the matched base-row strides of
    an events.csv whose (trial_id, foot, t0, t1) pair a matched stride of zupt_v2/real_strides.csv, persons only
    (reports/bench/scale/code/zupt_same_strides.py's join); (None, 0) when either file is absent."""
    import pandas as pd
    if not (os.path.exists(events) and os.path.exists(zupt_csv)):
        return None, 0
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(events, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[(ev["row"] == "base") & (ev["kind"] == "stride") & (ev["matched"].str.strip().str.lower() == "true")
            & ev["person_id"].isin(set(persons))]
    if set(ev["person_id"]) & set(locked):
        raise HarnessDefect("a lockbox person in the events")
    ev = ev.assign(t0=pd.to_numeric(ev["t0_s"]).round(2), t1=pd.to_numeric(ev["t1_s"]).round(2),
                   y=pd.to_numeric(ev["prediction"]))
    z = pd.read_csv(zupt_csv, encoding="utf-8")
    z = z[z["person_id"].isin(set(persons)) & (z["matched"].astype(str).str.lower() == "true")].copy()
    z["t0"], z["t1"] = z["t0"].round(2), z["t1"].round(2)
    j = ev.merge(z[["trial_id", "foot", "t0", "t1", "zupt_m"]], on=["trial_id", "foot", "t0", "t1"], how="inner")
    r = (j["y"] / (100.0 * j["zupt_m"])).to_numpy(np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    return (float(np.median(r)) if len(r) else None), int(len(r))


def t6_reading(tree, sets, locked):
    """T6 BEYOND THE INPUT: the dev12 decoded stride of the f5 end (diag events, base row) over the version-2 ZUPT
    stride, median over the strides both match; BEYOND above BEYOND. v1_B's value beside (descriptive)."""
    zc = os.path.join(tree, ZUPT_V2_STRIDES)
    out = {"line": "T6", "flag_name": FLAGS["T6"], "label": ROW_LABEL, "threshold": BEYOND, "ends": {}}
    for key, name in (("v1_B", V1_B), ("f5_B", F5_B)):
        g, n = decoded_over_zupt(os.path.join(tree, "reports", "gates", "g3", f"diag_{name}", "events.csv"), zc,
                                 sets["dev12"], locked)
        out["ends"][key] = {"decoded_over_zupt": g, "strides": n}
    g = out["ends"]["f5_B"]["decoded_over_zupt"]
    if g is None:
        out.update(read=False, flag=None, status="missing",
                   reason="no f5 diag events or no " + ZUPT_V2_STRIDES.replace("\\", "/"))
        return out
    flag = bool(g > BEYOND + TOL)
    out.update(read=True, flag=flag, status=FLAGS["T6"] if flag else "HELD", reason=None)
    return out


# ---------------------------------------------------------------- verdict and route
def verdict(r0, r1, r2, t):
    """Draft section 5, in lever 4's order (module docstring), with T1 to T6."""
    l4 = L4()
    if r0["stop"]:
        return {"verdict": NO_GO, "case": 1, "reasons": list(r0["stop"]), "missing": []}
    if r0["fail"]:
        return {"verdict": NOT_EVALUATED, "case": 2, "reasons": list(r0["fail"]), "missing": list(r0["fail"])}
    failed = [k for k in l4.R2_LINES if r2[k]["read"] and not r2[k]["holds"]]
    unread = [f"{k}: {r2[k]['reason']}" for k in l4.R2_LINES if not r2[k]["read"]]
    if failed:
        return {"verdict": NO_GO, "case": 3, "reasons": [f"{k} FAILS" for k in failed], "missing": unread}
    if all(r1[L]["read"] for L in l4.PRIMARY) and not any(r1[L]["meets"] for L in l4.PRIMARY):
        return {"verdict": NO_GO, "case": 4, "reasons": ["neither S(12) nor V(12) MEETS"], "missing": unread}
    missing = [f"{L}: {r1[L]['reason']}" for L in l4.PRIMARY if not r1[L]["read"]] + unread
    if missing:
        return {"verdict": NOT_EVALUATED, "case": 5, "reasons": missing, "missing": missing}
    met = [L for L in l4.PRIMARY if r1[L]["meets"]]
    flags = [t[k]["status"] for k in T_LINES if t[k]["read"] and t[k]["flag"]]
    t_missing = [f"{k}: {t[k]['reason']}" for k in T_LINES if not t[k]["read"]]
    if len(met) == 2:
        v = GO_QUALIFIED if flags else (GO_UNTESTED if t_missing else GO)
        return {"verdict": v, "case": 6, "reasons": [f"{L} MEETS" for L in met] + flags, "missing": t_missing,
                "met": met, "flags": flags}
    return {"verdict": PARTIAL, "case": 7, "reasons": [f"{L} MEETS" for L in met], "missing": t_missing, "met": met,
            "flags": flags}


def route(v):
    """The routes of the rule (section 9), as text."""
    x = v["verdict"]
    if x == GO:
        return ("GO: the f5 B end is proposed to the owner as an inertially anchored body beside v1's B end, its "
                "selected numbers labelled 'post-selection, optimistic', every T line and the K decomposition printed "
                "with it.")
    if x == GO_QUALIFIED:
        return ("GO, qualified (" + ", ".join(v.get("flags") or []) + "): reported with each flag named; the end is "
                "not proposed for what a flag excludes (LAYOUT KEYED: inputs without foot sensors; FEET SLIDE: the feet; "
                "BEYOND THE INPUT: the stride scale; PERSON GAP, CONTROL ONLY or SIM OVERSHOOT: as lever 4's "
                "'walkway-domain only').")
    if x == GO_UNTESTED:
        return ("GO, transfer not evaluated: reported; the end is not proposed until the missing T input exists. "
                "Missing: " + "; ".join(v.get("missing") or []) + ".")
    if x == PARTIAL:
        return "PARTIAL: reported; the end is not proposed; the line that met: " + ", ".join(v.get("met") or []) + "."
    if x == NO_GO:
        return ("NO-GO: the lever drops at this weight; the upstream alternative of the draft (section 8 item 5) is "
                "the body track's next candidate, after its own diagnostic.")
    return "NOT EVALUATED: nothing is proposed; missing: " + "; ".join(v.get("missing") or []) + "."


# ---------------------------------------------------------------- the reading
def run(tree=None, sets=None, locked=None, m=None, job_log=None, log=print):
    """Every reading of the rule; returns (results dict, exit code). RuleDraft while the rule is a DRAFT."""
    t0 = time.time()
    tree = tree or TREE
    rule_sha = check_rule(tree)
    l4 = L4()
    f2 = l4.F2()
    if sets is None or locked is None:
        sets, locked = load_sets(tree)
    m = m if m is not None else margins(tree, sets, locked)
    table = read_json(os.path.join(tree, BENCH, TABLE_JSON))["table"]
    res = {"what": LABEL, "row_label": ROW_LABEL, "task": TASK, "rule": {"path": RULE_REL.replace("\\", "/"),
                                                                         "sha256": rule_sha}, "margins": m}
    recs, unparsed = f2.run_log(tree, RUN_DIR)
    f5_sha = f2.verify_sha(recs)
    v1_sha = f2.verify_sha(f2.run_log(tree, V1_RUN_DIR)[0])
    res["R0"] = r0_reading(tree, recs, unparsed, job_log, table)
    v1_set, why_v1 = f2.diag_set(tree, V1_B, sets, locked, v1_sha)
    if v1_set is None:
        raise HarnessDefect(f"v1's diag base row: {why_v1}")
    f5_set, why_f5 = f2.diag_set(tree, F5_B, sets, locked, f5_sha)
    res["R1"] = renamed(l4.r1_reading(v1_set, f5_set, m, why_f5))
    wb = l4.WB()
    vals_v1 = {"S14": v1_set["subsets"][wb.SUB14]["S_point"], "V14": v1_set["subsets"][wb.SUB14]["V_point"]}
    vals_f5, reasons = {}, {}
    if f5_set is not None:
        vals_f5.update({"S14": f5_set["subsets"].get(wb.SUB14, {}).get("S_point"),
                        "V14": f5_set["subsets"].get(wb.SUB14, {}).get("V_point")})
    for k in ("S14", "V14"):
        if vals_f5.get(k) is None:
            reasons[k] = why_f5 or f"no {k} in the diag base row"
    for k in ("IC", "MAPE"):
        vals_v1[k], _ = f2.g3dev_line(tree, V1_B, l4.G3_LINE_OF[k], v1_sha)
        vals_f5[k], reasons[k] = f2.g3dev_line(tree, F5_B, l4.G3_LINE_OF[k], f5_sha)
    g1v1, _ = f2.g1a_of(tree, V1_B, sha=v1_sha)
    g1f5, reasons["G1a"] = f2.g1a_of(tree, F5_B, sha=f5_sha)
    vals_v1["G1a"], vals_f5["G1a"] = (g1v1 or {}).get("point"), (g1f5 or {}).get("point")
    r2 = l4.r2_reading(vals_v1, vals_f5, m, reasons)
    res["R2"] = renamed(r2)
    thr = m["transfer"]
    t = {"T1": l4.t1_reading(tree, sets, locked, thr), "T2": l4.t2_reading(tree, sets, locked, thr),
         "T3": l4.t3_reading(tree, sets, locked), "T4": l4.t4_reading(tree, thr, f5_sha),
         "T5": t5_reading(tree, sets, locked, thr), "T6": t6_reading(tree, sets, locked)}
    res["T"] = renamed(t)
    v = verdict(res["R0"], res["R1"], r2, t)
    res["verdict"], res["route"] = v, route(v)
    res["reported"] = {"label": "reported, reading nothing", "row_label": ROW_LABEL,
                       "k_decomposition": {"present": False, "reason": "the K decomposition of the f5 end against v1_B "
                                           "(reports/bench/scale/code/k_run.py) is a CPU job of the reading task"},
                       "seconds_per_phase_b_step": {"f5": f2.seconds_per_step(recs)}}
    res["timing_s"] = round(time.time() - t0, 1)
    log(f"verdict of the rule: {v['verdict']} (case {v['case']})")
    return res, (EXIT_NOT_EVALUATED if v["verdict"] == NOT_EVALUATED else EXIT_VERDICT)


# ---------------------------------------------------------------- --table
def _foot_channels(record):
    """{foot: (acc (n, 3) g, gyro (n, 3) deg/s)} of a canonical IMU record, NaN at an invalid sample (the SCALE
    zupt_run.foot_channels)."""
    from mdfm.data import zupt_anchor as ZA
    sig = np.asarray(record["signal"], dtype=np.float64)
    loc = [int(x) for x in np.asarray(record["location_ids"])]
    cm = np.asarray(record["channel_mask"], dtype=bool)
    valid = np.asarray(record["sample_valid"], dtype=bool) if record.get("sample_valid") is not None \
        else np.ones(sig.shape[:2], bool)
    out = {}
    for foot, site in ZA.FOOT_SITES.items():
        if site in loc and cm[loc.index(site), 0:6].all():
            j = loc.index(site)
            a, g = sig[:, j, 0:3].copy(), sig[:, j, 3:6].copy()
            a[~valid[:, j]] = np.nan
            g[~valid[:, j]] = np.nan
            out[foot] = (a, g)
    return out


def raw_agreement(ws, table, train, log=print):
    """The ZUPT of the canonical raw records of the table's recordings against the table's strides (module docstring,
    --table): per stride of the table, the raw stride of the same foot whose instants lie within PAIR_SAMPLES."""
    from mdfm.data import canonical
    from mdfm.data import zupt_anchor as ZA
    from mdfm.gates import g3
    from mdfm.readouts import zupt as Z
    locked = set(g3.lockbox_persons())
    if set(train) & locked:
        raise HarnessDefect("a lockbox person in the training fold")
    df = canonical.index()
    rec = ws.recordings
    ratios, pp, n_tab, n_pair, n_raw = [], {}, 0, 0, 0
    for r, s in sorted(table.strides.items()):
        g = rec.iloc[int(r)]
        pid, key = str(g["person_id"]), str(g["key"])
        if pid not in set(train):
            raise HarnessDefect(f"{pid} is not a training-fold person")
        sel = df[(df["source"] == ZA.SOURCE) & (df["modality"] == "imu") & (df["person_id"] == pid)
                 & (df["key"] == key)]
        if len(sel) != 1 or bool(sel["lockbox"].any()):
            raise HarnessDefect(f"{pid} {key}: {len(sel)} canonical imu rows")
        raw = canonical.load(sel.iloc[0].to_dict())
        fps = float(raw["fps"])
        for foot, (a, gy) in _foot_channels(raw).items():
            zs = [z for z in Z.strides(a, gy, fps) if z["length_m"] >= table.min_length_m]
            n_raw += len(zs)
            fid = ZA.FOOT_IDS[foot]
            for i in np.flatnonzero(s.foot == fid):
                n_tab += 1
                m = [z for z in zs if abs(z["m_a"] - int(s.m_a[i])) <= PAIR_SAMPLES
                     and abs(z["m_b"] - int(s.m_b[i])) <= PAIR_SAMPLES]
                if m:
                    n_pair += 1
                    q = float(s.length_m[i]) / float(m[0]["length_m"])
                    ratios.append(q)
                    pp.setdefault(pid, []).append(q)
    q = np.asarray(ratios, dtype=np.float64)
    med = {p: float(np.median(v)) for p, v in pp.items()}
    log(f"raw agreement: {n_pair} of {n_tab} table strides paired, median band over raw {np.median(q):.4f}")
    return {"table_strides": n_tab, "raw_strides_over_min": n_raw, "paired": n_pair,
            "paired_share": n_pair / n_tab if n_tab else None,
            "band_over_raw_median": float(np.median(q)) if len(q) else None,
            "band_over_raw_p05_p95": [float(np.quantile(q, 0.05)), float(np.quantile(q, 0.95))] if len(q) else None,
            "person_median_min_max": [min(med.values()), max(med.values())] if med else None,
            "persons": len(med), "pair_samples": PAIR_SAMPLES}


def table_mode(tree=None, raw=True, log=print):
    """--table (module docstring)."""
    from mdfm.data import walkway_anchor as WA
    from mdfm.data import window_store as WS
    from mdfm.data import zupt_anchor as ZA
    from mdfm.train import config as CF
    t0 = time.time()
    tree = tree or TREE
    cfg = CF.load(os.path.join(tree, CONFIG_REL))
    if CF.sha256(cfg) != CONFIG_SHA256:
        raise HarnessDefect(f"{CONFIG_REL} is not the config of this script ({CF.sha256(cfg)[:12]})")
    ws = WS.WindowStore()
    table = ZA.build_table(ws, cfg.zupt_anchor)
    train = ws.person_lists("train")[ZA.SOURCE]
    check = ZA.assert_table_clean(table, train, WA.refused_persons(ws), cfg.zupt_anchor.persons_sha256)
    t_build = round(time.time() - t0, 1)
    L = np.concatenate([s.length_m for s in table.strides.values()])
    T = np.concatenate([s.t_b - s.t_a for s in table.strides.values()])
    turn = np.abs(np.concatenate([s.turn_deg for s in table.strides.values()]))
    out = {"what": "the inertial anchor's table of configs/pilot_f2l5.yaml on the window store (DRAFT rule section 1)",
           "label": LABEL, "task": TASK, "config_sha256": CONFIG_SHA256, "window_store": ws.root, "table": check,
           "lengths_m": {"median": float(np.median(L)), "p05_p95": [float(np.quantile(L, 0.05)),
                                                                    float(np.quantile(L, 0.95))]},
           "durations_s": {"median": float(np.median(T)), "total": float(T.sum())},
           "turn_deg_abs": {"median": float(np.median(turn)), "p95": float(np.quantile(turn, 0.95)),
                            "share_over_20": float((turn > 20).mean()), "share_over_45": float((turn > 45).mean())},
           "build_seconds": t_build}
    if raw:
        out["raw_agreement"] = raw_agreement(ws, table, train, log)
    out["seconds"] = round(time.time() - t0, 1)
    log(f"table: {check['counts']['recordings']} recordings, {check['counts']['strides']} strides, "
        f"{check['persons']} persons, SHA-256 {check['table_sha256'][:12]}, {t_build} s")
    return out


# ---------------------------------------------------------------- --replay
def anchor_replay(tree=None, n_micro=None, log=print):
    """v1's phase B plans through the table of configs/pilot_f2l5.yaml (module docstring, --replay)."""
    from types import SimpleNamespace
    from mdfm.data import body_store as BS, sampler as S, walkway_anchor as WA, window_store as WS
    from mdfm.data import zupt_anchor as ZA
    from mdfm.train import config as CF, loop as LP
    t0 = time.time()
    tree = tree or TREE
    cfg = CF.load(os.path.join(tree, CONFIG_REL))
    assert CF.sha256(cfg) == CONFIG_SHA256, f"{CONFIG_REL} is not the config of this script"
    ws = WS.WindowStore()
    table = ZA.build_table(ws, cfg.zupt_anchor)
    train = ws.person_lists("train")[ZA.SOURCE]
    check = ZA.assert_table_clean(table, train, WA.refused_persons(ws), cfg.zupt_anchor.persons_sha256)
    smp = S.MixtureSampler(BS.BodyStore(split="train"), ws, None, "B", LP.phase_seed(cfg, "B"), cfg.mixture,
                           guard=False)
    total = int(CF.step_counts(cfg)[1]) * int(cfg.schedule.micro_per_step)
    n = total if n_micro is None else min(int(n_micro), total)
    keys = {int(i): k for i, k in ws.recordings["key"].items()}
    who = {int(i): p for i, p in ws.recordings["person_id"].items()}
    edge = float(cfg.zupt_anchor.edge_s)
    per = {p: {"windows": 0, "anchored": 0, "strides": 0, "seconds": 0.0} for p in train}
    tot = {"weargait_windows": 0, "anchored_windows": 0, "strides": 0, "anchored_seconds": 0.0}
    per_micro_w, per_micro_s = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for k in range(n):
        plan = smp.plan()
        recs, wins = [], []
        for it in plan.items:
            if it.source == ZA.SOURCE:
                recs.append(int(it.recording))
                wins.append(SimpleNamespace(synthetic=False, source=ZA.SOURCE, key=keys[int(it.recording)],
                                            t0=float(it.t0), duration=float(it.dur)))
                per[who[int(it.recording)]]["windows"] += 1
        tot["weargait_windows"] += len(wins)
        tg = ZA.targets_for_windows(table, recs, wins, edge)
        if not tg.n:
            continue
        w = tg.window.numpy()
        sec = (tg.t_b - tg.t_a).double().numpy()
        aw = sorted(set(w.tolist()))
        per_micro_w[k], per_micro_s[k] = len(aw), tg.n
        tot["anchored_windows"] += len(aw)
        tot["strides"] += int(tg.n)
        tot["anchored_seconds"] += float(sec.sum())
        for i in aw:
            p = who[recs[i]]
            per[p]["anchored"] += 1
            per[p]["strides"] += int((w == i).sum())
            per[p]["seconds"] += float(sec[w == i].sum())
    a = np.array([per[p]["anchored"] for p in train], dtype=np.float64)
    out = {"what": "anchor replay of v1's phase B plans under configs/pilot_f2l5.yaml (plan only)", "task": TASK,
           "config_sha256": CONFIG_SHA256, "micro_steps": n, "micro_steps_of_phase_b": total, "table": check,
           "totals": tot,
           "per_micro_step": {"anchored_windows_mean": float(per_micro_w.mean()),
                              "anchored_windows_max": int(per_micro_w.max()),
                              "strides_mean": float(per_micro_s.mean()),
                              "strides_p95": float(np.quantile(per_micro_s, 0.95)),
                              "strides_max": int(per_micro_s.max()),
                              "share_with_anchor": float((per_micro_w > 0).mean())},
           "per_person": {"persons": len(train), "persons_anchored": int((a > 0).sum()), "anchored_min": int(a.min()),
                          "anchored_median": float(np.median(a)), "anchored_max": int(a.max()), "rows": per},
           "seconds": round(time.time() - t0, 1)}
    log(f"replay: {n} micro-steps, {tot['anchored_windows']} anchored windows, {tot['strides']} strides, persons "
        f"anchored {out['per_person']['persons_anchored']} of {len(train)}")
    return out


# ---------------------------------------------------------------- --preflight
def preflight(tree=None, run_dir=None, n_anchored=PREFLIGHT_ANCHORED, max_micro=PREFLIGHT_MAX_MICRO,
              others=PREFLIGHT_OTHERS, source=None, log=print):
    """The CPU pre-flight of the term's size at the start (module docstring, --preflight)."""
    import torch
    from mdfm.data import body_store as BS, collate as CO, window_store as WS
    from mdfm.data import zupt_anchor as ZA
    from mdfm.factory import synth as SY
    from mdfm.train import config as CF, losses as LS, loop as LP
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:                                  # pragma: no cover
        proc = None
    l4 = L4()
    t0 = time.time()
    tree = tree or TREE
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2") or 2)))
    cfg = CF.load(os.path.join(tree, CONFIG_REL))
    assert CF.sha256(cfg) == CONFIG_SHA256
    cpu = torch.device("cpu")
    bs, ws = BS.BodyStore(split="train"), WS.WindowStore()
    synth = SY.SynthSampler(bs, device=cpu)
    rd = run_dir or os.path.join(tree, PREFLIGHT_RUN_DIR)
    tr = LP.Trainer(cfg, bs, ws, synth, cpu, rd, prefetch=0, dev_hook=None, echo=None)
    t_table = round(time.time() - t0, 1)
    src = source or os.path.join(tree, SOURCE_PATH)
    with l4.drift_log_in(rd):                  # the load's drift line goes to the scratch folder, not v1's run folder
        tr.init_from(src)
    model, table, lc = tr.model, tr.zupt_table, tr.loss_cfg
    model.train()
    wB = LS.term_weights("B", 1.0, lc)
    enc = [i for i, (n, _) in enumerate(model.named_parameters()) if n.startswith("encoder.")]
    dec = [i for i, (n, _) in enumerate(model.named_parameters()) if n.startswith("decoder.")]
    params = list(model.parameters())
    groups = {"encoder": enc, "decoder": dec}
    tr._start_sampler("B")
    rng = np.random.default_rng(20261015)
    edge = float(cfg.zupt_anchor.edge_s)
    micro, done = [], 0
    for m_i in range(int(max_micro)):
        if done >= int(n_anchored):
            break
        it = tr.next_item()
        tg_full = ZA.targets_for_step(table, it.step, edge)
        rec = {"micro_step": m_i, "windows": len(it.step.windows), "strides": int(tg_full.n)}
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
        tg = ZA.targets_for_windows(table, recs, wins, edge)
        mb = CO.collate(wins, cfg=tr.collate_cfg, gen=it.step.seed)
        batch, targets = mb.batch.to(cpu), mb.targets.to(cpu)
        N, n_a, m = len(it.step.windows), len(anch), len(pick)
        c = torch.tensor([1.0 / N] * n_a + [(N - n_a) / (max(m, 1) * N)] * m, dtype=torch.float32)
        gen = torch.Generator(device=cpu).manual_seed(int(it.step.seed) % 2 ** 63)
        post = LP.encode(model, batch)
        body, pred = LP.forward_draw(model, batch, post, gen, truth_windows=LS.truth_mask(batch, targets, cpu))
        out = LS.stage1_losses(model, batch, post, body, pred, "B", targets=targets, cfg=lc)
        ta = time.perf_counter()
        zp = LS.zupt_anchor_parts(model, body.z, body.beta, batch, tg, float(cfg.zupt_anchor.sd))
        t_anchor = time.perf_counter() - ta
        wz = float(cfg.zupt_anchor.weight)
        objs = {"anchor": (c * wz * zp["term"]).sum(), "rest": (c * out["per_window"]).sum(),
                "sup_root_rate": (c * wB["sup_root_rate"] * out["sup_root_rate"]).sum(),
                "imu_nll": (c * wB["imu_nll"] * out["imu_nll"]).sum()}
        objs["full"] = objs["rest"] + objs["anchor"]
        # an objective without a graph (sup_root_rate on a sub-batch without a body truth) has no gradient: norm 0
        grads = {k: (torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) if v.requires_grad
                     else [None] * len(params)) for k, v in objs.items()}
        norms = {k: l4._norms(g, groups) for k, g in grads.items()}
        ratio = {k: {grp: (norms["anchor"][grp] / norms[k][grp] if norms[k][grp] > 0 else None) for grp in groups}
                 for k in objs if k != "anchor"}
        aw = torch.tensor(list(range(n_a)))
        S, D = zp["S"].detach(), zp["D"].detach()
        rec.update({"sub_batch": {"anchored_windows": n_a, "other_windows_sampled": m, "other_windows": N - n_a,
                                  "tokens": int(mb.useful), "strides": int(tg.n)},
                    "anchor": {"term_sum": float(zp["term"].detach().sum()),
                               "distance_sum": float(zp["distance"].detach().sum()),
                               "still_sum": float(zp["still"].detach().sum()),
                               "per_anchored_window": float(zp["term"].detach().sum()) / n_a,
                               "per_stride": float(zp["term"].detach().sum()) / int(tg.n),
                               "per_window_log_mean_estimate": float(zp["term"].detach().sum()) / N,
                               "gain_decoded_over_zupt": float(D.sum() / S.sum()),
                               "gain_median": float((D / S).median()),
                               "speed_at_instants_mean": float(torch.cat([zp["v_a"], zp["v_b"]]).mean()),
                               "seconds": t_anchor},
                    "rest_per_anchored_window": float(out["per_window"].detach()[aw].mean()),
                    "rest_per_window_sub_batch": float(out["per_window"].detach().mean()),
                    "values": {k: float(v.detach()) for k, v in objs.items()}, "grad_norms": norms,
                    "truth_windows_in_sub_batch": int(LS.truth_mask(batch, targets, cpu).sum()),
                    "anchor_over": ratio, "rss_gb": proc.memory_info().rss / 1e9 if proc else None})
        micro.append(rec)
        done += 1
        del grads, objs, out, body, pred, post, mb, batch
        log(f"pre-flight micro-step {m_i}: {n_a} anchored windows of {N}, {tg.n} strides; term per anchored window "
            f"{rec['anchor']['per_anchored_window']:.1f}, rest {rec['rest_per_anchored_window']:.1f}; gain "
            f"{rec['anchor']['gain_decoded_over_zupt']:.4f}; anchor over rest encoder {ratio['rest']['encoder']}")
    tr._close_prefetch()
    return {"what": "CPU pre-flight of the inertial anchor at the start (estimate on sub-batches; module docstring)",
            "task": TASK, "config_sha256": CONFIG_SHA256, "source": SOURCE_PATH, "table": tr.zupt_check,
            "device": "cpu", "threads": torch.get_num_threads(), "refinement": "none (amortized posterior)",
            "weights_B": {k: wB[k] for k in ("sup_root_rate", "imu_nll")}, "zupt_anchor_weight": wz,
            "micro_steps": micro, "trainer_and_table_seconds": t_table,
            "peak_rss_gb": max([r.get("rss_gb") or 0.0 for r in micro] or [0.0]), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- --pin
def input_pin(tree=None, out_dir=None):
    """The input pin of the run's job (mdfm.data.zupt_anchor.PIN_ENV): the inputs and the table SHA-256 of table.json,
    after asserting that the store files hash to them now."""
    from mdfm.data import window_store as WS
    from mdfm.data import zupt_anchor as ZA
    tree = tree or TREE
    tab = read_json(os.path.join(out_dir or os.path.join(tree, BENCH), TABLE_JSON))["table"]
    now = ZA.store_inputs(WS.WindowStore())
    if now != tab["inputs"]:
        raise HarnessDefect(f"the window store's files differ from table.json: "
                            f"{sorted(k for k in now if now[k] != tab['inputs'].get(k))}")
    return {"what": "input pin of the inertial-anchor run (mdfm/data/zupt_anchor.py, MDFM_ZUPT_ANCHOR_PIN); DRAFT, "
                    "for the job file of a rule not yet hashed", "task": TASK, "inputs": dict(tab["inputs"]),
            "table_sha256": tab["table_sha256"]}


# ---------------------------------------------------------------- --cost
def _hours(a, b):
    f = "%Y-%m-%dT%H:%M:%S"
    return (datetime.datetime.strptime(b, f) - datetime.datetime.strptime(a, f)).total_seconds() / 3600.0


def cost(tree=None):
    """GPU hours of lever 4's jobs (job_log.csv) and the estimate of this lever's from them (module docstring)."""
    tree = tree or TREE
    rows = list(csv.DictReader(open(os.path.join(tree, JOB_LOG_REL), encoding="utf-8", newline="")))
    l4 = [r for r in rows if "f2l4" in r["name"] and r["state"] == "done" and r["exit_code"] == "0"]
    train = [r for r in l4 if r["name"].startswith("f2l4_train")]
    post = [r for r in l4 if not r["name"].startswith("f2l4_train") and r["cuda_visible_devices"] != "-1"]
    th = sum(_hours(r["start"], r["end"]) for r in train)
    ph = {r["name"]: round(_hours(r["start"], r["end"]), 4) for r in post}
    smoke_h = SMOKE_STEPS * th / PHASE_B_STEPS + STARTUP_HOURS
    out = {"what": "GPU hours (lever 4 measured, job_log.csv start to end; lever 5 estimated from them)", "task": TASK,
           "lever4": {"train_jobs": [r["name"] for r in train], "train_hours": round(th, 3),
                      "train_seconds_per_step": round(th * 3600.0 / PHASE_B_STEPS, 3),
                      "post_train_gpu_jobs": ph, "post_train_gpu_hours": round(sum(ph.values()), 3)},
           "lever5_estimate": {"smoke_steps": SMOKE_STEPS, "startup_hours": STARTUP_HOURS,
                               "smoke_hours": round(smoke_h, 2), "train_hours": round(th, 2),
                               "post_train_hours": round(sum(ph.values()), 2),
                               "total_hours": round(smoke_h + th + sum(ph.values()), 2),
                               "note": "the train hours are lever 4's measured run of the same steps; the extra "
                                       "decoder call of the anchor is not measured on the GPU (the smoke job measures "
                                       "it)"}}
    return out


# ---------------------------------------------------------------- --smoke-check
def smoke_check(run_dir, tree=None, n_v1=200):
    """The smoke run's log (module docstring, --smoke-check)."""
    tree = tree or TREE
    f2 = L4().F2()
    recs, _ = f2.run_log(tree, run_dir)
    v1, _ = f2.run_log(tree, V1_RUN_DIR)
    if recs is None:
        return {"present": False, "reason": f"no {run_dir}/log.jsonl"}
    starts = [r for r in recs if r.get("event") == "start"]
    logs = [r for r in recs if r.get("event") == "log" and r.get("phase") == "B"]
    v1b = [r for r in (v1 or []) if r.get("event") == "log" and r.get("phase") == "B" and r.get("step", 0) <= n_v1]

    def s_per_step(rs):
        w = sum(float(r.get("wall_s") or 0.0) for r in rs[1:])
        n = sum(int(r.get("optimizer_steps") or 0) for r in rs[1:])
        return w / n if n else None

    za = [r.get("zupt_anchor") or {} for r in logs]
    return {"present": True, "task": TASK, "start": [s.get("zupt_anchor") for s in starts],
            "config_sha256": [s.get("config_sha256") for s in starts], "log_entries": za,
            "seconds_per_step": s_per_step(logs), "v1_seconds_per_step_first_steps": s_per_step(v1b),
            "peak_reserved_gb": L4().WB().peak_reserved_gb(recs),
            "alarms": len([r for r in recs if r.get("event") == "alarm"])}


# ---------------------------------------------------------------- outputs and main
def write_outputs(out_dir, res, tree, started, name):
    from mdfm.gates import common
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(path, json.dumps(L4().F2().clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    prov = {"script": "scripts/f2_lever5.py", "task": TASK, "label": LABEL, "row_label": ROW_LABEL, "run_at": started,
            "output": name, "moveatlas_root": paths.MOVEATLAS, "python": sys.version.split()[0],
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "f2_lever4.py"),
                                               os.path.join(TREE, "mdfm", "data", "zupt_anchor.py"),
                                               os.path.join(TREE, "mdfm", "readouts", "zupt.py"),
                                               os.path.join(TREE, "mdfm", "train", "losses.py"),
                                               os.path.join(TREE, "mdfm", "train", "loop.py")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "cache": paths.CACHE.replace("\\", "/")}}
    try:
        prov["moveatlas_stamp"] = L4().GG().stamp_moveatlas()
    except Exception as e:  # noqa: BLE001  (the stamp is reported, never required)
        prov["moveatlas_stamp"] = {"error": str(e)}
    common.write_text(os.path.join(out_dir, f"_provenance_{os.path.splitext(name)[0]}.json"),
                      json.dumps(L4().F2().clean(prov), indent=1, ensure_ascii=False) + "\n")
    return path


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None)
    p.add_argument("--out", default=None, help="output folder (default reports/bench/f2_lever5/ of the tree)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--table", action="store_true")
    g.add_argument("--replay", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--pin", action="store_true")
    g.add_argument("--cost", action="store_true")
    g.add_argument("--smoke-check", default=None, metavar="RUN_DIR")
    g.add_argument("--rule-numbers", action="store_true")
    p.add_argument("--no-raw", action="store_true", help="--table: skip the raw-record agreement")
    p.add_argument("--micro-steps", type=int, default=None, help="--replay: the first N micro-steps only")
    p.add_argument("--run-dir", default=None, help="--preflight: the scratch run folder")
    p.add_argument("--source", default=None, help="--preflight: the phase A end (default rule section 1's)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("scripts/f2_lever5.py runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    tree = os.path.abspath(a.tree) if a.tree else TREE
    out_dir = a.out or os.path.join(tree, BENCH)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if a.rule_numbers:
        nums, checks = rule_numbers(tree)
        print(json.dumps({"numbers": nums, "checks": checks}, indent=1))
        return 0
    modes = (("table", TABLE_JSON, lambda: table_mode(tree, raw=not a.no_raw)),
             ("replay", REPLAY_JSON, lambda: anchor_replay(tree, a.micro_steps)),
             ("preflight", PREFLIGHT_JSON, lambda: preflight(tree, a.run_dir, source=a.source)),
             ("pin", PIN_NAME, lambda: input_pin(tree, out_dir)), ("cost", COST_JSON, lambda: cost(tree)),
             ("smoke_check", SMOKE_JSON, lambda: smoke_check(a.smoke_check, tree)))
    for flag, name, fn in modes:
        if getattr(a, flag):
            try:
                res = fn()
            except HarnessDefect as e:
                print(f"HARNESS DEFECT: {e}")
                return EXIT_HARNESS
            res["run_started"] = started
            try:
                print(f"written {rel(write_outputs(out_dir, res, tree, started, name), tree)}")
            except FileExistsError as e:
                print(f"{e} exists: nothing rewritten; pass --out with a new folder")
                return EXIT_EXISTS
            return 0
    try:
        res, code = run(tree)
    except RuleDraft as e:
        print(f"NOT EVALUATED: {e}")
        return EXIT_NOT_EVALUATED
    except HarnessDefect as e:
        print(f"HARNESS DEFECT: {e}")
        return EXIT_HARNESS
    try:
        write_outputs(out_dir, res, tree, started, "results.json")
    except FileExistsError as e:
        print(f"{e} exists: nothing rewritten; pass --out with a new folder")
        return EXIT_EXISTS
    print(json.dumps({"verdict": res["verdict"], "route": res["route"]}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
