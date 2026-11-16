"""F2-L1 (2026-10-05): the readings of the F2 lever 1 rule, the training-fold calibration head of the G3 stride and
speed readouts. Label of every output: dev diagnostic, not a G3 read.

Rule: reports/bench/f2_lever1/rule.txt (version 1, approved 2026-10-05: docs/proposals/F2_LEVER1_RULE_DRAFT.md with
the recommendations of task F2-CHAL). Its SHA-256 sits in rule.sha256 beside it and here in RULE_SHA256; check_rule
asserts that the three agree before any target value is read (run calls it first, analyse_end again before it opens
an events file), so a rule edited after the fact stops the run.

What it computes (CPU only, existing files only; CUDA_VISIBLE_DEVICES=-1 is set before torch can load):
- per end of scripts/g3_gain_check.py ENDS (v1_B, s2_B, v1_A): the matched dev units of row base of its diag run
  (reports/gates/g3/diag_<name>/events.csv) and the raw lines, asserted equal to diag.json (g3_gain_check.consistency);
  when its training-fold decode exists (reports/bench/g3_gain_check/decode_<name>/train_events.csv, read by
  g3_gain_check.read_decode, checkpoint SHA-256 asserted): the head (g3_gain_check.fit_gain on train54, never on a dev
  or lockbox person), the calibrated dev lines and their paired differences (g3_gain_check.recalibration), the
  in-sample and leave-one-person-out train54 lines (cross_fit), and the IC and G1 equality asserts (ic_g1_asserts);
- R0 (r0_check): at v1_B and s2_B the gains, raw, calibrated and paired lines equal F1's results.json, and the
  cross-fit lines equal F2-CHAL's results.json (test X), within R0_TOL;
- R1 (r1_reading): per end with a decode, TRANSFERS when S54_LOPO - S54 <= max(X_REL x S54, 0.10 cm) and
  V54_LOPO - V54 <= max(X_REL x V54, 0.002 m/s), DOES NOT TRANSFER otherwise; seen at the B ends, blind at v1_A;
- R2 (r2_reading): the v1_A confirmation, its dev part (dev_part: margins m_L = max(M_REL x raw, 2 x d_L) with d_L
  from the step-4,000 curve folders of the two seeds, read as scripts/weekend_branch.py reads them, fallback the B-end
  differences of reports/gpu_queue/branch_seeds_20261002_211426.json; non-inferiority of S(14) and V(14)) and its R1;
  when a registered success margin is at least the raw line itself (badly_posed: no calibration can meet it), the
  output says the criterion is badly posed (rule section 11) and states two facts beside it: whether the badly posed
  margins were computable before the hash (pre_hash_check: the same margins from the raw v1_A lines of F1's
  results.json, quoted in rule section 0, and the step-4,000 curve folders, with the modification times of those
  files against rule.sha256), and that the basis of the second reading kept beside it (the B-end seed differences)
  was chosen after the v1_A dev lines were seen and is not backed by the rule unless a step-4,000 value is missing;
  the registered reading and the route it gives are unchanged, the second reading proposes nothing by itself, and
  the owner decides;
- the route of rule section 9 (route) and the side tables of rule section 8, which read nothing (beside).
Writes reports/bench/f2_lever1/results.json, summary.md and _provenance.json (MoveAtlas stamp of
scripts/moveatlas_stamp.py). Exit code EXIT_OK (0) when R2 was evaluated, EXIT_NO_V1A (3) when the v1_A training-fold
decode does not exist (R0 and R1 written, R2 NOT EVALUATED), EXIT_HARNESS (4) when R0 or an equality assert failed
(written, the reading stopped).

Data discipline: sets dev14, dev12, train54 recomputed from mdfm.gates.g3.controls and asserted equal (SHA-256) to
F1's results.json; none in mdfm.gates.g3.lockbox_persons() (Refused); the person column of every events file is read
and guarded before any other column (g3_gain_check.read_events). No lockbox, EVAL, CODY or care_pd record; nothing
written outside reports/bench/f2_lever1/.

--print-job prints the scripts/queue_job.py command of the v1_A training-fold decode (job_command: the command of
scripts/g3_gain_check.py --print-jobs for the v1_A End, batch_windows read from the v1_A diag run's inference block).

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_lever1.py [--print-job]
"""
import argparse
import glob
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
OUT = os.path.join(TREE, "reports", "bench", "f2_lever1")
RULE_PATH = os.path.join(OUT, "rule.txt")
RULE_SHA_PATH = os.path.join(OUT, "rule.sha256")
RULE_SHA256 = "30af09f0068aae421293e5c7a915e4eaf5078b52379ea0e4b03b642c5d39dccf"
F1_DIR = os.path.join(TREE, "reports", "bench", "g3_gain_check")
F1_RESULTS = os.path.join(F1_DIR, "results.json")
CHAL_RESULTS = os.path.join(TREE, "reports", "bench", "f2_scale", "CHAL", "results.json")
SEEDS_JSON = os.path.join(TREE, "reports", "gpu_queue", "branch_seeds_20261002_211426.json")
SUPERVISED = os.path.join(TREE, "reports", "bench", "g3_supervised_baseline", "results.json")
FROZEN = os.path.join(TREE, "reports", "bench", "frozen_tokens_summary.json")
H1_RESULTS = os.path.join(TREE, "reports", "bench", "f2_scale", "H1shape", "results.json")
DUMP_GLOB = os.path.join(TREE, "reports", "bench", "f2_scale", "*", "dump_{name}")
A4000 = ("phaseA_root_full_ckpt_A_004000", "pilot_v1_s2_ckpt_A_004000")
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2L1"
R0_TOL = 1e-6
X_REL = 0.05
X_ABS = {"stride": 0.10, "speed": 0.002}
M_REL = 0.10
NI_REL = 0.05
SEED_FACTOR = 2.0
B_ENDS = ("v1_B", "s2_B")
CONFIRM_END = "v1_A"
PRIMARY = ("S12", "V12")
NONINF = ("S14", "V14")
EXIT_OK, EXIT_NO_V1A, EXIT_HARNESS = 0, 3, 4


class HarnessDefect(AssertionError):
    """A departure that the rule reads as a harness defect (R0, or the IC and G1 equality asserts)."""


def _sibling(name):
    key = f"_f2l1_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def G():
    """scripts/g3_gain_check.py (F1): estimator, readers, lines, guards."""
    return _sibling("g3_gain_check")


def WB():
    """scripts/weekend_branch.py: the G3 and G1 folder readers of its seeds stage."""
    return _sibling("weekend_branch")


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:                                # another drive (tests on a temporary folder)
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- rule and sets
def check_rule(rule_path=None, sha_path=None, expected=None):
    """The SHA-256 of rule.txt, asserted equal to rule.sha256 and RULE_SHA256 (module values at call time)."""
    return G().check_rule(RULE_PATH if rule_path is None else rule_path,
                          RULE_SHA_PATH if sha_path is None else sha_path,
                          RULE_SHA256 if expected is None else expected)


def load_sets(f1_doc, locked, dev, train):
    """{dev14, dev12, train54} from the fold lists, each SHA-256 asserted equal to F1's results.json, checked by
    g3_gain_check.check_sets (sizes, disjointness, dev12 = dev14 without the two excluded, lockbox Refused)."""
    from mdfm.data import sources
    g = G()
    sets = {"dev14": sorted(dev), "dev12": sorted(p for p in dev if p not in g.EXCLUDED_FROM_12),
            "train54": sorted(train)}
    s = f1_doc["sets"]
    for k in sets:
        if sources.person_hash(sets[k]) != s[f"{k}_sha256"]:
            raise AssertionError(f"{k}: SHA-256 differs from F1's results.json")
    g.check_sets(sets, locked)
    return sets


# ---------------------------------------------------------------- R1: the cross-fit
def cross_fit(strides, speeds, train, dev, locked):
    """{stride, speed}: the head fitted on all train54 units (in sample) and, per person, refitted on the other
    persons (leave one person out), each with g3_gain_check.fit_gain (no dev, no lockbox person); the person mean
    absolute error after dividing by each; their means over persons, the difference and the range of the refitted
    gains."""
    g = G()
    out = {}
    for kind, units in (("stride", strides), ("speed", speeds)):
        u = g.subset(units, train)
        full = g.fit_gain(u, train, dev, locked)["gain"]
        pin = g.person_abs_error(u, full)
        p = u["person_id"].astype(str).to_numpy()
        plo, gains = {}, {}
        for pid in sorted(set(p)):
            gi = g.fit_gain(u[p != pid], train, dev, locked)["gain"]
            gains[pid] = gi
            plo.update(g.person_abs_error(u[p == pid], gi))
        a, b = float(np.mean(list(pin.values()))), float(np.mean(list(plo.values())))
        out[kind] = {"gain_in_sample": full, "in_sample": a, "lopo": b, "lopo_minus_in": b - a,
                     "lopo_gain_range": [float(min(gains.values())), float(max(gains.values()))],
                     "n_persons": len(pin), "unit": g.UNIT_OF[kind], "per_person_in_sample": pin,
                     "per_person_lopo": plo, "lopo_gains": gains}
    return out


def r1_reading(cf, rel_tol=None, abs_tol=None):
    """TRANSFERS when, on both lines, LOPO minus in sample is at most max(rel_tol x in sample, abs_tol[kind]);
    DOES NOT TRANSFER otherwise."""
    rel_tol = X_REL if rel_tol is None else rel_tol
    abs_tol = X_ABS if abs_tol is None else abs_tol
    rows = {}
    for kind, line in (("stride", "S54"), ("speed", "V54")):
        c = cf[kind]
        tol = max(rel_tol * c["in_sample"], abs_tol[kind])
        rows[line] = {"in_sample": c["in_sample"], "lopo": c["lopo"], "lopo_minus_in": c["lopo_minus_in"],
                      "tolerance": tol, "ok": bool(c["lopo_minus_in"] <= tol), "unit": c["unit"]}
    return {"reading": "TRANSFERS" if all(r["ok"] for r in rows.values()) else "DOES NOT TRANSFER", "lines": rows,
            "rule": f"TRANSFERS when S54_LOPO - S54 <= max({rel_tol:g} x S54, {abs_tol['stride']:g} cm) and "
                    f"V54_LOPO - V54 <= max({rel_tol:g} x V54, {abs_tol['speed']:g} m/s)"}


# ---------------------------------------------------------------- R2: margins and the dev part
def seed_diffs_b(path=None):
    """{S12, S14, V12, V14: |seed 1 - seed 2|} at the B ends (branch_seeds json, section B_end)."""
    doc = read_json(SEEDS_JSON if path is None else path)
    return {k: fin(doc["seed_diff"]["B_end"][k]["abs_diff"]) for k in ("S12", "S14", "V12", "V14")}


def _source_files(src, tree):
    """The files a weekend_branch G3 source was read from: the G3 json and the per_person.csv beside it (a G3
    folder), or the diag.json (a diag row)."""
    if not src:
        return []
    if src.get("json"):
        pp = os.path.join(os.path.dirname(src["json"]), "per_person.csv")
        return [src["json"]] + ([pp] if os.path.exists(pp) else [])
    p = os.path.join(tree, str(src.get("source") or ""))
    return [p] if src.get("source") and os.path.isfile(p) else []


def seed_diffs_a(tree=None, names=None, fallback=None):
    """{line: {value, source, seed1, seed2, fallback, files}} at step 4,000 of phase A: the G3 lines of the two
    seeds' curve folders read with scripts/weekend_branch.py (g3_sources, pick_pair, g3_line on no_whc003_whc026 and
    all14), with the files read (files); a line one side lacks takes the B-end difference of `fallback`, and says so
    (fallback True)."""
    wb = WB()
    tree = tree or TREE
    names = A4000 if names is None else names
    a, b = wb.pick_pair(wb.g3_sources(tree, names[0]), wb.g3_sources(tree, names[1]))
    files = _source_files(a, tree) + _source_files(b, tree)
    out = {}
    for key, tag, sub in (("S", "12", wb.SUB12), ("S", "14", wb.SUB14), ("V", "12", wb.SUB12),
                          ("V", "14", wb.SUB14)):
        _, pa = wb.g3_line(a, key, sub)
        _, pb = wb.g3_line(b, key, sub)
        line = f"{key}{tag}"
        if pa is not None and pb is not None:
            out[line] = {"value": abs(pa - pb), "seed1": pa, "seed2": pb,
                         "source": f"{a['source']} and {b['source']}", "fallback": False, "files": files}
        else:
            fb = (fallback or {}).get(line)
            out[line] = {"value": fb, "seed1": pa, "seed2": pb,
                         "source": "B-end difference (reports/gpu_queue/branch_seeds_20261002_211426.json): a step "
                                   "4,000 value is missing", "fallback": True, "files": []}
    return out


def margins(raw, d):
    """Success margins m_L = max(M_REL x raw L, SEED_FACTOR x d_L) of the primary lines and non-inferiority margins
    max(NI_REL x raw L, SEED_FACTOR x d_L) of S(14) and V(14); raw and d map line -> number."""
    m = {L: max(M_REL * raw[L], SEED_FACTOR * d[L]) for L in PRIMARY}
    ni = {L: max(NI_REL * raw[L], SEED_FACTOR * d[L]) for L in NONINF}
    return m, ni


def dev_part(rc, m, ni):
    """GO / PARTIAL / NO-GO of rule section 8 (R2, dev part) on a recalibration block, with the per-line rows."""
    rows = {}
    for L in PRIMARY:
        p = rc["paired"][L]
        lo = fin((p["interval"] or [None])[0])
        meets = bool(fin(p["point"]) is not None and p["point"] >= m[L])
        above = bool(lo is not None and lo > 0)
        rows[L] = {"raw": rc["raw"][L]["point"], "calibrated": rc["recalibrated"][L]["point"],
                   "raw_minus_calibrated": p["point"], "paired_interval": p["interval"], "margin": m[L],
                   "meets_margin": meets, "lower_above_0": above, "both": meets and above}
    nonin = {}
    for L in NONINF:
        worse = rc["recalibrated"][L]["point"] - rc["raw"][L]["point"]
        nonin[L] = {"raw": rc["raw"][L]["point"], "calibrated": rc["recalibrated"][L]["point"],
                    "calibrated_minus_raw": worse, "margin": ni[L], "ok": bool(worse <= ni[L])}
    n = sum(r["both"] for r in rows.values())
    ni_ok = all(r["ok"] for r in nonin.values())
    reading = "GO" if n == 2 and ni_ok else "PARTIAL" if n == 1 and ni_ok else "NO-GO"
    return {"reading": reading, "primary": rows, "noninferiority": nonin,
            "met_lines": [L for L in PRIMARY if rows[L]["both"]]}


def badly_posed(raw, m):
    """Rule section 11 on the success margins: the primary lines whose margin m_L is at least the raw line itself.
    A calibrated line is an MAE (at least 0), so raw minus calibrated is at most raw L: at m_L > raw L no calibration
    can meet the margin. Returns {lines, calibrated_at_most} (calibrated_at_most: raw L - m_L per primary line, the
    largest calibrated line that would meet)."""
    return {"lines": [L for L in PRIMARY if m[L] >= raw[L]],
            "calibrated_at_most": {L: raw[L] - m[L] for L in PRIMARY}}


def file_times(paths_):
    """{path relative to the tree: modification time, local ISO seconds, or None when the file is absent}."""
    out = {}
    for p in paths_:
        try:
            out[rel(p)] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p)))
        except OSError:
            out[rel(p)] = None
    return out


def pre_hash_check(f1_raw, dA, f1_results=None, rule_sha_path=None):
    """Whether the registered v1_A margins were computable, and attainable, before the rule was hashed: the margins
    from the raw v1_A lines of F1's results.json (quoted in rule section 0) and the step-4,000 seed differences dA
    (seed_diffs_a), the badly posed lines among them, and the modification times of those input files against that
    of rule.sha256 (inputs_before_hash: every input file present and not newer than rule.sha256). Reads no target."""
    if not f1_raw or any(fin(f1_raw.get(L)) is None for L in PRIMARY + NONINF):
        return {"evaluated": False, "reason": "the raw v1_A lines are not in F1's results.json"}
    d = {L: dA[L]["value"] for L in dA}
    m, ni = margins(f1_raw, d)
    bp = badly_posed(f1_raw, m)
    inputs = [F1_RESULTS if f1_results is None else f1_results] + sorted(
        {f for v in dA.values() for f in (v.get("files") or [])})
    times = file_times(inputs)
    sha = RULE_SHA_PATH if rule_sha_path is None else rule_sha_path
    hashed = file_times([sha])[rel(sha)]
    before = bool(hashed is not None and all(t is not None and t <= hashed for t in times.values()))
    return {"evaluated": True, "raw_from": rel(F1_RESULTS if f1_results is None else f1_results), "raw": f1_raw,
            "seed_differences": d, "any_fallback": any(v.get("fallback") for v in dA.values()), "margins": m,
            "noninferiority_margins": ni, "badly_posed_lines": bp["lines"],
            "calibrated_at_most": bp["calibrated_at_most"], "input_times": times,
            "rule_sha256_file": rel(sha), "rule_hashed_at": hashed, "inputs_before_hash": before}


def fixed_before_hash_text(pre, bp):
    """The sentence of the badly posed statement on what was computable before the hash (pre: pre_hash_check)."""
    if not pre.get("evaluated"):
        return f"Whether the margins were computable before the hash was not evaluated: {pre.get('reason')}."
    nd = {"S12": 2, "S14": 2, "V12": 3, "V14": 3}
    files = "; ".join(f"{p} ({t or 'absent'})" for p, t in pre["input_times"].items())
    lines = pre["badly_posed_lines"]
    raw = ", ".join(f"{L} {pre['raw'][L]:.{nd[L]}f}" for L in PRIMARY)
    mg = ", ".join(f"{L} {pre['margins'][L]:.{nd[L]}f}" for L in PRIMARY)
    head = (f"Computable before the hash: the registered margins follow from the raw v1_A lines of {pre['raw_from']} "
            f"(quoted in rule section 0: {raw}) and the step-4,000 seed differences, giving margins {mg}. Input files "
            f"and their modification times: {files}; {pre['rule_sha256_file']} written {pre['rule_hashed_at']}. ")
    if lines and pre["inputs_before_hash"]:
        same = ("the same badly posed " if lines == bp["lines"] else "badly posed ") + (
            "lines" if len(lines) > 1 else "line")
        return (head + f"Every input predates the hash, and they give {same} ({', '.join(lines)}): the registered R2 "
                "could not read CONFIRMED whatever the v1_A decode showed. The margins were not checked for "
                "attainability before the rule was hashed.")
    if lines:
        return head + (f"They give badly posed line(s) {', '.join(lines)}, but not every input file predates "
                       "rule.sha256 by its modification time.")
    return head + "From these pre-hash inputs no margin is badly posed; the badly posed line comes from the decode."


def second_basis_text(seeds_rel, any_fallback):
    """The basis of the second reading, said as it is: the rule names the B-end differences for v1_A only for a
    missing step-4,000 value."""
    return (f"the B-end seed differences of {seeds_rel}. Rule section 7 names them for v1_A only when a step-4,000 "
            "value is missing" + (", as one is here for some line." if any_fallback else
                                  "; none is missing here, so the rule does not back this basis."))


def r2_reading(r1, dev, reason=None):
    """CONFIRMED / PARTIAL / NOT CONFIRMED / NOT EVALUATED of rule section 8 from the v1_A R1 and dev part."""
    if r1 is None or dev is None:
        return {"reading": "NOT EVALUATED", "reason": reason or "the v1_A training-fold decode does not exist",
                "failed": []}
    failed = []
    if r1["reading"] != "TRANSFERS":
        failed.append("R1 at v1_A")
    if dev["reading"] == "NO-GO":
        failed.append("the dev part")
    if failed:
        return {"reading": "NOT CONFIRMED", "failed": failed, "reason": None}
    return {"reading": "CONFIRMED" if dev["reading"] == "GO" else "PARTIAL", "failed": [], "reason": None,
            "lines": dev["met_lines"]}


def route(r0_ok, r1_b, r2):
    """The route of rule section 9, as text."""
    if not r0_ok:
        return "R0 failed: nothing is proposed; the departure from F1 or F2-CHAL is a harness defect, investigated first."
    bad = [k for k, v in r1_b.items() if (v or {}).get("reading") != "TRANSFERS"]
    if bad:
        return (f"R1 does not read TRANSFERS at {', '.join(bad)}: nothing is proposed; a departure from F2-CHAL is a "
                "harness defect, investigated first.")
    r = r2["reading"]
    if r == "CONFIRMED":
        return ("R0 holds, R1 TRANSFERS at both B ends, R2 CONFIRMED: the owner may be asked for a dated revision of "
                "docs/rules/G3_RULE.md with a calibrated row (both lines), the raw row kept; until that revision exists, "
                "dated before any lockbox read, nothing of the head enters a G3 verdict.")
    if r == "PARTIAL":
        return (f"R2 PARTIAL: the same revision may be proposed for {', '.join(r2.get('lines') or [])} only; the head "
                "of the other line stays a diagnostic. Nothing enters a G3 verdict before a dated revision of "
                "docs/rules/G3_RULE.md.")
    if r == "NOT CONFIRMED":
        return (f"R2 NOT CONFIRMED ({' and '.join(r2['failed'])} failed): no revision is proposed; the head stays a "
                "diagnostic of the B ends and G3 reads the raw row only.")
    return f"R2 NOT EVALUATED ({r2.get('reason')}): nothing is proposed yet."


# ---------------------------------------------------------------- equality asserts and R0
def ic_g1_asserts(strides, speeds, sets, rc, g_s, g_v, diag_doc, g1_point):
    """The rule's IC and G1 asserts (section 7): the calibrated lines, recomputed on an independent path (decoded
    values divided in a copy of the unit frames), equal the recalibration block per person within 1e-9 on the same
    persons and units; the IC line and G1a carried unchanged (the head reads no IC or G1 input). HarnessDefect on a
    departure."""
    g = G()
    for line, (kind, set_name) in g.LINE_DEF.items():
        u = g.subset(strides if kind == "stride" else speeds, sets[set_name])
        cal = u.assign(y=u["y"] / (g_s if kind == "stride" else g_v))
        pp = g.person_abs_error(cal, 1.0)
        got = rc["recalibrated"][line]["per_person"]
        if set(pp) != set(got) or set(got) != set(rc["raw"][line]["per_person"]):
            raise HarnessDefect(f"{line}: the calibrated line reads other persons than the raw line")
        bad = [p for p in pp if abs(pp[p] - got[p]) > 1e-9]
        if bad:
            raise HarnessDefect(f"{line}: calibrated person values differ on an independent path ({bad[:3]})")
    ic = ((diag_doc.get("rows") or {}).get("base") or {}).get("ic_walkway") or {}
    icv = fin(ic.get("median_ms"))
    return {"same_units": True, "independent_path_tol": 1e-9,
            "ic_median_ms": {"raw": icv, "calibrated": icv, "reference": ic.get("reference")},
            "g1a_mm": {"raw": g1_point, "calibrated": g1_point},
            "note": "the head divides the matched stride and pass-speed values only; the IC rows and G1 are not read "
                    "by it, so their lines are the same numbers in both rows"}


def r0_check(end_res, f1_end, chal_end, tol=None):
    """R0 at a B end: gains and the raw, calibrated and paired line points equal F1's results.json, the in-sample and
    LOPO S54 and V54 equal F2-CHAL's test X, within tol. Returns {ok, differences, missing}."""
    tol = R0_TOL if tol is None else float(tol)
    diffs, missing = {}, []
    if not end_res.get("recalibration"):
        return {"ok": False, "differences": {}, "missing": ["training-fold decode"], "tolerance": tol}
    hd = end_res["head"]
    for kind in ("stride", "speed"):
        ref = (f1_end.get("gain") or {}).get(kind, {}).get("gain")
        diffs[f"g_{kind}"] = None if ref is None else abs(hd[kind] - ref)
    rc, f1rc = end_res["recalibration"], f1_end.get("recalibration") or {}
    for block in ("raw", "recalibrated", "paired"):
        for L in rc[block]:
            ref = ((f1rc.get(block) or {}).get(L) or {}).get("point")
            diffs[f"{block} {L}"] = None if ref is None else abs(rc[block][L]["point"] - ref)
    cf = end_res["cross_fit"]
    for key, kind, field in (("S_in", "stride", "in_sample"), ("S_lopo", "stride", "lopo"),
                             ("V_in", "speed", "in_sample"), ("V_lopo", "speed", "lopo")):
        ref = (chal_end or {}).get(key)
        diffs[f"CHAL {key}"] = None if ref is None else abs(cf[kind][field] - ref)
    missing = [k for k, v in diffs.items() if v is None]
    ok = not missing and all(v <= tol for v in diffs.values())
    return {"ok": bool(ok), "differences": diffs, "missing": missing, "tolerance": tol,
            "max_difference": max((v for v in diffs.values() if v is not None), default=None)}


# ---------------------------------------------------------------- one end
def analyse_end(end, sets, locked, tree=None, f1_out=None, n_boot=None, seed=None, log=print):
    """Dev units, raw lines (consistency with diag.json), and with a training-fold decode: the head, the calibrated
    lines, the cross-fit (R1) and the equality asserts. A decode that fails its checks (another checkpoint, other
    inference settings than the diag run) is reported with decode_ok False and no head."""
    g = G()
    check_rule()
    tree = tree or TREE
    dd = g.diag_dir(end.name, tree)
    diag = read_json(os.path.join(dd, "diag.json"))
    ck = diag.get("checkpoint") or {}
    if ck.get("name") != end.name or ck.get("weights", "ema") != "ema":
        raise ValueError(f"{rel(dd, tree)}: checkpoint {ck.get('name')} ({ck.get('weights')}), expected {end.name} (ema)")
    sha = ck.get("sha256")
    st, sp, counts = g.read_events(os.path.join(dd, "events.csv"), sets["dev14"], locked)
    raw = g.lines_for(st, sp, sets, n_boot=n_boot, seed=seed)
    res = {"key": end.key, "name": end.name, "checkpoint": {"path": end.checkpoint, "sha256": sha,
                                                            "phase": ck.get("phase"), "step": ck.get("step")},
           "diag": rel(dd, tree), "dev_counts": counts, "raw": raw, "consistency": g.consistency(raw, diag),
           "decode": {"path": rel(g.decode_dir(end.name, f1_out), tree), "present": False, "ok": False},
           "head": None, "recalibration": None, "cross_fit": None, "r1": None, "asserts": None}
    try:
        dec = g.read_decode(g.decode_dir(end.name, f1_out), sha, sets["train54"], locked)
    except ValueError as e:                                   # a decode of another checkpoint file
        res["decode"].update({"present": True, "reason": str(e)})
        return res
    if dec is None:
        res["decode"]["reason"] = "train_events.csv absent"
        return res
    tst, tsp, tcounts, doc = dec
    diag_inf = {k: (diag.get("inference") or {}).get(k) for k in g.INFER_KEYS}
    diffs = g.inference_differences(diag_inf, doc.get("inference"))
    res["decode"].update({"present": True, "counts": tcounts, "persons": int(tst["person_id"].nunique()),
                          "inference_differences": diffs,
                          "trained_on_among_train54": (doc.get("sets") or {}).get("trained_on_among_train54"),
                          "cross_check": {k: v for k, v in g.cross_check(tst, g.dev_report(end.name, tree),
                                                                          sets["train54"], expected_sha256=sha).items()
                                          if k != "difference_cm"}})
    if diffs:
        res["decode"]["reason"] = f"inference settings differ from the diag run: {diffs}"
        return res
    res["decode"]["ok"] = True
    tu = g.units_of(tst, tsp)
    gs = g.fit_gain(tu["stride_per_stride"], sets["train54"], sets["dev14"], locked)
    gv = g.fit_gain(tu["speed_per_pass"], sets["train54"], sets["dev14"], locked)
    res["head"] = {"stride": gs["gain"], "speed": gv["gain"], "stride_fit": gs, "speed_fit": gv, "fitted_on": "train54"}
    rc = g.recalibration(st, sp, sets, gs["gain"], gv["gain"], None, n_boot, seed)
    res["recalibration"] = rc
    res["cross_fit"] = cross_fit(tst, tsp, sets["train54"], sets["dev14"], locked)
    res["r1"] = r1_reading(res["cross_fit"])
    g1 = (WB().g1_reads(tree, end.name) or {}).get("full")
    res["asserts"] = ic_g1_asserts(st, sp, sets, rc, gs["gain"], gv["gain"], diag, (g1 or {}).get("point"))
    log(f"{end.key}: g_S {gs['gain']:.4f}, g_V {gv['gain']:.4f}; calibrated S(12) "
        f"{rc['recalibrated']['S12']['point']:.3f} cm; R1 {res['r1']['reading']}")
    return res


# ---------------------------------------------------------------- side tables (read nothing)
def dump_refit(name, sets, locked, f1_gain, pattern=None):
    """The stride head refit on the train54 rows of the queued F2 dumps of a checkpoint that are written in the
    events.csv columns (events_base.csv), against F1's g_S: a rerun stability check (reads nothing)."""
    import pandas as pd
    g = G()
    out = []
    allowed = set(sets["dev14"]) | set(sets["train54"])
    for folder in sorted(glob.glob((pattern or DUMP_GLOB).format(name=name))):
        if os.path.basename(folder).startswith("._"):
            continue
        p = os.path.join(folder, "events_base.csv")
        if not os.path.exists(p):
            out.append({"folder": rel(folder), "status": "no events_base.csv (not in the events.csv columns)"})
            continue
        who = pd.read_csv(p, encoding="utf-8", usecols=["person_id"], dtype=str)
        g.guard_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(p))
        text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
        ev = pd.read_csv(p, encoding="utf-8", dtype=text, keep_default_na=False)
        k = ev[(ev["row"] == "base") & (ev["kind"] == "stride") & ev["person_id"].isin(set(sets["train54"]))
               & (ev["matched"].str.strip().str.lower() == "true")]
        u = pd.DataFrame({"person_id": k["person_id"].to_numpy(),
                          "x": pd.to_numeric(k["reference"]).to_numpy(np.float64),
                          "y": pd.to_numeric(k["prediction"]).to_numpy(np.float64)})
        if not len(u):
            out.append({"folder": rel(folder), "status": "no matched train54 stride"})
            continue
        gs = g.fit_gain(u, sets["train54"], sets["dev14"], locked)
        out.append({"folder": rel(folder), "status": "refit", "g_S": gs["gain"], "n_units": gs["n_units"],
                    "n_persons": gs["n_persons"], "minus_f1": gs["gain"] - f1_gain})
    return out


def comparators(name, key):
    """Dev12 lines of the comparators fitted on the same fold (reading nothing): population and supervised baseline
    (reports/bench/g3_supervised_baseline/results.json), the frozen-token probe of the checkpoint
    (reports/bench/frozen_tokens_summary.json) and the post-hoc height-aware head of F2-H1shape."""
    out = {}
    if os.path.exists(SUPERVISED):
        ln = read_json(SUPERVISED)["lines"]["dev12"]
        for who in ("population", "supervised"):
            out[who] = {"S12": fin(ln[who]["stride_mae_cm"]["point"]), "V12": fin(ln[who]["speed_mae_m_s"]["point"]),
                        "source": rel(SUPERVISED)}
    if os.path.exists(FROZEN):
        rows = read_json(FROZEN)["main"].get(name) or []
        pick = {x["line"]: x for x in rows if x.get("set") == "dev12"}
        if pick:
            out["frozen_token_probe"] = {"S12": fin((pick.get("stride_mae_cm") or {}).get("probe")),
                                         "V12": fin((pick.get("speed_mae_m_s") or {}).get("probe")),
                                         "source": rel(FROZEN)}
    if os.path.exists(H1_RESULTS):
        e = (read_json(H1_RESULTS).get("ends") or {}).get(key) or {}
        h = ((e.get("posthoc_height_lever") or {}).get("dev12")) or {}
        if h:
            out["height_aware_head_posthoc"] = {"S12": fin(h.get("height_gain_cm")), "V12": None,
                                                "source": rel(H1_RESULTS)}
    return out


# ---------------------------------------------------------------- the run
def run(sets, locked, f1_doc, chal_doc, ends=None, tree=None, f1_out=None, seeds_json=None, n_boot=None, seed=None,
        log=print):
    """Every reading of the rule; returns (results dict, exit code). No file written."""
    g = G()
    rule_sha = check_rule()
    t0 = time.time()
    ends = g.ENDS if ends is None else ends
    res = {"what": LABEL, "task": TASK, "rule_sha256": rule_sha, "ends": {}, "R0": {}, "R1": {}, "R2": None,
           "route": None, "beside": {}}
    harness = []
    for e in ends:
        try:
            res["ends"][e.key] = analyse_end(e, sets, locked, tree, f1_out, n_boot, seed, log)
        except HarnessDefect as x:
            harness.append(f"{e.key}: {x}")
            res["ends"][e.key] = {"key": e.key, "name": e.name, "harness_defect": str(x)}
    chal_x = (chal_doc.get("X") or {}).get("ends") or {}
    for k in B_ENDS:
        er = res["ends"].get(k) or {}
        res["R0"][k] = (r0_check(er, (f1_doc.get("ends") or {}).get(k) or {}, chal_x.get(k))
                        if "harness_defect" not in er else {"ok": False, "missing": ["harness defect"]})
        res["R1"][k] = dict(er.get("r1") or {"reading": "NOT EVALUATED"}, seen=True,
                            label="seen before the rule (F2-CHAL test X); reproduced")
    r0_ok = all(v["ok"] for v in res["R0"].values()) and not harness
    a = res["ends"].get(CONFIRM_END) or {}
    seen = (((f1_doc.get("ends") or {}).get(CONFIRM_END) or {}).get("descriptive") or {}).get("dev12", {})
    seen_gain = (seen.get("stride_per_stride") or {}).get("gain") or {}
    r2 = {"seen": {"v1_A_dev12_stride_gain": seen_gain.get("point"), "interval": seen_gain.get("interval"),
                   "source": rel(F1_RESULTS), "statement": "the v1_A dev12 gain was seen before the rule: the direction "
                                                           "of R2 is not blind"}}
    if not r0_ok:
        r2.update(r2_reading(None, None, "R0 failed" + (f" ({'; '.join(harness)})" if harness else "")))
    elif not a.get("recalibration"):
        r2.update(r2_reading(None, None, (a.get("decode") or {}).get("reason") or "no training-fold decode of v1_A"))
    else:
        dB = seed_diffs_b(seeds_json)
        dA = seed_diffs_a(tree, fallback=dB)
        d = {L: dA[L]["value"] for L in dA}
        raw = {L: a["recalibration"]["raw"][L]["point"] for L in PRIMARY + NONINF}
        m, ni = margins(raw, d)
        dev = dev_part(a["recalibration"], m, ni)
        r2.update(r2_reading(a["r1"], dev))
        r2.update({"seed_differences": dA, "margins": m, "noninferiority_margins": ni, "dev_part": dev,
                   "r1": a["r1"]})
        bp = badly_posed(raw, m)
        if bp["lines"]:                     # rule section 11: say so and keep both readings
            m2, ni2 = margins(raw, dB)
            dev2 = dev_part(a["recalibration"], m2, ni2)
            f1_raw = {L: (((((f1_doc.get("ends") or {}).get(CONFIRM_END) or {}).get("raw") or {}).get(L) or {})
                          .get("point")) for L in PRIMARY + NONINF}
            pre = pre_hash_check(f1_raw, dA)
            r2["badly_posed"] = {
                **bp, "statement": (f"the registered success margin of {', '.join(bp['lines'])} is at least the raw "
                                    "line itself, so no calibration can meet it: the criterion is badly posed (rule "
                                    "section 11); both readings are kept"),
                "pre_hash": pre, "fixed_before_hash": fixed_before_hash_text(pre, bp),
                "second_reading": {"basis": second_basis_text(rel(SEEDS_JSON if seeds_json is None else seeds_json),
                                                              any(v.get("fallback") for v in dA.values())),
                                   "label": "not the registered reading; post hoc: its basis was chosen after the v1_A "
                                            "dev lines were seen",
                                   "chosen_after_v1A_dev_lines_seen": True, "seed_differences": dB, "margins": m2,
                                   "noninferiority_margins": ni2, "dev_part": dev2,
                                   "reading": r2_reading(a["r1"], dev2)["reading"]}}
    res["R2"] = r2
    res["route"] = route(r0_ok, res["R1"], r2)
    if (r2.get("badly_posed") or {}).get("lines"):
        bp = r2["badly_posed"]
        sr = bp["second_reading"]
        res["route"] += (" The registered v1_A margins proved badly posed (rule section 11): the margin of "
                         f"{', '.join(bp['lines'])} is at least the raw line, so no calibration can meet it"
                         + ("; this was computable before the rule was hashed, from files that predate it, so the "
                            "registered R2 could not read CONFIRMED whatever the v1_A decode showed"
                            if (bp.get("pre_hash") or {}).get("inputs_before_hash")
                            and (bp.get("pre_hash") or {}).get("badly_posed_lines") else "")
                         + f". On the B-end seed differences, a basis chosen after the v1_A dev lines were seen, the dev "
                         f"part reads {sr['dev_part']['reading']} and R2 {sr['reading']}: a post hoc reading, not the "
                         "registered one, that proposes nothing by itself. Both readings are kept; the registered "
                         "route stands unless the owner decides otherwise. Recommended: a version 2 rule whose margins "
                         "are checked for attainability before it is hashed, read blind at the next checkpoint "
                         "(v1_A can no longer be read blind).")
    res["harness_defects"] = harness
    # beside, reading nothing
    for k, er in res["ends"].items():
        if er.get("head"):
            res["beside"].setdefault("gains", {})[k] = {"stride": er["head"]["stride"], "speed": er["head"]["speed"]}
    vb = res["ends"].get("v1_B") or {}
    if vb.get("head"):
        res["beside"]["dump_refit_v1_B"] = dump_refit(vb["name"], sets, locked, vb["head"]["stride"])
    res["beside"]["comparators"] = {e.key: comparators(e.name, e.key) for e in ends}
    res["timing_s"] = round(time.time() - t0, 1)
    code = EXIT_HARNESS if not r0_ok else EXIT_OK if r2["reading"] != "NOT EVALUATED" else EXIT_NO_V1A
    return res, code


# ---------------------------------------------------------------- report
def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def _nd(unit):
    return 2 if unit == "cm" else 3


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


def strip_per_person(res):
    """The results without the per-person maps of the dev lines (kept: the R1 per-person lines)."""
    out = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
    for e in out["ends"].values():
        for blk in [e.get("raw") or {}] + [((e.get("recalibration") or {}).get(b) or {}) for b in
                                          ("raw", "recalibrated", "paired")]:
            for v in blk.values():
                if isinstance(v, dict):
                    v.pop("per_person", None)
    return out


def summary_md(r):
    L = [f"# F2 lever 1: the training-fold calibration head ({LABEL})", "",
         f"Written by scripts/f2_lever1.py (task {TASK}) under reports/bench/f2_lever1/rule.txt (SHA-256 "
         f"{r['rule_sha256']}, asserted before any target value was read). Every number below is read from results.json "
         "of this folder. No lockbox person, EVAL, CODY or care_pd record was read (asserted). The dev lines of v1_B and "
         "s2_B are post hoc (seen in F1); nothing here is a G3 read, and nothing of the head enters a G3 verdict until "
         "docs/rules/G3_RULE.md says so in a dated revision written before any lockbox read.", "",
         "## Route", "", r["route"], ""]
    r2 = r["R2"]
    L += [f"## R2, the v1_A confirmation: **{r2['reading']}**", "",
          f"Seen before the rule: the v1_A dev12 stride gain {fmt(r2['seen']['v1_A_dev12_stride_gain'], 3)} "
          f"{fmt_iv(r2['seen']['interval'], 3)} ({r2['seen']['source']}); the direction of R2 is not blind."]
    if r2["reading"] == "NOT EVALUATED":
        L += ["", f"Not evaluated: {r2['reason']}."]
    else:
        dv = r2["dev_part"]
        L += ["", f"R1 at v1_A: {r2['r1']['reading']}. Dev part: {dv['reading']}." +
              (f" Failed: {', '.join(r2['failed'])}." if r2["failed"] else ""), "",
              "| line | raw | calibrated | raw minus calibrated | 95 % paired | margin | meets | lower above 0 |",
              "|---|---|---|---|---|---|---|---|"]
        for Ln, x in dv["primary"].items():
            nd = 2 if Ln.startswith("S") else 3
            L.append(f"| {Ln} | {fmt(x['raw'], nd)} | {fmt(x['calibrated'], nd)} | {fmt(x['raw_minus_calibrated'], nd)} | "
                     f"{fmt_iv(x['paired_interval'], nd)} | {fmt(x['margin'], nd)} | {x['meets_margin']} | "
                     f"{x['lower_above_0']} |")
        L += ["", "| non-inferiority | raw | calibrated | calibrated minus raw | margin | ok |", "|---|---|---|---|---|---|"]
        for Ln, x in dv["noninferiority"].items():
            nd = 2 if Ln.startswith("S") else 3
            L.append(f"| {Ln} | {fmt(x['raw'], nd)} | {fmt(x['calibrated'], nd)} | {fmt(x['calibrated_minus_raw'], nd)} | "
                     f"{fmt(x['margin'], nd)} | {x['ok']} |")
        L += ["", "Seed differences d_L (margins): " + "; ".join(
            f"{k} {fmt(v['value'], 3)} (seed 1 {fmt(v.get('seed1'), 3)}, seed 2 {fmt(v.get('seed2'), 3)}; {v['source']})"
            for k, v in r2["seed_differences"].items()) + "."]
        bp = r2.get("badly_posed")
        if bp and bp.get("lines"):
            sr = bp["second_reading"]
            L += ["", f"**Badly posed criterion**: {bp['statement']}. Largest calibrated line that would meet the "
                      "registered margin: " + ", ".join(f"{k} {fmt(v, 3)}" for k, v in bp["calibrated_at_most"].items())
                  + ".", "", bp.get("fixed_before_hash") or "", "",
                  f"Second reading ({sr['label']}): margins on {sr['basis']} Dev part "
                  f"**{sr['dev_part']['reading']}**, R2 **{sr['reading']}**.", "",
                  "| line | raw minus calibrated | 95 % paired | margin | meets | lower above 0 |", "|---|---|---|---|---|---|"]
            for Ln, x in sr["dev_part"]["primary"].items():
                nd = 2 if Ln.startswith("S") else 3
                L.append(f"| {Ln} | {fmt(x['raw_minus_calibrated'], nd)} | {fmt_iv(x['paired_interval'], nd)} | "
                         f"{fmt(x['margin'], nd)} | {x['meets_margin']} | {x['lower_above_0']} |")
            for Ln, x in sr["dev_part"]["noninferiority"].items():
                nd = 2 if Ln.startswith("S") else 3
                L.append(f"| {Ln} (non-inferiority) | calibrated minus raw {fmt(x['calibrated_minus_raw'], nd)} | | "
                         f"{fmt(x['margin'], nd)} | ok {x['ok']} | |")
    L += ["", "## R1, cross-fit transfer on train54 (leave one person out)", "",
          "| end | seen | S54 in sample (cm) | S54 LOPO | difference | tolerance | V54 in sample (m/s) | V54 LOPO | "
          "difference | tolerance | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        r1 = e.get("r1")
        if not r1:
            L.append(f"| {k} | | no training-fold decode | | | | | | | | NOT EVALUATED |")
            continue
        s, v = r1["lines"]["S54"], r1["lines"]["V54"]
        L.append(f"| {k} | {'yes (F2-CHAL)' if k in B_ENDS else 'no (blind)'} | {fmt(s['in_sample'])} | {fmt(s['lopo'])} | "
                 f"{fmt(s['lopo_minus_in'], 3)} | {fmt(s['tolerance'], 3)} | {fmt(v['in_sample'], 4)} | "
                 f"{fmt(v['lopo'], 4)} | {fmt(v['lopo_minus_in'], 4)} | {fmt(v['tolerance'], 4)} | {r1['reading']} |")
    L += ["", "## R0, reproduction of F1 and F2-CHAL (B ends)", "",
          "| end | ok | largest difference | missing |", "|---|---|---|---|"]
    for k, v in r["R0"].items():
        L.append(f"| {k} | {v['ok']} | {fmt(v.get('max_difference'), 10) if v.get('max_difference') is not None else 'n/a'} | "
                 f"{', '.join(v.get('missing') or []) or 'none'} |")
    L += ["", "## Lines, raw and calibrated (dev; v1_B and s2_B post hoc)", "",
          "| end | line | unit | persons | raw | calibrated | 95 % | raw minus calibrated | 95 % paired |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        rc = e.get("recalibration")
        tag = " (post hoc)" if k in B_ENDS else ""
        for Ln in ("S12", "S14", "V12", "V14"):
            raw = (e.get("raw") or {}).get(Ln) or {}
            nd = _nd(raw.get("unit"))
            if rc:
                c, p = rc["recalibrated"][Ln], rc["paired"][Ln]
                L.append(f"| {k}{tag} | {Ln} | {raw.get('unit')} | {raw.get('n_persons')} | {fmt(raw.get('point'), nd)} | "
                         f"{fmt(c['point'], nd)} | {fmt_iv(c['interval'], nd)} | {fmt(p['point'], nd)} | "
                         f"{fmt_iv(p['interval'], nd)} |")
            else:
                L.append(f"| {k} | {Ln} | {raw.get('unit')} | {raw.get('n_persons')} | {fmt(raw.get('point'), nd)} | "
                         "no head | | | |")
    L += ["", "## Beside (reading nothing)", "", "Heads (g_S, g_V): " + "; ".join(
        f"{k} {fmt(v['stride'], 4)}, {fmt(v['speed'], 4)}" for k, v in (r["beside"].get("gains") or {}).items()) + "."]
    dr = r["beside"].get("dump_refit_v1_B") or []
    L.append("Head refit on the queued F2 dumps of v1_B: " + ("; ".join(
        f"{x['folder']}: {x['status']}" + (f", g_S {fmt(x['g_S'], 4)} (minus F1 {fmt(x['minus_f1'], 4)})"
                                           if x["status"] == "refit" else "") for x in dr) if dr else "no dump folder yet")
             + ".")
    L += ["", "| end | comparator | S(12) cm | V(12) m/s | source |", "|---|---|---|---|---|"]
    for k, c in r["beside"].get("comparators", {}).items():
        for who, x in c.items():
            L.append(f"| {k} | {who} | {fmt(x['S12'])} | {fmt(x['V12'], 3)} | {x['source']} |")
    L += ["", "## Checks", ""]
    for k, e in r["ends"].items():
        if "harness_defect" in e:
            L.append(f"- {k}: HARNESS DEFECT: {e['harness_defect']}")
            continue
        d = e["decode"]
        txt = (f"- {k} ({e['checkpoint']['path']}, SHA-256 {e['checkpoint']['sha256']}): raw lines equal diag.json "
               f"(consistency ok). Training-fold decode {d['path']}: ")
        if not d["present"]:
            txt += "absent."
        elif not d["ok"]:
            txt += f"present, not used: {d.get('reason')}."
        else:
            cc = d.get("cross_check") or {}
            txt += (f"{d['persons']} persons; inference settings equal to the diag run; per-person rerun check: "
                    + (f"{cc.get('persons_compared')} persons, largest {fmt(cc.get('max_abs_difference_cm'), 3)} cm"
                       if cc.get("evaluated") else str(cc.get("reading"))) + ".")
            a = e.get("asserts") or {}
            txt += (f" IC and G1 asserts hold (IC {fmt(a.get('ic_median_ms', {}).get('raw'), 1)} ms and G1a "
                    f"{fmt(a.get('g1a_mm', {}).get('raw'), 1)} mm in both rows).")
        L.append(txt)
    lb = r["lockbox"]
    L += [f"- Lockbox: persons of the sets in the weargait_pd lockbox list {lb['persons_in_lockbox']} (asserted 0).",
          f"- Sets: dev14 SHA-256 {r['sets']['dev14_sha256']}, train54 SHA-256 {r['sets']['train54_sha256']}.",
          f"- MoveAtlas: HEAD {r['moveatlas'].get('head') or 'n/a'}. Run started {r['run_started']}, {r['timing_s']} s.",
          ""]
    return "\n".join(L) + "\n"


def finalize(res, sets, locked, stamp, started):
    from mdfm.data import sources
    out = strip_per_person(res)
    out["run_started"] = started
    out["rule"] = {"path": rel(RULE_PATH), "sha256": res["rule_sha256"], "asserted_before_targets": True}
    out["sets"] = {k: sorted(v) for k, v in sets.items()}
    out["sets"].update({f"{k}_sha256": sources.person_hash(v) for k, v in sets.items()})
    n_lb = len((set(sets["dev14"]) | set(sets["train54"])) & set(locked))
    if n_lb:
        raise AssertionError(f"{n_lb} lockbox persons in the sets")
    out["lockbox"] = {"persons_in_lockbox": n_lb, "asserted": True, "list_size": len(locked)}
    out["moveatlas"] = {"head": (stamp or {}).get("head")}
    return clean(out)


def job_command(tree=None):
    """The scripts/queue_job.py argument list of the v1_A training-fold decode (rule section 4): the command of
    scripts/g3_gain_check.py --print-jobs for the v1_A End, with the batch_windows of the inference block of the
    v1_A diag run (asserted equal to the End entry), cls 1, priority 70, group F2L1, explicit --out."""
    g = G()
    end = next(e for e in g.ENDS if e.key == CONFIRM_END)
    inf = read_json(os.path.join(g.diag_dir(end.name, tree), "diag.json")).get("inference") or {}
    bw = int(inf["batch_windows"])
    if bw != end.batch_windows:
        raise AssertionError(f"batch_windows {bw} of the diag run differs from the End entry {end.batch_windows}")
    out = f"reports/bench/g3_gain_check/decode_{end.name}"
    return ["scripts/queue_job.py", "--name", "f2l1_gain_decode_v1_A", "--cls", "1", "--priority", "70",
            "--est-hours", str(g.JOB_EST_HOURS), "--group", "F2L1", "--requires", end.checkpoint,
            "--outputs", f"{out}/train_events.csv",
            "--note", f"F2-L1 rule R2: decode of the 54 training-fold controls by {end.name} (the v1_A confirmation)",
            "--", "scripts/g3_gain_check.py", "--decode", "--checkpoint", end.checkpoint, "--batch-windows", str(bw),
            "--memory-gb", g.JOB_MEMORY_GB, "--out", out]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boot", type=int, default=None, help="bootstrap resamples (default the rule's 10,000)")
    ap.add_argument("--print-job", action="store_true", help="print the queue_job.py command of the v1_A decode")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_job:
        py = "<conda>/envs/mdfm/python.exe"
        print(" ".join([py] + [f'"{x}"' if " " in x else x for x in job_command()]))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the readings run on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    from mdfm.gates import common, g3
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    check_rule()
    f1_doc, chal_doc = read_json(F1_RESULTS), read_json(CHAL_RESULTS)
    locked = g3.lockbox_persons()
    sets = load_sets(f1_doc, locked, g3.controls("dev"), g3.controls("train"))
    res, code = run(sets, locked, f1_doc, chal_doc, n_boot=a.boot)
    stamp = G().stamp_moveatlas()
    out = finalize(res, sets, locked, stamp, started)
    text = summary_md(out)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(OUT, "results.json"), json.dumps(out, indent=1, ensure_ascii=False,
                                                                    allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), text)
    inputs = [F1_RESULTS, CHAL_RESULTS, SEEDS_JSON, RULE_PATH] + sorted(
        glob.glob(os.path.join(F1_DIR, "decode_*", "train_events.csv")))
    prov = {"script": "scripts/f2_lever1.py", "task": TASK, "label": LABEL, "run_at": started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "numpy": np.__version__, "rule_sha256": res["rule_sha256"], "exit_code": code,
            "inputs": {rel(p): G().sha256_file(p) for p in inputs if os.path.exists(p)},
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "g3_gain_check.py"),
                                               os.path.join(SCRIPTS, "weekend_branch.py"), g3.__file__]),
            "code_files": ["scripts/f2_lever1.py", "scripts/g3_gain_check.py", "scripts/weekend_branch.py",
                           "mdfm/gates/g3.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    print(text)
    print(f"written {rel(OUT)}; exit code {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
