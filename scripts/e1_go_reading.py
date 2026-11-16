"""E1P review (2026-10-06): the numbers of E1's reading in the GO branch of the F2 lever-2 reading, their attainability
check, their hash, and the requires of the E1 job file of that branch. Label of every output: dev diagnostic, not a
G3 read.

Rule: reports/bench/jepa_smoke_f2l2/go_reading_rule.txt, SHA-256 in go_reading_rule.sha256 beside it and in
RULE_SHA256 here; check_rule asserts that the three agree before any other file is read. The rule was hashed before
the f2l2 B end existed. Sections below name the rule's sections.

Order of a run (run):
1. check_rule; then the order guard (e1_outputs): nothing of E1 in this branch may exist yet (its run folder
   cache/runs/pilot_e1_f2l2_jepa, an e1_jepa_*.job.json in the rule's folder, a report folder or a queue job naming
   pilot_e1_f2l2_jepa), and go_reading_numbers.sha256 must not exist (the numbers are never rewritten).
2. The F2 lever-2 verdict, verdict.rule of reports/bench/f2_lever2/results.json: anything but GO stops the run
   (EXIT_NOT_GO), nothing written.
3. Rule section 4: the lines of ref_B (pilot_f2l2_ckpt_B_032044) and of ref_4000 (pilot_f2l2_ckpt_B_004000, early,
   reported only) with the seed terms of rule section 3 (v1 against seed 2), as f2_lever2.margins_from reads them for
   v1 (scripts/f2_lever2.py and scripts/weekend_branch.py, by import); J0 at ref_B (P1, dP, CARRY or BOTTLENECK, the
   prediction band, the leakage number); the screen go line (Y1, d, m) in S_pass(12) through scripts/jp0_read.py
   (analyse_checkpoint: refit and refit check of J0's probes, pass unit).
4. Rule section 5: the floors of the success target and the power check (power) of the success line and of the screen
   go line, the outcome ATTAINABLE or NOT ATTAINABLE.
5. Writes go_reading_numbers.json and go_reading_numbers.txt into the rule's folder and the SHA-256 of the .txt into
   go_reading_numbers.sha256, with the time and the order-guard listing.
A missing input of the B-end lines, of J0 at ref_B or of the screen stops the run with EXIT_MISSING (NOT EVALUATED),
nothing written; a missing early input is reported as missing.

--add-requires JOB (rule section 6, step 3): after scripts/jepa_lambda.py --e1-job has written E1's job file of this
branch, add go_reading_rule.sha256 and go_reading_numbers.sha256 to its requires (queue_job.normalize, the file
rewritten in place), so that the runner holds E1 until both exist. Refused for a job file of another config or run
folder, before the numbers' hash holds, or when a job of that name is already queued.

Data discipline: dev14 and dev12 from reports/bench/g3_gain_check/results.json through f2_lever2.load_sets (SHA-256
of rule.txt section 3 of the F2 lever-2 rule, disjoint from mdfm.gates.g3.lockbox_persons()); the persons of every
diag row and G3 folder read are guarded. No lockbox, EVAL, CODY or care_pd record; CPU only (CUDA_VISIBLE_DEVICES=-1
is set before torch can load).

Run (mdfm environment, from the tree root):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/e1_go_reading.py
  ... scripts/e1_go_reading.py --add-requires reports/bench/jepa_smoke_f2l2/e1_jepa_train.job.json
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the reading never sees the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
LABEL = "dev diagnostic, not a G3 read"
TASK = "E1P (review)"
FOLDER = "reports/bench/jepa_smoke_f2l2"
RULE_REL = f"{FOLDER}/go_reading_rule.txt"
RULE_SHA_REL = f"{FOLDER}/go_reading_rule.sha256"
RULE_SHA256 = "93cf1216ee0b8f1c9b83e5d0e117f96ba426fade352184cfd4ea070ab429bd34"
NUMBERS_JSON, NUMBERS_TXT, NUMBERS_SHA = "go_reading_numbers.json", "go_reading_numbers.txt", \
    "go_reading_numbers.sha256"
VERDICT_REL = "reports/bench/f2_lever2/results.json"
SUPERVISED_REL = "reports/bench/g3_supervised_baseline/results.json"
F1_RESULTS_REL = "reports/bench/g3_gain_check/results.json"
SEEDS_REL = "reports/gpu_queue/branch_seeds_20261002_211426.json"
E1_RUN = "pilot_e1_f2l2_jepa"
E1_RUN_DIR = f"cache/runs/{E1_RUN}"
E1_CONFIG = "configs/stage1_pilot_f2l2_jepa.yaml"
POST_SELECTION = "post-selection, optimistic"
# the checkpoints of rule sections 2 and 3 (report-folder names)
NAMES = {"ref_B": "pilot_f2l2_ckpt_B_032044", "ref_4000": "pilot_f2l2_ckpt_B_004000",
         "v1_B": "phaseA_root_full_ckpt_B_032044", "s2_B": "pilot_v1_s2_ckpt_B_032044",
         "v1_4000": "phaseA_root_full_ckpt_B_004000", "s2_4000": "pilot_v1_s2_ckpt_B_004000"}
# the three checkpoints of the screen go line: ref, v1, s2 at ckpt_B_004000 (J0 outputs)
SCREEN = {"ref": NAMES["ref_4000"], "v1": NAMES["v1_4000"], "s2": NAMES["s2_4000"]}
TARGET_FRACTION, OTHER_FRACTION, SEED_MULTIPLE = 0.10, 0.05, 2.0
G10_POINTS = 5.0
BOOT_N, SEED, LEVEL = 2000, 20260929, 0.95
NONINFERIORITY = ("V12", "S14", "V14", "IC", "IC_insole", "G1a")
SV = ("S12", "V12", "S14", "V14")
IC_LINE = {"IC": "ic_median_ms_walkway", "IC_insole": "ic_median_ms_insole"}
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "IC": "ms", "IC_insole": "ms", "G1a": "mm"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "IC": 2, "IC_insole": 2, "G1a": 2}
ATTAINABLE, NOT_ATTAINABLE = "ATTAINABLE", "NOT ATTAINABLE"
EXIT_WRITTEN, EXIT_HASH, EXIT_NOT_GO, EXIT_MISSING, EXIT_ORDER, EXIT_EXISTS = 0, 2, 3, 4, 5, 6


class Stop(Exception):
    """A stop of the run with its exit code (main prints the message and returns the code)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = int(code)
        self.message = message


def _sibling(name):
    key = f"_e1go_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def F2():
    """scripts/f2_lever2.py: sets, diag rows, g3_dev lines, G1 reads, the persons guard."""
    return _sibling("f2_lever2")


def WB():
    return F2().WB()


def JP():
    """scripts/jp0_read.py: the pass unit (analyse_checkpoint) and its context."""
    return _sibling("jp0_read")


def QJ():
    return _sibling("queue_job")


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:
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


def first_word(path):
    with open(path, encoding="utf-8") as f:
        words = f.read().split()
    return words[0] if words else ""


# ---------------------------------------------------------------- 1. rule and order
def check_rule(tree=None, expected=None):
    """The SHA-256 of the rule under `tree`, asserted equal to the first word of its hash file and to `expected`
    (default RULE_SHA256); Stop(EXIT_HASH) otherwise. Reads nothing else."""
    tree = tree or TREE
    expected = RULE_SHA256 if expected is None else expected
    rp, sp = os.path.join(tree, RULE_REL), os.path.join(tree, RULE_SHA_REL)
    for p in (rp, sp):
        if not os.path.exists(p):
            raise Stop(EXIT_HASH, f"refused: {rel(p, tree)} is missing")
    got, stored = sha256_file(rp), first_word(sp)
    if not (got == stored == expected):
        raise Stop(EXIT_HASH, f"refused: {RULE_REL} hashes to {got}, {RULE_SHA_REL} holds {stored or 'nothing'}, this "
                              f"script expects {expected}")
    return got


def e1_outputs(tree=None):
    """What of E1 in this branch exists under `tree` (tree-relative paths; empty: nothing): its run folder, an E1 job
    file in the rule's folder, a folder under reports/gates/ or reports/bench/ named after its run, a queue job file
    naming its run folder."""
    tree = tree or TREE
    out = []
    if os.path.exists(os.path.join(tree, E1_RUN_DIR)):
        out.append(E1_RUN_DIR)
    for p in sorted(glob.glob(os.path.join(tree, FOLDER, "e1_jepa_*.job.json"))):
        if not os.path.basename(p).startswith("._"):
            out.append(rel(p, tree))
    for base in ("reports/gates/g1", "reports/gates/g3", "reports/gates/g3_sim", "reports/gates/g4", "reports/bench"):
        d = os.path.join(tree, base)
        if os.path.isdir(d):
            out += [f"{base}/{n}" for n in sorted(os.listdir(d)) if E1_RUN in n and not n.startswith("._")]
    jd = os.path.join(tree, "cache", "runs", "queue", "jobs")
    if os.path.isdir(jd):
        for p in sorted(glob.glob(os.path.join(jd, "*.json"))):
            if os.path.basename(p).startswith(("._", ".")):
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                if E1_RUN_DIR in f.read():
                    out.append(rel(p, tree))
    return out


def verdict_of(tree=None, path=None):
    """(verdict.rule of the F2 lever-2 results, SHA-256 of the file); (None, None) when the file is absent."""
    tree = tree or TREE
    p = path or os.path.join(tree, VERDICT_REL)
    doc = read_json(p)
    if doc is None:
        return None, None
    return ((doc.get("verdict") or {}).get("rule")), sha256_file(p)


# ---------------------------------------------------------------- 3. lines and seed terms
def g3json_line(path, line):
    """(point, None) of a results_dev line of a G3 json file, or (None, reason)."""
    doc = read_json(path)
    if doc is None:
        return None, f"missing: {rel(path)}"
    for r in doc.get("results_dev") or []:
        if isinstance(r, dict) and r.get("line") == line:
            v = fin(r.get("point"))
            return (v, None) if v is not None else (None, f"{line} not finite in {rel(path)}")
    return None, f"no {line} line in {rel(path)}"


def curve_json(tree, name):
    return os.path.join(tree, "reports", "gates", "g3", f"curve_{name}", "g3_dev.json")


def values_at(tree, name, step, sets, locked):
    """{line: value} and {line: reason} of one checkpoint: at the B end (step 'B') S and V from the diag base row,
    the IC lines from dev_<name>/g3_dev.json, G1a from the full G1 run; at step 4,000 ('4000') S and V from the curve
    folder (per_person.csv), the IC lines from its g3_dev.json, G1a from the --limit-takes 2 run. Also the per-person
    stride values on the 12 ('S12_per_person')."""
    f2, wb = F2(), WB()
    vals, why = {}, {}
    if step == "B":
        gset, reason = f2.diag_set(tree, name, sets, locked)
        ic_path = f2.g3dev_path(tree, name)
        g1 = wb.g1_reads(tree, name).get("full")
    else:
        folder = os.path.join(tree, "reports", "gates", "g3", f"curve_{name}")
        gset = wb.g3set_from_folder(folder)
        reason = None if gset is not None else f"missing: {rel(folder, tree)} (g3_dev.json and per_person.csv)"
        f2.guard_set(gset, sets, locked, rel(folder, tree))
        ic_path = curve_json(tree, name)
        g1 = wb.g1_reads(tree, name).get("lt2")
    for k, sub, key in (("S12", wb.SUB12, "S"), ("V12", wb.SUB12, "V"), ("S14", wb.SUB14, "S"),
                        ("V14", wb.SUB14, "V")):
        v = fin(((gset or {}).get("subsets", {}).get(sub) or {}).get(f"{key}_point"))
        vals[k] = v
        if v is None:
            why[k] = reason or f"no {k} in {name}"
    s12 = ((gset or {}).get("subsets", {}).get(wb.SUB12) or {}).get("S") or {}
    vals["S12_per_person"] = {p: fin(v) for p, v in s12.items() if fin(v) is not None}
    for k, line in IC_LINE.items():
        vals[k], r = g3json_line(ic_path, line)
        if r:
            why[k] = r
    vals["G1a"] = fin((g1 or {}).get("point"))
    if vals["G1a"] is None:
        why["G1a"] = f"no G1 dev run ({'full' if step == 'B' else 'lt2'}) of {name} under reports/gates/g1/"
    return vals, why


def seed_terms(tree, step, v1, s2):
    """{line: abs difference v1 against seed 2} (rule section 3): S and V lines from SEEDS_REL, the IC lines and G1a
    from the v1 and s2 values; {line: reason} for a term that cannot be formed."""
    seeds = read_json(os.path.join(tree, SEEDS_REL))
    key = "B_end" if step == "B" else "ckpt_B_004000"
    out, why = {}, {}
    for k in SV:
        v = fin((((seeds or {}).get("seed_diff") or {}).get(key) or {}).get(k, {}).get("abs_diff"))
        out[k] = v
        if v is None:
            why[k] = f"no seed_diff {key} {k} in {SEEDS_REL}"
    for k in ("IC", "IC_insole", "G1a"):
        a, b = fin(v1.get(k)), fin(s2.get(k))
        out[k] = None if a is None or b is None else abs(a - b)
        if out[k] is None:
            why[k] = f"v1 or seed 2 {k} missing at {key}"
    return out, why


def line_block(ref, sd):
    """Rule section 4: the success line of S(12) and the non-inferiority lines, from the reference's values and the
    seed terms. A line without its value or seed term is None."""
    out = {}
    s, t = fin(ref.get("S12")), fin(sd.get("S12"))
    if s is None or t is None:
        out["S12"] = None
    else:
        m = max(TARGET_FRACTION * s, SEED_MULTIPLE * t)
        out["S12"] = {"line": s, "seed_term": t, "ten_percent": TARGET_FRACTION * s, "two_seed": SEED_MULTIPLE * t,
                      "margin": m, "at_most": s - m, "kind": "success"}
    for k in NONINFERIORITY:
        v, t = fin(ref.get(k)), fin(sd.get(k))
        if v is None or t is None:
            out[k] = None
            continue
        m = max(OTHER_FRACTION * v, SEED_MULTIPLE * t)
        out[k] = {"line": v, "seed_term": t, "five_percent": OTHER_FRACTION * v, "two_seed": SEED_MULTIPLE * t,
                  "margin": m, "at_most": v + m, "kind": "non-inferiority"}
    return out


def j0_results(tree, name):
    """J0's results.json of a checkpoint, checked (full run, the checkpoint's name); (doc, None) or (None, reason)."""
    p = os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}", "results.json")
    doc = read_json(p)
    if doc is None:
        return None, f"missing: {rel(p, tree)}"
    if doc.get("partial") or (doc.get("checkpoint") or {}).get("name") != name:
        return None, f"{rel(p, tree)}: partial run or another checkpoint"
    return doc, None


def j0_latent_per_stride(doc):
    return fin((((((doc or {}).get("passes") or {}).get("latent") or {}).get("lines") or {}).get("dev12") or {})
               .get("stride_mae_cm", {}).get("point"))


def j0_section3(tree, names, D):
    """Rule section 4, J0 at ref_B: P1, dP (v1 against seed 2), the reading CARRY or BOTTLENECK, the prediction band
    of E1's probe, the leakage number of ref_B and the G10 caution line."""
    docs, why = {}, {}
    for k in ("ref_B", "v1_B", "s2_B"):
        docs[k], r = j0_results(tree, names[k])
        if r:
            why[k] = r
    if why:
        return None, why
    P1, a, b = (j0_latent_per_stride(docs[k]) for k in ("ref_B", "v1_B", "s2_B"))
    if None in (P1, a, b, fin(D)):
        return None, {"J0": "a latent per-stride line is not finite, or D is missing"}
    dP = abs(a - b)
    m = max(TARGET_FRACTION * D, SEED_MULTIPLE * dP)
    band = max(TARGET_FRACTION * P1, SEED_MULTIPLE * dP)
    leak = fin(((docs["ref_B"].get("leakage") or {}).get("leakage_points")))
    return {"D": D, "P1": P1, "P1_v1": a, "P2_s2": b, "dP": dP, "margin": m, "D_minus_P1": D - P1,
            "reading": "CARRY" if D - P1 >= m else "BOTTLENECK", "band": band,
            "probe_lower_by_at_least_to": P1 - band, "probe_within": [P1 - band, P1 + band],
            "leakage_points_ref_B": leak,
            "g10_caution_above": None if leak is None else leak + G10_POINTS,
            "checkpoint_sha256": {k: (docs[k].get("checkpoint") or {}).get("sha256") for k in docs},
            "sources": {k: f"reports/bench/frozen_tokens_{names[k]}/results.json" for k in docs}}, {}


# ---------------------------------------------------------------- the pass unit (screen)
def jp0_context(tree):
    """The context of jp0_read.analyse_checkpoint, built as jp0_read.run builds it (controls, targets, pass units,
    population and supervised baselines), with the same guards."""
    jp = JP()
    from mdfm.eval import probes
    from mdfm.gates import g3
    train, dev = list(g3.controls("train")), list(g3.controls("dev"))
    locked = g3.lockbox_persons()
    if len(train) != 54 or len(dev) != 14 or set(train) & set(dev) or (set(train) | set(dev)) & set(locked):
        raise Stop(EXIT_MISSING, "controls: not 54 and 14 disjoint non-lockbox persons")
    sets = {"train": train, "dev14": dev, "dev12": [p for p in dev if p not in jp.EXCLUDED_FROM_12]}
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise Stop(EXIT_MISSING, "a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    tgs = {"dev14": tg_dev, "dev12": probes.subset_targets(tg_dev, sets["dev12"])}
    sup_s, sup_v, _ = jp.supervised_passes(tree, u_dev)
    return {"u_tr": u_tr, "u_dev": u_dev, "sets": sets, "tgs": tgs, "locked": locked,
            "population": g3.population_baseline(tg_train), "supervised": (sup_s, sup_v)}


def latent_pass_line(tree, name, ctx):
    """The latent probe's S_pass(12) line of a checkpoint ({point, interval, n_persons, per_person}), through
    jp0_read.analyse_checkpoint (refit and refit check included); Stop(EXIT_MISSING) when J0's output is missing."""
    jp = JP()
    try:
        a = jp.analyse_checkpoint(tree, name, name, ctx, {"gains": {}})
    except jp.Stop as s:
        raise Stop(EXIT_MISSING, f"screen, {name}: {s.message}")
    line = a["stride"]["lines"]["dev12"]["latent"]
    return {"point": fin(line["point"]), "interval": line.get("interval"), "n_persons": line.get("n_persons"),
            "per_person": line.get("per_person") or {}, "refit_check": a["refit"]["check"],
            "checkpoint_sha256": (a.get("checkpoint") or {}).get("sha256"),
            "pass_set": a["stride"]["pass_set"]["dev12"]}


# ---------------------------------------------------------------- 4. attainability
def lower_bound(values_by_person, n_boot=BOOT_N, seed=SEED):
    """The lower bound of the person bootstrap of the mean (mdfm.gates.common.cluster_bootstrap, percentile)."""
    from mdfm.gates import common
    ids = sorted(values_by_person)
    v = np.array([values_by_person[i] for i in ids], np.float64)
    return float(common.cluster_bootstrap(v, np.array(ids), n=n_boot, seed=seed, level=LEVEL)[0])


def power(a_pp, b_pp, m, n_boot=BOOT_N, seed=SEED):
    """Rule section 5, power: e = a minus b per person (two runs that differ by their seed only), c = e minus its
    mean; g_star = minus the lower bound of the bootstrap of c; lower_at_m = the lower bound of the bootstrap of c + m,
    computed directly; clears when lower_at_m is above 0 (the same as m above g_star, up to rounding)."""
    ids = sorted(k for k in set(a_pp) & set(b_pp) if fin(a_pp[k]) is not None and fin(b_pp[k]) is not None)
    if len(ids) < 2 or fin(m) is None:
        return {"n_persons": len(ids), "g_star": None, "lower_at_m": None, "clears": False, "m": fin(m)}
    e = np.array([a_pp[k] - b_pp[k] for k in ids], np.float64)
    c = e - e.mean()
    g_star = -lower_bound(dict(zip(ids, c)), n_boot, seed)
    lam = lower_bound(dict(zip(ids, c + float(m))), n_boot, seed)
    return {"n_persons": len(ids), "mean_difference": float(e.mean()), "sd_difference": float(e.std(ddof=1)),
            "g_star": g_star, "m": float(m), "lower_at_m": lam, "clears": bool(lam > 0)}


def attainability(success, P1, supervised, pw_success, pw_screen):
    """Rule section 5: the floors of the success target (P1 and the supervised baseline, per stride on the 12), the
    power checks; outcome ATTAINABLE when every check holds."""
    T = (success or {}).get("at_most")
    floors = {"P1_latent_probe_ref_B": P1, "supervised_baseline": supervised}
    fl = {k: {"floor": v, "target_at_or_above": None if v is None or T is None else bool(T >= v)}
          for k, v in floors.items()}
    failed = [f"success target {T if T is None else round(T, 2)} cm below the floor {k} "
              f"({v['floor'] if v['floor'] is None else round(v['floor'], 2)} cm)"
              for k, v in fl.items() if v["target_at_or_above"] is not True]
    for what, pw in (("success line", pw_success), ("screen go line", pw_screen)):
        if not pw.get("clears"):
            failed.append(f"{what}: a gap of m = {pw.get('m')} does not clear the paired lower bound under the seed "
                          f"noise (g_star {pw.get('g_star')})")
    return {"target": T, "floors": fl, "recalibrated_floor": "not available: F1 fitted no stride gain for ref_B "
            "(its train54 decode does not exist, F2 lever-2 rule.txt section 9), and the recalibrated row has no "
            "per-stride line", "power_success": pw_success, "power_screen": pw_screen,
            "noninferiority": "attainable by construction (a run equal to its reference meets each line); printed, "
                              "not checked",
            "outcome": ATTAINABLE if not failed else NOT_ATTAINABLE, "failed": failed}


# ---------------------------------------------------------------- the run
def compute(tree, names=None, screen=None, sets=None, locked=None, ctx=None):
    """Every number of rule sections 3 to 5 (no file written). Stop(EXIT_MISSING) when a needed input is missing."""
    names = dict(NAMES if names is None else names)
    screen = dict(SCREEN if screen is None else screen)
    f2 = F2()
    if sets is None or locked is None:
        from mdfm.gates import g3
        locked = g3.lockbox_persons()
        sets = f2.load_sets(read_json(os.path.join(tree, F1_RESULTS_REL)), locked)
    vals, why = {}, {}
    for k in ("ref_B", "v1_B", "s2_B"):
        vals[k], why[k] = values_at(tree, names[k], "B", sets, locked)
    for k in ("ref_4000", "v1_4000", "s2_4000"):
        vals[k], why[k] = values_at(tree, names[k], "4000", sets, locked)
    sd_B, why_sd_B = seed_terms(tree, "B", vals["v1_B"], vals["s2_B"])
    sd_4, why_sd_4 = seed_terms(tree, "4000", vals["v1_4000"], vals["s2_4000"])
    missing = [f"ref_B {k}: {r}" for k, r in why["ref_B"].items()] + \
              [f"seed term {k} (B end): {r}" for k, r in why_sd_B.items()]
    if not vals["v1_B"]["S12_per_person"] or not vals["s2_B"]["S12_per_person"]:
        missing.append("per-person S(12) of v1's or seed 2's B end (diag base rows)")
    b_lines = line_block(vals["ref_B"], sd_B)
    early = line_block(vals["ref_4000"], sd_4)
    D = (b_lines.get("S12") or {}).get("line")
    j0, why_j0 = j0_section3(tree, names, D)
    missing += [f"J0 {k}: {r}" for k, r in why_j0.items()]
    if missing:
        raise Stop(EXIT_MISSING, "NOT EVALUATED, nothing written; missing: " + "; ".join(missing))
    ctx = ctx if ctx is not None else jp0_context(tree)
    lat = {k: latent_pass_line(tree, screen[k], ctx) for k in ("ref", "v1", "s2")}
    if None in (lat["ref"]["point"], lat["v1"]["point"], lat["s2"]["point"]):
        raise Stop(EXIT_MISSING, "NOT EVALUATED, nothing written; a latent S_pass(12) is not finite")
    Y1, d = lat["ref"]["point"], abs(lat["v1"]["point"] - lat["s2"]["point"])
    m = max(TARGET_FRACTION * Y1, SEED_MULTIPLE * d)
    scr = {"Y1": Y1, "v1": lat["v1"]["point"], "s2": lat["s2"]["point"], "d": d, "ten_percent": TARGET_FRACTION * Y1,
           "two_d": SEED_MULTIPLE * d, "m": m, "threshold": Y1 - m, "names": screen,
           "lines": {k: {x: v[x] for x in ("point", "interval", "n_persons", "refit_check", "checkpoint_sha256",
                                           "pass_set")} for k, v in lat.items()}}
    pw_success = power(vals["v1_B"]["S12_per_person"], vals["s2_B"]["S12_per_person"], b_lines["S12"]["margin"])
    pw_screen = power(lat["v1"]["per_person"], lat["s2"]["per_person"], m)
    sup = read_json(os.path.join(tree, SUPERVISED_REL))
    sup12 = fin((((sup or {}).get("lines") or {}).get("dev12") or {}).get("supervised", {}).get("stride_mae_cm", {})
                .get("point"))
    att = attainability(b_lines["S12"], j0["P1"], sup12, pw_success, pw_screen)
    return {"names": names, "values": {k: {x: y for x, y in v.items() if x != "S12_per_person"}
                                       for k, v in vals.items()},
            "missing_early": {k: why[k] for k in ("ref_4000", "v1_4000", "s2_4000") if why[k]},
            "seed_terms": {"B_end": sd_B, "ckpt_B_004000": sd_4, "missing_ckpt_B_004000": why_sd_4},
            "b_end": b_lines, "early": early, "j0": j0, "screen": scr, "attainability": att,
            "supervised_per_stride_12": sup12}


def fmt(x, nd=2):
    return "n/a" if fin(x) is None else f"{x:.{nd}f}"


def numbers_text(res):
    """The text that is hashed: every number of rule sections 3 to 5, the reference's labelled post-selection."""
    L = []
    L.append("E1 in the GO branch of the F2 lever-2 reading: the numbers (scripts/e1_go_reading.py)")
    L.append(f"Label: {LABEL}. Rule: {RULE_REL}, SHA-256 {res['rule_sha256']}. F2 lever-2 verdict (verdict.rule of "
             f"{VERDICT_REL}): {res['verdict']} (file SHA-256 {res['verdict_sha256']}).")
    L.append(f"Reference ref_B {res['names']['ref_B']}, ref_4000 {res['names']['ref_4000']}: every reference number "
             f"below is {POST_SELECTION}; E1's comparisons against these lines are against a post-selection "
             "reference. Seed terms: v1 against seed 2 (rule section 3).")
    L.append("")
    L.append("B end (E1's B end against ref_B):")
    for k in ("S12",) + NONINFERIORITY:
        b = res["b_end"][k]
        nd, u = ND[k], UNIT[k]
        if k == "S12":
            L.append(f"- success S(12): line D {fmt(b['line'], nd)} {u} ({POST_SELECTION}), margin max(10 percent "
                     f"{fmt(b['ten_percent'], nd)}, 2 x seed term {fmt(b['two_seed'], nd)}) = {fmt(b['margin'], nd)} {u}, "
                     f"E1 at most {fmt(b['at_most'], nd)} {u}, with the paired lower bound of ref_B minus E1 above 0")
        else:
            L.append(f"- non-inferiority {k}: line {fmt(b['line'], nd)} {u} ({POST_SELECTION}), seed term "
                     f"{fmt(b['seed_term'], nd)}, margin max(5 percent {fmt(b['five_percent'], nd)}, 2 x seed term "
                     f"{fmt(b['two_seed'], nd)}) = {fmt(b['margin'], nd)} {u}, E1 at most {fmt(b['at_most'], nd)} {u}")
    L.append("")
    L.append("Early check at ckpt_B_004000 (E1 against ref_4000), reported only:")
    for k in ("S12",) + NONINFERIORITY:
        b = res["early"].get(k)
        nd, u = ND[k], UNIT[k]
        if b is None:
            L.append(f"- {k}: not computed (input missing, listed in go_reading_numbers.json)")
        else:
            L.append(f"- {k}: line {fmt(b['line'], nd)} {u} ({POST_SELECTION}), margin {fmt(b['margin'], nd)} {u}, "
                     f"E1 at most {fmt(b['at_most'], nd)} {u}")
    j = res["j0"]
    L.append("")
    L.append(f"J0 at ref_B: P1 {fmt(j['P1'])} cm ({POST_SELECTION}), dP {fmt(j['dP'])} cm (v1 {fmt(j['P1_v1'])}, "
             f"seed 2 {fmt(j['P2_s2'])}), D minus P1 {fmt(j['D_minus_P1'])} against max(10 percent of D, 2 dP) = "
             f"{fmt(j['margin'])}: {j['reading']}. Prediction band of E1's probe: max(10 percent of P1, 2 dP) = "
             f"{fmt(j['band'])} cm (BOTTLENECK: E1's probe at most {fmt(j['probe_lower_by_at_least_to'])} cm; CARRY: "
             f"within [{fmt(j['probe_within'][0])}, {fmt(j['probe_within'][1])}] cm).")
    L.append(f"G10 caution: leakage of ref_B {fmt(j['leakage_points_ref_B'])} points; the caution applies when E1's "
             f"exceeds {fmt(j['g10_caution_above'])} points.")
    s = res["screen"]
    L.append(f"Screen go line (S_pass(12), latent probe at ckpt_B_004000): Y1 {fmt(s['Y1'])} cm ({POST_SELECTION}), "
             f"d {fmt(s['d'])} cm (v1 {fmt(s['v1'])}, seed 2 {fmt(s['s2'])}), m = max({fmt(s['ten_percent'])}, "
             f"{fmt(s['two_d'])}) = {fmt(s['m'])} cm: go when E1's latent S_pass(12) is at most {fmt(s['threshold'])} cm "
             "and the paired lower bound of ref_4000 minus E1 is above 0.")
    a = res["attainability"]
    L.append("")
    L.append(f"Attainability (rule section 5): {a['outcome']}")
    for k, v in a["floors"].items():
        L.append(f"- floor {k}: {fmt(v['floor'])} cm; success target {fmt(a['target'])} cm at or above it: "
                 f"{v['target_at_or_above']}")
    L.append(f"- recalibrated floor: {a['recalibrated_floor']}")
    for what, key in (("success line", "power_success"), ("screen go line", "power_screen")):
        p = a[key]
        L.append(f"- power, {what}: {p['n_persons']} persons, g_star {fmt(p['g_star'])} cm, m {fmt(p['m'])} cm, lower "
                 f"bound of c + m {fmt(p['lower_at_m'])} cm, clears: {p['clears']}")
    L.append(f"- non-inferiority: {a['noninferiority']}")
    for f_ in a["failed"]:
        L.append(f"- failed: {f_}")
    L.append("")
    return "\n".join(L)


def write_numbers(out, res, text, listing, stamp_time):
    """go_reading_numbers.json, .txt and .sha256 into `out`; FileExistsError when the hash file exists."""
    sp = os.path.join(out, NUMBERS_SHA)
    if os.path.exists(sp):
        raise FileExistsError(sp)
    for t in (text, json.dumps(res)):
        if chr(0x2014) in t or chr(0x2013) in t:
            raise ValueError("a long dash in the output")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, NUMBERS_JSON), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    tp = os.path.join(out, NUMBERS_TXT)
    with open(tp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    sha = sha256_file(tp)
    body = (f"{sha}  {NUMBERS_TXT}\n"
            f"what: SHA-256 of the UTF-8 bytes of {NUMBERS_TXT} ({text.count(chr(10))} lines, "
            f"{len(text.encode('utf-8'))} bytes, LF)\n"
            f"time: {stamp_time}, local time of this PC\n"
            f"how: scripts/e1_go_reading.py under {RULE_REL} (SHA-256 {res['rule_sha256']})\n"
            f"E1 outputs at that time: {'none' if not listing else ', '.join(listing)} (run folder {E1_RUN_DIR}, "
            f"e1_jepa_*.job.json in {FOLDER}, reports and queue jobs naming {E1_RUN})\n")
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return sha


def run(tree=None, out=None, names=None, screen=None, guard_tree=None, verdict_path=None, sets=None, locked=None,
        ctx=None, expected=None):
    """Rule section 6, step 2. guard_tree: the tree whose E1 outputs are listed (default `tree`). Returns the
    results dict; raises Stop."""
    tree = os.path.abspath(tree or TREE)
    out = os.path.abspath(out or os.path.join(tree, FOLDER))
    rule_sha = check_rule(tree, expected)                                         # before anything else
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise Stop(EXIT_MISSING, "the reading runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    if os.path.exists(os.path.join(out, NUMBERS_SHA)):
        raise Stop(EXIT_EXISTS, f"{rel(os.path.join(out, NUMBERS_SHA), tree)} exists: the numbers are never "
                                "rewritten")
    listing = e1_outputs(guard_tree or tree)
    if listing:
        raise Stop(EXIT_ORDER, "refused: E1 of the GO branch has outputs already, the numbers must be hashed before "
                               "any of them: " + ", ".join(listing))
    verdict, vsha = verdict_of(tree, verdict_path)
    if verdict != "GO":
        raise Stop(EXIT_NOT_GO, f"not the GO branch: verdict.rule of {VERDICT_REL} is "
                                f"{verdict if verdict is not None else 'absent'}; nothing written")
    t0 = time.time()
    res = compute(tree, names, screen, sets, locked, ctx)
    res.update(what=LABEL, task=TASK, rule_sha256=rule_sha, verdict=verdict, verdict_sha256=vsha,
               label_of_reference=POST_SELECTION, e1_outputs_before=listing,
               constants={"target_fraction": TARGET_FRACTION, "other_fraction": OTHER_FRACTION,
                          "seed_multiple": SEED_MULTIPLE, "boot_n": BOOT_N, "seed": SEED, "level": LEVEL,
                          "g10_points": G10_POINTS},
               moveatlas_head=_sibling("moveatlas_stamp").stamp().get("head"), timing_s=round(time.time() - t0, 1))
    stamp_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    res["computed_at"] = stamp_time
    res = F2().clean(res)
    text = numbers_text(res)
    paths.assert_not_moveatlas(out)
    res["numbers_sha256"] = write_numbers(out, res, text, listing, stamp_time)
    print(text)
    print(f"written {rel(out, tree)}/{NUMBERS_TXT}, {NUMBERS_JSON}, {NUMBERS_SHA} (SHA-256 {res['numbers_sha256']})")
    return res


# ---------------------------------------------------------------- step 3: the E1 job's requires
def add_requires(job_path, tree=None, expected=None):
    """Rule section 6, step 3: go_reading_rule.sha256 and go_reading_numbers.sha256 added to the requires of E1's
    job file of this branch (normalized, rewritten in place; a second call changes nothing). Returns the job."""
    tree = os.path.abspath(tree or TREE)
    check_rule(tree, expected)
    qj = QJ()
    sp, tp = os.path.join(tree, FOLDER, NUMBERS_SHA), os.path.join(tree, FOLDER, NUMBERS_TXT)
    if not (os.path.exists(sp) and os.path.exists(tp)) or first_word(sp) != sha256_file(tp):
        raise Stop(EXIT_ORDER, f"refused: {FOLDER}/{NUMBERS_SHA} does not hold the hash of {NUMBERS_TXT} (run this "
                               "script without --add-requires first)")
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    argv = job.get("argv") or []

    def arg(flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
    if arg("--config") != E1_CONFIG or arg("--run-dir") != E1_RUN_DIR or job.get("run_dir") != E1_RUN_DIR:
        raise Stop(EXIT_ORDER, f"refused: {job_path} is not E1's job of the GO branch ({E1_CONFIG}, {E1_RUN_DIR})")
    if job.get("name") in qj.existing_names(qj.jobs_dir(tree)):
        raise Stop(EXIT_ORDER, f"refused: a job named {job.get('name')} is queued already")
    req = list(job.get("requires") or [])
    for r in (RULE_SHA_REL, f"{FOLDER}/{NUMBERS_SHA}"):
        if r not in req:
            req.append(r)
    job["requires"] = req
    job = qj.normalize(job)
    tmp = f"{job_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(job, f, indent=1)
        f.write("\n")
    os.replace(tmp, job_path)
    return job


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="the tree whose reports are read (default this tree)")
    p.add_argument("--out", default=None, help=f"output folder (default {FOLDER} of the tree)")
    p.add_argument("--add-requires", default=None, metavar="JOB",
                   help="add the two hash files to the requires of E1's job file (rule section 6, step 3)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        if a.add_requires:
            job = add_requires(a.add_requires, a.tree)
            print(f"requires of {a.add_requires}: {job['requires']}")
            return EXIT_WRITTEN
        run(a.tree, a.out)
        return EXIT_WRITTEN
    except Stop as s:
        print(s.message, file=sys.stderr)
        return s.code


if __name__ == "__main__":
    sys.exit(main())
