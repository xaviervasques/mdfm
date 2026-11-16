"""E1R (2026-10-08): the reading of E1, the JEPA auxiliary loss run (cache/runs/pilot_e1_jepa, configs/
stage1_pilot_jepa.yaml at lambda 728009, phase B only from cache/runs/phaseA_root_full/ckpt_A_010681.pt), judged
against pilot v1 as sections 3 and 3a of docs/proposals/JEPA_EXPLORATION_DRAFT.md fix it: the NO-GO branch of
reports/bench/jepa_smoke_f2l2/README.md (decisions.md, 2026-10-06 17:05). Label of every output: LABEL.

Order of a run (run), each step before the next reads anything:
1. Hash of section 3a (scripts/jp0_read.py check_addendum): SHA-256 6083e31e... of the section body, equal to the
   first word of reports/bench/jepa_jp0_addendum.sha256; otherwise EXIT_HASH (2) before any other file is read.
2. <out>/results.json must not exist (EXIT_EXISTS, 5): the reading is never rewritten.
3. Margins, from v1's and seed 2's files only (margins): the lines of section 3 at the B end and at ckpt_B_004000
   (scripts/e1_go_reading.py values_at, seed_terms and line_block, with v1's ends as the reference: S and V from the
   diag base row at the B end and from the curve folder at ckpt_B_004000, the IC lines from g3_dev.json, G1a from the
   full G1 run and the --limit-takes 2 run; the S and V seed terms from reports/gpu_queue/branch_seeds_20261002_211426
   .json, the IC and G1a seed terms from v1 against seed 2). Each margin is formatted as section 3 prints it and must
   appear in its own part of that section's text (whitespace normalized), check_printed: the B-end strings before the
   'Early check,' bullet, the ckpt_B_004000 strings inside it (up to 'G10 caution.'); J0's section-3 numbers (D, P1,
   P2, dP, the margin, CARRY or BOTTLENECK) are recomputed (e1_go_reading.j0_section3) and must equal
   reports/bench/jp0_reading/results.json. A departure is a harness defect: EXIT_HARNESS (4), nothing written. A
   missing v1 or seed-2 input: EXIT_MISSING (3).
4. Section 3a: the J0 reading in the pass unit is read from reports/bench/jp0_reading/results.json (its addendum
   SHA-256 asserted equal to the constant): GAIN, D, R, LM, SL, the route and the reading that section 3a routes by
   (sub_readings.main_reading: R under GAIN, D otherwise). The pass-unit lines of v1's and seed 2's B ends are
   recomputed with scripts/jp0_read.py analyse_checkpoint (J0's probes refitted, its refit check) and must equal
   that file within PASS_TOL_CM (EXIT_HARNESS otherwise); the 3a margins are formed from them: m_D and m_R of
   relations D and R (Y1 = v1's decoded or recalibrated S_pass(12), d = the larger seed difference of the two rows),
   m_P of E1's latent probe against v1's (Y1 = v1's latent S_pass(12), d = |v1 minus seed 2| of the latent row), and
   the JP2 go line at ckpt_B_004000 when v1's and seed 2's J0 outputs there exist (reported).
5. E1's inputs (the names of the post-train templates in cache/runs/queue/post_train/ and of the job files of groups
   post_pilot_e1_jepa and J0; <B> is <run folder>_<stem> of the run's last phase B verify event, else
   pilot_e1_jepa_ckpt_B_032044):
   required: cache/runs/pilot_e1_jepa/log.jsonl; reports/gates/g3/diag_<B>/diag.json and events.csv (post_diag);
   reports/gates/g3/dev_<B>/g3_dev.json and reports/gates/g1/dev_<B>/g1.json (post_j1);
   reports/bench/frozen_tokens_<B>/results.json and per_pass.csv (job j0_frozen_tokens_e1_B_032044);
   reported when present (the early check): reports/gates/g3/curve_pilot_e1_jepa_ckpt_B_004000/ (post_curve),
   reports/gates/g1/dev_pilot_e1_jepa_ckpt_B_004000_lt2/ (job pt_g1_lt2_pilot_e1_jepa_B_004000), reports/bench/
   frozen_tokens_pilot_e1_jepa_ckpt_B_004000/ (job j0_frozen_tokens_e1_B_004000).
   A required input missing: EXIT_MISSING (3), the missing paths named, NOT EVALUATED, nothing written.
6. R0 (r0_reading), from the run's log.jsonl and reports/gpu_queue/job_log.csv, the four conditions of the branch
   rule (scripts/weekend_branch.py r0_integrity) and nothing else: the phase B verify event at step 32,044, no alarm,
   no report mismatch (every start event and the verify event), exit 0 of the call that wrote the verify event (its
   job_log.csv row, matched by its --resume argument; inferred 0 from the verify event without an alarm when that
   row does not exist yet, as the branch rule does). Beside R0, not a condition of it, the resume lineage (lineage):
   the first call starts from v1's phase A end (one init_from event, its path and SHA-256), every later call resumes
   from a checkpoint that an earlier call of this run logged at the same step, never beyond the last step reached,
   every start event carries E1's config SHA-256, the steps rise inside a call and continue after its resume point,
   and the verify event names a checkpoint written by the last call with the SHA-256 that call logged. An
   inconsistent lineage is an integrity question for the owner: EXIT_HARNESS (4), its problems named, nothing
   written. The lineage is reported call by call, with the record of decisions.md 2026-10-07 21:00 (two resumes, the
   last from ckpt_B_024000) beside it (agreement reported, not a condition). The log is read line by line
   (parse_log): a line that does not parse whole is searched for a complete record that ends it (a record appended
   after NUL padding or after a fragment that a crash left without its newline); the record is kept, the fragment is
   counted as unparsed, and the line numbers of both are reported, as the branch rule reports its unparsed count.
7. Section 3 at the B end against v1: the success line S(12) (point at most D minus the margin and the paired person
   bootstrap of v1 minus E1 over the 12 persons, 2,000 resamples, seed 20260929, 95 percent percentile, with its
   lower bound above 0, per-person values from the base rows of the two diag.json files) and the six non-inferiority
   lines; the verdict MET when R0 holds and every line holds, NOT MET otherwise with the failed items named. The early
   check at ckpt_B_004000 is reported only (early_check): its reports are checked against the SHA-256 that E1's log
   gives the checkpoint ckpt_B_004000, and a failure there (a missing file, a report of another checkpoint, a person
   guard) is reported as not evaluated with its reason, never a stop.
8. Section 3, predictions, in both units: in the per-stride unit (section 3) with J0's per-stride reading and the band
   max(0.10 P1, 2 dP) on E1's per-stride probe line; in the pass unit (section 3a, which routes) with the reading
   section 3a routes by and the margin rule (m_P, paired lower bound above 0) on E1's latent S_pass(12). Readings D
   and R on E1's frozen-token probes at its B end, with the pre-fixed margins m_D and m_R (one end: CARRY or
   BOTTLENECK; R pending without an F1 stride gain for E1's checkpoint, never fitted here). E1's stem row and its
   chain are reported. The G10 caution: E1's J0 leakage number against v1's (5 points). The route (route_of): the
   pass unit's prediction outcome; when it makes E1 a candidate for the main recipe, E1 is one only when the verdict
   is MET and the G10 caution is evaluated and does not apply, otherwise the route names what blocks adoption (the
   failed items, R0 among them, and the G10 caution); when the JP0 reading is GAIN, routing rule 1 of section 3a is
   stated first (F2's scale sources come first, reading R routes what remains once the gain is explained). The JP2
   go line at ckpt_B_004000 is reported; a failure of its read is reported as not evaluated.
9. Beside, never substituted: the matched-settings seed differences of scripts/seed_difference.py (reports/bench/
   seed_difference/results.json, the rows of document JEPA) when that file exists, with E1's status under each
   threshold; the supervised baseline per stride on the 12.
Writes <out> (default reports/bench/e1_reading/): results.json, summary.md, _provenance.json (MoveAtlas stamp, the
section 3a hash, the SHA-256 of every input). CPU only (CUDA_VISIBLE_DEVICES=-1); no lockbox, EVAL, CODY or care_pd
record; no gain fitted.

--margins: steps 1, 3 and 4 only (v1's and seed 2's files, the proposal, the JP0 reading); prints the margins, opens
no E1 file and writes nothing.

Run (mdfm environment, from the tree root):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/e1_read.py [--margins] [--out DIR]
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the reading never sees the GPU

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
LABEL = "dev diagnostic, not a G3 read; reading of E1 against pilot v1 (JEPA proposal sections 3 and 3a)"
TASK = "E1R"
ADDENDUM_SHA256 = "6083e31e7d765797acd5062b70b622e46d45fe09521b260b2e5d17c107567c94"
PROPOSAL_REL = "docs/proposals/JEPA_EXPLORATION_DRAFT.md"
JP0_RESULTS_REL = "reports/bench/jp0_reading/results.json"
F1_RESULTS_REL = "reports/bench/g3_gain_check/results.json"
SEEDS_REL = "reports/gpu_queue/branch_seeds_20261002_211426.json"
SUPERVISED_REL = "reports/bench/g3_supervised_baseline/results.json"
SEED_DIFF_REL = "reports/bench/seed_difference/results.json"
JOB_LOG_REL = "reports/gpu_queue/job_log.csv"
OUT_REL = "reports/bench/e1_reading"

E1_RUN = "pilot_e1_jepa"
E1_RUN_DIR = f"cache/runs/{E1_RUN}"
E1_CONFIG_SHA256 = "ca9d1dd4b8f53cdd216bfbd2d5f3dc5c53bbb530cbc7f902d244da219fc6e166"   # job note, E1P check 4
SOURCE_PATH = "cache/runs/phaseA_root_full/ckpt_A_010681.pt"
SOURCE_SHA256 = "ad224565725e6ec973d9b4b2af0e67d2c658241d8644b7d879a8730d70fb5c10"
PHASE_B_STEPS = 32044
E1_B_DEFAULT = f"{E1_RUN}_ckpt_B_{PHASE_B_STEPS:06d}"
E1_4000 = f"{E1_RUN}_ckpt_B_004000"
# decisions.md 2026-10-07 21:00 and the job_log.csv row of e1_jepa_train: two host-memory crashes, the runner's
# automatic resume, then e1_jepa_train_r3 from ckpt_B_024000. Reported beside the lineage, not a condition.
RECORD = {"resume_events": 2, "last_resume": f"{E1_RUN}/ckpt_B_024000.pt",
          "source": "reports/gpu_queue/decisions.md, 2026-10-07 21:00"}
NAMES = {"v1_B": "phaseA_root_full_ckpt_B_032044", "s2_B": "pilot_v1_s2_ckpt_B_032044",
         "v1_4000": "phaseA_root_full_ckpt_B_004000", "s2_4000": "pilot_v1_s2_ckpt_B_004000"}
LINES = ("S12", "V12", "S14", "V14", "IC", "IC_insole", "G1a")
NONINFERIORITY = LINES[1:]
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "IC": "ms", "IC_insole": "ms", "G1a": "mm"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "IC": 2, "IC_insole": 2, "G1a": 2}
IC_LINE = {"IC": "ic_median_ms_walkway", "IC_insole": "ic_median_ms_insole"}
CHAIN = ("stem", "latent", "mu_z", "decoded", "recalibrated")
TARGET_FRACTION, SEED_MULTIPLE, G10_POINTS = 0.10, 2.0, 5.0
EARLY_MARK, G10_MARK = "Early check,", "G10 caution."      # the bullet of section 3 that holds the early numbers
TOL = 1e-9
PASS_TOL_CM = 1e-6                  # v1's and seed 2's S_pass(12) recomputed against the JP0 reading (same code)
CARRY, BOTTLENECK, SPLIT, PENDING = "CARRY", "BOTTLENECK", "SPLIT", "pending"
MET, NOT_MET = "MET", "NOT MET"
EXIT_WRITTEN, EXIT_HASH, EXIT_MISSING, EXIT_HARNESS, EXIT_EXISTS = 0, 2, 3, 4, 5
# section 3 of the proposal: the strings that hold the margins (formatted from the recomputed values)
SEED_DIFF_KEYS = {"S12": "S12", "V12": "V12", "S14": "S14", "V14": "V14", "IC": "IC", "ICins": "IC_insole",
                  "G1a": "G1a"}


class Stop(Exception):
    """A stop of the run with its exit code (main prints the message and returns the code)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = int(code)
        self.message = message


# ---------------------------------------------------------------- helpers
def _sibling(name):
    key = f"_e1r_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def JP():
    """scripts/jp0_read.py: the section 3a hash, the pass unit, the paired bootstrap of section 3a."""
    return _sibling("jp0_read")


def GO():
    """scripts/e1_go_reading.py: the lines of section 3 with a reference end, J0 at a reference, the 3a context."""
    return _sibling("e1_go_reading")


def F2():
    return GO().F2()


def WB():
    return GO().WB()


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
    if path is None or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(path):
    """'<run folder>/<file>' of a checkpoint path, whatever its separators or its absolute prefix."""
    parts = [p for p in str(path or "").replace("\\", "/").split("/") if p]
    return "/".join(parts[-2:])


def name_of(path):
    """'<run folder>_<stem>' of a checkpoint path (queue_job.checkpoint_name)."""
    c = canon(path)
    run, _, f = c.partition("/")
    return f"{run}_{os.path.splitext(f)[0]}"


def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def stop_from(e, what):
    """A Stop of this script from a stop of a sibling script (anything with .message): EXIT_MISSING when it names a
    missing input, EXIT_HARNESS otherwise."""
    msg = getattr(e, "message", None) or str(e)
    code = EXIT_MISSING if "missing" in msg.lower() or "does not exist" in msg.lower() else EXIT_HARNESS
    return Stop(code, f"{what}: {msg}")


# ---------------------------------------------------------------- 1. the hash of section 3a
def check_addendum(tree=None):
    tree = tree or TREE
    jp = JP()
    try:
        return jp.check_addendum(tree, ADDENDUM_SHA256)
    except jp.Stop as s:
        raise Stop(EXIT_HASH, s.message)


# ---------------------------------------------------------------- 3. margins from v1 and seed 2
def default_sets(tree):
    """(sets {dev14, dev12} of F1's results.json, asserted as the F2 lever-2 rule quotes them; the lockbox list)."""
    from mdfm.gates import g3
    locked = g3.lockbox_persons()
    doc = read_json(os.path.join(tree, F1_RESULTS_REL))
    if doc is None:
        raise Stop(EXIT_MISSING, f"NOT EVALUATED: missing {F1_RESULTS_REL}")
    try:
        return F2().load_sets(doc, locked), locked
    except AssertionError as e:
        raise Stop(EXIT_HARNESS, f"dev sets: {e}")


def margins(tree, sets, locked):
    """The lines of section 3 from v1's and seed 2's files: {'values', 'seed_terms', 'b_end', 'early'}. Stop
    (EXIT_MISSING) when an input of a printed line is missing; Stop(EXIT_HARNESS) on a persons guard."""
    go = GO()
    vals, why = {}, {}
    try:
        for k, step in (("v1_B", "B"), ("s2_B", "B"), ("v1_4000", "4000"), ("s2_4000", "4000")):
            vals[k], why[k] = go.values_at(tree, NAMES[k], step, sets, locked)
    except AssertionError as e:
        raise Stop(EXIT_HARNESS, f"v1 or seed 2 rows: {e}")
    sd_B, why_B = go.seed_terms(tree, "B", vals["v1_B"], vals["s2_B"])
    sd_4, why_4 = go.seed_terms(tree, "4000", vals["v1_4000"], vals["s2_4000"])
    missing = [f"{k} {line}: {r}" for k in vals for line, r in why[k].items()]
    missing += [f"seed term {k} (B end): {r}" for k, r in why_B.items()]
    missing += [f"seed term {k} (ckpt_B_004000): {r}" for k, r in why_4.items()]
    if len(vals["v1_B"]["S12_per_person"]) < 2:
        missing.append("per-person S(12) of v1's B end (diag base row)")
    if missing:
        raise Stop(EXIT_MISSING, "NOT EVALUATED, nothing written; v1 or seed-2 inputs of the margins missing: "
                   + "; ".join(missing))
    return {"values": vals, "seed_terms": {"B_end": sd_B, "ckpt_B_004000": sd_4},
            "b_end": go.line_block(vals["v1_B"], sd_B), "early": go.line_block(vals["v1_4000"], sd_4)}


def section_3_text(tree):
    """Section 3 of the proposal (from its heading to the heading of section 3a), whitespace normalized."""
    with open(os.path.join(tree, PROPOSAL_REL), encoding="utf-8") as f:
        t = f.read()
    try:
        body = t[t.index("\n## 3. Pre-registered"):t.index("\n## 3a. ")]
    except ValueError:
        raise Stop(EXIT_HARNESS, f"{PROPOSAL_REL}: section 3 not found")
    return " ".join(body.split())


def section_3_parts(text):
    """{'b_end': section 3 before its 'Early check,' bullet, 'early': that bullet up to 'G10 caution.'}."""
    try:
        i = text.index(EARLY_MARK)
        j = text.index(G10_MARK, i)
    except ValueError:
        raise Stop(EXIT_HARNESS, f"{PROPOSAL_REL}: section 3 has no '{EARLY_MARK}' bullet followed by '{G10_MARK}'")
    return {"b_end": text[:i], "early": text[i:j]}


def printed_strings(m):
    """The margins of section 3 formatted as the proposal prints them (computed, then looked up in its text):
    {'b_end': strings of the B-end reading, 'early': strings of the 'Early check' bullet}."""
    b, e, v = m["b_end"], m["early"], m["values"]
    s, s4 = b["S12"], e["S12"]
    b_end = [f"D: the decoded S(12) of v1's B end, {s['line']:.2f} cm",
             f"max({s['ten_percent']:.2f}, {s['two_seed']:.2f}) = {s['margin']:.2f} cm, so E1's S(12) at most "
             f"{s['at_most']:.2f} cm",
             f"V(12) at most {b['V12']['at_most']:.4f} m/s", f"S(14) at most {b['S14']['at_most']:.2f} cm",
             f"V(14) at most {b['V14']['at_most']:.3f} m/s",
             f"the walkway IC median at most {b['IC']['at_most']:.2f} ms",
             f"the insole IC median at most {b['IC_insole']['at_most']:.2f} ms (v1 {v['v1_B']['IC_insole']:.2f}, "
             f"seed 2 {v['s2_B']['IC_insole']:.2f} ms: 2 x the seed difference, {b['IC_insole']['two_seed']:.2f} ms, "
             "is the margin)",
             f"G1a at most {b['G1a']['at_most']:.1f} mm",
             f"G1a {v['s2_B']['G1a']:.2f} mm against v1's {v['v1_B']['G1a']:.2f} mm, so 2 x the seed difference, "
             f"{b['G1a']['two_seed']:.2f} mm, stays under 5 percent, {b['G1a']['five_percent']:.2f} mm"]
    early = [f"S(12) at most {s4['at_most']:.2f} cm (v1 {s4['line']:.2f}, margin max({s4['ten_percent']:.2f}, "
             f"{s4['two_seed']:.2f}))",
             f"V(12) at most {e['V12']['at_most']:.4f} m/s", f"S(14) at most {e['S14']['at_most']:.2f} cm",
             f"V(14) at most {e['V14']['at_most']:.4f} m/s",
             f"walkway IC median at most {e['IC']['at_most']:.2f} ms",
             f"insole IC median at most {e['IC_insole']['at_most']:.2f} ms (v1 {v['v1_4000']['IC_insole']:.2f}, "
             f"seed 2 {v['s2_4000']['IC_insole']:.2f} ms)",
             f"at most {e['G1a']['at_most']:.1f} mm (seed 2 {v['s2_4000']['G1a']:.2f} against v1's "
             f"{v['v1_4000']['G1a']:.2f} mm: 2 x the seed difference, {e['G1a']['two_seed']:.2f} mm, under 5 "
             f"percent, {e['G1a']['five_percent']:.2f} mm)"]
    return {"b_end": b_end, "early": early}


def check_printed(m, text):
    """Every margin string of printed_strings found in its own part of section 3 (section_3_parts: a B-end string
    before the 'Early check,' bullet, an early string inside it); Stop(EXIT_HARNESS) naming the others. Returns the
    strings found, B end first."""
    want, parts = printed_strings(m), section_3_parts(text)
    absent = [f"{x} (in the {'B-end reading' if k == 'b_end' else 'Early check bullet'})"
              for k in ("b_end", "early") for x in want[k] if x not in parts[k]]
    if absent:
        raise Stop(EXIT_HARNESS, "harness defect: margins recomputed from v1's and seed 2's files differ from the "
                                 f"numbers section 3 of {PROPOSAL_REL} prints: " + " | ".join(absent))
    return want["b_end"] + want["early"]


def check_section3_j0(tree, D, jp0doc):
    """J0's section-3 numbers recomputed from v1's and seed 2's J0 outputs (e1_go_reading.j0_section3 with v1's B
    end as the reference) and checked equal to the JP0 reading's section3 block."""
    go = GO()
    names = {"ref_B": NAMES["v1_B"], "v1_B": NAMES["v1_B"], "s2_B": NAMES["s2_B"]}
    j0, why = go.j0_section3(tree, names, D)
    if j0 is None:
        raise Stop(EXIT_MISSING, "NOT EVALUATED, nothing written; J0 of v1 or seed 2: " + "; ".join(
            f"{k}: {v}" for k, v in why.items()))
    s3 = (jp0doc or {}).get("section3") or {}
    bad = [k for k in ("D", "P1", "dP", "margin", "D_minus_P1", "reading") if s3.get(k) != j0[k]]
    if s3.get("P2") != j0["P2_s2"]:
        bad.append("P2")
    if bad:
        raise Stop(EXIT_HARNESS, f"harness defect: J0's section-3 numbers recomputed differ from {JP0_RESULTS_REL} "
                                 f"on {bad}")
    return j0


def jp0_reading(tree):
    """The J0 reading in the pass unit (scripts/jp0_read.py output), its addendum hash asserted."""
    p = os.path.join(tree, JP0_RESULTS_REL)
    doc = read_json(p)
    if doc is None:
        raise Stop(EXIT_MISSING, f"NOT EVALUATED, nothing written; missing {JP0_RESULTS_REL}")
    if (doc.get("addendum") or {}).get("sha256") != ADDENDUM_SHA256:
        raise Stop(EXIT_HARNESS, f"{JP0_RESULTS_REL} was not read under section 3a of SHA-256 {ADDENDUM_SHA256}")
    sub = doc.get("sub_readings") or {}
    return {"path": JP0_RESULTS_REL, "sha256": sha256_file(p), "readings": doc.get("readings"),
            "route": doc.get("route"), "main_reading": sub.get("main_reading"), "sub_readings": sub,
            "relations": {k: {x: (doc.get("relations") or {}).get(k, {}).get(x) for x in ("relation", "m", "d", "Y1")}
                          for k in ("D", "R", "LM", "SL")},
            "section3_reading": (doc.get("section3") or {}).get("reading"), "doc": doc}


# ---------------------------------------------------------------- the pass unit
def summarize_analysis(a):
    """The rows of jp0_read.analyse_checkpoint used here: S_pass(12) lines of the chain."""
    lines = a["stride"]["lines"]["dev12"]
    rows = {r: {k: lines[r].get(k) for k in ("point", "interval", "n_persons", "per_person")}
            for r in CHAIN if r in lines}
    return {"rows": rows, "checkpoint_sha256": (a.get("checkpoint") or {}).get("sha256"),
            "refit_check": (a.get("refit") or {}).get("check"), "pass_set": a["stride"]["pass_set"]["dev12"],
            "has_decoded": a.get("has_decoded"), "has_recalibrated": a.get("has_recalibrated")}


class PassReader:
    """Default pass reader: jp0_read.analyse_checkpoint on the J0 outputs of a checkpoint under a tree, with the
    context of e1_go_reading.jp0_context and F1's gains of the main tree, built once."""

    def __init__(self, tree):
        self.tree = tree
        self.ctx = None
        self.f1 = None

    def __call__(self, name, where):
        jp, go = JP(), GO()
        try:
            if self.f1 is None:
                self.f1 = jp.read_f1(self.tree)
            if self.ctx is None:
                self.ctx = go.jp0_context(self.tree)
            return summarize_analysis(jp.analyse_checkpoint(where, name, name, self.ctx, self.f1))
        except (jp.Stop, go.Stop) as s:
            raise stop_from(s, f"pass unit, {name}")


def pass_margins(v1, s2, jp0doc):
    """m_D, m_R (section 3a, margin rule, one margin for both ends) and m_P, m_stem (E1's row against v1's, d the
    v1 to seed-2 difference of that row), each {'Y1', 'd', 'm'}; m_D and m_R checked against the JP0 reading."""
    def pt(x, r):
        return fin(((x["rows"].get(r)) or {}).get("point"))

    out = {}
    for key, y in (("D", "decoded"), ("R", "recalibrated")):
        X1, X2, Y1, Y2 = pt(v1, "latent"), pt(s2, "latent"), pt(v1, y), pt(s2, y)
        if None in (X1, X2, Y1, Y2):
            out[key] = None
            continue
        d = max(abs(X1 - X2), abs(Y1 - Y2))
        out[key] = {"Y1": Y1, "d": d, "m": max(TARGET_FRACTION * Y1, SEED_MULTIPLE * d), "x": "latent", "y": y}
    for key, r in (("P", "latent"), ("stem", "stem")):
        Y1, Y2 = pt(v1, r), pt(s2, r)
        out[key] = None if None in (Y1, Y2) else {"Y1": Y1, "d": abs(Y1 - Y2), "x": f"E1 {r}", "y": f"v1 {r}",
                                                  "m": max(TARGET_FRACTION * Y1, SEED_MULTIPLE * abs(Y1 - Y2))}
    rels = (jp0doc or {}).get("relations") or {}
    bad = []
    for key in ("D", "R"):
        mine, theirs = out.get(key), rels.get(key) or {}
        if mine is None:
            if fin(theirs.get("m")) is not None:
                bad.append(f"{key}: not formed here, {theirs.get('m')} in the JP0 reading")
            continue
        for k in ("Y1", "d", "m"):
            t = fin(theirs.get(k))
            if t is None or abs(mine[k] - t) > PASS_TOL_CM:
                bad.append(f"{key} {k}: {mine[k]} here, {theirs.get(k)} in the JP0 reading")
    if out.get("P") is None:
        bad.append("P: v1's or seed 2's latent S_pass(12) is not finite")
    if bad:
        raise Stop(EXIT_HARNESS, "harness defect: section 3a margins recomputed differ: " + "; ".join(bad))
    return out


def check_pass_lines(key, summ, jp0doc):
    """v1's or seed 2's S_pass(12) rows recomputed here equal those of the JP0 reading within PASS_TOL_CM."""
    theirs = (((jp0doc.get("checkpoints") or {}).get(key) or {}).get("stride") or {}).get("lines", {}).get("dev12", {})
    bad = []
    for r, line in summ["rows"].items():
        t = fin((theirs.get(r) or {}).get("point"))
        if t is None or fin(line.get("point")) is None or abs(line["point"] - t) > PASS_TOL_CM:
            bad.append(f"{r} {line.get('point')} against {t}")
    for r in CHAIN:
        if r in theirs and r not in summ["rows"]:
            bad.append(f"{r} absent here")
    if bad:
        raise Stop(EXIT_HARNESS, f"harness defect: S_pass(12) of {key} recomputed differs from {JP0_RESULTS_REL}: "
                                 + "; ".join(bad))


def one_end(x, y, m):
    """X better than Y by the margin at one end (section 3a's relation read at a single end): G = Y minus X, the
    paired person bootstrap of Y minus X (jp0_read.paired); better when G >= m and the lower bound is above 0, worse
    when X minus Y >= m with the lower bound of X minus Y above 0, within when |G| < m."""
    jp = JP()
    if x is None or y is None or fin(x.get("point")) is None or fin(y.get("point")) is None or fin(m) is None:
        return {"evaluated": False}
    G = float(y["point"] - x["point"])
    pr = jp.paired(y.get("per_person") or {}, x.get("per_person") or {})
    pw = jp.paired(x.get("per_person") or {}, y.get("per_person") or {})
    return {"evaluated": True, "X": float(x["point"]), "Y": float(y["point"]), "G": G, "m": float(m), "paired": pr,
            "better": bool(G >= m and pr["lower_above_0"]), "worse": bool(-G >= m and pw["lower_above_0"]),
            "within": bool(abs(G) < m)}


def e1_carry(rel_):
    if not rel_ or not rel_.get("evaluated"):
        return PENDING
    return CARRY if rel_["better"] else BOTTLENECK


# ---------------------------------------------------------------- R0 and the resume lineage
def parse_log(path):
    """(records, info) of a log.jsonl, read line by line; (None, info) when absent. A line that does not parse whole,
    once its NUL padding is removed, is searched for a complete record that ends it (the leftmost '{' from which the
    rest of the line parses as an object with an event): a record appended after a fragment that a crash left
    without its newline. That record is kept and the fragment counted as unparsed. info: {'unparsed': n,
    'unparsed_lines': [line numbers], 'recovered_lines': [line numbers of records read after padding or a
    fragment]}."""
    info = {"unparsed": 0, "unparsed_lines": [], "recovered_lines": []}
    if path is None or not os.path.exists(path):
        return None, info
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            raw = line.strip()
            s = raw.strip("\x00").strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except ValueError:
                r = None
            if isinstance(r, dict):
                recs.append(r)
                if s != raw:
                    info["recovered_lines"].append(n)
                continue
            info["unparsed"] += 1
            info["unparsed_lines"].append(n)
            for i in [k for k, c in enumerate(s) if c == "{"][1:]:
                try:
                    r = json.loads(s[i:])
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("event"):
                    recs.append(r)
                    info["recovered_lines"].append(n)
                break
    return recs, info


def read_log(e1_tree):
    """(records, info) of E1's log.jsonl (parse_log); (None, info) when absent."""
    return parse_log(os.path.join(e1_tree, E1_RUN_DIR, "log.jsonl"))


def segments(recs):
    """The calls of a run: the records from one 'start' event to the next. Returns (calls, records before any
    start)."""
    calls, pre = [], []
    for r in recs or []:
        if r.get("event") == "start":
            calls.append([r])
        elif calls:
            calls[-1].append(r)
        else:
            pre.append(r)
    return calls, pre


def lineage(recs, run=E1_RUN, config_sha256=E1_CONFIG_SHA256, steps=PHASE_B_STEPS):
    """The resume lineage of a phase-B-only run, call by call, and its consistency (module docstring, step 6)."""
    calls, pre = segments(recs)
    problems, rows, seen = [], [], []          # seen: (call index, canon path, step, sha256) of checkpoint events
    if pre:
        problems.append(f"{len(pre)} records before the first start event")
    if not calls:
        problems.append("no start event")
    prev_last = None
    for i, call in enumerate(calls):
        start = call[0]
        ev = [r.get("event") for r in call]
        inits = [r for r in call if r.get("event") == "init_from"]
        resumes = [r for r in call if r.get("event") == "resume"]
        logs = [r for r in call if r.get("event") == "log" and fin(r.get("step")) is not None]
        cks = [r for r in call if r.get("event") == "checkpoint"]
        trained = bool(logs or cks)
        row = {"call": i + 1, "time": next((r.get("time") for r in call if r.get("time")), None),
               "config_sha256": start.get("config_sha256"), "how": None, "from": None, "from_step": None,
               "first_log_step": int(logs[0]["step"]) if logs else None,
               "last_step": max([int(r["step"]) for r in logs + cks if fin(r.get("step")) is not None], default=None),
               "checkpoints": [int(r["step"]) for r in cks if fin(r.get("step")) is not None],
               "verify": None, "end_event": "end" in ev, "alarms": ev.count("alarm"),
               "nonfinite": ev.count("nonfinite")}
        if start.get("config_sha256") != config_sha256:
            problems.append(f"call {i + 1}: start event at config {start.get('config_sha256')}, expected "
                            f"{config_sha256}")
        if i == 0:
            row["how"] = "init_from"
            if len(inits) != 1 or resumes:
                problems.append(f"call 1: {len(inits)} init_from and {len(resumes)} resume events (one init_from "
                                "expected)")
            for r in inits[:1]:
                src = r.get("source") or {}
                row["from"] = str(src.get("path") or "").replace("\\", "/")
                row["from_step"] = r.get("step")
                if not row["from"].endswith(SOURCE_PATH) or src.get("sha256") != SOURCE_SHA256:
                    problems.append(f"call 1: init_from {row['from']} ({src.get('sha256')}), expected {SOURCE_PATH} "
                                    f"({SOURCE_SHA256})")
                if r.get("phase") != "B" or r.get("step") != 0:
                    problems.append(f"call 1: init_from at phase {r.get('phase')} step {r.get('step')}")
        else:
            if inits:
                problems.append(f"call {i + 1}: an init_from event in a resumed call")
            if not resumes:
                row["how"] = "no resume event"
                if trained:
                    problems.append(f"call {i + 1}: trains without a resume event")
            elif len(resumes) > 1:
                row["how"] = "resume"
                problems.append(f"call {i + 1}: {len(resumes)} resume events")
            if resumes:
                r = resumes[0]
                c, s = canon(r.get("path")), r.get("step")
                row.update(how="resume", **{"from": c, "from_step": s})
                if not c.startswith(f"{run}/"):
                    problems.append(f"call {i + 1}: resumes {c}, not a checkpoint of {run}")
                if r.get("phase") != "B":
                    problems.append(f"call {i + 1}: resume at phase {r.get('phase')}")
                if "config_sha256" in (r.get("drift") or []):
                    problems.append(f"call {i + 1}: the resumed checkpoint has another config")
                earlier = [x for x in seen if x[1] == c]
                if not earlier:
                    problems.append(f"call {i + 1}: resumes {c}, which no earlier call of the run logged")
                elif earlier[-1][2] != s:
                    problems.append(f"call {i + 1}: resumes {c} at step {s}, logged at step {earlier[-1][2]}")
                else:
                    row["from_call"] = earlier[-1][0]
                    row["from_sha256"] = earlier[-1][3]
                if prev_last is not None and fin(s) is not None and s > prev_last:
                    problems.append(f"call {i + 1}: resumes at step {s}, beyond the last step reached, {prev_last}")
                row["steps_trained_again"] = None if prev_last is None or fin(s) is None else int(prev_last - s)
                if logs and fin(s) is not None and logs[0]["step"] <= s:
                    problems.append(f"call {i + 1}: first log step {logs[0]['step']} not after the resume step {s}")
        st = [int(r["step"]) for r in logs]
        if any(b <= a for a, b in zip(st[:-1], st[1:])):
            problems.append(f"call {i + 1}: log steps do not rise")
        for r in cks:
            seen.append((i + 1, canon(r.get("path")), r.get("step"), r.get("sha256")))
        vb = [r for r in call if r.get("event") == "verify" and r.get("phase") == "B"]
        if vb:
            v = vb[-1]
            row["verify"] = {"step": v.get("step"), "path": canon(v.get("path")), "sha256": v.get("sha256")}
        if row["last_step"] is not None:
            prev_last = row["last_step"]
        rows.append(row)
    verify = WB().b_end_verify(recs or [])
    final = None
    if verify is not None:
        c = canon(verify.get("path"))
        last_call = len(calls)
        same = [x for x in seen if x[1] == c and x[0] == last_call]
        final = {"path": c, "step": verify.get("step"), "sha256": verify.get("sha256"), "call": last_call}
        if c != f"{run}/ckpt_B_{steps:06d}.pt":
            problems.append(f"the verify event names {c}")
        if not same or same[-1][3] != verify.get("sha256"):
            problems.append(f"the verify event's checkpoint {c} ({verify.get('sha256')}) is not the one the last "
                            "call logged")
        if not any(r.get("event") == "verify" and r.get("phase") == "B" for r in calls[-1]):
            problems.append("the phase B verify event is not in the last call")
    resumes = [r for r in rows if r["how"] == "resume"]
    record = {"resume_events": len(resumes), "last_resume": resumes[-1]["from"] if resumes else None}
    agrees = record["resume_events"] == RECORD["resume_events"] and record["last_resume"] == RECORD["last_resume"]
    return {"calls": rows, "consistent": not problems, "problems": problems, "final_verify": final,
            "resumes": [r["from"] for r in resumes], "record": dict(RECORD), "log_says": record,
            "agrees_with_record": agrees}


def job_rows(job_log, run=E1_RUN):
    """The job_log.csv rows of scripts/train_stage1.py jobs on the run folder: [{name, state, exit_code, resume}]."""
    wb = WB()
    out = []
    if job_log is None or not os.path.exists(job_log):
        return out
    with open(job_log, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            argv = wb.argv_list(r.get("argv", ""))
            if not any("train_stage1" in x for x in argv) or wb.job_run_folder(argv) != run:
                continue
            res = next((argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "--resume"), None)
            code = r.get("exit_code")
            try:
                code = int(float(code)) if code not in (None, "") else None
            except ValueError:
                pass
            out.append({"name": r.get("name"), "state": r.get("state"), "exit_code": code,
                        "resume": canon(res) if res else None, "start": r.get("start"), "end": r.get("end")})
    return out


def r0_reading(recs, unparsed, lin, rows):
    """R0 of the branch rule for E1 (section 3), its four conditions only: verify event at step 32,044, no report
    mismatch, no alarm, exit 0 of the call that wrote the verify event. unparsed: the count of unparsed log lines or
    parse_log's info; reported, not a condition (as the branch rule reports it). The lineage is used only to find the
    job_log.csv row of the final call."""
    wb = WB()
    info = unparsed if isinstance(unparsed, dict) else {"unparsed": int(unparsed or 0), "unparsed_lines": [],
                                                          "recovered_lines": []}
    v = wb.b_end_verify(recs or [])
    starts = wb.log_events(recs, "start")
    alarms = wb.log_events(recs, "alarm")
    mism = [s.get("report_mismatches") for s in starts if s.get("report_mismatches")]
    if v is not None and v.get("report_mismatches") != []:
        mism.append(v.get("report_mismatches", "absent from the verify event"))
    last_resume = lin["resumes"][-1] if lin["resumes"] else None
    final_rows = [r for r in rows if r["resume"] == last_resume]
    if final_rows:
        code = final_rows[-1]["exit_code"]
        how = f"job_log.csv row {final_rows[-1]['name']} (state {final_rows[-1]['state']})"
    elif v is not None and not alarms:
        code, how = 0, ("inferred: train_stage1.py exits non-zero after its verify event only on an alarm (no "
                        "job_log.csv row of the final call yet)")
    else:
        code, how = None, "no job_log.csv row of the final call and no verify event"
    checks = {"verify_B_32044": bool(v is not None and v.get("phase") == "B" and v.get("step") == PHASE_B_STEPS),
              "report_mismatches_empty": v is not None and not mism, "no_alarm": not alarms, "exit_0": code == 0}
    return {"holds": all(checks.values()), "checks": checks,
            "failed": [k for k, x in checks.items() if not x],
            "verify": None if v is None else {"step": v.get("step"), "path": canon(v.get("path")),
                                              "sha256": v.get("sha256")},
            "report_mismatches": mism, "alarm_events": len(alarms),
            "nonfinite_events": len(wb.log_events(recs, "nonfinite")), "unparsed_log_lines": info["unparsed"],
            "unparsed_line_numbers": info["unparsed_lines"], "recovered_line_numbers": info["recovered_lines"],
            "exit_code": code, "exit_from": how, "job_rows": rows}


def lineage_stop(lin, info):
    """Stop(EXIT_HARNESS) when the resume lineage is not consistent: an integrity question for the owner, not a
    criterion of section 3, so nothing is written."""
    if lin["consistent"]:
        return
    extra = ""
    if info.get("unparsed_lines") or info.get("recovered_lines"):
        extra = (f"; log lines unparsed {info.get('unparsed_lines')}, records recovered from lines "
                 f"{info.get('recovered_lines')} (a crash fragment can hide or join a record)")
    raise Stop(EXIT_HARNESS, "integrity question for the owner, nothing written: the resume lineage of "
                             f"{E1_RUN_DIR}/log.jsonl is not consistent: " + "; ".join(lin["problems"]) + extra)


# ---------------------------------------------------------------- E1's values
def e1_values_B(e1_tree, name, sets, locked, sha):
    """{line: value} and {line: reason} of E1's B end: S and V from the diag base row, the IC lines from g3_dev.json,
    G1a from the full G1 run; every report checked to be of the verify event's checkpoint (SHA-256) and EMA."""
    f2 = F2()
    vals, why = {}, {}
    try:
        gset, r = f2.diag_set(e1_tree, name, sets, locked, sha)
    except AssertionError as e:
        raise Stop(EXIT_HARNESS, f"E1 diag rows: {e}")
    wb = WB()
    for k, sub, key in (("S12", wb.SUB12, "S"), ("V12", wb.SUB12, "V"), ("S14", wb.SUB14, "S"),
                        ("V14", wb.SUB14, "V")):
        vals[k] = fin(((gset or {}).get("subsets", {}).get(sub) or {}).get(f"{key}_point"))
        if vals[k] is None:
            why[k] = r or f"no {k} in the diag base row of {name}"
    s12 = ((gset or {}).get("subsets", {}).get(wb.SUB12) or {}).get("S") or {}
    vals["S12_per_person"] = {p: fin(v) for p, v in s12.items() if fin(v) is not None}
    for k, line in IC_LINE.items():
        vals[k], r2 = f2.g3dev_line(e1_tree, name, line, sha)
        if r2:
            why[k] = r2
    g, r3 = f2.g1a_of(e1_tree, name, "full", sha)
    vals["G1a"] = fin((g or {}).get("point"))
    if vals["G1a"] is None:
        why["G1a"] = r3 or f"no G1a in the G1 run of {name}"
    return vals, why


def checkpoint_sha(recs, step, run=E1_RUN):
    """The SHA-256 of the last phase B checkpoint event of the run at `step` (None when the log has none)."""
    want = f"{run}/ckpt_B_{int(step):06d}.pt"
    cks = [r for r in recs or [] if r.get("event") == "checkpoint" and r.get("phase") == "B"
           and canon(r.get("path")) == want]
    return cks[-1].get("sha256") if cks else None


def early_values(e1_tree, sets, locked, sha4):
    """E1's values at ckpt_B_004000 (e1_go_reading.values_at at step '4000') and {line: reason}, the curve folder's
    g3_dev.json and the --limit-takes 2 G1 run checked against the log's SHA-256 of that checkpoint (sha4). Reported
    only: a failure is a reason, never a stop. Returns (values, why, error or None)."""
    go, f2, wb = GO(), F2(), WB()
    try:
        v4, why4 = go.values_at(e1_tree, E1_4000, "4000", sets, locked)
    except (AssertionError, ValueError, KeyError, TypeError, go.Stop) as e:
        return {}, {}, f"E1 curve rows at ckpt_B_004000: {getattr(e, 'message', None) or e}"
    why4 = dict(why4)
    cj = go.curve_json(e1_tree, E1_4000)
    doc = read_json(cj)
    issue = f2.checkpoint_issue(doc, E1_4000, sha4, rel(cj, e1_tree)) if doc is not None else None
    if issue:
        for k in ("S12", "V12", "S14", "V14", "IC", "IC_insole"):
            v4[k] = None
            why4[k] = issue
        v4["S12_per_person"] = {}
    g = wb.g1_reads(e1_tree, E1_4000).get("lt2")
    gdoc = read_json(g["json"]) if g else None
    issue = f2.checkpoint_issue(gdoc, E1_4000, sha4, rel(g["json"], e1_tree)) if gdoc is not None else None
    if issue:
        v4["G1a"] = None
        why4["G1a"] = issue
    if sha4 is None:
        why4["sha256"] = "the log holds no checkpoint event of ckpt_B_004000: its reports were not checked by SHA-256"
    return v4, why4, None


def line_reads(vals, block, v1_pp=None):
    """Each line of a block (section 3) read on E1's values: the success line with its paired bootstrap (v1 minus E1
    per person over the persons both hold) and the non-inferiority lines."""
    jp = JP()
    out = {}
    for k in LINES:
        b, v = block.get(k), fin(vals.get(k))
        if b is None or v is None:
            out[k] = {"evaluated": False, "value": v, "at_most": (b or {}).get("at_most"), "holds": None}
            continue
        r = {"evaluated": True, "value": v, "line_v1": b["line"], "margin": b["margin"], "at_most": b["at_most"],
             "kind": b["kind"], "point_holds": bool(v <= b["at_most"] + TOL)}
        if k == "S12":
            pr = jp.paired(v1_pp or {}, vals.get("S12_per_person") or {})
            r["paired_v1_minus_e1"] = pr
            r["holds"] = bool(r["point_holds"] and pr["lower_above_0"])
        else:
            r["holds"] = r["point_holds"]
        out[k] = r
    return out


# ---------------------------------------------------------------- predictions
NEXT = {
    "carry_holds": "decoding is the next lever (decoder, root integration, contact readout); JEPA as an auxiliary "
                   "objective is dropped",
    "carry_failed_success": "the prediction failed and is reported as failed: the term helped by a route this "
                            "reading does not describe; section 3 fixes no route for this case",
    "carry_probe_moved": "the decoded line missed as predicted, the probe moved beyond its band; section 3 fixes no "
                         "route for this combination; decoding comes next under both readings when the decoded line "
                         "misses",
    "bottleneck_holds": "E1 repaired the representation and the decoded S(12) passes its line through it",
    "bottleneck_both": "both the representation and decoding were limiting: decoding comes next",
    "bottleneck_neither": "JEPA did not repair the representation: the auxiliary objective is dropped",
    "bottleneck_failed_success": "the decoded S(12) passed without the probe improving by its band: the prediction "
                                 "failed, reported as failed; section 3 fixes no route for this case",
    "none": "section 3 fixes no prediction under this reading",
}


def predict(reading, success, improved, within):
    """Section 3's prediction under a J0 reading, given E1's success line (S(12)), whether its probe improved by the
    band or margin, and whether it stayed within it. {'reading', 'predicted', 'holds', 'outcome', 'next'}."""
    if reading == CARRY:
        pred = "E1 misses the success criterion and its probe stays within the band of v1's"
        if success:
            return {"reading": reading, "predicted": pred, "holds": False, "outcome": "E1 passes the criterion",
                    "next": NEXT["carry_failed_success"]}
        if within:
            return {"reading": reading, "predicted": pred, "holds": True, "outcome": "E1 changes little",
                    "next": NEXT["carry_holds"]}
        return {"reading": reading, "predicted": pred, "holds": False,
                "outcome": "E1 misses the criterion, its probe moved beyond the band"
                           + (" (improved)" if improved else ""), "next": NEXT["carry_probe_moved"]}
    if reading == BOTTLENECK:
        pred = "E1 lowers the probe by at least the band and, through it, the decoded S(12) passes the criterion"
        if improved and success:
            return {"reading": reading, "predicted": pred, "holds": True, "outcome": "probe improved, criterion met",
                    "next": NEXT["bottleneck_holds"]}
        if improved:
            return {"reading": reading, "predicted": pred, "holds": False,
                    "outcome": "probe improved, decoded S(12) did not pass", "next": NEXT["bottleneck_both"]}
        if success:
            return {"reading": reading, "predicted": pred, "holds": False,
                    "outcome": "decoded S(12) passed, probe did not improve by the band",
                    "next": NEXT["bottleneck_failed_success"]}
        return {"reading": reading, "predicted": pred, "holds": False, "outcome": "neither improved",
                "next": NEXT["bottleneck_neither"]}
    return {"reading": reading, "predicted": None, "holds": None, "outcome": "no prediction", "next": NEXT["none"]}


CANDIDATE = ("a candidate for the main recipe on the owner's agreement (spec section 1.5 departure); the G10 caution "
             "was evaluated and does not apply")
GAIN_RULE_1 = ("Routing rule 1 of section 3a comes first, since the JP0 reading is GAIN: F2 works on the scale sources "
               "first (shape and leg length under IMU-only input, the decoder's root-velocity path, root integration), "
               "and reading R routes what remains only once the gain is explained")


def route_of(pred3a, verdict, failed, leak, jp0_readings):
    """The route of the reading: the pass unit's prediction outcome (pred3a['next']). Where that outcome makes E1 a
    candidate for the main recipe (BOTTLENECK, prediction held), E1 is one only when the verdict is MET and the G10
    caution was evaluated and does not apply; otherwise the route names what blocks adoption: the failed items (R0
    and the non-inferiority lines among them) and the G10 caution. Under GAIN, routing rule 1 is stated first.
    Returns (text, parts)."""
    blocks = list(failed or [])
    if leak.get("caution"):
        blocks.append(f"the G10 caution applies (E1's leakage {fmt(leak.get('e1'))} against v1's "
                      f"{fmt(leak.get('v1'))} points, a rise of {fmt(leak.get('rise'))}, at least {G10_POINTS:g})")
    elif not leak.get("evaluated"):
        blocks.append("the G10 caution was not evaluated")
    adoptable = pred3a.get("next") == NEXT["bottleneck_holds"]
    candidate = bool(adoptable and verdict == MET and leak.get("evaluated") and leak.get("caution") is False)
    text = pred3a.get("next") or NEXT["none"]
    if adoptable:
        text += (": " + CANDIDATE) if candidate else \
            ("; E1 is not a candidate for the main recipe while " + "; ".join(blocks))
    gain = (jp0_readings or {}).get("GAIN") == "GAIN"
    if gain:
        text = GAIN_RULE_1 + ". Under it: " + text
    return text, {"prediction_next": pred3a.get("next"), "adoption_possible": adoptable, "candidate": candidate,
                  "blocks": blocks, "gain_rule_1": GAIN_RULE_1 if gain else None}


def per_stride_probe(j0, e1_p):
    """Section 3's probe line in the per-stride unit: band max(0.10 P1, 2 dP) around v1's P1."""
    P1, dP = j0["P1"], j0["dP"]
    band = max(TARGET_FRACTION * P1, SEED_MULTIPLE * dP)
    if fin(e1_p) is None:
        return {"evaluated": False, "band": band, "P1": P1}
    diff = float(P1 - e1_p)
    return {"evaluated": True, "P1": P1, "dP": dP, "band": band, "E1": float(e1_p), "P1_minus_E1": diff,
            "improved": bool(diff >= band), "worse": bool(-diff >= band), "within": bool(abs(diff) < band)}


# ---------------------------------------------------------------- beside: the matched seed differences
def seed_difference_beside(path, b_reads, early_reads):
    """The rows of document JEPA of scripts/seed_difference.py's results.json, with E1's status under the threshold
    of each d (mismatched, the proposal's; matched; same code). Reported, never substituted."""
    doc = read_json(path)
    if doc is None:
        return {"present": False, "path": rel(path)}
    rows = []
    for u in doc.get("uses") or []:
        if u.get("document") != "JEPA":
            continue
        lid = str(u.get("line") or "")
        head, _, end = lid.partition("_")
        key = SEED_DIFF_KEYS.get(head)
        reads = b_reads if end == "B" else early_reads
        mine = (reads or {}).get(key) or {}
        v = fin(mine.get("value"))
        row = {"item": u.get("item"), "line": lid, "kind": u.get("kind"), "e1_value": v,
               "threshold_here": mine.get("at_most")}
        for tag in ("mismatched", "matched", "same_code"):
            t = fin(u.get(f"threshold_{tag}"))
            row[f"d_{tag}"] = u.get(f"d_{tag}")
            row[f"threshold_{tag}"] = t
            ok = None if t is None or v is None else bool(v <= t + TOL)
            if ok is not None and key == "S12" and "paired_v1_minus_e1" in mine:
                ok = ok and bool(mine["paired_v1_minus_e1"]["lower_above_0"])
            row[f"e1_holds_{tag}"] = ok
        th = fin(mine.get("at_most"))
        row["mismatched_equals_here"] = None if th is None or row["threshold_mismatched"] is None else \
            bool(abs(th - row["threshold_mismatched"]) <= 1e-9)
        rows.append(row)
    return {"present": True, "path": rel(path), "sha256": sha256_file(path), "complete": doc.get("complete"),
            "label": "beside the reading, never substituted: the margins that section 3 prints stand", "rows": rows,
            "changes": [r["item"] for r in rows if r.get("e1_holds_matched") is not None
                        and r["e1_holds_matched"] != r["e1_holds_mismatched"]]}


# ---------------------------------------------------------------- the run
def required_inputs(e1_tree, name):
    return [os.path.join(e1_tree, E1_RUN_DIR, "log.jsonl"),
            os.path.join(e1_tree, "reports", "gates", "g3", f"diag_{name}", "diag.json"),
            os.path.join(e1_tree, "reports", "gates", "g3", f"diag_{name}", "events.csv"),
            os.path.join(e1_tree, "reports", "gates", "g3", f"dev_{name}", "g3_dev.json"),
            os.path.join(e1_tree, "reports", "gates", "g1", f"dev_{name}", "g1.json"),
            os.path.join(e1_tree, "reports", "bench", f"frozen_tokens_{name}", "results.json"),
            os.path.join(e1_tree, "reports", "bench", f"frozen_tokens_{name}", "per_pass.csv")]


def early_inputs(e1_tree):
    return [os.path.join(e1_tree, "reports", "gates", "g3", f"curve_{E1_4000}", "g3_dev.json"),
            os.path.join(e1_tree, "reports", "gates", "g3", f"curve_{E1_4000}", "per_person.csv"),
            os.path.join(e1_tree, "reports", "gates", "g1", f"dev_{E1_4000}_lt2", "g1.json"),
            os.path.join(e1_tree, "reports", "bench", f"frozen_tokens_{E1_4000}", "results.json"),
            os.path.join(e1_tree, "reports", "bench", f"frozen_tokens_{E1_4000}", "per_pass.csv")]


def j0_present(tree, name):
    d = os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}")
    return all(os.path.exists(os.path.join(d, f)) for f in ("results.json", "per_pass.csv"))


def prepare(tree, sets=None, locked=None, pass_reader=None):
    """Steps 3 and 4: every number fixed before E1 is read (margins, J0's section-3 numbers, the J0 reading in the
    pass unit, the 3a margins and the go line). Reads v1's and seed 2's files only."""
    if sets is None or locked is None:
        sets, locked = default_sets(tree)
    m = margins(tree, sets, locked)
    printed = check_printed(m, section_3_text(tree))
    jr = jp0_reading(tree)
    j0 = check_section3_j0(tree, m["b_end"]["S12"]["line"], jr["doc"])
    reader = pass_reader or PassReader(tree)
    v1, s2 = reader(NAMES["v1_B"], tree), reader(NAMES["s2_B"], tree)
    check_pass_lines("v1_B", v1, jr["doc"])
    check_pass_lines("s2_B", s2, jr["doc"])
    pm = pass_margins(v1, s2, jr["doc"])
    go_line = {"evaluated": False, "missing": []}
    for k in ("v1_4000", "s2_4000"):
        if not j0_present(tree, NAMES[k]):
            go_line["missing"].append(f"reports/bench/frozen_tokens_{NAMES[k]}/")
    if not go_line["missing"]:
        a4, b4 = reader(NAMES["v1_4000"], tree), reader(NAMES["s2_4000"], tree)
        Y1, Y2 = fin(a4["rows"]["latent"]["point"]), fin(b4["rows"]["latent"]["point"])
        d = abs(Y1 - Y2)
        mm = max(TARGET_FRACTION * Y1, SEED_MULTIPLE * d)
        go_line = {"evaluated": True, "Y1": Y1, "latent_s2": Y2, "d": d, "m": mm, "threshold": Y1 - mm,
                   "v1_line": a4["rows"]["latent"], "missing": []}
    return {"sets": sets, "locked": locked, "margins": m, "printed": printed, "jp0": jr, "j0_section3": j0,
            "pass": {"v1_B": v1, "s2_B": s2}, "pass_margins": pm, "go_line": go_line, "reader": reader}


def run(tree=None, out=None, e1_tree=None, sets=None, locked=None, pass_reader=None, job_log=None,
        seed_difference=None, stamp=None):
    """Steps 1 to 9 (module docstring). tree: v1's and seed 2's files, the proposal, the JP0 reading, F1; e1_tree:
    E1's run folder and reports (default tree). Returns the results dict; raises Stop."""
    tree = os.path.abspath(tree or TREE)
    e1_tree = os.path.abspath(e1_tree or tree)
    out = os.path.abspath(out or os.path.join(tree, OUT_REL))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    sha3a = check_addendum(tree)                                                       # 1: before anything else
    if os.path.exists(os.path.join(out, "results.json")):                              # 2
        raise Stop(EXIT_EXISTS, f"{rel(os.path.join(out, 'results.json'), tree)} exists: the reading is never "
                                "rewritten (pass --out for a new folder)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise Stop(EXIT_HARNESS, "the reading runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    pre = prepare(tree, sets, locked, pass_reader)                                     # 3 and 4
    sets, locked, reader = pre["sets"], pre["locked"], pre["reader"]
    # 5: E1's inputs
    recs, log_info = read_log(e1_tree)
    verify = WB().b_end_verify(recs or []) if recs is not None else None
    name = name_of(verify.get("path")) if verify is not None else E1_B_DEFAULT
    missing = [rel(p, e1_tree) for p in required_inputs(e1_tree, name) if not os.path.exists(p)]
    if missing:
        raise Stop(EXIT_MISSING, f"NOT EVALUATED, nothing written; missing E1 inputs of {name}: " + ", ".join(missing))
    # 6: the resume lineage (an integrity question when inconsistent, nothing written), then R0 of the branch rule
    lin = lineage(recs)
    lineage_stop(lin, log_info)
    r0 = r0_reading(recs, log_info, lin, job_rows(job_log or os.path.join(e1_tree, JOB_LOG_REL)))
    sha_B = (verify or {}).get("sha256")
    sha_4000 = checkpoint_sha(recs, 4000)
    # 7: section 3 at the B end
    vals, why = e1_values_B(e1_tree, name, sets, locked, sha_B)
    if why:
        code = EXIT_MISSING if all(str(v).startswith("missing") for v in why.values()) else EXIT_HARNESS
        raise Stop(code, ("NOT EVALUATED, nothing written; " if code == EXIT_MISSING else "harness defect: ")
                   + f"E1 lines of {name}: " + "; ".join(f"{k}: {v}" for k, v in why.items()))
    m = pre["margins"]
    b_reads = line_reads(vals, m["b_end"], m["values"]["v1_B"]["S12_per_person"])
    n12 = len(sets["dev12"])
    if b_reads["S12"]["paired_v1_minus_e1"]["n_persons"] != n12:
        n_pp = b_reads["S12"]["paired_v1_minus_e1"]["n_persons"]
        raise Stop(EXIT_HARNESS, f"the paired bootstrap of S(12) holds {n_pp} persons, not the {n12} of dev12")
    failed = ([] if r0["holds"] else ["R0: " + ", ".join(r0["failed"])]) + \
        [f"{'success line' if k == 'S12' else 'non-inferiority'} {k}" for k in LINES if not b_reads[k]["holds"]]
    verdict = MET if not failed else NOT_MET
    # early check, reported only: a failure is reported as not evaluated, never a stop
    go = GO()
    early_missing = [rel(p, e1_tree) for p in early_inputs(e1_tree) if not os.path.exists(p)]
    v4, why4, early_error = early_values(e1_tree, sets, locked, sha_4000)
    early_reads = line_reads(v4, m["early"], m["values"]["v1_4000"]["S12_per_person"])
    # J0 of E1 at the B end (per-stride unit, leakage)
    j0doc = read_json(os.path.join(e1_tree, "reports", "bench", f"frozen_tokens_{name}", "results.json"))
    if j0doc.get("partial") or (j0doc.get("checkpoint") or {}).get("name") != name:
        raise Stop(EXIT_HARNESS, f"J0 of {name}: partial run or another checkpoint")
    if sha_B and (j0doc.get("checkpoint") or {}).get("sha256") != sha_B:
        raise Stop(EXIT_HARNESS, f"J0 of {name} read checkpoint {(j0doc.get('checkpoint') or {}).get('sha256')}, the "
                                 f"verify event names {sha_B}")
    e1_p = go.j0_latent_per_stride(j0doc)
    if e1_p is None:
        raise Stop(EXIT_HARNESS, f"J0 of {name}: no finite latent per-stride stride line on the 12")
    j0v1 = read_json(os.path.join(tree, "reports", "bench", f"frozen_tokens_{NAMES['v1_B']}", "results.json"))
    leak = {"e1": fin((j0doc.get("leakage") or {}).get("leakage_points")),
            "v1": fin(((j0v1 or {}).get("leakage") or {}).get("leakage_points"))}
    if None in leak.values():
        leak.update(evaluated=False, caution=None)
    else:
        leak.update(evaluated=True, rise=leak["e1"] - leak["v1"], points=G10_POINTS,
                    caution=bool(leak["e1"] - leak["v1"] >= G10_POINTS - TOL))
    leak["text"] = ("G10 caution applies: E1 is not adopted into the main recipe, whatever its G3 lines, until the "
                    "source of the rise is found; the G3 reading does not change" if leak.get("caution") else
                    "G10 caution does not apply" if leak.get("evaluated") else "G10 caution not evaluated")
    # 8: the pass unit of E1 and the predictions
    e1p = reader(name, e1_tree)
    if sha_B and e1p.get("checkpoint_sha256") not in (None, sha_B):
        raise Stop(EXIT_HARNESS, f"pass unit of {name}: checkpoint {e1p.get('checkpoint_sha256')}, verify {sha_B}")
    pm, v1p = pre["pass_margins"], pre["pass"]["v1_B"]
    rel_D = one_end(e1p["rows"].get("latent"), e1p["rows"].get("decoded"), (pm.get("D") or {}).get("m"))
    rel_R = one_end(e1p["rows"].get("latent"), e1p["rows"].get("recalibrated"), (pm.get("R") or {}).get("m"))
    rel_P = one_end(e1p["rows"].get("latent"), v1p["rows"].get("latent"), pm["P"]["m"])
    rel_stem = one_end(e1p["rows"].get("stem"), v1p["rows"].get("stem"), (pm.get("stem") or {}).get("m"))
    if not rel_P.get("evaluated"):
        raise Stop(EXIT_HARNESS, f"pass unit of {name}: no finite latent S_pass(12)")
    e1_readings = {"D": e1_carry(rel_D), "R": e1_carry(rel_R),
                   "R_note": None if rel_R.get("evaluated") else
                   "F1 gives no stride gain for E1's checkpoint and none is fitted here"}
    success = bool(b_reads["S12"]["holds"])
    ps = per_stride_probe(pre["j0_section3"], e1_p)
    pred3 = predict(pre["j0_section3"]["reading"], success, ps.get("improved"), ps.get("within"))
    main3a = pre["jp0"]["main_reading"]
    pred3a = predict(main3a, success, rel_P.get("better"), rel_P.get("within"))
    agree = pred3["next"] == pred3a["next"]
    route, route_parts = route_of(pred3a, verdict, failed, leak, pre["jp0"]["readings"])
    go_line = dict(pre["go_line"], e1=None, go=None, reason=None)
    if go_line.get("evaluated"):
        if j0_present(e1_tree, E1_4000):
            try:
                e4 = reader(E1_4000, e1_tree)
                got = e4.get("checkpoint_sha256")
                if sha_4000 and got not in (None, sha_4000):
                    raise Stop(EXIT_HARNESS, f"checkpoint {got}, the log's ckpt_B_004000 is {sha_4000}")
                rel_4 = one_end((e4.get("rows") or {}).get("latent"), go_line["v1_line"], go_line["m"])
                if not rel_4.get("evaluated"):
                    raise Stop(EXIT_HARNESS, "no finite latent S_pass(12)")
                go_line.update(e1=rel_4, go=bool(rel_4.get("better")))
            except (Stop, AssertionError, ValueError, KeyError, TypeError) as e:
                go_line["reason"] = f"not evaluated: pass unit of {E1_4000}: {getattr(e, 'message', None) or e}"
        else:
            go_line["missing"] = [f"reports/bench/frozen_tokens_{E1_4000}/"]
            go_line["reason"] = "not evaluated: missing " + go_line["missing"][0]
    sd_path = seed_difference or os.path.join(tree, SEED_DIFF_REL)
    beside_sd = seed_difference_beside(sd_path, b_reads, early_reads)
    sup = read_json(os.path.join(tree, SUPERVISED_REL))
    sup12 = fin((((sup or {}).get("lines") or {}).get("dev12") or {}).get("supervised", {}).get("stride_mae_cm", {})
                .get("point"))
    # 9: results
    inputs = [os.path.join(tree, PROPOSAL_REL), os.path.join(tree, "reports", "bench", "jepa_jp0_addendum.sha256"),
              os.path.join(tree, JP0_RESULTS_REL), os.path.join(tree, F1_RESULTS_REL), os.path.join(tree, SEEDS_REL),
              os.path.join(tree, SUPERVISED_REL)] + required_inputs(e1_tree, name) + \
        [p for p in early_inputs(e1_tree) if os.path.exists(p)]
    job_log_path = job_log or os.path.join(e1_tree, JOB_LOG_REL)
    if os.path.exists(job_log_path):
        inputs.append(job_log_path)
    if beside_sd.get("present"):
        inputs.append(sd_path)
    res = {"what": LABEL, "task": TASK, "run_started": started,
           "addendum": {"sha256": sha3a, "asserted_first": True, "proposal": PROPOSAL_REL},
           "verdict": verdict, "failed": failed, "name": name, "run_dir": E1_RUN_DIR,
           "r0": r0, "lineage": lin,
           "b_end": {"lines": b_reads, "margins": m["b_end"]},
           "early": {"label": "reported only: it neither stops E1 nor replaces the B-end reading",
                     "lines": early_reads, "margins": m["early"], "missing": early_missing,
                     "why": {k: v for k, v in why4.items()}, "error": early_error,
                     "checkpoint_sha256_from_log": sha_4000},
           "margins": {"values": {k: {x: y for x, y in v.items() if x != "S12_per_person"}
                                  for k, v in m["values"].items()},
                       "seed_terms": m["seed_terms"], "printed_strings_found": pre["printed"]},
           "j0_section3": pre["j0_section3"],
           "jp0_reading": {k: pre["jp0"][k] for k in ("path", "sha256", "readings", "route", "main_reading",
                                                      "relations", "section3_reading")},
           "pass_unit": {"margins": pm, "v1_B": v1p, "s2_B": pre["pass"]["s2_B"], "e1_B": e1p,
                         "relations": {"D": rel_D, "R": rel_R, "probe_latent": rel_P, "stem": rel_stem},
                         "e1_readings": e1_readings, "v1_readings": pre["jp0"]["readings"]},
           "per_stride_probe": ps,
           "predictions": {"section3_per_stride": pred3, "section3a_pass_unit": pred3a, "agree": agree,
                           "routes": "section 3a (pass unit)",
                           "note": None if agree else ("DISAGREEMENT between the two units: section 3a's reading "
                                                       "routes the next step")},
           "route": route, "route_parts": route_parts, "g10": leak, "jp2_go_line": go_line,
           "seed_difference": beside_sd, "log_read": log_info,
           "supervised_per_stride_12": sup12,
           "constants": {"target_fraction": TARGET_FRACTION, "seed_multiple": SEED_MULTIPLE, "g10_points": G10_POINTS,
                         "boot_n": JP().BOOT_N, "seed": JP().SEED, "level": JP().LEVEL, "tol": TOL,
                         "pass_tol_cm": PASS_TOL_CM, "e1_config_sha256": E1_CONFIG_SHA256,
                         "source": {"path": SOURCE_PATH, "sha256": SOURCE_SHA256}},
           "inputs": {rel(p, tree if p.startswith(tree) else e1_tree): sha256_file(p) for p in inputs
                      if os.path.exists(p)}}
    st = stamp if stamp is not None else _sibling("moveatlas_stamp").stamp()
    res.update(moveatlas_head=st.get("head"), timing_s=round(time.time() - t0, 1))
    res = JP().clean(res)
    text = summary_md(res)
    for t, w in ((text, "summary.md"), (json.dumps(res), "results.json")):
        if chr(0x2014) in t or chr(0x2013) in t:
            raise Stop(EXIT_HARNESS, f"{w} would hold a long dash")
    paths.assert_not_moveatlas(out)
    if os.path.exists(os.path.join(out, "results.json")):
        raise Stop(EXIT_EXISTS, f"{rel(os.path.join(out, 'results.json'), tree)} exists")
    os.makedirs(out, exist_ok=True)
    prov = {"script": "scripts/e1_read.py", "task": TASK, "label": LABEL, "run_at": started, "addendum_sha256": sha3a,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": st.get("head"), "moveatlas_stamp": st,
            "python": sys.version.split()[0],
            "code_sha256": {rel(p): sha256_file(p) for p in (os.path.abspath(__file__),) + tuple(
                os.path.join(SCRIPTS, f"{n}.py")
                for n in ("jp0_read", "e1_go_reading", "f2_lever2", "weekend_branch"))},
            "inputs": res["inputs"],
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    for fname, body in (("results.json", json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"),
                        ("summary.md", text),
                        ("_provenance.json", json.dumps(JP().clean(prov), indent=1, ensure_ascii=False) + "\n")):
        with open(os.path.join(out, fname), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    return res


# ---------------------------------------------------------------- summary
def summary_md(r):
    L = [f"# E1 reading ({LABEL})", "",
         "Written by scripts/e1_read.py (task E1R) under sections 3 and 3a of "
         f"docs/proposals/JEPA_EXPLORATION_DRAFT.md; section 3a SHA-256 {r['addendum']['sha256']}, asserted before any "
         "other file was read. Every number below is read from results.json of this folder. Dev only; no lockbox, "
         "EVAL, CODY or care_pd record; no gain fitted.",
         "", f"## Verdict: **{r['verdict']}** (E1 {r['name']} against v1 at the B end)", ""]
    if r["failed"]:
        L.append("Failed: " + "; ".join(r["failed"]) + ".")
    r0 = r["r0"]
    L += ["", "## R0", "",
          f"R0 {'holds' if r0['holds'] else 'fails'}: " + ", ".join(f"{k} {v}" for k, v in r0["checks"].items()) + ".",
          f"Verify: {r0['verify']}. Alarm events {r0['alarm_events']}, nonfinite events {r0['nonfinite_events']}, "
          f"unparsed log lines {r0['unparsed_log_lines']} (line numbers {r0['unparsed_line_numbers']}; records "
          f"recovered from lines {r0['recovered_line_numbers']}), reported and not a condition, as the branch rule "
          f"reports them; exit {r0['exit_code']} ({r0['exit_from']}).", "",
          "Resume lineage (call by call; checked beside R0, not a condition of it; an inconsistent lineage stops the "
          "reading before anything is written):", "",
          "| call | how | from | from step | first log step | last step | steps trained again | verify | end |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in r["lineage"]["calls"]:
        L.append(f"| {c['call']} | {c['how']} | {c['from']} | {c['from_step']} | {c['first_log_step']} | "
                 f"{c['last_step']} | {c.get('steps_trained_again', 'n/a')} | {(c.get('verify') or {}).get('path')} | "
                 f"{c['end_event']} |")
    lin = r["lineage"]
    probs = f" ({'; '.join(lin['problems'])})" if lin["problems"] else ""
    L += ["", f"Lineage consistent: {lin['consistent']}{probs}. The log shows {lin['log_says']['resume_events']} "
          f"resumes, the last from {lin['log_says']['last_resume']}; "
          f"the record ({lin['record']['source']}): {lin['record']['resume_events']}, the last from "
          f"{lin['record']['last_resume']}; agreement: {lin['agrees_with_record']}.", "",
          "## Section 3 at the B end (the 12 and, for the IC lines, the 14)", "",
          "| line | E1 | v1 | margin | E1 at most | holds | paired v1 minus E1 |", "|---|---|---|---|---|---|---|"]
    for k in LINES:
        x = r["b_end"]["lines"][k]
        pr = x.get("paired_v1_minus_e1")
        L.append(f"| {k} ({UNIT[k]}) | {fmt(x.get('value'), ND[k])} | {fmt(x.get('line_v1'), ND[k])} | "
                 f"{fmt(x.get('margin'), ND[k])} | {fmt(x.get('at_most'), ND[k])} | {x.get('holds')} | "
                 + (f"{fmt(pr['mean'])} {fmt_iv(pr['interval'])}, n {pr['n_persons']}" if pr else "") + " |")
    e = r["early"]
    L += ["", "## Early check at ckpt_B_004000 (reported only)", "",
          "| line | E1 | v1 | E1 at most | holds |", "|---|---|---|---|---|"]
    for k in LINES:
        x = e["lines"][k]
        L.append(f"| {k} ({UNIT[k]}) | {fmt(x.get('value'), ND[k])} | {fmt(x.get('line_v1'), ND[k])} | "
                 f"{fmt(x.get('at_most'), ND[k])} | {x.get('holds')} |")
    if e["missing"]:
        L.append("Missing early inputs: " + ", ".join(e["missing"]) + ".")
    if e.get("error"):
        L.append(f"Early check not evaluated: {e['error']}.")
    why_e = {k: v for k, v in (e.get("why") or {}).items() if not str(v).startswith("missing")}
    if why_e:
        L.append("Early lines not evaluated: " + "; ".join(f"{k}: {v}" for k, v in why_e.items()) + ".")
    L.append(f"SHA-256 of ckpt_B_004000 in E1's log: {e.get('checkpoint_sha256_from_log') or 'none'}.")
    s3, ps = r["j0_section3"], r["per_stride_probe"]
    pu = r["pass_unit"]
    L += ["", "## Predictions (section 3) in both units", "",
          f"- Per-stride unit (section 3): J0 reads {s3['reading']} (D {fmt(s3['D'])}, P1 {fmt(s3['P1'])}, dP "
          f"{fmt(s3['dP'])} cm). E1's probe {fmt(ps.get('E1'))} cm, P1 minus E1 {fmt(ps.get('P1_minus_E1'))} against "
          f"the band {fmt(ps.get('band'))} cm (improved {ps.get('improved')}, within {ps.get('within')}). "
          f"Prediction holds: {r['predictions']['section3_per_stride']['holds']} "
          f"({r['predictions']['section3_per_stride']['outcome']}).",
          "- Pass unit (section 3a, routes): the J0 reading section 3a routes by is "
          f"{r['jp0_reading']['main_reading']} "
          f"(GAIN {r['jp0_reading']['readings'].get('GAIN')}, D {r['jp0_reading']['readings'].get('D')}, R "
          f"{r['jp0_reading']['readings'].get('R')}). E1's latent S_pass(12) "
          f"{fmt(pu['relations']['probe_latent'].get('X'))} against v1's "
          f"{fmt(pu['relations']['probe_latent'].get('Y'))} "
          f"cm, gap {fmt(pu['relations']['probe_latent'].get('G'))} against m_P {fmt(pu['margins']['P']['m'])} "
          f"(paired {fmt_iv(((pu['relations']['probe_latent'].get('paired')) or {}).get('interval'))}; better "
          f"{pu['relations']['probe_latent'].get('better')}, within {pu['relations']['probe_latent'].get('within')}). "
          f"Prediction holds: {r['predictions']['section3a_pass_unit']['holds']} "
          f"({r['predictions']['section3a_pass_unit']['outcome']}).",
          f"- {r['predictions']['note'] or 'The two units route alike.'}", "",
          f"Route: {r['route']}.", "",
          f"Candidate for the main recipe: {r['route_parts']['candidate']}"
          + (f" (blocked by: {'; '.join(r['route_parts']['blocks'])})" if r['route_parts']['adoption_possible']
             and not r['route_parts']['candidate'] else "") + ".", "",
          "## Readings D and R on E1's frozen-token probes (one end, margins fixed from v1 and seed 2)", "",
          f"- D (latent against decoded, m_D {fmt((pu['margins'].get('D') or {}).get('m'))} cm): "
          f"**{pu['e1_readings']['D']}** (G {fmt(pu['relations']['D'].get('G'))}, paired "
          f"{fmt_iv(((pu['relations']['D'].get('paired')) or {}).get('interval'))}); v1 read "
          f"{pu['v1_readings'].get('D')}.",
          f"- R (latent against recalibrated): **{pu['e1_readings']['R']}**"
          + (f" ({pu['e1_readings']['R_note']})" if pu['e1_readings']['R_note'] else "") + ".",
          f"- E1's stem row against v1's: G {fmt(pu['relations']['stem'].get('G'))} cm, margin "
          f"{fmt((pu['margins'].get('stem') or {}).get('m'))} (better {pu['relations']['stem'].get('better')}).", "",
          "| row | E1 S_pass(12) cm | v1 | seed 2 |", "|---|---|---|---|"]
    for row in CHAIN:
        a = (pu["e1_B"]["rows"].get(row) or {}).get("point")
        b = (pu["v1_B"]["rows"].get(row) or {}).get("point")
        c = (pu["s2_B"]["rows"].get(row) or {}).get("point")
        L.append(f"| {row} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    g = r["g10"]
    L += ["", f"G10-style leakage (J0, weargait subject-id family): E1 {fmt(g.get('e1'))}, v1 {fmt(g.get('v1'))} "
          "points: "
          f"{g['text']}."]
    jp = r["jp2_go_line"]
    if jp.get("evaluated"):
        e1g = jp.get("e1") or {}
        L.append(f"JP2 go line at ckpt_B_004000 (reported): v1 latent S_pass(12) {fmt(jp['Y1'])} cm, seed 2 "
                 f"{fmt(jp['latent_s2'])}, m {fmt(jp['m'])}: E1 at most {fmt(jp['threshold'])} cm with the paired "
                 f"lower bound above 0; "
                 + (f"E1 {fmt(e1g.get('X'))} cm, go {jp.get('go')}." if jp.get("e1") else f"E1 {jp.get('reason')}."))
    else:
        L.append("JP2 go line at ckpt_B_004000: not evaluated (missing " + ", ".join(jp.get("missing") or []) + ").")
    sd = r["seed_difference"]
    L += ["", "## Beside, never substituted", ""]
    if sd.get("present"):
        L += [f"Matched-settings seed differences ({sd['path']}, {sd['label']}):", "",
              "| item | E1 | threshold here | mismatched | matched | same code | E1 holds (mismatched, matched, same "
              "code) |", "|---|---|---|---|---|---|---|"]
        for x in sd["rows"]:
            L.append(f"| {x['item']} | {fmt(x['e1_value'], 4)} | {fmt(x['threshold_here'], 4)} | "
                     f"{fmt(x['threshold_mismatched'], 4)} | {fmt(x['threshold_matched'], 4)} | "
                     f"{fmt(x['threshold_same_code'], 4)} | {x['e1_holds_mismatched']}, {x['e1_holds_matched']}, "
                     f"{x['e1_holds_same_code']} |")
    else:
        L.append(f"Matched-settings seed differences: {sd['path']} does not exist.")
    L += [f"Supervised baseline per stride on the 12: {fmt(r['supervised_per_stride_12'])} cm.", "",
          "## Inputs", ""] + [f"- {k} (SHA-256 {v})" for k, v in r["inputs"].items()]
    L += ["", f"MoveAtlas: HEAD {r.get('moveatlas_head') or 'n/a'}. Run started {r['run_started']}, {r['timing_s']} s.",
          ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def margins_text(pre):
    m, pm, j0 = pre["margins"], pre["pass_margins"], pre["j0_section3"]
    L = ["E1 reading: the numbers fixed before E1 is read (v1's and seed 2's files only)", ""]
    for block, title in (("b_end", "B end"), ("early", "ckpt_B_004000 (reported only)")):
        L.append(f"{title}:")
        for k in LINES:
            b = m[block][k]
            L.append(f"  {k}: v1 {fmt(b['line'], ND[k])} {UNIT[k]}, seed term {fmt(b['seed_term'], ND[k])}, margin "
                     f"{fmt(b['margin'], ND[k])}, E1 at most {fmt(b['at_most'], ND[k])} ({b['kind']})")
    L.append(f"Section 3 strings found in the proposal: {len(pre['printed'])}")
    L.append(f"J0 section 3 (per stride): D {fmt(j0['D'])}, P1 {fmt(j0['P1'])}, P2 {fmt(j0['P2_s2'])}, "
             f"dP {fmt(j0['dP'])}, "
             f"margin {fmt(j0['margin'])}: {j0['reading']}; band of E1's probe {fmt(j0['band'])} cm")
    for k in ("D", "R", "P", "stem"):
        x = pm.get(k)
        L.append(f"Section 3a margin {k}: " + ("n/a" if x is None else
                                                f"Y1 {fmt(x['Y1'])}, d {fmt(x['d'])}, m {fmt(x['m'])} cm"))
    L.append(f"J0 reading section 3a routes by: {pre['jp0']['main_reading']} (readings {pre['jp0']['readings']})")
    gl = pre["go_line"]
    L.append("JP2 go line at ckpt_B_004000: " + (f"Y1 {fmt(gl['Y1'])}, seed 2 {fmt(gl['latent_s2'])}, "
                                                 f"d {fmt(gl['d'])}, "
                                                 f"m {fmt(gl['m'])}, E1 at most {fmt(gl['threshold'])} cm"
                                                 if gl.get("evaluated") else f"missing {gl['missing']}"))
    return "\n".join(L) + "\n"


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="the tree whose files are read (default this tree)")
    p.add_argument("--out", default=None, help=f"output folder (default {OUT_REL} of the tree)")
    p.add_argument("--margins", action="store_true", help="steps 1, 3 and 4 only: print the margins, open no E1 file")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    tree = os.path.abspath(a.tree or TREE)
    try:
        if a.margins:
            check_addendum(tree)
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
                raise Stop(EXIT_HARNESS, "the reading runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
            print(margins_text(prepare(tree)))
            return EXIT_WRITTEN
        res = run(tree, a.out)
        print(summary_md(res))
        print(f"written {rel(os.path.abspath(a.out or os.path.join(tree, OUT_REL)), tree)}")
        return EXIT_WRITTEN
    except Stop as s:
        print(s.message, file=sys.stderr)
        return s.code


if __name__ == "__main__":
    sys.exit(main())
