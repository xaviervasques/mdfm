"""S2.10 (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 4.12 and 10): the healthy non-inferiority read of a stage-2
run against its start checkpoint, the attainability function of its rule, and the post-train set of a stage-2 run.
Label of every output: dev diagnostic, not a G3 read.

Rule: reports/bench/stage2_read/rule.txt, DRAFT until the owner reads it and the start checkpoint is chosen (spec
section 1 item 2: after the E1 and F2 lever-4 readings). Its machine-read fields are three lines at the top: 'Status:
DRAFT' or 'Status: final', 'Start: not chosen' or 'Start: <tree-relative checkpoint> SHA-256 <64 hex>', 'Config: not
fixed' or 'Config: <config file> SHA-256 <config SHA-256>' (rule_fields). The rule binds from the moment its SHA-256 is
written in rule.sha256 beside it, which only --hash does, and only when hash_refusals finds nothing (below).

Lines (LINES; spec section 4.12): S(12), V(12), S(14), V(14) from the base row of the G3 diag (diag.json, subsets
no_whc003_whc026 and all14, per-person stride_cm and speed_ms), stride_mape_pct and the walkway IC from g3_dev.json
(results_dev points, the model's per-person values of per_person.csv, row dev), G1a from the full G1 dev run
(g1.json, line G1a; per take from windows.csv, weighted by knots). The insole IC line is read the same way and decides
nothing (amendment A1 of the F2 lever-2 rule). Persons: dev14 and dev12 of reports/bench/g3_gain_check/results.json,
asserted by scripts/f2_lever2.py (load_sets, guard_set); no lockbox person (mdfm.gates.g3.lockbox_persons) in any row
read.

Statistic: the point, end minus start (positive is worse); a line HOLDS when its point is at most its margin (TOL
1e-9), FORGETS otherwise. The paired interval of each line (end minus start per person, or per take for G1a; the
v2check bootstrap of scripts/weekend_branch.py: 10,000 resamples in blocks of 1,000, numpy default_rng(20261001)) is
reported beside the point and decides nothing (spec section 10). A line whose end and start decodes ran under other
inference settings (the 'inference' entry of their files, the memory entry left out; the device is compared) is not
read. A report is read only when its checkpoint entry names the checkpoint, EMA weights and, once the checkpoint's
SHA-256 is known (the read: the Start line and the phase C verify event), that SHA-256 (report_issue); a report that
names no SHA-256 then counts as missing.

Margins and their attainability (attainability; spec section 10): per line, max(FIVE_PERCENT x the start's line,
SEED_MULTIPLE x the B-end seed difference); the seed difference is v1's recipe pair, v1_B against s2_B (the seeds file
reports/gpu_queue/branch_seeds_20261002_211426.json for S and V, |v1_B - s2_B| of their g3_dev.json and g1.json for
stride_mape_pct, the IC lines and G1a), the only seed pair of the program, whatever the start. The function reads two
decodes of the start at the read's settings (the first under the start's own report names, the second under the same
names with the suffix ATT_SUFFIX, or folders given), takes the point difference of the two decodes per line and its
paired interval, prints them beside every margin, and raises a margin that falls below the absolute point difference
to that difference. It is attainable when every line has both decodes at equal settings. Beside each margin it flags
a line whose paired interval of the two decodes reaches beyond the margin on either side (decode_interval_beyond): on
that line decoding alone can move the point past the margin, so a model equal to its start may read FORGETS on it;
the flag decides nothing (the point decides, spec section 10) and is printed for the owner before the hash.
section7_lines renders the lines the rule's section 7 must hold verbatim before the hash.

Hash (--hash; hash_refusals): refused while the rule reads Status DRAFT, the Start or the Config line is not filled,
rule.sha256 exists, the attainability of the rule's start is not attainable, or a line of section7_lines is missing
from the rule's section 7. rule.sha256 holds the rule's SHA-256 and the local time of the hash (a line 'hashed_at
<ISO time>', the clock and format of the run's log lines). Never after a number of the run exists: the read refuses a
rule whose recorded hash time is not before the run's first log line (the file's mtime, which a checkout or a copy
moves, is reported beside it and decides nothing).

Read (--read --run-dir; read_run): asserts rule.sha256 and the rule's Status final; R0 (r0_reading) from the run's
log.jsonl and reports/gpu_queue/job_log.csv, in the two parts of amendment A2 of the F2 lever-2 rule. Stops: an
alarm event (the stage-2 config stops the run on it, schedule.stop_on_alarm), a non-finite stop (F2's NONFINITE_RUN
nonfinite events at consecutive optimizer steps) or the memory bound (a memory_exceeded event: the loop's
schedule.max_reserved_gb of the Config). Failures: no init_stage2 event, or one whose source is not the rule's start
at its SHA-256; no start event, or one at another config SHA-256; a report mismatch in a start or the verify event;
no phase C verify event at the phase C step count; no job_log.csv row of a scripts/train_stage1.py job on the run
folder, or a last such row that did not exit 0 (a crash resumed by the queue is allowed when the rest holds). The
margins are recomputed (attainability on the rule's start) and their section7_lines asserted in the rule (a departure
is a harness defect, exit EXIT_HARNESS). Verdict (verdict), in A2's order: (1) a stop: STOPPED, a decided verdict
(A2 case 1 gives the lever's NO-GO; this read's verdicts name forgetting on the lines, which a stopped run has not
measured, so the stop has its own name); (2) any other R0 failure: NOT EVALUATED (the lines of another run, or of no
end, read nothing about this one); (3) a line read that FORGETS: FORGETS, whatever other input is missing; (4) a
missing input: NOT EVALUATED, each named; (5) KEEP. A deciding line that FORGETS while the attainability check
flagged its decode noise is named "possibly decode noise" beside the verdict, which it does not change (decode_noise;
option (a) of the rule's section 7, chosen under the owner's delegation on 2026-10-08, task S2.11). --expect-rule-sha256
(the queued read job carries it) makes a rule that hashes otherwise a harness defect. With --write the read writes
nothing while it is pending (pending): no stop, and either the run has no end while its training job has no exit in
the job log, or one of the end's three deciding decodes is missing; it then exits NOT EVALUATED, so that the read of
the finished run can still be written once.
Amendment 1 (S2.11 review; reports/bench/stage2_read/rule_amendment_1.txt, hashed by --hash-amendment into
rule_amendment_1.sha256 beside the rule, whose text it leaves unchanged; the form of the F2 lever-2 amendment): the
verdict of the rule is the one under the rule with amendment 1, and the rule alone is read beside it (rule section
10: "keeps both readings"). Its margin of a deciding line is max(the rule's margin, the largest
absolute bound of the paired interval of the start's two decodes) (second_reading; on v1's two decodes only the
walkway IC line rises). Beside each line it prints the share of sign flips of the start's per-unit decode differences
whose mean passes the margin and the second-reading margin (null_share: every flip up to FLIP_EXACT_MAX units, else
FLIPS draws of default_rng(FLIP_SEED)), what a model equal to its start reads FORGETS on when decoding alone moves it.
--hash-amendment refuses while the rule is not hashed and final, the amendment is not final, does not name the rule
at its SHA-256, is hashed already, the start is not attainable, or a line of amendment_lines is missing from it.
The read gives the verdict under the rule with amendment 1 (the verdict function on the same R0 and lines, with the
second-reading margins; its exit code) and, beside it, the rule alone (rule_alone, with its decode-noise naming); it
says whether they agree and names the lines where they differ. It refuses (harness defect) an amendment hashed at or
after the run's first log line, naming another rule SHA-256, or whose lines differ from the recomputation; an
amendment present but not hashed is not read (the rule alone decides, and the output says so).
--expect-amendment-sha256 (the queued read job carries it) makes an absent, unhashed or other amendment a harness
defect. --attainability --amendment prints the second reading too (with --write:
attainability_<start>_amendment_1.json and .md).
Amendment 2 (task S2B; reports/bench/stage2_read/rule_amendment_2.txt, hashed by --hash-amendment-2 into
rule_amendment_2.sha256; rule.txt and amendment 1 unchanged): names the run S2b (its Run and Smoke folders), its
configuration (Config: <config file> SHA-256 <configuration SHA-256>) and recipe, on the rule's Start; machine-read
fields Status, Rule, Amendment 1, Start, Config, Run and Smoke (amendment2_state). For a run folder it names, the read
and the smoke check take its Config in place of the rule's (R0's start-event check and the Config's memory bound and
units; amendment2_binding), in both readings of the read; every margin, line, verdict case and route stays the rule's
and amendment 1's, and a run it does not name (S2a) reads as before. --hash-amendment-2 refuses while the rule is not
hashed and final, the amendment is not final, its Rule or Amendment 1 line does not name the hashed file at its
SHA-256, its Start differs from the rule's, its Config does not load to the configuration SHA-256 it names or is not a
stage-2 config, it names no Run, the Run or Smoke folder holds a log.jsonl, it is hashed already, the start is not
attainable, or it lacks a line of amendment_lines (amendment 1's margins, verbatim). The read refuses (EXIT_REFUSED) a
run that an unhashed amendment 2 names, and (harness defect) an amendment 2 hashed at or after the run's first log
line or one that does not bind; --expect-amendment-2-sha256 (the S2b read job carries it) makes an absent, unhashed or
other amendment 2, or one that does not name the run, a harness defect.
Amendment 3 (task S2C; reports/bench/stage2_read/rule_amendment_3.txt, hashed by --hash-amendment-3 into
rule_amendment_3.sha256; rule.txt and amendments 1 and 2 unchanged): names the run S2c (its Run and Smoke folders) and
its configuration, amendment 2's with name, status and schedule.alarm_consecutive {beta_error: 3} changed (A3_CHANGES,
A3_ALARM_CONSECUTIVE): the beta_error alarm raised at three log records in a row. Machine-read fields Status, Rule,
Amendment 1, Amendment 2, Start, Config, Run and Smoke (amendment3_state). For a run folder it names, the read and the
smoke check take its Config in place of the rule's, as amendment 2 does for S2b (run_binding), and the read reports
the log's 'alarm_watch' events beside the verdict, deciding nothing (alarm_watch_events); S2a and S2b read as before.
--hash-amendment-3 refuses while the rule is not hashed and final, the amendment is not final, its Rule, Amendment 1 or
Amendment 2 line does not name the hashed file at its SHA-256, its Start differs from the rule's, its Config does not
load to the configuration SHA-256 it names, is not a stage-2 config, differs from amendment 2's Config elsewhere than in
A3_CHANGES or reads the alarm otherwise, it names no Run, it names a run folder of amendment 2, the Run or Smoke folder
holds a log.jsonl, it is hashed already, the start is not attainable, amendment 2 lacks a line of amendment_lines
(amendment 1's margins, unchanged), or it lacks a line of alarm_lines (its section 3: the false alarms of the reading,
rendered from reports/bench/stage2_beta_alarm/results.json under the Config). The read refuses (EXIT_REFUSED) a run
that an unhashed amendment 3 names, and (harness defect) an amendment 3 hashed at or after the run's first log line or
one that does not bind; --expect-amendment-3-sha256 (the S2c read job carries it) makes an absent, unhashed or other
amendment 3, or one that does not name the run, a harness defect, and the job's --expect-amendment-2-sha256 then checks
amendment 2's identity only.
Amendment 4 (task S2D; reports/bench/stage2_read/rule_amendment_4.txt, hashed by --hash-amendment-4 into
rule_amendment_4.sha256; rule.txt and amendments 1, 2 and 3 unchanged): names the run S2d (its Run and Smoke folders)
and its configuration, amendment 3's with name, status and schedule.alarm_consecutive {beta_error: 3, unit_nll_rise:
2} changed (A4_CHANGES, A4_ALARM_CONSECUTIVE): the unit_nll_rise alarm raised at two dev reads in a row, beta_error
read as in S2c. Machine-read fields Status, Rule, Amendment 1, Amendment 2, Amendment 3, Start, Config, Run and Smoke
(amendment4_state). For a run folder it names, the read and the smoke check take its Config in place of the rule's
(run_binding4), and the read reports the log's 'alarm_watch' events beside the verdict with the rising units of each
unit_nll_rise entry, deciding nothing; S2a, S2b and S2c read as before. --hash-amendment-4 refuses while the rule is not
hashed and final, the amendment is not final, its Rule, Amendment 1, 2 or 3 line does not name the hashed file at its
SHA-256, amendment 3 does not bind on its own points, its Start differs from the rule's, its Config does not load to
the configuration SHA-256 it names, is not a stage-2 config, differs from amendment 3's Config elsewhere than in
A4_CHANGES or reads the alarms otherwise, it names no Run, it names a run folder of amendment 2 or 3, the Run or Smoke
folder holds a log.jsonl, it is hashed already, the start is not attainable, amendment 2 lacks a line of
amendment_lines, amendment 3 lacks a line of alarm_lines under the Config (the beta_error reading, unchanged), or it
lacks a line of unit_alarm_lines (its section 3: the false alarms of the unit_nll_rise reading, rendered from
reports/bench/stage2_unit_nll_alarm/results.json under the Config). The read refuses (EXIT_REFUSED) a run that an
unhashed amendment 4 names, and (harness defect) an amendment 4 hashed at or after the run's first log line or one that
does not bind; --expect-amendment-4-sha256 (the S2d read job carries it) makes an absent, unhashed or other amendment 4,
or one that does not name the run, a harness defect, and the job's --expect-amendment-2-sha256 and
--expect-amendment-3-sha256 then check those amendments' identity only.
Reported beside, deciding nothing (spec section 4.12): the insole IC line; the G3 PD half (g3_pd_dev.json, row
pd_t19dev, model) of the end and the start; G4 on the stroke fold (g4.json of eval_g4_dev.py --stroke-set fold) of the
end and the start; the SP3 families (reports/synth/sp3/<name>/summary.json) of the end and the start; the per-unit dev
NLL and outside-limit share of the run's last phase C dev event; the frozen-token cohort probe, whose script does not
exist (named, not read). The PD half, G4 and SP3 files are read under report_issue as the lines are. Writes
reports/bench/stage2_read/<run>/results.json, summary.md and _provenance.json (never over an existing results.json).

Post-train set (--post-train RUN_DIR; expand_post_train): the stage-2 templates live in
cache/runs/queue/post_train/stage2/ (TEMPLATE_DIR), a folder that scripts/gpu_queue.py does not read (its template
glob is not recursive), because the runner expands templates on a phase B verify event only and a stage-2 run logs
phase C; written by --write-templates from stage2_template(). The expansion runs as a CPU queue job after the
training job (S2.11): it reads the run's init_stage2 and phase C verify events, fills {ckpt}, {name}, {run_dir},
{run}, {start_ckpt} and {start_name}, leaves out a job whose outputs all exist (the start's rows read
before), drops the names of jobs left out from the after and not_if lists of the others, and writes the job files
through scripts/queue_job.py (write_job). The set: the end's G3 dev, G3 diag and G1 dev (the read's inputs), the G3
PD half (scripts/eval_g3_pd_dev.py), G4 on the stroke fold and the SP3 families of the end and of the start, and the
read itself (a CPU job after the three deciding inputs). --attainability-jobs CKPT writes the start's second decodes
(and its first ones when absent), the inputs of --hash. --post-train-files RUN_DIR (S2.11) writes the same set as
job files into --out before the run starts (planned_post_train: the end is ckpt_C_<--end-step>.pt of RUN_DIR, the
start --start-ckpt; the end's jobs also wait for the training job --after, the read job carries the --second-* folders
and --expect-rule-sha256), each to be checked with scripts/queue_job.py --file F --dry-run and queued with --file F:
the runner expands templates on a phase B verify event only.

Smoke check (--smoke-check RUN_DIR --train-steps N; smoke_check, S2.11): the checks of a short phase C run before the
stage-2 training may start (lineage, config, verify event, no stop, finite losses and per-unit NLL, peak reserved
memory within the Config's bound, and the phase C rate projected over N steps against 24 GPU-hours times the
stage-1 run-to-run spread of reports/bench/stage2_budget.json); exit 0 when it passes, EXIT_SMOKE_FAILED otherwise;
--write rewrites reports/bench/stage2_read/smoke_<run>/results.json and summary.md.

Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --attainability [--start NAME] [--second-diag DIR ...] [--write]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --hash
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --attainability --amendment [--second-diag DIR ...] [--write]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --hash-amendment [--second-diag DIR ...]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --hash-amendment-2 [--second-diag DIR ...]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --hash-amendment-3 [--second-diag DIR ...]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --hash-amendment-4 [--second-diag DIR ...]
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --read --run-dir cache/runs/<run> [--write]
      [--expect-rule-sha256 HEX] [--expect-amendment-sha256 HEX] [--expect-amendment-2-sha256 HEX]
      [--expect-amendment-3-sha256 HEX] [--expect-amendment-4-sha256 HEX]
  python scripts/stage2_read.py --post-train cache/runs/<run> [--write]
  python scripts/stage2_read.py --attainability-jobs cache/runs/<start run>/ckpt_B_<step>.pt [--write]
  python scripts/stage2_read.py --write-templates
  python scripts/stage2_read.py --post-train-files cache/runs/<run> --end-step N --start-ckpt CKPT --after JOB
      --expect-rule-sha256 HEX [--expect-amendment-sha256 HEX] [--expect-amendment-2-sha256 HEX]
      [--expect-amendment-3-sha256 HEX] [--expect-amendment-4-sha256 HEX] [--second-diag DIR ...] --out FOLDER
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_read.py --smoke-check cache/runs/<smoke run> --train-steps N [--write]
Exit codes: EXIT_VERDICT (0) KEEP, FORGETS or STOPPED, or a command done; EXIT_NOT_EVALUATED (3); EXIT_HARNESS (4);
EXIT_EXISTS (5); EXIT_REFUSED (6) a hash refused or a read under an unhashed rule; EXIT_SMOKE_FAILED (7).
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

if __name__ == "__main__" and not {"--post-train", "--post-train-files"} & set(sys.argv[1:]):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # the readings never see the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
SCRIPTS = HERE

TASK = "S2.10"
LABEL = "dev diagnostic, not a G3 read"
BENCH = os.path.join("reports", "bench", "stage2_read")
RULE_REL = os.path.join(BENCH, "rule.txt")
RULE_SHA_REL = os.path.join(BENCH, "rule.sha256")
TEMPLATE_DIR = os.path.join("cache", "runs", "queue", "post_train", "stage2")
TEMPLATE_NAME = "post_stage2"
TEMPLATE_FORMAT = "mdfm-queue-template/1"
STAGE2_PLACEHOLDERS = ("ckpt", "name", "run_dir", "run", "start_ckpt", "start_name")
PYTHON = "<conda>/envs/mdfm/python.exe"
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
SEEDS_REL = os.path.join("reports", "gpu_queue", "branch_seeds_20261002_211426.json")
F1_RESULTS_REL = os.path.join("reports", "bench", "g3_gain_check", "results.json")
ATT_SUFFIX = "_att2"
LINES = ("S12", "V12", "S14", "V14", "MAPE", "IC", "G1a")
REPORTED = ("IC_insole",)
UNIT = {"S12": "cm", "V12": "m/s", "S14": "cm", "V14": "m/s", "MAPE": "percent", "IC": "ms", "G1a": "mm",
        "IC_insole": "ms"}
ND = {"S12": 2, "V12": 4, "S14": 2, "V14": 4, "MAPE": 2, "IC": 2, "G1a": 2, "IC_insole": 2}
NAMES = {"S12": "S(12)", "V12": "V(12)", "S14": "S(14)", "V14": "V(14)", "MAPE": "stride_mape_pct",
         "IC": "walkway IC", "G1a": "G1a", "IC_insole": "insole IC"}
G3_LINE_OF = {"MAPE": "stride_mape_pct", "IC": "ic_median_ms_walkway", "IC_insole": "ic_median_ms_insole"}
DIAG_LINES = {"S12": ("S", "twelve"), "V12": ("V", "twelve"), "S14": ("S", "fourteen"), "V14": ("V", "fourteen")}
FIVE_PERCENT, SEED_MULTIPLE, TOL = 0.05, 2.0, 1e-9
SETTINGS_IGNORED = ("memory",)
KEEP, FORGETS, NOT_EVALUATED, STOPPED = "KEEP", "FORGETS", "NOT EVALUATED", "STOPPED"
HASHED_AT = re.compile(r"^hashed_at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)$", re.M)
EXIT_VERDICT, EXIT_NOT_EVALUATED, EXIT_HARNESS, EXIT_EXISTS, EXIT_REFUSED = 0, 3, 4, 5, 6
EXIT_SMOKE_FAILED = 7
BUDGET_REL = os.path.join("reports", "bench", "stage2_budget.json")
WALL_BUDGET_H = 24.0            # rule (i) of spec section 6: a stage-2 run trains for at most 24 GPU-hours
END_DECODE_FILES = (("diag", "diag.json"), ("g3dev", "g3_dev.json"), ("g1", "g1.json"))
COHORT_PROBE = ("not built: no script reads the clinical_gait_multipatho group from frozen tokens of the start and "
                "the end (spec section 4.12); it is named here and read by nothing")
AMEND_TASK = "S2.11 review, amendment 1"
AMEND_REL = os.path.join(BENCH, "rule_amendment_1.txt")
AMEND_SHA_REL = os.path.join(BENCH, "rule_amendment_1.sha256")
AMEND_SUFFIX = "_amendment_1"
FLIP_EXACT_MAX, FLIPS, FLIP_SEED = 16, 10000, 20261001
AMEND2_TASK = "S2B, amendment 2"
AMEND2_REL = os.path.join(BENCH, "rule_amendment_2.txt")
AMEND2_SHA_REL = os.path.join(BENCH, "rule_amendment_2.sha256")
A2FIELD = re.compile(r"^(Status|Rule|Amendment 1|Start|Config|Run|Smoke): (.*)$", re.M)
AMEND3_TASK = "S2C, amendment 3"
AMEND3_REL = os.path.join(BENCH, "rule_amendment_3.txt")
AMEND3_SHA_REL = os.path.join(BENCH, "rule_amendment_3.sha256")
A3FIELD = re.compile(r"^(Status|Rule|Amendment 1|Amendment 2|Start|Config|Run|Smoke): (.*)$", re.M)
# amendment 3, section 1: its Config is amendment 2's with these canonical-JSON fields changed, and this reading
A3_CHANGES = ("name", "schedule.alarm_consecutive", "status")
A3_ALARM_CONSECUTIVE = {"beta_error": 3}
A3_ALARM = "beta_error"
BETA_ALARM_REL = os.path.join("reports", "bench", "stage2_beta_alarm", "results.json")
AMEND4_TASK = "S2D, amendment 4"
AMEND4_REL = os.path.join(BENCH, "rule_amendment_4.txt")
AMEND4_SHA_REL = os.path.join(BENCH, "rule_amendment_4.sha256")
A4FIELD = re.compile(r"^(Status|Rule|Amendment 1|Amendment 2|Amendment 3|Start|Config|Run|Smoke): (.*)$", re.M)
# amendment 4, section 1: its Config is amendment 3's with these canonical-JSON fields changed (config_differences
# names the key added to schedule.alarm_consecutive), and this reading
A4_CHANGES = ("name", "schedule.alarm_consecutive.unit_nll_rise", "status")
A4_ALARM_CONSECUTIVE = {"beta_error": 3, "unit_nll_rise": 2}
A4_ALARM = "unit_nll_rise"
UNIT_ALARM_REL = os.path.join("reports", "bench", "stage2_unit_nll_alarm", "results.json")
UNIT_FIRST_READ = 1000          # the first dev read that scripts/stage2_unit_nll_alarm.py models (its FIRST_READ)


class HarnessDefect(AssertionError):
    """A departure of the harness itself: margins that differ from the rule's printed lines, a person outside the
    expected set."""


def _sibling(name):
    key = f"_s2r_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def WB():
    """scripts/weekend_branch.py: G3 and G1 folder readers, the paired bootstrap, log helpers."""
    return _sibling("weekend_branch")


def F2():
    """scripts/f2_lever2.py: the dev sets, the person guard, the checkpoint check of a report."""
    return _sibling("f2_lever2")


def QJ():
    """scripts/queue_job.py: job and template format."""
    return _sibling("queue_job")


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def fin(x):
    if isinstance(x, bool) or x is None:
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
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_name(path):
    """'<run folder>_<file stem>' of a checkpoint path (mdfm.eval.infer.checkpoint_name, without the import)."""
    p = os.path.abspath(str(path).replace("/", os.sep))
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"


# ---------------------------------------------------------------- decodes
def decode_spec(tree, name, suffix="", diag=None, g3dev=None, g1=None):
    """{name, diag, g3dev, g1}: the report folders of one decode of the checkpoint `name` (default: the names of the
    post-train jobs, with `suffix`; a folder given replaces its default)."""
    g3, g1d = os.path.join(tree, "reports", "gates", "g3"), os.path.join(tree, "reports", "gates", "g1")
    return {"name": name,
            "diag": diag or os.path.join(g3, f"diag_{name}{suffix}"),
            "g3dev": g3dev or os.path.join(g3, f"dev_{name}{suffix}"),
            "g1": g1 or os.path.join(g1d, f"dev_{name}{suffix}")}


def settings_of(doc):
    """The inference entry of a report, the fields of SETTINGS_IGNORED left out (None when absent)."""
    inf = (doc or {}).get("inference")
    if not isinstance(inf, dict):
        return None
    return {k: v for k, v in sorted(inf.items()) if k not in SETTINGS_IGNORED}


def report_issue(doc, name, sha, what):
    """None, or why a report is not read: f2_lever2.checkpoint_issue (another checkpoint name, other weights, another
    SHA-256), and, when `sha` is given, a report whose checkpoint entry names no SHA-256 (a stale report of the same
    checkpoint name cannot pass)."""
    issue = F2().checkpoint_issue(doc, name, sha, what)
    if issue is None and sha and not ((doc or {}).get("checkpoint") or {}).get("sha256"):
        issue = f"{what}: no checkpoint SHA-256, expected {sha[:12]}"
    return issue


def _missing(lines, reason):
    return {k: {"point": None, "units": None, "weights": None, "source": None, "settings": None, "reason": reason}
            for k in lines}


def read_decode(tree, dec, sets, locked, sha=None):
    """{line: {point, units {person or take: value}, weights, source, settings, reason}} of one decode (LINES and
    REPORTED); the persons of every row read are guarded (dev14, dev12, no lockbox person)."""
    wb, f2 = WB(), F2()
    out = {}
    # the G3 diag base row: S and V on the 12 and the 14
    p = os.path.join(dec["diag"], "diag.json")
    doc = read_json(p)
    issue = None if doc is None else report_issue(doc, dec["name"], sha, rel(p, tree))
    if doc is None or issue:
        out.update(_missing(DIAG_LINES, issue or f"missing: {rel(p, tree)}"))
    else:
        s = wb.g3set_from_diag_row((doc.get("rows") or {}).get("base"), p)
        if s is None:
            out.update(_missing(DIAG_LINES, f"no base row in {rel(p, tree)}"))
        else:
            f2.guard_set(s, sets, locked, rel(p, tree))
            subs = {"twelve": wb.SUB12, "fourteen": wb.SUB14}
            for k, (key, sub) in DIAG_LINES.items():
                units, point = wb.g3_line(s, key, subs[sub])
                out[k] = {"point": fin(point), "units": units, "weights": None, "source": rel(p, tree),
                          "settings": settings_of(doc),
                          "reason": None if fin(point) is not None else f"no {k} in {rel(p, tree)}"}
    # g3_dev.json: stride_mape_pct and the IC lines, per person from per_person.csv (row dev, the model)
    keys = ("MAPE", "IC", "IC_insole")
    p = os.path.join(dec["g3dev"], "g3_dev.json")
    doc = read_json(p)
    issue = None if doc is None else report_issue(doc, dec["name"], sha, rel(p, tree))
    if doc is None or issue:
        out.update(_missing(keys, issue or f"missing: {rel(p, tree)}"))
    else:
        pts = {r.get("line"): fin(r.get("point")) for r in doc.get("results_dev") or [] if isinstance(r, dict)}
        pp = {}
        ppath = os.path.join(dec["g3dev"], "per_person.csv")
        if os.path.exists(ppath):
            with open(ppath, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("row") == "dev" and r.get("predictor") == "model":
                        pp.setdefault(r["line"], {})[r["person_id"]] = fin(r.get("value"))
        allowed = set(sets["dev14"])
        for k in keys:
            units = pp.get(G3_LINE_OF[k]) or {}
            if set(units) & set(locked):
                raise HarnessDefect(f"{rel(ppath, tree)}: a lockbox person in the dev row")
            if set(units) - allowed:
                raise HarnessDefect(f"{rel(ppath, tree)}: persons outside dev14 in the dev row")
            point = pts.get(G3_LINE_OF[k])
            out[k] = {"point": point, "units": units or None, "weights": None, "source": rel(p, tree),
                      "settings": settings_of(doc),
                      "reason": None if point is not None else f"no {G3_LINE_OF[k]} line in {rel(p, tree)}"}
    # G1a of the full G1 dev run, per take weighted by knots
    p = os.path.join(dec["g1"], "g1.json")
    doc = read_json(p)
    g = wb.g1_read(dec["g1"]) if doc is not None else None
    issue = None if doc is None else report_issue(doc, dec["name"], sha, rel(p, tree))
    if doc is None or g is None or issue:
        out.update(_missing(("G1a",), issue or f"missing: {rel(p, tree)}"))
    elif g["variant"] != "full":
        out.update(_missing(("G1a",), f"{rel(p, tree)} is a partial run ({g['variant']})"))
    else:
        out["G1a"] = {"point": fin(g["point"]), "units": g["takes"] or None, "weights": g["knots"] or None,
                      "source": rel(p, tree), "settings": settings_of(doc),
                      "reason": None if fin(g["point"]) is not None else f"no G1a line in {rel(p, tree)}"}
    return out


def input_hashes(tree, decodes):
    """{tree-relative path: SHA-256} of the files the decodes are read from and of the seeds file."""
    out = {}
    for dec in decodes:
        for key, files in (("diag", ("diag.json",)), ("g3dev", ("g3_dev.json", "per_person.csv")),
                           ("g1", ("g1.json", "windows.csv"))):
            for fn in files:
                p = os.path.join(dec[key], fn)
                if os.path.exists(p):
                    out[rel(p, tree)] = sha256_file(p)
    p = os.path.join(tree, SEEDS_REL)
    if os.path.exists(p):
        out[rel(p, tree)] = sha256_file(p)
    return dict(sorted(out.items()))


def paired_interval(a, b):
    """WB.paired of the per-unit values of two line reads (a minus b; weights for G1a), or None."""
    if not a.get("units") or not b.get("units"):
        return None
    p = WB().paired(a["units"], b["units"], a.get("weights"), b.get("weights"))
    return {k: p.get(k) for k in ("n", "point", "interval", "reason")}


def settings_issue(a, b, what):
    """None, or why two line reads ran under other inference settings."""
    if a.get("settings") is None or b.get("settings") is None:
        return f"{what}: no inference settings in a report"
    if a["settings"] != b["settings"]:
        diff = sorted(k for k in set(a["settings"]) | set(b["settings"])
                      if a["settings"].get(k) != b["settings"].get(k))
        return f"{what}: inference settings differ ({', '.join(diff)})"
    return None


# ---------------------------------------------------------------- margins
def seed_differences(tree, sets, locked):
    """{line: {d, source}}: the B-end seed difference of v1's recipe pair (module docstring), every line of LINES and
    REPORTED; FileNotFoundError when an input is absent (they exist before any stage-2 run)."""
    seeds = read_json(os.path.join(tree, SEEDS_REL))
    if seeds is None:
        raise FileNotFoundError(SEEDS_REL)
    out = {k: {"d": float(seeds["seed_diff"]["B_end"][k]["abs_diff"]), "source": SEEDS_REL.replace("\\", "/")}
           for k in DIAG_LINES}
    a = read_decode(tree, decode_spec(tree, V1_B), sets, locked)
    b = read_decode(tree, decode_spec(tree, S2_B), sets, locked)
    for k in ("MAPE", "IC", "IC_insole", "G1a"):
        if a[k]["point"] is None or b[k]["point"] is None:
            raise FileNotFoundError(a[k]["reason"] or b[k]["reason"])
        out[k] = {"d": abs(a[k]["point"] - b[k]["point"]), "source": f"|{a[k]['source']} - {b[k]['source']}|"}
    return out


def decode_interval_beyond(di, margin):
    """True when the paired interval of two decodes of the start reaches beyond the margin on either side (module
    docstring); None when that interval does not exist (fewer than 2 paired units)."""
    iv = (di or {}).get("interval")
    if not iv or fin(iv[0]) is None or fin(iv[1]) is None:
        return None
    return bool(max(abs(iv[0]), abs(iv[1])) > margin + TOL)


def attainability(tree, first, second, sets, locked, seeds=None):
    """The margins of the read and their attainability check (module docstring; spec section 10) for the start
    decoded twice (decode_spec dicts `first` and `second`): {start, lines {line: row}, attainable, reasons}."""
    a = read_decode(tree, first, sets, locked)
    b = read_decode(tree, second, sets, locked)
    sd = seeds or seed_differences(tree, sets, locked)
    rows, reasons = {}, []
    for k in LINES + REPORTED:
        x, y = a[k], b[k]
        row = {"line": x["point"], "unit": UNIT[k], "seed_diff": sd[k]["d"], "seed_source": sd[k]["source"],
               "first": x["source"], "second": y["source"], "second_point": y["point"], "decides": k in LINES}
        if x["point"] is None:
            row.update(checked=False, reason=f"first decode: {x['reason']}", margin=None)
        else:
            five, two = FIVE_PERCENT * x["point"], SEED_MULTIPLE * sd[k]["d"]
            row.update(five_percent=five, two_seed=two, form_margin=max(five, two))
            issue = None if y["point"] is None else settings_issue(x, y, NAMES[k])
            if y["point"] is None or issue:
                row.update(checked=False, reason=issue or f"second decode: {y['reason']}", margin=None,
                           decode_difference=None)
            else:
                dd = y["point"] - x["point"]
                m = max(row["form_margin"], abs(dd))
                di = paired_interval(y, x)
                row.update(checked=True, reason=None, decode_difference=dd, decode_interval=di,
                           margin=m, raised=bool(abs(dd) > row["form_margin"]),
                           decode_interval_beyond=decode_interval_beyond(di, m))
        if k in LINES and not row["checked"]:
            reasons.append(f"{NAMES[k]}: {row['reason']}")
        rows[k] = row
    return {"start": first["name"], "first": {k: rel(v, tree) for k, v in first.items() if k != "name"},
            "second": {k: rel(v, tree) for k, v in second.items() if k != "name"}, "lines": rows,
            "five_percent": FIVE_PERCENT, "seed_multiple": SEED_MULTIPLE, "seed_pair": [V1_B, S2_B],
            "attainable": not reasons, "reasons": reasons,
            "decode_interval_beyond": [k for k in LINES if rows[k].get("decode_interval_beyond")]}


def fmt(x, nd=2, sign=False):
    v = fin(x)
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    if not iv or iv[0] is None:
        return "n/a"
    return f"[{fmt(iv[0], nd, True)}, {fmt(iv[1], nd, True)}]"


def section7_lines(att):
    """The lines that the rule's section 7 must hold verbatim before the hash: one per line of LINES."""
    out = []
    for k in LINES:
        r = att["lines"][k]
        nd, u = ND[k] + 1, UNIT[k]
        if not r.get("checked"):
            out.append(f"- {NAMES[k]}: not checked ({r.get('reason')})")
            continue
        iv = (r.get("decode_interval") or {}).get("interval")
        out.append(f"- {NAMES[k]}: start {fmt(r['line'], nd)} {u}; 5 percent {fmt(r['five_percent'], nd)} {u}; "
                   f"seed difference {fmt(r['seed_diff'], nd)} {u}, twice {fmt(r['two_seed'], nd)} {u}; decode "
                   f"difference {fmt(r['decode_difference'], nd, True)} {u}, paired interval {fmt_iv(iv, nd)}; "
                   f"margin {fmt(r['margin'], nd)} {u}" + (" (raised to the decode difference)" if r["raised"] else "")
                   + ("; the paired interval of the two decodes reaches beyond the margin"
                      if r.get("decode_interval_beyond") else ""))
    return out


def attainability_text(att):
    """Every margin with its terms, the decode difference and its paired interval, then the result."""
    L = [f"Attainability of the stage-2 read margins (spec section 10), start {att['start']}",
         f"first decode: {att['first']}", f"second decode: {att['second']}",
         f"form: max({att['five_percent']:g} x the start's line, {att['seed_multiple']:g} x the B-end seed difference "
         f"of {att['seed_pair'][0]} against {att['seed_pair'][1]}); a margin under the absolute decode difference is "
         "raised to it", ""]
    L += section7_lines(att)
    r = att["lines"]["IC_insole"]
    L.append(f"- reported, deciding nothing: {NAMES['IC_insole']} start {fmt(r['line'], 3)} ms, seed difference "
             f"{fmt(r['seed_diff'], 3)} ms, decode difference {fmt(r.get('decode_difference'), 3, True)} ms")
    L += ["", "attainable: every line has two decodes at equal settings" if att["attainable"] else
          "NOT attainable: " + "; ".join(att["reasons"])]
    if att.get("decode_interval_beyond"):
        L.append("decode noise: on " + ", ".join(NAMES[k] for k in att["decode_interval_beyond"])
                 + " the paired interval of the two decodes of the start reaches beyond the margin, so decoding "
                 "alone can carry the point past it and a model equal to its start may read FORGETS there; the "
                 "point decides (spec section 10), this flag decides nothing and is the owner's to read before the "
                 "hash")
    return "\n".join(L)


# ---------------------------------------------------------------- the rule
FIELD = re.compile(r"^(Status|Start|Config): (.*)$", re.M)
PATH_SHA = re.compile(r"^(\S+) SHA-256 ([0-9a-f]{64})$")


def rule_fields(text):
    """{status, start, config} of the rule's three field lines; start and config {path, sha256} or None."""
    got = {}
    for k, v in FIELD.findall(text):
        got.setdefault(k.lower(), v.strip())
    out = {"status": got.get("status"), "start": None, "config": None}
    for k in ("start", "config"):
        m = PATH_SHA.match(got.get(k) or "")
        if m:
            out[k] = {"path": m.group(1), "sha256": m.group(2)}
    return out


def section7(text):
    nl = chr(10)
    i = text.find(nl + "7. ")
    j = text.find(nl + "8. ")
    return text[i:j] if 0 <= i < j else ""


def rule_state(tree):
    """{text, sha256, fields, hashed, stored, hashed_at, hash_mtime}: the rule file and its hash file; hashed when
    rule.sha256 exists and names the file's SHA-256 (a hash file that names another raises); hashed_at the time that
    --hash recorded in it (None when absent), hash_mtime the file's mtime as a local ISO time (reported only)."""
    p = os.path.join(tree, RULE_REL)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    sha = sha256_file(p)
    sp = os.path.join(tree, RULE_SHA_REL)
    stored = hashed_at = mtime = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            body = f.read()
        words = body.split()
        stored = words[0] if words else ""
        if stored != sha:
            raise HarnessDefect(f"rule.sha256 names {stored[:12]}, the rule hashes {sha[:12]}")
        m = HASHED_AT.search(body)
        hashed_at = m.group(1) if m else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(sp)).isoformat(timespec="seconds")
    return {"text": text, "sha256": sha, "fields": rule_fields(text), "hashed": stored is not None, "stored": stored,
            "hashed_at": hashed_at, "hash_mtime": mtime}


def start_decodes(tree, fields, second=None):
    """(first, second) decode_spec of the rule's start (second: folders given, else the ATT_SUFFIX names)."""
    name = checkpoint_name(fields["start"]["path"])
    first = decode_spec(tree, name)
    sec = decode_spec(tree, name, ATT_SUFFIX, **(second or {}))
    return first, sec


def hash_refusals(tree, sets, locked, second=None):
    """(reasons the rule may not be hashed, attainability of its start or None)."""
    st = rule_state(tree)
    f = st["fields"]
    reasons = []
    if f["status"] != "final":
        reasons.append(f"Status is {f['status']!r}, not 'final'")
    if f["start"] is None:
        reasons.append("the Start line names no checkpoint with its SHA-256")
    if f["config"] is None:
        reasons.append("the Config line names no config with its SHA-256")
    if st["hashed"]:
        reasons.append("rule.sha256 exists: the rule is hashed already")
    att = None
    if f["start"] is not None:
        att = attainability(tree, *start_decodes(tree, f, second), sets, locked)
        if not att["attainable"]:
            reasons += [f"not attainable: {x}" for x in att["reasons"]]
        sec = section7(st["text"])
        reasons += [f"section 7 lacks the line: {x}" for x in section7_lines(att) if x not in sec]
    return reasons, att


def write_hash(tree, sets, locked, second=None, now=None):
    """rule.sha256 written when hash_refusals finds nothing, with the local time of the hash (`now`, a datetime, for
    the tests); returns (written, reasons, attainability)."""
    reasons, att = hash_refusals(tree, sets, locked, second)
    if reasons:
        return False, reasons, att
    st = rule_state(tree)
    stamp = (now or datetime.datetime.now()).isoformat(timespec="seconds")
    with open(os.path.join(tree, RULE_SHA_REL), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{st['sha256']}  rule.txt\nhashed_at {stamp}\n")
    return True, [], att


def hash_time_issue(st, recs):
    """None, or why the rule's recorded hash time does not precede the run's first log line (module docstring):
    no recorded time, a time not parsed, or a time at or after that line (both local times, to the second)."""
    first = next((r.get("time") for r in recs or [] if r.get("time")), None)
    if not st.get("hashed_at"):
        return "rule.sha256 records no hash time (a line 'hashed_at <ISO time>' written by --hash)"
    if first is None:
        return None
    try:
        h = datetime.datetime.fromisoformat(st["hashed_at"])
        t = datetime.datetime.fromisoformat(str(first)[:19])
    except ValueError as e:
        return f"a time not parsed: {e}"
    if h >= t:
        return f"rule.sha256 records a hash at {st['hashed_at']}, not before the run's first log line {first}"
    return None


# ---------------------------------------------------------------- R0
def r0_reading(recs, fields, unparsed=0, tree=None, run_dir=None, job_log=None):
    """R0 of the read (module docstring): {holds, stop [reasons], fail [reasons], checks, values, end {path, sha256,
    name, step}}; stop: the run stopped on an alarm, a non-finite stop or the memory bound; fail: every other R0
    failure. The job_log.csv check reads `job_log` (default the tree's reports/gpu_queue/job_log.csv)."""
    wb = WB()
    stop, fail, checks, values = [], [], {}, {"unparsed_log_lines": unparsed}
    if recs is None:
        return {"holds": False, "stop": stop, "fail": ["no run log"], "checks": checks, "values": values,
                "end": None}
    inits = wb.log_events(recs, "init_stage2")
    values["init_stage2_events"] = len(inits)
    start = fields.get("start") or {}
    bad = [str((r.get("source") or {}).get("path")) for r in inits
           if not str((r.get("source") or {}).get("path") or "").replace("\\", "/").endswith(start.get("path") or "\0")
           or (r.get("source") or {}).get("sha256") != start.get("sha256")]
    checks["init_stage2"] = bool(inits) and not bad
    if not inits:
        fail.append("no init_stage2 event")
    elif bad:
        fail.append("init_stage2 source other than the rule's start: " + "; ".join(bad))
    starts = wb.log_events(recs, "start")
    cfg = (fields.get("config") or {}).get("sha256")
    bad_cfg = sorted({str(s.get("config_sha256")) for s in starts if s.get("config_sha256") != cfg})
    checks["start_config"] = bool(starts) and not bad_cfg
    if not starts:
        fail.append("no start event")
    elif bad_cfg:
        fail.append("start event at another config SHA-256: " + ", ".join(x[:12] for x in bad_cfg))
    steps = None
    for r in reversed(inits + starts):
        st = r.get("steps")
        if isinstance(st, list) and len(st) >= 3 and fin(st[2]) is not None:
            steps = int(st[2])
            break
    values["phase_c_steps"] = steps
    v = wb.last_verify(recs, "C")
    end = None
    if v is not None:
        path = str(v.get("path") or "").replace("\\", "/")
        end = {"path": path, "sha256": v.get("sha256"), "name": checkpoint_name(path) if path else None,
               "step": v.get("step")}
    checks["verify_C"] = bool(v is not None and steps is not None and v.get("step") == steps)
    if not checks["verify_C"]:
        fail.append(f"no phase C verify event at step {steps}")
    mism = [s.get("report_mismatches") for s in starts if s.get("report_mismatches")]
    if v is not None and v.get("report_mismatches") != []:
        mism.append(v.get("report_mismatches", "absent from the verify event"))
    checks["report_mismatches_empty"] = not mism
    if mism:
        fail.append(f"report_mismatches not empty: {mism[0]}")
    # the stops (A2 case 1)
    alarms = wb.log_events(recs, "alarm")
    values["alarm_events"] = len(alarms)
    checks["no_alarm"] = not alarms
    if alarms:
        names = sorted({str(h.get("name")) for a in alarms for h in (a.get("alarms") or []) if isinstance(h, dict)})
        stop.append(f"alarm ({len(alarms)} alarm events" + (f": {', '.join(names)}" if names else "") + ")")
    run = F2().NONFINITE_RUN
    values["nonfinite_events"] = len(wb.log_events(recs, "nonfinite"))
    checks["no_nonfinite_stop"] = not wb.nonfinite_stop(recs, run)
    if not checks["no_nonfinite_stop"]:
        stop.append(f"non-finite stop ({run} nonfinite events at consecutive optimizer steps)")
    mem = wb.log_events(recs, "memory_exceeded")
    values["peak_reserved_gb"] = wb.peak_reserved_gb(recs)
    values["memory_exceeded_events"] = len(mem)
    checks["no_memory_stop"] = not mem
    if mem:
        stop.append(f"memory bound (memory_exceeded at step {mem[-1].get('step')}, bound "
                    f"{mem[-1].get('bound_gb')} GB)")
    # the training job of the run folder in the queue's job log
    code, jpath = wb.job_exit(tree or TREE, job_log, run_dir or "")
    values["job_exit"] = code
    values["job_log"] = rel(jpath, tree) if jpath else None
    checks["job_exit_0"] = code == 0
    if code is None:
        fail.append(f"no job_log.csv row of a scripts/train_stage1.py job on {run_dir}")
    elif code != 0:
        fail.append(f"last train_stage1.py job on {run_dir} exited {code}")
    return {"holds": not stop and not fail, "stop": stop, "fail": fail, "checks": checks, "values": values,
            "end": end}


# ---------------------------------------------------------------- lines and verdict
def lines_reading(end, start, att):
    """{line: {read, start, end, difference, interval, margin, holds, reason}} for LINES and REPORTED: the point
    end minus start against the margin of the attainability check; the paired interval beside it."""
    out = {}
    for k in LINES + REPORTED:
        a, b = end.get(k) or {}, start.get(k) or {}
        margin = att["lines"][k].get("margin") if k in LINES else None
        row = {"read": False, "start": b.get("point"), "end": a.get("point"), "difference": None, "interval": None,
               "margin": margin, "holds": None, "unit": UNIT[k], "decides": k in LINES, "reason": None}
        if a.get("point") is None or b.get("point") is None:
            row["reason"] = a.get("reason") or b.get("reason") or f"no {NAMES[k]} value"
        elif settings_issue(a, b, NAMES[k]):
            row["reason"] = settings_issue(a, b, NAMES[k])
        elif k in LINES and margin is None:
            row["reason"] = f"no margin: {att['lines'][k].get('reason')}"
        else:
            row["read"] = True
            row["difference"] = a["point"] - b["point"]
            row["interval"] = paired_interval(a, b)
            if k in LINES:
                row["holds"] = bool(row["difference"] <= margin + TOL)
            else:
                row["seed_diff"] = att["lines"][k].get("seed_diff")
        out[k] = row
    return out


def verdict(r0, lines):
    """The verdict of the read (module docstring, A2's order): {verdict, case, reasons, missing}."""
    if r0.get("stop"):
        return {"verdict": STOPPED, "case": 1, "reasons": list(r0["stop"]), "missing": []}
    if r0.get("fail") or not r0["holds"]:
        fail = list(r0.get("fail") or ["R0 does not hold"])
        return {"verdict": NOT_EVALUATED, "case": 2, "reasons": fail, "missing": fail}
    missing = [f"{NAMES[k]}: {lines[k]['reason']}" for k in LINES if not lines[k]["read"]]
    forgets = [k for k in LINES if lines[k]["read"] and not lines[k]["holds"]]
    if forgets:
        return {"verdict": FORGETS, "case": 3, "reasons": [f"{NAMES[k]} FORGETS" for k in forgets],
                "missing": missing}
    if missing:
        return {"verdict": NOT_EVALUATED, "case": 4, "reasons": missing, "missing": missing}
    return {"verdict": KEEP, "case": 5, "reasons": [f"every line HOLDS ({len(LINES)})"], "missing": []}


# ---------------------------------------------------------------- reported beside
def _beside_doc(tree, p, name, sha):
    """(doc, None) of a reported-beside file, or (None, why it is not read): missing, or report_issue."""
    doc = read_json(p)
    if doc is None:
        return None, f"missing: {rel(p, tree)}"
    issue = report_issue(doc, name, sha, rel(p, tree))
    return (None, issue) if issue else (doc, None)


def pd_half(tree, name, sha=None):
    p = os.path.join(tree, "reports", "gates", "g3", f"dev_pd_{name}", "g3_pd_dev.json")
    doc, why = _beside_doc(tree, p, name, sha)
    if doc is None:
        return {"read": False, "reason": why}
    model =(((doc.get("lines") or {}).get("pd_t19dev") or {}).get("model")) or {}
    return {"read": True, "source": rel(p, tree), "row": "pd_t19dev",
            "n_persons": ((doc.get("rows") or {}).get("pd_t19dev") or {}).get("n_persons"),
            "excluded_trained": len((doc.get("persons") or {}).get("excluded_trained") or []),
            "lines": {k: {"point": fin((v or {}).get("point")), "n_persons": (v or {}).get("n_persons")}
                      for k, v in model.items()}}


def g4_fold(tree, name, sha=None):
    p = os.path.join(tree, "reports", "gates", "g4", f"dev_{name}_strokefold", "g4.json")
    doc, why = _beside_doc(tree, p, name, sha)
    if doc is None:
        return {"read": False, "reason": why}
    s = (doc.get("sets") or {})
    if (s.get("stroke_set") or {}).get("set") != "fold":
        return {"read": False, "reason": f"{rel(p, tree)} is not a stroke fold run"}
    return {"read": True, "source": rel(p, tree), "stroke_dev": s.get("stroke_dev"),
            "stroke_scored": s.get("stroke_scored"),
            "lines": {r.get("line"): {"point": fin(r.get("point")), "n_persons": r.get("n_persons")}
                      for r in doc.get("lines") or [] if isinstance(r, dict)},
            "metric3d_mpjpe_mm": fin(((doc.get("metric3d") or {}).get("mpjpe_mm") or {}).get("point"))}


def sp3(tree, name, sha=None):
    p = os.path.join(tree, "reports", "synth", "sp3", name, "summary.json")
    doc, why = _beside_doc(tree, p, name, sha)
    if doc is None:
        return {"read": False, "reason": why}
    return {"read": True, "source": rel(p, tree),
            "families": {k: {"reading": (v or {}).get("reading"), "prior": (v or {}).get("prior")}
                         for k, v in sorted((doc.get("families") or {}).items())}}


def units_of(recs):
    """The per-unit numbers of the last phase C dev event: {step, unit_nll_rise, units {unit: {held_out_nll,
    train_nll, held_out_outside_share, train_outside_share}}}."""
    devs = [r for r in WB().log_events(recs or [], "dev", "C")
            if ((r.get("dev") or {}).get("model") or {}).get("units")]
    if not devs:
        return {"read": False, "reason": "no phase C dev event with unit numbers"}
    r = devs[-1]
    m = r["dev"]["model"]
    out = {}
    for u, e in sorted(m["units"].items()):
        ho, tr = e.get("held_out") or {}, e.get("train") or {}
        out[u] = {"held_out_nll": fin(ho.get("nll")), "train_nll": fin(tr.get("nll")),
                  "held_out_outside_share": fin(ho.get("outside_limits_share")),
                  "train_outside_share": fin(tr.get("outside_limits_share"))}
    return {"read": True, "step": r.get("step"), "unit_nll_rise": m.get("unit_nll_rise"), "units": out}


def reported_beside(tree, end_name, start_name, recs, end_sha=None, start_sha=None):
    out = {}
    for key, fn in (("pd_half", pd_half), ("g4_stroke_fold", g4_fold), ("sp3", sp3)):
        out[key] = {"end": fn(tree, end_name, end_sha) if end_name else {"read": False, "reason": "no end (R0)"},
                    "start": fn(tree, start_name, start_sha)}
    out.update(units=units_of(recs), cohort_probe={"read": False, "reason": COHORT_PROBE})
    return out


# ---------------------------------------------------------------- the smoke check (S2.11)
def clean_intervals(recs, phase):
    """[(step0, step1, seconds)] between consecutive 'log' events of `phase` within one call, the first interval of
    a call and every interval holding a dev or checkpoint step left out; seconds from wall_s of the closing event
    (the rule of scripts/gpu_queue.py clean_intervals, by which the runner times a training run)."""
    marks = {(r.get("phase"), r.get("step")) for r in recs if r.get("event") in ("dev", "checkpoint")}
    out, prev = [], None
    for r in recs:
        ev = r.get("event")
        if ev in ("start", "resume"):
            prev = None
            continue
        if ev != "log" or r.get("phase") != phase:
            continue
        if prev is not None and r.get("step", 0) > prev.get("step", 0):
            s0, s1 = int(prev["step"]), int(r["step"])
            secs = fin(r.get("wall_s"))
            if not any((phase, k) in marks for k in range(s0, s1)) and secs is not None and secs > 0:
                out.append((s0, s1, secs))
        prev = r
    return out


def run_spread(budget):
    """(largest over smallest s_per_step of the stage-1 whole phase B runs of the budget, their labels): the
    run-to-run spread of one phase on this PC (reports/bench/stage2_budget.json, rates of kind stage1_run)."""
    runs = [r for r in (budget or {}).get("rates") or [] if r.get("kind") == "stage1_run" and fin(r.get("s_per_step"))]
    if len(runs) < 2:
        return None, []
    v = [float(r["s_per_step"]) for r in runs]
    return max(v) / min(v), [r.get("label") for r in runs]


def smoke_check(tree, run_dir, train_steps, budget=None):
    """The checks of a phase C smoke run (S2.11) from its log.jsonl, before the stage-2 training is let run:
    {passes, reasons, checks, values, rule}. Checks: an init_stage2 event naming the rule's Start at its SHA-256;
    start events at the rule's Config SHA-256; a phase C verify event; no report mismatch in a start or the verify
    event; no alarm event, no memory_exceeded event, no non-finite stop (F2's NONFINITE_RUN nonfinite events at
    consecutive steps); every phase C 'log' loss finite; the peak reserved memory of the log present and at most
    the Config's schedule.max_reserved_gb; at the last phase C dev event every unit of the Config with a finite
    train NLL, and a finite held-out NLL wherever a dev event gave the unit a held-out entry; the rate: seconds per
    optimizer step over the clean intervals (the runner's rule), and the projected training of train_steps steps at
    that rate at most WALL_BUDGET_H times the stage-1 run-to-run spread (run_spread of the budget). The step count
    was set at a stage-1 rate under rule (i) of spec section 6; a phase C rate inside that spread is not
    distinguishable from it, while one beyond it calls for a new budget before 24 GPU-hours are spent. Nonfinite
    events that stop nothing are counted, not refused (a skipped optimizer step)."""
    wb = WB()
    st = rule_state(tree)
    f = st["fields"]
    reasons, checks = [], {}
    values = {"run_dir": run_dir.replace("\\", "/"), "train_steps": int(train_steps)}
    if not st["hashed"] or f["status"] != "final" or f["start"] is None or f["config"] is None:
        return {"passes": False, "reasons": ["the rule is not hashed and final with its Start and Config"],
                "checks": checks, "values": values, "rule": {"sha256": st["sha256"], "fields": f}}
    recs, unparsed = wb.read_jsonl(os.path.join(tree, run_dir, "log.jsonl"))
    values["unparsed_log_lines"] = unparsed
    if not recs:
        return {"passes": False, "reasons": ["no run log"], "checks": checks, "values": values,
                "rule": {"sha256": st["sha256"], "fields": f}}
    a3 = a4 = None
    try:                                     # a run that amendment 2, 3 or 4 names reads its Config (S2B, S2C, S2D)
        f2, a2, a3, a4 = run_binding4(tree, st, run_dir, recs)
    except HarnessDefect as e:
        f2, a2 = None, {"read": False, "reason": f"harness defect: {e}"}
    if f2 is None:
        rule = {"sha256": st["sha256"], "fields": f, "amendment_2": a2}
        if a3 is not None:
            rule["amendment_3"] = a3
        if a4 is not None:
            rule["amendment_4"] = a4
        return {"passes": False, "reasons": [(a4 or a3 or a2)["reason"]], "checks": checks, "values": values,
                "rule": rule}
    f = f2
    inits = wb.log_events(recs, "init_stage2")
    src = (inits[-1].get("source") or {}) if inits else {}
    checks["init_stage2"] = bool(inits) and str(src.get("path") or "").replace("\\", "/").endswith(
        f["start"]["path"]) and src.get("sha256") == f["start"]["sha256"]
    if not checks["init_stage2"]:
        reasons.append("no init_stage2 event naming the rule's Start at its SHA-256")
    starts = wb.log_events(recs, "start")
    checks["start_config"] = bool(starts) and all(s.get("config_sha256") == f["config"]["sha256"] for s in starts)
    if not checks["start_config"]:
        reasons.append("no start event, or one at another config SHA-256 than the rule's Config")
    v = wb.last_verify(recs, "C")
    values["verify_step"] = None if v is None else v.get("step")
    mism = [s.get("report_mismatches") for s in starts if s.get("report_mismatches")]
    if v is not None and v.get("report_mismatches") != []:
        mism.append(v.get("report_mismatches", "absent from the verify event"))
    checks["verify_C"] = v is not None
    checks["report_mismatches_empty"] = not mism
    if v is None:
        reasons.append("no phase C verify event")
    if mism:
        reasons.append(f"report_mismatches not empty: {mism[0]}")
    alarms = wb.log_events(recs, "alarm")
    mem = wb.log_events(recs, "memory_exceeded")
    values["alarm_events"], values["memory_exceeded_events"] = len(alarms), len(mem)
    if a3 is not None or a4 is not None:     # amendments 3 and 4: counted beside, deciding nothing
        values["alarm_watch_events"] = len(alarm_watch_events(recs))
    values["nonfinite_events"] = len(wb.log_events(recs, "nonfinite"))
    checks["no_alarm"], checks["no_memory_stop"] = not alarms, not mem
    checks["no_nonfinite_stop"] = not wb.nonfinite_stop(recs, F2().NONFINITE_RUN)
    for ok, why in ((checks["no_alarm"], f"{len(alarms)} alarm events"),
                    (checks["no_memory_stop"], f"{len(mem)} memory_exceeded events"),
                    (checks["no_nonfinite_stop"], "a non-finite stop")):
        if not ok:
            reasons.append(why)
    logs = wb.log_events(recs, "log", "C")
    values["log_events"] = len(logs)
    checks["losses_finite"] = bool(logs) and all(fin(r.get("loss")) is not None for r in logs)
    if not checks["losses_finite"]:
        reasons.append("no phase C log event, or a non-finite logged loss")
    from mdfm.train import config as CF
    cfg = CF.load(os.path.join(tree, f["config"]["path"]))
    bound = float(cfg.schedule.max_reserved_gb)
    peak = wb.peak_reserved_gb(recs)
    values["peak_reserved_gb"], values["max_reserved_gb"] = peak, bound
    checks["peak_memory"] = peak is not None and peak <= bound
    if not checks["peak_memory"]:
        reasons.append(f"peak reserved memory {peak} GB absent or above {bound} GB")
    u = units_of(recs)
    values["units"] = u
    held = set()
    for r in wb.log_events(recs, "dev", "C"):
        for k, e in (((r.get("dev") or {}).get("model") or {}).get("units") or {}).items():
            if (e or {}).get("held_out"):
                held.add(k)
    got = u.get("units") or {}
    bad = sorted({k for k, x in got.items() if x["train_nll"] is None or (k in held and x["held_out_nll"] is None)})
    checks["unit_nll_finite"] = bool(u.get("read")) and set(got) == set(cfg.stage2.units) and not bad
    if not checks["unit_nll_finite"]:
        reasons.append("per-unit dev NLL absent, not of every unit of the Config, or not finite"
                       + (": " + ", ".join(bad) if bad else ""))
    iv = clean_intervals(recs, "C")
    steps = sum(b - a for a, b, _ in iv)
    rate = sum(x for _, _, x in iv) / steps if steps else None
    spread, labels = run_spread(budget if budget is not None else read_json(os.path.join(tree, BUDGET_REL)))
    values.update(rate_steps=steps, s_per_step=rate, spread=spread, spread_runs=labels,
                  projected_hours=None if rate is None else int(train_steps) * rate / 3600.0,
                  bound_hours=None if spread is None else WALL_BUDGET_H * spread)
    checks["rate"] = rate is not None and spread is not None and values["projected_hours"] <= values["bound_hours"]
    if not checks["rate"]:
        reasons.append("no clean interval or no stage-1 spread" if rate is None or spread is None else
                       f"projected training {values['projected_hours']:.2f} h at {rate:.4f} s per step, above "
                       f"{values['bound_hours']:.2f} h ({WALL_BUDGET_H:g} h x the stage-1 spread {spread:.4f})")
    rule = {"sha256": st["sha256"], "fields": f, "amendment_2": a2}
    if a3 is not None:
        rule["amendment_3"] = a3
    if a4 is not None:
        rule["amendment_4"] = a4
    return {"passes": not reasons, "reasons": reasons, "checks": checks, "values": values, "rule": rule}


def smoke_md(res):
    v = res["values"]
    a2 = (res.get("rule") or {}).get("amendment_2")
    a3 = (res.get("rule") or {}).get("amendment_3")
    a4 = (res.get("rule") or {}).get("amendment_4")
    L = [f"# Phase C smoke check ({LABEL})", "",
         f"Run {v.get('run_dir')}: {'PASSES' if res['passes'] else 'FAILS'}. Rule SHA-256 "
         f"{(res.get('rule') or {}).get('sha256')}."
         + (f" Amendment 2 SHA-256 {a2.get('sha256')}: the Config {(a2.get('config') or {}).get('path')} SHA-256 "
            f"{(a2.get('config') or {}).get('sha256')}." if a2 and a2.get("read") else "")
         + (f" Amendment 3 SHA-256 {a3.get('sha256')}: the Config {(a3.get('config') or {}).get('path')} SHA-256 "
            f"{(a3.get('config') or {}).get('sha256')}." if a3 and a3.get("read") else "")
         + (f" Amendment 4 SHA-256 {a4.get('sha256')}: the Config {(a4.get('config') or {}).get('path')} SHA-256 "
            f"{(a4.get('config') or {}).get('sha256')}." if a4 and a4.get("read") else ""), ""]
    L += [f"- {k}: {'yes' if ok else 'no'}" for k, ok in res["checks"].items()]
    L += [f"- reason: {x}" for x in res["reasons"]]
    L += ["", f"Peak reserved memory {fmt(v.get('peak_reserved_gb'), 2)} GB (bound {fmt(v.get('max_reserved_gb'), 1)} "
          f"GB); nonfinite events {v.get('nonfinite_events')}; rate {fmt(v.get('s_per_step'), 4)} s per optimizer "
          f"step over {v.get('rate_steps')} steps; {v.get('train_steps')} steps project to "
          f"{fmt(v.get('projected_hours'), 2)} h against {fmt(v.get('bound_hours'), 2)} h."]
    u = v.get("units") or {}
    if u.get("read"):
        L.append(f"Per-unit dev NLL at step {u['step']}: " + "; ".join(
            f"{k} train {fmt(x['train_nll'], 4)}, held out {fmt(x['held_out_nll'], 4)}" for k, x in u["units"].items()))
    return "\n".join(L) + "\n"


def write_smoke(out_dir, res, tree):
    """results.json and summary.md of a smoke check (rewritten at each check: a check decides a launch, not a
    verdict)."""
    from mdfm.gates import common
    text = smoke_md(res)
    common.assert_no_em_dash(text, "smoke summary.md")
    common.write_text(os.path.join(out_dir, "results.json"),
                      json.dumps(clean(dict(res, task="S2.11", label=LABEL,
                                            run_at=time.strftime("%Y-%m-%dT%H:%M:%S"))), indent=1,
                                 allow_nan=False) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    return text


# ---------------------------------------------------------------- the read
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def decode_noise(v, lines, att):
    """The deciding lines that read FORGETS while the attainability check flagged their decode noise (the paired
    interval of the start's two decodes reaches beyond the margin): option (a) of the rule's section 7 names each
    one as possibly decode noise beside the verdict, which it does not change."""
    if v.get("verdict") != FORGETS:
        return []
    flagged = set(att.get("decode_interval_beyond") or [])
    return [k for k in LINES if lines[k]["read"] and not lines[k]["holds"] and k in flagged]


# ---------------------------------------------------------------- amendment 1: the second reading (S2.11 review)
def decode_bound(row):
    """The largest absolute bound of the paired interval of the start's two decodes in an attainability row, None
    when that interval does not exist."""
    iv = ((row or {}).get("decode_interval") or {}).get("interval")
    if not iv or fin(iv[0]) is None or fin(iv[1]) is None:
        return None
    return max(abs(iv[0]), abs(iv[1]))


def paired_differences(a, b):
    """(ids, d, w) of two line reads as WB.paired pairs them: the units both hold a finite value for, d = a - b,
    w the mean of both sides' weights (ones without weights)."""
    ua, ub = a.get("units") or {}, b.get("units") or {}
    ids = sorted(k for k in set(ua) & set(ub) if fin(ua[k]) is not None and fin(ub[k]) is not None)
    d = np.array([ua[k] - ub[k] for k in ids], dtype=float)
    wa, wb = a.get("weights"), b.get("weights")
    if wa is not None or wb is not None:
        w = np.array([0.5 * ((wa or {}).get(k, 1.0) + (wb or {}).get(k, (wa or {}).get(k, 1.0))) for k in ids])
    else:
        w = np.ones(len(ids))
    return ids, d, w


def null_share(d, w, margin):
    """The share of sign flips of the per-unit decode differences d (weights w) whose weighted mean exceeds the
    margin (positive is worse, TOL as the read): what a model equal to its start reads FORGETS on, when decoding
    is all that moves it. Every flip with at most FLIP_EXACT_MAX units, else FLIPS draws of default_rng(FLIP_SEED)."""
    d, w = np.asarray(d, dtype=float), np.asarray(w, dtype=float)
    n = len(d)
    if n == 0:
        return None
    if n <= FLIP_EXACT_MAX:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(FLIP_SEED).choice(np.array([-1, 1]), size=(FLIPS, n))
    means = (signs * d * w).sum(axis=1) / w.sum()
    return float((means > margin + TOL).mean())


def second_reading(tree, first, second, sets, locked, att):
    """Amendment 1 (module docstring): {line: {margin, decode_bound, margin2, raised2, n, null_share, null_share2}}
    of the deciding lines: margin2 = max(margin, the largest absolute bound of the start's two-decode paired
    interval); the null shares at the margin and at margin2 from the start's decodes `first` and `second`."""
    a = read_decode(tree, first, sets, locked)
    b = read_decode(tree, second, sets, locked)
    out = {}
    for k in LINES:
        r = att["lines"][k]
        m, h = r.get("margin"), decode_bound(r)
        row = {"margin": m, "decode_bound": h, "margin2": None, "raised2": None, "n": None, "null_share": None,
               "null_share2": None}
        if m is not None:
            raised = bool(h is not None and h > m + TOL)
            row.update(margin2=h if raised else m, raised2=raised)
            _, d, w = paired_differences(b[k], a[k])
            if len(d) >= 2:
                row.update(n=int(len(d)), null_share=null_share(d, w, m), null_share2=null_share(d, w, row["margin2"]))
        out[k] = row
    return out


def _pct(x):
    return "n/a" if fin(x) is None else f"{100.0 * x:.1f}"


def amendment_lines(sr):
    """The lines amendment 1 holds verbatim before its hash, one per deciding line."""
    out = []
    for k in LINES:
        r, nd, u = sr[k], ND[k], UNIT[k]
        if r["margin"] is None:
            out.append(f"- {NAMES[k]}: no margin, no second reading")
            continue
        out.append(f"- {NAMES[k]}: margin {fmt(r['margin'], nd + 1)} {u}; largest decode bound "
                   f"{fmt(r['decode_bound'], nd + 1)} {u}; second-reading margin {fmt(r['margin2'], nd + 1)} {u}"
                   f"{' (raised)' if r['raised2'] else ''}; sign flips of the start's {r['n']} decode differences "
                   f"beyond the margin {_pct(r['null_share'])} percent, beyond the second-reading margin "
                   f"{_pct(r['null_share2'])} percent")
    return out


def amendment_text(sr):
    return "\n".join(["Amendment 1, second reading of the margins of section 7 (deciding lines):"]
                     + amendment_lines(sr))


AFIELD = re.compile(r"^(Status|Rule): (.*)$", re.M)


def amendment_state(tree):
    """None when amendment 1 is absent; else {text, sha256, status, rule {path, sha256} or None, hashed, stored,
    hashed_at, hash_mtime}, as rule_state (a hash file that names another SHA-256 raises)."""
    p = os.path.join(tree, AMEND_REL)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    got = {}
    for k, v in AFIELD.findall(text):
        got.setdefault(k.lower(), v.strip())
    m = PATH_SHA.match(got.get("rule") or "")
    sha = sha256_file(p)
    sp = os.path.join(tree, AMEND_SHA_REL)
    stored = hashed_at = mtime = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            body = f.read()
        words = body.split()
        stored = words[0] if words else ""
        if stored != sha:
            raise HarnessDefect(f"rule_amendment_1.sha256 names {stored[:12]}, the amendment hashes {sha[:12]}")
        hm = HASHED_AT.search(body)
        hashed_at = hm.group(1) if hm else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(sp)).isoformat(timespec="seconds")
    return {"text": text, "sha256": sha, "status": got.get("status"),
            "rule": {"path": m.group(1), "sha256": m.group(2)} if m else None, "hashed": stored is not None,
            "stored": stored, "hashed_at": hashed_at, "hash_mtime": mtime}


def amendment_refusals(tree, sets, locked, second=None):
    """(reasons amendment 1 may not be hashed, second reading or None): the rule not hashed and final; the
    amendment absent, not final, not naming the rule at its SHA-256, hashed already; the start not attainable; a
    line of amendment_lines missing from it."""
    st = rule_state(tree)
    am = amendment_state(tree)
    reasons = []
    if not st["hashed"] or st["fields"]["status"] != "final" or st["fields"]["start"] is None:
        reasons.append("the rule is not hashed and final with its Start")
    if am is None:
        return reasons + [f"{AMEND_REL.replace(os.sep, '/')} is absent"], None
    if am["status"] != "final":
        reasons.append(f"Status is {am['status']!r}, not 'final'")
    if am["rule"] is None or am["rule"]["path"] != RULE_REL.replace(os.sep, "/") or \
            am["rule"]["sha256"] != st["sha256"]:
        reasons.append(f"the Rule line does not name {RULE_REL.replace(os.sep, '/')} SHA-256 {st['sha256']}")
    if am["hashed"]:
        reasons.append("rule_amendment_1.sha256 exists: the amendment is hashed already")
    sr = None
    if st["fields"]["start"] is not None:
        first, sec = start_decodes(tree, st["fields"], second)
        att = attainability(tree, first, sec, sets, locked)
        if not att["attainable"]:
            reasons += [f"not attainable: {x}" for x in att["reasons"]]
        sr = second_reading(tree, first, sec, sets, locked, att)
        reasons += [f"the amendment lacks the line: {x}" for x in amendment_lines(sr) if x not in am["text"]]
    return reasons, sr


def write_amendment_hash(tree, sets, locked, second=None, now=None):
    """rule_amendment_1.sha256 written when amendment_refusals finds nothing, with the local time of the hash;
    returns (written, reasons, second reading)."""
    reasons, sr = amendment_refusals(tree, sets, locked, second)
    if reasons:
        return False, reasons, sr
    am = amendment_state(tree)
    stamp = (now or datetime.datetime.now()).isoformat(timespec="seconds")
    with open(os.path.join(tree, AMEND_SHA_REL), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{am['sha256']}  rule_amendment_1.txt\nhashed_at {stamp}\n")
    return True, [], sr


def amendment_reading(tree, st, recs, r0, end_lines, start_lines, first, sec, sets, locked, att, v, lines):
    """The reading under the rule with amendment 1 (module docstring), None when the amendment is absent: {read
    False, reason} while it is not hashed; HarnessDefect when it is hashed but not final, names another rule SHA-256,
    was hashed at or after the run's first log line, or lacks a recomputed line; else {read True, margins, lines,
    verdict, agrees, differs_on} against the rule alone (v, lines)."""
    am = amendment_state(tree)
    if am is None:
        return None
    if not am["hashed"]:
        return {"read": False, "reason": "amendment 1 is not hashed: no second reading", "sha256": am["sha256"]}
    if am["status"] != "final" or am["rule"] is None or am["rule"]["sha256"] != st["sha256"]:
        raise HarnessDefect("amendment 1 is hashed but not final, or names another rule SHA-256")
    issue = hash_time_issue(am, recs)
    if issue:
        raise HarnessDefect(f"amendment 1: {issue}")
    sr = second_reading(tree, first, sec, sets, locked, att)
    missing = [x for x in amendment_lines(sr) if x not in am["text"]]
    if missing:
        raise HarnessDefect("the second-reading margins differ from amendment 1: " + "; ".join(missing))
    att2 = dict(att, lines={k: (dict(r, margin=sr[k]["margin2"]) if k in LINES else r)
                            for k, r in att["lines"].items()})
    lines2 = lines_reading(end_lines, start_lines, att2)
    v2 = verdict(r0, lines2)
    return {"read": True, "path": AMEND_REL.replace(os.sep, "/"), "sha256": am["sha256"], "hashed_at": am["hashed_at"],
            "hash_file_mtime": am["hash_mtime"], "margins": sr, "lines": lines2, "verdict": v2,
            "agrees": v2["verdict"] == v["verdict"],
            "differs_on": [k for k in LINES if lines[k].get("holds") != lines2[k].get("holds")]}


# ---------------------------------------------------------------- amendment 2: the S2b run and its configuration (S2B)
def _tree_path(tree, p):
    """A path as the tree-relative, forward-slash form without a trailing slash (absolute or relative input)."""
    p = str(p or "").replace("\\", "/").rstrip("/")
    return rel(p if os.path.isabs(p) else os.path.join(tree, p), tree).rstrip("/")


def amendment2_state(tree):
    """None when amendment 2 is absent; else {text, sha256, status, rule, amendment_1, start, config ({path, sha256}
    or None each), run, smoke (tree-relative run folders or None), hashed, stored, hashed_at, hash_mtime}, as
    amendment_state (a hash file that names another SHA-256 raises)."""
    p = os.path.join(tree, AMEND2_REL)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    got = {}
    for k, v in A2FIELD.findall(text):
        got.setdefault(k.lower(), v.strip())
    out = {"text": text, "sha256": sha256_file(p), "status": got.get("status")}
    for k in ("rule", "amendment 1", "start", "config"):
        m = PATH_SHA.match(got.get(k) or "")
        out[k.replace(" ", "_")] = {"path": m.group(1), "sha256": m.group(2)} if m else None
    for k in ("run", "smoke"):
        out[k] = _tree_path(tree, got[k]) if got.get(k) else None
    sp = os.path.join(tree, AMEND2_SHA_REL)
    stored = hashed_at = mtime = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            body = f.read()
        words = body.split()
        stored = words[0] if words else ""
        if stored != out["sha256"]:
            raise HarnessDefect(f"rule_amendment_2.sha256 names {stored[:12]}, the amendment hashes "
                                f"{out['sha256'][:12]}")
        hm = HASHED_AT.search(body)
        hashed_at = hm.group(1) if hm else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(sp)).isoformat(timespec="seconds")
    out.update(hashed=stored is not None, stored=stored, hashed_at=hashed_at, hash_mtime=mtime)
    return out


def amendment2_names(tree, a2, run_dir):
    """True when amendment 2 names `run_dir` as its Run or its Smoke run."""
    r = _tree_path(tree, run_dir)
    return a2 is not None and r in {a2.get("run"), a2.get("smoke")}


def amendment2_issues(tree, a2, st, fields):
    """Why amendment 2 may not bind (list of str; empty when it binds on the points it names): not final; its Rule
    line not the rule at the rule's SHA-256; its Amendment 1 line not amendment 1, hashed, at its SHA-256; its Start
    not the rule's Start; its Config not a stage-2 config file whose configuration SHA-256 is the one named."""
    out = []
    if a2["status"] != "final":
        out.append(f"Status is {a2['status']!r}, not 'final'")
    rp = RULE_REL.replace(os.sep, "/")
    if a2["rule"] is None or a2["rule"]["path"] != rp or a2["rule"]["sha256"] != st["sha256"]:
        out.append(f"the Rule line does not name {rp} SHA-256 {st['sha256']}")
    a1 = amendment_state(tree)
    ap = AMEND_REL.replace(os.sep, "/")
    if a1 is None or not a1["hashed"] or a2["amendment_1"] is None or a2["amendment_1"]["path"] != ap or \
            a2["amendment_1"]["sha256"] != a1["sha256"]:
        out.append(f"the Amendment 1 line does not name {ap}, hashed, at its SHA-256")
    if a2["start"] is None or fields.get("start") is None or a2["start"] != fields["start"]:
        out.append("the Start line differs from the rule's Start")
    c = a2["config"]
    if c is None:
        out.append("the Config line names no config with its SHA-256")
    else:
        try:
            from mdfm.train import config as CF
            cfg = CF.load(os.path.join(tree, c["path"]))
            if CF.sha256(cfg) != c["sha256"]:
                out.append(f"the Config {c['path']} has configuration SHA-256 {CF.sha256(cfg)}, not {c['sha256']}")
            elif not CF.stage2_on(cfg):
                out.append(f"the Config {c['path']} is not a stage-2 config")
        except (OSError, ValueError) as e:
            out.append(f"the Config {c['path']} does not load: {type(e).__name__}: {e}")
    if not a2["run"]:
        out.append("the Run line names no run folder")
    return out


def amendment2_refusals(tree, sets, locked, second=None):
    """(reasons amendment 2 may not be hashed, second reading or None): the rule not hashed and final with its Start and
    Config; amendment2_issues; a Run or Smoke folder whose log.jsonl exists (a number of the run exists); hashed
    already; the start not attainable; a line of amendment_lines (amendment 1's margins, unchanged) missing from it."""
    st = rule_state(tree)
    f = st["fields"]
    a2 = amendment2_state(tree)
    reasons = []
    if not st["hashed"] or f["status"] != "final" or f["start"] is None or f["config"] is None:
        reasons.append("the rule is not hashed and final with its Start and Config")
    if a2 is None:
        return reasons + [f"{AMEND2_REL.replace(os.sep, '/')} is absent"], None
    reasons += amendment2_issues(tree, a2, st, f)
    for k in ("run", "smoke"):
        if a2.get(k) and os.path.exists(os.path.join(tree, a2[k], "log.jsonl")):
            reasons.append(f"{a2[k]}/log.jsonl exists: a number of the run exists before the hash")
    if a2["hashed"]:
        reasons.append("rule_amendment_2.sha256 exists: the amendment is hashed already")
    sr = None
    if f["start"] is not None:
        first, sec = start_decodes(tree, f, second)
        att = attainability(tree, first, sec, sets, locked)
        if not att["attainable"]:
            reasons += [f"not attainable: {x}" for x in att["reasons"]]
        sr = second_reading(tree, first, sec, sets, locked, att)
        reasons += [f"the amendment lacks the line: {x}" for x in amendment_lines(sr) if x not in a2["text"]]
    return reasons, sr


def write_amendment2_hash(tree, sets, locked, second=None, now=None):
    """rule_amendment_2.sha256 written when amendment2_refusals finds nothing, with the local time of the hash;
    returns (written, reasons, second reading)."""
    reasons, sr = amendment2_refusals(tree, sets, locked, second)
    if reasons:
        return False, reasons, sr
    a2 = amendment2_state(tree)
    stamp = (now or datetime.datetime.now()).isoformat(timespec="seconds")
    with open(os.path.join(tree, AMEND2_SHA_REL), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{a2['sha256']}  rule_amendment_2.txt\nhashed_at {stamp}\n")
    return True, [], sr


def amendment2_binding(tree, st, run_dir, recs=None, expect=None):
    """(fields, record) of a run under amendment 2 (module docstring): for a run folder that amendment 2 names (Run or
    Smoke) and that is hashed and binds (amendment2_issues empty, hashed before the run's first log line), the rule's
    fields with the Config of amendment 2 and the record {path, sha256, hashed_at, hash_file_mtime, run, smoke,
    config}; (the rule's fields, None) for a run it does not name. HarnessDefect: `expect` given while amendment 2 is
    absent, not hashed, hashes otherwise or does not name the run; a named run whose amendment does not bind or was
    hashed at or after the run's first log line. A named run under an amendment 2 that is not hashed: (None, record
    with read False), which the read and the smoke check refuse."""
    f = st["fields"]
    a2 = amendment2_state(tree)
    if expect and (a2 is None or not a2["hashed"] or a2["sha256"] != expect):
        raise HarnessDefect(f"amendment 2 is absent, not hashed or hashes otherwise; the job expects {expect[:12]}")
    if not amendment2_names(tree, a2, run_dir):
        if expect:
            raise HarnessDefect(f"amendment 2 does not name the run {run_dir}")
        return f, None
    rec = {"path": AMEND2_REL.replace(os.sep, "/"), "sha256": a2["sha256"], "hashed_at": a2["hashed_at"],
           "hash_file_mtime": a2["hash_mtime"], "run": a2["run"], "smoke": a2["smoke"], "config": a2["config"],
           "read": bool(a2["hashed"])}
    if not a2["hashed"]:
        return None, dict(rec, reason="amendment 2 names this run and is not hashed")
    issues = amendment2_issues(tree, a2, st, f)
    if issues:
        raise HarnessDefect("amendment 2 is hashed but does not bind: " + "; ".join(issues))
    issue = hash_time_issue(a2, recs)
    if issue:
        raise HarnessDefect(f"amendment 2: {issue}")
    return dict(f, config=dict(a2["config"])), rec


# ---------------------------------------------------------------- amendment 3: the S2c run, beta_error on three records
def amendment3_state(tree):
    """None when amendment 3 is absent; else {text, sha256, status, rule, amendment_1, amendment_2, start, config
    ({path, sha256} or None each), run, smoke (tree-relative run folders or None), hashed, stored, hashed_at,
    hash_mtime}, as amendment2_state (a hash file that names another SHA-256 raises)."""
    p = os.path.join(tree, AMEND3_REL)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    got = {}
    for k, v in A3FIELD.findall(text):
        got.setdefault(k.lower(), v.strip())
    out = {"text": text, "sha256": sha256_file(p), "status": got.get("status")}
    for k in ("rule", "amendment 1", "amendment 2", "start", "config"):
        m = PATH_SHA.match(got.get(k) or "")
        out[k.replace(" ", "_")] = {"path": m.group(1), "sha256": m.group(2)} if m else None
    for k in ("run", "smoke"):
        out[k] = _tree_path(tree, got[k]) if got.get(k) else None
    sp = os.path.join(tree, AMEND3_SHA_REL)
    stored = hashed_at = mtime = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            body = f.read()
        words = body.split()
        stored = words[0] if words else ""
        if stored != out["sha256"]:
            raise HarnessDefect(f"rule_amendment_3.sha256 names {stored[:12]}, the amendment hashes "
                                f"{out['sha256'][:12]}")
        hm = HASHED_AT.search(body)
        hashed_at = hm.group(1) if hm else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(sp)).isoformat(timespec="seconds")
    out.update(hashed=stored is not None, stored=stored, hashed_at=hashed_at, hash_mtime=mtime)
    return out


def amendment3_names(tree, a3, run_dir):
    """True when amendment 3 names `run_dir` as its Run or its Smoke run."""
    r = _tree_path(tree, run_dir)
    return a3 is not None and r in {a3.get("run"), a3.get("smoke")}


def config_differences(a, b, prefix=""):
    """The dotted names of the fields where two canonical-JSON dicts of configurations differ, sorted."""
    out = []
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out += config_differences(a[k], b[k], prefix + k + ".")
        elif a.get(k) != b.get(k):
            out.append(prefix + k)
    return out


def _load_config(tree, c):
    """(configuration, None) of a {path, sha256} line, or (None, why it does not hold): it does not load, its
    configuration SHA-256 is another or it is not a stage-2 config."""
    from mdfm.train import config as CF
    try:
        cfg = CF.load(os.path.join(tree, c["path"]))
    except (OSError, ValueError) as e:
        return None, f"the Config {c['path']} does not load: {type(e).__name__}: {e}"
    if CF.sha256(cfg) != c["sha256"]:
        return None, f"the Config {c['path']} has configuration SHA-256 {CF.sha256(cfg)}, not {c['sha256']}"
    if not CF.stage2_on(cfg):
        return None, f"the Config {c['path']} is not a stage-2 config"
    return cfg, None


def amendment3_issues(tree, a3, st, fields):
    """Why amendment 3 may not bind (list of str; empty when it binds on the points it names): not final; its Rule
    line not the rule at the rule's SHA-256; its Amendment 1 and Amendment 2 lines not those files, hashed, at their
    SHA-256; its Start not the rule's Start; its Config not a stage-2 config file at the configuration SHA-256 named,
    or one that differs from amendment 2's Config elsewhere than in A3_CHANGES or reads schedule.alarm_consecutive
    otherwise than A3_ALARM_CONSECUTIVE (section 1 of the amendment); no Run line; a Run or Smoke folder that
    amendment 2 names (S2b is never read under amendment 3)."""
    from mdfm.train import config as CF
    out = []
    if a3["status"] != "final":
        out.append(f"Status is {a3['status']!r}, not 'final'")
    rp = RULE_REL.replace(os.sep, "/")
    if a3["rule"] is None or a3["rule"]["path"] != rp or a3["rule"]["sha256"] != st["sha256"]:
        out.append(f"the Rule line does not name {rp} SHA-256 {st['sha256']}")
    a1 = amendment_state(tree)
    ap = AMEND_REL.replace(os.sep, "/")
    if a1 is None or not a1["hashed"] or a3["amendment_1"] is None or a3["amendment_1"]["path"] != ap or \
            a3["amendment_1"]["sha256"] != a1["sha256"]:
        out.append(f"the Amendment 1 line does not name {ap}, hashed, at its SHA-256")
    a2 = amendment2_state(tree)
    ap2 = AMEND2_REL.replace(os.sep, "/")
    if a2 is None or not a2["hashed"] or a3["amendment_2"] is None or a3["amendment_2"]["path"] != ap2 or \
            a3["amendment_2"]["sha256"] != a2["sha256"]:
        out.append(f"the Amendment 2 line does not name {ap2}, hashed, at its SHA-256")
    if a3["start"] is None or fields.get("start") is None or a3["start"] != fields["start"]:
        out.append("the Start line differs from the rule's Start")
    c = a3["config"]
    if c is None:
        out.append("the Config line names no config with its SHA-256")
    else:
        cfg, why = _load_config(tree, c)
        if why:
            out.append(why)
        elif a2 is None or a2.get("config") is None:
            out.append("amendment 2 names no Config to compare the Config with")
        else:
            base, why2 = _load_config(tree, a2["config"])
            if why2:
                out.append(f"amendment 2: {why2}")
            else:
                d = config_differences(CF.as_json_dict(base), CF.as_json_dict(cfg))
                if d != sorted(A3_CHANGES):
                    out.append(f"the Config differs from {a2['config']['path']} in {d}, not in {sorted(A3_CHANGES)}")
                if dict(cfg.schedule.alarm_consecutive) != A3_ALARM_CONSECUTIVE:
                    out.append(f"the Config reads schedule.alarm_consecutive {dict(cfg.schedule.alarm_consecutive)}, "
                               f"not {A3_ALARM_CONSECUTIVE}")
    if not a3["run"]:
        out.append("the Run line names no run folder")
    if a2 is not None:
        both = sorted({x for x in (a3["run"], a3["smoke"]) if x} & {x for x in (a2["run"], a2["smoke"]) if x})
        if both:
            out.append(f"amendment 2 names the run folders {both} too")
    return out


def alarm_lines(res, steps, log_every, k=A3_ALARM_CONSECUTIVE[A3_ALARM], sim_n=None, name=A3_ALARM):
    """The lines amendment 3 holds verbatim in its section 3 (they decide nothing): from the results.json of
    scripts/stage2_beta_alarm.py (`res`), the readings of the alarm's line on every log that carries it, the reference
    runs pooled per phase, the projection on one phase C of `steps` steps (records every `log_every` steps from the
    alarm's first step) and the cluster model under fixed weights; k the records in a row of the reading (the
    Config's schedule.alarm_consecutive), sim_n the resampled phases of the projection (that script's SIM_N)."""
    logs = res.get("logs") or {}
    line, first = float(res["line"]), int(res["first_step"])
    ck = f"consecutive_{int(k)}"
    phases = [(r, ph, x) for r, v in sorted((logs.get("runs") or {}).items())
              for ph, x in sorted((v.get("phases") or {}).items()) if x.get("from_first")]
    if not phases or any(ck not in (x.get("readings") or {}) for _, _, x in phases):
        return [f"- {name}: no reading of {k} records in a row in {BETA_ALARM_REL.replace(os.sep, '/')}"]
    fired_k = [f"{r} phase {ph} at step {x['readings'][ck]['first_step']}" for r, ph, x in phases
               if x["readings"][ck]["fired"]]
    fired_1 = [f"{r} phase {ph} at step {x['readings']['single']['first_step']}" for r, ph, x in phases
               if x["readings"]["single"]["fired"]]
    out = [f"- The logs: {len(phases)} phase logs of {len({r for r, _, _ in phases})} runs hold {name} records from "
           f"step {first} on; {k} records in a row above {line:.1f} in {len(fired_k)} of them"
           + (f" ({', '.join(fired_k)})" if fired_k else "") + f"; the single-record reading fires in {len(fired_1)}"
           + (f" ({', '.join(fired_1)})" if fired_1 else "")]
    for ph, p in sorted((logs.get("pools") or {}).items()):
        runs = list(p["runs"])
        per = {r: logs["runs"][r]["phases"][ph] for r in runs}
        out.append(f"- Phase {ph}, the reference runs pooled: {p['records']} records of {len(runs)} runs, "
                   f"{p['above']} above {line:.1f}, share {p['share']:.4f} [{p['interval'][0]:.4f}, "
                   f"{p['interval'][1]:.4f}]; per run the largest smallest-of-{k} and the lag-1 autocorrelation: "
                   + ", ".join(f"{r} {fmt(per[r]['readings'][ck]['max_statistic'], 3)} and "
                               f"{fmt(per[r].get('lag1_autocorrelation'), 3, True)}" for r in runs))
    pc = logs.get("projection_C") or {}
    n = len([s for s in range(int(log_every), int(steps) + 1, int(log_every)) if s >= first])
    rd = pc.get("readings") or {}
    if pc.get("records_per_phase") != n or ck not in rd:
        out.append(f"- The projection of {BETA_ALARM_REL.replace(os.sep, '/')} holds {pc.get('records_per_phase')} "
                   f"records per phase C, the Config {n}")
    else:
        lo, hi = pc["p_interval"]
        out.append(f"- One phase C of {steps} steps holds {n} records from step {first}. Its records drawn i.i.d. "
                   f"from the {pc['pool']} pooled phase C records ({pc['above']} above {line:.1f}, share "
                   f"{pc['p']:.4f} [{lo:.4f}, {hi:.4f}]): the single-record reading fires in "
                   f"{rd['single']['resampled_fire_share']:.4f} of {sim_n} resampled phases, 2 records in a row in "
                   f"{rd['consecutive_2']['resampled_fire_share']:.4f}, {k} in a row in "
                   f"{rd[ck]['resampled_fire_share']:.4f}; exactly, at the share, "
                   f"{rd['single']['exact_fire_at_p']:.4f}, {rd['consecutive_2']['exact_fire_at_p']:.4f} and "
                   f"{rd[ck]['exact_fire_at_p']:.4f}; at its upper bound, {rd['single']['exact_fire_at_p_high']:.4f}, "
                   f"{rd['consecutive_2']['exact_fire_at_p_high']:.4f} and {rd[ck]['exact_fire_at_p_high']:.4f}")
    for w, fw in ((res.get("panel") or {}).get("fixed_weights") or {}).items():
        tm = (fw or {}).get("tail_model")
        if not tm:
            continue
        pr = tm["projection"]
        c = {j: pr.get(f"consecutive_{j}") or {} for j in (1, 2, int(k))}
        f4 = (lambda j, key: "n/a" if fin(c[j].get(key)) is None else f"{c[j][key]:.4f}")  # noqa: E731
        out.append(f"- Under the fixed weights {w}, by the cluster model of the record ({tm['plans']} replayed S2b "
                   f"plans, {tm['records'] // max(tm['plans'], 1)} draws each): share of records above {line:.1f} "
                   f"{tm['p']:.5f} [{tm['interval'][0]:.5f}, {tm['interval'][1]:.5f}]; over {pr['records_per_phase']} "
                   f"records the single-record reading fires in {f4(1, 'fire_at_p')} of phases "
                   f"({f4(1, 'fire_at_p_high')} at the upper bound), 2 in a row in {f4(2, 'fire_at_p')} "
                   f"({f4(2, 'fire_at_p_high')}), {k} in a row in {f4(int(k), 'fire_at_p')} "
                   f"({f4(int(k), 'fire_at_p_high')})")
    out.append(f"- Detection: a level above {line:.1f} that lasts is raised {int(k) - 1} records "
               f"({(int(k) - 1) * int(log_every)} steps) after the single-record reading would raise it")
    return out


def alarm_reading(tree, cfg):
    """(alarm_lines of the tree's beta_alarm results under the configuration cfg, None) or ([], why not)."""
    res = read_json(os.path.join(tree, BETA_ALARM_REL))
    if res is None:
        return [], f"{BETA_ALARM_REL.replace(os.sep, '/')} is absent"
    k = dict(cfg.schedule.alarm_consecutive).get(A3_ALARM)
    if not k:
        return [], f"the Config reads {A3_ALARM} on one record"
    sim_n = getattr(_sibling("stage2_beta_alarm"), "SIM_N", None)
    return alarm_lines(res, int(cfg.stage2.steps), int(cfg.schedule.log_every), int(k), sim_n), None


def amendment3_refusals(tree, sets, locked, second=None):
    """(reasons amendment 3 may not be hashed, {second_reading, alarm_lines} or None): the rule not hashed and final
    with its Start and Config; amendment3_issues; a Run or Smoke folder whose log.jsonl exists (a number of the run
    exists); hashed already; the start not attainable; a line of amendment_lines (amendment 1's margins, unchanged)
    missing from amendment 2, which prints them; a line of alarm_lines (the false alarms of the reading, from
    reports/bench/stage2_beta_alarm/results.json under the Config) missing from amendment 3."""
    st = rule_state(tree)
    f = st["fields"]
    a3 = amendment3_state(tree)
    reasons = []
    if not st["hashed"] or f["status"] != "final" or f["start"] is None or f["config"] is None:
        reasons.append("the rule is not hashed and final with its Start and Config")
    if a3 is None:
        return reasons + [f"{AMEND3_REL.replace(os.sep, '/')} is absent"], None
    reasons += amendment3_issues(tree, a3, st, f)
    for k in ("run", "smoke"):
        if a3.get(k) and os.path.exists(os.path.join(tree, a3[k], "log.jsonl")):
            reasons.append(f"{a3[k]}/log.jsonl exists: a number of the run exists before the hash")
    if a3["hashed"]:
        reasons.append("rule_amendment_3.sha256 exists: the amendment is hashed already")
    out = {"second_reading": None, "alarm_lines": []}
    if f["start"] is not None:
        first, sec = start_decodes(tree, f, second)
        att = attainability(tree, first, sec, sets, locked)
        if not att["attainable"]:
            reasons += [f"not attainable: {x}" for x in att["reasons"]]
        sr = second_reading(tree, first, sec, sets, locked, att)
        out["second_reading"] = sr
        a2 = amendment2_state(tree)
        reasons += [f"amendment 2 lacks the line: {x}" for x in amendment_lines(sr)
                    if a2 is None or x not in a2["text"]]
    if a3["config"] is not None:
        cfg, why = _load_config(tree, a3["config"])
        if cfg is not None:
            lines, why = alarm_reading(tree, cfg)
            out["alarm_lines"] = lines
            reasons += [f"the amendment lacks the line: {x}" for x in lines if x not in a3["text"]]
        if why:
            reasons.append(f"no false-alarm lines: {why}")
    return reasons, out


def write_amendment3_hash(tree, sets, locked, second=None, now=None):
    """rule_amendment_3.sha256 written when amendment3_refusals finds nothing, with the local time of the hash;
    returns (written, reasons, {second_reading, alarm_lines})."""
    reasons, got = amendment3_refusals(tree, sets, locked, second)
    if reasons:
        return False, reasons, got
    a3 = amendment3_state(tree)
    stamp = (now or datetime.datetime.now()).isoformat(timespec="seconds")
    with open(os.path.join(tree, AMEND3_SHA_REL), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{a3['sha256']}  rule_amendment_3.txt\nhashed_at {stamp}\n")
    return True, [], got


def run_binding(tree, st, run_dir, recs=None, expect2=None, expect3=None):
    """(fields, amendment 2 record, amendment 3 record) of a run (module docstring). A run folder that amendment 3
    names (Run or Smoke): hashed and binding (amendment3_issues empty, hashed before the run's first log line), the
    rule's fields with the Config of amendment 3 and its record {path, sha256, hashed_at, hash_file_mtime, run, smoke,
    config, amendment_2, read}; while it is not hashed (None, None, record with read False and the reason), which the
    read and the smoke check refuse; `expect2` then names amendment 2's identity only (present, hashed, at that
    SHA-256: amendment 3 binds together with it), not a run it names. Any other run: amendment2_binding, unchanged.
    HarnessDefect: `expect3` given while amendment 3 is absent, not hashed, hashes otherwise or does not name the run;
    `expect2` given while amendment 2 is absent, not hashed or hashes otherwise; a named run whose amendment 3 does
    not bind or was hashed at or after the run's first log line."""
    f = st["fields"]
    a3 = amendment3_state(tree)
    if expect3 and (a3 is None or not a3["hashed"] or a3["sha256"] != expect3):
        raise HarnessDefect(f"amendment 3 is absent, not hashed or hashes otherwise; the job expects {expect3[:12]}")
    if not amendment3_names(tree, a3, run_dir):
        if expect3:
            raise HarnessDefect(f"amendment 3 does not name the run {run_dir}")
        f2, a2 = amendment2_binding(tree, st, run_dir, recs, expect2)
        return f2, a2, None
    if expect2:
        a2s = amendment2_state(tree)
        if a2s is None or not a2s["hashed"] or a2s["sha256"] != expect2:
            raise HarnessDefect(f"amendment 2 is absent, not hashed or hashes otherwise; the job expects "
                                f"{expect2[:12]}")
    rec = {"path": AMEND3_REL.replace(os.sep, "/"), "sha256": a3["sha256"], "hashed_at": a3["hashed_at"],
           "hash_file_mtime": a3["hash_mtime"], "run": a3["run"], "smoke": a3["smoke"], "config": a3["config"],
           "amendment_2": a3["amendment_2"], "read": bool(a3["hashed"])}
    if not a3["hashed"]:
        return None, None, dict(rec, reason="amendment 3 names this run and is not hashed")
    issues = amendment3_issues(tree, a3, st, f)
    if issues:
        raise HarnessDefect("amendment 3 is hashed but does not bind: " + "; ".join(issues))
    issue = hash_time_issue(a3, recs)
    if issue:
        raise HarnessDefect(f"amendment 3: {issue}")
    return dict(f, config=dict(a3["config"])), None, rec


def alarm_watch_events(recs, units=False):
    """The 'alarm_watch' events of a run's log (schedule.alarm_consecutive, task S2BA): [{phase, step, source, alarms
    [{name, value, consecutive, of}]}], in log order; reported beside the verdict under amendment 3, deciding
    nothing. units (amendment 4): each unit_nll_rise entry also names the rising units of the 'dev' event of its phase
    and step (the last one; unit_nll_rise_units of the dev hook's 'model' entry)."""
    rising = {}
    if units:
        for r in WB().log_events(recs or [], "dev"):
            m = (r.get("dev") or {}).get("model") or {}
            rising[(r.get("phase"), r.get("step"))] = list(m.get("unit_nll_rise_units") or [])
    out = []
    for r in WB().log_events(recs or [], "alarm_watch"):
        hs = []
        for h in r.get("alarms") or []:
            if isinstance(h, dict):
                e = {k: h.get(k) for k in ("name", "value", "consecutive", "of")}
                if units and h.get("name") == A4_ALARM:
                    e["units"] = rising.get((r.get("phase"), r.get("step")), [])
                hs.append(e)
        out.append({"phase": r.get("phase"), "step": r.get("step"), "source": r.get("source"), "alarms": hs})
    return out


# ---------------------------------------------------------------- amendment 4: the S2d run, unit_nll_rise on two reads
def amendment4_state(tree):
    """None when amendment 4 is absent; else {text, sha256, status, rule, amendment_1, amendment_2, amendment_3, start,
    config ({path, sha256} or None each), run, smoke (tree-relative run folders or None), hashed, stored, hashed_at,
    hash_mtime}, as amendment3_state (a hash file that names another SHA-256 raises)."""
    p = os.path.join(tree, AMEND4_REL)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        text = f.read()
    got = {}
    for k, v in A4FIELD.findall(text):
        got.setdefault(k.lower(), v.strip())
    out = {"text": text, "sha256": sha256_file(p), "status": got.get("status")}
    for k in ("rule", "amendment 1", "amendment 2", "amendment 3", "start", "config"):
        m = PATH_SHA.match(got.get(k) or "")
        out[k.replace(" ", "_")] = {"path": m.group(1), "sha256": m.group(2)} if m else None
    for k in ("run", "smoke"):
        out[k] = _tree_path(tree, got[k]) if got.get(k) else None
    sp = os.path.join(tree, AMEND4_SHA_REL)
    stored = hashed_at = mtime = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            body = f.read()
        words = body.split()
        stored = words[0] if words else ""
        if stored != out["sha256"]:
            raise HarnessDefect(f"rule_amendment_4.sha256 names {stored[:12]}, the amendment hashes "
                                f"{out['sha256'][:12]}")
        hm = HASHED_AT.search(body)
        hashed_at = hm.group(1) if hm else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(sp)).isoformat(timespec="seconds")
    out.update(hashed=stored is not None, stored=stored, hashed_at=hashed_at, hash_mtime=mtime)
    return out


def amendment4_names(tree, a4, run_dir):
    """True when amendment 4 names `run_dir` as its Run or its Smoke run."""
    r = _tree_path(tree, run_dir)
    return a4 is not None and r in {a4.get("run"), a4.get("smoke")}


def amendment4_issues(tree, a4, st, fields):
    """Why amendment 4 may not bind (list of str; empty when it binds on the points it names): not final; its Rule
    line not the rule at the rule's SHA-256; its Amendment 1, 2 and 3 lines not those files, hashed, at their SHA-256;
    amendment 3 not binding on its own points (amendment3_issues, named 'amendment 3: ...'); its Start not the rule's
    Start; its Config not a stage-2 config file at the configuration SHA-256 named, or one that differs from amendment
    3's Config elsewhere than in A4_CHANGES or reads schedule.alarm_consecutive otherwise than A4_ALARM_CONSECUTIVE
    (section 1 of the amendment); no Run line; a Run or Smoke folder that amendment 2 or 3 names (S2b and S2c are never
    read under amendment 4)."""
    from mdfm.train import config as CF
    out = []
    if a4["status"] != "final":
        out.append(f"Status is {a4['status']!r}, not 'final'")
    rp = RULE_REL.replace(os.sep, "/")
    if a4["rule"] is None or a4["rule"]["path"] != rp or a4["rule"]["sha256"] != st["sha256"]:
        out.append(f"the Rule line does not name {rp} SHA-256 {st['sha256']}")
    a2, a3 = amendment2_state(tree), amendment3_state(tree)
    for key, label, s, path in (("amendment_1", "Amendment 1", amendment_state(tree), AMEND_REL),
                                ("amendment_2", "Amendment 2", a2, AMEND2_REL),
                                ("amendment_3", "Amendment 3", a3, AMEND3_REL)):
        p = path.replace(os.sep, "/")
        if s is None or not s["hashed"] or a4[key] is None or a4[key]["path"] != p or a4[key]["sha256"] != s["sha256"]:
            out.append(f"the {label} line does not name {p}, hashed, at its SHA-256")
    if a3 is not None and a3["hashed"]:
        out += [f"amendment 3: {x}" for x in amendment3_issues(tree, a3, st, fields)]
    if a4["start"] is None or fields.get("start") is None or a4["start"] != fields["start"]:
        out.append("the Start line differs from the rule's Start")
    c = a4["config"]
    if c is None:
        out.append("the Config line names no config with its SHA-256")
    else:
        cfg, why = _load_config(tree, c)
        if why:
            out.append(why)
        elif a3 is None or a3.get("config") is None:
            out.append("amendment 3 names no Config to compare the Config with")
        else:
            base, why3 = _load_config(tree, a3["config"])
            if why3:
                out.append(f"amendment 3: {why3}")
            else:
                d = config_differences(CF.as_json_dict(base), CF.as_json_dict(cfg))
                if d != sorted(A4_CHANGES):
                    out.append(f"the Config differs from {a3['config']['path']} in {d}, not in {sorted(A4_CHANGES)}")
                if dict(cfg.schedule.alarm_consecutive) != A4_ALARM_CONSECUTIVE:
                    out.append(f"the Config reads schedule.alarm_consecutive {dict(cfg.schedule.alarm_consecutive)}, "
                               f"not {A4_ALARM_CONSECUTIVE}")
    if not a4["run"]:
        out.append("the Run line names no run folder")
    mine = {x for x in (a4["run"], a4["smoke"]) if x}
    for n, s in ((2, a2), (3, a3)):
        if s is not None:
            both = sorted(mine & {x for x in (s["run"], s["smoke"]) if x})
            if both:
                out.append(f"amendment {n} names the run folders {both} too")
    return out


def unit_reading_name(k):
    """The reading of scripts/stage2_unit_nll_alarm.py (its READINGS) that is unit_nll_rise read on k dev reads in a
    row: the loop's own for k = 1, consecutive_<k> otherwise."""
    return "loop_3_reads" if int(k) == 1 else f"consecutive_{int(k)}"


def unit_phase_reads(steps, dev_every, first=UNIT_FIRST_READ):
    """The dev reads of a phase C that scripts/stage2_unit_nll_alarm.py models (phase_reads): every dev_every steps from
    `first` before the phase end, and the phase end."""
    return len(range(int(first), int(steps), int(dev_every))) + 1


def unit_alarm_lines(res, steps, dev_every, k=A4_ALARM_CONSECUTIVE[A4_ALARM], sim_n=None):
    """The lines amendment 4 holds verbatim in its section 3 (they decide nothing): from the results.json of
    scripts/stage2_unit_nll_alarm.py (`res`, task S2CA), the readings of unit_nll_rise on the logs of S2a, S2b and S2c
    and where each log's stop censors them, the three-read windows of the logs against the noise models, the
    read-to-read spreads, the false alarms of the loop's reading and of k dev reads in a row over one phase C of `steps`
    steps read every `dev_every` steps (the Config's), an S2a-like and a steady overfit, the cost of a missed stop and
    the detection delay; sim_n the simulated phases of the noise model (that script's SIM_N, or its --n-sim)."""
    name, rel_ = A4_ALARM, UNIT_ALARM_REL.replace(os.sep, "/")
    M, L = res.get("model") or {}, res.get("logs") or {}
    rk, r1 = unit_reading_name(k), unit_reading_name(1)
    null = M.get("null") or {}
    if rk not in (null.get("readings") or {}):
        return [f"- {name}: no reading of {k} dev reads in a row in {rel_}"]
    n = unit_phase_reads(steps, dev_every)
    if null.get("reads_per_phase") != n:
        return [f"- The model of {rel_} holds {null.get('reads_per_phase')} dev reads per phase C, the Config {n}"]
    sim_n = sim_n or "n/a"
    sim10 = sim_n // 10 if isinstance(sim_n, int) else "n/a"
    reads = [r1] + ([rk] if rk != r1 else [])
    label = {r1: "the loop's one-read reading", rk: f"{k} dev reads in a row"}
    runs = sorted(M.get("logs") or {})
    fired = {r: [f"{run} at step {M['logs'][run][r]['step']}, {', '.join(M['logs'][run][r]['units'])}"
                 for run in runs if (M["logs"][run] or {}).get(r)] for r in reads}
    cen = M.get("logs_censoring") or {}
    at_stop = []
    for run in runs:
        c = ((cen.get(run) or {}).get("readings") or {}).get(rk) or {}
        last = (cen.get(run) or {}).get("last_read")
        if c.get("fires") is not None:
            at_stop.append(f"{run} fires at {c['fires']}")
        else:
            at_stop.append(f"{run} {c.get('streak_at_last_read')} of {c.get('consecutive')} at {last}"
                           + (f" (open: {', '.join(c.get('units_at_last_read') or [])})" if c.get("open") else ""))
    fires = [f"{label[r]} fires in {len(fired[r])}" + (f" ({'; '.join(fired[r])})" if fired[r] else "")
             for r in reads]
    out = [f"- The logs: {len(runs)} phase C logs ({', '.join(runs)}) hold {name} reads from step {UNIT_FIRST_READ} "
           f"on; " + "; ".join(fires) + f"; each log ends at its run's stop, where {label[rk]} stood at: "
           + ", ".join(at_stop)]
    tp = (L.get("triples_pooled") or {}).get("from_1000") or {}
    ex = M.get("expected_on_logs") or {}
    iv = tp.get("interval") or [None, None]
    out.append(f"- Three-read windows of the logs from step {UNIT_FIRST_READ} (the units with a val fold): "
               f"{tp.get('windows')}, held-out rising in {tp.get('held_out_rising')}, train falling in "
               f"{tp.get('train_falling')}, both (the loop's pattern) in {tp.get('both')}, share "
               f"{fmt(tp.get('share'), 4)} [{fmt(iv[0], 4)}, {fmt(iv[1], 4)}]; expected under the noise models "
               f"{fmt(ex.get('trend'), 2)} (trend), {fmt(ex.get('replicate'), 2)} (replicate), "
               f"{fmt(ex.get('residual'), 2)} (residual)")
    rep = L.get("replicate") or {}
    pr = M.get("params") or {}
    nd = sorted({(rep.get(u) or {}).get("held_out", {}).get("reads_from_2000") for u in pr} - {None})
    out.append(f"- Read-to-read spread, S2b minus S2c read by read from step 2000 (rms over sqrt 2, "
               f"{'/'.join(str(x) for x in nd)} differences per unit): " + "; ".join(
                   f"{u} held-out {fmt(p.get('sd_ho'), 0)}, train {fmt(p.get('sd_tr'), 0)}, correlation "
                   f"{fmt(p.get('rho'), 2, True)}" for u, p in sorted(pr.items())))

    def fp(var, r):
        x = ((M.get(var) or {}).get("readings") or {}).get(r) or {}
        return f"{fmt(x.get('fire_phase'), 4)} ({fmt(x.get('fire_by_8000'), 4)})"

    out.append(f"- One phase C of {steps} steps holds {n} dev reads from step {UNIT_FIRST_READ} (every {dev_every} "
               f"steps and the phase end). Reads drawn independently around a level with these spreads and "
               f"correlations, the units independent ({sim_n} simulated phases), the share of phases stopped (by step "
               f"8000): " + ", ".join(f"{label[r]} {fp('null', r)}" for r in reads)
               + "; with the detrended residual spreads " + ", ".join(fp("null_resid", r) for r in reads)
               + "; with the trend to step 8000 " + ", ".join(fp("null_trend", r) for r in reads)
               + "; held-out and train independent " + ", ".join(fp("null_rho0", r) for r in reads))
    sa = M.get("overfit_s2a_like") or {}

    def of(sc, r):
        x = (((sa.get(sc) or {}).get("readings") or {}).get(r)) or {}
        return f"{label[r]} by step 8000 in {fmt(x.get('fire_by_8000'), 4)}, never in {fmt(x.get('never_share'), 4)}"

    out.append(f"- An S2a-like overfit ({sa.get('unit')} held-out NLL rising "
               f"{fmt(sa.get('held_out_slope_per_1000'), 0, True)} (se {fmt(sa.get('held_out_slope_se'), 0)}) per 1000 "
               f"steps, {fmt(sa.get('drift_sd_per_read'), 2)} of its held-out spread per read, from step "
               f"{UNIT_FIRST_READ}; {sim10} simulated phases), the share of phases stopped: held-out only (S2a's train "
               f"NLL did not fall): " + "; ".join(of("B", r) for r in reads)
               + "; the train NLL falling at the same multiple of its spread: " + "; ".join(of("A", r) for r in reads))
    ov = M.get("overfit") or {}
    unit = (res.get("alarm") or {}).get("unit")

    def med(key, r):
        x = (((ov.get(key) or {}).get("readings") or {}).get(r)) or {}
        return f"{fmt(x.get('reads_to_fire_median'), 0)} ({fmt(x.get('never_share'), 4)})"

    parts = []
    for sc, what in (("A", "held-out up and train down"), ("B", "held-out up only")):
        for d in ("1.0", "0.5"):
            parts.append(f"{what} by {d} spread per read: " + ", ".join(f"{label[r]} {med(sc + d, r)}" for r in reads))
    out.append(f"- A steady overfit of {unit} from step {UNIT_FIRST_READ} ({sim10} simulated phases), the median dev "
               f"read at which a reading fires (1 is step {UNIT_FIRST_READ}, {n + 1} never) and the share of phases it "
               f"never fires: " + "; ".join(parts))
    mc = M.get("missed_stop_cost_s2a") or {}
    out.append(f"- A stop not made ({mc.get('run')} at its last read {mc.get('last_read')}) costs at least one dev "
               f"read, {mc.get('one_read_steps')} steps ({fmt((mc.get('one_read_gpu_h') or {}).get('s2b_rate'), 2)} "
               f"GPU-h at S2b's rate), at most the rest of phase C, {mc.get('rest_steps')} steps "
               f"({fmt((mc.get('rest_gpu_h') or {}).get('s2b_rate'), 2)} GPU-h); the run is then read at its end")
    late = int(k) - 1
    out.append(f"- Detection: a pattern that lasts is raised {late} dev read{'' if late == 1 else 's'} "
               f"({late * int(dev_every)} steps) after the loop's one-read reading would raise it")
    return out


def unit_sim_n(res):
    """The simulated phases of the noise model of scripts/stage2_unit_nll_alarm.py's results.json: its --n-sim when
    the call named one, else that script's SIM_N."""
    argv = [str(x) for x in (res.get("argv") or [])]
    for i, x in enumerate(argv):
        if x == "--n-sim" and i + 1 < len(argv):
            return int(argv[i + 1])
        if x.startswith("--n-sim="):
            return int(x.split("=", 1)[1])
    return getattr(_sibling("stage2_unit_nll_alarm"), "SIM_N", None)


def unit_alarm_reading(tree, cfg):
    """(unit_alarm_lines of the tree's unit_nll_rise results under the configuration cfg, None) or ([], why not)."""
    res = read_json(os.path.join(tree, UNIT_ALARM_REL))
    if res is None:
        return [], f"{UNIT_ALARM_REL.replace(os.sep, '/')} is absent"
    k = dict(cfg.schedule.alarm_consecutive).get(A4_ALARM)
    if not k:
        return [], f"the Config reads {A4_ALARM} on one read"
    return unit_alarm_lines(res, int(cfg.stage2.steps), int(cfg.schedule.dev_every), int(k), unit_sim_n(res)), None


def amendment4_refusals(tree, sets, locked, second=None):
    """(reasons amendment 4 may not be hashed, {second_reading, alarm_lines, unit_alarm_lines} or None): the rule not
    hashed and final with its Start and Config; amendment4_issues; a Run or Smoke folder whose log.jsonl exists (a
    number of the run exists); hashed already; the start not attainable; a line of amendment_lines (amendment 1's
    margins, unchanged) missing from amendment 2, which prints them; a line of alarm_lines (the beta_error reading,
    unchanged, under the Config) missing from amendment 3, which prints them; a line of unit_alarm_lines (the false
    alarms of the unit_nll_rise reading, from reports/bench/stage2_unit_nll_alarm/results.json under the Config) missing
    from amendment 4."""
    st = rule_state(tree)
    f = st["fields"]
    a4 = amendment4_state(tree)
    reasons = []
    if not st["hashed"] or f["status"] != "final" or f["start"] is None or f["config"] is None:
        reasons.append("the rule is not hashed and final with its Start and Config")
    if a4 is None:
        return reasons + [f"{AMEND4_REL.replace(os.sep, '/')} is absent"], None
    reasons += amendment4_issues(tree, a4, st, f)
    for k in ("run", "smoke"):
        if a4.get(k) and os.path.exists(os.path.join(tree, a4[k], "log.jsonl")):
            reasons.append(f"{a4[k]}/log.jsonl exists: a number of the run exists before the hash")
    if a4["hashed"]:
        reasons.append("rule_amendment_4.sha256 exists: the amendment is hashed already")
    out = {"second_reading": None, "alarm_lines": [], "unit_alarm_lines": []}
    if f["start"] is not None:
        first, sec = start_decodes(tree, f, second)
        att = attainability(tree, first, sec, sets, locked)
        if not att["attainable"]:
            reasons += [f"not attainable: {x}" for x in att["reasons"]]
        sr = second_reading(tree, first, sec, sets, locked, att)
        out["second_reading"] = sr
        a2 = amendment2_state(tree)
        reasons += [f"amendment 2 lacks the line: {x}" for x in amendment_lines(sr)
                    if a2 is None or x not in a2["text"]]
    if a4["config"] is not None:
        cfg, why = _load_config(tree, a4["config"])
        if cfg is not None:
            a3 = amendment3_state(tree)
            lines, why3 = alarm_reading(tree, cfg)
            out["alarm_lines"] = lines
            reasons += [f"amendment 3 lacks the line: {x}" for x in lines if a3 is None or x not in a3["text"]]
            if why3:
                reasons.append(f"no beta_error false-alarm lines: {why3}")
            lines, why = unit_alarm_reading(tree, cfg)
            out["unit_alarm_lines"] = lines
            reasons += [f"the amendment lacks the line: {x}" for x in lines if x not in a4["text"]]
        if why:
            reasons.append(f"no unit_nll_rise false-alarm lines: {why}")
    return reasons, out


def write_amendment4_hash(tree, sets, locked, second=None, now=None):
    """rule_amendment_4.sha256 written when amendment4_refusals finds nothing, with the local time of the hash;
    returns (written, reasons, {second_reading, alarm_lines, unit_alarm_lines})."""
    reasons, got = amendment4_refusals(tree, sets, locked, second)
    if reasons:
        return False, reasons, got
    a4 = amendment4_state(tree)
    stamp = (now or datetime.datetime.now()).isoformat(timespec="seconds")
    with open(os.path.join(tree, AMEND4_SHA_REL), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{a4['sha256']}  rule_amendment_4.txt\nhashed_at {stamp}\n")
    return True, [], got


def run_binding4(tree, st, run_dir, recs=None, expect2=None, expect3=None, expect4=None):
    """(fields, amendment 2 record, amendment 3 record, amendment 4 record) of a run (module docstring). A run folder
    that amendment 4 names (Run or Smoke): hashed and binding (amendment4_issues empty, hashed before the run's first
    log line), the rule's fields with the Config of amendment 4 and its record {path, sha256, hashed_at,
    hash_file_mtime, run, smoke, config, amendment_2, amendment_3, read}; while it is not hashed (None, None, None,
    record with read False and the reason), which the read and the smoke check refuse; `expect2` and `expect3` then
    name amendments 2's and 3's identity only (present, hashed, at that SHA-256: amendment 4 binds together with them),
    not a run they name. Any other run: run_binding, unchanged, and None for amendment 4. HarnessDefect: `expect4`
    given while amendment 4 is absent, not hashed, hashes otherwise or does not name the run; `expect2` or `expect3`
    given while that amendment is absent, not hashed or hashes otherwise; a named run whose amendment 4 does not bind or
    was hashed at or after the run's first log line."""
    f = st["fields"]
    a4 = amendment4_state(tree)
    if expect4 and (a4 is None or not a4["hashed"] or a4["sha256"] != expect4):
        raise HarnessDefect(f"amendment 4 is absent, not hashed or hashes otherwise; the job expects {expect4[:12]}")
    if not amendment4_names(tree, a4, run_dir):
        if expect4:
            raise HarnessDefect(f"amendment 4 does not name the run {run_dir}")
        f2, a2, a3 = run_binding(tree, st, run_dir, recs, expect2, expect3)
        return f2, a2, a3, None
    for n, exp, state in ((2, expect2, amendment2_state), (3, expect3, amendment3_state)):
        if exp:
            s = state(tree)
            if s is None or not s["hashed"] or s["sha256"] != exp:
                raise HarnessDefect(f"amendment {n} is absent, not hashed or hashes otherwise; the job expects "
                                    f"{exp[:12]}")
    rec = {"path": AMEND4_REL.replace(os.sep, "/"), "sha256": a4["sha256"], "hashed_at": a4["hashed_at"],
           "hash_file_mtime": a4["hash_mtime"], "run": a4["run"], "smoke": a4["smoke"], "config": a4["config"],
           "amendment_2": a4["amendment_2"], "amendment_3": a4["amendment_3"], "read": bool(a4["hashed"])}
    if not a4["hashed"]:
        return None, None, None, dict(rec, reason="amendment 4 names this run and is not hashed")
    issues = amendment4_issues(tree, a4, st, f)
    if issues:
        raise HarnessDefect("amendment 4 is hashed but does not bind: " + "; ".join(issues))
    issue = hash_time_issue(a4, recs)
    if issue:
        raise HarnessDefect(f"amendment 4: {issue}")
    return dict(f, config=dict(a4["config"])), None, None, rec


def read_run(tree, run_dir, sets, locked, second=None, job_log=None, expect_sha256=None, expect_amendment=None,
             expect_amendment2=None, expect_amendment3=None, expect_amendment4=None):
    """(results, exit code) of the read of a stage-2 run (module docstring); EXIT_REFUSED when the rule is not
    hashed or not final; job_log: the queue's job log read by R0 (default the tree's); expect_sha256: the rule's
    SHA-256 that the queued read job carries (S2.11), a HarnessDefect when the rule file hashes otherwise;
    expect_amendment: the SHA-256 of amendment 1 that the job carries, a HarnessDefect when the amendment is absent,
    not hashed, or hashes otherwise. With amendment 1 hashed, the verdict, its lines and the exit code are those of
    the rule with amendment 1, and the rule alone stands beside them (rule_alone), as the F2 lever-2 amendment
    did. Amendment 2 (task S2B; amendment2_binding): for the run it names, the Config of amendment 2 replaces the
    rule's in R0 (both readings), everything else is the rule's and amendment 1's; expect_amendment2, the SHA-256 the
    S2b read job carries, makes an absent, unhashed or other amendment 2, or one that does not name the run, a
    HarnessDefect; a run that amendment 2 names while it is not hashed is refused (EXIT_REFUSED). Amendment 3 (task
    S2C; run_binding): for the run it names, the Config of amendment 3 replaces the rule's in R0 (both readings), the
    'alarm_watch' events of the log stand beside the verdict (alarm_watch_events, deciding nothing), everything else is
    the rule's and amendment 1's; expect_amendment3 (the S2c read job's) makes an absent, unhashed or other amendment 3,
    or one that does not name the run, a HarnessDefect, and expect_amendment2 then names amendment 2's identity only; a
    run that amendment 3 names while it is not hashed is refused (EXIT_REFUSED). Amendment 4 (task S2D; run_binding4):
    for the run it names, the Config of amendment 4 replaces the rule's in R0 (both readings), the 'alarm_watch' events
    stand beside the verdict with the rising units of each unit_nll_rise entry (deciding nothing), everything else is
    the rule's and amendment 1's; expect_amendment4 (the S2d read job's) makes an absent, unhashed or other amendment 4,
    or one that does not name the run, a HarnessDefect, and expect_amendment2 and expect_amendment3 then name those
    amendments' identity only; a run that amendment 4 names while it is not hashed is refused (EXIT_REFUSED). A run no
    amendment names reads as before: the results gain the keys amendment_3 and alarm_watch only under amendment 3 or 4
    or with 'alarm_watch' events, and amendment_4 only under amendment 4."""
    st = rule_state(tree)
    f = st["fields"]
    if expect_sha256 and st["sha256"] != expect_sha256:
        raise HarnessDefect(f"the rule hashes {st['sha256'][:12]}, the read job expects {expect_sha256[:12]}")
    if expect_amendment:
        am = amendment_state(tree)
        if am is None or not am["hashed"] or am["sha256"] != expect_amendment:
            raise HarnessDefect(f"amendment 1 is absent, not hashed or hashes otherwise; the read job expects "
                                f"{expect_amendment[:12]}")
    if not st["hashed"] or f["status"] != "final" or f["start"] is None or f["config"] is None:
        return {"rule": {"sha256": st["sha256"], "hashed": st["hashed"], "fields": f},
                "verdict": {"verdict": NOT_EVALUATED, "case": 0,
                            "reasons": ["the rule is not hashed and final with its Start and Config"]}}, EXIT_REFUSED
    log_path = os.path.join(tree, run_dir, "log.jsonl")
    recs, unparsed = WB().read_jsonl(log_path)
    issue = hash_time_issue(st, recs)
    if issue:
        raise HarnessDefect(issue)
    f, a2, a3, a4 = run_binding4(tree, st, run_dir, recs, expect_amendment2, expect_amendment3, expect_amendment4)
    if f is None:
        out = {"rule": {"sha256": st["sha256"], "hashed": st["hashed"], "fields": st["fields"]}, "amendment_2": a2,
               "verdict": {"verdict": NOT_EVALUATED, "case": 0, "reasons": [(a4 or a3 or a2)["reason"]]}}
        if a3 is not None:
            out["amendment_3"] = a3
        if a4 is not None:
            out["amendment_4"] = a4
        return out, EXIT_REFUSED
    r0 = r0_reading(recs, f, unparsed, tree, run_dir, job_log)
    first, sec = start_decodes(tree, f, second)
    att = attainability(tree, first, sec, sets, locked)
    missing = [x for x in section7_lines(att) if x not in section7(st["text"])]
    if missing:
        raise HarnessDefect("the margins differ from the rule's section 7: " + "; ".join(missing))
    start_name = first["name"]
    end_name = (r0.get("end") or {}).get("name")
    start_lines = read_decode(tree, first, sets, locked, f["start"]["sha256"])
    if end_name:
        end_lines = read_decode(tree, decode_spec(tree, end_name), sets, locked, (r0["end"] or {}).get("sha256"))
    else:
        end_lines = {k: {"point": None, "reason": "no end checkpoint (R0)"} for k in LINES + REPORTED}
    lines = lines_reading(end_lines, start_lines, att)
    v = verdict(r0, lines)
    noise = decode_noise(v, lines, att)
    a1 = amendment_reading(tree, st, recs, r0, end_lines, start_lines, first, sec, sets, locked, att, v, lines)
    bound = bool(a1 and a1.get("read"))
    res = {"task": TASK, "label": LABEL, "run_dir": run_dir.replace("\\", "/"), "start": start_name,
           "end": end_name, "rule": {"path": RULE_REL.replace("\\", "/"), "sha256": st["sha256"],
                                     "fields": st["fields"], "hashed_at": st["hashed_at"],
                                     "hash_file_mtime": st["hash_mtime"]},
           "r0": r0, "attainability": att, "lines": a1["lines"] if bound else lines,
           "verdict": a1["verdict"] if bound else v, "decode_noise": [] if bound else noise,
           "amendment_1": None if a1 is None else {k: x for k, x in a1.items() if k not in ("lines", "verdict")},
           "amendment_2": a2,
           "rule_alone": {"lines": lines, "verdict": v, "decode_noise": noise} if bound else None,
           "beside": reported_beside(tree, end_name, start_name, recs, (r0.get("end") or {}).get("sha256"),
                                     f["start"]["sha256"])}
    watch = alarm_watch_events(recs, units=a4 is not None)
    if a3 is not None:
        res["amendment_3"] = a3
    if a4 is not None:
        res["amendment_4"] = a4
    if a3 is not None or a4 is not None or watch:   # beside the verdict, deciding nothing (amendments 3 and 4)
        res["alarm_watch"] = watch
    code = EXIT_VERDICT if res["verdict"]["verdict"] in (KEEP, FORGETS, STOPPED) else EXIT_NOT_EVALUATED
    return clean(res), code


def pending(res, tree):
    """None, or why the read must not be written yet (S2.11): no stop, and either the run's training job has no exit
    in the queue's job log while the run has no end, or the end exists and one of its three deciding decodes
    (END_DECODE_FILES under decode_spec) is missing. A run that stopped, or whose job exited without an end, is
    written: its R0 failure is final."""
    r0 = res.get("r0") or {}
    if r0.get("stop"):
        return None
    end = res.get("end")
    if not end:
        if (r0.get("values") or {}).get("job_exit") is None:
            return "the run has no end and its training job no exit in the job log"
        return None
    dec = decode_spec(tree, end)
    gone = [rel(os.path.join(dec[k], fn), tree) for k, fn in END_DECODE_FILES
            if not os.path.exists(os.path.join(dec[k], fn))]
    return ("the end's evaluations are missing: " + ", ".join(gone)) if gone else None


def summary_md(res):
    v = res["verdict"]
    L = [f"# Stage-2 healthy non-inferiority read ({LABEL})", "",
         f"Run {res.get('run_dir')}: end {res.get('end')}, start {res.get('start')}. Rule {res['rule']['path']} "
         f"(SHA-256 {res['rule']['sha256']}, hashed at {res['rule'].get('hashed_at')}; rule.sha256 file time "
         f"{res['rule'].get('hash_file_mtime')}, reported only).", "", f"Verdict: {v['verdict']} (case {v['case']}).",
         ""]
    L += [f"- {x}" for x in v.get("reasons") or []]
    for k in res.get("decode_noise") or []:
        L.append(f"- {NAMES[k]}: possibly decode noise (the paired interval of the start's two decodes reaches beyond "
                 "its margin; option (a) of the rule's section 7: named, the verdict unchanged)")
    a2 = res.get("amendment_2")
    if a2:
        L += ["", f"Run identity under amendment 2 ({a2['path']}, SHA-256 {a2['sha256']}, hashed at {a2['hashed_at']}): "
                  f"the run {a2['run']} under {(a2.get('config') or {}).get('path')} (configuration SHA-256 "
                  f"{(a2.get('config') or {}).get('sha256')}) in place of the rule's Config line, in both readings "
                  "below; the margins, lines, verdict cases and routes are those of the rule and amendment 1."]
    a3 = res.get("amendment_3")
    if a3:
        L += ["", f"Run identity under amendment 3 ({a3['path']}, SHA-256 {a3['sha256']}, hashed at {a3['hashed_at']}; "
                  f"with amendment 2 {(a3.get('amendment_2') or {}).get('sha256')}): the run {a3['run']} under "
                  f"{(a3.get('config') or {}).get('path')} (configuration SHA-256 "
                  f"{(a3.get('config') or {}).get('sha256')}: beta_error raised at three log records in a row) in "
                  "place of the rule's Config line, in both readings below; the margins, lines, verdict cases and "
                  "routes are those of the rule and amendment 1."]
    a4 = res.get("amendment_4")
    if a4:
        L += ["", f"Run identity under amendment 4 ({a4['path']}, SHA-256 {a4['sha256']}, hashed at {a4['hashed_at']}; "
                  f"with amendments 2 {(a4.get('amendment_2') or {}).get('sha256')} and 3 "
                  f"{(a4.get('amendment_3') or {}).get('sha256')}): the run {a4['run']} under "
                  f"{(a4.get('config') or {}).get('path')} (configuration SHA-256 "
                  f"{(a4.get('config') or {}).get('sha256')}: unit_nll_rise raised at two dev reads in a row, "
                  "beta_error at three log records in a row) in place of the rule's Config line, in both readings "
                  "below; the margins, lines, verdict cases and routes are those of the rule and amendment 1."]
    if "alarm_watch" in res:
        w = res.get("alarm_watch") or []
        L += ["", f"Beside the verdict, deciding nothing ({'amendment 4' if a4 else 'amendment 3'}, section 2): "
                  f"{len(w)} 'alarm_watch' events" + (":" if w else ".")]
        L += [f"- phase {e.get('phase')} step {e.get('step')}: " + "; ".join(
            f"{h.get('name')} {fmt(h.get('value'), 4)} ({h.get('consecutive')} of {h.get('of')} in a row"
            + (f"; rising: {', '.join(h['units']) or 'none named'}" if "units" in h else "") + ")"
            for h in e.get("alarms") or []) for e in w]
    a1, alone = res.get("amendment_1"), res.get("rule_alone")
    if a1 is not None and not a1.get("read"):
        L += ["", f"Amendment 1: not read ({a1.get('reason')}); the verdict above is the rule's alone."]
    elif a1 is not None and alone is not None:
        L += ["", f"The verdict above is the rule's with amendment 1 ({a1['path']}, SHA-256 {a1['sha256']}, hashed at "
                  f"{a1['hashed_at']}): each margin floored at the start's decode noise."]
        for k in LINES:
            m = a1["margins"][k]
            if m.get("raised2"):
                L.append(f"- {NAMES[k]}: margin {fmt(m['margin2'], ND[k] + 1)} {UNIT[k]} with amendment 1, "
                         f"{fmt(m['margin'], ND[k] + 1)} {UNIT[k]} under the rule alone")
        va = alone["verdict"]
        L += ["", f"Beside it, the rule alone (version 1; its section 10 keeps both readings): {va['verdict']} "
                  f"(case {va['case']})."]
        L += [f"- {x}" for x in va.get("reasons") or []]
        for k in alone.get("decode_noise") or []:
            L.append(f"- {NAMES[k]}: possibly decode noise (the paired interval of the start's two decodes reaches "
                     "beyond its margin under the rule alone)")
        if a1["agrees"]:
            L.append("- the two readings agree")
        else:
            L.append("- the two readings differ, on " + (", ".join(NAMES[k] for k in a1["differs_on"]) or "no line")
                     + ": there decoding alone can carry the point past the rule's margin (amendment 1, section 1)")
    L += ["", "| line | start | end | end minus start | paired interval | margin | holds |",
          "|---|---|---|---|---|---|---|"]
    for k in LINES + REPORTED:
        r = res["lines"][k]
        nd = ND[k]
        iv = (r.get("interval") or {}).get("interval")
        holds = ("HOLDS" if r["holds"] else "FORGETS") if r["read"] and r["decides"] else (
            "reported" if r["read"] else f"not read: {r['reason']}")
        L.append(f"| {NAMES[k]} ({UNIT[k]}) | {fmt(r['start'], nd)} | {fmt(r['end'], nd)} | "
                 f"{fmt(r['difference'], nd, True)} | {fmt_iv(iv, nd)} | {fmt(r['margin'], nd)} | {holds} |")
    b = res["beside"]
    L += ["", "## Reported beside, deciding nothing", ""]
    for key, label in (("pd_half", "G3 PD half, row pd_t19dev"), ("g4_stroke_fold", "G4 on the stroke fold"),
                       ("sp3", "SP3 families")):
        for side in ("end", "start"):
            e = b[key][side]
            if not e.get("read"):
                L.append(f"- {label}, {side}: not read ({e.get('reason')})")
            elif key == "sp3":
                L.append(f"- {label}, {side}: " + ", ".join(f"{f} {x['reading']}/{x['prior']}"
                                                            for f, x in e["families"].items()))
            else:
                L.append(f"- {label}, {side}: " + ", ".join(f"{ln} {fmt(x['point'], 4)} ({x['n_persons']} persons)"
                                                            for ln, x in e["lines"].items()))
    u = b["units"]
    if u.get("read"):
        L.append(f"- per-unit dev numbers at step {u['step']} (unit_nll_rise {u['unit_nll_rise']}): " + "; ".join(
            f"{k} held-out NLL {fmt(x['held_out_nll'], 4)}, train NLL {fmt(x['train_nll'], 4)}, outside limits "
            f"{fmt(x['held_out_outside_share'], 4)}" for k, x in u["units"].items()))
    else:
        L.append(f"- per-unit dev numbers: not read ({u.get('reason')})")
    L.append(f"- cohort probe: {b['cohort_probe']['reason']}")
    return "\n".join(L) + "\n"


def write_read(out_dir, res, tree, stamp):
    from mdfm.gates import common
    if os.path.exists(os.path.join(out_dir, "results.json")):
        raise FileExistsError(os.path.join(out_dir, "results.json"))
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(out_dir, "results.json"),
                      json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    prov = {"script": "scripts/stage2_read.py", "task": TASK, "label": LABEL,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "rule_sha256": res["rule"]["sha256"],
            "amendment_2_sha256": (res.get("amendment_2") or {}).get("sha256"),
            **({"amendment_3_sha256": res["amendment_3"].get("sha256")} if res.get("amendment_3") else {}),
            **({"amendment_4_sha256": res["amendment_4"].get("sha256")} if res.get("amendment_4") else {}),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "weekend_branch.py"),
                                               os.path.join(SCRIPTS, "f2_lever2.py")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    return text


# ---------------------------------------------------------------- post-train set
def _job(name, priority, est, argv, outputs, requires=None, gpu=True, uses=None, after=None, not_if=None, note=""):
    j = {"name": name, "priority": priority, "group": "post_{run}", "gpu": gpu, "est_hours": est,
         "argv": [PYTHON] + argv, "outputs": outputs, "requires": requires or [], "note": note}
    if uses:
        j["uses"] = uses
    if after:
        j["after"] = after
    if not_if:
        j["not_if"] = not_if
    return j


G3_DIAG_USES = ["scripts/eval_g3_diag.py", "scripts/eval_g3_dev.py", "scripts/eval_gait_insim.py",
                "scripts/moveatlas_stamp.py"]
READ_USES = ["scripts/stage2_read.py", "scripts/f2_lever2.py", "scripts/weekend_branch.py",
             "scripts/g3_gain_check.py", "scripts/queue_job.py"]


def decode_jobs(ck, name, suffix="", prefix="pt2", run_key="{run}", priority=1):
    """The three decodes the read reads (G3 dev, G3 diag, G1 dev) of checkpoint `ck`, report names `name` + suffix,
    at the settings of the stage-1 templates post_j1 and post_diag."""
    n = f"{name}{suffix}"
    tag = suffix.strip("_")
    nm = (lambda k: f"{prefix}_{k}{('_' + tag) if tag else ''}_{run_key}")
    return [
        _job(nm("g3_dev"), priority, 0.3,
             ["scripts/eval_g3_dev.py", "--checkpoint", ck, "--out", f"reports/gates/g3/dev_{n}", "--memory-gb", "12",
              "--batch-windows", "32"], [f"reports/gates/g3/dev_{n}"], [ck],
             note="S2.10: the read's stride_mape_pct and IC lines (G3 dev, EMA)"),
        _job(nm("diag"), priority + 1, 0.25,
             ["scripts/eval_g3_diag.py", "--checkpoint", ck, "--out", f"reports/gates/g3/diag_{n}", "--memory-gb",
              "12", "--batch-windows", "32"], [f"reports/gates/g3/diag_{n}"], [ck], uses=G3_DIAG_USES,
             note="S2.10: the read's S(12), V(12), S(14), V(14) (G3 diag base row)"),
        _job(nm("g1_dev"), priority + 2, 0.75,
             ["scripts/eval_g1_dev.py", "--checkpoint", ck, "--out", f"reports/gates/g1/dev_{n}", "--memory-gb", "12",
              "--batch-windows", "64", "--block-windows", "64"], [f"reports/gates/g1/dev_{n}"], [ck],
             note="S2.10: the read's G1a line (full G1 dev run)"),
    ]


def stage2_template():
    """The post-train template of a stage-2 run (module docstring, Post-train set)."""
    jobs = decode_jobs("{ckpt}", "{name}")
    jobs += [
        _job("pt2_g3_pd_dev_{run}", 10, 0.3,
             ["scripts/eval_g3_pd_dev.py", "--checkpoint", "{ckpt}", "--batch-windows", "32", "--memory-gb", "12",
              "--out", "reports/gates/g3/dev_pd_{name}"], ["reports/gates/g3/dev_pd_{name}"], ["{ckpt}"],
             note="S2.10: G3 PD half of the end on the PD dev fold (spec section 4.12), reported beside the read"),
        _job("pt2_g4_fold_{run}", 11, 0.5,
             ["scripts/eval_g4_dev.py", "--checkpoint", "{ckpt}", "--stroke-set", "fold", "--out",
              "reports/gates/g4/dev_{name}_strokefold", "--memory-gb", "12"],
             ["reports/gates/g4/dev_{name}_strokefold"], ["{ckpt}"],
             note="S2.10: G4 of the end on the stroke dev fold of 8 (spec section 4.12), reported beside the read"),
        _job("pt2_sp3_{run}", 12, 0.5,
             ["scripts/sp3_decoder_prior_probe.py", "--checkpoint", "{ckpt}", "--weights", "ema", "--device", "cuda",
              "--memory-gb", "12", "--dev-windows", "32", "--out", "reports/synth/sp3/{name}"],
             ["reports/synth/sp3/{name}"], ["{ckpt}"],
             note="S2.10: SP3 families of the end (spec section 4.12), reported beside the read"),
        _job("pt2_start_g3_pd_dev_{run}", 13, 0.3,
             ["scripts/eval_g3_pd_dev.py", "--checkpoint", "{start_ckpt}", "--batch-windows", "32", "--memory-gb",
              "12", "--out", "reports/gates/g3/dev_pd_{start_name}"], ["reports/gates/g3/dev_pd_{start_name}"],
             ["{start_ckpt}"], note="S2.10: G3 PD half of the start, left out when its folder exists"),
        _job("pt2_start_g4_fold_{run}", 14, 0.5,
             ["scripts/eval_g4_dev.py", "--checkpoint", "{start_ckpt}", "--stroke-set", "fold", "--out",
              "reports/gates/g4/dev_{start_name}_strokefold", "--memory-gb", "12"],
             ["reports/gates/g4/dev_{start_name}_strokefold"], ["{start_ckpt}"],
             note="S2.10: G4 of the start on the stroke fold, left out when its folder exists"),
        _job("pt2_start_sp3_{run}", 15, 0.5,
             ["scripts/sp3_decoder_prior_probe.py", "--checkpoint", "{start_ckpt}", "--weights", "ema", "--device",
              "cuda", "--memory-gb", "12", "--dev-windows", "32", "--out", "reports/synth/sp3/{start_name}"],
             ["reports/synth/sp3/{start_name}"], ["{start_ckpt}"],
             note="S2.10: SP3 families of the start, left out when its folder exists"),
    ]
    if not os.path.isfile(os.path.join(TREE, "scripts", "sp3_decoder_prior_probe.py")):
        # the SP3 probe belongs to the synthetic patient catalogue, which is not in every copy of this tree; its two
        # jobs are reported beside the read, never inputs of it, so the template leaves them out there
        jobs = [j for j in jobs if "_sp3_" not in j["name"]]
    beside = ["pt2_g3_pd_dev_{run}", "pt2_g4_fold_{run}", "pt2_sp3_{run}", "pt2_start_g3_pd_dev_{run}",
              "pt2_start_g4_fold_{run}", "pt2_start_sp3_{run}"]
    jobs.append(_job("pt2_read_{run}", 90, 0.05,
                     ["scripts/stage2_read.py", "--read", "--run-dir", "{run_dir}", "--write"],
                     ["reports/bench/stage2_read/{run}"], gpu=False, uses=READ_USES,
                     after=["pt2_g3_dev_{run}", "pt2_diag_{run}", "pt2_g1_dev_{run}"], not_if=beside,
                     note="S2.10: the healthy non-inferiority read under the hashed rule (CPU)"))
    return {"name": TEMPLATE_NAME, "cls": 1, "runs": ["*"], "jobs": jobs, "format": TEMPLATE_FORMAT,
            "placeholders": list(STAGE2_PLACEHOLDERS),
            "note": "stage-2 post-train set (scripts/stage2_read.py, task S2.10); expanded by stage2_read.py "
                    "--post-train after a phase C verify event, never by scripts/gpu_queue.py"}


def write_templates(tree):
    """Write stage2_template() as TEMPLATE_DIR/post_stage2.json (checked by queue_job.normalize_template)."""
    t = stage2_template()
    QJ().normalize_template(t)
    folder = os.path.join(tree, TEMPLATE_DIR)
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, f"{TEMPLATE_NAME}.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(t, indent=1) + "\n")
    os.replace(tmp, p)
    return p


def load_templates(tree):
    folder = os.path.join(tree, TEMPLATE_DIR)
    out = []
    if not os.path.isdir(folder):
        return out
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json") and not fn.startswith((".", "_")):
            with open(os.path.join(folder, fn), encoding="utf-8") as f:
                out.append(QJ().normalize_template(json.load(f)))
    return out


def run_values(tree, run_dir):
    """{placeholder: value} of a stage-2 run from its log.jsonl: the phase C verify event (the end) and the
    init_stage2 event (the start); ValueError for a run without them."""
    qj, wb = QJ(), WB()
    recs, _ = wb.read_jsonl(os.path.join(tree, run_dir, "log.jsonl"))
    if not recs:
        raise ValueError(f"no log.jsonl in {run_dir}")
    init = wb.log_events(recs, "init_stage2")
    v = wb.last_verify(recs, "C")
    if not init:
        raise ValueError(f"{run_dir}: no init_stage2 event, not a stage-2 run")
    if v is None:
        raise ValueError(f"{run_dir}: no phase C verify event")
    run_rel = qj.tree_rel(os.path.join(tree, run_dir), tree).rstrip("/")
    ck = qj.tree_rel(os.path.join(tree, str(v["path"])) if not os.path.isabs(str(v["path"])) else str(v["path"]),
                     tree)
    src = str((init[-1].get("source") or {}).get("path") or "")
    sck = qj.tree_rel(os.path.join(tree, src) if not os.path.isabs(src) else src, tree)
    return {"ckpt": ck, "name": checkpoint_name(os.path.join(tree, ck)), "run_dir": run_rel,
            "run": os.path.basename(run_rel), "start_ckpt": sck,
            "start_name": checkpoint_name(os.path.join(tree, sck))}


def expand_post_train(tree, run_dir, write=False, jobs_folder=None):
    """{values, jobs, skipped, written} of the stage-2 templates for a run (module docstring)."""
    qj = QJ()
    vals = run_values(tree, run_dir)
    jobs, skipped = [], []
    for t in load_templates(tree):
        if not qj.runs_match(t["runs"], vals["run"]):
            continue
        for tj in t["jobs"]:
            jd = qj.substitute(dict(tj, cls=t["cls"]), vals)
            jd.update(template=t["name"], run=vals["run"])
            left = qj.placeholders({k: jd.get(k) for k in ("argv", "requires", "outputs", "after", "not_if")})
            if left:
                raise ValueError(f"{jd['name']}: placeholders left {sorted(left)}")
            outs = jd.get("outputs") or []
            if outs and all(os.path.exists(os.path.join(tree, o)) for o in outs):
                skipped.append(jd["name"])
                continue
            jobs.append(jd)
    for jd in jobs:
        for f in ("after", "not_if"):
            if jd.get(f):
                jd[f] = [x for x in jd[f] if x not in skipped]
    for jd in jobs:
        qj.normalize(jd)
    written = []
    if write:
        for jd in jobs:
            written.append(qj.write_job(jd, tree=tree, folder=jobs_folder))
    return {"values": vals, "jobs": jobs, "skipped": skipped, "written": written}


def planned_post_train(tree, run_dir, end_step, start_ckpt, after, second=None, expect_sha256=None,
                       expect_amendment=None, expect_amendment2=None, expect_amendment3=None, expect_amendment4=None):
    """{values, jobs, skipped} of the stage-2 templates for a run not trained yet (S2.11): the values of run_values
    from what the queue fixes before the run (the end ckpt_C_<end_step>.pt of run_dir and the start), so that the
    job files are written, and checked, before the training starts. As expand_post_train: a job whose outputs all
    exist is left out and dropped from the after and not_if lists. Added: every job of the end waits for the
    training jobs `after` (exit 0) beside its requires on the end checkpoint; the read job carries the second
    decodes of the start (`second`, the folders of --hash), --expect-rule-sha256 and, when given,
    --expect-amendment-sha256 and --expect-amendment-2-sha256 (task S2B; a placeholder such as
    '{amendment_2_sha256}' keeps the job waiting in the runner until it is replaced by the hash), and
    --expect-amendment-3-sha256 (task S2C; '{amendment_3_sha256}' likewise) and --expect-amendment-4-sha256 (task S2D;
    '{amendment_4_sha256}' likewise). The runner expands templates on a phase B verify event only
    (scripts/gpu_queue.py), so a phase C run gets its set as these files."""
    qj = QJ()
    run_rel = qj.tree_rel(os.path.join(tree, run_dir), tree).rstrip("/")
    ck = f"{run_rel}/ckpt_C_{int(end_step):06d}.pt"
    sck = qj.tree_rel(start_ckpt if os.path.isabs(start_ckpt) else os.path.join(tree, start_ckpt), tree)
    if not os.path.exists(os.path.join(tree, sck)):
        raise ValueError(f"start checkpoint {sck} not found")
    vals = {"ckpt": ck, "name": checkpoint_name(os.path.join(tree, ck)), "run_dir": run_rel,
            "run": os.path.basename(run_rel), "start_ckpt": sck, "start_name": checkpoint_name(os.path.join(tree, sck))}
    jobs, skipped = [], []
    for t in load_templates(tree):
        if not qj.runs_match(t["runs"], vals["run"]):
            continue
        for tj in t["jobs"]:
            jd = qj.substitute(dict(tj, cls=t["cls"]), vals)
            jd.update(template=t["name"], run=vals["run"])
            left = qj.placeholders({k: jd.get(k) for k in ("argv", "requires", "outputs", "after", "not_if")})
            if left:
                raise ValueError(f"{jd['name']}: placeholders left {sorted(left)}")
            outs = jd.get("outputs") or []
            if outs and all(os.path.exists(os.path.join(tree, o)) for o in outs):
                skipped.append(jd["name"])
                continue
            if ck in (jd.get("requires") or []) or "--read" in jd["argv"]:
                jd["after"] = list(dict.fromkeys(list(after) + list(jd.get("after") or [])))
            if "--read" in jd["argv"]:
                for k in ("diag", "g3dev", "g1"):
                    if (second or {}).get(k):
                        jd["argv"] += [f"--second-{k}", qj.tree_rel(second[k], tree)]
                if expect_sha256:
                    jd["argv"] += ["--expect-rule-sha256", expect_sha256]
                if expect_amendment:
                    jd["argv"] += ["--expect-amendment-sha256", expect_amendment]
                if expect_amendment2:
                    jd["argv"] += ["--expect-amendment-2-sha256", expect_amendment2]
                if expect_amendment3:
                    jd["argv"] += ["--expect-amendment-3-sha256", expect_amendment3]
                if expect_amendment4:
                    jd["argv"] += ["--expect-amendment-4-sha256", expect_amendment4]
            jobs.append(jd)
    for jd in jobs:
        for f in ("after", "not_if"):
            if jd.get(f):
                jd[f] = [x for x in jd[f] if x not in skipped]
    return {"values": vals, "jobs": [qj.normalize(jd) for jd in jobs], "skipped": skipped}


def attainability_jobs(tree, start_ckpt, write=False, jobs_folder=None):
    """The start's decodes for the attainability check: the second ones (ATT_SUFFIX) always, the first ones when
    their folder is absent; class 1, group stage2_att. Returns (jobs, written)."""
    qj = QJ()
    ck = qj.tree_rel(start_ckpt, tree)
    name = checkpoint_name(os.path.join(tree, ck))
    jobs = []
    for suffix in ("", ATT_SUFFIX):
        for j in decode_jobs(ck, name, suffix, prefix="att", run_key=name, priority=40 if suffix else 30):
            if suffix == "" and all(os.path.exists(os.path.join(tree, o)) for o in j["outputs"]):
                continue
            j = dict(j, cls=1, group="stage2_att",
                     note=j["note"] + f"; S2.10 attainability decode{' 2' if suffix else ' 1'} of the start")
            jobs.append(qj.normalize(j))
    written = [qj.write_job(j, tree=tree, folder=jobs_folder) for j in jobs] if write else []
    return jobs, written


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--attainability", action="store_true")
    g.add_argument("--hash", action="store_true")
    g.add_argument("--read", action="store_true")
    g.add_argument("--post-train", default=None, metavar="RUN_DIR")
    g.add_argument("--attainability-jobs", default=None, metavar="CKPT")
    g.add_argument("--write-templates", action="store_true")
    g.add_argument("--smoke-check", default=None, metavar="RUN_DIR")
    g.add_argument("--post-train-files", default=None, metavar="RUN_DIR")
    g.add_argument("--hash-amendment", action="store_true", help="hash amendment 1 (its second reading)")
    g.add_argument("--hash-amendment-2", action="store_true",
                   help="hash amendment 2 (the S2b run, its config and recipe; task S2B)")
    g.add_argument("--hash-amendment-3", action="store_true",
                   help="hash amendment 3 (the S2c run, beta_error on three records in a row; task S2C)")
    g.add_argument("--hash-amendment-4", action="store_true",
                   help="hash amendment 4 (the S2d run, unit_nll_rise on two dev reads in a row; task S2D)")
    p.add_argument("--start", default=None, help="--attainability: checkpoint name (default the rule's Start)")
    p.add_argument("--second-diag", default=None)
    p.add_argument("--second-g3dev", default=None)
    p.add_argument("--second-g1", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument("--expect-rule-sha256", default=None, help="--read: the rule's SHA-256 the job expects")
    p.add_argument("--expect-amendment-sha256", default=None,
                   help="--read, --post-train-files: the SHA-256 of amendment 1 the read job expects")
    p.add_argument("--expect-amendment-2-sha256", default=None,
                   help="--read, --post-train-files: the SHA-256 of amendment 2 the read job expects (task S2B)")
    p.add_argument("--expect-amendment-3-sha256", default=None,
                   help="--read, --post-train-files: the SHA-256 of amendment 3 the read job expects (task S2C)")
    p.add_argument("--expect-amendment-4-sha256", default=None,
                   help="--read, --post-train-files: the SHA-256 of amendment 4 the read job expects (task S2D)")
    p.add_argument("--amendment", action="store_true",
                   help="--attainability: also the second reading of amendment 1 (written with the suffix "
                        "_amendment_1)")
    p.add_argument("--train-steps", type=int, default=None, help="--smoke-check: the training run's phase C steps")
    p.add_argument("--end-step", type=int, default=None, help="--post-train-files: the phase C step of the end")
    p.add_argument("--start-ckpt", default=None, help="--post-train-files: the start checkpoint")
    p.add_argument("--after", nargs="+", default=None, help="--post-train-files: the training job (exit 0)")
    return p.parse_args(argv)


def _sets(tree):
    from mdfm.gates import g3
    locked = g3.lockbox_persons()
    return F2().load_sets(read_json(os.path.join(tree, F1_RESULTS_REL)), locked), locked


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    tree = os.path.abspath(a.tree) if a.tree else TREE
    second = {k: getattr(a, f"second_{k}") for k in ("diag", "g3dev", "g1") if getattr(a, f"second_{k}")}
    if a.write_templates:
        print(f"written {rel(write_templates(tree), tree)}")
        return EXIT_VERDICT
    if a.post_train:
        res = expand_post_train(tree, a.post_train, write=a.write)
        for j in res["jobs"]:
            print(f"{'written' if a.write else 'would write'} {j['name']}: {' '.join(j['argv'][1:])}")
        for s in res["skipped"]:
            print(f"left out (outputs exist): {s}")
        return EXIT_VERDICT
    if a.post_train_files:
        if not (a.end_step and a.start_ckpt and a.after and a.out and a.expect_rule_sha256):
            raise SystemExit("--post-train-files needs --end-step, --start-ckpt, --after, --expect-rule-sha256 and "
                             "--out (the folder of the job files)")
        res = planned_post_train(tree, a.post_train_files, a.end_step, a.start_ckpt, a.after, second,
                                 a.expect_rule_sha256, a.expect_amendment_sha256, a.expect_amendment_2_sha256,
                                 a.expect_amendment_3_sha256, a.expect_amendment_4_sha256)
        os.makedirs(a.out, exist_ok=True)
        for j in res["jobs"]:
            fn = os.path.join(a.out, QJ().job_filename(j))
            with open(fn, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(j, indent=1) + "\n")
            print(f"job file {fn}: {j['name']}: {' '.join(j['argv'][1:])}")
        for x in res["skipped"]:
            print(f"left out (outputs exist): {x}")
        return EXIT_VERDICT
    if a.attainability_jobs:
        jobs, _ = attainability_jobs(tree, a.attainability_jobs, write=a.write)
        for j in jobs:
            print(f"{'written' if a.write else 'would write'} {j['name']}: {' '.join(j['argv'][1:])}")
        return EXIT_VERDICT
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the readings run on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    if a.smoke_check:
        if not a.train_steps:
            raise SystemExit("--smoke-check needs --train-steps")
        res = smoke_check(tree, a.smoke_check, a.train_steps)
        text = smoke_md(res)
        if a.write:
            out = a.out or os.path.join(tree, BENCH, "smoke_" + os.path.basename(os.path.normpath(a.smoke_check)))
            write_smoke(out, res, tree)
        print(text)
        return EXIT_VERDICT if res["passes"] else EXIT_SMOKE_FAILED
    sets, locked = _sets(tree)
    if a.attainability:
        fields = rule_state(tree)["fields"]
        if a.start:
            first, sec = decode_spec(tree, a.start), decode_spec(tree, a.start, ATT_SUFFIX, **second)
        elif fields["start"] is not None:
            first, sec = start_decodes(tree, fields, second)
        else:
            raise SystemExit("the rule names no start: pass --start NAME")
        att = attainability(tree, first, sec, sets, locked)
        text = attainability_text(att)
        extra = {}
        if a.amendment:
            sr = second_reading(tree, first, sec, sets, locked, att)
            text += "\n\n" + amendment_text(sr)
            am = amendment_state(tree)
            extra = {"second_reading": sr, "task": AMEND_TASK,
                     "amendment_sha256": None if am is None else am["sha256"]}
        print(text)
        if a.write:
            from mdfm.gates import common
            out = a.out or os.path.join(tree, BENCH)
            stem = f"attainability_{att['start']}" + (AMEND_SUFFIX if a.amendment else "")
            doc = dict(att, task=TASK, label=LABEL, run_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       rule_sha256=rule_state(tree)["sha256"],
                       moveatlas=_sibling("g3_gain_check").stamp_moveatlas(),
                       inputs=input_hashes(tree, [first, sec, decode_spec(tree, V1_B), decode_spec(tree, S2_B)]))
            doc = clean(dict(doc, **extra))
            common.write_text(os.path.join(out, stem + ".json"), json.dumps(doc, indent=1, allow_nan=False) + "\n")
            common.assert_no_em_dash(text, "attainability")
            common.write_text(os.path.join(out, stem + ".md"), "```\n" + text + "\n```\n")
            print(f"written {rel(os.path.join(out, stem), tree)}.json and .md")
        return EXIT_VERDICT if att["attainable"] else EXIT_NOT_EVALUATED
    if a.hash:
        ok, reasons, att = write_hash(tree, sets, locked, second)
        if att is not None:
            print(attainability_text(att))
        if not ok:
            print("HASH REFUSED:\n" + "\n".join(f"- {r}" for r in reasons))
            return EXIT_REFUSED
        print(f"written {RULE_SHA_REL}")
        return EXIT_VERDICT
    if a.hash_amendment:
        ok, reasons, sr = write_amendment_hash(tree, sets, locked, second)
        if sr is not None:
            print(amendment_text(sr))
        if not ok:
            print("HASH REFUSED:\n" + "\n".join(f"- {r}" for r in reasons))
            return EXIT_REFUSED
        print(f"written {AMEND_SHA_REL}")
        return EXIT_VERDICT
    if a.hash_amendment_2:
        try:
            ok, reasons, sr = write_amendment2_hash(tree, sets, locked, second)
        except HarnessDefect as e:
            print(f"HARNESS DEFECT: {e}")
            return EXIT_HARNESS
        if sr is not None:
            print(amendment_text(sr))
        if not ok:
            print("HASH REFUSED:\n" + "\n".join(f"- {r}" for r in reasons))
            return EXIT_REFUSED
        print(f"written {AMEND2_SHA_REL}")
        return EXIT_VERDICT
    if a.hash_amendment_3:
        try:
            ok, reasons, got = write_amendment3_hash(tree, sets, locked, second)
        except HarnessDefect as e:
            print(f"HARNESS DEFECT: {e}")
            return EXIT_HARNESS
        if got is not None:
            if got.get("second_reading") is not None:
                print(amendment_text(got["second_reading"]))
            print("\nAmendment 3, section 3 (the false alarms of the reading, deciding nothing):\n"
                  + "\n".join(got.get("alarm_lines") or []))
        if not ok:
            print("HASH REFUSED:\n" + "\n".join(f"- {r}" for r in reasons))
            return EXIT_REFUSED
        print(f"written {AMEND3_SHA_REL}")
        return EXIT_VERDICT
    if a.hash_amendment_4:
        try:
            ok, reasons, got = write_amendment4_hash(tree, sets, locked, second)
        except HarnessDefect as e:
            print(f"HARNESS DEFECT: {e}")
            return EXIT_HARNESS
        if got is not None:
            if got.get("second_reading") is not None:
                print(amendment_text(got["second_reading"]))
            print("\nAmendment 3, section 3 (the beta_error reading, unchanged; deciding nothing):\n"
                  + "\n".join(got.get("alarm_lines") or []))
            print("\nAmendment 4, section 3 (the false alarms of the unit_nll_rise reading, deciding nothing):\n"
                  + "\n".join(got.get("unit_alarm_lines") or []))
        if not ok:
            print("HASH REFUSED:\n" + "\n".join(f"- {r}" for r in reasons))
            return EXIT_REFUSED
        print(f"written {AMEND4_SHA_REL}")
        return EXIT_VERDICT
    if a.read:
        if not a.run_dir:
            raise SystemExit("--read needs --run-dir")
        try:
            res, code = read_run(tree, a.run_dir, sets, locked, second, expect_sha256=a.expect_rule_sha256,
                                 expect_amendment=a.expect_amendment_sha256,
                                 expect_amendment2=a.expect_amendment_2_sha256,
                                 expect_amendment3=a.expect_amendment_3_sha256,
                                 expect_amendment4=a.expect_amendment_4_sha256)
        except HarnessDefect as e:
            print(f"HARNESS DEFECT: {e}")
            return EXIT_HARNESS
        if code == EXIT_REFUSED:
            print("NOT EVALUATED: " + "; ".join(res["verdict"]["reasons"]))
            return code
        res["moveatlas"] = {"head": (_sibling("g3_gain_check").stamp_moveatlas() or {}).get("head")}
        text = summary_md(res)
        if code == EXIT_NOT_EVALUATED and not a.write:
            print(text)
            return code
        wait = pending(res, tree)
        if wait:
            print(f"NOT EVALUATED, nothing written: {wait}\n" + text)
            return EXIT_NOT_EVALUATED
        out = a.out or os.path.join(tree, BENCH, os.path.basename(os.path.normpath(a.run_dir)))
        try:
            write_read(out, res, tree, res["moveatlas"])
        except FileExistsError as e:
            print(f"{e} exists: nothing rewritten")
            return EXIT_EXISTS
        print(text)
        return code
    return EXIT_VERDICT


if __name__ == "__main__":
    sys.exit(main())
