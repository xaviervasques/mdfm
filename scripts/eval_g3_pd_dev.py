"""T27: the G3 PD-half dev estimate, the first patient estimate of the program, on the non-lockbox weargait_pd PD
persons (design section 9: "G3 IMU reads the pathological body, weargait_pd lockbox (37 subjects, PD and controls
separately)"; the healthy half is docs/rules/G3_RULE.md, DRAFT, whose section 2 leaves the 20 PD persons of the G3
lockbox to the pathological half). A dev estimate, not a gate read: no PD-half rule exists, so the pass and kill values
printed beside the lines are those of design section 9, the same for PD persons and controls, and every verdict is a
reading aid. No lockbox row or record is read.

Persons (pd_persons, in this script):
- pool: the non-lockbox weargait_pd persons of group PD with an IMU record in the canonical index (the pool of T19's
  proposed fold, reports/stage2_sources/summary.md, whose SHA-256 is compared and reported);
- scored: the persons of the pool whose SelfPace and HurriedPace trials have both an IMU record and a CSV trial in
  reports/weargait_targets/trials.csv (the same tasks on both sides); the others are listed as not scored with the
  reason (MAT-only: no CSV of theirs is on disk, so no walkway target; the release lists their task files,
  reports/weargait_targets/summary.md);
- a requested lockbox person (the 20 PD persons of the lockbox, or a lockbox control) or an id outside weargait_pd
  (CODY above all) is refused (canonical.Refused) before any file opens; a requested non-lockbox control is refused
  (ValueError) after the index is read (controls are the healthy half).
Controls: the 14 dev controls of mdfm.data.sources.folds fit the IC offsets (G3_RULE.md section 5: "fitted on the
dev controls only") and are scored beside the PD rows as the reference of the same run; the 54 training-fold
controls give the population baseline (rule section 7).

Rows (sets of trials, mdfm.gates.g3.evaluate with sets): pd_all (every scored PD person, every trial) and pd_t19dev
(the persons of T19's proposed PD dev fold, reports/stage2_sources/folds_proposed.csv, 16 persons), each also on the
trials of valid input records (pd_all_valid, pd_t19dev_valid: mdfm.gates.g3.input_validity, the check that
G3_RULE.md section 2 makes the primary row of the healthy half). Stage 2 will train on the other PD persons (the
proposed train fold, 64 persons), so pd_all is only a pre-stage-2 out-of-sample estimate: on a checkpoint that trained
on PD persons of the pool, those persons leave every PD row (not scored, reason NOT_SCORED_TRAINED, persons.csv
column trained_on) and pd_all keeps the persons it did not train on; pd_t19dev stays out-of-sample after stage 2 if the
owner adopts T19's fold, since a checkpoint trained on a person of that fold is refused (Inference below).
Control rows: dev14 and dev_valid (the 12 on dev, rule section 2).

Predictors (mdfm.gates.g3): the walkway (a harness check, zero on the walkway lines), the population baseline of the
54 training-fold controls (population, the trivial baseline of rule section 7, out-of-sample for every PD person),
the population baseline of T19's proposed PD train fold persons with targets (population_pd, out-of-sample on
pd_t19dev only, in-sample for the train-fold persons of pd_all; reported), the shank-gyroscope detector (the trivial
baseline and margin reference of the IC line) and the model. Lines: stride_mae_cm, speed_mae_m_s, stride_mape_pct,
ic_median_ms_walkway (the walkway contact decides the IC line, OD12) and ic_median_ms_insole (the second reference,
reported, decides nothing: rule section 6; its verdict cell reads 'reported (does not decide)'), with person
bootstrap intervals and matched shares. The GateResult rows come from g3.gate_results per row (rule_sha256 empty: no
PD-half rule). The matched-share floor of rule section 6 (0.9) is applied by g3.gate_results since task G3H
(2026-10-08): a stride, speed or MAPE row inside its pass line with the model's matched share under it reads
undetermined (extra floor_ok); the shares are reported beside every line.

F1 gain diagnostic (model runs only): the stride and speed gains that scripts/g3_gain_check.py fitted on the 54
training-fold controls for this checkpoint (reports/bench/g3_gain_check/results.json, the end whose checkpoint
SHA-256 is this one) divide the model's matched strides and pass speeds; per row the raw and recalibrated stride and
speed MAE and their paired difference. Labelled GAIN_LABEL: never a G3 verdict, and a head fitted on controls is not
carried to PD persons without its own fit and rule (docs/notes/F2_SCALE_DIAGNOSIS_2026-10-05.md section 6 item 5).
An end without a fitted gain (the phase A end, a new run) reads 'not computed'.

Inference: that of scripts/eval_g3_dev.py (mdfm.eval.infer.load_checkpoint and G3Predictor: 13 IMUs, magnetometer
masked by g3.model_input, 10 s windows every 5 s, 20 refinement steps of 256 points, shape pooled per trial), one
trial at a time. A dev control or a person of T19's proposed PD dev fold (all 16, scored or not) among the
checkpoint's training persons refuses the run (canonical.Refused, before the targets load); a scored PD person
among them (a stage-2 checkpoint trained on T19's train fold) leaves every PD row, so the model never reads a person
it trained on. --no-model runs the baselines only, on the CPU, and writes the same files without the model.

Writes <out> (default reports/gates/g3/dev_pd_<checkpoint name>, suffix _model for the raw weights): g3_pd_dev.json
and g3_pd_dev.md, per_person.csv (row, predictor, line, person_id, value), persons.csv, validity.csv, offsets.csv,
trials.csv (model runs: the predictor log) and _provenance.json (MoveAtlas HEAD read from its .git files, no git
command; script_sha256, the SHA-256 of this script as it ran).

The T2 producer of the F2 lever-4 rule (reports/bench/f2_lever4/rule_amendment_1.txt, A2; model runs only): also
events.csv, the rows of scripts/eval_g3_diag.py events_frame (imported, not rewritten) under the row name 'base', on
the model's matched tables of mdfm.gates.g3.evaluate over the trials of the scored PD persons, with the IC offsets
fitted on the 14 dev controls (strides, passes and walkway ICs matched by g3.evaluate alone; no control row), and
_provenance_events.json beside it (script, script_sha256, run_at, events_sha256, diag_sha256 (scripts/eval_g3_diag.py
as imported, the events layout), the SHA-256 of every code file, the checkpoint and the inference settings), the file
that the rule's amendment 2 (reports/bench/f2_lever4/rule_amendment_2.txt, B1 and B2) reads as the provenance of the
events. Both texts are built and checked (em-dash, MoveAtlas path) before the first file opens; events.csv is written
first and _provenance_events.json last, so that the provenance marks a complete pair. --events-only reruns a
checkpoint into the folder of its first run and writes these two files only: it stops before loading anything when
either exists, refuses a folder whose _provenance.json names another checkpoint or that holds no g3_pd_dev.json, opens
each new file in exclusive-create mode (an existing file is never replaced, also when it appears during the run) and
leaves every other file of the folder as it is; the folder's model lines and offsets are set beside this run's in
_provenance_events.json (folder_check, reported: GPU inference is not bit-reproducible). Amendment 2 B4 freezes this
script and scripts/eval_g3_diag.py (tests/test_f2_lever4.py asserts their SHA-256) until the f2l4 end's
_provenance_events.json exists.

Run (mdfm environment; GPU jobs only through scripts/queue_job.py, --print-jobs prints their commands):
  <conda>/envs/mdfm/python.exe scripts/eval_g3_pd_dev.py --checkpoint <run>/ckpt_B_<step>.pt \
      --batch-windows 32 --memory-gb 12 --out reports/gates/g3/dev_pd_<name>
  CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/eval_g3_pd_dev.py --no-model \
      --out reports/gates/g3/dev_pd_baselines
  <conda>/envs/mdfm/python.exe scripts/eval_g3_pd_dev.py --checkpoint <run>/ckpt_B_<step>.pt \
      --batch-windows 32 --memory-gb 12 --out reports/gates/g3/dev_pd_<name> --events-only
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

if __name__ == "__main__" and "--no-model" in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the baselines run never sees the GPU

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

TASK = "T27"
STEM = "g3_pd_dev"
WHAT = "G3 PD-half dev estimate on non-lockbox weargait_pd PD persons, not a gate read"
WHAT_BASE = "G3 PD-half dev baselines on non-lockbox weargait_pd PD persons (no model), not a gate read"
GROUP_PD = "PD"
FOLDS_CSV = os.path.join(TREE, "reports", "stage2_sources", "folds_proposed.csv")
STAGE2_SUMMARY = os.path.join(TREE, "reports", "stage2_sources", "summary.json")
FOLD_KEY = "weargait_pd:PD"
GAIN_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
GAIN_LABEL = ("F1 training-fold gain applied as a diagnostic row: the gains fitted on the 54 training-fold controls "
              "by scripts/g3_gain_check.py divide the model's strides and pass speeds; never a G3 verdict, and a head "
              "fitted on controls is not carried to PD persons without its own fit and rule")
GAIN_PREDICTOR = "model_recalibrated_f1_gain_diagnostic"   # per_person.csv rows of the gain diagnostic
NOT_SCORED_MAT = ("MAT-only: no CSV trial of SelfPace or HurriedPace in reports/weargait_targets/trials.csv (the "
                  "release lists the task files of such persons, absent on disk: reports/weargait_targets/summary.md)")
NOT_SCORED_TRAINED = "a training person of this checkpoint (in-sample): left out of every PD row"
PD_ALL, PD_T19 = "pd_all", "pd_t19dev"
PD_ALL_VALID, PD_T19_VALID = "pd_all_valid", "pd_t19dev_valid"
PD_ROWS = (PD_ALL, PD_T19, PD_ALL_VALID, PD_T19_VALID)
CTRL_ALL, CTRL_VALID = "dev14", "dev_valid"
CTRL_ROWS = (CTRL_ALL, CTRL_VALID)
PREDICTORS = ("walkway", "population", "population_pd", "shank_gyro", "model")
LINE_ORDER = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "ic_median_ms_insole")
GAIN_LINES = (("stride_mae_cm", "strides", "length_m", "pred_m", 100.0, "stride"),
              ("speed_mae_m_s", "speeds", "speed_m_s", "pred_m_s", 1.0, "speed"))
CONSISTENCY_TOL = 1e-9
SCRIPT_REL = "scripts/eval_g3_pd_dev.py"
DIAG_REL = "scripts/eval_g3_diag.py"            # the events layout (events_frame), pinned by amendment 2 B2
EVENTS_CSV = "events.csv"                       # T2 input of the F2 lever-4 rule (amendment 1 A2)
EVENTS_PROV = "_provenance_events.json"
EVENTS_ROW = "base"                             # the row name of A2 (eval_g3_diag's base row: the model as it runs)
EVENTS_SOURCE = "reports/bench/f2_lever4/rule_amendment_1.txt, A2 (the producer of T2)"
FOLDER_CHECK_LINES = ("stride_mae_cm", "speed_mae_m_s", "ic_median_ms_walkway")
STATEMENT_STAGE2 = ("Stage 2 will train on the other PD persons (T19's proposed PD train fold, 64 persons), so the "
                    "pd_all row is only a pre-stage-2 out-of-sample estimate: on a checkpoint that trained on PD "
                    "persons, those persons leave every PD row (persons.csv, columns trained_on and "
                    "not_scored_reason) and pd_all keeps the persons the checkpoint did not train on. The pd_t19dev "
                    "row (T19's proposed PD dev fold) stays out-of-sample after stage 2 if the owner adopts that "
                    "fold: a checkpoint trained on a person of that fold, or on a dev control, is refused.")


@dataclass(frozen=True)
class End:
    """A checkpoint of the T27 jobs: key, tree-relative checkpoint path, the batch_windows of its G3 diag run (the
    post_diag template's value for a run without one yet), the queue job name and priority, and whether this task
    queues it (False: the run does not exist yet, the command is only prepared)."""
    key: str
    checkpoint: str
    batch_windows: int
    job: str
    priority: int
    queued: bool


ENDS = (End("v1_A", "cache/runs/phaseA_root_full/ckpt_A_010681.pt", 32, "t27_g3_pd_v1_A", 30, True),
        End("v1_B", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", 32, "t27_g3_pd_v1_B", 31, True),
        End("s2_B", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", 16, "t27_g3_pd_s2_B", 32, True),
        End("f2l2_B", "cache/runs/pilot_f2l2/ckpt_B_032044.pt", 32, "t27_g3_pd_f2l2_B", 33, False))
JOB_CLS = 3
JOB_MEMORY_GB = "12"
JOB_EST_HOURS = 0.3
PYTHON = "<conda>/envs/mdfm/python.exe"


# ---------------------------------------------------------------- helpers
def rel(path):
    try:
        r = os.path.relpath(os.path.abspath(path), TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def checkpoint_name(path):
    """'<run folder>_<file stem>' (mdfm.eval.infer.checkpoint_name, without importing torch)."""
    p = os.path.abspath(str(path).replace("/", os.sep))
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"


def out_of(end):
    return f"reports/gates/g3/dev_pd_{checkpoint_name(end.checkpoint)}"


def job_argv(end):
    """The scripts/queue_job.py command (argv after the python) that queues the job of an end."""
    out = out_of(end)
    return ["scripts/queue_job.py", "--name", end.job, "--cls", str(JOB_CLS), "--priority", str(end.priority),
            "--est-hours", str(JOB_EST_HOURS), "--group", TASK, "--requires", end.checkpoint,
            "--outputs", f"{out}/{STEM}.json",
            "--note", f"T27 (reports/weekend_2026-10-01/tasks/T27.md): G3 PD-half dev estimate of "
                      f"{checkpoint_name(end.checkpoint)} on the non-lockbox weargait_pd PD persons",
            "--", "scripts/eval_g3_pd_dev.py", "--checkpoint", end.checkpoint, "--batch-windows",
            str(end.batch_windows), "--memory-gb", JOB_MEMORY_GB, "--out", out]


def _sibling(name):
    """The module of scripts/<name>.py of the tree (loaded once, under a private name)."""
    key = f"_t27_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def EG():
    """scripts/eval_g3_dev.py (formatting helpers and the insole verdict text)."""
    return _sibling("eval_g3_dev")


def DG():
    """scripts/eval_g3_diag.py (events_frame, the events layout that amendment 1 A2 names)."""
    return _sibling("eval_g3_diag")


# ---------------------------------------------------------------- the PD events (T2 producer, amendment 1 A2)
def pd_events(model_tables, targets, scored):
    """The events.csv frame of A2: eval_g3_diag.events_frame(EVENTS_ROW, model_tables, targets), where model_tables
    are the model's matched tables of g3.evaluate over the scored PD persons' trials. Every person of the frame is
    guarded (a lockbox or non-weargait id is refused, canonical.Refused) and must be a scored PD person (SystemExit
    otherwise: a control or another person would be a harness defect)."""
    ev = DG().events_frame(EVENTS_ROW, model_tables, targets)
    who = {str(p) for p in ev["person_id"]}
    guard(who)
    extra = sorted(who - {str(p) for p in scored})
    if extra:
        raise SystemExit(f"{len(extra)} persons in the PD events outside the scored PD persons "
                         f"({', '.join(extra[:3])})")
    return ev


def write_new(path, text):
    """Write UTF-8 text with LF line ends in exclusive-create mode (FileExistsError when the file exists), after the
    em-dash and MoveAtlas checks of mdfm.gates.common.write_text."""
    from mdfm import paths
    from mdfm.gates import common
    common.assert_no_em_dash(text, rel(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(paths.assert_not_moveatlas(path), "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def events_preflight(out):
    """--events-only, before anything loads: SystemExit when events.csv or _provenance_events.json already exists in
    out, or when out holds no g3_pd_dev.json and _provenance.json of a first run. Returns that _provenance.json."""
    for name in (EVENTS_CSV, EVENTS_PROV):
        if os.path.exists(os.path.join(out, name)):
            raise SystemExit(f"--events-only: {rel(os.path.join(out, name))} exists; an existing file is never "
                             "replaced")
    for name in (f"{STEM}.json", "_provenance.json"):
        if not os.path.exists(os.path.join(out, name)):
            raise SystemExit(f"--events-only adds the events to the folder of a first run of the same checkpoint; "
                             f"{rel(os.path.join(out, name))} is absent")
    with open(os.path.join(out, "_provenance.json"), encoding="utf-8") as f:
        return json.load(f)


def folder_check(out, res):
    """--events-only: the model lines of the PD rows (FOLDER_CHECK_LINES) and the model's IC offsets of the folder's
    g3_pd_dev.json beside this run's (res, the cleaned JSON of this run): reported, never asserted."""
    with open(os.path.join(out, f"{STEM}.json"), encoding="utf-8") as f:
        doc = json.load(f)
    lines = {}
    for row in PD_ROWS:
        for line in FOLDER_CHECK_LINES:
            a = (((doc.get("lines") or {}).get(row) or {}).get("model") or {}).get(line, {}).get("point")
            b = (((res.get("lines") or {}).get(row) or {}).get("model") or {}).get(line, {}).get("point")
            lines[f"{row}|{line}"] = {"folder": a, "this_run": b,
                                      "difference": None if a is None or b is None else b - a}
    offs = {}
    for key in ("model|walkway", "model|insole"):
        a = ((doc.get("offsets_ms") or {}).get(key) or {}).get("offset_ms")
        b = ((res.get("offsets_ms") or {}).get(key) or {}).get("offset_ms")
        offs[key] = {"folder": a, "this_run": b, "difference": None if a is None or b is None else b - a}
    inf_a, inf_b = doc.get("inference") or {}, res.get("inference") or {}
    keys = ("batch_windows", "refine_steps", "points_per_stream", "seed", "shape_pooling", "weights")
    return {"folder_json": rel(os.path.join(out, f"{STEM}.json")), "lines": lines, "offsets_ms": offs,
            "inference_differences": {k: [inf_a.get(k), inf_b.get(k)] for k in keys if inf_a.get(k) != inf_b.get(k)},
            "note": "reported, never asserted: GPU inference is not bit-reproducible"}


# ---------------------------------------------------------------- persons
@dataclass
class PDPersons:
    """pool: non-lockbox PD persons with an IMU record (any task); scored: those whose tasks (SelfPace, HurriedPace)
    have an IMU record and a CSV trial alike; not_scored {person: reason}; imu_tasks and csv_tasks {person: sorted
    tasks among the scored tasks}."""
    pool: list
    scored: list
    not_scored: dict
    imu_tasks: dict = field(default_factory=dict)
    csv_tasks: dict = field(default_factory=dict)


def guard(persons):
    """The set of requested ids, refused (canonical.Refused) before any target file or the index opens when one is
    outside weargait_pd (CODY named) or a weargait_pd lockbox person (PD or control); only the lockbox list
    (reports/lockbox/lockbox_persons.csv, checked against its hash by the g3 reader) is read."""
    from mdfm.data import canonical
    from mdfm.data.canonical import Refused
    from mdfm.gates import g3
    ps = {str(p) for p in persons}
    other = sorted(p for p in ps if not p.startswith(g3.PREFIX))
    if other:
        cody = [p for p in other if canonical.is_cody(p) or p.split(":", 1)[0] in canonical.CODY_NAMESPACES]
        raise Refused(f"the G3 PD half reads weargait_pd persons only; {len(other)} other ids"
                      + (f", {len(cody)} of them CODY (evaluation-only, never a G3 input)" if cody else ""))
    locked = ps & set(g3.lockbox_persons())
    if locked:
        raise Refused(f"G3 PD half: {len(locked)} lockbox persons requested; a dev estimate never reads the lockbox")
    return ps


def pd_persons(persons=None, index=None, trials_csv=None, tasks=None):
    """PDPersons of the module docstring. persons (optional): restrict to these ids, guarded first (guard: a
    lockbox or non-weargait id is refused before the index or a target file opens), then each must be a PD person
    of the pool (a control or an id outside the pool raises ValueError). index: the canonical index (default
    canonical.index()); trials_csv: default reports/weargait_targets/trials.csv, streamed by g3.read_rows (rows of
    the pool only, lockbox column asserted)."""
    from mdfm.data import canonical
    from mdfm.gates import g3
    want = guard(persons) if persons is not None else None
    tasks = tuple(g3.TASKS if tasks is None else tasks)
    df = canonical.index() if index is None else index
    sel = df[(df["source"] == g3.SOURCE) & (df["modality"] == "imu")]
    pd_rows = sel[(sel["group"] == GROUP_PD) & ~sel["lockbox"].astype(bool)]
    keys = {}
    for p, k in zip(pd_rows["person_id"], pd_rows["key"]):
        keys.setdefault(str(p), set()).add(str(k))
    pool = sorted(keys)
    locked = set(g3.lockbox_persons())
    if set(pool) & locked:
        raise ValueError(f"{len(set(pool) & locked)} lockbox persons among the non-lockbox PD rows of the index")
    if want is not None:
        ctrl = sorted(want & set(sel.loc[sel["group"] == "control", "person_id"].astype(str)))
        if ctrl:
            raise ValueError(f"{len(ctrl)} controls requested: controls are the healthy half (scripts/eval_g3_dev.py)")
        outside = sorted(want - set(pool))
        if outside:
            raise ValueError(f"{len(outside)} requested persons are not non-lockbox PD persons with an IMU record")
        pool = sorted(want)
    imu_tasks = {p: keys[p] & set(tasks) for p in pool}
    rows = g3.read_rows(trials_csv or g3.TRIALS_CSV, pool)
    csv_tasks = {}
    for r in rows:
        if r["task"] in tasks:
            csv_tasks.setdefault(r["person_id"], set()).add(r["task"])
    scored, not_scored = [], {}
    for p in pool:
        have, imu = csv_tasks.get(p, set()), imu_tasks[p]
        if not have and not imu:
            not_scored[p] = (f"{NOT_SCORED_MAT}; no IMU record of these tasks either (index IMU keys: "
                             f"{', '.join(sorted(keys[p]))})")
        elif not have:
            not_scored[p] = NOT_SCORED_MAT
        elif imu - have:
            not_scored[p] = f"no CSV trial for {', '.join(sorted(imu - have))} (IMU record present)"
        elif have - imu:
            not_scored[p] = f"no IMU record of {', '.join(sorted(have - imu))} (CSV trial present)"
        else:
            scored.append(p)
    return PDPersons(pool, scored, not_scored, {p: sorted(v) for p, v in imu_tasks.items()},
                     {p: sorted(v) for p, v in csv_tasks.items()})


def t19_folds(path=None, summary=None):
    """(dev, train, info) of T19's proposed weargait_pd PD fold: the person ids of folds_proposed.csv (group PD) by
    fold, their SHA-256 (sources.person_hash) asserted equal to those recorded in summary.json, and none of them a
    lockbox person (Refused otherwise)."""
    from mdfm.data import sources
    from mdfm.data.canonical import Refused
    from mdfm.gates import g3
    path = path or FOLDS_CSV
    summary = summary or STAGE2_SUMMARY
    folds = {"dev": [], "train": []}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == g3.SOURCE and r["group"] == GROUP_PD:
                if r["fold"] not in folds:
                    raise ValueError(f"fold {r['fold']!r} in {rel(path)}")
                folds[r["fold"]].append(r["person_id"])
    with open(summary, encoding="utf-8") as f:
        rec = json.load(f)["folds"][FOLD_KEY]
    dev, train = sorted(folds["dev"]), sorted(folds["train"])
    for name, ids in (("dev", dev), ("train", train)):
        if sources.person_hash(ids) != rec[f"sha256_{name}"]:
            raise ValueError(f"T19 {name} fold of {rel(path)} does not match its SHA-256 in {rel(summary)}")
    if set(dev) & set(train):
        raise ValueError("T19 dev and train folds overlap")
    locked = (set(dev) | set(train)) & set(g3.lockbox_persons())
    if locked:
        raise Refused(f"{len(locked)} lockbox persons in T19's proposed PD fold")
    info = {"path": rel(path), "summary": rel(summary), "pool": rec["pool"], "persons": rec["persons"],
            "dev": len(dev), "train": len(train), "sha256_pool": rec["sha256_pool"], "sha256_dev": rec["sha256_dev"],
            "sha256_train": rec["sha256_train"], "seed": rec.get("seed")}
    return dev, train, info


# ---------------------------------------------------------------- sets of trials
def pd_sets(trials, t19dev, valid=None):
    """{row: frozenset of trial ids} of the PD rows over a trials frame (trial_id, person_id): pd_all, pd_t19dev and,
    when valid (trial ids of valid records) is given, pd_all_valid and pd_t19dev_valid."""
    t19 = {str(p) for p in t19dev}
    ids = frozenset(trials["trial_id"])
    dev = frozenset(trials.loc[trials["person_id"].isin(t19), "trial_id"])
    out = {PD_ALL: ids, PD_T19: dev}
    if valid is not None:
        v = frozenset(valid)
        out[PD_ALL_VALID] = ids & v
        out[PD_T19_VALID] = dev & v
    return out


def control_sets(trials, valid=None):
    out = {CTRL_ALL: frozenset(trials["trial_id"])}
    if valid is not None:
        out[CTRL_VALID] = frozenset(trials["trial_id"]) & frozenset(valid)
    return out


def persons_of(trials, ids):
    return sorted(set(trials.loc[trials["trial_id"].isin(set(ids)), "person_id"]))


# ---------------------------------------------------------------- F1 gain diagnostic
def load_gain(checkpoint_sha256, path=None):
    """(gain dict or None, info): the gains of scripts/g3_gain_check.py for the end whose checkpoint SHA-256 is
    checkpoint_sha256 ({stride, speed, end, fitted_on}); None with the reason when that file, end or gain is
    missing."""
    path = path or GAIN_RESULTS
    info = {"path": rel(path), "label": GAIN_LABEL}
    if not os.path.exists(path):
        return None, {**info, "reason": "results.json absent"}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    info.update({"rule_sha256": doc.get("rule_sha256"), "what": doc.get("what")})
    for key, e in (doc.get("ends") or {}).items():
        if (e.get("checkpoint") or {}).get("sha256") != checkpoint_sha256:
            continue
        g = e.get("gain")
        if not g or not all(_usable((g.get(k) or {}).get("gain")) for k in ("stride", "speed")):
            return None, {**info, "end": key, "reason": f"end {key}: no gain fitted (no training-fold decode)"}
        return ({"stride": float(g["stride"]["gain"]), "speed": float(g["speed"]["gain"]), "end": key,
                 "fitted_on": g.get("fitted_on"), "n_persons": g["stride"].get("n_persons")},
                {**info, "end": key})
    return None, {**info, "reason": "no end of results.json holds this checkpoint"}


def _usable(g):
    try:
        v = float(g)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def person_line(per_person):
    """{point (mean over persons), interval (common.cluster_bootstrap, g3.BOOT_N resamples, g3.SEED; None under 2
    persons), n_persons, per_person} of {person: value}."""
    import numpy as np

    from mdfm.gates import common, g3
    ids = sorted(per_person)
    if not ids:
        return {"point": float("nan"), "interval": None, "n_persons": 0, "per_person": {}}
    v = np.array([per_person[i] for i in ids], np.float64)
    iv = list(common.cluster_bootstrap(v, np.array(ids), n=g3.BOOT_N, seed=g3.SEED)) if len(ids) >= 2 else None
    return {"point": float(v.mean()), "interval": iv, "n_persons": len(ids),
            "per_person": {i: float(x) for i, x in zip(ids, v)}}


def _person_abs(df, ref, pred, scale, gain):
    import numpy as np
    if not len(df):
        return {}
    e = scale * np.abs(df[pred].to_numpy(np.float64) / float(gain) - df[ref].to_numpy(np.float64))
    s = df.assign(_e=e).groupby(df["person_id"].astype(str))["_e"].mean()
    return {str(k): float(x) for k, x in s.sort_index().items()}


def gain_diagnostic(tables, sets, gain):
    """Per row of sets {row: trial ids}: for the stride and speed lines, the raw MAE of the model's matched events
    (equal to g3's line), the MAE after dividing the prediction by the gain, and their paired difference (raw minus
    recalibrated, with lower_above_0). tables: the model's matched tables of g3.evaluate (strides, speeds)."""
    out = {}
    for row, trials in sets.items():
        r = {}
        for line, table, ref, pred, scale, kind in GAIN_LINES:
            df = tables[table]
            df = df[df["matched"].astype(bool) & df["trial_id"].isin(set(trials))]
            raw = _person_abs(df, ref, pred, scale, 1.0)
            rec = _person_abs(df, ref, pred, scale, gain[kind])
            d = person_line({p: raw[p] - rec[p] for p in raw})
            d["lower_above_0"] = bool(d["interval"] is not None and d["interval"][0] > 0)
            r[line] = {"gain": gain[kind], "raw": person_line(raw), "recalibrated": person_line(rec), "paired": d}
        out[row] = r
    return out


def gain_consistency(diag, summ_by_row):
    """{row|line: {raw, g3, difference, ok}}: the raw line of gain_diagnostic against g3's model line of the same row
    (equal within CONSISTENCY_TOL)."""
    out = {}
    for row, lines in diag.items():
        for line, d in lines.items():
            a = d["raw"]["point"]
            b = summ_by_row.get(row, {}).get("model", {}).get(line, {}).get("point", float("nan"))
            both_nan = not math.isfinite(a) and not math.isfinite(b)
            diff = 0.0 if both_nan else a - b
            out[f"{row}|{line}"] = {"raw": a, "g3": b, "difference": diff,
                                    "ok": bool(both_nan or abs(diff) <= CONSISTENCY_TOL)}
    return out


# ---------------------------------------------------------------- report helpers
def strip_stats(st):
    return {k: v for k, v in st.items() if k not in ("per_person", "trials", "sets")}


def view_lines(view):
    return {name: {line: strip_stats(st) for line, st in s.items()} for name, s in view.items()}


def per_person_rows(row, view):
    out = []
    for name, lines in view.items():
        for line, st in lines.items():
            for pid, v in st.get("per_person", {}).items():
                out.append((row, name, line, pid, v))
    return out


def prior_dev_report(name, tree=None):
    """The G3 dev json of the same checkpoint (scripts/eval_g3_dev.py), the first that exists, or None."""
    for folder, stem in (("g3", "g3_dev"), ("g3_sim", "g3_sim")):
        p = os.path.join(tree or TREE, "reports", "gates", folder, f"dev_{name}", f"{stem}.json")
        if os.path.exists(p):
            return p
    return None


def rerun_check(prior_path, offsets, dev_lines, checkpoint_sha256):
    """The IC offsets and the dev14 model lines of this run against those of the healthy-half dev report of the same
    checkpoint (reported, never asserted: the inference settings may differ)."""
    if prior_path is None:
        return {"evaluated": False, "reason": "no healthy-half dev report of this checkpoint"}
    with open(prior_path, encoding="utf-8") as f:
        doc = json.load(f)
    if (doc.get("checkpoint") or {}).get("sha256") != checkpoint_sha256:
        return {"evaluated": False, "reason": f"{rel(prior_path)} holds another checkpoint"}
    out = {"evaluated": True, "reference": rel(prior_path),
           "reference_inference": {k: (doc.get("inference") or {}).get(k) for k in ("batch_windows", "refine_steps",
                                                                                   "points_per_stream", "seed")},
           "offsets_ms": {}, "lines": {}}
    for key, o in (doc.get("offsets_ms") or {}).items():
        n, r = key.split("|")
        mine = offsets.get((n, r))
        out["offsets_ms"][key] = {"reference": o.get("offset_ms"),
                                  "this_run": 1000.0 * mine[0] if mine is not None else None}
    ref_lines = ((doc.get("lines") or {}).get("dev") or {}).get("model") or {}
    for line in LINE_ORDER:
        a = (ref_lines.get(line) or {}).get("point")
        b = (dev_lines.get("model") or {}).get(line, {}).get("point")
        out["lines"][line] = {"reference": a, "this_run": b}
    return out


def checkpoint_kind(loaded):
    """The kind of a loaded checkpoint, as scripts/eval_g3_dev.py words it (None without a checkpoint)."""
    if loaded is None:
        return None
    if loaded.phase == "A" and loaded.phase_a_complete:
        return "closes phase A: the simulator-only model"
    if loaded.phase == "B":
        return "phase B checkpoint"
    if loaded.phase == "C":
        return "phase C checkpoint (stage 2)"
    return f"partial phase A (step {loaded.step}): not the simulator-only model"


def side_summary(table):
    """{trial_feet, flagged, flagged_list} of a g3.side_check table."""
    flagged = table[table["flagged"].astype(bool)] if len(table) else table
    return {"trial_feet": int(len(table)), "flagged": int(len(flagged)),
            "flagged_list": [f"{r.trial_id} {r.foot}" for r in flagged.itertuples()]}


# ---------------------------------------------------------------- run
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=None, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--no-model", action="store_true", help="the baselines only, on the CPU (no checkpoint)")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256, help="kernel points per stream-window and refinement step")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a trial")
    p.add_argument("--gain-results", default=None, help="results.json of scripts/g3_gain_check.py")
    p.add_argument("--out", default=None)
    p.add_argument("--print-jobs", action="store_true", help="print the scripts/queue_job.py commands and exit")
    p.add_argument("--events-only", action="store_true",
                   help="write only events.csv and _provenance_events.json into --out, the folder of a first run of "
                        "the same checkpoint (amendment 1 A2 of the F2 lever-4 rule); never replaces a file")
    a = p.parse_args(argv)
    if not a.print_jobs and (a.checkpoint is None) == (not a.no_model):
        p.error("give --checkpoint or --no-model (exactly one)")
    if a.no_model and a.out is None:
        p.error("--no-model needs --out")
    if a.events_only and (a.no_model or a.out is None):
        p.error("--events-only needs --checkpoint and --out (the events are the model's)")
    return a


def print_jobs():
    for e in ENDS:
        print(f"# {e.key} ({'queued by T27' if e.queued else 'prepared, not queued: the run does not exist yet'})")
        print(" ".join([PYTHON] + [f'"{x}"' if " " in x else x for x in job_argv(e)]))


def main(argv=None):
    a = parse(argv)
    if a.print_jobs:
        print_jobs()
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import pandas as pd

    from mdfm import paths
    from mdfm.data import sources
    from mdfm.data.canonical import Refused
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    eg = EG()

    t_start = time.time()
    script_sha = common.sha256_file(os.path.abspath(__file__))     # this script as it runs (amendment 1 A2)
    prior_prov = events_preflight(a.out) if a.events_only else None
    # the events layout as imported, hashed at its import (amendment 2 B2); model runs only
    diag_sha = None if a.no_model else common.sha256_file(os.path.abspath(DG().__file__))
    code_files = [g3.__file__, g3.gait.__file__, os.path.abspath(__file__), os.path.join(TREE, "scripts",
                                                                                         "eval_g3_dev.py")]
    loaded = pred = None
    mem, cap, dev = {}, None, None
    if not a.no_model:
        import torch

        from mdfm.eval import infer as I
        from mdfm.train import loop as LP
        code_files.append(I.__file__)
        dev = torch.device(a.device)
        cap = LP.cap_memory(dev, a.memory_gb)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
        print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step "
              f"{loaded.step}", flush=True)
        if prior_prov is not None and prior_prov.get("checkpoint_sha256") != loaded.sha256:
            raise SystemExit(f"--events-only: {rel(os.path.join(a.out, '_provenance.json'))} names checkpoint SHA-256 "
                             f"{str(prior_prov.get('checkpoint_sha256'))[:12]}, not {loaded.sha256[:12]}")
    code_sha = common.code_sha256(code_files)
    out = a.out or os.path.join(TREE, "reports", "gates", "g3",
                                f"dev_pd_{loaded.name}{'_model' if loaded.weights == 'model' else ''}")
    paths.assert_not_moveatlas(out)
    t_load = time.time() - t_start

    # ---------------- persons
    dev_ids, train_ids = g3.controls("dev"), g3.controls("train")
    pdp = pd_persons()
    t19dev, t19train, t19info = t19_folds()
    if set(pdp.scored) & (set(dev_ids) | set(train_ids)):
        raise SystemExit("a control among the PD persons")
    if set(pdp.scored) & set(g3.lockbox_persons()):
        raise SystemExit("a lockbox person among the scored PD persons")
    if set(t19dev) - set(pdp.pool) or set(t19train) - set(pdp.pool):
        print("note: T19's fold holds persons outside this pool", flush=True)
    trained_pd, train_sha = [], ""
    if loaded is not None:
        trained = loaded.trained_persons()
        hit = sorted((set(dev_ids) | set(t19dev)) & trained)
        if hit:
            raise Refused(f"{len(hit)} dev controls or persons of T19's proposed PD dev fold among the checkpoint's "
                          "training persons: the reference rows or the pd_t19dev row would be in-sample")
        trained_pd = sorted(set(pdp.pool) & trained)
        train_sha = I.train_persons_sha256(loaded)
        pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                             seed=a.seed, pool=not a.no_pool)
    excluded = sorted(set(pdp.scored) & set(trained_pd))       # in-sample (stage 2): out of every PD row
    scored = [p for p in pdp.scored if p not in set(excluded)]
    not_scored = {**pdp.not_scored, **{p: NOT_SCORED_TRAINED for p in excluded}}
    if excluded:
        print(f"{len(excluded)} scorable PD persons are training persons of the checkpoint: left out of every PD row",
              flush=True)

    def model(tid, rec):
        tb = pred(tid, rec)
        r = pred.log[-1]
        print(f"  {tid}: {r['windows']} windows, {r['seconds']:.1f} s", flush=True)
        return tb

    def run_records(persons, targets):
        """({trial_id: shank gyro}, validity frame, model Predictions or None), one record at a time."""
        gyros, checks, parts = {}, [], []
        info = targets.trials.set_index("trial_id")
        for tid, rec in g3.imu_records(persons, targets.trials):
            gyros[tid] = g3.shank_gyro(rec)
            checks.append(g3.validity_row(tid, info.at[tid, "person_id"], info.at[tid, "task"], rec))
            if pred is not None:
                parts.append(g3.body_predictions([model(tid, g3.model_input(rec))], targets))
        cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
        mp = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds")) if parts else None
        return gyros, g3.validity_frame(checks), mp

    # ---------------- targets
    t0 = time.time()
    tg_dev, tg_train, tg_pd = g3.load_targets(dev_ids), g3.load_targets(train_ids), g3.load_targets(scored)
    t19train_targets = sorted(set(t19train) & set(pdp.scored))   # population_pd: targets only, no model output
    tg_pdtrain = g3.load_targets(t19train_targets)
    for name, tg in (("dev", tg_dev), ("train", tg_train), ("pd", tg_pd), ("pd_train", tg_pdtrain)):
        if tg.counts["lockbox_rows"]:
            raise SystemExit(f"a lockbox row was kept ({name})")
    base = g3.population_baseline(tg_train)
    base_pd = g3.population_baseline(tg_pdtrain)

    # ---------------- controls: the offsets (fitted on the 14 dev controls only) and the reference rows
    gyros_dev, val_dev, model_dev = run_records(dev_ids, tg_dev)
    preds_dev = [g3.walkway_predictions(tg_dev), g3.population_predictions(base, tg_dev),
                 g3.shank_predictions(gyros_dev)] + ([model_dev] if model_dev is not None else [])
    offsets = g3.fit_offsets(preds_dev, tg_dev)
    unfitted = [f"{n}|{r}" for (n, r), (o, _) in offsets.items() if not g3.offset_fitted(o)]
    off_s = {k: v[0] for k, v in offsets.items()}
    sets_dev = control_sets(tg_dev.trials, g3.valid_trials(val_dev))
    summ_dev, tables_dev = g3.evaluate(tg_dev, preds_dev, off_s, sets=sets_dev)
    t_dev = time.time() - t0

    # ---------------- PD persons
    t0 = time.time()
    gyros_pd, val_pd, model_pd = run_records(scored, tg_pd)
    shank_pd = g3.shank_predictions(gyros_pd)
    preds_pd = [g3.walkway_predictions(tg_pd), g3.population_predictions(base, tg_pd),
                g3.population_predictions(base_pd, tg_pd, name="population_pd"), shank_pd] \
        + ([model_pd] if model_pd is not None else [])
    sets = pd_sets(tg_pd.trials, t19dev, g3.valid_trials(val_pd))
    summ_pd, tables_pd = g3.evaluate(tg_pd, preds_pd, off_s, sets=sets)
    events = pd_events(tables_pd["model"], tg_pd, scored) if model_pd is not None else None
    t_pd = time.time() - t0

    # ---------------- views, GateResult rows (dev estimates), checks
    views = {r: g3.set_view(summ_pd, tables_pd, r) for r in sets}
    views.update({r: g3.set_view(summ_dev, tables_dev, r) for r in sets_dev})
    model_name = "model" if model_pd is not None else None
    results = {}
    if model_name:
        ck_sha = loaded.sha256
        for r, (sv, tv) in views.items():
            gr = g3.gate_results(sv, tv, model_name, "", ck_sha, train_sha)
            for x in gr:
                x.extra["set"] = r
                x.extra["role_of_lines"] = "design section 9 lines, no PD-half rule: a reading aid, not a verdict"
            results[r] = [x.to_dict() for x in gr]
    side = {"pd_all": side_summary(g3.side_check(shank_pd, g3.reference_ics(tg_pd, "walkway"),
                                                 off_s[("shank_gyro", "walkway")])),
            "dev14": side_summary(g3.side_check(preds_dev[2], g3.reference_ics(tg_dev, "walkway"),
                                                off_s[("shank_gyro", "walkway")]))}
    gain, gain_info, gain_rows, gain_check = None, None, None, None
    if model_name:
        gain, gain_info = load_gain(loaded.sha256, a.gain_results)
        if gain is not None:
            gain_rows = gain_diagnostic(tables_pd["model"], sets, gain)
            gain_rows.update(gain_diagnostic(tables_dev["model"], sets_dev, gain))
            gain_check = gain_consistency(gain_rows, {r: v[0] for r, v in views.items()})
            if not all(c["ok"] for c in gain_check.values()):
                raise SystemExit(f"gain diagnostic raw lines differ from g3's: {gain_check}")
    rerun = (rerun_check(prior_dev_report(loaded.name), offsets, view_lines(views[CTRL_ALL][0]), loaded.sha256)
             if loaded is not None else {"evaluated": False, "reason": "no model"})

    # ---------------- tables
    pp = []
    for r, (sv, _) in views.items():
        pp += per_person_rows(r, sv)
    for r, lines in (gain_rows or {}).items():          # the F1 gain diagnostic, labelled by its predictor name
        for line, d in lines.items():
            pp += [(r, GAIN_PREDICTOR, line, pid, x) for pid, x in d["recalibrated"]["per_person"].items()]
    per_person = pd.DataFrame(pp, columns=["row", "predictor", "line", "person_id", "value"])
    off = pd.DataFrame([(n, r, 1000.0 * o, k) for (n, r), (o, k) in sorted(offsets.items())],
                       columns=["predictor", "reference", "offset_ms", "n_matched"])
    vt = pd.concat([val_dev.assign(group="control"), val_pd.assign(group="PD")], ignore_index=True)
    trials_pd = tg_pd.trials
    prow = []
    for p in pdp.pool:
        t = trials_pd[trials_pd["person_id"] == p]
        v = val_pd[val_pd["person_id"] == p]
        prow.append((p, p in scored, not_scored.get(p, ""),
                     "dev" if p in t19dev else "train" if p in t19train else "", ";".join(pdp.imu_tasks.get(p, [])),
                     ";".join(pdp.csv_tasks.get(p, [])), int(len(t)), int(v["valid"].astype(bool).sum()),
                     int(len(v)), p in trained_pd))
    persons_df = pd.DataFrame(prow, columns=["person_id", "scored", "not_scored_reason", "t19_fold", "imu_tasks",
                                             "csv_tasks", "trials", "valid_records", "records", "trained_on"])
    if not a.events_only:               # --events-only leaves every file of the first run as it is
        common.write_csv(per_person, os.path.join(out, "per_person.csv"))
        common.write_csv(off, os.path.join(out, "offsets.csv"))
        common.write_csv(vt, os.path.join(out, "validity.csv"))
        common.write_csv(persons_df, os.path.join(out, "persons.csv"))
        if pred is not None:
            tl = pd.DataFrame(pred.log)
            ctrl_trials = set(tg_dev.trials["trial_id"])
            tl["group"] = ["control" if t in ctrl_trials else "PD" for t in tl["trial_id"]]
            common.write_csv(tl, os.path.join(out, "trials.csv"))
    if dev is not None and dev.type == "cuda":
        import torch
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}

    # ---------------- JSON
    head = PV.moveatlas_head()
    with open(g3.RULE_PATH, "rb") as f:
        g3_rule_sha = common.sha256_bytes(f.read())
    with open(g3.RULE_PATH, encoding="utf-8") as f:
        g3_rule_status = common.rule_status(f.read())[0]
    row_info = {}
    for r, ids in {**sets, **sets_dev}.items():
        tg = tg_pd if r in sets else tg_dev
        row_info[r] = {"trials": len(ids), "persons": persons_of(tg.trials, ids),
                       "n_persons": len(persons_of(tg.trials, ids))}
    invalid_pd = val_pd[~val_pd["valid"].astype(bool)]
    res = {
        "run": "g3_pd_dev", "task": TASK, "gate": g3.GATE, "what": WHAT if model_name else WHAT_BASE,
        "checkpoint": loaded.describe() if loaded is not None else None,
        "checkpoint_kind": checkpoint_kind(loaded),
        "inference": ({"refine_steps": a.refine_steps, "points_per_stream": a.points, "batch_windows": a.batch_windows,
                       "seed": a.seed, "shape_pooling": "per trial" if not a.no_pool else "none", "weights": a.weights,
                       "device": str(dev), "amp_bf16": dev.type == "cuda", "memory": mem}
                      if loaded is not None else None),
        "rule": {"pd_half_rule": None,
                 "note": "no PD-half rule exists; the pass and kill values are those of design section 9 (the same "
                         "for PD persons and controls) and every verdict is a reading aid",
                 "g3_healthy_half_rule": {"path": common.rel(g3.RULE_PATH), "sha256": g3_rule_sha,
                                          "status": g3_rule_status}},
        "statement_stage2": STATEMENT_STAGE2,
        "persons": {"pool": len(pdp.pool), "pool_sha256": sources.person_hash(pdp.pool),
                    "pool_equals_t19_pool": sources.person_hash(pdp.pool) == t19info["sha256_pool"],
                    "scorable": len(pdp.scored), "scored": len(scored), "scored_sha256": sources.person_hash(scored),
                    "not_scored": not_scored, "trained_on_among_pool": trained_pd, "excluded_trained": excluded,
                    "t19": t19info, "t19_dev_scored": sorted(set(t19dev) & set(scored)),
                    "t19_dev_not_scored": sorted(set(t19dev) - set(scored)),
                    "t19_train_scored": len(set(t19train) & set(scored)),
                    "t19_train_with_targets": len(t19train_targets),
                    "controls_dev_sha256": sources.person_hash(dev_ids),
                    "controls_train_sha256": sources.person_hash(train_ids)},
        "rows": row_info,
        "counts": {"pd": tg_pd.counts, "pd_t19_train": tg_pdtrain.counts, "controls_dev": tg_dev.counts},
        "imu_records": {"pd": int(len(val_pd)), "controls_dev": int(len(val_dev))},
        "validity": {"pd_records": int(len(val_pd)), "pd_valid": int(val_pd["valid"].astype(bool).sum()),
                     "pd_invalid": [f"{r.trial_id}: {r.reasons}" for r in invalid_pd.itertuples()],
                     "dev_records": int(len(val_dev)), "dev_valid": int(val_dev["valid"].astype(bool).sum()),
                     "dev_valid_persons": persons_of(tg_dev.trials, sets_dev[CTRL_VALID])},
        "constants": {"stride_match_s": g3.STRIDE_MATCH_S, "ic_match_s": g3.IC_MATCH_S, "ic_margin": g3.IC_MARGIN,
                      "lines": g3.LINES, "primary_ic_reference": g3.PRIMARY_REFERENCE, "boot_n": g3.BOOT_N,
                      "seed": g3.SEED},
        "offsets_ms": {f"{n}|{r}": {"offset_ms": 1000.0 * o, "n_matched": k} for (n, r), (o, k) in offsets.items()},
        "offsets_fitted_on": "the 14 dev controls (G3_RULE.md section 5)",
        "offsets_unfitted": unfitted,
        "population_baseline": {t: {"stride_m": s, "speed_m_s": v} for t, (s, v) in base.items()},
        "population_pd_baseline": {t: {"stride_m": s, "speed_m_s": v} for t, (s, v) in base_pd.items()},
        "model_events": ({"pd": {"ics": int(len(model_pd.ics)), "strides": int(len(model_pd.strides)),
                                 "pass_speeds": int(len(model_pd.speeds))},
                          "controls_dev": {"ics": int(len(model_dev.ics)), "strides": int(len(model_dev.strides)),
                                           "pass_speeds": int(len(model_dev.speeds))}} if model_name else None),
        "lines": {r: view_lines(v[0]) for r, v in views.items()},
        "results": results,
        "side_check_shank_gyro_walkway": side,
        "gain_diagnostic": {"label": GAIN_LABEL, "info": gain_info, "gain": gain,
                            "rows": ({r: {line: {k: (strip_stats(v) if isinstance(v, dict) else v)
                                                 for k, v in d.items()} for line, d in lines.items()}
                                      for r, lines in gain_rows.items()} if gain_rows else None),
                            "consistency": gain_check,
                            "status": ("computed" if gain_rows else
                                       "not computed: " + (gain_info or {}).get("reason", "no model"))},
        "rerun_check": rerun,
        "timing_s": {"load": round(t_load, 1), "targets_and_controls": round(t_dev, 1), "pd": round(t_pd, 1),
                     "all": round(time.time() - t_start, 1)},
        "moveatlas_head": head,
        "code_sha256": code_sha,
    }
    raw = res
    res = eg.clean(res)                 # the JSON holds null for a non-finite value, the markdown prints inf
    text = markdown(raw, res, views, sets, sets_dev, pdp, gain_rows, eg, g3, off, unfitted, model_name, loaded)
    run_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not a.events_only:
        common.write(res, out, STEM, text)
        prov = {"script": SCRIPT_REL, "script_sha256": script_sha, "task": TASK, "moveatlas_root": paths.MOVEATLAS,
                "moveatlas_head": head, "run_at": run_at, "python": sys.version.split()[0],
                "checkpoint": res["checkpoint"]["path"] if loaded is not None else None,
                "checkpoint_sha256": loaded.sha256 if loaded is not None else None, "code_sha256": code_sha,
                "code_files": [common.rel(p) for p in code_files]}
        common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
        print(text)
    if events is not None:
        write_events(out, events, a, res, loaded, script_sha, diag_sha, code_files, head, run_at, scored, tg_pd)
    print(f"written {common.rel(out)}" + (f" ({EVENTS_CSV} and {EVENTS_PROV} only)" if a.events_only else ""))


def write_events(out, events, a, res, loaded, script_sha, diag_sha, code_files, head, run_at, scored, targets):
    """events.csv and _provenance_events.json (amendment 1 A2; amendment 2 B1 and B2). Both texts are built and
    checked (em-dash, MoveAtlas path) before the first file opens; events.csv is written first and the provenance last,
    so that the provenance marks a complete pair. --events-only: both in exclusive-create mode, with the folder check;
    a default run writes them beside its other files, as it writes those."""
    from mdfm import paths
    from mdfm.gates import common
    text = events.to_csv(index=False, lineterminator="\n")
    files = list(code_files) + [os.path.join(TREE, "scripts", "eval_g3_diag.py")]
    st = events[events["kind"] == "stride"]
    prov = {"script": SCRIPT_REL, "script_sha256": script_sha, "task": TASK, "produces": EVENTS_SOURCE,
            "mode": "events_only" if a.events_only else "default", "run_at": run_at, "python": sys.version.split()[0],
            "file": EVENTS_CSV, "events_sha256": common.sha256_bytes(text.encode("utf-8")), "row": EVENTS_ROW,
            "diag_script": DIAG_REL, "diag_sha256": diag_sha, "columns": list(events.columns), "rows": int(len(events)),
            "kinds": {k: int(n) for k, n in events["kind"].value_counts().sort_index().items()},
            "matched_strides": int(st["matched"].astype(bool).sum()),
            "persons": int(events["person_id"].nunique()),
            "persons_sha256": _person_hash(sorted(set(events["person_id"]))),
            "scored_pd_persons": len(scored), "trials": int(targets.trials["trial_id"].nunique()),
            "matching": "mdfm.gates.g3.evaluate (strides, passes, walkway ICs); IC offsets fitted on the 14 dev "
                        "controls",
            "checkpoint": res["checkpoint"]["path"], "checkpoint_sha256": loaded.sha256,
            "inference": {k: (res.get("inference") or {}).get(k) for k in ("refine_steps", "points_per_stream",
                                                                           "batch_windows", "seed", "shape_pooling",
                                                                           "weights", "device")},
            "offsets_ms": {k: v for k, v in (res.get("offsets_ms") or {}).items() if k.startswith("model|")},
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "code_sha256": common.code_sha256(files), "code_files": [common.rel(p) for p in files],
            "code_file_sha256": {common.rel(p): common.sha256_file(p) for p in files}}
    if a.events_only:
        prov["folder_check"] = folder_check(out, res)
    body = json.dumps(EG().clean(prov), indent=1, default=str) + "\n"
    pair = ((os.path.join(out, EVENTS_CSV), text), (os.path.join(out, EVENTS_PROV), body))
    for path, t in pair:                # both texts checked before the first file opens
        common.assert_no_em_dash(t, rel(path))
        paths.assert_not_moveatlas(path)
    write = write_new if a.events_only else common.write_text
    for path, t in pair:                # events.csv first, its provenance last
        write(path, t)
    print(f"{EVENTS_CSV}: {prov['rows']} rows ({prov['kinds']}), {prov['matched_strides']} matched strides, "
          f"{prov['persons']} persons, SHA-256 {prov['events_sha256'][:12]}", flush=True)


def _person_hash(ids):
    from mdfm.data import sources
    return sources.person_hash(list(ids))


# ---------------------------------------------------------------- markdown
def markdown(raw, res, views, sets, sets_dev, pdp, gain_rows, eg, g3, off, unfitted, model_name, loaded):
    fmt, fmt_iv = eg.fmt, eg.fmt_iv
    L = [f"# {raw['what']}", "",
         f"Written by scripts/eval_g3_pd_dev.py (task {TASK}; harness mdfm/gates/g3.py). Every number below is read "
         f"from {STEM}.json, per_person.csv, persons.csv, validity.csv and offsets.csv of this folder. A dev estimate "
         "on non-lockbox PD persons, not a gate read: no lockbox row or record was read. No PD-half rule exists: the "
         "pass and kill values are those of design section 9, the same for PD persons and controls, and every verdict "
         "below is a reading aid.", "", STATEMENT_STAGE2, ""]
    if loaded is not None:
        ck = res["checkpoint"]
        inf = res["inference"]
        L += [f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
              f"{ck['step']:,}; {res['checkpoint_kind']}. Training persons: "
              + ", ".join(f"{s} {n}" for s, n in ck["persons"].items())
              + f"; PD persons of the pool among them: {len(res['persons']['trained_on_among_pool'])}. Inference: "
              f"{inf['refine_steps']} refinement steps of {inf['points_per_stream']} points per stream, batch_windows "
              f"{inf['batch_windows']}, shape pooled {inf['shape_pooling']}, device {inf['device']}.", ""]
    else:
        L += ["No model: the walkway, the two population baselines and the shank-gyroscope detector only.", ""]
    p = res["persons"]
    L += ["## Persons", "",
          f"- Pool: {p['pool']} non-lockbox weargait_pd PD persons with an IMU record in the canonical index "
          f"(SHA-256 {p['pool_sha256']}; equal to T19's pool: {p['pool_equals_t19_pool']}).",
          f"- Scored: {p['scored']} of the {p['scorable']} with targets (SHA-256 {p['scored_sha256']}). Not scored: "
          f"{len(p['not_scored'])}"
          + (": " + "; ".join(f"{k} ({v})" for k, v in sorted(p["not_scored"].items())) if p["not_scored"] else "")
          + "."]
    if loaded is not None:
        L += [f"- Training persons of the checkpoint among the scorable PD persons, left out of every PD row as "
              f"in-sample: {len(p['excluded_trained'])}" + (f" ({', '.join(p['excluded_trained'])})"
                                                            if p["excluded_trained"] else "") + "."]
    L += [f"- T19's proposed PD fold ({p['t19']['path']}, dev SHA-256 {p['t19']['sha256_dev']}): {p['t19']['dev']} dev "
          f"persons, {len(p['t19_dev_scored'])} scored"
          + (f" (not scored: {', '.join(p['t19_dev_not_scored'])})" if p["t19_dev_not_scored"] else "")
          + f"; {p['t19']['train']} train persons, {p['t19_train_with_targets']} with targets (population_pd learns "
          f"from them), {p['t19_train_scored']} of them scored.",
          "- Controls: the 14 dev controls fit the IC offsets and give the reference rows dev14 and dev_valid; the 54 "
          "training-fold controls give the population baseline.", ""]
    L += ["| row | persons | trials |", "|---|---|---|"]
    for r, info in res["rows"].items():
        L.append(f"| {r} | {info['n_persons']} | {info['trials']} |")
    v = res["validity"]
    L += ["", "Input validity (mdfm.gates.g3.input_validity, the blind check of G3_RULE.md section 2): "
          f"{v['pd_valid']} of {v['pd_records']} PD records valid; {v['dev_valid']} of {v['dev_records']} dev control "
          "records valid "
          f"({len(v['dev_valid_persons'])} persons). Invalid PD records: "
          + ("; ".join(v["pd_invalid"]) if v["pd_invalid"] else "none") + ".", ""]
    c = res["counts"]["pd"]
    L += [f"PD targets: {c['trials']} trials, {c['walkway_ic']} walkway ICs, {c['insole_ic']} insole ICs, "
          f"{c['strides']} walkway strides, {c['passes']} walkway passes; IMU records read: "
          f"{res['imu_records']['pd']}.", ""]
    if model_name:
        L += ["## GateResult rows per row (dev estimates; design section 9 lines, no PD-half rule)", ""]
        for r in list(sets) + list(sets_dev):
            L += [f"### {r}", "",
                  "| line | unit | pass | kill | model | 95 % interval | matched share | trivial baseline | "
                  "margin model / detector (ms) | verdict (dev estimate) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for x in raw["results"][r]:
                e = x["extra"]
                mg = (f"{fmt(e.get('margin_model_ms'))} / {fmt(e.get('margin_detector_ms'))}"
                      if "margin_model_ms" in e else "")
                L.append(f"| {x['line']} | {x['unit']} | {fmt(x['pass_line'])} | {fmt(x['kill_line'])} | "
                         f"{fmt(x['point'])} | {fmt_iv(x['interval'])} | {fmt(e.get('matched_share'), 3)} | "
                         f"{fmt(x['trivial_baseline'])} | {mg} | {eg.verdict_text(x['verdict'], e)} |")
            L.append("")
        L += [f"Rows marked '{eg.REPORTED}': the insole IC row, the second initial-contact reference, scored with the "
              "same statistic, interval and margin and reported; it decides nothing (G3_RULE.md section 6). Trivial "
              "baselines: the population baseline of the 54 training-fold controls (stride, speed, MAPE) and the "
              f"shank-gyroscope detector (IC; margin {g3.IC_MARGIN}). The matched-share floor of G3_RULE.md section 6 "
              f"({g3.MATCHED_SHARE_FLOOR}) is applied by g3.gate_results: a stride, speed or MAPE row inside its pass "
              "line with the model's matched share under it reads undetermined; the shares stand beside each line.", ""]
    L += ["## Every predictor per row", ""]
    for r in list(sets) + list(sets_dev):
        L += [f"### {r}", "", "| predictor | line | point | 95 % interval | matched share | reference events | persons "
              "| offset (ms) |", "|---|---|---|---|---|---|---|---|"]
        lines = raw["lines"][r]
        for name in PREDICTORS:
            for line in LINE_ORDER:
                st = lines.get(name, {}).get(line)
                if st is None:
                    continue
                L.append(f"| {name} | {line} | {fmt(st['point'])} | {fmt_iv(st['interval'])} | "
                         f"{fmt(st['matched_share'], 3)} | {st['n_events']} | {st['n_persons']} | "
                         f"{fmt(st.get('offset_ms'), 1) if 'offset_ms' in st else ''} |")
        L.append("")
    L += ["population: the 54 training-fold controls (out-of-sample for every PD person); population_pd: T19's "
          "proposed PD train fold persons with targets (out-of-sample on pd_t19dev, in-sample for the train-fold "
          "persons of pd_all). Population baselines (stride m, speed m/s): "
          + "; ".join(f"{t} {fmt(b['stride_m'], 3)} and {fmt(b['speed_m_s'], 3)}"
                      for t, b in raw["population_baseline"].items())
          + " (controls); "
          + "; ".join(f"{t} {fmt(b['stride_m'], 3)} and {fmt(b['speed_m_s'], 3)}"
                      for t, b in raw["population_pd_baseline"].items()) + " (PD train fold).", ""]
    L += ["Offsets (fitted on the 14 dev controls, median of predicted minus reference over the nearest predictions "
          f"within {g3.IC_MATCH_S} s): " + "; ".join(f"{r.predictor} against {r.reference} {r.offset_ms:.1f} ms "
                                                    f"(n {r.n_matched})" for r in off.itertuples())
          + f". Not fitted: {', '.join(unfitted) or 'none'}.", ""]
    s = raw["side_check_shank_gyro_walkway"]
    L += ["Side check of the shank-gyroscope detector against the walkway (g3.side_check; a trial foot is flagged when "
          "the other foot's ICs match more reference ICs than its own): PD "
          f"{s['pd_all']['flagged']} of {s['pd_all']['trial_feet']} trial feet"
          + (f" ({', '.join(s['pd_all']['flagged_list'])})" if s["pd_all"]["flagged_list"] else "")
          + f"; dev controls {s['dev14']['flagged']} of {s['dev14']['trial_feet']}"
          + (f" ({', '.join(s['dev14']['flagged_list'])})" if s["dev14"]["flagged_list"] else "") + ".", ""]
    gd = raw["gain_diagnostic"]
    L += ["## F1 training-fold gain, a diagnostic row (never a G3 verdict)", "", GAIN_LABEL + ".", ""]
    if gain_rows:
        g = gd["gain"]
        L += [f"Gains of end {g['end']} ({gd['info']['path']}, fitted on {g['fitted_on']}): stride {g['stride']:.4f}, "
              f"speed {g['speed']:.4f}. Raw equals g3's model line in every row (consistency within "
              f"{CONSISTENCY_TOL:g}).", "",
              "| row | line | persons | raw | 95 % | recalibrated | 95 % | raw minus recalibrated | 95 % paired |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r, lines in gain_rows.items():
            for line, d in lines.items():
                nd = 2 if line == "stride_mae_cm" else 3
                L.append(f"| {r} | {line} | {d['raw']['n_persons']} | {fmt(d['raw']['point'], nd)} | "
                         f"{fmt_iv(d['raw']['interval'], nd)} | {fmt(d['recalibrated']['point'], nd)} | "
                         f"{fmt_iv(d['recalibrated']['interval'], nd)} | {fmt(d['paired']['point'], nd)} | "
                         f"{fmt_iv(d['paired']['interval'], nd)} |")
        L.append("")
    else:
        L += [f"Status: {gd['status']}.", ""]
    rc = raw["rerun_check"]
    if rc.get("evaluated"):
        L += [f"Rerun check against {rc['reference']} (dev14 model lines and offsets, reported): "
              + "; ".join(f"{k} {fmt(v['reference'])} there, {fmt(v['this_run'])} here" for k, v in rc["lines"].items())
              + ". Offsets: " + "; ".join(f"{k} {fmt(v['reference'], 1)} there, {fmt(v['this_run'], 1)} here"
                                          for k, v in rc["offsets_ms"].items()) + ".", ""]
    t = raw["timing_s"]
    L += [f"Time: load {t['load']} s, targets and controls {t['targets_and_controls']} s, PD {t['pd']} s, all "
          f"{t['all']} s. MoveAtlas HEAD "
          f"{raw['moveatlas_head']}. G3 healthy-half rule {raw['rule']['g3_healthy_half_rule']['path']} status "
          f"'{raw['rule']['g3_healthy_half_rule']['status']}' (SHA-256 {raw['rule']['g3_healthy_half_rule']['sha256']}"
          "), read for its lines only.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
