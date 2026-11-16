"""G3 harness, healthy half: IMU-only gait readouts of weargait_pd controls against the walkway and the insoles
(design section 9; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.7 and B4.12; rule docs/rules/G3_RULE.md,
DRAFT until the owner approves and registers it, revision 2026-10-05; OD12, the initial-contact reference, closed
on 2026-10-05: the walkway contact decides the IC line, the insole reference is reported and decides nothing).

Set: weargait_pd controls, tasks SelfPace and HurriedPace, 13 Xsens IMUs as model input (acc and gyro).
Dev: the 14 dev controls of mdfm.data.sources.folds (drawn from the G2 TEST half). Lockbox: the 17 lockbox
controls of reports/lockbox/lockbox_persons.csv, read once through read_lockbox after the rule is registered.
The trivial baseline learns from the training fold (the other 54 non-lockbox controls).

Targets (reports/weargait_targets/, scripts/build_weargait_targets.py, rule b3a-1): events.csv (walkway ICs
with their passes, heel positions and strides), cache/insole_events.csv (every insole IC) and trials.csv
(samples per trial). Every row of the three files carries a lockbox column. The reader streams each file
and keeps the rows of the requested persons only; a requested lockbox person is refused before any file
opens unless allow_lockbox comes with the row of a logged lockbox read (a row of common.read_log for G3
whose rule is the registered docs/rules/G3_RULE.md, checked before any file opens), and a kept row whose
lockbox column disagrees with the lockbox list raises. A person id outside weargait_pd (CODY above all) is
refused before any file opens.
Model input: model_input(record), a copy of the IMU record with the magnetometer channels zeroed and
masked, is what the model callable of dev_run and read_lockbox receives (13 IMUs, acc and gyro only).
- Walkway ICs: event IC with phase Walk (non-censored contacts under the Walk label).
- Walkway strides: the footfalls of a straight pass (pass_id) in time order; the stride of footfall i
  (stride_cm filled: same foot two footfalls earlier, the other foot between, both non-partial) spans
  t0 = the IC of footfall i - 2 and t1 = the IC of footfall i; length stride_cm / 100 m.
- Walkway pass speed: passes with at least 3 non-partial footfalls, |heel(last) - heel(first)| x cell /
  (t_last - t_first) over the first and last non-partial footfalls, cell size from summary.json: the B3a
  speed rule applied per pass.
- Insole ICs: event IC with phase Walk, minus the trial's own offset (median of insole minus walkway time
  over the trial's matched walkway ICs of every phase, trials with at least 4: B3a's offset_removed_ms).
  A trial without an offset gives no insole reference.

Lines (lower is better; lines of design section 9):
- stride_mae_cm: a predicted stride matches a walkway stride of the same trial and foot when both ends lie
  within 0.15 s (the nearest by the larger end difference); absolute error in cm, mean per person, then
  mean over persons. Pass 5, kill above 10. A person without any matched stride drops out of the mean and
  is listed (persons_unmatched), as for speed.
- speed_mae_m_s: per walkway pass, the predicted speed over the pass interval; absolute error, mean per
  person, mean over persons. Pass 0.08, kill above 0.15.
- ic_median_ms, per reference (walkway, the primary reference, whose row decides the IC verdict: OD12, closed
  2026-10-05; insole with its per-trial offset removed, the second reference, reported with the same statistic,
  interval and margin, whose row decides nothing: gate_results marks it decides False and deciding() leaves it
  out): one constant offset per detector and reference, fitted on the dev persons only (fit_offset:
  the median of predicted minus reference over the nearest predictions within 0.3 s), is subtracted from
  the predicted times; each reference IC takes the nearest shifted prediction of the same foot, matched
  within 0.3 s (the walkway to insole pairing window of B3a); the error of an unmatched reference IC is
  infinite. A detector with no IC within 0.3 s of any dev reference IC (a model without contacts included)
  has no offset: fit_offsets records it as not fitted (UNFITTED, NaN with 0 matched), and its line leaves
  every reference IC unmatched (infinite, a kill, matched share 0 reported) instead of raising. Per person
  the median absolute error, then the mean over persons. Pass 30, kill above 50 (an infinite value, a
  person with more than half of its reference ICs unmatched, included), and kill unless the value is at
  most 0.8 times the shank-gyroscope detector's on the reference ICs both detectors matched (margin). The
  reference ICs of a trial and foot whose input a detector lacks (a record without that shank's gyroscope)
  are left out of that detector's line and counted.
- stride_mape_pct (simulator-only line): absolute stride error over the walkway stride in percent, mean per
  person, mean over persons, on the strides of stride_mae_cm. Pass 25, kill above 40.
Matched shares are reported beside every line. Intervals: person-clustered percentile bootstrap of the
mean over persons (common.cluster_bootstrap, BOOT_N resamples, SEED), 95 percent.
Matched-share floor (rule section 6, revision 2026-10-05, evening; section 10, item 4; task G3H): stride_mae_cm,
speed_mae_m_s and stride_mape_pct pass only when the model's matched share on the line (the matched walkway
strides, or passes, over all of them, as reported in the stats of the row being read) is at least
MATCHED_SHARE_FLOOR (0.9): gate_results reads the floor as a pass condition (common.verdict conditions), so below it
a line inside its pass line reads undetermined, never pass, a value beyond the kill line stays a kill and a line
with no finite point stays not evaluated. The floor only removes passes: no stat, interval or matched share
changes, and a baseline is never judged by it. The three rows carry extra floor_ok (True, False, or None when the
share is not a number) and matched_share_floor. The IC rows keep their own rule (an unmatched reference IC is an
infinite error; predicted ICs that match nothing cost nothing).

Dev rows on two sets (rule section 2, revision 2026-10-05), opt-in: evaluate(..., sets=...) and
dev_run(sets=True) add to the stats of every line a key "sets", {set name: the line's stats on the trials of
that set}, computed by the same code on the matched tables restricted to those trials, offsets as fitted (on the
14), each with its "trials" and "n_trials". dev_run(sets=True) uses DEV_SETS: dev14 (every dev trial) and dev12
(the 14 without EXCLUDED_FROM_12, WHC003 and WHC026: the subset no_whc003_whc026 of scripts/eval_g3_diag.py).
set_view(summ, tables, name) returns the (summ, tables) of one set, which gate_results reads as it reads the
default ones (gate_results_by_set: the GateResult lines of every set). Without sets, evaluate and dev_run return
what they returned before 2026-10-05.

Input validity (plan task T32; docs/proposals/G3_INPUT_VALIDITY_PROPOSAL.md, adopted as the item "Input validity
(blind)" of rule section 2 and section 10, item 5, closed 2026-10-05; applied by read_lockbox since task G3H, opt-in
in dev_run): input_validity(record) reads one IMU record and nothing else (no target, no model output, no G3
value) and returns its checks:
- site completeness: shank_L, shank_R, foot_L and foot_R (REQUIRED_SITES) each hold their accelerometer and
  gyroscope channels in the channel mask with at least MIN_VALID_SHARE of their samples valid (sample_valid and
  finite), the definition of a held site in reports/stage2_sources/summary.md;
- shank sides, on a complete record: (sign) per shank, the signed sagittal axis of its gyroscope
  (readouts.gait.sagittal_rate: the first principal axis, mid-swing peaks positive) has a positive dot product
  with SWING_AXIS of its side, the forward-swing axis of that side's xsens13 shank sensor in its own frame (-z
  left, +z right: the sensor z axis is the outward normal of a lateral shank sensor, as the mount prior
  reports/mount_priors/priors.json places it); (phase) per shank, the envelope of its gyroscope norm (centred
  moving mean over ENVELOPE_S) correlates more with the same-side foot's envelope than with the other foot's
  (Pearson over the samples valid on the four sites; fewer than MIN_TEST_S of them: untestable).
A record is valid when it is complete, testable and passes both tests on both shanks; reasons lists every
failure. validity_table(persons, trials) checks the records of a set of persons; dev_run(validity=True) checks
every dev record, keeps the per-trial table in Targets.validity and adds the set valid_inputs (the trials of
valid records) beside dev14.
read_lockbox applies it (rule sections 2 and 9; task G3H), inside the logged read: each lockbox record the read
opens goes through validity_row before the model sees it and before any line is computed; evaluate then scores two
sets, SET_VALID (the trials of valid records, the primary row) and SET_ALL (every trial of the 17 controls, the
all-trials row), the GateResult rows of both come from gate_results_by_set with extra set, the primary rows decide
(the insole IC row still decides nothing), the all-trials rows are reported beside them (extra decides False,
set_role), and the validity table with its accounting (validity_report: invalid trials with their reasons, trials
without a record, persons left without a valid trial, also in extra input_validity of every row) comes back with
the read (LockboxRead.validity and .report). The offsets stay those fitted on the 14 dev controls. A lockbox
trial without an IMU record has no valid record: it is out of the primary row, in the all-trials row and listed.

Predictions: Predictions(name, ics, strides, speeds) from bodies (body_predictions: mdfm.readouts.gait on
the stitched model output of each trial), from the shank gyroscopes (shank_predictions: ICs only, one per
foot from mdfm.readouts.gait.shank_gyro_ic) or from the training-fold population (population_predictions:
per task the person-weighted mean stride and pass speed of the training fold, on every walkway stride and
pass; no IC).

Readout self-check (readout_check, on the first N_WALKS forward walks of the AMASS dev body store, never a
weargait person): per take the readout speed (median pass_speed over the strides between contact ICs,
strides up to 2 s) against factory.gait.speed_and_cadence, and the lag of the contact-label ICs after the
kinematic heel strikes (gait.heel_strikes); pooled, the share of steady straight steps whose reference
event (heel strike, or contact IC) has a shank_gyro_ic detection of the same foot within 50 ms after one
pooled offset, on shank gyroscopes rendered from the truth body without device noise. A steady straight
step: pelvis speed at the step at least STEADY x the take speed, and the pelvis chord over the 1 s around it
at least STRAIGHT x its path.
"""
import copy
import csv
import functools
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from mdfm import paths
from mdfm.data import canonical, sources
from mdfm.data.canonical import Refused
from mdfm.readouts import gait

GATE = "G3"
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G3_RULE.md")
TARGETS_DIR = os.path.join(paths.TREE, "reports", "weargait_targets")
EVENTS_CSV = os.path.join(TARGETS_DIR, "events.csv")
INSOLE_CSV = os.path.join(TARGETS_DIR, "cache", "insole_events.csv")
TRIALS_CSV = os.path.join(TARGETS_DIR, "trials.csv")
SUMMARY_JSON = os.path.join(TARGETS_DIR, "summary.json")
TARGET_FILES = (EVENTS_CSV, INSOLE_CSV, TRIALS_CSV)
SOURCE = "weargait_pd"
PREFIX = "weargait:"
TASKS = ("SelfPace", "HurriedPace")
WALK = "Walk"
TARGET_RULE = "b3a-1"                   # rule_version of the targets (summary.json)
STRIDE_MATCH_S = 0.15
IC_MATCH_S = 0.3
UNFITTED = (float("nan"), 0)            # (offset_s, n matched) of a detector whose offset could not be fitted
MIN_MATCHED_FOR_OFFSET = 4              # B3a MIN_MATCHED_FOR_OFFSET
MIN_PASS_FOOTFALLS = 3
LINES = {"stride_mae_cm": (5.0, 10.0), "speed_mae_m_s": (0.08, 0.15), "ic_median_ms": (30.0, 50.0),
         "stride_mape_pct": (25.0, 40.0)}
UNITS = {"stride_mae_cm": "cm", "speed_mae_m_s": "m/s", "ic_median_ms": "ms", "stride_mape_pct": "percent"}
IC_MARGIN = 0.8
MATCHED_SHARE_FLOOR = 0.9               # rule section 6 (revision 2026-10-05, evening; section 10, item 4): a pass needs it
FLOORED_LINES = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct")   # the lines the floor conditions
REFERENCES = ("walkway", "insole")
PRIMARY_REFERENCE = "walkway"           # decides the IC verdict (OD12 closed 2026-10-05, rule sections 1, 5 and 6)
SHANK_SITES = {"L": 7, "R": 8}          # canonical_imu.yaml locations shank_L, shank_R
FOOT_SITES = {"L": 11, "R": 12}         # canonical_imu.yaml locations foot_L, foot_R
ACC = slice(0, 3)
GYRO = slice(3, 6)
MAG = slice(6, 9)
# dev rows on two sets (rule section 2, revision 2026-10-05); opt-in (evaluate sets, dev_run sets=True)
SET_14 = "dev14"
SET_12 = "dev12"
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")   # subset no_whc003_whc026 of scripts/eval_g3_diag.py
DEV_SETS = {SET_14: (), SET_12: EXCLUDED_FROM_12}          # set name -> persons left out
SET_VALID = "valid_inputs"              # the trials of valid records: the primary row (rule section 2, input validity)
SET_ALL = "all_trials"                  # every trial of the set read: the all-trials row of the lockbox read
# input validity (T32; rule section 2 "Input validity (blind)", docs/proposals/G3_INPUT_VALIDITY_PROPOSAL.md)
REQUIRED_SITES = {"shank_L": SHANK_SITES["L"], "shank_R": SHANK_SITES["R"], "foot_L": FOOT_SITES["L"],
                  "foot_R": FOOT_SITES["R"]}
MIN_VALID_SHARE = 0.5                   # a held site: at least half of its samples valid (stage2_sources summary)
SWING_AXIS = {"L": (0.0, 0.0, -1.0), "R": (0.0, 0.0, 1.0)}   # xsens13 shank sensor frame, forward swing
ENVELOPE_S = 0.2                        # s, centred moving mean of the gyroscope norm (phase test)
MIN_TEST_S = 10.0                       # s of samples valid on the four required sites, at least (one window)
SEED = 20260929
BOOT_N = 2000
HEX64 = re.compile(r"[0-9a-f]{64}")     # a lowercase hex SHA-256, whole string (fullmatch)
WALK_STYLES = ("normal", "fast", "handbag", "shoulder bag", "backpack", "baggage on", "Walking0")
N_WALKS = 20
STRAIGHT = 0.95                         # pelvis chord over path within 0.5 s of a step, at least
STEADY = 0.7                            # pelvis speed at a step over the take's speed, at least
AGREE_S = 0.05                          # s, detector agreement tolerance of the self-check


# ---------------------------------------------------------------- persons
@functools.lru_cache(maxsize=1)
def _folds():
    f = sources.folds()
    return f[f["source"] == SOURCE].copy()


def controls(fold):
    """Sorted person ids of the non-lockbox weargait_pd controls of a fold of sources.folds: 'dev' (14) or
    'train' (54)."""
    f = _folds()
    if fold not in set(f["fold"]):
        raise ValueError(f"fold {fold!r} not in {sorted(set(f['fold']))}")
    return sorted(f.loc[f["fold"] == fold, "person_id"])


@functools.lru_cache(maxsize=1)
def _lockbox_rows():
    from mdfm.data import amass_index
    amass_index.load()                          # checks lockbox_persons.csv against lockbox.sha256
    with open(amass_index.LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        return tuple(r for r in csv.DictReader(f) if r["source"] == SOURCE)


def lockbox_persons():
    """Every weargait_pd lockbox person (controls and PD), sorted."""
    return sorted(r["person_id"] for r in _lockbox_rows())


def lockbox_controls():
    """The weargait_pd lockbox controls (17), sorted."""
    return sorted(r["person_id"] for r in _lockbox_rows() if r["group"] == "control")


def _check_read_row(read_row):
    """Refused unless read_row is a row of the lockbox read log (common.read_log) for G3 whose rule SHA-256
    is that of docs/rules/G3_RULE.md as registered now (common.rule): a flag alone, or a row made up by the
    caller, opens nothing."""
    from mdfm.gates import common
    if not isinstance(read_row, dict) or read_row.get("gate") != GATE:
        raise Refused("allow_lockbox needs the row of a logged G3 lockbox read (common.log_lockbox_read)")
    _, sha = common.rule(RULE_PATH, gate=GATE)
    if read_row.get("rule_sha256") != sha:
        raise Refused("the lockbox read row names another rule than the registered docs/rules/G3_RULE.md")
    keys = ("read_at", "gate", "checkpoint_sha256", "rule_sha256")
    if not any(all(r[k] == str(read_row.get(k, "")) for k in keys) for r in common.read_log()):
        raise Refused("the lockbox read row is not in reports/gates/lockbox_reads.csv")


def _guard(persons, allow_lockbox, read_row):
    """Refuse a person outside weargait_pd (CODY included: G3 never reads it), lockbox persons without a
    logged read, and a lockbox flag without one (_check_read_row)."""
    other = sorted(p for p in persons if not str(p).startswith(PREFIX))
    if other:
        cody = [p for p in other if canonical.is_cody(p) or str(p).split(":", 1)[0] in canonical.CODY_NAMESPACES]
        raise Refused(f"G3 reads weargait_pd persons only; {len(other)} other ids"
                      + (f", {len(cody)} of them CODY (evaluation-only, never a G3 input)" if cody else ""))
    locked = set(persons) & set(lockbox_persons())
    if allow_lockbox:
        _check_read_row(read_row)
    if locked and not allow_lockbox:
        raise Refused(f"G3 lockbox persons requested outside a logged lockbox read: {len(locked)} persons")
    return locked


# ---------------------------------------------------------------- targets
def read_rows(path, persons, allow_lockbox=False, read_row=None):
    """Rows (dicts of strings) of a target CSV whose person_id is one of `persons`, streamed; refusals and
    the lockbox-column check of the module docstring. A dev read returns no row with lockbox True."""
    persons = set(persons)
    locked = _guard(persons, allow_lockbox, read_row)
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            p = row["person_id"]
            if p not in persons:
                continue
            flag = row["lockbox"]
            if flag not in ("True", "False"):
                raise ValueError(f"{os.path.basename(path)}: lockbox column {flag!r} for {p}")
            if (flag == "True") != (p in locked):
                raise ValueError(f"{os.path.basename(path)}: lockbox column {flag} for {p} disagrees with the "
                                 "lockbox list")
            out.append(row)
    if not allow_lockbox and any(r["lockbox"] != "False" for r in out):
        raise Refused(f"{os.path.basename(path)}: a lockbox row in a dev read")
    return out


def _f(v):
    return float(v) if v not in ("", None) else float("nan")


@functools.lru_cache(maxsize=1)
def target_meta():
    """(cell size in cm, rule version) of reports/weargait_targets/summary.json."""
    with open(SUMMARY_JSON, encoding="utf-8") as f:
        s = json.load(f)
    return float(s["cell_cm"]), str(s["rule_version"])


@dataclass
class Targets:
    """Reference quantities of a set of persons (module docstring). Frames: trials (trial_id, person_id,
    task, samples), walkway_ic and insole_ic (trial_id, person_id, task, foot, t_s; insole also t_raw_s
    and offset_ms), strides (trial_id, person_id, task, pass_id, foot, t0, t1, length_m), passes
    (trial_id, person_id, task, pass_id, t0, t1, n_footfalls, speed_m_s), offsets (trial_id, offset_ms,
    n_matched). counts: rows read and kept. validity: None, or the per-trial input-validity table
    (validity_table) when dev_run checked the records (validity=True)."""
    persons: tuple
    lockbox: bool
    trials: object
    walkway_ic: object
    insole_ic: object
    strides: object
    passes: object
    offsets: object
    counts: dict = field(default_factory=dict)
    validity: object = None


def _pass_tables(ic, cell_cm):
    """Walkway strides and pass speeds from the IC rows of events.csv (module docstring)."""
    strides, passes = [], []
    rows = [r for r in ic if r["pass_id"] != ""]
    groups = {}
    for r in rows:
        groups.setdefault((r["trial_id"], r["pass_id"]), []).append(r)
    for (trial, pid), P in sorted(groups.items()):
        P.sort(key=lambda r: (float(r["walkway_t_s"]), r["foot"]))
        t = [float(r["walkway_t_s"]) for r in P]
        for i, r in enumerate(P):
            if r["stride_cm"] == "":
                continue
            if i < 2 or P[i - 2]["foot"] != r["foot"] or P[i - 1]["foot"] == r["foot"]:
                raise ValueError(f"{trial} pass {pid}: stride at footfall {i} without a same-foot footfall two "
                                 "before")
            h0, h1 = _f(P[i - 2]["heel_x_cells"]), _f(r["heel_x_cells"])
            if abs(abs(h1 - h0) - _f(r["stride_cells"])) > 1e-6:
                raise ValueError(f"{trial} pass {pid}: stride_cells {r['stride_cells']} is not |{h1} - {h0}|")
            strides.append((trial, r["person_id"], r["task"], pid, r["foot"], t[i - 2], t[i],
                            _f(r["stride_cm"]) / 100.0))
        full = [i for i, r in enumerate(P) if r["partial"] == "False" and r["heel_x_cells"] != ""]
        if len(full) >= MIN_PASS_FOOTFALLS:
            i0, i1 = full[0], full[-1]
            dt = t[i1] - t[i0]
            if dt > 0:
                d = abs(_f(P[i1]["heel_x_cells"]) - _f(P[i0]["heel_x_cells"])) * cell_cm / 100.0
                passes.append((trial, P[i0]["person_id"], P[i0]["task"], pid, t[i0], t[i1], len(P), d / dt))
    return strides, passes


def load_targets(persons, allow_lockbox=False, read_row=None, tasks=TASKS):
    """Targets of the given persons for the given tasks (module docstring)."""
    import pandas as pd
    persons = tuple(sorted(set(persons)))
    _guard(persons, allow_lockbox, read_row)
    cell_cm, rule_version = target_meta()
    if rule_version != TARGET_RULE:
        raise ValueError(f"targets rule {rule_version}, the harness reads {TARGET_RULE}")
    trows = [r for r in read_rows(TRIALS_CSV, persons, allow_lockbox, read_row) if r["task"] in tasks]
    trials = pd.DataFrame([(r["trial_id"], r["person_id"], r["task"], int(r["samples"])) for r in trows],
                          columns=["trial_id", "person_id", "task", "samples"])
    tid = set(trials["trial_id"])
    ev = [r for r in read_rows(EVENTS_CSV, persons, allow_lockbox, read_row) if r["trial_id"] in tid]
    ins = [r for r in read_rows(INSOLE_CSV, persons, allow_lockbox, read_row) if r["trial_id"] in tid]
    ic = [r for r in ev if r["event"] == "IC"]
    walkway = pd.DataFrame([(r["trial_id"], r["person_id"], r["task"], r["foot"], float(r["walkway_t_s"]))
                            for r in ic if r["phase"] == WALK],
                           columns=["trial_id", "person_id", "task", "foot", "t_s"])
    diffs = {}
    for r in ic:
        if r["state"] == "matched":
            diffs.setdefault(r["trial_id"], []).append(float(r["diff_ms"]))
    offsets = pd.DataFrame([(k, float(np.median(v)), len(v)) for k, v in sorted(diffs.items())
                            if len(v) >= MIN_MATCHED_FOR_OFFSET], columns=["trial_id", "offset_ms", "n_matched"])
    off = dict(zip(offsets["trial_id"], offsets["offset_ms"]))
    task = dict(zip(trials["trial_id"], trials["task"]))
    insole = pd.DataFrame([(r["trial_id"], r["person_id"], task[r["trial_id"]], r["foot"],
                            float(r["t_s"]) - off[r["trial_id"]] / 1000.0, float(r["t_s"]), off[r["trial_id"]])
                           for r in ins if r["event"] == "IC" and r["phase"] == WALK and r["trial_id"] in off],
                          columns=["trial_id", "person_id", "task", "foot", "t_s", "t_raw_s", "offset_ms"])
    st, ps = _pass_tables(ic, cell_cm)
    strides = pd.DataFrame(st, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1", "length_m"])
    passes = pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                       "speed_m_s"])
    no_offset = sum(1 for r in ins if r["event"] == "IC" and r["phase"] == WALK and r["trial_id"] not in off)
    counts = {"trials": len(trials), "events_rows": len(ev), "insole_rows": len(ins), "walkway_ic": len(walkway),
              "insole_ic": len(insole), "insole_ic_without_offset": no_offset,
              "trials_with_offset": len(offsets), "strides": len(strides), "passes": len(passes),
              "lockbox_rows": sum(r["lockbox"] == "True" for r in trows + ev + ins)}
    return Targets(persons, bool(allow_lockbox), trials, walkway, insole, strides, passes, offsets, counts)


# ---------------------------------------------------------------- IMU records
def trial_id(person_id, task):
    """events.csv trial id of a person and task: '<ID>_<task>' with the ID as in the file name."""
    if not person_id.startswith(PREFIX):
        raise ValueError(f"not a weargait_pd person id: {person_id}")
    return f"{person_id[len(PREFIX):]}_{task}"


def imu_rows(persons, allow_lockbox=False, read_row=None, tasks=TASKS):
    """Index rows (dicts) of the weargait_pd IMU records of the persons and tasks, refused like read_rows."""
    _guard(persons, allow_lockbox, read_row)
    df = canonical.index()
    sel = df[(df["source"] == SOURCE) & (df["modality"] == "imu") & df["person_id"].isin(set(persons))
             & df["key"].isin(tasks)]
    if not allow_lockbox and sel["lockbox"].any():
        raise Refused("a lockbox IMU row in a dev read")
    return sel.sort_values(["person_id", "key"], kind="stable").to_dict("records")


def imu_records(persons, trials, allow_lockbox=False, read_row=None, tasks=TASKS):
    """Generator of (trial_id, record) over the IMU records of the persons (canonical.load with the flag),
    each checked against the samples of trials.csv."""
    samples = dict(zip(trials["trial_id"], trials["samples"]))
    for row in imu_rows(persons, allow_lockbox, read_row, tasks):
        rec = canonical.load(row, allow_lockbox=allow_lockbox)
        tid = trial_id(row["person_id"], row["key"])
        n = int(np.asarray(rec["signal"]).shape[0])
        if tid not in samples or samples[tid] != n or float(rec["fps"]) != 100.0:
            raise ValueError(f"{tid}: record of {n} samples at {rec['fps']} Hz, trials.csv {samples.get(tid)} "
                             "at 100 Hz")
        yield tid, rec


def model_input(record):
    """The record as the model reads it in G3: a shallow copy whose signal and channel_mask are copies with
    the magnetometer channels (6 to 8) zeroed and masked; the record itself is left unchanged."""
    rec = dict(record)
    sig = np.array(record["signal"], copy=True)
    cm = np.array(record["channel_mask"], dtype=bool, copy=True)
    sig[..., MAG] = 0
    cm[:, MAG] = False
    rec["signal"], rec["channel_mask"] = sig, cm
    return rec


def shank_gyro(record):
    """{"L": (T, 3), "R": (T, 3)} deg/s of an IMU record's shank sensors, NaN where sample_valid is false; a
    shank without its gyroscope channels is left out."""
    sig = np.asarray(record["signal"], dtype=np.float64)
    loc = [int(x) for x in np.asarray(record["location_ids"])]
    cm = np.asarray(record["channel_mask"], dtype=bool)
    valid = np.asarray(record["sample_valid"], dtype=bool) if record.get("sample_valid") is not None \
        else np.ones(sig.shape[:2], bool)
    out = {}
    for foot, site in SHANK_SITES.items():
        if site in loc:
            j = loc.index(site)
            if cm[j, GYRO].all():
                g = sig[:, j, GYRO].copy()
                g[~valid[:, j]] = np.nan
                out[foot] = g
    return out


# ---------------------------------------------------------------- input validity (T32; rule section 2)
VALIDITY_COLUMNS = ("trial_id", "person_id", "task", "valid", "reasons", "missing_sites", "test_s", "sign_L",
                    "sign_R", "phase_same_L", "phase_other_L", "phase_same_R", "phase_other_R")


def _envelope(x, valid, n):
    """Centred moving mean over n (odd) samples of x (T,) over its valid samples; NaN where none is valid."""
    from scipy.ndimage import uniform_filter1d
    num = uniform_filter1d(np.where(valid, x, 0.0), n, mode="constant")
    den = uniform_filter1d(valid.astype(np.float64), n, mode="constant")
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def _corr(a, b):
    """Pearson correlation of two finite series; NaN when one is constant or shorter than 3."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def input_validity(record):
    """The input-validity checks of one weargait_pd IMU record (module docstring; rule section 2): reads the
    record only. Returns {valid, reasons (tuple: 'incomplete: <site>', 'untestable: ...', 'sign: shank_<side>',
    'phase: shank_<side>'), missing_sites (tuple), test_s (s of samples valid on the four required sites),
    sign_L, sign_R (dot product of each shank's signed sagittal axis with SWING_AXIS of its side), phase_same_L,
    phase_other_L, phase_same_R, phase_other_R (envelope correlations of each shank with the same-side and the
    other foot)}; a value whose test did not run is NaN. A record of another source raises ValueError."""
    src = record.get("source")
    if src is not None and src != SOURCE:
        raise ValueError(f"input_validity reads {SOURCE} records (xsens13 shank mounts), got {src}")
    sig = np.asarray(record["signal"], dtype=np.float64)
    loc = [int(x) for x in np.asarray(record["location_ids"]).ravel()]
    cm = np.asarray(record["channel_mask"], dtype=bool)
    sv = (np.asarray(record["sample_valid"], dtype=bool) if record.get("sample_valid") is not None
          else np.ones(sig.shape[:2], bool))
    rate = float(record.get("fps", 100.0))
    out = {"valid": False, "reasons": (), "missing_sites": (), "test_s": 0.0,
           **{f"sign_{s}": float("nan") for s in gait.FEET},
           **{f"phase_{k}_{s}": float("nan") for s in gait.FEET for k in ("same", "other")}}
    held, missing = {}, []
    for name, site in REQUIRED_SITES.items():
        if site in loc:
            j = loc.index(site)
            if cm[j, ACC].all() and cm[j, GYRO].all():
                v = sv[:, j] & np.isfinite(sig[:, j, :6]).all(axis=1)
                if len(v) and float(v.mean()) >= MIN_VALID_SHARE:
                    held[name] = (j, v)
                    continue
        missing.append(name)
    out["missing_sites"] = tuple(missing)
    if missing:
        out["reasons"] = tuple(f"incomplete: {n}" for n in missing)
        return out
    both = np.logical_and.reduce([v for _, v in held.values()])
    out["test_s"] = float(both.sum()) / rate
    if both.sum() < MIN_TEST_S * rate:
        out["reasons"] = (f"untestable: {out['test_s']:.2f} s valid on the four required sites",)
        return out
    gyro = {}
    for name, (j, v) in held.items():
        g = sig[:, j, GYRO].copy()
        g[~v] = np.nan
        gyro[name] = g
    reasons = []
    for side in gait.FEET:                              # sign: the mounting side of each shank sensor
        _, axis = gait.sagittal_rate(gyro[f"shank_{side}"])
        out[f"sign_{side}"] = d = float(np.dot(axis, SWING_AXIS[side]))
        if not d > 0:
            reasons.append(f"sign: shank_{side}")
    n = 2 * int(round(ENVELOPE_S * rate / 2.0)) + 1
    env = {name: _envelope(np.linalg.norm(np.nan_to_num(g), axis=1), held[name][1], n)[both]
           for name, g in gyro.items()}
    for side, other in (("L", "R"), ("R", "L")):        # phase: each shank swings with its own foot
        same = _corr(env[f"shank_{side}"], env[f"foot_{side}"])
        oth = _corr(env[f"shank_{side}"], env[f"foot_{other}"])
        out[f"phase_same_{side}"], out[f"phase_other_{side}"] = same, oth
        if not same > oth:
            reasons.append(f"phase: shank_{side}")
    out["reasons"] = tuple(reasons)
    out["valid"] = not reasons
    return out


def validity_row(trial, person_id, task, record):
    """One row of the input-validity table: the trial, its person and task, and input_validity(record) with
    reasons and missing_sites joined by '; '."""
    v = input_validity(record)
    row = {"trial_id": trial, "person_id": person_id, "task": task, **v}
    row["reasons"], row["missing_sites"] = "; ".join(v["reasons"]), "; ".join(v["missing_sites"])
    return {k: row[k] for k in VALIDITY_COLUMNS}


def validity_frame(rows):
    """DataFrame of validity_row rows (VALIDITY_COLUMNS), sorted by trial."""
    import pandas as pd
    return pd.DataFrame(list(rows), columns=list(VALIDITY_COLUMNS)).sort_values("trial_id", kind="stable") \
        .reset_index(drop=True)


def validity_table(persons, trials, allow_lockbox=False, read_row=None):
    """The input-validity table of the persons' records (imu_records with the same refusals; trials: the trials
    frame of their Targets)."""
    info = trials.set_index("trial_id")
    return validity_frame(validity_row(tid, info.at[tid, "person_id"], info.at[tid, "task"], rec)
                          for tid, rec in imu_records(persons, trials, allow_lockbox, read_row))


def valid_trials(table):
    """The trial ids of the valid records of a validity table."""
    return frozenset(table.loc[table["valid"].astype(bool), "trial_id"])


def validity_report(table, trials):
    """The accounting that rule section 2 asks of the input-validity check: the invalid trials counted and listed
    with their reasons, and the persons left without a valid trial. table: a validity_table or validity_frame;
    trials: the trials frame of the Targets read. Returns {n_trials, n_records, n_valid, n_invalid, invalid (list of
    {trial_id, person_id, task, reasons}), trials_without_record (trials of the targets that no record was read for:
    not valid, so out of the primary row, and listed), persons_without_valid_trial (sorted)}."""
    valid = table["valid"].astype(bool)
    invalid = [{"trial_id": str(r.trial_id), "person_id": str(r.person_id), "task": str(r.task),
                "reasons": str(r.reasons)} for r, v in zip(table.itertuples(), valid) if not v]
    persons = set(trials["person_id"])
    return {"n_trials": int(len(trials)), "n_records": int(len(table)), "n_valid": int(valid.sum()),
            "n_invalid": len(invalid), "invalid": invalid,
            "trials_without_record": sorted(set(trials["trial_id"]) - set(table["trial_id"])),
            "persons_without_valid_trial": sorted(persons - set(table.loc[valid, "person_id"]))}


# ---------------------------------------------------------------- predictions
@dataclass
class Predictions:
    """A predictor's outputs over trials: ics DataFrame(trial_id, foot, t_s), strides DataFrame(trial_id,
    foot, t0, t1, length_m), speeds DataFrame(trial_id, pass_id, speed_m_s); None where it predicts
    nothing. covered: the (trial_id, foot) pairs whose input the predictor had (None: every pair); the IC
    lines score the reference ICs of covered pairs only and count the others."""
    name: str
    ics: object = None
    strides: object = None
    speeds: object = None
    covered: object = None


@dataclass
class TrialBody:
    """A model's body over one trial on the 100 Hz trial clock: joints (T, 22, 3) m in one frame, or (W, T,
    22, 3) with window_of_frame (T,) (readouts.gait.on_clock and stitch); contact (T, 2) probabilities."""
    trial_id: str
    joints: object
    contact: object
    window_of_frame: object = None
    rate: float = 100.0


def stitched_body(tid, joints_windows, contact_windows, hop, starts=None, rate=100.0):
    """TrialBody from per-window model outputs (each window in its own frame): contacts stitched, joints on
    the trial clock with the window of each frame (readouts.gait)."""
    contact, wof = gait.stitch(contact_windows, hop, starts)
    return TrialBody(tid, gait.on_clock(joints_windows, hop, starts), contact, wof, rate)


def body_predictions(bodies, targets, name="model"):
    """Predictions of bodies (TrialBody iterable): ICs from the contact probabilities, strides between
    successive ICs of a foot, pass speeds over the walkway pass intervals of the targets."""
    import pandas as pd
    ics, st, sp = [], [], []
    passes = targets.passes.groupby("trial_id")
    for b in bodies:
        ic = gait.initial_contacts(b.contact, b.rate)
        for foot, times in ic.items():
            ics += [(b.trial_id, foot, float(t)) for t in times]
        s = gait.strides(b.joints, ic, b.window_of_frame, b.rate)
        st += [(b.trial_id, r.foot, r.t0, r.t1, r.length_m) for r in s.itertuples()]
        if b.trial_id in passes.groups:
            n = np.asarray(b.contact).shape[0]
            for r in passes.get_group(b.trial_id).itertuples():
                if gait.frame_of(r.t1, b.rate) < n:
                    sp.append((b.trial_id, r.pass_id, gait.pass_speed(b.joints, r.t0, r.t1, b.window_of_frame, b.rate)))
    return Predictions(name, pd.DataFrame(ics, columns=["trial_id", "foot", "t_s"]),
                       pd.DataFrame(st, columns=["trial_id", "foot", "t0", "t1", "length_m"]),
                       pd.DataFrame(sp, columns=["trial_id", "pass_id", "speed_m_s"]))


def shank_predictions(gyros, rate=100.0, name="shank_gyro"):
    """ICs of the shank-gyroscope detector: gyros {trial_id: {"L": (T, 3), "R": (T, 3)} deg/s}; a foot
    without its shank gyroscope is not covered."""
    import pandas as pd
    rows = []
    for tid, g in sorted(gyros.items()):
        for foot, x in sorted(g.items()):
            rows += [(tid, foot, float(t)) for t in gait.shank_gyro_ic(x, rate)]
    covered = frozenset((tid, foot) for tid, g in gyros.items() for foot in g)
    return Predictions(name, pd.DataFrame(rows, columns=["trial_id", "foot", "t_s"]), covered=covered)


def walkway_predictions(targets, name="walkway"):
    """The walkway itself as a predictor (its ICs, strides and pass speeds): zero error on the walkway lines
    and, against the insole reference, the floor that the two references leave (OD12)."""
    return Predictions(name, targets.walkway_ic[["trial_id", "foot", "t_s"]].copy(),
                       targets.strides[["trial_id", "foot", "t0", "t1", "length_m"]].copy(),
                       targets.passes[["trial_id", "pass_id", "speed_m_s"]].copy())


def population_baseline(train_targets):
    """Per task, the person-weighted mean stride (m) and pass speed (m/s) of the training fold: {task:
    (stride_m, speed_m_s)}."""
    from mdfm.gates import common
    out = {}
    for task in sorted(set(train_targets.trials["task"])):
        s = train_targets.strides[train_targets.strides["task"] == task]
        p = train_targets.passes[train_targets.passes["task"] == task]
        out[task] = (common.person_weighted_mean(s["length_m"], s["person_id"]),
                     common.person_weighted_mean(p["speed_m_s"], p["person_id"]))
    return out


def population_predictions(baseline, targets, name="population"):
    """The population baseline on every walkway stride and pass of the targets (same end times, so every
    stride and pass matches)."""
    s = targets.strides[["trial_id", "foot", "t0", "t1"]].copy()
    s["length_m"] = [baseline[t][0] for t in targets.strides["task"]]
    p = targets.passes[["trial_id", "pass_id"]].copy()
    p["speed_m_s"] = [baseline[t][1] for t in targets.passes["task"]]
    return Predictions(name, None, s, p)


# ---------------------------------------------------------------- matching
def match_strides(pred, targets, tol=STRIDE_MATCH_S):
    """targets.strides with pred_m (nearest predicted stride of the same trial and foot by the larger end
    difference), end_diff_s, matched (both ends within tol), err_cm and ape_pct (NaN when unmatched)."""
    df = targets.strides.copy()
    by = {k: g for k, g in pred.strides.groupby(["trial_id", "foot"])} if len(pred.strides) else {}
    pm, dd = [], []
    for r in df.itertuples():
        g = by.get((r.trial_id, r.foot))
        if g is None:
            pm.append(np.nan)
            dd.append(np.inf)
            continue
        d = np.maximum(np.abs(g["t0"].to_numpy() - r.t0), np.abs(g["t1"].to_numpy() - r.t1))
        k = int(np.argmin(d))
        pm.append(float(g["length_m"].iloc[k]))
        dd.append(float(d[k]))
    df["pred_m"] = pm
    df["end_diff_s"] = dd
    df["matched"] = df["end_diff_s"] <= tol + 1e-9
    df["err_cm"] = np.where(df["matched"], 100.0 * (df["pred_m"] - df["length_m"]), np.nan)
    df["ape_pct"] = np.where(df["matched"], 100.0 * np.abs(df["pred_m"] - df["length_m"]) / df["length_m"], np.nan)
    return df


def match_speeds(pred, targets):
    """targets.passes with pred_m_s, matched (a finite prediction for the pass) and err_m_s."""
    df = targets.passes.copy()
    p = {(r.trial_id, str(r.pass_id)): r.speed_m_s for r in pred.speeds.itertuples()}
    df["pred_m_s"] = [p.get((r.trial_id, str(r.pass_id)), np.nan) for r in df.itertuples()]
    df["matched"] = np.isfinite(df["pred_m_s"])
    df["err_m_s"] = df["pred_m_s"] - df["speed_m_s"]
    return df


def ic_differences(pred_ics, ref_ics, offset_s=0.0, tol=IC_MATCH_S):
    """ref_ics with pred_t_s (nearest shifted prediction t - offset_s of the same trial and foot), diff_s
    (shifted prediction minus reference) and matched (|diff_s| <= tol); NaN and False without a prediction."""
    df = ref_ics.copy()
    by = {k: np.sort(g["t_s"].to_numpy()) - offset_s for k, g in pred_ics.groupby(["trial_id", "foot"])}
    pt = []
    for r in df.itertuples():
        t = by.get((r.trial_id, r.foot))
        pt.append(float(t[np.argmin(np.abs(t - r.t_s))]) if t is not None and len(t) else np.nan)
    df["pred_t_s"] = pt
    df["diff_s"] = df["pred_t_s"] - df["t_s"]
    df["matched"] = np.abs(df["diff_s"]) <= tol + 1e-9
    return df


def _fit(pred_ics, ref_ics, tol):
    """(offset_s, n) of fit_offset, UNFITTED when no prediction lies within tol of a reference IC; a
    reference IC of a person outside the dev controls raises ValueError."""
    dev = set(controls("dev"))
    other = sorted(set(ref_ics["person_id"]) - dev)
    if other:
        raise ValueError(f"the offset is fitted on the dev controls only; {len(other)} other persons given")
    d = ic_differences(pred_ics, ref_ics, 0.0, tol)
    d = d[d["matched"]]
    if not len(d):
        return UNFITTED
    return float(np.median(d["diff_s"])), int(len(d))


def fit_offset(pred_ics, ref_ics, tol=IC_MATCH_S):
    """The detector's constant offset (s): median of predicted minus reference over the nearest predictions
    within tol. Fitted on dev persons only: a reference IC of any other person raises ValueError, and so
    does a detector without any prediction within tol (fit_offsets records that case as not fitted).
    Returns (offset_s, n)."""
    off, n = _fit(pred_ics, ref_ics, tol)
    if not n:
        raise ValueError("no matched IC to fit an offset")
    return off, n


def offset_fitted(offset_s):
    """True for a fitted offset (a finite number of seconds), False for the NaN of UNFITTED (or None)."""
    return offset_s is not None and math.isfinite(float(offset_s))


def fit_offsets(predictions, targets, tol=IC_MATCH_S):
    """{(name, reference): (offset_s, n matched)} over the predictors with ICs (Predictions.ics not None) and
    the IC references of dev targets, each as fit_offset; a predictor with no IC within tol of any reference IC
    of a reference (a model without contacts included) gets UNFITTED there instead of an error. A reference
    IC of a person outside the dev controls raises ValueError."""
    out = {}
    for p in predictions:
        if p.ics is None:
            continue
        for ref in REFERENCES:
            out[(p.name, ref)] = _fit(p.ics, reference_ics(targets, ref), tol)
    return out


# ---------------------------------------------------------------- lines
def _person_summary(values, persons, how):
    """(ids, per-person values) with how 'mean' or 'median' (inf values propagate into a median)."""
    import pandas as pd
    s = pd.Series(np.asarray(values, dtype=np.float64)).groupby(np.asarray(persons).astype(str))
    v = s.mean() if how == "mean" else s.median()
    return v.index.to_numpy(), v.to_numpy()


def _line(values, persons, how="mean", seed=SEED, n_boot=BOOT_N):
    """Point (mean over persons of the per-person summary), interval (person bootstrap), n persons and
    the per-person table."""
    from mdfm.gates import common
    ids, v = _person_summary(values, persons, how)
    if not len(ids):
        return {"point": float("nan"), "interval": None, "n_persons": 0, "per_person": {}}
    point = float(np.mean(v))
    interval = common.cluster_bootstrap(v, ids, n=n_boot, seed=seed) if np.isfinite(v).all() else None
    return {"point": point, "interval": interval, "n_persons": int(len(ids)),
            "per_person": {str(i): float(x) for i, x in zip(ids, v)}}


def _stride_stats(m):
    """{stride_mae_cm, stride_mape_pct} stats of a match_strides table (every row, or a set's rows)."""
    k = m[m["matched"]]
    share = float(m["matched"].mean()) if len(m) else float("nan")
    unmatched = sorted(set(m["person_id"]) - set(k["person_id"]))
    out = {}
    for line, col in (("stride_mae_cm", "err_cm"), ("stride_mape_pct", "ape_pct")):
        out[line] = {**_line(np.abs(k[col]), k["person_id"]), "matched_share": share, "n_events": int(len(m)),
                     "persons_unmatched": unmatched}
    return out


def stride_lines(pred, targets):
    """stride_mae_cm and stride_mape_pct of a predictor, with the matched share."""
    m = match_strides(pred, targets)
    return _stride_stats(m), m


def _speed_stats(m):
    """speed_mae_m_s stats of a match_speeds table (every row, or a set's rows)."""
    k = m[m["matched"]]
    return {**_line(np.abs(k["err_m_s"]), k["person_id"]),
            "matched_share": float(m["matched"].mean()) if len(m) else float("nan"), "n_events": int(len(m)),
            "persons_unmatched": sorted(set(m["person_id"]) - set(k["person_id"]))}


def speed_line(pred, targets):
    """speed_mae_m_s of a predictor, with the matched share."""
    m = match_speeds(pred, targets)
    return _speed_stats(m), m


def _covered(pred, df):
    if pred.covered is None:
        return np.ones(len(df), bool)
    return np.array([(t, f) in pred.covered for t, f in zip(df["trial_id"], df["foot"])], dtype=bool)


def ic_line(pred, ref_ics, offset_s):
    """ic_median_ms of a predictor against one reference after its offset: per covered reference IC the
    absolute error in ms (infinite when unmatched), per person the median, mean over persons. An offset not
    fitted (UNFITTED's NaN, see fit_offsets) matches nothing: every covered reference IC is unmatched, the
    value is infinite and the matched share 0 (offset_fitted False, offset_ms NaN). The table keeps every
    reference IC with columns covered, matched and abs_err_ms."""
    fitted = offset_fitted(offset_s)
    d = ic_differences(pred.ics, ref_ics, float(offset_s) if fitted else 0.0)
    if not fitted:
        d["pred_t_s"], d["diff_s"], d["matched"] = np.nan, np.nan, False
    d["covered"] = _covered(pred, d)
    d["abs_err_ms"] = np.where(d["matched"], 1000.0 * np.abs(d["diff_s"]), np.inf)
    return _ic_stats(d, offset_s), d


def _ic_stats(d, offset_s):
    """ic_median_ms stats of an ic_line table (every row, or a set's rows) after the offset offset_s."""
    fitted = offset_fitted(offset_s)
    k = d[d["covered"]]
    return {**_line(k["abs_err_ms"], k["person_id"], how="median"),
            "matched_share": float(k["matched"].mean()) if len(k) else float("nan"), "n_events": int(len(k)),
            "n_uncovered": int((~d["covered"]).sum()),
            "persons_uncovered": sorted(set(d["person_id"]) - set(k["person_id"])),
            "offset_ms": 1000.0 * float(offset_s) if fitted else float("nan"), "offset_fitted": fitted}


def side_check(pred, ref_ics, offset_s=0.0, tol=IC_MATCH_S):
    """Per trial and foot of the reference ICs a predictor covers: the share of those reference ICs matched by
    the predictor's ICs of the same foot (same) and by its ICs of the other foot (other), after its offset.
    flagged: other above same, a left-right exchange of the predictor's input or of the reference labels. A
    quality check, reported; no line uses it."""
    import pandas as pd
    swap = pred.ics.copy()
    swap["foot"] = swap["foot"].map({"L": "R", "R": "L"})
    d = ic_differences(pred.ics, ref_ics, offset_s, tol)
    o = ic_differences(swap, ref_ics, offset_s, tol)
    d["other"] = o["matched"].to_numpy()
    d = d[_covered(pred, d)]
    if not len(d):
        return pd.DataFrame(columns=["trial_id", "foot", "person_id", "n", "same", "other", "flagged"])
    t = d.groupby(["trial_id", "foot"]).agg(person_id=("person_id", "first"), n=("matched", "size"),
                                            same=("matched", "mean"), other=("other", "mean")).reset_index()
    t["flagged"] = t["other"] > t["same"]
    return t


def ic_margin(model_diff, gyro_diff):
    """The margin condition on the reference ICs both detectors matched: (model value, gyro value, model <=
    IC_MARGIN x gyro), values as ic_line on that common set."""
    both = model_diff["matched"].to_numpy() & gyro_diff["matched"].to_numpy() & gyro_diff["covered"].to_numpy()
    if not both.any():
        return float("nan"), float("nan"), None
    p = model_diff["person_id"].to_numpy()[both]
    m = _person_summary(1000.0 * np.abs(model_diff["diff_s"].to_numpy()[both]), p, "median")[1].mean()
    g = _person_summary(1000.0 * np.abs(gyro_diff["diff_s"].to_numpy()[both]), p, "median")[1].mean()
    return float(m), float(g), bool(m <= IC_MARGIN * g)


def reference_ics(targets, reference):
    """The reference IC frame of a Targets: 'walkway' or 'insole' (per-trial offset removed)."""
    if reference == "walkway":
        return targets.walkway_ic
    if reference == "insole":
        return targets.insole_ic[["trial_id", "person_id", "task", "foot", "t_s"]]
    raise ValueError(f"unknown IC reference {reference!r}")


def evaluate(targets, predictions, offsets, sets=None):
    """Lines of every predictor on a Targets. predictions: list of Predictions; offsets: {(name, reference):
    offset_s} for every predictor with ICs (fitted on dev with fit_offset). Returns (summary {name: {line:
    stats}}, matched tables {name: {table: DataFrame}}). sets (opt-in, default None: the output is the one
    without it): {set name: trial ids of the targets}; every line's stats then carry "sets" {set name: the
    line's stats on the rows of those trials, with trials and n_trials} (module docstring); a trial id outside
    the targets raises ValueError."""
    sets = _check_sets(targets, sets)
    summ, tables = {}, {}
    for p in predictions:
        s, t = {}, {}
        if p.strides is not None:
            lines, t["strides"] = stride_lines(p, targets)
            s.update(lines)
        if p.speeds is not None:
            s["speed_mae_m_s"], t["speeds"] = speed_line(p, targets)
        if p.ics is not None:
            for ref in REFERENCES:
                s[f"ic_median_ms_{ref}"], t[f"ic_{ref}"] = ic_line(p, reference_ics(targets, ref),
                                                                   offsets[(p.name, ref)])
        if sets:
            for line, st in s.items():
                st["sets"] = {name: _set_stats(line, t[_table_of(line)], trials,
                                               offsets.get((p.name, _reference_of(line))))
                              for name, trials in sets.items()}
        summ[p.name], tables[p.name] = s, t
    return summ, tables


# ---------------------------------------------------------------- sets of trials (dev rows on the 14 and the 12)
def trial_sets(targets, sets=None):
    """{set name: frozenset of trial ids} of a Targets: per set (default DEV_SETS: dev14, dev12) every trial of the
    targets whose person the set does not leave out."""
    sets = DEV_SETS if sets is None else sets
    t = targets.trials
    return {str(name): frozenset(t.loc[~t["person_id"].isin(set(out)), "trial_id"]) for name, out in sets.items()}


def _check_sets(targets, sets):
    if not sets:
        return None
    known = set(targets.trials["trial_id"])
    out = {}
    for name, trials in sets.items():
        trials = frozenset(trials)
        if trials - known:
            raise ValueError(f"set {name!r}: {len(trials - known)} trial ids outside the targets")
        out[str(name)] = trials
    return out


def _table_of(line):
    """The matched table (evaluate's tables) a line is computed from."""
    if line in ("stride_mae_cm", "stride_mape_pct"):
        return "strides"
    if line == "speed_mae_m_s":
        return "speeds"
    return "ic_" + _reference_of(line)


def _reference_of(line):
    return line[len("ic_median_ms_"):] if line.startswith("ic_median_ms_") else None


def _set_stats(line, table, trials, offset_s=None):
    """The stats of `line` on the rows of `table` whose trial is in `trials` (the code of the default stats)."""
    sub = table[table["trial_id"].isin(trials)]
    if line in ("stride_mae_cm", "stride_mape_pct"):
        st = _stride_stats(sub)[line]
    elif line == "speed_mae_m_s":
        st = _speed_stats(sub)
    else:
        st = _ic_stats(sub, offset_s)
    return {**st, "trials": sorted(trials), "n_trials": len(trials)}


def set_names(summ):
    """The set names of a summary evaluated with sets (in their order), () without sets."""
    for lines in summ.values():
        for st in lines.values():
            if st.get("sets"):
                return tuple(st["sets"])
    return ()


def set_view(summ, tables, name):
    """(summary, tables) of one set of a summary evaluated with sets: each line's stats on that set, each table
    restricted to the set's trials (row order kept, so ic_margin pairs the same reference ICs). gate_results
    reads them as it reads the default ones. KeyError when a line lacks the set."""
    view, trials = {}, None
    for pred, lines in summ.items():
        view[pred] = {}
        for line, st in lines.items():
            s = (st.get("sets") or {}).get(name)
            if s is None:
                raise KeyError(f"no set {name!r} in the stats of {pred} {line} (evaluate with sets)")
            view[pred][line] = s
            trials = frozenset(s["trials"])
    tabs = {pred: {k: (df[df["trial_id"].isin(trials)] if trials is not None else df) for k, df in t.items()}
            for pred, t in tables.items()}
    return view, tabs


def gate_results_by_set(summ, tables, model, *args, **kwargs):
    """{set name: gate_results on set_view(summ, tables, name)} over the sets of a summary evaluated with sets,
    each GateResult with extra["set"] the set name; ValueError without sets."""
    names = set_names(summ)
    if not names:
        raise ValueError("the summary carries no set (evaluate with sets)")
    out = {}
    for name in names:
        res = gate_results(*set_view(summ, tables, name), model, *args, **kwargs)
        for r in res:
            r.extra["set"] = name
        out[name] = res
    return out


def matched_share_floor_ok(share, floor=MATCHED_SHARE_FLOOR):
    """The matched-share floor of rule section 6 on one share (module docstring): True when share >= floor (a
    share of exactly the floor is enough), False below it, None when the share is not a number (not evaluable; as a
    condition of common.verdict, None blocks a pass like False does)."""
    try:
        s = float(share)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(s) else bool(s >= floor)


def gate_results(summ, tables, model, rule_sha256="", checkpoint_sha256="", train_persons_sha256="",
                 baseline="population", detector="shank_gyro"):
    """GateResult lines of mdfm.gates.common: the model's point when `model` names a predictor of summ, else
    not evaluated; the population baseline (stride, speed, MAPE) and the shank-gyroscope detector (IC) as
    trivial baselines; the IC margin as a pass condition, and the matched-share floor (MATCHED_SHARE_FLOOR, module
    docstring) as a pass condition of the stride, speed and MAPE lines, read on the matched_share of the stats of
    `summ` (a set's stats through set_view) with extra floor_ok. The IC line has one row per reference: the
    walkway row (PRIMARY_REFERENCE, extra decides True) carries the verdict of the line; the insole row is
    computed the same way and reported (extra decides False, role), and its verdict decides nothing (rule
    section 6, revision 2026-10-05): a gate-level reading takes deciding(results)."""
    from mdfm.gates import common

    def get(name, line, key="point"):
        return summ.get(name, {}).get(line, {}).get(key, float("nan")) if name else float("nan")

    def iv(name, line):
        v = summ.get(name, {}).get(line, {}).get("interval") if name else None
        return tuple(v) if v else None

    def n_persons(name, line):
        n = get(model, line, "n_persons") if model else float("nan")
        n = get(name, line, "n_persons") if not (isinstance(n, int) and n > 0) else n
        return int(n) if isinstance(n, int) else 0

    out = []
    for line in FLOORED_LINES:
        pl, kl = LINES[line]
        pt = get(model, line)
        share = get(model, line, "matched_share")
        floor_ok = matched_share_floor_ok(share) if model else None
        extra = {"matched_share": share, "baseline_interval": iv(baseline, line),
                 "matched_share_floor": MATCHED_SHARE_FLOOR, "floor_ok": floor_ok}
        if line == "stride_mape_pct":
            extra["role"] = "simulator-only line (phase A model)"
        out.append(common.GateResult(GATE, line, pt, iv(model, line), pl, kl,
                                     common.verdict(pt if model else None, pl, kl, conditions=(floor_ok,)),
                                     get(baseline, line), rule_sha256,
                                     checkpoint_sha256, train_persons_sha256, n_persons(baseline, line),
                                     unit=UNITS[line], extra=extra))
    pl, kl = LINES["ic_median_ms"]
    for ref in REFERENCES:
        line = f"ic_median_ms_{ref}"
        pt = get(model, line)
        m = g = cond = None
        if model and model in tables and f"ic_{ref}" in tables[model]:
            m, g, cond = ic_margin(tables[model][f"ic_{ref}"], tables[detector][f"ic_{ref}"])
        v = common.verdict(pt if model else None, pl, kl, conditions=(cond,))
        if model and pt == float("inf"):
            v = "kill"                       # unmatched reference ICs (unfitted offset too): beyond the kill line
        if v == "undetermined" and cond is False:
            v = "kill"                       # not 20 percent better than the shank-gyroscope detector [D§9]
        fitted = summ.get(model, {}).get(line, {}).get("offset_fitted") if model else None
        extra = {"reference": ref, "primary": ref == PRIMARY_REFERENCE, "offset_fitted": fitted,
                 "matched_share": get(model, line, "matched_share"),
                 "detector_matched_share": get(detector, line, "matched_share"),
                 "detector_interval": iv(detector, line),
                 "margin_model_ms": m, "margin_detector_ms": g, "margin_ok": cond, "margin": IC_MARGIN,
                 "decides": ref == PRIMARY_REFERENCE}
        if ref != PRIMARY_REFERENCE:
            extra["role"] = "second IC reference: reported, decides nothing (rule section 6, revision 2026-10-05)"
        out.append(common.GateResult(GATE, line, pt, iv(model, line), pl, kl, v, get(detector, line),
                                     rule_sha256, checkpoint_sha256, train_persons_sha256,
                                     n_persons(detector, line), unit="ms", extra=extra))
    return out


def deciding(results):
    """The GateResult rows whose verdict decides the gate: every row but those marked decides False (the insole IC
    row)."""
    return [r for r in results if r.extra.get("decides", True)]


# ---------------------------------------------------------------- readout self-check (AMASS dev walks)
def amass_dev_walks(n=N_WALKS, rate=100.0):
    """The first n forward walks (task walk, a style of WALK_STYLES in the take id) of the AMASS dev body
    store, sorted by take id: dicts take, i, traj (BodyTrajectory, float64, the take's gendered rest joints),
    J (T, 22, 3) m at the knots, contact (T, 2) stored labels, speed (m/s, factory.gait.speed_and_cadence),
    gyro {"L": (T - 1, 3), "R": (T - 1, 3)} deg/s rendered at shank_L and shank_R without device noise,
    store. FileNotFoundError when the dev body store is not built."""
    import torch

    from mdfm.body.smplh import SkeletonBank
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data import body_store
    from mdfm.factory import imu_render
    from mdfm.factory.gait import speed_and_cadence
    if not os.path.isdir(os.path.join(body_store.store_root(), "dev")):
        raise FileNotFoundError("dev body store not built (scripts/build_body_store.py)")
    store = body_store.BodyStore(split="dev")
    tk = store.takes
    sel = [i for i in tk.sort_values("take_id", kind="stable").index
           if tk.at[i, "task_name"] == "walk" and any(x in tk.at[i, "take_id"] for x in WALK_STYLES)][:n]
    if len(sel) < n:
        raise ValueError(f"{len(sel)} forward walks in the dev body store, {n} asked")
    f64 = torch.float64
    bank = SkeletonBank(f64)
    out = []
    for i in sel:
        a = store.take(i)
        r6 = torch.from_numpy(np.asarray(a["rot6d"], np.float64))
        rest = bank.rest(torch.from_numpy(store.betas[i:i + 1]), torch.from_numpy(store.gender[i:i + 1]))[0]
        traj = BodyTrajectory(None, r6, torch.from_numpy(a["transl"]), None, rate=rate, rest=rest)
        t = torch.arange(r6.shape[0], dtype=f64) / rate
        with torch.no_grad():
            J = traj.pose_at(t)[0].numpy()
            sig, _, _ = imu_render.render(traj, t[:-1], sites=["shank_L", "shank_R"])
        speed, _ = speed_and_cadence(traj)
        gy = sig.numpy().astype(np.float64)
        out.append({"i": int(i), "take": tk.at[i, "take_id"], "traj": traj, "J": J,
                    "contact": np.asarray(a["contact"], np.float64), "speed": float(speed),
                    "gyro": {f: gy[:, s, GYRO] for f, s in SHANK_SITES.items()}, "store": store})
    return out


# the forward walks of the open body store (step M4 of docs/MODEL_OPEN_PORT_PLAN.md), by source and take key: the rule
# of scripts/g2_open_render.WALK_RULES (tests/test_open_g3_selfcheck.py asserts the two equal)
OPEN_WALK_RULES = {
    "ceti_age_kinematics": r"_w0[12]_",
    "stroke_gait_vancriekinge": r"^gait_",
    "gait120": r"^level_walking_trial\d+$",
    "lencioni_gait_stairs": r"^walking_\d+$",
    "style100": r"^Neutral_FW$",
}


OPEN_DRAW_SEED = "20260928:g3_open_selfcheck:"
OPEN_STRAIGHT_SHARE = 0.5      # a drawn walk's share of frames on a straight path (G2 mask), at least


def _seeded_rng(source):
    import hashlib
    return np.random.default_rng(int(hashlib.sha256((OPEN_DRAW_SEED + source).encode("utf-8")).hexdigest()[:16], 16))


def open_walk_draw(store, n=N_WALKS, rate=100.0):
    """Row indices of the seeded draw of the open self-check walks (draw='seeded' of open_dev_walks): one walk per
    person; per source (sorted) its persons in a permutation of default_rng(SHA-256 of OPEN_DRAW_SEED + source), the
    sources taken in turn until n walks; a person's walk the first of its forward walks (OPEN_WALK_RULES, at least 2 s,
    in a permutation of the same generator) whose share of straight frames (mdfm/readouts/straight.py, the G2 mask)
    is at least OPEN_STRAIGHT_SHARE; a person without one is passed over."""
    import re
    from mdfm.body.joints import fk
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.readouts.straight import straight_frames
    tk = store.takes
    walks = [i for i in range(len(tk)) if tk.at[i, "source"] in OPEN_WALK_RULES
             and re.search(OPEN_WALK_RULES[tk.at[i, "source"]], tk.at[i, "key"]) and int(tk.at[i, "n_knots"]) >= int(2 * rate)]
    queues = {}
    for src in sorted({tk.at[i, "source"] for i in walks}):
        rng = _seeded_rng(src)
        rows = [i for i in walks if tk.at[i, "source"] == src]
        persons = sorted({tk.at[i, "person_id"] for i in rows})
        q = []
        for j in rng.permutation(len(persons)):
            mine = sorted(i for i in rows if tk.at[i, "person_id"] == persons[j])
            q.append([mine[k] for k in rng.permutation(len(mine))])
        queues[src] = q

    def straight(i):
        import torch
        a = store.take(i)
        with torch.no_grad():
            pos, _ = fk(rot6d_to_matrix(torch.as_tensor(np.asarray(a["rot6d"], np.float64))),
                        torch.as_tensor(a["transl"]), torch.as_tensor(store.rest_joints[i], dtype=torch.float64))
        return float(straight_frames(pos[:, 0].numpy()[:, [0, 2]], rate).mean())

    out = []
    while len(out) < n and any(queues.values()):
        for src in sorted(queues):
            if not queues[src] or len(out) >= n:
                continue
            for i in queues[src].pop(0):
                if straight(i) >= OPEN_STRAIGHT_SHARE:
                    out.append(int(i))
                    break
    return out


def open_dev_walks(n=N_WALKS, rate=100.0, store=None, draw="seeded"):
    """The open counterpart of amass_dev_walks (step M4): n forward walks (OPEN_WALK_RULES) of the open body store's
    dev folds (mdfm/data/open_body_store.OpenTrainStore(split='dev'), the takes over the residual bound refused),
    each at least 2 s long, drawn by open_walk_draw (draw 'seeded', the default since 2026-11-14) or, draw 'first',
    the first n by take id (the first reading, reports/gates/g3/dev_dry_run_open/, where all 20 were ceti turning
    walks); the rest joints of the take's phenotype (P3 basis); the same dicts as amass_dev_walks. FileNotFoundError
    when the dev store is not built (scripts/build_open_body_store.py --split dev)."""
    import re
    import torch

    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data import open_body_store as OS
    from mdfm.factory import imu_render
    from mdfm.factory.gait import speed_and_cadence
    if store is None:
        root = OS.store_root(None, "dev")
        if not os.path.exists(os.path.join(root, "hashes.json")):
            raise FileNotFoundError(f"open dev body store not built at {root} (scripts/build_open_body_store.py "
                                    "--split dev)")
        store = OS.OpenTrainStore(split="dev")
    tk = store.takes
    if draw == "seeded":
        sel = open_walk_draw(store, n, rate)
    elif draw == "first":
        sel = [i for i in tk.sort_values("take_id", kind="stable").index
               if tk.at[i, "source"] in OPEN_WALK_RULES and re.search(OPEN_WALK_RULES[tk.at[i, "source"]],
                                                                    tk.at[i, "key"])
               and int(tk.at[i, "n_knots"]) >= int(2 * rate)][:n]
    else:
        raise ValueError(f"draw {draw!r}, expected 'seeded' or 'first'")
    if len(sel) < n:
        raise ValueError(f"{len(sel)} forward walks in the open dev body store, {n} asked")
    f64 = torch.float64
    out = []
    for i in sel:
        a = store.take(i)
        r6 = torch.from_numpy(np.asarray(a["rot6d"], np.float64))
        rest = torch.as_tensor(store.rest_joints[i], dtype=f64)
        traj = BodyTrajectory(None, r6, torch.from_numpy(a["transl"]), None, rate=rate, rest=rest)
        t = torch.arange(r6.shape[0], dtype=f64) / rate
        with torch.no_grad():
            J = traj.pose_at(t)[0].numpy()
            sig, _, _ = imu_render.render(traj, t[:-1], sites=["shank_L", "shank_R"])
        speed, _ = speed_and_cadence(traj)
        gy = sig.numpy().astype(np.float64)
        out.append({"i": int(i), "take": tk.at[i, "take_id"], "traj": traj, "J": J,
                    "contact": np.asarray(a["contact"], np.float64), "speed": float(speed),
                    "gyro": {f: gy[:, s, GYRO] for f, s in SHANK_SITES.items()}, "store": store})
    return out


def steady_straight(walk, t, rate=100.0):
    """True when the step at time t (s) of a walk (amass_dev_walks) is steady and straight (module
    docstring)."""
    J = walk["J"]
    if "_pelvis_speed" not in walk:
        v = np.gradient(J[:, gait.PELVIS][:, [0, 2]], 1.0 / rate, axis=0)
        walk["_pelvis_speed"] = np.linalg.norm(v, axis=1)
    f, h = int(round(t * rate)), int(round(0.5 * rate))
    p = J[max(0, f - h):min(len(J) - 1, f + h) + 1, gait.PELVIS][:, [0, 2]]
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    return bool(walk["_pelvis_speed"][f] >= STEADY * walk["speed"] and np.linalg.norm(p[-1] - p[0]) >= STRAIGHT * path)


def detector_agreement(walks, reference, tol=AGREE_S, rate=100.0):
    """(share, offset_s, n) of the shank-gyroscope detector over the steady straight reference events of the
    walks (reference: walk -> {"L": times, "R": times} s): the share with a detection of the same foot within
    tol s after one pooled offset (median of detection minus reference over the nearest detections)."""
    diffs = []
    for w in walks:
        ref = reference(w)
        for foot in gait.FEET:
            det = gait.shank_gyro_ic(w["gyro"][foot], rate)
            for t in ref[foot]:
                if steady_straight(w, t, rate):
                    diffs.append(det[np.argmin(np.abs(det - t))] - t if len(det) else np.inf)
    diffs = np.asarray(diffs, dtype=np.float64)
    off = float(np.median(diffs[np.isfinite(diffs)]))
    return float(np.mean(np.abs(diffs - off) <= tol + 1e-9)), off, int(len(diffs))


def readout_check(walks, rate=100.0):
    """Readout self-check on AMASS dev walks (module docstring). Returns (DataFrame per take: take,
    n_strides, speed_readout_m_s, speed_factory_m_s, speed_abs_diff_m_s, contact_lag_ms (median over the
    take's steady straight contact ICs of the IC minus the nearest heel strike of the same foot), n_lag;
    summary dict)."""
    import pandas as pd
    rows = []
    for w in walks:
        ic = gait.initial_contacts(w["contact"], rate)
        st = gait.strides(w["J"], ic, rate=rate)
        short = st[st["t1"] - st["t0"] <= 2.0]
        v = float(np.median([gait.pass_speed(w["J"], r.t0, r.t1, rate=rate) for r in short.itertuples()]))
        hs = gait.heel_strikes(w["J"], rate)
        lag = [t - hs[foot][np.argmin(np.abs(hs[foot] - t))] for foot in gait.FEET for t in ic[foot]
               if len(hs[foot]) and steady_straight(w, t, rate)]
        rows.append((w["take"], len(st), v, w["speed"], abs(v - w["speed"]),
                     1000.0 * float(np.median(lag)) if lag else float("nan"), len(lag)))
    df = pd.DataFrame(rows, columns=["take", "n_strides", "speed_readout_m_s", "speed_factory_m_s",
                                     "speed_abs_diff_m_s", "contact_lag_ms", "n_lag"])
    agree = {}
    for key, ref in (("heel_strike", lambda w: gait.heel_strikes(w["J"], rate)),
                     ("contact_ic", lambda w: gait.initial_contacts(w["contact"], rate))):
        share, off, n = detector_agreement(walks, ref, AGREE_S, rate)
        agree[key] = {"share_within": share, "offset_ms": 1000.0 * off, "n_steps": n}
    d = df["speed_abs_diff_m_s"].to_numpy()
    lag = df["contact_lag_ms"].dropna().to_numpy()
    summary = {"n_walks": int(len(df)), "agree_tol_ms": 1000.0 * AGREE_S,
               "speed_abs_diff_median_m_s": float(np.median(d)), "speed_abs_diff_max_m_s": float(d.max()),
               "speed_within_0_05": int((d <= 0.05).sum()), "shank_gyro_agreement": agree,
               "contact_lag_ms": {"takes": int(len(lag)), "min": float(lag.min()) if len(lag) else float("nan"),
                                  "median": float(np.median(lag)) if len(lag) else float("nan"),
                                  "max": float(lag.max()) if len(lag) else float("nan")}}
    return df, summary


# ---------------------------------------------------------------- runs
def shank_gyros(persons, trials, allow_lockbox=False, read_row=None):
    """{trial_id: shank_gyro(record)} of the persons' records."""
    return {tid: shank_gyro(rec) for tid, rec in imu_records(persons, trials, allow_lockbox, read_row)}


def dev_run(model_bodies=None, sets=None, validity=False):
    """The dev evaluation on the 14 dev controls: the walkway as a predictor (reference floor), the
    population baseline (training fold), the shank detector and, when model_bodies (callable: trial_id,
    model_input(record) -> TrialBody) is given, the model; offsets fitted on the same dev persons
    (fit_offsets: a predictor with no IC within IC_MATCH_S of a reference IC, a model without contacts
    included, is recorded as UNFITTED and its IC line reads inf). Returns (summary, tables, offsets
    {(name, reference): (offset_s, n matched)}, dev Targets, train Targets, baseline, the list of
    Predictions).
    Opt-in, both off by default (the output is then the one without them): sets=True adds the rows of DEV_SETS
    (dev14 and dev12) to every line (evaluate sets; a mapping {set name: persons left out} gives other sets);
    validity=True checks every dev record (input_validity), keeps the table in the dev Targets' validity and adds
    the set valid_inputs (the trials of valid records) and dev14. The offsets stay fitted on the 14."""
    dev, train = controls("dev"), controls("train")
    if set(dev) & set(train):
        raise ValueError("dev and training fold overlap")
    tg_dev = load_targets(dev)
    tg_train = load_targets(train)
    base = population_baseline(tg_train)
    preds = [walkway_predictions(tg_dev), population_predictions(base, tg_dev)]
    gyros, bodies, checks = {}, [], []
    info = tg_dev.trials.set_index("trial_id") if validity else None
    for tid, rec in imu_records(dev, tg_dev.trials):
        gyros[tid] = shank_gyro(rec)
        if validity:
            checks.append(validity_row(tid, info.at[tid, "person_id"], info.at[tid, "task"], rec))
        if model_bodies is not None:
            bodies.append(model_bodies(tid, model_input(rec)))
    preds.append(shank_predictions(gyros))
    if model_bodies is not None:
        preds.append(body_predictions(bodies, tg_dev))
    offsets = fit_offsets(preds, tg_dev)
    defs = None
    if sets or validity:
        defs = trial_sets(tg_dev, DEV_SETS if sets is True else (sets or {SET_14: ()}))
        if validity:
            tg_dev.validity = validity_frame(checks)
            defs.setdefault(SET_14, frozenset(tg_dev.trials["trial_id"]))
            defs[SET_VALID] = valid_trials(tg_dev.validity)
    summ, tables = evaluate(tg_dev, preds, {k: v[0] for k, v in offsets.items()}, sets=defs)
    return summ, tables, offsets, tg_dev, tg_train, base, preds


def _preflight(common, checkpoint_sha256, rule_sha, model_bodies, reread, reason, code_files):
    """The checks of read_lockbox that need no lockbox row or record: the arguments, the read log state (the
    refusals of common.log_lockbox_read, raised here before the dev run and again there), the target files
    and their rule version, the lockbox controls and the code hash. Returns (lockbox controls, code
    SHA-256)."""
    if not HEX64.fullmatch(str(checkpoint_sha256)):
        raise ValueError(f"checkpoint_sha256 must be a lowercase hex SHA-256, got {checkpoint_sha256!r}")
    if not callable(model_bodies):
        raise TypeError("model_bodies must be a callable (trial_id, model_input(record)) -> TrialBody")
    prior = [r for r in common.read_log() if (r["gate"], r["checkpoint_sha256"], r["rule_sha256"])
             == (GATE, checkpoint_sha256, rule_sha)]
    if prior and not reread:
        raise Refused(f"{GATE} was read on the lockbox with checkpoint {checkpoint_sha256[:12]} and rule "
                      f"{rule_sha[:12]} on {prior[0]['read_at']}: a second read needs reread and a reason")
    if reread and not prior:
        raise ValueError("reread asked for a read that was never logged")
    if reread and not str(reason).strip():
        raise Refused("a lockbox reread needs a logged reason")
    missing = [common.rel(p) for p in TARGET_FILES + (SUMMARY_JSON,) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"G3 targets missing: {missing} (scripts/build_weargait_targets.py)")
    _, version = target_meta()
    if version != TARGET_RULE:
        raise ValueError(f"targets rule {version}, the harness reads {TARGET_RULE}")
    persons = lockbox_controls()
    if not persons:
        raise ValueError(f"no {SOURCE} lockbox control in reports/lockbox/lockbox_persons.csv")
    if any(not p.startswith(PREFIX) for p in persons):
        raise ValueError(f"a G3 lockbox control outside {SOURCE}")
    folds = set(controls("dev")) | set(controls("train"))
    if set(persons) & folds:
        raise ValueError(f"{len(set(persons) & folds)} lockbox controls sit in the dev or training fold")
    return persons, common.code_sha256([__file__, gait.__file__, *code_files])


class LockboxRead(tuple):
    """What read_lockbox returns: the 3-tuple (GateResult list, summary, tables), so that existing callers keep
    unpacking it, with the input-validity outputs of the read as attributes: validity (the per-trial validity table
    of every record the read opened, VALIDITY_COLUMNS: the validity.csv of rule section 9), report (validity_report:
    the invalid trials and their reasons, trials without a record, persons without a valid trial) and sets (the
    two sets scored, {SET_VALID: trial ids, SET_ALL: trial ids})."""

    def __new__(cls, results, summary, tables, validity, report, sets):
        self = super().__new__(cls, (results, summary, tables))
        self.validity, self.report, self.sets = validity, report, sets
        return self


def read_lockbox(checkpoint_sha256, model_bodies, reread=False, reason="", code_files=()):
    """The sealed G3 read (plan task B4.21). Refused unless docs/rules/G3_RULE.md is registered for G3
    (common.rule, checked first). Then every check that can fail without lockbox data runs before the read is
    logged: _preflight (arguments, read log state, targets, lockbox controls, code hash), dev_run with the
    same model (the model on every dev trial, the offsets fitted on the dev controls, an unfitted one
    recorded as UNFITTED, never an error), gate_results on the dev lines, the training-person hash and the rule
    again (a rule file changed during the dev run is refused here, not by the readers after the log). Only
    then is the read logged once for this checkpoint and rule (common.log_lockbox_read) and are the lockbox
    rows and records read; what can still fail after the log depends on them (a lockbox target row or record
    that fails its checks, the model or a readout on a lockbox trial). The model receives
    model_input(record).
    Input validity (rule section 2, "Input validity (blind)"; G3_RULE.md section 10, item 5): inside the logged
    read, each lockbox record the read opens is checked by validity_row (input_validity: the record only) before
    the model sees it, hence before any line is computed. The lockbox set stays the 17 controls; the offsets stay
    those fitted on the 14 dev controls. The primary rows score the trials of valid records (SET_VALID), the
    all-trials rows every trial (SET_ALL), both through evaluate with sets and gate_results_by_set (so with the
    matched-share floor); the all-trials rows are marked extra decides False (reported beside the primary rows,
    deciding(results) leaves them out) and every row carries extra input_validity (validity_report). A record that
    fails the check is not refused: its trial is left out of the primary row, counted and listed with its reasons.
    Returns a LockboxRead: (GateResult list, summary, tables) with the primary rows first and the all-trials rows
    after them, plus validity, report and sets as attributes. The summary's default stats (no set) are those of
    every trial, as in dev_run(validity=True); the primary stats are stats['sets'][SET_VALID]."""
    from mdfm.gates import common
    _, rule_sha = common.rule(RULE_PATH, gate=GATE)
    persons, code_sha = _preflight(common, checkpoint_sha256, rule_sha, model_bodies, reread, reason, code_files)
    summ_dev, tables_dev, offsets, _, tg_train, base, _ = dev_run(model_bodies)
    train_sha = sources.person_hash(tg_train.persons)
    gate_results(summ_dev, tables_dev, "model", rule_sha, checkpoint_sha256, train_sha)
    if common.rule(RULE_PATH, gate=GATE)[1] != rule_sha:         # the readers check it again after the log
        raise Refused("docs/rules/G3_RULE.md changed during the dev run: no lockbox read")
    row = common.log_lockbox_read(GATE, checkpoint_sha256, rule_sha, reread=reread, reason=reason,
                                  code_sha256=code_sha)
    tg = load_targets(persons, allow_lockbox=True, read_row=row)
    info = tg.trials.set_index("trial_id")
    gyros, bodies, checks = {}, [], []
    for tid, rec in imu_records(persons, tg.trials, allow_lockbox=True, read_row=row):
        checks.append(validity_row(tid, info.at[tid, "person_id"], info.at[tid, "task"], rec))   # the record only
        gyros[tid] = shank_gyro(rec)
        bodies.append(model_bodies(tid, model_input(rec)))
    tg.validity = validity_frame(checks)
    sets = {SET_VALID: valid_trials(tg.validity), SET_ALL: frozenset(tg.trials["trial_id"])}
    report = validity_report(tg.validity, tg.trials)
    preds = [population_predictions(base, tg), shank_predictions(gyros), body_predictions(bodies, tg)]
    summ, tables = evaluate(tg, preds, {k: v[0] for k, v in offsets.items()}, sets=sets)
    by_set = gate_results_by_set(summ, tables, "model", rule_sha, checkpoint_sha256, train_sha)
    res = []
    for name in (SET_VALID, SET_ALL):
        for r in by_set[name]:
            r.extra["input_validity"] = copy.deepcopy(report)
            if name == SET_ALL:
                r.extra["decides"] = False
                r.extra["set_role"] = ("all-trials row: reported beside the primary row, decides nothing "
                                       "(rule section 2, input validity)")
            res.append(r)
    return LockboxRead(res, summ, tables, tg.validity, report, sets)

