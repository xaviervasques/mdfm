"""Walkway-anchor targets of F2 lever 4 (docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md, sections A to C; task L4C):
the anchor table built once from the walkway targets of the weargait_pd training-fold controls, its checks
(assert_anchor_clean) and the per-micro-step targets (AnchorTargets) that train/loop.py hands to
train/losses.walkway_anchor_parts.

Persons. The anchored persons are the window store's weargait_pd training fold (train54 on the pilot's store), whose
SHA-256 (sources.person_hash) must equal the config's walkway_anchor.persons_sha256. Before any target file opens,
check_persons refuses a person outside that fold, a person the window store holds out (dev), a person of the lockbox
list (reports/lockbox/lockbox_persons.csv, hash-checked, its person_id column), a lockbox, EVAL, CODY or EXCLUDED_V1
person or the canary of the deny lists every trained window passes (train/provenance.problems), and a person of the
proposed PD folds of the reviewed stage-2 sources (PD_FOLDS_CSV: its source and person_id columns only).
Every reader then streams the CSV rows and drops the rows of other persons first, before any other value of the row
is used, and asserts lockbox False on each kept row.

Values (section B, item 1: a column whitelist). From reports/weargait_targets/events.csv only ANCHOR_COLUMNS reach the
table (project): never the group, phase, state or direction columns. The heel positions are in walkway cells; the
cell size is refit with the rule of scripts/build_weargait_targets.py (the median over SelfPace trials with a finite
PKMAS stride length and a positive walkway stride of their ratio, cm per cell) on the training-fold rows of
reports/weargait_targets/trials.csv, through CELL_COLUMNS only; the cell of summary.json (fitted with dev and PD
persons) is never read. The table records the fit's (PKMAS cm, walkway cells) pairs and the SHA-256 of the fit
persons, so that assert_anchor_clean can recompute it.

Clock (section B, item 5). The walkway timeline and the window clock are one clock (MoveAtlas builds the IMU record
from the same per-trial CSV). A recording is left out when its store clock is not the targets' (span_t0 0, record
offset 0, every sample kept at 100 Hz: clock_ok), and a trial is left out when its shank-gyroscope contacts sit more
than CLOCK_MAX_MS from the pooled offset of the probe's alignment check (reports/walkway_anchor/alignment_trials.csv,
scripts/probe_walkway_anchor.py: column away_from_pooled_ms of the training-fold trials; a trial without a finite
value is left out too).

Segments (section A; the probe's segment_of). A pass is the IC rows of one pass_id of a trial; a footfall is full when
it is not partial and has a heel position. For a window [t0, t0 + dur] on the recording clock, each pass gives at most
one segment: per foot, its first and last full footfall inside the window; the foot with the longer interval (ties:
more strides, then L) holds the segment (t_a, t_b), its distance |heel(t_b) - heel(t_a)| x cell (m) and its strides
(same-foot footfalls minus one, at least 1). A window holds a segment of each pass that has one.

Targets (AnchorTargets): one row per segment of the micro-step's windows, keyed by the window's index in the
micro-step (never a person, a trial or a recording): window (int64), k_a and k_b (float32, the segment's ends as
fractional knots of the window clock, rate x (t - t0)), dist_m (float32) and n_strides (int64). targets_for_step reads
the sampler's Step: its plan items (the recording index of each real window) and its windows (source, key, t0,
duration) in item order. Synthetic windows, other sources, other tasks and windows without a segment carry no row.

assert_anchor_clean(table, train, refused, persons_sha256) refuses (AnchorRefused, a canonical.Refused): a person of
the table outside `train`, a person of any refused set, a persons SHA-256 other than the given one, a row (a Pass)
that carries a person field, a cell fit with a person outside `train`, a cell size that differs from the refit of its
recorded pairs by more than CELL_TOL_CM, and label keys (train/losses.label_paths) in the rows.

Input pin (task FIXL4, the review of L4C). When the environment variable PIN_ENV names a JSON file (absolute or
tree-relative) whose 'inputs' maps tree-relative paths to SHA-256 values, build_anchor_table hashes its four input
files before it reads a value from them and raises train/provenance.ProvenanceError unless they equal the pin (an
unreadable pin raises too), so that a run whose rule fixes the inputs fails at launch rather than after training.
Unset (the default), nothing is checked and nothing changes.
"""
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields

import numpy as np
import torch

from .. import paths
from .canonical import Refused, is_cody

SOURCE = "weargait_pd"
RATE = 100.0
CELL_TASK = "SelfPace"                       # the builder's cell-size fit task (scripts/build_weargait_targets.py)
CLOCK_MAX_MS = 50.0                          # a trial's shank-gyroscope offset away from the pooled one, at most
CELL_TOL_CM = 1e-9                           # refit against recorded cell size
ANCHOR_COLUMNS = ("trial_id", "person_id", "task", "lockbox", "foot", "event", "walkway_t_s", "pass_id",
                  "heel_x_cells", "partial")
CELL_COLUMNS = ("trial_id", "person_id", "task", "lockbox", "pkmas_stride_length_cm", "ww_stride_cells_mean")
ALIGN_COLUMNS = ("trial_id", "person_id", "task", "matched", "away_from_pooled_ms")
PD_FOLD_COLUMNS = ("source", "person_id")
TARGETS_DIR = os.path.join(paths.TREE, "reports", "weargait_targets")
EVENTS_CSV = os.path.join(TARGETS_DIR, "events.csv")
TRIALS_CSV = os.path.join(TARGETS_DIR, "trials.csv")
ALIGNMENT_CSV = os.path.join(paths.TREE, "reports", "walkway_anchor", "alignment_trials.csv")
PD_FOLDS_CSV = os.path.join(paths.TREE, "reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06",
                            "folds_proposed.csv")
FORMAT = "mdfm-walkway-anchor/1"
PIN_ENV = "MDFM_WALKWAY_ANCHOR_PIN"          # a JSON file {"inputs": {tree-relative path: SHA-256}} (module docstring)


class AnchorRefused(Refused):
    """A person, a value or a table the walkway anchor may not use (module docstring)."""


def person_hash(ids):
    """SHA-256 of the sorted ids joined by newlines (data.sources.person_hash)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), paths.TREE).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def trial_id(person_id, key):
    """The targets' trial id of a recording: the person id after its namespace, '_', the recording key
    (weargait:HC100, SelfPace -> HC100_SelfPace)."""
    return f"{str(person_id).split(':', 1)[-1]}_{key}"


# ---------------------------------------------------------------- passes and segments
@dataclass(frozen=True)
class Pass:
    """One straight walkway pass of a trial: pass_id; t (n,) IC times (s, recording clock), foot (n,) 'L' or 'R',
    heel_m (n,) heel position along the walkway (m, NaN when missing), full (n,) bool (not partial, heel present);
    sorted by time. No person field (module docstring)."""
    pass_id: str
    t: np.ndarray
    foot: np.ndarray
    heel_m: np.ndarray
    full: np.ndarray


@dataclass(frozen=True)
class Segment:
    """A segment of a pass inside a window: t_a, t_b (s, recording clock), dist_m, n_strides, foot."""
    t_a: float
    t_b: float
    dist_m: float
    n_strides: int
    foot: str

    @property
    def dur_s(self):
        return self.t_b - self.t_a


def project(row):
    """The anchor's view of an events.csv row: ANCHOR_COLUMNS only (section B, item 1)."""
    return {k: row[k] for k in ANCHOR_COLUMNS}


def passes_of(rows, cell_cm):
    """{trial_id: (Pass, ...)} from projected events.csv rows: the IC rows with a pass_id, heel cells in metres with
    cell_cm."""
    groups = {}
    for r in rows:
        if r["event"] != "IC" or r["pass_id"] == "":
            continue
        groups.setdefault((r["trial_id"], r["pass_id"]), []).append(r)
    out = {}
    for (tid, pid), P in sorted(groups.items()):
        P.sort(key=lambda r: (float(r["walkway_t_s"]), r["foot"]))
        t = np.array([float(r["walkway_t_s"]) for r in P], dtype=np.float64)
        foot = np.array([r["foot"] for r in P])
        heel = np.array([float(r["heel_x_cells"]) * float(cell_cm) / 100.0 if r["heel_x_cells"] != "" else np.nan
                         for r in P], dtype=np.float64)
        full = np.array([r["partial"] == "False" and r["heel_x_cells"] != "" for r in P], dtype=bool)
        out.setdefault(tid, []).append(Pass(str(pid), t, foot, heel, full))
    return {k: tuple(v) for k, v in out.items()}


def segment_of(p, a, b):
    """Segment of pass p inside [a, b] (module docstring), or None when no foot holds two full footfalls there."""
    inside = p.full & (p.t >= a - 1e-9) & (p.t <= b + 1e-9)
    best = None
    for foot in ("L", "R"):
        idx = np.flatnonzero(inside & (p.foot == foot))
        if len(idx) < 2:
            continue
        i, j = int(idx[0]), int(idx[-1])
        cand = (p.t[j] - p.t[i], len(idx) - 1, foot == "L", foot, i, j)
        if best is None or cand[:3] > best[:3]:
            best = cand
    if best is None:
        return None
    _, n_str, _, foot, i, j = best
    return Segment(float(p.t[i]), float(p.t[j]), float(abs(p.heel_m[j] - p.heel_m[i])), int(n_str), str(foot))


def window_segments(passes, t0, dur):
    """Segments of the passes (a trial's Pass tuple) inside the window [t0, t0 + dur]."""
    out = []
    for p in passes:
        s = segment_of(p, float(t0), float(t0) + float(dur))
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------- persons
def pd_fold_persons(path=PD_FOLDS_CSV):
    """The weargait_pd persons of the proposed PD folds (person_id of the rows of SOURCE; PD_FOLD_COLUMNS only)."""
    out = set()
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE:
                out.add(r["person_id"])
    return frozenset(out)


def lockbox_persons():
    """Every person id of reports/lockbox/lockbox_persons.csv (its person_id column), the file's SHA-256 checked
    against reports/lockbox/lockbox.sha256 first (data.amass_index)."""
    from . import amass_index
    with open(amass_index.LOCKBOX_SHA_FILE, encoding="utf-8") as f:
        want = f.read().split()[0]
    if amass_index.sha256_file(amass_index.LOCKBOX_CSV) != want:
        raise RuntimeError(f"{amass_index.LOCKBOX_CSV} differs from {amass_index.LOCKBOX_SHA_FILE}")
    with open(amass_index.LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        return frozenset(r["person_id"] for r in csv.DictReader(f))


def refused_persons(window_store=None, pd_folds=PD_FOLDS_CSV):
    """{name: frozenset of person ids} the anchor refuses besides the deny lists: 'held_out' (the window store's
    persons of SOURCE outside the training fold: dev), 'lockbox' (lockbox_persons) and 'pd_folds' (the proposed PD
    folds)."""
    out = {"pd_folds": pd_fold_persons(pd_folds) if pd_folds else frozenset(), "lockbox": lockbox_persons()}
    held = set()
    if window_store is not None:
        r = window_store.records
        held = set(r.loc[(r["source"] == SOURCE) & (r["fold"] != "train"), "person_id"])
    out["held_out"] = frozenset(held)
    return out


def check_persons(persons, train, refused, persons_sha256=None, deny=True):
    """AnchorRefused unless every person lies in `train`, none in a refused set, none flagged by the deny lists of
    train/provenance (lockbox, EVAL, CODY, EXCLUDED_V1, canary; deny=False skips them, for a table already checked)
    and, given persons_sha256, the SHA-256 of `train` equals it. Returns the sorted persons."""
    persons = sorted(set(persons))
    train = set(train)
    out = [p for p in persons if p not in train]
    if out:
        raise AnchorRefused(f"walkway anchor: {len(out)} persons outside the training fold ({out[:3]})")
    for name, s in sorted((refused or {}).items()):
        bad = sorted(set(persons) & set(s))
        if bad:
            raise AnchorRefused(f"walkway anchor: {len(bad)} {name} persons ({bad[:3]})")
    cody = [p for p in persons if is_cody(p)]
    if cody:
        raise AnchorRefused(f"walkway anchor: {len(cody)} CODY persons (evaluation-only)")
    if persons_sha256 is not None and person_hash(train) != str(persons_sha256):
        raise AnchorRefused(f"walkway anchor: the training fold's SHA-256 {person_hash(train)[:12]} is not the "
                            f"configuration's {str(persons_sha256)[:12]}")
    if deny:
        from ..train import provenance as PV
        bad = PV.problems({SOURCE: persons})
        if bad:
            raise AnchorRefused(f"walkway anchor: {len(bad)} persons on a deny list ({bad[:3]})")
    return persons


def read_rows(path, persons, columns, refused=None):
    """Rows (dicts of the given columns) of a CSV whose person_id is in `persons`; the persons are checked against
    `refused` before the file opens, and each row's person_id is read first, the row dropped for another person
    before any other value is used; every kept row must read lockbox False when the file has the column."""
    persons = set(persons)
    for name, s in sorted((refused or {}).items()):
        bad = sorted(persons & set(s))
        if bad:
            raise AnchorRefused(f"{os.path.basename(path)}: {name} persons requested ({bad[:3]})")
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        missing = [c for c in columns if c not in (rd.fieldnames or ())]
        if missing:
            raise ValueError(f"{os.path.basename(path)}: columns {missing} absent")
        for row in rd:
            if row["person_id"] not in persons:
                continue
            if "lockbox" in row and row["lockbox"] != "False":
                raise AnchorRefused(f"{os.path.basename(path)}: a row of {row['person_id']} reads lockbox "
                                    f"{row['lockbox']!r}")
            out.append({c: row[c] for c in columns})
    return out


# ---------------------------------------------------------------- cell size
def cell_fit(trial_rows, task=CELL_TASK):
    """The walkway cell size refit on trials.csv rows (CELL_COLUMNS; module docstring): {cell_cm, n, pairs ((PKMAS
    stride cm, walkway stride cells), ...), persons}. cell_cm NaN without a usable trial."""
    pairs, persons = [], set()
    for r in trial_rows:
        if r["task"] != task:
            continue
        a, b = _num(r["pkmas_stride_length_cm"]), _num(r["ww_stride_cells_mean"])
        if math.isfinite(a) and math.isfinite(b) and b > 0:
            pairs.append((a, b))
            persons.add(r["person_id"])
    if not pairs:
        return {"cell_cm": float("nan"), "n": 0, "pairs": (), "persons": ()}
    return {"cell_cm": cell_of_pairs(pairs), "n": len(pairs), "pairs": tuple(pairs), "persons": tuple(sorted(persons))}


def cell_of_pairs(pairs):
    """The median of PKMAS cm over walkway cells of (cm, cells) pairs (float64)."""
    x = np.asarray([a / b for a, b in pairs], dtype=np.float64)
    return float(np.median(x))


# ---------------------------------------------------------------- the table
@dataclass
class AnchorTable:
    """The anchor's source table (module docstring): passes {recording index of the window store: (Pass, ...)} and
    keys {recording index: recording key} of the anchored recordings; tasks; persons (sorted ids of the persons whose
    trials were read, for the checks; never in a row) and persons_sha256 (of the training fold); the cell fit
    (cell_cm, cell_n, cell_pairs, cell_persons); excluded {trial_id: why} (clock); inputs {path: SHA-256}; counts."""
    passes: dict
    keys: dict
    tasks: tuple
    persons: tuple
    persons_sha256: str
    cell_cm: float
    cell_n: int
    cell_pairs: tuple
    cell_persons: tuple
    excluded: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


def clock_ok(window_store, recording):
    """True when the recording's clock is the targets' (span_t0 0; its first record at offset 0, every sample kept
    (k 1) at 100 Hz)."""
    g = window_store.recordings.iloc[int(recording)]
    r = window_store.records.iloc[int(g["rows"][0])]
    try:
        return (float(g["span_t0"]) == 0.0 and float(r["offset"]) == 0.0 and int(r["k"]) == 1
                and float(r["fps"]) == RATE)
    except (KeyError, TypeError, ValueError):
        return False


def check_input_pin(inputs, pin=None):
    """The input pin (module docstring): pin (default the file named by PIN_ENV; empty or unset: no check) is the path
    of a JSON file whose 'inputs' must equal `inputs` ({tree-relative path: SHA-256}); ProvenanceError otherwise or
    when the file cannot be read. Returns the pin's tree-relative path, or None when nothing was checked."""
    pin = os.environ.get(PIN_ENV, "") if pin is None else pin
    if not pin:
        return None
    from ..train.provenance import ProvenanceError
    path = pin if os.path.isabs(pin) else os.path.join(paths.TREE, pin)
    try:
        with open(path, encoding="utf-8") as f:
            want = json.load(f)["inputs"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ProvenanceError(f"walkway anchor: the input pin {pin} cannot be read ({type(e).__name__}: {e})")
    got = dict(inputs)
    if not isinstance(want, dict) or got != want:
        keys = sorted(set(got) | set(want if isinstance(want, dict) else {}))
        bad = [k for k in keys if not isinstance(want, dict) or got.get(k) != want.get(k)]
        raise ProvenanceError(f"walkway anchor: the anchor table's inputs differ from the pin {_rel(path)} ({bad})")
    return _rel(path)


def build_anchor_table(window_store, section, *, events=EVENTS_CSV, trials=TRIALS_CSV, alignment=ALIGNMENT_CSV,
                       pd_folds=PD_FOLDS_CSV, deny=True, check_clock=True):
    """The AnchorTable of a window store under a config.WalkwayAnchorSection (module docstring), checked with
    assert_anchor_clean; AnchorRefused before any target file opens when the training fold's SHA-256 is not
    section.persons_sha256 or a person is refused; ProvenanceError before any target value is read when PIN_ENV names
    a pin that the input files' SHA-256 do not match. deny=False skips the deny lists of train/provenance (tests on
    fake persons without the tiers records file); check_clock=False skips clock_ok (fake stores)."""
    tasks = tuple(section.tasks)
    rec = window_store.recordings
    train = sorted(set(rec.loc[(rec["source"] == SOURCE) & (rec["fold"] == "train"), "person_id"]))
    refused = refused_persons(window_store, pd_folds)
    check_persons(train, train, refused, section.persons_sha256, deny=deny)
    inputs = {_rel(p): sha256_file(p) for p in (events, trials, alignment) + ((pd_folds,) if pd_folds else ())}
    check_input_pin(inputs)
    trial_rows = read_rows(trials, train, CELL_COLUMNS, refused)
    fit = cell_fit(trial_rows)
    if not fit["n"] or not math.isfinite(fit["cell_cm"]):
        raise AnchorRefused("walkway anchor: no training-fold SelfPace trial to fit the cell size on")
    align = {r["trial_id"]: _num(r["away_from_pooled_ms"]) for r in read_rows(alignment, train, ALIGN_COLUMNS, refused)}
    ev = [project(r) for r in read_rows(events, train, ANCHOR_COLUMNS, refused) if r["task"] in tasks]
    by_trial = passes_of(ev, fit["cell_cm"])
    sel = rec[(rec["source"] == SOURCE) & (rec["fold"] == "train") & rec["key"].isin(tasks)]
    passes, keys, excluded, persons = {}, {}, {}, set()
    for i, g in sel.iterrows():
        tid = trial_id(g["person_id"], g["key"])
        if tid not in by_trial:
            continue
        away = align.get(tid, float("nan"))
        if not math.isfinite(away) or abs(away) > CLOCK_MAX_MS:
            excluded[tid] = "no alignment value" if not math.isfinite(away) else f"shank offset {away:g} ms"
            continue
        if check_clock and not clock_ok(window_store, i):
            excluded[tid] = "store clock is not the targets' clock"
            continue
        passes[int(i)] = by_trial[tid]
        keys[int(i)] = str(g["key"])
        persons.add(str(g["person_id"]))
    n_pass = sum(len(v) for v in passes.values())
    table = AnchorTable(passes=passes, keys=keys, tasks=tasks, persons=tuple(sorted(persons)),
                        persons_sha256=person_hash(train), cell_cm=fit["cell_cm"], cell_n=fit["n"],
                        cell_pairs=fit["pairs"], cell_persons=fit["persons"], excluded=dict(sorted(excluded.items())),
                        inputs=inputs,
                        counts={"training_fold_persons": len(train), "anchored_persons": len(persons),
                                "recordings": len(passes), "passes": n_pass, "event_rows": len(ev),
                                "trials_with_passes": len(by_trial), "excluded_trials": len(excluded)})
    assert_anchor_clean(table, train, refused, section.persons_sha256, deny=deny)
    return table


def _row_fields(row):
    if hasattr(row, "__dataclass_fields__"):
        return [f.name for f in fields(row)]
    if isinstance(row, dict):
        return [str(k) for k in row]
    return [str(k) for k in vars(row)] if hasattr(row, "__dict__") else []


def assert_anchor_clean(table, train, refused=None, persons_sha256=None, deny=True):
    """The checks of section B, item 2 (module docstring) on an AnchorTable; returns the summary that the training
    log records ('start', walkway_anchor.table)."""
    from ..train.losses import label_paths
    persons = check_persons(table.persons, train, refused, persons_sha256, deny=deny)
    if persons_sha256 is not None and table.persons_sha256 != str(persons_sha256):
        raise AnchorRefused("walkway anchor: the table's persons SHA-256 is not the configuration's")
    for rec, rows in table.passes.items():
        if not isinstance(rec, (int, np.integer)) or isinstance(rec, bool):
            raise AnchorRefused(f"walkway anchor: a row key {rec!r} is not a recording index")
        for row in rows:
            named = [f for f in _row_fields(row) if "person" in f.lower() or "subject" in f.lower()]
            if named:
                raise AnchorRefused(f"walkway anchor: a row carries a person column ({named})")
        bad = label_paths(list(rows))
        if bad:
            raise AnchorRefused(f"walkway anchor: label keys in the rows ({bad[:3]})")
    out = sorted(set(table.cell_persons) - set(train))
    if out:
        raise AnchorRefused(f"walkway anchor: the cell size was fitted on {len(out)} persons outside the training "
                            f"fold ({out[:3]})")
    if not table.cell_pairs or int(table.cell_n) != len(table.cell_pairs):
        raise AnchorRefused("walkway anchor: the cell fit records no pairs or another count")
    refit = cell_of_pairs(table.cell_pairs)
    if not (math.isfinite(float(table.cell_cm)) and abs(refit - float(table.cell_cm)) <= CELL_TOL_CM):
        raise AnchorRefused(f"walkway anchor: cell size {table.cell_cm} cm, refit {refit} cm")
    return {"format": FORMAT, "check": "passed", "persons": len(persons), "persons_sha256": table.persons_sha256,
            "anchored_persons_sha256": person_hash(persons), "cell_cm": float(table.cell_cm),
            "cell_n": int(table.cell_n), "cell_persons": len(table.cell_persons),
            "cell_persons_sha256": person_hash(table.cell_persons), "tasks": list(table.tasks),
            "excluded": dict(table.excluded), "inputs": dict(table.inputs), "counts": dict(table.counts),
            "refused_sets": {k: len(v) for k, v in sorted((refused or {}).items())}}


# ---------------------------------------------------------------- per-micro-step targets
@dataclass
class AnchorTargets:
    """Segments of a micro-step's windows (module docstring): window (S,) int64, k_a and k_b (S,) float32 fractional
    knots of the window clock, dist_m (S,) float32 walkway distance, n_strides (S,) int64."""
    window: torch.Tensor
    k_a: torch.Tensor
    k_b: torch.Tensor
    dist_m: torch.Tensor
    n_strides: torch.Tensor

    @property
    def n(self):
        return int(self.window.numel())

    def to(self, device):
        return AnchorTargets(*(getattr(self, f.name).to(device) for f in fields(self)))

    @staticmethod
    def empty():
        z = torch.zeros(0)
        return AnchorTargets(z.long(), z.float(), z.float(), z.float(), z.long())


def targets_for_windows(table, recordings, windows, rate=RATE):
    """AnchorTargets of windows (windows.Window, batch order) whose recording index (WindowStore.recordings row; -1
    for a synthetic window) is recordings[i]. A window of SOURCE whose key differs from the table's key of its
    recording raises ValueError."""
    if len(recordings) != len(windows):
        raise ValueError(f"{len(recordings)} recording indices for {len(windows)} windows")
    w, ka, kb, d, n = [], [], [], [], []
    for i, (r, wd) in enumerate(zip(recordings, windows)):
        if wd.synthetic or wd.source != SOURCE or int(r) < 0:
            continue
        ps = table.passes.get(int(r))
        if ps is None:
            continue
        if str(wd.key) != table.keys[int(r)]:
            raise ValueError(f"window {i}: key {wd.key!r} differs from the anchor table's {table.keys[int(r)]!r}")
        t0 = float(wd.t0)
        for s in window_segments(ps, t0, float(wd.duration)):
            w.append(i)
            ka.append(rate * (s.t_a - t0))
            kb.append(rate * (s.t_b - t0))
            d.append(s.dist_m)
            n.append(s.n_strides)
    if not w:
        return AnchorTargets.empty()
    return AnchorTargets(torch.tensor(w, dtype=torch.int64), torch.tensor(ka, dtype=torch.float32),
                         torch.tensor(kb, dtype=torch.float32), torch.tensor(d, dtype=torch.float32),
                         torch.tensor(n, dtype=torch.int64))


def targets_for_step(table, step, rate=RATE):
    """AnchorTargets of a sampler Step (data.sampler.Step): its plan items give each window's recording index (-1
    for a synthetic item), its windows the source, key, t0 and duration, in item order."""
    items = list(step.plan.items)
    if len(items) != len(step.windows):
        raise ValueError(f"{len(items)} plan items for {len(step.windows)} windows")
    for i, (it, wd) in enumerate(zip(items, step.windows)):
        if it.source != wd.source or (not wd.synthetic and it.person != wd.person_id):
            raise ValueError(f"window {i} is not the plan's item {i}")
    recs = [int(it.recording) if it.source == wd.source and not wd.synthetic else -1
            for it, wd in zip(items, step.windows)]
    return targets_for_windows(table, recs, step.windows, rate)
