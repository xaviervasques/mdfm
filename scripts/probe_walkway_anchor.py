"""Feasibility probe of the walkway-anchor lever (F2 lever 4, DRAFT; task L4, 2026-10-06).

Label of every output: LABEL. CPU only (CUDA_VISIBLE_DEVICES=-1), no model, no checkpoint, no training.

Question. The DRAFT lever (docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md) adds, in phase B, a loss on real
weargait_pd windows of the 54 training-fold controls (train54) that anchors the decoded forward distance over each
walkway pass segment a window covers to the walkway-measured distance. Before any code exists this probe measures what
that loss would have to work with:
1. Data volume: per train54 person, how many phase B real windows of the pilot's own sampler would overlap a walkway
   pass with targets, and what share of the weargait_pd windows and of all phase B tokens that is.
2. Alignment: whether the walkway timeline and the IMU clock of a window agree (the time convention of
   scripts/build_weargait_targets.py and of mdfm/gates/g3.py, the recording clock of the window store, and an
   empirical check: the shank-gyroscope initial contacts of each train54 trial against its walkway contacts).
3. Targets: the anchor's values available per anchored window (distance, duration, speed, strides).
4. Attainability: the margins of the lever-2 rule form from the v1 and seed-2 files, the calibration head's dev
   reading as the ceiling, the smallest share of the gain gap the lever must close to meet each margin, and the
   matched-settings seed differences of scripts/seed_difference.py beside them when they exist.

Inputs (read only):
- configs/stage1_pilot.yaml; the window store (MDFM_WINDOW_STORE, else cache/window_store: its tables only, no shard
  is read) and the AMASS train body store (its takes table only): mdfm.data.sampler.MixtureSampler in phase B, seed
  loop.phase_seed (20260929, v1's and every phase-B-only continuation's), plan() only (no window is read or
  rendered), for the micro-steps of phase B (step_counts x micro_per_step = 64,088).
- reports/weargait_targets/: events.csv, trials.csv (through mdfm.gates.g3.read_rows: train54 rows only, lockbox rows
  refused) and phases.csv (csv module, train54 rows only, each asserted lockbox False); summary.json (the G3 cell size,
  fitted by the builder on every non-lockbox SelfPace trial, dev and PD persons included; used here only for the
  pass-speed check against mdfm.gates.g3, never for an anchor value).
- reports/bench/g3_gain_check/results.json (person sets and their SHA-256, the fitted gains, the dev12 and in-sim gains
  of v1_B and s2_B for the transfer thresholds); the window store's weargait_pd/source.json (train person SHA-256);
  cache/runs/phaseA_root_full/log.jsonl (v1's logged phase B token shares, window counts and terms: the replay check
  and the size of the term); reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/folds_proposed.csv (person ids and
  fold of the proposed PD folds only, for the disjointness check of the anchor persons).
- With records on (default): the IMU records of train54 for the anchored tasks through mdfm.gates.g3.imu_records
  (canonical.load with lockbox and EVAL refused; each index row checked for lockbox False and a tier other than EVAL
  and EXCLUDED_V1 before its file opens). Only the shank gyroscopes are used.
- Attainability: reports/gates/g3/diag_<end>/ (diag.json, events.csv) and dev_<end>/g3_dev.json, reports/gates/g1/
  dev_<end>/g1.json of v1_B and s2_B, reports/gpu_queue/branch_seeds_20261002_211426.json, through the functions of
  scripts/weekend_branch.py and scripts/g3_gain_check.py; reports/bench/f2_lever2/results.json (seconds per phase B
  step); reports/bench/seed_difference/results.json when it exists.
- Transfer references: reports/gates/g3/diag_<v1_B>_ablations/events.csv (rows base, sites6 and the families, dev14),
  through g3_gain_check.read_events; the presence of the matched s2_B ablations and of the PD dev folders of T27.

Never read: a dev walkway value for the anchor (the dev rows enter the attainability arithmetic and the transfer
references only, from the existing v1 and s2 diag files that every F2 rule has read); a lockbox, EVAL, CODY or care_pd
record.

Definitions.
- Passes and footfalls: the IC rows of events.csv with a pass_id (the straight passes of the B3a build), read through
  the column whitelist ANCHOR_COLUMNS; a footfall is full when it is not partial and has a heel position.
- Cell size of the anchor (cell_fit): the builder's rule (scripts/build_weargait_targets.py, the median over SelfPace
  trials with a matched PKMAS row of PKMAS stride length in cm over walkway stride length in cells) refit on the
  train54 rows of trials.csv only, so that no dev or PD walkway value enters an anchor target. The summary.json cell
  (all non-lockbox persons) is reported beside it and used only by the pass-speed check.
- Window clock: a real window of the sampler starts at t0 on the recording clock (origin the first input sample) and
  lasts dur s; its knots sit at t0 + j / 100 s, j = 0 .. ceil(100 dur) (mdfm.model.mfm.batch_from_windows). A walkway
  event at time t (the CSV Time column) lies inside the window when t0 <= t <= t0 + dur.
- Anchor segment of a window and pass: the full footfalls of the pass inside the window; per foot its first and last
  such footfall; the foot with the longer interval (ties: more strides, then L) gives the segment (t_a, t_b), its
  distance |heel(t_b) - heel(t_a)| x cell, its duration, speed and number of strides (same-foot footfalls minus one).
  At least one stride. The any-foot segment (first and last full footfall of any foot, at least 3 footfalls, the G3
  pass-speed form) is reported beside. A window is anchored when it holds at least one segment.
- Walkway strides inside a window: the G3 strides (same foot two footfalls apart, the other foot between, both full)
  with both ends inside the window.
- Scopes: g3 = SelfPace and HurriedPace (the G3 tasks); walkway = g3 plus SelfPace_mat, HurriedPace_mat and
  SelfPace_matTURN (straight Walk-labelled passes on the walkway); TUG and TandemGait are reported per task and anchor
  nothing.
- Analytic expectation (beside the replay): for a recording of span s and window length dur = min(10, s), t0 uniform
  on [0, s - dur]; the window is anchored when it holds a consecutive same-foot pair (t_i, t_j) of full footfalls of
  one pass, i.e. t0 in [t_j - dur, t_i]; the share is the measure of the union of these intervals within [0, s - dur]
  over s - dur (at s <= dur, the indicator at t0 = 0). Per person the recording share is span / sum of spans, as the
  sampler draws; persons are drawn uniformly.

Outputs (reports/walkway_anchor/ by default): results.json, summary.md, per_person.csv, per_task.csv,
alignment_trials.csv (records on), _provenance.json.

Options: --micro-steps N (default: all of phase B), --no-records (skip the shank-gyroscope check), --out DIR,
--moveatlas ROOT.
"""
import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

LABEL = ("feasibility probe of a DRAFT lever (F2 lever 4, walkway anchor): dev diagnostic, not a G3 read; anchor "
         "targets from training-fold controls only")
TASK = "L4"
OUT = os.path.join(TREE, "reports", "walkway_anchor")
SOURCE = "weargait_pd"
RATE = 100.0
G3_TASKS = ("SelfPace", "HurriedPace")
MAT_TASKS = ("SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN")
OTHER_TASKS = ("TUG", "TandemGait")
TARGET_TASKS = G3_TASKS + MAT_TASKS + OTHER_TASKS
SCOPES = {"g3": G3_TASKS, "walkway": G3_TASKS + MAT_TASKS}
ANCHOR_COLUMNS = ("trial_id", "person_id", "task", "lockbox", "foot", "event", "walkway_t_s", "pass_id",
                  "heel_x_cells", "partial")
MIN_ANY_FOOTFALLS = 3                     # mdfm.gates.g3.MIN_PASS_FOOTFALLS
IC_TOL_S = 0.3                            # mdfm.gates.g3.IC_MATCH_S
ALIGN_FLAG_MS = (50.0, 100.0)             # per-trial offset away from the pooled one, reported
CELL_TASK = "SelfPace"                    # the builder's cell-size fit task
FOLDS_REVIEWED = os.path.join(TREE, "reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06", "folds_proposed.csv")
G3_DIR = os.path.join(TREE, "reports", "gates", "g3")
ABLATION_ROWS = ("base", "sites6", "family_legs", "family_lower_back", "family_wrists")
MIN_ROW_PERSONS = 10                      # a row's gain is read on a quantity with at least this many dev14 persons
COMPARE_TERMS = ("sup_root_rate", "imu_nll", "sup_deriv", "hsic")     # weighted phase B terms beside the anchor
V1_LOG = os.path.join(TREE, "cache", "runs", "phaseA_root_full", "log.jsonl")
CONFIG = os.path.join(TREE, "configs", "stage1_pilot.yaml")
GAIN_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
SEEDS = os.path.join(TREE, "reports", "gpu_queue", "branch_seeds_20261002_211426.json")
LEVER2_RESULTS = os.path.join(TREE, "reports", "bench", "f2_lever2", "results.json")
SEED_DIFF = os.path.join(TREE, "reports", "bench", "seed_difference", "results.json")
V1_B = "phaseA_root_full_ckpt_B_032044"
S2_B = "pilot_v1_s2_ckpt_B_032044"
TARGET_FRACTION, OTHER_FRACTION, SEED_MULTIPLE = 0.10, 0.05, 2.0     # the lever-2 rule form
NAMES_TO_CHECK = ("anchor_t_a", "anchor_t_b", "anchor_dist_m", "anchor_n_strides", "anchor_window", "anchor_valid")


# ---------------------------------------------------------------- small helpers
def _module(name):
    key = f"_l4_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rel(path):
    try:
        return os.path.relpath(path, TREE).replace("\\", "/")
    except ValueError:                    # another drive (a test's temporary folder)
        return os.path.abspath(path).replace("\\", "/")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _q(x, qs=(10, 50, 90)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{q}": None for q in qs} | {"n": 0}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(len(x)), "mean": float(x.mean())}


# ---------------------------------------------------------------- footfalls and segments
@dataclass
class Pass:
    """One straight walkway pass: pass_id; t (n,) walkway IC times (s, CSV clock), foot (n,) 'L'/'R', heel_m (n,)
    heel position along the walkway (m, NaN when missing), full (n,) bool (not partial, heel present); sorted by
    time."""
    pass_id: str
    t: np.ndarray
    foot: np.ndarray
    heel_m: np.ndarray
    full: np.ndarray


@dataclass
class Segment:
    pass_id: str
    foot: str
    t_a: float
    t_b: float
    dist_m: float
    n_strides: int
    n_full: int
    any_t_a: float = float("nan")
    any_t_b: float = float("nan")
    any_dist_m: float = float("nan")

    @property
    def dur_s(self):
        return self.t_b - self.t_a

    @property
    def speed_m_s(self):
        return self.dist_m / self.dur_s if self.dur_s > 0 else float("nan")


def project(row):
    """The anchor's view of an events.csv row: the ANCHOR_COLUMNS only (the whitelist of the label-free extension of
    the DRAFT, section 2.3)."""
    return {k: row[k] for k in ANCHOR_COLUMNS}


def pass_footfalls(rows, cell_cm):
    """{trial_id: [Pass]} from events.csv rows (dicts with ANCHOR_COLUMNS): IC rows with a pass_id."""
    groups = {}
    for r in rows:
        if r["event"] != "IC" or r["pass_id"] == "":
            continue
        groups.setdefault((r["trial_id"], r["pass_id"]), []).append(r)
    out = {}
    for (tid, pid), P in sorted(groups.items()):
        P.sort(key=lambda r: (float(r["walkway_t_s"]), r["foot"]))
        t = np.array([float(r["walkway_t_s"]) for r in P])
        foot = np.array([r["foot"] for r in P])
        heel = np.array([float(r["heel_x_cells"]) * cell_cm / 100.0 if r["heel_x_cells"] != "" else np.nan
                         for r in P])
        full = np.array([r["partial"] == "False" and r["heel_x_cells"] != "" for r in P])
        out.setdefault(tid, []).append(Pass(str(pid), t, foot, heel, full))
    return out


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
    seg = Segment(p.pass_id, foot, float(p.t[i]), float(p.t[j]), float(abs(p.heel_m[j] - p.heel_m[i])), int(n_str),
                  int(inside.sum()))
    anyi = np.flatnonzero(inside)
    if len(anyi) >= MIN_ANY_FOOTFALLS:
        i0, i1 = int(anyi[0]), int(anyi[-1])
        seg.any_t_a, seg.any_t_b = float(p.t[i0]), float(p.t[i1])
        seg.any_dist_m = float(abs(p.heel_m[i1] - p.heel_m[i0]))
    return seg


def window_segments(passes, a, dur):
    """Segments of the passes (a trial's list of Pass) inside the window [a, a + dur]."""
    out = []
    for p in passes:
        s = segment_of(p, a, a + dur)
        if s is not None:
            out.append(s)
    return out


def strides_inside(passes, a, dur):
    """Number of G3 walkway strides (module docstring) with both ends inside [a, a + dur]."""
    n = 0
    b = a + dur
    for p in passes:
        for i in range(2, len(p.t)):
            if (p.foot[i] == p.foot[i - 2] and p.foot[i - 1] != p.foot[i] and p.full[i] and p.full[i - 2]
                    and p.t[i - 2] >= a - 1e-9 and p.t[i] <= b + 1e-9):
                n += 1
    return n


def anchor_intervals(passes, dur):
    """Intervals [lo, hi] of window starts t0 for which a window of length dur holds a consecutive same-foot pair of
    full footfalls of one pass (t0 in [t_j - dur, t_i])."""
    iv = []
    for p in passes:
        for foot in ("L", "R"):
            idx = np.flatnonzero(p.full & (p.foot == foot))
            for i, j in zip(idx[:-1], idx[1:]):
                lo, hi = float(p.t[j]) - dur, float(p.t[i])
                if hi >= lo:
                    iv.append((lo, hi))
    return iv


def union_measure(iv, lo, hi):
    """Measure of the union of the intervals iv clipped to [lo, hi]."""
    clipped = sorted((max(a, lo), min(b, hi)) for a, b in iv if min(b, hi) > max(a, lo))
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def anchored_share(passes, span, win_s=10.0):
    """Probability that a window of the sampler on a recording of this span (t0 uniform on [0, span - dur], dur =
    min(win_s, span)) is anchored."""
    dur = min(float(win_s), float(span))
    iv = anchor_intervals(passes, dur)
    width = float(span) - dur
    if width <= 1e-9:
        return 1.0 if any(lo <= 0.0 <= hi for lo, hi in iv) else 0.0
    return union_measure(iv, 0.0, width) / width


def expected_share(recordings, tasks, win_s=10.0):
    """Analytic share of a person's windows that are anchored: recordings is the person's list of (task, span_s,
    passes) over every recording the sampler can draw (any task); the sampler picks a recording with probability
    span / sum of spans, so the share is the span-weighted sum of anchored_share over the recordings of `tasks`."""
    total = float(sum(span for _, span, _ in recordings))
    if total <= 0:
        return 0.0
    return float(sum(span / total * anchored_share(passes, span, win_s)
                     for task, span, passes in recordings if task in tasks))


# ---------------------------------------------------------------- persons and guards
def person_sets():
    """(train54, dev14, dev12, locked, set block): the sets of F1's results.json, train54 from mdfm.gates.g3 (fold
    train), each SHA-256 asserted against F1's and the window store's; every set disjoint from the lockbox."""
    from mdfm.data import window_store as WS
    from mdfm.gates import g3
    f1 = _json(GAIN_RESULTS)["sets"]
    train = sorted(g3.controls("train"))
    dev14, dev12 = sorted(f1["dev14"]), sorted(f1["dev12"])
    locked = set(g3.lockbox_persons())
    src = _json(os.path.join(WS.store_root(), SOURCE, "source.json"))
    block = check_sets(train, dev14, dev12, locked, f1, src)
    return train, dev14, dev12, locked, block


def check_sets(train, dev14, dev12, locked, f1, src):
    """Assertions of person_sets on given sets: train54 and dev14 equal to F1's (SHA-256) and to the window store's
    folds, dev12 inside dev14, no lockbox person in any set, train54 and dev14 disjoint. Returns the set block."""
    from mdfm.data import sources
    assert sources.person_hash(train) == f1["train54_sha256"], "train54 differs from F1's"
    assert sources.person_hash(dev14) == f1["dev14_sha256"], "dev14 differs from F1's"
    assert src["person_sha256"]["train"] == f1["train54_sha256"], "the window store's train fold is not train54"
    assert src["person_sha256"]["dev"] == f1["dev14_sha256"], "the window store's dev fold is not dev14"
    assert set(dev12) <= set(dev14), "dev12 is not inside dev14"
    for name, s in (("train54", train), ("dev14", dev14), ("dev12", dev12)):
        assert not set(s) & set(locked), f"{name} holds a lockbox person"
    assert not set(train) & set(dev14), "train54 and dev14 overlap"
    return {"train54_sha256": f1["train54_sha256"], "dev14_sha256": f1["dev14_sha256"],
            "window_store_train_sha256": src["person_sha256"]["train"], "n_train54": len(train),
            "lockbox_in_sets": 0}


def reviewed_pd_folds(train, dev14, locked, path=FOLDS_REVIEWED):
    """The proposed PD folds of the reviewed stage-2 sources (person ids and fold only): asserted disjoint from train54,
    dev14 and the lockbox (the persons assert_anchor_clean of the DRAFT refuses). Returns counts and SHA-256."""
    from mdfm.data import sources
    folds = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE:
                folds.setdefault(r["fold"], set()).add(r["person_id"])
    pd_all = set().union(*folds.values()) if folds else set()
    assert not pd_all & set(train), "a proposed PD-fold person is in train54"
    assert not pd_all & set(dev14), "a proposed PD-fold person is in dev14"
    assert not pd_all & set(locked), "a lockbox person in the proposed PD folds"
    return {"path": rel(path), "sha256": _sha_file(path), "persons": {k: len(v) for k, v in sorted(folds.items())},
            "dev_sha256": sources.person_hash(sorted(folds.get("dev", ()))) if folds.get("dev") else None,
            "overlap_train54": 0, "overlap_dev14": 0, "overlap_lockbox": 0}, pd_all


def guard_replayed(rows, train, refused):
    """Every replayed weargait_pd window belongs to a train54 person and to none of `refused` (dev14, lockbox, the
    proposed PD folds)."""
    persons = {r[1] for r in rows}
    assert persons <= set(train), "a replayed weargait window of a person outside train54"
    assert not persons & set(refused), "a replayed weargait window of a refused person"
    return len(persons)


def cell_fit(trial_rows):
    """The walkway cell size refit on the given trials.csv rows (module docstring, cell_fit): the median over SelfPace
    trials with a finite PKMAS stride length and a positive walkway stride in cells of their ratio. trials.csv keeps
    the cells to 3 decimals. Every row must read lockbox False."""
    ratios, persons = [], set()
    for r in trial_rows:
        if r["task"] != CELL_TASK:
            continue
        assert r["lockbox"] == "False", r["trial_id"]
        a, b = _num(r["pkmas_stride_length_cm"]), _num(r["ww_stride_cells_mean"])
        if math.isfinite(a) and math.isfinite(b) and b > 0:
            ratios.append(a / b)
            persons.add(r["person_id"])
    if not ratios:
        return {"n": 0, "cell_cm": float("nan"), "p10": None, "p90": None, "persons": []}
    x = np.asarray(ratios, dtype=np.float64)
    return {"n": int(len(x)), "cell_cm": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "persons": sorted(persons)}


def read_targets(train):
    """(passes {trial_id: [Pass]} with the train54 cell of cell_fit (the anchor's), passes_g3 the same with the G3 cell
    of summary.json (the pass-speed check only), trials rows, phases first-sample times, counts) of train54 for
    TARGET_TASKS."""
    from mdfm.gates import g3
    cell_g3, rule_version = g3.target_meta()
    assert rule_version == g3.TARGET_RULE, rule_version
    ev = [project(r) for r in g3.read_rows(g3.EVENTS_CSV, train) if r["task"] in TARGET_TASKS]
    trials = [r for r in g3.read_rows(g3.TRIALS_CSV, train) if r["task"] in TARGET_TASKS]
    assert all(r["lockbox"] == "False" for r in ev + trials)
    assert {r["person_id"] for r in ev + trials} <= set(train), "a target row of a person outside train54"
    first = {}
    keep = set(train)
    with open(os.path.join(g3.TARGETS_DIR, "phases.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["person_id"] not in keep or r["task"] not in TARGET_TASKS:
                continue
            assert r["lockbox"] == "False", r["trial_id"]
            if r["run_index"] == "0":
                first[r["trial_id"]] = float(r["start_s"])
    fit = cell_fit(trials)
    assert fit["n"] > 0 and math.isfinite(fit["cell_cm"]), "no train54 SelfPace trial to fit the cell size on"
    assert set(fit["persons"]) <= set(train), "the cell size fit holds a person outside train54"
    passes = pass_footfalls(ev, fit["cell_cm"])
    passes_g3 = pass_footfalls(ev, cell_g3)
    fit_block = {k: v for k, v in fit.items() if k != "persons"} | {"n_persons": len(fit["persons"]),
                                                                   "persons_inside_train54": True}
    counts = {"events_rows": len(ev), "trials": len(trials), "cell_cm": fit["cell_cm"], "cell_fit_train54": fit_block,
              "cell_cm_g3_summary": cell_g3, "cell_relative_difference": fit["cell_cm"] / cell_g3 - 1.0,
              "passes": sum(len(v) for v in passes.values()), "trials_with_passes": len(passes)}
    return passes, passes_g3, trials, first, counts


# ---------------------------------------------------------------- sampler replay
def make_sampler():
    from mdfm.data import body_store as BS, sampler as S, window_store as WS
    from mdfm.train import config as C, loop as L
    cfg = C.load(CONFIG)
    bs, ws = BS.BodyStore(split="train"), WS.WindowStore()
    smp = S.MixtureSampler(bs, ws, None, "B", L.phase_seed(cfg, "B"), cfg.mixture, guard=False)
    n_micro = int(C.step_counts(cfg)[1]) * int(cfg.schedule.micro_per_step)
    return cfg, ws, smp, n_micro


def replay(smp, n_micro):
    """Plans of the first n_micro phase B micro-steps (plan only): (real windows of weargait_pd as rows (micro, person,
    recording, t0, dur, tokens), per micro-step arrays {windows, tokens, wg_tokens, wg_windows}, totals)."""
    from mdfm.data.sampler import SYNTHETIC
    rows = []
    micro = {k: np.zeros(n_micro, dtype=np.int64) for k in ("windows", "tokens", "wg_tokens", "wg_windows")}
    tot = {"tokens": 0, "windows": 0, "wg_tokens": 0, "wg_windows": 0, "syn_tokens": 0, "syn_seconds": 0.0,
           "syn_bodies": 0}
    for m in range(n_micro):
        plan = smp.plan()
        bodies = set()
        for it in plan.items:
            tot["tokens"] += it.tokens
            tot["windows"] += 1
            micro["tokens"][m] += it.tokens
            micro["windows"][m] += 1
            if it.source == SOURCE:
                rows.append((m, it.person, it.recording, it.t0, it.dur, it.tokens))
                tot["wg_tokens"] += it.tokens
                tot["wg_windows"] += 1
                micro["wg_tokens"][m] += it.tokens
                micro["wg_windows"][m] += 1
            elif it.source == SYNTHETIC:
                tot["syn_tokens"] += it.tokens
                if it.body not in bodies:
                    bodies.add(it.body)
                    tot["syn_seconds"] += it.dur
                    tot["syn_bodies"] += 1
    return rows, micro, tot


def v1_log_records():
    """v1's phase B 'log' records in file order: (step, windows, weargait token share, micro_steps, tokens_per_micro)."""
    out = []
    if not os.path.isfile(V1_LOG):
        return out
    with open(V1_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "log" and d.get("phase") == "B":
                out.append((int(d["step"]), int(d["windows"]), float(d["token_share"].get(SOURCE, 0.0)),
                            int(d["micro_steps"]), float(d["tokens_per_micro"])))
    return out


def replay_check(micro, log):
    """The plan replay against v1's log, interval by interval: each log record takes the next micro_steps planned
    micro-steps (every interval, the last partial one included). Compared: the window count (the log counts the
    materialized windows), the weargait token share, and the weargait tokens, the log's being share x tokens_per_micro x
    micro_steps (useful tokens). The plan counts a real window's tokens exactly (mdfm/data/sampler.py), so equal
    weargait tokens on an interval mean that no weargait window of the plan was dropped or added there."""
    n_plan = len(micro["windows"])
    pos, dw, dshare, dwg, dtok, n_partial, ps, ls = 0, [], [], [], [], 0, [], []
    for _step, w, share, n, tpm in log:
        if pos + n > n_plan:
            break
        sl = slice(pos, pos + n)
        pos += n
        pw, pt, pg = (int(micro[k][sl].sum()) for k in ("windows", "tokens", "wg_tokens"))
        useful = tpm * n
        dw.append(pw - w)
        ps.append(pg / pt if pt else float("nan"))
        ls.append(share)
        dshare.append(ps[-1] - share)
        dwg.append(pg - share * useful)
        dtok.append(pt - useful)
        n_partial += int(n != 100)
    if not dw:
        return {"intervals_compared": 0, "micro_steps_compared": 0}
    dw, dwg = np.asarray(dw, dtype=np.float64), np.asarray(dwg, dtype=np.float64)
    return {"intervals_compared": len(dw), "micro_steps_compared": pos, "micro_steps_planned": n_plan,
            "intervals_partial": n_partial, "plan_weargait_token_share_mean": float(np.mean(ps)),
            "log_weargait_token_share_mean": float(np.mean(ls)),
            "windows_plan_minus_log": _q(dw, (0, 50, 100)), "intervals_windows_differ": int((dw != 0).sum()),
            "intervals_plan_fewer_windows": int((dw < 0).sum()),
            "weargait_share_plan_minus_log": _q(dshare, (0, 50, 100)),
            "weargait_tokens_plan_minus_log": _q(dwg, (0, 50, 100)),
            "intervals_weargait_tokens_equal": int((np.abs(dwg) < 0.5).sum()),
            "tokens_plan_minus_log_useful": _q(dtok, (0, 50, 100))}


# ---------------------------------------------------------------- alignment
def store_clock(ws, train, trials):
    """Checks of the recording clock against the targets' clock for the train54 recordings of TARGET_TASKS."""
    rec = ws.recordings
    records = ws.records
    samples = {r["trial_id"]: int(r["samples"]) for r in trials}
    g = rec[(rec["source"] == SOURCE) & (rec["fold"] == "train")]
    assert set(g["person_id"]) <= set(train)
    bad, n = [], 0
    for _, r in g.iterrows():
        if r["key"] not in TARGET_TASKS:
            continue
        n += 1
        ri = records.iloc[int(r["rows"][0])]
        tid = f"{r['person_id'][len('weargait:'):]}_{r['key']}"
        ok = (float(r["span_t0"]) == 0.0 and float(ri["offset"]) == 0.0 and int(ri["k"]) == 1
              and float(ri["fps"]) == RATE and samples.get(tid) == int(ri["n"]))
        if not ok:
            bad.append(tid)
    return {"recordings": n, "clock_equal_to_targets": n - len(bad), "differ": bad[:20], "n_differ": len(bad)}


def shank_alignment(train, tasks):
    """Per train54 trial of `tasks`: the shank-gyroscope ICs (mdfm.readouts.gait.shank_gyro_ic) against the walkway
    ICs of the Walk phase (nearest same-foot prediction within 0.3 s): median difference and matched count."""
    import pandas as pd
    from mdfm.data import canonical
    from mdfm.gates import g3
    from mdfm.readouts import gait
    tg = g3.load_targets(train, tasks=tasks)
    for row in g3.imu_rows(train, tasks=tasks):
        assert row["lockbox"] in (False, "False", 0) and str(row["tier"]) not in ("EVAL", "EXCLUDED_V1"), row
        assert not canonical.is_cody(row["person_id"])
    pred = []
    for tid, rec in g3.imu_records(train, tg.trials, tasks=tasks):
        assert rec.get("lockbox") in (False, "False") and str(rec.get("tier")) not in ("EVAL", "EXCLUDED_V1")
        for foot, x in g3.shank_gyro(rec).items():
            pred += [(tid, foot, float(t)) for t in gait.shank_gyro_ic(x, RATE)]
    pred = pd.DataFrame(pred, columns=["trial_id", "foot", "t_s"])
    d = g3.ic_differences(pred, tg.walkway_ic, 0.0, IC_TOL_S)
    m = d[d["matched"]]
    pooled = float(np.median(m["diff_s"])) * 1000.0
    per = []
    for tid, g in d.groupby("trial_id"):
        gm = g[g["matched"]]
        med = float(np.median(gm["diff_s"])) * 1000.0 if len(gm) else float("nan")
        per.append({"trial_id": tid, "person_id": g["person_id"].iloc[0], "task": g["task"].iloc[0],
                    "walkway_ic": int(len(g)), "matched": int(len(gm)), "median_diff_ms": med,
                    "away_from_pooled_ms": med - pooled if math.isfinite(med) else float("nan")})
    away = np.array([p["away_from_pooled_ms"] for p in per], dtype=np.float64)
    fin = np.isfinite(away)
    out = {"tasks": list(tasks), "trials": len(per), "walkway_ic": int(len(d)), "matched": int(len(m)),
           "matched_share": float(len(m) / len(d)) if len(d) else None, "pooled_median_ms": pooled,
           "per_trial_median_ms": _q([p["median_diff_ms"] for p in per]),
           "abs_away_from_pooled_ms": _q(np.abs(away[fin]), (50, 90, 100)),
           "trials_without_match": int((~fin).sum())}
    for lim in ALIGN_FLAG_MS:
        out[f"trials_away_over_{int(lim)}ms"] = int((np.abs(away[fin]) > lim).sum())
    out["trials_away_over_50ms_list"] = [{k: p[k] for k in ("trial_id", "walkway_ic", "matched", "median_diff_ms")}
                                         for p in per if math.isfinite(p["away_from_pooled_ms"])
                                         and abs(p["away_from_pooled_ms"]) > ALIGN_FLAG_MS[0]]
    return out, per


# ---------------------------------------------------------------- the anchor over the replayed windows
def anchor_table(rows, ws, passes):
    """Per replayed weargait window: person, task, t0, dur, tokens, segments (list of Segment), strides inside."""
    rec = ws.recordings
    keys = {int(i): (r["person_id"], r["key"]) for i, r in rec.iterrows()}
    out = []
    for m, person, recording, t0, dur, tokens in rows:
        p, key = keys[int(recording)]
        assert p == person
        tid = f"{person[len('weargait:'):]}_{key}"
        ps = passes.get(tid, []) if key in TARGET_TASKS else []
        segs = window_segments(ps, t0, dur) if ps else []
        out.append({"micro": m, "person": person, "task": key, "trial_id": tid, "t0": t0, "dur": dur,
                    "tokens": tokens, "segments": segs, "strides": strides_inside(ps, t0, dur) if ps else 0})
    return out


def scope_summary(table, scope_tasks, tot, n_micro):
    w = [r for r in table if r["task"] in scope_tasks and r["segments"]]
    segs = [s for r in w for s in r["segments"]]
    seconds = sum(s.dur_s for s in segs)
    anyd = [s.any_dist_m for s in segs if math.isfinite(s.any_dist_m)]
    return {"anchored_windows": len(w), "share_of_weargait_windows": len(w) / max(1, len(table)),
            "anchored_tokens": int(sum(r["tokens"] for r in w)),
            "share_of_weargait_tokens": sum(r["tokens"] for r in w) / max(1, tot["wg_tokens"]),
            "share_of_all_phase_b_tokens": sum(r["tokens"] for r in w) / max(1, tot["tokens"]),
            "anchored_windows_per_micro_step": len(w) / max(1, n_micro),
            "segments": len(segs), "segments_per_anchored_window": _q([len(r["segments"]) for r in w], (50, 90)),
            "segment_dist_m": _q([s.dist_m for s in segs]), "segment_dur_s": _q([s.dur_s for s in segs]),
            "segment_speed_m_s": _q([s.speed_m_s for s in segs]), "segment_strides": _q([s.n_strides for s in segs]),
            "segment_any_foot_dist_m": _q(anyd), "segments_with_any_foot_form": len(anyd),
            "walkway_strides_per_anchored_window": _q([r["strides"] for r in w]),
            "anchored_seconds": seconds, "anchored_seconds_per_micro_step": seconds / max(1, n_micro),
            "anchored_seconds_over_synthetic_seconds": seconds / tot["syn_seconds"] if tot["syn_seconds"] else None,
            "anchored_share_of_window_seconds": _q([sum(s.dur_s for s in r["segments"]) / r["dur"] for r in w])}


def per_person_rows(table, train, ws, passes):
    rec = ws.recordings
    g = rec[(rec["source"] == SOURCE) & (rec["fold"] == "train") & (rec["span_s"] >= 1.0 - 1e-9)]
    out = []
    for person in train:
        mine = [r for r in table if r["person"] == person]
        row = {"person_id": person, "windows": len(mine), "tokens": int(sum(r["tokens"] for r in mine))}
        gp = g[g["person_id"] == person]
        recs = [(r["key"], float(r["span_s"]), passes.get(f"{person[len('weargait:'):]}_{r['key']}", []))
                for _, r in gp.iterrows()]
        for scope, tasks in SCOPES.items():
            w = [r for r in mine if r["task"] in tasks and r["segments"]]
            segs = [s for r in w for s in r["segments"]]
            row[f"{scope}_anchored"] = len(w)
            row[f"{scope}_share"] = len(w) / max(1, len(mine))
            row[f"{scope}_segments"] = len(segs)
            row[f"{scope}_strides_in_segments"] = int(sum(s.n_strides for s in segs))
            row[f"{scope}_median_dist_m"] = float(np.median([s.dist_m for s in segs])) if segs else float("nan")
            row[f"{scope}_median_speed_m_s"] = float(np.median([s.speed_m_s for s in segs])) if segs else float("nan")
            row[f"{scope}_expected_share"] = expected_share(recs, tasks)
        tids = {f"{person[len('weargait:'):]}_{t}" for t in SCOPES["walkway"]}
        row["walkway_passes_available"] = sum(len(passes.get(t, [])) for t in tids)
        out.append(row)
    return out


def per_task_rows(table, passes, ws, train):
    rec = ws.recordings
    g = rec[(rec["source"] == SOURCE) & (rec["fold"] == "train")]
    out = []
    for task in sorted(set(r["task"] for r in table)):
        mine = [r for r in table if r["task"] == task]
        w = [r for r in mine if r["segments"]]
        segs = [s for r in w for s in r["segments"]]
        gt = g[g["key"] == task]
        out.append({"task": task, "windows": len(mine), "share_of_weargait_windows": len(mine) / max(1, len(table)),
                    "with_segment": len(w), "share_with_segment": len(w) / max(1, len(mine)),
                    "anchored_in_draft": task in SCOPES["walkway"], "recordings": int(len(gt)),
                    "median_span_s": float(gt["span_s"].median()) if len(gt) else float("nan"),
                    "passes": sum(len(passes.get(f"{p[len('weargait:'):]}_{task}", [])) for p in train),
                    "median_segment_dist_m": float(np.median([s.dist_m for s in segs])) if segs else float("nan"),
                    "median_segment_speed_m_s": float(np.median([s.speed_m_s for s in segs])) if segs else float("nan"),
                    "median_segment_strides": float(np.median([s.n_strides for s in segs])) if segs else float("nan")})
    return out


def pass_speed_check(passes, train):
    """The any-foot segment of a whole pass against mdfm.gates.g3's pass speed (same rule, same cell: pass the
    passes built with the G3 cell of summary.json): largest difference."""
    from mdfm.gates import g3
    tg = g3.load_targets(train, tasks=G3_TASKS)
    ref = {(r.trial_id, str(r.pass_id)): r.speed_m_s for r in tg.passes.itertuples()}
    diffs = []
    for tid, ps in passes.items():
        for p in ps:
            if (tid, p.pass_id) not in ref:
                continue
            s = segment_of(p, -1e9, 1e9)
            if s is not None and math.isfinite(s.any_dist_m):
                diffs.append(abs(s.any_dist_m / (s.any_t_b - s.any_t_a) - ref[(tid, p.pass_id)]))
    return {"passes_compared": len(diffs), "largest_abs_diff_m_s": float(max(diffs)) if diffs else None}


# ---------------------------------------------------------------- attainability
def success_margin(line, d):
    """The lever-2 rule's success margin: max(0.10 x line, 2 x seed difference)."""
    return max(TARGET_FRACTION * float(line), SEED_MULTIPLE * float(d))


def ni_margin(line, d):
    """The lever-2 rule's non-inferiority margin: max(0.05 x line, 2 x seed difference)."""
    return max(OTHER_FRACTION * float(line), SEED_MULTIPLE * float(d))


def gain_scan(units, g, margin):
    """Pure-gain arithmetic on one line's units (person_id, x walkway, y decoded): for f = 0, 0.01, .., 1 the decoded
    values are multiplied by g_f / g, g_f = g + f (1 - g) (the train54 gain moved a share f toward 1); per f the line
    (mean over persons of the mean |y g_f / g - x|), the paired improvement over f = 0 and whether it meets `margin`
    with a lower bound above 0. f = 1 is the calibration head (y / g)."""
    WB, GG = _module("weekend_branch"), _module("g3_gain_check")
    raw = GG.person_abs_error(units, 1.0)
    rows, first = [], None
    for f in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        gf = g + f * (1.0 - g)
        sc = GG.person_abs_error(units, g / gf)          # y / (g / gf) = y x gf / g
        p = WB.paired(raw, sc)
        line = float(np.mean(list(sc.values())))
        met = bool(p["point"] is not None and p["point"] >= margin - 1e-9 and p["interval"][0] > 0)
        rows.append({"f": float(f), "gain": gf, "line": line, "improvement": p["point"],
                     "interval": list(p["interval"]) if p["interval"] is not None else None, "meets": met})
        if met and first is None:
            first = rows[-1]
    head = rows[-1]
    return {"g_train54": g, "raw_recomputed": float(np.mean(list(raw.values()))), "head_line": head["line"],
            "head_improvement": head["improvement"], "head_interval": head["interval"],
            "smallest_share_meeting": first, "persons": len(raw), "at_quarter": rows[25], "at_half": rows[50]}


def _median_gain(units):
    """F1's gain form on given units: the median of y / x over the units with x > 0 (None without one)."""
    if not len(units):
        return None
    x, y = units["x"].to_numpy(np.float64), units["y"].to_numpy(np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return float(np.median(y[ok] / x[ok])) if ok.any() else None


def transfer_references(dev14, dev12, locked):
    """What the transfer lines T1 to T4 of the DRAFT can read today (dev diagnostic, reference values only):
    - thresholds from F1 (reports/bench/g3_gain_check/results.json): d_gap, |v1_B minus s2_B| of the train54-minus-dev12
      stride gain gap; d_dev12_stride and d_dev12_speed, |v1_B minus s2_B| of the dev12 gains; d_insim, of the in-sim
      stride gain;
    - per row of the v1_B ablations diag (ABLATION_ROWS): persons with a matched stride or pass speed on dev14 and on
      dev12, and the dev12 and dev14 stride and speed gains (F1's median form) when at least MIN_ROW_PERSONS dev14
      persons hold the quantity; the default diag's base row beside (same checkpoint, the code of the B-end reads);
    - presence of the inputs the DRAFT's gate requires: the s2_B ablations, the PD dev folders of T27 for v1_B and
      s2_B, and the code SHA-256 of each diag folder."""
    GG = _module("g3_gain_check")
    ends = _json(GAIN_RESULTS)["ends"]

    t = {}
    for end in ("v1_B", "s2_B"):
        e = ends[end]
        d12 = e["descriptive"]["dev12"]
        t[end] = {"train54_stride": float(e["gain"]["stride"]["gain"]), "train54_speed": float(e["gain"]["speed"]["gain"]),
                  "dev12_stride": float(d12["stride_per_stride"]["gain"]["point"]),
                  "dev12_speed": float(d12["speed_per_pass"]["gain"]["point"]),
                  "insim_stride": float(e["insim"]["stride_gain"])}
        t[end]["gap_stride"] = t[end]["train54_stride"] - t[end]["dev12_stride"]
    thr = {"d_gap": abs(t["v1_B"]["gap_stride"] - t["s2_B"]["gap_stride"]),
           "d_dev12_stride": abs(t["v1_B"]["dev12_stride"] - t["s2_B"]["dev12_stride"]),
           "d_dev12_speed": abs(t["v1_B"]["dev12_speed"] - t["s2_B"]["dev12_speed"]),
           "d_insim": abs(t["v1_B"]["insim_stride"] - t["s2_B"]["insim_stride"]), "ends": t}

    def folder(name):
        p = os.path.join(G3_DIR, name)
        prov = os.path.join(p, "_provenance.json")
        code = _json(prov).get("code_sha256") if os.path.isfile(prov) else None
        return {"path": rel(p), "present": os.path.isdir(p), "code_sha256": code}

    rows = {}
    abl = os.path.join(G3_DIR, f"diag_{V1_B}_ablations", "events.csv")
    if os.path.isfile(abl):
        for row in ABLATION_ROWS:
            st, sp, _ = GG.read_events(abl, dev14, locked, row=row)
            r = {}
            for q, units in (("stride", st), ("speed", sp)):
                n14 = int(units["person_id"].nunique())
                u12 = GG.subset(units, dev12)
                r[q] = {"dev14_persons": n14, "dev12_persons": int(u12["person_id"].nunique()),
                        "readable": n14 >= MIN_ROW_PERSONS,
                        "dev14_gain": _median_gain(units) if n14 >= MIN_ROW_PERSONS else None,
                        "dev12_gain": _median_gain(u12) if n14 >= MIN_ROW_PERSONS else None}
            rows[row] = r
    st, sp, _ = GG.read_events(os.path.join(G3_DIR, f"diag_{V1_B}", "events.csv"), dev14, locked)
    default_base = {"stride_dev12_gain": _median_gain(GG.subset(st, dev12)),
                    "speed_dev12_gain": _median_gain(GG.subset(sp, dev12))}
    gate = {"v1_B_diag": folder(f"diag_{V1_B}"), "v1_B_ablations": folder(f"diag_{V1_B}_ablations"),
            "s2_B_ablations": folder(f"diag_{S2_B}_ablations"),
            "v1_B_pd_dev": folder(f"dev_pd_{V1_B}"), "s2_B_pd_dev": folder(f"dev_pd_{S2_B}")}
    for k in ("v1_B_pd_dev", "s2_B_pd_dev"):
        gate[k]["result_present"] = os.path.isfile(os.path.join(TREE, gate[k]["path"], "g3_pd_dev.json"))
    return {"thresholds": thr, "v1_B_ablation_rows": rows, "v1_B_default_diag_base": default_base, "gate_inputs": gate}


def attainability(dev14, dev12, locked):
    """The margins of the lever-2 rule form from the v1 and s2 files, the calibration head as the ceiling, and the
    smallest share of the gain gap the lever must close to meet each success margin (pure-gain arithmetic on v1's dev12
    units, not a prediction)."""
    WB, GG = _module("weekend_branch"), _module("g3_gain_check")
    out = {}
    srcs = WB.g3_sources(TREE, V1_B)
    d1 = next(x for x in srcs if x["kind"] == "diag")
    lines = {"S12": d1["subsets"][WB.SUB12]["S_point"], "V12": d1["subsets"][WB.SUB12]["V_point"],
             "S14": d1["subsets"][WB.SUB14]["S_point"], "V14": d1["subsets"][WB.SUB14]["V_point"]}
    seeds = _json(SEEDS)["seed_diff"]["B_end"]
    sd = {k: float(seeds[k]["abs_diff"]) for k in ("S12", "V12", "S14", "V14")}

    def g3row(name, line):
        doc = _json(os.path.join(TREE, "reports", "gates", "g3", f"dev_{name}", "g3_dev.json"))
        return float(next(r for r in doc["results_dev"] if r["line"] == line)["point"])

    def g1a(name):
        g = WB.g1_read(os.path.join(TREE, "reports", "gates", "g1", f"dev_{name}"))
        assert g is not None and g["variant"] == "full"
        return float(g["point"])

    for k, line in (("IC", "ic_median_ms_walkway"), ("MAPE", "stride_mape_pct")):
        a, b = g3row(V1_B, line), g3row(S2_B, line)
        lines[k], sd[k] = a, abs(a - b)
    a, b = g1a(V1_B), g1a(S2_B)
    lines["G1a"], sd["G1a"] = a, abs(a - b)
    succ = {}
    for k in ("S12", "V12"):
        m = success_margin(lines[k], sd[k])
        succ[k] = {"line": lines[k], "seed_diff": sd[k], "ten_percent": TARGET_FRACTION * lines[k],
                   "two_seed": SEED_MULTIPLE * sd[k], "margin": m, "needed_line_at_most": lines[k] - m,
                   "below_line": m < lines[k]}
    ni = {k: {"line": lines[k], "seed_diff": sd[k], "margin": ni_margin(lines[k], sd[k])}
          for k in ("S14", "V14", "IC", "G1a", "MAPE")}
    # the calibration head (F1) as the ceiling, and the gain-closing scan
    gain = _json(GAIN_RESULTS)["ends"]["v1_B"]["gain"]
    assert gain["fitted_on"] == "train54"
    g_s, g_v = float(gain["stride"]["gain"]), float(gain["speed"]["gain"])
    ev = os.path.join(TREE, "reports", "gates", "g3", f"diag_{V1_B}", "events.csv")
    strides, speeds, _ = GG.read_events(ev, dev14, locked)
    scan = {}
    for k, units, g in (("S12", strides, g_s), ("V12", speeds, g_v)):
        scan[k] = gain_scan(GG.subset(units, dev12), g, succ[k]["margin"])
    seed_matched = None
    if os.path.isfile(SEED_DIFF):
        seed_matched = {"path": rel(SEED_DIFF), "sha256": _sha_file(SEED_DIFF)}
        try:
            sdj = _json(SEED_DIFF)
            seed_matched["lines"] = {k: sdj["lines"][k] for k in ("S12", "V12", "S14", "V14") if k in sdj.get("lines", {})}
        except (KeyError, ValueError, TypeError) as e:
            seed_matched["read_error"] = str(e)
    lv2 = _json(LEVER2_RESULTS)["reported"]["seconds_per_phase_b_step"]
    return {"v1_B": {"lines": lines, "seed_diff": sd, "source_diag": d1.get("source")}, "success": succ,
            "non_inferiority": ni, "gain_scan": scan, "seed_difference_matched": seed_matched or "absent: "
            + rel(SEED_DIFF), "seconds_per_phase_b_step_v1": float(lv2["v1"]),
            "attainable": all(v["below_line"] for v in succ.values())
            and all(scan[k]["smallest_share_meeting"] is not None for k in scan)}


def v1_last_record(weights_b=None):
    """v1's last phase B log record (the log's per-window means over the interval's windows): its step, the loss
    (per-window total, weighted), sup_root_rate and its parts u and yaw (diagnostics), hsic_active, and the phase B
    weighted value of each of COMPARE_TERMS (term x its phase B weight of configs/stage1_pilot.yaml)."""
    last = None
    if os.path.isfile(V1_LOG):
        with open(V1_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("event") == "log" and d.get("phase") == "B":
                    last = d
    if last is None:
        return None
    w = weights_b or {}
    out = {"step": int(last["step"]), "loss": float(last["loss"]),
           "sup_root_rate_u": float(last["diagnostics"]["sup_root_rate_u"]),
           "sup_root_rate_yaw": float(last["diagnostics"]["sup_root_rate_yaw"]),
           "sup_root_rate": float(last["terms"]["sup_root_rate"]), "hsic_active": float(last["hsic_active"]),
           "weights_b": {t: w.get(t) for t in COMPARE_TERMS}}
    out["weighted"] = {t: float(last["terms"][t]) * float(w[t]) for t in COMPARE_TERMS if w.get(t) is not None}
    return out


def weight_arithmetic(scopes, tot, n_micro, g_v, sd=0.1, win_s=10.0, weights_b=None):
    """Size of the DRAFT's term at the start of the run (arithmetic, not a measurement): per scope, the speed error at
    the train54 pass-speed gain g_v on the median segment speed, its value per knot ((err)^2 / (2 sd^2)), per
    micro-step (anchored knots), per window as the training log averages (over the micro-step's windows), per anchored
    window at the median anchored share of a window, and the per-window value over v1's last logged values: the
    sup_root_rate total and its u and yaw parts, the weighted imu_nll, sup_deriv and hsic, and the loss."""
    ref = v1_last_record(weights_b)
    wpm = tot["windows"] / max(1, n_micro)
    out = {"g_v": g_v, "sd_m_s": sd, "windows_per_micro_step": wpm, "v1_last_log": ref}
    for k, v in scopes.items():
        speed = v["segment_speed_m_s"]["p50"]
        err = (1.0 - g_v) * speed
        pk = err * err / (2.0 * sd * sd)
        pm = v["anchored_seconds_per_micro_step"] * RATE * pk
        pw = pm / wpm
        out[k] = {"median_segment_speed_m_s": speed, "speed_error_m_s": err, "per_knot": pk, "per_micro_step": pm,
                  "per_window_log_mean": pw,
                  "per_anchored_window": v["anchored_share_of_window_seconds"]["p50"] * win_s * RATE * pk,
                  "per_window_over_v1_sup_root_rate_u": pw / ref["sup_root_rate_u"] if ref else None}
        if ref:
            over = {"sup_root_rate": pw / ref["sup_root_rate"], "sup_root_rate_u": pw / ref["sup_root_rate_u"],
                    "sup_root_rate_yaw": pw / ref["sup_root_rate_yaw"], "loss": pw / ref["loss"]}
            over.update({f"{t}_weighted": pw / x for t, x in ref["weighted"].items() if x})
            out[k]["per_window_over_v1"] = over
    return out


def label_free_names():
    """label_key of mdfm.train.losses on the DRAFT's target field names (each must read False) and on two control names
    (each must read True)."""
    from mdfm.train.losses import label_key
    out = {n: bool(label_key(n)) for n in NAMES_TO_CHECK}
    ctl = {n: bool(label_key(n)) for n in ("anchor_group", "walk_phase")}
    return {"draft_names": out, "controls": ctl, "ok": not any(out.values()) and all(ctl.values())}


# ---------------------------------------------------------------- run
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--micro-steps", type=int, default=None)
    p.add_argument("--no-records", action="store_true")
    p.add_argument("--out", default=OUT)
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        sys.exit("run with CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    started = _dt.datetime.now().isoformat(timespec="seconds")
    train, dev14, dev12, locked, sets = person_sets()
    folds_pd, pd_all = reviewed_pd_folds(train, dev14, locked)
    sets["proposed_pd_folds"] = folds_pd
    passes, passes_g3, trials, first, tcounts = read_targets(train)
    cfg, ws, smp, n_full = make_sampler()
    n_micro = n_full if a.micro_steps is None else int(a.micro_steps)
    t0 = time.time()
    rows, micro, tot = replay(smp, n_micro)
    t_replay = time.time() - t0
    sets["replayed_weargait_persons"] = guard_replayed(rows, train, set(dev14) | set(locked) | pd_all)
    table = anchor_table(rows, ws, passes)
    # replay check against v1's log
    check = {"log": rel(V1_LOG)} | replay_check(micro, v1_log_records())
    clock = store_clock(ws, train, trials)
    first_t = [first.get(r["trial_id"]) for r in trials]
    clock.update({"trials": len(trials), "trials_first_sample_time_0": int(sum(1 for x in first_t if x == 0.0)),
                  "trials_without_phase_row": int(sum(1 for x in first_t if x is None)),
                  "dt_irregular_trials": int(sum(1 for r in trials if r["dt_irregular"] not in ("0", "", "0.0"))),
                  "time_nonfinite_trials": int(sum(1 for r in trials if r["time_nonfinite"] not in ("0", "", "0.0")))})
    align, align_rows = None, []
    if not a.no_records:
        align, align_rows = shank_alignment(train, G3_TASKS + MAT_TASKS)
    scopes = {s: scope_summary(table, t, tot, n_micro) for s, t in SCOPES.items()}
    pp = per_person_rows(table, train, ws, passes)
    pt = per_task_rows(table, passes, ws, train)
    exp = {s: float(np.mean([r[f"{s}_expected_share"] for r in pp])) for s in SCOPES}
    att = attainability(dev14, dev12, locked)
    names = label_free_names()
    psc = pass_speed_check(passes_g3, train)
    trf = transfer_references(dev14, dev12, locked)
    weights_b = {t: float(cfg.losses.weights[t][1]) for t in COMPARE_TERMS}
    res = {"what": LABEL, "task": TASK, "run_started": started, "sets": sets, "targets": tcounts,
           "config": {"path": rel(CONFIG), "seed_phase_b": smp.seed, "micro_steps_phase_b": n_full,
                      "micro_steps_replayed": n_micro, "data_sha256": smp.data_sha256(),
                      "config_sha256": smp.config_sha256()},
           "replay": {"totals": tot, "seconds": t_replay, "check_against_v1_log": check},
           "clock": clock, "shank_alignment": align, "pass_speed_check": psc, "scopes": scopes,
           "expected_share_analytic": exp,
           "per_person_anchored": {s: _q([r[f"{s}_anchored"] for r in pp], (0, 10, 50, 90, 100)) for s in SCOPES},
           "per_person_share": {s: _q([r[f"{s}_share"] for r in pp], (0, 10, 50, 90, 100)) for s in SCOPES},
           "per_person_segments": {s: _q([r[f"{s}_segments"] for r in pp], (0, 10, 50, 90, 100)) for s in SCOPES},
           "attainability": att, "label_free_names": names, "transfer_references": trf,
           "weight_arithmetic": weight_arithmetic(scopes, tot, n_micro, att["gain_scan"]["V12"]["g_train54"],
                                                  weights_b=weights_b),
           "cost": {"phase_b_steps": int(n_full // int(cfg.schedule.micro_per_step)),
                    "seconds_per_step_v1": att["seconds_per_phase_b_step_v1"],
                    "gpu_hours": n_full // int(cfg.schedule.micro_per_step) * att["seconds_per_phase_b_step_v1"] / 3600.0},
           "lockbox": {"persons_in_sets": 0}, "timing_s": time.time() - t_start}
    _write(a, res, pp, pt, align_rows)
    print(json.dumps({"scopes": {s: {k: scopes[s][k] for k in ("anchored_windows", "share_of_weargait_windows",
                                                                   "share_of_all_phase_b_tokens")} for s in scopes},
                      "expected": exp, "timing_s": res["timing_s"]}, indent=1))


def _write(a, res, pp, pt, align_rows):
    head = _moveatlas_head(a.moveatlas)
    res["moveatlas_head"] = head
    for name, rows in (("per_person.csv", pp), ("per_task.csv", pt), ("alignment_trials.csv", align_rows)):
        if not rows:
            continue
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summary_md(res, pt))
    prov = {"script": "scripts/probe_walkway_anchor.py", "task": TASK, "label": LABEL, "moveatlas_root": a.moveatlas,
            "moveatlas_head": head, "run_at": res["run_started"], "python": sys.version.split()[0],
            "inputs": {rel(x): _sha_file(x) for x in (GAIN_RESULTS, SEEDS, LEVER2_RESULTS, CONFIG, FOLDS_REVIEWED)
                       if os.path.isfile(x)}}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)


def _moveatlas_head(root):
    import subprocess
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or "(git failed)"
    except Exception as e:  # noqa: BLE001
        return f"(git unavailable: {e})"


def _f(x, nd=3):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def summary_md(res, pt):
    s, att, c = res["scopes"], res["attainability"], res["clock"]
    L = [f"# Walkway-anchor feasibility probe (task {TASK})", "",
         f"Label: {LABEL}. Written by scripts/probe_walkway_anchor.py; every number below is read from results.json of "
         "this folder. No lockbox, EVAL, CODY or care_pd record was read; the anchor values come from the 54 "
         "training-fold controls (train54) only.", "",
         "## Sets and replay", "",
         f"- train54 SHA-256 {res['sets']['train54_sha256']} (equal to F1's and to the window store's train fold); "
         f"lockbox persons in the sets 0. Proposed PD folds ({res['sets']['proposed_pd_folds']['path']}): persons "
         + ", ".join(f"{k} {v}" for k, v in res['sets']['proposed_pd_folds']['persons'].items())
         + "; overlap with train54, dev14 and the lockbox 0. Replayed weargait_pd persons "
         f"{res['sets']['replayed_weargait_persons']}, all of train54 and none of dev14, the lockbox or those folds.",
         f"- Cell size of the anchor, the builder's rule refit on train54 SelfPace trials: "
         f"{res['targets']['cell_fit_train54']['cell_cm']:.5f} cm (n {res['targets']['cell_fit_train54']['n']} trials, "
         f"{res['targets']['cell_fit_train54']['n_persons']} persons, p10 {res['targets']['cell_fit_train54']['p10']:.4f}, "
         f"p90 {res['targets']['cell_fit_train54']['p90']:.4f}); the G3 cell of summary.json (all non-lockbox persons) "
         f"{res['targets']['cell_cm_g3_summary']:.5f} cm; relative difference "
         f"{res['targets']['cell_relative_difference']:.6f}. Anchor targets use the train54 cell; the pass-speed check "
         "uses the G3 cell.",
         f"- Sampler: configs/stage1_pilot.yaml, phase B seed {res['config']['seed_phase_b']}, "
         f"{res['config']['micro_steps_replayed']} of {res['config']['micro_steps_phase_b']} micro-steps replayed "
         f"(plan only), data SHA-256 {res['config']['data_sha256']}.",
         f"- Totals of the replay: {res['replay']['totals']['windows']} windows, {res['replay']['totals']['tokens']} "
         f"tokens; weargait_pd {res['replay']['totals']['wg_windows']} windows, {res['replay']['totals']['wg_tokens']} "
         f"tokens; synthetic {res['replay']['totals']['syn_bodies']} bodies, "
         f"{res['replay']['totals']['syn_seconds']:.0f} s."]
    chk = res["replay"]["check_against_v1_log"]
    if chk.get("intervals_compared"):
        dw, ds, dg, dt = (chk[k] for k in ("windows_plan_minus_log", "weargait_share_plan_minus_log",
                                            "weargait_tokens_plan_minus_log", "tokens_plan_minus_log_useful"))
        L.append(f"- Against v1's log: {chk['intervals_compared']} intervals ({chk['intervals_partial']} partial), "
                 f"{chk['micro_steps_compared']} of {chk['micro_steps_planned']} planned micro-steps. Windows plan minus "
                 f"log: min {dw['p0']:.0f}, median {dw['p50']:.0f}, max {dw['p100']:.0f}, mean {dw['mean']:.4f}; "
                 f"intervals that differ {chk['intervals_windows_differ']}, of which the plan has fewer "
                 f"{chk['intervals_plan_fewer_windows']}. Weargait tokens plan minus log (share x useful tokens): min "
                 f"{dg['p0']:.3g}, max {dg['p100']:.3g}; intervals equal within 0.5 token "
                 f"{chk['intervals_weargait_tokens_equal']}. Weargait share plan minus log: min {ds['p0']:.3g}, max "
                 f"{ds['p100']:.3g}; mean share plan {chk['plan_weargait_token_share_mean']:.5f}, log "
                 f"{chk['log_weargait_token_share_mean']:.5f}. All tokens plan minus log useful: min {dt['p0']:.0f}, "
                 f"median {dt['p50']:.0f}, max {dt['p100']:.0f}.")
    L += ["", "## Anchored windows", "", "| scope | anchored windows | share of weargait windows | share of weargait "
          "tokens | share of all phase B tokens | per micro-step | analytic share | anchored s per micro-step | "
          "anchored s over synthetic s |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s.items():
        L.append(f"| {k} | {v['anchored_windows']} | {v['share_of_weargait_windows']:.4f} | "
                 f"{v['share_of_weargait_tokens']:.4f} | {v['share_of_all_phase_b_tokens']:.4f} | "
                 f"{v['anchored_windows_per_micro_step']:.3f} | {res['expected_share_analytic'][k]:.4f} | "
                 f"{v['anchored_seconds_per_micro_step']:.2f} | {_f(v['anchored_seconds_over_synthetic_seconds'], 4)} |")
    L += ["", "Per person (54): anchored windows and share of the person's weargait windows.", "",
          "| scope | anchored min | p10 | median | p90 | max | share min | median | max | segments median |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in s:
        a_, sh, sg = res["per_person_anchored"][k], res["per_person_share"][k], res["per_person_segments"][k]
        L.append(f"| {k} | {a_['p0']:.0f} | {a_['p10']:.0f} | {a_['p50']:.0f} | {a_['p90']:.0f} | {a_['p100']:.0f} | "
                 f"{sh['p0']:.3f} | {sh['p50']:.3f} | {sh['p100']:.3f} | {sg['p50']:.0f} |")
    L += ["", "## Targets per anchored window (segments)", "",
          "| scope | segments | per window median (p90) | distance m p10 / median / p90 | duration s p10 / median / p90 | "
          "speed m/s p10 / median / p90 | strides p10 / median / p90 | G3 strides inside per window median | "
          "segment share of the window, median |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s.items():
        d, du, sp, st = v["segment_dist_m"], v["segment_dur_s"], v["segment_speed_m_s"], v["segment_strides"]
        L.append(f"| {k} | {v['segments']} | {v['segments_per_anchored_window']['p50']:.0f} "
                 f"({v['segments_per_anchored_window']['p90']:.0f}) | {d['p10']:.2f} / {d['p50']:.2f} / {d['p90']:.2f} | "
                 f"{du['p10']:.2f} / {du['p50']:.2f} / {du['p90']:.2f} | {sp['p10']:.3f} / {sp['p50']:.3f} / "
                 f"{sp['p90']:.3f} | {st['p10']:.0f} / {st['p50']:.0f} / {st['p90']:.0f} | "
                 f"{v['walkway_strides_per_anchored_window']['p50']:.0f} | "
                 f"{v['anchored_share_of_window_seconds']['p50']:.3f} |")
    L += ["", "Per task (every weargait window of the replay):", "",
          "| task | windows | share of weargait windows | with a segment | share | in the DRAFT's scope | passes "
          "(train54) | median segment m | median speed m/s | median strides |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in pt:
        L.append(f"| {r['task']} | {r['windows']} | {r['share_of_weargait_windows']:.4f} | {r['with_segment']} | "
                 f"{r['share_with_segment']:.3f} | {r['anchored_in_draft']} | {r['passes']} | "
                 f"{_f(r['median_segment_dist_m'], 2)} | {_f(r['median_segment_speed_m_s'], 3)} | "
                 f"{_f(r['median_segment_strides'], 0)} |")
    L += ["", "## Alignment of the walkway timeline and the window clock", "",
          f"- Recording clock against the targets' clock: {c['clock_equal_to_targets']} of {c['recordings']} train54 "
          "recordings of the target tasks start at 0 s with offset 0, one sample in one (k 1) at 100 Hz and the "
          f"sample count of trials.csv ({c['n_differ']} differ). Trials whose first sample has Time 0 (phases.csv): "
          f"{c['trials_first_sample_time_0']} of {c['trials']}; irregular sample interval {c['dt_irregular_trials']}; "
          f"non-finite Time {c['time_nonfinite_trials']}.",
          f"- Pass speed rule: the any-foot segment of a whole pass equals mdfm.gates.g3's pass speed on "
          f"{res['pass_speed_check']['passes_compared']} train54 passes, largest difference "
          f"{_f(res['pass_speed_check']['largest_abs_diff_m_s'], 9)} m/s."]
    al = res["shank_alignment"]
    if al:
        L.append(f"- Shank-gyroscope ICs against walkway ICs (train54, {', '.join(al['tasks'])}): {al['matched']} of "
                 f"{al['walkway_ic']} walkway ICs matched within 0.3 s ({al['matched_share']:.3f}) over {al['trials']} "
                 f"trials; pooled median {al['pooled_median_ms']:.1f} ms; per-trial median p10 / p50 / p90 "
                 f"{al['per_trial_median_ms']['p10']:.1f} / {al['per_trial_median_ms']['p50']:.1f} / "
                 f"{al['per_trial_median_ms']['p90']:.1f} ms; |per-trial minus pooled| p50 / p90 / max "
                 f"{al['abs_away_from_pooled_ms']['p50']:.1f} / {al['abs_away_from_pooled_ms']['p90']:.1f} / "
                 f"{al['abs_away_from_pooled_ms']['p100']:.1f} ms; trials beyond 50 ms {al['trials_away_over_50ms']}, "
                 f"beyond 100 ms {al['trials_away_over_100ms']}; trials without a match {al['trials_without_match']}"
                 + ("; beyond 50 ms: " + ", ".join(f"{x['trial_id']} (median {x['median_diff_ms']:.0f} ms, "
                                                    f"{x['matched']} of {x['walkway_ic']} matched)"
                                                    for x in al["trials_away_over_50ms_list"])
                    if al["trials_away_over_50ms_list"] else "") + ".")
    else:
        L.append("- Shank-gyroscope check not run (--no-records).")
    L += ["", "## Attainability (lever-2 rule form; v1_B reference)", ""]
    for k, v in att["success"].items():
        sc = att["gain_scan"][k]
        sm = sc["smallest_share_meeting"]
        L.append(f"- {k}: v1 {v['line']:.4f}, seed difference {v['seed_diff']:.4f}, margin {v['margin']:.4f} "
                 f"(10 percent {v['ten_percent']:.4f}, 2 x d {v['two_seed']:.4f}), so at most {v['needed_line_at_most']:.4f}. "
                 f"Calibration head (gain {sc['g_train54']:.4f} closed fully): line {sc['head_line']:.4f}, improvement "
                 f"{sc['head_improvement']:.4f} [{sc['head_interval'][0]:.4f}, {sc['head_interval'][1]:.4f}]. "
                 f"Smallest share of the gain gap closed that meets the margin with a lower bound above 0: "
                 + (f"{sm['f']:.2f} (gain {sm['gain']:.4f}, line {sm['line']:.4f}, improvement {sm['improvement']:.4f} "
                    f"[{sm['interval'][0]:.4f}, {sm['interval'][1]:.4f}])" if sm else "none") + ".")
    L.append("- Non-inferiority margins, max(0.05 x line, 2 x d): " + "; ".join(
        f"{k} {v['margin']:.4f} (line {v['line']:.4f}, d {v['seed_diff']:.4f})" for k, v in att["non_inferiority"].items())
        + ".")
    sm = att["seed_difference_matched"]
    L.append(f"- Matched-settings seed differences (scripts/seed_difference.py): "
             + (sm if isinstance(sm, str) else f"{sm['path']} present, SHA-256 {sm['sha256']}") + ".")
    L.append(f"- Cost: {res['cost']['phase_b_steps']} phase B steps x {res['cost']['seconds_per_step_v1']:.4f} s "
             f"(v1, reports/bench/f2_lever2/results.json) = {res['cost']['gpu_hours']:.2f} GPU-h.")
    wa = res["weight_arithmetic"]
    ref = wa["v1_last_log"]
    L += ["", "## Size of the term at the start (arithmetic, weight 1.0, sd 0.1 m/s)", "",
          f"Speed error at the train54 pass-speed gain {wa['g_v']:.4f} on the median segment speed; "
          f"{wa['windows_per_micro_step']:.2f} windows per micro-step; v1's last phase B log record (step "
          f"{ref['step'] if ref else 'n/a'}): sup_root_rate_u {_f(ref['sup_root_rate_u'] if ref else None, 1)}.", "",
          "| scope | median segment speed m/s | speed error m/s | per knot | per micro-step | per window (log mean) | "
          "per anchored window | per window over v1's sup_root_rate_u |", "|---|---|---|---|---|---|---|---|"]
    for k in SCOPES:
        x = wa[k]
        L.append(f"| {k} | {x['median_segment_speed_m_s']:.3f} | {x['speed_error_m_s']:.3f} | {x['per_knot']:.2f} | "
                 f"{x['per_micro_step']:.1f} | {x['per_window_log_mean']:.2f} | {x['per_anchored_window']:.0f} | "
                 f"{_f(x['per_window_over_v1_sup_root_rate_u'], 4)} |")
    if ref:
        L += ["", f"v1's last phase B record, per window (log means): loss {ref['loss']:.1f}; sup_root_rate "
              f"{ref['sup_root_rate']:.1f} (u {ref['sup_root_rate_u']:.1f}, yaw {ref['sup_root_rate_yaw']:.1f}); weighted "
              + ", ".join(f"{t} {v:.4g} (weight {ref['weights_b'][t]})" for t, v in ref["weighted"].items())
              + f"; hsic_active {ref['hsic_active']:.3f}. The anchor's per-window value over each:", "",
              "| scope | " + " | ".join(wa[next(iter(SCOPES))]["per_window_over_v1"]) + " |",
              "|---|" + "---|" * len(wa[next(iter(SCOPES))]["per_window_over_v1"])]
        for k in SCOPES:
            L.append(f"| {k} | " + " | ".join(f"{v:.5f}" for v in wa[k]["per_window_over_v1"].values()) + " |")
    tr = res["transfer_references"]
    th = tr["thresholds"]
    L += ["", "## Transfer references (dev diagnostic; the DRAFT's T lines)", "",
          f"- From F1: d_gap {th['d_gap']:.4f} (train54-minus-dev12 stride gain gap, v1_B "
          f"{th['ends']['v1_B']['gap_stride']:.4f}, s2_B {th['ends']['s2_B']['gap_stride']:.4f}); d_dev12_stride "
          f"{th['d_dev12_stride']:.4f} (v1_B {th['ends']['v1_B']['dev12_stride']:.4f}, s2_B "
          f"{th['ends']['s2_B']['dev12_stride']:.4f}); d_dev12_speed {th['d_dev12_speed']:.4f}; d_insim "
          f"{th['d_insim']:.4f} (v1_B {th['ends']['v1_B']['insim_stride']:.4f}, s2_B "
          f"{th['ends']['s2_B']['insim_stride']:.4f}).",
          f"- Default diag of v1_B, base row: dev12 stride gain {_f(tr['v1_B_default_diag_base']['stride_dev12_gain'], 4)}, "
          f"speed gain {_f(tr['v1_B_default_diag_base']['speed_dev12_gain'], 4)}.", "",
          "| v1_B ablations row | stride dev14 persons | stride dev12 gain | speed dev14 persons | speed dev12 gain |",
          "|---|---|---|---|---|"]
    for row, r in tr["v1_B_ablation_rows"].items():
        L.append(f"| {row} | {r['stride']['dev14_persons']} | {_f(r['stride']['dev12_gain'], 4)} | "
                 f"{r['speed']['dev14_persons']} | {_f(r['speed']['dev12_gain'], 4)} |")
    L += ["", "Gate inputs: " + "; ".join(
        f"{k} {v['path']} present {v['present']}" + (f", result {v['result_present']}" if "result_present" in v else "")
        + (f", code {v['code_sha256'][:12]}" if v.get("code_sha256") else "") for k, v in tr["gate_inputs"].items())
        + "."]
    nm = res["label_free_names"]
    L += ["", "## Label-free names", "",
          "mdfm.train.losses.label_key on the DRAFT's target field names: " + ", ".join(
              f"{k} {v}" for k, v in nm["draft_names"].items()) + "; controls (must read True): " + ", ".join(
              f"{k} {v}" for k, v in nm["controls"].items()) + f". OK: {nm['ok']}.", "",
          f"MoveAtlas HEAD: {res.get('moveatlas_head', 'see _provenance.json')}. Time {res['timing_s']:.0f} s.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
