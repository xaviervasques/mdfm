"""pd_meg_stn_lfp_ds004998: header-pass inventory of the OpenNeuro ds004998 raw tree.

Task T24 of the weekend plan of 2026-10-01. It checks section 1.5 of
docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md and feeds NR-2.12, NR-6.4.4, OPEN-12, OPEN-43 and
OPEN-80 of docs/rules/NEURAL_RULES_DRAFT.md.

Source (read-only): MoveAtlas datasets/eeg/pd_meg_stn_lfp_ds004998/raw/, a BIDS 1.6 tree of
Elekta FIF files, one session (ses-PeriOp) per subject. Headers only: every FIF is opened with
mne.io.read_raw_fif(preload=False), which reads the measurement info, the annotations stored in
the file and the directory of data buffers, never a data sample. The BIDS side files
(participants.tsv, scans.tsv, events.tsv, channels.tsv, meg.json, montage.tsv) are read as text.

What it measures:
  1. Split parts (the header half of the G15 line, NR-2.12). The FIF files of one BIDS run (same
     sub, ses, task, acq and run entities) that differ only by the split entity are merged into
     the run. The parts must be numbered 1..n, carry one meas_date and one sampling rate, and abut
     on the acquisition sample clock: first_samp of part k+1 = last_samp of part k + 1. Data-skip
     entries of the buffer directory (MNE's BAD_ACQ_SKIP) are counted per part. The run's sample
     count is compared with RecordingDuration of its meg.json and its channel list with its
     channels.tsv. Every channel of a FIF (MEG, the LFP contacts stored as EEG0xx, EMG, EOG,
     triggers) shares the file's data buffers, so all of them have one sample count per run by
     construction; what can break the line is a gap, an overlap, a missing part or a skip.
  2. Acquisitions. Runs of one subject with the same meas_date come from one recording; their
     sample ranges are compared (identical, contained, overlapping, disjoint). A run contained in
     another one repeats samples of it and is not a separate acquisition. Where such copies start
     gives the size of the first acquisition file per channel count; every other recording is
     compared with it (shorter, equal, longer), since one exactly a file long may be truncated.
  3. State-by-task table: BIDS runs and acquisitions per acq entity (MedOff, MedOn) and task,
     against addendum 1.5. These run counts are the only quantity split by medication state.
  4. Order census per subject: units sorted by acq_time (scans.tsv); whether every MedOff unit
     precedes every MedOn unit; the gap between the blocks; the feasibility of the order-only
     null of NR-6.4.4 under both readings of OPEN-43 (task class hold, move, rest; or the
     side-specific task label), counted over BIDS runs (the unit of addendum 1.5) and over
     acquisitions with distinct acq_time.
  5. Contact map from montage.tsv, checked against the FIF channel names; the EMG channels and
     whether any file on disk names a muscle.
  6. Rest and task segments from events.tsv (run_events): when the events.tsv of the first part
     reaches past the part's end it holds the run's events, onsets from the run's first sample,
     and the later parts' own events.tsv are compared with it; otherwise the parts' files are
     joined on the sample clock and the continuations across a boundary merged. The run events
     are checked against the annotations stored in each part's FIF. segments.csv has one row per
     acquisition, keyed by subject, task and a blind key (a hash of the recording's own events,
     never of its run name), with no state, run or part column, so that no segment quantity is
     split by MedOff or MedOn and no row joins the run list; segments_by_task.csv pools all
     acquisitions of a task; runs.csv carries event checks, never event counts.
  7. Empty-room runs (task-noise); subjects of participants.tsv missing on disk.
  8. Person map (OPEN-80): the source's subject id is the person id; evidence from the folder,
     participants.tsv and the fiducial geometry of the FIF digitization (within subject across
     runs, nearest other subject).

Outputs (reports/ds004998/ by default):
  files.csv                one row per FIF file: entities, size, sample range, buffers, skips
  runs.csv                 one row per BIDS run: parts, continuity, skips, G15 header verdict,
                           acquisition relation, acq_time, events QA
  acquisitions.csv         one row per acquisition (subject, meas_date) and its runs
  state_task_counts.csv    runs and acquisitions per acq entity and task
  order_census.csv         one row per subject: order of the MedOff and MedOn units, gaps
  order_null_feasibility.csv  one row per subject, unit and reading of OPEN-43
  contact_map.csv          montage rows checked against the FIF channel names
  contact_map_subjects.csv contacts per side, FIF EEG channels outside the montage, per subject
  emg_channels.csv         EMG and MISC channels per run
  segments.csv             rest and task segments per acquisition (blind key, no state)
  segments_by_task.csv     the same pooled per task over all acquisitions
  event_labels.csv         trial_type census over every events.tsv
  empty_room.csv           task-noise runs
  subjects.csv             participants.tsv against the disk
  admitted_runs.csv        draft input to OPEN-12: runs passing the header line, one per
                           acquisition
  addendum_checks.csv      each statement of the addendum on ds004998, measured value, verdict
  summary.md               the same in prose
  cache/                   one JSON of header facts per FIF (resume; gitignored)
  _provenance.json         the MoveAtlas stamp of scripts/moveatlas_stamp.py
  and reports/person_ids/map_pd_meg_stn_lfp_ds004998.csv (--person-map).

No spectral quantity, no signal statistic, and no quantity split by medication state other than
run counts. The script pauses while cache/runs/queue/SLOW exists and lowers its own priority to
below normal on Windows.

Run: python scripts/probe_ds004998.py [--moveatlas <moveatlas>] [--out DIR]
         [--person-map CSV] [--force]
"""
import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_REL = "datasets/eeg/pd_meg_stn_lfp_ds004998/raw"
SOURCE = "pd_meg_stn_lfp_ds004998"
CACHE_VERSION = "probe_ds004998 v1"
SLOW_FLAG = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
BLIND_SALT = "ds004998-segments-v2|"
MUSCLE_WORDS = ("extensor", "flexor", "muscle", "muscul", "deltoid", "biceps", "triceps", "carpi",
                "digitorum", "brachioradialis", "interosse", "tibialis", "gastrocnem")

BIDS_RE = re.compile(
    r"^sub-(?P<sub>[A-Za-z0-9]+)_ses-(?P<ses>[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)"
    r"(?:_acq-(?P<acq>[A-Za-z0-9]+))?(?:_run-(?P<run>\d+))?(?:_split-(?P<split>\d+))?"
    r"_(?P<suffix>[A-Za-z0-9]+)\.(?P<ext>[A-Za-z0-9.]+)$")

# Addendum 1.5 (docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md), the statements checked here.
ADDENDUM_STATE_TASK = {("MedOff", "HoldL"): 16, ("MedOff", "HoldR"): 9, ("MedOff", "MoveL"): 20,
                       ("MedOff", "MoveR"): 8, ("MedOff", "Rest"): 3,
                       ("MedOn", "HoldL"): 17, ("MedOn", "HoldR"): 9, ("MedOn", "MoveL"): 16,
                       ("MedOn", "MoveR"): 6, ("MedOn", "Rest"): 4}
ADDENDUM_MISSING = ("BgojEx", "zxEhes")


# ----------------------------------------------------------------------------------------------
# Pure functions (tested in tests/test_probe_ds004998.py)
# ----------------------------------------------------------------------------------------------

def parse_bids_name(name):
    """BIDS entities of a file name, or None (AppleDouble sidecars and other names)."""
    base = os.path.basename(name)
    if base.startswith("._"):
        return None
    m = BIDS_RE.match(base)
    if not m:
        return None
    return m.groupdict()


def run_id(ent):
    """The run identity: the file name entities without split, suffix and extension."""
    s = f"sub-{ent['sub']}_ses-{ent['ses']}_task-{ent['task']}"
    if ent.get("acq"):
        s += f"_acq-{ent['acq']}"
    if ent.get("run"):
        s += f"_run-{ent['run']}"
    return s


def task_class(task):
    """OPEN-43, first reading: hold, move or rest (side dropped); None for the empty room."""
    t = task.lower()
    if t.startswith("hold"):
        return "hold"
    if t.startswith("move"):
        return "move"
    if t.startswith("rest"):
        return "rest"
    return None


def merge_split_parts(parts):
    """Merge the FIF parts of a listing into BIDS runs.

    parts: iterable of dicts with 'name' (file name) and, when headers were read, 'first_samp',
    'last_samp', 'sfreq', 'meas_date' and optionally 'skip_entries', 'skip_samples'. Names that
    are not BIDS *_meg.fif files are ignored.

    Returns one dict per run, sorted by run id, with: run_id, sub, ses, task, acq, run, parts
    (names in split order), split_numbers, n_parts, problems (list of strings), continuity
    ('single', 'contiguous', 'gap', 'overlap', 'missing_part', 'mixed', 'duplicate_part',
    'header_mismatch' or 'no_header'), joins (first_samp of part k+1 - last_samp of part k - 1,
    per join: 0 abuts, > 0 a gap, < 0 an overlap), and, with headers, first_samp, last_samp,
    span_samples, n_samples (sum of part lengths), meas_date, sfreq, skip_entries, skip_samples.
    """
    groups = defaultdict(list)
    for p in parts:
        ent = parse_bids_name(p["name"])
        if ent is None or ent["suffix"] != "meg" or ent["ext"] != "fif":
            continue
        groups[run_id(ent)].append((ent, p))
    runs = []
    for rid in sorted(groups):
        items = groups[rid]
        ent0 = items[0][0]
        problems = []
        splits = [e["split"] for e, _ in items]
        n_unsplit = sum(s is None for s in splits)
        nums = [int(s) for s in splits if s is not None]
        status = None
        if n_unsplit and nums:
            status = "mixed"
            problems.append("split and unsplit files share the run")
        elif len(set(nums)) != len(nums):
            status = "duplicate_part"
            problems.append("a split number repeats")
        elif nums and sorted(nums) != list(range(1, max(nums) + 1)):
            status = "missing_part"
            missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
            problems.append("missing split " + ",".join(f"{k:02d}" for k in missing))
        ordered = sorted(items, key=lambda x: (x[0]["split"] is not None, int(x[0]["split"] or 0)))
        run = {"run_id": rid, "sub": ent0["sub"], "ses": ent0["ses"], "task": ent0["task"],
               "acq": ent0["acq"] or "", "run": ent0["run"] or "",
               "parts": [p["name"] for _, p in ordered],
               "split_numbers": [e["split"] or "" for e, _ in ordered],
               "n_parts": len(ordered), "joins": [], "problems": problems}
        have = all("first_samp" in p and p["first_samp"] is not None for _, p in ordered)
        if have:
            firsts = [int(p["first_samp"]) for _, p in ordered]
            lasts = [int(p["last_samp"]) for _, p in ordered]
            dates = {str(p.get("meas_date")) for _, p in ordered}
            rates = {float(p.get("sfreq")) for _, p in ordered}
            joins = [firsts[k + 1] - lasts[k] - 1 for k in range(len(ordered) - 1)]
            run.update({"joins": joins, "first_samp": firsts[0], "last_samp": lasts[-1],
                        "span_samples": lasts[-1] - firsts[0] + 1,
                        "n_samples": sum(b - a + 1 for a, b in zip(firsts, lasts)),
                        "meas_date": sorted(dates)[0] if len(dates) == 1 else "|".join(sorted(dates)),
                        "sfreq": sorted(rates)[0],
                        "skip_entries": sum(int(p.get("skip_entries", 0) or 0) for _, p in ordered),
                        "skip_samples": sum(int(p.get("skip_samples", 0) or 0) for _, p in ordered)})
            if status is None:
                if len(dates) > 1 or len(rates) > 1:
                    status = "header_mismatch"
                    problems.append("parts differ in meas_date or sampling rate")
                elif any(j < 0 for j in joins):
                    status = "overlap"
                    problems.append("parts overlap by " + ",".join(str(-j) for j in joins if j < 0) + " samples")
                elif any(j > 0 for j in joins):
                    status = "gap"
                    problems.append("gap of " + ",".join(str(j) for j in joins if j > 0) + " samples between parts")
                else:
                    status = "single" if len(ordered) == 1 else "contiguous"
        elif status is None:
            status = "no_header"
        run["continuity"] = status
        runs.append(run)
    return runs


def range_relation(a0, a1, b0, b1):
    """Relation of sample range [a0, a1] to [b0, b1] (inclusive)."""
    if a0 == b0 and a1 == b1:
        return "identical"
    if b0 <= a0 and a1 <= b1:
        return "contained"
    if a0 <= b0 and b1 <= a1:
        return "contains"
    if a1 < b0 or b1 < a0:
        return "disjoint"
    return "overlapping"


def group_acquisitions(runs):
    """Group runs with headers by (sub, meas_date). The primary run of a group is the one with
    the longest span (ties: lowest run number, then name). Each run gets 'acquisition_id',
    'relation_to_primary' ('primary', 'identical', 'contained', 'overlapping', 'disjoint',
    'contains') and 'primary_run_id'; 'separate_acquisition' is True for the primary and for a
    run disjoint from every other run of the group. Returns the groups as
    {acquisition_id: [runs, primary first]}."""
    groups = defaultdict(list)
    for r in runs:
        if r.get("first_samp") is None:
            continue
        groups[(r["sub"], r["meas_date"])].append(r)
    out = {}
    for (sub, md), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: (-r["span_samples"], int(r["run"] or 0), r["run_id"]))
        prim = rs[0]
        aid = f"sub-{sub}@{md}"
        for r in rs:
            r["acquisition_id"] = aid
            r["primary_run_id"] = prim["run_id"]
            if r is prim:
                r["relation_to_primary"] = "primary"
                r["separate_acquisition"] = True
                continue
            rel = range_relation(r["first_samp"], r["last_samp"], prim["first_samp"], prim["last_samp"])
            r["relation_to_primary"] = rel
            others = [o for o in rs if o is not r]
            r["separate_acquisition"] = all(
                range_relation(r["first_samp"], r["last_samp"], o["first_samp"], o["last_samp"]) == "disjoint"
                for o in others)
        out[aid] = rs
    return out


def order_census(units):
    """Order of MedOff and MedOn units of one subject.

    units: list of dicts with 'state' ('MedOff' or 'MedOn'), 'start' and 'end' (seconds on one
    clock), 'label'. Returns n_off, n_on, all_off_before_on (every MedOff unit starts before
    every MedOn unit starts), interleaved, the start-to-start gap (first MedOn start - last
    MedOff start) and the end-to-start gap (first MedOn start - end of the last MedOff unit),
    in seconds, and the sequence of labels in start order. With one state missing the gaps are
    None and all_off_before_on is None.
    """
    off = [u for u in units if u["state"] == "MedOff"]
    on = [u for u in units if u["state"] == "MedOn"]
    seq = [u["label"] for u in sorted(units, key=lambda u: (u["start"], u["label"]))]
    res = {"n_off": len(off), "n_on": len(on), "sequence": seq, "all_off_before_on": None,
           "interleaved": None, "gap_start_to_start_s": None, "gap_end_to_start_s": None,
           "off_after_first_on": []}
    if not off or not on:
        return res
    last_off = max(off, key=lambda u: u["start"])
    first_on = min(on, key=lambda u: u["start"])
    before = last_off["start"] < first_on["start"]
    res["all_off_before_on"] = before
    res["interleaved"] = not before
    res["off_after_first_on"] = [u["label"] for u in sorted(off, key=lambda u: u["start"])
                                 if u["start"] > first_on["start"]]
    res["gap_start_to_start_s"] = first_on["start"] - last_off["start"]
    res["gap_end_to_start_s"] = first_on["start"] - max(u["end"] for u in off)
    return res


def order_null_feasibility(units, reading):
    """NR-6.4.4 feasibility for one subject under one reading of OPEN-43.

    units: dicts with 'state' and 'task' (side-specific label, e.g. HoldL) and 'start'.
    reading 'class': tasks matched as hold, move, rest; 'side': matched on the label itself.
    Only MedOff units with distinct start times count as ordered: units sharing a start time
    cannot be ordered by acq_time. Returns per task key the MedOff count and the count of
    distinct start times; feasible: some key has at least two ordered MedOff units;
    feasible_counting_ties: some key has at least two MedOff units, ordered or not (the count of
    addendum 1.5); any_task_two_or_more (ordered) and any_task_two_or_more_counting_ties: at
    least two MedOff units over all tasks.
    """
    key = (lambda t: task_class(t)) if reading == "class" else (lambda t: t)
    by = defaultdict(list)
    for u in units:
        if u["state"] != "MedOff" or task_class(u["task"]) is None:
            continue
        by[key(u["task"])].append(u["start"])
    per_key = {k: {"n": len(v), "n_distinct_start": len(set(v))} for k, v in sorted(by.items())}
    all_starts = [s for v in by.values() for s in v]
    return {"per_key": per_key,
            "feasible": any(d["n_distinct_start"] >= 2 for d in per_key.values()),
            "feasible_counting_ties": any(d["n"] >= 2 for d in per_key.values()),
            "any_task_two_or_more": len(set(all_starts)) >= 2,
            "any_task_two_or_more_counting_ties": len(all_starts) >= 2}


def merge_part_events(part_events, part_starts_s, tol_s=0.5):
    """Put the events of the parts of a run on one clock and merge part-boundary continuations.

    part_events: per part, a list of (onset_s, duration_s, label) relative to the part's first
    sample, or None when the part has no events file. part_starts_s: the part's first sample in
    seconds on the acquisition clock. An event of part k+1 that starts within tol_s of the part
    start continues an event of part k with the same label whose end reaches the boundary
    (within tol_s); the two are merged (end = the later end). Returns (events, n_merged), events
    being (start_s, end_s, label) on the acquisition clock sorted by start.
    """
    merged = []
    n_merged = 0
    for k, evs in enumerate(part_events):
        if evs is None:
            continue
        p0 = part_starts_s[k]
        for on, dur, lab in evs:
            start, end = p0 + float(on), p0 + float(on) + float(dur)
            if k > 0 and float(on) <= tol_s:
                cands = [i for i, (s, e, l) in enumerate(merged)
                         if l == lab and s < p0 and e >= p0 - tol_s]
                if cands:
                    i = max(cands, key=lambda j: merged[j][1])
                    s, e, l = merged[i]
                    merged[i] = (s, max(e, end), l)
                    n_merged += 1
                    continue
            merged.append((start, end, lab))
    merged.sort(key=lambda x: (x[0], x[2]))
    return merged, n_merged


def clip_events(evs, a, b):
    """Events (start, end, label) on one clock, clipped to [a, b) and returned as (onset - a, duration,
    label), the form of an events.tsv of a part that starts at a and ends at b."""
    out = []
    for s, e, lab in evs:
        if e <= a or s >= b:
            continue
        s2, e2 = max(s, a), min(e, b)
        out.append((s2 - a, e2 - s2, lab))
    return out


def run_events(part_events, part_bounds_s, tol_s=0.5):
    """Run-level events of a run whose parts may each carry an events.tsv.

    part_events: per part, (onset_s, duration_s, label) relative to the part's first sample, or None.
    part_bounds_s: per part, (start_s, end_s) on the acquisition clock.
    When the first part's events reach past its end into the later parts, that file holds the
    run's events with onsets from the run start (the layout of ds004998, where those events also
    equal the annotations of every part's FIF) and it is the run's events; the later parts' files
    are then only compared with it. Otherwise the parts' files are joined on the clock by
    merge_part_events. Returns (events as (start, end, label) on the clock, source, comparisons),
    comparisons holding per later part with a file (index, n_unpaired, max onset difference s,
    max duration difference s, equal within 10 ms) against the run's events clipped to the part.
    """
    first = part_events[0] if part_events else None
    if first is not None and len(part_bounds_s) > 1:
        a0, b0 = part_bounds_s[0]
        reach = max((a0 + on + du for on, du, lab in first), default=a0)
        if reach > b0 + 1e-6:
            evs = sorted((a0 + on, a0 + on + du, lab) for on, du, lab in first)
            comps = []
            for k in range(1, len(part_events)):
                if part_events[k] is None:
                    continue
                ref = clip_events(evs, *part_bounds_s[k])
                d = events_diff(part_events[k], ref)
                comps.append((k, d[0], d[1], d[2], events_equal(part_events[k], ref)))
            return evs, "first_part_covers_run", comps
    evs, _ = merge_part_events(part_events, [a for a, b in part_bounds_s], tol_s=tol_s)
    return evs, ("single_part" if len(part_bounds_s) == 1 else "parts_joined"), []


def blind_key(sub, task, events):
    """Key of a segments.csv row: a hash of the subject, the task and the recording's own events
    (start, end, label; seconds from its first sample, rounded to 1 ms). The run name, which carries
    the state, does not enter it, so the key cannot be recomputed from the run names listed in
    runs.csv or admitted_runs.csv; only the events of the raw recording give it back."""
    body = ";".join(f"{float(s):.3f},{float(e):.3f},{lab}" for s, e, lab in sorted(events))
    return hashlib.sha256(f"{BLIND_SALT}{sub}|{task}|{body}".encode("utf-8")).hexdigest()[:10]


def union_intervals(iv):
    """Union of (start, end) intervals, sorted and disjoint."""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def overlap_s(a, b):
    """Overlap in seconds between the unions of two lists of (start, end) intervals."""
    tot = 0.0
    for s0, e0 in union_intervals(a):
        for s1, e1 in union_intervals(b):
            tot += max(0.0, min(e0, e1) - max(s0, s1))
    return tot


def events_equal(evs, anns, tol_s=0.01):
    """True when two lists of (onset, duration, label) hold the same events within tol_s."""
    if len(evs) != len(anns):
        return False
    pool = list(anns)
    for on, du, lab in evs:
        hit = [i for i, (o2, d2, l2) in enumerate(pool)
               if l2 == lab and abs(o2 - on) <= tol_s and abs(d2 - du) <= tol_s]
        if not hit:
            return False
        pool.pop(hit[0])
    return True


def events_diff(evs, anns):
    """Pair each event with the unused annotation of the same label nearest in onset.
    Returns (n_unpaired, max |onset difference| s, max |duration difference| s) over the pairs."""
    pool = list(anns)
    d_on = d_du = 0.0
    unpaired = 0
    for on, du, lab in sorted(evs):
        cand = [i for i, (o2, d2, l2) in enumerate(pool) if l2 == lab]
        if not cand:
            unpaired += 1
            continue
        i = min(cand, key=lambda j: abs(pool[j][0] - on))
        o2, d2, _ = pool.pop(i)
        d_on = max(d_on, abs(o2 - on))
        d_du = max(d_du, abs(d2 - du))
    return unpaired + len(pool), d_on, d_du


# ----------------------------------------------------------------------------------------------
# Reading (headers and side files)
# ----------------------------------------------------------------------------------------------

def set_below_normal():
    if os.name != "nt":
        return "not windows"
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass.restype = ctypes.c_int
        ok = k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        return "below normal" if ok else "SetPriorityClass failed"
    except Exception as e:  # noqa: BLE001
        return f"priority unchanged ({e})"


def wait_if_slow():
    announced = False
    while os.path.exists(SLOW_FLAG):
        if not announced:
            print(f"paused: {SLOW_FLAG} exists", flush=True)
            announced = True
        time.sleep(20)
    if announced:
        print("resumed", flush=True)


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_events(path):
    if not os.path.isfile(path):
        return None
    rows = read_tsv(path)
    return [(float(r["onset"]), float(r["duration"]), r["trial_type"]) for r in rows]


def read_fif_header(path):
    """Header facts of one FIF (preload=False: no data sample is read)."""
    import mne
    import numpy as np
    from mne.io.constants import FIFF
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raw = mne.io.read_raw_fif(path, preload=False, allow_maxshield=True,
                                  on_split_missing="warn", verbose="ERROR")
    info = raw.info
    ex = raw._raw_extras[0]
    bounds = np.asarray(ex["bounds"], dtype=np.int64)
    sizes = bounds[1:] - bounds[:-1]
    skip = np.array([e is None for e in ex["ent"]], dtype=bool)
    sf = float(info["sfreq"])
    md = info["meas_date"]
    types = raw.get_channel_types()
    anns = []
    for a in raw.annotations:
        if a["description"] == "BAD_ACQ_SKIP":
            continue
        anns.append([str(a["description"]), float(a["onset"]), float(a["duration"])])
    fids = {}
    dig_kinds = Counter()
    for d in info["dig"] or []:
        dig_kinds[int(d["kind"])] += 1
        if int(d["kind"]) == FIFF.FIFFV_POINT_CARDINAL:
            fids[{1: "lpa", 2: "nasion", 3: "rpa"}.get(int(d["ident"]), str(d["ident"]))] = \
                [float(x) for x in d["r"]]
    si = info.get("subject_info") or {}
    return {
        "cache_version": CACHE_VERSION,
        "name": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "chain": [os.path.basename(str(f)) for f in raw.filenames],
        "first_samp": int(raw._first_samps[0]),
        "last_samp": int(raw._last_samps[0]),
        "sfreq": sf,
        "meas_date": md.isoformat() if md is not None else None,
        "n_buffers": int((~skip).sum()),
        "buffer_samples_median": float(np.median(sizes[~skip])) if (~skip).any() else None,
        "skip_entries": int(skip.sum()),
        "skip_samples": int(sizes[skip].sum()),
        "orig_format": raw.orig_format,
        "nchan": int(info["nchan"]),
        "ch_names": list(raw.ch_names),
        "ch_types": list(types),
        "bads": list(info["bads"]),
        "highpass": float(info["highpass"]),
        "lowpass": float(info["lowpass"]),
        "line_freq": None if info["line_freq"] is None else float(info["line_freq"]),
        "dev_head_t": info["dev_head_t"] is not None,
        "n_hpi_results": len(info.get("hpi_results") or []),
        "n_proc_history": len(info.get("proc_history") or []),
        "maxshield": bool(info.get("maxshield", False)) if hasattr(info, "get") else False,
        "his_id": str(si.get("his_id", "")) if si else "",
        "proj_name": info.get("proj_name") or "",
        "description": info.get("description") or "",
        "fiducials": fids,
        "dig_kinds": {str(k): v for k, v in sorted(dig_kinds.items())},
        "annotations": anns,
        "warnings": sorted({str(x.message)[:300] for x in w}),
    }


def load_headers(raw_root, fif_paths, cache_dir, force):
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for i, p in enumerate(fif_paths):
        name = os.path.basename(p)
        cpath = os.path.join(cache_dir, name + ".json")
        st = os.stat(p)
        if not force and os.path.isfile(cpath):
            with open(cpath, encoding="utf-8") as f:
                h = json.load(f)
            if h.get("cache_version") == CACHE_VERSION and h.get("size_bytes") == st.st_size \
                    and h.get("mtime") == int(st.st_mtime):
                out[name] = h
                continue
        wait_if_slow()
        t0 = time.time()
        h = read_fif_header(p)
        h["mtime"] = int(st.st_mtime)
        h["read_s"] = round(time.time() - t0, 3)
        with open(cpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(h, f, indent=1)
        out[name] = h
        print(f"[{i + 1}/{len(fif_paths)}] {name}: {h['read_s']} s", flush=True)
    return out


def list_files(raw_root):
    allf = []
    for dp, dns, fns in os.walk(raw_root):
        dns[:] = [d for d in dns if not d.startswith("._")]
        for fn in fns:
            if fn.startswith("._"):
                continue
            allf.append(os.path.join(dp, fn).replace("\\", "/"))
    return sorted(allf)


# ----------------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------------

def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c, "")) for c in cols])


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.1f}"
    if isinstance(v, (list, tuple)):
        return ";".join(fmt(x) for x in v)
    return str(v)


def hhmmss(dt):
    return dt.strftime("%H:%M:%S") if dt is not None else ""


def parse_dt(s):
    if not s:
        return None
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def tree_relative(path):
    """path relative to this tree with forward slashes; the absolute path when it lies outside the tree
    (on Windows os.path.relpath raises ValueError across drives)."""
    ap = os.path.abspath(path)
    try:
        rel = os.path.relpath(ap, TREE)
    except ValueError:
        return ap.replace("\\", "/")
    return ap.replace("\\", "/") if rel.startswith("..") else rel.replace("\\", "/")


def stamp_provenance(moveatlas):
    here = os.path.join(TREE, "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)
    import moveatlas_stamp  # scripts/moveatlas_stamp.py (task T00): every _provenance.json uses it, no fallback
    return {"moveatlas_stamp": moveatlas_stamp.stamp(moveatlas)}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "ds004998"))
    ap.add_argument("--person-map", default=os.path.join(TREE, "reports", "person_ids",
                                                         "map_pd_meg_stn_lfp_ds004998.csv"))
    ap.add_argument("--force", action="store_true", help="re-read every header (ignore the cache)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    raw_root = f"{a.moveatlas}/{RAW_REL}"
    if not os.path.isdir(raw_root):
        sys.exit(f"raw tree not found: {raw_root}")
    os.makedirs(a.out, exist_ok=True)
    prio = set_below_normal()
    print("priority:", prio)
    t_start = time.time()

    files = list_files(raw_root)
    rel = {f: f[len(raw_root) + 1:] for f in files}
    fifs = [f for f in files if f.endswith("_meg.fif")]
    headers = load_headers(raw_root, fifs, os.path.join(a.out, "cache"), a.force)

    # ---------------- subjects and side files
    parts_tsv = read_tsv(f"{raw_root}/participants.tsv")
    listed = [r["participant_id"] for r in parts_tsv]
    on_disk = sorted(d for d in os.listdir(raw_root) if d.startswith("sub-") and os.path.isdir(f"{raw_root}/{d}"))
    updrs = {}
    for st in ("off", "on"):
        p = f"{raw_root}/participants_updrs_{st}.tsv"
        updrs[st] = {r["participant_id"] for r in read_tsv(p)} if os.path.isfile(p) else set()

    scans = {}
    for f in files:
        if f.endswith("_scans.tsv"):
            for r in read_tsv(f):
                scans[os.path.basename(r["filename"])] = r["acq_time"]

    # ---------------- files and runs
    file_rows = []
    parts = []
    for f in fifs:
        name = os.path.basename(f)
        h = headers[name]
        ent = parse_bids_name(name)
        parts.append({"name": name, "first_samp": h["first_samp"], "last_samp": h["last_samp"],
                      "sfreq": h["sfreq"], "meas_date": h["meas_date"],
                      "skip_entries": h["skip_entries"], "skip_samples": h["skip_samples"]})
        file_rows.append({
            "file": rel[f], "sub": ent["sub"], "task": ent["task"], "acq": ent["acq"] or "",
            "run": ent["run"] or "", "split": ent["split"] or "", "size_bytes": h["size_bytes"],
            "meas_date": h["meas_date"], "acq_time_scans": scans.get(name, ""),
            "first_samp": h["first_samp"], "last_samp": h["last_samp"],
            "n_samples": h["last_samp"] - h["first_samp"] + 1, "sfreq": h["sfreq"],
            "duration_s": (h["last_samp"] - h["first_samp"] + 1) / h["sfreq"],
            "n_buffers": h["n_buffers"], "buffer_samples": h["buffer_samples_median"],
            "skip_entries": h["skip_entries"], "skip_samples": h["skip_samples"],
            "orig_format": h["orig_format"], "nchan": h["nchan"],
            "chain_from_header": h["chain"], "has_events_tsv": os.path.isfile(f.replace("_meg.fif", "_events.tsv")),
            # whether the FIF stores annotations, not how many: a count per file would join the blind rows of
            # segments.csv to their run name, hence to the state
            "annotations_in_fif": bool(h["annotations"]), "n_proc_history": h["n_proc_history"],
            "warnings": h["warnings"]})
    runs = merge_split_parts(parts)
    acqs = group_acquisitions(runs)
    hdr_of = {os.path.basename(f): headers[os.path.basename(f)] for f in fifs}
    dir_of = {os.path.basename(f): os.path.dirname(f) for f in fifs}

    # side files per run, G15 header verdict, events QA
    label_census = Counter()
    events_files = [f for f in files if f.endswith("_events.tsv")]
    for f in events_files:
        for on, dur, lab in read_events(f):
            label_census[lab] += 1
    seg_rows = []
    for r in runs:
        d = dir_of[r["parts"][0]]
        h0 = hdr_of[r["parts"][0]]
        r["chain_ok"] = h0["chain"] == r["parts"]
        r["chain_from_header"] = h0["chain"]
        js = f"{d}/{r['run_id']}_meg.json"
        r["meg_json"] = os.path.isfile(js)
        dur_json = None
        if r["meg_json"]:
            with open(js, encoding="utf-8") as fh:
                jd = json.load(fh)
            dur_json = jd.get("RecordingDuration")
            r["json_sfreq"] = jd.get("SamplingFrequency")
            r["json_counts"] = {k: jd.get(k) for k in ("MEGChannelCount", "EEGChannelCount", "EMGChannelCount",
                                                       "EOGChannelCount", "MiscChannelCount", "TriggerChannelCount")}
        r["json_duration_s"] = dur_json
        r["json_samples"] = int(round(dur_json * r["sfreq"])) + 1 if dur_json is not None else None
        r["samples_minus_json"] = (r["n_samples"] - r["json_samples"]) if r["json_samples"] is not None else None
        chf = f"{d}/{r['run_id']}_channels.tsv"
        r["channels_tsv"] = os.path.isfile(chf)
        r["channels_match"] = None
        r["tsv_bad"] = []
        if r["channels_tsv"]:
            ch = read_tsv(chf)
            r["channels_match"] = [c["name"] for c in ch] == h0["ch_names"]
            r["tsv_bad"] = [c["name"] for c in ch if c.get("status") == "bad"]
        names_same = all(hdr_of[p]["ch_names"] == h0["ch_names"] for p in r["parts"])
        r["parts_same_channels"] = names_same
        r["nchan"] = h0["nchan"]
        tcount = Counter(h0["ch_types"])
        r["n_meg"] = tcount.get("mag", 0) + tcount.get("grad", 0)
        r["n_eeg"] = tcount.get("eeg", 0)
        r["n_emg"] = tcount.get("emg", 0)
        r["n_eog"] = tcount.get("eog", 0)
        r["n_misc"] = tcount.get("misc", 0)
        r["n_stim"] = tcount.get("stim", 0)
        r["fif_bads"] = h0["bads"]
        r["acq_time"] = scans.get(r["parts"][0], "")
        r["acq_time_parts_equal"] = len({scans.get(p, "") for p in r["parts"]}) == 1
        md = parse_dt(r["meas_date"])
        at = parse_dt(r["acq_time"])
        r["acq_time_equals_meas_date"] = (md is not None and at is not None and abs((md - at).total_seconds()) < 1e-3)
        r["record_start"] = md + _dt.timedelta(seconds=r["first_samp"] / r["sfreq"]) if md else None
        r["record_end"] = md + _dt.timedelta(seconds=(r["last_samp"] + 1) / r["sfreq"]) if md else None
        r["duration_s"] = r["n_samples"] / r["sfreq"]
        # G15 header line
        reasons = []
        if r["continuity"] not in ("single", "contiguous"):
            reasons.append(r["continuity"])
        if r["skip_entries"]:
            reasons.append(f"{r['skip_entries']} skip entries ({r['skip_samples']} samples)")
        if not r["chain_ok"]:
            reasons.append("header chain differs from the listing")
        if r["samples_minus_json"] is None:
            reasons.append("no meg.json duration")
        elif abs(r["samples_minus_json"]) > 1:
            reasons.append(f"sample count differs from meg.json by {r['samples_minus_json']}")
        if r["channels_match"] is False:
            reasons.append("channels.tsv differs from the FIF channel list")
        if not names_same:
            reasons.append("parts differ in channel list")
        if float(r["sfreq"]) != 2000.0:
            reasons.append(f"sfreq {r['sfreq']}")
        r["g15_header_verdict"] = "pass" if not reasons else "fail"
        r["g15_reasons"] = reasons
        # events of the run on the acquisition clock (run_events), and their checks
        pev = [read_events(f"{d}/{p.replace('_meg.fif', '_events.tsv')}") for p in r["parts"]]
        bounds = [(hdr_of[p]["first_samp"] / r["sfreq"], (hdr_of[p]["last_samp"] + 1) / r["sfreq"])
                  for p in r["parts"]]
        r["events_parts_missing"] = [p for p, e in zip(r["parts"], pev) if e is None]
        evs, src, comps = run_events(pev, bounds)
        r["events"] = evs
        r["events_source"] = src if any(e is not None for e in pev) else "none"
        r["n_events"] = len(evs)
        t0, t1 = bounds[0][0], bounds[-1][1]
        beyond = [e - t1 for s, e, l in evs if e > t1 + 1e-6]
        before = [t0 - s for s, e, l in evs if s < t0 - 1e-6]
        r["events_beyond_end"] = len(beyond)
        r["events_beyond_end_max_s"] = max(beyond) if beyond else 0.0
        r["events_before_start"] = len(before)
        # later parts' own events.tsv against the run's events clipped to the part
        r["later_part_tsv_compared"] = len(comps)
        r["later_part_tsv_equal"] = sum(1 for c in comps if c[4])
        r["later_part_tsv_unpaired"] = sum(c[1] for c in comps)
        r["later_part_tsv_max_onset_diff_s"] = max((c[2] for c in comps), default=None)
        r["later_part_tsv_max_duration_diff_s"] = max((c[3] for c in comps), default=None)
        # the annotations stored in each part's FIF: the first part holds the run's events, a later part
        # the run's events clipped to it
        fe = []
        for k, p in enumerate(r["parts"]):
            hp = hdr_of[p]
            p0 = bounds[k][0]
            anns = [(on - p0, du, lab) for lab, on, du in hp["annotations"]]
            if k == 0:
                ref = [(s - p0, e - s, l) for s, e, l in evs]
            else:
                ref = clip_events(evs, *bounds[k])
            fe.append(events_equal(ref, anns))
        r["parts_fif_equal_run_events"] = sum(1 for x in fe if x)
        r["parts_fif_differ"] = [p for p, x in zip(r["parts"], fe) if not x]
        r["n_fif_annotations"] = sum(len(hdr_of[p]["annotations"]) for p in r["parts"])
        r["is_noise"] = task_class(r["task"]) is None
    # relation of contained runs to their primary: where they start in it, and whether the labels
    # of their events agree with the primary's labels on the shared samples
    for aid, rs in acqs.items():
        prim = rs[0]
        for r in rs[1:]:
            r["offset_in_primary_samples"] = r["first_samp"] - prim["first_samp"]
            r["ends_with_primary"] = r["last_samp"] == prim["last_samp"]
            ev2 = [(s, e) for s, e, l in r["events"] if not l.startswith("bad")]
            tot = sum(e - s for s, e in union_intervals(ev2))
            same = diff = 0.0
            pairs = set()
            for s, e, l in r["events"]:
                if l.startswith("bad"):
                    continue
                same += overlap_s([(s, e)], [(s1, e1) for s1, e1, l1 in prim["events"] if l1 == l])
                other = [(s1, e1, l1) for s1, e1, l1 in prim["events"] if l1 != l and not l1.startswith("bad")]
                diff += overlap_s([(s, e)], [(s1, e1) for s1, e1, l1 in other])
                pairs |= {f"{l} in the copy, {l1} in run-1" for s1, e1, l1 in other if min(e, e1) > max(s, s1)}
            r["label_pairs_in_conflict"] = sorted(pairs)
            r["events_same_label_in_primary"] = (same / tot) if tot > 0 else None
            r["events_other_label_in_primary"] = (diff / tot) if tot > 0 else None
            r["overlap_samples_with_primary"] = max(0, min(r["last_samp"], prim["last_samp"]) -
                                                    max(r["first_samp"], prim["first_samp"]) + 1)

    # the size of the first acquisition file, in samples per channel count, from where the copies start;
    # each recording without a copy is compared with it (a recording exactly one file long may stop at the
    # file boundary, its continuation not published)
    first_file = defaultdict(set)
    for r in runs:
        if r.get("relation_to_primary") == "contained":
            first_file[r["nchan"]].add(r["offset_in_primary_samples"])
    has_copy = {r["primary_run_id"] for r in runs if r.get("relation_to_primary") == "contained"}
    for r in runs:
        r["one_file_samples"] = ";".join(str(x) for x in sorted(first_file.get(r["nchan"], ())))
        lab_end = max((e for s, e, l in r["events"] if not l.startswith("bad")), default=None)
        r["last_label_end_to_run_end_s"] = (None if lab_end is None
                                            else (r["last_samp"] + 1) / r["sfreq"] - lab_end)
        if r.get("relation_to_primary") != "primary" or r["is_noise"]:
            r["length_vs_one_file"] = ""
        elif r["run_id"] in has_copy:
            r["length_vs_one_file"] = "longer_with_copy"
        elif not first_file.get(r["nchan"]):
            r["length_vs_one_file"] = "unknown"
        elif r["n_samples"] in first_file[r["nchan"]]:
            r["length_vs_one_file"] = "equal"
        elif r["n_samples"] > max(first_file[r["nchan"]]):
            r["length_vs_one_file"] = "longer_without_copy"
        else:
            r["length_vs_one_file"] = "shorter"

    task_runs = [r for r in runs if not r["is_noise"]]
    noise_runs = [r for r in runs if r["is_noise"]]
    subjects = sorted({r["sub"] for r in runs})

    # ---------------- runs.csv, files.csv, acquisitions.csv
    run_cols = ["run_id", "sub", "task", "acq", "run", "n_parts", "parts", "continuity", "joins", "problems",
                "skip_entries", "skip_samples", "first_samp", "last_samp", "n_samples", "duration_s",
                "json_duration_s", "samples_minus_json", "chain_ok", "channels_tsv", "channels_match",
                "parts_same_channels", "nchan", "n_meg", "n_eeg", "n_emg", "n_eog", "n_misc", "n_stim",
                "g15_header_verdict", "g15_reasons", "acquisition_id", "relation_to_primary", "primary_run_id",
                "separate_acquisition", "offset_in_primary_samples", "ends_with_primary",
                "overlap_samples_with_primary", "events_same_label_in_primary", "events_other_label_in_primary",
                "label_pairs_in_conflict", "one_file_samples", "length_vs_one_file", "last_label_end_to_run_end_s",
                "meas_date", "acq_time", "acq_time_parts_equal", "acq_time_equals_meas_date",
                "record_start", "record_end", "events_source", "events_parts_missing",
                "events_beyond_end", "events_beyond_end_max_s", "events_before_start",
                "parts_fif_equal_run_events", "parts_fif_differ", "later_part_tsv_compared",
                "later_part_tsv_equal", "later_part_tsv_unpaired", "later_part_tsv_max_onset_diff_s",
                "later_part_tsv_max_duration_diff_s"]
    rows = []
    for r in runs:
        rr = dict(r)
        rr["record_start"] = r["record_start"].isoformat() if r["record_start"] else ""
        rr["record_end"] = r["record_end"].isoformat() if r["record_end"] else ""
        rows.append(rr)
    write_csv(os.path.join(a.out, "runs.csv"), rows, run_cols)
    write_csv(os.path.join(a.out, "files.csv"), file_rows, list(file_rows[0].keys()))

    acq_rows = []
    for aid, rs in acqs.items():
        prim = rs[0]
        labels = sorted({(r["acq"], r["task"]) for r in rs})
        acq_rows.append({"acquisition_id": aid, "sub": prim["sub"], "task": prim["task"], "acq": prim["acq"],
                         "labels_consistent": len(labels) == 1, "n_runs": len(rs),
                         "runs": [r["run_id"] for r in rs],
                         "relations": [r["relation_to_primary"] for r in rs],
                         "n_separate": sum(1 for r in rs if r["separate_acquisition"]),
                         "acq_time": prim["acq_time"], "record_start": prim["record_start"].isoformat(),
                         "record_end": prim["record_end"].isoformat(), "duration_s": prim["duration_s"],
                         "primary_g15": prim["g15_header_verdict"]})
    write_csv(os.path.join(a.out, "acquisitions.csv"), acq_rows, list(acq_rows[0].keys()))

    # ---------------- state by task (run counts only)
    st_rows = []
    tasks = sorted({r["task"] for r in task_runs})
    states = sorted({r["acq"] for r in task_runs})
    for stt in states:
        for t in tasks:
            rr = [r for r in task_runs if r["acq"] == stt and r["task"] == t]
            if not rr and (stt, t) not in ADDENDUM_STATE_TASK:
                continue
            st_rows.append({"acq": stt, "task": t, "bids_runs": len(rr),
                            "bids_run_files": sum(r["n_parts"] for r in rr),
                            "acquisitions": sum(1 for r in rr if r["relation_to_primary"] == "primary"),
                            "separate_acquisitions": sum(1 for r in rr if r["separate_acquisition"]),
                            "subjects": len({r["sub"] for r in rr}),
                            "addendum_runs": ADDENDUM_STATE_TASK.get((stt, t), "")})
    write_csv(os.path.join(a.out, "state_task_counts.csv"), st_rows, list(st_rows[0].keys()))

    # ---------------- order census and feasibility
    oc_rows, fz_rows = [], []
    t_ref = None
    for s in subjects:
        rs = [r for r in task_runs if r["sub"] == s]
        if not rs:
            continue
        t_ref = min(parse_dt(r["acq_time"]) for r in rs)
        # unit A: BIDS runs (split parts merged), start = acq_time of scans.tsv
        u_runs = [{"state": r["acq"], "task": r["task"], "label": f"{r['acq']}:{r['task']}:run-{r['run']}",
                   "start": (parse_dt(r["acq_time"]) - t_ref).total_seconds(),
                   "end": (r["record_end"] - t_ref).total_seconds()} for r in rs]
        # unit B: acquisitions (primary runs only; a contained run repeats samples of its primary)
        u_acq = [u for u, r in zip(u_runs, rs) if r["relation_to_primary"] == "primary"]
        ocA = order_census(u_runs)
        ocB = order_census(u_acq)
        rec_gap = None
        if ocB["all_off_before_on"] is not None:
            prim = [r for r in rs if r["relation_to_primary"] == "primary"]
            off_end = max(r["record_end"] for r in prim if r["acq"] == "MedOff")
            on_start = min(r["record_start"] for r in prim if r["acq"] == "MedOn")
            rec_gap = (on_start - off_end).total_seconds()
        seq = sorted([(parse_dt(r["acq_time"]), r) for r in rs if r["relation_to_primary"] == "primary"],
                     key=lambda x: (x[0], x[1]["run_id"]))
        oc_rows.append({
            "sub": s, "n_bids_runs": len(rs), "n_acquisitions": len(u_acq),
            "n_medoff_runs": ocA["n_off"], "n_medon_runs": ocA["n_on"],
            "n_medoff_acquisitions": ocB["n_off"], "n_medon_acquisitions": ocB["n_on"],
            "all_medoff_before_medon": ocB["all_off_before_on"], "interleaved": ocB["interleaved"],
            "medoff_after_first_medon": ocB["off_after_first_on"],
            "gap_acq_time_last_off_to_first_on_min": None if ocB["gap_start_to_start_s"] is None
            else ocB["gap_start_to_start_s"] / 60.0,
            "gap_record_end_off_to_record_start_on_min": None if rec_gap is None else rec_gap / 60.0,
            "sequence_by_acq_time": [f"{hhmmss(t)} {r['acq']} {r['task']}" for t, r in seq],
            "acq_time_ties": sum(1 for r in rs if r["relation_to_primary"] != "primary")})
        for unit, us in (("bids_runs", u_runs), ("acquisitions", u_acq)):
            for reading in ("class", "side"):
                fz = order_null_feasibility(us, reading)
                fz_rows.append({"sub": s, "unit": unit, "reading": reading,
                                "medoff_per_task": [f"{k}:{v['n']}/{v['n_distinct_start']}" for k, v in fz["per_key"].items()],
                                "feasible": fz["feasible"], "feasible_counting_ties": fz["feasible_counting_ties"],
                                "any_task_two_or_more": fz["any_task_two_or_more"],
                                "any_task_two_or_more_counting_ties": fz["any_task_two_or_more_counting_ties"],
                                "max_medoff_same_task": max((v["n"] for v in fz["per_key"].values()), default=0),
                                "max_medoff_same_task_distinct_time": max((v["n_distinct_start"] for v in fz["per_key"].values()), default=0)})
    write_csv(os.path.join(a.out, "order_census.csv"), oc_rows, list(oc_rows[0].keys()))
    write_csv(os.path.join(a.out, "order_null_feasibility.csv"), fz_rows, list(fz_rows[0].keys()))

    # side-specific task matching of move runs (sub-iDpl28 statement): label sets per state
    side_sets = {}
    for s in subjects:
        rs = [r for r in task_runs if r["sub"] == s]
        side_sets[s] = {stt: sorted({r["task"] for r in rs if r["acq"] == stt}) for stt in ("MedOff", "MedOn")}

    # ---------------- contact map and EMG
    cm_rows = []
    unmapped_rows = []
    for s in subjects:
        mt = f"{raw_root}/sub-{s}/ses-PeriOp/montage/sub-{s}_ses-PeriOp_montage.tsv"
        rs = [r for r in runs if r["sub"] == s and not r["is_noise"]]
        ch_sets = [set(hdr_of[r["parts"][0]]["ch_names"]) for r in rs]
        mapped = set()
        if os.path.isfile(mt):
            for row in read_tsv(mt):
                for side in ("right", "left"):
                    old, new = row[f"{side}_contacts_old"], row[f"{side}_contacts_new"]
                    mapped.add(old)
                    m = re.match(r"LFP-(right|left)-(\d+)$", new)
                    types = {hdr_of[r["parts"][0]]["ch_types"][hdr_of[r["parts"][0]]["ch_names"].index(old)]
                             for r in rs if old in hdr_of[r["parts"][0]]["ch_names"]}
                    cm_rows.append({"sub": s, "side": side, "contact_index": int(m.group(2)) if m else "",
                                    "old_name": old, "new_name": new,
                                    "new_name_side_agrees": bool(m and m.group(1) == side),
                                    "runs_with_channel": sum(1 for c in ch_sets if old in c), "task_runs": len(rs),
                                    "fif_types": sorted(types),
                                    "runs_marked_bad": sum(1 for r in rs if old in r["fif_bads"] or old in r["tsv_bad"])})
        eeg_names = sorted({n for r in rs for n, t in zip(hdr_of[r["parts"][0]]["ch_names"], hdr_of[r["parts"][0]]["ch_types"]) if t == "eeg"})
        unmapped_rows.append({"sub": s, "montage_tsv": os.path.isfile(mt), "contacts_per_side": len(mapped) // 2,
                              "fif_eeg_channels": len(eeg_names),
                              "fif_eeg_unmapped": [n for n in eeg_names if n not in mapped],
                              "montage_not_in_fif": sorted(n for n in mapped if n not in eeg_names)})
    write_csv(os.path.join(a.out, "contact_map.csv"), cm_rows, list(cm_rows[0].keys()))
    write_csv(os.path.join(a.out, "contact_map_subjects.csv"), unmapped_rows, list(unmapped_rows[0].keys()))

    emg_rows = []
    for r in runs:
        h0 = hdr_of[r["parts"][0]]
        emg = [n for n, t in zip(h0["ch_names"], h0["ch_types"]) if t == "emg"]
        misc = [n for n, t in zip(h0["ch_names"], h0["ch_types"]) if t == "misc"]
        desc = set()
        chf = f"{dir_of[r['parts'][0]]}/{r['run_id']}_channels.tsv"
        if os.path.isfile(chf):
            desc = {c.get("description", "") for c in read_tsv(chf) if c["name"] in emg}
        emg_rows.append({"run_id": r["run_id"], "sub": r["sub"], "task": r["task"], "emg_channels": emg,
                         "n_emg": len(emg), "misc_channels": misc, "channels_tsv_description": sorted(desc),
                         "emg_marked_bad": [n for n in emg if n in r["fif_bads"] or n in r["tsv_bad"]]})
    write_csv(os.path.join(a.out, "emg_channels.csv"), emg_rows, list(emg_rows[0].keys()))
    muscle_hits = []
    for f in files:
        if f.endswith((".json", ".tsv", "README", "CHANGES")) and os.path.getsize(f) < 5_000_000:
            with open(f, encoding="utf-8", errors="replace") as fh:
                txt = fh.read().lower()
            hit = [w for w in MUSCLE_WORDS if w in txt]
            if hit:
                muscle_hits.append(f"{rel[f]}: {','.join(hit)}")

    # ---------------- segments (one row per acquisition, blind key, no state)
    for aid, rs in acqs.items():
        prim = rs[0]
        if prim["is_noise"]:
            continue
        t0 = prim["first_samp"] / prim["sfreq"]
        evs = [(s - t0, e - t0, l) for s, e, l in prim["events"]]
        rest = [(s, e) for s, e, l in evs if l == "rest"]
        task_lab = [(s, e, l) for s, e, l in evs if task_class(l) in ("hold", "move")]
        bad = [(s, e) for s, e, l in evs if l in ("bad_lfp", "bad_flp")]
        other = sorted({l for s, e, l in evs if l not in ("rest", "bad_lfp", "bad_flp") and task_class(l) not in ("hold", "move")})
        tdur = [e - s for s, e, l in task_lab]
        first_task = min((s for s, e, l in task_lab), default=None)
        first_rest = min(rest, default=None)
        seg_rows.append({
            "sub": prim["sub"], "task": prim["task"], "blind_key": blind_key(prim["sub"], prim["task"], evs),
            "has_events": len(prim["events_parts_missing"]) < prim["n_parts"],
            "n_rest": len(rest), "rest_total_s": sum(e - s for s, e in rest),
            "first_rest_onset_s": first_rest[0] if first_rest else None,
            "first_rest_duration_s": (first_rest[1] - first_rest[0]) if first_rest else None,
            "rest_before_first_task": (first_rest is not None and first_task is not None
                                       and first_rest[0] < first_task),
            "first_rest_end_to_first_task_s": (first_task - first_rest[1])
            if (first_rest is not None and first_task is not None and first_rest[0] < first_task) else None,
            "n_rest_after_first_task": sum(1 for s, e in rest if first_task is not None and s > first_task),
            "task_labels": sorted({l for s, e, l in task_lab}),
            "task_label_matches_run": all(l == prim["task"] for s, e, l in task_lab) if task_lab else None,
            "n_task_segments": len(task_lab), "task_total_s": sum(tdur),
            "task_segment_min_s": min(tdur) if tdur else None, "task_segment_max_s": max(tdur) if tdur else None,
            "n_bad_lfp": len(bad), "bad_lfp_total_s": sum(e - s for s, e in union_intervals(bad)),
            "bad_lfp_overlap_task_s": overlap_s(bad, [(s, e) for s, e, l in task_lab]),
            "bad_lfp_overlap_rest_s": overlap_s(bad, rest),
            "other_labels": other})
    seg_rows.sort(key=lambda x: (x["sub"], x["task"], x["blind_key"]))
    seg_cols = list(seg_rows[0].keys())
    write_csv(os.path.join(a.out, "segments.csv"), seg_rows, seg_cols)
    sbt = []
    for t in sorted({x["task"] for x in seg_rows}):
        xs = [x for x in seg_rows if x["task"] == t]
        frd = [x["first_rest_duration_s"] for x in xs if x["first_rest_duration_s"] is not None]
        sbt.append({"task": t, "acquisitions": len(xs), "subjects": len({x["sub"] for x in xs}),
                    "with_rest": sum(1 for x in xs if x["n_rest"] > 0),
                    "first_rest_300s": sum(1 for d in frd if abs(d - 300.0) <= 1.0),
                    "first_rest_s_median": median(frd), "first_rest_s_min": min(frd, default=None),
                    "first_rest_s_max": max(frd, default=None),
                    "rest_before_first_task": sum(1 for x in xs if x["rest_before_first_task"]),
                    "with_rest_after_first_task": sum(1 for x in xs if x["n_rest_after_first_task"] > 0),
                    "with_task_segments": sum(1 for x in xs if x["n_task_segments"] > 0),
                    "task_segments_total": sum(x["n_task_segments"] for x in xs),
                    "task_total_s_median": median([x["task_total_s"] for x in xs if x["n_task_segments"] > 0]),
                    "with_bad_lfp": sum(1 for x in xs if x["n_bad_lfp"] > 0),
                    "bad_lfp_total_s": sum(x["bad_lfp_total_s"] for x in xs),
                    "task_label_mismatch": sum(1 for x in xs if x["task_label_matches_run"] is False),
                    "other_labels": sorted({l for x in xs for l in x["other_labels"]})})
    write_csv(os.path.join(a.out, "segments_by_task.csv"), sbt, list(sbt[0].keys()))
    write_csv(os.path.join(a.out, "event_labels.csv"),
              [{"trial_type": k, "events": v} for k, v in sorted(label_census.items(), key=lambda x: -x[1])],
              ["trial_type", "events"])

    # ---------------- empty room
    er_rows = []
    for r in noise_runs:
        h0 = hdr_of[r["parts"][0]]
        tr = [x for x in task_runs if x["sub"] == r["sub"] and x["relation_to_primary"] == "primary"]
        first = min((x["record_start"] for x in tr), default=None)
        last = max((x["record_end"] for x in tr), default=None)
        pos = ""
        if first is not None:
            if r["record_end"] <= first:
                pos = f"before the first task run by {(first - r['record_end']).total_seconds() / 60:.1f} min"
            elif r["record_start"] >= last:
                pos = f"after the last task run by {(r['record_start'] - last).total_seconds() / 60:.1f} min"
            else:
                pos = "between task runs"
        er_rows.append({"run_id": r["run_id"], "sub": r["sub"], "acq_time": r["acq_time"],
                        "duration_s": r["duration_s"], "nchan": r["nchan"], "n_meg": r["n_meg"],
                        "n_eeg": r["n_eeg"], "n_emg": r["n_emg"], "n_misc": r["n_misc"],
                        "g15_header_verdict": r["g15_header_verdict"], "position": pos,
                        "dev_head_t": h0["dev_head_t"]})
    write_csv(os.path.join(a.out, "empty_room.csv"), er_rows, list(er_rows[0].keys()))

    # ---------------- subjects and person map
    fid_geom = {}
    fid_spread = {}
    for s in subjects:
        geoms = []
        for r in runs:
            if r["sub"] != s:
                continue
            fd = hdr_of[r["parts"][0]]["fiducials"]
            if all(k in fd for k in ("lpa", "nasion", "rpa")):
                geoms.append(fid_geometry(fd))
        if geoms:
            fid_geom[s] = [sum(g[i] for g in geoms) / len(geoms) for i in range(3)]
            fid_spread[s] = max(max(abs(g[i] - fid_geom[s][i]) for i in range(3)) for g in geoms)
    sub_rows = []
    for s in sorted(set(x[4:] for x in listed) | set(x[4:] for x in on_disk)):
        sid = f"sub-{s}"
        rs = [r for r in task_runs if r["sub"] == s]
        sub_rows.append({"participant_id": sid, "in_participants_tsv": sid in listed, "on_disk": sid in on_disk,
                         "bids_runs": len(rs), "acquisitions": sum(1 for r in rs if r["relation_to_primary"] == "primary"),
                         "fif_files": sum(r["n_parts"] for r in rs),
                         "empty_room": any(r["sub"] == s for r in noise_runs),
                         "tasks": sorted({r["task"] for r in rs}),
                         "in_updrs_off_tsv": sid in updrs["off"], "in_updrs_on_tsv": sid in updrs["on"]})
    write_csv(os.path.join(a.out, "subjects.csv"), sub_rows, list(sub_rows[0].keys()))

    # demographic twins in participants.tsv (same age, sex and disease duration): a weak flag only
    demo = defaultdict(list)
    for row in parts_tsv:
        demo[(row.get("age", ""), row.get("sex", ""), row.get("diseaseDuration", ""))].append(row["participant_id"])
    twins = {pid: [o for o in ids if o != pid] for ids in demo.values() if len(ids) > 1 for pid in ids}
    # for a twin pair: how many fields of their UPDRS rows are equal (no score is written)
    updrs_rows = {}
    for st in ("off", "on"):
        p = f"{raw_root}/participants_updrs_{st}.tsv"
        if os.path.isfile(p):
            updrs_rows[st] = {r["participant_id"]: r for r in read_tsv(p)}
    twin_updrs = {}
    for pid, others in twins.items():
        notes = []
        for o in others:
            for st, rows_ in sorted(updrs_rows.items()):
                a_, b_ = rows_.get(pid), rows_.get(o)
                if a_ is None or b_ is None:
                    continue
                keys = [k for k in a_ if k != "participant_id"]
                eqn = sum(1 for k in keys if a_.get(k, "") == b_.get(k, ""))
                x_, y_ = sorted([pid, o])
                notes.append(f"UPDRS {st} rows of {x_} and {y_} equal in {eqn} of {len(keys)} fields")
        twin_updrs[pid] = notes
    nn_all = {}
    for s in fid_geom:
        others = [(math.dist(fid_geom[s], fid_geom[o]), o) for o in fid_geom if o != s]
        nn_all[s] = min(others) if others else None
    pm_rows = []
    multi_dig = []
    for s in subjects:
        sid = f"sub-{s}"
        rs = [r for r in runs if r["sub"] == s]
        sessions = sorted(d for d in os.listdir(f"{raw_root}/{sid}") if d.startswith("ses-"))
        his = sorted({hdr_of[r["parts"][0]]["his_id"] for r in rs})
        ev = (f"raw folder {sid}: sessions {','.join(sessions)}; {sum(1 for r in rs if not r['is_noise'])} BIDS task runs "
              f"({sum(1 for r in rs if not r['is_noise'] and r['relation_to_primary'] == 'primary')} acquisitions, "
              f"{sum(r['n_parts'] for r in rs if not r['is_noise'])} FIF files), empty-room run "
              f"{'yes' if any(r['is_noise'] for r in rs) else 'no'}; participants.tsv row {'present' if sid in listed else 'absent'}; "
              f"FIF headers anonymized (his_id {','.join(his)}, mne_anonymize time shift), so no header id to cross-check")
        amb = ""
        if s in fid_geom:
            geos = {tuple(round(x, 2) for x in fid_geometry(hdr_of[r['parts'][0]]['fiducials']))
                    for r in rs if all(k in hdr_of[r['parts'][0]]['fiducials'] for k in ('lpa', 'nasion', 'rpa'))}
            ndig = len(geos)
            if ndig > 1:
                multi_dig.append((s, ndig, max(math.dist(g1, g2) for g1 in geos for g2 in geos)))
            ev += (f"; digitized fiducials: {ndig} distinct geometr{'y' if ndig == 1 else 'ies'} over its runs, "
                   f"spread {fid_spread[s]:.2f} mm (LPA-RPA, nasion-LPA, nasion-RPA)")
            if nn_all.get(s):
                ev += (f"; nearest other subject sub-{nn_all[s][1]} at {nn_all[s][0]:.1f} mm, a distance this "
                       f"three-number geometry does not resolve")
        if sid in twins:
            absent = [o for o in twins[sid] if o not in on_disk]
            amb = ("participants.tsv row identical in age, sex and disease duration to " + ",".join(twins[sid])
                   + (f" ({','.join(absent)} absent on disk, so no split is affected while it has no data)" if absent else "")
                   + ("; " + "; ".join(twin_updrs.get(sid, [])) if twin_updrs.get(sid) else ""))
        pm_rows.append({"source": SOURCE, "subject_ns": f"{SOURCE}:{sid}", "person_id": f"{SOURCE}:{sid}",
                        "rule": "bids_participant_single_session", "evidence": ev, "ambiguity": amb})
    os.makedirs(os.path.dirname(a.person_map), exist_ok=True)
    write_csv(a.person_map, pm_rows, ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"])

    # ---------------- admitted runs (draft input to OPEN-12)
    adm = [r for r in task_runs if r["relation_to_primary"] == "primary"]
    adm_rows = [{"run_id": r["run_id"], "sub": r["sub"], "task": r["task"], "acq": r["acq"],
                 "parts": r["parts"], "n_samples": r["n_samples"], "g15_header_verdict": r["g15_header_verdict"],
                 "g15_reasons": r["g15_reasons"],
                 "duplicates_left_out": [x["run_id"] for x in acqs[r["acquisition_id"]][1:]]} for r in adm]
    write_csv(os.path.join(a.out, "admitted_runs.csv"), adm_rows, list(adm_rows[0].keys()))

    # ---------------- addendum checks, provenance, summary
    facts = build_facts(files, fifs, headers, runs, task_runs, noise_runs, acqs, st_rows, oc_rows, fz_rows,
                        listed, on_disk, side_sets, emg_rows, muscle_hits, cm_rows, unmapped_rows, seg_rows, sbt,
                        label_census, fid_geom, twins)
    facts["twin_updrs"] = twin_updrs
    facts["missing_updrs"] = [(r["participant_id"], r["in_updrs_off_tsv"], r["in_updrs_on_tsv"])
                              for r in sub_rows if not r["on_disk"]]
    facts["multi_dig"] = [(s, n) for s, n, _ in multi_dig]
    facts["multi_dig_full"] = multi_dig
    facts["fid_nn_min"] = min((v[0] for v in nn_all.values() if v), default=float("nan"))
    checks = addendum_checks(facts)
    write_csv(os.path.join(a.out, "addendum_checks.csv"), checks,
              ["item", "addendum_statement", "measured", "verdict", "output"])
    prov = {"script": "scripts/probe_ds004998.py", "task": "T24", "raw_root": raw_root,
            "run_at": _dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "mne": __import__("mne").__version__, "priority": prio, "headers_only": True,
            "preload": False, "fif_files": len(fifs), "cache_version": CACHE_VERSION,
            "person_map": tree_relative(a.person_map)}
    prov.update(stamp_provenance(a.moveatlas))
    write_summary(os.path.join(a.out, "summary.md"), facts, checks, prov)
    prov["elapsed_s"] = round(time.time() - t_start, 1)
    prov["outputs"] = sorted(x for x in os.listdir(a.out) if not x.startswith("._")) + ["_provenance.json"]
    prov["outputs"] = sorted(set(prov["outputs"]))
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1, default=str)
    print("done in", prov["elapsed_s"], "s; outputs in", a.out)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def fid_geometry(fd):
    """Distances in mm: LPA-RPA, nasion-LPA, nasion-RPA."""
    l, n, r = fd["lpa"], fd["nasion"], fd["rpa"]
    return [1000 * math.dist(l, r), 1000 * math.dist(n, l), 1000 * math.dist(n, r)]


def build_facts(files, fifs, headers, runs, task_runs, noise_runs, acqs, st_rows, oc_rows, fz_rows,
                listed, on_disk, side_sets, emg_rows, muscle_hits, cm_rows, unmapped_rows, seg_rows, sbt_rows,
                label_census, fid_geom, twins):
    F = {}
    hd = [headers[os.path.basename(f)] for f in fifs]
    F["n_files"] = len(files)
    F["bytes_total"] = sum(os.path.getsize(f) for f in files)
    F["bytes_fif"] = sum(h["size_bytes"] for h in hd)
    F["n_fif"] = len(fifs)
    F["n_split_files"] = sum(1 for f in fifs if "_split-" in f)
    F["n_runs"] = len(runs)
    F["n_task_runs"] = len(task_runs)
    F["n_noise_runs"] = len(noise_runs)
    F["n_split_runs"] = sum(1 for r in runs if r["n_parts"] > 1)
    F["parts_per_split_run"] = sorted({r["n_parts"] for r in runs if r["n_parts"] > 1})
    F["continuity"] = Counter(r["continuity"] for r in runs)
    F["joins"] = Counter(j for r in runs for j in r["joins"])
    F["skip_entries"] = sum(r["skip_entries"] for r in runs)
    F["chain_ok"] = sum(1 for r in runs if r["chain_ok"])
    F["g15"] = Counter(r["g15_header_verdict"] for r in runs)
    F["g15_fail"] = [(r["run_id"], r["g15_reasons"]) for r in runs if r["g15_header_verdict"] != "pass"]
    F["samples_minus_json"] = Counter(r["samples_minus_json"] for r in runs)
    F["channels_match"] = sum(1 for r in runs if r["channels_match"])
    F["n_acquisitions_task"] = sum(1 for r in task_runs if r["relation_to_primary"] == "primary")
    F["relations"] = Counter(r["relation_to_primary"] for r in task_runs)
    cont = [r for r in task_runs if r["relation_to_primary"] != "primary"]
    F["contained"] = cont
    F["contained_tail"] = sum(1 for r in cont if r.get("ends_with_primary"))
    F["contained_run2"] = sum(1 for r in cont if r["run"] == "2")
    F["contained_subjects"] = sorted({r["sub"] for r in cont})
    F["contained_dur"] = sorted(r["duration_s"] for r in cont)
    F["offsets"] = Counter((r["nchan"], r["offset_in_primary_samples"]) for r in cont)
    F["contained_primary_split"] = sum(1 for r in cont
                                       if next(x for x in acqs[r["acquisition_id"]]
                                               if x["relation_to_primary"] == "primary")["n_parts"] > 1)
    F["label_conflict"] = [r for r in cont if (r.get("events_other_label_in_primary") or 0) > 0.5]
    F["label_agree"] = sum(1 for r in cont if (r.get("events_same_label_in_primary") or 0) >= 0.9)
    F["separate_run2"] = [r["run_id"] for r in task_runs if r["run"] == "2" and r["relation_to_primary"] == "primary"]
    F["vs_one_file"] = Counter(r["length_vs_one_file"] for r in task_runs if r["relation_to_primary"] == "primary")
    F["one_file_equal"] = [r for r in task_runs if r["length_vs_one_file"] == "equal"]
    F["one_file_longer_no_copy"] = [r["run_id"] for r in task_runs if r["length_vs_one_file"] == "longer_without_copy"]
    F["labels_inconsistent"] = [aid for aid, rs in acqs.items() if len({(r["acq"], r["task"]) for r in rs}) > 1]
    F["acq_time_mismatch"] = [r["run_id"] for r in runs if not r["acq_time_equals_meas_date"]]
    F["meas_days"] = sorted({str(r["meas_date"])[:10] for r in runs})
    F["acq_time_parts_unequal"] = [r["run_id"] for r in runs if not r["acq_time_parts_equal"]]
    F["subjects_on_disk"] = [s[4:] for s in on_disk]
    F["listed"] = [s[4:] for s in listed]
    F["missing"] = sorted(set(F["listed"]) - set(F["subjects_on_disk"]))
    F["extra"] = sorted(set(F["subjects_on_disk"]) - set(F["listed"]))
    F["noise_subjects"] = sorted({r["sub"] for r in noise_runs})
    F["noise"] = noise_runs
    F["noise_g15"] = Counter(r["g15_header_verdict"] for r in noise_runs)
    F["sfreq"] = Counter(r["sfreq"] for r in runs)
    F["n_meg"] = Counter(r["n_meg"] for r in task_runs)
    F["n_emg"] = Counter(r["n_emg"] for r in task_runs)
    F["n_eeg"] = Counter(r["n_eeg"] for r in task_runs)
    F["n_misc"] = Counter(r["n_misc"] for r in task_runs)
    F["emg_names"] = Counter(n for e in emg_rows for n in e["emg_channels"])
    F["emg_desc"] = sorted({d for e in emg_rows for d in e["channels_tsv_description"]})
    F["emg_extra_runs"] = [e["run_id"] for e in emg_rows if e["n_emg"] != 4]
    F["misc_runs"] = [(e["run_id"], e["misc_channels"]) for e in emg_rows if e["misc_channels"]]
    F["emg_bad"] = [(e["run_id"], e["emg_marked_bad"]) for e in emg_rows if e["emg_marked_bad"]]
    F["muscle_hits"] = muscle_hits
    stems = Counter()
    for h in hd:
        for st in sorted({re.sub(r"\d+$", "", n) for n in h["ch_names"]}):
            stems[st] += 1
    F["name_stems"] = stems
    F["proc_history"] = Counter(h["n_proc_history"] for h in hd)
    F["maxshield"] = sum(1 for h in hd if h["maxshield"])
    F["filters"] = Counter((round(h["highpass"], 3), h["lowpass"]) for h in hd)
    F["orig_format"] = Counter(h["orig_format"] for h in hd)
    F["warnings"] = Counter(w for h in hd for w in h["warnings"])
    F["anonymized"] = Counter(h["description"] for h in hd)
    F["contacts_per_side"] = Counter(u["contacts_per_side"] for u in unmapped_rows)
    F["unmapped"] = [(u["sub"], u["fif_eeg_unmapped"]) for u in unmapped_rows if u["fif_eeg_unmapped"]]
    F["montage_missing_in_fif"] = [(u["sub"], u["montage_not_in_fif"]) for u in unmapped_rows if u["montage_not_in_fif"]]
    F["contact_bad"] = [(c["sub"], c["old_name"], c["new_name"], c["runs_marked_bad"], c["task_runs"])
                        for c in cm_rows if c["runs_marked_bad"]]
    F["contact_side_disagree"] = [(c["sub"], c["old_name"], c["new_name"]) for c in cm_rows if not c["new_name_side_agrees"]]
    F["contact_absent_runs"] = [(c["sub"], c["old_name"], c["runs_with_channel"], c["task_runs"])
                                for c in cm_rows if c["runs_with_channel"] != c["task_runs"]]
    F["n_contact_rows"] = len(cm_rows)
    F["oc"] = oc_rows
    F["fz"] = fz_rows
    F["st_rows"] = st_rows
    F["side_sets"] = side_sets
    F["seg"] = seg_rows
    F["sbt"] = sbt_rows
    F["label_census"] = label_census
    F["events_missing"] = [(r["run_id"], r["events_parts_missing"]) for r in task_runs if r["events_parts_missing"]]
    F["events_beyond"] = [(r["run_id"], r["events_beyond_end"], r["events_beyond_end_max_s"])
                          for r in task_runs if r["events_beyond_end"]]
    F["events_before"] = [r["run_id"] for r in task_runs if r["events_before_start"]]
    F["parts_with_events"] = sum(r["n_parts"] - len(r["events_parts_missing"]) for r in task_runs)
    F["events_source"] = Counter(r["events_source"] for r in task_runs)
    F["parts_fif_equal"] = sum(r["parts_fif_equal_run_events"] for r in task_runs)
    F["n_parts_task"] = sum(r["n_parts"] for r in task_runs)
    F["parts_fif_differ"] = [p for r in task_runs for p in r["parts_fif_differ"]]
    F["later_compared"] = sum(r["later_part_tsv_compared"] for r in task_runs)
    F["later_equal"] = sum(r["later_part_tsv_equal"] for r in task_runs)
    F["later_unpaired"] = sum(r["later_part_tsv_unpaired"] for r in task_runs)
    F["later_max_on"] = max((r["later_part_tsv_max_onset_diff_s"] for r in task_runs
                             if r["later_part_tsv_max_onset_diff_s"] is not None), default=None)
    F["later_max_du"] = max((r["later_part_tsv_max_duration_diff_s"] for r in task_runs
                             if r["later_part_tsv_max_duration_diff_s"] is not None), default=None)
    F["later_over_1s"] = [r["run_id"] for r in task_runs if (r["later_part_tsv_max_onset_diff_s"] or 0) > 1.0
                          or (r["later_part_tsv_max_duration_diff_s"] or 0) > 1.0 or r["later_part_tsv_unpaired"]]
    F["split_runs_task"] = sum(1 for r in task_runs if r["n_parts"] > 1)
    F["fid_subjects"] = len(fid_geom)
    F["twins"] = twins
    F["durations_acq"] = sorted(r["duration_s"] for r in task_runs if r["relation_to_primary"] == "primary")
    F["n_subjects_runs"] = len({r["sub"] for r in task_runs})
    return F


def _fz_count(F, unit, reading, col):
    return sum(1 for r in F["fz"] if r["unit"] == unit and r["reading"] == reading and r[col])


def _fz_subjects(F, unit, reading, col):
    return sorted(r["sub"] for r in F["fz"] if r["unit"] == unit and r["reading"] == reading and r[col])


def _span(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def addendum_checks(F):
    C = []

    def add(item, statement, measured, verdict, output):
        C.append({"item": item, "addendum_statement": statement, "measured": measured, "verdict": verdict,
                  "output": output})

    gb = F["bytes_total"] / 1e9
    add("A1.1 size", "173.7 GB",
        f"{gb:.1f} GB ({F['bytes_total'] / 2**30:.1f} GiB) in {F['n_files']} files; FIF {F['bytes_fif'] / 1e9:.1f} GB "
        f"in {F['n_fif']} files", "holds" if abs(gb - 173.7) < 0.05 else "does not hold", "files.csv")
    n_on = len(F["subjects_on_disk"])
    add("A1.1 subjects", "20 PD on disk (22 in participants.tsv)",
        f"{n_on} subject folders on disk, {len(F['listed'])} rows in participants.tsv, {len(F['extra'])} folders "
        f"without a row", "holds" if (n_on == 20 and len(F["listed"]) == 22 and not F["extra"]) else "does not hold",
        "subjects.csv")
    add("A1.6 item 11", "2 of the 22 participants (sub-BgojEx, sub-zxEhes) are absent on disk",
        "absent: " + ", ".join("sub-" + s for s in F["missing"]),
        "holds" if tuple(F["missing"]) == ADDENDUM_MISSING else "does not hold", "subjects.csv")
    short = [r for r in F["noise"] if r["duration_s"] < 60]
    add("A1.1 empty room", "empty-room runs for 7 subjects",
        f"{len(F['noise_subjects'])} subjects with a task-noise run ("
        + ", ".join("sub-" + s for s in F["noise_subjects"]) + "); durations "
        + ", ".join(f"{r['duration_s']:.2f} s" for r in F["noise"])
        + (f"; {len(short)} shorter than 60 s ({', '.join('sub-' + r['sub'] for r in short)})" if short else ""),
        "holds" if len(F["noise_subjects"]) == 7 else "does not hold", "empty_room.csv")
    add("A1.1 MEG", "MEG 306 channels at 2000 Hz",
        f"MEG channels per task run {dict(F['n_meg'])}; sampling rate of every run {sorted(F['sfreq'])}",
        "holds" if set(F["n_meg"]) == {306} and set(F["sfreq"]) == {2000.0} else "does not hold", "runs.csv")
    add("A1.1 LFP", "bilateral STN LFP from externalized leads",
        f"montage.tsv maps a right and a left lead in all {len(F['subjects_on_disk'])} subjects; contacts per side "
        + ", ".join(f"{k}: {v} subjects" for k, v in sorted(F["contacts_per_side"].items()))
        + "; FIF EEG channels not in the montage: "
        + ("; ".join(f"sub-{s} {','.join(u)}" for s, u in F["unmapped"]) or "none")
        + f"; montage channels missing from a FIF: {len(F['montage_missing_in_fif'])}",
        "holds" if F["contacts_per_side"] and min(F["contacts_per_side"]) >= 4 else "does not hold",
        "contact_map.csv, contact_map_subjects.csv")
    add("A1.1 EMG; A1.7", "4 EMG at 2000 Hz, muscles not named; the muscles are not verified",
        f"EMG channels per task run {dict(sorted(F['n_emg'].items()))}: "
        + ", ".join(f"{k} in {v} runs" for k, v in sorted(F["emg_names"].items()))
        + f"; channels.tsv description {F['emg_desc']}; files on disk naming a muscle: {len(F['muscle_hits'])}",
        ("holds" if set(F["n_emg"]) == {4} else
         f"holds with a correction: {len(F['emg_extra_runs'])} runs carry "
         f"{'/'.join(str(k) for k in sorted(set(F['n_emg']) - {4}))} EMG channels"
         + (f", {', '.join(n for n, c in sorted(F['emg_names'].items()) if c < F['n_runs'])} not in every run"
            if any(c < F["n_runs"] for c in F["emg_names"].values()) else "")
         + f", subject(s) {', '.join(sorted({'sub-' + parse_bids_name(x + '_meg.fif')['sub'] for x in F['emg_extra_runs']}))}")
        if not F["muscle_hits"] else "does not hold", "emg_channels.csv")
    acc = [k for k in F["name_stems"] if re.search(r"acc|gyr|imu", k, re.I)]
    add("A1.1 accelerometer", "no accelerometer",
        "channel-name stems (files holding them): " + ", ".join(f"{k} {v}" for k, v in sorted(F["name_stems"].items()))
        + f"; MISC channels in {len(F['misc_runs'])} runs, their sensor not documented",
        "holds" if not acc else "does not hold", "emg_channels.csv")

    st = {(r["acq"], r["task"]): r for r in F["st_rows"]}
    mism = [f"{k[0]} {k[1]} {st.get(k, {}).get('bids_runs', 0)} vs {v}" for k, v in ADDENDUM_STATE_TASK.items()
            if st.get(k, {}).get("bids_runs", 0) != v]
    tot_runs = sum(r["bids_runs"] for r in F["st_rows"])
    tot_acq = sum(r["acquisitions"] for r in F["st_rows"])
    add("A1.5 runs by state and task",
        "MedOff HoldL 16, HoldR 9, MoveL 20, MoveR 8, Rest 3; MedOn HoldL 17, HoldR 9, MoveL 16, MoveR 6, Rest 4",
        f"BIDS runs (split parts merged): {tot_runs}, equal to the addendum in every cell"
        + (f" except {'; '.join(mism)}" if mism else "")
        + f"; acquisitions: {tot_acq} (MedOff "
        + ", ".join(f"{r['task']} {r['acquisitions']}" for r in F["st_rows"] if r["acq"] == "MedOff")
        + "; MedOn " + ", ".join(f"{r['task']} {r['acquisitions']}" for r in F["st_rows"] if r["acq"] == "MedOn")
        + f"); the other {tot_runs - tot_acq} BIDS runs lie inside the sample range of another run of their "
        f"recording ({F['contained_run2']} of them run-2 files, {F['contained_tail']} ending on its last sample)",
        ("does not hold" if mism else "holds" if tot_acq == tot_runs
         else "holds as a count of BIDS runs; does not hold as a count of recordings"),
        "state_task_counts.csv, runs.csv")

    oc = [r for r in F["oc"] if r["all_medoff_before_medon"] is not None]
    before = [r for r in oc if r["all_medoff_before_medon"]]
    inter = [r["sub"] for r in oc if r["interleaved"]]
    add("A1.5 order", "in 18 of 20 subjects every MedOff run precedes every MedOn run",
        f"{len(before)} of {len(oc)} subjects (acq_time of scans.tsv, acquisitions; the run-2 copies share the "
        f"acq_time of their run-1 and change nothing)", "holds" if len(before) == 18 and len(oc) == 20 else "does not hold",
        "order_census.csv")
    g0, g1 = _span([r["gap_acq_time_last_off_to_first_on_min"] for r in before])
    e0, e1 = _span([r["gap_record_end_off_to_record_start_on_min"] for r in before])
    add("A1.5 gap", "21 to 77 min between blocks",
        f"acq_time of the last MedOff acquisition to acq_time of the first MedOn acquisition: {g0:.1f} to {g1:.1f} min "
        f"over the {len(before)} ordered subjects; end of the last MedOff recording to start of the first MedOn "
        f"recording (header sample clock): {e0:.1f} to {e1:.1f} min",
        "holds (start to start); the recording-to-recording gap is shorter" if round(g0) == 21 and round(g1) == 77
        else "does not hold", "order_census.csv")
    add("A1.5 interleave", "sub-AbzsOg and sub-BYJoWR interleave",
        "interleaved: " + ", ".join("sub-" + s for s in inter),
        "holds" if sorted(inter) == ["AbzsOg", "BYJoWR"] else "does not hold", "order_census.csv")
    by = next((r for r in F["oc"] if r["sub"] == "BYJoWR"), None)
    if by is not None:
        after = by["medoff_after_first_medon"]
        add("A1.5 sub-BYJoWR", "one MedOff run at 09:20 follows the only MedOn run at 08:24, a possible label error",
            "order by acq_time: " + "; ".join(by["sequence_by_acq_time"])
            + f"; MedOn recordings {by['n_medon_acquisitions']} over {by['n_medon_runs']} MedOn BIDS runs (a run inside "
            "the sample range of another run of its recording is not counted); MedOff recordings after the first "
            f"MedOn one: {', '.join(after) or 'none'} (each a recording of its own, with its own meas_date)",
            "holds (one MedOn recording; the label question stays open)"
            if by["n_medon_acquisitions"] == 1 and len(after) == 1 else "does not hold", "order_census.csv, runs.csv")
    side = F["side_sets"].get("iDpl28")
    if side:
        ok = ("MoveL" in side["MedOff"] and "MoveL" not in side["MedOn"] and "MoveR" in side["MedOn"]
              and "MoveR" not in side["MedOff"])
        add("A1.5 sub-iDpl28", "MoveL runs only in MedOff and MoveR runs only in MedOn",
            f"MedOff tasks {side['MedOff']}; MedOn tasks {side['MedOn']}", "holds" if ok else "does not hold",
            "runs.csv")
    n_ties = _fz_count(F, "bids_runs", "side", "feasible_counting_ties")
    feas = {(u, rd): _fz_subjects(F, u, rd, "feasible") for u in ("bids_runs", "acquisitions") for rd in ("class", "side")}
    add("A1.5 and NR-6.4.4 order null",
        "an order-only null is feasible: 11 of 20 subjects have at least two MedOff runs of the same task",
        f"subjects with at least two MedOff BIDS runs of one task label: {n_ties} of {len(F['oc'])} (task class "
        f"reading: {_fz_count(F, 'bids_runs', 'class', 'feasible_counting_ties')}); in "
        f"{n_ties - len(feas[('bids_runs', 'side')])} of them the second run is the run-2 copy of the first, with the "
        f"same acq_time, so the two cannot be ordered; subjects with two ordered task-matched MedOff recordings: "
        f"{len(feas[('acquisitions', 'side')])} under the side-specific reading of OPEN-43 and "
        f"{len(feas[('acquisitions', 'class')])} under the hold, move, rest reading ("
        + ", ".join("sub-" + s for s in feas[("acquisitions", "class")]) + ")",
        ("holds" if n_ties == 11 and len(feas[("acquisitions", "side")]) == 11
         else ("does not hold: the count holds for BIDS runs, but " if n_ties == 11 else "does not hold: ")
         + "the order null is feasible in "
         + (f"{len(feas[('acquisitions', 'class')])} of {len(F['oc'])} subjects under either reading"
            if feas[("acquisitions", "class")] == feas[("acquisitions", "side")] else
            f"{len(feas[('acquisitions', 'side')])} (side-specific) and {len(feas[('acquisitions', 'class')])} "
            f"(hold, move, rest) of {len(F['oc'])} subjects")),
        "order_null_feasibility.csv")
    any_t = _fz_count(F, "bids_runs", "side", "any_task_two_or_more_counting_ties")
    any_a = _fz_count(F, "acquisitions", "side", "any_task_two_or_more")
    combos = Counter("+".join(x.split(":")[0] for x in r["medoff_per_task"]) for r in F["fz"]
                     if r["unit"] == "acquisitions" and r["reading"] == "class" and r["any_task_two_or_more"])
    add("A1.5 any task", "19 of 20 have at least two MedOff runs of any task",
        f"{any_t} of {len(F['oc'])} counting BIDS runs; {any_a} of {len(F['oc'])} counting ordered recordings, "
        "whose MedOff recordings span the task classes "
        + ", ".join(f"{k} in {v}" for k, v in sorted(combos.items()))
        + f", task-matched in {len(_fz_subjects(F, 'acquisitions', 'class', 'feasible'))}",
        (f"holds as a count of BIDS runs; {any_a} of {len(F['oc'])} as recordings" if any_t == 19 and len(F["oc"]) == 20
         else "does not hold"), "order_null_feasibility.csv")
    fails = F["g15_fail"]
    add("NR-2.12 G15 line (header half)", "matching sample counts per run; no report exists yet (OPEN-12)",
        f"{F['n_runs']} runs ({F['n_task_runs']} task, {F['n_noise_runs']} empty room) from {F['n_fif']} FIF files; "
        f"{F['n_split_runs']} runs in {F['parts_per_split_run']} parts, every join abutting (joins "
        f"{dict(F['joins'])}); data-skip entries {F['skip_entries']}; sample count minus meg.json "
        f"{dict(F['samples_minus_json'])}; channels.tsv equal to the FIF list in {F['channels_match']} runs; "
        f"header verdict {dict(F['g15'])}",
        "the header line passes for every run" if not fails else f"{len(fails)} runs fail the header line",
        "runs.csv, admitted_runs.csv")
    tk = [r for r in F["sbt"] if r["task"] != "Rest"]
    n_all = sum(r["acquisitions"] for r in tk)
    n300 = sum(r["first_rest_300s"] for r in tk)
    nbef = sum(r["rest_before_first_task"] for r in tk)
    add("A1.5 rest then task (README)", "5 min of rest followed by hold or move",
        "; ".join(f"{r['task']}: first rest 300 s in {r['first_rest_300s']} of {r['acquisitions']} acquisitions, "
                  f"rest before the first task segment in {r['rest_before_first_task']}, a rest segment after the "
                  f"first task segment in {r['with_rest_after_first_task']}" for r in tk),
        "holds" if n300 == n_all and nbef == n_all else
        f"partly: a 300 s first rest in {n300} of {n_all} task recordings, rest before the first task segment in "
        f"{nbef} of {n_all}", "segments.csv, segments_by_task.csv")
    add("A1.7 dyskinesia", "whether dyskinesia occurs in ds004998 MedOn runs is not verified",
        "event labels over every events.tsv: " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(F["label_census"].items(), key=lambda x: -x[1])),
        ("still not verified: no file labels dyskinesia, and headers cannot show it"
         if not any("dysk" in k.lower() for k in F["label_census"]) else
         "events labelled as dyskinesia exist: " + ", ".join(k for k in F["label_census"] if "dysk" in k.lower())),
        "event_labels.csv")
    add("NR-1.12 and OPEN-80 person ids", "draft reading: the source's own subject id is the person id",
        f"{len(F['subjects_on_disk'])} folders, one session (ses-PeriOp) each, one participants.tsv row each; FIF "
        f"headers anonymized ({', '.join(sorted(F['anonymized']))}), so no header id; the runs of a folder share one "
        f"digitization of the fiducials except in "
        + (", ".join(f"sub-{s} ({n})" for s, n in F.get("multi_dig", [])) or "none")
        + "; identical age, sex and disease duration in participants.tsv: "
        + ("; ".join(sorted({" and ".join(sorted([p] + o)) for p, o in F["twins"].items()})) if F["twins"] else "none"),
        "no contradiction found; map written", "reports/person_ids/map_pd_meg_stn_lfp_ds004998.csv")
    return C


def md_table(rows, cols, heads=None):
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def write_summary(path, F, checks, prov):
    stamp = prov.get("moveatlas_stamp") or {}
    head = stamp.get("head") if isinstance(stamp, dict) else None
    head = head or prov.get("moveatlas_head_log", "unknown")
    cont = F["contained"]
    fz_cls_acq = _fz_subjects(F, "acquisitions", "class", "feasible")
    L = []
    L.append("# ds004998 header-pass inventory")
    L.append("")
    L.append(f"Script: scripts/probe_ds004998.py (task T24). MoveAtlas commit: {head}. Run {prov['run_at']}, "
             f"mne {prov['mne']}. Source: {RAW_REL}.")
    L.append("")
    L.append("Headers only: every FIF was opened with mne.io.read_raw_fif(preload=False) and no data sample was "
             "read. No spectral quantity is computed, and no quantity is split by MedOff or MedOn other than run "
             "counts: the segment table carries no state column.")
    L.append("")
    L.append("## 1. Split parts and the header half of the G15 line (NR-2.12)")
    L.append("")
    L.append(f"The tree holds {F['n_files']} files, {F['bytes_total'] / 1e9:.1f} GB, of which {F['n_fif']} FIF files "
             f"({F['bytes_fif'] / 1e9:.1f} GB). {F['n_split_files']} FIF files are split parts; merged by run they "
             f"give {F['n_runs']} BIDS runs ({F['n_task_runs']} task runs, {F['n_noise_runs']} empty-room runs), "
             f"{F['n_split_runs']} of them in {', '.join(str(x) for x in F['parts_per_split_run'])} parts. In "
             f"{F['continuity'].get('contiguous', 0)} of the {F['n_split_runs']} split runs the parts share meas_date "
             f"and sampling rate and abut on the acquisition sample clock "
             f"(first_samp of part 2 = last_samp of part 1 + 1; join values {dict(F['joins'])}), and the chain of "
             f"files that MNE follows from the first part's next-file reference equals the listed parts in "
             f"{F['chain_ok']} of {F['n_runs']} runs. "
             f"The buffer directories hold {F['skip_entries']} data-skip entries, so MNE would mark no BAD_ACQ_SKIP. "
             f"The merged sample count equals RecordingDuration x sampling rate + 1 of meg.json in "
             f"{F['samples_minus_json'].get(0, 0)} of {F['n_runs']} runs ({dict(F['samples_minus_json'])} as count minus json), and channels.tsv lists the FIF channels in "
             f"order in {F['channels_match']} runs.")
    L.append("")
    L.append(f"Every channel of a FIF ({'/'.join(str(k) for k in sorted(F['n_meg']))} MEG, the LFP contacts stored as "
             "EEG0xx, EMG, EOG, triggers) is stored in the same data buffers, so all channels of a run have one "
             "sample count by construction. Processing history entries per file (an SSS or MaxFilter step would "
             f"leave one): {dict(F['proc_history'])}; stored value types {dict(F['orig_format'])}; acquisition "
             f"filters (high-pass, low-pass in Hz) {dict(F['filters'])}. The header "
             f"verdict is {dict(F['g15'])} over the {F['n_runs']} runs. What the header pass cannot show (a clock "
             "fault inside the amplifier chain, a flat or saturated LFP contact) needs the data and is left to the "
             "N2 reader.")
    L.append("")
    L.append("## 2. The run-2 files repeat the end of run-1")
    L.append("")
    offs = ", ".join(f"{o} samples at {n} channels ({c} runs)" for (n, o), c in sorted(F["offsets"].items()))
    nbytes = [n * o * 4 for (n, o) in F["offsets"]]
    b0, b1 = _span(nbytes)
    d0, d1 = _span(F["contained_dur"])
    L.append(f"{len(cont)} task runs carry the same meas_date (and the same scans.tsv acq_time) as another run of "
             f"the same subject, state and task, and their sample range lies inside that run's range: "
             f"{F['contained_run2']} of them are run-2 files, {F['contained_tail']} end on the same last sample as "
             f"their run-1, and they last {d0:.1f} to {d1:.1f} s. They start a fixed number of samples after the "
             f"start of their run-1 for a given channel count ({offs}); samples x channels x 4 bytes (the files "
             f"store {', '.join(sorted(F['orig_format']))} values) gives {b0 / 1e9:.4f} to {b1 / 1e9:.4f} GB, one "
             f"file size. The reading that fits: the acquisition wrote each recording in files of that size, each "
             f"run-2 is the second such file, and run-1 is the whole recording (the first file followed through to "
             f"the second). {F['contained_primary_split']} of the {len(cont)} run-1 files were split again on "
             f"writing (split-01, split-02); the others were not. Recordings whose runs carry different state or "
             f"task labels: {len(F['labels_inconsistent'])}.")
    L.append("")
    vs = F["vs_one_file"]
    L.append(f"Against that file size, the {F['n_acquisitions_task']} recordings are: "
             + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(vs.items()))
             + ". Longer than one file but without a copy: "
             + (", ".join(F["one_file_longer_no_copy"]) or "none")
             + ", so not every second file was published as a run-2. Exactly one file long: "
             + ("; ".join(f"{r['run_id']} ({r['n_samples']} samples, its last labelled segment ending "
                          f"{r['last_label_end_to_run_end_s']:.2f} s before the end)" for r in F["one_file_equal"])
                or "none")
             + (". A recording that stops on the exact file boundary, inside a labelled segment, reads as one whose "
                "second file is not in the dataset; the header pass cannot confirm it." if F["one_file_equal"] else "."))
    L.append("")
    L.append(f"The labels of the copies disagree with run-1 on shared samples in {len(F['label_conflict'])} "
             f"copy or copies: "
             + ("; ".join(f"{r['run_id']} ({100 * r['events_other_label_in_primary']:.1f} percent of its labelled "
                          f"time carries another label in run-1: {', '.join(r['label_pairs_in_conflict'])})" for r in F["label_conflict"]) or "none")
             + f". In {F['label_agree']} copies at least 90 percent of the labelled time carries the same label in "
             f"run-1. The one run-2 that is a separate recording is {', '.join(F['separate_run2']) or 'none'} "
             f"(its own meas_date).")
    L.append("")
    L.append(f"Consequence: the {F['n_task_runs']} task runs are {F['n_acquisitions_task']} recordings. "
             "admitted_runs.csv lists one run per recording (the run-1, parts merged) with the copies it leaves "
             "out, as a draft input to OPEN-12; a reader that loads both run-1 and run-2 counts the last "
             f"{d0:.0f} to {d1:.0f} s of a recording twice.")
    L.append("")
    L.append("## 3. State by task (run counts only)")
    L.append("")
    L.append(md_table(F["st_rows"], ["acq", "task", "bids_runs", "addendum_runs", "acquisitions", "subjects",
                                     "bids_run_files"],
                      ["acq", "task", "BIDS runs", "addendum 1.5", "recordings", "subjects", "FIF files"]))
    L.append("")
    L.append("## 4. Order census and the order-only null (NR-6.4.4, OPEN-43)")
    L.append("")
    oc = [r for r in F["oc"] if r["all_medoff_before_medon"] is not None]
    before = [r for r in oc if r["all_medoff_before_medon"]]
    g0, g1 = _span([r["gap_acq_time_last_off_to_first_on_min"] for r in before])
    e0, e1 = _span([r["gap_record_end_off_to_record_start_on_min"] for r in before])
    days = F["meas_days"]
    L.append((f"Every meas_date falls on {days[0]} after the anonymizing time shift, so the runs of a subject are "
              "ordered by time of day. " if len(days) == 1 else
              f"meas_date values fall on {len(days)} days after the anonymizing time shift. ")
             + f"In {len(before)} of {len(oc)} subjects every "
             f"MedOff recording starts before every MedOn recording; the gap from the acq_time of the last MedOff "
             f"recording to that of the first MedOn recording is {g0:.1f} to {g1:.1f} min, and from the end of the "
             f"last MedOff recording to the start of the first MedOn recording (header sample clock) {e0:.1f} to "
             f"{e1:.1f} min. Interleaved: "
             + "; ".join(f"sub-{r['sub']} ({', '.join(r['sequence_by_acq_time'])})" for r in oc if r["interleaved"])
             + ". The acq_time of scans.tsv equals the FIF meas_date in "
             f"{F['n_runs'] - len(F['acq_time_mismatch'])} of {F['n_runs']} runs.")
    L.append("")
    rows = []
    for unit in ("bids_runs", "acquisitions"):
        for rd in ("class", "side"):
            rows.append({"unit": "BIDS runs" if unit == "bids_runs" else "recordings",
                         "reading": "hold, move, rest" if rd == "class" else "side-specific label",
                         "ties": _fz_count(F, unit, rd, "feasible_counting_ties"),
                         "ordered": _fz_count(F, unit, rd, "feasible"),
                         "any": _fz_count(F, unit, rd, "any_task_two_or_more")})
    L.append(f"Subjects (of {len(F['oc'])}) with at least two task-matched MedOff units, counted with acq_time ties and with "
             "distinct acq_time only (an order null needs distinct times), and subjects with at least two MedOff "
             "units of distinct acq_time over all tasks:")
    L.append("")
    L.append(md_table(rows, ["unit", "reading", "ties", "ordered", "any"],
                      ["unit", "OPEN-43 reading", "two or more, ties counted", "two or more, distinct acq_time",
                       "two or more of any task, distinct acq_time"]))
    L.append("")
    feas_seq = "; ".join(
        f"sub-{r['sub']}: " + ", ".join(x for x in r["sequence_by_acq_time"] if " MedOff " in x)
        for r in F["oc"] if r["sub"] in fz_cls_acq)
    n_any = _fz_count(F, "acquisitions", "class", "any_task_two_or_more")
    n_one = sum(1 for r in F["fz"] if r["unit"] == "acquisitions" and r["reading"] == "class"
                and r["max_medoff_same_task"] == 1)
    L.append(f"The {_fz_count(F, 'bids_runs', 'side', 'feasible_counting_ties')} of 20 of addendum 1.5 is reproduced "
             "by counting BIDS runs with ties. Once the run-2 copies are set aside, the task-matched order null is "
             f"feasible in {len(fz_cls_acq)} subject(s) under either reading of OPEN-43 ({feas_seq}); in "
             f"{n_one} subjects every task has exactly one MedOff recording. In sub-BYJoWR the second MoveR MedOff "
             "recording is the one whose label addendum 1.5 already questions. Designs left to the owner (OPEN-12, "
             f"OPEN-43, OPEN-45): an order null between the hold and the move MedOff recordings ({n_any} subjects "
             "with two MedOff recordings of distinct acq_time, task and order confounded), an early against late "
             "contrast inside one MedOff recording, or no order null, with every medication result reported as "
             "'state or time' (NR-6.4.5).")
    L.append("")
    L.append("## 5. Contacts and EMG")
    L.append("")
    L.append(f"montage.tsv maps EEG0xx channel names to LFP-right-k and LFP-left-k in all {len(F['subjects_on_disk'])} "
             "subjects: " + ", ".join(f"{v} subjects with {k} contacts per side" for k, v in
                                     sorted(F["contacts_per_side"].items()))
             + ". Every mapped channel is present as an EEG channel in every task run of its subject ("
             f"{len(F['contact_absent_runs'])} exceptions); the side in the new name agrees with the column in "
             f"{F['n_contact_rows'] - len(F['contact_side_disagree'])} of {F['n_contact_rows']} rows. FIF EEG channels "
             "outside the montage: "
             + ("; ".join(f"sub-{s} {', '.join(u)}" for s, u in F["unmapped"]) or "none")
             + ". Contacts marked bad (FIF bads or channels.tsv status) in at least one run: "
             + ("; ".join(f"sub-{s} {o} ({n}) in {k} of {t} runs" for s, o, n, k, t in F["contact_bad"]) or "none")
             + ".")
    L.append("")
    L.append("EMG: " + ", ".join(f"{k} in {v} runs" for k, v in sorted(F["emg_names"].items()))
             + f"; channels.tsv describes them only as {F['emg_desc']}, and no file on disk names a muscle "
             f"({len(F['muscle_hits'])} hits over the json, tsv, README and CHANGES files). MISC channels: "
             + ("; ".join(f"{r} {', '.join(m)}" for r, m in F["misc_runs"]) or "none")
             + ". No channel name suggests an accelerometer.")
    L.append("")
    L.append("## 6. Rest and task segments (events.tsv)")
    L.append("")
    L.append(f"Event labels over every events.tsv: "
             + ", ".join(f"{k} {v}" for k, v in sorted(F["label_census"].items(), key=lambda x: -x[1]))
             + f". 'bad_flp' ({F['label_census'].get('bad_flp', 0)} events) reads as a typing variant of 'bad_lfp' "
             f"and is counted with it; 'HoldB' appears {F['label_census'].get('HoldB', 0)} time(s). "
             f"Each split part has its own events.tsv. In {F['events_source'].get('first_part_covers_run', 0)} of "
             f"{F['split_runs_task']} split task runs the events.tsv of split-01 reaches past the end of split-01 and "
             "holds the events of the whole run, with onsets from the run's first sample (events source over the "
             f"task runs: {dict(F['events_source'])}); these run events equal, within "
             f"10 ms, the FIF annotations as MNE reads them from every part (the whole run from the first part, "
             f"whose next-file reference MNE follows; the run events clipped to the part from the second, MNE "
             f"keeping only the annotations within the data it reads) in {F['parts_fif_equal']} of "
             f"{F['n_parts_task']} task-run parts. The events.tsv of "
             f"split-02 is a second, different labelling of the second part: it equals the run events clipped to "
             f"split-02 in {F['later_equal']} of {F['later_compared']} split-02 files, and paired events differ by up "
             f"to {fmt(F['later_max_on'])} s in onset and {fmt(F['later_max_du'])} s in duration, with "
             f"{F['later_unpaired']} events left unpaired; runs where the difference exceeds 1 s or an event is "
             f"unpaired: {len(F['later_over_1s'])}. Segments here use the run events; which labelling of the second "
             f"parts a reader takes is open (OPEN-12). {len(F['events_missing'])} split runs lack the events.tsv of "
             f"split-02 (" + ", ".join(r for r, p in F["events_missing"]) + "). Events beyond the end of their run: "
             f"{len(F['events_beyond'])} runs; before its start: {len(F['events_before'])}.")
    L.append("")
    L.append(md_table(F["sbt"], ["task", "acquisitions", "subjects", "first_rest_300s", "rest_before_first_task",
                                 "with_rest_after_first_task", "task_segments_total", "with_bad_lfp",
                                 "task_label_mismatch"],
                      ["task", "recordings", "subjects", "first rest 300 s", "rest before task",
                       "rest after first task", "task segments", "with bad_lfp", "label not the run's task"]))
    L.append("")
    L.append("segments.csv gives one row per recording (run-1, parts merged) keyed by subject, task and a blind key, "
             "without state, run or part; the pooled table above is over all recordings of a task.")
    L.append("")
    L.append("## 7. Empty room and missing subjects")
    L.append("")
    L.append(f"{len(F['noise'])} empty-room runs (task-noise): "
             + "; ".join(f"sub-{r['sub']} {r['duration_s']:.2f} s" for r in F["noise"])
             + ". Subjects of participants.tsv absent on disk: "
             + ", ".join(f"{sid} (UPDRS off row {fmt(a)}, on row {fmt(b)})" for sid, a, b in F.get("missing_updrs", []))
             + ".")
    L.append("")
    L.append("## 8. Person map (OPEN-80)")
    L.append("")
    L.append(f"reports/person_ids/map_pd_meg_stn_lfp_ds004998.csv maps the {len(F['subjects_on_disk'])} subject "
             "folders to person ids under the draft reading of NR-1.12 (the source's subject id is the person id). "
             "Each folder holds one session; the FIF headers are anonymized (random id, time shift), so no header "
             "field links two folders. All runs of a folder share one digitization of the fiducials, except in "
             + (", ".join(f"sub-{s} ({n} digitizations, {sp:.1f} mm apart)" for s, n, sp in F.get("multi_dig_full", []))
                or "none")
             + f". The three fiducial distances do not separate people: the nearest pair of subjects is "
             f"{F.get('fid_nn_min', float('nan')):.1f} mm apart, the scale of a re-digitization, so they support "
             "continuity within a folder and say nothing across folders. Participants with identical age, sex and disease "
             "duration in participants.tsv: "
             + ("; ".join(sorted({" and ".join(sorted([p] + o)) for p, o in F["twins"].items()})) if F["twins"]
                else "none")
             + ("; " + "; ".join(sorted({n for v in F.get("twin_updrs", {}).values() for n in v}))
                if F.get("twin_updrs") else "")
             + ". Absent subjects carry no data, so such a pair affects no split unless the absent one is fetched.")
    L.append("")
    L.append("## 9. Addendum checks")
    L.append("")
    L.append(md_table(checks, ["item", "addendum_statement", "verdict"], ["item", "statement", "verdict"]))
    L.append("")
    L.append("The measured values are in addendum_checks.csv.")
    L.append("")
    L.append("## 10. Open points")
    L.append("")
    qz = [r for r in F["noise"] if r["duration_s"] < 60]
    L.append(f"- OPEN-12: admitted_runs.csv is a draft list of {F['n_acquisitions_task']} task recordings, all "
             f"passing the header line, with the {len(cont)} run-2 copies left out; which labelling of the split-02 "
             "parts a reader uses (the run events of split-01, equal to the FIF annotations, or the split-02 "
             "events.tsv) is to be decided; the data half of the G15 line (sample-level checks) belongs to the "
             "N2 reader.")
    L.append(f"- OPEN-43 and NR-6.4.4: the task-matched order null is feasible in {len(fz_cls_acq)} of "
             f"{len(F['oc'])} subjects under either reading; addendum 1.5 and NR-6.4.4 count BIDS runs and need the "
             "owner's choice among the designs of section 4.")
    L.append("- Labels: "
             + ("; ".join(f"{r['run_id']} labels {100 * r['events_other_label_in_primary']:.1f} percent of its "
                          f"labelled time differently from run-1" for r in F["label_conflict"])
                or "no copy contradicts its run-1")
             + "; the sub-BYJoWR MoveR MedOff recording at 09:20 keeps its label question; rare labels: "
             f"'bad_flp' {F['label_census'].get('bad_flp', 0)}, 'HoldB' {F['label_census'].get('HoldB', 0)}.")
    if F["one_file_equal"]:
        L.append("- Possibly truncated: " + ", ".join(r["run_id"] for r in F["one_file_equal"])
                 + " end exactly on the acquisition-file boundary (section 2); a data reader can check whether the "
                 "signal ends mid-segment.")
    if qz:
        L.append("- Empty room: " + ", ".join(f"sub-{r['sub']} lasts {r['duration_s']:.2f} s" for r in qz) + ".")
    L.append("- EMG: no file names the muscles of EMG061 to EMG064 or EMG096 (addendum 1.7 stays open).")
    L.append("")
    txt = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
