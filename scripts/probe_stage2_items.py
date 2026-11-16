"""Stage-2 facts the build needs (task S2.0 of docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 7; sections 2, 4.1,
4.5 and 4.8 there). Reads non-lockbox RELEASE rows only, with mdfm.data.canonical and every flag at False.

Units (spec section 4.5, departure 5): weargait_pd:PD, pads:all, clinical_gait_multipatho:all,
imu_inthewild_iprognosis:all, stroke_gait_vancriekinge:stroke (S2a); physionet_gaitpdb:all, gaitndd:all,
gaitrec:A|C|H|K, modys_video:all (S2b). Rows: canonical._select(tier="RELEASE", lane="body", sources=<source>)
(the selection of canonical.iterate with allow_eval and allow_lockbox False: no lockbox, EVAL, CODY or
EXCLUDED_V1 row, and tier RELEASE so no INTERNAL row), restricted to the unit's groups. Records are read with
canonical.iterate(sources=<source>, modality=<m>, tier="RELEASE", lane="body") (canonical.check refuses each
row again before its record is read); rows of groups outside the unit are skipped unmeasured. Per source,
canonical.load on one lockbox row must raise canonical.Refused (checked before any container opens).

Items and their tables (reports/stage2_sources/items/):
1. recording_keys.csv: per unit and modality, the recording keys (mdfm.data.sources.recording_key of the index
   key, the string the task-vocabulary rules match) grouped by pattern (every run of digits written '#'):
   distinct keys, records, persons and up to three example keys. Index rows only, no record read. The input
   of the vocabulary-v2 rules (spec section 4.1).
2. Force zero runs (sources weargait_pd PD, physionet_gaitpdb, gaitndd, gaitrec A, C, H, K, stroke patients).
   Per record and foot whose total (channel 8) the sensor mask holds. A zero sample is a finite total equal to
   0. Since a sensor's unloaded level need not read 0 (an offset, plate noise around 0 cut at 0), the zeros
   are classed through the unloaded stretch that holds them: per foot, floor = min(0, p1 of the total), peak =
   its p99, threshold = floor + NEAR_ZERO [0.02] x (peak - floor); a sample is loaded above the threshold
   (non-finite samples are not loaded), and an unloaded stretch is a maximal run of samples not loaded. Each
   stretch is classed:
   - record_edge: it touches the first or the last sample of the record (no load on one of its sides, so the
     record holds no step around it);
   - both_unloaded: interior, and the other foot (when the mask holds it) is also unloaded at some sample of
     the stretch (walking always loads one foot);
   - short_gap: interior, the other foot loaded throughout or absent from the mask, shorter than SWING_MIN_S
     [cfg 0.15 s]: a dip of the signal or a plate-to-plate gap, too short for a swing (a swing lasts about 0.3
     to 0.6 s);
   - long: interior, the other foot loaded throughout or absent, longer than SWING_MAX_S [cfg 1.0 s];
   - swing: interior, the other foot loaded throughout, from SWING_MIN_S to SWING_MAX_S;
   - swing_other_unseen: interior, from SWING_MIN_S to SWING_MAX_S, the other foot absent from the mask.
   swing and swing_other_unseen are "inside a step" (a valid unloaded foot); the others are "off the plate"
   (no measurement of the foot, none that a step frames, or a gap too short to be a swing). force_zero_feet.csv:
   per record and foot the proposed fold of the person (train or dev, reports/stage2_sources/folds_proposed.csv;
   empty for a person without one), the zero, negative and unloaded samples, the floor, p99, threshold and
   low_mode, the stretches, unloaded samples and zero samples per class, the longest stretch, the first and
   last loaded sample, and for weargait_pd the zero samples its sample_valid marks invalid (gaitrec's per-foot
   zero counts sit in gaitrec_clock.csv instead, to keep the file small). A zero sample outside every unloaded
   stretch (a floor below 0, where 0 lies within the loaded range) is counted as zero_while_loaded. low_mode:
   the median of the samples in the fullest of LOW_MODE_BINS [20] equal bins spanning the foot's finite samples
   from their p0.5 to their median; it reads the unloaded level where the unloaded samples form the lowest
   mode (insoles, plates, gaitndd), not on gaitrec (unloaded share about 0.02, where it reads the stance
   plateau). Where the floor is below 0 (gaitndd) the unloaded level is a noisy band, the 2 percent threshold
   cuts inside it, and the class columns are not interpretable (summary.md says so per group);
   force_zero_summary.csv counts those feet (feet_floor_below_0). force_zero_summary.csv: per source and group,
   the same summed, the share of zero (and of unloaded) samples inside a step, the feet of the dev fold, and per
   class the stretch durations (s) p10, p50, p90 per stretch and per zero sample (each stretch weighted by the
   zero samples it holds). both_loaded_share per record (samples where both feet are loaded among samples where
   either is) is the double-support reference of the sources recorded on one clock. The thresholds were fixed
   before the first run; no rule may be tuned on these tables without leaving out the rows of fold dev.
3. gaitrec_clock.csv: per gaitrec record (groups A, C, H, K) with both feet in the mask: per foot the first and
   last loaded sample (total > 0: gaitrec totals never read below 0), the stance length, the total at the first loaded sample, and both_loaded_share. Rule,
   written before the run: on one clock, two consecutive contacts on two plates are a step apart (about half a
   gait cycle), so the two feet's first loaded samples differ by many samples in almost every record. Verdict
   NO SHARED CLOCK when at least CLOCK_SAME_ONSET [0.95] of the two-foot records start both feet at the same
   sample (each foot's stance cut at its own contact); SHARED CLOCK when at most 1 - CLOCK_SAME_ONSET do and
   the median both_loaded_share is at most 0.5; UNDETERMINED otherwise.
4. pads_pairs.csv: per pads person and task with both wrist records: n of each canonical record (length
   shared when equal); the raw time column of each wrist (datasets/imu/pads/raw/movement/timeseries/
   <id>_<task>_<LeftWrist|RightWrist>.txt, column Time in s, read only after raw_allowed has checked in the
   index that every row of the person is non-lockbox, RELEASE and not CODY): rows (equal to the canonical n
   or the pair is flagged), last time, median, p1, p99, min and max sample interval, the share of intervals
   outside 0.5 to 1.5 times the median; per pair the end-time difference right minus left and the largest
   |t_R(i) - t_L(i)| over the common samples (both columns start at 0, each on its watch's own clock); and
   the index lag (right against left, samples, +-XC_MAXLAG) maximizing the normalized cross-correlation of the
   gravity-free acceleration norm in the first and in the last XC_WIN samples, with its correlation (kept
   when at least XC_MIN_CORR); two wrists move differently, so the lag bears on neither the clock nor the start
   offset of the two watches (kept as a description of the movement, not as clock evidence). Rule, written
   before the run: length shared when n_L equals n_R in every pair; clock shared when the end-time difference
   is within one nominal interval (10 ms) in at least 0.95 of the pairs with raw time; otherwise NOT ONE CLOCK
   (spec section 4.5: two single-site windows). The verdict also counts the pairs per length (the deposit
   holds fixed lengths), the pairs whose two time columns start at 0, the median end-time difference per
   length, and the persons whose end-time difference keeps one sign over all their tasks. The start offset of
   the two watches is not measurable here (both columns start at 0, each on its own clock).
5. iprognosis_records.csv: per imu_inthewild_iprognosis record: the arrays and fields it carries, whether any
   of them is a time stamp (a key named time, times, t, timestamp or timestamps, or containing 'stamp'), fps,
   n, n / fps in s, the sample_valid share, and the share of consecutive samples identical on the three acc
   axes (a phone that repeats its last value when no new sample arrives shows here). The raw time stamps sit
   only in the source's raw pickle (MoveAtlas build_canonical_imu.py, ingest_imu_inthewild: column 0 in ns,
   fps = the session's median rate); that pickle holds every subject, the lockbox ones included, and a pickle
   cannot refuse a subject before it is parsed: it is not opened, so the interval spread is reported as not
   measurable from the canonical records.
6. _provenance.json: the MoveAtlas stamp (scripts/moveatlas_stamp.py) at start and end, HEAD included.
summary.json and summary.md: the verdicts, the done-when checks and the findings, every number read from the
CSVs. modys_video needs no item (reports/modys_structure/ lists both zips' members per training child).
CPU discipline: BelowNormal priority; while cache/runs/queue/SLOW exists the run sleeps 10 minutes between two
sources and logs it.
Run: CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/probe_stage2_items.py
[--out DIR] [--limit N: N records per source and modality, a dry run] [--md-only]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import collections  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm import paths  # noqa: E402
from mdfm.data import canonical as C  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from moveatlas_stamp import stamp  # noqa: E402

UNITS = (("weargait_pd", ("PD",)), ("pads", None), ("clinical_gait_multipatho", None),
         ("imu_inthewild_iprognosis", None), ("stroke_gait_vancriekinge", ("stroke",)),
         ("physionet_gaitpdb", None), ("gaitndd", None), ("gaitrec", ("A", "C", "H", "K")), ("modys_video", None))
RUN = {"weargait_pd": "S2a", "pads": "S2a", "clinical_gait_multipatho": "S2a", "imu_inthewild_iprognosis": "S2a",
       "stroke_gait_vancriekinge": "S2a", "physionet_gaitpdb": "S2b", "gaitndd": "S2b", "gaitrec": "S2b",
       "modys_video": "S2b"}
FORCE_SOURCES = ("weargait_pd", "physionet_gaitpdb", "gaitndd", "gaitrec", "stroke_gait_vancriekinge")
SWING_MAX_S = 1.0
SWING_MIN_S = 0.15
NEAR_ZERO = 0.02
LOW_MODE_BINS = 20
CLASSES = ("swing", "swing_other_unseen", "short_gap", "record_edge", "both_unloaded", "long")
INSIDE = ("swing", "swing_other_unseen")
FOLDS = os.path.join(TREE, "reports", "stage2_sources", "folds_proposed.csv")
CLOCK_SAME_ONSET = 0.95
CLOCK_OVERLAP_MAX = 0.5
PADS_RAW = "datasets/imu/pads/raw/movement/timeseries"
WRIST_FILE = {"wrist_L": "LeftWrist", "wrist_R": "RightWrist"}
PADS_DT = 0.01                       # nominal interval of the 100 Hz pads records (fps of the records)
PADS_SHARED_SHARE = 0.95
XC_WIN = 400
XC_MAXLAG = 50
XC_MIN_CORR = 0.5
IRREGULAR = (0.5, 1.5)
TIME_KEYS = ("time", "times", "t", "timestamp", "timestamps")
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
PAUSE_S = 600
NAN = float("nan")
OUTPUTS = sorted(["recording_keys.csv", "force_zero_feet.csv", "force_zero_summary.csv", "gaitrec_clock.csv",
                  "pads_pairs.csv", "iprognosis_records.csv", "summary.json", "summary.md", "_provenance.json",
                  "run.log", "pauses.csv"])
LOG = []


def log(msg):
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG[0].write(line + "\n")
        LOG[0].flush()


def set_below_normal():
    """BelowNormal priority for this process (Windows), so that training keeps the CPU."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetPriorityClass.restype = wintypes.BOOL
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000))
    os.nice(5)
    return True


def pause_if_slow(after, before, pauses):
    if not os.path.exists(SLOW):
        return
    t0 = dt.datetime.now()
    log(f"PAUSE {PAUSE_S} s between {after} and {before}: {SLOW} exists")
    time.sleep(PAUSE_S)
    t1 = dt.datetime.now()
    pauses.append({"started": t0.isoformat(timespec="seconds"), "ended": t1.isoformat(timespec="seconds"),
                   "seconds": round((t1 - t0).total_seconds(), 1), "after_source": after, "before_source": before})


def r4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if np.isfinite(x) else None


def qs(x, ps=(10, 50, 90)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return "" if not len(x) else "|".join(f"{r4(np.percentile(x, p)):g}" for p in ps)


def wqs(vals, weights, ps=(10, 50, 90)):
    """Percentiles of vals weighted by weights, written as qs writes them: each value counted weight times, the
    smallest value whose cumulative weight reaches p percent of the total (inverted CDF); values with a
    non-positive weight are left out; '' when none is left."""
    v = np.asarray(vals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(v) & (w > 0)
    v, w = v[keep], w[keep]
    if not len(v):
        return ""
    o = np.argsort(v, kind="stable")
    v, cw = v[o], np.cumsum(w[o])
    at = [min(int(np.searchsorted(cw, p / 100.0 * cw[-1], side="left")), len(v) - 1) for p in ps]
    return "|".join(f"{r4(v[i]):g}" for i in at)


# ---------------------------------------------------------------- units and rows
def unit_name(source):
    groups = dict(UNITS)[source]
    return f"{source}:{'|'.join(groups) if groups else 'all'}"


def in_unit(source, group):
    groups = dict(UNITS).get(source, ())
    return groups is None or group in groups


def unit_rows(sources=None):
    """Index rows of the units (module docstring): non-lockbox RELEASE rows of lane body, not EVAL, CODY or
    EXCLUDED_V1, one per record, in the unit's groups. No container is opened."""
    names = [s for s, _ in UNITS] if sources is None else list(sources)
    sel = C._select(tier="RELEASE", lane="body", sources=names)
    sel = sel[sel["source"].isin(names)]
    keep = [in_unit(s, g) for s, g in zip(sel["source"], sel["group"])]
    out = sel[keep].copy()
    assert not out["lockbox"].any() and (out["tier"] == "RELEASE").all() and not out["cody"].any()
    out["unit"] = [unit_name(s) for s in out["source"]]
    out["recording"] = [S.recording_key(s, m, k) for s, m, k in zip(out["source"], out["modality"], out["key"])]
    return out


def lockbox_refusals(idx, sources):
    """Per source: canonical.load on one lockbox row of the source must raise Refused (before any open)."""
    out = {}
    for s in sources:
        lb = idx[(idx["source"] == s) & idx["lockbox"]]
        tried = refused = 0
        if len(lb):
            tried = 1
            try:
                C.load(lb.iloc[0].to_dict())
            except C.Refused:
                refused = 1
        out[s] = {"lockbox_rows_tried": tried, "lockbox_rows_refused": refused}
    return out


def read_unit(source, modality, limit=0):
    """(row, record) pairs of a unit's rows of one modality, canonical.iterate with every flag False and tier
    RELEASE; rows outside the unit's groups are skipped before any measure."""
    n = 0
    for row, rec in C.iterate(sources=source, modality=modality, tier="RELEASE", lane="body"):
        if not in_unit(row["source"], row["group"]):
            continue
        assert not C._truthy(row["lockbox"]) and row["tier"] == "RELEASE"
        yield row, rec
        n += 1
        if limit and n >= limit:
            return


# ---------------------------------------------------------------- item 1: recording keys
def key_pattern(key):
    return re.sub(r"\d+", "#", str(key))


def key_table(rows):
    out = []
    d = rows.assign(pattern=[key_pattern(k) for k in rows["recording"]])
    for (unit, source, mod, pat), g in d.groupby(["unit", "source", "modality", "pattern"], sort=True):
        keys = sorted(set(g["recording"]))
        out.append({"unit": unit, "run": RUN[source], "source": source, "modality": mod, "key_pattern": pat,
                    "recording_keys": len(keys), "records": len(g), "persons": g["person_id"].nunique(),
                    "examples": "; ".join(keys[:3])})
    return pd.DataFrame(out, columns=["unit", "run", "source", "modality", "key_pattern", "recording_keys",
                                      "records", "persons", "examples"])


# ---------------------------------------------------------------- item 2: force zero runs
def runs_of(mask):
    """(starts, ends) of the maximal True runs of a 1-D bool array, ends exclusive."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def load_levels(tot):
    """(floor, peak, threshold) of one foot's total: floor = min(0, p1), peak = p99 over finite samples,
    threshold = floor + NEAR_ZERO x (peak - floor); NaN when no sample is finite."""
    tot = np.asarray(tot, dtype=np.float64)
    fin = np.isfinite(tot)
    if not fin.any():
        return NAN, NAN, NAN
    p1, p99 = np.percentile(tot[fin], [1, 99])
    floor = min(0.0, float(p1))
    return floor, float(p99), floor + NEAR_ZERO * (float(p99) - floor)


def loaded_mask(tot):
    """Samples of one foot's total above its threshold (load_levels); non-finite samples are not loaded."""
    tot = np.asarray(tot, dtype=np.float64)
    u = load_levels(tot)[2]
    return np.isfinite(tot) & (tot > u) if np.isfinite(u) else np.zeros(len(tot), bool)


def low_mode(tot, bins=LOW_MODE_BINS):
    """The level of the lowest mode of one foot's total (module docstring, item 2): the median of the samples
    in the fullest of `bins` equal bins spanning the finite samples from their p0.5 to their median; NaN when no
    sample is finite."""
    t = np.asarray(tot, dtype=np.float64)
    t = t[np.isfinite(t)]
    if not len(t):
        return NAN
    lo, med = (float(q) for q in np.percentile(t, [0.5, 50]))
    v = t[(t >= lo) & (t <= med)]
    if med <= lo:
        return float(np.median(v))
    h, e = np.histogram(v, bins=bins, range=(lo, med))
    k = int(np.argmax(h))
    return float(np.median(v[(v >= e[k]) & (v <= e[k + 1])]))


def classify_runs(tot, other, fps, swing_max_s=SWING_MAX_S, swing_min_s=SWING_MIN_S):
    """[(start, end, class)] of the unloaded stretches of one foot's total (module docstring, item 2). other:
    the other foot's total, or None when the mask does not hold it."""
    tot = np.asarray(tot, dtype=np.float64)
    n = len(tot)
    loaded_other = None if other is None else loaded_mask(other)
    out = []
    for a, b in zip(*runs_of(~loaded_mask(tot))):
        a, b = int(a), int(b)
        if a == 0 or b == n:
            cls = "record_edge"
        elif loaded_other is not None and not loaded_other[a:b].all():
            cls = "both_unloaded"
        elif (b - a) / fps < swing_min_s:
            cls = "short_gap"
        elif (b - a) / fps > swing_max_s:
            cls = "long"
        else:
            cls = "swing" if loaded_other is not None else "swing_other_unseen"
        out.append((a, b, cls))
    return out


def foot_facts(tot, other, fps, valid=None):
    """Per-foot row of force_zero_feet.csv (without identifiers) and, per class, the (duration s, zero samples)
    of each stretch."""
    tot = np.asarray(tot, dtype=np.float64)
    fin = np.isfinite(tot)
    zero = fin & (tot == 0)
    floor, peak, u = load_levels(tot)
    runs = classify_runs(tot, other, fps)
    ld = np.flatnonzero(loaded_mask(tot))
    f = {"n": len(tot), "nonfinite": int((~fin).sum()), "zero_samples": int(zero.sum()),
         "negative_samples": int((fin & (tot < 0)).sum()), "floor": r4(floor), "p99": r4(peak), "threshold": r4(u),
         "low_mode": r4(low_mode(tot)),
         "unloaded_samples": int(sum(b - a for a, b, _ in runs)), "stretches": len(runs),
         "stretches_with_a_zero": int(sum(bool(zero[a:b].any()) for a, b, _ in runs)),
         "first_loaded": int(ld[0]) if len(ld) else -1, "last_loaded": int(ld[-1]) if len(ld) else -1,
         "longest_stretch_s": r4(max(((b - a) / fps for a, b, _ in runs), default=0.0))}
    durs = collections.defaultdict(list)
    for c in CLASSES:
        sel = [(a, b) for a, b, k in runs if k == c]
        f[f"stretches_{c}"] = len(sel)
        f[f"unloaded_{c}"] = int(sum(b - a for a, b in sel))
        f[f"zero_{c}"] = int(sum(int(zero[a:b].sum()) for a, b in sel))
        durs[c] = [((b - a) / fps, int(zero[a:b].sum())) for a, b in sel]
    f["zero_inside_step"] = sum(f[f"zero_{c}"] for c in INSIDE)
    f["zero_off_plate"] = sum(f[f"zero_{c}"] for c in CLASSES if c not in INSIDE)
    f["zero_while_loaded"] = f["zero_samples"] - f["zero_inside_step"] - f["zero_off_plate"]
    f["unloaded_inside_step"] = sum(f[f"unloaded_{c}"] for c in INSIDE)
    f["unloaded_off_plate"] = f["unloaded_samples"] - f["unloaded_inside_step"]
    if valid is not None:
        f["zero_samples_marked_invalid"] = int((zero & ~np.asarray(valid, bool)).sum())
    return f, durs


def both_loaded_share(x, mask):
    """Samples where both feet are loaded (loaded_mask) among samples where either is, both feet in the mask;
    NaN otherwise."""
    if not (mask[0, 8] and mask[1, 8]):
        return NAN
    l0, l1 = loaded_mask(x[:, 0, 8]), loaded_mask(x[:, 1, 8])
    either = (l0 | l1).sum()
    return float((l0 & l1).sum() / either) if either else NAN


def force_record(row, rec):
    """(foot rows, durations per class, record facts) of one force record."""
    x = np.asarray(rec["force"], dtype=np.float64)
    m = np.asarray(rec["sensor_mask"], bool)
    fps = float(rec["fps"])
    sv = rec.get("sample_valid")
    sv = None if sv is None else np.asarray(sv, bool).reshape(x.shape[0], -1)
    feet, durs = [], collections.defaultdict(list)
    for j, side in enumerate("LR"):
        if not m[j, 8]:
            continue
        other = x[:, 1 - j, 8] if m[1 - j, 8] else None
        f, d = foot_facts(x[:, j, 8], other, fps, None if sv is None else sv[:, j])
        feet.append(dict({"source": row["source"], "group": row["group"], "person_id": row["person_id"],
                          "key": row["key"], "foot": side, "fps": fps, "other_foot_in_mask": other is not None}, **f))
        for c, v in d.items():
            durs[c] += v
    return feet, durs, {"both_loaded_share": both_loaded_share(x, m)}


def read_folds(path=FOLDS):
    """{(source, person_id): fold} of the proposed stage-2 folds (reports/stage2_sources/folds_proposed.csv, an
    MD_FM report: identifiers and folds only); {} when the file is absent."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    return {(s, p): f for s, p, f in zip(d["source"], d["person_id"], d["fold"])}


def sha256_of(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def add_folds(F, folds):
    """F with a column fold (after person_id): the person's proposed fold, '' when it has none."""
    F = F.copy()
    F.insert(int(F.columns.get_loc("person_id")) + 1, "fold",
             [folds.get((s, p), "") for s, p in zip(F["source"], F["person_id"])])
    return F


def force_summary(F, durs, recfacts):
    rows = []
    for (s, g), f in F.groupby(["source", "group"], sort=True):
        n, z, un = int(f["n"].sum()), int(f["zero_samples"].sum()), int(f["unloaded_samples"].sum())
        r = {"source": s, "group": g, "records": int(f[["person_id", "key"]].drop_duplicates().shape[0]),
             "feet": len(f), "feet_other_foot_unseen": int((~f["other_foot_in_mask"].astype(bool)).sum()),
             "feet_dev": int((f["fold"] == "dev").sum()) if "fold" in f else 0,
             "samples": n, "zero_samples": z, "zero_share": r4(z / max(1, n)), "negative_share": r4(f["negative_samples"].sum() / max(1, n)),
             "floor_p10_p50_p90": qs(f["floor"].astype(float)), "p99_p10_p50_p90": qs(f["p99"].astype(float)),
             "feet_floor_below_0": int((f["floor"].astype(float) < 0).sum()),
             "low_mode_p10_p50_p90": qs(f["low_mode"].astype(float)),
             "unloaded_share": r4(un / max(1, n)),
             "unloaded_inside_step_share": r4(f["unloaded_inside_step"].sum() / un) if un else None,
             "zero_inside_step_share": r4(f["zero_inside_step"].sum() / z) if z else None,
             "zero_off_plate_share": r4(f["zero_off_plate"].sum() / z) if z else None,
             "zero_while_loaded_share": r4(f["zero_while_loaded"].sum() / z) if z else None,
             "feet_without_a_zero": int((f["zero_samples"] == 0).sum())}
        for c in CLASSES:
            r[f"stretches_{c}"] = int(f[f"stretches_{c}"].sum())
            r[f"zero_{c}"] = int(f[f"zero_{c}"].sum())
            dz = durs.get((s, g, c), [])
            r[f"dur_s_{c}_per_stretch_p10_p50_p90"] = qs([d for d, _ in dz])
            r[f"dur_s_{c}_per_zero_p10_p50_p90"] = wqs([d for d, _ in dz], [k for _, k in dz])
        if "zero_samples_marked_invalid" in f and f["zero_samples_marked_invalid"].notna().any():
            r["zero_samples_marked_invalid"] = int(f["zero_samples_marked_invalid"].fillna(0).astype(int).sum())
        r["both_loaded_share_p10_p50_p90"] = qs(recfacts.get((s, g), []))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- item 3: gaitrec clock
GAITREC_FOOT_COLS = ("zero_samples", "zero_inside_step", "zero_off_plate", "zero_while_loaded", "stretches")


def gaitrec_clock_row(row, rec, feet=()):
    """One row of gaitrec_clock.csv; feet: the record's force_zero rows (force_record), whose zero counts are
    carried per foot (gaitrec's per-foot zero facts live here, not in force_zero_feet.csv)."""
    x = np.asarray(rec["force"], dtype=np.float64)
    m = np.asarray(rec["sensor_mask"], bool)
    r = {"group": row["group"], "person_id": row["person_id"], "key": row["key"], "n": x.shape[0],
         "two_feet": bool(m[0, 8] and m[1, 8])}
    for j, side in enumerate("LR"):
        t = x[:, j, 8]
        ld = np.flatnonzero(np.isfinite(t) & (t > 0)) if m[j, 8] else np.array([], int)
        r[f"first_loaded_{side}"] = int(ld[0]) if len(ld) else -1
        r[f"last_loaded_{side}"] = int(ld[-1]) if len(ld) else -1
        r[f"stance_{side}"] = int(ld[-1] - ld[0] + 1) if len(ld) else 0
        r[f"total_at_first_{side}"] = r4(t[ld[0]]) if len(ld) else None
    r["same_first_loaded"] = bool(r["two_feet"] and r["first_loaded_L"] == r["first_loaded_R"] >= 0)
    r["both_loaded_share"] = r4(both_loaded_share(x, m))
    for f in feet:
        for c in GAITREC_FOOT_COLS:
            r[f"{c}_{f['foot']}"] = f[c]
    return r


def clock_verdict(G):
    two = G[G["two_feet"].astype(bool)]
    n = len(two)
    same = int(two["same_first_loaded"].astype(bool).sum())
    share = same / n if n else NAN
    bl = two["both_loaded_share"].astype(float).to_numpy()
    ov = float(np.nanmedian(bl)) if np.isfinite(bl).any() else NAN
    if n and share >= CLOCK_SAME_ONSET:
        v = "NO SHARED CLOCK"
    elif n and share <= 1 - CLOCK_SAME_ONSET and ov <= CLOCK_OVERLAP_MAX:
        v = "SHARED CLOCK"
    else:
        v = "UNDETERMINED"
    lag = (two["first_loaded_R"] - two["first_loaded_L"]).abs()
    return {"verdict": v, "records": len(G), "two_feet_records": n, "same_first_loaded": same,
            "same_first_loaded_share": r4(share), "both_loaded_share_median": r4(ov),
            "first_loaded_diff_samples_p50_p90_max": qs(lag.astype(float), (50, 90, 100)),
            "total_at_first_p10_p50_p90": qs(pd.concat([two["total_at_first_L"], two["total_at_first_R"]]).astype(float)),
            "rule": f"NO SHARED CLOCK when >= {CLOCK_SAME_ONSET} of two-foot records start both feet at the same "
                    f"sample; SHARED CLOCK when <= {1 - CLOCK_SAME_ONSET:g} do and the median both_loaded_share "
                    f"<= {CLOCK_OVERLAP_MAX}; UNDETERMINED otherwise"}


# ---------------------------------------------------------------- item 4: pads pairs
class RawGate:
    """The persons whose raw files may be opened: listed in the index (subject_ns or person_id) with every row
    non-lockbox, RELEASE and not CODY. Built once from the index; no file is opened."""

    def __init__(self, idx):
        bad = idx["lockbox"] | (idx["tier"] != "RELEASE") | idx["cody"]
        self.known = frozenset(idx["subject_ns"]) | frozenset(idx["person_id"])
        self.bad = frozenset(idx.loc[bad, "subject_ns"]) | frozenset(idx.loc[bad, "person_id"])

    def check(self, subject_ns):
        """Refuse (canonical.Refused) unless the person's raw files may be opened (class docstring)."""
        if subject_ns not in self.known:
            raise C.Refused(f"{subject_ns}: not in the index")
        if subject_ns in self.bad or C.is_cody(subject_ns):
            raise C.Refused(f"{subject_ns}: a lockbox, non-RELEASE or CODY row; its raw files are never opened")
        return True


def raw_allowed(gate, subject_ns):
    """gate.check(subject_ns): called before any raw file of the person is opened."""
    return gate.check(subject_ns)


def pads_raw_path(subject_ns, task, wrist):
    return f"{paths.MOVEATLAS}/{PADS_RAW}/{subject_ns.split(':', 1)[1]}_{task}_{WRIST_FILE[wrist]}.txt"


def read_pads_time(gate, subject_ns, task, wrist):
    """Time column (s) of one raw pads wrist file, after raw_allowed."""
    raw_allowed(gate, subject_ns)
    return pd.read_csv(pads_raw_path(subject_ns, task, wrist), header=None, usecols=[0],
                       dtype=np.float64).to_numpy()[:, 0]


def time_facts(t):
    t = np.asarray(t, dtype=np.float64)
    d = np.diff(t)
    med = float(np.median(d)) if len(d) else NAN
    irr = ((d < IRREGULAR[0] * med) | (d > IRREGULAR[1] * med)).mean() if len(d) and med > 0 else NAN
    return {"rows": len(t), "t0": r4(t[0]) if len(t) else None, "t_end": float(t[-1]) if len(t) else NAN,
            "dt_median": med, "dt_p1": float(np.percentile(d, 1)) if len(d) else NAN,
            "dt_p99": float(np.percentile(d, 99)) if len(d) else NAN,
            "dt_min": float(d.min()) if len(d) else NAN, "dt_max": float(d.max()) if len(d) else NAN,
            "irregular_share": float(irr)}


def xcorr_lag(a, b, maxlag=XC_MAXLAG):
    """(lag, r): the lag in samples (b delayed by lag against a: a[t] best matches b[t + lag]) maximizing the
    Pearson correlation of the overlapping parts, |lag| <= maxlag, and that correlation."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    best = (0, -np.inf)
    for lag in range(-maxlag, maxlag + 1):
        x, y = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
        n = min(len(x), len(y))
        if n < 10:
            continue
        x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        r = float((x * y).sum() / den) if den > 0 else -np.inf
        if r > best[1]:
            best = (lag, r)
    return best[0], (best[1] if np.isfinite(best[1]) else NAN)


def acc_norm(rec):
    s = np.asarray(rec["signal"], np.float64)[:, 0, 0:3]
    return np.linalg.norm(s, axis=1)


def pads_pair(gate, subject_ns, person_id, task, recs, raw=True):
    """One row of pads_pairs.csv from the two canonical records {wrist: record}."""
    aL, aR = acc_norm(recs["wrist_L"]), acc_norm(recs["wrist_R"])
    r = {"person_id": person_id, "task": task, "n_L": len(aL), "n_R": len(aR), "same_length": len(aL) == len(aR)}
    for name, sl in (("first", slice(0, XC_WIN)), ("last", slice(-XC_WIN, None))):
        lag, c = xcorr_lag(aL[sl], aR[sl])
        r[f"lag_{name}"] = lag if np.isfinite(c) and c >= XC_MIN_CORR else None
        r[f"corr_{name}"] = r4(c)
    if raw:
        tf = {}
        for w, side in (("wrist_L", "L"), ("wrist_R", "R")):
            t = read_pads_time(gate, subject_ns, task, w)
            tf[side] = t
            f = time_facts(t)
            r.update({f"raw_rows_{side}": f["rows"], f"raw_t0_{side}": f["t0"], f"raw_t_end_{side}": r4(f["t_end"]),
                      f"dt_median_ms_{side}": r4(1e3 * f["dt_median"]), f"dt_p1_ms_{side}": r4(1e3 * f["dt_p1"]),
                      f"dt_p99_ms_{side}": r4(1e3 * f["dt_p99"]), f"dt_min_ms_{side}": r4(1e3 * f["dt_min"]),
                      f"dt_max_ms_{side}": r4(1e3 * f["dt_max"]), f"irregular_share_{side}": r4(f["irregular_share"])})
        k = min(len(tf["L"]), len(tf["R"]))
        r["raw_rows_equal_canonical"] = r["raw_rows_L"] == r["n_L"] and r["raw_rows_R"] == r["n_R"]
        r["end_diff_ms"] = r4(1e3 * (tf["R"][-1] - tf["L"][-1]))
        r["max_abs_time_diff_ms"] = r4(1e3 * float(np.abs(tf["R"][:k] - tf["L"][:k]).max())) if k else None
    return r


def pads_verdict(PP):
    n = len(PP)
    same_len = int(PP["same_length"].astype(bool).sum())
    raw = PP[PP["end_diff_ms"].notna()] if "end_diff_ms" in PP else PP.iloc[:0]
    within = int((raw["end_diff_ms"].astype(float).abs() <= 1e3 * PADS_DT).sum()) if len(raw) else 0
    share = within / len(raw) if len(raw) else NAN
    length = "LENGTH SHARED" if n and same_len == n else "LENGTH NOT SHARED"
    clock = ("ONE CLOCK" if len(raw) and share >= PADS_SHARED_SHARE else
             "NOT ONE CLOCK" if len(raw) else "CLOCK NOT MEASURED")
    out = {"verdict": f"{length}; {clock}", "pairs": n, "same_length": same_len, "pairs_with_raw_time": len(raw),
           "end_diff_within_one_interval": within, "end_diff_within_one_interval_share": r4(share),
           "pairs_per_n_L": {str(k): int(v) for k, v in PP["n_L"].value_counts().sort_index().items()} if n else {},
           "rule": f"length shared when n_L equals n_R in every pair; one clock when |end-time difference| <= "
                   f"{1e3 * PADS_DT:g} ms in >= {PADS_SHARED_SHARE} of the pairs with raw time",
           "lag_reading": "two wrists move differently: the acc-norm lag bears on neither the clock nor the start "
                          "offset of the two watches"}
    if len(raw):
        e = raw["end_diff_ms"].astype(float)
        sign = np.sign(e).groupby(raw["person_id"]).nunique()
        out.update({"raw_t0_zero_pairs": int(((raw["raw_t0_L"].astype(float) == 0)
                                              & (raw["raw_t0_R"].astype(float) == 0)).sum()),
                    "end_diff_ms_p50_per_n_L": {str(k): r4(g.median()) for k, g in e.groupby(raw["n_L"])},
                    "persons_with_raw_time": len(sign), "persons_end_diff_one_sign": int((sign == 1).sum())})
        out.update({"end_diff_ms_p10_p50_p90": qs(e), "end_diff_ms_min_max": qs(e, (0, 100)),
                    "end_diff_ms_per_s_p50": r4(np.median(e / raw["raw_t_end_L"].astype(float))),
                    "max_abs_time_diff_ms_p50": r4(raw["max_abs_time_diff_ms"].astype(float).median()),
                    "raw_rows_equal_canonical": int(raw["raw_rows_equal_canonical"].astype(bool).sum()),
                    "dt_median_ms_L_p10_p50_p90": qs(raw["dt_median_ms_L"].astype(float)),
                    "dt_median_ms_R_p10_p50_p90": qs(raw["dt_median_ms_R"].astype(float)),
                    "dt_p1_ms_p50_LR": [r4(raw["dt_p1_ms_L"].astype(float).median()),
                                        r4(raw["dt_p1_ms_R"].astype(float).median())],
                    "dt_p99_ms_p50_LR": [r4(raw["dt_p99_ms_L"].astype(float).median()),
                                         r4(raw["dt_p99_ms_R"].astype(float).median())],
                    "irregular_share_p50_LR": [r4(raw["irregular_share_L"].astype(float).median()),
                                               r4(raw["irregular_share_R"].astype(float).median())]})
    for name in ("first", "last"):
        lag = PP[f"lag_{name}"].dropna().astype(float)
        out[f"lag_{name}_pairs_corr_ok"] = len(lag)
        out[f"lag_{name}_p10_p50_p90"] = qs(lag)
        out[f"lag_{name}_within_1"] = int((lag.abs() <= 1).sum())
    return out


# ---------------------------------------------------------------- item 5: iprognosis
def is_time_key(k):
    k = str(k).lower()
    return k in TIME_KEYS or "stamp" in k


def ipr_row(row, rec):
    arrays = sorted(k for k, v in rec.items() if isinstance(v, np.ndarray))
    fields = sorted(k for k, v in rec.items() if not isinstance(v, np.ndarray) and k not in ("manifest",))
    tkeys = [k for k in arrays + fields if is_time_key(k)]
    x = np.asarray(rec["signal"], np.float64)[:, 0, 0:3]
    sv = rec.get("sample_valid")
    rep = (np.all(np.diff(x, axis=0) == 0, axis=1).mean()) if len(x) > 1 else NAN
    fps = float(rec["fps"])
    return {"person_id": row["person_id"], "key": row["key"], "n": len(x), "fps": r4(fps),
            "seconds_at_fps": r4(len(x) / fps) if fps > 0 else None,
            "has_time_stamps": bool(tkeys), "time_keys": "|".join(tkeys), "arrays": "|".join(arrays),
            "sample_valid_share": r4(np.asarray(sv, bool).mean()) if sv is not None else None,
            "repeat_share": r4(rep)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "stage2_sources", "items"))
    ap.add_argument("--limit", type=int, default=0, help="records per source and modality (dry run)")
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args()
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.md_only:
        write_md(a.out)
        return
    LOG.append(open(os.path.join(a.out, "run.log"), "w", encoding="utf-8", newline="\n"))
    t_start = time.time()
    log(f"start, BelowNormal priority set: {set_below_normal()}, limit {a.limit}, out {a.out}")
    st0 = stamp(paths.MOVEATLAS)
    idx = C.index()
    w = lambda df, name: df.to_csv(os.path.join(a.out, name), index=False, lineterminator="\n")  # noqa: E731
    pauses, stats = [], {}
    names = [s for s, _ in UNITS]
    refusals = lockbox_refusals(idx, names)
    log(f"lockbox refusals: {refusals}")

    # item 1
    rows = unit_rows()
    w(key_table(rows), "recording_keys.csv")
    log(f"recording keys: {len(rows)} rows of {rows['unit'].nunique()} units")

    # item 2 and 3
    feet, durs, recfacts, clock = [], collections.defaultdict(list), collections.defaultdict(list), []
    prev = None
    for s in FORCE_SOURCES:
        if prev:
            pause_if_slow(prev, s, pauses)
        prev = s
        t0, n = time.time(), 0
        for row, rec in read_unit(s, "force", a.limit):
            fr, d, rf = force_record(row, rec)
            feet += fr
            for c, v in d.items():
                durs[(s, row["group"], c)] += v
            recfacts[(s, row["group"])].append(rf["both_loaded_share"])
            if s == "gaitrec":
                clock.append(gaitrec_clock_row(row, rec, fr))
            n += 1
        stats[f"{s}|force"] = {"records": n, "seconds": round(time.time() - t0, 1)}
        log(f"{s} force: {n} records in {time.time() - t0:.0f} s")
    F = add_folds(pd.DataFrame(feet), read_folds())
    log(f"proposed folds: {int((F['fold'] == 'dev').sum())} dev feet, {int((F['fold'] == 'train').sum())} train feet")
    w(F[F["source"] != "gaitrec"], "force_zero_feet.csv")      # gaitrec's per-foot zeros: gaitrec_clock.csv
    w(force_summary(F, durs, recfacts), "force_zero_summary.csv")
    G = pd.DataFrame(clock)
    w(G, "gaitrec_clock.csv")
    gv = clock_verdict(G)

    # item 4
    pause_if_slow("gaitrec", "pads", pauses)
    t0, pend, pairs, n = time.time(), {}, [], 0
    gate = RawGate(idx)
    for row, rec in read_unit("pads", "imu", a.limit):
        m = re.fullmatch(r"(.+)_(wrist_[LR])", str(row["key"]))
        assert m and m.group(1) == rec["task"], row["key"]
        k = (row["subject_ns"], m.group(1))
        pend.setdefault(k, {"person_id": row["person_id"], "recs": {}})["recs"][m.group(2)] = rec
        if len(pend[k]["recs"]) == 2:
            p = pend.pop(k)
            pairs.append(pads_pair(gate, k[0], p["person_id"], k[1], p["recs"]))
        n += 1
    unpaired = len(pend)
    pend.clear()
    stats["pads|imu"] = {"records": n, "pairs": len(pairs), "unpaired": unpaired,
                         "seconds": round(time.time() - t0, 1)}
    log(f"pads: {n} records, {len(pairs)} pairs, {unpaired} unpaired in {time.time() - t0:.0f} s")
    PP = pd.DataFrame(pairs)
    w(PP, "pads_pairs.csv")
    pv = pads_verdict(PP)
    pv["unpaired_records"] = unpaired

    # item 5
    pause_if_slow("pads", "imu_inthewild_iprognosis", pauses)
    t0 = time.time()
    IP = pd.DataFrame([ipr_row(row, rec) for row, rec in read_unit("imu_inthewild_iprognosis", "imu", a.limit)])
    stats["imu_inthewild_iprognosis|imu"] = {"records": len(IP), "seconds": round(time.time() - t0, 1)}
    w(IP, "iprognosis_records.csv")
    iv = {"records": len(IP), "persons": int(IP["person_id"].nunique()),
          "records_with_time_stamps": int(IP["has_time_stamps"].sum()),
          "arrays": sorted({x for v in IP["arrays"] for x in v.split("|")}),
          "fps_min_median_max": qs(IP["fps"].astype(float), (0, 50, 100)),
          "seconds_at_fps_p10_p50_p90": qs(IP["seconds_at_fps"].astype(float)),
          "seconds_at_fps_min_max": qs(IP["seconds_at_fps"].astype(float), (0, 100)),
          "sample_valid_share_p10_p50_p90": qs(IP["sample_valid_share"].astype(float)),
          "repeat_share_p10_p50_p90": qs(IP["repeat_share"].astype(float)),
          "interval_spread": "not measurable from the canonical records (no time stamps); the raw pickle holds "
                             "them and holds lockbox subjects, so it is not opened"}
    log(f"iprognosis: {len(IP)} records in {time.time() - t0:.0f} s")

    pd.DataFrame(pauses, columns=["started", "ended", "seconds", "after_source", "before_source"]).to_csv(
        os.path.join(a.out, "pauses.csv"), index=False, lineterminator="\n")
    st1 = stamp(paths.MOVEATLAS)
    summary = {"limit": a.limit, "moveatlas_head": st0["head"], "moveatlas_diff_sha256": st0["diff_sha256"],
               "moveatlas_unchanged_during_run": st0["head"] == st1["head"] and st0["diff_sha256"] == st1["diff_sha256"],
               "manifest_note": st0.get("note", ""), "lockbox_refusals": refusals,
               "units": {u: {"rows": int((rows["unit"] == u).sum()), "persons": int(rows.loc[rows["unit"] == u, "person_id"].nunique())}
                         for u in sorted(set(rows["unit"]))},
               "rows_lockbox": int(rows["lockbox"].sum()), "rows_not_release": int((rows["tier"] != "RELEASE").sum()),
               "swing_max_s": SWING_MAX_S, "swing_min_s": SWING_MIN_S, "near_zero": NEAR_ZERO,
               "low_mode_bins": LOW_MODE_BINS, "gaitrec_clock": gv, "pads_pairing": pv,
               "iprognosis": iv, "reader": stats, "pauses": len(pauses), "runtime_s": round(time.time() - t_start, 1)}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, default=str)
    prov = {"script": "scripts/probe_stage2_items.py", "task": "S2.0 (docs/MFM_STAGE2_CODE_SPEC_DRAFT.md section 7)",
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": st0["head"], "moveatlas_stamp": st0,
            "moveatlas_stamp_end": st1, "run_at": dt.datetime.now().isoformat(timespec="seconds"),
            "folds_proposed_sha256": sha256_of(FOLDS),
            "python": sys.version.split()[0], "limit": a.limit,
            "reader": "mdfm.data.canonical.iterate(sources=<source>, modality=<m>, tier='RELEASE', lane='body'), "
                      "every flag False; canonical._select for the index rows",
            "inputs": ["cache/manifest/manifest_records.csv.gz", "cache/manifest/record_index.csv.gz",
                       "reports/stage2_sources/folds_proposed.csv (the fold column of force_zero_feet.csv)",
                       "MoveAtlas canonical records of the units (mdfm.data.canonical)",
                       f"{PADS_RAW}/<id>_<task>_<LeftWrist|RightWrist>.txt (Time column; non-lockbox RELEASE persons, "
                       "checked by raw_allowed before each open)"],
            "not_opened": ["datasets/imu/imu_inthewild_iprognosis/raw/i-PROGNOSIS_DS2.10_Sub1_IMUdataPhonecalls.zip "
                           "(tremor_dataset.pickle holds lockbox subjects)"],
            "outputs": OUTPUTS}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1, default=str)
    write_md(a.out)
    log(f"done in {summary['runtime_s']} s")


# ---------------------------------------------------------------- summary.md
def md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else str(int(v)) if v.is_integer() else f"{v:g}"
        return str(v).replace("|", "\\|")

    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return lines


def write_md(out):
    rd = lambda name: pd.read_csv(os.path.join(out, name), encoding="utf-8", keep_default_na=False)  # noqa: E731
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    keys, fz = rd("recording_keys.csv"), rd("force_zero_summary.csv")
    gv, pv, iv = s["gaitrec_clock"], s["pads_pairing"], s["iprognosis"]
    ref = s["lockbox_refusals"]
    L = ["# Stage-2 items (task S2.0)", "",
         "Generated by scripts/probe_stage2_items.py; every number below is read from summary.json and the CSVs of "
         f"this directory. MoveAtlas HEAD {s['moveatlas_head']}, working-tree diff SHA-256 {s['moveatlas_diff_sha256']} "
         f"(unchanged during the run: {s['moveatlas_unchanged_during_run']}). Definitions and rules in the script "
         "docstring." + (f" DRY RUN (--limit {s['limit']})." if s["limit"] else ""), "",
         "## Done-when checks", "", "| check | measured |", "|---|---|",
         f"| rows non-lockbox RELEASE | {sum(v['rows'] for v in s['units'].values())} index rows of "
         f"{len(s['units'])} units; lockbox rows {s['rows_lockbox']}, non-RELEASE rows {s['rows_not_release']} |",
         "| canonical.load refuses a lockbox row | " + ", ".join(
             f"{k} {v['lockbox_rows_refused']} of {v['lockbox_rows_tried']}" for k, v in ref.items()) + " |",
         f"| recording keys per unit | {len(keys)} pattern rows in recording_keys.csv over "
         f"{keys['unit'].nunique()} units |",
         f"| force zero runs per record and foot | {int(pd.to_numeric(fz['feet']).sum())} feet of "
         f"{int(pd.to_numeric(fz['records']).sum())} records, summed in force_zero_summary.csv; per foot in "
         f"force_zero_feet.csv (column fold: {int(pd.to_numeric(fz['feet_dev']).sum())} feet of the proposed dev "
         "fold), gaitrec's in gaitrec_clock.csv |",
         f"| gaitrec clock | {gv['verdict']} ({gv['two_feet_records']} two-foot records) |",
         f"| pads wrist pairing | {pv['verdict']} ({pv['pairs']} pairs) |",
         f"| iprognosis time stamps | {iv['records_with_time_stamps']} of {iv['records']} records carry one |",
         "| MoveAtlas HEAD in _provenance.json | moveatlas_head and the stamps at start and end |", "",
         "## Findings", ""]
    L.append(f"- gaitrec: {gv['verdict']}. Of {gv['two_feet_records']} two-foot records, {gv['same_first_loaded']} "
             f"({gv['same_first_loaded_share']}) start both feet at the same sample; the absolute difference of the two "
             f"first loaded samples p50/p90/max {gv['first_loaded_diff_samples_p50_p90_max']} samples; the total at each foot's first "
             f"loaded sample p10/p50/p90 {gv['total_at_first_p10_p50_p90']} N; median both_loaded_share "
             f"{gv['both_loaded_share_median']}. Each foot's stance is cut at its own load threshold and both start "
             "at sample 0: the record holds two stances, not one clock (rule: " + gv["rule"] + ").")
    per_n = ", ".join(f"{v} pairs of {k} samples" for k, v in pv.get("pairs_per_n_L", {}).items())
    per_n_e = ", ".join(f"{v} ms at {k} samples" for k, v in pv.get("end_diff_ms_p50_per_n_L", {}).items())
    L.append(f"- pads: {pv['verdict']}. {pv['same_length']} of {pv['pairs']} pairs have equal lengths ({per_n}): the "
             f"deposit holds fixed lengths, so the equal length says nothing of the clock; raw time on "
             f"{pv['pairs_with_raw_time']} pairs (raw rows equal the canonical n on {pv.get('raw_rows_equal_canonical')}; "
             f"both time columns start at 0 on {pv.get('raw_t0_zero_pairs')}, each on its own watch, so the start "
             f"offset of the two watches is not measured); end time right minus left p10/p50/p90 "
             f"{pv.get('end_diff_ms_p10_p50_p90')} ms (min/max {pv.get('end_diff_ms_min_max')}), p50 per length "
             f"{per_n_e}; {pv['end_diff_within_one_interval']} pairs within one 10 ms interval; the sign of the "
             f"end-time difference is the same over all tasks for {pv.get('persons_end_diff_one_sign')} of "
             f"{pv.get('persons_with_raw_time')} persons; drift per second of record p50 {pv.get('end_diff_ms_per_s_p50')} "
             f"ms; median interval of the left watch p10/p50/p90 {pv.get('dt_median_ms_L_p10_p50_p90')} ms and of the "
             f"right {pv.get('dt_median_ms_R_p10_p50_p90')} ms; intervals p1 and p99 (medians over pairs, L and R) "
             f"{pv.get('dt_p1_ms_p50_LR')} and {pv.get('dt_p99_ms_p50_LR')} ms. Index lag at the best acc-norm "
             f"correlation (>= {XC_MIN_CORR}), not clock evidence ({pv['lag_reading']}): first {XC_WIN} samples "
             f"{pv['lag_first_pairs_corr_ok']} pairs, p10/p50/p90 {pv['lag_first_p10_p50_p90']}, within 1 sample "
             f"{pv['lag_first_within_1']}; last {XC_WIN} samples {pv['lag_last_pairs_corr_ok']} pairs, p10/p50/p90 "
             f"{pv['lag_last_p10_p50_p90']}, within 1 sample {pv['lag_last_within_1']}. Unpaired records "
             f"{pv.get('unpaired_records')}.")
    L.append(f"- imu_inthewild_iprognosis: {iv['records_with_time_stamps']} of {iv['records']} records ({iv['persons']} "
             f"persons) carry a time stamp; arrays {', '.join(iv['arrays'])}; fps min/median/max "
             f"{iv['fps_min_median_max']}; n / fps p10/p50/p90 {iv['seconds_at_fps_p10_p50_p90']} s (min/max "
             f"{iv['seconds_at_fps_min_max']}); sample_valid share p10/p50/p90 {iv['sample_valid_share_p10_p50_p90']}; "
             f"share of consecutive samples repeated on all three axes p10/p50/p90 {iv['repeat_share_p10_p50_p90']}. "
             f"Interval spread: {iv['interval_spread']}.")
    def durs_txt(r, c):
        return (f"{c} {r[f'stretches_{c}']} (s per stretch p10/p50/p90 {r[f'dur_s_{c}_per_stretch_p10_p50_p90'] or '-'}, "
                f"per zero {r[f'dur_s_{c}_per_zero_p10_p50_p90'] or '-'})")

    for r in fz.to_dict("records"):
        caveat = (f" CLASS COLUMNS NOT INTERPRETABLE: the floor is below 0 on {r['feet_floor_below_0']} of {r['feet']} "
                  f"feet, the unloaded level is a noisy band (low_mode p10/p50/p90 {r['low_mode_p10_p50_p90']}, the "
                  "baseline an offset would subtract) and the threshold floor + "
                  f"{NEAR_ZERO} x (p99 - floor) cuts inside it; the zero shares hold." if int(r["feet_floor_below_0"])
                  else "")
        L.append(f"- {r['source']} {r['group']} force: {r['records']} records, {r['feet']} feet "
                 f"({r['feet_other_foot_unseen']} without the other foot, {r['feet_dev']} of the proposed dev fold); "
                 f"zero share {r['zero_share']}, negative share "
                 f"{r['negative_share']}, floor p10/p50/p90 {r['floor_p10_p50_p90']}, p99 p10/p50/p90 "
                 f"{r['p99_p10_p50_p90']}; zero samples inside a step {r['zero_inside_step_share']}, off the plate "
                 f"{r['zero_off_plate_share']}, while loaded (the floor below 0) {r['zero_while_loaded_share']} "
                 f"(zeros per class: swing {r['zero_swing']}, swing_other_unseen {r['zero_swing_other_unseen']}, "
                 f"short_gap {r['zero_short_gap']}, record_edge {r['zero_record_edge']}, both_unloaded "
                 f"{r['zero_both_unloaded']}, long {r['zero_long']}); unloaded share {r['unloaded_share']}, of it inside "
                 f"a step {r['unloaded_inside_step_share']}; stretches {durs_txt(r, 'swing')}, "
                 f"{durs_txt(r, 'swing_other_unseen')}, {durs_txt(r, 'short_gap')}, both_unloaded "
                 f"{r['stretches_both_unloaded']}, long {r['stretches_long']}, record_edge {r['stretches_record_edge']}; "
                 f"both_loaded_share p10/p50/p90 {r['both_loaded_share_p10_p50_p90']}"
                 + (f"; zero samples marked invalid by sample_valid {int(float(r['zero_samples_marked_invalid']))}"
                    if str(r.get('zero_samples_marked_invalid', '')) != '' else "") + "." + caveat)
    i0 = L.index("## Findings") + 2
    L[i0:] = [x.replace("|", "/") for x in L[i0:]]           # percentile triples a|b|c read as a/b/c in prose
    L += ["", "## Recording keys per unit", "", "Pattern: every run of digits written '#'. Index rows only.", ""]
    L += md_table(keys)
    L += ["", "## Force zero runs per source and group", "",
          f"Classes and rule in the script docstring (swing from SWING_MIN_S {s['swing_min_s']} s to SWING_MAX_S "
          f"{s['swing_max_s']} s; a sample is loaded above floor + {s['near_zero']} x (p99 - floor), floor = min(0, "
          "p1), per foot). Durations in s, per stretch and per zero sample (each stretch weighted by the zero "
          "samples it holds). The class columns of a group with feet_floor_below_0 above 0 are not "
          "interpretable (findings above). fold: the proposed stage-2 fold of the person "
          "(reports/stage2_sources/folds_proposed.csv); no rule may be tuned on these tables without leaving out "
          "the dev rows.", ""]
    L += md_table(fz)
    kv = lambda d: [f"| {k} | {str(v).replace('|', chr(92) + '|')} |" for k, v in d.items()]  # noqa: E731
    L += ["", "## gaitrec clock", "", "| key | value |", "|---|---|"] + kv(gv)
    L += ["", "## pads wrist pairing", "", "| key | value |", "|---|---|"] + kv(pv)
    L += ["", "## imu_inthewild_iprognosis", "", "| key | value |", "|---|---|"] + kv(iv)
    L += ["", "## Reader", "", "; ".join(f"{k} {v}" for k, v in s["reader"].items()) + ".", "",
          f"Pauses: {s['pauses']}. Runtime: {s['runtime_s']} s.", ""]
    text = "\n".join(L)
    assert "\u2014" not in text and "\u2013" not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
