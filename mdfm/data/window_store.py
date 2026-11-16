"""Window store: band copies of the real stage-1 records as float16 memmaps (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.2 and task B4.6).

Band copy (band_plan, band_copy): the observation on the band the body carries [D§3]. One function
serves the store, factory/synth.py (every synthetic stream) and the G1 input builder, so that real,
synthetic and gate inputs share one sample grid and one band.
- imu, force: 4th-order Butterworth low-pass at 40 Hz run forward and backward (zero phase, scipy
  sosfiltfilt) when fps > 80 Hz; unfiltered at 80 Hz and below.
- emg: 4th-order Butterworth band-pass 20 to 450 Hz (zero phase), rectified moving RMS over a centred
  window of 2 round(0.01 fps) + 1 samples (20 ms; the window shrinks at a run edge), natural log with a
  per-channel floor at the 5th percentile of the stream's valid envelope samples (at least 1e-6 mV).
  Values in log mV, meta signal_kind 'log_envelope'. A rate at or below 900 Hz is refused, and so is a
  stream whose meta signal_kind is set and not 'raw' (a band copy is never taken twice; a stream without
  the key is read as raw). A run of at least rms_n samples (20 ms) where a raw channel reads exactly 0 is
  zero-fill, not signal (the stroke_gait_vancriekinge C3D trials pad their EMG with 0 before or after the
  acquisition, on every channel at once): those samples are gaps (zero_runs).
- pose2d, pose3d: as recorded.
Then every k-th native sample from the stream's first sample, k = max(1, floor(fps / 100)) (k = 1 for
pose), so kept timestamps are native; the output Stream has fps = fps / k. Filters run per group
(IMU site, foot, EMG channel) on each contiguous run of valid samples (sample_valid, the weargait
insole gaps of mdfm.data.sources, NaN, EMG zero-fill; a non-finite entry in a kept channel makes its group
sample invalid); a run of at most padlen samples, padlen = max(3 (2 sections + 1), ceil(fps / lowest
corner)), is masked. Invalid samples keep valid False and value 0, so a gap never spreads past its own
edges. kernels/emg.log_envelope repeats the EMG steps on one run; a test holds the two equal.

Store (build_source, build, WindowStore): each admitted recording of a pilot source goes through
sources.paired (canonical.load with its flags at False: no lockbox, EVAL, CODY or EXCLUDED_V1 row is
read) and windows.cut over the whole recording (the field whitelist, the span [t0, t0 + duration) of the
input and the covariates of a window; a recording under 1 s is dropped and counted). band_copy then runs
on every native sample of each stream's record, not on the cut copy, so that a target reaching past the
input span (the gait120 EMG trial around its pose steps) is filtered on its own contiguous valid run
without edge transients at the span; the kept samples inside the span are stored, with half a native
period of tolerance at both edges (a sample that float rounding puts just inside the end edge is left
out). The decimation phase and the EMG floor are those of the whole record. Layout under the root:
  <source>/<stream>/values.f16   float16 (n, G, C) blocks of every record, concatenated
  <source>/<stream>/valid.u1     uint8 (n, G) blocks
  <source>/pose3d/root.f32       float32 (n, 3) root-track blocks, m y up (only when a record carries one)
  <source>/pose3d/root_valid.u1  uint8 (n,) blocks
  <source>/records.csv           one row per stream of a recording (RECORD_COLUMNS)
  <source>/source.json           band configuration, counts, person-list SHA-256, file SHA-256
  store.json                     the per-source hashes (build())
Names (task S2.3): build_source and build take specs, a source registry {name: SourceSpec} that replaces
sources.SOURCES (sources.py, Registry), and vocabulary, the task vocabulary file of the windows' task covariate
(windows.cut and windows.validate; a relative path is read under the tree), whose SHA-256, version and task
count source.json records (task_vocabulary) and store.json repeats (task_vocabulary_sha256); both None build
the stage-1 store as before, with no record (no record reads as the version-1 file). WindowStore refuses a source
built under another vocabulary than the reader's (vocabulary_sha256, default the version-1 file's).
Root track (plan task B4.13b): the raw pelvis track of mdfm.data.mocap_raw (gait120 PELV_C of the TRC,
stroke_gait_vancriekinge PELO of the C3D; windows.attach_root, meta['root'] and meta['root_valid'] of the
pose3d Stream) is attached by build_source to every recording it reads through sources.paired
(attach_raw_root, reader flags at False) and stored in float32 (the lab-frame pelvis lies metres from the
origin, where float16 steps by 2^-9 m, 2 mm, from 2 to 4 m): root.f32 and root_valid.u1 run parallel to
the pose3d shard, sample by sample, so the root block of a pose3d row starts at sample valid_at / G and
has its n samples; a pose3d record without a track holds zero rows with root_valid 0. Nothing else
changes: records.csv and the other shards are those of a build without roots. The reader returns
meta['root'] and meta['root_valid'] for a pose3d record whose root block has a valid sample, and raises
when the root files on disk are not those source.json lists (files_sha256). A recording
whose track cannot be read (LookupError, ValueError) is stored without it and listed in source.json
(root.missing).
Record times: the kept sample j of a record sits at offset + j k / fps_native s on the recording clock
(origin: the first input sample, mdfm.data.sources). float16 keeps 11 significant bits: the round trip
is within 1 mg below 4 g and within 0.1 deg/s below 256 deg/s, within 2^-11 of the value above (the
build reports the share of entries beyond 1 mg and 0.1 deg/s); a value that overflows marks its group
sample invalid (counted). The root is MDFM_WINDOW_STORE, else cache/window_store of this tree; a root
inside MoveAtlas, inside the gates-only store (mdfm.paths.GATES_STORE, <store>/store/lockbox or
cache/gates_only) or inside this tree outside cache/ is refused.
Store roots (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.4, task S2.5): three roots, mdfm.paths
.window_store_root. The stage-1 root (MDFM_WINDOW_STORE, else cache/window_store) is never rebuilt for stage 2 and
carries no marker; it reads as RELEASE. RELEASE stage-2 windows go to MDFM_WINDOW_STORE_S2, else
cache/window_store_s2; INTERNAL windows to MDFM_WINDOW_STORE_INTERNAL, else cache/window_store_internal. A stage-2
build (build_source and build with tier and names) refuses overlapping roots, the stage-1 root, the other tier's
root, a root marked with the other tier and an unmarked root holding sources; it refuses a person with an index row
of another tier (an INTERNAL row in a RELEASE root and the reverse), writes the root's MARKER (store_tier.json:
format and tier) and records the tier, the name tables' SHA-256 and the vocabulary in source.json; store.json, the
manifest, adds the tier and the root's name-table and vocabulary SHA-256. WindowStore refuses a root of another
tier than the run's (its marker), a root that is or lies in a root of the other tier by path (other_tier_roots), a
source whose source.json records another tier (none reads RELEASE) and, given the run's names, a source built under
other name tables; store.json refuses a source of another tier than its root's.
"""
import collections
import csv
import functools
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass

import numpy as np

from mdfm import paths
from mdfm.data import canonical
from mdfm.data import windows as W

FORMAT = "mdfm-window-store/1"
ENV = "MDFM_WINDOW_STORE"
TARGET_HZ = 100.0                 # k = max(1, floor(fps / TARGET_HZ))
FILTER_MIN_FPS = 80.0             # imu and force are low-passed above this rate only
LOWPASS_HZ = 40.0
ORDER = 4
EMG_BAND_HZ = (20.0, 450.0)
EMG_RMS_S = 0.020
EMG_FLOOR_PCT = 5.0
EMG_FLOOR_MIN = 1e-6              # mV
MIN_S = 1.0                       # windows.cut min_s: a recording under 1 s is dropped
WHOLE_S = 1e9                     # win_s and stride_s of the whole-recording cut
GRID_TOL_S = 1e-6                 # native times must sit on t[0] + i / fps within this
BAND = {"target_hz": TARGET_HZ, "filter_min_fps": FILTER_MIN_FPS, "lowpass_hz": LOWPASS_HZ, "order": ORDER,
        "emg_band_hz": list(EMG_BAND_HZ), "emg_rms_s": EMG_RMS_S, "emg_floor_pct": EMG_FLOOR_PCT,
        "emg_floor_min_mv": EMG_FLOOR_MIN, "emg_zero_run": "rms_n samples of exact 0 in a raw channel are a gap",
        "zero_phase": "scipy.signal.sosfiltfilt"}
RECORD_COLUMNS = ("source", "stream", "role", "person_id", "subject_ns", "key", "fold", "sex", "age_band", "task",
                  "span_t0", "span_s", "fps_native", "k", "fps", "offset", "n", "G", "C", "values_at", "valid_at",
                  "n_valid", "masked_short", "f16_overflow", "flags", "chan_mask", "meta")
_INT = ("sex", "age_band", "task", "k", "n", "G", "C", "values_at", "valid_at", "n_valid", "masked_short",
        "f16_overflow")
_FLOAT = ("span_t0", "span_s", "fps_native", "fps", "offset")
LOCKBOX_CSV = os.path.join(paths.TREE, "reports", "lockbox", "lockbox_persons.csv")
LOCKBOX_SHA_FILE = os.path.join(paths.TREE, "reports", "lockbox", "lockbox.sha256")
# float16 round-trip bounds reported by the build (acc g, gyro deg/s)
F16_BOUNDS = {"acc": 1e-3, "gyro": 0.1}
# root-track files of a stream folder, parallel to its shard sample by sample (module docstring)
ROOT_FILES = (("root.f32", "<f4"), ("root_valid.u1", np.uint8))
TIERS = ("RELEASE", "INTERNAL")   # the tiers of train/provenance.TIERS (a store serves a run of one tier)
# store roots (task S2.5): the marker file of a stage-2 root and the root kind of each tier (mdfm.paths)
MARKER = "store_tier.json"
MARKER_FORMAT = "mdfm-window-store-tier/1"
TIER_KIND = {"RELEASE": "s2", "INTERNAL": "internal"}


# ---------------------------------------------------------------- roots and refusals
def store_root(environ=None, kind="stage1"):
    """Root of the window store of `kind` (mdfm.paths.WINDOW_STORE_KINDS). stage1 (the default): MDFM_WINDOW_STORE
    of `environ` (default os.environ), not created here; else mdfm.paths.cache_dir('window_store') of this tree
    (gitignored, created). s2 and internal: mdfm.paths.window_store_root (MDFM_WINDOW_STORE_S2, else
    cache/window_store_s2; MDFM_WINDOW_STORE_INTERNAL, else cache/window_store_internal), nothing created.
    Forward slashes, no trailing slash."""
    if kind != "stage1":
        return paths.window_store_root(kind, environ)
    env = os.environ if environ is None else environ
    root = env.get(ENV) or paths.cache_dir("window_store")
    return root.replace("\\", "/").rstrip("/")


def assert_store_root(root):
    """Refuse (canonical.Refused) a store root inside MoveAtlas, the gates-only store (its current root,
    <store>/store/lockbox and cache/gates_only whatever MDFM_GATES_STORE says) or this tree outside cache/;
    return the root with forward slashes otherwise."""
    root = str(root).replace("\\", "/").rstrip("/")
    try:
        paths.assert_not_moveatlas(root)
    except PermissionError as e:
        raise canonical.Refused(str(e)) from None
    gated = (paths.GATES_STORE, paths.GATES_STORE_TARGET, os.path.join(paths.CACHE, "gates_only"))
    if paths.in_gates_store(root) or any(paths._inside(root, g) for g in gated):
        raise canonical.Refused(f"the window store may not sit inside the gates-only (lockbox) store: {root}")
    if paths._inside(root, paths.TREE) and not paths._inside(root, paths.CACHE):
        raise canonical.Refused(f"the window store may sit in this tree only under cache/ (gitignored): {root}")
    return root


# ---------------------------------------------------------------- stage-2 roots and their tier (task S2.5)
def _nested(a, b):
    """True when one of two paths lies inside the other (or they are equal)."""
    return paths._inside(a, b) or paths._inside(b, a)


def read_marker(root):
    """The marker of a store root ({'format', 'tier'}), None when the root holds no MARKER file; ValueError for a
    marker of another format or tier."""
    p = os.path.join(root, MARKER)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("format") != MARKER_FORMAT or doc.get("tier") not in TIERS:
        raise ValueError(f"{p}: not a {MARKER_FORMAT} marker with a tier of {TIERS}")
    return doc


def root_tier(root):
    """Tier of a store root: its marker's; RELEASE for a root without one (the stage-1 root, whose rows are RELEASE
    by selection: sources.rows reads canonical.iterate(tier='RELEASE'))."""
    m = read_marker(root)
    return "RELEASE" if m is None else m["tier"]


def write_marker(root, tier):
    """Write the MARKER of a stage-2 root (format and tier) when it holds none; ValueError for another tier."""
    m = read_marker(root)
    if m is not None:
        if m["tier"] != tier:
            raise ValueError(f"{root}: marked {m['tier']}, not {tier}")
        return m
    doc = {"format": MARKER_FORMAT, "tier": tier}
    with open(os.path.join(root, MARKER), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    return doc


def assert_roots_apart(environ=None):
    """Refuse (canonical.Refused) when two of the three window store roots (stage1, s2, internal, mdfm.paths) are
    equal or nested; return {kind: root} otherwise."""
    roots = {k: (paths.window_store_root(k, environ)) for k in paths.WINDOW_STORE_KINDS}
    kinds = list(roots)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            if _nested(roots[a], roots[b]):
                raise canonical.Refused(f"the {a} and {b} window store roots overlap: {roots[a]}, {roots[b]}")
    return roots


def other_tier_roots(tier, environ=None):
    """{kind: root} of the window store roots (mdfm.paths.window_store_root of `environ`) whose kind serves the other
    tier (mdfm.paths.WINDOW_STORE_TIER): the internal root for a RELEASE run, the stage1 and s2 roots for an INTERNAL
    run. A reader refuses them by path, whatever their marker says."""
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
    return {k: paths.window_store_root(k, environ) for k in paths.WINDOW_STORE_KINDS
            if paths.WINDOW_STORE_TIER[k] != tier}


def source_tier(doc):
    """Tier of a source from its source.json content: the recorded one, RELEASE when none is recorded (a stage-1
    build, whose rows are RELEASE by selection)."""
    return doc.get("tier") or "RELEASE"


def assert_stage2_root(root, tier, environ=None):
    """Refuse (canonical.Refused) a root that a build of `tier` may not write: a root assert_store_root refuses; the
    roots overlapping (assert_roots_apart); the stage-1 root (MDFM_WINDOW_STORE of `environ` and cache/window_store
    whatever it says), never rebuilt for stage 2; the root of the other tier; a root marked with the other tier; an
    unmarked root that already holds a source (a stage-1 kind of store). Returns the root (forward slashes)."""
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
    root = assert_store_root(root)
    roots = assert_roots_apart(environ)
    for s1 in (roots["stage1"], os.path.join(paths.CACHE, paths.WINDOW_STORE_DEFAULT["stage1"])):
        if _nested(root, s1):
            raise canonical.Refused(f"the stage-1 window store root is never rebuilt for stage 2: {root} ({s1})")
    other = [t for t in TIERS if t != tier][0]
    if _nested(root, roots[TIER_KIND[other]]):
        raise canonical.Refused(f"a {tier} build into the {other} window store root: {root}")
    m = read_marker(root)
    if m is not None and m["tier"] != tier:
        raise canonical.Refused(f"{root} is a store root of tier {m['tier']}, refused for a build of tier {tier}")
    if m is None and _source_dirs(root):
        raise canonical.Refused(f"{root} holds sources and no {MARKER}: not a stage-2 root, refused for a build of "
                                f"tier {tier}")
    return root


def person_tiers(index=None):
    """{person_id: frozenset of the tiers of its rows} of a canonical.index() frame (default: read it); the
    index's person_id, tier columns only."""
    df = canonical.index() if index is None else index
    out = collections.defaultdict(set)
    for pid, tier in zip(df["person_id"].astype(str), df["tier"].astype(str)):
        out[pid].add(tier)
    return {k: frozenset(v) for k, v in out.items()}


def refuse_tier(source, person_id, tier, tiers):
    """canonical.Refused unless every index row of the person (tiers: person_tiers) has the root's tier: an INTERNAL
    row in a RELEASE root and a RELEASE row in an INTERNAL root are refused, and so are a row of any other tier and
    a person absent from the index."""
    got = tiers.get(person_id)
    if not got:
        raise canonical.Refused(f"{source} {person_id}: not in the record index, refused in a store root of tier "
                                f"{tier}")
    other = sorted(got - {tier})
    if other:
        raise canonical.Refused(f"{source} {person_id}: a row of tier {', '.join(other)} in a store root of tier "
                                f"{tier}")


def _stage2_names(source, names, vocabulary):
    """(vocabulary path, names record) of a stage-2 build: the source must be a name of names.sources, the
    vocabulary defaults to names.vocabulary_path, and a given one must have its SHA-256."""
    if source not in names.sources:
        raise KeyError(f"{source} is not a source of the run's name tables ({list(names.sources)})")
    want = names.vocabulary_sha256()
    vocab = names.vocabulary_path if vocabulary is None else vocabulary
    got = W.task_vocabulary(vocabulary_file(vocab))["sha256"]
    if got != want:
        raise ValueError(f"vocabulary {vocab} SHA-256 {got} differs from the name tables' {want}")
    return vocab, {"sha256": names.sha256(), "tables": names.to_dict()}


@functools.lru_cache(maxsize=None)
def lockbox_persons():
    """Person ids of every source in reports/lockbox/lockbox_persons.csv, checked against lockbox.sha256."""
    with open(LOCKBOX_SHA_FILE, encoding="utf-8") as f:
        want = f.read().split()[0]
    got = sha256_file(LOCKBOX_CSV)
    if got != want:
        raise RuntimeError(f"{LOCKBOX_CSV} SHA-256 {got} differs from {LOCKBOX_SHA_FILE} ({want})")
    with open(LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        return frozenset(r["person_id"] for r in csv.DictReader(f))


def _refuse_names(source, subject_ns, person_id):
    """Refused for a CODY name or a lockbox person (the names alone, no index read)."""
    if canonical.is_cody(source, subject_ns, person_id):
        raise canonical.Refused(f"CODY is evaluation-only and never enters the window store: {source} {person_id}")
    if person_id in lockbox_persons():
        raise canonical.Refused(f"lockbox person in the window store: {source} {person_id}")


def attach_raw_root(pr):
    """The PairedRecord with the raw root track of mdfm.data.mocap_raw.root_for_paired on its pose3d stream
    (windows.attach_root; reader flags at False) for the sources mocap_raw reads, unchanged for the others. A
    CODY name or a lockbox person is refused before any index or zip read; mocap_raw applies canonical.check
    to every index row of the recording before it opens a zip."""
    from mdfm.data import mocap_raw
    _refuse_names(pr.source, pr.subject_ns, pr.person_id)
    if pr.source not in mocap_raw.SOURCES or "pose3d" not in pr.streams:
        return pr
    root, valid, _ = mocap_raw.root_for_paired(pr)
    return W.attach_root(pr, root, valid)


def _has_root(pr):
    """True when the PairedRecord's pose3d meta already carries a root track key (windows.ROOT_KEYS)."""
    m = ((getattr(pr, "meta", None) or {}).get("pose3d")) or {}
    return any(k in m for k in W.ROOT_KEYS)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- task vocabulary of a build (task S2.3)
def vocabulary_file(vocabulary):
    """Path of a task vocabulary file: windows.VOCABULARY for None, else train/names.resolve (an absolute path as
    given, a relative one under the tree, whatever the working directory)."""
    if vocabulary is None:
        return W.VOCABULARY
    from mdfm.train import names as NM
    return NM.resolve(vocabulary)


def vocabulary_record(path):
    """The task_vocabulary entry of source.json: SHA-256 of the file bytes, version and task count
    (windows.task_vocabulary)."""
    v = W.task_vocabulary(path)
    return {"sha256": v["sha256"], "version": v["version"], "tasks": len(v["tasks"])}


def source_vocabulary_sha256(doc):
    """SHA-256 of the task vocabulary a source was built under, from its source.json content: the recorded one, or
    the version-1 file's when none is recorded (a build without a vocabulary)."""
    rec = doc.get("task_vocabulary")
    return str(rec["sha256"]) if rec else W.task_vocabulary()["sha256"]


def person_hash(ids):
    """SHA-256 of the sorted unique ids joined by newlines (mdfm.data.sources.person_hash)."""
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- band copy
@dataclass(frozen=True, eq=False)
class BandPlan:
    """How band_copy treats a stream (module docstring): kind 'lowpass', 'unfiltered', 'emg_envelope' or
    'as_recorded'; k, the decimation factor; sos, the filter sections (None when unfiltered); padlen, the
    sosfiltfilt padding in samples (a valid run of at most padlen samples is masked); rms_n, the samples
    of the EMG moving RMS window."""
    modality: str
    fps: float
    kind: str
    k: int
    sos: np.ndarray = None
    padlen: int = 0
    rms_n: int = 0

    @property
    def fps_out(self):
        """Rate of the kept samples (Hz)."""
        return self.fps / self.k


def _padlen(sos, fps, corner):
    return max(3 * (2 * len(sos) + 1), int(math.ceil(fps / corner)))


@functools.lru_cache(maxsize=128)
def band_plan(modality, fps):
    """BandPlan of a stream of `modality` sampled at `fps` Hz (module docstring)."""
    from scipy import signal
    fps = float(fps)
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"band_plan needs a positive fps, got {fps}")
    if modality not in W.MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    if modality in ("pose2d", "pose3d"):
        return BandPlan(modality, fps, "as_recorded", 1)
    k = max(1, int(math.floor(fps / TARGET_HZ)))
    if modality == "emg":
        lo, hi = EMG_BAND_HZ
        if fps <= 2 * hi:
            raise ValueError(f"EMG at {fps:g} Hz cannot carry the {lo:g} to {hi:g} Hz band")
        sos = signal.butter(ORDER, [lo, hi], btype="bandpass", fs=fps, output="sos")
        rms_n = 2 * int(round(EMG_RMS_S * fps / 2)) + 1
        return BandPlan(modality, fps, "emg_envelope", k, sos, _padlen(sos, fps, lo), rms_n)
    if fps <= FILTER_MIN_FPS:
        return BandPlan(modality, fps, "unfiltered", k)
    sos = signal.butter(ORDER, LOWPASS_HZ, btype="lowpass", fs=fps, output="sos")
    return BandPlan(modality, fps, "lowpass", k, sos, _padlen(sos, fps, LOWPASS_HZ))


def runs(mask):
    """(start, stop) index pairs of the contiguous True runs of a (T,) bool mask."""
    m = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _patterns(valid):
    """Groups sharing one validity column, with the runs of that column."""
    by = collections.OrderedDict()
    for g in range(valid.shape[1]):
        by.setdefault(valid[:, g].tobytes(), []).append(g)
    for gs in by.values():
        yield np.asarray(gs), runs(valid[:, gs[0]])


def zero_runs(values, chan_mask, n):
    """(T, G) bool: the group samples inside a run of at least n consecutive samples whose kept channels (chan_mask
    (G, C)) all read exactly 0; a group without a kept channel is never marked."""
    cm = np.asarray(chan_mask, dtype=bool)
    z = ((np.asarray(values) == 0) | ~cm[None]).all(axis=2) & cm.any(axis=1)[None]
    out = np.zeros(z.shape, dtype=bool)
    for g in range(z.shape[1]):
        for a, b in runs(z[:, g]):
            if b - a >= n:
                out[a:b, g] = True
    return out


def moving_mean(x, n):
    """Centred moving mean over n samples (n odd) along axis 0, over the samples of x only: the window
    shrinks at both ends."""
    h = n // 2
    L = x.shape[0]
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    i = np.arange(L)
    lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, L)
    return (cs[hi] - cs[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


@dataclass
class BandInfo:
    """What band_copy did: plan, idx (kept input sample indices), masked_short (input group samples
    masked because their valid run was too short to filter), zero_fill (valid input group samples of raw EMG
    masked as zero-fill, zero_runs)."""
    plan: BandPlan
    idx: np.ndarray
    masked_short: int
    zero_fill: int = 0


def band_copy(stream, return_info=False):
    """The band copy of a windows.Stream (module docstring): a new Stream at the kept native samples,
    values float32 in the stream's units (EMG in log mV), masked or invalid entries 0; a root track in the
    meta (windows.ROOT_KEYS) is taken at the kept samples; truth is passed through (time-indexed truth is
    sliced by the caller with BandInfo.idx). With return_info, returns (Stream, BandInfo)."""
    from scipy.signal import sosfiltfilt
    plan = band_plan(stream.modality, stream.fps)
    if plan.kind == "emg_envelope":
        kind = (stream.meta or {}).get("signal_kind")
        if kind not in (None, "raw"):
            raise ValueError(f"band_copy takes raw EMG, got signal_kind {kind!r} (a band copy is never taken twice)")
    v = np.asarray(stream.values)
    if v.ndim != 3:
        raise ValueError(f"values of shape {v.shape}, expected (T, G, C)")
    T, G, C = v.shape
    valid = np.asarray(stream.valid, dtype=bool)
    cm = np.asarray(stream.chan_mask, dtype=bool)
    t = np.asarray(stream.t)
    if valid.shape != (T, G) or cm.shape != (G, C) or t.shape != (T,):
        raise ValueError(f"t {t.shape}, valid {valid.shape} and chan_mask {cm.shape} for values {v.shape}")
    valid = valid & (np.isfinite(v) | ~cm[None]).all(axis=2)          # a non-finite entry is a gap
    zero_fill = 0
    if plan.kind == "emg_envelope":
        zf = zero_runs(v, cm, plan.rms_n) & valid                       # zero-fill is a gap
        zero_fill = int(zf.sum())
        valid &= ~zf
    keep = valid.copy()
    masked = 0
    meta = dict(stream.meta or {})
    if plan.sos is not None:
        y = np.zeros((T, G, C), dtype=np.float64)
        for gs, rs in _patterns(valid):
            chans = np.flatnonzero(cm[gs].any(axis=0))
            if not len(chans):
                continue
            for a, b in rs:
                if b - a <= plan.padlen:
                    keep[a:b, gs] = False
                    masked += (b - a) * len(gs)
                    continue
                sel = np.ix_(np.arange(a, b), gs, chans)
                out = sosfiltfilt(plan.sos, v[sel].astype(np.float64), axis=0, padlen=plan.padlen)
                if plan.kind == "emg_envelope":
                    out = np.sqrt(np.maximum(moving_mean(out * out, plan.rms_n), 0.0))
                y[sel] = out
        if plan.kind == "emg_envelope":
            for g in range(G):
                if keep[:, g].any():
                    floor = max(float(np.percentile(y[keep[:, g], g, 0], EMG_FLOOR_PCT)), EMG_FLOOR_MIN)
                    y[:, g] = np.log(np.maximum(y[:, g], floor))
            meta["signal_kind"] = "log_envelope"
    else:
        y = v
    idx = np.arange(0, T, plan.k)
    ok = keep[idx]
    vals = np.where(ok[..., None] & cm[None], y[idx], 0.0).astype(np.float32)
    W.check_root(meta, stream.modality, T, f"band_copy {stream.modality}")
    meta = W.slice_meta(meta, idx)                                      # a root track follows the kept samples
    out = W.Stream(modality=stream.modality, role=stream.role, fps=float(plan.fps_out), t=t[idx].copy(), values=vals,
                   chan_mask=cm.copy(), valid=ok, meta=meta, truth=stream.truth)
    return (out, BandInfo(plan, idx, int(masked), zero_fill)) if return_info else out


def to_float16(values, valid):
    """(values float16, valid, overflow): a group sample holding a finite value beyond the float16 range is
    marked invalid and zeroed; overflow counts those group samples."""
    with np.errstate(over="ignore"):
        h = np.asarray(values, dtype=np.float32).astype(np.float16)
    bad = (~np.isfinite(h)).any(axis=2) & valid
    if bad.any():
        valid = valid & ~bad
        h[bad] = 0
    return h, valid, int(bad.sum())


# ---------------------------------------------------------------- build
def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o).__name__}")


def _dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _mask_str(cm):
    return "|".join("".join("1" if x else "0" for x in row) for row in np.asarray(cm, dtype=bool))


def _mask_arr(text):
    return np.array([[c == "1" for c in row] for row in str(text).split("|")], dtype=bool)


def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def _record_stream(pr, s, t0):
    """(Stream, t64): every native sample of the record behind window stream `s` of a PairedRecord, with the
    arrays of windows.arrays (the same validity and cleaning as the cut) and the window stream's chan_mask
    and meta; t64 float64 native times on the recording clock, the Stream's t float32 from t0. A root track
    comes from the record's paired meta (windows.attach_root), over every native sample."""
    rec = W.whitelist(pr.streams[s.modality])
    values, cm, valid, _ = W.arrays(s.modality, rec)
    t64 = np.asarray(pr.t[s.modality], dtype=np.float64)
    if t64.shape != (values.shape[0],) or not np.array_equal(cm, s.chan_mask):
        raise ValueError(f"{pr.key} {s.modality}: record arrays differ from the cut stream")
    meta = {k: v for k, v in s.meta.items() if k not in W.ROOT_KEYS}
    extra = (getattr(pr, "meta", None) or {}).get(s.modality) or {}
    meta.update({k: extra[k] for k in W.ROOT_KEYS if k in extra})
    st = W.Stream(modality=s.modality, role=s.role, fps=s.fps, t=(t64 - t0).astype(np.float32), values=values,
                  chan_mask=s.chan_mask, valid=valid, meta=meta)
    return st, t64


def _in_span(kept_t, t0, dur, fps):
    """Positions of the kept times inside [t0, t0 + dur) with half a native period of tolerance at each edge."""
    half = 0.5 / fps
    return np.flatnonzero((kept_t >= t0 - half) & (kept_t < t0 + dur - half))


def _f16_family(modality, C):
    """Channel families whose float16 error the build reports."""
    if modality == "imu":
        return {"acc": slice(0, 3), "gyro": slice(3, 6)}
    return {"all": slice(0, C)}


class _Shard:
    """Append-only values.f16 and valid.u1 files of one stream kind; offsets in elements."""

    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.fv = open(os.path.join(folder, "values.f16"), "wb")
        self.fm = open(os.path.join(folder, "valid.u1"), "wb")
        self.nv = self.nm = 0

    def append(self, h, valid):
        at = (self.nv, self.nm)
        self.fv.write(np.ascontiguousarray(h, dtype="<f2").tobytes())
        self.fm.write(np.ascontiguousarray(valid, dtype=np.uint8).tobytes())
        self.nv += h.size
        self.nm += valid.size
        return at

    def close(self):
        self.fv.close()
        self.fm.close()


def build_source(source, out=None, recordings=None, persons=None, limit=None, paired=None, folds=None, stats=None,
                 raw_roots=None, specs=None, vocabulary=None, tier=None, names=None, index=None):
    """Band copies of the admitted recordings of a pilot source into <root>/<source>/ (module docstring),
    replacing a previous build of that source (a folder of that name that is not one is refused before any
    record is read). recordings, persons and limit narrow sources.paired; paired replaces it by an iterable
    of PairedRecords (tests); folds maps person_id to fold (default sources.folds()). A CODY or non-pilot
    source raises before the index is read. masked_short counts the group samples of the whole record whose
    valid run was too short to filter. raw_roots: attach the raw root tracks (attach_raw_root); None means
    when the records come from sources.paired (paired is None); a record that already carries a track keeps
    it. specs: the source registry {name: SourceSpec} admitting `source` (None: sources.SOURCES, the pilot's);
    vocabulary: the task vocabulary file of the windows' task covariate, absolute or relative to the tree (the
    config form, train/names.NameTables.vocabulary_path), recorded in source.json (task_vocabulary: its SHA-256,
    version and task count); None: windows.VOCABULARY, version 1, and no record (a stage-1 build's source.json
    unchanged). Returns the content of source.json.
    tier and names (task S2.5, given together or not at all): a stage-2 build of that tier (TIERS) under the run's
    name tables (train/names.NameTables). Its root (out, default the root of TIER_KIND[tier]) passes
    assert_stage2_root; `source` must be in names.sources; vocabulary defaults to names.vocabulary_path (a given one
    must have its SHA-256); every recording's person must have index rows of `tier` only (refuse_tier; index: a
    canonical.index() frame, default read), refused before its record is cut; another source of the root built under
    other name tables is refused; source.json records tier, names (SHA-256 and tables) and the vocabulary; the root
    gets its MARKER before the source folder takes its place; a refused or failed recording loop leaves no building
    folder behind (the root folder itself may stay, empty and unmarked). Without them, a
    root that carries a MARKER is refused (a stage-2 root takes builds of its tier only)."""
    from mdfm.data import sources as S
    if canonical.is_cody(source):
        raise canonical.Refused(f"CODY is evaluation-only and never enters the window store: {source}")
    if source not in (S.SOURCES if specs is None else specs):
        raise KeyError(f"not a pilot source: {source}" if specs is None else
                       f"not a source of the given specs: {source}")
    if (tier is None) != (names is None):
        raise ValueError("tier and names go together: a stage-2 build records the run's tier and name tables")
    names_record, tiers = None, None
    if tier is None:
        root = assert_store_root(store_root() if out is None else out)
        if read_marker(root) is not None:
            raise canonical.Refused(f"{root} is a {root_tier(root)} stage-2 root: a build without tier and names "
                                    f"is refused there")
    else:
        if tier not in TIERS:
            raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
        vocabulary, names_record = _stage2_names(source, names, vocabulary)
        root = assert_stage2_root(store_root(kind=TIER_KIND[tier]) if out is None else out, tier)
        for other in _source_dirs(root):
            if other == source:
                continue
            with open(os.path.join(root, other, "source.json"), encoding="utf-8") as f:
                got = (json.load(f).get("names") or {}).get("sha256")
            if got != names_record["sha256"]:
                raise canonical.Refused(f"{root}/{other} was built under the name tables {got}, this build under "
                                        f"{names_record['sha256']}: one root holds one set of name tables")
        tiers = person_tiers(index)
    stats = collections.Counter() if stats is None else stats
    vocab = vocabulary_file(vocabulary)
    vocab_record = None if vocabulary is None else vocabulary_record(vocab)
    n_tasks = None if vocabulary is None else vocab_record["tasks"]
    if folds is None:
        f = S.folds() if specs is None else S.folds(specs=specs)
        folds = dict(zip(f.loc[f["source"] == source, "person_id"], f.loc[f["source"] == source, "fold"]))
    read_roots = (paired is None) if raw_roots is None else bool(raw_roots)
    if paired is None:
        paired = S.paired(source, persons=persons, recordings=recordings, limit=limit, stats=stats, specs=specs)
    os.makedirs(root, exist_ok=True)
    final = os.path.join(root, source)
    if os.path.isdir(final):
        _assert_built_source(final, source)
    tmp = os.path.join(root, f".{source}.building")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    shards, rows = {}, []
    agg = collections.defaultdict(collections.Counter)
    grid_dev = collections.defaultdict(float)
    f16_err = collections.defaultdict(float)
    f16_over = collections.Counter()
    f16_n = collections.Counter()
    roots = collections.defaultdict(list)          # stream kind -> [(n, root or None, root_valid, pose sample valid)]
    root_missing = []
    failed = False
    try:
        for pr in paired:
            if pr.source != source:
                raise ValueError(f"a {pr.source} recording in the {source} build")
            _refuse_names(source, pr.subject_ns, pr.person_id)
            if tiers is not None:
                refuse_tier(source, pr.person_id, tier, tiers)
            fold = folds.get(pr.person_id)
            if fold is None:
                raise ValueError(f"{source} {pr.person_id}: no fold")
            root_err = None
            if read_roots and not _has_root(pr):
                try:
                    attach_raw_root(pr)
                except (LookupError, ValueError) as e:
                    root_err = f"{type(e).__name__}: {e}"[:300]
            ws = W.cut(pr, win_s=WHOLE_S, stride_s=WHOLE_S, min_s=MIN_S, stats=stats, vocabulary=vocab)
            if not ws:
                continue
            w = W.validate(ws[0], n_tasks=n_tasks)
            entries = []
            for s in w.streams:
                full, t64 = _record_stream(pr, s, w.t0)
                if np.abs(t64 - (t64[0] + np.arange(len(t64)) / s.fps)).max() > GRID_TOL_S:
                    entries = None
                    break
                fb, info = band_copy(full, return_info=True)
                j = _in_span(t64[info.idx], w.t0, w.duration, s.fps)
                if not len(j):
                    stats["stream_windows_no_kept_sample"] += 1
                    continue
                idx = info.idx[j]
                b = W.Stream(modality=fb.modality, role=fb.role, fps=fb.fps, t=fb.t[j], values=fb.values[j],
                             chan_mask=fb.chan_mask, valid=fb.valid[j], meta=W.slice_meta(fb.meta, j), truth=fb.truth)
                entries.append((s, b, info, t64[idx], full.valid[idx]))
            if entries is None:
                stats["recordings_irregular_grid"] += 1
                continue
            stats["recordings_stored"] += 1
            if root_err and any(s.modality == "pose3d" for s, *_ in entries):
                root_missing.append([pr.subject_ns, pr.key, root_err])
            for s, b, info, kept, valid_in in entries:
                grid = kept[0] + np.arange(len(kept)) * (info.plan.k / s.fps)
                grid_dev[s.modality] = max(grid_dev[s.modality], float(np.abs(grid - kept).max()))
                offset = float(kept[0])
                h, ok, nover = to_float16(b.values, b.valid)
                d = np.abs(h.astype(np.float32) - b.values)
                inside = ok[..., None] & b.chan_mask[None]
                for fam, sl in _f16_family(s.modality, h.shape[2]).items():
                    e = d[..., sl][inside[..., sl]]
                    if e.size:
                        key = f"{s.modality}.{fam}"
                        f16_err[key] = max(f16_err[key], float(e.max()))
                        f16_n[key] += int(e.size)
                        if fam in F16_BOUNDS:
                            f16_over[key] += int((e > F16_BOUNDS[fam]).sum())
                if s.modality not in shards:
                    shards[s.modality] = _Shard(os.path.join(tmp, s.modality))
                va, ma = shards[s.modality].append(h, ok)
                flags = []
                if info.plan.kind == "unfiltered":
                    flags.append("unfiltered")
                if info.masked_short:
                    flags.append("short_runs_masked")
                if info.zero_fill:
                    flags.append("zero_fill_masked")
                if nover:
                    flags.append("f16_overflow_masked")
                if not ok.any():
                    flags.append("no_valid_sample")
                meta = {k: v for k, v in b.meta.items() if k in W.META_KEYS and k not in W.ROOT_KEYS}
                has = "root" in b.meta
                roots[s.modality].append((h.shape[0], b.meta.get("root"), b.meta.get("root_valid"),
                                          ok.any(axis=1) if has else None))
                rows.append({"source": source, "stream": s.modality, "role": s.role, "person_id": pr.person_id,
                             "subject_ns": pr.subject_ns, "key": pr.key, "fold": fold, "sex": w.cov["sex"],
                             "age_band": w.cov["age_band"], "task": w.cov["task"], "span_t0": w.t0,
                             "span_s": w.duration, "fps_native": s.fps, "k": info.plan.k, "fps": b.fps,
                             "offset": offset, "n": h.shape[0], "G": h.shape[1], "C": h.shape[2], "values_at": va,
                             "valid_at": ma, "n_valid": int(ok.sum()), "masked_short": info.masked_short,
                             "f16_overflow": nover, "flags": ";".join(flags), "chan_mask": _mask_str(b.chan_mask),
                             "meta": _dumps(meta)})
                a = agg[s.modality]
                a["records"] += 1
                a["samples"] += h.shape[0]
                a["group_samples"] += h.shape[0] * h.shape[1]
                a["group_samples_valid"] += int(ok.sum())
                a["native_group_samples_masked_short"] += info.masked_short
                a["native_group_samples_zero_fill"] += info.zero_fill
                a["records_zero_fill"] += bool(info.zero_fill)
                a["group_samples_f16_overflow"] += nover
                invalid_in = ~np.asarray(valid_in, dtype=bool)     # the gaps windows.arrays marks (sample_valid, NaN)
                a["group_samples_invalid_in_input"] += int(invalid_in.sum())
                a["group_samples_made_valid"] += int((ok & invalid_in).sum())          # 0 by construction
                a[f"k={info.plan.k}"] += 1
                a[f"fps_native={s.fps:g}"] += 1
                a["seconds"] += h.shape[0] * info.plan.k / s.fps
    except BaseException:
        failed = tier is not None
        raise
    finally:
        for sh in shards.values():
            sh.close()
        if failed:
            shutil.rmtree(tmp, ignore_errors=True)
    with open(os.path.join(tmp, "records.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(RECORD_COLUMNS)
        for r in rows:
            wr.writerow([_cell(r[c]) for c in RECORD_COLUMNS])
    files = {"records.csv": sha256_file(os.path.join(tmp, "records.csv"))}
    for kind in sorted(shards):
        for name in ("values.f16", "valid.u1"):
            files[f"{kind}/{name}"] = sha256_file(os.path.join(tmp, kind, name))
    root_doc = {}
    for kind in sorted(roots):
        if any(r is not None for _, r, _, _ in roots[kind]):
            root_doc[kind] = _write_roots(os.path.join(tmp, kind), roots[kind])
            for name, _ in ROOT_FILES:
                files[f"{kind}/{name}"] = sha256_file(os.path.join(tmp, kind, name))
    by_fold = collections.defaultdict(set)
    for r in rows:
        by_fold["all"].add(r["person_id"])
        by_fold[r["fold"]].add(r["person_id"])
    summary = {"format": FORMAT, "source": source, "band": BAND, "cut": {"min_s": MIN_S},
               "stats": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in sorted(stats.items())},
               "streams": {k: dict({kk: (round(float(vv), 6) if kk == "seconds" else int(vv))
                                    for kk, vv in sorted(a.items())}, grid_max_dev_s=grid_dev[k])
                           for k, a in sorted(agg.items())},
               "f16": {k: {"max_abs_err": f16_err[k], "entries": f16_n[k],
                           "over_bound": f16_over[k] if k.split(".")[1] in F16_BOUNDS else None,
                           "bound": F16_BOUNDS.get(k.split(".")[1])} for k in sorted(f16_n)},
               "records": len(rows), "recordings": len({(r["subject_ns"], r["key"]) for r in rows}),
               "persons": {k: len(v) for k, v in sorted(by_fold.items())},
               "person_sha256": {k: person_hash(v) for k, v in sorted(by_fold.items())}, "files_sha256": files}
    if root_doc or root_missing:
        summary["root"] = {"streams": root_doc, "missing": root_missing, "dtype": "float32", "unit": "m, y up"}
    if vocab_record is not None:
        summary["task_vocabulary"] = vocab_record
    if tier is not None:
        summary["tier"] = tier
        summary["names"] = names_record
    with open(os.path.join(tmp, "source.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=_json_default)
        f.write("\n")
    if tier is not None:
        write_marker(root, tier)                   # before the source lands: a stage-2 root never shows unmarked
    if os.path.isdir(final):
        _assert_built_source(final, source)
        shutil.rmtree(final)
    os.replace(tmp, final)
    return summary


def _write_roots(folder, items):
    """root.f32 (float32 (N, 3)) and root_valid.u1 (uint8 (N,)) of a stream folder from its rows in shard order,
    items [(n, root (n, 3) float32 or None, root_valid (n,) or None, stored sample has a valid group (n,) or
    None)]: a row without a track holds zeros. Returns the counts of source.json (root.streams)."""
    parts, masks = [], []
    c = collections.Counter()
    top = 0.0
    for n, r, ok, pose_ok in items:
        c["records"] += 1
        c["samples"] += n
        if r is None:
            parts.append(np.zeros((n, 3), np.float32))
            masks.append(np.zeros(n, np.uint8))
            continue
        r = np.asarray(r, dtype=np.float32)
        ok = np.asarray(ok, dtype=bool)
        if r.shape != (n, 3) or ok.shape != (n,):
            raise ValueError(f"root {r.shape} and valid {ok.shape} for {n} samples")
        parts.append(np.where(ok[:, None], r, 0.0).astype(np.float32))
        masks.append(ok.astype(np.uint8))
        c["records_with_root"] += 1
        c["records_root_all_valid"] += bool(ok.all())
        c["records_root_none_valid"] += not ok.any()
        c["samples_with_root"] += n
        c["samples_valid"] += int(ok.sum())
        c["samples_pose_valid"] += int(pose_ok.sum())
        c["samples_pose_valid_root_invalid"] += int((pose_ok & ~ok).sum())
        if ok.any():
            top = max(top, float(np.abs(r[ok]).max()))
    with open(os.path.join(folder, ROOT_FILES[0][0]), "wb") as f:
        f.write(np.ascontiguousarray(np.concatenate(parts) if parts else np.zeros((0, 3)), dtype="<f4").tobytes())
    with open(os.path.join(folder, ROOT_FILES[1][0]), "wb") as f:
        f.write(np.ascontiguousarray(np.concatenate(masks) if masks else np.zeros(0), dtype=np.uint8).tobytes())
    return dict({k: int(v) for k, v in sorted(c.items())}, max_abs_m=top)


def _assert_built_source(folder, source):
    """Refuse to replace a folder that is not a window-store build of `source` (its source.json carries FORMAT
    and the source name): a store root pointed at a folder holding other data never loses it."""
    try:
        with open(os.path.join(folder, "source.json"), encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        doc = {}
    if doc.get("format") != FORMAT or doc.get("source") != source:
        raise canonical.Refused(f"{folder} exists and is not a {FORMAT} build of {source}: not replaced")


def build(sources=None, out=None, limit=None, recordings=None, stats=None, specs=None, vocabulary=None, tier=None,
          names=None, index=None):
    """build_source for each pilot source (default all six), folds drawn once; writes store.json with the
    per-source hashes. recordings: {source: collection of (subject_ns, recording)}; stats: {source:
    Counter}; specs and vocabulary as build_source (with specs, the default sources are those of specs).
    tier, names and index as build_source (a stage-2 build; out defaults to the root of TIER_KIND[tier], checked by
    assert_stage2_root before any source is built). Returns {source: source.json content}."""
    from mdfm.data import sources as S
    default = S.PILOT if specs is None else tuple(specs)
    names_ = list(default) if sources is None else ([sources] if isinstance(sources, str) else list(sources))
    for n in names_:
        if canonical.is_cody(n):
            raise canonical.Refused(f"CODY is evaluation-only and never enters the window store: {n}")
        if n not in (S.SOURCES if specs is None else specs):
            raise KeyError(f"not a pilot source: {n}" if specs is None else f"not a source of the given specs: {n}")
    if (tier is None) != (names is None):
        raise ValueError("tier and names go together: a stage-2 build records the run's tier and name tables")
    if tier is None:
        root = assert_store_root(store_root() if out is None else out)
    else:
        if tier not in TIERS:
            raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
        root = assert_stage2_root(store_root(kind=TIER_KIND[tier]) if out is None else out, tier)
        index = canonical.index() if index is None else index
    f = S.folds() if specs is None else S.folds(specs=specs)
    out_ = {}
    for n in names_:
        folds = dict(zip(f.loc[f["source"] == n, "person_id"], f.loc[f["source"] == n, "fold"]))
        st = None if stats is None else stats.setdefault(n, collections.Counter())
        rec = None if recordings is None else recordings.get(n)
        kw = {} if tier is None else {"tier": tier, "names": names, "index": index}
        out_[n] = build_source(n, out=root, recordings=rec, limit=limit, folds=folds, stats=st, specs=specs,
                               vocabulary=vocabulary, **kw)
    write_store_json(root)
    return out_


def write_store_json(root):
    """store.json: format and, per source present under the root, its person and file SHA-256 (and its task
    vocabulary's SHA-256 when source.json records one). A stage-2 root (one with a MARKER, task S2.5) adds its tier,
    each source's name-table SHA-256, and the name-table and vocabulary SHA-256 of the root, which a run compares
    with its config (ValueError when a source records none or the sources disagree). A source whose tier
    (source_tier) is not the root's (root_tier: RELEASE without a marker) is refused (canonical.Refused)."""
    root = assert_store_root(root)
    marker = read_marker(root)
    tier = "RELEASE" if marker is None else marker["tier"]
    doc = {"format": FORMAT, "band": BAND, "sources": {}}
    for name in _source_dirs(root):
        with open(os.path.join(root, name, "source.json"), encoding="utf-8") as f:
            s = json.load(f)
        if source_tier(s) != tier:
            raise canonical.Refused(f"{root}/{name}: a source of tier {source_tier(s)} in a store root of tier {tier}")
        doc["sources"][name] = {"person_sha256": s["person_sha256"], "files_sha256": s["files_sha256"],
                                "records": s["records"], "recordings": s["recordings"]}
        if s.get("task_vocabulary"):
            doc["sources"][name]["task_vocabulary_sha256"] = s["task_vocabulary"]["sha256"]
        if marker is not None:
            got = (s.get("names") or {}).get("sha256")
            if not got or not s.get("task_vocabulary") or s.get("tier") != marker["tier"]:
                raise ValueError(f"{root}/{name}: a source of a {marker['tier']} stage-2 root without its tier, name "
                                 f"tables or vocabulary record")
            doc["sources"][name]["names_sha256"] = got
    if marker is not None:
        doc["tier"] = marker["tier"]
        for key in ("names_sha256", "task_vocabulary_sha256"):
            vals = sorted({v[key] for v in doc["sources"].values()})
            if len(vals) > 1:
                raise ValueError(f"{root}: the sources disagree on {key}: {vals}")
            doc[key] = vals[0] if vals else None
    with open(os.path.join(root, "store.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    return doc


def _source_dirs(root):
    """Source folders of a store root that hold a source.json, in pilot order then by name."""
    from mdfm.data import sources as S
    if not os.path.isdir(root):
        return []
    names = [n for n in os.listdir(root) if not n.startswith((".", "_"))
             and os.path.isfile(os.path.join(root, n, "source.json"))]
    for n in names:
        if canonical.is_cody(n):
            raise canonical.Refused(f"a CODY folder in the window store: {root}/{n}")
    order = {n: i for i, n in enumerate(S.PILOT)}
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))


# ---------------------------------------------------------------- reader
class WindowStore:
    """A built store (module docstring). records: one row per stream of a recording (RECORD_COLUMNS, typed),
    rows of all sources in pilot order; recordings: one row per recording with the positions of its
    records. store[rec], store[rec, t0] and store[rec, t0, dur] give the Stream of records row `rec` between
    t0 and t0 + dur s on the recording clock; window(i, t0, dur) the Window of recording i; root_block(rec)
    the stored root track of a pose3d row. Opening refuses a root inside the gates-only store and any record
    of a CODY name or a lockbox person. vocabulary_sha256: the SHA-256 of the task vocabulary the reader expects
    (train/names.NameTables.vocabulary_sha256(); None: the version-1 file's); opening refuses (ValueError) a source
    built under another (source_vocabulary_sha256), so that a window's task id always means the reader's task.
    tier (task S2.4): the tier of the run the store serves, RELEASE (default) or INTERNAL (TIERS), kept as
    self.tier; check() refuses an INTERNAL person in a RELEASE store, and provenance.record refuses a RELEASE run
    on a store whose tier is INTERNAL.
    Store roots (task S2.5): opening refuses (canonical.Refused) a root whose tier (root_tier: its MARKER's, RELEASE
    without one) is not the reader's, so a stage-2 root is refused by a run of the other tier and the unmarked
    stage-1 root by an INTERNAL run; a root that is or lies in a root of the other tier by path (other_tier_roots:
    the internal root for a RELEASE run, the stage1 and s2 roots for an INTERNAL run), whatever its marker says; and a
    source whose source.json records another tier than the reader's (source_tier: RELEASE when none is recorded), so
    a source folder copied across roots or a marker removed does not let a run read the other tier's windows.
    check() of an INTERNAL store refuses a RELEASE person. names: the run's
    train/names.NameTables; it sets vocabulary_sha256 (a given one must agree) and every source must have been built
    under its SHA-256 (a source without a names record reads as names.STAGE1, the stage-1 build); a marked root is
    refused without names. self.names_sha256 holds the reader's (None without names).
    strata (task S2.8; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7): a data.strata.StrataTable, or None (the
    default: every window carries stratum 0, the stage-1 behaviour). Given one, window() sets each window's
    hsic_stratum to its person's stratum, and opening refuses (ValueError) a store holding a person the table lacks;
    self.strata_sha256 holds the table's SHA-256 (None without one)."""

    def __init__(self, root=None, vocabulary_sha256=None, tier="RELEASE", names=None, strata=None):
        import pandas as pd
        if tier not in TIERS:
            raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
        self.tier = tier
        self.root = assert_store_root(store_root() if root is None else root)
        got_tier = root_tier(self.root)
        if got_tier != tier:
            raise canonical.Refused(f"{self.root} is a store root of tier {got_tier}, refused by a run of tier {tier}")
        for kind, other in other_tier_roots(tier).items():
            if paths._inside(self.root, other):
                raise canonical.Refused(f"{self.root} lies in the {kind} window store root {other} (tier "
                                        f"{paths.WINDOW_STORE_TIER[kind]}), refused by a run of tier {tier}")
        self.marker = read_marker(self.root)
        names_sha = None
        if names is not None:
            names_sha = names.sha256()
            nv = names.vocabulary_sha256()
            if vocabulary_sha256 is not None and str(vocabulary_sha256) != nv:
                raise ValueError(f"vocabulary_sha256 {vocabulary_sha256} differs from the name tables' {nv}")
            vocabulary_sha256 = nv
        elif self.marker is not None:
            raise ValueError(f"{self.root} is a stage-2 root: it is read under the run's name tables (names)")
        want = W.task_vocabulary()["sha256"] if vocabulary_sha256 is None else str(vocabulary_sha256)
        parts, self.sources = [], {}
        stage1_sha = None
        for name in _source_dirs(self.root):
            with open(os.path.join(self.root, name, "source.json"), encoding="utf-8") as f:
                self.sources[name] = json.load(f)
            if source_tier(self.sources[name]) != tier:
                raise canonical.Refused(f"{self.root}/{name}: a source of tier {source_tier(self.sources[name])} "
                                        f"(source.json), refused by a run of tier {tier}")
            got = source_vocabulary_sha256(self.sources[name])
            if got != want:
                raise ValueError(f"{name}: built under the task vocabulary SHA-256 {got}, the reader expects {want} "
                                 f"(WindowStore vocabulary_sha256)")
            if names_sha is not None:
                got = (self.sources[name].get("names") or {}).get("sha256")
                if got is None:
                    if stage1_sha is None:
                        from mdfm.train import names as NM
                        stage1_sha = NM.STAGE1.sha256()
                    got = stage1_sha
                if got != names_sha:
                    raise ValueError(f"{name}: built under the name tables SHA-256 {got}, the reader expects "
                                     f"{names_sha} (WindowStore names)")
            d = pd.read_csv(os.path.join(self.root, name, "records.csv"), dtype=str, keep_default_na=False,
                            encoding="utf-8")
            if tuple(d.columns) != RECORD_COLUMNS:
                raise ValueError(f"{name}/records.csv: columns {list(d.columns)}")
            if (d["source"] != name).any():
                raise ValueError(f"{name}/records.csv holds rows of another source")
            parts.append(d)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(RECORD_COLUMNS))
        for c in _INT:
            df[c] = df[c].astype(int)
        for c in _FLOAT:
            df[c] = df[c].map(float).astype(float)
        for r in df[["source", "subject_ns", "person_id"]].drop_duplicates().itertuples(index=False):
            _refuse_names(r.source, r.subject_ns, r.person_id)
        df["recording"] = (df.groupby(["source", "subject_ns", "key"], sort=False).ngroup().astype(int)
                           if len(df) else pd.Series([], dtype=int))
        self.records = df
        cols = ("source", "person_id", "subject_ns", "key", "fold", "span_t0", "span_s")
        if len(df):
            self.recordings = (df.reset_index().groupby("recording", sort=True)
                               .agg(**{c: (c, "first") for c in cols}, rows=("index", list)).reset_index(drop=True))
        else:
            self.recordings = pd.DataFrame(columns=[*cols, "rows"])
        self.vocabulary_sha256 = want
        self.names_sha256 = names_sha
        self.strata, self.strata_sha256 = None, None
        if strata is not None:
            lost = strata.missing(df["person_id"]) if len(df) else []
            if lost:
                raise ValueError(f"{self.root}: {len(lost)} persons of the store are not in the HSIC strata table "
                                 f"{strata.path}, first {lost[:5]}")
            self.strata, self.strata_sha256 = strata, strata.sha256
        self._maps = {}
        self._root_maps = {}

    def __len__(self):
        return len(self.records)

    def _shard(self, source, stream):
        key = (source, stream)
        if key not in self._maps:
            folder = os.path.join(self.root, source, stream)
            self._maps[key] = tuple(_memmap(os.path.join(folder, n), dt) for n, dt in
                                    (("values.f16", "<f2"), ("valid.u1", np.uint8)))
        return self._maps[key]

    def _root_shard(self, source, stream):
        """(root float32 (N * 3,), root_valid uint8 (N,)) memmaps of a stream folder, None without root files.
        ValueError when the root files on disk differ from those source.json lists (files_sha256): a lost or
        half-present pair never reads as a store without roots."""
        key = (source, stream)
        if key not in self._root_maps:
            files = [os.path.join(self.root, source, stream, n) for n, _ in ROOT_FILES]
            listed = (self.sources.get(source) or {}).get("files_sha256") or {}
            on_disk = [os.path.isfile(p) for p in files]
            declared = [f"{stream}/{n}" in listed for n, _ in ROOT_FILES]
            if on_disk != declared:
                names = [n for n, _ in ROOT_FILES]
                raise ValueError(f"{source}/{stream}: root files on disk {[n for n, x in zip(names, on_disk) if x]}, "
                                 f"in source.json {[n for n, x in zip(names, declared) if x]}")
            self._root_maps[key] = (tuple(_memmap(p, dt) for p, (_, dt) in zip(files, ROOT_FILES))
                                    if all(on_disk) else None)
        return self._root_maps[key]

    def close(self):
        """Drop the memmaps (Windows keeps a mapped file open)."""
        self._maps.clear()
        self._root_maps.clear()

    def root_block(self, rec):
        """(root float32 (n, 3) m y up, root_valid uint8 (n,)) memmap views of the root track of records row
        `rec` (module docstring), None when its stream folder holds no root files."""
        r = self.records.iloc[int(rec)]
        sh = self._root_shard(r["source"], r["stream"])
        if sh is None:
            return None
        n, G = int(r["n"]), int(r["G"])
        at, rem = divmod(int(r["valid_at"]), G)
        if rem:
            raise ValueError(f"record {rec}: valid_at {r['valid_at']} is not a whole number of samples")
        x, m = sh[0][3 * at:3 * (at + n)], sh[1][at:at + n]
        if x.size != 3 * n or m.size != n:
            raise ValueError(f"record {rec}: root files shorter than its block")
        return x.reshape(n, 3), m

    def block(self, rec):
        """(values float16 (n, G, C), valid uint8 (n, G)) memmap views of records row `rec`."""
        r = self.records.iloc[int(rec)]
        vals, valid = self._shard(r["source"], r["stream"])
        n, G, C = int(r["n"]), int(r["G"]), int(r["C"])
        v = vals[r["values_at"]:r["values_at"] + n * G * C]
        m = valid[r["valid_at"]:r["valid_at"] + n * G]
        if v.size != n * G * C or m.size != n * G:
            raise ValueError(f"record {rec}: shard shorter than its block")
        return v.reshape(n, G, C), m.reshape(n, G)

    def times(self, rec):
        """float64 (n,) times (s, recording clock) of the kept samples of records row `rec`."""
        r = self.records.iloc[int(rec)]
        return r["offset"] + np.arange(int(r["n"])) * (int(r["k"]) / r["fps_native"])

    def stream(self, rec, t0=None, dur=None):
        """windows.Stream of records row `rec` with the kept samples t0 <= t < t0 + dur (s, recording clock;
        defaults: the recording span); t is float32 from t0, values float32. A record whose root block has a
        valid sample gets meta['root'] (float32 (T, 3) m y up) and meta['root_valid'] (bool (T,)), copies."""
        r = self.records.iloc[int(rec)]
        t0 = float(r["span_t0"]) if t0 is None else float(t0)
        dur = float(r["span_t0"] + r["span_s"] - t0) if dur is None else float(dur)
        times = self.times(rec)
        j0, j1 = (int(j) for j in np.searchsorted(times, [t0, t0 + dur], side="left"))
        v, m = self.block(rec)
        t = (times[j0:j1] - t0).astype(np.float32)
        meta = json.loads(r["meta"])
        rb = self.root_block(rec)
        if rb is not None and rb[1].any():
            meta["root"] = np.array(rb[0][j0:j1], dtype=np.float32)
            meta["root_valid"] = np.array(rb[1][j0:j1], dtype=bool)
        return W.Stream(modality=r["stream"], role=r["role"], fps=float(r["fps"]), t=t,
                        values=np.asarray(v[j0:j1], dtype=np.float32), chan_mask=_mask_arr(r["chan_mask"]),
                        valid=np.asarray(m[j0:j1], dtype=bool), meta=meta)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.stream(*key)
        return self.stream(key)

    def window(self, recording, t0=None, dur=None):
        """windows.Window of recording `recording` (a row of self.recordings) between t0 and t0 + dur s on the
        recording clock (defaults: the whole span); streams without a sample there are left out; hsic_stratum its
        person's stratum under the store's strata table (0 without one)."""
        g = self.recordings.iloc[int(recording)]
        t0 = float(g["span_t0"]) if t0 is None else float(t0)
        dur = float(g["span_t0"] + g["span_s"] - t0) if dur is None else float(dur)
        if not dur > 0:
            raise ValueError(f"window of duration {dur}")
        streams = [s for s in (self.stream(i, t0, dur) for i in g["rows"]) if len(s.t)]
        r = self.records.iloc[g["rows"][0]]
        wd = W.Window(source=g["source"], person_id=g["person_id"], key=g["key"], synthetic=False, t0=t0,
                      duration=dur, streams=streams,
                      cov={"sex": int(r["sex"]), "age_band": int(r["age_band"]), "task": int(r["task"])})
        if self.strata is not None:
            wd.hsic_stratum = int(self.strata.stratum(g["person_id"]))
        return wd

    def person_lists(self, fold=None):
        """{source: sorted person ids} of the store (of one fold when given)."""
        d = self.records if fold is None else self.records[self.records["fold"] == fold]
        return {s: sorted(set(g["person_id"])) for s, g in d.groupby("source", sort=False)}

    def person_hashes(self):
        """{source: {'all' and each fold: SHA-256 of the sorted person list}}."""
        out = {}
        for s, g in self.records.groupby("source", sort=False):
            out[s] = {"all": person_hash(g["person_id"])}
            for fold, gf in g.groupby("fold", sort=True):
                out[s][fold] = person_hash(gf["person_id"])
        return out

    def check(self, index=None):
        """Read-back of every shard. Each block must lie inside its shard, the blocks of a shard must tile it
        exactly, values must be finite with masked and invalid entries 0, n_valid must match; every person
        must be in the record index with no lockbox, EVAL, EXCLUDED_V1 or CODY row (nor, in a RELEASE store, an
        INTERNAL row, nor, in an INTERNAL store, a RELEASE row), outside the lockbox list and outside the CODY
        namespaces. Raises canonical.Refused for a person, ValueError for a shard; returns the counts otherwise,
        with max_abs: per source and channel family (imu acc and gyro, all for the others) the largest stored
        |value|, the median of the record maxima and the record holding the largest, and the signed range of the
        valid entries, lo and hi, with the medians of the record lows and highs (EMG log-envelopes are mostly
        negative: a large |value| there is a quiet floor, not a peak); and root: per source and stream folder
        holding root files, their read-back (_check_roots); root files on disk that differ from those source.json
        lists are a shard error.
        index: a canonical.index() frame (default: read it)."""
        df = canonical.index() if index is None else index
        bad = {"lockbox": set(df.loc[df["lockbox"].astype(bool), "person_id"]),
               "EVAL": set(df.loc[df["tier"] == "EVAL", "person_id"]),
               "EXCLUDED_V1": set(df.loc[df["tier"] == "EXCLUDED_V1", "person_id"]),
               "CODY": set(df.loc[df["cody"].astype(bool), "person_id"])}
        if getattr(self, "tier", "RELEASE") == "RELEASE":
            bad["INTERNAL"] = set(df.loc[df["tier"] == "INTERNAL", "person_id"])
        else:                                          # an INTERNAL store holds INTERNAL rows only (task S2.5)
            bad["RELEASE"] = set(df.loc[df["tier"] == "RELEASE", "person_id"])
        known = set(df["person_id"])
        cody_ns = set(canonical.cody_namespaces())
        problems = []
        persons = self.records[["source", "subject_ns", "person_id"]].drop_duplicates()
        for r in persons.itertuples(index=False):
            why = [k for k, s in bad.items() if r.person_id in s]
            if r.person_id in lockbox_persons():
                why.append("lockbox list")
            if canonical.is_cody(r.source, r.subject_ns, r.person_id) or (
                    {canonical.namespace(r.subject_ns), canonical.namespace(r.person_id)} & cody_ns):
                why.append("CODY name")
            if r.person_id not in known:
                why.append("not in the record index")
            if why:
                problems.append(f"{r.source} {r.person_id}: {', '.join(why)}")
        if problems:
            raise canonical.Refused(f"{len(problems)} persons that may not be in the window store: {problems[:5]}")
        shard_errors, counts, roots = [], collections.Counter(), {}
        peaks = collections.defaultdict(list)          # (source, stream.family) -> [(record max |value|, row)]
        for (source, stream), g in self.records.groupby(["source", "stream"], sort=False):
            vals, valid = self._shard(source, stream)
            ev, em = 0, 0
            for i, r in zip(g.index, g.itertuples(index=False)):
                if r.values_at != ev or r.valid_at != em:
                    shard_errors.append(f"{source}/{stream} row {i}: block at {r.values_at}, {r.valid_at} after "
                                        f"{ev}, {em}")
                ev, em = r.values_at + r.n * r.G * r.C, r.valid_at + r.n * r.G
                try:
                    v, m = self.block(i)
                except ValueError as e:
                    shard_errors.append(f"{source}/{stream}: {e}")
                    continue
                v = np.asarray(v, dtype=np.float32)
                m = np.asarray(m)
                cm = _mask_arr(r.chan_mask)
                if not np.isfinite(v).all():
                    shard_errors.append(f"{source}/{stream} row {i}: non-finite values")
                if (v[~m.astype(bool)] != 0).any() or (v[:, ~cm] != 0).any() or m.max(initial=0) > 1:
                    shard_errors.append(f"{source}/{stream} row {i}: masked or invalid entries are not 0")
                if int(m.sum()) != r.n_valid:
                    shard_errors.append(f"{source}/{stream} row {i}: {int(m.sum())} valid, row says {r.n_valid}")
                ok = m.astype(bool)[..., None] & cm[None]
                for fam, sl in _f16_family(stream, r.C).items():
                    if cm[:, sl].any():
                        x = v[..., sl][ok[..., sl]]
                        lo, hi = (float(x.min()), float(x.max())) if x.size else (math.nan, math.nan)
                        peaks[(source, f"{stream}.{fam}")].append((float(np.abs(v[..., sl]).max(initial=0.0)), i, lo, hi))
                counts["records"] += 1
                counts["values"] += v.size
            if ev != vals.size or em != valid.size:
                shard_errors.append(f"{source}/{stream}: blocks end at {ev}, {em}; shard holds {vals.size}, {valid.size}")
            counts["shards"] += 1
            counts["bytes"] += vals.size * 2 + valid.size
            try:
                has_roots = self._root_shard(source, stream) is not None
            except ValueError as e:
                shard_errors.append(str(e))
                continue
            if has_roots:
                roots[f"{source}/{stream}"] = self._check_roots(source, stream, g, shard_errors)
        if shard_errors:
            raise ValueError(f"{len(shard_errors)} shard errors: {shard_errors[:5]}")
        mags = {}
        for (source, fam), items in peaks.items():
            top, row = max(items, key=lambda it: (it[0], -it[1]))[:2]
            r = self.records.iloc[row]
            lows = [it[2] for it in items if math.isfinite(it[2])]
            highs = [it[3] for it in items if math.isfinite(it[3])]
            mags[f"{source}/{fam}"] = {"records": len(items), "max": top, "median_record_max":
                                       float(np.median([it[0] for it in items])), "max_person_id": r["person_id"],
                                       "max_key": r["key"], "lo": min(lows) if lows else None,
                                       "hi": max(highs) if highs else None,
                                       "median_record_lo": float(np.median(lows)) if lows else None,
                                       "median_record_hi": float(np.median(highs)) if highs else None}
        return {"records": int(counts["records"]), "recordings": int(len(self.recordings)),
                "shards": int(counts["shards"]), "bytes": int(counts["bytes"]), "values_read": int(counts["values"]),
                "persons": {s: len(v) for s, v in self.person_lists().items()}, "persons_refused": 0,
                "max_abs": mags, "root": roots}

    def _check_roots(self, source, stream, g, errors):
        """Read-back of the root files of one stream folder (rows g of records, in shard order): a pose3d folder
        only, one block per row tiling the files exactly, finite values, root_valid 0 or 1 and invalid rows 0.
        Appends to errors; returns the counts (records, records_with_root, samples, samples_valid, bytes,
        max_abs_m)."""
        c = collections.Counter()
        top = 0.0
        name = f"{source}/{stream}"
        root, valid = self._root_shard(source, stream)
        if stream != "pose3d":
            errors.append(f"{name}: root files outside a pose3d folder")
        for i, r in zip(g.index, g.itertuples(index=False)):
            try:
                x, m = self.root_block(i)
            except ValueError as e:
                errors.append(f"{name} root: {e}")
                continue
            x = np.asarray(x, dtype=np.float32)
            m = np.asarray(m)
            ok = m == 1
            if m.max(initial=0) > 1:
                errors.append(f"{name} row {i}: root_valid above 1")
            if not np.isfinite(x).all():
                errors.append(f"{name} row {i}: non-finite root")
            if (x[~ok] != 0).any():
                errors.append(f"{name} row {i}: invalid root rows are not 0")
            c["records"] += 1
            c["records_with_root"] += bool(ok.any())
            c["samples"] += r.n
            c["samples_valid"] += int(ok.sum())
            if ok.any():
                top = max(top, float(np.abs(x[ok]).max()))
        if c["samples"] != valid.size or 3 * c["samples"] != root.size:
            errors.append(f"{name}: root blocks end at {c['samples']} samples; files hold {valid.size}, {root.size // 3}")
        return dict({k: int(v) for k, v in sorted(c.items())}, bytes=int(root.size * 4 + valid.size), max_abs_m=top)


def _memmap(path, dtype):
    if os.path.getsize(path) == 0:
        return np.zeros(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")
