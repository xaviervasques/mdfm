"""Windows of real recordings at native samples (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.2).

A Window is what the sampler, collate and the losses see: streams at their native sample times, in
contract units, with masks, and the covariates kept outside the encoder input. cut() copies a
whitelist of record fields (RECORD_FIELDS) and nothing else: the index row under "manifest", tier,
lockbox, scope, lane, group, label, analog_labels, label_prefix_stripped, notes, aux_streams and any
other field of a loaded record never reach a window. gait_events serves the pairing of
mdfm.data.sources only. The window keeps person_id and key for balancing and hashing; the encoder
never reads them (collate).

Stream values are (T, G, C) float32:
- imu: G sites (meta site_ids into the 14-site taxonomy), C = 9 (acc g, gyro deg/s, then the
  magnetometer, always masked and zeroed);
- pose2d, pose3d: G = 15 canonical joints, C = 3, canonical torso units (coords x scale gives the
  source unit, metres for a 3d_meter record: unit_to_si 1.0);
- force: G = 2 feet (foot_L, foot_R), C = 9 (8 insole positions, then the total), unit N or unknown;
  valid also honours the record's sample_valid (T, 2) (the weargait zero gaps of mdfm.data.sources);
- emg: G channels (meta channels: muscle and side), C = 1, mV with unit_scale recorded.
t (T,) float32 s from the window start (record times stay float64 until t0 is subtracted);
chan_mask (G, C) and valid (T, G) bool; a masked or invalid entry holds 0.

Root track (pose3d only, optional; spec 2.2 and plan task B4.13b): meta['root'] (T, 3) float32, the raw
pelvis track in metres, y up, in the recording's world frame (mdfm.data.mocap_raw: gait120 PELV_C, stroke
PELO), one row per pose sample; meta['root_valid'] (T,) bool, a root row that is not valid holds 0. The two
keys come together (ROOT_KEYS). attach_root() puts them on a PairedRecord; cut() and the window store slice
them with the pose samples; canonical coords x scale plus meta['root'] gives the world joints in metres.

Covariates, outside the encoder input: sex (0 female, 1 male, 2 unknown) from the source demographic
table (mdfm.data.sources.demographics); age_band, the decade 0 to 7 (7 is 70 years and over, 8
unknown), logged and never a condition in the pilot; task, the id of docs/rules/task_vocabulary.yaml, or of the
vocabulary file a run names (train/names.py; task_name, task_id, covariates and cut take its path, validate its
task count; task S2.3).
"""
import collections
import functools
import hashlib
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from mdfm import paths

MODALITIES = ("imu", "pose2d", "pose3d", "force", "emg")
ROLES = ("input", "target")
# record fields a window may carry, all others are dropped by cut()
RECORD_FIELDS = ("coords", "coord_mask", "joint_mask", "valid_mask", "signal", "channel_mask", "sample_valid",
                 "location_ids", "force", "sensor_mask", "fps", "source", "subject_ns", "person_id", "key", "task",
                 "dim", "modality", "scale", "acc_content", "mag_unit", "unit", "unit_scale", "signal_kind",
                 "channels")
PAIRING_FIELDS = ("gait_events",)
# keys a stream's meta may hold (spec 2.2: site ids, acc_content, dim, scale, unit_to_si, families, source,
# muscle table, optional root track; tilt_rotation: the per-site turn of a stage-2 IMU record, mdfm.data.mounting,
# stage-2 spec section 4.5)
META_KEYS = frozenset({"source", "site_ids", "acc_content", "mag_unit", "dim", "space", "scale", "unit_to_si",
                       "unit", "unit_scale", "signal_kind", "channels", "layout", "device", "camera", "root",
                       "root_valid", "tilt_rotation"})
# time-indexed meta (one row per sample of the stream): the raw root track of a pose3d stream
ROOT_KEYS = ("root", "root_valid")
# names that never appear at any depth of a window (assert_label_free of B4.16 uses the same stems)
FORBIDDEN = ("label", "group", "cohort", "severity", "diagnosis", "phase", "manifest", "tier", "lockbox", "scope",
             "lane", "analog_labels", "notes", "aux")
SEX = {"F": 0, "M": 1}
SEX_UNKNOWN = 2
AGE_UNKNOWN = 8
N_JOINTS = 15
VOCABULARY = os.path.join(paths.TREE, "docs", "rules", "task_vocabulary.yaml")
_SUFFIX = re.compile(r"(_poses|_stageii)?\.npz$", re.I)


@dataclass
class Stream:
    """One stream of a window (module docstring). truth is set for synthetic streams only."""
    modality: str
    role: str
    fps: float
    t: np.ndarray
    values: np.ndarray
    chan_mask: np.ndarray
    valid: np.ndarray
    meta: dict = field(default_factory=dict)
    truth: dict = None


@dataclass
class Window:
    """Streams of one recording between t0 and t0 + duration (s, on the recording clock of
    mdfm.data.sources.PairedRecord). cov holds sex, age_band and task; body is synthetic truth only; hsic_stratum the
    opaque HSIC stratum of the window's person (task S2.8, mdfm.data.strata: a non-negative integer that only
    stratifies losses.hsic, never an encoder input; 0, the default and every stage-1 window's, is the one stratum of
    stage 1)."""
    source: str
    person_id: str
    key: str
    synthetic: bool
    t0: float
    duration: float
    streams: list
    cov: dict
    body: dict = None
    hsic_stratum: int = 0


# ---------------------------------------------------------------- task vocabulary
@functools.lru_cache(maxsize=None)
def task_vocabulary(path=VOCABULARY):
    """docs/rules/task_vocabulary.yaml parsed: tasks (names in id order), ids, default, rules per source
    (task, compiled regex) and the SHA-256 of the file bytes."""
    import yaml
    with open(path, "rb") as f:
        raw = f.read()
    doc = yaml.safe_load(raw.decode("utf-8"))
    tasks = list(doc["tasks"])
    ids = {name: i for i, name in enumerate(tasks)}
    rules = {}
    for source, items in (doc.get("sources") or {}).items():
        for item in items:
            if item["task"] not in ids:
                raise ValueError(f"{path}: task {item['task']!r} of {source} is not in the vocabulary")
        rules[source] = tuple((item["task"], re.compile(item["match"], re.I)) for item in items)
    if doc["default"] not in ids:
        raise ValueError(f"{path}: default {doc['default']!r} is not in the vocabulary")
    return {"tasks": tasks, "ids": ids, "default": doc["default"], "rules": rules, "status": doc.get("status"),
            "version": doc.get("version"), "sha256": hashlib.sha256(raw).hexdigest()}


def task_name(source, key, path=VOCABULARY):
    """Task name of a recording key under the rules of `source` (first match; the default otherwise). An
    amass key is matched on its take file name without _poses.npz or _stageii.npz. path: the vocabulary file
    (default docs/rules/task_vocabulary.yaml; a run's own is train/names.NameTables.vocabulary_path, resolved)."""
    vocab = task_vocabulary(path)
    key = str(key)
    if source == "amass":
        key = _SUFFIX.sub("", key.replace("\\", "/").rsplit("/", 1)[-1])
    for task, rx in vocab["rules"].get(source, ()):
        if rx.search(key):
            return task
    return vocab["default"]


def task_id(source, key, path=VOCABULARY):
    """Integer task id (position in the vocabulary of the file `path`) of a recording key."""
    return task_vocabulary(path)["ids"][task_name(source, key, path)]


# ---------------------------------------------------------------- covariates
def age_band(age):
    """Decade 0 to 7 (7: 70 years and over); 8 when the age is unknown."""
    try:
        age = float(age)
    except (TypeError, ValueError):
        return AGE_UNKNOWN
    if not math.isfinite(age) or age < 0:
        return AGE_UNKNOWN
    return int(min(age // 10, 7))


def covariates(record, source, vocabulary=VOCABULARY):
    """sex, age_band and task of a record. Sex and age come from the source demographic table
    (mdfm.data.sources.demographics); a record's own age field (sixmwt_lifespan) fills a missing age.
    The task comes from the record key (the recording key of mdfm.data.sources when given as 'key'), under the
    vocabulary file `vocabulary` (task_id)."""
    from mdfm.data import sources
    info = sources.demographic(source, record.get("subject_ns"))
    age = info.get("age")
    if age is None or (isinstance(age, float) and math.isnan(age)):
        age = record.get("age")
    key = record.get("key")
    if key is None:
        key = next((record[k] for k in ("task", "take", "sequence") if record.get(k) is not None), "")
    return {"sex": SEX.get(info.get("sex"), SEX_UNKNOWN), "age_band": age_band(age),
            "task": task_id(source, key, vocabulary)}


# ---------------------------------------------------------------- records to arrays
def whitelist(record):
    """The fields of RECORD_FIELDS present in a record (a new dict; arrays shared)."""
    return {k: record[k] for k in RECORD_FIELDS if k in record}


def record_fps(record):
    """The record's fps as a float; ValueError when absent, not finite or not positive."""
    try:
        fps = float(record.get("fps"))
    except (TypeError, ValueError):
        fps = float("nan")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"record without fps: {record.get('source')} {record.get('subject_ns')} "
                         f"{record.get('key')}")
    return fps


def _clean(x, keep):
    """x with every entry outside `keep` (broadcast bool) set to 0, NaN and inf included."""
    return np.where(keep, np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float32)


def arrays(kind, record):
    """values (T, G, C) float32, chan_mask (G, C), valid (T, G) and meta of a whitelisted record of stream
    kind `kind` (module docstring)."""
    if kind == "imu":
        x = np.asarray(record["signal"], dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 9:
            raise ValueError(f"imu signal of shape {x.shape}, expected (T, L, 9)")
        cm = np.asarray(record["channel_mask"], dtype=bool).copy()
        cm[:, 6:9] = False                                          # never the magnetometer
        sv = record.get("sample_valid")
        valid = np.ones(x.shape[:2], bool) if sv is None else np.asarray(sv, dtype=bool).copy()
        valid &= (np.isfinite(x) | ~cm[None]).all(axis=2)
        values = _clean(x, cm[None] & valid[..., None])
        meta = {"site_ids": [int(i) for i in np.asarray(record["location_ids"]).ravel()],
                "acc_content": record.get("acc_content") or "with_gravity"}
        if record.get("mag_unit"):
            meta["mag_unit"] = str(record["mag_unit"])
        return values, cm, valid, meta
    if kind in ("pose2d", "pose3d"):
        x = np.asarray(record["coords"], dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (N_JOINTS, 3):
            raise ValueError(f"pose coords of shape {x.shape}, expected (T, 15, 3)")
        dim = int(record.get("dim", 3))
        if (kind == "pose3d") != (dim == 3):
            raise ValueError(f"a dim {dim} record in a {kind} stream")
        cm = np.asarray(record["joint_mask"], dtype=bool)[:, None] & np.asarray(record["coord_mask"], dtype=bool)[None]
        if dim == 2:
            cm = cm.copy()
            cm[:, 2] = False
        valid = np.asarray(record["valid_mask"], dtype=bool).copy()
        valid &= (np.isfinite(x) | ~cm[None]).all(axis=2)
        values = _clean(x, cm[None] & valid[..., None])
        space = str(record.get("modality", ""))
        meta = {"dim": dim, "space": space, "scale": float(record["scale"]),
                "unit_to_si": 1.0 if space == "3d_meter" else None}
        return values, cm, valid, meta
    if kind == "force":
        x = np.asarray(record["force"], dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (2, 9):
            raise ValueError(f"force of shape {x.shape}, expected (T, 2, 9)")
        cm = np.asarray(record["sensor_mask"], dtype=bool).copy()
        valid = (np.isfinite(x) | ~cm[None]).all(axis=2)
        sv = record.get("sample_valid")
        if sv is not None:
            sv = np.asarray(sv, dtype=bool)
            if sv.shape != valid.shape:
                raise ValueError(f"force sample_valid of shape {sv.shape}, expected {valid.shape}")
            valid &= sv
        values = _clean(x, cm[None] & valid[..., None])
        return values, cm, valid, {"unit": str(record.get("unit") or "N")}
    if kind == "emg":
        x = np.asarray(record["signal"], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"emg signal of shape {x.shape}, expected (T, C)")
        chans = list(record.get("channels") or [])
        if len(chans) != x.shape[1]:
            raise ValueError(f"emg with {x.shape[1]} columns and {len(chans)} channel entries")
        cm = np.ones((x.shape[1], 1), bool)
        valid = np.isfinite(x)
        values = _clean(x[..., None], valid[..., None])
        meta = {"unit": record.get("unit"), "unit_scale": record.get("unit_scale"),
                "signal_kind": record.get("signal_kind"),
                "channels": [{"muscle": str(c.get("muscle")), "side": str(c.get("side"))} for c in chans]}
        return values, cm, valid, meta
    raise ValueError(f"unknown stream kind {kind!r}")


# ---------------------------------------------------------------- root track
def attach_root(paired, root, valid):
    """The PairedRecord with the root track on its pose3d stream (module docstring): root (T, 3) m y up and valid
    (T,) bool, T the pose3d samples of the record (paired.t['pose3d']). A non-finite row is invalid; invalid
    rows hold 0. paired.meta is replaced by a new dict (a meta dict shared with other records is not
    touched); the record arrays are not touched. Returns paired."""
    if "pose3d" not in paired.streams:
        raise ValueError(f"{paired.source} {paired.key}: a root track needs a pose3d stream")
    T = len(paired.t["pose3d"])
    root = np.asarray(root, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if root.shape != (T, 3) or valid.shape != (T,):
        raise ValueError(f"{paired.source} {paired.key}: root {root.shape} and valid {valid.shape} for {T} pose samples")
    ok = valid & np.isfinite(root).all(axis=1)
    meta = dict(getattr(paired, "meta", None) or {})
    meta["pose3d"] = dict(meta.get("pose3d") or {}, root=_clean(root, ok[:, None]), root_valid=ok)
    paired.meta = meta
    return paired


def root_problems(meta, modality, T):
    """Contract violations of the root track in a stream meta (module docstring), as a list of strings."""
    have = [k for k in ROOT_KEYS if k in (meta or {})]
    if not have:
        return []
    if len(have) != len(ROOT_KEYS):
        return [f"root keys {have} without {sorted(set(ROOT_KEYS) - set(have))}"]
    if modality != "pose3d":
        return [f"a root track on a {modality} stream"]
    r, ok = meta["root"], meta["root_valid"]
    if not (isinstance(r, np.ndarray) and r.dtype == np.float32 and r.shape == (T, 3)):
        return [f"root must be float32 ({T}, 3)"]
    if not (isinstance(ok, np.ndarray) and ok.dtype == bool and ok.shape == (T,)):
        return [f"root_valid must be bool ({T},)"]
    if not np.isfinite(r).all():
        return ["non-finite root"]
    if np.any(r[~ok] != 0):
        return ["invalid root rows are not 0"]
    return []


def check_root(meta, modality, T, where):
    """ValueError naming `where` when root_problems finds a violation."""
    bad = root_problems(meta, modality, T)
    if bad:
        raise ValueError(f"{where}: " + "; ".join(bad))


def slice_meta(meta, sel):
    """A copy of a stream meta with the time-indexed entries (ROOT_KEYS) taken at `sel` (a slice or an index
    array into the stream's samples); the other entries are shared."""
    out = dict(meta or {})
    for k in ROOT_KEYS:
        if k in out:
            out[k] = np.asarray(out[k])[sel]
    return out


# ---------------------------------------------------------------- cut
def cut(paired, win_s=10.0, stride_s=10.0, min_s=1.0, stats=None, vocabulary=VOCABULARY):
    """Windows of a paired record (mdfm.data.sources.PairedRecord) at native samples, no resampling.

    Windows start at the first input sample and every stride_s after it; each lasts win_s or up to the
    end of the input span (last input sample plus one period), and a window shorter than min_s (the tail,
    or a whole record shorter than min_s) is dropped and counted in `stats` (a Counter: windows,
    tails_dropped, tail_s_dropped, records_short, stream_windows_empty). A stream holds the samples with
    t0 <= t < t0 + duration; a stream with none is left out of that window. Every stream needs an fps
    (ValueError otherwise). Only RECORD_FIELDS reach the window. A root track attached to the pose3d stream
    (attach_root) is sliced with its samples. vocabulary: the task vocabulary file of the task covariate
    (covariates)."""
    stats = collections.Counter() if stats is None else stats
    names = list(paired.streams)
    inputs = [n for n in names if paired.roles[n] == "input"]
    if not inputs:
        raise ValueError(f"{paired.source} {paired.key}: no input stream")
    full = {}
    for n in names:
        if n not in MODALITIES:
            raise ValueError(f"unknown stream {n!r}")
        rec = whitelist(paired.streams[n])
        fps = record_fps(rec)
        values, cm, valid, meta = arrays(n, rec)
        t = np.asarray(paired.t[n], dtype=np.float64)
        if t.shape != (values.shape[0],):
            raise ValueError(f"{paired.key} {n}: {t.shape[0]} times for {values.shape[0]} samples")
        extra = (getattr(paired, "meta", None) or {}).get(n, {})
        meta.update({k: v for k, v in extra.items() if k in META_KEYS})
        meta["source"] = paired.source
        check_root(meta, n, values.shape[0], f"{paired.key} {n}")
        full[n] = (fps, t, values, cm, valid, meta)
    start = min(full[n][1][0] for n in inputs)
    end = max(full[n][1][-1] + 1.0 / full[n][0] for n in inputs)
    first = paired.streams[inputs[0]]
    cov = covariates(dict(first, key=paired.key), paired.source, vocabulary)
    out = []
    k = 0
    while True:
        t0 = start + k * stride_s
        if t0 >= end - 1e-9:
            break
        dur = min(win_s, end - t0)
        if dur < min_s - 1e-9:
            stats["tails_dropped"] += 1
            stats["tail_s_dropped"] += dur
            if k == 0:
                stats["records_short"] += 1
            break
        streams = []
        for n in names:
            fps, t, values, cm, valid, meta = full[n]
            i0, i1 = np.searchsorted(t, [t0, t0 + dur], side="left")
            if i1 <= i0:
                stats["stream_windows_empty"] += 1
                continue
            streams.append(Stream(modality=n, role=paired.roles[n], fps=fps, t=(t[i0:i1] - t0).astype(np.float32),
                                  values=values[i0:i1], chan_mask=cm.copy(), valid=valid[i0:i1],
                                  meta=slice_meta(meta, slice(i0, i1))))
        out.append(Window(source=paired.source, person_id=paired.person_id, key=paired.key, synthetic=False,
                          t0=float(t0), duration=float(dur), streams=streams, cov=dict(cov)))
        k += 1
    stats["windows"] += len(out)
    return out


# ---------------------------------------------------------------- validation
def keys_at_any_depth(obj, _seen=None):
    """Every dict key and dataclass field name inside obj (windows, streams, dicts, lists)."""
    out = set()
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            out.add(name)
            out |= keys_at_any_depth(getattr(obj, name))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out |= keys_at_any_depth(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= keys_at_any_depth(v)
    return out


def forbidden_keys(obj):
    """Keys inside obj whose name starts with a FORBIDDEN stem (case-insensitive)."""
    return sorted(k for k in keys_at_any_depth(obj) if k.lower().startswith(FORBIDDEN))


def validate(window, n_tasks=None):
    """Raise ValueError on any contract violation of a window (module docstring); return it otherwise. n_tasks:
    the task count of the run's vocabulary (None: the tasks of docs/rules/task_vocabulary.yaml)."""
    bad = []
    if not isinstance(window, Window):
        raise TypeError(f"not a Window: {type(window).__name__}")
    if not window.source or not window.person_id or not window.key:
        bad.append("source, person_id and key are required")
    if not (math.isfinite(window.t0) and math.isfinite(window.duration) and window.duration > 0):
        bad.append(f"t0 {window.t0} and duration {window.duration}")
    ntasks = len(task_vocabulary()["tasks"]) if n_tasks is None else int(n_tasks)
    cov = window.cov or {}
    if set(cov) != {"sex", "age_band", "task"}:
        bad.append(f"cov keys {sorted(cov)}")
    elif not (cov["sex"] in (0, 1, 2) and 0 <= cov["age_band"] <= AGE_UNKNOWN and 0 <= cov["task"] < ntasks):
        bad.append(f"cov values {cov}")
    hs = getattr(window, "hsic_stratum", 0)
    if isinstance(hs, bool) or not isinstance(hs, (int, np.integer)) or int(hs) < 0:
        bad.append(f"hsic_stratum {hs!r}: a non-negative integer")
    if not any(s.role == "input" for s in window.streams):
        bad.append("no input stream")
    for s in window.streams:
        name = f"stream {s.modality}"
        if s.modality not in MODALITIES or s.role not in ROLES:
            bad.append(f"{name}: modality {s.modality!r} role {s.role!r}")
            continue
        if not (isinstance(s.fps, float) and math.isfinite(s.fps) and s.fps > 0):
            bad.append(f"{name}: fps {s.fps!r}")
        v, t, cm, valid = s.values, s.t, s.chan_mask, s.valid
        if not (isinstance(v, np.ndarray) and v.dtype == np.float32 and v.ndim == 3):
            bad.append(f"{name}: values must be float32 (T, G, C)")
            continue
        T, G, C = v.shape
        if not (t.dtype == np.float32 and t.shape == (T,) and T > 0):
            bad.append(f"{name}: t must be float32 ({T},)")
        elif not (np.all(np.diff(t) > 0) and t[0] >= 0 and t[-1] < window.duration + 1e-4):
            bad.append(f"{name}: times not increasing inside [0, duration)")
        if not (cm.dtype == bool and cm.shape == (G, C)):
            bad.append(f"{name}: chan_mask must be bool ({G}, {C})")
            continue
        if not (valid.dtype == bool and valid.shape == (T, G)):
            bad.append(f"{name}: valid must be bool ({T}, {G})")
            continue
        if not np.isfinite(v).all():
            bad.append(f"{name}: non-finite values")
        if np.any(v[:, ~cm] != 0) or np.any(v[~valid] != 0):
            bad.append(f"{name}: masked or invalid entries are not 0")
        extra = set(s.meta) - META_KEYS
        if extra:
            bad.append(f"{name}: meta keys outside the whitelist {sorted(extra)}")
        bad += [f"{name}: {p}" for p in root_problems(s.meta, s.modality, T)]
        m = s.meta
        if s.modality == "imu":
            if C != 9 or cm[:, 6:9].any():
                bad.append(f"{name}: 9 channels with the magnetometer masked")
            ids = m.get("site_ids") or []
            if len(ids) != G or not all(0 <= i < 14 for i in ids):
                bad.append(f"{name}: site_ids {ids}")
        elif s.modality in ("pose2d", "pose3d"):
            if (G, C) != (N_JOINTS, 3) or not (m.get("scale") or 0) > 0:
                bad.append(f"{name}: (15, 3) joints and a positive scale")
            if m.get("dim") != (3 if s.modality == "pose3d" else 2):
                bad.append(f"{name}: dim {m.get('dim')}")
        elif s.modality == "force":
            if (G, C) != (2, 9) or m.get("unit") not in ("N", "unknown"):
                bad.append(f"{name}: (2, 9) and unit N or unknown")
        elif s.modality == "emg":
            if C != 1 or len(m.get("channels") or []) != G:
                bad.append(f"{name}: one column per channel entry")
        if s.truth is not None and not window.synthetic:
            bad.append(f"{name}: truth on a real window")
    if window.body is not None and not window.synthetic:
        bad.append("body truth on a real window")
    leaked = forbidden_keys(window)
    if leaked:
        bad.append(f"forbidden keys {leaked}")
    if bad:
        raise ValueError(f"{window.source} {window.key} t0 {window.t0}: " + "; ".join(bad))
    return window
