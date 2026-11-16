"""Inertial-anchor targets of F2 lever 5 (docs/proposals/F2_LEVER5_DRAFT.md, section 2; task L5): the anchor table built
once from the foot sensors of the weargait_pd training-fold recordings of the window store, its checks
(assert_table_clean) and the per-micro-step targets (ZuptTargets) that train/loop.py hands to
train/losses.zupt_anchor_parts. Label-free: the targets come from the IMU records alone, through a fixed strapdown
integration (mdfm/readouts/zupt.py, version 2 of task SCALE), with no walkway, insole, PKMAS or other reference value.

Persons. The anchored persons are the window store's weargait_pd training fold (train54 on the pilot's store), whose
SHA-256 (sources.person_hash) must equal the config's zupt_anchor.persons_sha256. Before any value is read the
builder refuses, as the walkway anchor does (data/walkway_anchor.check_persons and refused_persons, reused): a person
outside that fold, a person the window store holds out (dev), a person of the lockbox list, a lockbox, EVAL, CODY or
EXCLUDED_V1 person or the canary of the deny lists, and a person of the proposed PD folds.

IMU only (draft section 2: "No walkway, insole or PKMAS value enters the table"). The builder opens no file of
reports/ beyond the person lists above. It reads, through the window store, only records rows of stream 'imu' of
weargait_pd training-fold recordings whose key is in the section's tasks (imu_block refuses any other row before
its values are read), and of those only the foot sites FOOT_SITES with their specific force (channels 0 to 2, g with
gravity) and angular rate (channels 3 to 5, deg/s). The insole stream of weargait_pd (stream 'force') is never read.
The values are the store's band copy (data/window_store.py: 40 Hz zero-phase low-pass at 100 Hz, float16), the
model's own input; an invalid sample is NaN, through which no stride is formed. Inputs: the SHA-256 of the store's
weargait_pd records.csv, imu/values.f16 and imu/valid.u1, hashed at the build and required equal to the listing of
the source's source.json (files_sha256), whose own SHA-256 is recorded too.

Strides (readouts/zupt.strides, its constants unchanged): per foot of each anchored recording, the version-2 ZUPT
strides between consecutive still runs, each with its two still instants m_a and m_b (sample indices; the times
t_a and t_b on the recording clock, window_store.times), its length (m), its heading change (turn_deg) and its foot;
the table keeps those of length at least section.min_length_m. Rows are keyed by the window store's recording index,
never by a person.

Targets (ZuptTargets): one row per stride of a micro-step's real weargait_pd windows whose two instants lie at least
section.edge_s inside the window [t0, t0 + duration]: window (int64, the window's index in the micro-step), foot
(int64, 0 left, 1 right), t_a and t_b (float32, s from the window's start, the knot clock k / rate), dist_m (float32,
the ZUPT stride). Synthetic windows, other sources and recordings outside the table carry no row.

assert_table_clean(table, train, refused, persons_sha256) refuses (ZuptAnchorRefused, a canonical.Refused): a person
of the table outside `train` or in a refused set, a persons SHA-256 other than the given one, a row key that is not a
recording index, a stride field named like a person or a label (train/losses.label_paths), a stream other than imu
among the records read, a non-finite stride, an instant order or a length below the table's minimum, and a content
SHA-256 that is not the recomputed one; it returns the summary that the training log records ('start',
zupt_anchor.table).

Input pin (as the walkway anchor's, task FIXL4). When the environment variable PIN_ENV names a JSON file (absolute or
tree-relative) holding 'inputs' ({name: SHA-256}) and optionally 'table_sha256', build_table checks the inputs before
it reads a value and the table's content SHA-256 after the build, and raises train/provenance.ProvenanceError on a
difference, so that a run whose rule fixes its targets fails at launch. Unset (the default), nothing is checked.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, fields

import numpy as np
import torch

from .. import paths
from ..readouts import zupt as Z
from . import walkway_anchor as WA
from .canonical import Refused

SOURCE = "weargait_pd"
STREAM = "imu"
RATE = 100.0
FOOT_SITES = {"L": 11, "R": 12}             # factory.imu_render.SITES: foot_L, foot_R (the canonical location ids)
FOOT_IDS = {"L": 0, "R": 1}
ACC, GYRO = slice(0, 3), slice(3, 6)        # band-copy channels: specific force (g), angular rate (deg/s)
FORMAT = "mdfm-zupt-anchor/1"
PIN_ENV = "MDFM_ZUPT_ANCHOR_PIN"
STRIDE_FIELDS = ("m_a", "m_b", "t_a", "t_b", "length_m", "foot", "turn_deg")


class ZuptAnchorRefused(Refused):
    """A person, a row or a table the inertial anchor may not use (module docstring)."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the table
@dataclass(frozen=True)
class Strides:
    """The ZUPT strides of one recording (module docstring): m_a, m_b (n,) int64 sample indices; t_a, t_b (n,)
    float64 s on the recording clock; length_m (n,) float64; foot (n,) int64 (0 left, 1 right); turn_deg (n,)
    float64. No person field."""
    m_a: np.ndarray
    m_b: np.ndarray
    t_a: np.ndarray
    t_b: np.ndarray
    length_m: np.ndarray
    foot: np.ndarray
    turn_deg: np.ndarray

    @property
    def n(self):
        return int(len(self.t_a))


@dataclass
class ZuptTable:
    """The anchor's source table (module docstring): strides {recording index: Strides} and keys {recording index:
    recording key} of the anchored recordings; tasks; min_length_m; persons (sorted ids of the persons whose records
    were read, for the checks; never in a row) and persons_sha256 (of the training fold); streams_read (the record
    streams whose values were read); excluded {'<person>_<key>': why}; inputs {name: SHA-256}; counts; zupt (the
    method's version and constants); sha256, the content SHA-256 (content_sha256)."""
    strides: dict
    keys: dict
    tasks: tuple
    min_length_m: float
    persons: tuple
    persons_sha256: str
    streams_read: tuple = ()
    excluded: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    zupt: dict = field(default_factory=dict)
    sha256: str = ""


def content_sha256(strides, keys):
    """SHA-256 of the table's rows, one line per stride sorted by recording, foot and m_a: recording index, key, foot,
    m_a, m_b and the length in m to 6 decimals (robust to the last bits of the float arithmetic)."""
    lines = []
    for rec in sorted(strides):
        s = strides[rec]
        for i in np.lexsort((s.m_a, s.foot)):
            lines.append(f"{int(rec)}\t{keys[rec]}\t{int(s.foot[i])}\t{int(s.m_a[i])}\t{int(s.m_b[i])}\t"
                         f"{float(s.length_m[i]):.6f}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def store_inputs(window_store):
    """{name: SHA-256} of the files the builder reads values from (module docstring): the weargait_pd source.json and
    its records.csv, imu/values.f16 and imu/valid.u1, hashed now; ValueError when a shard differs from the listing of
    source.json (files_sha256)."""
    folder = os.path.join(window_store.root, SOURCE)
    listed = (window_store.sources.get(SOURCE) or {}).get("files_sha256") or {}
    out = {f"{SOURCE}/source.json": sha256_file(os.path.join(folder, "source.json"))}
    for name in ("records.csv", f"{STREAM}/values.f16", f"{STREAM}/valid.u1"):
        got = sha256_file(os.path.join(folder, name))
        if listed.get(name) != got:
            raise ValueError(f"{SOURCE}/{name}: SHA-256 {got[:12]} differs from source.json's listing "
                             f"{str(listed.get(name))[:12]}")
        out[f"{SOURCE}/{name}"] = got
    return out


def check_input_pin(inputs, table_sha=None, pin=None):
    """The input pin (module docstring): pin (default the file named by PIN_ENV; empty or unset: no check) is a JSON
    file whose 'inputs' must equal `inputs` and, when it holds 'table_sha256' and table_sha is given, whose
    'table_sha256' must equal table_sha; ProvenanceError otherwise or when the file cannot be read. Returns the pin's
    tree-relative path, or None when nothing was checked."""
    pin = os.environ.get(PIN_ENV, "") if pin is None else pin
    if not pin:
        return None
    from ..train.provenance import ProvenanceError
    path = pin if os.path.isabs(pin) else os.path.join(paths.TREE, pin)
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        want = doc["inputs"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ProvenanceError(f"zupt anchor: the input pin {pin} cannot be read ({type(e).__name__}: {e})")
    if not isinstance(want, dict) or dict(inputs) != want:
        keys = sorted(set(inputs) | set(want if isinstance(want, dict) else {}))
        bad = [k for k in keys if not isinstance(want, dict) or inputs.get(k) != want.get(k)]
        raise ProvenanceError(f"zupt anchor: the table's inputs differ from the pin {WA._rel(path)} ({bad})")
    if table_sha is not None and doc.get("table_sha256") not in (None, table_sha):
        raise ProvenanceError(f"zupt anchor: the table's content SHA-256 {table_sha[:12]} differs from the pin's "
                              f"{str(doc['table_sha256'])[:12]}")
    return WA._rel(path)


def imu_rows(window_store, recording):
    """The records rows of a recording that the builder may read: stream imu of SOURCE (module docstring)."""
    g = window_store.recordings.iloc[int(recording)]
    r = window_store.records
    return [int(i) for i in g["rows"] if r.iloc[int(i)]["stream"] == STREAM and r.iloc[int(i)]["source"] == SOURCE]


def imu_block(window_store, row):
    """(values float16 (n, G, C), valid (n, G), site ids, chan mask, fps) of records row `row`, refused
    (ZuptAnchorRefused) before any value is read unless the row is an imu row of SOURCE (the IMU-only guard)."""
    r = window_store.records.iloc[int(row)]
    if r["stream"] != STREAM or r["source"] != SOURCE:
        raise ZuptAnchorRefused(f"zupt anchor: records row {row} is a {r['source']} {r['stream']} row; the anchor "
                                f"reads {SOURCE} {STREAM} rows only")
    meta = json.loads(r["meta"])
    cm = np.array([[c == "1" for c in g] for g in str(r["chan_mask"]).split("|")], dtype=bool)
    v, m = window_store.block(int(row))
    return v, m, [int(s) for s in meta.get("site_ids", [])], cm, float(r["fps"])


def foot_signals(values, valid, site_ids, chan_mask):
    """{foot: (acc (n, 3) g, gyro (n, 3) deg/s) float64, NaN at an invalid sample} of the foot sites present with
    both channel groups."""
    out = {}
    for foot, site in FOOT_SITES.items():
        if site not in site_ids:
            continue
        j = site_ids.index(site)
        if not (chan_mask[j, ACC].all() and chan_mask[j, GYRO].all()):
            continue
        a = np.asarray(values[:, j, ACC], dtype=np.float64).copy()
        g = np.asarray(values[:, j, GYRO], dtype=np.float64).copy()
        bad = ~np.asarray(valid[:, j], dtype=bool)
        a[bad] = np.nan
        g[bad] = np.nan
        out[foot] = (a, g)
    return out


def recording_strides(window_store, row, min_length_m):
    """Strides of records row `row` (an imu row; module docstring) of length at least min_length_m, or None when the
    record holds no foot site with both channel groups."""
    v, m, sites, cm, fps = imu_block(window_store, row)
    feet = foot_signals(v, m, sites, cm)
    if not feet:
        return None
    times = window_store.times(int(row))
    cols = {k: [] for k in STRIDE_FIELDS}
    for foot, (a, g) in sorted(feet.items()):
        for s in Z.strides(a, g, fps):
            if not s["length_m"] >= float(min_length_m):
                continue
            cols["m_a"].append(int(s["m_a"]))
            cols["m_b"].append(int(s["m_b"]))
            cols["t_a"].append(float(times[s["m_a"]]))
            cols["t_b"].append(float(times[s["m_b"]]))
            cols["length_m"].append(float(s["length_m"]))
            cols["foot"].append(FOOT_IDS[foot])
            cols["turn_deg"].append(float(s["turn_deg"]))
    dt = {"m_a": np.int64, "m_b": np.int64, "foot": np.int64}
    return Strides(**{k: np.asarray(cols[k], dtype=dt.get(k, np.float64)) for k in STRIDE_FIELDS})


def build_table(window_store, section, *, pd_folds=WA.PD_FOLDS_CSV, deny=True):
    """The ZuptTable of a window store under a config.ZuptAnchorSection (module docstring), checked with
    assert_table_clean; ZuptAnchorRefused (WA.AnchorRefused for the person checks) before any value is read when
    the training fold's SHA-256 is not section.persons_sha256 or a person is refused; ProvenanceError when PIN_ENV
    names a pin that the inputs or the table's content SHA-256 do not match. deny=False skips the deny lists of
    train/provenance (tests on fake persons without the tiers records file)."""
    tasks = tuple(section.tasks)
    rec = window_store.recordings
    train = sorted(set(rec.loc[(rec["source"] == SOURCE) & (rec["fold"] == "train"), "person_id"]))
    refused = WA.refused_persons(window_store, pd_folds)
    WA.check_persons(train, train, refused, section.persons_sha256, deny=deny)
    inputs = store_inputs(window_store)
    check_input_pin(inputs)
    sel = rec[(rec["source"] == SOURCE) & (rec["fold"] == "train") & rec["key"].isin(tasks)]
    strides, keys, excluded, persons, read = {}, {}, {}, set(), set()
    per_task = {}
    for i, g in sel.iterrows():
        name = WA.trial_id(g["person_id"], g["key"])
        rows = imu_rows(window_store, i)
        if not rows:
            excluded[name] = "no imu record"
            continue
        s = recording_strides(window_store, rows[0], section.min_length_m)
        read.add(STREAM)
        if s is None:
            excluded[name] = "no foot site with acc and gyro"
            continue
        strides[int(i)] = s
        keys[int(i)] = str(g["key"])
        persons.add(str(g["person_id"]))
        pt = per_task.setdefault(str(g["key"]), {"recordings": 0, "strides": 0, "stride_seconds": 0.0})
        pt["recordings"] += 1
        pt["strides"] += s.n
        pt["stride_seconds"] += float((s.t_b - s.t_a).sum())
    sha = content_sha256(strides, keys)
    check_input_pin(inputs, sha)
    n = np.array([s.n for s in strides.values()], dtype=np.int64)
    by_person = {}
    for r, s in strides.items():
        p = str(rec.iloc[int(r)]["person_id"])
        by_person[p] = by_person.get(p, 0) + s.n
    bp = np.array(list(by_person.values()), dtype=np.int64) if by_person else np.zeros(1, np.int64)
    turn = np.abs(np.concatenate([s.turn_deg for s in strides.values()])) if strides else np.zeros(0)
    table = ZuptTable(strides=strides, keys=keys, tasks=tasks, min_length_m=float(section.min_length_m),
                      persons=tuple(sorted(persons)), persons_sha256=WA.person_hash(train),
                      streams_read=tuple(sorted(read)), excluded=dict(sorted(excluded.items())), inputs=inputs,
                      counts={"training_fold_persons": len(train), "anchored_persons": len(persons),
                              "recordings": len(strides), "strides": int(n.sum()),
                              "excluded_recordings": len(excluded),
                              "strides_per_person": {"min": int(bp.min()), "median": float(np.median(bp)),
                                                     "max": int(bp.max())},
                              "turn_over_20_deg": int((turn > 20.0).sum()),
                              "by_task": {k: dict(v, stride_seconds=round(v["stride_seconds"], 2))
                                          for k, v in sorted(per_task.items())}},
                      zupt={"version": Z.VERSION, "constants": dict(Z.CONSTANTS)}, sha256=sha)
    assert_table_clean(table, train, refused, section.persons_sha256, deny=deny)
    return table


def _row_fields(row):
    return [f.name for f in fields(row)] if hasattr(row, "__dataclass_fields__") else \
        ([str(k) for k in row] if isinstance(row, dict) else [str(k) for k in vars(row)])


def assert_table_clean(table, train, refused=None, persons_sha256=None, deny=True):
    """The checks of the module docstring on a ZuptTable; returns the summary that the training log records ('start',
    zupt_anchor.table)."""
    from ..train.losses import label_paths
    persons = WA.check_persons(table.persons, train, refused, persons_sha256, deny=deny)
    if persons_sha256 is not None and table.persons_sha256 != str(persons_sha256):
        raise ZuptAnchorRefused("zupt anchor: the table's persons SHA-256 is not the configuration's")
    if tuple(table.streams_read) not in ((), (STREAM,)):
        raise ZuptAnchorRefused(f"zupt anchor: the table read the streams {list(table.streams_read)}, not {STREAM} "
                                "alone")
    for rec, s in table.strides.items():
        if not isinstance(rec, (int, np.integer)) or isinstance(rec, bool):
            raise ZuptAnchorRefused(f"zupt anchor: a row key {rec!r} is not a recording index")
        named = [f for f in _row_fields(s) if "person" in f.lower() or "subject" in f.lower()]
        if named:
            raise ZuptAnchorRefused(f"zupt anchor: a row carries a person column ({named})")
        bad = label_paths([s])
        if bad:
            raise ZuptAnchorRefused(f"zupt anchor: label keys in the rows ({bad[:3]})")
        if s.n and not (np.isfinite(s.t_a).all() and np.isfinite(s.t_b).all() and np.isfinite(s.length_m).all()
                        and (s.t_b > s.t_a).all() and (s.length_m >= float(table.min_length_m)).all()
                        and np.isin(s.foot, (0, 1)).all()):
            raise ZuptAnchorRefused(f"zupt anchor: recording {rec} holds a stride that is not finite, ordered and at "
                                    f"least {table.min_length_m} m")
    sha = content_sha256(table.strides, table.keys)
    if table.sha256 and sha != table.sha256:
        raise ZuptAnchorRefused(f"zupt anchor: the table's content SHA-256 {table.sha256[:12]} is not its rows' "
                                f"{sha[:12]}")
    return {"format": FORMAT, "check": "passed", "persons": len(persons), "persons_sha256": table.persons_sha256,
            "anchored_persons_sha256": WA.person_hash(persons), "tasks": list(table.tasks),
            "min_length_m": float(table.min_length_m), "streams_read": list(table.streams_read),
            "excluded": dict(table.excluded), "inputs": dict(table.inputs), "counts": dict(table.counts),
            "zupt": dict(table.zupt), "table_sha256": sha,
            "refused_sets": {k: len(v) for k, v in sorted((refused or {}).items())}}


# ---------------------------------------------------------------- per-micro-step targets
@dataclass
class ZuptTargets:
    """Strides of a micro-step's windows (module docstring): window (S,) int64, foot (S,) int64, t_a and t_b (S,)
    float32 s from the window's start, dist_m (S,) float32."""
    window: torch.Tensor
    foot: torch.Tensor
    t_a: torch.Tensor
    t_b: torch.Tensor
    dist_m: torch.Tensor

    @property
    def n(self):
        return int(self.window.numel())

    def to(self, device):
        return ZuptTargets(*(getattr(self, f.name).to(device) for f in fields(self)))

    @staticmethod
    def empty():
        z = torch.zeros(0)
        return ZuptTargets(z.long(), z.long(), z.float(), z.float(), z.float())


def window_strides(s, t0, dur, edge_s):
    """Indices of the strides of `s` (Strides) whose two instants lie at least edge_s inside [t0, t0 + dur]."""
    a, b = s.t_a - float(t0), s.t_b - float(t0)
    return np.flatnonzero((a >= float(edge_s) - 1e-9) & (b <= float(dur) - float(edge_s) + 1e-9))


def targets_for_windows(table, recordings, windows, edge_s):
    """ZuptTargets of windows (windows.Window, batch order) whose recording index (WindowStore.recordings row; -1 for
    a synthetic window) is recordings[i]. A window of SOURCE whose key differs from the table's key of its recording
    raises ValueError."""
    if len(recordings) != len(windows):
        raise ValueError(f"{len(recordings)} recording indices for {len(windows)} windows")
    w, ft, ta, tb, d = [], [], [], [], []
    for i, (r, wd) in enumerate(zip(recordings, windows)):
        if wd.synthetic or wd.source != SOURCE or int(r) < 0:
            continue
        s = table.strides.get(int(r))
        if s is None:
            continue
        if str(wd.key) != table.keys[int(r)]:
            raise ValueError(f"window {i}: key {wd.key!r} differs from the anchor table's {table.keys[int(r)]!r}")
        t0 = float(wd.t0)
        for j in window_strides(s, t0, float(wd.duration), edge_s):
            w.append(i)
            ft.append(int(s.foot[j]))
            ta.append(float(s.t_a[j]) - t0)
            tb.append(float(s.t_b[j]) - t0)
            d.append(float(s.length_m[j]))
    if not w:
        return ZuptTargets.empty()
    return ZuptTargets(torch.tensor(w, dtype=torch.int64), torch.tensor(ft, dtype=torch.int64),
                       torch.tensor(ta, dtype=torch.float32), torch.tensor(tb, dtype=torch.float32),
                       torch.tensor(d, dtype=torch.float32))


def targets_for_step(table, step, edge_s):
    """ZuptTargets of a sampler Step (data.sampler.Step): its plan items give each window's recording index (-1 for a
    synthetic item), its windows the source, key, t0 and duration, in item order."""
    items = list(step.plan.items)
    if len(items) != len(step.windows):
        raise ValueError(f"{len(items)} plan items for {len(step.windows)} windows")
    for i, (it, wd) in enumerate(zip(items, step.windows)):
        if it.source != wd.source or (not wd.synthetic and it.person != wd.person_id):
            raise ValueError(f"window {i} is not the plan's item {i}")
    recs = [int(it.recording) if it.source == wd.source and not wd.synthetic else -1
            for it, wd in zip(items, step.windows)]
    return targets_for_windows(table, recs, step.windows, edge_s)
