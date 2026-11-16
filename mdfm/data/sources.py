"""Real healthy sources of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 1.3 and 2.2):
admitted index rows, pairing of the streams of one recording on one clock, person folds, and the
recorded demographics.

Rows: the selection of canonical.iterate(tier="RELEASE", lane="body", sources=...) (canonical._select:
no EXCLUDED_V1, EVAL, CODY or lockbox row, no unresolved row, one row per record), then the stream
modalities of the source and the healthy filter on the index column group (weargait_pd control,
stroke_gait_vancriekinge able_bodied; the other four sources are healthy by construction). The group
column selects rows here and nowhere else. Records are read one by one with canonical.load and its
flags left at False, so the reader refuses a lockbox, EVAL, CODY or EXCLUDED_V1 row again at read time.

Clock: each stream of a PairedRecord carries its sample times t (float64, s) on the recording clock,
whose origin is the first input sample; offsets_s[name] = t[0].
- weargait_pd: 13 Xsens IMUs (input) and insole totals (target) come from one CSV at 100 Hz: same n,
  offset 0. The insole unit is not stated in the raw folder (reports/weargait_targets): unit unknown.
  The canonical insole totals hold 0 where the raw LTotalForce and RTotalForce columns are NaN (the
  insoles not streaming: every record starts with such a run; scripts/probe_stage1_sources.py compares
  the records with the raw CSV files). A sample where both foot totals read exactly 0 is therefore
  marked invalid in the force record's sample_valid (T, 2) (SourceSpec.force_zero_gaps); a one-foot
  dropout cannot be told from swing and stays valid.
- stroke_gait_vancriekinge: pose (input, 100 Hz), plate force totals and EMG (targets, 1,000 Hz) of one
  C3D trial: n_target = n_pose x fps_target / fps_pose, offset 0. The canonical pose records are z-up
  (canonical_skeleton.yaml pig_vancriekinge); pair() rotates them to y-up by UP_ROTATION, the matrix
  of mdfm.body.rotations.Z_UP_TO_Y_UP ((x, y, z) -> (x, z, -y)), applied to the root-centred
  coordinates.
- gait120: the step records of a trial (steps 01 and 02 where both exist) carry no frame field; step
  k covers frames gait_events.step_target_frames_100hz[k] of the EMG record of the same trial, checked
  by n_frames. Each step is un-normalized by its own scale, a boundary frame shared by consecutive steps
  is kept once (the copy of the earlier step), and the joined pose is re-normalized once by the median
  pelvis-neck length over frames where both joints are valid (the MoveAtlas rule). EMG sample 0 is
  frame total_frame_100hz[0] and the trial holds (last - first + 1) x fps_emg / 100 samples. Steps
  that are not contiguous keep their gap in t.
- nonan_gaitprint, nonan_gaitprint_young, sixmwt_lifespan: one IMU stream.
A recording without its input stream is dropped and counted; a target without its input never trains.

Folds (folds()): weargait_pd controls, 14 dev (20 percent of 68, rounded half up) drawn among the TEST
half of scripts/g2_renderer_realism.py (g2_halves() replicates its prepare(): the sorted subject ids of
the non-lockbox controls with SelfPace or HurriedPace IMU rows, numpy default_rng(20260928).permutation,
the first half ALIGN), so that no mounting, device-noise or residual-spectrum statistic of the renderer
saw them; gait120 and stroke able-bodied dev 20 percent; nonan and sixmwt val 10 percent [cfg]. Draws
use numpy default_rng seeded by the SHA-256 of "<seed>:<source>"; each sorted person list is hashed.

Registry: SOURCES holds the SourceSpec of each pilot source and refuses any other name ("not a pilot source").
rows, pair, paired and folds take specs, another registry {name: SourceSpec} (a stage-2 run's, spec
docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 4.1 and 4.5, task S2.3), whose names alone they then admit; None gives
SOURCES and the stage-1 behaviour.

Stage 2, run S2a (spec section 4.5, task S2.6): STAGE2_S2A holds the five sampling units of S2a, one per source,
keyed by the source name (the store folder and the model's source code stay the source name, departure 5); unit
names the (source, groups) pair. weargait_pd:PD (the controls stay in the stage-1 replay) and
stroke_gait_vancriekinge:stroke keep their stage-1 streams, pairing and layout families (xsens13, mocap3d); their
dev folds are those of reports/stage2_sources/folds_proposed.csv, drawn by folds() with the seed key of the unit
(SourceSpec.fold_key, "<source>:<group>", the key that file was drawn with). pads, clinical_gait_multipatho and
imu_inthewild_iprognosis: every group, one IMU record per recording ('single'; pads as two single-site windows per
task, since its two watches share no clock, reports/stage2_sources/items), a val fold of 10 percent per unit [cfg]
drawn by the rule of folds() with the source name as seed key, layout families pads_wrists, multipatho4 and
pocket_acc (mdfm/factory/layouts.py, STAGE2_FAMILY_RULES). SourceSpec.tilt turns each site of each record of
clinical_gait_multipatho and imu_inthewild_iprognosis once (mdfm.data.mounting.turn_record, departure 6): the
rotations sit in the IMU stream meta ('tilt_rotation') and the pairing facts ('tilt'); never pads (no gravity) nor
weargait_pd (its controls trained unturned). SourceSpec.bulk_read reads the records of a recording through one
open container (_Containers: canonical.load itself, every flag False, with an opener that keeps the container
open), for the sources of thousands of records in one container. s2a_names() gives the name tables of S2a
(train/names.NameTables: the stage-1 tables with the three new sources and layout families appended and the
version-2 vocabulary).
STAGE2_PENDING lists the stage-2 sources outside S2a with the owner's item that holds them (spec section 9).
"""
import collections
import csv
import functools
import glob
import hashlib
import io
import math
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

from mdfm import paths
from mdfm.data import canonical

PILOT = ("weargait_pd", "gait120", "stroke_gait_vancriekinge", "nonan_gaitprint", "nonan_gaitprint_young",
         "sixmwt_lifespan")
SEED = 20260928
G2_SEED = 20260928                     # scripts/g2_renderer_realism.py prepare(seed=20260928)
G2_TASKS = ("SelfPace", "HurriedPace")
# z-up to y-up, equal to mdfm.body.rotations.Z_UP_TO_Y_UP (tests/test_sources.py checks it)
UP_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
UP_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class StreamSpec:
    """One stream of a source: kind (window stream: imu, pose3d, force, emg), the index modality, role
    (input or target), and whether a recording without it is dropped."""
    kind: str
    modality: str
    role: str
    required: bool = False


@dataclass(frozen=True)
class SourceSpec:
    """A real source of the pilot. groups: admitted values of the index column group (None: every row).
    fold: 'dev' or 'val', held out at fold_share of the persons, drawn among fold_pool when set. cap:
    share of all tokens of a micro-step [D§5]. up_axis: up axis of the canonical 3D pose records as
    recorded (measured by scripts/probe_stage1_sources.py). layout: family name of the stream layout.
    force_unit: unit of the force totals when the record states none. force_zero_gaps: the records write 0
    for a missing force sample, so a sample where both feet read exactly 0 is invalid (module docstring).
    Stage 2 (task S2.6, defaults leave a stage-1 spec as it was): fold_key, the seed key of the fold draw (None: the
    source name); tilt, turn each IMU site of the input record (mdfm.data.mounting); bulk_read, read the records
    through one open container (_Containers)."""
    name: str
    streams: tuple
    groups: tuple
    pairing: str
    fold: str
    fold_share: float
    fold_pool: str = None
    cap: float = 0.15
    up_axis: str = None
    layout: str = None
    force_unit: str = None
    force_zero_gaps: bool = False
    fold_key: str = None
    tilt: bool = False
    bulk_read: bool = False

    @property
    def unit(self):
        """Sampling unit name: '<source>:<groups joined by |>', or '<source>:all' without a group filter."""
        return f"{self.name}:{'|'.join(self.groups) if self.groups else 'all'}"

    def stream(self, modality):
        for s in self.streams:
            if s.modality == modality:
                return s
        raise KeyError(f"{self.name}: no {modality} stream")


_IMU_IN = StreamSpec("imu", "imu", "input", True)
_POSE_IN = StreamSpec("pose3d", "pose", "input", True)
SOURCES = {
    "weargait_pd": SourceSpec("weargait_pd", (_IMU_IN, StreamSpec("force", "force", "target")), ("control",),
                              "same_clock", "dev", 0.2, fold_pool="g2_test", layout="xsens13",
                              force_unit="unknown", force_zero_gaps=True),
    "gait120": SourceSpec("gait120", (_POSE_IN, StreamSpec("emg", "emg", "target")), None, "gait120_steps",
                          "dev", 0.2, up_axis="y", layout="mocap3d"),
    "stroke_gait_vancriekinge": SourceSpec("stroke_gait_vancriekinge",
                                           (_POSE_IN, StreamSpec("force", "force", "target"),
                                            StreamSpec("emg", "emg", "target")), ("able_bodied",), "c3d_trial",
                                           "dev", 0.2, up_axis="z", layout="mocap3d", force_unit="N"),
    "nonan_gaitprint": SourceSpec("nonan_gaitprint", (_IMU_IN,), None, "single", "val", 0.1, layout="nonan3"),
    "nonan_gaitprint_young": SourceSpec("nonan_gaitprint_young", (_IMU_IN,), None, "single", "val", 0.1,
                                        layout="nonan3"),
    "sixmwt_lifespan": SourceSpec("sixmwt_lifespan", (_IMU_IN,), None, "single", "val", 0.1, layout="lower_back"),
}

# stage 2, run S2a (module docstring; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.5)
STAGE2_S2A = {
    "weargait_pd": SourceSpec("weargait_pd", (_IMU_IN, StreamSpec("force", "force", "target")), ("PD",),
                              "same_clock", "dev", 0.2, layout="xsens13", force_unit="unknown", force_zero_gaps=True,
                              fold_key="weargait_pd:PD"),
    "pads": SourceSpec("pads", (_IMU_IN,), None, "single", "val", 0.1, layout="pads_wrists", bulk_read=True),
    "clinical_gait_multipatho": SourceSpec("clinical_gait_multipatho", (_IMU_IN,), None, "single", "val", 0.1,
                                           layout="multipatho4", tilt=True, bulk_read=True),
    "imu_inthewild_iprognosis": SourceSpec("imu_inthewild_iprognosis", (_IMU_IN,), None, "single", "val", 0.1,
                                           layout="pocket_acc", tilt=True, bulk_read=True),
    "stroke_gait_vancriekinge": SourceSpec("stroke_gait_vancriekinge",
                                           (_POSE_IN, StreamSpec("force", "force", "target"),
                                            StreamSpec("emg", "emg", "target")), ("stroke",), "c3d_trial",
                                           "dev", 0.2, up_axis="z", layout="mocap3d", force_unit="N",
                                           fold_key="stroke_gait_vancriekinge:stroke"),
}
# stage-2 sources outside S2a and the owner's item of spec section 9 that holds them (force-only sources: item 4,
# the force input stem; 2D video: item 5, reader, postures and tracker); koa_pd_nm is not in stage 2 (item 6)
STAGE2_PENDING = {"physionet_gaitpdb": 4, "gaitndd": 4, "gaitrec": 4, "modys_video": 5}
S2A_NEW_LAYOUTS = ("pads_wrists", "multipatho4", "pocket_acc")


def s2a_names():
    """train/names.NameTables of S2a: names.STAGE1 with the S2a sources outside it and S2A_NEW_LAYOUTS appended, the
    version-2 vocabulary (names.VOCABULARY_V2); devices, cameras and mount families as STAGE1 (the S2a families use
    the generic device and mount family until their priors are measured)."""
    from dataclasses import replace
    from mdfm.train import names as NM
    new = tuple(s for s in STAGE2_S2A if s not in NM.STAGE1.sources)
    return replace(NM.STAGE1, sources=NM.STAGE1.sources + new, vocabulary_path=NM.VOCABULARY_V2,
                   layouts=NM.STAGE1.layouts + tuple(x for x in S2A_NEW_LAYOUTS if x not in NM.STAGE1.layouts))


class PairingError(ValueError):
    """A recording whose streams do not pair under its source's rule; reason is a short tag."""

    def __init__(self, reason, message):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass
class PairedRecord:
    """The streams of one recording on one clock (module docstring). streams: kind -> record dict as
    loaded (pose rotated or joined where the rule says); t: kind -> sample times (s, float64);
    roles: kind -> input or target; meta: kind -> window meta from the source spec (layout, unit_to_si);
    pairing: facts of the join (frames, rotation) that never reach a window."""
    source: str
    person_id: str
    subject_ns: str
    key: str
    streams: dict
    t: dict
    roles: dict
    offsets_s: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    pairing: dict = field(default_factory=dict)


# ---------------------------------------------------------------- rows
def _names(sources, specs=None):
    """Source names of a call: `sources` (a name or names; None: every source of the registry), each refused
    (KeyError) unless the registry holds it. specs: the registry, {name: SourceSpec} (None: SOURCES, the pilot's,
    whose default order is PILOT)."""
    known = SOURCES if specs is None else specs
    default = PILOT if specs is None else tuple(specs)
    names = list(default) if sources is None else ([sources] if isinstance(sources, str) else list(sources))
    unknown = [n for n in names if n not in known]
    if unknown:
        raise KeyError(f"not a pilot source: {unknown}" if specs is None else
                       f"not a source of the given specs: {unknown}")
    return names


def recording_key(source, modality, key):
    """Recording id of an index row within its subject: the trial of a gait120 pose step, the trial name
    at the end of an EMG record path (<source>__<subject>__<trial>.npz), the key otherwise."""
    key = str(key)
    if modality == "emg" and key.endswith(".npz"):
        key = key.replace("\\", "/").rsplit("/", 1)[-1][:-4].rsplit("__", 1)[-1]
    if source == "gait120" and modality == "pose":
        key = re.sub(r"_step\d+$", "", key)
    return key


def rows(sources=None, specs=None):
    """Admitted index rows of the pilot sources (module docstring), with the columns stream (window
    stream kind) and recording (recording_key). No container is opened. specs: the source registry
    ({name: SourceSpec}; None: SOURCES)."""
    import pandas as pd
    names = _names(sources, specs)
    known = SOURCES if specs is None else specs
    sel = canonical._select(tier="RELEASE", lane="body", sources=names)
    parts = []
    for n in names:
        spec = known[n]
        d = sel[(sel["source"] == n) & sel["modality"].isin([s.modality for s in spec.streams])]
        if spec.groups is not None:
            d = d[d["group"].isin(spec.groups)]
        d = d.copy()
        d["stream"] = [spec.stream(m).kind for m in d["modality"]]
        d["recording"] = [recording_key(n, m, k) for m, k in zip(d["modality"], d["key"])]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["source", "person_id", "recording", "stream", "key"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- pairing
def _one(items, kind, recording):
    if len(items) != 1:
        raise PairingError("duplicate_stream", f"{recording}: {len(items)} {kind} records")
    return items[0]


def _n(rec):
    for k in ("coords", "signal", "force"):
        if k in rec:
            return int(np.asarray(rec[k]).shape[0])
    raise PairingError("no_array", "record without a main array")


def _fps(rec, recording):
    try:
        fps = float(rec.get("fps"))
    except (TypeError, ValueError):
        fps = float("nan")
    if not math.isfinite(fps) or fps <= 0:
        raise PairingError("no_fps", f"{recording}: record without fps")
    return fps


def pair(source, items, specs=None):
    """PairedRecord of one recording from its (row, record) pairs (module docstring for the rules).
    Raises PairingError when the streams do not pair. specs: the source registry (None: SOURCES)."""
    spec = (SOURCES if specs is None else specs)[source]
    if not items:
        raise PairingError("empty", f"{source}: no record")
    by = collections.defaultdict(list)
    ids = set()
    for row, rec in items:
        kind = spec.stream(row["modality"]).kind
        by[kind].append((row, rec))
        ids.add((row["person_id"], row["subject_ns"], recording_key(source, row["modality"], row["key"])))
    if len(ids) != 1:
        raise PairingError("mixed", f"{source}: records of several recordings {sorted(ids)[:3]}")
    person_id, subject_ns, recording = ids.pop()
    for s in spec.streams:
        if s.required and s.kind not in by:
            raise PairingError("no_input", f"{source} {subject_ns} {recording}: no {s.kind} record")
    roles = {s.kind: s.role for s in spec.streams if s.kind in by}
    streams, times, meta, info = {}, {}, {}, {}
    if spec.pairing == "gait120_steps":
        streams, times, info = _pair_gait120(by, recording)
    else:
        main = spec.streams[0].kind
        _, rec = _one(by[main], main, recording)
        fps = _fps(rec, recording)
        n = _n(rec)
        rec = dict(rec)
        if spec.pairing == "c3d_trial":
            rec["coords"] = (np.asarray(rec["coords"], dtype=np.float64) @ UP_ROTATION.T).astype(np.float32)
            info["up_rotation"] = UP_ROTATION.tolist()
        if spec.tilt:
            if main != "imu":
                raise ValueError(f"{source}: tilt canonicalization needs an IMU input, not {main}")
            from mdfm.data import mounting
            rec, rots, info["tilt"] = mounting.turn_record(rec)
            info["tilt_rotation"] = mounting.rotation_meta(rots)
        streams[main] = rec
        times[main] = np.arange(n, dtype=np.float64) / fps
        for s in spec.streams[1:]:
            if s.kind not in by:
                continue
            _, trec = _one(by[s.kind], s.kind, recording)
            tf = _fps(trec, recording)
            nt = _n(trec)
            want = n * tf / fps
            if abs(nt - want) > 1e-6:
                raise PairingError("clock", f"{source} {subject_ns} {recording}: {s.kind} {nt} samples at {tf:g} "
                                   f"Hz for {n} input samples at {fps:g} Hz")
            streams[s.kind] = dict(trec)
            times[s.kind] = np.arange(nt, dtype=np.float64) / tf
    for kind, rec in streams.items():
        m = {"layout": spec.layout} if spec.layout and roles[kind] == "input" else {}
        if kind == "force" and not rec.get("unit") and spec.force_unit:
            rec["unit"] = spec.force_unit
        if kind == "force" and spec.force_zero_gaps:
            gap = zero_gaps(rec)
            sv = np.repeat(~gap[:, None], 2, axis=1)
            if rec.get("sample_valid") is not None:
                sv &= np.asarray(rec["sample_valid"], dtype=bool)
            rec["sample_valid"] = sv
            info["force_zero_gap_samples"] = int(gap.sum())
        if kind == "pose3d":
            m["unit_to_si"] = 1.0
        if kind == "imu" and "tilt_rotation" in info:
            m["tilt_rotation"] = info["tilt_rotation"]
        meta[kind] = m
    return PairedRecord(source=source, person_id=person_id, subject_ns=subject_ns, key=recording, streams=streams,
                        t=times, roles=roles, offsets_s={k: float(v[0]) for k, v in times.items()}, meta=meta,
                        pairing=info)


def zero_gaps(rec):
    """(T,) bool: samples of a force record whose channels in sensor_mask read exactly 0 on both feet (a
    missing sample written as 0, SourceSpec.force_zero_gaps). A record with a foot absent from sensor_mask
    raises: one foot at 0 is swing, not a gap."""
    x = np.asarray(rec["force"])
    cm = np.asarray(rec.get("sensor_mask", np.zeros((0,), bool)), dtype=bool)
    if x.ndim != 3 or x.shape[1:] != cm.shape or not cm.any(axis=1).all():
        raise PairingError("force_mask", f"zero-gap rule needs both feet: force {x.shape}, sensor_mask {cm.shape}")
    return np.where(cm[None], x == 0, True).all(axis=(1, 2))


def _pair_gait120(by, recording):
    """Join the pose steps of one gait120 trial and place its EMG (module docstring)."""
    steps = sorted(by["pose3d"], key=lambda it: str(it[0]["key"]))
    if "emg" not in by:
        raise PairingError("no_gait_events", f"gait120 {recording}: no EMG record, so no step frames")
    _, emg = _one(by["emg"], "emg", recording)
    if str(emg.get("task")) != recording:
        raise PairingError("emg_task", f"gait120 {recording}: EMG record of task {emg.get('task')}")
    ge = emg.get("gait_events") or {}
    total = ge.get("total_frame_100hz")
    targets = ge.get("step_target_frames_100hz")
    if not total or not targets:
        raise PairingError("no_gait_events", f"gait120 {recording}: gait_events {sorted(ge)}")
    if len(targets) != len(steps):
        raise PairingError("step_count", f"gait120 {recording}: {len(steps)} pose steps, {len(targets)} targets")
    fps = {_fps(rec, recording) for _, rec in steps}
    if len(fps) != 1:
        raise PairingError("clock", f"gait120 {recording}: step rates {sorted(fps)}")
    fps = fps.pop()
    frames, xs, vs, last = [], [], [], None
    boundary = []           # metres between the two copies of each shared frame, over joints valid in both
    masks = {(tuple(np.asarray(r["joint_mask"], bool)), tuple(np.asarray(r["coord_mask"], bool))) for _, r in steps}
    if len(masks) != 1:
        raise PairingError("masks", f"gait120 {recording}: joint or coord masks differ between steps")
    for (a, b), (row, st) in zip(targets, steps):
        a, b = int(round(float(a))), int(round(float(b)))
        n = _n(st)
        if n != b - a + 1:
            raise PairingError("step_frames", f"gait120 {recording} {row['key']}: {n} frames for {a} to {b}")
        x = np.asarray(st["coords"], dtype=np.float64) * float(st["scale"])
        f = np.arange(a, b + 1)
        v = np.asarray(st["valid_mask"], dtype=bool)
        if last is not None:
            if a == last:
                both = vs[-1][-1] & v[0]
                boundary.append(float(np.abs(xs[-1][-1][both] - x[0][both]).max()) if both.any() else float("nan"))
                f, x, v = f[1:], x[1:], v[1:]
            elif a < last:
                raise PairingError("step_overlap", f"gait120 {recording}: step starts at {a} before {last}")
        frames.append(f)
        xs.append(x)
        vs.append(v)
        last = b
    frames = np.concatenate(frames)
    X = np.concatenate(xs)
    V = np.concatenate(vs)
    both = V[:, 0] & V[:, 1]
    if not both.any():
        raise PairingError("no_torso", f"gait120 {recording}: pelvis and neck never valid together")
    scale = float(np.median(np.linalg.norm(X[both, 1] - X[both, 0], axis=1)))
    if not (math.isfinite(scale) and scale > 0):
        raise PairingError("no_torso", f"gait120 {recording}: pelvis-neck length {scale}")
    first = steps[0][1]
    pose = dict(first, coords=(X / scale).astype(np.float32), valid_mask=V, scale=scale, n_frames=int(len(frames)),
                key=recording, take=recording)
    t_pose = (frames - frames[0]) / fps
    f0, f1 = int(round(float(total[0]))), int(round(float(total[-1])))
    efps = _fps(emg, recording)
    ne = _n(emg)
    want = (f1 - f0 + 1) * efps / fps
    if abs(ne - want) > 1e-6:
        raise PairingError("emg_frames", f"gait120 {recording}: {ne} EMG samples for frames {f0} to {f1}")
    if frames[0] < f0 or frames[-1] > f1:
        raise PairingError("emg_frames", f"gait120 {recording}: pose frames {frames[0]} to {frames[-1]} outside "
                           f"the EMG frames {f0} to {f1}")
    t_emg = (f0 - frames[0]) / fps + np.arange(ne, dtype=np.float64) / efps
    info = {"pose_frames": frames.tolist(), "emg_frames": [f0, f1], "step_scales": [float(r["scale"]) for _, r in steps],
            "step_frames": [[int(round(float(a))), int(round(float(b)))] for a, b in targets], "scale": scale,
            "boundary_diff_m": boundary}
    return {"pose3d": pose, "emg": dict(emg)}, {"pose3d": t_pose, "emg": t_emg}, info


def load(recording_rows):
    """(row, record) pairs of the rows of one recording, each read by canonical.load with its flags at
    False (the reader refuses lockbox, EVAL, CODY and EXCLUDED_V1 rows at read time)."""
    return [(row, canonical.load(row)) for row in recording_rows]


class _Containers:
    """canonical.load through one open container at a time (SourceSpec.bulk_read): load(row) is canonical.load(row)
    with every flag False and this object's open() as its opener, so each row passes canonical.load's own refusals
    (a CODY name, then the index row's check: no lockbox, EVAL, CODY or EXCLUDED_V1 row) before any file is opened;
    the container stays open until a row of another container comes, and close() closes it."""

    def __init__(self):
        self.path, self.zf = None, None

    def open(self, path):
        """The open container at path (the one held, or a new one after closing it)."""
        if path != self.path:
            self.close()
            self.zf, self.path = zipfile.ZipFile(path), path
        return self.zf

    def load(self, row):
        return canonical.load(row, opener=self.open)

    def close(self):
        if self.zf is not None:
            self.zf.close()
        self.path, self.zf = None, None


def paired(source, persons=None, recordings=None, limit=None, stats=None, specs=None):
    """Generator of the PairedRecords of a source, in (person, recording) order. persons: a collection of
    person ids to keep; recordings: a collection of (subject_ns, recording) to keep; stats: a Counter
    updated with paired, dropped_<reason> and rows_dropped_<reason>; specs: the source registry (None:
    SOURCES). A spec with bulk_read reads through _Containers (the same records, one container open at a time)."""
    stats = collections.Counter() if stats is None else stats
    spec = (SOURCES if specs is None else specs)[source]
    df = rows(source, specs)
    if persons is not None:
        df = df[df["person_id"].isin(set(persons))]
    required = {s.kind for s in spec.streams if s.required}
    done = 0
    bulk = _Containers() if spec.bulk_read else None
    try:
        for (pid, sns, rk), g in df.groupby(["person_id", "subject_ns", "recording"], sort=True):
            if recordings is not None and (sns, rk) not in recordings:
                continue
            if not required <= set(g["stream"]):
                stats["dropped_no_input"] += 1
                stats["rows_dropped_no_input"] += len(g)
                continue
            items = g.to_dict("records")
            try:
                pr = pair(source, load(items) if bulk is None else [(r, bulk.load(r)) for r in items], specs)
            except PairingError as e:
                stats[f"dropped_{e.reason}"] += 1
                stats[f"rows_dropped_{e.reason}"] += len(g)
                continue
            stats["paired"] += 1
            if "tilt" in pr.pairing:
                for k, v in pr.pairing["tilt"].items():
                    if k == "angle_max_deg":
                        stats["tilt_angle_max_deg"] = max(stats["tilt_angle_max_deg"], v)
                    else:
                        stats[f"tilt_sites_{k}"] += v
            yield pr
            done += 1
            if limit is not None and done >= limit:
                return
    finally:
        if bulk is not None:
            bulk.close()


# ---------------------------------------------------------------- folds
def _seed(seed, source):
    return int(hashlib.sha256(f"{seed}:{source}".encode("utf-8")).hexdigest()[:16], 16)


def _round_half_up(x):
    return int(math.floor(x + 0.5))


def person_hash(ids):
    """SHA-256 of the sorted ids joined by newlines."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def g2_halves(index=None):
    """(ALIGN, TEST) halves of the weargait_pd controls of scripts/g2_renderer_realism.py version 4, as
    sorted subject ids (module docstring)."""
    df = canonical.index() if index is None else index
    sel = ((df["source"] == "weargait_pd") & (df["modality"] == "imu") & (df["group"] == "control")
           & ~df["lockbox"] & df["key"].isin(G2_TASKS))
    persons = sorted(set(df.loc[sel, "subject_ns"]))
    perm = np.random.default_rng(G2_SEED).permutation(len(persons))
    align = {persons[i] for i in perm[: len(persons) // 2]}
    return sorted(align), sorted(set(persons) - align)


def folds(seed=SEED, specs=None):
    """DataFrame(source, person_id, fold) over the admitted persons of every pilot source (module
    docstring); fold is train, dev or val; attrs["sha256"] holds fold_hashes() of the frame. specs: the source
    registry, every source of it drawn by the same rule (None: SOURCES, in PILOT order), the seed of a source's draw
    from SourceSpec.fold_key when set (the unit key of a stage-2 dev fold), else from the source name."""
    import pandas as pd
    df = rows(specs=specs)
    out = []
    for source in _names(None, specs):
        spec = (SOURCES if specs is None else specs)[source]
        persons = sorted(set(df.loc[df["source"] == source, "person_id"]))
        n_hold = _round_half_up(spec.fold_share * len(persons))
        rng = np.random.default_rng(_seed(seed, spec.fold_key or source))
        if spec.fold_pool == "g2_test":
            pool = sorted(set(g2_halves()[1]) & set(persons))
            if len(pool) < n_hold:
                raise ValueError(f"{source}: {len(pool)} persons in the G2 TEST half for {n_hold} dev persons")
            hold = {pool[i] for i in rng.permutation(len(pool))[:n_hold]}
        elif spec.fold_pool is None:
            hold = {persons[i] for i in rng.permutation(len(persons))[:n_hold]}
        else:
            raise ValueError(f"unknown fold pool {spec.fold_pool}")
        out += [(source, p, spec.fold if p in hold else "train") for p in persons]
    frame = pd.DataFrame(out, columns=["source", "person_id", "fold"])
    frame.attrs["sha256"] = fold_hashes(frame)
    return frame


def fold_hashes(frame):
    """{source: {fold: SHA-256 of its sorted person list}} of a folds() frame."""
    out = {}
    for (source, fold), g in frame.groupby(["source", "fold"], sort=True):
        out.setdefault(source, {})[fold] = person_hash(g["person_id"])
    return out


# ---------------------------------------------------------------- demographics
def _raw(*parts):
    return "/".join([paths.MOVEATLAS, "datasets", *parts])


DEMOGRAPHIC_FILES = {
    "weargait_pd": ("imu/weargait_pd/raw/*/CONTROLS - Demographic+Clinical - datasetV1.csv",
                    "imu/weargait_pd/raw/*/PD - Demographic+Clinical - datasetV1.csv"),
    "gait120": ("emg/gait120/raw/Subject_Demographics.xlsx",),
    "stroke_gait_vancriekinge": ("pose/stroke_gait_vancriekinge/raw/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat",),
    "nonan_gaitprint": ("imu/nonan_gaitprint/raw/subject_trial_characteristics.zip",),
    "nonan_gaitprint_young": ("imu/nonan_gaitprint_young/raw/GaitPrint_Subject_Characteristics.xlsx",),
    "sixmwt_lifespan": ("imu/sixmwt_lifespan/raw/6MWT_IMU_Dataset.zip",),
}


def _sex(v):
    """'M', 'F' or '' from a text value (Male, M, Female, F); numeric codes are decoded by each table."""
    s = str(v).strip().lower()
    return {"male": "M", "m": "M", "female": "F", "f": "F"}.get(s, "")


def _num(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _weargait_tables():
    """CONTROLS and PD tables of the raw folder (header row 'Subject ID'): columns Sex (Male, Female), Age
    or Age (years), Weight (kg). Subject IDs carry no ' (control)' suffix (demographic() strips it)."""
    out = []
    for pattern in DEMOGRAPHIC_FILES["weargait_pd"]:
        files = sorted(x.replace("\\", "/") for x in glob.glob(_raw(pattern)))
        if len(files) != 1:
            raise FileNotFoundError(f"{pattern}: {len(files)} files")
        with open(files[0], encoding="utf-8-sig", newline="") as f:
            table = list(csv.reader(f))
        head = next(i for i, r in enumerate(table) if r and r[0].strip() == "Subject ID")
        cols = [c.strip() for c in table[head]]
        at = {c: cols.index(c) for c in cols if c}
        age = at.get("Age", at.get("Age (years)"))
        for r in table[head + 1:]:
            if not r or not r[0].strip():
                continue
            out.append({"subject_ns": f"weargait:{r[0].strip()}", "sex": _sex(r[at["Sex"]]), "age": _num(r[age]),
                        "mass_kg": _num(r[at["Weight (kg)"]]), "file": files[0]})
    return out


def _gait120_table():
    import pandas as pd
    p = _raw(DEMOGRAPHIC_FILES["gait120"][0])
    d = pd.read_excel(p, dtype=str)
    return [{"subject_ns": f"gait120:{s.strip()}", "sex": _sex(x), "age": _num(a), "mass_kg": _num(m), "file": p}
            for s, x, a, m in zip(d["Subject"], d["Sex"], d["Age"], d["BodyMass (Kg)"])]


def _stroke_table():
    """sub_char of Data.Sub(i) (MAT v7.3, HDF5) of the able-bodied file: Sub(i) is SUBJ i of the C3D zip
    (the xlsx ReadMe: 'numbered consistently with the MAT structure'; scripts/probe_stage1_sources.py
    checks it: the force-plate landings of events.Trial and ICstart meet a loading onset of the same foot
    in the canonical force records, reports/stage1_sources). Male: 1 male, 0 female; Weight in kg."""
    import h5py
    p = _raw(DEMOGRAPHIC_FILES["stroke_gait_vancriekinge"][0])
    out = []
    with h5py.File(p, "r") as f:
        refs = f["Sub"]["sub_char"]
        for i in range(refs.shape[0]):
            g = f[refs[i, 0]]
            val = {k: float(np.asarray(g[k]).ravel()[0]) for k in ("Male", "Age", "Weight") if k in g}
            out.append({"subject_ns": f"stroke_gait:able_bodied:S{i + 1:03d}",
                        "sex": {1.0: "M", 0.0: "F"}.get(val.get("Male"), ""),
                        "age": val.get("Age", float("nan")), "mass_kg": val.get("Weight", float("nan")), "file": p})
    return out


def _nonan_table(source):
    import pandas as pd
    p = _raw(DEMOGRAPHIC_FILES[source][0])
    if source == "nonan_gaitprint":
        with zipfile.ZipFile(p) as z:
            member = "subject_trial_characteristics/Gaitprint_subject_characteristics.csv"
            d = pd.read_csv(io.BytesIO(z.read(member)), dtype=str, keep_default_na=False, encoding="utf-8")
        ids, age, mass = d["id"], d["age"], [float("nan")] * len(d)     # height and weight carry no unit there
    else:
        d = pd.read_excel(p, dtype=str)
        ids, age, mass = [f"S{int(float(x)):03d}" for x in d["id"]], d["age (years)"], d["weight (kg)"]
    return [{"subject_ns": f"{source}:{str(s).strip()}", "sex": _sex(x), "age": _num(a), "mass_kg": _num(m), "file": p}
            for s, x, a, m in zip(ids, d["gender"], age, mass)]


def _mcos_string(u):
    """Text of a MATLAB string object's saved data: uint64 [1, 2, 1, 1, n characters, UTF-16 code units
    packed four per word]; None for any other layout."""
    u = np.asarray(u).ravel()
    if u.dtype != np.uint64 or len(u) < 5 or u[:4].tolist() != [1, 2, 1, 1]:
        return None
    return u[5:].astype("<u8").tobytes().decode("utf-16-le")[: int(u[4])]


def _mat_string(m, opaque):
    """A MATLAB string object of a v5 MAT file read by scipy.io.loadmat (m, the whole dict): its reference
    ends with (object id, class id); the MCOS subsystem in __function_workspace__ holds the objects' data in
    cells 2, 3, ... for object ids 1, 2, ... (one property each for strings)."""
    from scipy.io.matlab._mio5 import MatFile5Reader
    oid = int(np.asarray(opaque["arr"][0]).ravel()[-2])
    ws = m["__function_workspace__"].tobytes()
    head = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + ws[:4]
    r = MatFile5Reader(io.BytesIO(head + ws[8:]))
    r.mat_stream.seek(0)
    r.initialize_read()
    r.read_file_header()
    h, _ = r.read_var_header()
    cells = np.ravel(r.read_var_array(h, process=True)["MCOS"][0, 0][0]["arr"])
    return _mcos_string(cells[oid + 1])


def _sixmwt_table():
    """DATA/Subject k/Data.mat of 6MWT_IMU_Dataset.zip (subject sixmwt_lifespan:S<k, 2 digits>): Age (years),
    BodyWeight (kg) and Sex, a MATLAB string object ('M' or 'F', README: 30 males, 30 females) decoded from
    the MCOS subsystem with scipy (_mat_string)."""
    import scipy.io as sio
    p = _raw(DEMOGRAPHIC_FILES["sixmwt_lifespan"][0])
    out = []
    with zipfile.ZipFile(p) as z:
        for name in sorted(z.namelist()):
            mt = re.fullmatch(r"6MWT_IMU_Dataset/DATA/Subject (\d+)/Data\.mat", name)
            if not mt:
                continue
            m = sio.loadmat(io.BytesIO(z.read(name)))
            d = m["Data"][0, 0]
            sex = _mat_string(m, d["Sex"])
            if sex not in ("M", "F"):
                raise ValueError(f"{name}: Sex decodes to {sex!r}")
            out.append({"subject_ns": f"sixmwt_lifespan:S{int(mt.group(1)):02d}", "sex": sex,
                        "age": _num(np.asarray(d["Age"]).ravel()[0]),
                        "mass_kg": _num(np.asarray(d["BodyWeight"]).ravel()[0]), "file": p})
    return out


@functools.lru_cache(maxsize=None)
def _demographics(source):
    if source == "weargait_pd":
        return _weargait_tables()
    if source == "gait120":
        return _gait120_table()
    if source == "stroke_gait_vancriekinge":
        return _stroke_table()
    if source in ("nonan_gaitprint", "nonan_gaitprint_young"):
        return _nonan_table(source)
    if source == "sixmwt_lifespan":
        return _sixmwt_table()
    return []


def demographics(source):
    """DataFrame(subject_ns, sex 'F', 'M' or '', age years, mass_kg, file) from the source's own table
    (DEMOGRAPHIC_FILES)."""
    import pandas as pd
    return pd.DataFrame(_demographics(source), columns=["subject_ns", "sex", "age", "mass_kg", "file"])


def _lookup_id(source, subject_ns):
    s = str(subject_ns)
    return s.replace(" (control)", "") if source == "weargait_pd" else s


@functools.lru_cache(maxsize=None)
def _by_subject(source):
    return {r["subject_ns"]: r for r in _demographics(source)}


def demographic(source, subject_ns):
    """{sex, age, mass_kg} of one subject from its source table; {} when the source or subject has none."""
    if source not in SOURCES or subject_ns is None:
        return {}
    r = _by_subject(source).get(_lookup_id(source, subject_ns))
    return {} if r is None else {"sex": r["sex"] or None, "age": r["age"], "mass_kg": r["mass_kg"]}
