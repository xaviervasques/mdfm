"""Raw mocap root tracks and body mass (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.2; plan task B4.13).

Canonical pose records are root-centred per frame and divided by a torso scale (mdfm.kernels.pose), so the
pelvis track is gone from them. This module reads it back from the raw files in the MoveAtlas zips
(read-only), with the rules of the MoveAtlas ingest scripts that made the records:
- gait120 (ingest_gait120_pose.py): PELV_C, the nanmean of LASI, RASI, LPSI and RPSI of the step TRC
  (<zip>/S###/MotionCapture/<Task>/TRC/Trial<NN>/Step<KK>.trc, mm, 100 Hz, y up as recorded). Frames are the
  TRC Frame# column, the frames of the EMG gait_events.step_target_frames_100hz (mdfm.data.sources).
- stroke_gait_vancriekinge (ingest_stroke_gait.py): PELO of the Plug-in Gait C3D (<cohort zip>/<SUBJxx or
  TVCxx>/<trial>.c3d, mm, 100 Hz, lab z up), read with mdfm.data.c3d; a 'ID:' label prefix is stripped, an
  invalid point and a coordinate exactly 0 read NaN. The track is rotated to y up by sources.UP_ROTATION
  ((x, y, z) -> (x, z, -y)), the rotation sources.pair applies to the pose, so that canonical coords x scale
  plus the track gives world joints in metres, y up. Frames count from 0 at the first C3D frame.
Keys are the index keys: gait120 step keys <task>_trialNN_stepKK, or a trial key <task>_trialNN whose steps
are joined as sources.pair joins them (a boundary frame shared by consecutive steps kept once, from the
earlier step; a gap keeps its time); stroke keys gait_<trial>, the C3D stem with each run of
non-alphanumerics replaced by '_'.

Refusals come before any zip opens: a CODY name or namespace unless allow_eval, a source without a raw
reader, a key without index rows, and every index row of the key goes through canonical.check (EXCLUDED_V1
always, EVAL and CODY without allow_eval, lockbox rows without allow_lockbox, which only mdfm.gates passes).

Body mass (spec question 7, adopted 2026-09-29; body_mass): stroke, the median C3D PROCESSING:BODYMASS (kg)
over the subject's trials where present and positive; else sub_char Weight of the cohort MAT file (mat_char);
else NaN, and mdfm.gates.g4 estimates the mass from the plate force. gait120: BodyMass of
Subject_Demographics.xlsx (sources.demographic).

The C3D subject names (SUBJECTS:NAMES, birthdate-based on the healthy files) and label prefixes never leave
this module.
"""
import functools
import re
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np

from mdfm import paths
from mdfm.data import c3d as C3D
from mdfm.data import canonical

GAIT120 = "gait120"
STROKE = "stroke_gait_vancriekinge"
SOURCES = (GAIT120, STROKE)
PELV_C = ("LASI", "RASI", "LPSI", "RPSI")
HEAD_C = ("LFHD", "RFHD", "LBHD", "RBHD")
STROKE_ZIPS = {"able_bodied": "138_HealthyPiG.zip", "stroke": "50_StrokePiG.zip"}
STROKE_MATS = {"able_bodied": "MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat",
               "stroke": "MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"}
# z-up to y-up, equal to mdfm.data.sources.UP_ROTATION and mdfm.body.rotations.Z_UP_TO_Y_UP
UP_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
STROKE_CORE = ("PELO", "LSHO", "RSHO", "LFEP", "RFEP", "LFEO", "RFEO", "LTIO", "RTIO")
STROKE_NANFILL = ("HEDO", "LHUO", "RHUO", "LRAO", "RRAO")
_TRIAL = re.compile(r"^.+_trial\d+$")


@dataclass
class RootTrack:
    """Pelvis track of one recording. t (T,) s from the first frame (float64); pelvis (T, 3) m, y up, NaN
    where invalid; valid (T,) bool; fps Hz; frames (T,) int64 (TRC frame numbers for gait120, C3D frame
    index from 0 for stroke); mass_kg, the trial's C3D PROCESSING:BODYMASS (NaN when absent, and for
    gait120); members, the raw zip members read."""
    source: str
    subject_ns: str
    key: str
    t: np.ndarray
    pelvis: np.ndarray
    valid: np.ndarray
    fps: float
    frames: np.ndarray
    mass_kg: float = float("nan")
    members: tuple = ()


# ---------------------------------------------------------------- refusals
def _raw_dir(source):
    sub = {GAIT120: "emg/gait120/raw", STROKE: "pose/stroke_gait_vancriekinge/raw"}[source]
    return f"{paths.MOVEATLAS}/datasets/{sub}"


def index_rows(source, subject_ns, key, allow_eval=False, allow_lockbox=False):
    """The pose index rows of a key (module docstring), each passed through canonical.check; Refused and
    LookupError come before any zip opens."""
    rows = subject_rows(source, subject_ns, allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    key = str(key)
    if source == GAIT120 and _TRIAL.match(key):
        rows = [r for r in rows if re.fullmatch(re.escape(key) + r"_step\d+", str(r["key"]))]
    else:
        rows = [r for r in rows if str(r["key"]) == key]
    if not rows:
        raise LookupError(f"no pose index row for {source} {subject_ns} {key}")
    return rows


def subject_rows(source, subject_ns, allow_eval=False, allow_lockbox=False):
    """Every pose index row of one subject, each passed through canonical.check: a CODY name refused by name
    before the index is read, then the CODY namespaces, tiers and the lockbox."""
    if not allow_eval and canonical.is_cody(source, subject_ns):
        raise canonical.Refused(f"CODY is evaluation-only: {source} {subject_ns} (pass allow_eval=True in an "
                                "evaluation stage)")
    canonical.check({"source": source, "subject_ns": subject_ns, "person_id": subject_ns},
                    allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    if source not in SOURCES:
        raise KeyError(f"no raw mocap reader for source {source!r} (readers: {SOURCES})")
    rows = _pose_rows(source).get(subject_ns, [])
    if not rows:
        raise LookupError(f"no pose index row for {source} {subject_ns}")
    return [canonical.check(r, allow_eval=allow_eval, allow_lockbox=allow_lockbox) for r in rows]


@functools.lru_cache(maxsize=None)
def _pose_rows(source):
    """{subject_ns: [index row dicts]} of the pose rows of a source (canonical.index, read once)."""
    df = canonical.index()
    out = {}
    for r in df[(df["source"] == source) & (df["modality"] == "pose")].to_dict("records"):
        out.setdefault(r["subject_ns"], []).append(r)
    return out


# ---------------------------------------------------------------- gait120 TRC
def snake(s):
    """CamelCase task folder to the snake_case of the record keys (ingest_gait120_pose.py)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def gait120_zip(subject_ns):
    """Zip name of a gait120 subject (Gait120_<lo>_to_<lo + 9>.zip, ten subjects per zip)."""
    num = int(str(subject_ns).rsplit(":S", 1)[1])
    lo = (num - 1) // 10 * 10 + 1
    return f"Gait120_{lo:03d}_to_{lo + 9:03d}.zip"


def parse_trc(raw):
    """(markers (M,), unit, fps, frames (T,) int64, xyz (T, M, 3) float64 in the file unit, NaN for an
    empty cell, header dict) of TRC bytes, as ingest_gait120_pose.parse_trc reads them."""
    lines = raw.decode("utf-8", "replace").splitlines()
    keys, vals = lines[1].split("\t"), lines[2].split("\t")
    header = {k.strip(): v.strip() for k, v in zip(keys, vals) if k.strip()}
    fps = float(vals[0])
    unit = vals[4].strip().lower()
    names = [x.strip() for x in lines[3].split("\t")]
    markers = [n for n in names[2:] if n]
    rows, frames = [], []
    for ln in lines[5:]:
        if not ln.strip():
            continue
        f = ln.split("\t")
        v = [float(x) if x.strip() not in ("", "nan", "NaN") else np.nan for x in f[2:2 + 3 * len(markers)]]
        v += [np.nan] * (3 * len(markers) - len(v))
        rows.append(v)
        frames.append(int(float(f[0])))
    xyz = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(markers), 3)
    return markers, unit, fps, np.asarray(frames, dtype=np.int64), xyz, header


def _centroid(xyz, markers, names):
    idx = {n: i for i, n in enumerate(markers)}
    missing = [n for n in names if n not in idx]
    if missing:
        raise ValueError(f"TRC without markers {missing}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all four markers missing: NaN
        return np.nanmean(np.stack([xyz[:, idx[n]] for n in names], axis=1), axis=1)


@functools.lru_cache(maxsize=16)
def _gait120_members(zname):
    """{(subject_ns, step key): member} of the step TRC files of one zip (static trials left out)."""
    out = {}
    with zipfile.ZipFile(f"{_raw_dir(GAIT120)}/{zname}") as z:
        for name in z.namelist():
            m = re.fullmatch(r"S(\d+)/MotionCapture/([^/]+)/TRC/Trial(\d+)/Step(\d+)\.trc", name, re.I)
            if not m or "/Static/" in name:
                continue
            key = f"{snake(m.group(2))}_trial{int(m.group(3)):02d}_step{int(m.group(4)):02d}"
            out[(f"gait120:S{int(m.group(1)):03d}", key)] = name
    return out


def _gait120_steps(subject_ns, keys):
    """[(step key, markers, fps, frames, xyz m, member)] of the step keys, read from their zip."""
    zname = gait120_zip(subject_ns)
    members = _gait120_members(zname)
    out = []
    with zipfile.ZipFile(f"{_raw_dir(GAIT120)}/{zname}") as z:
        for key in keys:
            member = members.get((subject_ns, key))
            if member is None:
                raise LookupError(f"gait120 {subject_ns} {key}: no TRC member in {zname}")
            markers, unit, fps, frames, xyz, _ = parse_trc(z.read(member))
            if unit != "mm":
                raise ValueError(f"{member}: unit {unit!r}, expected mm")
            out.append((key, markers, fps, frames, xyz / 1000.0, f"{zname}/{member}"))
    return out


def _join_steps(steps):
    """Frames, per-step arrays and fps of consecutive steps joined (module docstring)."""
    fps = {s[2] for s in steps}
    if len(fps) != 1:
        raise ValueError(f"step rates {sorted(fps)}")
    keep, last = [], None
    for key, markers, _, frames, xyz, member in steps:
        sel = np.ones(len(frames), dtype=bool)
        if last is not None:
            if frames[0] < last:
                raise ValueError(f"{key} starts at frame {frames[0]} before {last}")
            if frames[0] == last:
                sel[0] = False
        keep.append((markers, frames[sel], xyz[sel], member))
        last = int(frames[-1])
    return keep, fps.pop()


# ---------------------------------------------------------------- stroke C3D
def _stroke_key(member):
    base = member.rsplit("/", 1)[-1][:-4]
    return "gait_" + re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


@functools.lru_cache(maxsize=4)
def _stroke_members(group):
    """{(subject_ns, key): member} of the trial C3D files of one cohort zip (calibration trials left out)."""
    zname = STROKE_ZIPS[group]
    out = {}
    with zipfile.ZipFile(f"{_raw_dir(STROKE)}/{zname}") as z:
        for name in z.namelist():
            parts = name.split("/")
            if not name.lower().endswith(".c3d") or "cal" in parts[-1].lower() or len(parts) < 2:
                continue
            m = re.search(r"(\d+)", parts[-2])
            if not m:
                continue
            k = (f"stroke_gait:{group}:S{int(m.group(1)):03d}", _stroke_key(name))
            if k in out:
                raise ValueError(f"{zname}: {out[k]} and {name} map to one key {k}")
            out[k] = name
    return out


def _stroke_group(subject_ns):
    group = str(subject_ns).split(":")[1] if str(subject_ns).count(":") >= 2 else ""
    if group not in STROKE_ZIPS:
        raise LookupError(f"stroke subject {subject_ns}: cohort {group!r} not in {sorted(STROKE_ZIPS)}")
    return group


def _stroke_c3d(subject_ns, key):
    """(C3D, labels without prefix, member) of one stroke trial."""
    group = _stroke_group(subject_ns)
    member = _stroke_members(group).get((subject_ns, key))
    if member is None:
        raise LookupError(f"stroke {subject_ns} {key}: no C3D member in {STROKE_ZIPS[group]}")
    with zipfile.ZipFile(f"{_raw_dir(STROKE)}/{STROKE_ZIPS[group]}") as z:
        c = C3D.read(z.read(member))
    unit = str(c.units).strip().lower()
    if unit != "mm":
        raise ValueError(f"{member}: point unit {unit!r}, expected mm")
    labels = [x.split(":", 1)[-1] for x in c.labels]
    return c, labels, f"{STROKE_ZIPS[group]}/{member}"


def _stroke_point(c, labels, name):
    hits = [i for i, x in enumerate(labels) if x == name]
    if len(hits) != 1:
        raise ValueError(f"C3D label {name!r} on {len(hits)} points")
    x = c.points[:, hits[0]].copy()
    x[x == 0.0] = np.nan                                # the ingest reads a zero coordinate as missing
    return x / 1000.0


def c3d_mass(c):
    """PROCESSING:BODYMASS in kg of a C3D, NaN when absent or not positive."""
    v = c.param("PROCESSING", "BODYMASS")
    if v is None:
        return float("nan")
    v = float(np.asarray(v, dtype=np.float64).ravel()[0]) if np.size(v) else float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


@functools.lru_cache(maxsize=4)
def _stroke_folders(group):
    """Subject numbers of the subject folders of one cohort zip, in number order (member names only)."""
    nums = set()
    with zipfile.ZipFile(f"{_raw_dir(STROKE)}/{STROKE_ZIPS[group]}") as z:
        for name in z.namelist():
            parts = name.split("/")
            m = re.fullmatch(r"(?:SUBJ|TVC)(\d+)", parts[-2]) if len(parts) >= 2 else None
            if m and name.lower().endswith(".c3d"):
                nums.add(int(m.group(1)))
    return tuple(sorted(nums))


def mat_char(subject_ns, allow_lockbox=False, allow_eval=False):
    """sub_char of one stroke subject from its cohort MAT file (MAT v7.3), {name: float} (Weight kg, Height
    mm, LegLength mm, Male, Age, ...): Data.Sub(i) with i the rank of the subject's folder number among the
    folders of its cohort zip (able-bodied: SUBJ i is Sub(i), found by B4.5; post-stroke: the TVC folders in
    number order, checked by scripts/gate_g4.py against the C3D PROCESSING parameters). Only that entry is
    read. Refusals: the subject's pose index rows through canonical.check."""
    import h5py
    subject_rows(STROKE, subject_ns, allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    group = _stroke_group(subject_ns)
    num = int(str(subject_ns).rsplit(":S", 1)[1])
    folders = _stroke_folders(group)
    if num not in folders:
        raise LookupError(f"{subject_ns}: no folder {num} in {STROKE_ZIPS[group]}")
    with h5py.File(f"{_raw_dir(STROKE)}/{STROKE_MATS[group]}", "r") as f:
        refs = f["Sub"]["sub_char"]
        if refs.shape[0] != len(folders):
            raise ValueError(f"{STROKE_MATS[group]}: {refs.shape[0]} subjects for {len(folders)} folders")
        g = f[refs[folders.index(num), 0]]
        return {k: float(np.asarray(g[k]).ravel()[0]) for k in g.keys()
                if np.asarray(g[k]).dtype.kind in "fiu" and np.asarray(g[k]).size == 1}


def body_mass(source, subject_ns, allow_lockbox=False, allow_eval=False):
    """(mass kg, how) of one subject (module docstring): stroke, the median of the positive C3D
    PROCESSING:BODYMASS over the subject's pose trials ('c3d'), else the MAT sub_char Weight ('mat'), else
    (NaN, ''); gait120, the BodyMass of its demographic table ('table')."""
    rows = subject_rows(source, subject_ns, allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    if source == GAIT120:
        from mdfm.data import sources
        m = sources.demographic(source, subject_ns).get("mass_kg")
        ok = m is not None and np.isfinite(m) and m > 0
        return (float(m), "table") if ok else (float("nan"), "")
    masses = []
    for r in rows:
        c, _, _ = _stroke_c3d(subject_ns, str(r["key"]))
        masses.append(c3d_mass(c))
    masses = np.asarray([m for m in masses if np.isfinite(m)])
    if len(masses):
        return float(np.median(masses)), "c3d"
    w = mat_char(subject_ns, allow_lockbox=allow_lockbox, allow_eval=allow_eval).get("Weight", float("nan"))
    return (float(w), "mat") if np.isfinite(w) and w > 0 else (float("nan"), "")


# ---------------------------------------------------------------- public API
def root_track(source, subject_ns, take, allow_lockbox=False, allow_eval=False):
    """RootTrack of one recording (module docstring): gait120 PELV_C of a step or a joined trial, stroke
    PELO of one C3D trial with the trial's C3D body mass."""
    rows = index_rows(source, subject_ns, take, allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    if source == GAIT120:
        keys = sorted(str(r["key"]) for r in rows)
        parts, fps = _join_steps(_gait120_steps(subject_ns, keys))
        frames = np.concatenate([p[1] for p in parts])
        pelvis = np.concatenate([_centroid(p[2], p[0], PELV_C) for p in parts])
        members = tuple(p[3] for p in parts)
        mass = float("nan")
    else:
        c, labels, member = _stroke_c3d(subject_ns, str(take))
        pelvis = _stroke_point(c, labels, "PELO") @ UP_ROTATION.T
        fps = float(c.rate)
        frames = np.arange(c.n_frames, dtype=np.int64)
        members = (member,)
        mass = c3d_mass(c)
    valid = np.isfinite(pelvis).all(axis=1)
    t = (frames - frames[0]).astype(np.float64) / fps
    return RootTrack(source=source, subject_ns=subject_ns, key=str(take), t=t, pelvis=pelvis, valid=valid,
                     fps=float(fps), frames=frames, mass_kg=float(mass), members=members)


@functools.lru_cache(maxsize=1)
def _skeleton():
    import yaml
    with open(f"{paths.MOVEATLAS}/configs/canonical_skeleton.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["canonical_joints"], {GAIT120: cfg["sources"]["gait120_pig_markers"]["joint_map"],
                                     STROKE: cfg["sources"]["pig_vancriekinge"]["joint_map"]}


def _resolve(rule, point):
    if "direct" in rule:
        return point(rule["direct"])
    if "proxy" in rule:
        return point(rule["proxy"])
    a, b = rule["midpoint"]
    return 0.5 * (point(a) + point(b))


def raw_joints15(source, subject_ns, take, allow_lockbox=False, allow_eval=False):
    """(T, 15, 3) m, y up, NaN where missing: the canonical 15 joints of one recording from the raw
    markers or model outputs through the joint map of MoveAtlas configs/canonical_skeleton.yaml, before
    root-centring and scaling (the records' coords x scale plus the root track). Same keys and refusals
    as root_track."""
    rows = index_rows(source, subject_ns, take, allow_eval=allow_eval, allow_lockbox=allow_lockbox)
    joints, maps = _skeleton()
    if source == GAIT120:
        parts, _ = _join_steps(_gait120_steps(subject_ns, sorted(str(r["key"]) for r in rows)))
        out = []
        for markers, _, xyz, _ in parts:
            idx = {n: i for i, n in enumerate(markers)}
            virtual = {"PELV_C": _centroid(xyz, markers, PELV_C), "HEAD_C": _centroid(xyz, markers, HEAD_C)}
            point = (lambda n, xyz=xyz, idx=idx, virtual=virtual: virtual[n] if n in virtual else xyz[:, idx[n]])
            out.append(np.stack([_resolve(maps[GAIT120][j], point) for j in joints], axis=1))
        return np.concatenate(out)
    c, labels, _ = _stroke_c3d(subject_ns, str(take))
    missing = [n for n in STROKE_CORE if n not in labels]
    if missing:
        raise ValueError(f"stroke {subject_ns} {take}: no model output {missing}")

    def point(n):
        if n not in labels and n in STROKE_NANFILL:
            return np.full((c.n_frames, 3), np.nan)
        return _stroke_point(c, labels, n)

    X = np.stack([_resolve(maps[STROKE][j], point) for j in joints], axis=1)
    return X @ UP_ROTATION.T


def root_for_paired(pr, allow_lockbox=False, allow_eval=False):
    """(root (T, 3) m y up, valid (T,) bool, RootTrack) aligned with the pose samples of a
    sources.PairedRecord (gait120 by frame number through pairing['pose_frames'], stroke sample by sample)."""
    rt = root_track(pr.source, pr.subject_ns, pr.key, allow_lockbox=allow_lockbox, allow_eval=allow_eval)
    n = len(pr.t["pose3d"])
    if pr.source == GAIT120:
        frames = np.asarray(pr.pairing["pose_frames"], dtype=np.int64)
        pos = {int(f): i for i, f in enumerate(rt.frames)}
        miss = [int(f) for f in frames if int(f) not in pos]
        if miss:
            raise ValueError(f"gait120 {pr.subject_ns} {pr.key}: pose frames {miss[:3]} not in the TRC frames")
        sel = np.array([pos[int(f)] for f in frames])
    else:
        if rt.frames.shape[0] != n:
            raise ValueError(f"stroke {pr.subject_ns} {pr.key}: {rt.frames.shape[0]} C3D frames for {n} pose samples")
        sel = np.arange(n)
    return rt.pelvis[sel], rt.valid[sel], rt
