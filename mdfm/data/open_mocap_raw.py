"""Raw landmarks of the open body store's takes (plan docs/MODEL_OPEN_PORT_PLAN.md, step M3), in metres, y up, with
the root kept.

The canonical pose records are root-centred and divided by a torso length (MoveAtlas build_canonical.make_record), so
the body fit reads the raw files again, with the rules that made the records:
- the open motion capture of 2026-11-12 (MoveAtlas ingest_open_mocap.py, imported read-only with bytecode writing off:
  its file lists, its readers mocap_readers.py and its joint maps configs/canonical_skeleton.yaml open_*);
- gait120 and stroke_gait_vancriekinge able-bodied through mdfm.data.mocap_raw (raw_joints15 and its raw helpers).
Per take: the 15 canonical landmarks (T, 15, 3) and one toe landmark per foot (T, 2, 3) (TOE_NAMES: the marker on the
forefoot, or the BVH toe joint), NaN where missing; the take's rate from its file; the up axis measured on the take
(ingest_open_mocap.measure_up) turned to +y (UP_TO_Y); units by the file, and for style100, whose BVH states no unit,
by STYLE100_UNIT (centimetres, by the segment lengths it gives); a style100 take is cut to the span of the dataset's
Frame_Cuts.csv (style100_cut).

Refusals come before any file opens: every take goes through mdfm.data.canonical.check with its tiers row (EVAL, a
lockbox row, CODY and EXCLUDED_V1 refused; a gait120 or stroke lockbox row opens only for mdfm.gates code inside the
open G1 lockbox read, load_take's allow_lockbox); open_takes() lists the takes a source's files hold by their names only.
A take of a source outside SOURCES raises KeyError.
"""
import functools
import os
import re
import sys

import numpy as np

from mdfm import paths

OPEN_SOURCES = ("style100", "ceti_age_kinematics", "cmu_mocap", "deepdance", "umons_taichi", "moore_perturbed_gait",
                "push_recovery_perturbations", "notredame_noncyclic", "lencioni_gait_stairs")
GAIT120 = "gait120"
STROKE = "stroke_gait_vancriekinge"
SOURCES = OPEN_SOURCES + (GAIT120, STROKE)
C3D_SOURCES = ("cmu_mocap", "umons_taichi", "push_recovery_perturbations", "notredame_noncyclic")
# the toe landmark of each source, per side: a list of rules, the first whose markers exist wins (a tuple: midpoint)
TOE_NAMES = {
    "cmu_mocap": ["{S}TOE"], "push_recovery_perturbations": ["{S}TOE"], "umons_taichi": ["{S}_FM2", ("{S}_FM1", "{S}_FM5")],
    "notredame_noncyclic": [("{S}FM1", "{S}FM5"), "{S}TOE"], "deepdance": ["DD_{S}ToeTip"],
    "lencioni_gait_stairs": ["{S}xToe1"], "style100": ["{Side}Toe"], "ceti_age_kinematics": ["Character1_{Side}Foot_End"],
    "moore_perturbed_gait": ["{S}TOE"], GAIT120: ["{S}TOE"], STROKE: ["{S}TOE"]}
# sources whose hip landmark is a pelvis marker in MoveAtlas configs/canonical_skeleton.yaml (gait120 LASI, cmu and
# push recovery LFWT, deepdance DD_WaistLFront, lencioni LxAsis): the fit puts it on the pelvis (openfit.segments)
HIP_ON_PELVIS = ("gait120", "cmu_mocap", "push_recovery_perturbations", "deepdance", "lencioni_gait_stairs")
STYLE100_UNIT = 0.01        # centimetres: read so, a style100 take has hips to neck 0.54 m, thigh 0.44 m, shank 0.43 m
#                             (probe of 2026-11-12 on WildLegs_FW); the stored takes record pelvis_neck_m
_AXES = {"x": 0, "y": 1, "z": 2}


def _ma():
    """MoveAtlas ingest_open_mocap and mocap_readers, imported from the tree read-only (no bytecode written)."""
    root = paths.MOVEATLAS
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if root not in sys.path:
            sys.path.insert(0, root)
        import ingest_open_mocap as IO
        import mocap_readers as MR
    finally:
        sys.dont_write_bytecode = prev
    return IO, MR


def up_to_y(axis):
    """(3, 3) the rotation taking the signed lab axis ('+z', '-y', ...) to +y (a quarter or half turn about a lab
    axis, right-handed)."""
    sign = -1.0 if axis.startswith("-") else 1.0
    a = _AXES[axis[-1]]
    if a == 1:
        return np.eye(3) if sign > 0 else np.diag([1.0, -1.0, -1.0])
    if a == 2:                                   # z up: (x, y, z) -> (x, z, -y), or its mirror for -z
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        return R if sign > 0 else np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # x up: (x, y, z) -> (-y, x, z)
    return R if sign > 0 else np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _point(labels, pts, name):
    idx = {n: i for i, n in enumerate(labels)}
    return pts[:, idx[name]] if name in idx else None


def toe_landmarks(source, labels, pts):
    """(T, 2, 3) [left, right] toe landmarks by TOE_NAMES (NaN when no rule's markers exist)."""
    out = np.full((pts.shape[0], 2, 3), np.nan)
    for k, (S, Side) in enumerate((("L", "Left"), ("R", "Right"))):
        for rule in TOE_NAMES[source]:
            names = [rule] if isinstance(rule, str) else list(rule)
            got = [_point(labels, pts, n.format(S=S, Side=Side)) for n in names]
            if all(g is not None for g in got):
                out[:, k] = np.mean(np.stack(got), axis=0)
                break
    return out


@functools.lru_cache(maxsize=None)
def open_takes(source):
    """{(subject_ns, take): (path, extra)} of an open motion-capture source, from its file names only (the take
    names of ingest_open_mocap.py; a multi-actor CMU take is listed under its file take and resolved on load)."""
    IO, _ = _ma()
    out = {}
    if source in C3D_SOURCES:
        for subject, take, p, extra, skip in IO.items_c3d(source):
            if not skip:
                out[(f"{source}:{subject}", take)] = (p, extra)
    elif source == "deepdance":
        for p in IO.walk(source, (".tsv",)):
            m = re.match(r"Subject0*(\d+)_(.+)\.tsv$", os.path.basename(p))
            k = (f"{source}:S{int(m.group(1)):02d}", m.group(2))
            out.setdefault(k, (p, {}))               # the second file of a repeated take is a possible duplicate
    elif source in ("style100", "ceti_age_kinematics"):
        for p in IO.walk(source, (".bvh",)):
            stem = os.path.basename(p)[:-4]
            if source == "style100":
                out[(f"{source}:P01", stem)] = (p, {})
            else:
                m = re.match(r"(sub-d\d+)_ses-(\d+)_task-(\w+?)_tracksys-[^_]+_run-(\d+)_motion$", stem)
                if m and not m.group(3).startswith("c"):
                    out[(f"{source}:{m.group(1)}", f"ses{m.group(2)}_{m.group(3)}_run{m.group(4)}")] = (p, {})
    elif source == "lencioni_gait_stairs":
        for p in IO.walk(source, (".mat",)):
            m = re.match(r"Subject(\d+)\.mat$", os.path.basename(p))
            if m:
                out[(f"{source}:S{int(m.group(1)):02d}", "*")] = (p, {})
    elif source == "moore_perturbed_gait":
        import glob
        import yaml
        for meta in sorted(glob.glob(os.path.join(IO.rawdir(source), "*", "T*", "meta-*.yml"))):
            with open(meta, encoding="utf-8") as f:
                y = yaml.safe_load(f)
            if int(y["subject"]["id"]) == 0:
                continue
            out[(f"{source}:S{int(y['subject']['id']):02d}", f"T{int(y['trial']['id']):03d}")] = (os.path.dirname(meta), {})
    else:
        raise KeyError(f"no open raw reader for {source!r}")
    return out


@functools.lru_cache(maxsize=1)
def _style100_cuts():
    import csv
    IO, _ = _ma()
    with open(os.path.join(IO.rawdir("style100"), "100STYLE", "100STYLE", "Frame_Cuts.csv"), encoding="utf-8",
              newline="") as f:
        return {r["STYLE_NAME"]: r for r in csv.DictReader(f)}


def style100_cut(take):
    """(start, stop) native frames, inclusive, of a style100 take by the dataset's Frame_Cuts.csv (the span its
    authors give as the style's usable motion; the frames outside it hold the capture's entry and exit, a jump at the
    end of some takes), None when the table gives none (N/A)."""
    style, _, kind = take.rpartition("_")
    r = _style100_cuts().get(style)
    if r is None or r.get(f"{kind}_START") in (None, "", "N/A"):
        return None
    return int(r[f"{kind}_START"]), int(r[f"{kind}_STOP"])


def _lencioni(path, take):
    import collections
    import scipy.io as sio
    s = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["s"]
    fps = float(s.KinFreq)
    vn = [str(x).strip() for x in np.atleast_1d(s.MarkerVarName)]
    names = [v[1:] for v in vn[::3]]
    count = collections.Counter()
    for x in np.atleast_1d(s.Data):
        task = re.sub(r"[^a-z0-9]+", "_", str(x.Task).strip().lower()).strip("_")
        count[task] += 1
        if f"{task}_{count[task]:02d}" == take:
            mk = np.asarray(x.Marker, dtype=float)
            pts = mk.T.reshape(mk.shape[1], len(names), 3)
            pts[np.all(pts == 0.0, axis=-1)] = np.nan
            return names, pts, fps
    raise LookupError(f"lencioni {path}: no take {take}")


def _load_open(source, subject_ns, take):
    """(labels, pts (T, M, 3) m, fps, joint-map key) of one open take, by ingest_open_mocap's rules."""
    IO, MR = _ma()
    key = IO.CFG["open_mocap_sources"][source]
    takes = open_takes(source)
    if source == "lencioni_gait_stairs":
        path, _ = takes[(subject_ns, "*")]
        labels, pts, fps = _lencioni(path, take)
        return labels, pts, fps, key
    base = re.sub(r"_actor\d+$", "", take) if source == "cmu_mocap" else take
    if (subject_ns, base) not in takes:
        raise LookupError(f"{source} {subject_ns} {take}: no raw file")
    path, _ = takes[(subject_ns, base)]
    if source in C3D_SOURCES:
        labels, pts, fps, _, _, _ = IO.load_c3d(path)
        if source == "cmu_mocap":
            groups = {}
            for i, lab in enumerate(labels):
                pre, _, name = lab.rpartition(":")
                if name.startswith("*") or not name:
                    continue
                groups.setdefault(pre, []).append((name, i))
            actors = [g for pre, g in groups.items() if {"LFWT", "RFWT", "LBWT", "RBWT"} <= {nm for nm, _ in g}]
            m = re.search(r"_actor(\d+)$", take)
            k = int(m.group(1)) - 1 if m else 0
            if (m is None) != (len(actors) == 1):
                raise LookupError(f"cmu {subject_ns} {take}: {len(actors)} actors in the file")
            g = actors[k]
            return [nm for nm, _ in g], pts[:, [i for _, i in g], :], fps, key
        return labels, pts, fps, key
    if source == "deepdance":
        q = MR.read_qualisys_tsv(path)
        return q["labels"], q["points"] * 1e-3, q["rate"], key
    if source in ("style100", "ceti_age_kinematics"):
        bvh = MR.read_bvh(path)
        pos, names = MR.bvh_positions(bvh)
        unit = STYLE100_UNIT if source == "style100" else 0.01
        return names, pos * unit, bvh["rate"], key
    if source == "moore_perturbed_gait":
        import glob
        want = sorted({n for rule in IO.CFG["sources"][key]["joint_map"].values() for n in IO._all_names(rule)}
                      | {"LTOE", "RTOE"})
        txt = glob.glob(os.path.join(path, "mocap-*.txt"))
        r = MR.read_dflow_mocap_txt(txt[0], [IO.MOORE_FILE_NAME.get(x, x) for x in want])
        present = [IO.MOORE_CONFIG_NAME.get(x, x) for x in r["labels_present"]]
        return present, r["points"][:, [want.index(x) for x in present], :], r["rate"], key
    raise KeyError(source)


def _gates_granted(allow_lockbox, caller_frame):
    """allow_lockbox as granted: False when not asked, True when the calling module is mdfm.gates or one of its
    submodules, Refused otherwise: mdfm/data/lockbox_flag.gates_override, the shipped rule (no AMASS module)."""
    from mdfm.data.lockbox_flag import gates_override
    return gates_override(allow_lockbox, caller_frame)


def load_take(row, measure_up=True, allow_lockbox=False):
    """One take of the store (row: a tiers or index row with source, subject_ns, person_id, key and the tier
    columns), after canonical.check -> {'landmarks' (T, 15, 3) m, 'toes' (T, 2, 3) m, both y up, NaN missing;
    'fps', 'up_axis' (the lab axis turned to +y), 'labels'}. allow_lockbox=True opens a lockbox row of gait120 or
    stroke_gait_vancriekinge and is honoured for a caller in mdfm.gates only (lockbox_flag.gates_override: the open
    G1 lockbox read, mdfm/gates/g1_open.py); the open sources' takes never pass it."""
    from mdfm.data import canonical
    from mdfm.data import mocap_raw as MRAW
    granted = _gates_granted(allow_lockbox, sys._getframe(1))
    source = row["source"]
    if source not in SOURCES:
        raise KeyError(f"no raw reader for {source!r}")
    if granted and source not in (GAIT120, STROKE):
        raise canonical.Refused(f"{source}: no lockbox row of an open source is read")
    canonical.check(dict(row), allow_lockbox=granted)
    if source == GAIT120:
        rows = MRAW.index_rows(source, row["subject_ns"], row["key"], allow_lockbox=granted)
        parts, fps = MRAW._join_steps(MRAW._gait120_steps(row["subject_ns"], sorted(str(r["key"]) for r in rows)))
        lm = MRAW.raw_joints15(source, row["subject_ns"], row["key"], allow_lockbox=granted)
        toes = np.concatenate([toe_landmarks(source, list(markers), xyz) for markers, _, xyz, _ in parts])
        return {"landmarks": lm, "toes": toes, "fps": float(fps), "up_axis": "+y", "labels": None}
    if source == STROKE:
        lm = MRAW.raw_joints15(source, row["subject_ns"], row["key"], allow_lockbox=granted)
        c, labels, _ = MRAW._stroke_c3d(row["subject_ns"], str(row["key"]))
        pts = np.stack([MRAW._stroke_point(c, labels, n) if labels.count(n) == 1 else
                        np.full((c.n_frames, 3), np.nan) for n in ("LTOE", "RTOE")], axis=1)
        toes = toe_landmarks(source, ["LTOE", "RTOE"], pts) @ MRAW.UP_ROTATION.T
        return {"landmarks": lm, "toes": toes, "fps": float(c.rate), "up_axis": "+z", "labels": None}
    IO, _ = _ma()
    labels, pts, fps, key = _load_open(source, row["subject_ns"], str(row["key"]).partition("#")[0])
    cut = None
    if source == "style100":
        cut = style100_cut(str(row["key"]))
        if cut is not None:
            pts = pts[cut[0]:cut[1] + 1]
    canon, provided, _ = IO.to_canonical(key, labels, pts)
    toes = toe_landmarks(source, list(labels), pts)
    up, _ = IO.measure_up(canon) if measure_up else ("+y", "")
    if not up:
        raise ValueError(f"{source} {row['subject_ns']} {row['key']}: up axis not measurable")
    R = up_to_y(up)
    return {"landmarks": canon @ R.T, "toes": toes @ R.T, "fps": float(fps), "up_axis": up, "labels": list(labels),
            "provided": provided, "frame_cut": cut}
